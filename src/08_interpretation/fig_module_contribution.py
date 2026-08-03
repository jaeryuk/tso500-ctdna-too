#!/usr/bin/env python3
"""Weighted MODULE-CONTRIBUTION decomposition for objectively chosen representative samples,
plus per-sample repeated-CV uncertainty (v1/v2).

Reconstructs the benchmark's OWN per-class shrinkage-NNLS combiner (shrink60) from
benchmark_nested_v2/{coh}_phase1.pkl on the exact outer folds (RepeatedStratifiedKFold 5x20, seed 42):
  For outer fold f: Wk = shrink_fit(trainOOF[f]); the held-out testpred[f] gives, for class c,
  module m:  contribution_{m,c}(sample i) = Wk[c][m] * P_m(i,c)   (the true NNLS-weighted module input).
Each sample is in outer-test once per repeat; we average its contribution over the 20 repeats.
Sum_m contribution_{m,c} = the (un-normalised) ensemble score for class c; primary[i,c] (from oof.npz,
the row-normalised mean over repeats) is overlaid as the final ensemble probability.

Combiner helpers (_norm/wnnls/global_weights/shrink_fit/shrink_apply) are copied VERBATIM from
rc_benchmark_nested_v2 so nothing heavy (xgboost/sklearn base learners) is imported.

Representative samples per cohort (objective):
  * median-confidence CORRECT      : correct sample whose signed margin is nearest the median correct margin
  * most-ambiguous CORRECT         : correct sample with the smallest positive margin
  * highest-confidence MISCLASSIFIED: misclassified sample with the most negative margin

Out (results/plot):
  Fig_oof_module_contribution_{coh}.png   3 representative samples x stacked module contributions
  Tab_module_contribution.tsv             the decomposition behind the figure
  Tab_oof_repeat_uncertainty.tsv          per-sample p_true / p_top1 mean+sd across the 20 repeats
"""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_oof_probability import gather, SHORT, COHORTS

PROJ = "/home/jrkim/TSO_TFBS/project"; PLOT = f"{PROJ}/results/plot"
OUT = f"{PROJ}/results/rule_conformant/benchmark_nested_v2"
MODULES = ["Genome_wide_CNA", "E1_entropy", "Somatic_mutation_profile", "All_exon_depth", "SHAPE_nep300"]
MODLAB = {"E1_entropy": "E1 entropy", "All_exon_depth": "All-exon depth", "Genome_wide_CNA": "Broad CNA",
          "Somatic_mutation_profile": "Somatic mutation", "SHAPE_nep300": "TFBS SHAPE"}
MODCOL = {"E1_entropy": "#8c6bb1", "All_exon_depth": "#41ab5d", "Genome_wide_CNA": "#4292c6",
          "Somatic_mutation_profile": "#ef6548", "SHAPE_nep300": "#f0a202"}
N_SPLITS, N_REPEATS, SEED, LAMBDA = 5, 20, 42, 60


# --------- combiner (verbatim from rc_benchmark_nested_v2) ---------
def _norm(P):
    P = np.clip(P, 1e-12, None); return P / P.sum(1, keepdims=True)


def wnnls(Z, t):
    w = (t == 1).mean(); sw = np.where(t == 1, 0.5 / max(w, 1e-6), 0.5 / max(1 - w, 1e-6))
    s = np.sqrt(sw)[:, None]; Zc = np.column_stack([Z, np.ones(len(Z))])
    c, _ = nnls(Zc * s, t * np.sqrt(sw)); return c[:-1]


def global_weights(oof_tr, y_tr, K, mods):
    Zg = []; tg = []
    for k in range(K):
        Zg.append(np.column_stack([oof_tr[m][:, k] for m in mods])); tg.append((y_tr == k).astype(int))
    wg = wnnls(np.vstack(Zg), np.concatenate(tg))
    return wg / wg.sum() if wg.sum() > 1e-9 else np.full(len(mods), 1 / len(mods))


def shrink_fit(oof_tr, y_tr, K, mods, const):
    nm = len(mods); wg = global_weights(oof_tr, y_tr, K, mods)
    cnt = np.bincount(y_tr, minlength=K)
    alpha = np.ones(K) if const == 0 else (np.zeros(K) if np.isinf(const) else np.clip(cnt / (cnt + const), 0, 1))
    Wk = []
    for k in range(K):
        if alpha[k] <= 0:
            Wk.append(wg.copy()); continue
        Z = np.column_stack([oof_tr[m][:, k] for m in mods]); wk = wnnls(Z, (y_tr == k).astype(int))
        wk = wk / wk.sum() if wk.sum() > 1e-9 else wg.copy()
        W = alpha[k] * wk + (1 - alpha[k]) * wg; Wk.append(W / max(W.sum(), 1e-9))
    return Wk, wg


def shrink_apply(pred, Wk, K, mods):
    P = np.zeros((len(pred[mods[0]]), K))
    for k in range(K):
        Z = np.column_stack([pred[m][:, k] for m in mods]); P[:, k] = Z @ Wk[k]
    return _norm(P)


def decompose(coh, g):
    """-> contrib (N,nmod,K) mean-over-repeats weighted module contribution; ens_sd, ptrue_sd per sample."""
    y = g["y"]; N = len(y); K = g["K"]; nmod = len(MODULES)
    with open(f"{OUT}/{coh}_phase1.pkl", "rb") as f:
        D = pickle.load(f)
    trainOOF, testpred = D["trainOOF"], D["testpred"]
    outer = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(np.zeros(N), y))
    contrib = np.zeros((N, nmod, K)); cnt = np.zeros(N)
    ens_rep = np.full((N_REPEATS, N, K), np.nan)
    for f_, (OTR, OTE) in enumerate(outer):
        Wk, _ = shrink_fit({m: trainOOF[f_][m] for m in MODULES}, y[OTR], K, MODULES, LAMBDA)
        pred = testpred[f_]
        for mi, m in enumerate(MODULES):
            Wrow = np.array([Wk[k][mi] for k in range(K)])        # weight of module m per class
            contrib[OTE, mi, :] += pred[m] * Wrow[None, :]
        ens_rep[f_ // N_SPLITS][OTE] = shrink_apply(pred, Wk, K, MODULES)
        cnt[OTE] += 1
    contrib /= cnt[:, None, None]
    ptrue = ens_rep[:, np.arange(N), y]                            # (N_REPEATS, N)
    ptop = np.nanmax(ens_rep, axis=2)                              # (N_REPEATS, N)
    return contrib, np.nanstd(ptrue, 0), np.nanmean(ptrue, 0), np.nanstd(ptop, 0), np.nanmean(ptop, 0)


def pick(g):
    y = g["y"]; mg = g["margin"]; corr = g["top1"] == y
    ci = np.where(corr)[0]; wi = np.where(~corr)[0]
    cm = mg[ci]; med = np.median(cm)
    p_med = int(ci[np.argmin(np.abs(cm - med))])
    p_amb = int(ci[np.argmin(cm)])
    p_wrong = int(wi[np.argmin(mg[wi])]) if len(wi) else int(ci[np.argmin(cm)])
    return [("median-confidence correct", p_med), ("most-ambiguous correct", p_amb),
            ("highest-confidence misclassification", p_wrong)]


def contribution_fig(coh, g, contrib):
    cl = [SHORT.get(c, c) for c in g["classes"]]; K = g["K"]
    reps = pick(g)
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.6), sharey=False)
    rows = []
    for ax, (cat, i) in zip(axes, reps):
        bottoms_pos = np.zeros(K); bottoms_neg = np.zeros(K)
        for mi, m in enumerate(MODULES):
            vals = contrib[i, mi, :]
            base = np.where(vals >= 0, bottoms_pos, bottoms_neg)
            ax.bar(range(K), vals, bottom=base, color=MODCOL[m], width=0.8,
                   label=MODLAB[m], edgecolor="white", linewidth=0.4)
            bottoms_pos += np.where(vals >= 0, vals, 0); bottoms_neg += np.where(vals < 0, vals, 0)
            for k in range(K):
                rows.append(dict(cohort=coh, sid=g["sids"][i], category=cat, cancer_type=cl[k],
                                 module=MODLAB[m], contribution=round(float(vals[k]), 5)))
        ax.scatter(range(K), g["P"][i], color="black", zorder=5, s=42, marker="D", label="ensemble probability")
        tl = g["y"][i]; pl = g["top1"][i]
        ax.set_xticks(range(K)); ax.set_xticklabels(cl, rotation=45, ha="right", fontsize=10)
        for xt, k in zip(ax.get_xticklabels(), range(K)):
            if k == tl: xt.set_color("#c0392b"); xt.set_fontweight("bold")
        ax.axhline(0, color="0.6", lw=0.8)
        mark = "correct" if pl == tl else "WRONG"
        ax.set_title(f"{cat}\n{g['sids'][i]}  true={cl[tl]}  pred={cl[pl]} [{mark}]  margin={g['margin'][i]:+.2f}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Weighted module contribution  (w$_{m,c}$·P$_{m,c}$)", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{coh}: NNLS-weighted module contributions for representative samples "
                 f"(red x-label = true class; diamonds = final ensemble probability)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(f"{PLOT}/Fig_oof_module_contribution_{coh}.png", dpi=190, bbox_inches="tight")
    fig.savefig(f"{PLOT}/Fig_oof_module_contribution_{coh}.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"WROTE Fig_oof_module_contribution_{coh}.png", flush=True)
    return rows


def main():
    all_rows = []; unc = []
    for coh in COHORTS:
        g = gather(coh)
        contrib, ptrue_sd, ptrue_mean, ptop_sd, ptop_mean = decompose(coh, g)
        all_rows += contribution_fig(coh, g, contrib)
        for i, s in enumerate(g["sids"]):
            unc.append(dict(sid=s, cohort=coh, true=SHORT.get(g["classes"][g["y"][i]], ""),
                            p_true_mean=round(float(ptrue_mean[i]), 5), p_true_sd=round(float(ptrue_sd[i]), 5),
                            p_top1_mean=round(float(ptop_mean[i]), 5), p_top1_sd=round(float(ptop_sd[i]), 5)))
        print(f"[{coh}] decomposed; median p_true_sd={np.median(ptrue_sd):.4f}", flush=True)
    pd.DataFrame(all_rows).to_csv(f"{PLOT}/Tab_module_contribution.tsv", sep="\t", index=False)
    pd.DataFrame(unc).to_csv(f"{PLOT}/Tab_oof_repeat_uncertainty.tsv", sep="\t", index=False)
    print("WROTE Tab_module_contribution.tsv + Tab_oof_repeat_uncertainty.tsv")
    print("MODULE_CONTRIBUTION_FIGS_DONE")


if __name__ == "__main__":
    main()
