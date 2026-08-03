#!/usr/bin/env python3
"""Per-cancer, per-module Shapley contribution to the ensemble, STRATIFIED by tumor-fraction bin.

REWIRED 2026-07-28 onto the CURRENT definitive benchmark (benchmark_nested_v2/{coh}_phase1.pkl):
  - 5 modules = rc_benchmark_nested_v2.MODULES (Genome_wide_CNA, E1_entropy, Somatic_mutation_profile
    [NO-FILTER], All_exon_depth, SHAPE_nep300 [percell]) — supersedes the stale _oof_cache version that
    used SHAPE_gini5_len5 + old mutation and an IN-SAMPLE NNLS combiner.
  - Combiner is the benchmark's OWN nested shrink60 Super-Learner (shrink_fit on outer-TRAIN meta-OOF,
    shrink_apply to held-out outer-TEST) — leakage-free, exactly matching rc_shapley_percancer.py.
For each subset S of modules build the nested ensemble OOF; for each (cancer c, TF bin b): v(S) = repeat-mean
OVR-AUROC of cancer c evaluated ONLY on bin-b samples (v(empty):=0.5). EXACT Shapley over the 2^5 subsets ->
phi_{c,b,module}. Cells with < MINPOS positive bin-samples blanked. Sum_module phi = (c OVR-AUROC in b) - 0.5.
Out: results/plot/Fig_shapley_percancer_tfbin_{v1,v2}.png (+ .pdf) and fig_shapley_percancer_tfbin.csv
"""
import os, sys, itertools, math, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_benchmark_nested_v2 import (shrink_fit, shrink_apply, MODULES, SHRINK_PRIMARY,
                                    N_SPLITS, N_REPEATS, SEED, OUT)

PROJ = "/home/jrkim/TSO_TFBS/project"
RC = f"{PROJ}/results/rule_conformant"; FEAT = f"{PROJ}/results/auto_plan/feat_rc"; MAN = f"{RC}/manifest_dev.tsv"
PLOT = f"{PROJ}/results/plot"
MIN_CLASS = 20; RC_EXPECT = {"v1": 1093, "v2": 796}
FILE_MODULES = ["Genome_wide_CNA", "E1_entropy", "Somatic_mutation_profile", "All_exon_depth"]
LABEL = {"E1_entropy": "E1 entropy", "All_exon_depth": "All-exon depth", "Genome_wide_CNA": "Broad CNA",
         "Somatic_mutation_profile": "Somatic mutation", "SHAPE_nep300": "TFBS SHAPE"}
MODLABS = [LABEL[m] for m in MODULES]
BINS = [(-np.inf, 0.03, "<3%"), (0.03, 0.10, "3-10%"), (0.10, 0.20, "10-20%"), (0.20, np.inf, ">20%")]
BINLABS = [b[2] for b in BINS]
MINPOS = 5
NJOB = int(os.environ.get("NJOB", "12"))
V_EMPTY = 0.5


def labels(coh):
    return {l.split("\t")[0]: l.split("\t")[2] for l in open(MAN).read().splitlines()[1:] if l.split("\t")[1] == coh}


def sids_of(p): return [str(s) for s in np.load(p, allow_pickle=True)["sids"]]


def rebuild_order(coh):
    lab = labels(coh); common = set(lab) & set(sids_of(f"{FEAT}/{coh}/X_SHAPE.npz"))
    for m in FILE_MODULES: common &= set(sids_of(f"{FEAT}/{coh}/X_{m}.npz"))
    cnt = {}
    for s in common: cnt[lab[s]] = cnt.get(lab[s], 0) + 1
    classes = sorted([c for c, nn in cnt.items() if nn >= MIN_CLASS]); ci = {c: i for i, c in enumerate(classes)}
    order = sorted([s for s in common if lab[s] in ci])
    return order, np.array([ci[lab[s]] for s in order]), classes


def subset_ens(S, trainOOF, testpred, outer, y, K, n):
    """nested shrink60 ensemble OOF for module subset S -> (N_REPEATS, n, K); empty subset handled by caller."""
    mods = [MODULES[j] for j in S]
    ens = np.zeros((N_REPEATS, n, K))
    for f_, (OTR, OTE) in enumerate(outer):
        Wk, _ = shrink_fit(trainOOF[f_], y[OTR], K, mods, SHRINK_PRIMARY)
        ens[f_ // N_SPLITS][OTE] = shrink_apply(testpred[f_], Wk, K, mods)
    return ens


# present cancer labels like Fig2D (Shapley): short Title-Case organ names (user 2026-07-29); fallback = capitalise.
CMAP = {"biliary tract cancer": "Biliary", "bladder cancer": "Bladder", "breast cancer": "Breast",
        "colorectal cancer": "Colorectal", "gastric cancer": "Stomach", "liver cancer": "Liver",
        "lung cancer": "Lung", "melanoma": "Melanoma", "pancreatic cancer": "Pancreas",
        "prostate cancer": "Prostate", "sarcoma": "Sarcoma"}
def cap(s): return CMAP.get(s, s[:1].upper() + s[1:])


def main():
    n = len(MODULES); idx = list(range(n))
    coef = {s: math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)}
    rows = []
    for coh in ("v1", "v2"):
        z = np.load(f"{OUT}/{coh}_oof.npz", allow_pickle=True)
        y = z["y"]; classes = [str(c) for c in z["classes"]]; K = len(classes); N = len(y)
        order, y_re, cls_re = rebuild_order(coh)
        if len(order) != RC_EXPECT[coh]: sys.exit(f"ABORT {coh}: n={len(order)} != {RC_EXPECT[coh]}")
        if cls_re != classes or not np.array_equal(y_re, y): sys.exit(f"ABORT {coh}: order/y mismatch")
        d = pd.read_csv(f"{PROJ}/results/rcv{coh[-1]}_cna/cna_feature_table.tsv", sep="\t",
                        dtype={"Unnamed: 0": str}).rename(columns={"Unnamed: 0": "sid"})
        tfm = dict(zip(d.sid, pd.to_numeric(d.tumor_fraction, errors="coerce")))
        tf = np.array([tfm.get(s, np.nan) for s in order])
        print(f"[{coh}] order verified n={N} K={K}; tf finite={int(np.isfinite(tf).sum())}", flush=True)

        with open(f"{OUT}/{coh}_phase1.pkl", "rb") as f: D = pickle.load(f)
        trainOOF, testpred = D["trainOOF"], D["testpred"]
        outer = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(np.zeros(N), y))
        subsets = [S for size in range(1, n + 1) for S in itertools.combinations(idx, size)]
        ens_list = Parallel(n_jobs=NJOB, prefer="processes", max_nbytes="100M")(
            delayed(subset_ens)(S, trainOOF, testpred, outer, y, K, N) for S in subsets)
        ENS = {S: e for S, e in zip(subsets, ens_list)}                 # each (N_REPEATS, n, K)
        print(f"[{coh}] {len(subsets)} nested subset-ensembles built", flush=True)

        for lo, hi, blab in BINS:
            m = np.isfinite(tf) & (tf > lo) & (tf <= hi)
            yb = y[m]
            for k in range(K):
                pos = (yb == k); npos = int(pos.sum())
                if npos < MINPOS or npos == len(yb):
                    for i in idx: rows.append(dict(cohort=coh, cancer=cap(classes[k]), bin=blab, module=LABEL[MODULES[i]], shapley=np.nan, npos=npos))
                    continue
                # v(S)(k,bin) = repeat-mean OVR-AUROC on bin-b samples; empty subset := 0.5
                vS = {(): V_EMPTY}
                for S in subsets:
                    e = ENS[S]
                    vS[S] = float(np.mean([roc_auc_score(pos.astype(int), e[r][m][:, k]) for r in range(N_REPEATS)]))
                for i in idx:
                    others = [j for j in idx if j != i]; phi = 0.0
                    for size in range(len(others) + 1):
                        for S in itertools.combinations(others, size):
                            phi += coef[size] * (vS[tuple(sorted(S + (i,)))] - vS[tuple(sorted(S))])
                    rows.append(dict(cohort=coh, cancer=cap(classes[k]), bin=blab, module=LABEL[MODULES[i]], shapley=phi, npos=npos))
        print(f"[{coh}] done", flush=True)
    Dd = pd.DataFrame(rows); Dd.to_csv(f"{PLOT}/fig_shapley_percancer_tfbin.csv", index=False)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "svg.fonttype": "none"})
    vmax = float(np.nanmax(np.abs(Dd.shapley.values)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("#eeeeee")
    for c in ("v1", "v2"):
        d = Dd[Dd.cohort == c]
        order = d.groupby("cancer").npos.max().sort_values(ascending=False).index.tolist()
        fig, axes = plt.subplots(1, len(BINLABS), figsize=(4.2 * len(BINLABS), 0.62 * len(order) + 2.2), sharey=True)
        for bi, blab in enumerate(BINLABS):
            ax = axes[bi]; M = np.full((len(order), len(MODLABS)), np.nan)
            for ri, ca in enumerate(order):
                for ci, mo in enumerate(MODLABS):
                    v = d[(d.cancer == ca) & (d.bin == blab) & (d.module == mo)].shapley.values
                    M[ri, ci] = v[0] if len(v) else np.nan
            ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
            for ri in range(len(order)):
                for ci in range(len(MODLABS)):
                    if np.isfinite(M[ri, ci]):
                        ax.text(ci, ri, f"{M[ri,ci]:.2f}", ha="center", va="center", fontsize=9,
                                color="white" if abs(M[ri, ci]) > 0.6 * vmax else "black", fontweight="bold")
            ax.set_xticks(range(len(MODLABS))); ax.set_xticklabels(MODLABS, rotation=35, ha="right", fontsize=10)
            ax.set_title(f"{blab}", fontsize=14, fontweight="bold")
            if bi == 0: ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=11)
            ax.set_xticks(np.arange(-.5, len(MODLABS), 1), minor=True); ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
            ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.015); cb.set_label("Shapley (Δ AUROC)", fontsize=13)
        fig.suptitle(f"Per-cancer module contribution to ensemble across tumor-fraction bins — {c}",
                     y=1.0, fontsize=16, fontweight="bold")
        out = f"{PLOT}/Fig_shapley_percancer_tfbin_{c}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight"); print("WROTE", out); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight"); plt.close(fig)
    print("SHAPLEY_PERCANCER_TFBIN_DONE")


if __name__ == "__main__":
    main()
