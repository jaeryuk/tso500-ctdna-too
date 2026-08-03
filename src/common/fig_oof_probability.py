#!/usr/bin/env python3
"""Sample-level OUT-OF-FOLD ensemble prediction figures (paper-ready), per cohort v1/v2.

Uses the definitive nested benchmark posterior benchmark_nested_v2/{coh}_oof.npz -> `primary`
(shape (n,K), already the MEAN over the 20 repeated-CV OOF passes, rows sum to 1) and the
per-module OOF posteriors `mod_*`. Sample order rebuilt+verified by predict_proba_sample.rebuild_order.

Deliverables (all in results/plot):
  Tab_oof_probability_long.tsv   tidy: sid,cohort,true,top1,correct,signed_margin,tumor_fraction,
                                 source(ensemble/module),cancer_type,probability
  Tab_soft_confusion.tsv         cohort,true,pred,mean_prob,n   (row = true class, mean posterior)
  Fig_oof_soft_confusion.png     (a) soft confusion matrices v1|v2 (mean OOF prob per true class)
  Fig_oof_margin.png             (b) signed probability margin per cancer type (box+strip), v1|v2
  Fig_oof_sample_heatmap_{coh}.png (c) all-sample probability heatmap, grouped by true class,
                                 within class sorted by signed margin; true-class strip + top-1 hit +
                                 tumor-fraction annotation columns
  Fig_oof_calibration.png        (d) reliability curve of top-1 confidence + Brier + ECE, v1|v2

signed margin_i = P(true_i) - max_{k != true_i} P(k)   (>0  <=> top-1 correct).
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_proba_sample import load_cohort, labels, SHORT

PROJ = "/home/jrkim/TSO_TFBS/project"; PLOT = f"{PROJ}/results/plot"
MODKEYS = ["mod_E1_entropy", "mod_All_exon_depth", "mod_Genome_wide_CNA",
           "mod_Somatic_mutation_profile", "mod_SHAPE_nep300"]
MODNAME = {"mod_E1_entropy": "E1_entropy", "mod_All_exon_depth": "All_exon_depth",
           "mod_Genome_wide_CNA": "Broad_CNA", "mod_Somatic_mutation_profile": "Somatic_mutation",
           "mod_SHAPE_nep300": "TFBS_SHAPE"}
# muted steel-blue ramp echoing the house confusion-matrix style
BLUES = LinearSegmentedColormap.from_list("mutedblue",
        ["#ffffff", "#e8eef4", "#cddbe8", "#a7c3da", "#7aa6cc", "#4d84b8", "#2f5f8f"])
COHORTS = ("v1", "v2")


def tumor_frac(coh):
    d = pd.read_csv(f"{PROJ}/results/rcv{coh[-1]}_cna/cna_feature_table.tsv", sep="\t",
                    dtype={"Unnamed: 0": str}).rename(columns={"Unnamed: 0": "sid"})
    return dict(zip(d.sid, pd.to_numeric(d.tumor_fraction, errors="coerce")))


def gather(coh):
    """-> dict with sids, y, classes, P(ens n,K), mods{name:(n,K)}, tf(n,), margin(n,), top1(n,)."""
    z, order, classes = load_cohort(coh)
    y = z["y"].astype(int); P = z["primary"]; n, K = P.shape
    tfm = tumor_frac(coh); tf = np.array([tfm.get(s, np.nan) for s in order])
    top1 = P.argmax(1)
    pt = P[np.arange(n), y]
    Pm = P.copy(); Pm[np.arange(n), y] = -1.0          # mask true col to get max over others
    margin = pt - Pm.max(1)
    mods = {MODNAME[k]: z[k] for k in MODKEYS if k in z.files}
    return dict(coh=coh, sids=order, y=y, classes=classes, K=K, P=P, mods=mods,
                tf=tf, margin=margin, top1=top1)


# ------------------------------------------------------------------ TSVs
def write_tsv(G):
    rows = []
    for coh in COHORTS:
        g = G[coh]; cl = g["classes"]
        sh = [SHORT.get(c, c) for c in cl]
        for i, s in enumerate(g["sids"]):
            base = dict(sid=s, cohort=coh, true_label=sh[g["y"][i]],
                        top1_pred=sh[g["top1"][i]], correct=int(g["top1"][i] == g["y"][i]),
                        signed_margin=round(float(g["margin"][i]), 5),
                        tumor_fraction=round(float(g["tf"][i]), 5) if np.isfinite(g["tf"][i]) else "")
            for k in range(g["K"]):
                rows.append({**base, "source": "ensemble", "cancer_type": sh[k],
                             "probability": round(float(g["P"][i, k]), 5)})
            for mname, M in g["mods"].items():
                for k in range(g["K"]):
                    rows.append({**base, "source": mname, "cancer_type": sh[k],
                                 "probability": round(float(M[i, k]), 5)})
    pd.DataFrame(rows).to_csv(f"{PLOT}/Tab_oof_probability_long.tsv", sep="\t", index=False)
    print("WROTE Tab_oof_probability_long.tsv", flush=True)

    sc = []
    for coh in COHORTS:
        g = G[coh]; cl = [SHORT.get(c, c) for c in g["classes"]]
        for ti in range(g["K"]):
            m = g["y"] == ti; nrow = int(m.sum())
            mp = g["P"][m].mean(0)
            for pj in range(g["K"]):
                sc.append(dict(cohort=coh, true=cl[ti], pred=cl[pj],
                               mean_prob=round(float(mp[pj]), 5), n=nrow))
    pd.DataFrame(sc).to_csv(f"{PLOT}/Tab_soft_confusion.tsv", sep="\t", index=False)
    print("WROTE Tab_soft_confusion.tsv", flush=True)


# ------------------------------------------------------------------ (a) soft confusion
def soft_confusion(G):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))
    for ax, coh in zip(axes, COHORTS):
        g = G[coh]; cl = [SHORT.get(c, c) for c in g["classes"]]; K = g["K"]
        Msc = np.zeros((K, K)); nrow = np.zeros(K, int)
        for ti in range(K):
            m = g["y"] == ti; nrow[ti] = int(m.sum()); Msc[ti] = g["P"][m].mean(0)
        # True on x (with n=), Predicted on y  -> transpose so x=true(col index ti), y=pred
        im = ax.imshow(Msc.T, cmap=BLUES, vmin=0, vmax=1, aspect="equal")
        for ti in range(K):
            for pj in range(K):
                v = Msc[ti, pj]
                if v >= 0.005:
                    ax.text(ti, pj, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                            color="white" if v > 0.55 else "0.15", fontweight="bold")
        ax.set_xticks(range(K)); ax.set_xticklabels([f"{c}\n(n={nrow[i]})" for i, c in enumerate(cl)],
                                                    rotation=40, ha="right", fontsize=9.5)
        ax.set_yticks(range(K)); ax.set_yticklabels(cl, fontsize=9.5)
        ax.set_xlabel("True", fontsize=13); ax.set_ylabel("Predicted", fontsize=13)
        ax.set_title(coh, fontsize=15, fontweight="bold")
        ax.set_xticks(np.arange(-.5, K, 1), minor=True); ax.set_yticks(np.arange(-.5, K, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.0); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.set_label("Mean OOF probability", fontsize=12)
    fig.suptitle("Soft confusion matrix — mean out-of-fold probability assigned per true cancer type",
                 fontsize=15.5, fontweight="bold", y=1.02)
    fig.savefig(f"{PLOT}/Fig_oof_soft_confusion.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{PLOT}/Fig_oof_soft_confusion.pdf", bbox_inches="tight"); plt.close(fig)
    print("WROTE Fig_oof_soft_confusion.png", flush=True)


# ------------------------------------------------------------------ (b) signed margin
def margin_fig(G):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    for ax, coh in zip(axes, COHORTS):
        g = G[coh]; cl = [SHORT.get(c, c) for c in g["classes"]]; K = g["K"]
        data = [g["margin"][g["y"] == k] for k in range(K)]
        med = np.array([np.median(d) for d in data])
        acc = np.array([float(np.mean(d > 0)) for d in data])
        oo = np.argsort(med)                                   # worst at bottom, best at top
        bp = ax.boxplot([data[i] for i in oo], vert=False, widths=0.62, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.6),
                        whiskerprops=dict(color="0.4"), boxprops=dict(color="0.4"))
        for patch in bp["boxes"]: patch.set(facecolor="#dbe6f0", alpha=0.9)
        for yi, i in enumerate(oo, start=1):
            d = data[i]; jit = (np.random.RandomState(i).rand(len(d)) - 0.5) * 0.36
            col = np.where(d > 0, "#2c7a7b", "#c05621")
            ax.scatter(d, np.full(len(d), yi) + jit, s=7, c=col, alpha=0.55, linewidths=0)
            ax.text(1.02, yi, f"{acc[i]*100:.0f}%", va="center", ha="left", fontsize=9,
                    transform=ax.get_yaxis_transform())
        ax.axvline(0, color="0.3", ls="--", lw=1.2)
        ax.set_yticks(range(1, K + 1)); ax.set_yticklabels([cl[i] for i in oo], fontsize=11)
        ax.set_xlim(-1.02, 1.02); ax.set_xlabel("Signed probability margin  (P$_{true}$ − max P$_{other}$)", fontsize=12)
        ax.set_title(f"{coh}   (right: top-1 accuracy)", fontsize=14, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", ls="", color="#2c7a7b", label="correct (margin > 0)"),
           Line2D([0], [0], marker="o", ls="", color="#c05621", label="misclassified (margin < 0)")]
    fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Per-cancer-type signed probability margin (out-of-fold ensemble)",
                 fontsize=15.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(f"{PLOT}/Fig_oof_margin.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{PLOT}/Fig_oof_margin.pdf", bbox_inches="tight"); plt.close(fig)
    print("WROTE Fig_oof_margin.png", flush=True)


# ------------------------------------------------------------------ (c) sample heatmap
def sample_heatmap(G):
    # distinct per-class strip colors
    strip_cols = plt.cm.tab20.colors
    for coh in COHORTS:
        g = G[coh]; cl = [SHORT.get(c, c) for c in g["classes"]]; K = g["K"]; n = len(g["sids"])
        # order: by true class, then signed margin desc within class
        idx = sorted(range(n), key=lambda i: (g["y"][i], -g["margin"][i]))
        Pm = g["P"][idx]; yy = g["y"][idx]; corr = (g["top1"][idx] == yy); tf = g["tf"][idx]
        fig = plt.figure(figsize=(K * 0.62 + 3.2, 9.6))
        gs = fig.add_gridspec(1, 5, width_ratios=[0.30, K, 0.28, 0.28, 0.5], wspace=0.06)
        axs = fig.add_subplot(gs[0, 0]); axm = fig.add_subplot(gs[0, 1])
        axc = fig.add_subplot(gs[0, 2]); axt = fig.add_subplot(gs[0, 3]); axcb = fig.add_subplot(gs[0, 4])
        # main probability heatmap
        im = axm.imshow(Pm, cmap=BLUES, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        axm.set_xticks(range(K)); axm.set_xticklabels(cl, rotation=40, ha="right", fontsize=10)
        axm.set_yticks([]); axm.set_title(f"{coh}: sample-level OOF probability (n={n})", fontsize=13, fontweight="bold")
        axm.set_xlabel("Predicted cancer type", fontsize=12)
        # true-class strip + block boundaries + centered labels
        strip = yy.reshape(-1, 1)
        axs.imshow(strip, cmap=matplotlib.colors.ListedColormap([strip_cols[k % 20] for k in range(K)]),
                   aspect="auto", vmin=0, vmax=K - 1)
        axs.set_xticks([]); axs.set_yticks([]); axs.set_title("True", fontsize=11)
        bounds = np.where(np.diff(yy) != 0)[0] + 0.5
        for b in bounds:
            for a in (axm, axs, axc, axt): a.axhline(b, color="black", lw=0.7)
        for k in range(K):
            r = np.where(yy == k)[0]
            if len(r): axs.text(-0.9, r.mean(), cl[k], ha="right", va="center", fontsize=9.5, clip_on=False)
        # correct/wrong strip
        axc.imshow(corr.reshape(-1, 1), cmap=matplotlib.colors.ListedColormap(["#c05621", "#2c7a7b"]),
                   aspect="auto", vmin=0, vmax=1); axc.set_xticks([]); axc.set_yticks([])
        axc.set_title("top-1", fontsize=9, rotation=0)
        # tumor fraction strip
        tfc = np.clip(np.nan_to_num(tf, nan=0.0), 0, 0.5).reshape(-1, 1)
        imt = axt.imshow(tfc, cmap="magma", aspect="auto", vmin=0, vmax=0.5)
        axt.set_xticks([]); axt.set_yticks([]); axt.set_title("TF", fontsize=9)
        # colorbars
        cb = fig.colorbar(im, cax=axcb); cb.set_label("OOF probability", fontsize=11)
        fig.savefig(f"{PLOT}/Fig_oof_sample_heatmap_{coh}.png", dpi=190, bbox_inches="tight")
        fig.savefig(f"{PLOT}/Fig_oof_sample_heatmap_{coh}.pdf", bbox_inches="tight"); plt.close(fig)
        print(f"WROTE Fig_oof_sample_heatmap_{coh}.png", flush=True)


# ------------------------------------------------------------------ (d) calibration
def brier_ece(P, y, K, nb=10):
    n = len(y); onehot = np.eye(K)[y]
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
    conf = P.max(1); hit = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, nb + 1); ece = 0.0; xs = []; ys = []; ws = []
    for b in range(nb):
        m = (conf > edges[b]) & (conf <= edges[b + 1]) if b else (conf >= edges[0]) & (conf <= edges[1])
        if m.sum():
            c = conf[m].mean(); a = hit[m].mean(); xs.append(c); ys.append(a); ws.append(m.sum())
            ece += (m.sum() / n) * abs(a - c)
    return brier, ece, np.array(xs), np.array(ys), np.array(ws)


def calibration_fig(G):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for ax, coh in zip(axes, COHORTS):
        g = G[coh]; br, ece, xs, ys, ws = brier_ece(g["P"], g["y"], g["K"])
        acc = float(np.mean(g["top1"] == g["y"]))
        ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1.2, label="perfect")
        ax.scatter(xs, ys, s=ws / ws.max() * 320 + 20, color="#4d84b8", alpha=0.85, edgecolor="white", zorder=3)
        ax.plot(xs, ys, color="#2f5f8f", lw=1.6, zorder=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("Mean top-1 predicted probability", fontsize=12)
        ax.set_ylabel("Empirical accuracy", fontsize=12)
        ax.set_title(f"{coh}   top-1 acc={acc:.2f}  Brier={br:.3f}  ECE={ece:.3f}", fontsize=12.5, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=10, loc="upper left", frameon=False)
    fig.suptitle("Reliability of out-of-fold top-1 confidence (bubble size ∝ n samples in bin)",
                 fontsize=14.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{PLOT}/Fig_oof_calibration.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{PLOT}/Fig_oof_calibration.pdf", bbox_inches="tight"); plt.close(fig)
    print("WROTE Fig_oof_calibration.png", flush=True)


def main():
    G = {coh: gather(coh) for coh in COHORTS}
    for coh in COHORTS:
        g = G[coh]
        print(f"[{coh}] n={len(g['sids'])} K={g['K']} top1_acc={np.mean(g['top1']==g['y']):.3f} "
              f"mean_margin={g['margin'].mean():.3f} tf_finite={int(np.isfinite(g['tf']).sum())}", flush=True)
    write_tsv(G)
    soft_confusion(G)
    margin_fig(G)
    sample_heatmap(G)
    calibration_fig(G)
    print("OOF_PROBABILITY_FIGS_DONE")


if __name__ == "__main__":
    main()
