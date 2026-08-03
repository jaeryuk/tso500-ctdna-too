#!/usr/bin/env python
"""Per-TF enrichment ranking from the per-site contribution lens (padding window, raw features). For each
cancer type k and TF t, what fraction of t's TFBS are INFORMATIVE (|one-vs-rest site-AUROC - 0.5| >= DELTA)
via gini and/or lenent, normalised by the TF's own site count so abundance (e.g. CTCF) no longer dominates.
Also cancer-SPECIFICITY = frac(k,t) - mean_k' frac(k',t).

2026-07-18 (user directive): RULE-CONFORMANT cohort (rc_cohort.rc_table, keep_t1) restricted to
TUMOR FRACTION > 0.03 samples (tumor_fraction from results/rcv{1,2}_cna/cna_feature_table.tsv), and v1/v2
shown SIDE BY SIDE in a single figure with a shared TF row order.
Out: results/plot/Tab_perTF_enrichment_padding_rc.tsv + Fig_perTF_enrichment_rc.png"""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import build_conc_metrics_bench as B
from per_cancer_contrib import site_meta, prep_shared, col_auroc, DELTA, WINKEY
from rc_cohort import rc_table

WIN = "padding"; MIN_TF_SITES = 20; TOPN_FIG = 25
TF_MIN = 0.03            # tumor-fraction gate (user directive 2026-07-18)
MIN_CLASS = 10           # after the tf>0.03 gate the cohort shrinks; keep types with >=10 samples
PROJ = "/home/jrkim/TSO_TFBS/project"
CNA = {"v1": f"{PROJ}/results/rcv1_cna/cna_feature_table.tsv",
       "v2": f"{PROJ}/results/rcv2_cna/cna_feature_table.tsv"}


def have_cvp():
    return set(os.path.basename(p)[:-len(".cvp.npz")] for p in glob.glob(f"{B.NPZD}/*.cvp.npz"))


def rc_cohort_tf(cohort, have):
    """Rule-conformant sids with tumor_fraction > TF_MIN and a per-site cvp npz; class>=MIN_CLASS."""
    rc = rc_table(cohort)
    tf = pd.read_csv(CNA[cohort], sep="\t", index_col=0)["tumor_fraction"]; tf.index = tf.index.astype(str)
    rc["tf"] = rc.num_sid.map(tf)
    rc = rc[(rc.tf > TF_MIN) & rc.num_sid.isin(have)].copy()
    cnt = rc.cancer.value_counts(); classes = sorted([c for c in cnt.index if cnt[c] >= MIN_CLASS])
    rc = rc[rc.cancer.isin(classes)].reset_index(drop=True)
    sids = rc.num_sid.tolist(); y = np.array([classes.index(c) for c in rc.cancer])
    return sids, y, classes


def main():
    mask = B.unique_mask(); have = have_cvp()
    chrom, pos, tf_all, edge = site_meta(mask)
    rows = []; nsamp = {}
    for cohort in ("v1", "v2"):
        sids, y, classes = rc_cohort_tf(cohort, have); K = len(classes); nsamp[cohort] = len(sids)
        G, L, Xr, keep = prep_shared(sids, mask, WINKEY[WIN])
        tf = tf_all[keep]
        aucG = np.stack([col_auroc(G, y == k) for k in range(K)])   # (K,P)
        aucL = np.stack([col_auroc(L, y == k) for k in range(K)])
        aucX = np.stack([col_auroc(Xr, y == k) for k in range(K)])  # interaction channel
        infoG = np.abs(aucG - 0.5) >= DELTA; infoL = np.abs(aucL - 0.5) >= DELTA
        infoX = np.abs(aucX - 0.5) >= DELTA
        infoAny = infoG | infoL
        ampG = np.abs(aucG - 0.5); ampL = np.abs(aucL - 0.5); ampX = np.abs(aucX - 0.5)
        tfs = pd.Series(tf); counts = tfs.value_counts()
        keep_tfs = counts[counts >= MIN_TF_SITES].index.tolist()
        print(f"[{cohort}] RC tf>{TF_MIN} n={len(sids)} K={K} classes={classes} sites={G.shape[1]} "
              f"TFs(>= {MIN_TF_SITES} sites)={len(keep_tfs)}", flush=True)
        for t in keep_tfs:
            col = np.where(tf == t)[0]; nt = len(col)
            for k in range(K):
                rows.append(dict(cohort=cohort, cancer_type=classes[k], TF=t, n_sites=nt,
                    frac_gini=round(float(infoG[k, col].mean()), 4),
                    frac_lenent=round(float(infoL[k, col].mean()), 4),
                    frac_inter=round(float(infoX[k, col].mean()), 4),
                    frac_any=round(float(infoAny[k, col].mean()), 4),
                    mean_ampG=round(float(ampG[k, col].mean()), 4),
                    mean_ampL=round(float(ampL[k, col].mean()), 4),
                    mean_ampX=round(float(ampX[k, col].mean()), 4)))
    df = pd.DataFrame(rows)
    # cancer-specificity per channel: frac(k,t) minus mean over cancers of the same TF (within cohort)
    for ch in ("gini", "lenent", "inter", "any"):
        df[f"spec_{ch}"] = df[f"frac_{ch}"] - df.groupby(["cohort", "TF"])[f"frac_{ch}"].transform("mean")
    df = df.round(4)
    df.to_csv(f"{B.PLOT}/Tab_perTF_enrichment_{WIN}_rc.tsv", sep="\t", index=False)

    # ---- side-by-side figure: shared TF rows, v1 | v2 heatmaps of informative fraction ----
    disc = {c: df[df.cohort == c].groupby("TF")["spec_any"].max() for c in ("v1", "v2")}
    common = set(disc["v1"].index) & set(disc["v2"].index)
    comb = pd.Series({t: float(disc["v1"][t] + disc["v2"][t]) for t in common}).sort_values(ascending=False)
    tfsel = comb.head(TOPN_FIG).index.tolist()                     # shared row order, most discriminating first
    pivs = {}
    for c in ("v1", "v2"):
        d = df[(df.cohort == c) & (df.TF.isin(tfsel))]
        piv = d.pivot_table(index="TF", columns="cancer_type", values="frac_any").reindex(tfsel)
        pivs[c] = piv
    vmax = max(np.nanmax(pivs["v1"].values), np.nanmax(pivs["v2"].values))
    nc = {c: pivs[c].shape[1] for c in ("v1", "v2")}
    fig = plt.figure(figsize=(2.6 + 0.62 * (nc["v1"] + nc["v2"]), 0.42 * len(tfsel) + 2.4))
    gs = GridSpec(1, 3, width_ratios=[nc["v1"], nc["v2"], 0.5], wspace=0.12)
    axes = {}
    for i, c in enumerate(("v1", "v2")):
        ax = fig.add_subplot(gs[0, i]); axes[c] = ax
        im = ax.imshow(pivs[c].values, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set_xticks(range(nc[c])); ax.set_xticklabels(pivs[c].columns, rotation=40, ha="right", fontsize=12)
        ax.set_title(c, fontsize=17, fontweight="bold", pad=8)
        if i == 0:
            ax.set_yticks(range(len(tfsel))); ax.set_yticklabels(tfsel, fontsize=12)
        else:
            ax.set_yticks(range(len(tfsel))); ax.set_yticklabels([])
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax, fraction=1.0)
    cb.set_label("Fraction of TF sites informative", fontsize=14); cb.ax.tick_params(labelsize=11)
    fig.suptitle("Lineage-TF enrichment in SHAPE features across cancer types\n"
                 f"(rule-conformant cohort, tumor fraction > {TF_MIN}: v1 n={nsamp['v1']}, v2 n={nsamp['v2']}; "
                 f"top {len(tfsel)} discriminating TFs)", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.16, right=0.9, top=0.86, bottom=0.16)
    fig.savefig(f"{B.PLOT}/Fig_perTF_enrichment_rc.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> Tab_perTF_enrichment_{WIN}_rc.tsv + Fig_perTF_enrichment_rc.png\nPERTF_ENRICH_DONE", flush=True)


if __name__ == "__main__":
    main()
