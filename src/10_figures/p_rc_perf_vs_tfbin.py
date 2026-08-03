#!/usr/bin/env python3
"""RC-cohort macro-AUROC vs tumour-fraction bin, v1 & v2 stacked (shared axes). Plot only.
Reads Tab_rc_perf_vs_tfbin.tsv (built by rc_perf_vs_tfbin.py from the frozen nested-CV OOF).
Styling deliberately mirrors the old keep_t1 Fig_perf_vs_tfbin_macroAUROC.png so the two are comparable.
Out: results/plot/Fig_rc_perf_vs_tfbin_macroAUROC.png (+ .pdf)
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TB = "/home/jrkim/TSO_TFBS/project/results/auto_plan/nc_readiness/tables"
PLOT = "/home/jrkim/TSO_TFBS/project/results/plot"
TSV = f"{TB}/Tab_rc_perf_vs_tfbin.tsv"
BINLABS = ["<3%", "3-10%", "10-20%", ">20%"]
COMPS = ["Exon1 entropy", "All-exon depth", "Broad CNA (>10Mb)", "Somatic mutation", "SHAPE_nep300",
         "LATE-FUSION (all 5)"]
DISPLAY = {"Exon1 entropy": "Exon1 entropy", "All-exon depth": "All-exon depth",
           "Broad CNA (>10Mb)": "Broad CNA", "Somatic mutation": "Somatic mutation",
           "SHAPE_nep300": "TFBS SHAPE", "LATE-FUSION (all 5)": "Ensemble"}
COL = {"Exon1 entropy": "#4C9F70", "All-exon depth": "#E1A730", "Broad CNA (>10Mb)": "#C0392B",
       "Somatic mutation": "#8E44AD", "SHAPE_nep300": "#2C6FB2", "LATE-FUSION (all 5)": "#111111"}

plt.rcParams.update({"font.size": 14, "axes.linewidth": 1.4, "axes.edgecolor": "#222222",
                     "xtick.direction": "out", "ytick.direction": "out",
                     "font.family": "DejaVu Sans", "svg.fonttype": "none"})


def main():
    R = pd.read_csv(TSV, sep="\t")
    xr = np.arange(len(BINLABS))
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 9.2), sharex=True, sharey=True)
    for j, c in enumerate(("v1", "v2")):
        ax = axes[j]; d = R[R.cohort == c]
        nmap = {r["bin"]: int(r["n"]) for _, r in d[d.component == "SHAPE_nep300"].iterrows()}
        for name in COMPS:
            vals = [d[(d.component == name) & (d["bin"] == bl)]["macroAUROC"].values for bl in BINLABS]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ens = name.startswith("LATE")
            ax.plot(xr, vals, color=COL[name], lw=3.4 if ens else 2.0, marker="o", ms=9 if ens else 6,
                    alpha=1.0 if ens else 0.9, zorder=5 if ens else 3, label=DISPLAY[name])
        ax.grid(ls=":", alpha=0.4)
        ax.text(0.015, 0.94, c, transform=ax.transAxes, ha="left", va="top", fontsize=19, fontweight="bold")
        for bl, xi in zip(BINLABS, xr):
            ax.text(xi, 0.03, f"n={nmap.get(bl,'?')}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10.5, color="0.45")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=14)
    axes[0].set_ylim(0.55, 1.0)
    axes[1].set_xticks(xr); axes[1].set_xticklabels(BINLABS, fontsize=15)
    axes[1].set_xlabel("Tumor-fraction bin", fontsize=17)
    fig.supylabel("Macro-AUROC", fontsize=18)
    fig.subplots_adjust(top=0.94, bottom=0.14, left=0.13, right=0.985, hspace=0.10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.075))
    fig.suptitle("Tumor fraction–stratified TOO performance (RC)", y=0.985, fontsize=19, fontweight="bold")
    out = f"{PLOT}/Fig_rc_perf_vs_tfbin_macroAUROC.png"
    fig.savefig(out, dpi=160); print("WROTE", out)
    fig.savefig(out.replace(".png", ".pdf")); print("WROTE", out.replace(".png", ".pdf"))
    print("RC_TFBIN_FIG_DONE")


if __name__ == "__main__":
    main()
