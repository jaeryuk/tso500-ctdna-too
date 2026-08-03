#!/usr/bin/env python
"""PILLAR 4 (revised) — TFBS attribution vs cancer lineage biology using the EXTERNAL CaCTS master-TF catalog
(sciadv.abf6123 Table S6) instead of a hand-curated list, with multiple-testing correction, restricted to
samples with ichorCNA tumor fraction > 10% (reduces haematopoietic dilution).
Per cancer: |OVR-AUROC-0.5| per-TF association on the high-TF subset; Mann-Whitney enrichment of CaCTS master TFs
vs background; BH-FDR + Bonferroni across all cohort x cancer tests.
Out: tables/Tab_P4cacts_highTF_enrichment.tsv + figure."""
import sys, os; sys.path.insert(0, "/home/jrkim/TSO_TFBS/project/scripts/auto"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import manuscript_figures as MF
from ncr_common import COH, FG, TB, perTF_from_sites

XLSX = "/home/jrkim/TSO_TFBS/project/results/sciadv.abf6123_tables_s1_to_s14.xlsx"
TF_MIN = 0.10        # restrict to ichorCNA tumor fraction > 10%
MIN_N = 10           # min samples per cancer in the high-TF subset to compute OVR-AUROC
TCGA = {"lung cancer": ["LUAD", "LUSC"], "colorectal cancer": ["COAD", "READ"], "gastric cancer": ["STAD"],
        "pancreatic cancer": ["PAAD"], "biliary tract cancer": ["CHOL"], "melanoma": ["SKCM"],
        "liver cancer": ["LIHC"], "prostate cancer": ["PRAD"], "breast cancer": ["BRCA"],
        "bladder cancer": ["BLCA"], "sarcoma": ["SARC"]}


def cacts_catalog():
    df = pd.read_excel(XLSX, "Table S6", header=2).dropna(subset=["Tumor Type", "Candidate MTF"])
    df["TF"] = df["Candidate MTF"].astype(str).str.upper(); df["TT"] = df["Tumor Type"].astype(str).str.upper()
    out = {}
    for cls, tts in TCGA.items():
        out[cls] = set(df[df.TT.isin(tts)]["TF"])
    return out


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); ranked = p[order]
    bh = ranked * n / (np.arange(n) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(bh, 0, 1); return out


def main():
    R, PC, Z, SID, LAB, TF = MF.load()
    cacts = cacts_catalog()
    rows = []
    IMP = {}
    for c in COH:
        X, tfs, sids, y, classes = perTF_from_sites(c)
        tfs_u = [t.upper() for t in tfs]
        tfu_idx = {t: i for i, t in enumerate(tfs_u)}
        ytype = np.array([classes[i] for i in y])
        tfrac = np.array([TF.get(s, np.nan) for s in sids])
        hi = tfrac > TF_MIN
        Xh = X[hi]; yth = ytype[hi]
        Xz = (Xh - Xh.mean(0)) / (Xh.std(0) + 1e-9)
        cohort_imp = {}
        for cls in TCGA:
            m = yth == cls; n = int(m.sum())
            if n < MIN_N or (~m).sum() < MIN_N: continue
            assoc = np.array([abs(roc_auc_score(m.astype(int), Xz[:, j]) - 0.5) for j in range(Xz.shape[1])])
            cohort_imp[cls] = pd.Series(assoc, index=tfs_u)
            mtf = sorted(cacts[cls] & set(tfs_u))
            if len(mtf) < 2:
                rows.append(dict(cohort=c, cancer=cls, n_highTF=n, n_cacts_in_panel=len(mtf),
                                 cacts_median_rank_pct=np.nan, p_raw=np.nan, cacts_in_top10="—",
                                 cacts_TFs=",".join(mtf) if mtf else "—")); continue
            s = pd.Series(assoc, index=tfs_u); ranks = s.rank(pct=True)
            cur = s[mtf]; bg = s[[t for t in tfs_u if t not in mtf]]
            U, p = mannwhitneyu(cur, bg, alternative="greater")
            top10 = list(s.sort_values(ascending=False).head(10).index)
            rows.append(dict(cohort=c, cancer=cls, n_highTF=n, n_cacts_in_panel=len(mtf),
                             cacts_median_rank_pct=round(float(ranks[mtf].median()), 3),
                             p_raw=p, cacts_in_top10=",".join([t for t in mtf if t in top10]) or "—",
                             cacts_TFs=",".join(mtf)))
        IMP[c] = cohort_imp
    T = pd.DataFrame(rows)
    # multiple-testing correction across all computable tests
    has_p = T.p_raw.notna()
    T.loc[has_p, "p_BH"] = bh_fdr(T.loc[has_p, "p_raw"].values)
    T.loc[has_p, "p_bonferroni"] = np.clip(T.loc[has_p, "p_raw"].values * has_p.sum(), 0, 1)
    T["sig_BH_0.1"] = (T["p_BH"] <= 0.10)
    for col in ("p_raw", "p_BH", "p_bonferroni"):
        T[col] = T[col].map(lambda x: f"{x:.3g}" if pd.notna(x) else "NA")
    T = T[["cohort", "cancer", "n_highTF", "n_cacts_in_panel", "cacts_TFs", "cacts_median_rank_pct",
           "p_raw", "p_BH", "p_bonferroni", "sig_BH_0.1", "cacts_in_top10"]]
    T.to_csv(f"{TB}/Tab_P4cacts_highTF_enrichment.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200); print(T.to_string(index=False))
    print(f"\n[P4cacts] tests={int(has_p.sum())}  BH-sig(<=0.1)={int(T['sig_BH_0.1'].sum())}  "
          f"-> {TB}/Tab_P4cacts_highTF_enrichment.tsv")

    # figure: -log10(p_raw) per cohort x cancer with BH threshold reference
    plt.figure(figsize=(12, 5.5))
    Tp = pd.DataFrame(rows); Tp = Tp[Tp.p_raw.notna()].copy()
    Tp["nl"] = -np.log10(Tp.p_raw.astype(float)); Tp["lab"] = Tp.cohort.str.upper() + " " + Tp.cancer.str.replace(" cancer", "")
    Tp = Tp.sort_values("nl", ascending=True)
    colors = ["#1b9e77" if s else "#bbbbbb" for s in (bh_fdr(Tp.p_raw.astype(float).values) <= 0.10)]
    plt.barh(range(len(Tp)), Tp.nl, color=colors)
    plt.yticks(range(len(Tp)), Tp.lab, fontsize=8)
    plt.axvline(-np.log10(0.05), color="k", ls=":", lw=0.8, label="p=0.05 (raw)")
    plt.xlabel("-log10 enrichment p (CaCTS master TFs vs background)")
    plt.title(f"Pillar 4 (CaCTS, TF>10%, FDR): lineage-TF enrichment\ngreen = BH-FDR≤0.1", fontweight="bold", fontsize=11)
    plt.legend(frameon=False, fontsize=8); plt.tight_layout()
    plt.savefig(f"{FG}/Fig_P4cacts_enrichment.png", dpi=150, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
