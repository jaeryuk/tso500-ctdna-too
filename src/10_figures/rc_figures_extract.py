#!/usr/bin/env python3
"""Regenerate per-cohort figure tables on the RULE-CONFORMANT (keep_t1) cohort.

Membership + labels come from rc_cohort.rc_table (keep_t1, tumor_type1, resolved to numeric feature sids).
Tumor fraction from results/rcv{1,2}_cna/cna_feature_table.tsv (RC ichorCNA, covers BOTH cohorts).
(The old docstring said sample_metadata.tsv -- WRONG and dangerous: that file has TF for 0% of v2 and
only 92% of v1, so gating on it would silently empty every v2 panel. The code below is correct.)
Cohort-independent metadata (gene panels, known-event / driver-home flags) reused from the existing
pooled CSVs. Outputs (results/plot/):
  Tab_fraglen_bins_rc.tsv                       (short-fragment 35-80bp, per sample)
  e1_entropy_vs_expression_byversion.csv        (E1 entropy vs TCGA expr, per gene x cancer x cohort)
  cna_arm_concordance_byversion.csv             (arm-level CNA, high-TF, per cohort)
  mut_driver_heatmap_rc.csv                     (driver-gene OVR-logistic coef + freq, per cohort)
  depth_focal_cna_concordance_byversion.csv     (gene-level focal depth, high-TF, per cohort)
"""
import os, sys, re
import os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from rc_cohort import rc_table
from sklearn.linear_model import LogisticRegression

PROJ = "/home/jrkim/TSO_TFBS/project"
PLOT = f"{PROJ}/results/plot"
FEAT = f"{PROJ}/results/auto_plan/feat"
FEAT_RC = f"{PROJ}/results/auto_plan/feat_rc"
FB = f"{PROJ}/results/auto_plan/fbins"

# Tumor fraction: RC ichorCNA off-probe extraction, numeric-sid keyed, covers BOTH cohorts
# (sample_metadata.tsv only covers the old cohort / v1). See rcv{1,2}_cna/cna_feature_table.tsv.
TF = {}
for _c in ("rcv1", "rcv2"):
    _d = pd.read_csv(f"{PROJ}/results/{_c}_cna/cna_feature_table.tsv", sep="\t")
    _d = _d.rename(columns={_d.columns[0]: "sid"})
    for s, t in zip(_d["sid"], _d["tumor_fraction"]):
        if pd.notna(t):
            TF[str(int(s)) if str(s).replace(".", "", 1).isdigit() else str(s)] = float(t)
# Tumour-fraction gate for the two CNA concordance figures. Was >=0.05; user directive
# 2026-07-17 -> >0.03. Lowering it ADDS samples, so no RC class can drop out.
TFCUT = float(os.environ.get("TFCUT", "0.03"))
RC = {c: rc_table(c) for c in ("v1", "v2")}
for c in RC:
    print(f"[rc {c}] n={len(RC[c])} types={RC[c].cancer.nunique()}", flush=True)


def load_feat(c, name, root=FEAT):
    z = np.load(f"{root}/{c}/X_{name}.npz", allow_pickle=True)
    return z["X"].astype(float), [str(x) for x in z["cols"]], {str(s): i for i, s in enumerate(z["sids"])}


def labmap(c):
    return dict(zip(RC[c].num_sid.astype(str), RC[c].cancer))


# ---------------------------------------------------------------- 1. fraglen 35-80bp
def fraglen():
    rows = []
    for c in ("v1", "v2"):
        for s in RC[c].num_sid.astype(str):
            p = f"{FB}/{s}.fbins"
            if not os.path.exists(p):
                continue
            x = open(p).read().strip().split("\t")
            if len(x) < 7:
                continue
            n = np.array(x[2:6], float); tot = float(x[6])
            if tot < 2000:
                continue
            fr = n / tot
            rows.append((c, s, *fr, tot))
    df = pd.DataFrame(rows, columns=["cohort", "sid", "f35_80", "f80_120", "f120_180", "f180p", "total"])
    df.to_csv(f"{PLOT}/Tab_fraglen_bins_rc.tsv", sep="\t", index=False)
    print("FRAGLEN", df.groupby("cohort").size().to_dict(),
          "median f35_80%", (df.groupby("cohort").f35_80.median() * 100).round(3).to_dict(), flush=True)


# ---------------------------------------------------------------- 2. E1 entropy vs expression
def e1_expr():
    pooled = pd.read_csv(f"{PLOT}/e1_entropy_vs_expression_rna.csv")
    expr = {(r.cancer, r.gene): r.expr for r in pooled.itertuples()}
    rows = []
    for c in ("v1", "v2"):
        X, cols, pos = load_feat(c, "E1_entropy", FEAT_RC)
        lab = labmap(c)
        for cancer in sorted(set(lab.values())):
            sids = [s for s, cc in lab.items() if cc == cancer and s in pos]
            if len(sids) < 8:
                continue
            mean = np.nanmean(X[[pos[s] for s in sids]], 0)
            for gi, g in enumerate(cols):
                e = expr.get((cancer, g))
                if e is None:
                    continue
                rows.append(dict(version=c, cancer=cancer, gene=g,
                                 e1=round(float(mean[gi]), 5), expr=round(float(e), 5)))
    D = pd.DataFrame(rows)
    D["e1_c"] = D.groupby(["version", "cancer"])["e1"].transform(lambda v: v - v.mean())
    D["expr_c"] = D.groupby(["version", "cancer"])["expr"].transform(lambda v: v - v.mean())
    D.to_csv(f"{PLOT}/e1_entropy_vs_expression_byversion.csv", index=False)
    print("E1EXPR cancers/cohort", D.groupby("version").cancer.unique().apply(list).to_dict(), flush=True)


# ---------------------------------------------------------------- 3. arm-level CNA concordance
def cna_arm():
    known = pd.read_csv(f"{PLOT}/cna_arm_concordance.csv")
    kmap = {(r.cancer, r.arm): int(r.known) for r in known.itertuples()}
    rows = []
    for c in ("v1", "v2"):
        X, cols, pos = load_feat(c, "Genome_wide_CNA", FEAT_RC)   # RC-specific, 43 arms, full coverage
        arms = [a.replace("chr", "") for a in cols]
        lab = labmap(c)
        for cancer in sorted(set(lab.values())):
            sids = [s for s, cc in lab.items() if cc == cancer and s in pos]   # ALL tumor fraction (user 2026-07-21; depth_focal stays > TFCUT)
            if len(sids) < 5:
                continue
            mean = np.nanmean(X[[pos[s] for s in sids]], 0)
            for a, arm in enumerate(arms):
                rows.append(dict(version=c, cancer=cancer, arm=arm, mean_cna=round(float(mean[a]), 5),
                                 known=kmap.get((cancer, arm), 0), n=len(sids)))
    pd.DataFrame(rows).to_csv(f"{PLOT}/cna_arm_concordance_byversion.csv", index=False)
    print("CNA cancers/cohort",
          pd.DataFrame(rows).groupby("version").cancer.nunique().to_dict(), flush=True)


# ---------------------------------------------------------------- 4. mutation driver heatmap
def mut_driver():
    meta = pd.read_csv(f"{PLOT}/mut_driver_heatmap.csv")
    gmeta = meta.drop_duplicates("gene")[["gene", "home", "home_i"]]
    genes = gmeta.gene.tolist()
    isdrv = set((r.cancer, r.gene) for r in meta[meta.is_driver == 1].itertuples())
    rows = []
    for c in ("v1", "v2"):
        X, cols, pos = load_feat(c, "Somatic_mutation_profile", FEAT_RC)  # RC-specific, full coverage
        cidx = {g: cols.index(f"mut_{g}") for g in genes if f"mut_{g}" in cols}
        gg = [g for g in genes if g in cidx]
        lab = labmap(c)
        sids = [s for s in RC[c].num_sid.astype(str) if s in pos]
        y = np.array([lab[s] for s in sids])
        Mb = (np.vstack([X[pos[s]] for s in sids])[:, [cidx[g] for g in gg]] > 0).astype(float)
        cancers = sorted([cc for cc, n in pd.Series(y).value_counts().items() if n >= 20])
        for cancer in cancers:
            yb = (y == cancer).astype(int)
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Mb, yb)
            coef = clf.coef_[0]; freq = Mb[y == cancer].mean(0)
            for gi, g in enumerate(gg):
                rows.append(dict(cohort=c, cancer=cancer, gene=g, coef=round(float(coef[gi]), 3),
                                 freq_in=round(float(freq[gi]), 3),
                                 is_driver=int((cancer, g) in isdrv)))
    D = pd.DataFrame(rows).merge(gmeta, on="gene", how="left")
    D["cancer_s"] = D.cancer.str.replace(" cancer", "", regex=False).str.title()
    D.to_csv(f"{PLOT}/mut_driver_heatmap_rc.csv", index=False)
    print("MUT cancers/cohort", D.groupby("cohort").cancer.nunique().to_dict(), flush=True)


# ---------------------------------------------------------------- 5. focal (gene-level) depth concordance
def depth_focal():
    meta = pd.read_csv(f"{PLOT}/depth_focal_cna_concordance.csv")
    genes = list(dict.fromkeys(meta.gene.tolist()))
    kmap = {(r.cancer, r.gene): int(r.known) for r in meta.itertuples()}
    rows = []
    for c in ("v1", "v2"):
        X, cols, pos = load_feat(c, "All_exon_depth", FEAT_RC)   # RC-specific, full coverage
        gcols = {}
        for i, cn in enumerate(cols):
            m = re.match(r"exon:(.+?)_Exon", cn)
            if m:
                gcols.setdefault(m.group(1), []).append(i)
        lab = labmap(c)
        sids = [s for s in RC[c].num_sid.astype(str) if s in pos
                and np.isfinite(TF.get(s, np.nan)) and TF.get(s, 0) > TFCUT]
        y = np.array([lab[s] for s in sids])
        Xs = np.vstack([X[pos[s]] for s in sids])
        smed = np.nanmedian(Xs, axis=1, keepdims=True)
        gmat = np.full((len(sids), len(genes)), np.nan)
        for gi, g in enumerate(genes):
            idx = gcols.get(g)
            if idx:
                gmat[:, gi] = np.log2((np.nanmean(Xs[:, idx], axis=1) + 1) / (smed[:, 0] + 1))
        cancers = sorted([cc for cc, n in pd.Series(y).value_counts().items() if n >= 5])
        rawM = np.vstack([np.nanmean(gmat[y == cc], axis=0) for cc in cancers])
        gc = np.nanmean(rawM, axis=0)
        for ci, cancer in enumerate(cancers):
            for gi, g in enumerate(genes):
                rows.append(dict(version=c, cancer=cancer, gene=g,
                                 focal_raw=round(float(rawM[ci, gi]), 5),
                                 focal=round(float(rawM[ci, gi] - gc[gi]), 5),
                                 known=kmap.get((cancer, g), 0)))
    pd.DataFrame(rows).to_csv(f"{PLOT}/depth_focal_cna_concordance_byversion.csv", index=False)
    print("DEPTHFOCAL cancers/cohort",
          pd.DataFrame(rows).groupby("version").cancer.nunique().to_dict(), flush=True)


if __name__ == "__main__":
    fraglen()
    e1_expr()
    cna_arm()
    mut_driver()
    depth_focal()
    print("RC_FIGURES_EXTRACT_DONE", flush=True)
