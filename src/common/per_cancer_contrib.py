#!/usr/bin/env python
"""Per-cancer-type contribution analysis of the gini+lenent+interaction fusion, at the SAME TFBS.
For each cancer type k and TFBS j we ask whether the endpoint-concentration (gini), fragment-length
(lenent) and their per-site interaction (gini x lenent) contribute DIFFERENTIALLY or CO-contribute.

Cohort = manuscript blood_B01 ∩ 5-module ∩ class>=20 (n=1034 v1 / 708 v2), 17,178 unique TFBS.

Two complementary lenses (collinearity of the interaction term makes coefficient magnitude alone
unreliable, so we pair a model-free lens with a stability-selected model lens):
  A) MODEL-FREE per-site univariate one-vs-rest AUROC of gini_j, lenent_j, (gini*lenent)_j for class k.
       co-contribution : both gini_j and lenent_j individually informative
       differential    : only one of them
       synergy/interaction : the product beats BOTH parents (super-additive)
  B) STABILITY-SELECTED elastic-net (saga, l1_ratio/C matched to benchmark) refit on B subsamples;
       per (class, site, block) non-zero selection frequency -> which block the model keeps at site j.

Usage: per_cancer_contrib.py <center|padding> [both]
Out: results/plot/Tab_contrib_site_<win>.tsv, Tab_contrib_summary_<win>.tsv, Fig_contrib_<win>.png
"""
import os, sys, glob, itertools, numpy as np, pandas as pd
os.environ.setdefault("MCT_XGB_DEVICE", "cpu")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
sys.path.insert(0, "/home/jrkim/TSO_TFBS/project/scripts/auto/nc_readiness")
import build_conc_metrics_bench as B
from bench_ctr_vs_pad_cohortA import cohort_sids, load_n2tso

PLOT = B.PLOT; NPZD = B.NPZD
REG = f"{B.SCR}/centers_edge0_regions.tsv"
SEED = 42
DELTA = 0.08          # |AUROC-0.5| threshold for "informative"
SYN_MARGIN = 0.02     # interaction must beat both parents by this to be called synergy
NSUB = 20             # stability-selection refits
HP = dict(C=1.0, l1=0.5)   # middle of benchmark grid C in {.3,1,3}, l1 in {.2,.5,.8}
WINKEY = {"center": "g_ctr", "padding": "g_pad"}


def site_meta(mask):
    rows = [ln.rstrip("\n").split("\t") for ln in open(REG)]
    rows = [rows[i] for i in np.where(mask)[0]]
    chrom = np.array([r[0] for r in rows]); pos = np.array([(int(r[1]) + int(r[2])) // 2 for r in rows])
    tf = np.array([r[4] for r in rows]); edge = np.array([int(r[5]) for r in rows])
    return chrom, pos, tf, edge


def prep_shared(sids, mask, gininame):
    """Load raw gini/lenent/interaction, one SHARED keep-mask (<=50% missing in all), col-median impute, robust-z."""
    G = B.load_raw(sids, mask, lambda z: z[gininame])
    L = B.load_raw(sids, mask, lambda z: z["lenent"])
    Xr = G * L
    keep = (np.isnan(G).mean(0) <= 0.5) & (np.isnan(L).mean(0) <= 0.5) & (np.isnan(Xr).mean(0) <= 0.5)
    def rz(M):
        M = M[:, keep]
        med = np.nanmedian(M, 0); med = np.where(np.isfinite(med), med, 0.0)
        ii = np.where(np.isnan(M)); M[ii] = np.take(med, ii[1])
        m = np.median(M, 1, keepdims=True)
        q1 = np.percentile(M, 25, 1, keepdims=True); q3 = np.percentile(M, 75, 1, keepdims=True)
        iqr = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
        return ((M - m) / iqr).astype(np.float32)
    return rz(G), rz(L), rz(Xr), keep


def col_auroc(F, ypos):
    """Vectorized one-vs-rest AUROC per column of F (n x p) for boolean ypos."""
    n, p = F.shape; npos = int(ypos.sum()); nneg = n - npos
    if npos == 0 or nneg == 0: return np.full(p, 0.5, np.float32)
    ranks = np.empty_like(F, dtype=np.float64)
    order = np.argsort(F, axis=0, kind="mergesort")
    ar = np.arange(1, n + 1, dtype=np.float64)
    for j in range(p):                                  # ordinal ranks per column
        ranks[order[:, j], j] = ar
    rpos = ranks[ypos].sum(0)
    return ((rpos - npos * (npos + 1) / 2.0) / (npos * nneg)).astype(np.float32)


def stability_select(design, y, K):
    """Non-zero selection frequency per (class, feature) across NSUB balanced subsamples."""
    n = len(y); rng = np.random.RandomState(SEED)
    def one(seed):
        r = np.random.RandomState(seed); idx = r.choice(n, int(0.8 * n), replace=False)
        clf = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=HP["l1"], C=HP["C"],
                                 class_weight="balanced", max_iter=300, tol=1e-2, n_jobs=1, random_state=seed)
        clf.fit(design[idx], y[idx])
        C = clf.coef_
        if C.shape[0] == 1:  # binary safety
            C = np.vstack([-C[0], C[0]])
        return (np.abs(C) > 1e-8).astype(np.float32), C
    res = Parallel(n_jobs=min(NSUB, B.NCORE))(delayed(one)(int(s)) for s in rng.randint(0, 1 << 30, NSUB))
    freq = np.mean([r[0] for r in res], 0)              # (K, 3P)
    coef = np.mean([r[1] for r in res], 0)              # (K, 3P)
    return freq, coef


def run_window(win, cohorts):
    gininame = WINKEY[win]; mask = B.unique_mask(); n2tso = load_n2tso()
    chrom, pos, tf, edge = site_meta(mask)
    site_rows = []; summ_rows = []
    for cohort in cohorts:
        sids, y, classes = cohort_sids(cohort, n2tso); K = len(classes)
        G, L, Xr, keep = prep_shared(sids, mask, gininame)
        cm, cp, ct, ce = chrom[keep], pos[keep], tf[keep], edge[keep]; P = G.shape[1]
        print(f"[{win}/{cohort}] n={len(sids)} K={K} sites={P}", flush=True)
        # ---- lens A: model-free per-site univariate AUROC ----
        aucG = np.stack([col_auroc(G, y == k) for k in range(K)])   # (K,P)
        aucL = np.stack([col_auroc(L, y == k) for k in range(K)])
        aucX = np.stack([col_auroc(Xr, y == k) for k in range(K)])
        # ---- lens B: stability-selected coefficients on [G|L|X] ----
        design = np.hstack([G, L, Xr]).astype(np.float32)
        freq, coef = stability_select(design, y, K)                 # (K,3P)
        fG, fL, fX = freq[:, :P], freq[:, P:2 * P], freq[:, 2 * P:]
        for k in range(K):
            dG = np.abs(aucG[k] - 0.5); dL = np.abs(aucL[k] - 0.5); dX = np.abs(aucX[k] - 0.5)
            infoG = dG >= DELTA; infoL = dL >= DELTA
            cat = np.full(P, "none", dtype=object)
            cat[infoG & ~infoL] = "gini_only"
            cat[~infoG & infoL] = "lenent_only"
            cat[infoG & infoL] = "co"
            syn = (dX >= DELTA) & (dX >= dG + SYN_MARGIN) & (dX >= dL + SYN_MARGIN)
            cat[syn] = "synergy"
            info = cat != "none"
            summ_rows.append(dict(cohort=cohort, window=win, cancer_type=classes[k],
                n_info=int(info.sum()), n_co=int((cat == "co").sum()),
                n_gini_only=int((cat == "gini_only").sum()), n_lenent_only=int((cat == "lenent_only").sum()),
                n_synergy=int((cat == "synergy").sum()),
                top_TFs=";".join(pd.Series(ct[info]).value_counts().head(6).index.tolist())))
            imp = np.maximum.reduce([dG, dL, dX])
            topj = np.argsort(-imp)[:60]
            for j in topj:
                if not info[j]: continue
                site_rows.append(dict(cohort=cohort, window=win, cancer_type=classes[k], TF=ct[j],
                    chrom=cm[j], pos=int(cp[j]), edge=int(ce[j]), category=cat[j],
                    aucG=round(float(aucG[k, j]), 4), aucL=round(float(aucL[k, j]), 4),
                    aucX=round(float(aucX[k, j]), 4),
                    selfreq_gini=round(float(fG[k, j]), 3), selfreq_lenent=round(float(fL[k, j]), 3),
                    selfreq_inter=round(float(fX[k, j]), 3)))
        pd.DataFrame(summ_rows).to_csv(f"{PLOT}/Tab_contrib_summary_{win}.tsv", sep="\t", index=False)
        pd.DataFrame(site_rows).to_csv(f"{PLOT}/Tab_contrib_site_{win}.tsv", sep="\t", index=False)
    # ---- figure: per-class category composition (last cohort loop keeps all) ----
    S = pd.DataFrame(summ_rows)
    for cohort in cohorts:
        sc = S[S.cohort == cohort]
        if not len(sc): continue
        cats = ["n_co", "n_gini_only", "n_lenent_only", "n_synergy"]
        cols = {"n_co": "#4C72B0", "n_gini_only": "#DD8452", "n_lenent_only": "#55A868", "n_synergy": "#C44E52"}
        fig, ax = plt.subplots(figsize=(10, 4.5)); x = np.arange(len(sc)); bot = np.zeros(len(sc))
        tot = sc[cats].sum(1).replace(0, 1).values
        for c in cats:
            frac = sc[c].values / tot
            ax.bar(x, frac, bottom=bot, color=cols[c], label=c.replace("n_", ""))
            bot += frac
        ax.set_xticks(x); ax.set_xticklabels(sc.cancer_type, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("fraction of informative TFBS"); ax.set_ylim(0, 1)
        ax.legend(ncol=4, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01))
        ax.set_title(f"{win} ±40 — per-cancer gini/lenent/interaction contribution mode ({cohort})")
        fig.tight_layout(); fig.savefig(f"{PLOT}/Fig_contrib_{win}_{cohort}.png", dpi=160); plt.close(fig)
    print(f"CONTRIB_DONE {win} -> Tab_contrib_summary_{win}.tsv / Tab_contrib_site_{win}.tsv", flush=True)


if __name__ == "__main__":
    win = sys.argv[1] if len(sys.argv) > 1 else "center"
    wins = ["center", "padding"] if (len(sys.argv) > 2 and sys.argv[2] == "both") else [win]
    for w in wins:
        run_window(w, ("v1", "v2"))
    print("ALL_CONTRIB_DONE", flush=True)
