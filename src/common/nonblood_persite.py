#!/usr/bin/env python
"""
Per-TFBS (per-site) SHAPE on NON-BLOOD UniBind sites — instead of per-TF aggregation.
Idea: cfDNA is dominated by haematopoietic fragments, so blood-cell TFBS footprints are background; tissue
(non-blood) TFBS footprints carry the tumour tissue-of-origin signal. Use each individual TFBS as its own
feature (per-site short-fraction), restricted to non-blood UniBind centres.

Per-site feature = SF_site = n_short/(n_short+n_long) − sample-global short-fraction  (±30bp; CN-free/PoN-free),
support-gated (≥MIN per site), one column per UNIQUE TFBS centre, within-sample robust-z across that site set.
UniBind centre annotation (intermediate/blood_tf/*.centers.tsv) tags each SHAPE centre as blood / non-blood.

Site sets compared (all per-site), against the per-TF SHAPE baseline and the depth backbone:
  persite_nonblood       : non-blood UniBind centres                        (the proposal)
  persite_nonblood_excl  : non-blood AND NOT blood (specificity-strict)
  persite_blood          : blood UniBind centres                            (negative control = background)
  persite_all            : all unique TFBS centres                          (per-site without selection)
Leakage-free within-chemistry OOF (version-aware). Outputs -> results/auto_plan/primary/nonblood/ + fig.
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
PROJ = "/home/jrkim/TSO_TFBS/project"
sys.path.insert(0, f"{PROJ}/scripts/auto"); sys.path.insert(0, f"{PROJ}/scripts")
import primary as P
import chem_separated as CSx
import v2_frag_shape as V
OUT = f"{PROJ}/results/auto_plan/primary/nonblood"; os.makedirs(OUT, exist_ok=True)
FIG = f"{PROJ}/results/auto_plan/report"
BT = f"{PROJ}/intermediate/blood_tf"
AMP = {"v1": f"{PROJ}/results/v1_fragshape/_amp", "v2": f"{PROJ}/results/v2_fragshape/_amp"}
SEED = 42; EPS = 1e-15; REPS = int(os.environ.get("PS_REPS", 4)); MIN_SUPP = 30
INCLUDE_ALL = os.environ.get("PS_ALL", "1") == "1"
rng = np.random.default_rng(SEED)
def log(m): print(m, flush=True)


def site_geom():
    g, tkeys, tf, tgt = V.site_table()
    chrom = []; cen = []
    for k in tkeys:
        c0, ts, te = k
        for r in g[k]["f"]: chrom.append(c0); cen.append(int(r["center"]))
    return np.array(chrom), np.array(cen, np.int64)

def load_centers(path):
    s = set()
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            try: s.add((p[0], int(p[1])))
            except ValueError: pass
    return s

def rep_rows_for(centerset, all_rep):
    return np.array([all_rep[c] for c in centerset if c in all_rep], np.int64)

def robustz_rows(X):
    med = np.nanmedian(X, 1, keepdims=True); mad = np.nanmedian(np.abs(X - med), 1, keepdims=True) * 1.4826
    sd = np.nanstd(X, 1, keepdims=True); mad[mad == 0] = sd[mad == 0] + 1e-9
    return (X - med) / mad

def finalize_cols(X, frac=0.5):
    keep = np.isnan(X).mean(0) < frac
    X = X[:, keep]; cm = np.nanmedian(X, 0); ii = np.where(np.isnan(X)); X[ii] = np.take(cm, ii[1])
    return np.nan_to_num(X), keep

def macro_auc(Pm, y, K):
    a = [roc_auc_score((y == k).astype(int), Pm[:, k]) for k in range(K) if 0 < (y == k).sum() < len(y)]
    return float(np.mean(a)) if a else np.nan
def per_class_auc(Pm, y, K):
    return np.array([roc_auc_score((y == k).astype(int), Pm[:, k]) if 0 < (y == k).sum() < len(y) else np.nan
                     for k in range(K)])
def topk(Pm, y, k): o = np.argsort(-Pm, 1)[:, :k]; return float(np.mean([y[i] in o[i] for i in range(len(y))]))
def topk_b(Pm, y, k): o = np.argsort(-Pm, 1)[:, :k]; return np.array([y[i] in o[i] for i in range(len(y))])
def tprob(Pm, y): return Pm[np.arange(len(y)), y]
def trank(Pm, y): tp = Pm[np.arange(len(y)), y][:, None]; return 1 + (Pm > tp).sum(1)
def perm_p(v, B=10000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3 or np.allclose(v, 0): return 1.0
    obs = abs(np.mean(v)); signs = rng.integers(0, 2, (B, len(v))) * 2 - 1
    return float((np.sum(np.abs((signs * v).mean(1)) >= obs) + 1) / (B + 1))


def main():
    P.METHODS = ["All_exon_depth", "SHAPE"]
    man, M, y, cohort, classes, ci = P.load_all(); K = len(classes)
    short = [c.replace(" cancer", "").replace("biliary tract", "biliary") for c in classes]
    chrom, cen = site_geom()
    val0 = np.load(sorted(glob.glob(f"{AMP['v2']}/*.npz"))[0])["valid"]
    all_rep = {}                                                       # unique centre -> first valid row
    for i in np.where(val0)[0]:
        ck = (chrom[i], int(cen[i]))
        if ck not in all_rep: all_rep[ck] = i

    nb = load_centers(f"{BT}/nonblood_feature.centers.tsv"); bl = load_centers(f"{BT}/blood_feature.centers.tsv")
    SETS = {"persite_nonblood": rep_rows_for(nb, all_rep),
            "persite_nonblood_excl": rep_rows_for(nb - bl, all_rep),
            "persite_blood": rep_rows_for(bl, all_rep)}
    if INCLUDE_ALL: SETS["persite_all"] = np.array(sorted(all_rep.values()), np.int64)
    log("[nb] site-set sizes: " + " ".join(f"{k}={len(v)}" for k, v in SETS.items()))
    union = np.array(sorted(set(np.concatenate(list(SETS.values())).tolist())), np.int64)
    colpos = {r: j for j, r in enumerate(union)}                       # row idx -> column in X_union

    # ---- compute per-site SF at union rows for every sample ----
    Xu = np.full((len(y), len(union)), np.nan, np.float32)
    for ridx in range(len(y)):
        cn = "v2" if cohort[ridx] == 1 else "v1"; sid = str(man.sid.values[ridx])
        p = f"{AMP[cn]}/{sid}.npz"
        if not os.path.exists(p): continue
        z = np.load(p); ns = z["n_short"].astype(np.float64); nl = z["n_long"].astype(np.float64)
        tot = ns + nl; sfg = float(z["tot_short"][0]) / max(float(z["tot_short"][0]) + float(z["tot_long"][0]), 1)
        sf = np.where(tot >= MIN_SUPP, ns / np.maximum(tot, 1) - sfg, np.nan)
        Xu[ridx] = sf[union]
        if (ridx + 1) % 300 == 0: log(f"[nb] persite SF {ridx+1}/{len(y)}")
    log("[nb] per-site SF computed")

    # ---- build per-set matrices (slice union, finalize cols, within-sample robust-z) ----
    XSET = {}
    for nm, rows in SETS.items():
        cols = [colpos[r] for r in rows]
        Xs, keep = finalize_cols(Xu[:, cols].copy())
        Xs = np.nan_to_num(robustz_rows(Xs)).astype(np.float32)
        XSET[nm] = Xs
        log(f"[nb] {nm}: {Xs.shape[1]} sites kept")

    # ---- OOF: depth + per-TF SHAPE + each per-site set, within-chemistry ----
    OOF = {"v1": {}, "v2": {}}
    for c, cn in [(0, "v1"), (1, "v2")]:
        rows = np.where(cohort == c)[0]
        OOF[cn]["rows"] = rows; OOF[cn]["y"] = y[rows]
        OOF[cn]["depth"] = P.base_oof(M["All_exon_depth"], y, cohort, K, rows, reps=REPS)
        OOF[cn]["SHAPE_perTF"] = P.base_oof(M["SHAPE"], y, cohort, K, rows, reps=REPS)
        for nm in SETS: OOF[cn][nm] = P.base_oof(XSET[nm], y, cohort, K, rows, reps=REPS)
        log(f"[nb] {cn}: OOF built ({len(rows)} samples)")

    MODELS = ["SHAPE_perTF"] + list(SETS.keys())
    def pooled(key):
        Pp = np.zeros((len(y), K))
        for cn, c in [("v1", 0), ("v2", 1)]: Pp[OOF[cn]["rows"]] = OOF[cn][key]
        return Pp

    # (A) standalone
    rowsA = []; pc = {}
    for scope in ["v1", "v2", "pooled"]:
        yy = OOF[scope]["y"] if scope != "pooled" else y
        for key in MODELS:
            Pm = OOF[scope][key] if scope != "pooled" else pooled(key)
            rowsA.append(dict(scope=scope, model=key, n_sites=(XSET[key].shape[1] if key in XSET else "perTF"),
                              macro_AUROC=round(macro_auc(Pm, yy, K), 4),
                              top1=round(topk(Pm, yy, 1), 4), top3=round(topk(Pm, yy, 3), 4)))
            if scope == "pooled": pc[key] = per_class_auc(Pm, yy, K)
    tA = pd.DataFrame(rowsA); tA.to_csv(f"{OUT}/table_standalone.tsv", sep="\t", index=False)
    tPC = pd.DataFrame({"cancer": short, **{k: np.round(pc[k], 4) for k in MODELS}})
    tPC.to_csv(f"{OUT}/table_perclass_pooled.tsv", sep="\t", index=False)
    log("[nb] (A) standalone:\n" + tA.to_string(index=False))
    log("[nb] (A) per-class pooled:\n" + tPC.to_string(index=False))

    # (B) depth add-on
    rowsB = []
    for cn in ["v1", "v2"]:
        yc = OOF[cn]["y"]; Pd = OOF[cn]["depth"]; a_d = macro_auc(Pd, yc, K)
        for key in MODELS:
            Pf, _, _ = CSx.fuse_subset({"All_exon_depth": Pd, "S": OOF[cn][key]}, yc, K)
            dll = (-np.log(np.clip(tprob(Pd, yc), EPS, 1))) - (-np.log(np.clip(tprob(Pf, yc), EPS, 1)))
            dt3 = topk_b(Pf, yc, 3).astype(int) - topk_b(Pd, yc, 3).astype(int)
            resc = int(((~topk_b(Pd, yc, 3)) & topk_b(Pf, yc, 3)).sum())
            harm = int((topk_b(Pd, yc, 3) & (~topk_b(Pf, yc, 3))).sum())
            rowsB.append(dict(chemistry=cn, model=f"depth+{key}",
                              standalone_macroAUROC=round(macro_auc(OOF[cn][key], yc, K), 4),
                              delta_macro_vs_depth=round(macro_auc(Pf, yc, K) - a_d, 4),
                              mean_Dlogloss=round(float(np.mean(dll)), 4), Dtop3=round(float(dt3.mean()), 4),
                              net_rescue3=resc - harm, perm_p=round(perm_p(dll), 4)))
    tB = pd.DataFrame(rowsB); tB.to_csv(f"{OUT}/table_depth_addon.tsv", sep="\t", index=False)
    log("[nb] (B) depth add-on:\n" + tB.to_string(index=False))

    make_fig(tA, tPC, short, MODELS); write_summary(tA, tPC, tB, SETS)
    log(f"[nb] DONE -> {OUT}")


def make_fig(tA, tPC, short, MODELS):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    a = ax[0]; scopes = ["v1", "v2", "pooled"]; w = 0.15
    cols = {"SHAPE_perTF": "#777", "persite_nonblood": "#2ca02c", "persite_nonblood_excl": "#98df8a",
            "persite_blood": "#d62728", "persite_all": "#1f77b4"}
    mm = [m for m in MODELS]
    for i, m in enumerate(mm):
        vals = [tA[(tA.scope == s) & (tA.model == m)].macro_AUROC.iloc[0] for s in scopes]
        a.bar(np.arange(3) + (i - len(mm) / 2) * w, vals, w, label=m.replace("persite_", "ps:"),
              color=cols.get(m, None))
    a.set_xticks(range(3)); a.set_xticklabels(scopes); a.set_ylim(0.5, None); a.axhline(0.5, color="grey", ls=":")
    a.set_ylabel("standalone macro-AUROC"); a.set_title("(A) per-site non-blood SHAPE vs per-TF / blood")
    a.legend(fontsize=7, frameon=False)
    b = ax[1]; xs = np.arange(len(short)); mm2 = [m for m in ["SHAPE_perTF", "persite_nonblood", "persite_blood"] if m in MODELS]
    w = 0.26
    for i, m in enumerate(mm2):
        b.bar(xs + (i - 1) * w, tPC[m].values, w, label=m.replace("persite_", "ps:"), color=cols.get(m, None))
    b.set_xticks(xs); b.set_xticklabels(short, rotation=30, ha="right"); b.set_ylim(0.5, None)
    b.axhline(0.5, color="grey", ls=":"); b.set_ylabel("per-class OVR AUROC (pooled)"); b.legend(fontsize=8, frameon=False)
    b.set_title("(B) per-cancer OVR AUROC")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_nonblood_persite.png", dpi=120); plt.close()
    log("[nb] figure written")


def write_summary(tA, tPC, tB, SETS):
    L = ["# Per-TFBS (per-site) SHAPE on non-blood UniBind sites\n",
         "Per-site short-fraction (±30bp, CN-free/PoN-free), one column per unique TFBS centre, within-sample "
         "robust-z. Site sets: " + ", ".join(f"{k}={len(v)}" for k, v in SETS.items()) + ".\n",
         "## (A) Standalone macro-AUROC\n", tA.to_string(index=False),
         "\n\n## (A) Per-class OVR AUROC (pooled)\n", tPC.to_string(index=False),
         "\n\n## (B) Depth add-on\n", tB.to_string(index=False), "\n"]
    open(f"{OUT}/nonblood_persite_summary.md", "w").write("\n".join(L))
    log("[nb] summary written")


if __name__ == "__main__":
    main()
