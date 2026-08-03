#!/usr/bin/env python
"""
Train + evaluate a baseline classifier ON the GRAIL cfDNA cohort (not transfer), with healthy controls.

Cohort: 198 plasma cfDNA = 54 prostate + 49 lung + 48 breast + 47 non-cancer (EGAD00001005302 / Razavi 2019).
Feasible feature views on the collapsed (capture-only, no-cfDNA-fragment) BAMs:
  Genome_wide_CNA          on-target PoN-normalized arm log2-ratios  (grail_cna_features.py)
  Mutation_signature       96->47 channel COSMIC exposures           (existing GRAIL feat)
  Somatic_mutation_profile per-gene nonsyn flags                     (existing GRAIL feat)
(Fragmentomic SHAPE/E1 are NOT computable here: the analysis2 BAMs carry no 30-180bp cfDNA fragments.)

Leakage-free repeated stratified 5-fold OOF; within-fold robust-z fit on train only; L2 logistic (balanced).
Reports, class-imbalance-aware (macro over equally-weighted classes):
  (a) 4-class TOO+detection : breast/lung/prostate/non-cancer  -> macro-AUROC, per-class AUROC, balanced acc
  (b) 3-class TOO           : cancers only (breast/lung/prostate)
  (c) binary detection      : cancer vs non-cancer             -> AUROC
Per view + simple-average + per-class NNLS fusion.

Out: results/auto_plan/grail/baseline/*.tsv
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.optimize import nnls

PROJ = "/home/jrkim/TSO_TFBS/project"
G = f"{PROJ}/results/auto_plan/grail"
FEAT = f"{G}/feat"
OUT = f"{G}/baseline"; os.makedirs(OUT, exist_ok=True)
VIEWS = ["Genome_wide_CNA", "Mutation_signature", "Somatic_mutation_profile"]
SEED = 42; NREP = 10
def log(m): print(m, flush=True)


def norm_sid(s):
    s = str(s).replace("collapsed", "")
    if s.endswith("CH"): s = s[:-2]
    return s


def load_view(name):
    p = f"{FEAT}/X_{name}.npz"
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True)
    sids = np.array([norm_sid(s) for s in z["sids"]])
    return z["X"].astype(np.float32), sids, [str(c) for c in z["cols"]]


def robustz_fit(X, tr):
    med = np.median(X[tr], 0); iqr = np.subtract(*np.percentile(X[tr], [75, 25], 0))
    iqr[iqr == 0] = 1.0; return med, iqr


def base_oof(X, y, K, reps=NREP):
    n = len(y); P = np.zeros((n, K))
    for rep in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + rep)
        for tr, te in skf.split(X, y):
            med, iqr = robustz_fit(X, tr)
            Xtr = np.clip((X[tr] - med) / iqr, -8, 8); Xte = np.clip((X[te] - med) / iqr, -8, 8)
            clf = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced", solver="lbfgs").fit(Xtr, y[tr])
            pr = np.zeros((len(te), K)); pr[:, clf.classes_] = clf.predict_proba(Xte)
            P[te] += pr
    return P / reps


def macro_auc(P, y, K):
    a = [roc_auc_score((y == k).astype(int), P[:, k]) for k in range(K) if 0 < (y == k).sum() < len(y)]
    return float(np.mean(a)) if a else np.nan


def per_class_auc(P, y, K):
    return [roc_auc_score((y == k).astype(int), P[:, k]) if 0 < (y == k).sum() < len(y) else np.nan
            for k in range(K)]


def wnnls(Z, t):
    w = (t == 1).mean(); sw = np.where(t == 1, 0.5 / max(w, 1e-6), 0.5 / max(1 - w, 1e-6)); s = np.sqrt(sw)[:, None]
    c, _ = nnls(np.column_stack([Z, np.ones(len(Z))]) * s, t * np.sqrt(sw)); return c[:-1]


def fuse_nnls(oof, y, K):
    P = np.zeros((len(y), K))
    for k in range(K):
        Z = np.column_stack([oof[m][:, k] for m in oof]); w = wnnls(Z, (y == k).astype(int))
        w = w / w.sum() if w.sum() > 1e-9 else np.full(Z.shape[1], 1 / Z.shape[1]); P[:, k] = Z @ w
    P = np.clip(P, 1e-12, None); return P / P.sum(1, keepdims=True)


def boot_ci(y, P, K, fn, n=1000):
    rng = np.random.default_rng(0); v = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        try: v.append(fn(P[ix], y[ix], K))
        except Exception: pass
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    man = pd.read_csv(f"{G}/manifest.tsv", sep="\t", dtype=str)
    man["key"] = man.sid.map(norm_sid)
    lab_of = dict(zip(man.key, man.cancer_type))
    views = {}
    for v in VIEWS:
        r = load_view(v)
        if r is None: log(f"[grail-clf] MISSING view {v}"); continue
        views[v] = r
    # common samples across all loaded views + manifest
    common = set(man.key)
    for v, (X, sids, cols) in views.items(): common &= set(sids)
    common = sorted(common)
    log(f"[grail-clf] samples usable across {list(views)}: {len(common)}")
    y_lab = np.array([lab_of[s] for s in common])
    classes = ["breast cancer", "lung cancer", "prostate cancer", "non-cancer"]
    ci = {c: k for k, c in enumerate(classes)}
    y4 = np.array([ci[l] for l in y_lab])
    log("[grail-clf] class counts: " + " ".join(f"{c}={int((y4==k).sum())}" for k, c in enumerate(classes)))
    # assemble per-view matrices aligned to `common`
    M = {}
    for v, (X, sids, cols) in views.items():
        pos = {s: i for i, s in enumerate(sids)}; M[v] = np.vstack([X[pos[s]] for s in common]).astype(np.float32)

    # ---------- (a) 4-class TOO+detection ----------
    K4 = 4; oof4 = {v: base_oof(M[v], y4, K4) for v in M}
    rows = []
    for v in M:
        rows.append(dict(view=v, task="4class_macroAUROC", value=round(macro_auc(oof4[v], y4, K4), 4)))
    P_avg4 = np.mean([oof4[v] for v in M], 0); P_nnls4 = fuse_nnls(oof4, y4, K4)
    lo, hi = boot_ci(y4, P_nnls4, K4, macro_auc)
    rows.append(dict(view="FUSION_avg", task="4class_macroAUROC", value=round(macro_auc(P_avg4, y4, K4), 4)))
    rows.append(dict(view="FUSION_nnls", task="4class_macroAUROC", value=round(macro_auc(P_nnls4, y4, K4), 4)))
    rows.append(dict(view="FUSION_nnls", task="4class_macroAUROC_CI", value=f"[{lo:.3f},{hi:.3f}]"))
    top = P_nnls4.argmax(1)
    rows.append(dict(view="FUSION_nnls", task="4class_balanced_acc", value=round(balanced_accuracy_score(y4, top), 4)))
    pc = per_class_auc(P_nnls4, y4, K4)
    pd.DataFrame({"class": classes, "n": [int((y4 == k).sum()) for k in range(K4)],
                  "fusion_AUROC": np.round(pc, 3)}).to_csv(f"{OUT}/grail_perclass_4class.tsv", sep="\t", index=False)

    # ---------- (b) 3-class TOO (cancers only) ----------
    cm = y_lab != "non-cancer"; classes3 = ["breast cancer", "lung cancer", "prostate cancer"]
    ci3 = {c: k for k, c in enumerate(classes3)}; y3 = np.array([ci3[l] for l in y_lab[cm]])
    oof3 = {v: base_oof(M[v][cm], y3, 3) for v in M}
    for v in M:
        rows.append(dict(view=v, task="3class_TOO_macroAUROC", value=round(macro_auc(oof3[v], y3, 3), 4)))
    P_nnls3 = fuse_nnls(oof3, y3, 3)
    rows.append(dict(view="FUSION_nnls", task="3class_TOO_macroAUROC", value=round(macro_auc(P_nnls3, y3, 3), 4)))

    # ---------- (c) binary cancer vs healthy detection ----------
    yb = (y_lab != "non-cancer").astype(int)
    oofb = {v: base_oof(M[v], yb, 2) for v in M}
    for v in M:
        rows.append(dict(view=v, task="detection_AUROC", value=round(roc_auc_score(yb, oofb[v][:, 1]), 4)))
    Pb_avg = np.mean([oofb[v] for v in M], 0)
    abi = roc_auc_score(yb, Pb_avg[:, 1]); lo2, hi2 = boot_ci(yb, Pb_avg, 2, lambda P, y, K: roc_auc_score(y, P[:, 1]))
    rows.append(dict(view="FUSION_avg", task="detection_AUROC", value=round(abi, 4)))
    rows.append(dict(view="FUSION_avg", task="detection_AUROC_CI", value=f"[{lo2:.3f},{hi2:.3f}]"))

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/grail_baseline_metrics.tsv", sep="\t", index=False)
    log("\n=== GRAIL baseline (trained on GRAIL, leakage-free CV) ===")
    log(df.to_string(index=False))
    log(f"\n[grail-clf] -> {OUT}/grail_baseline_metrics.tsv , grail_perclass_4class.tsv")


if __name__ == "__main__":
    main()
