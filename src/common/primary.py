#!/usr/bin/env python
"""
Phases 3-6 (autonomous, headless) — domain-aware TSO500 v1+v2 primary analysis. No Claude tokens.

  3. Locked 20% internal test  : 1000 candidate splits stratified by cancer_type x cohort(version);
     pick the metadata-best-balanced split (class + cohort + class*cohort proportion match). No peeking.
  4. Development repeated 5-fold CV (reps configurable): 6 base models (L2 multinomial logistic for fast,
     robust unattended completion), version-aware robust-z normalization fit on the TRAIN fold only.
     -> out-of-fold (OOF) class-probabilities per method, leakage-free.
  5. Late fusion: per-class non-negative least squares on OOF (balanced), shrunk toward global weights by
     alpha_c set from class size. Compare: simple-avg, global-NNLS, class-NNLS, shrinkage. Refit base models
     + norm on full development set, lock fusion, apply ONCE to the locked test.
  6. Cross-version transfer: train v1 -> test v2 and v2 -> v1, per method + fusion (rank-AUROC, robust).

Notes / documented simplifications for unattended reliability:
  * base learner = L2 multinomial logistic (lbfgs); fast and stable even for the 9k-col depth matrix.
    (Plan suggested elastic-net; L2 chosen so the headless run completes in ~1-2h without saga stalls.)
  * primary endpoint = 9 primary cancer types; exploratory (sarcoma, bladder) reported separately.

Outputs -> results/auto_plan/primary/ : locked_split.json, dev_oof.npz, basemodel_cv.tsv, fusion_compare.tsv,
  locked_test_metrics.tsv, perclass_locked.tsv, fusion_weights.tsv, transfer.tsv, confusion_locked.tsv
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
from scipy.optimize import nnls

PROJ = "/home/jrkim/TSO_TFBS/project"
FEAT = f"{PROJ}/results/auto_plan/feat"
OUT = os.environ.get("PRIMARY_OUT", f"{PROJ}/results/auto_plan/primary"); os.makedirs(OUT, exist_ok=True)
METHODS = ["E1_entropy", "All_exon_depth", "Genome_wide_CNA",
           "Mutation_signature", "Somatic_mutation_profile", "SHAPE"]
# Optional ablation: drop named methods (comma-separated) without touching the 6-feature artifacts.
_DROP = {m.strip() for m in os.environ.get("METHODS_DROP", "").split(",") if m.strip()}
if _DROP:
    METHODS = [m for m in METHODS if m not in _DROP]
PRIMARY = ["lung cancer", "colorectal cancer", "gastric cancer", "pancreatic cancer",
           "biliary tract cancer", "melanoma", "liver cancer", "prostate cancer", "breast cancer"]
SEED = 42
NREP = int(os.environ.get("CV_REPS", "10"))
NSPLIT = int(os.environ.get("N_SPLITS", "1000"))
def log(m): print(m, flush=True)


# ---------- data ----------
def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return z["X"].astype(np.float32), [str(s) for s in z["sids"]], [str(c) for c in z["cols"]]


def load_all():
    man = pd.read_csv(f"{PROJ}/results/auto_plan/manifest_dev.tsv", sep="\t", dtype=str)
    man = man[man.cancer_type.isin(PRIMARY)].copy()
    # per method/cohort: (X, sid->row, shared-col index). Columns INTERSECTED by name across v1+v2 so the
    # stacked dev matrix is column-comparable (MutSig 49 vs 47, MutProfile 1217 vs 854, depth filters differ).
    feats = {}
    for m in METHODS:
        loaded = {}
        for cohort in ["v1", "v2"]:
            p = f"{FEAT}/{cohort}/X_{m}.npz"
            if os.path.exists(p):
                X, sids, cols = load_npz(p); loaded[cohort] = (X, {s: i for i, s in enumerate(sids)}, cols)
        if len(loaded) < 2:
            feats[m] = None; continue
        shared = [c for c in loaded["v1"][2] if c in set(loaded["v2"][2])]
        if not shared: feats[m] = None; continue
        sel = {}
        for cohort in ["v1", "v2"]:
            X, idx, cols = loaded[cohort]; ci = {c: i for i, c in enumerate(cols)}
            keep = [ci[c] for c in shared]
            sel[cohort] = (X[:, keep], idx)
        feats[m] = sel
    METH = [m for m in METHODS if feats[m] is not None]
    if len(METH) < len(METHODS):
        log(f"[primary] WARNING only methods present cross-cohort: {METH}")
    globals()["METHODS"] = METH

    def has_all(sid, cohort):
        return all(sid in feats[m][cohort][1] for m in METH)
    man = man[[has_all(r.sid, r.cohort) for r in man.itertuples()]].reset_index(drop=True)
    classes = sorted(man.cancer_type.unique()); ci = {c: k for k, c in enumerate(classes)}
    y = man.cancer_type.map(ci).values
    cohort = (man.cohort == "v2").astype(int).values        # 0=v1, 1=v2
    M = {}
    for m in METH:
        rows = []
        for r in man.itertuples():
            X, idx = feats[m][r.cohort]; rows.append(X[idx[r.sid]])
        M[m] = np.vstack(rows).astype(np.float32)
    return man, M, y, cohort, classes, ci


# ---------- version-aware robust-z (fit on train rows only) ----------
def fit_vznorm(X, cohort, tr):
    par = {}
    for c in (0, 1):
        m = tr[cohort[tr] == c]
        if len(m) < 3: m = tr
        med = np.median(X[m], 0); iqr = np.subtract(*np.percentile(X[m], [75, 25], 0))
        iqr[iqr == 0] = 1.0; par[c] = (med, iqr)
    return par


def apply_vznorm(X, cohort, par):
    Z = np.empty_like(X, dtype=np.float32)
    for c in (0, 1):
        med, iqr = par[c]; m = cohort == c
        Z[m] = ((X[m] - med) / iqr).astype(np.float32)
    return np.clip(Z, -8, 8)


def base_oof(X, y, cohort, K, idx_rows, reps=NREP):
    """OOF probs over the given dev rows. Version-aware norm + L2 multinomial logistic, leakage-free."""
    n = len(idx_rows); P = np.zeros((n, K)); pos = {r: i for i, r in enumerate(idx_rows)}
    ya = y[idx_rows]
    for rep in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + rep)
        for tr_l, te_l in skf.split(idx_rows, ya):
            tr = idx_rows[tr_l]; te = idx_rows[te_l]
            par = fit_vznorm(X, cohort, tr)
            Xtr = apply_vznorm(X[tr], cohort[tr], par); Xte = apply_vznorm(X[te], cohort[te], par)
            clf = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced",
                                     solver="lbfgs").fit(Xtr, y[tr])
            pr = clf.predict_proba(Xte)
            full = np.zeros((len(te), K)); full[:, clf.classes_] = pr
            for j, r in enumerate(te): P[pos[r]] += full[j]
    return P / reps


def macro_auc(P, y, K):
    a = [roc_auc_score((y == k).astype(int), P[:, k]) for k in range(K)
         if 0 < (y == k).sum() < len(y)]
    return float(np.mean(a)) if a else np.nan


def per_class_auc(P, y, K):
    return np.array([roc_auc_score((y == k).astype(int), P[:, k]) if 0 < (y == k).sum() < len(y) else np.nan
                     for k in range(K)])


# ---------- per-class NNLS fusion (balanced), shrunk toward global ----------
def wnnls(Z, t):
    # Explicit intercept column: without it, a near-constant (low-variance) method whose probabilities sit at
    # the class base-rate gets loaded up by NNLS as a stand-in intercept, inflating its weight far above its
    # discriminative value (observed for Mutation_signature). The intercept absorbs that offset; we then return
    # the method coefficients only, so downstream fuse() stays a convex combination over the methods.
    w = (t == 1).mean(); sw = np.where(t == 1, 0.5 / max(w, 1e-6), 0.5 / max(1 - w, 1e-6))
    s = np.sqrt(sw)[:, None]
    Zc = np.column_stack([Z, np.ones(len(Z))])
    c, _ = nnls(Zc * s, t * np.sqrt(sw))
    return c[:-1]


def fusion_weights(oof, y, K, alpha_by_class):
    """NNLS stacking on per-class PROBABILITIES (not log): non-neg weights that combine method probs to
    predict the one-vs-rest target. Class-specific weights shrunk toward global by alpha_c."""
    nm = len(METHODS); Wc = np.zeros((K, nm))
    Zg = []; tg = []
    for k in range(K):
        Zg.append(np.column_stack([oof[m][:, k] for m in METHODS])); tg.append((y == k).astype(int))
    wg = wnnls(np.vstack(Zg), np.concatenate(tg))
    wg = wg / wg.sum() if wg.sum() > 1e-9 else np.full(nm, 1.0 / nm)
    for k in range(K):
        Z = np.column_stack([oof[m][:, k] for m in METHODS])
        wk = wnnls(Z, (y == k).astype(int))
        wk = wk / wk.sum() if wk.sum() > 1e-9 else wg.copy()
        a = alpha_by_class[k]
        Wc[k] = a * wk + (1 - a) * wg
        Wc[k] = Wc[k] / max(Wc[k].sum(), 1e-9)
    return Wc, wg


def fuse(oof, W, K):
    """weighted average of per-class method probabilities -> renormalized class distribution."""
    n = len(next(iter(oof.values()))); P = np.zeros((n, K))
    for k in range(K):
        Z = np.column_stack([oof[m][:, k] for m in METHODS])
        P[:, k] = Z @ W[k]
    P = np.clip(P, 1e-12, None)
    return P / P.sum(1, keepdims=True)


# ---------- locked split (metadata balance only) ----------
def choose_locked_split(man, y, cohort, K):
    n = len(y); rng = np.random.default_rng(SEED)
    strata = np.array([f"{a}_{b}" for a, b in zip(y, cohort)])
    best = None
    p_class = np.bincount(y, minlength=K) / n
    p_coh = np.array([(cohort == 0).mean(), (cohort == 1).mean()])
    for s in range(NSPLIT):
        test = np.zeros(n, bool)
        for st in np.unique(strata):                       # stratified 20% per (class,cohort) cell
            ix = np.where(strata == st)[0]; rng.shuffle(ix)
            ntest = max(1, int(round(0.20 * len(ix)))) if len(ix) >= 3 else 0
            test[ix[:ntest]] = True
        dev = ~test
        if test.sum() < 20 or dev.sum() < 50: continue
        bc = np.abs(np.bincount(y[dev], minlength=K) / dev.sum() - p_class).sum()
        bh = np.abs(np.bincount(y[test], minlength=K) / test.sum() - p_class).sum()
        cc = abs((cohort[dev] == 1).mean() - p_coh[1]) + abs((cohort[test] == 1).mean() - p_coh[1])
        score = bc + bh + cc
        if best is None or score < best[0]:
            best = (score, test.copy())
    test = best[1]
    return np.where(~test)[0], np.where(test)[0], float(best[0])


def boot_ci(fn, y, P, K, n=1000):
    rng = np.random.default_rng(0); vals = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        try: vals.append(fn(P[ix], y[ix], K))
        except Exception: pass
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    man, M, y, cohort, classes, ci = load_all()
    K = len(classes); n = len(y)
    log(f"[primary] dev cohort n={n} (v1={int((cohort==0).sum())} v2={int((cohort==1).sum())}) K={K}")
    log("[primary] class counts: " + " ".join(f"{c}={int((y==k).sum())}" for k, c in enumerate(classes)))

    # ---- Phase 3: locked split ----
    dev, test, bscore = choose_locked_split(man, y, cohort, K)
    json.dump({"balance_score": bscore, "n_dev": int(len(dev)), "n_test": int(len(test)),
               "dev_sids": man.sid.values[dev].tolist(), "test_sids": man.sid.values[test].tolist(),
               "classes": classes}, open(f"{OUT}/locked_split.json", "w"))
    log(f"[primary] locked split: dev={len(dev)} test={len(test)} balance={bscore:.4f}")

    # ---- Phase 4: dev CV base models (OOF on dev rows) ----
    oof = {}; rows = []
    for m in METHODS:
        oof[m] = base_oof(M[m], y, cohort, K, dev)
        a = macro_auc(oof[m], y[dev], K)
        rows.append(dict(method=m, dev_macro_AUROC=round(a, 4)))
        log(f"[primary] base {m:24s} dev macro AUROC={a:.4f}")
    pd.DataFrame(rows).to_csv(f"{OUT}/basemodel_cv.tsv", sep="\t", index=False)
    np.savez(f"{OUT}/dev_oof.npz", y=y[dev], classes=np.array(classes),
             **{m: oof[m] for m in METHODS})

    # ---- Phase 5: fusion (shrinkage) ----
    cnt = np.bincount(y[dev], minlength=K)
    alpha = np.clip(cnt / (cnt + 60.0), 0.0, 0.8)            # large class -> trust class-specific weights more
    Wc, wg = fusion_weights(oof, y[dev], K, alpha)
    P_simple = np.mean([oof[m] for m in METHODS], 0)
    P_global = fuse(oof, np.tile(wg, (K, 1)), K)
    P_shrink = fuse(oof, Wc, K)
    fr = [dict(fusion="simple_avg", dev_macro=round(macro_auc(P_simple, y[dev], K), 4)),
          dict(fusion="global_NNLS", dev_macro=round(macro_auc(P_global, y[dev], K), 4)),
          dict(fusion="shrinkage_classspecific", dev_macro=round(macro_auc(P_shrink, y[dev], K), 4))]
    pd.DataFrame(fr).to_csv(f"{OUT}/fusion_compare.tsv", sep="\t", index=False)
    log("[primary] fusion dev macro: " + " ".join(f"{d['fusion']}={d['dev_macro']}" for d in fr))
    pd.DataFrame(Wc, index=classes, columns=METHODS).to_csv(f"{OUT}/fusion_weights.tsv", sep="\t")

    # ---- Phase 5b: lock & evaluate on locked test ONCE ----
    # refit base models on full dev, version-aware norm on dev only, predict test
    Ptest = {}
    for m in METHODS:
        par = fit_vznorm(M[m], cohort, dev)
        Xd = apply_vznorm(M[m][dev], cohort[dev], par); Xt = apply_vznorm(M[m][test], cohort[test], par)
        clf = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced",
                                 solver="lbfgs").fit(Xd, y[dev])
        pr = np.zeros((len(test), K)); pr[:, clf.classes_] = clf.predict_proba(Xt); Ptest[m] = pr
    Pt = fuse(Ptest, Wc, K); yt = y[test]
    top = np.argsort(-Pt, 1)
    acc1 = float((top[:, 0] == yt).mean())
    acc2 = float(np.mean([yt[i] in top[i, :2] for i in range(len(yt))]))
    acc3 = float(np.mean([yt[i] in top[i, :3] for i in range(len(yt))]))
    macro = macro_auc(Pt, yt, K)
    wts = np.bincount(yt, minlength=K) / len(yt)
    wauc = float(np.nansum(per_class_auc(Pt, yt, K) * wts))
    lo, hi = boot_ci(macro_auc, yt, Pt, K)
    ba = balanced_accuracy_score(yt, top[:, 0])
    pd.DataFrame([dict(metric="top1_acc", value=round(acc1, 4)), dict(metric="top2_acc", value=round(acc2, 4)),
                  dict(metric="top3_acc", value=round(acc3, 4)), dict(metric="macro_AUROC", value=round(macro, 4)),
                  dict(metric="macro_AUROC_CI_lo", value=round(lo, 4)), dict(metric="macro_AUROC_CI_hi", value=round(hi, 4)),
                  dict(metric="weighted_AUROC", value=round(wauc, 4)), dict(metric="balanced_acc", value=round(ba, 4)),
                  dict(metric="n_test", value=int(len(yt)))]).to_csv(f"{OUT}/locked_test_metrics.tsv", sep="\t", index=False)
    pc = pd.DataFrame({"class": classes, "n_test": [int((yt == k).sum()) for k in range(K)],
                       "AUROC": per_class_auc(Pt, yt, K).round(3)})
    pc.to_csv(f"{OUT}/perclass_locked.tsv", sep="\t", index=False)
    pd.DataFrame(confusion_matrix(yt, top[:, 0], labels=range(K)), index=classes, columns=classes
                 ).to_csv(f"{OUT}/confusion_locked.tsv", sep="\t")
    log(f"[primary] LOCKED TEST: top1={acc1:.3f} top2={acc2:.3f} top3={acc3:.3f} "
        f"macroAUROC={macro:.3f}[{lo:.3f},{hi:.3f}] wAUROC={wauc:.3f} balAcc={ba:.3f}")

    # ---- Phase 6: cross-version transfer (per method + fusion), rank-AUROC ----
    tr_rows = []
    for src, dst, name in [(0, 1, "v1->v2"), (1, 0, "v2->v1")]:
        si = np.where(cohort == src)[0]; di = np.where(cohort == dst)[0]
        Pm = {}
        for m in METHODS:
            par = fit_vznorm(M[m], cohort, si)
            Xs = apply_vznorm(M[m][si], cohort[si], par); Xd = apply_vznorm(M[m][di], cohort[di], par)
            clf = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced",
                                     solver="lbfgs").fit(Xs, y[si])
            pr = np.zeros((len(di), K)); pr[:, clf.classes_] = clf.predict_proba(Xd); Pm[m] = pr
            tr_rows.append(dict(direction=name, model=m, macro_AUROC=round(macro_auc(pr, y[di], K), 4)))
        Pf = fuse(Pm, Wc, K)
        tr_rows.append(dict(direction=name, model="FUSION_shrinkage", macro_AUROC=round(macro_auc(Pf, y[di], K), 4)))
        log(f"[primary] transfer {name}: fusion macro={macro_auc(Pf, y[di], K):.3f}")
    pd.DataFrame(tr_rows).to_csv(f"{OUT}/transfer.tsv", sep="\t", index=False)
    log("[primary] DONE -> results/auto_plan/primary/")


if __name__ == "__main__":
    main()
