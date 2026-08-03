#!/usr/bin/env python
"""
BEST-base + BEST-combiner version of the our-style six-feature ensemble, to try to beat the strongest
single view (exon depth). Within v1 and v2 SEPARATELY, locked 70/30 protocol, nested + leakage-free.

Stage 1 (per feature, chosen by TRAIN-OOF macro-AUROC):
  candidate library = {L2-logistic, elastic-net(saga), XGBoost, HistGradientBoosting}, small tuned grids.
  Wide matrices (>1500 feats) get a fold-fit TruncatedSVD(200) for the tree/saga learners (leakage-free).
Stage 2 (late fusion, chosen by INNER StratifiedKFold CV on the train rows only):
  C1 per-class NNLS-shrinkage | C2 logistic stacker | C3 XGB stacker | C4 mean | C5 best-base floor
  | C6 guarded NNLS (per class take fusion only where it beats best-base on inner OOF).
Stage 3: refit chosen bases on full 70%, apply chosen combiner, score the held-out 30% ONCE.
Two splits: stratified-random (SEED=42) and ctDNA-fraction low-burden. v1 and v2 reported separately.

Reuses our_ensemble_v1v2.load_cohort and primary.py fusion math. No feature re-extraction.
Outputs -> results/auto_plan/our_ensemble/best_*.tsv + report/fig24_best_*.png
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix
import xgboost as xgb

PROJ = "/home/jrkim/TSO_TFBS/project"
sys.path.insert(0, f"{PROJ}/scripts/auto")
import primary as P
import our_ensemble_v1v2 as OE
OUT = OE.OUT; FIG = OE.FIG
METHODS, NICE, short = OE.METHODS, OE.NICE, OE.short
P.METHODS = METHODS
SEED = 42; REPS_SEL = 2; REPS_FIN = 3; WIDE = 1500; SVDK = 200
def log(m): print(m, flush=True)


# ---------------- base candidate library ----------------
# Only All_exon_depth (7567) is wide -> raw logL2 keeps the genuine strong baseline (the 0.860 bar);
# enet/xgb use a fold-fit SVD(200) on wide so they stay cheap. Tree depth is where error-diversity vs
# depth's linear fit comes from, so we keep two XGBoost configs.
BASE_GRID = [
    ("logL2",  dict(C=1.0)),
    ("enet",   dict(C=1.0, l1=0.5)),
    ("xgb",    dict(depth=3, eta=0.1,  n=400)),
    ("xgb",    dict(depth=4, eta=0.05, n=600)),
]

def make_est(kind, pr, K):
    if kind == "logL2":
        return LogisticRegression(max_iter=300, C=pr["C"], class_weight="balanced", solver="lbfgs")
    if kind == "enet":
        return LogisticRegression(max_iter=300, C=pr["C"], l1_ratio=pr["l1"], penalty="elasticnet",
                                  solver="saga", class_weight="balanced", tol=2e-3, n_jobs=4)
    if kind == "xgb":
        return xgb.XGBClassifier(max_depth=pr["depth"], learning_rate=pr["eta"], n_estimators=pr["n"],
                                 subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
                                 num_class=K, tree_method="hist", n_jobs=4, eval_metric="mlogloss",
                                 reg_lambda=1.0, verbosity=0)
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_leaf_nodes=pr["leaf"], learning_rate=pr["lr"],
                                              max_iter=pr["it"], l2_regularization=1.0, random_state=SEED)


def prep_fold(Xtr, Xte, kind, use_svd):
    """version-agnostic here (single cohort): robust-z by median/IQR on train; optional train-fit SVD."""
    med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0.0, med)
    Atr = np.where(np.isnan(Xtr), med, Xtr); Ate = np.where(np.isnan(Xte), med, Xte)
    q1, q3 = np.percentile(Atr, [25, 75], 0); iqr = q3 - q1; iqr[iqr == 0] = 1.0
    Ztr = np.clip((Atr - med) / iqr, -8, 8); Zte = np.clip((Ate - med) / iqr, -8, 8)
    if use_svd and Ztr.shape[1] > WIDE:
        k = min(SVDK, Ztr.shape[1] - 1, len(Ztr) - 1)
        svd = TruncatedSVD(n_components=k, random_state=SEED).fit(Ztr)
        Ztr, Zte = svd.transform(Ztr), svd.transform(Zte)
    return Ztr.astype(np.float32), Zte.astype(np.float32)


def oof_for(Xm, y, K, idx, kind, pr, reps):
    """leakage-free OOF probs for one base config over rows `idx` (a single cohort)."""
    n = len(idx); Pr = np.zeros((n, K)); pos = {r: i for i, r in enumerate(idx)}; ya = y[idx]
    use_svd = kind in ("enet", "xgb")
    for rep in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + rep)
        for tl, vl in skf.split(idx, ya):
            tr, te = idx[tl], idx[vl]
            Xt, Xv = prep_fold(Xm[tr], Xm[te], kind, use_svd)
            est = make_est(kind, pr, K).fit(Xt, y[tr])
            pp = est.predict_proba(Xv); full = np.zeros((len(te), K)); full[:, est.classes_] = pp
            for j, r in enumerate(te): Pr[pos[r]] += full[j]
    return Pr / reps


def refit_predict(Xm, y, K, tr, va, kind, pr):
    use_svd = kind in ("enet", "xgb")
    Xt, Xv = prep_fold(Xm[tr], Xm[va], kind, use_svd)
    est = make_est(kind, pr, K).fit(Xt, y[tr])
    pp = est.predict_proba(Xv); full = np.zeros((len(va), K)); full[:, est.classes_] = pp
    return full


def macro_auc(Pm, yy, K):
    a = [roc_auc_score((yy == k).astype(int), Pm[:, k]) for k in range(K) if 0 < (yy == k).sum() < len(yy)]
    return float(np.mean(a)) if a else np.nan


# ---------------- combiners (fit on train-OOF dict, return predictor) ----------------
def sub(oof, idx): return {m: oof[m][idx] for m in METHODS}
def stack_mat(oof): return np.column_stack([oof[m] for m in METHODS])   # n x (6K)

def fit_combiner(name, oof, ytr, K):
    """Return a function val_oof_dict -> (n x K) probs."""
    if name == "nnls":
        cnt = np.bincount(ytr, minlength=K); alpha = np.clip(cnt / (cnt + 60.0), 0.0, 0.8)
        Wc, _ = P.fusion_weights(oof, ytr, K, alpha)
        return lambda vo: P.fuse(vo, Wc, K), Wc
    if name == "mean":
        return lambda vo: _norm(sum(vo[m] for m in METHODS) / len(METHODS)), None
    if name == "logit":
        clf = LogisticRegression(max_iter=400, C=1.0, class_weight="balanced", solver="lbfgs").fit(stack_mat(oof), ytr)
        def f(vo):
            pp = clf.predict_proba(stack_mat(vo)); full = np.zeros((len(next(iter(vo.values()))), K)); full[:, clf.classes_] = pp; return full
        return f, None
    if name == "xgbstack":
        clf = xgb.XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=200, subsample=0.8,
                                objective="multi:softprob", num_class=K, tree_method="hist", n_jobs=4,
                                eval_metric="mlogloss", reg_lambda=2.0, verbosity=0).fit(stack_mat(oof), ytr)
        def f(vo):
            pp = clf.predict_proba(stack_mat(vo)); full = np.zeros((len(next(iter(vo.values()))), K)); full[:, clf.classes_] = pp; return full
        return f, None
    if name == "bestbase":
        sc = {m: macro_auc(oof[m], ytr, K) for m in METHODS}; bm = max(sc, key=sc.get)
        return lambda vo: _norm(vo[bm]), bm
    if name == "guarded":   # per-class: fusion where it beats best-base on this train OOF, else best-base
        cnt = np.bincount(ytr, minlength=K); alpha = np.clip(cnt / (cnt + 60.0), 0.0, 0.8)
        Wc, _ = P.fusion_weights(oof, ytr, K, alpha)
        Pf = P.fuse(oof, Wc, K)
        sc = {m: macro_auc(oof[m], ytr, K) for m in METHODS}; bm = max(sc, key=sc.get)
        use_fus = np.array([roc_auc_score((ytr == k).astype(int), Pf[:, k]) >=
                            roc_auc_score((ytr == k).astype(int), oof[bm][:, k])
                            if 0 < (ytr == k).sum() < len(ytr) else True for k in range(K)])
        def f(vo):
            Pfu = P.fuse(vo, Wc, K); Pb = _norm(vo[bm]); out = np.where(use_fus[None, :], Pfu, Pb); return _norm(out)
        return f, (bm, use_fus.sum())

def _norm(Pm):
    Pm = np.clip(Pm, 1e-12, None); return Pm / Pm.sum(1, keepdims=True)

COMBINERS = ["nnls", "logit", "xgbstack", "mean", "bestbase", "guarded"]


def inner_cv_combiner(name, oof, ytr, K, folds=5):
    """leakage-safe combiner selection: inner SKF over train rows; combiner never sees its inner-test rows."""
    n = len(ytr); preds = np.zeros((n, K)); idx = np.arange(n)
    for tl, vl in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(idx, ytr):
        fpredict, _ = fit_combiner(name, sub(oof, tl), ytr[tl], K)
        preds[vl] = fpredict(sub(oof, vl))
    return macro_auc(preds, ytr, K)


# ---------------- per-cohort pipeline ----------------
def run_split(coh, X, y, classes, tr, va, split_name, store=None):
    K = len(classes); ytr, yva = y[tr], y[va]
    # Stage 1: pick best base per feature by train-OOF macro-AUROC
    chosen, oof_tr, base_tr_auc = {}, {}, {}
    for m in METHODS:
        best = None
        for kind, pr in BASE_GRID:
            o = oof_for(X[m], y, K, tr, kind, pr, REPS_SEL)
            a = macro_auc(o, ytr, K)
            if best is None or a > best[0]: best = (a, kind, pr, o)
        chosen[m] = (best[1], best[2]); base_tr_auc[m] = round(best[0], 3)
        # final OOF at higher reps for the winner (for combiner fitting)
        oof_tr[m] = oof_for(X[m], y, K, tr, best[1], best[2], REPS_FIN)
        log(f"   [{coh}/{split_name}] base {NICE[m]:12s} -> {best[1]:5s} {best[2]} trainOOF={base_tr_auc[m]}")
    # Stage 2: pick combiner by inner-CV on train OOF
    comb_score = {}
    for cname in COMBINERS:
        try: comb_score[cname] = round(inner_cv_combiner(cname, oof_tr, ytr, K), 4)
        except Exception as e: comb_score[cname] = -1; log(f"      combiner {cname} failed: {e}")
    best_comb = max(comb_score, key=comb_score.get)
    log(f"   [{coh}/{split_name}] combiner innerCV: " +
        " ".join(f"{c}={comb_score[c]}" for c in COMBINERS) + f"  -> WIN {best_comb}")
    # Stage 3: refit bases on full train, build val OOF, apply chosen combiner, score once
    oof_va = {m: refit_predict(X[m], y, K, tr, va, chosen[m][0], chosen[m][1]) for m in METHODS}
    fpredict, meta = fit_combiner(best_comb, oof_tr, ytr, K)
    Pval = fpredict(oof_va)
    base_val = {m: macro_auc(oof_va[m], yva, K) for m in METHODS}
    depth_val = round(base_val["All_exon_depth"], 3)
    bb = max(base_val, key=base_val.get)
    rec = dict(cohort=coh, split=split_name, n_train=len(tr), n_val=len(va),
               combiner=best_comb, fusion_val_AUROC=round(macro_auc(Pval, yva, K), 3),
               fusion_val_acc=round(accuracy_score(yva, Pval.argmax(1)), 3),
               fusion_val_balacc=round(balanced_accuracy_score(yva, Pval.argmax(1)), 3),
               exon_depth_val_AUROC=depth_val, best_single=NICE[bb],
               best_single_val_AUROC=round(base_val[bb], 3),
               delta_vs_depth=round(macro_auc(Pval, yva, K) - depth_val, 3))
    log(f"   [{coh}/{split_name}] FUSION({best_comb}) val={rec['fusion_val_AUROC']} "
        f"vs depth={depth_val}  delta={rec['delta_vs_depth']:+.3f}  acc={rec['fusion_val_acc']}")
    if store is not None:
        store["chosen"][(coh, split_name)] = {NICE[m]: f"{chosen[m][0]} {chosen[m][1]}" for m in METHODS}
        store["combscore"][(coh, split_name)] = comb_score
        store["basetrauc"][(coh, split_name)] = {NICE[m]: base_tr_auc[m] for m in METHODS}
        store["baseval"][(coh, split_name)] = {NICE[m]: round(base_val[m], 3) for m in METHODS}
        if split_name == "stratified_random":
            store["cm"][coh] = (confusion_matrix(yva, Pval.argmax(1), labels=range(K)), classes)
    return rec


def main():
    rows = []; store = dict(chosen={}, combscore={}, basetrauc={}, baseval={}, cm={})
    for coh in ["v1", "v2"]:
        X, y, classes, tf = OE.load_cohort(coh); K = len(classes)
        log(f"\n=== {coh}: n={len(y)} K={K} {[short(c) for c in classes]}")
        tr, va = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=SEED).split(np.zeros(len(y)), y))
        t0 = time.time(); rows.append(run_split(coh, X, y, classes, tr, va, "stratified_random", store))
        log(f"   ...stratified_random done ({time.time()-t0:.0f}s)")
        okt = np.isfinite(tf)
        if okt.sum() >= 30:
            thr = np.quantile(tf[okt], 0.30); va2 = np.where(okt & (tf <= thr))[0]; tr2 = np.where(okt & (tf > thr))[0]
            trc = set(y[tr2].tolist()); va2 = np.array([i for i in va2 if y[i] in trc])
            if len(va2) >= 10:
                t0 = time.time(); rows.append(run_split(coh, X, y, classes, tr2, va2, "ctDNA_fraction", store))
                log(f"   ...ctDNA_fraction done ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows); res.to_csv(f"{OUT}/best_results.tsv", sep="\t", index=False)
    # chosen base method table (stratified_random)
    ch = pd.DataFrame({coh: store["chosen"][(coh, "stratified_random")] for coh in ["v1", "v2"]})
    ch.to_csv(f"{OUT}/best_chosen_bases.tsv", sep="\t")
    cs = pd.DataFrame({coh: store["combscore"][(coh, "stratified_random")] for coh in ["v1", "v2"]})
    cs.to_csv(f"{OUT}/best_combiner_innercv.tsv", sep="\t")
    bv = pd.DataFrame({coh: store["baseval"][(coh, "stratified_random")] for coh in ["v1", "v2"]})
    bv.to_csv(f"{OUT}/best_baseval.tsv", sep="\t")
    log(f"\n[best] wrote best_results.tsv ({len(res)}) + chosen/combiner/baseval tables")
    print(res.to_string(index=False))

    # fig24: per-cohort base val AUROC (best learner) + fusion + depth marker
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for j, coh in enumerate(["v1", "v2"]):
        bvc = store["baseval"][(coh, "stratified_random")]
        order = [NICE[m] for m in METHODS]
        vals = [bvc[o] for o in order]
        fusrow = res[(res.cohort == coh) & (res.split == "stratified_random")].iloc[0]
        bars = axs[j].bar(range(len(order)), vals, color="#9ecae1")
        axs[j].bar(len(order), fusrow.fusion_val_AUROC, color="#d98c5f")
        axs[j].axhline(fusrow.exon_depth_val_AUROC, color="k", ls="--", lw=1, alpha=.7)
        axs[j].set_xticks(range(len(order) + 1)); axs[j].set_xticklabels(order + ["FUSION"], rotation=22, ha="right", fontsize=8)
        axs[j].set_ylim(0.5, 1.0); axs[j].set_title(f"{coh}: best-learner bases vs tuned fusion ({fusrow.combiner})")
        axs[j].grid(alpha=.3, axis="y")
    axs[0].set_ylabel("validation macro AUROC")
    plt.suptitle("Best base learner per feature + best late-fusion combiner (dashed = exon depth)")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig24_best_bases_fusion.png", dpi=120); plt.close()
    log("[best] figure -> fig24_best_bases_fusion")


if __name__ == "__main__":
    main()
