#!/usr/bin/env python
"""Per-feature MODEL-ARCHITECTURE benchmark with hyperparameter tuning + 20x5 repeated CV (mean +- 95% CI).

For EACH cell (cohort x feature x pipeline):
  1) inner 3-fold CV grid-search -> best hyperparameters (tuned ONCE per cell, not per outer fold)
       A Elastic net          : C in {.3,1,3}, l1 in {.2,.5,.8}                       (9)
       B XGBoost (GPU)        : depth in {3,4,6}, lr in {.05,.1}, n_est in {200,400}  (12)
       C EN-select -> XGB     : selector enet fixed (C=.5,l1=.5, top-300); tune XGB    (12)
       D SVD -> XGB           : k in {100,200} x XGB grid                              (24)
       E SVD -> Elastic net   : k in {100,200} x enet grid                            (18)
  2) 20x5 repeated stratified outer CV with the best HPs -> macro-AUROC / weighted-AUROC / top-1 / macro-F1,
     each as mean +- 95% CI over the 20 repeats.
Leakage-free: scaler / SVD / selector fit on the train fold only, inside every fit.

Features: E1_entropy, All_exon_depth, Genome_wide_CNA(=broad >10Mb), Somatic_mutation_profile, SHAPE,
plus EARLY_FUSION (all concatenated) and LATE_FUSION (per-cancer NNLS over each feature's BEST pipeline's OOF).

Parallelism: saga is single-threaded -> process pool (loky) at 80% of cores, 1 BLAS thread/worker.
GPU used ONLY for Model B (full high-dim XGBoost); C/D run XGBoost on the reduced <=300 dims on CPU.
Resumable: every cell's 20-repeat OOF is cached to _oof_cache/{cohort}_{feature}_{model}.npz.

Out: tables/Tab_model_comparison_tuned.tsv, Tab_fusion_tuned.tsv, Tab_best_hparams.tsv + Fig_model_comparison_tuned.png
"""
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
NPROC = max(1, int(os.cpu_count() * 0.8))        # 80% of cores
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"                          # process-level parallelism -> 1 thread per worker
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
sys.path.insert(0, "/home/jrkim/TSO_TFBS/project/scripts/auto"); sys.path.insert(0, os.path.dirname(__file__))
import manuscript_figures as MF
from ncr_common import FEAT, FG, TB, OUT, SEED, MODS, NICE, COH, macro_auc, weighted_auc, topk_acc, wnnls

CACHE = os.environ.get("MCT_CACHE", f"{OUT}/_oof_cache"); os.makedirs(CACHE, exist_ok=True)
N_REP = int(os.environ.get("MCT_REP", 20)); N_SPLIT = 5; INNER = 3
SMOKE = os.environ.get("MCT_SMOKE") == "1"

MODELS = ["A: Elastic net", "B: XGBoost", "C: EN-select->XGB", "D: SVD->XGB", "E: SVD->Elastic net"]
MKEY = {"A: Elastic net": "A", "B: XGBoost": "B", "C: EN-select->XGB": "C", "D: SVD->XGB": "D", "E: SVD->Elastic net": "E"}
MCOL = {"A: Elastic net": "#2c7fb8", "B: XGBoost": "#d95f0e", "C: EN-select->XGB": "#1b9e77",
        "D: SVD->XGB": "#e7298a", "E: SVD->Elastic net": "#7570b3"}
ENET_GRID = [{"C": c, "l1": l} for c in (0.3, 1.0, 3.0) for l in (0.2, 0.5, 0.8)]
XGB_GRID = [{"max_depth": d, "lr": lr, "n_estimators": n} for d in (3, 4, 6) for lr in (0.05, 0.1) for n in (200, 400)]
D_GRID = [{"k": k, **g} for k in (100, 200) for g in XGB_GRID]
E_GRID = [{"k": k, **g} for k in (100, 200) for g in ENET_GRID]
GRID = {"A": ENET_GRID, "B": XGB_GRID, "C": XGB_GRID, "D": D_GRID, "E": E_GRID}
# N=1034 is too small to amortize per-fit GPU overhead across 136 fits/cell; CPU process-parallelism wins.
XGB_DEVICE = os.environ.get("MCT_XGB_DEVICE", "cpu")
BACKEND = {"A": "loky", "B": "loky", "C": "loky", "D": "loky", "E": "loky"}
NJOBS = {"A": NPROC, "B": NPROC, "C": NPROC, "D": NPROC, "E": NPROC}
if SMOKE:
    N_REP = 2; GRID = {k: v[:2] for k, v in GRID.items()}


# ---------- fit primitives (run inside workers) ----------
def _scale(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9; return (Xtr - mu) / sd, (Xte - mu) / sd


def _enet(C, l1, max_iter=400):
    return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1, C=C, class_weight="balanced",
                              max_iter=max_iter, tol=1e-2, n_jobs=1, random_state=SEED)


def _xgb(hp, device):
    return XGBClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], learning_rate=hp["lr"],
                         subsample=0.8, colsample_bytree=0.6, tree_method="hist", device=device, max_bin=128,
                         reg_lambda=1.0, n_jobs=int(os.environ.get("MCT_XGB_THREADS", "1")), eval_metric="mlogloss", verbosity=0, random_state=SEED)


def _proba(clf, Zte, K):
    P = np.zeros((len(Zte), K)); P[:, clf.classes_.astype(int)] = clf.predict_proba(Zte); return P


def _balw(y):
    """inverse-class-frequency sample weights (sklearn 'balanced'): w_i = n / (K * count[y_i])."""
    cls, cnt = np.unique(y, return_counts=True); m = {c: len(y) / (len(cls) * ct) for c, ct in zip(cls, cnt)}
    return np.array([m[v] for v in y], float)


def fit_A(Xtr, ytr, Xte, K, hp):
    Ztr, Zte = _scale(Xtr, Xte); return _proba(_enet(hp["C"], hp["l1"]).fit(Ztr, ytr), Zte, K)


def fit_B(Xtr, ytr, Xte, K, hp):
    return _proba(_xgb(hp, XGB_DEVICE).fit(Xtr, ytr, sample_weight=_balw(ytr)), Xte, K)


def fit_C(Xtr, ytr, Xte, K, hp):
    Ztr, Zte = _scale(Xtr, Xte)
    sel = SelectFromModel(_enet(0.5, 0.5), max_features=min(300, Xtr.shape[1]), threshold=-np.inf).fit(Ztr, ytr)
    Ts, Te = sel.transform(Ztr), sel.transform(Zte)
    return _proba(_xgb(hp, "cpu").fit(Ts, ytr, sample_weight=_balw(ytr)), Te, K)


def fit_D(Xtr, ytr, Xte, K, hp):
    Ztr, Zte = _scale(Xtr, Xte); k = min(hp["k"], Xtr.shape[1] - 1, len(Xtr) - 1)
    s = TruncatedSVD(k, random_state=SEED).fit(Ztr)
    return _proba(_xgb(hp, "cpu").fit(s.transform(Ztr), ytr, sample_weight=_balw(ytr)), s.transform(Zte), K)


def fit_E(Xtr, ytr, Xte, K, hp):
    Ztr, Zte = _scale(Xtr, Xte); k = min(hp["k"], Xtr.shape[1] - 1, len(Xtr) - 1)
    s = TruncatedSVD(k, random_state=SEED).fit(Ztr)
    return _proba(_enet(hp["C"], hp["l1"]).fit(s.transform(Ztr), ytr), s.transform(Zte), K)


FIT = {"A": fit_A, "B": fit_B, "C": fit_C, "D": fit_D, "E": fit_E}


def _tune_task(mk, X, y, K, tr, te, hp):
    return macro_auc(FIT[mk](X[tr], y[tr], X[te], K, hp), y[te], K)


def _fit_task(mk, X, y, K, tr, te, hp):
    return te, FIT[mk](X[tr], y[tr], X[te], K, hp).astype(np.float32)


# ---------- per-cell tune + repeated outer ----------
def metrics_from_oof(P, y, K):
    return (macro_auc(P, y, K), weighted_auc(P, y, K), topk_acc(P, y, 1),
            float(f1_score(y, P.argmax(1), average="macro")))


def ci(v):
    v = np.asarray(v, float); m = v.mean(); h = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return m, m - h, m + h


def process_cell(c, feat, mk, X, y, K):
    y = np.asarray(y).astype(int)                         # guarantee integer class labels (fit_*/_proba index P[:, classes_])
    cf = f"{CACHE}/{c}_{feat}_{mk}.npz"
    if os.path.exists(cf):
        z = np.load(cf, allow_pickle=True); return z["oof"], str(z["hp"])
    grid = GRID[mk]; bk = BACKEND[mk]; nj = NJOBS[mk]
    # let XGB models (B,C,D) run multi-threaded inside loky workers; saga (A,E) stays 1-thread (loky default forces OMP=1)
    imt = int(os.environ.get("MCT_XGB_THREADS", "1")) if mk in ("B", "C", "D") else 1
    # --- inner tuning ---
    inner = list(StratifiedKFold(INNER, shuffle=True, random_state=SEED).split(X, y))
    tt = [(mk, X, y, K, tr, te, hp) for hp in grid for (tr, te) in inner]
    sc = Parallel(n_jobs=nj, backend=bk, batch_size=1, inner_max_num_threads=imt)(delayed(_tune_task)(*t) for t in tt)
    g = len(inner); means = [np.nanmean(sc[i * g:(i + 1) * g]) for i in range(len(grid))]
    best = grid[int(np.nanargmax(means))]
    # --- 20x5 repeated outer with best HP ---
    rkf = list(RepeatedStratifiedKFold(n_splits=N_SPLIT, n_repeats=N_REP, random_state=SEED).split(X, y))
    ot = [(mk, X, y, K, tr, te, best) for (tr, te) in rkf]
    res = Parallel(n_jobs=nj, backend=bk, batch_size=1, inner_max_num_threads=imt)(delayed(_fit_task)(*t) for t in ot)
    oof = np.zeros((N_REP, len(y), K), np.float32)
    for i, (te, Pte) in enumerate(res):
        oof[i // N_SPLIT, te] = Pte
    np.savez_compressed(cf, oof=oof, hp=json.dumps(best))
    return oof, json.dumps(best)


def summarize(oof, y, K):
    R = oof.shape[0]; M = np.array([metrics_from_oof(oof[r], y, K) for r in range(R)])
    names = ["macroAUROC", "weightedAUROC", "top1", "macroF1"]; out = {}
    for j, nm in enumerate(names):
        m, lo, hi = ci(M[:, j]); out[nm] = round(m, 4); out[nm + "_lo"] = round(lo, 4); out[nm + "_hi"] = round(hi, 4)
    return out


def load_mod(c, m, sids):
    z = np.load(f"{FEAT}/{c}/X_{m}.npz", allow_pickle=True); pos = {str(s): i for i, s in enumerate(z["sids"])}
    return np.nan_to_num(np.vstack([z["X"][pos[s]] for s in sids]).astype(float))


def late_fuse_rep(oofs, y, K):
    """oofs: list of (n,K) OOF prob matrices (one per feature, single repeat). Per-cancer NNLS."""
    F = np.zeros((len(y), K))
    for k in range(K):
        Zc = np.column_stack([P[:, k] for P in oofs]); w = wnnls(Zc, (y == k).astype(int))
        w = w / w.sum() if w.sum() > 1e-9 else np.full(len(oofs), 1 / len(oofs)); F[:, k] = Zc @ w
    F = np.clip(F, 1e-12, None); return F / F.sum(1, keepdims=True)


def main():
    R, PC, Z, SID, LAB, TF = MF.load()
    rows = []; hp_rows = []; fus_rows = []
    feat_groups = MODS + ["EARLY_FUSION"]
    for c in COH:
        sids = [str(s) for s in SID[c]]; y = np.asarray(Z[c]["y"]); classes = [str(x) for x in Z[c]["classes"]]; K = len(classes)
        print(f"\n=== {c}: n={len(sids)} K={K} | NPROC={NPROC} N_REP={N_REP} ===", flush=True)
        Xcache = {m: load_mod(c, m, sids) for m in MODS}
        Xcache["EARLY_FUSION"] = np.hstack([Xcache[m] for m in MODS])
        oof_best = {}                                   # feat -> (20,n,K) OOF of its best model (for late fusion)
        for feat in feat_groups:
            X = Xcache[feat]; best_macro = -1; best_oof = None; best_model = None
            for model in MODELS:
                mk = MKEY[model]
                oof, hp = process_cell(c, feat, mk, X, y, K)
                s = summarize(oof, y, K)
                rows.append(dict(cohort=c, feature=feat, feature_nice=NICE.get(feat, "Early fusion (all feat)"),
                                 model=model, n_features=X.shape[1], **s))
                hp_rows.append(dict(cohort=c, feature=feat, model=model, best_hp=hp))
                print(f"  {c} {feat:<24} {model:<20} mAUROC={s['macroAUROC']:.3f} "
                      f"[{s['macroAUROC_lo']:.3f},{s['macroAUROC_hi']:.3f}] top1={s['top1']:.3f} hp={hp}", flush=True)
                pd.DataFrame(rows).to_csv(f"{TB}/Tab_model_comparison_tuned.tsv", sep="\t", index=False)  # incremental
                pd.DataFrame(hp_rows).to_csv(f"{TB}/Tab_best_hparams.tsv", sep="\t", index=False)
                if feat in MODS and s["macroAUROC"] > best_macro:
                    best_macro = s["macroAUROC"]; best_oof = oof; best_model = model
            if feat in MODS:
                oof_best[feat] = (best_oof, best_model)
        # ---- late fusion (best pipeline per feature), per repeat -> CI ----
        feats = MODS; mats = [oof_best[f][0] for f in feats]
        latem = np.zeros((N_REP, 4))
        for r in range(N_REP):
            F = late_fuse_rep([mats[i][r] for i in range(len(feats))], y, K)
            latem[r] = metrics_from_oof(F, y, K)
        names = ["macroAUROC", "weightedAUROC", "top1", "macroF1"]; lf = {}
        for j, nm in enumerate(names):
            m, lo, hi = ci(latem[:, j]); lf[nm] = round(m, 4); lf[nm + "_lo"] = round(lo, 4); lf[nm + "_hi"] = round(hi, 4)
        comp = "; ".join(f"{NICE[f]}={oof_best[f][1].split(':')[0]}" for f in feats)
        fus_rows.append(dict(cohort=c, kind="LATE_FUSION (best-per-feat)", components=comp, **lf))
        ef = [r for r in rows if r["cohort"] == c and r["feature"] == "EARLY_FUSION"]
        ebest = max(ef, key=lambda r: r["macroAUROC"])
        fus_rows.append(dict(cohort=c, kind="EARLY_FUSION (best pipeline)", components=ebest["model"],
                             **{k: ebest[k] for k in sum([[n, n + "_lo", n + "_hi"] for n in names], [])}))
        print(f"  {c} LATE_FUSION  mAUROC={lf['macroAUROC']:.3f} [{lf['macroAUROC_lo']:.3f},{lf['macroAUROC_hi']:.3f}]  [{comp}]", flush=True)
        pd.DataFrame(fus_rows).to_csv(f"{TB}/Tab_fusion_tuned.tsv", sep="\t", index=False)

    T = pd.DataFrame(rows); T.to_csv(f"{TB}/Tab_model_comparison_tuned.tsv", sep="\t", index=False)
    FUS = pd.DataFrame(fus_rows); FUS.to_csv(f"{TB}/Tab_fusion_tuned.tsv", sep="\t", index=False)
    figure(T, FUS, feat_groups)
    print(f"\n[tuned-benchmark] -> {TB}/Tab_model_comparison_tuned.tsv, Tab_fusion_tuned.tsv, Tab_best_hparams.tsv + figure")


def figure(T, FUS, feat_groups):
    gnice = [NICE[m] for m in MODS] + ["Early fusion\n(all feat)"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 10.5))
    for ax, c in zip(axes, COH):
        d = T[T.cohort == c]; x = np.arange(len(feat_groups)); w = 0.16
        for j, model in enumerate(MODELS):
            vals, los, his = [], [], []
            for g in feat_groups:
                rr = d[(d.feature == g) & (d.model == model)]
                if len(rr): vals.append(rr.macroAUROC.iloc[0]); los.append(rr.macroAUROC_lo.iloc[0]); his.append(rr.macroAUROC_hi.iloc[0])
                else: vals.append(np.nan); los.append(np.nan); his.append(np.nan)
            vals = np.array(vals); err = np.vstack([vals - np.array(los), np.array(his) - vals])
            ax.bar(x + (j - 2) * w, vals, w, yerr=err, capsize=1.5, error_kw=dict(lw=0.6), label=model, color=MCOL[model])
        lf = FUS[(FUS.cohort == c) & (FUS.kind.str.startswith("LATE"))]
        if len(lf):
            v = float(lf.macroAUROC.iloc[0]); ax.axhline(v, color="k", ls="--", lw=1.4)
            ax.text(len(feat_groups) - 0.5, v + 0.003, f"late fusion = {v:.3f} [{lf.macroAUROC_lo.iloc[0]:.3f},{lf.macroAUROC_hi.iloc[0]:.3f}]",
                    ha="right", va="bottom", fontsize=9, fontweight="bold")
        ax.axvline(len(MODS) - 0.5, color="grey", lw=0.8, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(gnice, fontsize=9); ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("macro-AUROC (20x5 CV, 95% CI)"); ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{c.upper()}: tuned model architecture per feature + early/late fusion", fontweight="bold")
        if c == "v1": ax.legend(frameon=False, ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.suptitle("Tuned model-architecture benchmark (inner-CV grid-search + 20x5 repeated CV)", fontweight="bold", y=0.995)
    fig.tight_layout(); fig.savefig(f"{FG}/Fig_model_comparison_tuned.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
