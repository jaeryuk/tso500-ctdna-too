#!/usr/bin/env python
"""LOCKED full-rigor multimodal TOO benchmark (user directive 2026-07-12, addressing methodological review).

THREE-LEVEL NESTING (leakage-controlled end to end):
  OUTER : RepeatedStratifiedKFold(5 x N_REPEATS)      -> the ONLY thing scored; outer-test never touched
    within each outer-train:
      META  : StratifiedKFold(META_K)                 -> cross-fit base learners => clean train-OOF for the stacker
        INNER : GridSearchCV(cv=3, scoring=macro-AUROC) -> hyperparameter tuning, per meta-train fold
      -> fit NNLS-shrinkage combiner on train-OOF
      -> refit base learners on FULL outer-train (inner-3 tuned)
      -> predict untouched outer-test; apply the FROZEN combiner
  => nothing that touches an outer-test sample was fit using it (base models, tuning, or combiner weights).

Fixes vs the preliminary (rc_benchmark_nested):
  * Pipeline([StandardScaler, LogReg]) INSIDE GridSearchCV -> scaler refits per inner fold (no inner-fold leak).
  * fully nested cross-fitted stacking (above) -> removes the subtle CV-on-OOF combiner leak.
  * convergence tracked (n_iter_ captured; converged fraction reported); warnings NOT globally suppressed.
  * class-STRATIFIED patient-level bootstrap CI.
  * shrinkage constant kept at 60 (pre-specified from prior work) + {30,60,100} sensitivity.
  * combiner ablations on IDENTICAL folds: best-single / equal-weight / global-NNLS / per-class no-shrink /
    per-class shrink(30,60,100) / leave-one-module-out.
  * metric panel: macro-AUROC, log-loss, multiclass Brier, top-1 acc, macro-recall, per-class AUROC, confusion.
Base learners tuned by macro-AUROC (roc_auc_ovr); the NNLS combiner minimizes CV squared-probability error
(Brier-type) under non-negativity -> named "Shrinkage NNLS Super-Learner (custom)".
enet class_weight=balanced (primary); xgb unweighted (better-calibrated probs for the NNLS input).
Crash-safe: phase-1 base predictions pickled per cohort so the multi-day compute survives restarts.
Out: results/rule_conformant/benchmark_nested_v2/
"""
import os, json, pickle, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, log_loss, recall_score, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.optimize import nnls
from joblib import Parallel, delayed
import xgboost as xgb

PROJ = "/home/jrkim/TSO_TFBS/project"; RC = f"{PROJ}/results/rule_conformant"
FEAT = f"{PROJ}/results/auto_plan/feat_rc"; MAN = f"{RC}/manifest_dev.tsv"
OUT = os.environ.get("OUT_DIR", f"{RC}/benchmark_nested_v2"); os.makedirs(OUT, exist_ok=True)
MIN_CLASS = 20; SEED = 42
N_SPLITS = 5; N_REPEATS = int(os.environ.get("N_REPEATS", "20")); META_K = int(os.environ.get("META_K", "5"))
INNER = 3; MAXIT = int(os.environ.get("MAXIT", "1500")); TOLV = 1e-3
ENET_CS = [0.03, 0.1, 0.3, 1.0]
ENET_L1 = [float(x) for x in os.environ.get("ENET_L1", "0.5").split(",")]  # fix l1_ratio=0.5, tune C only (3x faster; C is the dominant knob)
XGB_GRID = {"max_depth": [3, 4], "learning_rate": [0.03, 0.1], "n_estimators": [300, 600]}
NJOB = int(os.environ.get("NJOB", "72")); NBOOT = int(os.environ.get("NBOOT", "2000"))
TCRIT = 2.093 if N_REPEATS == 20 else 2.262
COHORTS = tuple(c for c in os.environ.get("COHORTS", "v1,v2").split(",") if c)
# ENSEMBLE members. File-backed modules load from X_<m>.npz; the SHAPE member is the CONSTRUCTED variant
# SHAPE_nep300 (all-frag endpoint-Gini + length-entropy over nep>300 sites), built on the fly from
# X_SHAPE.npz channels (there is no X_SHAPE_nep300.npz file). SHAPE(primary)/SHAPE_long/SHAPE_long_nep300 dropped
# (user directive 2026-07-13: keep only SHAPE_nep300, and use it as the ensemble SHAPE module).
FILE_MODULES = [m for m in os.environ.get(
    "FILE_MODULES", "Genome_wide_CNA,E1_entropy,Somatic_mutation_profile,All_exon_depth").split(",") if m]
SHAPE_MODULE = os.environ.get("SHAPE_MODULE", "SHAPE_nep300")     # constructed SHAPE variant used in the ensemble
MODULES = FILE_MODULES + ([SHAPE_MODULE] if SHAPE_MODULE else [])
LEARNER = {"E1_entropy": "enet", "All_exon_depth": "enet", SHAPE_MODULE: "enet",
           "Genome_wide_CNA": "xgb",
           "Somatic_mutation_profile": os.environ.get("MUT_LEARNER", "xgb")}  # env-overridable for the xgb/enet bake-off
# eval-only SHAPE variants (module-level nested CV, EXCLUDED from the ensemble); empty by default now.
EVAL_MODULES = [m for m in os.environ.get("EVAL_MODULES", "").split(",") if m]
for _m in EVAL_MODULES: LEARNER[_m] = "enet"
ALLMODS = MODULES + EVAL_MODULES
PADD = f"{PROJ}/results/auto_plan/padgini/npz"; REG = f"{RC}/centers_edge0_regions.tsv"; IPAD = 6; NEP_MIN = 300
# nep>300 support gate mode. 'percell' (default 2026-07-21): PER-SAMPLE PER-SITE gate — keep a SHAPE cell iff THAT
# sample's endpoint support at that site > NEP_MIN, else NaN (imputed per-fold in the pipeline). No cohort-level
# site selection -> outer-test support never informs the feature set. 'cohort' = OLD leaky cohort-median site subset.
NEP_MODE = os.environ.get("NEP_MODE", "percell")
SHRINK_CONSTS = [60]; SHRINK_PRIMARY = 60   # λ fixed at 60 (user 2026-07-21); sensitivity {30,60,100} was a wash (<0.001) — dropped


def make_estimator(learner, K):
    if learner == "enet":
        pipe = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),  # per-fold TRAIN median for percell-NaN cells
                         ("sc", StandardScaler()),
                         ("clf", LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced",
                                                    max_iter=MAXIT, tol=TOLV))])
        grid = {"clf__C": ENET_CS, "clf__l1_ratio": ENET_L1}
        base = pipe
    else:
        base = xgb.XGBClassifier(objective="multi:softprob", num_class=K, tree_method="hist", device="cpu",
                                 subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, min_child_weight=2,
                                 eval_metric="mlogloss", verbosity=0, n_jobs=1)
        grid = XGB_GRID
    return GridSearchCV(base, grid, cv=StratifiedKFold(INNER, shuffle=True, random_state=SEED + 1),
                        scoring="roc_auc_ovr", n_jobs=int(os.environ.get("GRID_NJOBS", "1")),
                        refit=True, error_score="raise")


def fit_predict(Xm, y, fit_g, pred_g, learner, K):
    """fit a tuned base learner on fit_g, return (predict_proba on pred_g mapped to K cols, max n_iter for enet)."""
    est = make_estimator(learner, K); est.fit(Xm[fit_g], y[fit_g])
    P = np.zeros((len(pred_g), K)); P[:, est.classes_] = est.predict_proba(Xm[pred_g])
    niter = -1
    if learner == "enet":
        clf = est.best_estimator_.named_steps["clf"]
        niter = int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else -1
    return P, niter


# ------------------------------- combiner (NNLS + shrinkage) -------------------------------
def _norm(P):
    P = np.clip(P, 1e-12, None); return P / P.sum(1, keepdims=True)


def wnnls(Z, t):
    w = (t == 1).mean(); sw = np.where(t == 1, 0.5 / max(w, 1e-6), 0.5 / max(1 - w, 1e-6))
    s = np.sqrt(sw)[:, None]; Zc = np.column_stack([Z, np.ones(len(Z))])
    c, _ = nnls(Zc * s, t * np.sqrt(sw)); return c[:-1]


def global_weights(oof_tr, y_tr, K, mods):
    Zg = []; tg = []
    for k in range(K):
        Zg.append(np.column_stack([oof_tr[m][:, k] for m in mods])); tg.append((y_tr == k).astype(int))
    wg = wnnls(np.vstack(Zg), np.concatenate(tg))
    return wg / wg.sum() if wg.sum() > 1e-9 else np.full(len(mods), 1 / len(mods))


def shrink_fit(oof_tr, y_tr, K, mods, const):
    """per-class NNLS weights shrunk toward the global convex weights; alpha_k = n_k/(n_k+const).
    const=0 -> pure per-class (no shrink); const=inf -> pure global."""
    nm = len(mods); wg = global_weights(oof_tr, y_tr, K, mods)
    cnt = np.bincount(y_tr, minlength=K)
    alpha = np.ones(K) if const == 0 else (np.zeros(K) if np.isinf(const) else np.clip(cnt / (cnt + const), 0, 1))
    Wk = []
    for k in range(K):
        if alpha[k] <= 0:
            Wk.append(wg.copy()); continue
        Z = np.column_stack([oof_tr[m][:, k] for m in mods]); wk = wnnls(Z, (y_tr == k).astype(int))
        wk = wk / wk.sum() if wk.sum() > 1e-9 else wg.copy()
        W = alpha[k] * wk + (1 - alpha[k]) * wg; Wk.append(W / max(W.sum(), 1e-9))
    return Wk, wg


def shrink_apply(pred, Wk, K, mods):
    P = np.zeros((len(pred[mods[0]]), K))
    for k in range(K):
        Z = np.column_stack([pred[m][:, k] for m in mods]); P[:, k] = Z @ Wk[k]
    return _norm(P)


# ------------------------------- metrics -------------------------------
def macro_auc(P, y, K):
    return float(np.mean([roc_auc_score((y == k).astype(int), P[:, k]) for k in range(K) if 0 < (y == k).sum() < len(y)]))


def per_class_auc(P, y, K):
    return [round(float(roc_auc_score((y == k).astype(int), P[:, k])), 4) if 0 < (y == k).sum() < len(y) else None
            for k in range(K)]


def strat_boot(P, y, K):
    rng = np.random.RandomState(SEED); byc = [np.where(y == k)[0] for k in range(K)]; vals = []
    for _ in range(NBOOT):
        idx = np.concatenate([rng.choice(c, len(c), replace=True) for c in byc])
        try: vals.append(macro_auc(P[idx], y[idx], K))
        except Exception: pass
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


def metric_panel(P, y, K):
    pred = P.argmax(1); oh = np.eye(K)[y]
    return dict(macroAUROC=round(macro_auc(P, y, K), 4),
                logloss=round(float(log_loss(y, P, labels=list(range(K)))), 4),
                brier=round(float(np.mean(np.sum((P - oh) ** 2, 1))), 4),
                top1=round(float(accuracy_score(y, pred)), 4),
                macro_recall=round(float(recall_score(y, pred, average="macro")), 4))


def summ(aucs):
    a = np.array(aucs); m = a.mean(); sd = a.std(ddof=1)
    return dict(mean=round(float(m), 4), sd=round(float(sd), 4),
                stab_lo=round(float(m - TCRIT * sd / np.sqrt(len(a))), 4),
                stab_hi=round(float(m + TCRIT * sd / np.sqrt(len(a))), 4))


# --- Sex-chromosome exclusion for the CNA modules (user directive 2026-07-21) --------------------------------
# chrX/chrY copy number tracks patient SEX, which is confounded with tissue (breast~female, prostate~male), so a
# CNA module could classify via sex rather than tumour biology. Drop chrX/chrY features from BOTH CNA modules:
# broad Genome_wide_CNA (chrXp/chrXq arms) and focal All_exon_depth (exons of chrX/chrY genes, mapped via the
# panel manifest). Label-blind genomic-location filter applied as a fixed column mask -> no leakage.
DROP_SEXCHROM = os.environ.get("DROP_SEXCHROM", "1") == "1"
# Per-module control so a MIXED winner can be deployed (e.g. broad-excluded + focal-included). SEXCHROM_EXCL is the
# set of module names to strip chrX/chrY from; default = both CNA modules when DROP_SEXCHROM, else none. After the
# variant selection (rc_cna_sexchrom_select.py) set SEXCHROM_EXCL to the winning config for the ensemble re-run.
_DEFAULT_SEX = "Genome_wide_CNA,All_exon_depth" if DROP_SEXCHROM else ""
SEXCHROM_EXCL = set(m for m in os.environ.get("SEXCHROM_EXCL", _DEFAULT_SEX).split(",") if m)
MANIFEST_BED = os.environ.get("MANIFEST_BED", "/home/jrkim/TSO_TFBS/TST500C_manifest.bed")
import re as _re
_ARM_SEX_RE = _re.compile(r"^chr(X|Y)[pq]$")
_EXON2CHR = None


def _exon2chr():
    global _EXON2CHR
    if _EXON2CHR is None:
        _EXON2CHR = {}
        for ln in open(MANIFEST_BED):
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 4:
                _EXON2CHR[f[3]] = f[0]
    return _EXON2CHR


def sexchrom_keep(module, cols):
    """Boolean keep-mask dropping chrX/chrY features for the two CNA modules; all-True otherwise."""
    cols = [str(c) for c in cols]
    if module == "Genome_wide_CNA":                                   # arm cols: chr1p..chrXq (+ tumor_fraction, ploidy)
        return np.array([_ARM_SEX_RE.match(c) is None for c in cols])
    if module == "All_exon_depth":                                    # exon:GENE_ExonN_... -> chrom via manifest
        n2c = _exon2chr()
        return np.array([n2c.get(c[5:] if c.startswith("exon:") else c, "") not in ("chrX", "chrY", "X", "Y")
                         for c in cols])
    return np.ones(len(cols), bool)


def loadnpz(p, module=None):
    z = np.load(p, allow_pickle=True)
    X, sids = z["X"], [str(s) for s in z["sids"]]
    if module in SEXCHROM_EXCL and "cols" in z.files:
        keep = sexchrom_keep(module, z["cols"])
        if not keep.all():
            X = X[:, keep]
            print(f"  [sexchrom] {module}: dropped {int((~keep).sum())}/{len(keep)} chrX/chrY cols "
                  f"-> {int(keep.sum())} kept", flush=True)
    return X, sids


def labels(coh):
    return {l.split("\t")[0]: l.split("\t")[2] for l in open(MAN).read().splitlines()[1:] if l.split("\t")[1] == coh}


def reg_idx():
    d = {}
    for i, ln in enumerate(open(REG)):
        r = ln.rstrip("\n").split("\t"); d[f"{r[0]}_{(int(r[1]) + int(r[2])) // 2}"] = i
    return d


def median_nep(sids, cvp_idx):
    nn = []
    for s in sids:
        p = f"{PADD}/{s}.padgini.npz"
        if os.path.exists(p): nn.append(np.load(p, allow_pickle=True)["nep"][:, IPAD].astype(float)[cvp_idx])
    return np.nanmedian(np.array(nn), 0)


def per_sample_nep(sids, cvp_idx):
    """n x nsite matrix of EACH sample's own endpoint support (nep) at each site; NaN row if padgini missing."""
    out = []
    for s in sids:
        p = f"{PADD}/{s}.padgini.npz"
        if os.path.exists(p):
            out.append(np.load(p, allow_pickle=True)["nep"][:, IPAD].astype(float)[cvp_idx])
        else:
            out.append(np.full(len(cvp_idx), np.nan))
    return np.array(out)


def load_channels(coh, npz):
    z = np.load(f"{FEAT}/{coh}/{npz}", allow_pickle=True)
    cols = [str(c) for c in z["cols"]]; sids = [str(s) for s in z["sids"]]
    gi = [j for j, c in enumerate(cols) if c.startswith("gr:")]
    li = [j for j, c in enumerate(cols) if c.startswith("lenhzsp:")]
    return np.nan_to_num(z["X"][:, gi]), np.nan_to_num(z["X"][:, li]), sids, [cols[j][3:] for j in gi]


def order_y(coh):
    lab = labels(coh); mats = {m: loadnpz(f"{FEAT}/{coh}/X_{m}.npz", m) for m in FILE_MODULES}
    # SHAPE channels (all-frag endpoint-Gini + length-entropy) — source for the constructed SHAPE variant(s)
    gA, lnA, sidsS, sitesS = load_channels(coh, "X_SHAPE.npz")
    common = set(lab) & set(sidsS)                            # keep the X_SHAPE intersection the old SHAPE module gave
    for m in FILE_MODULES: common &= set(mats[m][1])
    cnt = {}
    for s in common: cnt[lab[s]] = cnt.get(lab[s], 0) + 1
    classes = sorted([c for c, nn in cnt.items() if nn >= MIN_CLASS]); ci = {c: i for i, c in enumerate(classes)}
    order = sorted([s for s in common if lab[s] in ci]); y = np.array([ci[lab[s]] for s in order])
    Xmods = {}
    for m in FILE_MODULES:
        X, sids = mats[m]; idx = {s: i for i, s in enumerate(sids)}
        Xmods[m] = np.vstack([X[idx[s]] for s in order]).astype(np.float64)
    # ---- constructed SHAPE variant(s), aligned to the SAME order; frag-gate x nep>300 site-filter ----
    need_shape = ([SHAPE_MODULE] if SHAPE_MODULE else []) + EVAL_MODULES
    if need_shape:
        iS = {s: i for i, s in enumerate(sidsS)}; rS = np.array([iS[s] for s in order], np.int64)
        gA, lnA = gA[rS], lnA[rS]
        cvp = np.array([reg_idx()[s] for s in sitesS], np.int64)
        if NEP_MODE == "cohort":                                      # OLD: cohort-median site subset (leaks outer-test support)
            hi = median_nep([sidsS[i] for i in rS], cvp) > NEP_MIN
            def gate(M): return M[:, hi]
            gate_info = f"cohort nep>300 sites={int(hi.sum())}/{len(sitesS)}"
        else:                                                         # PER-SAMPLE PER-SITE gate (default 2026-07-21)
            keep = per_sample_nep([sidsS[i] for i in rS], cvp) > NEP_MIN   # n x nsite; nan support -> False
            def gate(M): return np.where(keep, M, np.nan)             # ALL sites kept as cols; unreliable cells NaN -> per-fold median impute
            gate_info = f"percell all {len(sitesS)} sites, cell keep-rate={100*float(keep.mean()):.1f}%"
        # SHAPE ablation arms: the two SHAPE channels ALONE, on the SAME nep gate as SHAPE_nep300.
        var = {"SHAPE": lambda: np.hstack([gA, lnA]),
               "SHAPE_nep300": lambda: np.hstack([gate(gA), gate(lnA)]),
               "GINI_nep300": lambda: gate(gA),           # endpoint-Gini only
               "LENENT_nep300": lambda: gate(lnA)}        # fragment-length-entropy only
        if any("long" in m for m in need_shape):             # long-frag variants need X_SHAPE_long.npz
            gL, _, sidsL, sitesL = load_channels(coh, "X_SHAPE_long.npz")
            assert sitesS == sitesL, f"[{coh}] X_SHAPE vs X_SHAPE_long site order mismatch"
            iL = {s: i for i, s in enumerate(sidsL)}; gL = gL[np.array([iL[s] for s in order], np.int64)]
            var["SHAPE_long"] = lambda: np.hstack([gL, lnA])
            var["SHAPE_long_nep300"] = lambda: np.hstack([gate(gL), gate(lnA)])
        for m in need_shape:
            Xmods[m] = var[m]().astype(np.float64)
        print(f"[{coh}] SHAPE variants built {need_shape}: nep-gate={NEP_MODE} ({gate_info})", flush=True)
    return order, y, classes, Xmods


def run_cohort(coh):
    order, y, classes, Xmods = order_y(coh); K = len(classes); n = len(y)
    print(f"[{coh}] n={n} K={K} classes={classes}", flush=True)
    outer = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(np.zeros(n), y))

    p1 = f"{OUT}/{coh}_phase1.pkl"
    if os.path.exists(p1):
        with open(p1, "rb") as f: D = pickle.load(f)
        trainOOF, testpred, niters = D["trainOOF"], D["testpred"], D["niters"]
        print(f"[{coh}] phase-1 loaded from checkpoint", flush=True)
    else:
        specs = []
        for f_, (OTR, OTE) in enumerate(outer):
            meta = list(StratifiedKFold(META_K, shuffle=True, random_state=SEED + 1000 + f_).split(np.zeros(len(OTR)), y[OTR]))
            for m in MODULES:                                    # meta cross-fit ONLY for ensemble members
                for (mtr, mva) in meta:
                    specs.append((f_, m, "meta", mva, OTR[mtr], OTR[mva]))
            for m in ALLMODS:                                    # refit -> outer-test pred for ALL (incl eval-only)
                specs.append((f_, m, "refit", None, OTR, OTE))
        print(f"[{coh}] phase-1: {len(specs)} tasks (outer={len(outer)}; meta{META_K} x {len(MODULES)} ens-mods + "
              f"refit x {len(ALLMODS)} all-mods) NJOB={NJOB}", flush=True)
        res = Parallel(n_jobs=NJOB, prefer="processes", max_nbytes="20M", verbose=5)(
            delayed(fit_predict)(Xmods[m], y, fg, pg, LEARNER[m], K) for (f_, m, kind, pos, fg, pg) in specs)
        trainOOF = [{m: np.zeros((len(outer[f_][0]), K)) for m in MODULES} for f_ in range(len(outer))]
        testpred = [{m: None for m in ALLMODS} for f_ in range(len(outer))]
        niters = []
        for (f_, m, kind, pos, fg, pg), (P, ni) in zip(specs, res):
            if kind == "meta": trainOOF[f_][m][pos] = P
            else: testpred[f_][m] = P
            if ni >= 0: niters.append(ni)
        with open(p1, "wb") as f: pickle.dump(dict(trainOOF=trainOOF, testpred=testpred, niters=niters), f)
        print(f"[{coh}] phase-1 done + checkpointed", flush=True)

    conv = np.array(niters); nonconv = float(np.mean(conv >= MAXIT)) if len(conv) else 0.0
    print(f"[{coh}] enet convergence: {100*(1-nonconv):.1f}% converged (< max_iter={MAXIT}); "
          f"median n_iter={int(np.median(conv)) if len(conv) else 0}", flush=True)

    # -------- phase 2: assemble per-repeat OOF for every combiner variant + each module --------
    variants = (["best_single", "equal_weight", "global_nnls", "perclass_noshrink"] +
                [f"shrink{c}" for c in SHRINK_CONSTS] + [f"LOMO_drop_{m}" for m in MODULES])
    ens = {v: np.zeros((N_REPEATS, n, K)) for v in variants}
    modOOF = {m: np.zeros((N_REPEATS, n, K)) for m in ALLMODS}
    for f_, (OTR, OTE) in enumerate(outer):
        r = f_ // N_SPLITS; ytr = y[OTR]
        tp = testpred[f_]; to = trainOOF[f_]
        for m in ALLMODS: modOOF[m][r][OTE] = tp[m]
        # equal weight
        ens["equal_weight"][r][OTE] = _norm(np.mean([tp[m] for m in MODULES], 0))
        # best single (by train-OOF macro-AUROC on this outer-train)
        bm = max(MODULES, key=lambda m: macro_auc(to[m], ytr, K)); ens["best_single"][r][OTE] = _norm(tp[bm])
        # global nnls
        wg = global_weights(to, ytr, K, MODULES)
        ens["global_nnls"][r][OTE] = _norm(sum(wg[i] * tp[MODULES[i]] for i in range(len(MODULES))))
        # per-class no-shrink + shrink variants
        Wk0, _ = shrink_fit(to, ytr, K, MODULES, 0); ens["perclass_noshrink"][r][OTE] = shrink_apply(tp, Wk0, K, MODULES)
        for c in SHRINK_CONSTS:
            Wk, _ = shrink_fit(to, ytr, K, MODULES, c); ens[f"shrink{c}"][r][OTE] = shrink_apply(tp, Wk, K, MODULES)
        # leave-one-module-out (shrink primary)
        for drop in MODULES:
            mods = [m for m in MODULES if m != drop]
            Wk, _ = shrink_fit(to, ytr, K, mods, SHRINK_PRIMARY); ens[f"LOMO_drop_{drop}"][r][OTE] = shrink_apply(tp, Wk, K, mods)

    # -------- scoring --------
    summary = {"n": n, "K": K, "classes": classes, "convergence_frac": round(1 - nonconv, 4),
               "modules": {}, "ensemble": {}, "primary": f"shrink{SHRINK_PRIMARY}"}
    for m in ALLMODS:
        aucs = [macro_auc(modOOF[m][r], y, K) for r in range(N_REPEATS)]; pooled = _norm(modOOF[m].mean(0))
        blo, bhi = strat_boot(pooled, y, K)
        summary["modules"][m] = {**summ(aucs), "boot_lo": blo, "boot_hi": bhi, "learner": LEARNER[m],
                                 "eval_only": m in EVAL_MODULES,
                                 "panel": metric_panel(pooled, y, K), "perclass_auc": per_class_auc(pooled, y, K)}
    for v in variants:
        aucs = [macro_auc(ens[v][r], y, K) for r in range(N_REPEATS)]; pooled = _norm(ens[v].mean(0))
        blo, bhi = strat_boot(pooled, y, K)
        summary["ensemble"][v] = {**summ(aucs), "boot_lo": blo, "boot_hi": bhi, "panel": metric_panel(pooled, y, K)}
    # paired delta vs primary ensemble (patient-level stratified bootstrap on averaged OOF)
    prim = _norm(ens[f"shrink{SHRINK_PRIMARY}"].mean(0))
    summary["confusion_primary"] = confusion_matrix(y, prim.argmax(1)).tolist()
    np.savez(f"{OUT}/{coh}_oof.npz", y=y, classes=np.array(classes),
             primary=prim, **{f"mod_{m}": _norm(modOOF[m].mean(0)) for m in ALLMODS})
    json.dump(summary, open(f"{OUT}/{coh}_summary.json", "w"), indent=2)
    return summary


def write_tsv(all_):
    with open(f"{OUT}/benchmark_v2.tsv", "w") as f:
        f.write("cohort\ttype\tname\tlearner\tmacroAUROC\tsd\tstab_lo\tstab_hi\tboot_lo\tboot_hi\tlogloss\tbrier\ttop1\tmacro_recall\n")
        for coh, s in all_.items():
            for m, d in s["modules"].items():
                p = d["panel"]; f.write(f"{coh}\tmodule\t{m}\t{d['learner']}\t{d['mean']}\t{d['sd']}\t{d['stab_lo']}\t{d['stab_hi']}\t{d['boot_lo']}\t{d['boot_hi']}\t{p['logloss']}\t{p['brier']}\t{p['top1']}\t{p['macro_recall']}\n")
            for v, d in s["ensemble"].items():
                p = d["panel"]; f.write(f"{coh}\tensemble\t{v}\tnnls-SL\t{d['mean']}\t{d['sd']}\t{d['stab_lo']}\t{d['stab_hi']}\t{d['boot_lo']}\t{d['boot_hi']}\t{p['logloss']}\t{p['brier']}\t{p['top1']}\t{p['macro_recall']}\n")


def main():
    all_ = {}
    for coh in COHORTS:
        sp = f"{OUT}/{coh}_summary.json"
        all_[coh] = json.load(open(sp)) if os.path.exists(sp) else run_cohort(coh)
        write_tsv(all_)
        d = all_[coh]; print(f"[{coh}] primary ensemble (shrink{SHRINK_PRIMARY}) macroAUROC="
                              f"{d['ensemble'][f'shrink{SHRINK_PRIMARY}']['mean']}", flush=True)
    write_tsv(all_)
    print("\n=== FULL-RIGOR NESTED BENCHMARK v2 (3-level nesting; shrinkage NNLS SL; ablations + panel) ===", flush=True)
    with open(f"{OUT}/benchmark_v2.tsv") as f: print(f.read(), flush=True)
    print("RC_BENCH_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
