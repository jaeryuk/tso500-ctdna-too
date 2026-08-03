#!/usr/bin/env python
"""GRAIL external tissue-of-origin validation.

The TSO500 model has no non-cancer class, so the primary endpoint is three-class
TOO among the GRAIL cancer samples overlapping TSO500 labels:

  lung cancer, breast cancer, prostate cancer

Non-cancer GRAIL controls are scored separately as forced cancer-class probabilities;
those rows are not used as negatives for AUROC because the trained classifier is not a
cancer-detection model.

Outputs:
  results/auto_plan/grail/external_metrics.tsv
  results/auto_plan/grail/external_perclass.tsv
  results/auto_plan/grail/external_predictions.tsv
  results/auto_plan/grail/external_control_scores.tsv
  results/auto_plan/grail/external_feature_summary.tsv
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJ = "/home/jrkim/TSO_TFBS/project"
FEAT = f"{PROJ}/results/auto_plan/feat"
GFEAT = f"{PROJ}/results/auto_plan/grail/feat"
OUT = f"{PROJ}/results/auto_plan/grail"
CLASSES = ["lung cancer", "breast cancer", "prostate cancer"]
METHODS = ["Genome_wide_CNA", "Mutation_signature", "Somatic_mutation_profile"]
SEED = 42


def log(msg):
    print(msg, flush=True)


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z["X"].astype(np.float32), [str(s) for s in z["sids"]], [str(c) for c in z["cols"]]


def safe_auc(ybin, score):
    if ybin.sum() == 0 or ybin.sum() == len(ybin):
        return np.nan
    return float(roc_auc_score(ybin, score))


def macro_auc(P, y):
    vals = [safe_auc((y == k).astype(int), P[:, k]) for k in range(len(CLASSES))]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def weighted_auc(P, y):
    vals = []
    weights = []
    for k in range(len(CLASSES)):
        v = safe_auc((y == k).astype(int), P[:, k])
        if np.isfinite(v):
            vals.append(v)
            weights.append(float((y == k).mean()))
    return float(np.average(vals, weights=weights)) if vals else np.nan


def topk_acc(P, y, k):
    order = np.argsort(-P, axis=1)[:, :k]
    return float(np.mean([y[i] in order[i] for i in range(len(y))]))


def clf_fit_predict(Xtrain, ytrain, Xtest):
    scaler = StandardScaler().fit(Xtrain)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="lbfgs")
    clf.fit(scaler.transform(Xtrain), ytrain)
    pred = np.zeros((len(Xtest), len(CLASSES)), dtype=np.float64)
    pred[:, clf.classes_] = clf.predict_proba(scaler.transform(Xtest))
    pred = np.clip(pred, 1e-12, None)
    return pred / pred.sum(axis=1, keepdims=True)


def oof_predict(X, y):
    P = np.zeros((len(y), len(CLASSES)), dtype=np.float64)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        P[te] = clf_fit_predict(X[tr], y[tr], X[te])
    return P


def shared_columns(method):
    gpath = f"{GFEAT}/X_{method}.npz"
    paths = {cohort: f"{FEAT}/{cohort}/X_{method}.npz" for cohort in ["v1", "v2"]}
    if not os.path.exists(gpath):
        log(f"[grail-ext] {method}: missing GRAIL matrix -> skip")
        return None
    if any(not os.path.exists(path) for path in paths.values()):
        log(f"[grail-ext] {method}: missing TSO v1/v2 matrix -> skip")
        return None
    _, _, gcols = load_npz(gpath)
    v1cols = set(load_npz(paths["v1"])[2])
    v2cols = set(load_npz(paths["v2"])[2])
    cols = [c for c in gcols if c in v1cols and c in v2cols]
    if not cols:
        log(f"[grail-ext] {method}: no shared columns -> skip")
        return None
    return cols


def load_tso(method, cols):
    man = pd.read_csv(f"{PROJ}/results/auto_plan/manifest_dev.tsv", sep="\t", dtype=str)
    man = man[man.cancer_type.isin(CLASSES)].copy()
    ymap = {c: i for i, c in enumerate(CLASSES)}
    rows = []
    y = []
    keys = []
    for cohort in ["v1", "v2"]:
        X, sids, all_cols = load_npz(f"{FEAT}/{cohort}/X_{method}.npz")
        sid_idx = {s: i for i, s in enumerate(sids)}
        col_idx = {c: i for i, c in enumerate(all_cols)}
        keep = [col_idx[c] for c in cols]
        sub = man[man.cohort == cohort]
        for r in sub.itertuples(index=False):
            if r.sid not in sid_idx:
                continue
            rows.append(X[sid_idx[r.sid], keep])
            y.append(ymap[r.cancer_type])
            keys.append(f"{cohort}:{r.sid}")
    return np.vstack(rows).astype(np.float32), np.array(y, dtype=int), keys


def load_grail(method, cols, gman):
    X, sids, all_cols = load_npz(f"{GFEAT}/X_{method}.npz")
    sid_idx = {s: i for i, s in enumerate(sids)}
    col_idx = {c: i for i, c in enumerate(all_cols)}
    keep_cols = [col_idx[c] for c in cols]
    row_ix = []
    rows = []
    for i, r in enumerate(gman.itertuples(index=False)):
        if r.sid in sid_idx:
            row_ix.append(i)
            rows.append(X[sid_idx[r.sid], keep_cols])
    return np.vstack(rows).astype(np.float32), np.array(row_ix, dtype=int)


def wnnls(Z, t):
    pos = max(float((t == 1).mean()), 1e-6)
    neg = max(1.0 - pos, 1e-6)
    sw = np.where(t == 1, 0.5 / pos, 0.5 / neg)
    Zc = np.column_stack([Z, np.ones(len(Z))])
    coef, _ = nnls(Zc * np.sqrt(sw)[:, None], t * np.sqrt(sw))
    return coef[:-1]


def fit_fusion(oof_by_method, y):
    methods = list(oof_by_method)
    K = len(CLASSES)
    global_Z = []
    global_t = []
    for k in range(K):
        global_Z.append(np.column_stack([oof_by_method[m][:, k] for m in methods]))
        global_t.append((y == k).astype(int))
    wg = wnnls(np.vstack(global_Z), np.concatenate(global_t))
    wg = wg / wg.sum() if wg.sum() > 1e-12 else np.full(len(methods), 1.0 / len(methods))
    counts = np.bincount(y, minlength=K)
    alpha = np.clip(counts / (counts + 60.0), 0.0, 0.8)
    W = np.zeros((K, len(methods)), dtype=np.float64)
    for k in range(K):
        Z = np.column_stack([oof_by_method[m][:, k] for m in methods])
        wk = wnnls(Z, (y == k).astype(int))
        wk = wk / wk.sum() if wk.sum() > 1e-12 else wg.copy()
        W[k] = alpha[k] * wk + (1.0 - alpha[k]) * wg
        W[k] = W[k] / W[k].sum()
    return W, wg, methods


def fuse(prob_by_method, W, methods):
    n = len(next(iter(prob_by_method.values())))
    P = np.zeros((n, len(CLASSES)), dtype=np.float64)
    for k in range(len(CLASSES)):
        Z = np.column_stack([prob_by_method[m][:, k] for m in methods])
        P[:, k] = Z @ W[k]
    P = np.clip(P, 1e-12, None)
    return P / P.sum(axis=1, keepdims=True)


def add_metric_rows(name, P, y, n_train, n_cols, n_controls, control_scores, metrics_rows, perclass_rows):
    pred = P.argmax(axis=1)
    metrics_rows.append({
        "method": name,
        "n_tso_train": int(n_train),
        "n_grail_cancer_eval": int(len(y)),
        "n_grail_controls_scored": int(n_controls),
        "n_shared_cols": int(n_cols),
        "top1_acc": round(float((pred == y).mean()), 4),
        "top2_acc": round(topk_acc(P, y, 2), 4),
        "macro_AUROC": round(macro_auc(P, y), 4),
        "weighted_AUROC": round(weighted_auc(P, y), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
        "control_median_max_cancer_prob": round(float(np.median(control_scores)), 4) if len(control_scores) else np.nan,
        "control_p90_max_cancer_prob": round(float(np.percentile(control_scores, 90)), 4) if len(control_scores) else np.nan,
        "control_frac_max_prob_lt_0.50": round(float((control_scores < 0.5).mean()), 4) if len(control_scores) else np.nan,
    })
    for k, cls in enumerate(CLASSES):
        mask = y == k
        call = pred == k
        perclass_rows.append({
            "method": name,
            "cancer_type": cls,
            "n": int(mask.sum()),
            "AUROC": round(safe_auc(mask.astype(int), P[:, k]), 4),
            "sensitivity_top1": round(float((pred[mask] == k).mean()), 4) if mask.sum() else np.nan,
            "PPV_top1": round(float((y[call] == k).mean()), 4) if call.sum() else np.nan,
        })


def main():
    gman = pd.read_csv(f"{OUT}/manifest.tsv", sep="\t", dtype=str)
    if "cancer_type" not in gman.columns:
        raise SystemExit("GRAIL manifest is missing cancer_type column")
    gman = gman[gman.cancer_type.isin(CLASSES + ["non-cancer"])].reset_index(drop=True)
    ymap = {c: i for i, c in enumerate(CLASSES)}
    log("[grail-ext] GRAIL labels: " + str(gman.cancer_type.value_counts().to_dict()))

    method_data = {}
    feature_rows = []
    for method in METHODS:
        cols = shared_columns(method)
        if cols is None:
            continue
        Xt, yt, tkeys = load_tso(method, cols)
        Xg, grow = load_grail(method, cols, gman)
        if len(np.unique(yt)) < len(CLASSES):
            log(f"[grail-ext] {method}: TSO training lacks all classes -> skip")
            continue
        Poof = oof_predict(Xt, yt)
        Pg = clf_fit_predict(Xt, yt, Xg)
        method_data[method] = {
            "cols": cols, "Xt": Xt, "yt": yt, "tkeys": tkeys, "Poof": Poof,
            "Xg": Xg, "grow": grow, "Pg": Pg,
        }
        feature_rows.append({
            "method": method,
            "n_shared_cols": len(cols),
            "n_tso_train": len(yt),
            "n_grail_scored": len(grow),
            "tso_oof_macro_AUROC": round(macro_auc(Poof, yt), 4),
        })
        log(f"[grail-ext] {method}: train={len(yt)} grail={len(grow)} cols={len(cols)} tso_oof_macro={macro_auc(Poof, yt):.4f}")

    if not method_data:
        raise SystemExit("No usable GRAIL feature matrices were available")

    metrics_rows = []
    perclass_rows = []
    pred_tables = []
    control_tables = []

    for method, d in method_data.items():
        grow = d["grow"]
        cancer_mask = gman.cancer_type.iloc[grow].isin(CLASSES).to_numpy()
        control_mask = gman.cancer_type.iloc[grow].eq("non-cancer").to_numpy()
        y = gman.cancer_type.iloc[grow[cancer_mask]].map(ymap).to_numpy(dtype=int)
        P = d["Pg"][cancer_mask]
        control_scores = d["Pg"][control_mask].max(axis=1) if control_mask.any() else np.array([])
        add_metric_rows(method, P, y, len(d["yt"]), len(d["cols"]), int(control_mask.sum()),
                        control_scores, metrics_rows, perclass_rows)
        pt = gman.iloc[grow].copy()
        pt["method"] = method
        for k, cls in enumerate(CLASSES):
            pt[f"prob_{cls.replace(' ', '_')}"] = d["Pg"][:, k]
        pt["predicted_cancer_type"] = [CLASSES[i] for i in d["Pg"].argmax(axis=1)]
        pt["max_cancer_prob"] = d["Pg"].max(axis=1)
        pred_tables.append(pt)
        control_tables.append(pt[pt.cancer_type.eq("non-cancer")].copy())

    common_train = sorted(set.intersection(*[set(d["tkeys"]) for d in method_data.values()]))
    common_grail = sorted(set.intersection(*[set(map(int, d["grow"])) for d in method_data.values()]))
    if len(method_data) >= 2 and common_train and common_grail:
        y_by_key = {}
        oof_by_method = {}
        for method, d in method_data.items():
            key_pos = {k: i for i, k in enumerate(d["tkeys"])}
            ix = [key_pos[k] for k in common_train]
            oof_by_method[method] = d["Poof"][ix]
            for k, i in zip(common_train, ix):
                y_by_key[k] = int(d["yt"][i])
        y_fuse_train = np.array([y_by_key[k] for k in common_train], dtype=int)
        W, wg, fuse_methods = fit_fusion(oof_by_method, y_fuse_train)
        pd.DataFrame(W, index=CLASSES, columns=fuse_methods).to_csv(f"{OUT}/external_fusion_weights.tsv", sep="\t")

        prob_by_method = {}
        for method, d in method_data.items():
            grow_pos = {int(ix): i for i, ix in enumerate(d["grow"])}
            ix = [grow_pos[i] for i in common_grail]
            prob_by_method[method] = d["Pg"][ix]
        Pg_fuse = fuse(prob_by_method, W, fuse_methods)
        common_grail_arr = np.array(common_grail, dtype=int)
        cancer_mask = gman.cancer_type.iloc[common_grail_arr].isin(CLASSES).to_numpy()
        control_mask = gman.cancer_type.iloc[common_grail_arr].eq("non-cancer").to_numpy()
        y = gman.cancer_type.iloc[common_grail_arr[cancer_mask]].map(ymap).to_numpy(dtype=int)
        P = Pg_fuse[cancer_mask]
        control_scores = Pg_fuse[control_mask].max(axis=1) if control_mask.any() else np.array([])
        add_metric_rows("FUSION_NNLS", P, y, len(common_train),
                        sum(len(method_data[m]["cols"]) for m in fuse_methods),
                        int(control_mask.sum()), control_scores, metrics_rows, perclass_rows)

        pt = gman.iloc[common_grail_arr].copy()
        pt["method"] = "FUSION_NNLS"
        for k, cls in enumerate(CLASSES):
            pt[f"prob_{cls.replace(' ', '_')}"] = Pg_fuse[:, k]
        pt["predicted_cancer_type"] = [CLASSES[i] for i in Pg_fuse.argmax(axis=1)]
        pt["max_cancer_prob"] = Pg_fuse.max(axis=1)
        pred_tables.append(pt)
        control_tables.append(pt[pt.cancer_type.eq("non-cancer")].copy())
        log(f"[grail-ext] FUSION_NNLS: train={len(common_train)} grail={len(common_grail)} methods={','.join(fuse_methods)} macro={macro_auc(P, y):.4f}")

    pd.DataFrame(metrics_rows).to_csv(f"{OUT}/external_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(perclass_rows).to_csv(f"{OUT}/external_perclass.tsv", sep="\t", index=False)
    pd.DataFrame(feature_rows).to_csv(f"{OUT}/external_feature_summary.tsv", sep="\t", index=False)
    pd.concat(pred_tables, ignore_index=True).to_csv(f"{OUT}/external_predictions.tsv", sep="\t", index=False)
    pd.concat(control_tables, ignore_index=True).to_csv(f"{OUT}/external_control_scores.tsv", sep="\t", index=False)
    log("[grail-ext] DONE -> results/auto_plan/grail/external_*.tsv")


if __name__ == "__main__":
    main()
