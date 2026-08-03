#!/usr/bin/env python
"""Shared helpers for the NC-readiness analysis program (Pillars 1-4)."""
import os, numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score

PROJ = "/home/jrkim/TSO_TFBS/project"
D = f"{PROJ}/results/auto_plan/blood_nested2"
FEAT = f"{PROJ}/results/auto_plan/feat"
OUT = f"{PROJ}/results/auto_plan/nc_readiness"
FG = f"{OUT}/figures"; TB = f"{OUT}/tables"
for d in (FG, TB): os.makedirs(d, exist_ok=True)
COH = ("v1", "v2")
MODS = ["E1_entropy", "All_exon_depth", "Genome_wide_CNA", "Somatic_mutation_profile", "SHAPE"]
NICE = {"E1_entropy": "Exon1 entropy", "All_exon_depth": "All-exon depth", "Genome_wide_CNA": "Broad CNA (>10Mb)",
        "Somatic_mutation_profile": "Somatic mutation", "SHAPE": "per-TFBS SHAPE"}
SEED = 42


def load_oof(c):
    z = np.load(f"{D}/{c}_oof.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def wnnls(Zc, t):
    w = (t == 1).mean(); sw = np.where(t == 1, 0.5 / max(w, 1e-6), 0.5 / max(1 - w, 1e-6)); s = np.sqrt(sw)[:, None]
    c, _ = nnls(np.column_stack([Zc, np.ones(len(Zc))]) * s, t * np.sqrt(sw)); return c[:-1]


def fuse_subset(z, mods):
    """Per-cancer OVR NNLS late fusion over the given modality OOF probabilities (matches blood_nested2)."""
    y = z["y"]; K = len(z["classes"]); P = np.zeros((len(y), K))
    for k in range(K):
        Zc = np.column_stack([z[f"P_{m}"][:, k] for m in mods]); w = wnnls(Zc, (y == k).astype(int))
        w = w / w.sum() if w.sum() > 1e-9 else np.full(len(mods), 1 / len(mods)); P[:, k] = Zc @ w
    P = np.clip(P, 1e-12, None); return P / P.sum(1, keepdims=True)


def onehot(y, K):
    Y = np.zeros((len(y), K)); Y[np.arange(len(y)), y] = 1; return Y


def macro_auc(P, y, K):
    aucs = []
    for k in range(K):
        yk = (y == k).astype(int)
        if yk.sum() == 0 or yk.sum() == len(yk): continue
        aucs.append(roc_auc_score(yk, P[:, k]))
    return float(np.mean(aucs)) if aucs else np.nan


def weighted_auc(P, y, K):
    aucs, ws = [], []
    for k in range(K):
        yk = (y == k).astype(int)
        if yk.sum() == 0 or yk.sum() == len(yk): continue
        aucs.append(roc_auc_score(yk, P[:, k])); ws.append(yk.sum())
    return float(np.average(aucs, weights=ws)) if aucs else np.nan


def macro_auprc(P, y, K):
    aps = []
    for k in range(K):
        yk = (y == k).astype(int)
        if yk.sum() == 0: continue
        aps.append(average_precision_score(yk, P[:, k]))
    return float(np.mean(aps)) if aps else np.nan


def topk_acc(P, y, k=1):
    return float(np.mean([y[i] in np.argsort(-P[i])[:k] for i in range(len(y))]))


def all_metrics(P, y, K):
    pred = P.argmax(1)
    return dict(macroAUROC=macro_auc(P, y, K), weightedAUROC=weighted_auc(P, y, K),
                macroAUPRC=macro_auprc(P, y, K), top1=topk_acc(P, y, 1), top2=topk_acc(P, y, 2),
                balAcc=float(balanced_accuracy_score(y, pred)),
                macroF1=float(f1_score(y, pred, average="macro")))


def ece_toplabel(P, y, nbins=15):
    conf = P.max(1); pred = P.argmax(1); correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, nbins + 1); e = 0.0
    for b in range(nbins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0: continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def classwise_ece(P, y, K, nbins=15):
    Y = onehot(y, K); es = []
    bins = np.linspace(0, 1, nbins + 1)
    for k in range(K):
        e = 0.0
        for b in range(nbins):
            m = (P[:, k] > bins[b]) & (P[:, k] <= bins[b + 1])
            if m.sum() == 0: continue
            e += m.mean() * abs(Y[m, k].mean() - P[m, k].mean())
        es.append(e)
    return float(np.mean(es))


def brier(P, y, K): return float(np.mean(np.sum((P - onehot(y, K)) ** 2, axis=1)))
def nll(P, y): return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


# ---- label + SHAPE feature loaders (for Pillars 3-4) ----
MANIFEST = f"{PROJ}/results/auto_plan/manifest_dev.tsv"
IMMUNE = f"{PROJ}/results/annotations/tfbs_unique_site_immune_annotation.tsv"
MIN_CLASS = 20


def load_labels(c):
    """Return dict sid->cancer_type for cohort c (string sids)."""
    m = pd.read_csv(MANIFEST, sep="\t", dtype=str); m = m[m.cohort == c]
    return dict(zip(m.sid.astype(str), m.cancer_type))


def _align(Xz_path, c):
    z = np.load(Xz_path, allow_pickle=True)
    sids = [str(s) for s in z["sids"]]; cols = [str(x) for x in z["cols"]]
    lab = load_labels(c)
    keep = [i for i, s in enumerate(sids) if s in lab]
    X = np.nan_to_num(z["X"][keep].astype(float)); ks = [sids[i] for i in keep]
    yt = np.array([lab[s] for s in ks])
    # restrict to classes with n>=MIN_CLASS
    vc = pd.Series(yt).value_counts(); classes = sorted(vc[vc >= MIN_CLASS].index)
    m2 = np.isin(yt, classes)
    X = X[m2]; yt = yt[m2]; ks = [ks[i] for i in range(len(ks)) if m2[i]]
    y = np.array([classes.index(t) for t in yt])
    return X, np.array(cols, dtype=object), ks, y, classes


def load_persite_shape(c): return _align(f"{FEAT}/{c}/X_SHAPE.npz", c)        # cols = chr_center
def load_perTF_shape(c):   return _align(f"{FEAT}/{c}/X_SHAPE_perTF.npz", c)   # NOTE: combined fragshape matrix


def perTF_from_sites(c):
    """Clean per-TF SHAPE matrix: aggregate the per-SITE short-fraction matrix to per-TF (mean of a TF's sites),
    using the site->TF annotation. Avoids the contaminated X_SHAPE_perTF.npz. Returns (X_perTF, tf_list, sids, y, classes)."""
    X, cols, sids, y, classes = load_persite_shape(c)
    a = site_annotation(); keys = [str(k) for k in cols]
    site_tf = a["TF"].reindex(keys).values
    idx = {}
    for j, t in enumerate(site_tf):
        if isinstance(t, str): idx.setdefault(t, []).append(j)
    tf_list = sorted(idx)
    Xtf = np.column_stack([X[:, idx[t]].mean(1) for t in tf_list])
    return Xtf, tf_list, sids, y, classes


_ANN = None
def site_annotation():
    """Per-site annotation indexed by 'chr_center' -> TF, edge_dist, immune(bool)."""
    global _ANN
    if _ANN is None:
        a = pd.read_csv(IMMUNE, sep="\t", dtype=str); a["key"] = a["chr"] + "_" + a["center"]
        a["edge_dist"] = pd.to_numeric(a["edge_dist"], errors="coerce")
        a["immune"] = a["immune_overlap_pm50bp"].isin(["True", "TRUE", "1"])
        _ANN = a.drop_duplicates("key").set_index("key")
    return _ANN
