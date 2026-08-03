#!/usr/bin/env python3
"""RULE-CONFORMANT late-fusion ensemble confusion matrix, restricted to tumor fraction > TFCUT (default 0.10).
The frozen RC OOF (benchmark_nested_v2/{coh}_oof.npz) stores y/primary but NOT sample IDs, so we RECONSTRUCT
the benchmark sample ORDER cheaply (from the sids of the same feature npz that rc_benchmark_nested_v2.order_y
intersects), ASSERT the reconstructed y/classes match the frozen oof.npz (guarantees row alignment), then map
each row to its tumor fraction (rcv{1,2}_cna/cna_feature_table.tsv, keyed by num_sid) and filter TF>TFCUT.
`primary` = shrink60 NNLS-shrinkage Super-Learner (late-fusion ensemble) OOF; argmax vs y. Row-normalised
K x K confusion (true rows present in the subset; predicted columns = all K). SUPERSEDES the 2026-07-14
old-cohort Tab_confusion_ensemble_tf10.tsv.
Usage: fig_confusion_ensemble_rc_tf10_extract.py [TFCUT=0.10] [SUF=tf10]
Out: results/plot/Tab_confusion_ensemble_{SUF}.tsv (+ Tab_confusion_shape_{SUF}.tsv)."""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_benchmark_nested_v2 as B

PROJ = "/home/jrkim/TSO_TFBS/project"; OOFD = f"{PROJ}/results/rule_conformant/benchmark_nested_v2"
PLOT = f"{PROJ}/results/plot"; FEAT = B.FEAT
CNA = {"v1": f"{PROJ}/results/rcv1_cna/cna_feature_table.tsv",
       "v2": f"{PROJ}/results/rcv2_cna/cna_feature_table.tsv"}
TFCUT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
SUF = sys.argv[2] if len(sys.argv) > 2 else "tf10"
SHAPE_MOD = "mod_SHAPE_nep300"


def sids_of(coh, npz):
    return [str(s) for s in np.load(f"{FEAT}/{coh}/{npz}", allow_pickle=True)["sids"]]


def reconstruct_order(coh):
    """Replicate rc_benchmark_nested_v2.order_y's order/y/classes using ONLY the sids (no heavy matrix loads)."""
    lab = B.labels(coh)
    common = set(lab) & set(sids_of(coh, "X_SHAPE.npz"))
    for m in B.FILE_MODULES: common &= set(sids_of(coh, f"X_{m}.npz"))
    cnt = {}
    for s in common: cnt[lab[s]] = cnt.get(lab[s], 0) + 1
    classes = sorted([c for c, n in cnt.items() if n >= B.MIN_CLASS]); ci = {c: i for i, c in enumerate(classes)}
    order = sorted([s for s in common if lab[s] in ci]); y = np.array([ci[lab[s]] for s in order])
    return order, y, classes


def cap(s): return s[:1].upper() + s[1:]


def disp_round(v):
    h = v * 100.0; f = np.floor(h).astype(int); need = int(round(100.0 * v.sum())) - int(f.sum())
    order = np.argsort(-(h - f)); f[order[:max(need, 0)]] += 1
    return f / 100.0


def long_rows(pred, y, mask, classes, K, c):
    cm = confusion_matrix(y[mask], pred[mask], labels=range(K)).astype(float)
    rowsum = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, rowsum, out=np.zeros_like(cm), where=rowsum > 0)
    out = []
    for i in [i for i in range(K) if rowsum[i, 0] > 0]:
        disp = disp_round(cmn[i])
        for j in range(K):
            out.append(dict(cohort=c, true=cap(classes[i]), pred=cap(classes[j]),
                            value=round(float(cmn[i, j]), 4), disp=round(float(disp[j]), 2), n=int(rowsum[i, 0])))
    return out


def main():
    ens, shp = [], []
    for c in ("v1", "v2"):
        z = np.load(f"{OOFD}/{c}_oof.npz", allow_pickle=True)
        y_oof = np.asarray(z["y"]); classes = [str(x) for x in z["classes"]]; K = len(classes)
        order, y, cls2 = reconstruct_order(c)
        assert cls2 == classes and np.array_equal(y, y_oof), \
            f"[{c}] order reconstruction mismatch (classes {cls2==classes}, y {np.array_equal(y, y_oof)})"
        tfser = pd.read_csv(CNA[c], sep="\t", index_col=0)["tumor_fraction"]; tfser.index = tfser.index.astype(str)
        tf = np.array([tfser.get(s, np.nan) for s in order], float)
        mask = np.isfinite(tf) & (tf > TFCUT)
        ens += long_rows(z["primary"].argmax(1), y, mask, classes, K, c)
        if SHAPE_MOD in z.files: shp += long_rows(z[SHAPE_MOD].argmax(1), y, mask, classes, K, c)
        print(f"[{c}] TFCUT={TFCUT}: {int(mask.sum())}/{len(y)} samples  ({int(np.isnan(tf).sum())} TF-missing)", flush=True)
    pd.DataFrame(ens).to_csv(f"{PLOT}/Tab_confusion_ensemble_{SUF}.tsv", sep="\t", index=False)
    if shp: pd.DataFrame(shp).to_csv(f"{PLOT}/Tab_confusion_shape_{SUF}.tsv", sep="\t", index=False)
    print(f"WROTE Tab_confusion_ensemble_{SUF}.tsv" + (f" + Tab_confusion_shape_{SUF}.tsv" if shp else ""), flush=True)
    print("CONFUSION_RC_TF_DONE", flush=True)


if __name__ == "__main__":
    main()
