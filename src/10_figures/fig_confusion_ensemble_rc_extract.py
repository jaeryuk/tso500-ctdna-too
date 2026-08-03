#!/usr/bin/env python3
"""RULE-CONFORMANT late-fusion ensemble confusion matrix, ALL tumor fraction (no TF filter).
Reads the frozen RC nested-benchmark OOF results/rule_conformant/benchmark_nested_v2/{coh}_oof.npz:
`primary` = the shrink60 NNLS-shrinkage Super-Learner (late-fusion ensemble) mean-over-repeats OOF
probabilities; argmax vs the true labels y. Row-normalised K x K confusion, long format for the existing
ggpubr renderer (fig_confusion_ensemble_tf10_ggpubr.R). SUPERSEDES the 2026-07-14 Tab_confusion_ensemble_all.tsv
which was built on the OLD manuscript cohort (v1=1034 / v2=708, 7 v2 classes) — RC is v1=1093 / v2=796 (8 v2 classes).
Out: results/plot/Tab_confusion_ensemble_all.tsv (+ Tab_confusion_shape_all.tsv for the SHAPE module)."""
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix

PROJ = "/home/jrkim/TSO_TFBS/project"; OOFD = f"{PROJ}/results/rule_conformant/benchmark_nested_v2"
PLOT = f"{PROJ}/results/plot"
SHAPE_MOD = "mod_SHAPE_nep300"      # the SHAPE ensemble module OOF stored in the same npz


def cap(s): return s[:1].upper() + s[1:]


def disp_round(v):
    """Largest-remainder round a probability row to 2 decimals so the printed labels sum to exactly 1.00."""
    h = v * 100.0; f = np.floor(h).astype(int); need = int(round(100.0 * v.sum())) - int(f.sum())
    order = np.argsort(-(h - f))
    f[order[:max(need, 0)]] += 1
    return f / 100.0


def long_rows(pred, y, classes, K, c):
    cm = confusion_matrix(y, pred, labels=range(K)).astype(float)
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
        y = np.asarray(z["y"]); classes = [str(x) for x in z["classes"]]; K = len(classes)
        ens += long_rows(z["primary"].argmax(1), y, classes, K, c)
        if SHAPE_MOD in z.files:
            shp += long_rows(z[SHAPE_MOD].argmax(1), y, classes, K, c)
        print(f"[{c}] RC n={len(y)} K={K}  ensemble top1={np.mean(z['primary'].argmax(1)==y):.4f}", flush=True)
    pd.DataFrame(ens).to_csv(f"{PLOT}/Tab_confusion_ensemble_all.tsv", sep="\t", index=False)
    if shp:
        pd.DataFrame(shp).to_csv(f"{PLOT}/Tab_confusion_shape_all.tsv", sep="\t", index=False)
    print("WROTE Tab_confusion_ensemble_all.tsv" + (" + Tab_confusion_shape_all.tsv" if shp else ""), flush=True)
    print("CONFUSION_RC_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
