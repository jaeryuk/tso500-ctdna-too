#!/usr/bin/env python
"""
OUR-STYLE ensemble of the 6 features, evaluated WITHIN v1 and v2 SEPARATELY, under a rigorous
70%-train / 30%-independent-validation split. Unlike a single elastic-net on stacked raw features,
this uses our actual architecture:
  * one L2 multinomial-logistic BASE model per feature family,
  * leakage-free out-of-fold (OOF) base probabilities on the training 70%,
  * CANCER-TYPE-SPECIFIC fusion weights via non-negative least squares (NNLS) in probability space,
    shrunk toward a global weight by class size (small classes trust the global pattern more),
  * fuse -> renormalize, refit bases on the full 70%, score the held-out 30% ONCE.
Two split styles (as in the Helzer protocol): stratified-random, and ctDNA-fraction (train high-burden
70%, validate lowest-burden 30%). No feature re-extraction; reuses results/auto_plan/feat/{v1,v2}/X_*.npz
and the fusion math from primary.py.

Outputs -> results/auto_plan/our_ensemble/*.tsv + report/fig19-22 (our-ensemble versions).
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, confusion_matrix, roc_curve, auc

PROJ = "/home/jrkim/TSO_TFBS/project"
sys.path.insert(0, f"{PROJ}/scripts/auto")
import primary as P
FEAT = f"{PROJ}/results/auto_plan/feat"
OUT = f"{PROJ}/results/auto_plan/our_ensemble"; os.makedirs(OUT, exist_ok=True)
FIG = f"{PROJ}/results/auto_plan/report"
MAN = f"{PROJ}/results/auto_plan/manifest_dev.tsv"
METHODS = ["E1_entropy", "All_exon_depth", "Genome_wide_CNA", "Mutation_signature",
           "Somatic_mutation_profile", "SHAPE"]
NICE = {"E1_entropy": "E1 entropy", "All_exon_depth": "Exon depth", "Genome_wide_CNA": "CNA",
        "Mutation_signature": "Mut signature", "Somatic_mutation_profile": "Mut profile", "SHAPE": "SHAPE"}
PRIMARY = ["lung cancer", "colorectal cancer", "gastric cancer", "pancreatic cancer",
           "biliary tract cancer", "melanoma", "liver cancer", "prostate cancer", "breast cancer"]
P.METHODS = METHODS                       # primary.fusion_weights/fuse iterate this list
MINN = 8; SEED = 42; REPS = 5
def log(m): print(m, flush=True)
def short(c): return {"biliary tract cancer":"biliary","colorectal cancer":"colorectal","gastric cancer":"gastric",
    "pancreatic cancer":"pancreatic","liver cancer":"liver","lung cancer":"lung","breast cancer":"breast",
    "prostate cancer":"prostate","melanoma":"melanoma"}.get(c, c)


def load_tf():
    tf = {}
    for coh in ["v1", "v2"]:
        base = f"/data/TSO500/ichorCNA_offProbe_result/{coh}/ichorCNA"
        if not os.path.isdir(base): continue
        for d in os.listdir(base):
            pf = os.path.join(base, d, d + ".params.txt")
            if not os.path.exists(pf): continue
            sid = d.split("_")[0]
            try:
                for ln in open(pf):
                    if ln.startswith("Tumor Fraction:"): tf[sid] = float(ln.strip().split()[-1]); break
            except Exception: pass
    return tf
TF = load_tf()


def load_cohort(coh):
    man = pd.read_csv(MAN, sep="\t", dtype=str)
    man = man[(man.cohort == coh) & (man.cancer_type.isin(PRIMARY))].copy()
    feats = {}
    for m in METHODS:
        z = np.load(f"{FEAT}/{coh}/X_{m}.npz", allow_pickle=True)
        sids = [str(s) for s in z["sids"]]
        feats[m] = (z["X"].astype(np.float32), {s: i for i, s in enumerate(sids)})
    keep = [all(s in feats[m][1] for m in METHODS) for s in man.sid]
    man = man[keep].reset_index(drop=True)
    vc = man.cancer_type.value_counts(); classes = sorted(vc[vc >= MINN].index)
    man = man[man.cancer_type.isin(classes)].reset_index(drop=True)
    ci = {c: k for k, c in enumerate(classes)}; y = man.cancer_type.map(ci).values
    X = {m: np.vstack([feats[m][0][feats[m][1][s]] for s in man.sid]).astype(np.float32) for m in METHODS}
    tf = np.array([TF.get(s, np.nan) for s in man.sid])
    return X, y, classes, tf


def macro_auc(Pm, yy, K):
    a = [roc_auc_score((yy == k).astype(int), Pm[:, k]) for k in range(K) if 0 < (yy == k).sum() < len(yy)]
    return float(np.mean(a)) if a else np.nan
def pc_auc(Pm, yy, K):
    return [round(roc_auc_score((yy == k).astype(int), Pm[:, k]), 3) if 0 < (yy == k).sum() < len(yy) else np.nan
            for k in range(K)]


def our_fusion(X, y, classes, tr, va):
    """Train base models + per-class shrinkage NNLS fusion on `tr`; score `va`."""
    K = len(classes); coh0 = np.zeros(len(y), int)
    ytr, yva = y[tr], y[va]
    oof = {m: P.base_oof(X[m], y, coh0, K, tr, reps=REPS) for m in METHODS}     # leakage-free train OOF
    cnt = np.bincount(ytr, minlength=K); alpha = np.clip(cnt / (cnt + 60.0), 0.0, 0.8)
    Wc, wg = P.fusion_weights(oof, ytr, K, alpha)                                # per-class weights
    Ptr_f = P.fuse(oof, Wc, K)
    Pval = {}
    for m in METHODS:                                                           # refit on full train -> val
        par = P.fit_vznorm(X[m], coh0, tr)
        Xt = P.apply_vznorm(X[m][tr], coh0[tr], par); Xv = P.apply_vznorm(X[m][va], coh0[va], par)
        clf = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced", solver="lbfgs").fit(Xt, ytr)
        pr = np.zeros((len(va), K)); pr[:, clf.classes_] = clf.predict_proba(Xv); Pval[m] = pr
    Pval_f = P.fuse(Pval, Wc, K)
    base_val = {m: macro_auc(Pval[m], yva, K) for m in METHODS}
    rec = dict(train_cv_macroAUROC=round(macro_auc(Ptr_f, ytr, K), 3),
               val_acc=round(accuracy_score(yva, Pval_f.argmax(1)), 3),
               val_balanced_acc=round(balanced_accuracy_score(yva, Pval_f.argmax(1)), 3),
               val_macroAUROC=round(macro_auc(Pval_f, yva, K), 3),
               best_base=max(base_val, key=base_val.get), best_base_val_AUROC=round(max(base_val.values()), 3))
    return rec, Wc, Pval_f, yva, base_val


def main():
    rows, pc_rows, base_rows, store_w, store_cm, store_roc = [], [], [], {}, {}, {}
    for coh in ["v1", "v2"]:
        X, y, classes, tf = load_cohort(coh); K = len(classes)
        log(f"\n=== {coh}: n={len(y)} K={K} {[short(c) for c in classes]}")
        # ---- stratified random 70/30 ----
        tr, va = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=SEED).split(np.zeros(len(y)), y))
        t0 = time.time(); rec, Wc, Pval_f, yva, base_val = our_fusion(X, y, classes, tr, va)
        rec.update(cohort=coh, split="stratified_random", n_train=len(tr), n_val=len(va))
        rows.append(rec); store_w[coh] = (Wc, classes); store_cm[coh] = (confusion_matrix(yva, Pval_f.argmax(1), labels=range(K)), classes)
        store_roc[coh] = (Pval_f, yva, classes)
        for m in METHODS: base_rows.append(dict(cohort=coh, model=NICE[m], val_macroAUROC=round(base_val[m], 3)))
        base_rows.append(dict(cohort=coh, model="FUSION (per-class)", val_macroAUROC=rec["val_macroAUROC"]))
        for k in range(K):
            pc_rows.append(dict(cohort=coh, cancer=short(classes[k]), n_val=int((yva == k).sum()),
                                fusion_val_AUROC=pc_auc(Pval_f, yva, K)[k]))
        log(f"   [random] fusion val AUROC={rec['val_macroAUROC']} (best base {rec['best_base']}={rec['best_base_val_AUROC']}) "
            f"acc={rec['val_acc']} ({time.time()-t0:.0f}s)")
        # ---- ctDNA-fraction low-burden split ----
        okt = np.isfinite(tf)
        if okt.sum() >= 30:
            thr = np.quantile(tf[okt], 0.30); va_mask = okt & (tf <= thr); tr_mask = okt & (tf > thr)
            tr2 = np.where(tr_mask)[0]; va2 = np.where(va_mask)[0]
            trc = set(y[tr2].tolist()); va2 = np.array([i for i in va2 if y[i] in trc])
            if len(va2) >= 10:
                t0 = time.time(); rec2, _, _, _, _ = our_fusion(X, y, classes, tr2, va2)
                rec2.update(cohort=coh, split="ctDNA_fraction", n_train=len(tr2), n_val=len(va2),
                            val_TF_max=round(float(np.nanmax(tf[va2])), 4))
                rows.append(rec2)
                log(f"   [low-ctDNA] fusion val AUROC={rec2['val_macroAUROC']} acc={rec2['val_acc']} ({time.time()-t0:.0f}s)")

    res = pd.DataFrame(rows); res.to_csv(f"{OUT}/our_ensemble_results.tsv", sep="\t", index=False)
    pd.DataFrame(pc_rows).to_csv(f"{OUT}/our_ensemble_perclass.tsv", sep="\t", index=False)
    pd.DataFrame(base_rows).to_csv(f"{OUT}/our_ensemble_basevsfusion.tsv", sep="\t", index=False)
    for coh in ["v1", "v2"]:
        if coh in store_w:
            Wc, classes = store_w[coh]
            pd.DataFrame(Wc, index=[short(c) for c in classes], columns=[NICE[m] for m in METHODS]).round(3) \
                .to_csv(f"{OUT}/weights_{coh}.tsv", sep="\t")
    log(f"\n[our-ensemble] wrote {OUT}/our_ensemble_results.tsv ({len(res)} rows)")

    # fig19: base models vs fusion, validation AUROC, v1 & v2
    bdf = pd.DataFrame(base_rows); order = [NICE[m] for m in METHODS] + ["FUSION (per-class)"]
    fig, ax = plt.subplots(figsize=(11, 4.8)); x = np.arange(len(order)); w = 0.38
    for j, coh in enumerate(["v1", "v2"]):
        sub = bdf[bdf.cohort == coh].set_index("model").reindex(order)
        bars = ax.bar(x + (j - 0.5) * w, sub.val_macroAUROC.values, w, label=coh)
        bars[-1].set_edgecolor("black"); bars[-1].set_linewidth(2)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=22, ha="right"); ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("validation macro AUROC"); ax.set_title("Our per-class-weighted ensemble vs single base models (30% validation), v1 & v2")
    ax.grid(alpha=.3); ax.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/fig19_ourens_v1v2.png", dpi=120); plt.close()

    # fig20: per-class fusion weight heatmaps v1 & v2 (the "weighting by cancer type")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
    for j, coh in enumerate(["v1", "v2"]):
        if coh not in store_w: continue
        Wc, classes = store_w[coh]
        im = axs[j].imshow(Wc, cmap="viridis", aspect="auto", vmin=0, vmax=float(np.max(Wc)))
        axs[j].set_xticks(range(len(METHODS))); axs[j].set_xticklabels([NICE[m] for m in METHODS], rotation=30, ha="right", fontsize=8)
        axs[j].set_yticks(range(len(classes))); axs[j].set_yticklabels([short(c) for c in classes], fontsize=8)
        axs[j].set_title(f"{coh}: per-cancer fusion weights")
        for a in range(len(classes)):
            for b in range(len(METHODS)):
                axs[j].text(b, a, f"{Wc[a,b]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if Wc[a, b] < np.max(Wc) * 0.6 else "black")
        plt.colorbar(im, ax=axs[j], fraction=.046)
    plt.suptitle("Cancer-type-specific fusion weights (each row sums to 1 across the 6 features)")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig20_ourens_weights.png", dpi=120); plt.close()

    # fig21: ROC (fusion) v1 & v2
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))
    for j, coh in enumerate(["v1", "v2"]):
        if coh not in store_roc: continue
        Pv, yv, classes = store_roc[coh]; K = len(classes)
        for k in range(K):
            yk = (yv == k).astype(int)
            if 0 < yk.sum() < len(yk):
                fpr, tpr, _ = roc_curve(yk, Pv[:, k]); axs[j].plot(fpr, tpr, lw=1.5, label=f"{short(classes[k])} ({auc(fpr,tpr):.2f})")
        axs[j].plot([0, 1], [0, 1], "k--", alpha=.4); axs[j].set_title(f"{coh}: fusion ROC (30% validation)")
        axs[j].set_xlabel("FPR"); axs[j].set_ylabel("TPR"); axs[j].legend(fontsize=7, loc="lower right"); axs[j].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig21_ourens_roc.png", dpi=120); plt.close()

    # fig22: confusion (fusion) v1 & v2
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    for j, coh in enumerate(["v1", "v2"]):
        if coh not in store_cm: continue
        cm, classes = store_cm[coh]; cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        im = axs[j].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        axs[j].set_xticks(range(len(classes))); axs[j].set_xticklabels([short(c) for c in classes], rotation=45, ha="right", fontsize=8)
        axs[j].set_yticks(range(len(classes))); axs[j].set_yticklabels([short(c) for c in classes], fontsize=8)
        axs[j].set_title(f"{coh}: fusion confusion (30% validation)"); axs[j].set_xlabel("predicted"); axs[j].set_ylabel("actual")
        for a in range(len(classes)):
            for b in range(len(classes)):
                if cmn[a, b] > 0.01: axs[j].text(b, a, f"{cmn[a,b]:.2f}", ha="center", va="center",
                                                 fontsize=6, color="white" if cmn[a, b] > 0.5 else "black")
    plt.suptitle("Our-ensemble confusion matrices (row-normalized)"); plt.tight_layout()
    plt.savefig(f"{FIG}/fig22_ourens_confusion.png", dpi=120); plt.close()
    log("[our-ensemble] figures -> fig19_ourens, fig20_ourens, fig21_ourens, fig22_ourens")


if __name__ == "__main__":
    main()
