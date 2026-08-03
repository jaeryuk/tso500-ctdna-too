#!/usr/bin/env python
"""
REPORT HARDENING — the statistically-defensible analyses the critique demands, on the POOLED locked test.
Reconstructs the EXACT locked-test predictions used by primary.py (same locked_split.json + fusion_weights.tsv),
then computes:
  T2  paired bootstrap of macro-AUROC: FUSION vs All-exon-depth (overall + per cancer), CI + two-sided p.
  T3  per-cancer top-1/top-3 sensitivity, specificity, PPV, NPV (bootstrap CI), macro-F1, balanced acc, top-k.
  T4  tumor-fraction confounding: per-cancer TF distribution; TF-only / mean-depth-only / CNA-burden-only
      baselines vs the full depth view; TF-stratified locked-test performance.
  T5  batch-effect magnitude: how well each feature family PREDICTS the chemistry (v1 vs v2), 5-fold OOF AUROC
      (1.0 = the feature is dominated by batch). Also fusion v1<->v2 transfer recap.
Leakage-safe: bases refit on dev only; TF/baseline models fit on dev, scored on locked test; nothing is tuned
on the locked test. Outputs -> results/auto_plan/primary/hardening/*.tsv + report/fig25-28_*.png
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

PROJ = "/home/jrkim/TSO_TFBS/project"
sys.path.insert(0, f"{PROJ}/scripts/auto")
import primary as P
import our_ensemble_v1v2 as OE          # TF dict
PR = f"{PROJ}/results/auto_plan/primary"; OUT = f"{PR}/hardening"; os.makedirs(OUT, exist_ok=True)
FIG = f"{PROJ}/results/auto_plan/report"
SEED = 42; NBOOT = 1000
def log(m): print(m, flush=True)
def short(c): return c.replace(" cancer", "").replace("biliary tract", "biliary")


def macro_auc(Pm, yy, K):
    a = [roc_auc_score((yy == k).astype(int), Pm[:, k]) for k in range(K) if 0 < (yy == k).sum() < len(yy)]
    return float(np.mean(a)) if a else np.nan
def pc_auc(Pm, yy, K, k):
    return roc_auc_score((yy == k).astype(int), Pm[:, k]) if 0 < (yy == k).sum() < len(yy) else np.nan


def reconstruct_locked():
    """Rebuild the exact locked-test base + fusion probabilities used in the report."""
    man, M, y, cohort, classes, ci = P.load_all(); K = len(classes)
    sp = json.load(open(f"{PR}/locked_split.json"))
    sid2row = {s: i for i, s in enumerate(man.sid.values)}
    dev = np.array([sid2row[s] for s in sp["dev_sids"] if s in sid2row])
    test = np.array([sid2row[s] for s in sp["test_sids"] if s in sid2row])
    Wc = pd.read_csv(f"{PR}/fusion_weights.tsv", sep="\t", index_col=0).reindex(index=classes)[P.METHODS].values
    Ptest = {}
    for m in P.METHODS:
        par = P.fit_vznorm(M[m], cohort, dev)
        Xd = P.apply_vznorm(M[m][dev], cohort[dev], par); Xt = P.apply_vznorm(M[m][test], cohort[test], par)
        clf = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced", solver="lbfgs").fit(Xd, y[dev])
        pr = np.zeros((len(test), K)); pr[:, clf.classes_] = clf.predict_proba(Xt); Ptest[m] = pr
    Pf = P.fuse(Ptest, Wc, K)
    tf = np.array([OE.TF.get(s, np.nan) for s in man.sid.values])
    return dict(man=man, M=M, y=y, cohort=cohort, classes=classes, K=K, dev=dev, test=test,
                Ptest=Ptest, Pf=Pf, tf=tf)


def fit_predict_single(M_col, y, cohort, dev, test, K):
    """version-aware robust-z logistic on a single/low-dim confound feature; dev->test probs."""
    par = P.fit_vznorm(M_col, cohort, dev)
    Xd = P.apply_vznorm(M_col[dev], cohort[dev], par); Xt = P.apply_vznorm(M_col[test], cohort[test], par)
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", solver="lbfgs").fit(Xd, y[dev])
    pr = np.zeros((len(test), K)); pr[:, clf.classes_] = clf.predict_proba(Xt); return pr


def main():
    D = reconstruct_locked()
    y, classes, K, test = D["y"], D["classes"], D["K"], D["test"]
    yt = y[test]; Pf = D["Pf"]; Pdepth = D["Ptest"]["All_exon_depth"]
    rng = np.random.default_rng(SEED)
    log(f"[hardening] locked test n={len(test)} K={K}")

    # ---------- T2: paired bootstrap FUSION vs All-exon-depth ----------
    deltas, fus_b, dep_b = [], [], []
    for _ in range(NBOOT):
        ix = rng.integers(0, len(yt), len(yt))
        f = macro_auc(Pf[ix], yt[ix], K); d = macro_auc(Pdepth[ix], yt[ix], K)
        if np.isfinite(f) and np.isfinite(d): deltas.append(f - d); fus_b.append(f); dep_b.append(d)
    deltas = np.array(deltas)
    p_two = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    t2 = dict(metric="macroAUROC", fusion=round(macro_auc(Pf, yt, K), 4), depth=round(macro_auc(Pdepth, yt, K), 4),
              delta=round(float(deltas.mean()), 4), delta_CI_lo=round(float(np.percentile(deltas, 2.5)), 4),
              delta_CI_hi=round(float(np.percentile(deltas, 97.5)), 4), p_two_sided=round(float(p_two), 4),
              fusion_winrate=round(float((deltas > 0).mean()), 3))
    pd.DataFrame([t2]).to_csv(f"{OUT}/T2_fusion_vs_depth_paired.tsv", sep="\t", index=False)
    # per-cancer paired
    pcr = []
    for k in range(K):
        dl = []
        for _ in range(400):
            ix = rng.integers(0, len(yt), len(yt))
            a = pc_auc(Pf[ix], yt[ix], K, k); b = pc_auc(Pdepth[ix], yt[ix], K, k)
            if np.isfinite(a) and np.isfinite(b): dl.append(a - b)
        dl = np.array(dl) if dl else np.array([0.0])
        pcr.append(dict(cancer=short(classes[k]), n_test=int((yt == k).sum()),
                        fusion_AUROC=round(pc_auc(Pf, yt, K, k), 3) if np.isfinite(pc_auc(Pf, yt, K, k)) else np.nan,
                        depth_AUROC=round(pc_auc(Pdepth, yt, K, k), 3) if np.isfinite(pc_auc(Pdepth, yt, K, k)) else np.nan,
                        delta=round(float(dl.mean()), 3), delta_CI_lo=round(float(np.percentile(dl, 2.5)), 3),
                        delta_CI_hi=round(float(np.percentile(dl, 97.5)), 3)))
    pd.DataFrame(pcr).to_csv(f"{OUT}/T2_perclass_fusion_vs_depth.tsv", sep="\t", index=False)
    log(f"[T2] fusion {t2['fusion']} vs depth {t2['depth']} delta {t2['delta']} "
        f"[{t2['delta_CI_lo']},{t2['delta_CI_hi']}] p={t2['p_two_sided']}")

    # ---------- T3: per-cancer clinical metrics (top-1 confusion-derived) + bootstrap CI ----------
    preds = Pf.argmax(1); top3 = np.argsort(-Pf, 1)[:, :3]
    def class_metrics(yy, pp, t3, k):
        tp = int(((pp == k) & (yy == k)).sum()); fp = int(((pp == k) & (yy != k)).sum())
        fn = int(((pp != k) & (yy == k)).sum()); tn = int(((pp != k) & (yy != k)).sum())
        sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan
        ppv = tp / (tp + fp) if tp + fp else np.nan; npv = tn / (tn + fn) if tn + fn else np.nan
        s3 = float(np.mean([k in t3[i] for i in range(len(yy)) if yy[i] == k])) if (yy == k).any() else np.nan
        return sens, spec, ppv, npv, s3
    rows = []
    for k in range(K):
        sens, spec, ppv, npv, s3 = class_metrics(yt, preds, top3, k)
        # bootstrap CIs for sensitivity, PPV, top3-sens
        bs = {"sens": [], "ppv": [], "s3": []}
        for _ in range(NBOOT):
            ix = rng.integers(0, len(yt), len(yt))
            se, _, pv, _, ss = class_metrics(yt[ix], preds[ix], top3[ix], k)
            if np.isfinite(se): bs["sens"].append(se)
            if np.isfinite(pv): bs["ppv"].append(pv)
            if np.isfinite(ss): bs["s3"].append(ss)
        def ci(a): return (round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)) if a else (np.nan, np.nan)
        sl, sh = ci(bs["sens"]); pl, ph = ci(bs["ppv"]); s3l, s3h = ci(bs["s3"])
        rows.append(dict(cancer=short(classes[k]), n_test=int((yt == k).sum()),
                         top1_sensitivity=round(sens, 3) if np.isfinite(sens) else np.nan, sens_CI=f"[{sl},{sh}]",
                         top3_sensitivity=round(s3, 3) if np.isfinite(s3) else np.nan, top3_CI=f"[{s3l},{s3h}]",
                         specificity=round(spec, 3) if np.isfinite(spec) else np.nan,
                         PPV=round(ppv, 3) if np.isfinite(ppv) else np.nan, PPV_CI=f"[{pl},{ph}]",
                         NPV=round(npv, 3) if np.isfinite(npv) else np.nan))
    pd.DataFrame(rows).to_csv(f"{OUT}/T3_perclass_clinical.tsv", sep="\t", index=False)
    macro_f1 = f1_score(yt, preds, average="macro"); bal = balanced_accuracy_score(yt, preds)
    pd.DataFrame([dict(metric="macro_F1", value=round(macro_f1, 4)),
                  dict(metric="balanced_acc", value=round(bal, 4)),
                  dict(metric="top1_acc", value=round(float((preds == yt).mean()), 4)),
                  dict(metric="top3_acc", value=round(float(np.mean([yt[i] in top3[i] for i in range(len(yt))])), 4))
                  ]).to_csv(f"{OUT}/T3_overall_metrics.tsv", sep="\t", index=False)
    log(f"[T3] macro-F1 {macro_f1:.3f} balanced-acc {bal:.3f}")

    # ---------- T4: tumor-fraction confounding ----------
    M, cohort, dev, tf = D["M"], D["cohort"], D["dev"], D["tf"]
    # per-cancer TF distribution (whole cohort)
    tfd = []
    for k in range(K):
        v = tf[(y == k) & np.isfinite(tf)]
        tfd.append(dict(cancer=short(classes[k]), n_TF=int(len(v)),
                        TF_median=round(float(np.median(v)), 4) if len(v) else np.nan,
                        TF_q25=round(float(np.percentile(v, 25)), 4) if len(v) else np.nan,
                        TF_q75=round(float(np.percentile(v, 75)), 4) if len(v) else np.nan))
    pd.DataFrame(tfd).to_csv(f"{OUT}/T4_perclass_TF_dist.tsv", sep="\t", index=False)
    # confound baselines vs full depth view (locked test macro-AUROC)
    base_rows = [dict(view="All_exon_depth (full)", macro_AUROC=round(macro_auc(Pdepth, yt, K), 4))]
    # TF-only (samples with finite TF in BOTH dev and test)
    fin = np.isfinite(tf)
    devf = dev[fin[dev]]; testf = test[fin[test]]
    if len(devf) > 30 and len(testf) > 20:
        Ptf = fit_predict_single(tf.reshape(-1, 1), y, cohort, devf, testf, K)
        base_rows.append(dict(view="tumor_fraction_only", macro_AUROC=round(macro_auc(Ptf, y[testf], K), 4),
                              note=f"n_dev={len(devf)} n_test={len(testf)}"))
    # mean-exon-depth only (1 feature = mean across exon columns)
    meandepth = M["All_exon_depth"].mean(1, keepdims=True)
    Pmd = fit_predict_single(meandepth, y, cohort, dev, test, K)
    base_rows.append(dict(view="mean_exon_depth_only", macro_AUROC=round(macro_auc(Pmd, yt, K), 4)))
    # CNA-burden only (mean abs CNA)
    cna_burden = np.abs(M["Genome_wide_CNA"]).mean(1, keepdims=True)
    Pcb = fit_predict_single(cna_burden, y, cohort, dev, test, K)
    base_rows.append(dict(view="CNA_burden_only", macro_AUROC=round(macro_auc(Pcb, yt, K), 4)))
    pd.DataFrame(base_rows).to_csv(f"{OUT}/T4_confound_baselines.tsv", sep="\t", index=False)
    # TF-stratified locked-test performance (fusion)
    strat = []
    tft = tf[test]
    for lab, lo, hi in [("TF<0.03", -1, 0.03), ("0.03-0.10", 0.03, 0.10), ("TF>=0.10", 0.10, 9)]:
        msk = np.isfinite(tft) & (tft >= lo) & (tft < hi)
        if msk.sum() >= 10:
            strat.append(dict(stratum=lab, n=int(msk.sum()),
                              top1_acc=round(float((preds[msk] == yt[msk]).mean()), 3),
                              macro_AUROC=round(macro_auc(Pf[msk], yt[msk], K), 3)))
        else:
            strat.append(dict(stratum=lab, n=int(msk.sum()), top1_acc=np.nan, macro_AUROC=np.nan))
    pd.DataFrame(strat).to_csv(f"{OUT}/T4_TF_stratified.tsv", sep="\t", index=False)
    log("[T4] confound baselines: " + " ".join(f"{r['view']}={r['macro_AUROC']}" for r in base_rows))

    # ---------- T5: batch-effect magnitude (predict chemistry from each feature) ----------
    bt = []
    for m in P.METHODS:
        X = M[m]; oof = np.zeros(len(cohort))
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for tl, vl in skf.split(X, cohort):
            par = P.fit_vznorm(X, np.zeros(len(cohort), int), tl)   # single-pool norm so we don't erase batch
            Xt = P.apply_vznorm(X[tl], np.zeros(len(tl), int), par); Xv = P.apply_vznorm(X[vl], np.zeros(len(vl), int), par)
            clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs").fit(Xt, cohort[tl])
            oof[vl] = clf.predict_proba(Xv)[:, 1]
        auc = roc_auc_score(cohort, oof)
        bt.append(dict(feature=m, chemistry_predict_AUROC=round(float(auc), 4)))
        log(f"[T5] {m:24s} predicts v1-vs-v2 AUROC={auc:.3f}")
    pd.DataFrame(bt).to_csv(f"{OUT}/T5_batch_predictability.tsv", sep="\t", index=False)

    # ---------- figures ----------
    # fig25: per-cancer fusion vs depth AUROC with delta CI
    dfp = pd.DataFrame(pcr)
    fig, ax = plt.subplots(figsize=(10, 4.6)); x = np.arange(len(dfp))
    ax.bar(x - 0.2, dfp.fusion_AUROC, 0.4, label="Fusion", color="#d98c5f")
    ax.bar(x + 0.2, dfp.depth_AUROC, 0.4, label="All-exon depth", color="#9ecae1")
    ax.set_xticks(x); ax.set_xticklabels(dfp.cancer, rotation=25, ha="right"); ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("locked-test per-cancer AUROC"); ax.legend()
    ax.set_title(f"Per-cancer AUROC: fusion vs all-exon depth (locked test, n={len(test)})")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig25_perclass_fusion_vs_depth.png", dpi=120); plt.close()
    # fig26: confound baselines
    dfb = pd.DataFrame(base_rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(dfb.view[::-1], dfb.macro_AUROC[::-1], color="#74a9cf")
    ax.set_xlim(0.5, 0.95); ax.set_xlabel("locked-test macro-AUROC")
    ax.set_title("Confound baselines vs full depth view")
    for i, v in enumerate(dfb.macro_AUROC[::-1]): ax.text(v + .005, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig26_confound_baselines.png", dpi=120); plt.close()
    # fig27: batch predictability
    dft = pd.DataFrame(bt).sort_values("chemistry_predict_AUROC")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cols = ["#cb181d" if v > 0.9 else "#fb6a4a" if v > 0.75 else "#74c476" for v in dft.chemistry_predict_AUROC]
    ax.barh([short(s.replace("_", " ")) for s in dft.feature], dft.chemistry_predict_AUROC, color=cols)
    ax.axvline(0.5, color="k", ls="--", alpha=.5); ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC predicting chemistry (v1 vs v2)  — higher = stronger batch effect")
    ax.set_title("How batch-confounded is each feature family?")
    for i, v in enumerate(dft.chemistry_predict_AUROC): ax.text(v + .005, i, f"{v:.2f}", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig27_batch_predictability.png", dpi=120); plt.close()
    # fig28: per-cancer TF distribution (box)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    data = [tf[(y == k) & np.isfinite(tf)] for k in range(K)]
    ax.boxplot(data, labels=[short(c) for c in classes], showfliers=False)
    ax.set_ylabel("tumor fraction (ichorCNA)"); ax.set_title("Per-cancer tumor-fraction distribution (whole cohort)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(f"{FIG}/fig28_perclass_TF.png", dpi=120); plt.close()

    log(f"[hardening] wrote {OUT}/*.tsv + figs 25-28")


if __name__ == "__main__":
    main()
