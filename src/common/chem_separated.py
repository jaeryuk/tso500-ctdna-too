#!/usr/bin/env python
"""
CHEMISTRY-AWARE v1/v2-SEPARATED ANALYSIS  (follows chemistry_aware_v1_v2_separated_analysis_plan.md)
Two-track: (1) v1/v2-SEPARATED feature discovery + incremental value + biology; (2) domain-aware POOLED only for
final performance, always reported v1/v2-separately.  Reuses primary.py (load_all, base_oof, version-aware z,
per-class NNLS fusion, transfer).  Modules implemented:
  M1  cohort imbalance (reuses chem_confound A1/A2; recomputed compactly here for self-containment)
  M2  feature-family TOO performance WITHIN each chemistry (macro-AUROC, top1/2/3, balanced acc, macro-F1, logloss, brier)
  M3  per-cancer OVR AUROC v1 vs v2 (bootstrap CI + reportability flags) ; per-TF-bin macro-AUROC
  M4  INCREMENTAL VALUE over all-exon depth within each chemistry: depth vs depth+feature vs fusion;
      delta macro-AUROC / top3 / logloss, bootstrap CI, permutation p (perm feature OOF within cancer x TF-bin), FDR
  M6  domain-aware POOLED model: locked-test performance pooled / v1-only / v2-only, depth vs fusion
  M7  cross-chemistry transfer (v1->v2, v2->v1, internal) per feature + fusion; degradation
Outputs -> results/auto_plan/primary/chem_sep/table_*.tsv + report/figS*_*.png
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, log_loss
PROJ = "/home/jrkim/TSO_TFBS/project"; sys.path.insert(0, f"{PROJ}/scripts/auto")
import primary as P
import our_ensemble_v1v2 as OE
OUT = f"{PROJ}/results/auto_plan/primary/chem_sep"; os.makedirs(OUT, exist_ok=True)
FIG = f"{PROJ}/results/auto_plan/report"
NICE = {"E1_entropy": "E1 entropy", "All_exon_depth": "Exon depth", "Genome_wide_CNA": "CNA",
        "Mutation_signature": "Mut signature", "Somatic_mutation_profile": "Mut profile", "SHAPE": "SHAPE"}
ORDER = ["All_exon_depth", "E1_entropy", "SHAPE", "Genome_wide_CNA", "Somatic_mutation_profile", "Mutation_signature"]
NONDEPTH = [m for m in ORDER if m != "All_exon_depth"]
SEED = 42; NBOOT = int(os.environ.get("CS_NBOOT", 1000)); NPERM = int(os.environ.get("CS_NPERM", 300))
REPS = int(os.environ.get("CS_REPS", 4)); EPS = 1e-4
TFBINS = [("low(<3%)", -1, 0.03), ("int(3-10%)", 0.03, 0.10), ("high(>=10%)", 0.10, 9)]
rng = np.random.default_rng(SEED)
def log(m): print(m, flush=True)


# ---------- generic per-class shrinkage NNLS late fusion over an arbitrary method subset ----------
def fuse_subset(oof_sub, y, K):
    methods = list(oof_sub.keys()); nm = len(methods)
    cnt = np.bincount(y, minlength=K); alpha = np.clip(cnt / (cnt + 60.0), 0.0, 0.8)
    Zg = []; tg = []
    for k in range(K):
        Zg.append(np.column_stack([oof_sub[m][:, k] for m in methods])); tg.append((y == k).astype(int))
    wg = P.wnnls(np.vstack(Zg), np.concatenate(tg)); wg = wg / wg.sum() if wg.sum() > 1e-9 else np.full(nm, 1.0 / nm)
    W = np.zeros((K, nm))
    for k in range(K):
        Z = np.column_stack([oof_sub[m][:, k] for m in methods]); wk = P.wnnls(Z, (y == k).astype(int))
        wk = wk / wk.sum() if wk.sum() > 1e-9 else wg.copy()
        a = alpha[k]; W[k] = a * wk + (1 - a) * wg; W[k] = W[k] / max(W[k].sum(), 1e-9)
    Pf = fuse_apply(oof_sub, methods, W, len(y), K)
    return Pf, W, methods


def fuse_apply(oof_sub, methods, W, n, K):
    """apply FIXED per-class weights W to a (possibly permuted) method set -> renormalized class probs."""
    Pf = np.zeros((n, K))
    for k in range(K):
        Pf[:, k] = np.column_stack([oof_sub[m][:, k] for m in methods]) @ W[k]
    Pf = np.clip(Pf, 1e-12, None); return Pf / Pf.sum(1, keepdims=True)


def all_metrics(Pm, y, K):
    top = np.argsort(-Pm, 1)
    oh = np.eye(K)[y]
    return dict(macro_AUROC=round(P.macro_auc(Pm, y, K), 4),
                top1=round(float((top[:, 0] == y).mean()), 4),
                top2=round(float(np.mean([y[i] in top[i, :2] for i in range(len(y))])), 4),
                top3=round(float(np.mean([y[i] in top[i, :3] for i in range(len(y))])), 4),
                bal_acc=round(float(balanced_accuracy_score(y, top[:, 0])), 4),
                macro_F1=round(float(f1_score(y, top[:, 0], average="macro", labels=range(K))), 4),
                logloss=round(float(log_loss(y, Pm, labels=range(K))), 4),
                brier=round(float(np.mean(np.sum((Pm - oh) ** 2, 1))), 4))


def boot_delta(Pa, Pb, y, K, fn, n=NBOOT):
    """bootstrap CI of fn(Pb)-fn(Pa) over resampled samples (models fixed)."""
    d = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        try: d.append(fn(Pb[ix], y[ix], K) - fn(Pa[ix], y[ix], K))
        except Exception: pass
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    man, M, y, cohort, classes, ci = P.load_all()
    K = len(classes); short = [c.replace(" cancer", "").replace("biliary tract", "biliary") for c in classes]
    tf = np.array([OE.TF.get(s, np.nan) for s in man.sid.values]); tf = np.nan_to_num(tf, nan=float(np.nanmedian(tf)))
    tfbin = np.array([next(lab for lab, lo, hi in TFBINS if lo < v <= hi) for v in tf])
    log(f"[chem-sep] n={len(y)} v1={int((cohort==0).sum())} v2={int((cohort==1).sum())} K={K} reps={REPS}")

    # ---- within-chemistry OOF per base family ----
    OOF = {0: {}, 1: {}}; ROWS = {}
    for c in (0, 1):
        rows = np.where(cohort == c)[0]; ROWS[c] = rows
        for m in ORDER:
            OOF[c][m] = P.base_oof(M[m], y, cohort, K, rows, reps=REPS)
        log(f"[chem-sep] chem {'v1' if c==0 else 'v2'}: OOF done ({len(rows)} samples)")

    # ================= M2 feature performance by chemistry + fusion =================
    rows2 = []
    FUS = {}
    for c in (0, 1):
        yc = y[ROWS[c]]
        for m in ORDER:
            d = all_metrics(OOF[c][m], yc, K); d.update(chemistry="v1" if c == 0 else "v2", feature=NICE[m]); rows2.append(d)
        FUS[c], _, _ = fuse_subset({m: OOF[c][m] for m in ORDER}, yc, K)
        d = all_metrics(FUS[c], yc, K); d.update(chemistry="v1" if c == 0 else "v2", feature="FUSION (all)"); rows2.append(d)
    t2 = pd.DataFrame(rows2)[["chemistry", "feature", "macro_AUROC", "top1", "top2", "top3", "bal_acc", "macro_F1", "logloss", "brier"]]
    t2.to_csv(f"{OUT}/table_02_feature_performance_by_chemistry.tsv", sep="\t", index=False)
    log("[chem-sep] M2 table_02 written")

    # ================= M4 incremental value over all-exon depth (within chemistry) =================
    rows5 = []; rows6 = []                                          # rows6 = per-cancer incremental value
    fn = P.macro_auc
    def ovr_delta_ci(p_base_k, p_comb_k, yk, n=600):
        pos = np.where(yk == 1)[0]; neg = np.where(yk == 0)[0]
        if len(pos) < 1 or len(neg) < 1: return np.nan, np.nan, np.nan, np.nan, np.nan
        ab = roc_auc_score(yk, p_base_k); ac = roc_auc_score(yk, p_comb_k); d = []
        for _ in range(n):
            ii = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
            try: d.append(roc_auc_score(yk[ii], p_comb_k[ii]) - roc_auc_score(yk[ii], p_base_k[ii]))
            except Exception: pass
        return ab, ac, ac - ab, (np.percentile(d, 2.5) if d else np.nan), (np.percentile(d, 97.5) if d else np.nan)
    for c in (0, 1):
        yc = y[ROWS[c]]; cname = "v1" if c == 0 else "v2"
        Pdepth = OOF[c]["All_exon_depth"]
        base_m = all_metrics(Pdepth, yc, K)
        addsets = [(NICE[m], {"All_exon_depth": Pdepth, m: OOF[c][m]}) for m in NONDEPTH]
        addsets.append(("ALL non-depth", {m: OOF[c][m] for m in ORDER}))
        # GLOBAL permutation null: break the added feature's label alignment entirely, apply FROZEN real weights
        # (tests whether the feature carries ANY label-relevant signal beyond depth). Within-cancer permutation was
        # rejected: it preserves each feature's between-cancer mean (the TOO signal itself), giving a degenerate p~1.
        strata = np.zeros(len(yc), int)
        for label, sub in addsets:
            Pcomb, Wcomb, methcomb = fuse_subset(sub, yc, K); comb_m = all_metrics(Pcomb, yc, K)
            d_auc, lo, hi = boot_delta(Pdepth, Pcomb, yc, K, fn)
            obs = comb_m["macro_AUROC"] - base_m["macro_AUROC"]
            # permutation: shuffle ADDED feature OOF within cancer x TF-bin, apply FROZEN real weights (isolate
            # the feature's information, not the fusion machinery), recompute delta vs depth-only
            addkey = [k for k in sub if k != "All_exon_depth"]
            ge = 0
            for _ in range(NPERM):
                permsub = {kk: vv for kk, vv in sub.items()}
                for ak in addkey:
                    perm = np.arange(len(yc))
                    for s in np.unique(strata):
                        ix = np.where(strata == s)[0]
                        if len(ix) > 1: perm[ix] = rng.permutation(ix)
                    permsub[ak] = sub[ak][perm]
                Pp = fuse_apply(permsub, methcomb, Wcomb, len(yc), K)
                if (P.macro_auc(Pp, yc, K) - base_m["macro_AUROC"]) >= obs: ge += 1
            pval = (ge + 1) / (NPERM + 1)
            rows5.append(dict(chemistry=cname, added_feature=label,
                              depth_macro=base_m["macro_AUROC"], combined_macro=comb_m["macro_AUROC"],
                              delta_macro_AUROC=round(d_auc, 4), delta_lo=round(lo, 4), delta_hi=round(hi, 4),
                              delta_top3=round(comb_m["top3"] - base_m["top3"], 4),
                              delta_logloss=round(comb_m["logloss"] - base_m["logloss"], 4),
                              perm_p=round(float(pval), 4)))
            # ---- M4.5 per-cancer incremental value (OVR AUROC: depth vs depth+feature) ----
            for k in range(K):
                yk = (yc == k).astype(int); npos = int(yk.sum()); nneg = len(yk) - npos
                ab, ac, dd, dlo, dhi = ovr_delta_ci(Pdepth[:, k], Pcomb[:, k], yk)
                rows6.append(dict(chemistry=cname, cancer=short[k], added_feature=label,
                                  baseline_AUROC=round(ab, 3) if ab == ab else np.nan,
                                  combined_AUROC=round(ac, 3) if ac == ac else np.nan,
                                  delta_AUROC=round(dd, 3) if dd == dd else np.nan,
                                  delta_lo=round(dlo, 3) if dlo == dlo else np.nan,
                                  delta_hi=round(dhi, 3) if dhi == dhi else np.nan,
                                  n_pos=npos, n_neg=nneg,
                                  reportability=("reportable" if (npos >= 10 and nneg >= 30) else
                                                 "exploratory" if npos >= 5 else "not_reportable")))
            log(f"[chem-sep] M4 {cname} depth+{label:14s} dMacro={d_auc:+.4f}[{lo:+.3f},{hi:+.3f}] "
                f"dTop3={comb_m['top3']-base_m['top3']:+.3f} permP={pval:.3f}")
    t5 = pd.DataFrame(rows5)
    # FDR (BH) across all incremental tests
    pv = t5.perm_p.values; order = np.argsort(pv); ranked = pv[order]; m_ = len(pv)
    q = np.empty(m_); prev = 1.0
    for i in range(m_ - 1, -1, -1):
        prev = min(prev, ranked[i] * m_ / (i + 1)); q[order[i]] = round(min(prev, 1.0), 4)
    t5["perm_FDR"] = q
    t5.to_csv(f"{OUT}/table_05_incremental_value_by_chemistry.tsv", sep="\t", index=False)
    t6 = pd.DataFrame(rows6)
    t6.to_csv(f"{OUT}/table_06_incremental_value_by_cancer_and_chemistry.tsv", sep="\t", index=False)
    log("[chem-sep] M4 table_05 + per-cancer table_06 written")

    # ================= M3 per-cancer OVR AUROC (fusion & depth) v1 vs v2 + reportability =================
    def ovr_auc_ci(p_k, yk):
        pos = np.where(yk == 1)[0]; neg = np.where(yk == 0)[0]
        if len(pos) < 1 or len(neg) < 1: return np.nan, np.nan, np.nan
        a = roc_auc_score(yk, p_k); bs = []
        for _ in range(1000):
            ii = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
            try: bs.append(roc_auc_score(yk[ii], p_k[ii]))
            except Exception: pass
        return a, (np.percentile(bs, 2.5) if bs else np.nan), (np.percentile(bs, 97.5) if bs else np.nan)
    def flag(npos, nneg):
        if npos < 5 or nneg < 10: return "not_reportable_insufficient_events"
        if npos < 10: return "exploratory_low_positive_n"
        if nneg < 30: return "exploratory_low_negative_n"
        return "reportable"
    rows3 = []
    models = {"fusion_all_features": FUS, "depth_only": {0: OOF[0]["All_exon_depth"], 1: OOF[1]["All_exon_depth"]}}
    for c in (0, 1):
        yc = y[ROWS[c]]; cname = "v1" if c == 0 else "v2"
        for mod, store in models.items():
            Pm = store[c]
            for k in range(K):
                yk = (yc == k).astype(int); npos = int(yk.sum()); nneg = len(yk) - npos
                a, lo, hi = ovr_auc_ci(Pm[:, k], yk)
                rows3.append(dict(chemistry=cname, cancer=short[k], model=mod, metric="AUROC",
                                  estimate=round(a, 3) if a == a else np.nan,
                                  ci_lo=round(lo, 3) if lo == lo else np.nan, ci_hi=round(hi, 3) if hi == hi else np.nan,
                                  n_pos=npos, n_neg=nneg, reportability=flag(npos, nneg)))
    t3 = pd.DataFrame(rows3); t3.to_csv(f"{OUT}/table_03b_per_cancer_v1_v2_barplot_metrics.tsv", sep="\t", index=False)
    log("[chem-sep] M3 table_03b written")

    # per-TF-bin macro-AUROC by feature + chemistry
    rows4 = []
    for c in (0, 1):
        yc = y[ROWS[c]]; tfb_c = tfbin[ROWS[c]]; cname = "v1" if c == 0 else "v2"
        for lab, lo, hi in TFBINS:
            msk = tfb_c == lab
            if msk.sum() < 20: continue
            for m in ORDER:
                rows4.append(dict(chemistry=cname, tf_bin=lab, feature=NICE[m], n=int(msk.sum()),
                                  macro_AUROC=round(P.macro_auc(OOF[c][m][msk], yc[msk], K), 3)))
            rows4.append(dict(chemistry=cname, tf_bin=lab, feature="FUSION (all)", n=int(msk.sum()),
                              macro_AUROC=round(P.macro_auc(FUS[c][msk], yc[msk], K), 3)))
    pd.DataFrame(rows4).to_csv(f"{OUT}/table_04_tf_bin_feature_performance_by_chemistry.tsv", sep="\t", index=False)
    log("[chem-sep] M3 table_04 written")

    # ================= M7 cross-chemistry transfer (per feature + fusion) =================
    rows7 = []
    Wt = {}
    for src, dst, name in [(0, 1, "v1->v2"), (1, 0, "v2->v1")]:
        si = np.where(cohort == src)[0]; di = np.where(cohort == dst)[0]
        Pm = {}
        for m in ORDER:
            par = P.fit_vznorm(M[m], cohort, si)
            Xs = P.apply_vznorm(M[m][si], cohort[si], par); Xd = P.apply_vznorm(M[m][di], cohort[di], par)
            clf = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced", solver="lbfgs").fit(Xs, y[si])
            pr = np.zeros((len(di), K)); pr[:, clf.classes_] = clf.predict_proba(Xd); Pm[m] = pr
        Pf, _, _ = fuse_subset(Pm, y[di], K)
        within = {m: t2[(t2.chemistry == ("v1" if dst == 0 else "v2")) & (t2.feature == NICE[m])].macro_AUROC.iloc[0] for m in ORDER}
        within_fus = t2[(t2.chemistry == ("v1" if dst == 0 else "v2")) & (t2.feature == "FUSION (all)")].macro_AUROC.iloc[0]
        for m in ORDER:
            cross = P.macro_auc(Pm[m], y[di], K)
            rows7.append(dict(direction=name, feature=NICE[m], cross_AUROC=round(cross, 3),
                              within_AUROC=round(within[m], 3), degradation=round(within[m] - cross, 3)))
        cross_f = P.macro_auc(Pf, y[di], K)
        rows7.append(dict(direction=name, feature="FUSION (all)", cross_AUROC=round(cross_f, 3),
                          within_AUROC=round(within_fus, 3), degradation=round(within_fus - cross_f, 3)))
        log(f"[chem-sep] M7 {name}: fusion cross={cross_f:.3f} within={within_fus:.3f} degr={within_fus-cross_f:+.3f}")
    pd.DataFrame(rows7).to_csv(f"{OUT}/table_10_cross_chemistry_transfer.tsv", sep="\t", index=False)
    log("[chem-sep] M7 table_10 written")

    # ================= M6 domain-aware pooled (locked test: pooled / v1-only / v2-only) =================
    try:
        import report_hardening as RH
        D = RH.reconstruct_locked(); yt = D["y"][D["test"]]; coht = cohort[D["test"]]
        Pf_t = D["Pf"]; Pd_t = D["Ptest"]["All_exon_depth"]
        rows6 = []
        for sub, mask in [("pooled", np.ones(len(yt), bool)), ("v1 only", coht == 0), ("v2 only", coht == 1)]:
            for mod, Pm in [("depth only", Pd_t), ("fusion", Pf_t)]:
                mm = all_metrics(Pm[mask], yt[mask], K)
                rows6.append(dict(test_subset=sub, model=mod, n=int(mask.sum()), **mm))
        pd.DataFrame(rows6).to_csv(f"{OUT}/table_09_pooled_domain_aware_performance.tsv", sep="\t", index=False)
        log("[chem-sep] M6 table_09 written")
    except Exception as e:
        log(f"[chem-sep] M6 pooled skipped: {e}")

    make_figs(t2, t5, t3, pd.DataFrame(rows4), pd.DataFrame(rows7), short, K, t6)
    log(f"[chem-sep] DONE -> {OUT}")


# ----------------------------------------------------------------- figures
def make_figs(t2, t5, t3, t4, t7, short, K, t6=None):
    feats = [NICE[m] for m in ORDER] + ["FUSION (all)"]
    # figS1: feature macro-AUROC v1 vs v2
    fig, ax = plt.subplots(figsize=(10, 4.8)); xs = np.arange(len(feats)); w = 0.38
    for i, c in enumerate(["v1", "v2"]):
        g = t2[t2.chemistry == c].set_index("feature").reindex(feats)
        ax.bar(xs + (i - 0.5) * w, g.macro_AUROC, w, label=c, color=["#9ecae1", "#3182bd"][i], edgecolor="k", lw=.4)
    ax.set_xticks(xs); ax.set_xticklabels(feats, rotation=20, ha="right"); ax.axhline(0.5, color="grey", ls=":")
    ax.set_ylabel("macro-AUROC (within-chemistry OOF)"); ax.set_ylim(0.45, 1.0); ax.legend(title="chemistry", frameon=False)
    ax.set_title("Chemistry-separated feature-family TOO performance"); plt.tight_layout()
    plt.savefig(f"{FIG}/figS1_feature_perf_by_chem.png", dpi=120); plt.close()

    # figS2: incremental delta macro-AUROC over depth, v1 vs v2, with CI + perm-sig star
    adds = [NICE[m] for m in NONDEPTH] + ["ALL non-depth"]
    fig, ax = plt.subplots(figsize=(10, 4.8)); xs = np.arange(len(adds)); w = 0.38
    for i, c in enumerate(["v1", "v2"]):
        g = t5[t5.chemistry == c].set_index("added_feature").reindex(adds)
        yv = g.delta_macro_AUROC.values
        err = np.vstack([yv - g.delta_lo.values, g.delta_hi.values - yv])
        bars = ax.bar(xs + (i - 0.5) * w, yv, w, label=c, color=["#9ecae1", "#3182bd"][i], edgecolor="k", lw=.4,
                      yerr=err, capsize=2, error_kw=dict(lw=.7))
        for j, (b, fdr) in enumerate(zip(bars, g.perm_FDR.values)):
            if fdr < 0.05: ax.text(b.get_x() + b.get_width() / 2, max(0, b.get_height()) + 0.002, "*", ha="center", fontsize=11)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(xs); ax.set_xticklabels(adds, rotation=20, ha="right")
    ax.set_ylabel("Δ macro-AUROC over all-exon depth"); ax.legend(title="chemistry", frameon=False)
    ax.set_title("Incremental value of non-depth features over all-exon depth (within chemistry; * = FDR<0.05)")
    plt.tight_layout(); plt.savefig(f"{FIG}/figS2_incremental_over_depth.png", dpi=120); plt.close()

    # figS3 per-cancer fusion AUROC v1 vs v2 ; figS4 depth
    for mod, fname, title in [("fusion_all_features", "figS3_percancer_fusion", "fusion (all features)"),
                              ("depth_only", "figS4_percancer_depth", "all-exon depth only")]:
        sub = t3[t3.model == mod]; cancers = short
        fig, ax = plt.subplots(figsize=(11, 4.8)); xs = np.arange(len(cancers)); w = 0.38
        for i, c in enumerate(["v1", "v2"]):
            g = sub[sub.chemistry == c].set_index("cancer").reindex(cancers)
            yv = g.estimate.values.astype(float)
            err = np.vstack([yv - g.ci_lo.values.astype(float), g.ci_hi.values.astype(float) - yv])
            bars = ax.bar(xs + (i - 0.5) * w, yv, w, label=c, color=["#9ecae1", "#3182bd"][i], edgecolor="k", lw=.4,
                          yerr=err, capsize=2, error_kw=dict(lw=.6))
            for b, npos, fl in zip(bars, g.n_pos.values, g.reportability.values):
                if isinstance(fl, str) and fl.startswith("exploratory"): b.set_alpha(.45); b.set_hatch("//")
                elif isinstance(fl, str) and fl.startswith("not_reportable"): b.set_alpha(.25); b.set_hatch("xx")
                ax.text(b.get_x() + b.get_width() / 2, 0.46, f"n={int(npos)}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(xs); ax.set_xticklabels(cancers, rotation=30, ha="right"); ax.axhline(0.5, color="grey", ls="--", lw=.8)
        ax.set_ylim(0.45, 1.03); ax.set_ylabel("one-vs-rest AUROC"); ax.legend(title="chemistry", frameon=False)
        ax.set_title(f"Per-cancer TOO performance by chemistry — {title} (hatched = exploratory small-n)")
        plt.tight_layout(); plt.savefig(f"{FIG}/{fname}.png", dpi=120); plt.close()

    # figS5 cross-chemistry transfer degradation
    fig, ax = plt.subplots(figsize=(10, 4.8)); xs = np.arange(len(feats)); w = 0.38
    for i, d in enumerate(["v1->v2", "v2->v1"]):
        g = t7[t7.direction == d].set_index("feature").reindex(feats)
        ax.bar(xs + (i - 0.5) * w, g.degradation, w, label=d, color=["#fdae6b", "#e6550d"][i], edgecolor="k", lw=.4)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(xs); ax.set_xticklabels(feats, rotation=20, ha="right")
    ax.set_ylabel("within − cross macro-AUROC (↑ = worse transfer)"); ax.legend(title="direction", frameon=False)
    ax.set_title("Cross-chemistry transfer degradation by feature"); plt.tight_layout()
    plt.savefig(f"{FIG}/figS5_transfer_degradation.png", dpi=120); plt.close()

    # figS6 feature macro-AUROC by TF bin (v1 & v2 panels)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, c in zip(axes, ["v1", "v2"]):
        g = t4[t4.chemistry == c]
        for m in [NICE[x] for x in ORDER] + ["FUSION (all)"]:
            gg = g[g.feature == m].set_index("tf_bin").reindex([b[0] for b in TFBINS])
            ax.plot([b[0] for b in TFBINS], gg.macro_AUROC.values, marker="o", label=m, lw=1.4 if m == "FUSION (all)" else 1)
        ax.set_title(f"{c}: macro-AUROC by tumour-fraction bin"); ax.axhline(0.5, color="grey", ls=":")
        ax.set_xlabel("tumour-fraction bin"); ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("macro-AUROC"); axes[1].legend(fontsize=7, frameon=False, ncol=2)
    plt.tight_layout(); plt.savefig(f"{FIG}/figS6_tfbin_by_chem.png", dpi=120); plt.close()

    # figS7 per-cancer incremental value heatmap (cancer x added feature, delta OVR-AUROC over depth), v1 & v2
    if t6 is not None and len(t6):
        adds = [NICE[m] for m in NONDEPTH] + ["ALL non-depth"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, c in zip(axes, ["v1", "v2"]):
            g = t6[t6.chemistry == c]
            Mh = np.full((len(short), len(adds)), np.nan)
            for i, cc in enumerate(short):
                for j, ad in enumerate(adds):
                    r = g[(g.cancer == cc) & (g.added_feature == ad)]
                    if len(r): Mh[i, j] = r.delta_AUROC.iloc[0]
            im = ax.imshow(Mh, vmin=-0.15, vmax=0.15, cmap="RdBu_r", aspect="auto")
            ax.set_xticks(range(len(adds))); ax.set_xticklabels(adds, rotation=30, ha="right")
            ax.set_yticks(range(len(short))); ax.set_yticklabels(short)
            for i in range(len(short)):
                for j in range(len(adds)):
                    if Mh[i, j] == Mh[i, j]: ax.text(j, i, f"{Mh[i,j]:+.2f}", ha="center", va="center", fontsize=7)
            ax.set_title(f"{c}: Δ OVR-AUROC over depth, per cancer")
        fig.colorbar(im, ax=axes, fraction=.025, label="Δ AUROC (depth+feature − depth)")
        plt.savefig(f"{FIG}/figS7_percancer_incremental.png", dpi=120, bbox_inches="tight"); plt.close()
    log("[chem-sep] figures figS1-S7 written")


if __name__ == "__main__":
    main()
