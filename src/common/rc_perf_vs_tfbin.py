#!/usr/bin/env python3
"""Tumour-fraction-stratified TOO performance on the RC cohort, from the finished nested-CV OOF.

NO re-run of the 27-h nested CV. The RC benchmark already wrote leakage-free OOF probabilities
(benchmark_nested_v2/{v1,v2}_oof.npz: `primary` = shrink60 ensemble, `mod_<M>` = each module).
Macro-AUROC is a ranking statistic, so it can be re-evaluated on any SAMPLE SUBSET of those same
frozen predictions. Binning by tumour fraction is exactly that. Nothing is refit -- in particular the
ensemble weights stay the ones learnt on the full cohort, which is the honest thing: refitting the
combiner per bin would be a different (bin-specialised) model and would leak the bin definition.

SAMPLE ORDER. The OOF npz stores y/classes but NOT sids, so the order has to be reconstructed. The
benchmark builds it as `order = sorted(common)` where common = manifest labels & X_SHAPE sids & every
FILE_MODULE's sids, restricted to classes with >=MIN_CLASS members (rc_benchmark_nested_v2.order_y).
That is deterministic, so it is reproducible here -- and it is VERIFIED, not assumed: the reconstructed
y must equal the saved y element-wise or this aborts. A wrong order would otherwise silently scramble
the tumour-fraction join and produce plausible-looking nonsense.

BINS. <3% / 3-10% / 10-20% / >20%, matching the ONLY tfbin table on disk
(Tab_ensemble_shape_gini40_tfbins.tsv, keep_t1 cohort) so the RC figure is comparable to the old one.
(An earlier note claimed balanced 3-7.5/7.5-19/>=19 cutoffs; no such table exists -- that note was wrong.)

CAVEAT the caption must carry: per-class AUROC needs both classes present IN THE BIN. High-TF bins are
small, so some cancer types drop out and the macro average is then over FEWER classes than the overall
figure -- `n_classes` is reported per cell for exactly this reason. Bins are not equal-difficulty.

Out: results/auto_plan/nc_readiness/tables/Tab_rc_perf_vs_tfbin.tsv
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

PROJ = "/home/jrkim/TSO_TFBS/project"
RC = f"{PROJ}/results/rule_conformant"
FEAT = f"{PROJ}/results/auto_plan/feat_rc"
MAN = f"{RC}/manifest_dev.tsv"
OOF = f"{RC}/benchmark_nested_v2"
TB = f"{PROJ}/results/auto_plan/nc_readiness/tables"
MIN_CLASS = 20
FILE_MODULES = ["Genome_wide_CNA", "E1_entropy", "Somatic_mutation_profile", "All_exon_depth"]
RC_EXPECT = {"v1": 1093, "v2": 796}

# (label, lo, hi] in tumour-fraction units (fraction, not %)
BINS = [("<3%", -np.inf, 0.03), ("3-10%", 0.03, 0.10), ("10-20%", 0.10, 0.20), (">20%", 0.20, np.inf)]
NICE = {"primary": "LATE-FUSION (all 5)", "mod_E1_entropy": "Exon1 entropy",
        "mod_All_exon_depth": "All-exon depth", "mod_Genome_wide_CNA": "Broad CNA (>10Mb)",
        "mod_Somatic_mutation_profile": "Somatic mutation", "mod_SHAPE_nep300": "SHAPE_nep300"}


def sids_of(p):
    """sids only -- npz members load lazily, so this never touches the big X arrays."""
    return [str(s) for s in np.load(p, allow_pickle=True)["sids"]]


def labels(coh):
    return {l.split("\t")[0]: l.split("\t")[2]
            for l in open(MAN).read().splitlines()[1:] if l.split("\t")[1] == coh}


def rebuild_order(coh):
    """Replicate rc_benchmark_nested_v2.order_y's ordering exactly (sids only, no X)."""
    lab = labels(coh)
    common = set(lab) & set(sids_of(f"{FEAT}/{coh}/X_SHAPE.npz"))
    for m in FILE_MODULES:
        common &= set(sids_of(f"{FEAT}/{coh}/X_{m}.npz"))
    cnt = {}
    for s in common:
        cnt[lab[s]] = cnt.get(lab[s], 0) + 1
    classes = sorted([c for c, nn in cnt.items() if nn >= MIN_CLASS])
    ci = {c: i for i, c in enumerate(classes)}
    order = sorted([s for s in common if lab[s] in ci])
    return order, np.array([ci[lab[s]] for s in order]), classes


def macro_auc(P, y, K):
    """Unweighted mean OVR AUROC; classes absent (or saturating) in this subset are skipped."""
    aucs = []
    for k in range(K):
        yk = (y == k).astype(int)
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        aucs.append(roc_auc_score(yk, P[:, k]))
    return (float(np.mean(aucs)) if aucs else np.nan), len(aucs)


def top1(P, y):
    return float((P.argmax(1) == y).mean())


def main():
    rows = []
    for coh in ("v1", "v2"):
        z = np.load(f"{OOF}/{coh}_oof.npz", allow_pickle=True)
        y_saved = z["y"]
        classes = [str(c) for c in z["classes"]]
        K = len(classes)

        order, y_re, cls_re = rebuild_order(coh)
        # --- the order is VERIFIED, never assumed ---
        if len(order) != RC_EXPECT[coh]:
            sys.exit(f"ABORT {coh}: rebuilt n={len(order)} != RC {RC_EXPECT[coh]}")
        if cls_re != classes:
            sys.exit(f"ABORT {coh}: class list mismatch\n rebuilt={cls_re}\n saved  ={classes}")
        if not np.array_equal(y_re, y_saved):
            sys.exit(f"ABORT {coh}: rebuilt y != saved y ({int((y_re != y_saved).sum())} mismatches) "
                     "-> sample order is wrong, tumour-fraction join would be scrambled")
        print(f"[{coh}] order verified: n={len(order)} K={K} y matches saved OOF exactly", flush=True)

        # tumour fraction (same source the RC figures use)
        d = pd.read_csv(f"{PROJ}/results/rcv{coh[-1]}_cna/cna_feature_table.tsv", sep="\t",
                        dtype={"Unnamed: 0": str}).rename(columns={"Unnamed: 0": "sid"})
        tfm = dict(zip(d.sid, pd.to_numeric(d.tumor_fraction, errors="coerce")))
        tf = np.array([tfm.get(s, np.nan) for s in order])
        miss = int((~np.isfinite(tf)).sum())
        print(f"[{coh}] tumour fraction: {len(tf)-miss}/{len(tf)} finite ({miss} missing -> dropped from bins)",
              flush=True)

        comps = ["primary"] + [f"mod_{m}" for m in FILE_MODULES + ["SHAPE_nep300"]]
        for comp in comps:
            P = z[comp]
            for lab_, lo, hi in [("overall", -np.inf, np.inf)] + BINS:
                sel = np.isfinite(tf) & (tf > lo) & (tf <= hi) if lab_ != "overall" else np.isfinite(tf)
                if sel.sum() < 10:
                    continue
                a, nk = macro_auc(P[sel], y_saved[sel], K)
                rows.append(dict(cohort=coh, component=NICE[comp], bin=lab_, n=int(sel.sum()),
                                 n_classes=nk, macroAUROC=round(a, 4) if np.isfinite(a) else np.nan,
                                 top1=round(top1(P[sel], y_saved[sel]), 4)))
    R = pd.DataFrame(rows)
    os.makedirs(TB, exist_ok=True)
    R.to_csv(f"{TB}/Tab_rc_perf_vs_tfbin.tsv", sep="\t", index=False)
    print("\n=== macro-AUROC by tumour-fraction bin (RC cohort) ===")
    for coh in ("v1", "v2"):
        piv = R[R.cohort == coh].pivot(index="component", columns="bin", values="macroAUROC")
        piv = piv[["overall"] + [b[0] for b in BINS]]
        nn = R[(R.cohort == coh) & (R.component == "SHAPE_nep300")].set_index("bin")["n"]
        print(f"\n[{coh}]  n per bin: " + "  ".join(f"{b}={nn.get(b,'-')}" for b in ["overall"] + [b[0] for b in BINS]))
        print(piv.to_string())
    print(f"\nsaved -> {TB}/Tab_rc_perf_vs_tfbin.tsv")
    print("RC_TFBIN_DONE")


if __name__ == "__main__":
    main()
