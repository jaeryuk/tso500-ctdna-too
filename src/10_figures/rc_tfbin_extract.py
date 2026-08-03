#!/usr/bin/env python3
"""Tumor-fraction-stratified TOO performance + module Shapley, on the RULE-CONFORMANT cohort.

Replaces the cohort source of two figures that violated the standing RC directive:
  p_ensemble_shape_gini40_tfbins.py / fig_shapley_tfbin.py read results/auto_plan/nc_readiness/_oof_cache,
  i.e. the OLD cohort (v1 n=1034 K=10; v2 n=708 K=7 -- a whole cancer type missing vs RC's K=8).
Here everything comes from results/rule_conformant/benchmark/{v1,v2}_oof.npz  -> v1 1093/K=10, v2 796/K=8.

Difference vs the originals: the RC store holds the repeat-AVERAGED OOF per module (n,K), not the
(20,n,K) per-repeat cube, so Shapley is computed once on the averaged OOF instead of per repeat and
averaged. Both R scripts only ever plotted the across-repeat MEAN, so the figures are unaffected; what
is lost is the per-repeat spread (which was never drawn).

Characteristic function, unchanged from fig_shapley_tfbin.py: per-cancer NNLS late fusion over each
subset's module OOFs (weights fit on ALL samples), macro-AUROC evaluated RESTRICTED to each TF bin;
v({}) = 0.5, so within a bin  sum_i phi_i = ensemble macroAUROC(bin) - 0.5.

Out: results/plot/Tab_rc_perf_by_tfbin.tsv , results/plot/Tab_rc_shapley_tfbin.tsv
"""
import os, sys, itertools, math
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/jrkim/TSO_TFBS/project/scripts/auto")
sys.path.insert(0, "/home/jrkim/TSO_TFBS/project/scripts/auto/nc_readiness")
from ncr_common import wnnls
from rc_perf_vs_tfbin import rebuild_order      # deterministic sid order + y self-check

PROJ = "/home/jrkim/TSO_TFBS/project"
# REPOINTED 2026-07-17 to the authoritative RC benchmark. This read
# results/rule_conformant/benchmark — a DIFFERENT, older run (2026-07-08) with OOF keys
# {E1_entropy, ..., SHAPE, late_fusion}. benchmark_nested_v2 (RC_BENCH_V2_DONE 2026-07-17) is the
# published one: SHAPE -> mod_SHAPE_nep300 (the standing nep>300 definition), late_fusion -> primary
# (shrink60 NNLS SL), and every module key gains a `mod_` prefix. Cohort is identical between the two
# (y/classes verified), so only the model changes. Same trap hit rc_top1_extract.py.
BEN = f"{PROJ}/results/rule_conformant/benchmark_nested_v2"
PLOT = f"{PROJ}/results/plot"
MODS = ["mod_E1_entropy", "mod_All_exon_depth", "mod_Genome_wide_CNA",
        "mod_Somatic_mutation_profile", "mod_SHAPE_nep300"]
ENS = "primary"
LABEL = {"mod_E1_entropy": "E1 entropy", "mod_All_exon_depth": "All-exon depth",
         "mod_Genome_wide_CNA": "Broad CNA",
         "mod_Somatic_mutation_profile": "Somatic mutation", "mod_SHAPE_nep300": "TFBS SHAPE"}
BINS = [(-1.0, 0.03, "<3%"), (0.03, 0.10, "3-10%"), (0.10, 0.20, "10-20%"), (0.20, 2.0, ">20%")]


def tumor_fraction(cohort):
    f = f"{PROJ}/results/rcv{cohort[-1]}_cna/cna_feature_table.tsv"
    d = pd.read_csv(f, sep="\t", dtype={"Unnamed: 0": str}).rename(columns={"Unnamed: 0": "sid"})
    return dict(zip(d.sid, pd.to_numeric(d.tumor_fraction, errors="coerce")))


def fuse_F(oofs, y, K):
    F = np.zeros((len(y), K))
    for k in range(K):
        Zc = np.column_stack([P[:, k] for P in oofs])
        w = wnnls(Zc, (y == k).astype(int))
        w = w / w.sum() if w.sum() > 1e-9 else np.full(len(oofs), 1 / len(oofs))
        F[:, k] = Zc @ w
    F = np.clip(F, 1e-12, None)
    return F / F.sum(1, keepdims=True)


def macro_bin(P, yb, valid):
    return float(np.mean([roc_auc_score((yb == k).astype(int), P[:, k]) for k in valid]))


def main():
    n_mod = len(MODS); idx = list(range(n_mod))
    coef = {s: math.factorial(s) * math.factorial(n_mod - s - 1) / math.factorial(n_mod) for s in range(n_mod)}
    perf_rows, shap_rows = [], []

    for c in ("v1", "v2"):
        z = np.load(f"{BEN}/{c}_oof.npz", allow_pickle=True)
        y = np.asarray(z["y"]); K = len(z["classes"])
        # benchmark_nested_v2's OOF drops the `order` key the old benchmark carried. Rebuild it
        # deterministically; rebuild_order re-derives y and ABORTS unless it matches the saved y
        # element-wise, so a wrong order cannot silently scramble the tumour-fraction join below.
        sids, y_re, _ = rebuild_order(c)
        if not np.array_equal(y_re, y):
            sys.exit(f"ABORT {c}: rebuilt order's y != saved y -> tumour-fraction join unsafe")
        tfmap = tumor_fraction(c)
        tf = np.array([tfmap.get(s, np.nan) for s in sids], float)
        miss = int(np.isnan(tf).sum())
        print(f"[{c}] RC n={len(y)} K={K}   tumor_fraction matched {len(y)-miss}/{len(y)}"
              f"{'  <-- ' + str(miss) + ' MISSING' if miss else ''}", flush=True)
        if miss:
            sys.exit(f"ABORT — {miss} {c} samples lack tumor_fraction; cohort/TF join is wrong")

        binmask = {lab: (tf > lo) & (tf <= hi) for lo, hi, lab in BINS}
        valid = {lab: [k for k in range(K) if 0 < (y[m] == k).sum() < m.sum()] for lab, m in binmask.items()}
        for lab, m in binmask.items():
            print(f"    {lab:7s} n={int(m.sum()):4d}  classes evaluable: {len(valid[lab])}/{K}", flush=True)

        mats = [np.asarray(z[mm]) for mm in MODS]

        # ---- per-module + ensemble macro-AUROC per bin (the perf figure) ----
        for mm in MODS + [ENS]:
            P = np.asarray(z[mm])
            for lab, m in binmask.items():
                if not valid[lab]: continue
                perf_rows.append(dict(cohort=c, bin=lab,
                                      module=LABEL.get(mm, "Ensemble"),
                                      macroAUROC=round(macro_bin(P[m], y[m], valid[lab]), 4),
                                      n=int(m.sum())))

        # ---- exact Shapley over all 2^5 subsets, evaluated per bin (the contribution figure) ----
        FS = {S: (fuse_F([mats[j] for j in S], y, K) if len(S) else None)
              for size in range(n_mod + 1) for S in itertools.combinations(idx, size)}
        for lab, m in binmask.items():
            if not valid[lab]: continue
            vS = {S: (macro_bin(FS[S][m], y[m], valid[lab]) if len(S) else 0.5) for S in FS}
            tot = 0.0
            for i in idx:
                others = [j for j in idx if j != i]; phi = 0.0
                for size in range(len(others) + 1):
                    for S in itertools.combinations(others, size):
                        phi += coef[size] * (vS[tuple(sorted(S + (i,)))] - vS[tuple(sorted(S))])
                shap_rows.append(dict(cohort=c, bin=lab, module=LABEL[MODS[i]],
                                      shapley=round(phi, 5), n=int(m.sum())))
                tot += phi
            # efficiency axiom: sum of Shapley values == v(all) - v({})
            full = vS[tuple(idx)] - 0.5
            if abs(tot - full) > 1e-6:
                sys.exit(f"ABORT — Shapley efficiency violated ({c}/{lab}): sum={tot:.6f} vs v(N)-0.5={full:.6f}")
            print(f"    [{c}/{lab}] Shapley OK: sum={tot:.4f} == NNLS-fusion macroAUROC-0.5={full:.4f}", flush=True)

    pd.DataFrame(perf_rows).to_csv(f"{PLOT}/Tab_rc_perf_by_tfbin.tsv", sep="\t", index=False)
    pd.DataFrame(shap_rows).to_csv(f"{PLOT}/Tab_rc_shapley_tfbin.tsv", sep="\t", index=False)
    print(f"\nsaved -> {PLOT}/Tab_rc_perf_by_tfbin.tsv ({len(perf_rows)} rows)")
    print(f"saved -> {PLOT}/Tab_rc_shapley_tfbin.tsv ({len(shap_rows)} rows)\nRC_TFBIN_DONE")


if __name__ == "__main__":
    main()
