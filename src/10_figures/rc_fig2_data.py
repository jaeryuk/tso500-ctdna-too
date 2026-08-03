#!/usr/bin/env python3
"""Rebuild the Fig2A/2B CSVs from the AUTHORITATIVE RC benchmark (benchmark_nested_v2), and re-emit the
Fig2C/2D CSVs from the RC Shapley tables — so all four manuscript figures come from one frozen result.

WHY THIS EXISTS. The Fig2A-D CSVs on disk were written 2026-07-06 by the pre-RC pipeline (fig2a_data.py
etc., keep_t1 cohort, _oof_cache, SHAPE=gini5_len5). The standing cohort is RC and the authoritative
benchmark is results/rule_conformant/benchmark_nested_v2 (RC_BENCH_V2_DONE 2026-07-17, ensemble
0.8680 v1 / 0.8784 v2, SHAPE_nep300). NOTE the two top-1 extractors point at the OLDER
results/rule_conformant/benchmark — a different, stale directory; they are repointed separately.

WHAT IS RECOMPUTED vs REUSED
  2A per-repeat macro-AUROC : RECOMPUTED. The benchmark json stores only mean/sd, not the 20 raw values
                              a box plot needs, so phase-2 is replayed from {coh}_phase1.pkl
                              (trainOOF/testpred) using the benchmark's OWN shrink_fit/shrink_apply.
                              No CV re-run, no model refits.
  2B per-cancer AUROC       : from the pooled OOF in {coh}_oof.npz (the `panel` metric).
  2C / 2D                   : REUSED from Tab_rc_shapley*.tsv (already verified: v(N) == published
                              shrink60, LOMO reproduced, efficiency 1.1e-16, linearity 4.9e-05).

CHECKS (asserted): mean over repeats of the 2A module values must equal the published module `mean`, and
the 2A Ensemble must equal the published shrink60 `mean`; the 2B module per-class AUROCs must equal the
json's `perclass_auc`. If either fails, the CSVs would disagree with the benchmark table in the paper.

GOTCHA: the R scripts use DIFFERENT module labels per figure — 2A wants "Mutation profile", 2B/2C/2D want
"Mutation"; all want "E1 entropy" (the RC tables say "Exon1 entropy"). Labels below match each R script's
factor levels exactly; a mismatch silently drops the row to NA and blanks a box/cell.

Out: results/plot/{fig2a_macroAUROC_perrepeat,fig2b_percancer_auroc_long,fig2c_shapley_perrepeat,
     fig2d_shapley_percancer_long}.csv  (+ .prerc.bak of whatever was there)
"""
import os, sys, json, pickle, shutil
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_benchmark_nested_v2 import (shrink_fit, shrink_apply, macro_auc, _norm, MODULES,
                                    SHRINK_PRIMARY, N_SPLITS, N_REPEATS, SEED, OUT)

PROJ = "/home/jrkim/TSO_TFBS/project"
PLOT = f"{PROJ}/results/plot"
TB = f"{PROJ}/results/auto_plan/nc_readiness/tables"

# per-figure labels — must match each R script's factor levels EXACTLY
LAB_2A = {"E1_entropy": "E1 entropy", "All_exon_depth": "All-exon depth",
          "Genome_wide_CNA": "Broad CNA", "Somatic_mutation_profile": "Mutation profile",
          "SHAPE_nep300": "TFBS SHAPE"}
LAB_2B = {**LAB_2A, "Somatic_mutation_profile": "Mutation"}
RC2FIG = {"Exon1 entropy": "E1 entropy"}          # RC Shapley tables -> R factor levels


def per_class_auc(P, y, K):
    return [roc_auc_score((y == k).astype(int), P[:, k]) if 0 < (y == k).sum() < len(y) else np.nan
            for k in range(K)]


def main():
    a_rows, b_rows = [], []
    for coh in ("v1", "v2"):
        z = np.load(f"{OUT}/{coh}_oof.npz", allow_pickle=True)
        y = z["y"]; classes = [str(c) for c in z["classes"]]; K = len(classes); n = len(y)
        summ = json.load(open(f"{OUT}/{coh}_summary.json"))
        with open(f"{OUT}/{coh}_phase1.pkl", "rb") as f:
            D = pickle.load(f)
        trainOOF, testpred = D["trainOOF"], D["testpred"]
        outer = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                             random_state=SEED).split(np.zeros(n), y))
        print(f"[{coh}] n={n} K={K}", flush=True)

        # ---- replay phase-2: per-repeat module OOF + shrink60 ensemble OOF ----
        modOOF = {m: np.zeros((N_REPEATS, n, K)) for m in MODULES}
        ens = np.zeros((N_REPEATS, n, K))
        for f_, (OTR, OTE) in enumerate(outer):
            r = f_ // N_SPLITS
            for m in MODULES:
                modOOF[m][r][OTE] = testpred[f_][m]
            Wk, _ = shrink_fit(trainOOF[f_], y[OTR], K, MODULES, SHRINK_PRIMARY)
            ens[r][OTE] = shrink_apply(testpred[f_], Wk, K, MODULES)

        # ---- 2A: per-repeat macro-AUROC (+ CHECK vs published means) ----
        for m in MODULES:
            aucs = [macro_auc(modOOF[m][r], y, K) for r in range(N_REPEATS)]
            ref = summ["modules"][m]["mean"]
            if abs(float(np.mean(aucs)) - ref) > 5e-4:
                sys.exit(f"ABORT {coh}/{m}: 2A mean {np.mean(aucs):.4f} != published {ref:.4f}")
            print(f"    {LAB_2A[m]:16s} mean={np.mean(aucs):.4f} (published {ref:.4f}) OK", flush=True)
            for r, v in enumerate(aucs):
                a_rows.append(dict(group=LAB_2A[m], cohort=coh, repeat=r + 1,
                                   macroAUROC=round(float(v), 5),
                                   best_model=summ["modules"][m]["learner"]))
        eaucs = [macro_auc(ens[r], y, K) for r in range(N_REPEATS)]
        eref = summ["ensemble"][f"shrink{SHRINK_PRIMARY}"]["mean"]
        if abs(float(np.mean(eaucs)) - eref) > 5e-4:
            sys.exit(f"ABORT {coh}: 2A Ensemble {np.mean(eaucs):.4f} != published {eref:.4f}")
        print(f"    {'Ensemble':16s} mean={np.mean(eaucs):.4f} (published {eref:.4f}) OK", flush=True)
        for r, v in enumerate(eaucs):
            a_rows.append(dict(group="Ensemble", cohort=coh, repeat=r + 1,
                               macroAUROC=round(float(v), 5), best_model="nnls-SL"))

        # ---- 2B: per-cancer AUROC from pooled OOF (+ CHECK vs json perclass_auc) ----
        cnt = np.bincount(y, minlength=K)
        for m in MODULES:
            pooled = _norm(modOOF[m].mean(0))
            pc = per_class_auc(pooled, y, K)
            ref = summ["modules"][m]["perclass_auc"]
            d = max(abs(round(float(a), 4) - b) for a, b in zip(pc, ref) if b is not None)
            if d > 5e-4:
                sys.exit(f"ABORT {coh}/{m}: 2B per-class AUROC != json perclass_auc (max diff {d:.2e})")
            for k in range(K):
                b_rows.append(dict(cohort=coh, cancer_type=classes[k], n=int(cnt[k]),
                                   module=LAB_2B[m], AUROC=round(float(pc[k]), 3)))
        epc = per_class_auc(_norm(ens.mean(0)), y, K)
        for k in range(K):
            b_rows.append(dict(cohort=coh, cancer_type=classes[k], n=int(cnt[k]),
                               module="Ensemble", AUROC=round(float(epc[k]), 3)))
        print(f"[{coh}] 2B per-class AUROC matches published perclass_auc for all modules", flush=True)

    def bak(p):
        if os.path.exists(p) and not os.path.exists(p + ".prerc.bak"):
            shutil.copy2(p, p + ".prerc.bak"); print(f"  backed up {os.path.basename(p)} -> .prerc.bak")

    # ---- 2A / 2B ----
    for p, df in ((f"{PLOT}/fig2a_macroAUROC_perrepeat.csv", pd.DataFrame(a_rows)),
                  (f"{PLOT}/fig2b_percancer_auroc_long.csv", pd.DataFrame(b_rows))):
        bak(p); df.to_csv(p, index=False); print(f"WROTE {os.path.basename(p)}  rows={len(df)}")

    # ---- 2C: reuse the verified RC Shapley per-repeat table ----
    c = pd.read_csv(f"{PLOT}/fig2c_rc_shapley_perrepeat.csv")
    c["module"] = c["module"].replace(RC2FIG)
    c = c[["cohort", "module", "repeat", "shapley"]]
    p = f"{PLOT}/fig2c_shapley_perrepeat.csv"; bak(p); c.to_csv(p, index=False)
    print(f"WROTE fig2c_shapley_perrepeat.csv  rows={len(c)}")

    # ---- 2D: reuse the verified RC per-cancer Shapley table ----
    d = pd.read_csv(f"{TB}/Tab_rc_shapley_percancer.tsv", sep="\t")
    d["module"] = d["module"].replace(RC2FIG)
    d = (d.rename(columns={"cancer": "cancer_type", "shapley_mean": "shapley"})
           [["cohort", "cancer_type", "n", "module", "shapley"]])
    p = f"{PLOT}/fig2d_shapley_percancer_long.csv"; bak(p); d.to_csv(p, index=False)
    print(f"WROTE fig2d_shapley_percancer_long.csv  rows={len(d)}")
    print("RC_FIG2_DATA_DONE")


if __name__ == "__main__":
    main()
