#!/usr/bin/env python3
"""EXACT module-level Shapley decomposition of the RC late-fusion macro-AUROC.

Game: 5 players = the RC ensemble modules. v(S) = macro-AUROC of the shrink60 NNLS late fusion built
from ONLY the modules in S. v(empty) := 0.5 (chance). Exact Shapley over all 2^5 = 32 subsets, computed
PER REPEAT (20) -> a distribution of phi_i, not a point estimate. Efficiency: sum_i phi_i = v(N) - 0.5.

WHY THIS IS NOT THE OLD fig2c_shapley.py (which was keep_t1, pre-RC, SHAPE=gini5_len5):
  1. NESTED, not in-sample. The old fuse_macro fit the NNLS combiner on the full OOF and scored the same
     rows -> the combiner saw y for the rows it was graded on. Small optimism (few non-negative weights,
     n~1000) but real. Here the combiner is fit on the outer-TRAIN meta-OOF (trainOOF) and applied to the
     held-out outer-TEST (testpred), exactly as rc_benchmark_nested_v2 does for its own ensembles.
  2. Uses the benchmark's OWN functions (imported, not copied): shrink_fit/shrink_apply/macro_auc/_norm,
     shrink const = SHRINK_PRIMARY. So v(N) IS the benchmark's shrink60 and v(N\\{i}) IS its LOMO_drop_i.

That last point gives a hard check, asserted below: the s=4 Shapley marginals must reproduce the
benchmark's published LOMO deltas to floating point. If they do not, the game is not the benchmark's
ensemble and the decomposition is measuring something else.

Input: benchmark_nested_v2/{coh}_phase1.pkl (trainOOF/testpred per outer fold) -- NO model refits, no CV
re-run. The outer folds are regenerated from RepeatedStratifiedKFold(5, 20, random_state=SEED), which is
deterministic given y, and y is verified against the saved OOF before use.

Out: results/plot/fig2c_rc_shapley_perrepeat.csv, tables/Tab_rc_shapley.tsv
"""
import os, sys, json, math, pickle, itertools, time
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_benchmark_nested_v2 import (shrink_fit, shrink_apply, macro_auc, MODULES,
                                    SHRINK_PRIMARY, N_SPLITS, N_REPEATS, SEED, OUT)

PROJ = "/home/jrkim/TSO_TFBS/project"
TB = f"{PROJ}/results/auto_plan/nc_readiness/tables"
PLOT = f"{PROJ}/results/plot"
V_EMPTY = 0.5
NJOB = int(os.environ.get("NJOB", "16"))        # shared host (GRAIL download + docker snvqual) -> modest
LABEL = {"E1_entropy": "Exon1 entropy", "All_exon_depth": "All-exon depth",
         "Genome_wide_CNA": "Broad CNA", "Somatic_mutation_profile": "Mutation",
         "SHAPE_nep300": "TFBS SHAPE"}


def subset_values(S, mods_all, trainOOF, testpred, outer, y, K, n):
    """Per-repeat macro-AUROC of the shrink60 fusion restricted to module subset S. Fully nested."""
    if len(S) == 0:
        return np.full(N_REPEATS, V_EMPTY)
    mods = [mods_all[j] for j in S]
    ens = np.zeros((N_REPEATS, n, K))
    for f_, (OTR, OTE) in enumerate(outer):
        r = f_ // N_SPLITS
        Wk, _ = shrink_fit(trainOOF[f_], y[OTR], K, mods, SHRINK_PRIMARY)   # fit on outer-TRAIN
        ens[r][OTE] = shrink_apply(testpred[f_], Wk, K, mods)               # apply to held-out outer-TEST
    return np.array([macro_auc(ens[r], y, K) for r in range(N_REPEATS)])


def main():
    n_mod = len(MODULES)
    coef = {s: math.factorial(s) * math.factorial(n_mod - s - 1) / math.factorial(n_mod)
            for s in range(n_mod)}
    idx = list(range(n_mod))
    rows, tab = [], []

    for coh in ("v1", "v2"):
        t0 = time.time()
        z = np.load(f"{OUT}/{coh}_oof.npz", allow_pickle=True)
        y = z["y"]; classes = [str(c) for c in z["classes"]]; K = len(classes); n = len(y)
        with open(f"{OUT}/{coh}_phase1.pkl", "rb") as f:
            D = pickle.load(f)
        trainOOF, testpred = D["trainOOF"], D["testpred"]
        outer = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                             random_state=SEED).split(np.zeros(n), y))
        if len(outer) != len(trainOOF):
            sys.exit(f"ABORT {coh}: regenerated {len(outer)} outer folds != checkpoint {len(trainOOF)}")
        print(f"[{coh}] n={n} K={K} modules={MODULES} folds={len(outer)}", flush=True)

        # ---- value of every subset, per repeat ----
        subsets = [S for size in range(n_mod + 1) for S in itertools.combinations(idx, size)]
        vals = Parallel(n_jobs=NJOB, prefer="processes", max_nbytes="50M")(
            delayed(subset_values)(S, MODULES, trainOOF, testpred, outer, y, K, n) for S in subsets)
        vS = {S: v for S, v in zip(subsets, vals)}
        print(f"[{coh}] {len(subsets)} subsets valued in {time.time()-t0:.0f}s", flush=True)

        # ---- VERIFY the game is the benchmark's own ensemble ----
        summ = json.load(open(f"{OUT}/{coh}_summary.json"))
        full = float(np.mean(vS[tuple(idx)]))
        ref_full = summ["ensemble"][f"shrink{SHRINK_PRIMARY}"]["mean"]
        d_full = abs(full - ref_full)
        print(f"[{coh}] v(N)={full:.4f} vs benchmark shrink{SHRINK_PRIMARY}={ref_full:.4f}  diff={d_full:.2e}",
              flush=True)
        if d_full > 5e-4:
            sys.exit(f"ABORT {coh}: v(N) != benchmark primary ensemble -> wrong game")
        worst = 0.0
        for i in idx:
            S_i = tuple(j for j in idx if j != i)
            got = float(np.mean(vS[S_i]))
            ref = summ["ensemble"][f"LOMO_drop_{MODULES[i]}"]["mean"]
            worst = max(worst, abs(got - ref))
            print(f"    v(N\\{{{MODULES[i]:24s}}})={got:.4f}  benchmark LOMO={ref:.4f}  diff={abs(got-ref):.2e}",
                  flush=True)
        if worst > 5e-4:
            sys.exit(f"ABORT {coh}: s=4 marginals do not reproduce published LOMO (max diff {worst:.2e})")
        print(f"[{coh}] LOMO reproduction OK (max diff {worst:.2e})", flush=True)

        # ---- exact Shapley, per repeat (kept UNROUNDED; see the efficiency check below) ----
        PHI = np.zeros((N_REPEATS, n_mod))
        for r in range(N_REPEATS):
            for i in idx:
                others = [j for j in idx if j != i]
                phi = 0.0
                for size in range(len(others) + 1):
                    for S in itertools.combinations(others, size):
                        phi += coef[size] * (vS[tuple(sorted(S + (i,)))][r] - vS[tuple(sorted(S))][r])
                PHI[r, i] = phi

        # ---- efficiency axiom, on UNROUNDED phi ----
        # Checking the ROUNDED phi (what the csv stores, 5 dp) fails a 1e-6 tolerance for a reason that
        # has nothing to do with the maths: summing 5 values each rounded to 5 dp admits up to 2.5e-5 of
        # pure rounding error. A first pass aborted here at 1.47e-05 -- the axiom was fine, the test was
        # measuring its own rounding. Round for output only.
        eff = float(np.abs(PHI.sum(1) - (vS[tuple(idx)] - V_EMPTY)).max())
        print(f"[{coh}] efficiency: max|sum(phi) - (v(N)-0.5)| = {eff:.2e}   "
              f"mean sum(phi)={PHI.sum(1).mean():.4f}  v(N)-0.5={full-0.5:.4f}", flush=True)
        if eff > 1e-6:
            sys.exit(f"ABORT {coh}: Shapley efficiency axiom violated ({eff:.2e})")

        for r in range(N_REPEATS):
            for i in idx:
                rows.append(dict(cohort=coh, module=LABEL[MODULES[i]], repeat=r + 1,
                                 shapley=round(PHI[r, i], 5)))
        for i in idx:
            s = PHI[:, i]
            lomo = full - float(np.mean(vS[tuple(j for j in idx if j != i)]))
            tab.append(dict(cohort=coh, module=LABEL[MODULES[i]], shapley_mean=round(float(s.mean()), 4),
                            shapley_sd=round(float(s.std(ddof=1)), 4),
                            shapley_pct=round(100 * float(s.mean()) / (full - V_EMPTY), 1),
                            lomo_delta=round(lomo, 4)))

    D = pd.DataFrame(rows)
    os.makedirs(TB, exist_ok=True)
    D.to_csv(f"{PLOT}/fig2c_rc_shapley_perrepeat.csv", index=False)
    D.to_csv(f"{TB}/fig2c_rc_shapley_perrepeat.csv", index=False)
    T = pd.DataFrame(tab); T.to_csv(f"{TB}/Tab_rc_shapley.tsv", sep="\t", index=False)
    print("\n=== RC Shapley (macro-AUROC points) — mean over 20 repeats ===")
    for coh in ("v1", "v2"):
        t = T[T.cohort == coh].sort_values("shapley_mean", ascending=False)
        print(f"\n[{coh}]"); print(t.to_string(index=False))
    print(f"\nrows={len(D)} -> {PLOT}/fig2c_rc_shapley_perrepeat.csv, {TB}/Tab_rc_shapley.tsv")
    print("RC_SHAPLEY_DONE")


if __name__ == "__main__":
    main()
