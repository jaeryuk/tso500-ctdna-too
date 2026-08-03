#!/usr/bin/env python3
"""PER-CANCER exact Shapley + LOMO for the RC late-fusion ensemble.

Characteristic function for cancer k: v_k(S) = one-vs-rest AUROC of cancer k from the shrink60 NNLS
late fusion built on module subset S. v_k(empty) := 0.5. Exact Shapley over all 2^5 subsets, per repeat
(20). Per-cancer LOMO delta = v_k(N) - v_k(N\\{i}) — the s=4 marginal, i.e. UNIQUE information for k.

Same machinery as rc_shapley.py (nested: combiner fit on outer-TRAIN meta-OOF, applied to held-out
outer-TEST, using the benchmark's OWN shrink_fit/shrink_apply). Runs off {coh}_phase1.pkl, no CV re-run.

THREE CHECKS, all asserted:
  1. LINEARITY. Shapley is a linear operator on v, and macro-AUROC is the unweighted mean of the K
     per-class AUROCs -> mean_k phi_i^(k) MUST equal the macro phi_i from rc_shapley.py
     (Tab_rc_shapley.tsv). This ties this run to the previous one and is the strongest check available.
  2. EFFICIENCY, per cancer: sum_i phi_i^(k) = v_k(N) - 0.5, for every k and every repeat.
  3. MACRO ANCHOR: mean_k v_k(N) = the published shrink60 mean (0.8680 / 0.8784).

GOTCHA carried over from rc_shapley.py: check efficiency on UNROUNDED phi. Rounding to 5 dp and then
summing 5 modules admits up to 2.5e-5 of pure rounding error, which alone fails a 1e-6 tolerance.

Out: tables/Tab_rc_shapley_percancer.tsv (cohort, cancer, n, module, shapley_mean, shapley_sd, lomo_delta)
"""
import os, sys, json, math, pickle, itertools, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_benchmark_nested_v2 import (shrink_fit, shrink_apply, MODULES, SHRINK_PRIMARY,
                                    N_SPLITS, N_REPEATS, SEED, OUT)

PROJ = "/home/jrkim/TSO_TFBS/project"
TB = f"{PROJ}/results/auto_plan/nc_readiness/tables"
V_EMPTY = 0.5
NJOB = int(os.environ.get("NJOB", "16"))
LABEL = {"E1_entropy": "Exon1 entropy", "All_exon_depth": "All-exon depth",
         "Genome_wide_CNA": "Broad CNA", "Somatic_mutation_profile": "Mutation",
         "SHAPE_nep300": "TFBS SHAPE"}
CMAP = {"lung cancer": "Lung cancer", "colorectal cancer": "Colorectal", "gastric cancer": "Stomach",
        "pancreatic cancer": "Pancreas", "biliary tract cancer": "Biliary", "melanoma": "Melanoma",
        "liver cancer": "Liver", "prostate cancer": "Prostate", "breast cancer": "Breast",
        "sarcoma": "Sarcoma", "bladder cancer": "Bladder"}


def perclass_auc_vec(P, y, K):
    return np.array([roc_auc_score((y == k).astype(int), P[:, k]) if 0 < (y == k).sum() < len(y) else np.nan
                     for k in range(K)])


def subset_values(S, mods_all, trainOOF, testpred, outer, y, K, n):
    """-> (N_REPEATS, K) per-class OVR AUROC of the shrink60 fusion restricted to subset S."""
    if len(S) == 0:
        return np.full((N_REPEATS, K), V_EMPTY)
    mods = [mods_all[j] for j in S]
    ens = np.zeros((N_REPEATS, n, K))
    for f_, (OTR, OTE) in enumerate(outer):
        r = f_ // N_SPLITS
        Wk, _ = shrink_fit(trainOOF[f_], y[OTR], K, mods, SHRINK_PRIMARY)
        ens[r][OTE] = shrink_apply(testpred[f_], Wk, K, mods)
    return np.array([perclass_auc_vec(ens[r], y, K) for r in range(N_REPEATS)])


def main():
    n_mod = len(MODULES)
    coef = {s: math.factorial(s) * math.factorial(n_mod - s - 1) / math.factorial(n_mod)
            for s in range(n_mod)}
    idx = list(range(n_mod))
    macro_ref = pd.read_csv(f"{TB}/Tab_rc_shapley.tsv", sep="\t")
    rows = []

    for coh in ("v1", "v2"):
        t0 = time.time()
        z = np.load(f"{OUT}/{coh}_oof.npz", allow_pickle=True)
        y = z["y"]; classes = [str(c) for c in z["classes"]]; K = len(classes); n = len(y)
        with open(f"{OUT}/{coh}_phase1.pkl", "rb") as f:
            D = pickle.load(f)
        trainOOF, testpred = D["trainOOF"], D["testpred"]
        outer = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                             random_state=SEED).split(np.zeros(n), y))
        print(f"[{coh}] n={n} K={K} classes={classes}", flush=True)

        subsets = [S for size in range(n_mod + 1) for S in itertools.combinations(idx, size)]
        vals = Parallel(n_jobs=NJOB, prefer="processes", max_nbytes="50M")(
            delayed(subset_values)(S, MODULES, trainOOF, testpred, outer, y, K, n) for S in subsets)
        vS = {S: v for S, v in zip(subsets, vals)}
        print(f"[{coh}] {len(subsets)} subsets valued in {time.time()-t0:.0f}s", flush=True)

        vN = vS[tuple(idx)]                                  # (R, K)
        # --- check 3: macro anchor ---
        summ = json.load(open(f"{OUT}/{coh}_summary.json"))
        ref_full = summ["ensemble"][f"shrink{SHRINK_PRIMARY}"]["mean"]
        got_macro = float(np.nanmean(vN, axis=1).mean())
        print(f"[{coh}] mean_k v_k(N) = {got_macro:.4f}  vs published shrink{SHRINK_PRIMARY} = {ref_full:.4f}"
              f"  diff={abs(got_macro-ref_full):.2e}", flush=True)
        if abs(got_macro - ref_full) > 5e-4:
            sys.exit(f"ABORT {coh}: per-class macro != published ensemble")

        # --- exact per-cancer Shapley (unrounded) ---
        PHI = np.zeros((N_REPEATS, n_mod, K))
        for r in range(N_REPEATS):
            for k in range(K):
                for i in idx:
                    others = [j for j in idx if j != i]
                    phi = 0.0
                    for size in range(len(others) + 1):
                        for S in itertools.combinations(others, size):
                            phi += coef[size] * (vS[tuple(sorted(S + (i,)))][r, k] - vS[tuple(sorted(S))][r, k])
                    PHI[r, i, k] = phi

        # --- check 2: per-cancer efficiency ---
        eff = float(np.abs(PHI.sum(1) - (vN - V_EMPTY)).max())
        print(f"[{coh}] per-cancer efficiency: max|sum_i phi_i^(k) - (v_k(N)-0.5)| = {eff:.2e}", flush=True)
        if eff > 1e-6:
            sys.exit(f"ABORT {coh}: per-cancer efficiency violated ({eff:.2e})")

        # --- check 1: linearity vs the macro Shapley ---
        worst = 0.0
        for i in idx:
            got = float(PHI[:, i, :].mean(axis=1).mean())       # mean over cancers, then repeats
            ref = float(macro_ref[(macro_ref.cohort == coh) &
                                  (macro_ref.module == LABEL[MODULES[i]])].shapley_mean.iloc[0])
            worst = max(worst, abs(got - ref))
            print(f"    mean_k phi[{LABEL[MODULES[i]]:15s}] = {got:.4f}  macro Shapley = {ref:.4f}"
                  f"  diff={abs(got-ref):.2e}", flush=True)
        if worst > 5e-4:
            sys.exit(f"ABORT {coh}: linearity check failed (max diff {worst:.2e}) -> per-cancer game "
                     "is not consistent with the macro game")
        print(f"[{coh}] linearity OK (max diff {worst:.2e})", flush=True)

        cnt = np.bincount(y, minlength=K)
        for k in range(K):
            for i in idx:
                s = PHI[:, i, k]
                lomo = float(np.mean(vN[:, k] - vS[tuple(j for j in idx if j != i)][:, k]))
                rows.append(dict(cohort=coh, cancer=CMAP.get(classes[k], classes[k]), n=int(cnt[k]),
                                 module=LABEL[MODULES[i]], shapley_mean=round(float(s.mean()), 4),
                                 shapley_sd=round(float(s.std(ddof=1)), 4),
                                 lomo_delta=round(lomo, 4),
                                 auc_full=round(float(vN[:, k].mean()), 4)))

    R = pd.DataFrame(rows)
    os.makedirs(TB, exist_ok=True)
    R.to_csv(f"{TB}/Tab_rc_shapley_percancer.tsv", sep="\t", index=False)
    for coh in ("v1", "v2"):
        print(f"\n=== [{coh}] per-cancer Shapley (macro-AUROC points) ===")
        print(R[R.cohort == coh].pivot(index="module", columns="cancer", values="shapley_mean").to_string())
        print(f"\n=== [{coh}] per-cancer LOMO delta ===")
        print(R[R.cohort == coh].pivot(index="module", columns="cancer", values="lomo_delta").to_string())
    print(f"\nsaved -> {TB}/Tab_rc_shapley_percancer.tsv")
    print("RC_SHAPLEY_PERCANCER_DONE")


if __name__ == "__main__":
    main()
