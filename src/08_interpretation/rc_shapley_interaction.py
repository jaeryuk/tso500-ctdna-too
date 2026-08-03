#!/usr/bin/env python3
"""Pairwise SHAPLEY INTERACTION INDEX (Grabisch & Roubens) for the RC late-fusion ensemble — global and
per cancer type.

For a pair {i,j}, the discrete second derivative on coalition S is
    delta_ij(S) = v(S u {i,j}) - v(S u {i}) - v(S u {j}) + v(S)
and the interaction index averages it over all S in N\\{i,j} with the Grabisch-Roubens weights:
    I(i,j) = sum_{S subset N\\{i,j}}  s!(n-s-2)!/(n-1)!  * delta_ij(S)          [n=5 -> 8 subsets per pair]

SIGN:  I < 0  = REDUNDANCY  (the two modules overlap; together they add less than the sum of their parts)
       I > 0  = SYNERGY     (complementary; together they add more than the sum of their parts)

Same game as rc_shapley.py: v(S) = macro-AUROC (global) or v_k(S) = OVR AUROC of cancer k (per-cancer)
of the shrink60 NNLS late fusion on subset S; v(empty)=0.5. Nested (combiner fit on outer-TRAIN
meta-OOF, applied to held-out outer-TEST) using the benchmark's OWN shrink_fit/shrink_apply. Runs off
{coh}_phase1.pkl — no CV re-run. Per-repeat (20) -> mean +- SD.

THE CHECK THAT MATTERS. The Grabisch-Roubens index is defined for any coalition T; for |T|=1 it
provably COLLAPSES to the ordinary Shapley value:
    I({i}) = sum_{S subset N\\{i}} s!(n-s-1)!/n! [v(S u {i}) - v(S)] = phi_i
So the SAME generic code path that computes the |T|=2 interactions is run at |T|=1 and asserted against
the already-verified macro phi in Tab_rc_shapley.tsv. If the interaction machinery (weights, subset
enumeration, sign convention) were wrong, the singleton case would not reproduce phi. This is what makes
the pairwise numbers trustworthy — they are otherwise unfalsifiable by eye.

Plus: linearity (mean_k I_k(i,j) == I_global(i,j), since macro-AUROC is the mean of the per-class AUROCs
and I is linear in v) and symmetry (I(i,j) == I(j,i)).

Out: tables/Tab_rc_shapley_interaction_global.tsv, tables/Tab_rc_shapley_interaction_percancer.tsv
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
    """-> (N_REPEATS, K) per-class OVR AUROC for module subset S. Global v = mean over K."""
    if len(S) == 0:
        return np.full((N_REPEATS, K), V_EMPTY)
    mods = [mods_all[j] for j in S]
    ens = np.zeros((N_REPEATS, n, K))
    for f_, (OTR, OTE) in enumerate(outer):
        r = f_ // N_SPLITS
        Wk, _ = shrink_fit(trainOOF[f_], y[OTR], K, mods, SHRINK_PRIMARY)
        ens[r][OTE] = shrink_apply(testpred[f_], Wk, K, mods)
    return np.array([perclass_auc_vec(ens[r], y, K) for r in range(N_REPEATS)])


def interaction(T, vS, idx, n):
    """Grabisch-Roubens interaction index for coalition T (generic |T|).

    I(T) = sum_{S subset N\\T} [s!(n-s-t)!/(n-t+1)!] * sum_{L subset T} (-1)^(t-|L|) v(S u L)

    |T|=1 collapses to the Shapley value -- which is exactly how this code path gets validated.
    Returns an array shaped like vS's value (per-repeat, or per-repeat x class).
    """
    T = tuple(sorted(T)); t = len(T)
    others = [j for j in idx if j not in T]
    acc = None
    for size in range(len(others) + 1):
        w = math.factorial(size) * math.factorial(n - size - t) / math.factorial(n - t + 1)
        for S in itertools.combinations(others, size):
            inner = None
            for L_size in range(t + 1):
                for L in itertools.combinations(T, L_size):
                    sign = (-1) ** (t - L_size)
                    term = sign * vS[tuple(sorted(S + L))]
                    inner = term if inner is None else inner + term
            contrib = w * inner
            acc = contrib if acc is None else acc + contrib
    return acc


def main():
    n = len(MODULES)
    idx = list(range(n))
    pairs = list(itertools.combinations(idx, 2))
    macro_ref = pd.read_csv(f"{TB}/Tab_rc_shapley.tsv", sep="\t")
    grows, prows = [], []

    for coh in ("v1", "v2"):
        t0 = time.time()
        z = np.load(f"{OUT}/{coh}_oof.npz", allow_pickle=True)
        y = z["y"]; classes = [str(c) for c in z["classes"]]; K = len(classes); nn = len(y)
        with open(f"{OUT}/{coh}_phase1.pkl", "rb") as f:
            D = pickle.load(f)
        trainOOF, testpred = D["trainOOF"], D["testpred"]
        outer = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                             random_state=SEED).split(np.zeros(nn), y))
        print(f"[{coh}] n={nn} K={K}", flush=True)

        subsets = [S for size in range(n + 1) for S in itertools.combinations(idx, size)]
        vals = Parallel(n_jobs=NJOB, prefer="processes", max_nbytes="50M")(
            delayed(subset_values)(S, MODULES, trainOOF, testpred, outer, y, K, nn) for S in subsets)
        vS_k = {S: v for S, v in zip(subsets, vals)}                       # (R, K)
        vS_g = {S: np.nanmean(v, axis=1) for S, v in vS_k.items()}         # (R,) macro
        print(f"[{coh}] {len(subsets)} subsets valued in {time.time()-t0:.0f}s", flush=True)

        # ---- CHECK: |T|=1 must reproduce the verified macro Shapley ----
        worst = 0.0
        for i in idx:
            got = float(interaction((i,), vS_g, idx, n).mean())
            ref = float(macro_ref[(macro_ref.cohort == coh) &
                                  (macro_ref.module == LABEL[MODULES[i]])].shapley_mean.iloc[0])
            worst = max(worst, abs(got - ref))
            print(f"    I({LABEL[MODULES[i]]:15s}) = {got:.4f}   macro phi = {ref:.4f}   diff={abs(got-ref):.2e}",
                  flush=True)
        if worst > 5e-4:
            sys.exit(f"ABORT {coh}: singleton interaction != Shapley value (max diff {worst:.2e}) "
                     "-> interaction machinery is wrong; pairwise values not trustworthy")
        print(f"[{coh}] singleton==Shapley OK (max diff {worst:.2e}) -> pairwise index validated", flush=True)

        # ---- pairwise, global + per cancer ----
        for (i, j) in pairs:
            Ig = interaction((i, j), vS_g, idx, n)                      # (R,)
            Ik = interaction((i, j), vS_k, idx, n)                      # (R, K)
            # symmetry (cheap, exact by construction — assert anyway)
            if abs(float(interaction((j, i), vS_g, idx, n).mean()) - float(Ig.mean())) > 1e-9:
                sys.exit(f"ABORT {coh}: interaction not symmetric for {MODULES[i]},{MODULES[j]}")
            # linearity: mean_k I_k == I_global
            dlin = abs(float(np.nanmean(Ik, axis=1).mean()) - float(Ig.mean()))
            if dlin > 5e-4:
                sys.exit(f"ABORT {coh}: linearity failed for pair ({MODULES[i]},{MODULES[j]}): {dlin:.2e}")
            grows.append(dict(cohort=coh, module_i=LABEL[MODULES[i]], module_j=LABEL[MODULES[j]],
                              interaction=round(float(Ig.mean()), 4),
                              sd=round(float(Ig.std(ddof=1)), 4)))
            cnt = np.bincount(y, minlength=K)
            for k in range(K):
                prows.append(dict(cohort=coh, cancer=CMAP.get(classes[k], classes[k]), n=int(cnt[k]),
                                  module_i=LABEL[MODULES[i]], module_j=LABEL[MODULES[j]],
                                  pair=f"{LABEL[MODULES[i]]} x {LABEL[MODULES[j]]}",
                                  interaction=round(float(Ik[:, k].mean()), 4),
                                  sd=round(float(Ik[:, k].std(ddof=1)), 4)))
        print(f"[{coh}] linearity + symmetry OK for all {len(pairs)} pairs", flush=True)

    G = pd.DataFrame(grows); P = pd.DataFrame(prows)
    os.makedirs(TB, exist_ok=True)
    G.to_csv(f"{TB}/Tab_rc_shapley_interaction_global.tsv", sep="\t", index=False)
    P.to_csv(f"{TB}/Tab_rc_shapley_interaction_percancer.tsv", sep="\t", index=False)
    for coh in ("v1", "v2"):
        print(f"\n=== [{coh}] GLOBAL pairwise interaction (neg = redundant, pos = synergistic) ===")
        print(G[G.cohort == coh].sort_values("interaction").to_string(index=False))
    print(f"\nsaved -> Tab_rc_shapley_interaction_global.tsv / _percancer.tsv")
    print("RC_INTERACTION_DONE")


if __name__ == "__main__":
    main()
