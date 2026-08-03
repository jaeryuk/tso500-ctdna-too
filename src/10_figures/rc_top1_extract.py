#!/usr/bin/env python3
"""Top-1 accuracy per module + ensemble on the RULE-CONFORMANT cohort, from the authoritative RC benchmark OOF
(results/rule_conformant/benchmark_nested_v2/{v1,v2}_oof.npz — per-module OOF class-probability matrices).
Outputs (results/plot/):
  Tab_rc_top1_bootstrap.tsv   (cohort, module, rep, top1)          -> boxplot (1000 stratified bootstraps)
  Tab_rc_top1_by_tfbin.tsv    (cohort, module, bin, n, top1)       -> lineplot vs tumor-fraction bin
Mirrors the macro-AUROC module boxplot / tumor-fraction lineplot, but with top-1 accuracy on the RC cohort."""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_perf_vs_tfbin import rebuild_order      # deterministic sid order + y self-check

PROJ = "/home/jrkim/TSO_TFBS/project"
PLOT = f"{PROJ}/results/plot"
# REPOINTED 2026-07-17 to the authoritative RC benchmark. This previously read
# results/rule_conformant/benchmark — a DIFFERENT, older run (2026-07-08) whose OOF keys were
# {E1_entropy..., SHAPE, late_fusion}. benchmark_nested_v2 (RC_BENCH_V2_DONE 2026-07-17) is the
# published one and renames both: SHAPE -> mod_SHAPE_nep300 (the nep>300 site-filtered SHAPE that is
# now the standing definition) and late_fusion -> primary (the shrink60 NNLS Super-Learner).
# Cohort is unchanged between the two (y and classes verified identical), so only the model changes.
BEN = f"{PROJ}/results/rule_conformant/benchmark_nested_v2"
MODS = [("mod_E1_entropy", "Exon1 entropy"), ("mod_All_exon_depth", "All-exon depth"),
        ("mod_Genome_wide_CNA", "Broad CNA"), ("mod_Somatic_mutation_profile", "Somatic mutation"),
        ("mod_SHAPE_nep300", "TFBS SHAPE"), ("primary", "Ensemble")]
NBOOT = 1000
BINS = [("<3%", 0.0, 0.03), ("3-10%", 0.03, 0.10), ("10-20%", 0.10, 0.20), (">20%", 0.20, 1.01)]

# tumor fraction (numeric sid -> tf), RC ichorCNA table, both cohorts
TF = {}
for c in ("rcv1", "rcv2"):
    d = pd.read_csv(f"{PROJ}/results/{c}_cna/cna_feature_table.tsv", sep="\t")
    d = d.rename(columns={d.columns[0]: "sid"})
    for s, t in zip(d.sid, d.tumor_fraction):
        if pd.notna(t):
            TF[str(int(s))] = float(t)


def strat_boot_idx(y, rng):
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        idx.append(ci[rng.integers(0, len(ci), len(ci))])
    return np.concatenate(idx)


def main():
    boot_rows, bin_rows = [], []
    for coh in ("v1", "v2"):
        z = np.load(f"{BEN}/{coh}_oof.npz", allow_pickle=True)
        y = np.asarray(z["y"])
        # benchmark_nested_v2's OOF drops the `order` key the old benchmark carried, so the sid order
        # (needed only to join tumour fraction for the by-TF-bin table) is rebuilt deterministically and
        # VERIFIED: rebuild_order re-derives y and aborts unless it matches the saved y element-wise.
        # Without that check a wrong order would silently scramble the TF join into plausible nonsense.
        sids, y_re, _ = rebuild_order(coh)
        if not np.array_equal(y_re, y):
            sys.exit(f"ABORT {coh}: rebuilt order's y != saved y -> tumour-fraction join unsafe")
        pred = {disp: np.asarray(z[key]).argmax(1) for key, disp in MODS}
        tf = np.array([TF.get(s, np.nan) for s in sids])
        # bootstrap overall top-1 accuracy
        rng = np.random.default_rng(42)
        for r in range(NBOOT):
            bi = strat_boot_idx(y, rng)
            yb = y[bi]
            for _, disp in MODS:
                boot_rows.append(dict(cohort=coh, module=disp, rep=r,
                                      top1=float((pred[disp][bi] == yb).mean())))
        # per-TF-bin top-1 accuracy (+overall)
        for blab, lo, hi in [("overall", -1, 2)] + BINS:
            m = np.ones(len(y), bool) if blab == "overall" else (np.isfinite(tf) & (tf >= lo) & (tf < hi))
            if m.sum() == 0:
                continue
            for _, disp in MODS:
                bin_rows.append(dict(cohort=coh, module=disp, bin=blab, n=int(m.sum()),
                                     top1=round(float((pred[disp][m] == y[m]).mean()), 4)))
        print(f"[{coh}] n={len(y)}  overall top1: "
              + ", ".join(f"{disp}={float((pred[disp]==y).mean()):.3f}" for _, disp in MODS), flush=True)
    pd.DataFrame(boot_rows).to_csv(f"{PLOT}/Tab_rc_top1_bootstrap.tsv", sep="\t", index=False)
    pd.DataFrame(bin_rows).to_csv(f"{PLOT}/Tab_rc_top1_by_tfbin.tsv", sep="\t", index=False)
    print("RC_TOP1_DONE", flush=True)


if __name__ == "__main__":
    main()
