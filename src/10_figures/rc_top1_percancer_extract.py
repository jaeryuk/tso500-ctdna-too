#!/usr/bin/env python3
"""Per-cancer, per-module TOP-1 ACCURACY on the RULE-CONFORMANT cohort, from the RC benchmark OOF
(results/rule_conformant/benchmark/{v1,v2}_oof.npz).

Per-cancer top-1 = among the samples whose true label is that cancer, the fraction where the module's
argmax class == the true class (i.e. per-class recall of the top-1 call). This is the top-1 analog of the
per-cancer AUROC heatmap (Fig2B_percancer_auroc_heatmap).

Out: results/plot/Tab_rc_top1_percancer.tsv  (cohort, cancer, n, module, top1)
"""
import numpy as np, pandas as pd

PROJ = "/home/jrkim/TSO_TFBS/project"
PLOT = f"{PROJ}/results/plot"
# REPOINTED 2026-07-17 to the authoritative RC benchmark. This previously read
# results/rule_conformant/benchmark — a DIFFERENT, older run (2026-07-08) whose OOF keys were
# {E1_entropy..., SHAPE, late_fusion}. benchmark_nested_v2 (RC_BENCH_V2_DONE 2026-07-17) is the
# published one and renames both: SHAPE -> mod_SHAPE_nep300 (the nep>300 site-filtered SHAPE that is
# now the standing definition) and late_fusion -> primary (the shrink60 NNLS Super-Learner).
# Cohort is unchanged between the two (y and classes verified identical), so only the model changes.
BEN = f"{PROJ}/results/rule_conformant/benchmark_nested_v2"
# labels match the reference figures (Fig2B/Fig2C)
MODS = [("mod_E1_entropy", "E1 entropy"), ("mod_All_exon_depth", "All-exon depth"),
        ("mod_Genome_wide_CNA", "Broad CNA"), ("mod_Somatic_mutation_profile", "Somatic mutation"),
        ("mod_SHAPE_nep300", "TFBS SHAPE"), ("primary", "Ensemble")]


def main():
    rows = []
    for coh in ("v1", "v2"):
        z = np.load(f"{BEN}/{coh}_oof.npz", allow_pickle=True)
        y = np.asarray(z["y"])
        classes = [str(c) for c in z["classes"]]
        pred = {disp: np.asarray(z[key]).argmax(1) for key, disp in MODS}
        for ci, cname in enumerate(classes):
            m = (y == ci)
            if m.sum() == 0:
                continue
            for _, disp in MODS:
                rows.append(dict(cohort=coh, cancer=cname, n=int(m.sum()), module=disp,
                                 top1=round(float((pred[disp][m] == ci).mean()), 4)))
        print(f"[{coh}] n={len(y)} K={len(classes)} classes={classes}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{PLOT}/Tab_rc_top1_percancer.tsv", sep="\t", index=False)
    # console view: cancer x module top-1 grid per cohort
    for coh in ("v1", "v2"):
        d = df[df.cohort == coh]
        piv = d.pivot_table(index=["cancer", "n"], columns="module", values="top1")
        piv = piv.reindex(columns=[m for _, m in MODS]).sort_index(level="n", ascending=False)
        print(f"\n=== {coh} per-cancer TOP-1 ===\n{piv.to_string()}", flush=True)
    print("\nRC_TOP1_PERCANCER_DONE", flush=True)


if __name__ == "__main__":
    main()
