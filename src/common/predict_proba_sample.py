#!/usr/bin/env python3
"""Per-cancer PREDICTION PROBABILITY for a specific sample (sid).

Surfaces the late-fusion ensemble's OUT-OF-FOLD posterior from the definitive nested benchmark
(benchmark_nested_v2/{coh}_oof.npz -> `primary`, shape (n, K), rows sum to 1). Because it is OOF,
each sample's probability comes from a model that never saw that sample in training (honest held-out).

The npz stores rows in the benchmark's sample order but not the sids; rebuild_order() reconstructs
that exact order (same code path the benchmark used) and we verify y matches before indexing.

Usage:
  predict_proba_sample.py <sid> [v1|v2]     # one sample; auto-detects cohort if omitted
  predict_proba_sample.py --list [v1|v2]    # list available sids
With no cohort given the sid is looked up in whichever cohort contains it (v1 tried first).
"""
import os, sys
import numpy as np

PROJ = "/home/jrkim/TSO_TFBS/project"
RC = f"{PROJ}/results/rule_conformant"
FEAT = f"{PROJ}/results/auto_plan/feat_rc"
MAN = f"{RC}/manifest_dev.tsv"
OUT = f"{RC}/benchmark_nested_v2"
FILE_MODULES = ["Genome_wide_CNA", "E1_entropy", "Somatic_mutation_profile", "All_exon_depth"]
MIN_CLASS = 20
MODLABEL = {"mod_E1_entropy": "E1 entropy", "mod_All_exon_depth": "All-exon depth",
            "mod_Genome_wide_CNA": "Broad CNA", "mod_Somatic_mutation_profile": "Somatic mutation",
            "mod_SHAPE_nep300": "TFBS SHAPE"}
SHORT = {"biliary tract cancer": "Biliary", "bladder cancer": "Bladder", "breast cancer": "Breast",
         "colorectal cancer": "Colorectal", "gastric cancer": "Stomach", "liver cancer": "Liver",
         "lung cancer": "Lung", "melanoma": "Melanoma", "pancreatic cancer": "Pancreas",
         "prostate cancer": "Prostate", "sarcoma": "Sarcoma"}


def labels(coh):
    return {l.split("\t")[0]: l.split("\t")[2] for l in open(MAN).read().splitlines()[1:]
            if l.split("\t")[1] == coh}


def sids_of(p):
    return [str(s) for s in np.load(p, allow_pickle=True)["sids"]]


def rebuild_order(coh):
    """sid order of oof.npz rows (must match the saved y)."""
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


def load_cohort(coh):
    z = np.load(f"{OUT}/{coh}_oof.npz", allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    order, y_re, cls_re = rebuild_order(coh)
    if cls_re != classes or not np.array_equal(y_re, z["y"]):
        sys.exit(f"ABORT {coh}: rebuilt order/labels do not match saved oof (benchmark drift)")
    return z, order, classes


def find_cohort(sid):
    for coh in ("v1", "v2"):
        _, order, _ = load_cohort(coh)
        if sid in order:
            return coh
    return None


def report(sid, coh):
    z, order, classes = load_cohort(coh)
    if sid not in order:
        sys.exit(f"sid {sid} not in cohort {coh} (n={len(order)}). Try --list {coh}.")
    i = order.index(sid)
    true_lab = classes[int(z["y"][i])]
    prim = z["primary"][i]
    modkeys = [k for k in z.files if k.startswith("mod_")]

    print(f"\nSample: {sid}   cohort: {coh}")
    print(f"True label:      {SHORT.get(true_lab, true_lab)}")
    top = int(np.argmax(prim))
    mark = "correct" if top == int(z["y"][i]) else "WRONG"
    print(f"Ensemble top-1:  {SHORT.get(classes[top], classes[top])}  (p={prim[top]:.3f})  [{mark}]")
    print("\nEnsemble posterior probability per cancer type (out-of-fold, sums to 1):")
    print(f"  {'cancer type':<12s} {'prob':>7s}   {'':<20s}")
    for k in np.argsort(-prim):
        bar = "#" * int(round(prim[k] * 40))
        flag = "  <- TRUE" if k == int(z["y"][i]) else ""
        print(f"  {SHORT.get(classes[k], classes[k]):<12s} {prim[k]:>7.3f}   {bar}{flag}")

    print("\nPer-module posterior (each row sums to 1; ensemble = shrinkage-weighted blend):")
    hdr = "  " + " ".join(f"{SHORT.get(c, c)[:7]:>7s}" for c in classes)
    print(hdr)
    for mk in ["mod_E1_entropy", "mod_All_exon_depth", "mod_Genome_wide_CNA",
               "mod_Somatic_mutation_profile", "mod_SHAPE_nep300"]:
        if mk in z.files:
            row = z[mk][i]
            print(f"  {' '.join(f'{v:>7.3f}' for v in row)}   {MODLABEL[mk]}")
    print(f"  {' '.join(f'{v:>7.3f}' for v in prim)}   ENSEMBLE")
    print()


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); sys.exit(0)
    if a[0] == "--list":
        coh = a[1] if len(a) > 1 else "v1"
        _, order, classes = load_cohort(coh)
        lab = labels(coh)
        print(f"cohort {coh}: {len(order)} samples, {len(classes)} classes")
        for s in order:
            print(f"  {s}\t{SHORT.get(lab[s], lab[s])}")
        sys.exit(0)
    sid = a[0]
    coh = a[1] if len(a) > 1 else find_cohort(sid)
    if coh is None:
        sys.exit(f"sid {sid} not found in v1 or v2. Try --list.")
    report(sid, coh)


if __name__ == "__main__":
    main()
