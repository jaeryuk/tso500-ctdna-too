#!/usr/bin/env python
"""
On-target, PoN-normalized chromosome-arm CNA features for GRAIL cfDNA.

WHY on-target (not off-probe): the GRAIL analysis2 (UMI/duplex-collapsed) BAMs are capture-only
(~0.08% of reads off-target; genome-wide 1Mb bins outside the panel hold a median of 3-4 reads), so true
off-probe ichorCNA is infeasible. We therefore use the panel-covered 1Mb bins: per bin, log2 of the sample
read count over the non-cancer-cfDNA panel-of-normals median, GC-detrended, aggregated to chromosome arms.
This is the standard capture-panel CNA readout (CNVkit/ichorCNA-on-target style) and yields the same arm-level
feature shape as the TSO500 v2_cna_features.py output, so it drops into the GRAIL classifier.

In : results/auto_plan/grail/wig/<sid>.wig   (readCounter 1Mb, from grail_readcounter.sh)
     ichorCNA gc_hg19_1000kb.wig             (GC per bin)
     manifest.tsv                            (labels; non-cancer rows define the PoN)
Out: results/auto_plan/grail/feat/X_Genome_wide_CNA.npz  (X, sids, cols=[arms... , cna_burden])
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

PROJ = "/home/jrkim/TSO_TFBS/project"
WIG = f"{PROJ}/results/auto_plan/grail/wig"
MAN = f"{PROJ}/results/auto_plan/grail/manifest.tsv"
OUT = f"{PROJ}/results/auto_plan/grail/feat"; os.makedirs(OUT, exist_ok=True)
GCWIG = "/home/jrkim/.conda/envs/cfse_cna/lib/R/library/ichorCNA/extdata/gc_hg19_1000kb.wig"
MINCOV = 50          # PoN-median read count to call a 1Mb bin "on-target / informative"
CEN = {"chr1":1.25e8,"chr2":9.33e7,"chr3":9.1e7,"chr4":5.04e7,"chr5":4.84e7,"chr6":6.1e7,"chr7":5.99e7,
       "chr8":4.56e7,"chr9":4.9e7,"chr10":4.02e7,"chr11":5.37e7,"chr12":3.58e7,"chr13":1.79e7,"chr14":1.76e7,
       "chr15":1.9e7,"chr16":3.66e7,"chr17":2.4e7,"chr18":1.72e7,"chr19":2.65e7,"chr20":2.75e7,"chr21":1.32e7,
       "chr22":1.47e7,"chrX":6.06e7}
ACRO_P = {"chr13p","chr14p","chr15p","chr21p","chr22p"}
ARMS = [f"{c}{a}" for c in CEN for a in ("p", "q") if f"{c}{a}" not in ACRO_P]
def log(m): print(m, flush=True)


def parse_wig(path):
    """fixedStep wig -> ordered list of (chrom, start0) and float values."""
    chrom = None; start = step = None; keys = []; vals = []
    for ln in open(path):
        if ln.startswith("fixedStep"):
            d = dict(kv.split("=") for kv in ln.split()[1:])
            chrom = d["chrom"]; start = int(d.get("start", 1)); step = int(d.get("step", 1000000))
            cur = start - 1
        elif ln.startswith("track") or not ln.strip():
            continue
        else:
            keys.append((chrom, cur)); vals.append(float(ln)); cur += step
    return keys, np.array(vals, np.float64)


def arm_of(chrom, start0):
    if chrom not in CEN: return None
    a = "p" if start0 < CEN[chrom] else "q"
    arm = f"{chrom}{a}"
    return None if arm in ACRO_P else arm


def main():
    man = pd.read_csv(MAN, sep="\t", dtype=str)
    wigs = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{WIG}/*.wig")}
    sids = [s for s in man.sid if s in wigs]
    log(f"[grail-cna] {len(sids)}/{len(man)} samples have wigs")
    # canonical bin layout from first wig; GC vector aligned to it
    keys0, _ = parse_wig(wigs[sids[0]])
    gck, gcv = parse_wig(GCWIG)
    # GC wig uses bare chrom names ('1') while readCounter uses 'chr1' -> match on (stripped chrom, start)
    def ck(c, s): return (c.replace("chr", ""), s)
    gcmap = {ck(c, s): v for (c, s), v in zip(gck, gcv)}
    gc = np.array([gcmap.get(ck(c, s), -1.0) for (c, s) in keys0])
    arms = np.array([arm_of(c, s) for (c, s) in keys0], dtype=object)
    # count matrix samples x bins
    C = np.full((len(sids), len(keys0)), np.nan)
    for i, s in enumerate(sids):
        k, v = parse_wig(wigs[s])
        if k == keys0: C[i] = v
        else:                                            # align by key (robust to chrom-order diffs)
            m = {kk: vv for kk, vv in zip(k, v)}; C[i] = [m.get(kk, np.nan) for kk in keys0]
    # PoN = non-cancer; informative bins = PoN median coverage >= MINCOV and valid GC
    lab = man.set_index("sid").loc[sids, "cancer_type"].values
    pon = C[lab == "non-cancer"]
    pon_med = np.nanmedian(pon, 0)
    info = (pon_med >= MINCOV) & (gc >= 0) & np.array([a is not None for a in arms])
    log(f"[grail-cna] informative on-target bins: {int(info.sum())}/{len(keys0)} "
        f"(PoN n={int((lab=='non-cancer').sum())})")
    # per-sample log2 ratio vs PoN median on informative bins, then GC detrend (deg-2 polyfit)
    R = np.full((len(sids), len(keys0)), np.nan)
    gi = gc[info]
    for i in range(len(sids)):
        r = np.log2((C[i, info] + 0.5) / (pon_med[info] + 0.5))
        ok = np.isfinite(r)
        if ok.sum() > 10:
            coef = np.polyfit(gi[ok], r[ok], 2); r = r - np.polyval(coef, gi)
            r = r - np.median(r[ok])             # center
        R[i, info] = r
    # aggregate informative bins -> arm median log2ratio
    arm_idx = {a: np.where(info & (arms == a))[0] for a in ARMS}
    X = np.full((len(sids), len(ARMS) + 1), np.nan, np.float32)
    for j, a in enumerate(ARMS):
        ix = arm_idx[a]
        if len(ix):
            X[:, j] = np.nanmedian(R[:, ix], 1)
    # CNA burden proxy (aneuploidy score): mean |arm log2ratio| across covered arms
    X[:, -1] = np.nanmean(np.abs(X[:, :len(ARMS)]), 1)
    # median-impute arms with no informative bins (per column)
    med = np.nanmedian(X, 0); nan = np.where(~np.isfinite(X)); X[nan] = np.take(np.nan_to_num(med), nan[1])
    cols = ARMS + ["cna_burden"]
    np.savez(f"{OUT}/X_Genome_wide_CNA.npz", X=X, sids=np.array(sids), cols=np.array(cols))
    pd.DataFrame(X, index=sids, columns=cols).to_csv(f"{OUT}/grail_cna_table.tsv", sep="\t")
    covered = [a for a in ARMS if len(arm_idx[a])]
    log(f"[grail-cna] X {X.shape}; arms with on-target coverage: {len(covered)}/{len(ARMS)}")
    log(f"[grail-cna] -> {OUT}/X_Genome_wide_CNA.npz")


if __name__ == "__main__":
    main()
