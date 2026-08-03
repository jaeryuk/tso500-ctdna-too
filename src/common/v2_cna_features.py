#!/usr/bin/env python
"""
Genome-wide copy-number TOO features from the ichorCNA off-probe results (ultima's run).

Per covered v2 sample:
  * chromosome-ARM mean of the per-1Mb log2 tumor/normal ratio  (.correctedDepth.txt) -> ~39 arm features
    (arm-level CNA is the classic tissue-of-origin readout: e.g. CRC +20q/+13q/-18q, lung +1q, ...)
  * tumor fraction + ploidy (.params.txt) as covariates
Median-impute arm features that are all-NA for a sample (low off-target coverage).

Writes -> results/v2_cna/X_CNA.npz  (X, sids, cols) aligned to whatever samples ichorCNA covers.
"""
import os, glob, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

PROJ = "/home/jrkim/TSO_TFBS/project"
OUT = os.environ.get("CNA_OUT", f"{PROJ}/results/v2_cna"); os.makedirs(OUT, exist_ok=True)
RES = os.environ.get("CNA_ICHOR", "/data/TSO500/ichorCNA_offProbe_result/v2/ichorCNA")
MAN = os.environ.get("CNA_MAN", f"{PROJ}/results/v2_ponbench/manifest.tsv")
# hg19 centromere midpoints (bp); acrocentric p-arms (13,14,15,21,22) dropped (uninformative)
CEN = {"chr1":1.25e8,"chr2":9.33e7,"chr3":9.1e7,"chr4":5.04e7,"chr5":4.84e7,"chr6":6.1e7,"chr7":5.99e7,
       "chr8":4.56e7,"chr9":4.9e7,"chr10":4.02e7,"chr11":5.37e7,"chr12":3.58e7,"chr13":1.79e7,"chr14":1.76e7,
       "chr15":1.9e7,"chr16":3.66e7,"chr17":2.4e7,"chr18":1.72e7,"chr19":2.65e7,"chr20":2.75e7,"chr21":1.32e7,
       "chr22":1.47e7,"chrX":6.06e7}
ACRO_P = {"chr13p","chr14p","chr15p","chr21p","chr22p"}
ARMS = [f"{c}{a}" for c in CEN for a in ("p", "q") if f"{c}{a}" not in ACRO_P]
def log(m): print(m, flush=True)


def load_manifest():
    rows = []
    with open(MAN) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for ln in f: rows.append(dict(zip(hdr, ln.rstrip("\n").split("\t"))))
    return rows


def index_results():
    """sample-name -> correctedDepth.txt path. Files are named <SAMPLE>_non_amplified_flank70.correctedDepth.txt
    where <SAMPLE> is the TSO id (e.g. TSO_00048_B_01) or a legacy numeric id. Key by that full sample name.
    (The old split('_')[0] collapsed every re-run TSO file to the key 'TSO' -> 0 matches.)"""
    d = {}
    for p in glob.glob(f"{RES}/**/*.correctedDepth.txt", recursive=True):
        k = os.path.basename(p).replace(".correctedDepth.txt", "")
        k = re.sub(r'_non_amplified_flank70$', '', k)
        d.setdefault(k, p)
    return d


def sample_arm_logr(cdpath):
    df = pd.read_csv(cdpath, sep="\t", names=["chr", "start", "end", "lr"], header=0)
    df["lr"] = pd.to_numeric(df["lr"], errors="coerce")
    df = df[df["chr"].isin(CEN)].dropna(subset=["lr"])
    if df.empty: return {a: np.nan for a in ARMS}
    df["arm"] = [f"{c}{'p' if s < CEN[c] else 'q'}" for c, s in zip(df["chr"], df["start"])]
    m = df.groupby("arm")["lr"].mean().to_dict()
    return {a: m.get(a, np.nan) for a in ARMS}


def parse_params(cdpath):
    pp = cdpath.replace(".correctedDepth.txt", ".params.txt")
    tf = ploidy = np.nan
    if os.path.exists(pp):
        for ln in open(pp):
            if ln.startswith("Tumor Fraction:"): tf = float(ln.split("\t")[1])
            elif ln.startswith("Ploidy:"): ploidy = float(ln.split("\t")[1])
    return tf, ploidy


def tso_of(r):
    """manifest bam path -> TSO id, e.g. /data/.../BAM/TSO_00048_B_01.bam -> TSO_00048_B_01"""
    return os.path.splitext(os.path.basename(r["bam"]))[0]


def canon(t):
    """strip a re-seq suffix that follows the _B_NN token: TSO_..._B_01_1 -> TSO_..._B_01 (safe: leaves _B_01 alone)."""
    m = re.match(r'^(.*_B_\d+)_\d+$', t)
    return m.group(1) if m else t


def main():
    man = load_manifest(); idx = index_results()
    cohort = []; paths = []; n_kept = 0
    for r in man:
        if not (r["keep_t1"] == "1" or r["keep_t2"] == "1"): continue
        n_kept += 1
        t = tso_of(r)
        p = idx.get(t) or idx.get(canon(t)) or idx.get(r["sid"])   # TSO id, re-seq-canon, or legacy numeric
        if p: cohort.append(r["sid"]); paths.append(p)   # OUTPUT keyed by numeric sid (aligns w/ other modules)
    log(f"[cna] cohort with ichorCNA: {len(cohort)} / {n_kept} kept")
    cols = ARMS + ["tumor_fraction", "ploidy"]
    X = np.full((len(cohort), len(cols)), np.nan, np.float32)
    for i, (sid, cd) in enumerate(zip(cohort, paths)):
        arm = sample_arm_logr(cd); tf, pl = parse_params(cd)
        X[i, :len(ARMS)] = [arm[a] for a in ARMS]; X[i, -2] = tf; X[i, -1] = pl
        if (i + 1) % 100 == 0: log(f"  {i+1}/{len(cohort)}")
    # median-impute remaining NA (per column)
    med = np.nanmedian(X, 0); nan = np.where(~np.isfinite(X)); X[nan] = np.take(np.nan_to_num(med), nan[1])
    np.savez(f"{OUT}/X_CNA.npz", X=X, sids=np.array(cohort), cols=np.array(cols))
    pd.DataFrame(X, index=cohort, columns=cols).to_csv(f"{OUT}/cna_feature_table.tsv", sep="\t")
    log(f"[cna] X_CNA {X.shape} ({len(ARMS)} arms + TF + ploidy) -> {OUT}/X_CNA.npz")


if __name__ == "__main__":
    main()
