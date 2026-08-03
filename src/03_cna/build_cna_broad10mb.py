#!/usr/bin/env python
"""Redefine the Genome_wide_CNA modality: a 'chromosomal CNA' is now a SEGMENT WIDER THAN 10 Mb (user definition),
not a chromosome-arm aggregate.

Source: ichorCNA off-probe segmentation (.seg.txt, column seg.median.logR), hg19, chr1-22 + chrX.
For each sample:
  * keep only segments with (end-start) > 10 Mb           <- 'chromosomal' / broad CNA
  * paint a genome-wide 1 Mb-bin log2 copy-ratio profile: bin value = seg.median.logR of the broad segment
    covering that bin's midpoint; bins inside focal (<=10 Mb) or neutral-gap segments stay 0.
  * append tumour_fraction + ploidy (.params.txt) for parity with the previous arm-level feature set.

seg<->sample mapping is rename-log aware (files are named either <numericSID>_<aliquot> or TSO_xxxxx_B_01).
Aligns to the EXACT feat sids of the existing X_Genome_wide_CNA.npz, backs up the arm-level matrix to
X_Genome_wide_CNA_armlevel.npz, and OVERWRITES X_Genome_wide_CNA.npz with the broad-10Mb matrix so the whole
pipeline (model-comparison benchmark, fusion, figures) picks up the new definition. Writes a QC table too."""
import os, sys, glob, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

PROJ = "/home/jrkim/TSO_TFBS/project"
FEAT = f"{PROJ}/results/auto_plan/feat"
ICHOR = {"v1": "/data/TSO500/ichorCNA_offProbe_result/v1/ichorCNA",
         "v2": "/data/TSO500/ichorCNA_offProbe_result/v2/ichorCNA"}
RENAME = "/data/TSO500/id_rename_success.log"
QC = f"{PROJ}/results/auto_plan/nc_readiness/tables"; os.makedirs(QC, exist_ok=True)
MINW = 10e6          # >10 Mb = chromosomal CNA
BIN = 1_000_000      # 1 Mb genome bins
SIZE = {"chr1":249250621,"chr2":243199373,"chr3":198022430,"chr4":191154276,"chr5":180915260,"chr6":171115067,
        "chr7":159138663,"chr8":146364022,"chr9":141213431,"chr10":135534747,"chr11":135006516,"chr12":133851895,
        "chr13":115169878,"chr14":107349540,"chr15":102531392,"chr16":90354753,"chr17":81195210,"chr18":78077248,
        "chr19":59128983,"chr20":63025520,"chr21":48129895,"chr22":51304566,"chrX":155270560}
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
SUF = ["_tumor_non_amplified_flank70", "_non_amplified_flank70", "_tumor_flank70", "_flank70"]


def log(m): print(m, flush=True)


# ---- genome 1 Mb bin grid ----
BINS = []                      # (chrom, bin_index, mid_bp)
BIN_SLICE = {}                 # chrom -> (start_col, n_bins)
for c in CHROMS:
    nb = int(np.ceil(SIZE[c] / BIN)); BIN_SLICE[c] = (len(BINS), nb)
    for j in range(nb): BINS.append((c, j, j * BIN + BIN / 2))
COLS_BIN = [f"{c}:{j}" for c, j, _ in BINS]
NBIN = len(BINS)


def rename_maps():
    tso2num = {}
    for ln in open(RENAME):
        if "->" in ln and not ln.startswith("#"):
            try:
                o, n = ln.rstrip().split("\t")[-1].split("->"); tso2num[n.strip()] = o.strip()
            except Exception: pass
    return tso2num


def core(bn):
    s = bn[:-len(".seg.txt")]
    for suf in SUF:
        if s.endswith(suf): return s[:-len(suf)]
    return s


def make_canon(tso2num):
    def canon(coreid):
        m = re.match(r"^(\d{10})", coreid)
        if m: return m.group(1)
        for v in (coreid, coreid.rsplit("_", 1)[0], coreid.rsplit("_", 2)[0]):
            if v in tso2num: return tso2num[v]
        return None
    return canon


def index_seg(tag, canon):
    """canonical numeric sid -> chosen seg.txt path (deterministic: most segment marks)."""
    cand = {}
    for p in glob.glob(f"{ICHOR[tag]}/**/*.seg.txt", recursive=True):
        cn = canon(core(os.path.basename(p)))
        if cn: cand.setdefault(cn, []).append(p)
    idx = {}
    for sid, paths in cand.items():
        idx[sid] = sorted(paths)[0] if len(paths) == 1 else _pick(paths)
    return idx


def _pick(paths):
    best, bn = None, -1
    for p in sorted(paths):
        try:
            n = len(pd.read_csv(p, sep="\t", usecols=["chrom"]))
        except Exception:
            n = 0
        if n > bn: bn, best = n, p
    return best


def broad_profile(segpath):
    """Return (bin_vector[NBIN], n_broad_segments, frac_genome_broad_altered)."""
    vec = np.zeros(NBIN, np.float32)
    try:
        df = pd.read_csv(segpath, sep="\t")
    except Exception:
        return None, 0, 0.0
    if "seg.median.logR" not in df.columns: return None, 0, 0.0
    df = df[df["chrom"].astype(str).isin(SIZE)].copy()
    df["lr"] = pd.to_numeric(df["seg.median.logR"], errors="coerce")
    df = df.dropna(subset=["lr"])
    nbroad = 0; altered_bins = 0
    for c, s, e, lr in zip(df["chrom"].astype(str), df["start"].astype(float), df["end"].astype(float), df["lr"]):
        if (e - s) <= MINW: continue                       # focal -> not a chromosomal CNA
        nbroad += 1
        c0, nb = BIN_SLICE[c]
        j0 = max(0, int(s // BIN)); j1 = min(nb - 1, int((e - 1) // BIN))
        for j in range(j0, j1 + 1):
            mid = j * BIN + BIN / 2
            if s <= mid < e:
                vec[c0 + j] = lr
                if abs(lr) >= 0.2: altered_bins += 1       # count only true gain/loss bins for QC
    return vec, nbroad, altered_bins / NBIN


def params_tf_ploidy(segpath):
    pp = segpath.replace(".seg.txt", ".params.txt"); tf = pl = np.nan
    if os.path.exists(pp):
        for ln in open(pp):
            if ln.startswith("Tumor Fraction:"): tf = float(ln.split("\t")[1])
            elif ln.startswith("Ploidy:"): pl = float(ln.split("\t")[1])
    return tf, pl


def build_tag(tag, canon):
    feat_npz = f"{FEAT}/{tag}/X_Genome_wide_CNA.npz"
    sids = [str(s) for s in np.load(feat_npz, allow_pickle=True)["sids"]]
    idx = index_seg(tag, canon)
    cols = COLS_BIN + ["tumor_fraction", "ploidy"]
    X = np.zeros((len(sids), len(cols)), np.float32)
    qc = []; n_have = 0; multi = 0
    for i, sid in enumerate(sids):
        if sid not in idx:
            qc.append(dict(sid=sid, has_seg=0, n_broad=0, frac_altered=0.0, tf=np.nan)); continue
        vec, nb, fa = broad_profile(idx[sid])
        if vec is None:
            qc.append(dict(sid=sid, has_seg=0, n_broad=0, frac_altered=0.0, tf=np.nan)); continue
        n_have += 1
        X[i, :NBIN] = vec
        tf, pl = params_tf_ploidy(idx[sid])
        X[i, NBIN] = tf if np.isfinite(tf) else 0.0; X[i, NBIN + 1] = pl if np.isfinite(pl) else 0.0
        qc.append(dict(sid=sid, has_seg=1, n_broad=nb, frac_altered=round(fa, 4), tf=round(tf, 4) if np.isfinite(tf) else np.nan))
    # backup arm-level then overwrite canonical feature
    bak = f"{FEAT}/{tag}/X_Genome_wide_CNA_armlevel.npz"
    if not os.path.exists(bak):
        z = np.load(feat_npz, allow_pickle=True); np.savez(bak, X=z["X"], sids=z["sids"], cols=z["cols"])
    np.savez(feat_npz, X=X, sids=np.array(sids), cols=np.array(cols))
    np.savez(f"{FEAT}/{tag}/X_Genome_wide_CNA_broad10mb.npz", X=X, sids=np.array(sids), cols=np.array(cols))
    qdf = pd.DataFrame(qc); qdf.to_csv(f"{QC}/Tab_cna_broad10mb_qc_{tag}.tsv", sep="\t", index=False)
    log(f"[{tag}] sids={len(sids)} with_seg={n_have} dims={len(cols)} (={NBIN} bins +TF+ploidy) | "
        f"mean broad>10Mb segs/sample={qdf[qdf.has_seg==1].n_broad.mean():.1f} | "
        f"mean genome frac altered={qdf[qdf.has_seg==1].frac_altered.mean():.3f}")
    return n_have


def main():
    tso2num = rename_maps(); canon = make_canon(tso2num)
    log(f"[setup] genome 1Mb bins={NBIN} over {len(CHROMS)} chroms; >10Mb broad-CNA definition; rename pairs={len(tso2num)}")
    for tag in ("v1", "v2"):
        build_tag(tag, canon)
    log("[cna-broad10mb] DONE -> X_Genome_wide_CNA.npz replaced (arm-level backed up to *_armlevel.npz)")


if __name__ == "__main__":
    main()
