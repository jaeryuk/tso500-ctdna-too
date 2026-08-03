#!/usr/bin/env python
"""
GDD-style mutational-signature features from TSO500 tumor-only calls (Penson 2020 / bergerm1 method).

Per sample, from <sid>.tmb.trace.tsv:
  somatic SNVs   = Status starts with "Somatic" (Somatic / Somatic_Putative_CH), single-base ref/alt
  96-channel trinucleotide spectrum (pyrimidine-centric, b37 FASTA context)
  NNLS fit to COSMIC v2 (96 x 30) -> non-negative exposures -> normalised to fractions
  collapse to GDD mechanism names: Age, APOBEC, BRCA, Smoking, MMR, UV, POLE, TMZ (+ 'other')
  GDD binary flag per mechanism: n_snv>=MIN_SIG_MUT (20) AND mechanism_fraction>SIG_DOM (0.40)

CLI:  python gdd_signatures.py one <sid> <tmb_trace_tsv>      # validate/print one sample
      (importable: spectrum_from_trace(), fit_exposures(), collapse_mechanisms())
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import nnls

PROJ = "/home/jrkim/TSO_TFBS/project"
FASTA = "/home/jrkim/cfdna_se_project/ref/genome/human_g1k_v37.fasta"   # b37 (no 'chr'); hg19 coords == b37 for SNVs
COSMIC = f"{PROJ}/data/cosmic/cosmic_v2_SBS.txt"
MIN_SIG_MUT = 20          # GDD: signature flag requires >=20 mutations
SIG_DOM = 0.40            # GDD: and mechanism fraction > 0.40
# COSMIC v2 -> GDD mechanism (unnamed signatures fold into 'other'); matches generate_feature_table.R
SIG2MECH = {1:"Age", 2:"APOBEC", 3:"BRCA", 4:"Smoking", 6:"MMR", 7:"UV", 10:"POLE", 11:"TMZ",
            13:"APOBEC", 15:"MMR", 20:"MMR", 24:"Smoking", 26:"MMR"}
MECHS = ["Age", "APOBEC", "BRCA", "Smoking", "MMR", "UV", "POLE", "TMZ", "other"]
_COMP = str.maketrans("ACGT", "TGCA")
_PUR = {"A", "G"}

_fa = None
_W = None; _CH = None; _SIGCOLS = None


def _fasta():
    global _fa
    if _fa is None:
        from pyfaidx import Fasta
        _fa = Fasta(FASTA, as_raw=True, sequence_always_upper=True)
    return _fa


def cosmic():
    """Return (W 96x30 ndarray, channel list of 96 'A[C>A]A', signature col names)."""
    global _W, _CH, _SIGCOLS
    if _W is None:
        df = pd.read_csv(COSMIC, sep="\t")
        _SIGCOLS = [c for c in df.columns if c != "Type"]
        _CH = df["Type"].tolist()
        _W = df[_SIGCOLS].to_numpy(float)
    return _W, _CH, _SIGCOLS


def _chan_index():
    _, ch, _ = cosmic()
    return {c: i for i, c in enumerate(ch)}


def spectrum_from_trace(trace_path):
    """96-channel SNV spectrum (counts) of somatic SNVs in one tmb.trace.tsv. Returns (vec96, n_snv)."""
    ci = _chan_index(); fa = _fasta()
    df = pd.read_csv(trace_path, sep="\t", dtype={"Chromosome": str})
    s = df[df["Status"].astype(str).str.startswith("Somatic")]
    s = s[(s["RefCall"].astype(str).str.len() == 1) & (s["AltCall"].astype(str).str.len() == 1)]
    vec = np.zeros(96, float); n = 0
    for chrom, pos, ref, alt in zip(s["Chromosome"], s["Position"], s["RefCall"], s["AltCall"]):
        ref = str(ref).upper(); alt = str(alt).upper()
        if ref not in "ACGT" or alt not in "ACGT" or ref == alt: continue
        contig = chrom[3:] if str(chrom).startswith("chr") else str(chrom)
        if contig in ("M", "MT", "chrM"): continue
        try:
            tri = fa[contig][int(pos) - 2:int(pos) + 1]            # 1-based pos -> [pos-1,pos,pos+1]
        except Exception:
            continue
        if not tri or len(tri) != 3 or tri[1] != ref: continue     # ref must match reference base
        up, _, dn = tri[0], tri[1], tri[2]
        if ref in _PUR:                                            # fold to pyrimidine-centric
            up, dn = dn.translate(_COMP), up.translate(_COMP)
            ref = ref.translate(_COMP); alt = alt.translate(_COMP)
        chan = f"{up}[{ref}>{alt}]{dn}"
        if chan in ci: vec[ci[chan]] += 1; n += 1
    return vec, n


def fit_exposures(vec96):
    """NNLS fit of a 96-count vector to COSMIC v2; return per-signature FRACTIONS (len 30, sum<=1)."""
    W, _, _ = cosmic()
    if vec96.sum() <= 0: return np.zeros(W.shape[1])
    expo, _ = nnls(W, vec96.astype(float))
    tot = expo.sum()
    return expo / tot if tot > 0 else expo


def collapse_mechanisms(expo_frac):
    """Collapse 30 COSMIC fractions to GDD mechanisms; return dict mech->fraction (incl 'other')."""
    _, _, sigcols = cosmic()
    out = {m: 0.0 for m in MECHS}
    for j, col in enumerate(sigcols):
        num = int(col.split("_")[1]); mech = SIG2MECH.get(num, "other")
        out[mech] += float(expo_frac[j])
    return out


def sample_features(trace_path):
    """Full per-sample signature feature dict:
       sigexp_S<k>  : full 30 continuous COSMIC v2 exposure fractions (TOO-informative; user's choice)
       sigfrac_<m>  : 9 GDD mechanism fractions (collapsed)
       sigflag_<m>  : 8 GDD binary dominant-mechanism flags (>=20 muts & frac>0.40; method-faithful)
       n_somatic_snv, sig_reconstruction_cos
    """
    _, _, sigcols = cosmic()
    vec, n = spectrum_from_trace(trace_path)
    frac = fit_exposures(vec)
    mech = collapse_mechanisms(frac)
    feat = {f"sigexp_S{col.split('_')[1]}": float(frac[j]) for j, col in enumerate(sigcols)}
    feat.update({f"sigfrac_{m}": mech[m] for m in MECHS})
    for m in MECHS:
        if m == "other": continue
        feat[f"sigflag_{m}"] = int(n >= MIN_SIG_MUT and mech[m] > SIG_DOM)
    feat["n_somatic_snv"] = n
    feat["sig_reconstruction_cos"] = _cosine(vec, frac)
    return feat


def _cosine(vec, frac):
    W, _, _ = cosmic()
    if vec.sum() <= 0: return np.nan
    recon = W @ (frac * vec.sum())
    a, b = vec, recon
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else np.nan


if __name__ == "__main__":
    if sys.argv[1] == "one":
        sid, trace = sys.argv[2], sys.argv[3]
        vec, n = spectrum_from_trace(trace)
        frac = fit_exposures(vec); mech = collapse_mechanisms(frac)
        print(f"[{sid}] somatic SNVs used: {n}")
        print(f"  spectrum total={int(vec.sum())}  top channels:")
        _, ch, _ = cosmic()
        order = np.argsort(vec)[::-1][:6]
        for i in order: print(f"    {ch[i]}: {int(vec[i])}")
        print(f"  reconstruction cosine: {_cosine(vec,frac):.3f}")
        print("  GDD mechanism fractions:")
        for m in MECHS: print(f"    {m:8s} {mech[m]:.3f}" + ("  *FLAG*" if (m!='other' and n>=MIN_SIG_MUT and mech[m]>SIG_DOM) else ""))
        _, _, sigcols = cosmic()
        top = np.argsort(frac)[::-1][:5]
        print("  top COSMIC signatures:", ", ".join(f"{sigcols[i]}={frac[i]:.2f}" for i in top if frac[i] > 0.01))
