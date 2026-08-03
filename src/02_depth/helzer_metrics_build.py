#!/usr/bin/env python
"""
Helzer et al. 2025 (Nat Commun 16:9122) fragmentomics metrics, reimplemented in Python over our
TSO500 b37 BAMs (the official R/snakemake repo expects hg38 BAMs + 8-col fragment BEDs; our BAMs are
b37, so we reproduce the documented metric definitions on the same target framework used by our
amplitude method -> identical CV / identical samples for a fair within-TSO500 benchmark).

Per sample, ONE BAM pass over the 9,232 TSO500 manifest targets (GENE_ExonN_RefSeq).  Per exon:
  depth      = (n_frag / exon_size) / (total_on_target_frag / 1e6)        [H01 / H02 gene / H03 E1]
  entropy    = Shannon entropy (bits) of the fragment-LENGTH distribution  [H04 / H05 E1]
  mds        = motif diversity score = norm. entropy of 4-mer fragment-END motifs [H06 / H07 E1]
  smallfrac  = frac fragments 1-150bp among 1-500bp                        [H08 / H09 E1]
  bins[6]    = proportion in 0-100,100-150,150-200,200-250,250-300,300-1000 [H10]
Plus, by overlapping fragments with our UniBind TFBS centres / TCGA-ATAC-open sites:
  tfbs_entropy[254]  = frag-length entropy per TF                          [H11]
  atac_entropy[23]   = frag-length entropy per TCGA cancer type           [H12]

Fragment filters (Helzer-style): read1 of a proper pair, primary, non-dup, non-supp/sec, MAPQ>=20,
template_length in [1,1000].  Fragment assigned to the exon whose [ts,te) contains its midpoint.

Modes: sample <sid> <bam> | list
"""
import os, sys, glob
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_gc as P

PROJ = "/home/jrkim/TSO_TFBS/project"
MANIFEST = "/home/jrkim/TSO_TFBS/TST500C_manifest.bed"
PRES = f"{PROJ}/results/cancer_classification/features/site_cancer_presence.parquet"
OUTD = os.environ.get("HELZER_METRICS_OUTD", f"{PROJ}/results/metaplots/_helzer_metrics")
os.makedirs(OUTD, exist_ok=True)
MAPQ = 20; LMAX = 1000; PADSEQ = 360
CANCERS = ['ACC','BLCA','BRCA','CESC','CHOL','COAD','ESCA','GBM','HNSC','KIRC','KIRP','LGG',
           'LIHC','LUAD','LUSC','MESO','PCPG','PRAD','SKCM','STAD','TGCT','THCA','UCEC']
BIN_EDGES = [0, 100, 150, 200, 250, 300, 1000]          # 6 bins
LBINS = np.arange(1, 502)                                # 1bp length bins 1..500 for entropy
_B2I = {b: i for i, b in enumerate("ACGT")}
_COMP = str.maketrans("ACGT", "TGCA")

def log(m): print(m, flush=True)
def tf_of(name):
    p = (name or "").split("_"); return p[2] if len(p) > 2 else None

def shannon(counts):
    c = np.asarray(counts, float); s = c.sum()
    if s <= 0: return np.nan
    p = c[c > 0] / s
    return float(-(p * np.log2(p)).sum())

_PLAN = None
def build_plan():
    """targets (b37) + per-target TFBS feature centres (center, tf_j, row_r) + presence rows."""
    global _PLAN
    if _PLAN is not None: return _PLAN
    import pandas as pd
    targets = []
    with open(MANIFEST) as f:
        for ln in f:
            x = ln.rstrip("\n").split("\t")
            chrom, ts, te, name = x[0], int(x[1]), int(x[2]), x[3]
            parts = name.split("_")
            gene = parts[0]; exonf = parts[1] if len(parts) > 1 else "NA"
            targets.append(dict(chrom=chrom, ts=ts, te=te, name=name, gene=gene,
                                is_exon=exonf.startswith("Exon"), is_E1=(exonf == "Exon1")))
    feats = P.load_centers(f"{P.CTRLD}/TFBS.feature.centers.tsv")
    pres = pd.read_parquet(PRES)[CANCERS].to_numpy(bool)
    tfs = sorted({tf_of(f.get("name")) for f in feats if tf_of(f.get("name"))})
    tfj = {t: j for j, t in enumerate(tfs)}
    tmp = defaultdict(list)
    for r, f in enumerate(feats):
        t = tf_of(f.get("name"))
        if t in tfj:
            tmp[(f["chrom"], f["ts"], f["te"])].append((f["center"], tfj[t], r))
    # pre-sort each target's centres into arrays (cpos, cj, cr) for searchsorted overlap
    feat_by_t = {}
    for tk, lst in tmp.items():
        lst.sort()
        feat_by_t[tk] = (np.array([c for c, _, _ in lst]),
                         np.array([j for _, j, _ in lst]),
                         np.array([r for _, _, r in lst]))
    _PLAN = dict(targets=targets, tfs=tfs, presence=pres, feat_by_t=feat_by_t)
    return _PLAN

def run_sample(sid, bampath):
    import pysam
    from pyfaidx import Fasta
    out = f"{OUTD}/{sid}.npz"
    if os.path.exists(out):
        log(f"[helz {sid}] cached, skip"); return
    pl = build_plan(); targets = pl["targets"]; tfs = pl["tfs"]; ntf = len(tfs)
    PRESM = pl["presence"]; feat_by_t = pl["feat_by_t"]
    fa = Fasta(P.FASTA, as_raw=True, sequence_always_upper=True)
    bam = pysam.AlignmentFile(bampath, "rb")
    nE = len(targets)
    depth = np.zeros(nE); entropy = np.zeros(nE); mds = np.zeros(nE)
    smallf = np.zeros(nE); bins = np.zeros((nE, 6)); ncount = np.zeros(nE, np.int64)
    esize = np.array([t["te"] - t["ts"] for t in targets], float)
    tf_lh = np.zeros((ntf, len(LBINS) + 1)); can_lh = np.zeros((23, len(LBINS) + 1))
    total = 0
    for ei, t in enumerate(targets):
        chrom, ts, te = t["chrom"], t["ts"], t["te"]
        contig = P.contig_of(chrom); clen = len(fa[contig])
        cs = max(0, ts - PADSEQ); ce = min(clen, te + PADSEQ)
        seq = fa[contig][cs:ce]
        Ls = []; motif = np.zeros(256, np.int64)
        tk = (chrom, ts, te); fc = feat_by_t.get(tk)
        cpos = fc[0] if fc is not None else None
        for read in bam.fetch(chrom, max(0, ts - 350), te + 1):
            if not read.is_read1 or not read.is_proper_pair: continue
            if read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail: continue
            if read.mapping_quality < MAPQ: continue
            L = read.template_length
            if L <= 0 or L > LMAX: continue
            fs = read.reference_start; fe = fs + L
            mid = (fs + fe) // 2
            if not (ts <= mid < te): continue            # assign to exon containing midpoint
            Ls.append(L)
            # fragment-end 4-mer motifs (both ends), MDS
            for s0, rc in ((fs, False), (fe - 4, True)):
                a = s0 - cs
                if a < 0 or a + 4 > len(seq): continue
                k = seq[a:a + 4]
                if rc: k = k.translate(_COMP)[::-1]
                idx = 0; ok = True
                for ch in k:
                    v = _B2I.get(ch, -1)
                    if v < 0: ok = False; break
                    idx = idx * 4 + v
                if ok: motif[idx] += 1
            # per-TF / per-cancer length histograms (centres spanned by this fragment)
            if cpos is not None:
                lo = np.searchsorted(cpos, fs, "left"); hi = np.searchsorted(cpos, fe, "left")
                if hi > lo:
                    lb = min(L, 500)
                    for idx in range(lo, hi):
                        tf_lh[fc[1][idx], lb] += 1
                        pm = PRESM[fc[2][idx]]
                        if pm.any(): can_lh[pm, lb] += 1
        n = len(Ls); ncount[ei] = n; total += n
        if n:
            La = np.array(Ls)
            h = np.bincount(np.clip(La, 1, 501), minlength=502)[1:]
            entropy[ei] = shannon(h)
            small = int((La <= 150).sum()); large = int(((La > 150) & (La <= 500)).sum())
            smallf[ei] = small / (small + large) if (small + large) else np.nan
            bi = np.clip(np.digitize(La, BIN_EDGES) - 1, 0, 5)
            bc = np.bincount(bi, minlength=6).astype(float)
            bins[ei] = bc / bc.sum()
            mds[ei] = shannon(motif) / 8.0 if motif.sum() else np.nan      # /log2(256)=8
    bam.close()
    permil = max(total, 1) / 1e6
    depth = (ncount / np.maximum(esize, 1)) / permil
    tfbs_entropy = np.array([shannon(tf_lh[j]) for j in range(ntf)])
    atac_entropy = np.array([shannon(can_lh[c]) for c in range(23)])
    np.savez(out, qc_n=np.array([total]),
             exon_name=np.array([t["name"] for t in targets]),
             gene=np.array([t["gene"] for t in targets]),
             is_exon=np.array([t["is_exon"] for t in targets]),
             is_E1=np.array([t["is_E1"] for t in targets]),
             exon_size=esize, ncount=ncount,
             depth=depth, entropy=entropy, mds=mds, smallfrac=smallf, bins=bins,
             tfs=np.array(tfs), tfbs_entropy=tfbs_entropy,
             cancers=np.array(CANCERS), atac_entropy=atac_entropy)
    log(f"[helz {sid}] SAVED {out} (qc_n={total}, exons={nE})")

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "sample":
        run_sample(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == "list":
        pl = build_plan()
        log(f"plan: {len(pl['targets'])} targets, {sum(t['is_exon'] for t in pl['targets'])} exons, "
            f"{sum(t['is_E1'] for t in pl['targets'])} E1, {len(pl['tfs'])} TFs")
    else:
        sys.exit("usage: sample <sid> <bam> | list")
