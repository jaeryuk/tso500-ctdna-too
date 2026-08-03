#!/usr/bin/env python3
"""Per-site endpoint-Gini over the REAL UniBind motif region padded by P on BOTH sides, ONE BAM pass.
Window for padding P = [region_start - P, region_end + P]; 5bp bins from the window's left edge; endpoint-Gini
of the binned endpoint histogram (all 1-800bp frags). P swept over PADS. Gini is order-invariant (strand-free).
In : regions TSV  chrom<TAB>start<TAB>end<TAB>strand<TAB>TF<TAB>edge   (hg19, 0-based motif span)
Out: <outdir>/<sid>.padgini.npz  ginipad (float32 nsite x len(PADS)), nep (int32 nsite x len(PADS)),
                                  pads (int32), tf_code (uint16), edge (int32), tf_names
Usage: extract_persite_padgini_region.py <sid> <bam> <regions_tsv> <outdir>
"""
import os, sys
from collections import defaultdict
import numpy as np

PADS = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50], np.int32); NP = len(PADS); PMAX = int(PADS.max())
LBIN = 5; MAPQ = 20; MARGIN = 260; GAP = 2 * MARGIN
def log(m): print(m, flush=True)


def gini_counts(c):
    s = c.sum()
    if s <= 0: return np.nan
    c = np.sort(c.astype(np.float64)); n = c.size; idx = np.arange(1, n + 1)
    return (2.0 * np.sum(idx * c) / (n * s)) - (n + 1.0) / n


def load_regions(path):
    tf_names = []; tfj = {}; by = defaultdict(list)
    for ln in open(path):
        x = ln.rstrip("\n").split("\t")
        if len(x) < 3: continue
        st, en = int(x[1]), int(x[2])
        sgn = -1 if (len(x) > 3 and x[3] == "-") else 1
        tf = x[4] if len(x) > 4 else "NA"
        if tf not in tfj: tfj[tf] = len(tf_names); tf_names.append(tf)
        e = int(x[5]) if len(x) > 5 else -1
        by[x[0]].append((st, en, sgn, tfj[tf], e))
    groups = []
    for ch in sorted(by):
        cs = sorted(by[ch]); cur = [cs[0]]
        for cc in cs[1:]:
            if cc[0] - cur[-1][0] <= GAP: cur.append(cc)
            else: groups.append((ch, cur)); cur = [cc]
        groups.append((ch, cur))
    return groups, tf_names


def main():
    sid, bampath, regions_tsv, outdir = sys.argv[1:5]
    import pysam
    os.makedirs(outdir, exist_ok=True); out = f"{outdir}/{sid}.padgini.npz"
    groups, tf_names = load_regions(regions_tsv); nsite = sum(len(c) for _, c in groups)
    log(f"[{sid}] {nsite} regions / {len(groups)} windows")
    ginip = np.full((nsite, NP), np.nan, np.float32); nepp = np.zeros((nsite, NP), np.int32)
    gc10 = np.full(nsite, np.nan, np.float32)                # center +-10bp, 1bp bins (20 bins)
    tfc = np.zeros(nsite, np.uint16); edge = np.zeros(nsite, np.int32)
    bam = pysam.AlignmentFile(bampath, "rb"); refset = set(bam.references)
    def res(c):
        if c in refset: return c
        if c.startswith("chr") and c[3:] in refset: return c[3:]
        return ("chr" + c) if ("chr" + c) in refset else None
    si = 0
    for chrom, regs in groups:
        fc = res(chrom); w0 = regs[0][0] - MARGIN; w1 = regs[-1][1] + MARGIN; span = w1 - w0
        if fc is None:
            for st, en, sgn, t, e in regs: tfc[si] = t; edge[si] = e; si += 1
            continue
        ep_l = []
        for r in bam.fetch(fc, max(0, w0), w1):
            if not r.is_read1 or not r.is_proper_pair: continue
            if r.is_secondary or r.is_supplementary or r.is_duplicate or r.is_qcfail: continue
            if r.mapping_quality < MAPQ: continue
            L = r.template_length
            if L < 1 or L > 800: continue
            fs = r.reference_start - w0; fe = fs + L - 1
            ep_l.append(fs); ep_l.append(fe)
        if ep_l:
            cc = np.asarray(ep_l, np.int64); ok = (cc >= 0) & (cc < span)
            ep_cov = np.bincount(cc[ok], minlength=span)
        else:
            ep_cov = np.zeros(span, np.int64)
        for st, en, sgn, t, e in regs:
            tfc[si] = t; edge[si] = e; rs = st - w0; re_ = en - w0
            rc = (st + en) // 2 - w0                          # TFBS center (motif midpoint)
            if rc - 10 >= 0 and rc + 10 <= span:
                gc10[si] = gini_counts(ep_cov[rc - 10:rc + 10])   # center +-10bp, 1bp bins (20 bins)
            for k in range(NP):
                a = rs - int(PADS[k]); b = re_ + int(PADS[k])
                if a < 0 or b > span: continue
                W = b - a; nb = W // LBIN
                if nb < 2: continue
                win = ep_cov[a:a + nb * LBIN]
                hb = win.reshape(nb, LBIN).sum(1)
                nepp[si, k] = int(hb.sum()); ginip[si, k] = gini_counts(hb)
            si += 1
    bam.close()
    np.savez_compressed(out, ginipad=ginip, gini_c10b1=gc10, nep=nepp, pads=PADS, tf_code=tfc, edge=edge,
                        tf_names=np.array(tf_names))
    cov = int(np.isfinite(ginip[:, -1]).sum())
    log(f"[{sid}] SAVED {out} nsite={nsite} covered@P50={cov}({100*cov/max(nsite,1):.0f}%) "
        f"med_gini@P40={np.nanmedian(ginip[:, 6]):.3f}")


if __name__ == "__main__":
    main()
