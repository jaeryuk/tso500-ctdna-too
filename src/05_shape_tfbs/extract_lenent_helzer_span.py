#!/usr/bin/env python3
"""Helzer-rule TFBS length-entropy over the MOTIF-EDGE window [motif_start-40, motif_end+40].

Identical Helzer fragment rules as extract_lenent_helzer.py:
  * fragment counts if it OVERLAPS the window by >= 1 bp  (NOT midpoint)
  * insert size kept if 20 <= size <= 500
  * lenent = Shannon entropy (natural log) of EXACT fragment SIZE frequencies
The ONLY change vs extract_lenent_helzer.py: the window is anchored to the motif EDGES (region start/end
padded by 40) instead of the collapsed center +-40. Reads a regions TSV (motif spans), labels each site by
its motif CENTER (=(start+end)//2) so it aligns to the X_SHAPE gr:chrom_center columns downstream.

Computes the lenent over BOTH windows in one BAM pass (same fetched fragments), so a single run yields the
motif-edge AND the center-anchored Helzer lenent for the window-effect comparison:
    lenent      : [motif_start-40, motif_end+40]   (motif-edge)
    lenent_ctr  : [center-40, center+40], center=(start+end)//2   (center-anchored)
In : regions TSV  chrom<TAB>start<TAB>end<TAB>strand<TAB>TF<TAB>edge   (hg19, 0-based motif span)
Out: <outdir>/<sid>.lenhz.npz  lenent,lenent_ctr(float32), nfr,nfr_ctr(int32), tf_code, edge, tf_names, chrom, pos(center)
Usage: extract_lenent_helzer_span.py <sid> <bam> <regions_tsv> <outdir>"""
import os, sys
from collections import defaultdict
import numpy as np
PM = 40; SZLO, SZHI = 20, 500; MAPQ = 20; MARGIN = 600; GAP = 2 * MARGIN
def log(m): print(m, flush=True)


def shannon_nat(counts):
    s = counts.sum()
    if s <= 0: return np.nan
    p = counts[counts > 0] / s
    return float(-(p * np.log(p)).sum())


def load_regions(path):
    tfj = {}; names = []; by = defaultdict(list)
    for ln in open(path):
        x = ln.rstrip("\n").split("\t")
        if len(x) < 3: continue
        st, en = int(x[1]), int(x[2])
        tf = x[4] if len(x) > 4 else "NA"
        if tf not in tfj: tfj[tf] = len(names); names.append(tf)
        e = int(x[5]) if len(x) > 5 else -1
        by[x[0]].append((st, en, tfj[tf], e))
    groups = []
    for ch in sorted(by):
        cs = sorted(by[ch]); cur = [cs[0]]
        for cc in cs[1:]:
            if cc[0] - cur[-1][0] <= GAP: cur.append(cc)
            else: groups.append((ch, cur)); cur = [cc]
        groups.append((ch, cur))
    return groups, names


def main():
    sid, bampath, regions_tsv, outdir = sys.argv[1:5]
    import pysam
    os.makedirs(outdir, exist_ok=True); out = f"{outdir}/{sid}.lenhz.npz"
    if os.path.exists(out): log(f"[{sid}] cached"); return
    groups, tf_names = load_regions(regions_tsv); nsite = sum(len(c) for _, c in groups)
    lenent = np.full(nsite, np.nan, np.float32); nfr = np.zeros(nsite, np.int32)
    lenent_c = np.full(nsite, np.nan, np.float32); nfr_c = np.zeros(nsite, np.int32)
    tfc = np.zeros(nsite, np.uint16); edge = np.zeros(nsite, np.int32)
    chrom_a = np.empty(nsite, object); pos_a = np.zeros(nsite, np.int64)
    bam = pysam.AlignmentFile(bampath, "rb"); refset = set(bam.references)
    def res(c):
        if c in refset: return c
        if c.startswith("chr") and c[3:] in refset: return c[3:]
        return ("chr" + c) if ("chr" + c) in refset else None
    si = 0
    for chrom, regs in groups:
        fc = res(chrom); w0 = regs[0][0] - MARGIN; w1 = regs[-1][1] + MARGIN
        if fc is None:
            for st, en, t, e in regs:
                tfc[si] = t; edge[si] = e; chrom_a[si] = chrom; pos_a[si] = (st + en) // 2; si += 1
            continue
        fs_l = []; fe_l = []; L_l = []
        for r in bam.fetch(fc, max(0, w0), w1):
            if not r.is_read1 or not r.is_proper_pair: continue
            if r.is_secondary or r.is_supplementary or r.is_duplicate or r.is_qcfail: continue
            if r.mapping_quality < MAPQ: continue
            L = r.template_length
            if L < SZLO or L > SZHI: continue        # 20-500bp (Helzer 2025)
            fs = r.reference_start - w0
            fs_l.append(fs); fe_l.append(fs + L - 1); L_l.append(L)
        fs_a = np.asarray(fs_l); fe_a = np.asarray(fe_l); L_a = np.asarray(L_l, np.int64)
        if fs_a.size:
            o = np.argsort(fs_a); fs_s = fs_a[o]; fe_s = fe_a[o]; L_s = L_a[o]
        for st, en, t, e in regs:
            tfc[si] = t; edge[si] = e; chrom_a[si] = chrom; pos_a[si] = (st + en) // 2
            cc = (st + en) // 2 - w0                            # motif center (window-relative)
            for (a, b, dst_le, dst_nf) in ((st - w0 - PM, en - w0 + PM, lenent, nfr),   # motif-edge [start-40,end+40]
                                           (cc - PM, cc + PM, lenent_c, nfr_c)):        # center [c-40,c+40]
                if not fs_a.size: continue
                lo = np.searchsorted(fs_s, a - SZHI, "left"); hi = np.searchsorted(fs_s, b, "right")
                if hi <= lo: continue
                m = fe_s[lo:hi] >= a                           # >=1bp overlap: fs<=b (slice) AND fe>=a
                if not m.any(): continue
                sizes = L_s[lo:hi][m]
                h = np.bincount(sizes - SZLO, minlength=SZHI - SZLO + 1)
                dst_le[si] = shannon_nat(h.astype(float)); dst_nf[si] = int(sizes.size)
            si += 1
    bam.close()
    np.savez_compressed(out, lenent=lenent, nfr=nfr, lenent_ctr=lenent_c, nfr_ctr=nfr_c,
                        tf_code=tfc, edge=edge, tf_names=np.array(tf_names),
                        chrom=chrom_a.astype(str), pos=pos_a)
    cl = int(np.isfinite(lenent).sum())
    log(f"[{sid}] SAVED {out} nsite={nsite} span_cov={cl}({100*cl/nsite:.0f}%) "
        f"med_nfr_span={int(np.median(nfr))} med_nfr_ctr={int(np.median(nfr_c))}")


if __name__ == "__main__":
    main()
