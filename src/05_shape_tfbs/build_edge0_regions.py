#!/usr/bin/env python3
"""Attach the REAL UniBind motif boundary (start,end) to each edge0 unique center.
Motif width is constant per UniBind hit-name (= PWM length, liftover-invariant); read it from the hg38 source
BED and apply to the hg19 center (region = center +- width/2). For centers with multiple motifs, prefer a hit
whose TF matches the center's TF, else the modal width at that locus.
Out: centers_edge0_regions.tsv  chrom<TAB>start<TAB>end<TAB>strand<TAB>TF<TAB>edge   (hg19, 0-based, region span)
"""
import sys, gzip
from collections import defaultdict, Counter

SCR = "/tmp/claude-1002/-home-jrkim/5ebb3ac4-50d5-4c66-a4d7-ce4c72a5252e/scratchpad/gw_metagene"
UNIQ = f"{SCR}/centers_edge0_unique.tsv"                       # chrom pos strand TF edge (26845)
EDGE0 = "/home/jrkim/TSO_TFBS/project/intermediate/TFBS/TSO500_UniBind_TFBS.edge0.centers.bed"
HG38 = "/home/jrkim/TSO_TFBS/hg38_compressed_TFBSs.bed.gz"
OUT = f"{SCR}/centers_edge0_regions.tsv"


def tf_of(name):                                              # EXP..._<cell>_<TF>_MA####.# -> TF token
    i = name.rfind("_MA")
    if i < 0: return None
    return name[:i].rsplit("_", 1)[-1]


def main():
    # 1) names present in edge0 centers -> only need widths for these
    need = set()
    for ln in open(EDGE0):
        c = ln.rstrip("\n").split("\t")
        if len(c) > 3: need.add(c[3])
    print(f"edge0 distinct hit-names: {len(need)}", flush=True)

    # 2) width per needed name from hg38 source (constant per name)
    width = {}
    with gzip.open(HG38, "rt") as f:
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < 4: continue
            nm = c[3]
            if nm in need and nm not in width:
                width[nm] = int(c[2]) - int(c[1])
    print(f"widths resolved: {len(width)}/{len(need)}", flush=True)

    # 3) (chrom,pos) -> list of (TF, width) from edge0 centers
    loc = defaultdict(list)
    for ln in open(EDGE0):
        c = ln.rstrip("\n").split("\t")
        if len(c) < 4: continue
        nm = c[3]; w = width.get(nm)
        if w is None: continue
        loc[(c[0], int(c[1]))].append((tf_of(nm), w))

    # 4) emit region per unique center (prefer TF-matching width, else modal)
    n = 0; miss = 0; wsum = 0
    with open(UNIQ) as fi, open(OUT, "w") as fo:
        for ln in fi:
            x = ln.rstrip("\n").split("\t")
            if len(x) < 4: continue
            chrom, pos, strand, tf = x[0], int(x[1]), x[2], x[3]
            edge = x[4] if len(x) > 4 else "-1"
            cand = loc.get((chrom, pos), [])
            ws = [w for t, w in cand if t == tf] or [w for _, w in cand]
            if not ws:
                miss += 1; w = 12                              # fallback: median motif width
            else:
                w = Counter(ws).most_common(1)[0][0]
            half = w // 2; start = pos - half; end = pos + (w - half)
            fo.write(f"{chrom}\t{start}\t{end}\t{strand}\t{tf}\t{edge}\n")
            n += 1; wsum += w
    print(f"wrote {OUT}  n={n}  no-width-fallback={miss}  mean_width={wsum/max(n,1):.1f}", flush=True)


if __name__ == "__main__":
    main()
