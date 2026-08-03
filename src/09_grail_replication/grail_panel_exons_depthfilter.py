#!/usr/bin/env python
"""
"exons INTERSECT empirical coverage" done at EXON resolution from direct BAM read depth.

The existing grail_panel_exons.covered.bed clips the gene-anchored exons to the coarse 1kb-bin,
single-sample coverage footprint (panel.bed) -- blocky and noisy (it false-negatived HLA-A).
This rebuilds the intersection from actual read depth: for each gene-anchored exon interval, compute
reads/kb/BAM pooled across N BAMs and KEEP intervals at/above a presence threshold. Same metric that
gave the trustworthy 491/508 gene-level call, now per exon -> trims genuinely-uncaptured exons
(uncaptured genes + untargeted UTR/alt exons within captured genes) at exon precision, multi-sample.

In : results/auto_plan/grail/panel/grail_panel_exons.labeled.bed
     results/auto_plan/grail/manifest.tsv
Out: grail_panel_exons.depth_covered.labeled.bed  (kept exon intervals, chr,start,end,gene)
     grail_panel_exons.depth_covered.merged.bed    (gene-agnostic merged; the empirical on-target target)
     grail_panel_exons.depth_per_interval.tsv       (every interval: depth + kept flag)
     grail_panel_depthfilter_report.json
"""
import os, json, csv
from collections import defaultdict
import pysam

PROJ = "/home/jrkim/TSO_TFBS/project"
PANEL = f"{PROJ}/results/auto_plan/grail/panel"
MAN = f"{PROJ}/results/auto_plan/grail/manifest.tsv"
NBAM = int(os.environ.get("NBAM", "4"))
THRESH = float(os.environ.get("KEEP_RDKB", "20"))     # reads/kb/BAM presence threshold (captured>>20, uncaptured<2)
def log(m): print(m, flush=True)


def merge_iv(iv):
    iv = sorted(iv); out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            if e > out[-1][1]: out[-1][1] = e
        else: out.append([s, e])
    return [(s, e) for s, e in out]


def chrom_order(c):
    cc = c.replace("chr", "")
    return {**{str(i): i for i in range(1, 23)}, "X": 23, "Y": 24}.get(cc, 99)


def main():
    intervals = []                       # (chrom, start, end, gene)
    for ln in open(f"{PANEL}/grail_panel_exons.labeled.bed"):
        c, s, e, g = ln.rstrip("\n").split("\t"); intervals.append((c, int(s), int(e), g))
    rows = open(MAN).read().splitlines()[1:]
    bams = [r.split("\t")[1] for r in rows]
    step = max(1, len(bams) // NBAM); sample = bams[::step][:NBAM]
    log(f"[depthfilter] {len(intervals)} exon intervals x {len(sample)} BAMs ; keep >= {THRESH} rd/kb/BAM")

    reads = [0] * len(intervals)
    for bp in sample:
        bam = pysam.AlignmentFile(bp, "rb")
        for i, (c, s, e, g) in enumerate(intervals):
            try: reads[i] += bam.count(c, s, e)
            except Exception: pass
        bam.close()

    kept = []; per_iv = []; bp_in = 0; bp_keep = 0
    by_gene_keep = defaultdict(int)
    for i, (c, s, e, g) in enumerate(intervals):
        L = e - s; bp_in += L
        rdkb = reads[i] / (L / 1000.0) / len(sample) if L > 0 else 0.0
        keep = rdkb >= THRESH
        per_iv.append(dict(chrom=c, start=s, end=e, gene=g, len=L,
                           reads=reads[i], rd_per_kb_per_bam=round(rdkb, 1), kept="yes" if keep else "no"))
        if keep:
            kept.append((c, s, e, g)); bp_keep += L; by_gene_keep[g] += 1

    # labeled kept BED
    kept.sort(key=lambda r: (chrom_order(r[0]), r[1], r[2]))
    with open(f"{PANEL}/grail_panel_exons.depth_covered.labeled.bed", "w") as f:
        for c, s, e, g in kept: f.write(f"{c}\t{s}\t{e}\t{g}\n")
    # gene-agnostic merged
    bych = defaultdict(list)
    for c, s, e, g in kept: bych[c].append((s, e))
    n_merged = 0; bp_merged = 0
    with open(f"{PANEL}/grail_panel_exons.depth_covered.merged.bed", "w") as f:
        for c in sorted(bych, key=chrom_order):
            for s, e in merge_iv(bych[c]):
                f.write(f"{c}\t{s}\t{e}\n"); n_merged += 1; bp_merged += e - s
    # per-interval table
    with open(f"{PANEL}/grail_panel_exons.depth_per_interval.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_iv[0]), delimiter="\t"); w.writeheader(); w.writerows(per_iv)

    all_genes = set(g for _, _, _, g in intervals)
    genes_dropped = sorted(all_genes - set(by_gene_keep))
    rep = dict(
        n_bams=len(sample), keep_threshold_rd_per_kb_per_bam=THRESH,
        exon_intervals_in=len(intervals), exon_intervals_kept=len(kept),
        Mb_in=round(bp_in / 1e6, 3), Mb_kept_merged=round(bp_merged / 1e6, 3),
        published_panel_Mb=2.13, footprint_intersection_Mb=1.898,
        genes_with_zero_kept=len(genes_dropped), genes_dropped=genes_dropped,
    )
    json.dump(rep, open(f"{PANEL}/grail_panel_depthfilter_report.json", "w"), indent=2)
    log("=== depth-filtered exon target (exons ∩ direct read depth) ===")
    log(json.dumps(rep, indent=2))
    log(f"[depthfilter] -> grail_panel_exons.depth_covered.merged.bed ({rep['Mb_kept_merged']} Mb, {n_merged} intervals)")


if __name__ == "__main__":
    main()
