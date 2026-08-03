#!/usr/bin/env python
"""
Validate the gene-anchored GRAIL panel BED against ACTUAL read depth in the collapsed BAMs.

For each of the 508 published-panel genes, count reads overlapping its (merged) exon intervals across
a spread of GRAIL cfDNA BAMs, and express as reads/kb/BAM. Panel-captured genes sit at hundreds-thousands
rd/kb/BAM; genuinely uncaptured loci sit < ~3. This is self-contained (does not rely on the coarse
coverage-threshold footprint) and reproducible.

In : results/auto_plan/grail/panel/grail_panel_exons.labeled.bed  (gene-anchored exons, chr-prefixed)
     results/auto_plan/grail/manifest.tsv                          (BAM paths)
Out: results/auto_plan/grail/panel/grail_panel_coverage_validation.tsv  (per-gene depth + captured flag)
     updates grail_panel_build_report.json with the read-depth validation summary
"""
import os, json, csv
from collections import defaultdict, OrderedDict
import pysam

PROJ = "/home/jrkim/TSO_TFBS/project"
PANEL = f"{PROJ}/results/auto_plan/grail/panel"
MAN = f"{PROJ}/results/auto_plan/grail/manifest.tsv"
NBAM = int(os.environ.get("NBAM", "6"))           # BAMs sampled (capture is ~sample-independent)
CAPTURED_RDKB = float(os.environ.get("CAPTURED_RDKB", "20"))   # rd/kb/BAM threshold (covered>>20, uncovered<3)
def log(m): print(m, flush=True)


def main():
    # per-gene merged exon intervals from the labeled BED
    gi = defaultdict(list)
    for ln in open(f"{PANEL}/grail_panel_exons.labeled.bed"):
        c, s, e, g = ln.rstrip("\n").split("\t")
        gi[g].append((c, int(s), int(e)))
    chrom_of = {g: v[0][0] for g, v in gi.items()}
    kb = {g: sum(e - s for _, s, e in v) / 1000.0 for g, v in gi.items()}

    # evenly-spread BAM sample
    rows = open(MAN).read().splitlines()[1:]
    bams = [r.split("\t")[1] for r in rows]
    step = max(1, len(bams) // NBAM)
    sample = bams[::step][:NBAM]
    log(f"[validate] {len(gi)} genes x {len(sample)} BAMs (rd/kb/BAM, captured>= {CAPTURED_RDKB})")

    reads = defaultdict(int)
    for bp in sample:
        bam = pysam.AlignmentFile(bp, "rb")
        for g, ivs in gi.items():
            n = 0
            for c, s, e in ivs:
                try: n += bam.count(c, s, e)        # reads overlapping the interval
                except Exception: pass
            reads[g] += n
        bam.close()

    out = []
    for g in sorted(gi):
        rdkb = reads[g] / kb[g] / len(sample) if kb[g] > 0 else 0.0
        out.append(dict(gene=g, chrom=chrom_of[g], exon_kb=round(kb[g], 2),
                        reads_total=reads[g], rd_per_kb_per_bam=round(rdkb, 1),
                        captured="yes" if rdkb >= CAPTURED_RDKB else "no"))
    with open(f"{PANEL}/grail_panel_coverage_validation.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter="\t"); w.writeheader(); w.writerows(out)

    cap = [r for r in out if r["captured"] == "yes"]
    notcap = [r for r in out if r["captured"] == "no"]
    log(f"[validate] captured: {len(cap)}/{len(out)}  not-captured: {len(notcap)}")
    log("[validate] NOT captured (rd/kb/BAM): " +
        ", ".join(f"{r['gene']}({r['chrom']},{r['rd_per_kb_per_bam']})" for r in sorted(notcap, key=lambda x: x['gene'])))

    # fold into the build report
    rp = f"{PANEL}/grail_panel_build_report.json"
    rep = json.load(open(rp)) if os.path.exists(rp) else {}
    rep["readdepth_validation"] = dict(
        n_bams=len(sample), captured_threshold_rd_per_kb_per_bam=CAPTURED_RDKB,
        genes_captured=len(cap), genes_not_captured=len(notcap),
        not_captured_genes=sorted(r["gene"] for r in notcap),
        median_rdkb_captured=round(sorted(r["rd_per_kb_per_bam"] for r in cap)[len(cap) // 2], 1) if cap else None,
    )
    json.dump(rep, open(rp, "w"), indent=2)
    log(f"[validate] -> {PANEL}/grail_panel_coverage_validation.tsv ; report updated")


if __name__ == "__main__":
    main()
