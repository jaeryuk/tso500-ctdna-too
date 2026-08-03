#!/usr/bin/env python
"""
Gene-anchored GRAIL (MSK-TechVal) targeted-panel BED, built from the PUBLISHED gene list.

Source of the gene list: Razavi et al. 2019 Nat Med, Supplementary Table 1
("List of genes assayed in the cfDNA targeted panel") -> 508 cancer-related genes (2.13 Mb;
GRAIL, Inc.). Panel design per Methods: "full exons except for TERT (promoter regions only);
additional intronic regions for rearrangement detection of 28 genes and CNA detection of 42 genes."

This is the canonical gene-anchoring GRAIL themselves use (MSK-GRAIL-TECHVAL repo get_target_lengths():
overlap the panel BED with hg19 gene SYMBOLs). Here we invert it: take the published 508 symbols and
emit their hg19 exon intervals from Ensembl GRCh37.75, then validate against the empirical capture
footprint recovered from the actual collapsed BAMs (results/grail_deepsomatic/ref/panel.bed).

Outputs (results/auto_plan/grail/panel/):
  grail_panel_genes.txt              the 508 published symbols (one per line)
  grail_panel_exons.labeled.bed      per-gene merged exon intervals (chr, start, end, gene)  [PAD-padded]
  grail_panel_exons.merged.bed       gene-agnostic merged intervals (for masking / binning)
  grail_panel_cds.merged.bed         CDS-only merged intervals (span comparison vs 2.13 Mb)
  grail_panel_per_gene.tsv           gene, ensembl_name, chrom, n_intervals, exon_bp, covered_in_data
  grail_panel_build_report.json      counts, spans, alias map, TERT handling, coverage-footprint validation
"""
import os, re, json
from collections import defaultdict

PROJ = "/home/jrkim/TSO_TFBS/project"
GTF = "/home/jrkim/cfdna_se_project/ref/genes/Homo_sapiens.GRCh37.75.gtf"
GENOME = "/home/jrkim/cfdna_se_project/ref/genome/b37.autosome.genome"     # chrom sizes (1..22), for clipping
PANEL_TXT = os.environ.get("PANEL_TXT",
            "/tmp/claude-1002/-home-jrkim/5ebb3ac4-50d5-4c66-a4d7-ce4c72a5252e/scratchpad/gsupp/panel_genes.txt")
COV_BED = f"{PROJ}/results/grail_deepsomatic/ref/panel.bed"                 # empirical capture footprint (validation)
OUT = f"{PROJ}/results/auto_plan/grail/panel"; os.makedirs(OUT, exist_ok=True)

PRIMARY = {str(i) for i in range(1, 23)} | {"X", "Y"}  # exclude alt-haplotype / patch contigs (not in BAMs)
PAD = int(os.environ.get("PANEL_PAD", "0"))           # bp padding added to each exon on both sides
ALIAS = {"FAM123B": "AMER1", "MYCL1": "MYCL"}         # panel symbol -> Ensembl GRCh37.75 symbol
# TERT is promoter-only in the GRAIL panel (NOT full exons). hg19 chr5, minus strand; TSS ~1,295,162.
# Core promoter incl. C228T (1,295,228) & C250T (1,295,250) hotspots:
TERT_PROMOTER = ("5", 1294900, 1295500)
def log(m): print(m, flush=True)


def merge_iv(iv):
    """merge list of (start,end) half-open intervals."""
    iv = sorted(iv); out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            if e > out[-1][1]: out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def chrom_order(c):
    cc = c.replace("chr", "")
    o = {**{str(i): i for i in range(1, 23)}, "X": 23, "Y": 24, "MT": 25, "M": 25}
    return o.get(cc, 99)


def overlap_bp(a, b):
    """total bp of overlap between two per-chrom dicts {chrom:[(s,e)..]} (each pre-merged & sorted)."""
    tot = 0
    for c in set(a) & set(b):
        A, B = a[c], b[c]; i = j = 0
        while i < len(A) and j < len(B):
            s = max(A[i][0], B[j][0]); e = min(A[i][1], B[j][1])
            if e > s: tot += e - s
            if A[i][1] < B[j][1]: i += 1
            else: j += 1
    return tot


def main():
    sizes = {}
    for ln in open(GENOME):
        c, n = ln.split()[:2]; sizes[c] = int(n)

    panel = [l.strip() for l in open(PANEL_TXT) if l.strip()]
    assert len(panel) == len(set(panel)), "duplicate panel genes"
    log(f"[panel] published genes: {len(panel)}")

    want = {}                         # ensembl_name -> panel_name
    for g in panel:
        want[ALIAS.get(g, g)] = g

    # ---- parse GRCh37.75 exon + CDS features for wanted genes ----
    exons = defaultdict(list)         # ensembl_name -> [(chrom,start0,end,gene_id)]
    cds = defaultdict(list)
    re_name = re.compile(r'gene_name "([^"]+)"')
    re_id = re.compile(r'gene_id "([^"]+)"')
    with open(GTF) as f:
        for ln in f:
            if ln[0] == "#": continue
            is_exon = "\texon\t" in ln
            is_cds = "\tCDS\t" in ln
            if not (is_exon or is_cds): continue
            m = re_name.search(ln)
            if not m or m.group(1) not in want: continue
            gn = m.group(1); fld = ln.split("\t")
            chrom = fld[0]
            if chrom not in PRIMARY: continue                          # skip alt-haplotype / patch contigs
            s0 = int(fld[3]) - 1; e = int(fld[4])                      # GTF 1-based incl -> 0-based half-open
            gid = re_id.search(ln).group(1)
            (exons if is_exon else cds)[gn].append((chrom, s0, e, gid))

    # ---- resolve each gene to its dominant locus, merge exons ----
    labeled = []                      # (chrom, start, end, panel_gene)
    per_gene = []                     # dict rows
    cds_iv = defaultdict(list)        # chrom -> [(s,e)] for CDS span calc
    for en, pan in want.items():
        if pan == "TERT":             # promoter-only; handled below
            continue
        lst = exons.get(en, [])
        if not lst:
            per_gene.append(dict(gene=pan, ensembl_name=en, chrom="NA", n_intervals=0, exon_bp=0)); continue
        bych = defaultdict(list)
        for c, s, e, gid in lst: bych[(c, gid)].append((s, e))
        best = None; best_bp = -1
        for (c, gid), iv in bych.items():
            mg = merge_iv(iv); bp = sum(e - s for s, e in mg)
            if bp > best_bp: best_bp = bp; best = (c, mg)
        c, mg = best
        cap = sizes.get(c, 10**9)
        for s, e in mg:
            s2 = max(0, s - PAD); e2 = min(cap, e + PAD)
            labeled.append((c, s2, e2, pan))
        per_gene.append(dict(gene=pan, ensembl_name=en, chrom=c, n_intervals=len(mg), exon_bp=best_bp))
        # CDS dominant locus (same chrom) for span comparison
        clst = [(s, e) for cc, s, e, gid in cds.get(en, []) if cc == c]
        if clst:
            for s, e in merge_iv(clst): cds_iv[c].append((s, e))

    # TERT promoter
    c, s, e = TERT_PROMOTER
    labeled.append((c, s, e, "TERT"))
    per_gene.append(dict(gene="TERT", ensembl_name="TERT(promoter-only)", chrom=c, n_intervals=1, exon_bp=e - s))

    # ---- sort + write labeled per-gene exon BED (chr-prefixed to match GRAIL BAMs) ----
    labeled.sort(key=lambda r: (chrom_order(r[0]), r[1], r[2]))
    with open(f"{OUT}/grail_panel_exons.labeled.bed", "w") as f:
        for c, s, e, g in labeled:
            f.write(f"chr{c}\t{s}\t{e}\t{g}\n")

    # ---- gene-agnostic merged BED (collapse overlaps across adjacent genes) ----
    bychrom = defaultdict(list)
    for c, s, e, g in labeled: bychrom[c].append((s, e))
    merged = {c: merge_iv(v) for c, v in bychrom.items()}
    n_merged = 0; bp_merged = 0
    with open(f"{OUT}/grail_panel_exons.merged.bed", "w") as f:
        for c in sorted(merged, key=chrom_order):
            for s, e in merged[c]:
                f.write(f"chr{c}\t{s}\t{e}\n"); n_merged += 1; bp_merged += e - s

    # ---- CDS-only merged (span comparison vs published 2.13 Mb) ----
    cds_merged = {c: merge_iv(v) for c, v in cds_iv.items()}
    bp_cds = sum(e - s for c in cds_merged for s, e in cds_merged[c])
    with open(f"{OUT}/grail_panel_cds.merged.bed", "w") as f:
        for c in sorted(cds_merged, key=chrom_order):
            for s, e in cds_merged[c]:
                f.write(f"chr{c}\t{s}\t{e}\n")

    # ---- validate against empirical capture footprint (coverage-derived panel.bed) ----
    cov = defaultdict(list)
    if os.path.exists(COV_BED):
        for ln in open(COV_BED):
            p = ln.split("\t");  cov[p[0].replace("chr", "")].append((int(p[1]), int(p[2])))
        cov = {c: merge_iv(v) for c, v in cov.items()}
    bp_cov = sum(e - s for c in cov for s, e in cov[c])
    ov = overlap_bp(merged, cov) if cov else 0

    # high-confidence on-target BED = gene-anchored exons clipped to the empirical coverage footprint
    def clip(intervals, mask):
        out = []
        for s, e in intervals:
            for ms, me in mask:
                a, b = max(s, ms), min(e, me)
                if b > a: out.append((a, b))
        return merge_iv(out)
    covered_bed = {c: clip(merged[c], cov.get(c, [])) for c in merged} if cov else {}
    with open(f"{OUT}/grail_panel_exons.covered.bed", "w") as f:
        for c in sorted(covered_bed, key=chrom_order):
            for s, e in covered_bed[c]:
                f.write(f"chr{c}\t{s}\t{e}\n")

    # per-gene coverage support
    covered = 0; uncov_x = 0
    for r in per_gene:
        c = r["chrom"]
        if c == "NA": r["covered_in_data"] = "no_annotation"; continue
        ge = [(s, e) for cc, s, e, g in labeled if cc == c and g == r["gene"]]
        sup = overlap_bp({c: merge_iv(ge)}, {c: cov.get(c, [])}) if cov else 0
        r["covered_in_data"] = "yes" if sup > 0 else "no"
        covered += sup > 0
        if sup == 0 and c == "X": uncov_x += 1

    # ---- write per-gene table + report ----
    import csv
    cols = ["gene", "ensembl_name", "chrom", "n_intervals", "exon_bp", "covered_in_data"]
    with open(f"{OUT}/grail_panel_per_gene.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); w.writeheader()
        for r in sorted(per_gene, key=lambda x: x["gene"]): w.writerow(r)
    open(f"{OUT}/grail_panel_genes.txt", "w").write("\n".join(panel) + "\n")

    n_no_annot = sum(1 for r in per_gene if r["chrom"] == "NA")
    rep = dict(
        source="Razavi 2019 Nat Med, Supplementary Table 1 (MOESM1 PDF); panel 508 genes / 2.13 Mb (GRAIL)",
        annotation="Ensembl GRCh37.75 (hg19)", pad_bp=PAD,
        n_genes_published=len(panel), n_genes_annotated=len(panel) - n_no_annot, n_genes_no_annotation=n_no_annot,
        alias_map=ALIAS, tert="promoter-only window chr5:%d-%d (hg19)" % (TERT_PROMOTER[1], TERT_PROMOTER[2]),
        exons_labeled_intervals=len(labeled),
        exons_merged_intervals=n_merged, exons_merged_Mb=round(bp_merged / 1e6, 3),
        cds_merged_Mb=round(bp_cds / 1e6, 3), published_panel_Mb=2.13,
        coverage_footprint_Mb=round(bp_cov / 1e6, 3),
        highconf_exons_covered_Mb=round(ov / 1e6, 3),                 # gene-exons clipped to coverage footprint
        pct_geneanchored_in_coverage=round(100 * ov / bp_merged, 1) if bp_merged else None,
        pct_coverage_explained_by_genes=round(100 * ov / bp_cov, 1) if bp_cov else None,
        genes_covered_in_data=int(covered), genes_not_covered=int((len(panel) - n_no_annot) - covered),
        genes_not_covered_on_chrX=int(uncov_x),
    )
    json.dump(rep, open(f"{OUT}/grail_panel_build_report.json", "w"), indent=2)
    log("=== gene-anchored GRAIL panel BED ===")
    log(json.dumps(rep, indent=2))
    log(f"[panel] -> {OUT}/grail_panel_exons.labeled.bed (+ merged/cds/per_gene/report)")
    if n_no_annot:
        log("[panel] NO-ANNOTATION genes: " +
            ", ".join(r["gene"] for r in per_gene if r["chrom"] == "NA"))
    nc = [r["gene"] for r in per_gene if r.get("covered_in_data") == "no"]
    if nc: log(f"[panel] {len(nc)} annotated genes with NO coverage support in data: " + ", ".join(sorted(nc)))


if __name__ == "__main__":
    main()
