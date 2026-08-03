#!/usr/bin/env python
"""
Faithful Helzer E1SE (exon-1 Shannon entropy), exactly per Helzer et al. 2023
("Fragmentomic analysis of circulating tumor DNA-targeted cancer panels", Ann Oncol 34:813).

Definition (paper Methods):
  * region   = first coding exon (E1) of each gene  -> TSO500 "_Exon1_" manifest targets, pooled per gene
  * fragment = properly-paired primary mapped read pair; span = [start, stop] (template start..stop),
               insert size = stop - start, kept if 1 <= size <= 1000
  * a fragment counts toward E1 if it OVERLAPS the exon by >= 1 bp  (NOT midpoint)
  * E1SE     = Shannon entropy (natural log, R 'entropy' pkg empirical estimator) of the FREQUENCIES
               of fragment SIZES overlapping the first coding exon, per gene per sample
  * feature gate (assemble) = keep a gene only if >= 500 reads overlap its E1 across ALL samples

mode one <sid>     -> results/v2_e1se/_raw/<sid>.json   (resumable, xargs-parallel)
mode assemble      -> results/v2_e1se/X_E1SE.npz + e1se_feature_table.tsv
"""
import os, sys, json, glob, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pysam

PROJ = "/home/jrkim/TSO_TFBS/project"
TAG = os.environ.get("TSO_TAG", "v2")                                    # cohort tag -> output dir
OUT = f"{PROJ}/results/{TAG}_e1se"; RAW = f"{OUT}/_raw"
MAN = os.environ.get("TSO_MANIFEST", f"{PROJ}/results/v2_ponbench/manifest.tsv")
MANIFEST = "/home/jrkim/TSO_TFBS/TST500C_manifest.bed"
PAD = 1000              # fetch window pad upstream (max fragment) so leftmost spanning reads are seen
# fragment insert-size inclusion range. Helzer 2023 Ann Oncol general metric = 1-1000; Helzer 2025 Nat Commun
# E1SE Methods = "fragment size inserts between 20 bp and 500 bp were retained". Set via env for a faithful-2025 run.
SZ_LO = int(os.environ.get("E1SE_SZLO", "1")); SZ_HI = int(os.environ.get("E1SE_SZHI", "1000"))
MIN_TOTAL_READS = int(os.environ.get("E1SE_MINREADS", "500"))   # feature-inclusion gate (2025 does not specify; keep as QC)
# NEAREST-EXON SUBSTITUTION (Helzer 2025 Nat Commun): when a gene's first coding exon is NOT on the panel, use its
# NEAREST captured coding exon (lowest captured exon number) as the E1 proxy. Off by default (=faithful-2023 Exon1-only).
E1SE_NEAREST = os.environ.get("E1SE_NEAREST", "0") == "1"
MAPQ = int(os.environ.get("E1SE_MAPQ", "20"))   # Helzer QC (matches all-exon-depth): MAPQ>=20 + duplicate removal
def log(m): print(m, flush=True)


def e1_genes():
    """gene -> list of (chrom, ts, te) first-coding-exon intervals from TSO500 manifest."""
    g = {}
    with open(MANIFEST) as f:
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < 4: continue
            name = c[3]; parts = name.split("_")
            if len(parts) < 2 or parts[1] != "Exon1": continue
            gene = parts[0]
            g.setdefault(gene, []).append((c[0], int(c[1]), int(c[2])))
    return dict(sorted(g.items()))


def e1_genes_nearest():
    """Helzer-2025 nearest-exon substitution: per gene use its FIRST captured coding exon (lowest exon number);
    Exon1 when captured, else the nearest captured coding exon (Exon2, Exon3, ...). Exon number parsed from the
    TSO500 manifest target name '<gene>_Exon<N>_<RefSeq>'. Multiple intervals of the chosen exon are pooled."""
    per = {}                                                     # gene -> {exon_num: [(chrom,ts,te),...]}
    with open(MANIFEST) as f:
        for ln in f:
            c = ln.rstrip("\n").split("\t")
            if len(c) < 4: continue
            parts = c[3].split("_")
            if len(parts) < 2: continue
            m = re.fullmatch(r"Exon(\d+)", parts[1])
            if not m: continue
            gene = parts[0]; en = int(m.group(1))
            per.setdefault(gene, {}).setdefault(en, []).append((c[0], int(c[1]), int(c[2])))
    g = {}; sub = 0
    for gene, exons in per.items():
        nmin = min(exons)                                        # nearest captured exon to the 5' first coding exon
        g[gene] = exons[nmin]
        if nmin != 1: sub += 1
    log(f"[genes] nearest-exon: {len(g)} genes ({sub} used a substituted exon, Exon1 not captured)")
    return dict(sorted(g.items()))


def pick_genes():
    return e1_genes_nearest() if E1SE_NEAREST else e1_genes()


def load_manifest():
    rows = []
    with open(MAN) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for ln in f: rows.append(dict(zip(hdr, ln.rstrip("\n").split("\t"))))
    return {r["sid"]: r for r in rows}


def shannon_nat(counts):
    """Shannon entropy in nats of a count vector (R entropy::entropy default unit='log')."""
    s = counts.sum()
    if s <= 0: return np.nan
    p = counts[counts > 0] / s
    return float(-(p * np.log(p)).sum())


def e1se_one_bam(bam, genes):
    ent = {}; cnt = {}
    bf = pysam.AlignmentFile(bam, "rb")
    refs = set(bf.references)
    for gene, ivs in genes.items():
        sizes = {}                          # qname -> insert size (dedupe across overlapping intervals)
        for (chrom, ts, te) in ivs:
            if chrom not in refs: continue
            for r in bf.fetch(chrom, max(0, ts - PAD), te):
                if r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_duplicate or r.is_qcfail or not r.is_proper_pair:
                    continue
                if r.mapping_quality < MAPQ:
                    continue
                tl = r.template_length
                if tl <= 0:                 # count each fragment once (leftmost mate, TLEN>0)
                    continue
                if tl < SZ_LO or tl > SZ_HI:
                    continue
                fs = r.reference_start; fe = fs + tl
                if fe > ts and fs < te:     # >=1bp overlap with the exon
                    sizes[r.query_name] = tl
        if sizes:
            h = np.bincount(np.fromiter(sizes.values(), dtype=int), minlength=SZ_HI + 1)[SZ_LO:SZ_HI + 1]
            ent[gene] = shannon_nat(h.astype(float)); cnt[gene] = int(h.sum())
        else:
            ent[gene] = np.nan; cnt[gene] = 0
    bf.close()
    return ent, cnt


def cmd_one(sid):
    os.makedirs(RAW, exist_ok=True)
    out = f"{RAW}/{sid}.json"
    if os.path.exists(out): log(f"[{sid}] cached"); return
    man = load_manifest()
    if sid not in man: log(f"[{sid}] not in manifest"); return
    bam = man[sid]["bam"]
    if not os.path.exists(bam): log(f"[{sid}] BAM missing {bam}"); return
    ent, cnt = e1se_one_bam(bam, pick_genes())
    json.dump({"ent": {g: (None if not np.isfinite(v) else v) for g, v in ent.items()}, "cnt": cnt},
              open(out, "w"))
    tot = sum(cnt.values()); ng = sum(1 for v in cnt.values() if v >= 100)
    log(f"[{sid}] OK genes={len(ent)} reads={tot} genes>=100frag={ng}")


def cmd_assemble():
    man = load_manifest()
    files = sorted(glob.glob(f"{RAW}/*.json"))
    raws = {os.path.basename(f)[:-5]: json.load(open(f)) for f in files}
    sids = [s for s in raws if s in man]
    genes = sorted({g for s in sids for g in raws[s]["ent"]})
    log(f"[assemble] {len(sids)} samples, {len(genes)} E1 genes (pre-gate)")
    total = {g: sum(int(raws[s]["cnt"].get(g, 0)) for s in sids) for g in genes}
    keep = [g for g in genes if total[g] >= MIN_TOTAL_READS]
    log(f"[assemble] keep {len(keep)} genes with >= {MIN_TOTAL_READS} reads across all samples")
    X = np.full((len(sids), len(keep)), np.nan, np.float32); gi = {g: i for i, g in enumerate(keep)}
    for r, s in enumerate(sids):
        e = raws[s]["ent"]
        for g in keep:
            v = e.get(g)
            if v is not None and np.isfinite(v): X[r, gi[g]] = v
    col_med = np.nanmedian(X, 0)
    nanidx = np.where(~np.isfinite(X)); X[nanidx] = np.take(np.nan_to_num(col_med), nanidx[1])
    np.savez(f"{OUT}/X_E1SE.npz", X=X, sids=np.array(sids), cols=np.array(keep))
    pd.DataFrame(X, index=sids, columns=keep).to_csv(f"{OUT}/e1se_feature_table.tsv", sep="\t")
    log(f"[assemble] X_E1SE {X.shape} -> {OUT}/X_E1SE.npz")


if __name__ == "__main__":
    if sys.argv[1] == "one": cmd_one(sys.argv[2])
    elif sys.argv[1] == "assemble": cmd_assemble()
    elif sys.argv[1] == "genes": g = pick_genes(); print(f"{len(g)} genes, {sum(len(v) for v in g.values())} exon intervals (nearest={E1SE_NEAREST})")
    else: sys.exit(f"unknown cmd {sys.argv[1]}")
