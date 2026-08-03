#!/usr/bin/env python
"""Mutation module = clinically-reportable variants (CombinedVariantOutput [Small Variants]) INTERSECTED with
the TMB-trace SOMATIC calls (user directive 2026-07-27: features from the reportable CVO; use the TMB trace
ONLY as the somatic gate -> 'use only tsv mutations that intersect with tmb trace to ensure somatic mutation').

Per sample: keep a CVO reportable variant iff its (chrom,pos,ref,alt) also appears in the TMB trace as Somatic.
From that somatic-reportable set build:
  gene-level : mut_<G> (>=1 nonsynonymous), trunc_<G> (>=1 truncating)   [gene + consequence from the CVO]
  hotspots   : hs_<GENE AA> matching the curated GDD hotspot_list (P-Dot HGVS 3->1-letter, substitutions)
Features kept if occurring in >= K samples (MINF, default 3).

Trace somatic: v2 & v1-adapted traces have a Status col (Somatic/Germline); v1 RAW TMB_Trace derives it as
Somatic = NOT(GermlineFilterDatabase or GermlineFilterProxi). Resolution via v1 manifest.tsv / v2
manifest_bamfix.tsv -> bam -> oid (strip _tumor.bam / .bam) -> {run}/Results/{oid}/.
Out: results/auto_plan/feat_rc/{coh}/X_Somatic_mutation_profile_cvo.npz + mut_hotspots/{coh}_curated_hotspot.npz.
"""
import os, sys, re, glob
import numpy as np, pandas as pd
from collections import Counter
from joblib import Parallel, delayed
NJOB = int(os.environ.get("NJOB", "40"))

PROJ = "/home/jrkim/TSO_TFBS/project"
FEAT = f"{PROJ}/results/auto_plan/feat_rc"; HOTD = f"{PROJ}/results/rule_conformant/mut_hotspots"
os.makedirs(HOTD, exist_ok=True)
K = int(os.environ.get("MINF", "3"))
TAG = os.environ.get("MUT_TAG", "")           # output suffix; "" = definitive, "_nofilter" = side experiment
HOT = set(x for x in open(f"{PROJ}/results/rule_conformant/gdd_hotspot_list.txt").read().split("\n") if x)
AA = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G','His':'H','Ile':'I',
      'Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Ter':'*'}
NONSYN = ("missense", "stop_gained", "stop_lost", "start_lost", "frameshift", "inframe", "protein_altering",
          "splice_acceptor", "splice_donor")
TRUNC = ("stop_gained", "frameshift", "splice_acceptor", "splice_donor", "start_lost", "stop_lost", "exon_loss")
MANP = {"v1": f"{PROJ}/results/v1_ponbench/manifest.tsv", "v2": f"{PROJ}/results/v2_ponbench/manifest_bamfix.tsv"}
V1TR = f"{PROJ}/results/v1_gdd/_trace"

def load_man(p):
    man = {}
    for i, ln in enumerate(open(p)):
        c = ln.rstrip("\n").split("\t")
        if i == 0: hdr = c; continue
        r = dict(zip(hdr, c)); man[str(r["sid"])] = r
    return man

def oid_run(r):
    bam = r.get("bam", ""); return os.path.basename(bam).replace("_tumor.bam", "").replace(".bam", ""), os.path.dirname(os.path.dirname(bam))

def cvo_path(sid, man):
    r = man.get(sid);
    if r is None: return None
    oid, run = oid_run(r); p = f"{run}/Results/{oid}/{oid}_CombinedVariantOutput.tsv"
    if os.path.exists(p): return p
    g = glob.glob(f"{run}/Results/{oid}*/{oid}*_CombinedVariantOutput.tsv"); return g[0] if g else None

def trace_path(coh, sid, man):
    if coh == "v1":
        a = f"{V1TR}/{sid}.tmb.trace.tsv"
        if os.path.exists(a): return a
    r = man.get(sid)
    if r is None: return None
    oid, run = oid_run(r)
    for cand in (f"{run}/Results/{oid}/{oid}.tmb.trace.tsv", f"{run}/Results/{oid}/{oid}_TMB_Trace.tsv"):
        if os.path.exists(cand): return cand
    return None

def norm(ch): return str(ch).replace("chr", "")

def trace_somatic(path):
    if not path or not os.path.exists(path): return None
    try: df = pd.read_csv(path, sep="\t", dtype=str)
    except Exception: return None
    if "Status" in df.columns:
        sub = df[df["Status"].astype(str).str.startswith("Somatic")]
    elif "GermlineFilterDatabase" in df.columns:
        gd = df["GermlineFilterDatabase"].astype(str); gp = df.get("GermlineFilterProxi", pd.Series([""] * len(df))).astype(str)
        sub = df[~((gd == "True") | (gp == "True"))]
    else: return set()
    return set(zip(sub["Chromosome"].map(norm), sub["Position"].astype(str), sub["RefCall"].astype(str), sub["AltCall"].astype(str)))

def pdot_allele(gene, pdot):
    m = re.search(r'p\.\(?([A-Za-z]{3})(\d+)([A-Za-z]{3})\)?', pdot)
    if not m: return None
    a1, pos, a2 = m.group(1), m.group(2), m.group(3)
    if a1 not in AA or a2 not in AA or AA[a1] == AA[a2]: return None
    return f"{gene} {AA[a1]}{pos}{AA[a2]}"

def cvo_reportable(path):
    """(chrom,pos,ref,alt) -> (gene, consequence, pdot) for reportable [Small Variants]."""
    out = {}
    if not path or not os.path.exists(path): return None
    lines = open(path, encoding="utf-8", errors="ignore").read().split("\n")
    i = 0
    while i < len(lines) and not lines[i].startswith("[Small Variants]"): i += 1
    if i >= len(lines) - 1: return out
    i += 1; hdr = lines[i].split("\t"); i += 1
    H = {c.strip(): j for j, c in enumerate(hdr)}
    need = ["Gene", "Chromosome", "Genomic Position", "Reference Call", "Alternative Call"]
    if not all(k in H for k in need): return out
    ci = H.get("Consequence(s)"); pi = H.get("P-Dot Notation")
    while i < len(lines) and not lines[i].startswith("["):
        c = lines[i].split("\t"); i += 1
        if len(c) <= max(H.values()): continue
        key = (norm(c[H["Chromosome"]]), c[H["Genomic Position"]], c[H["Reference Call"]], c[H["Alternative Call"]])
        out[key] = (c[H["Gene"]].strip(), (c[ci].lower() if ci is not None else ""), (c[pi] if pi is not None else ""))
    return out

def per_sample(coh, sid, man):
    cvo = cvo_reportable(cvo_path(sid, man)); som = trace_somatic(trace_path(coh, sid, man))
    mut = set(); tr = set(); hs = set(); inter = 0; ok = (cvo is not None and som is not None)
    if ok:
        for key, (gene, cons, pdot) in cvo.items():
            if key not in som: continue                  # SOMATIC gate: must be in trace-somatic
            inter += 1
            if not gene: continue
            if any(k in cons for k in NONSYN): mut.add(gene)
            if any(k in cons for k in TRUNC): tr.add(gene)
            a = pdot_allele(gene, pdot)
            if a and a in HOT: hs.add(a)
    return sid, mut, tr, hs, inter, ok

def build(cohorts=("v1", "v2")):
    ann = []
    for coh in cohorts:
        z = np.load(f"{FEAT}/{coh}/X_Somatic_mutation_profile.npz", allow_pickle=True)
        sids = [str(s) for s in z["sids"]]; man = load_man(MANP[coh])
        print(f"[{coh}] parsing {len(sids)} samples (NJOB={NJOB}) ...", flush=True)
        res = Parallel(n_jobs=NJOB, prefer="processes")(delayed(per_sample)(coh, sid, man) for sid in sids)
        M = {}; T = {}; Hs = {}; nres = 0; nnotr = 0; ninter = []
        for sid, mut, tr, hs, inter, ok in res:
            M[sid], T[sid], Hs[sid] = mut, tr, hs
            if ok: nres += 1; ninter.append(inter)
            else: nnotr += 1
        cm = Counter(g for s in sids for g in M[s]); ct = Counter(g for s in sids for g in T[s]); ch = Counter(h for s in sids for h in Hs[s])
        mg = sorted([g for g, n in cm.items() if n >= K]); tg = sorted([g for g, n in ct.items() if n >= K]); hg = sorted([h for h, n in ch.items() if n >= K])
        cols = [f"mut_{g}" for g in mg] + [f"trunc_{g}" for g in tg]; ci = {c: k for k, c in enumerate(cols)}
        X = np.zeros((len(sids), len(cols)), np.float32)
        for k, sid in enumerate(sids):
            for g in M[sid]:
                if f"mut_{g}" in ci: X[k, ci[f"mut_{g}"]] = 1
            for g in T[sid]:
                if f"trunc_{g}" in ci: X[k, ci[f"trunc_{g}"]] = 1
        np.savez(f"{FEAT}/{coh}/X_Somatic_mutation_profile_cvo{TAG}.npz", X=X, sids=np.array(sids), cols=np.array(cols))
        hci = {h: k for k, h in enumerate(hg)}; XH = np.zeros((len(sids), len(hg)), np.float32)
        for k, sid in enumerate(sids):
            for h in Hs[sid]:
                if h in hci: XH[k, hci[h]] = 1
        np.savez(f"{HOTD}/{coh}_curated_hotspot{TAG}.npz", X=XH, sids=np.array(sids), cols=np.array([f"hs_{h}" for h in hg]))
        mi = float(np.mean(ninter)) if ninter else 0
        print(f"[{coh}] resolved {nres}/{len(sids)} (noData {nnotr}); mean somatic-reportable variants/sample={mi:.1f}; "
              f"gene-level {len(cols)} cols ({len(mg)} mut_ + {len(tg)} trunc_, {len(set(mg)|set(tg))} genes); "
              f"curated hotspots {len(hg)} (mean/sample {XH.sum(1).mean():.2f})", flush=True)
        for h in hg: ann.append(dict(cohort=coh, hotspot=h, n_samples=int(ch[h])))
    pd.DataFrame(ann).sort_values(["cohort", "n_samples"], ascending=[True, False]).to_csv(f"{HOTD}/curated_hotspot_alleles{TAG}.tsv", sep="\t", index=False)
    print("EXTRACT_CVO_MUTATION_DONE", flush=True)

if __name__ == "__main__":
    build()
