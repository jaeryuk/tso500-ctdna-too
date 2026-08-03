#!/usr/bin/env python
"""Build GRAIL cfDNA feature matrices for external validation -> results/auto_plan/grail/feat/X_<M>.npz.
Reliable panel-agnostic classes (built from existing GRAIL outputs):
  Genome_wide_CNA          via v2_cna_features.py on GRAIL ichorCNA (subprocess)
  Mutation_signature       96-channel COSMIC exposures from GRAIL somatic calls (gdd_signatures)
  Somatic_mutation_profile per-gene nonsyn flags from GRAIL calls
Fragmentomic classes (E1/All_exon_depth/SHAPE) are panel-overlap dependent -> attempted fail-soft elsewhere.
Reads GRAIL calls from results/grail_deepsomatic/{calls,calls_tumoronly}/<sid>/raw.tsv.
"""
import os, sys, glob, json, subprocess, collections
import numpy as np, pandas as pd
sys.path.insert(0, "/home/jrkim/TSO_TFBS/project/results/grail_deepsomatic/scripts")
sys.path.insert(0, "/home/jrkim/TSO_TFBS/project/scripts")
PROJ = "/home/jrkim/TSO_TFBS/project"
GD = f"{PROJ}/results/grail_deepsomatic"
OUT = f"{PROJ}/results/auto_plan/grail/feat"; os.makedirs(OUT, exist_ok=True)
PY = "/home/jrkim/.conda/envs/cfse/bin/python"
def log(m): print(m, flush=True)


def build_cna():
    out = f"{PROJ}/results/auto_plan/grail/X_CNA_grail.npz"
    env = dict(os.environ, CNA_OUT=f"{PROJ}/results/auto_plan/grail/cna",
               CNA_ICHOR=f"{PROJ}/results/auto_plan/grail/ichorCNA",
               CNA_MAN=f"{PROJ}/results/auto_plan/grail/manifest.tsv")
    try:
        subprocess.run([PY, f"{PROJ}/scripts/v2_cna_features.py"], env=env, check=False)
        src = f"{PROJ}/results/auto_plan/grail/cna/X_CNA.npz"
        if os.path.exists(src):
            z = np.load(src, allow_pickle=True)
            np.savez(f"{OUT}/X_Genome_wide_CNA.npz", X=z["X"].astype(np.float32),
                     sids=np.array([str(s) for s in z["sids"]]), cols=np.array([str(c) for c in z["cols"]]))
            log(f"[grail-feat] CNA {z['X'].shape}")
    except Exception as e:
        log(f"[grail-feat] CNA failed: {e}")


def build_mutation():
    """mutation profile (gene flags) + signature (96-channel) from GRAIL calls."""
    try:
        import gdd_signatures as GS
    except Exception as e:
        log(f"[grail-feat] gdd_signatures import failed: {e}"); GS = None
    files = glob.glob(f"{GD}/calls/*/raw.tsv") + glob.glob(f"{GD}/calls_tumoronly/*/raw.tsv")
    if not files: log("[grail-feat] no GRAIL calls yet -> skip mutation features"); return
    sig_rows = {}; mut_genes = collections.defaultdict(set); sids = []
    for f in files:
        sid = os.path.basename(os.path.dirname(f)); sids.append(sid)
        df = pd.read_csv(f, sep="\t", dtype=str)
        # somatic-ish: PASS filter, plasma alt>=3, not flagged GDNA (germline/CH) -> tumor-derived candidates
        df = df[df["filter"].fillna("").str.contains("PASS") & (df["adnobaq"].astype(float) >= 3) &
                ~df["filter"].fillna("").str.contains("GDNA")]
        for g, ns in zip(df["gene"].fillna(""), df["is_nonsyn"].astype(int)):
            if g and ns == 1:
                for gg in str(g).split(";"): mut_genes[sid].add(gg.strip())
        # signature: write an adapted trace then call GS.spectrum (reuse) -- here build 96 directly
        if GS is not None:
            try:
                tr = f"{GD}/_grailtrace_{sid}.tsv"
                sub = df[["chrom", "pos", "ref", "alt"]].copy()
                sub.columns = ["Chromosome", "Position", "RefCall", "AltCall"]; sub["Status"] = "Somatic"
                sub.to_csv(tr, sep="\t", index=False)
                feat = GS.sample_features(tr); os.remove(tr)
                sig_rows[sid] = feat
            except Exception:
                sig_rows[sid] = {}
    # mutation profile matrix (genes seen in >=3 samples)
    cnt = collections.Counter(g for s in mut_genes for g in mut_genes[s])
    genes = sorted([g for g, c in cnt.items() if c >= 3])
    gi = {g: i for i, g in enumerate(genes)}
    Xm = np.zeros((len(sids), len(genes)), np.float32)
    for r, s in enumerate(sids):
        for g in mut_genes[s]:
            if g in gi: Xm[r, gi[g]] = 1
    np.savez(f"{OUT}/X_Somatic_mutation_profile.npz", X=Xm, sids=np.array(sids),
             cols=np.array([f"mut_{g}" for g in genes]))
    log(f"[grail-feat] mutation profile {Xm.shape}")
    if sig_rows:
        keys = sorted({k for d in sig_rows.values() for k in d if k.startswith(("sigexp_", "sigfrac_", "sigflag_"))})
        Xs = np.array([[float(sig_rows[s].get(k, 0) or 0) for k in keys] for s in sids], np.float32)
        np.savez(f"{OUT}/X_Mutation_signature.npz", X=Xs, sids=np.array(sids), cols=np.array(keys))
        log(f"[grail-feat] signature {Xs.shape}")


def main():
    build_cna()
    build_mutation()
    log("[grail-feat] DONE (portable classes). Fragmentomic E1/depth/SHAPE deferred (panel-overlap).")


if __name__ == "__main__":
    main()
