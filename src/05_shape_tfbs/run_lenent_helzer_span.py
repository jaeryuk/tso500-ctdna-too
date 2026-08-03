#!/usr/bin/env python3
"""Parallel driver: extract_lenent_helzer_span.py per sample for a cohort, over the MOTIF-EDGE window
(regions = centers_edge0_regions.tsv). Resumable. v2 uses manifest_bamfix.tsv (TSO-renamed BAM paths).
Usage: run_lenent_helzer_span.py <cohort> <NP>"""
import os, sys, subprocess
from joblib import Parallel, delayed
PROJ = "/home/jrkim/TSO_TFBS/project"
PY = "/home/jrkim/.conda/envs/cfse/bin/python"
EXTRACT = f"{PROJ}/scripts/auto/nc_readiness/extract_lenent_helzer_span.py"
SCR = "/tmp/claude-1002/-home-jrkim/5ebb3ac4-50d5-4c66-a4d7-ce4c72a5252e/scratchpad"
REG = f"{SCR}/gw_metagene/centers_edge0_regions.tsv"     # motif spans (start/end)


def load_manifest(p):
    sid2bam = {}
    with open(p) as f:
        hdr = f.readline().rstrip("\n").split("\t"); sc = hdr.index("sid"); bc = hdr.index("bam")
        for ln in f:
            r = ln.rstrip("\n").split("\t")
            if len(r) > max(sc, bc): sid2bam[r[sc]] = r[bc]
    return sid2bam


def main():
    coh, NP = sys.argv[1], int(sys.argv[2])
    outdir = f"{SCR}/lenhzspan_{coh}"; os.makedirs(outdir, exist_ok=True)
    fix = f"{PROJ}/results/{coh}_ponbench/manifest_bamfix.tsv"
    man = fix if os.path.exists(fix) else f"{PROJ}/results/{coh}_ponbench/manifest.tsv"
    print(f"[{coh}] manifest = {man}", flush=True)
    sid2bam = load_manifest(man)
    sids = [s.strip() for s in open(f"{PROJ}/results/{coh}_e1se/sample_list.txt") if s.strip()]
    todo = [s for s in sids if s in sid2bam and not os.path.exists(f"{outdir}/{s}.lenhz.npz")]
    print(f"[{coh}] {len(sids)} samples, {len(todo)} to extract (NP={NP})", flush=True)

    def run(sid):
        bam = sid2bam[sid]
        if not os.path.exists(bam): return f"MISS_BAM {sid}"
        r = subprocess.run([PY, EXTRACT, sid, bam, REG, outdir], capture_output=True, text=True)
        return f"{sid} rc={r.returncode}"

    Parallel(n_jobs=NP, backend="loky")(delayed(run)(s) for s in todo)
    done = len([1 for _ in os.listdir(outdir) if _.endswith(".lenhz.npz")])
    print(f"[{coh}] extraction complete: {done}/{len(sids)} npz present", flush=True)


if __name__ == "__main__":
    main()
