#!/usr/bin/env python
"""Surgical rebuild of ONLY feat_rc/{v1,v2}/X_E1_entropy.npz from the nearest-exon nc E1SE store
(results/{coh}_nc_e1se/X_E1SE.npz), replacing the old exon1-only definition.

Reuses rc_assemble.copy_npz (row-subset to dev sids + float32, cols passthrough) — the SAME transform
that built the old store — so the output schema (X/sids/cols, float32) is byte-compatible; only the gene
column set legitimately grows 362 -> 519 (old genes are a strict subset). Touches no other module.
Do NOT run rc_assemble.main() for this: its E1 source is the OLD rcv{V}_e1se store."""
import sys, numpy as np
sys.path.insert(0, "/home/jrkim/TSO_TFBS/project/scripts/auto")
from rc_assemble import copy_npz, dev_sids, OUT   # OUT = .../results/auto_plan/feat_rc

for coh in ("v1", "v2"):
    src = f"/home/jrkim/TSO_TFBS/project/results/{coh}_nc_e1se/X_E1SE.npz"   # NEAREST-EXON source
    dst = f"{OUT}/{coh}/X_E1_entropy.npz"
    n, p = copy_npz(src, dst, dev_sids(coh))
    print(f"[{coh}] E1_entropy <- {coh}_nc_e1se  kept {n} sids x {p} genes -> {dst}", flush=True)
print("REBUILD_E1_NC_DONE")
