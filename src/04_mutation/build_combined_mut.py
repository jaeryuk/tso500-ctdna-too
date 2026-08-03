#!/usr/bin/env python
"""Redefine Somatic_mutation_profile = CVO∩trace-somatic gene-level (mut_/trunc_) + curated hotspots (hs_).
Concatenates X_Somatic_mutation_profile_cvo.npz + {coh}_curated_hotspot.npz (same sid order) and OVERWRITES
X_Somatic_mutation_profile.npz (backing up the prior version as .pre_cvo_hotspot.bak.npz)."""
import numpy as np, os, shutil
FEAT = "/home/jrkim/TSO_TFBS/project/results/auto_plan/feat_rc"
HOTD = "/home/jrkim/TSO_TFBS/project/results/rule_conformant/mut_hotspots"
TAG = os.environ.get("MUT_TAG", "")            # "" = overwrite definitive (with backup); "_nofilter" = write side file, no clobber
for coh in ("v1", "v2"):
    base = f"{FEAT}/{coh}/X_Somatic_mutation_profile{TAG}.npz"
    g = np.load(f"{FEAT}/{coh}/X_Somatic_mutation_profile_cvo{TAG}.npz", allow_pickle=True)
    h = np.load(f"{HOTD}/{coh}_curated_hotspot{TAG}.npz", allow_pickle=True)
    assert list(map(str, g["sids"])) == list(map(str, h["sids"])), f"[{coh}] sid order mismatch"
    if not TAG and not os.path.exists(f"{base}.pre_cvo_hotspot.bak.npz"): shutil.copy2(base, f"{base}.pre_cvo_hotspot.bak.npz")
    X = np.hstack([g["X"], h["X"]]).astype(np.float32)
    cols = [str(c) for c in g["cols"]] + [str(c) for c in h["cols"]]
    np.savez(base, X=X, sids=g["sids"], cols=np.array(cols))
    print(f"[{coh}] combined Somatic_mutation_profile = {g['X'].shape[1]} gene-level + {h['X'].shape[1]} curated-hotspot "
          f"= {X.shape[1]} cols  (n={X.shape[0]})", flush=True)
print("BUILD_COMBINED_MUT_DONE")
