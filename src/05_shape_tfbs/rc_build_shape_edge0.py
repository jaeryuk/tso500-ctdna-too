#!/usr/bin/env python
"""Standing SHAPE for the RC cohort at the 26,845 EDGE0 UniBind site set (user 2026-07-08 switch):
  SHAPE = gpad40 gini (motif-region ±40, 5bp, residualised) ⊕ lenent_helzerspan (motif±40, residualised)
  over the EDGE0 sites (~26.8k), NOT the 17,178 gpad40 subset.

GINI  — REBUILT for ALL RC samples from padgini (native 26,845) under one RC-wide block() (user directive
        "rebuild all 1889 gini rows for consistency"). P=40 arm = ginipad[:, ipad=6].
LENENT— reuse VERBATIM from the old-cohort 26,845 matrix results/auto_plan/feat/{v}/X_SHAPE_edge0.npz (l: cols,
        already resid+robustz — cohort-portable) for samples present there; FRESH samples residualised from lenhz.
Canonical sites = edge0 sites present in BOTH gini(g:) and lenent(l:) of X_SHAPE_edge0 AND in padgini (chr_center key).
Out: results/auto_plan/feat_rc/{v}/X_SHAPE.npz  (gr:*Nsite + lenhzsp:*Nsite).
"""
import os, sys, numpy as np
sys.path.insert(0, "/home/jrkim/TSO_TFBS/project/scripts/auto/nc_readiness")
from build_lenent_b5_w40 import resid_logsupport, block

PROJ = "/home/jrkim/TSO_TFBS/project"
FEATOLD = f"{PROJ}/results/auto_plan/feat"          # old-cohort matrices (X_SHAPE_edge0)
OUT = f"{PROJ}/results/auto_plan/feat_rc"
RC = f"{PROJ}/results/rule_conformant"
PADD = f"{PROJ}/results/auto_plan/padgini/npz"
# LENENT_SRC: 'mixed' (default, current) = reuse old X_SHAPE_edge0 l: + fresh lenhz (center±40, mixed recipe);
#             'edge'  = UNIFORM edge-anchored [start-40,end+40] lenent for ALL samples from lenedge_span (no reuse).
LENENT_SRC = os.environ.get("LENENT_SRC", "mixed")
LHD = f"{PROJ}/results/auto_plan/lenhz_span/npz" if LENENT_SRC != "edge" else f"{PROJ}/results/auto_plan/lenedge_span/npz"
LSUF = "lenhz" if LENENT_SRC != "edge" else "lenedge"
REG = f"{RC}/centers_edge0_regions.tsv"
IPAD = 6


def robustz_row(v):
    med = np.nanmedian(v); mad = np.nanmedian(np.abs(v - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        sd = np.nanstd(v); mad = (sd if np.isfinite(sd) else 0.0) + 1e-9
    return np.nan_to_num((v - med) / mad).astype(np.float32)


def rc_sids(cohort):
    return [l.split("\t")[0] for l in open(f"{RC}/manifest_dev.tsv").read().splitlines()[1:]
            if l.split("\t")[1] == cohort]


def reg_name2idx():
    d = {}
    for i, ln in enumerate(open(REG)):
        r = ln.rstrip("\n").split("\t")
        d[f"{r[0]}_{(int(r[1]) + int(r[2])) // 2}"] = i
    return d


def build(cohort):
    e0 = np.load(f"{FEATOLD}/{cohort}/X_SHAPE_edge0.npz", allow_pickle=True)
    cols = [str(c) for c in e0["cols"]]
    gcol = {c[2:]: j for j, c in enumerate(cols) if c.startswith("g:")}
    lcol = {c[2:]: j for j, c in enumerate(cols) if c.startswith("l:")}
    e0sid = {str(s): i for i, s in enumerate(e0["sids"])}
    reg = reg_name2idx()
    sites = sorted(set(gcol) & set(lcol) & set(reg))          # canonical edge0 sites w/ gini+lenent+padgini
    cvp_idx = np.array([reg[s] for s in sites], np.int64)      # site -> padgini row
    l_from_e0 = np.array([lcol[s] for s in sites], np.int64)   # site -> e0 lenent col
    ncol = len(sites); sids = rc_sids(cohort)
    print(f"[{cohort}] canonical edge0 sites={ncol} (g∩l∩padgini)  RC n={len(sids)}", flush=True)

    # ---- GINI: rebuild EVERY RC row from padgini, then one RC-wide block() ----
    def gini_resid(sid):
        p = f"{PADD}/{sid}.padgini.npz"
        if not os.path.exists(p): return None
        z = np.load(p, allow_pickle=True)
        return resid_logsupport(z["ginipad"][:, IPAD].astype(float)[cvp_idx],
                                z["nep"][:, IPAD].astype(float)[cvp_idx])
    grows = [gini_resid(s) for s in sids]
    gmiss = [sids[i] for i, r in enumerate(grows) if r is None]
    if gmiss: print(f"[{cohort}] WARN gini padgini missing {len(gmiss)}: {gmiss[:5]}", flush=True)
    grows = [r if r is not None else np.full(ncol, np.nan) for r in grows]
    Xg = block(grows)

    # ---- LENENT: reuse verbatim from X_SHAPE_edge0; fresh residualised from lenhz ----
    Xl = np.zeros((len(sids), ncol), np.float32); reused = fresh = miss = 0
    for r, sid in enumerate(sids):
        if LENENT_SRC != "edge" and sid in e0sid:                 # edge mode = UNIFORM, no old-cohort reuse
            Xl[r] = e0["X"][e0sid[sid]][l_from_e0]; reused += 1; continue
        p = f"{LHD}/{sid}.{LSUF}.npz"                              # .lenhz.npz (mixed) or .lenedge.npz (edge)
        if not os.path.exists(p): miss += 1; continue
        fresh += 1
        z = np.load(p, allow_pickle=True); ch = [str(x) for x in z["chrom"]]; po = z["pos"]
        nidx = {f"{ch[i]}_{int(po[i])}": i for i in range(len(ch))}
        cmap = np.array([nidx.get(s, -1) for s in sites], np.int64); valid = cmap >= 0; cmap[~valid] = 0
        vl = z["lenent"].astype(float)[cmap]; sl = z["nfr"].astype(float)[cmap]
        vl[~valid] = np.nan; sl[~valid] = 0
        Xl[r] = robustz_row(resid_logsupport(vl, sl))

    X = np.hstack([Xg, Xl]).astype(np.float32)
    out_cols = [f"gr:{s}" for s in sites] + [f"lenhzsp:{s}" for s in sites]
    os.makedirs(f"{OUT}/{cohort}", exist_ok=True)
    np.savez(f"{OUT}/{cohort}/X_SHAPE.npz", X=X, sids=np.array(sids), cols=np.array(out_cols))
    print(f"[{cohort}] X_SHAPE {X.shape}  sites={ncol}  gini=ALL-{len(sids)}-rebuilt(miss={len(gmiss)})  "
          f"lenent=reuse{reused}+fresh{fresh}(miss{miss})  std={X.std():.3f}", flush=True)


if __name__ == "__main__":
    for c in ("v1", "v2"):
        build(c)
    print("RC_SHAPE_EDGE0_DONE")
