#!/usr/bin/env python
"""Rebuild X_SHAPE_lenent_b5.npz from the WIDENED +-40bp length window (_amp_lenbins40), making +-40bp/5bp
the canonical SHAPE length component. IDENTICAL recipe to build_lenent_bins.py (support gate >=30 ->
residualise on log(support) -> per-cohort column-median impute -> within-sample robust-z); only the source
amp dir changes _amp_lenbins (+-30) -> _amp_lenbins40 (+-40). Backs up the existing +-30 b5 feature once.
"""
import os, sys, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np
from joblib import Parallel, delayed
PROJ = "/home/jrkim/TSO_TFBS/project"
sys.path.insert(0, f"{PROJ}/scripts/auto"); sys.path.insert(0, f"{PROJ}/scripts")
from nonblood_persite import site_geom, robustz_rows
FEAT = f"{PROJ}/results/auto_plan/feat"
AMP = {"v1": f"{PROJ}/results/v1_fragshape/_amp_lenbins40", "v2": f"{PROJ}/results/v2_fragshape/_amp_lenbins40"}
W = 5; MIN_SUPP = 30
def log(m): print(m, flush=True)


def resid_logsupport(val, supp):
    out = np.full_like(val, np.nan, dtype=np.float64)
    m = np.isfinite(val) & (supp >= MIN_SUPP) & (supp > 0)
    if m.sum() < 20:
        out[m] = val[m] - np.nanmean(val[m]) if m.any() else np.nan
        return out
    x = np.log(supp[m].astype(float)); yv = val[m].astype(float)
    b1, b0 = np.polyfit(x, yv, 1); out[m] = yv - (b0 + b1 * x); return out


def per_sample(sid, ampdir, rep):
    p = os.path.join(ampdir, sid + ".npz")
    if not os.path.exists(p): return None
    z = np.load(p); supp = z["n_len"].astype(float)[rep]
    return resid_logsupport(z[f"len_ent_b{W}"].astype(float)[rep], supp)


def block(rows):
    X = np.vstack(rows).astype(np.float32)
    cmed = np.nanmedian(X, 0); ii = np.where(~np.isfinite(X)); X[ii] = np.take(np.nan_to_num(cmed), ii[1])
    return np.nan_to_num(robustz_rows(X)).astype(np.float32)


def main():
    chrom, cen = site_geom(); site_idx = {}
    for i in range(len(chrom)):
        k = (chrom[i], int(cen[i]))
        if k not in site_idx: site_idx[k] = i
    bkp = f"{FEAT}/_bak_lenent_w30"; os.makedirs(bkp, exist_ok=True)
    for c in ("v1", "v2"):
        sh = np.load(f"{FEAT}/{c}/X_SHAPE.npz", allow_pickle=True)
        cols = [str(x) for x in sh["cols"]]; sids = [str(s) for s in sh["sids"]]
        rep = np.array([site_idx[(cc.rsplit("_", 1)[0], int(cc.rsplit("_", 1)[1]))] for cc in cols], np.int64)
        log(f"[{c}] {len(sids)} samples x {len(cols)} sites; mapping +-40bp len-entropy b5")
        rows = Parallel(n_jobs=48)(delayed(per_sample)(s, AMP[c], rep) for s in sids)
        miss = [sids[i] for i, r in enumerate(rows) if r is None]
        if miss: log(f"[{c}] WARNING {len(miss)} missing _amp_lenbins40 (NaN->impute): {miss[:5]}")
        rows = [r if r is not None else np.full(len(rep), np.nan) for r in rows]
        X = block(rows)
        out = f"{FEAT}/{c}/X_SHAPE_lenent_b{W}.npz"
        if os.path.exists(out) and not os.path.exists(f"{bkp}/{c}_X_SHAPE_lenent_b{W}_w30.npz"):
            shutil.copy2(out, f"{bkp}/{c}_X_SHAPE_lenent_b{W}_w30.npz"); log(f"[{c}] backed up +-30 b5 -> {bkp}")
        wcols = np.array([f"len{W}:{x}" for x in cols])
        np.savez_compressed(out, X=X, sids=np.array(sids), cols=wcols)
        log(f"[{c}] WROTE +-40bp b5 {X.shape}  std={X.std():.3f}  -> {out}")
    log("[done] X_SHAPE_lenent_b5.npz rebuilt at +-40bp for v1,v2")


if __name__ == "__main__":
    main()
