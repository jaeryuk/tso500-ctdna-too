#!/usr/bin/env python
"""Benchmark endpoint-distribution metrics + length-entropy for TOO, all on the SAME frame:
TFBS window ±40 / 5bp, edge0-UNIQUE 17,178 sites, within-sample robust-z, elastic-net 20×5 nested CV, v1+v2.
Individual metrics (one value per site):
  gini_ineq   Lorenz endpoint-Gini INEQUALITY (order-stat)   -> reproduces padgini-uniq ±40/b5
  gini_impur  Simpson endpoint-Gini IMPURITY  = 1 - Σpᵢ²
  HHI         Herfindahl                     = Σpᵢ²   (= 1 - gini_impur; robust-z ⇒ same AUROC)
  conc_1mHn   endpoint entropy concentration = 1 - H/ln(B)
  spatial_S   SIGNED spatial index           = (1-H/ln B)·(D - D_uniform)   S>0 edge-mass, S<0 center-mass
  lenent      FRAGMENT-LENGTH entropy (bits) of 30-420bp/5bp lengths, fragments centred within ±40bp
Then EARLY-FUSION cell (feature-concatenation of robust-z blocks, elastic net):
  fuse_best5_len = (best-AUROC of the 5 endpoint metrics) ⊕ lenent   = 2 blocks ≈ 34k features
Out: results/plot/Tab_conc_metrics_bench.tsv, Fig_conc_metrics_bench.png
"""
import os, sys, glob
os.environ.setdefault("MCT_XGB_DEVICE", "cpu")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/jrkim/TSO_TFBS/project/scripts/auto/nc_readiness")
import p_model_comparison_tuned as MCT
from ncr_common import macro_auc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PROJ = "/home/jrkim/TSO_TFBS/project"; PLOT = f"{PROJ}/results/plot"
NPZD = f"{PROJ}/results/auto_plan/center_vs_pad/npz"; MANIFEST = f"{PROJ}/results/auto_plan/manifest_dev.tsv"
SCR = "/tmp/claude-1002/-home-jrkim/5ebb3ac4-50d5-4c66-a4d7-ce4c72a5252e/scratchpad/gw_metagene"
REG = f"{SCR}/centers_edge0_regions.tsv"; FEAT = f"{PROJ}/results/auto_plan/feat"
MINCLASS = 20
NCORE = max(1, int(os.cpu_count() * 0.80)); MCT.NJOBS = {k: NCORE for k in MCT.NJOBS}
MK = MCT.MKEY["A: Elastic net"]
METRICS = [("gini_ineq", "Gini inequality (Lorenz)", lambda z: z["g_pad"]),
           ("gini_impur", "Gini impurity (1-Σp²)",   lambda z: 1.0 - z["m2"]),
           ("HHI",        "HHI (Σp²)",                lambda z: z["m2"]),
           ("conc_1mHn",  "1 - H_norm (endpoint conc)", lambda z: z["m1"]),
           ("spatial_S",  "Signed spatial S",         lambda z: z["m3"]),
           ("lenent",     "Length entropy ±40 (30-420/5bp)", lambda z: z["lenent"])]
ENDPOINT5 = ["gini_ineq", "gini_impur", "HHI", "conc_1mHn", "spatial_S"]     # the "5 metrics currently running"
FN = {k: fn for k, _, fn in METRICS}


def unique_mask():
    # X_SHAPE.npz cols may carry a "<prefix>:" tag (e.g. g:/l: since the padding-gini⊕lenent repin);
    # strip it and dedup so this recovers the identical 17,178 unique-site mask for old & new formats.
    z = np.load(f"{FEAT}/v1/X_SHAPE.npz", allow_pickle=True); S = set(str(c).split(":", 1)[-1] for c in z["cols"])
    reg = [ln.rstrip("\n").split("\t") for ln in open(REG)]
    cid = [f"{r[0]}_{(int(r[1]) + int(r[2])) // 2}" for r in reg]
    m = np.array([c in S for c in cid]); assert m.sum() == len(S), f"{m.sum()}!={len(S)}"
    return m


def cohort_data(cohort, have):
    man = pd.read_csv(MANIFEST, sep="\t", dtype=str); man = man[(man.cohort == cohort) & (man.sid.isin(have))]
    vc = man.cancer_type.value_counts(); man = man[man.cancer_type.isin(vc[vc >= MINCLASS].index)].reset_index(drop=True)
    classes = np.array(sorted(man.cancer_type.unique()))
    y = np.array([{c: i for i, c in enumerate(classes)}[c] for c in man.cancer_type.values], dtype=int)
    return man.sid.tolist(), y, classes


def preprocess(X):
    keep = np.isnan(X).mean(0) <= 0.5; X = X[:, keep]
    med = np.nanmedian(X, 0); med = np.where(np.isfinite(med), med, 0.0)
    ii = np.where(np.isnan(X)); X[ii] = np.take(med, ii[1])
    m = np.median(X, 1, keepdims=True)
    q1 = np.percentile(X, 25, 1, keepdims=True); q3 = np.percentile(X, 75, 1, keepdims=True)
    iqr = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)
    return ((X - m) / iqr).astype(np.float32)


def load_raw(sids, mask, fn):
    n = int(mask.sum()); X = np.empty((len(sids), n), np.float32)
    for i, s in enumerate(sids):
        X[i] = np.asarray(fn(np.load(f"{NPZD}/{s}.cvp.npz")), np.float32)[mask]
    return X                                       # raw per-site (NaN kept)


def load_metric(sids, mask, fn):
    return preprocess(load_raw(sids, mask, fn))


def score(cohort, name, X, y, K):
    oof, hp = MCT.process_cell(cohort, name, MK, X, y, K)
    return MCT.summarize(oof, y, K)


def main():
    have = set(os.path.basename(p)[:-len(".cvp.npz")] for p in glob.glob(f"{NPZD}/*.cvp.npz"))
    mask = unique_mask()
    print(f"cvp npz={len(have)} unique sites={int(mask.sum())} NCORE={NCORE}", flush=True)
    rows = []; Xcache = {}
    # ---- pass 1: individual metrics (cache the robust-z matrices for fusion) ----
    for cohort in ("v1", "v2"):
        sids, y, classes = cohort_data(cohort, have); K = len(classes)
        print(f"\n=== {cohort}: n={len(sids)} K={K} ===", flush=True)
        Xcache[cohort] = {"_sids": sids, "_y": y, "_K": K}
        for key, label, fn in METRICS:
            X = load_metric(sids, mask, fn); Xcache[cohort][key] = X
            s = score(cohort, f"conc_{key}", X, y, K)
            rows.append(dict(cohort=cohort, kind="single", metric=key, label=label, n_features=X.shape[1], **s))
            pd.DataFrame(rows).to_csv(f"{PLOT}/Tab_conc_metrics_bench.tsv", sep="\t", index=False)
            print(f"  {label:32s} mAUROC={s['macroAUROC']:.4f} [{s['macroAUROC_lo']:.4f},{s['macroAUROC_hi']:.4f}]", flush=True)

    # ---- decide best performers (mean AUROC across cohorts) ----
    df = pd.DataFrame(rows)
    meanauc = df[df.kind == "single"].groupby("metric").macroAUROC.mean()
    best5 = meanauc[ENDPOINT5].idxmax()
    print(f"\nbest-of-5 endpoint metric = {best5}", flush=True)
    print(f"mean AUROC per metric:\n{meanauc.sort_values(ascending=False).round(4).to_string()}", flush=True)

    # ---- pass 2: early-fusion cells ----
    # fuse_best5_len   = best5 ⊕ lenent                             (main effects, ~34k)
    # fuse_best5_len_x = best5 ⊕ lenent ⊕ (best5 × lenent per-site) (main effects + per-TFBS interaction, ~51k)
    for cohort in ("v1", "v2"):
        C = Xcache[cohort]; sids = C["_sids"]; y = C["_y"]; K = C["_K"]
        inter = preprocess(load_raw(sids, mask, FN[best5]) * load_raw(sids, mask, FN["lenent"]))  # per-site interaction, robust-z
        specs = {"fuse_best5_len":   ([C[best5], C["lenent"]],        f"{best5}+lenent"),
                 "fuse_best5_len_x": ([C[best5], C["lenent"], inter], f"{best5}+lenent+({best5}×lenent)")}
        for fname, (blocks, desc) in specs.items():
            X = np.hstack(blocks).astype(np.float32)
            s = score(cohort, f"{fname}", X, y, K)
            rows.append(dict(cohort=cohort, kind="fusion", metric=fname, label=f"{fname}={desc}",
                             n_features=X.shape[1], **s))
            pd.DataFrame(rows).to_csv(f"{PLOT}/Tab_conc_metrics_bench.tsv", sep="\t", index=False)
            print(f"  [{cohort}] {fname} ({desc}) feats={X.shape[1]} "
                  f"mAUROC={s['macroAUROC']:.4f} [{s['macroAUROC_lo']:.4f},{s['macroAUROC_hi']:.4f}]", flush=True)
    fusions = {"fuse_best5_len": 1, "fuse_best5_len_x": 1}          # for the figure ordering below

    df = pd.DataFrame(rows); df.to_csv(f"{PLOT}/Tab_conc_metrics_bench.tsv", sep="\t", index=False)
    print("\n===== SUMMARY macro-AUROC =====")
    print(df.pivot_table(index=["kind", "metric"], columns="cohort", values="macroAUROC").round(4).to_string())

    # figure: singles then fusions, both cohorts
    order = [m[0] for m in METRICS] + list(fusions)
    lab = {m[0]: m[1] for m in METRICS}; lab.update({k: k for k in fusions})
    fig, ax = plt.subplots(figsize=(11, 5.4)); x = np.arange(len(order)); w = 0.38
    for k, c in enumerate(("v1", "v2")):
        vals = [float(df[(df.cohort == c) & (df.metric == o)].macroAUROC.iloc[0]) if
                len(df[(df.cohort == c) & (df.metric == o)]) else np.nan for o in order]
        ax.bar(x + (k - 0.5) * w, vals, w, label=c.upper(), color=["#4C72B0", "#DD8452"][k])
        for xi, mi in zip(x + (k - 0.5) * w, vals):
            if np.isfinite(mi): ax.text(xi, mi + 0.004, f"{mi:.3f}", ha="center", fontsize=7)
    ax.axhline(0.80, ls="--", c="grey", lw=0.8); ax.axvline(len(METRICS) - 0.5, ls=":", c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([lab[o] for o in order], rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("macro-AUROC"); ax.set_ylim(0.55, 0.85); ax.legend(frameon=False)
    ax.set_title("Endpoint metrics + length-entropy + early fusion — ±40/5bp, unique 17,178, robust-z, enet")
    fig.tight_layout(); out = f"{PLOT}/Fig_conc_metrics_bench.png"; fig.savefig(out, dpi=170); plt.close(fig)
    print(f"\nsaved -> {out} + Tab_conc_metrics_bench.tsv")


if __name__ == "__main__":
    main()
