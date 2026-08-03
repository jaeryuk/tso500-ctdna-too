#!/usr/bin/env python
"""
RC pipeline — assemble the 5 benchmark modules into results/auto_plan/feat_rc/{v1,v2}/ (numeric-keyed),
isolated from the shared feat/ tree. Modules (matching the user's learner spec):
  E1_entropy               <- results/rcv{V}_e1se/X_E1SE.npz            (copy; assembled by helzer_e1se.py)
  SHAPE                    <- results/rcv{V}_fragshape/X_SHAPE.npz      (copy; assembled by v2_frag_shape_features.py)
  All_exon_depth           <- results/metaplots/_helzer_metrics_rcv{V}/*.npz  (built here, = phase2 logic)
  Somatic_mutation_profile <- results/rcv{V}_gdd/_raw/*.json            (built here from _mut/_trunc, genes >=3 samples)
  Genome_wide_CNA          <- results/rcv{V}_cna/X_CNA.npz              (copy; re-derived from current ichorCNA)
Only samples present in the rule manifest_dev for that cohort are kept.
"""
import os, glob, json
import numpy as np

PROJ = "/home/jrkim/TSO_TFBS/project"
RC = f"{PROJ}/results/rule_conformant"
OUT = f"{PROJ}/results/auto_plan/feat_rc"
MINF = 3   # keep mutated/truncated genes seen in >= MINF samples (matches gdd_features.py)


def dev_sids(cohort):
    S = set()
    for ln in open(f"{RC}/manifest_dev.tsv").read().splitlines()[1:]:
        c = ln.split("\t")
        if c[1] == cohort:
            S.add(c[0])
    return S


def copy_npz(src, dst, keep):
    z = np.load(src, allow_pickle=True)
    sids = [str(s) for s in z["sids"]]
    idx = [i for i, s in enumerate(sids) if s in keep]
    cols = [str(c) for c in z["cols"]] if "cols" in z.files else [f"f{i}" for i in range(z["X"].shape[1])]
    np.savez(dst, X=z["X"][idx].astype(np.float32),
             sids=np.array([sids[i] for i in idx]), cols=np.array(cols))
    return len(idx), z["X"].shape[1]


def build_exon_depth(tag, keep):
    files = sorted(glob.glob(f"{PROJ}/results/metaplots/_helzer_metrics_{tag}/*.npz"))
    z0 = np.load(files[0], allow_pickle=True)
    names = np.array([str(x) for x in z0["exon_name"]])[z0["is_exon"]]
    rows, sids = [], []
    for f in files:
        sid = os.path.basename(f)[:-4]
        if sid not in keep:
            continue
        z = np.load(f, allow_pickle=True)
        d = z["depth"][z["is_exon"]].astype(float)
        med = np.nanmedian(d[d > 0]) if np.any(d > 0) else 1.0
        rows.append(np.log2((d + 1e-3) / (med + 1e-3))); sids.append(sid)
    X = np.vstack(rows).astype(np.float32)
    finite = np.isfinite(X); k = finite.mean(0) >= 0.95
    X = X[:, k]; cols = names[k]
    cm = np.nanmedian(np.where(np.isfinite(X), X, np.nan), 0)
    bad = ~np.isfinite(X); X[bad] = np.take(np.nan_to_num(cm), np.where(bad)[1])
    return X, sids, [f"exon:{c}" for c in cols]


def build_mut_profile(tag, keep):
    files = sorted(glob.glob(f"{PROJ}/results/{tag}_gdd/_raw/*.json"))
    raw = {}
    for f in files:
        sid = os.path.basename(f)[:-5]
        if sid in keep:
            raw[sid] = json.load(open(f))
    sids = sorted(raw)
    def freq(key):
        c = {}
        for s in sids:
            for g in raw[s].get(key, []) or []:
                c[g] = c.get(g, 0) + 1
        return sorted([g for g, n in c.items() if n >= MINF])
    mg, tg = freq("_mut"), freq("_trunc")
    cols = [f"mut_{g}" for g in mg] + [f"trunc_{g}" for g in tg]
    ci = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(sids), len(cols)), np.float32)
    for r, s in enumerate(sids):
        for g in raw[s].get("_mut", []) or []:
            if f"mut_{g}" in ci: X[r, ci[f"mut_{g}"]] = 1
        for g in raw[s].get("_trunc", []) or []:
            if f"trunc_{g}" in ci: X[r, ci[f"trunc_{g}"]] = 1
    return X, sids, cols


def main():
    for cohort in ["v1", "v2"]:
        V = cohort[-1]; tag = f"rcv{V}"; od = f"{OUT}/{cohort}"; os.makedirs(od, exist_ok=True)
        keep = dev_sids(cohort)
        n, p = copy_npz(f"{PROJ}/results/{tag}_e1se/X_E1SE.npz", f"{od}/X_E1_entropy.npz", keep)
        print(f"[{cohort}] E1_entropy   {n}x{p}->kept")
        n, p = copy_npz(f"{PROJ}/results/{tag}_fragshape/X_SHAPE.npz", f"{od}/X_SHAPE.npz", keep)
        print(f"[{cohort}] SHAPE        {n} kept")
        n, p = copy_npz(f"{PROJ}/results/{tag}_cna/X_CNA.npz", f"{od}/X_Genome_wide_CNA.npz", keep)
        print(f"[{cohort}] CNA          {n} kept")
        X, sids, cols = build_exon_depth(tag, keep)
        np.savez(f"{od}/X_All_exon_depth.npz", X=X, sids=np.array(sids), cols=np.array(cols))
        print(f"[{cohort}] All_exon_depth {X.shape}")
        X, sids, cols = build_mut_profile(tag, keep)
        np.savez(f"{od}/X_Somatic_mutation_profile.npz", X=X, sids=np.array(sids), cols=np.array(cols))
        print(f"[{cohort}] Somatic_mutation_profile {X.shape}")
    print(f"[rc_assemble] DONE -> {OUT}")


if __name__ == "__main__":
    main()
