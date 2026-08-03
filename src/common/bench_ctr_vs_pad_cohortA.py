#!/usr/bin/env python
"""CENTER ±40 vs PADDING ±40 gini inequality (5bp bins) on the MANUSCRIPT cohort
(Fig2A / blood_nested2): blood _B_01 ∩ all-5-modules present ∩ class>=20  -> n=1034(v1)/708(v2).
Same 17,178 unique sites, same within-sample robust-z, same elastic-net 20x5 nested CV.
Also the ⊕ lenent fusion for each window."""
import os, re, glob, numpy as np, pandas as pd
from collections import Counter
import build_conc_metrics_bench as B

MAN = f"{B.PROJ}/results/auto_plan/manifest_dev.tsv"
FEAT = f"{B.PROJ}/results/auto_plan/feat"
RENAME = "/data/TSO500/id_rename_success.log"
METHODS = ["E1_entropy", "All_exon_depth", "Genome_wide_CNA", "Somatic_mutation_profile", "SHAPE"]
MIN_CLASS = 20

def load_n2tso():
    n2tso = {}
    for ln in open(RENAME):
        for m in re.finditer(r'\b(\d{10})->(TSO_\d+_[A-Z]_\d+)', ln): n2tso[m.group(1)] = m.group(2)
    return n2tso

def basenum(s):
    m = re.match(r'(\d{10})', str(s)); return m.group(1) if m else None

def cohort_sids(cohort, n2tso):
    man = pd.read_csv(MAN, sep="\t", dtype=str); man = man[man.cohort == cohort]
    lab = dict(zip(man.sid, man.cancer_type))
    elig = {s for s in man.sid if n2tso.get(basenum(s), "").endswith("_B_01") and lab.get(s)}
    common = set(elig)
    for m in METHODS:
        z = np.load(f"{FEAT}/{cohort}/X_{m}.npz", allow_pickle=True)
        common &= set(str(s) for s in z["sids"])
    common = sorted(common)
    cnt = Counter(lab[s] for s in common); classes = sorted([c for c in cnt if cnt[c] >= MIN_CLASS])
    common = [s for s in common if lab[s] in classes]
    y = np.array([classes.index(lab[s]) for s in common], dtype=int)
    return common, y, classes

def main():
    have = set(os.path.basename(p)[:-len(".cvp.npz")] for p in glob.glob(f"{B.NPZD}/*.cvp.npz"))
    mask = B.unique_mask(); n2tso = load_n2tso()
    print(f"cvp npz={len(have)} unique sites={int(mask.sum())} NCORE={B.NCORE}  MANUSCRIPT cohort", flush=True)
    G_CTR = lambda z: z["g_ctr"]; G_PAD = lambda z: z["g_pad"]; LEN = lambda z: z["lenent"]
    rows = []
    for cohort in ("v1", "v2"):
        sids, y, classes = cohort_sids(cohort, n2tso); K = len(classes)
        miss = [s for s in sids if s not in have]
        assert not miss, f"{cohort}: {len(miss)} cohort sids lack cvp npz: {miss[:5]}"
        print(f"\n=== {cohort}: n={len(sids)} K={K}  (manuscript blood_B01 ∩ 5-module ∩ class≥20) ===", flush=True)
        ctr = B.preprocess(B.load_raw(sids, mask, G_CTR))
        pad = B.preprocess(B.load_raw(sids, mask, G_PAD))
        ln  = B.preprocess(B.load_raw(sids, mask, LEN))
        # per-TFBS interaction = robust-z of (raw gini * raw lenent) at the same site
        inter_ctr = B.preprocess(B.load_raw(sids, mask, G_CTR) * B.load_raw(sids, mask, LEN))
        inter_pad = B.preprocess(B.load_raw(sids, mask, G_PAD) * B.load_raw(sids, mask, LEN))
        cells = [
            ("fusion", "center_lenent",    "center gini ⊕ lenent",              np.hstack([ctr, ln])),
            ("fusion", "padding_lenent",   "padding gini ⊕ lenent",             np.hstack([pad, ln])),
            ("fusion", "center_lenent_x",  "center gini ⊕ lenent ⊕ interaction",  np.hstack([ctr, ln, inter_ctr])),
            ("fusion", "padding_lenent_x", "padding gini ⊕ lenent ⊕ interaction", np.hstack([pad, ln, inter_pad])),
        ]
        for kind, key, label, X in cells:
            X = X.astype(np.float32)
            s = B.score(cohort, f"{key}_{cohort}", X, y, K)
            rows.append(dict(cohort=cohort, kind=kind, metric=key, label=label, n_features=X.shape[1], **s))
            pd.DataFrame(rows).to_csv(f"{B.PLOT}/Tab_ctr_vs_pad_cohortA.tsv", sep="\t", index=False)
            print(f"  {label:30s} feats={X.shape[1]:5d}  mAUROC={s['macroAUROC']:.4f} "
                  f"[{s['macroAUROC_lo']:.4f},{s['macroAUROC_hi']:.4f}]  top1={s['top1']:.4f} "
                  f"[{s['top1_lo']:.4f},{s['top1_hi']:.4f}]", flush=True)
    print("\nsaved -> results/plot/Tab_ctr_vs_pad_cohortA.tsv\nCTR_VS_PAD_A_DONE", flush=True)

if __name__ == "__main__":
    main()
