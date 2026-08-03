#!/usr/bin/env python3
"""Shared helper: resolve the rule-conformant (keep_t1) cohort to numeric feature-sids.

The rule-conformant manifests (results/rule_conformant/{v1,v2}_manifest.tsv) key samples by TSO id
(sid == tso_id, e.g. TSO_00048_B_01) with tumor_type1 labels. The feature matrices / fbins key
samples by 10-digit numeric sids. /data/TSO500/id_rename_success.log provides numeric->TSO id.

rc_table(cohort) -> DataFrame[num_sid, tso_id, cancer]  (keep_t1 rows that resolve to a numeric sid)
"""
import os, re
import pandas as pd

PROJ = "/home/jrkim/TSO_TFBS/project"
RENAME = "/data/TSO500/id_rename_success.log"
RC_MAN = {c: f"{PROJ}/results/rule_conformant/{c}_manifest.tsv" for c in ("v1", "v2")}


def tso2num():
    """TSO id -> 10-digit numeric sid (inverse of the rename log)."""
    m = {}
    for ln in open(RENAME):
        for g in re.finditer(r'\b(\d{10})->(TSO_\d+_[A-Z]_\d+)', ln):
            m[g.group(2)] = g.group(1)
    return m


def rc_table(cohort, label_col="tumor_type1", keep_col="keep_t1"):
    t2n = tso2num()
    man = pd.read_csv(RC_MAN[cohort], sep="\t", dtype=str)
    man = man[man[keep_col].astype(str).str.lower().isin(("true", "1"))].copy()
    man["num_sid"] = man["tso_id"].map(t2n)
    out = man.dropna(subset=["num_sid", label_col])
    out = out[out[label_col].astype(str).str.len() > 0]
    return out[["num_sid", "tso_id", label_col]].rename(columns={label_col: "cancer"}).reset_index(drop=True)


if __name__ == "__main__":
    import numpy as np
    for c in ("v1", "v2"):
        rc = rc_table(c)
        nset = set(rc.num_sid)
        # coverage in the feature stores each downstream figure needs
        e1 = np.load(f"{PROJ}/results/auto_plan/feat_rc/{c}/X_E1_entropy.npz", allow_pickle=True)
        cna = np.load(f"{PROJ}/results/auto_plan/feat/{c}/X_Genome_wide_CNA_armlevel.npz", allow_pickle=True)
        fb = {os.path.basename(p).split(".")[0]
              for p in os.listdir(f"{PROJ}/results/auto_plan/fbins")}
        e1s = set(str(s) for s in e1["sids"]); cnas = set(str(s) for s in cna["sids"])
        print(f"[{c}] keep_t1 resolved={len(rc)}  types={rc.cancer.nunique()}  "
              f"in_E1={len(nset & e1s)}  in_CNA={len(nset & cnas)}  in_fbins={len(nset & fb)}")
        print(f"     type counts: {rc.cancer.value_counts().to_dict()}")
