#!/usr/bin/env python
"""
Rule-Conformant (RC) pipeline — STEP 1: prepare manifests, seed caches, compute missing worklists.

Keying: NUMERIC analysis-id throughout (matches every existing per-sample cache + the benchmark join),
mapped from the rule cohort's TSO ids via /data/TSO500/id_rename_success.log (col3 OLD->NEW).

Produces (all under results/rule_conformant/):
  {v}_ponbench.tsv   numeric-keyed manifest for the extractors (cols: sid run t1 t2 keep_t1 keep_t2 bam cvo exon)
  manifest_dev.tsv   sid cohort cancer_type group   (labels = case_list tumor_type1; for the benchmark)
  missing_{v}.tsv    BAM-heavy worklist for the not-yet-extracted samples: sid<TAB>bam
  num2tso.json / tso2num.json
Seeds isolated rule cache dirs by SYMLINKING existing per-sample caches for the reused (already-extracted) sids
so the extractors cache-hit them and only extract the missing:
  results/rcv{V}_e1se/_raw/<num>.json      <- results/{v}_e1se/_raw/<num>.json
  results/rcv{V}_fragshape/_amp/<num>.npz  <- results/{v}_fragshape/_amp/<num>.npz
  results/metaplots/_helzer_metrics_rcv{V}/<num>.npz <- .../_helzer_metrics_{v}/<num>.npz
  results/rcv{V}_gdd/_raw/<num>.json        <- results/{v}_gdd/_raw/<num>.json
CNA: reuse existing X_CNA rows; write cna_missing_{v}.txt (numeric sids to derive from /data ichorCNA only).
"""
import os, re, glob, json, collections
import pandas as pd, numpy as np

PROJ = "/home/jrkim/TSO_TFBS/project"
RC = f"{PROJ}/results/rule_conformant"
PRIMARY = {"lung cancer", "colorectal cancer", "gastric cancer", "pancreatic cancer",
           "biliary tract cancer", "melanoma", "liver cancer", "prostate cancer", "breast cancer"}
EXPLOR = {"sarcoma", "bladder cancer"}


def num_map():
    tso2num = collections.defaultdict(set)
    with open("/data/TSO500/id_rename_success.log") as f:
        for ln in f:
            if ln.startswith("#") or "->" not in ln:
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            old, new = p[2].split("->")[:2]
            m = re.match(r"(TSO_\d+_B_\d+)", new.strip()); n = re.match(r"(\d{6,})", old.strip())
            if m and n:
                tso2num[m.group(1)].add(n.group(1))
    tso2num = {k: sorted(v)[0] for k, v in tso2num.items()}
    return tso2num, {v: k for k, v in tso2num.items()}


def load(p):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)


def cache_pool(pattern, strip):
    return set(os.path.basename(p)[:-strip] for p in glob.glob(pattern))


def link(src, dst):
    if os.path.exists(src) and not os.path.lexists(dst):
        os.symlink(src, dst)


def main():
    tso2num, num2tso = num_map()
    json.dump(tso2num, open(f"{RC}/tso2num.json", "w"))
    json.dump(num2tso, open(f"{RC}/num2tso.json", "w"))
    devrows = []
    for v in ["v1", "v2"]:
        V = v[-1]
        rm = load(f"{RC}/{v}_manifest.tsv"); rk = rm[rm["keep_t1"] == "1"].copy()
        rk["sid"] = rk["tso_id"].map(tso2num)
        assert rk["sid"].notna().all(), f"{v}: some rule sids have no numeric id"
        # numeric-keyed ponbench-style manifest for the extractors
        pon = rk.rename(columns={"tumor_type1": "t1", "tumor_type2": "t2"})[
            ["sid", "run", "t1", "t2", "keep_t1", "keep_t2", "bam", "cvo", "exon"]]
        pon.to_csv(f"{RC}/{v}_ponbench.tsv", sep="\t", index=False)
        # manifest_dev rows (labels from case_list tumor_type1)
        for _, r in rk.iterrows():
            ct = r["tumor_type1"]
            grp = "primary" if ct in PRIMARY else ("exploratory" if ct in EXPLOR else "other")
            devrows.append((r["sid"], v, ct, grp))
        # seed caches + compute missing per module
        os.makedirs(f"{PROJ}/results/rcv{V}_e1se/_raw", exist_ok=True)
        os.makedirs(f"{PROJ}/results/rcv{V}_fragshape/_amp", exist_ok=True)
        os.makedirs(f"{PROJ}/results/metaplots/_helzer_metrics_rcv{V}", exist_ok=True)
        os.makedirs(f"{PROJ}/results/rcv{V}_gdd/_raw", exist_ok=True)
        pools = {
            "e1se": cache_pool(f"{PROJ}/results/{v}_e1se/_raw/*.json", 5),
            "frag": cache_pool(f"{PROJ}/results/{v}_fragshape/_amp/*.npz", 4),
            "helz": cache_pool(f"{PROJ}/results/metaplots/_helzer_metrics_{v}/*.npz", 4),
            "gdd":  cache_pool(f"{PROJ}/results/{v}_gdd/_raw/*.json", 5),
        }
        sids = list(rk["sid"]); bam_of = dict(zip(rk["sid"], rk["bam"]))
        for s in sids:
            if s in pools["e1se"]:
                link(f"{PROJ}/results/{v}_e1se/_raw/{s}.json", f"{PROJ}/results/rcv{V}_e1se/_raw/{s}.json")
            if s in pools["frag"]:
                link(f"{PROJ}/results/{v}_fragshape/_amp/{s}.npz", f"{PROJ}/results/rcv{V}_fragshape/_amp/{s}.npz")
            if s in pools["helz"]:
                link(f"{PROJ}/results/metaplots/_helzer_metrics_{v}/{s}.npz",
                     f"{PROJ}/results/metaplots/_helzer_metrics_rcv{V}/{s}.npz")
            if s in pools["gdd"]:
                link(f"{PROJ}/results/{v}_gdd/_raw/{s}.json", f"{PROJ}/results/rcv{V}_gdd/_raw/{s}.json")
        # BAM-heavy missing = union of the three BAM-derived caches (e1se/frag/helz)
        miss_bam = [s for s in sids if not (s in pools["e1se"] and s in pools["frag"] and s in pools["helz"])]
        with open(f"{RC}/missing_{v}.tsv", "w") as f:
            for s in miss_bam:
                f.write(f"{s}\t{bam_of[s]}\n")
        miss_gdd = [s for s in sids if s not in pools["gdd"]]
        with open(f"{RC}/missing_gdd_{v}.txt", "w") as f:
            f.write("\n".join(miss_gdd) + ("\n" if miss_gdd else ""))
        # CNA: EXCEPTION to skip-if-exists (user directive) -> RE-EXTRACT ALL rule samples from current
        # /data ichorCNA (results changed on disk). Handled in the extract step via CNA_MAN=rule ponbench.
        print(f"[{v}] rule keep_t1={len(sids)} | reuse caches: e1se {len(sids)-len([s for s in sids if s not in pools['e1se']])}, "
              f"frag {len(sids)-len([s for s in sids if s not in pools['frag']])}, helz {len(sids)-len([s for s in sids if s not in pools['helz']])}, "
              f"gdd {len(sids)-len(miss_gdd)}")
        print(f"     EXTRACT: BAM-heavy(e1se+frag+helz)={len(miss_bam)}  gdd={len(miss_gdd)}  cna=ALL {len(sids)} (re-derive, ichorCNA changed)")
    with open(f"{RC}/manifest_dev.tsv", "w") as f:
        f.write("sid\tcohort\tcancer_type\tgroup\n")
        for s, c, ct, g in devrows:
            f.write(f"{s}\t{c}\t{ct}\t{g}\n")
    print(f"[rc_prepare] manifest_dev rows={len(devrows)} -> {RC}/manifest_dev.tsv")


if __name__ == "__main__":
    main()
