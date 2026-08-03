#!/usr/bin/env python
"""
Build the RULE-CONFORMANT cohort manifest, sourced from TSO500_2202_case_list_v1.1.xlsx.

Rule (user directive 2026-07-07):
  - source of truth  = TSO500_2202_case_list_v1.1.xlsx (sheet 'data'): ID, TSO version, tumor_type1
  - keep only IDs containing 'B_01'
  - split by TSO version: ver_01 -> v1, ver_02 -> v2
  - QC filter: MEDIAN_EXON_COVERAGE >= MIN_COV (=1000), applied BEFORE the class count
  - keep cancer types (tumor_type1) with >= MIN_N (=20) samples, computed INDEPENDENTLY per version
  (tumor_type2 carried along; keep_t2 uses the same >=20-per-version rule for internal consistency.)

Membership is resolved against on-disk BAMs (authoritative). Vendor files (CombinedVariantOutput for the
mutation module; exon_cov_report for the v2 exon-depth module) are resolved from the sample's Results dir.
On-disk sample ids may carry a re-sequencing suffix (TSO_xxxxx_B_01_1); the canonical join key strips it.
v1 has no vendor exon_cov_report (its all-exon-depth is computed from the BAM directly) -> exon left blank.

Writes results/rule_conformant/{v1,v2}_manifest.tsv + cohort_summary.tsv.
"""
import os, re, glob, json, collections
import pandas as pd

PROJ = "/home/jrkim/TSO_TFBS/project"
OUT  = f"{PROJ}/results/rule_conformant"; os.makedirs(OUT, exist_ok=True)
XLSX = f"{PROJ}/TSO500_2202_case_list_v1.1.xlsx"
MIN_N = 20
MIN_COV = 1000            # QC: keep only samples with MEDIAN_EXON_COVERAGE >= MIN_COV (applied pre-count)
CANON = re.compile(r"(TSO_\d+_B_\d+)(?:_\d+)?$")   # strip a trailing _<n> re-seq suffix -> canonical id


def index_bams():
    """canonical TSO id -> list of (date, run, on_disk_id, bam_path), per version."""
    idx = {"v1": collections.defaultdict(list), "v2": collections.defaultdict(list)}
    for v, pat in [("v1", f"/data/TSO500/rawdata/v1/*/BAM/*.bam"),
                   ("v2", f"/data/TSO500/rawdata/v2/*/BAM/*_tumor.bam")]:
        for p in glob.glob(pat):
            if p.endswith(".bai") or "/V2_PoN/" in p:
                continue
            base = os.path.basename(p)
            oid = base[:-4] if v == "v1" else base.replace("_tumor.bam", "")   # on-disk id (may have _n)
            m = CANON.match(oid)
            if not m:
                continue
            canon = m.group(1)
            run = p.split(f"/rawdata/{v}/")[1].split("/")[0]
            dm = re.match(r"(\d{6})_", run); date = dm.group(1) if dm else "000000"
            idx[v][canon].append((date, run, oid, p))
    return idx


def pick(lst, version):
    lst = sorted(lst, key=lambda x: x[0])
    return lst[-1] if version == "v2" else lst[0]   # v2 latest run, v1 first run


def vendor_paths(version, run, oid):
    rdir = f"/data/TSO500/rawdata/{version}/{run}/Results/{oid}"
    cvo = f"{rdir}/{oid}_CombinedVariantOutput.tsv"
    exon = f"{rdir}/{oid}.exon_cov_report.tsv"
    cvo = cvo if os.path.exists(cvo) else ""
    exon = exon if (version == "v2" and os.path.exists(exon)) else ""   # v1 exon-depth is BAM-derived
    return cvo, exon


def main():
    cl = pd.read_excel(XLSX, sheet_name="data")
    cl["ID"] = cl["ID"].astype(str).str.strip()
    cl["ver"] = cl["TSO version"].map({"ver_01": "v1", "ver_02": "v2"})
    cl["t1"] = cl["tumor_type1"].astype(str).str.strip()
    cl["t2"] = cl["tumor_type2"].astype(str).str.strip()
    cl["cov"] = pd.to_numeric(cl["MEDIAN_EXON_COVERAGE"], errors="coerce")
    b = cl[cl["ID"].str.contains("B_01") & cl["ver"].isin(["v1", "v2"])].copy()
    b = b[~b["t1"].isin(["nan", "None", ""])]
    pre = len(b)
    b = b[b["cov"] >= MIN_COV]                          # QC filter, applied BEFORE the >=20 class count
    print(f"[rule] MEDIAN_EXON_COVERAGE>={MIN_COV} filter: {pre} -> {len(b)} "
          f"(dropped {pre - len(b)}: v1={(cl['ID'].str.contains('B_01') & (cl['ver']=='v1') & ~cl['t1'].isin(['nan','None','']) & (cl['cov']<MIN_COV)).sum()}, "
          f"v2={(cl['ID'].str.contains('B_01') & (cl['ver']=='v2') & ~cl['t1'].isin(['nan','None','']) & (cl['cov']<MIN_COV)).sum()})")

    bam_idx = index_bams()
    cols = ["sid", "tso_id", "version", "run", "tumor_type1", "tumor_type2",
            "median_exon_coverage", "keep_t1", "keep_t2", "bam", "cvo", "exon", "bam_found"]
    summ = {}
    for v in ["v1", "v2"]:
        sub = b[b["ver"] == v].copy()
        vc = sub["t1"].value_counts();  keep1 = set(vc[vc >= MIN_N].index)
        vc2 = sub["t2"].value_counts(); keep2 = set(vc2[vc2 >= MIN_N].index) - {"nan", "None", ""}
        rows = []
        for _, r in sub.iterrows():
            hit = bam_idx[v].get(r["ID"])
            if hit:
                _, run, oid, bam = pick(hit, v); found = 1
                cvo, exon = vendor_paths(v, run, oid)
            else:
                run = oid = bam = cvo = exon = ""; found = 0
            rows.append(dict(sid=oid or r["ID"], tso_id=r["ID"], version=v, run=run,
                             tumor_type1=r["t1"], tumor_type2=(r["t2"] if r["t2"] not in ("nan", "None", "") else ""),
                             median_exon_coverage=int(r["cov"]),
                             keep_t1=int(r["t1"] in keep1), keep_t2=int(r["t2"] in keep2),
                             bam=bam, cvo=cvo, exon=exon, bam_found=found))
        md = pd.DataFrame(rows)[cols].sort_values(["keep_t1", "tumor_type1", "tso_id"], ascending=[False, True, True])
        md.to_csv(f"{OUT}/{v}_manifest.tsv", sep="\t", index=False)
        k = md[md["keep_t1"] == 1]
        summ[v] = dict(md=md, keep_types=sorted(keep1), counts=vc[vc >= MIN_N])
        print(f"[{v}] kept(>=20 t1)={len(k)} in {len(keep1)} types | BAM found={int(k['bam_found'].sum())}/{len(k)} "
              f"| CVO={int((k['cvo']!='').sum())}/{len(k)}"
              + (f" | exon_cov={int((k['exon']!='').sum())}/{len(k)}" if v == "v2" else " | exon=BAM-derived")
              + f"  -> {OUT}/{v}_manifest.tsv")

    # cohort summary table
    allt = sorted(set(summ["v1"]["keep_types"]) | set(summ["v2"]["keep_types"]))
    rows = []
    for t in allt:
        rows.append(dict(cancer_type=t,
                         v1=int(summ["v1"]["counts"].get(t, 0)),
                         v2=int(summ["v2"]["counts"].get(t, 0))))
    st = pd.DataFrame(rows)
    st.loc["TOTAL"] = dict(cancer_type="TOTAL",
                           v1=int(st["v1"].sum()), v2=int(st["v2"].sum()))
    st.to_csv(f"{OUT}/cohort_summary.tsv", sep="\t", index=False)
    json.dump({v: summ[v]["keep_types"] for v in summ}, open(f"{OUT}/kept_types.json", "w"), indent=2)
    print("\n==== RULE-CONFORMANT COHORT (tumor_type1 >=20 per version) ====")
    print(st.to_string(index=False))


if __name__ == "__main__":
    main()
