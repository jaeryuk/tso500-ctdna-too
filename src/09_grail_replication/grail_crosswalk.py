#!/usr/bin/env python
"""
Build the GRAIL BAM -> phenotype crosswalk from the EGA metadata, the join key the old manifest was missing.

Join chain (all in /data/grail_EGAD00001005302/metadata):
  analysis2_manifest.tsv : EGAF(file) -> on-disk BAM filename   (downloaded analysis2, error-corrected)
  sample_file.tsv        : EGAF       -> EGAN(sample)
  samples.tsv            : EGAN       -> phenotype, title(material), subject_id, biological_sex

Writes results/auto_plan/grail/bam_cancer_type.tsv with the columns grail_manifest.py prefers:
  parsed_sample_id  bam_path  cancer_type  subject_id  sample_accession_id  sample_material  biological_sex

Phenotype map: 'Cancer breast/lung/prostate' -> '<x> cancer'; 'Non cancer' -> 'non-cancer'.
Keeps BOTH plasma cfDNA and WBC gDNA rows (material column distinguishes); the manifest builder filters to cfDNA.
"""
import os, csv, glob

META = "/data/grail_EGAD00001005302/metadata"
GBAM = "/data/grail_EGAD00001005302/bam_analysis2"
OUT = "/home/jrkim/TSO_TFBS/project/results/auto_plan/grail"
os.makedirs(OUT, exist_ok=True)
PHEN = {"cancer breast": "breast cancer", "cancer lung": "lung cancer",
        "cancer prostate": "prostate cancer", "non cancer": "non-cancer"}
MAT = {"plasma cfdna": "plasma_cfDNA", "wbc gdna": "WBC_gDNA"}


def main():
    # EGAF -> on-disk bam path (analysis2 dir name == EGAF)
    egaf_path = {}
    for p in glob.glob(f"{GBAM}/*/*.bam"):
        egaf_path[os.path.basename(os.path.dirname(p))] = p
    # EGAF -> EGAN
    f2s = {}
    with open(f"{META}/sample_file.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            f2s[r["file_accession_id"]] = r["sample_accession_id"]
    # EGAN -> info
    s2i = {}
    with open(f"{META}/samples.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            s2i[r["accession_id"]] = r
    rows = []
    for egaf, path in egaf_path.items():
        egan = f2s.get(egaf)
        if egan is None or egan not in s2i:
            continue
        s = s2i[egan]
        ph = (s.get("phenotype") or "").strip().lower()
        ti = (s.get("title") or "").strip().lower()
        cancer = PHEN.get(ph, "unknown")
        material = MAT.get(ti, ti)
        sid = os.path.basename(path)[:-4]              # filename without .bam, e.g. MRL0001CHcollapsed
        rows.append([sid, path, cancer, s.get("subject_id", ""), egan, material,
                     s.get("biological_sex", "")])
    rows.sort()
    out = f"{OUT}/bam_cancer_type.tsv"
    with open(out, "w") as f:
        f.write("parsed_sample_id\tbam_path\tcancer_type\tsubject_id\tsample_accession_id\t"
                "sample_material\tbiological_sex\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    import collections
    cf = collections.Counter(r[2] for r in rows if r[5] == "plasma_cfDNA")
    print(f"[crosswalk] {len(rows)} BAMs total; cfDNA label coverage: {dict(cf)}")
    print(f"[crosswalk] -> {out}")


if __name__ == "__main__":
    main()
