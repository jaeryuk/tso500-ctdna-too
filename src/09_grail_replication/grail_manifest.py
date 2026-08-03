#!/usr/bin/env python
"""GRAIL cfDNA sample manifest for external validation.

Preferred source is results/auto_plan/grail/bam_cancer_type.tsv, which is built from the
new EGA metadata crosswalk:

  BAM EGAF accession -> sample_file.tsv -> samples.tsv phenotype.

Fallback retains the old best-effort filename rules only if the direct metadata manifest is
not available. Outputs results/auto_plan/grail/manifest.tsv.
"""
import os, re, json, glob, collections
import csv
PROJ = "/home/jrkim/TSO_TFBS/project"
GBAM = "/data/grail_EGAD00001005302/bam_analysis2"
META = "/data/grail_EGAD00001005302/metadata"
OUT = f"{PROJ}/results/auto_plan/grail"; os.makedirs(OUT, exist_ok=True)
BAM_CANCER_TYPE = f"{OUT}/bam_cancer_type.tsv"
MAP = {"lung": "lung cancer", "breast": "breast cancer", "prostate": "prostate cancer",
       "non cancer": "non-cancer", "non_cancer": "non-cancer"}


def main():
    if os.path.exists(BAM_CANCER_TYPE):
        rows = []
        with open(BAM_CANCER_TYPE, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r.get("sample_material") != "plasma_cfDNA":
                    continue
                rows.append((r["parsed_sample_id"], r["bam_path"], r["cancer_type"],
                             r.get("subject_id", ""), r.get("sample_accession_id", "")))
        rows = sorted(rows)
        cov = collections.Counter(r[2] for r in rows)
        with open(f"{OUT}/manifest.tsv", "w") as f:
            f.write("sid\tcfDNA_bam\tcancer_type\tsubject_id\tsample_accession_id\n")
            for sid, bam, lab, subject, sample_acc in rows:
                f.write(f"{sid}\t{bam}\t{lab}\t{subject}\t{sample_acc}\n")
        print(f"[grail-manifest] {len(rows)} cfDNA samples from direct metadata; label coverage: {dict(cov)}", flush=True)
        print(f"[grail-manifest] -> {OUT}/manifest.tsv", flush=True)
        return

    # subject_id (W...) -> phenotype  from metadata
    phen = {}
    try:
        for s in json.load(open(f"{META}/dataset_samples.json")):
            sid = str(s.get("subject_id", "")); ph = (s.get("phenotype") or "").strip().lower()
            if sid: phen[sid] = ph
    except Exception:
        pass
    ch = collections.defaultdict(list)
    for p in glob.glob(f"{GBAM}/*/*CHcollapsed.bam"):
        b = os.path.basename(p); m = re.match(r"^(.*?)CH\d*collapsed\.bam$", b)
        if m: ch[m.group(1)].append(p)
    rows = []; cov = collections.Counter()
    for base, paths in sorted(ch.items()):
        lab = "unknown"
        u = base.upper()
        if u.startswith("MSKVB"): lab = "breast cancer"
        elif u.startswith("MSKVL"): lab = "lung cancer"
        elif u.startswith("MSKVP"): lab = "prostate cancer"
        elif base.startswith("SDBBW"):
            w = "W" + base[5:]                          # SDBBW044... -> W044...
            ph = phen.get(w, "")
            lab = MAP.get(ph, "non-cancer" if "non" in ph else "unknown")
        rows.append((base, sorted(paths)[0], lab)); cov[lab] += 1
    with open(f"{OUT}/manifest.tsv", "w") as f:
        f.write("sid\tcfDNA_bam\tcancer_type\n")
        for b, p, l in rows: f.write(f"{b}\t{p}\t{l}\n")
    print(f"[grail-manifest] {len(rows)} cfDNA samples; label coverage: {dict(cov)}", flush=True)
    print(f"[grail-manifest] -> {OUT}/manifest.tsv", flush=True)


if __name__ == "__main__":
    main()
