#!/usr/bin/env bash
# readCounter 1Mb wig for every GRAIL cfDNA BAM (input to PoN-normalized on-target CNA).
#
# NOTE on method: the GRAIL analysis2 (UMI-collapsed) BAMs are CAPTURE-ONLY (~0.08% of reads are
# off-target; genome-wide 1Mb bins outside the panel hold a median of 3-4 reads). True off-probe
# ichorCNA (which needs millions of genome-wide off-target reads) is therefore INFEASIBLE on these
# BAMs, and the raw analysis1 BAMs that retain off-target reads are not on disk. We instead bin all
# reads at 1Mb; the ~570 panel-covered bins carry the usable on-target signal, which grail_cna_features.py
# normalizes against the non-cancer-cfDNA panel-of-normals + GC/map to derive arm-level CNA.
set -uo pipefail
cd /home/jrkim/TSO_TFBS/project
ENVB=/home/jrkim/.conda/envs/cfse_cna/bin
MAN=results/auto_plan/grail/manifest.tsv
OD=results/auto_plan/grail/wig; mkdir -p "$OD"
P="${1:-8}"
CHRS="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX"

one(){
  local sid="$1" bam="$2"
  local out="$OD/${sid}.wig"
  [ -s "$out" ] && { echo "[rc] skip $sid"; return 0; }
  "$ENVB/readCounter" --window 1000000 --quality 20 --chromosome "$CHRS" "$bam" > "$out.tmp" 2>/dev/null \
    && mv "$out.tmp" "$out" && echo "[rc] done $sid" || { echo "[rc] FAIL $sid"; rm -f "$out.tmp"; }
}
export -f one; export ENVB OD CHRS

awk -F'\t' 'NR>1 && $2!=""{print $1"\t"$2}' "$MAN" > "$OD/_work.tsv"
echo "[rc] $(wc -l < "$OD/_work.tsv") BAMs (P=$P)"
nice -n 19 ionice -c3 xargs -a "$OD/_work.tsv" -P "$P" -L1 bash -c 'one "$1" "$2"' _
echo "[rc] DONE -> $OD"
