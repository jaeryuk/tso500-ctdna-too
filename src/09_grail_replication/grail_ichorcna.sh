#!/usr/bin/env bash
# Genome-wide CNA for GRAIL cfDNA via ichorCNA off-probe: readCounter (1Mb bins) -> runIchorCNA.R.
# Uses cfse_cna env (readCounter + ichorCNA R pkg + its extdata GC/map wigs). Stages each BAM to local
# /home (dodge /data contention). Output mirrors ultima's layout so v2_cna_features.py can read it.
# Usage: grail_ichorcna.sh [P]
set -uo pipefail
cd /home/jrkim/TSO_TFBS/project
ENVB=/home/jrkim/.conda/envs/cfse_cna/bin
SAM=/usr/bin/samtools
MAN=results/auto_plan/grail/manifest.tsv
OD=results/auto_plan/grail/ichorCNA; ST=results/grail_deepsomatic/_stage_cna; mkdir -p "$OD" "$ST"
P="${1:-6}"
CHRS="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX"

# locate ichorCNA resources (runIchorCNA.R + GC/map wigs) inside the installed package
EXT=$("$ENVB/Rscript" -e 'cat(system.file("extdata", package="ichorCNA"))' 2>/dev/null)
RUN=$("$ENVB/Rscript" -e 'p=system.file("scripts/runIchorCNA.R", package="ichorCNA"); if(!nzchar(p)) p=Sys.glob(file.path(.libPaths(),"ichorCNA","scripts","runIchorCNA.R"))[1]; cat(p)' 2>/dev/null)
GC=$(ls "$EXT"/gc_hg19_1000kb.wig 2>/dev/null | head -1)
MAPW=$(ls "$EXT"/map_hg19_1000kb.wig 2>/dev/null | head -1)
CENT=$(ls "$EXT"/GRCh37.p13_centromere_UCSC-gapTable.txt 2>/dev/null | head -1)
echo "[grail-ichor] runIchorCNA=$RUN gc=$GC map=$MAPW"
[ -s "$GC" ] && [ -s "$MAPW" ] && [ -s "$RUN" ] || { echo "[grail-ichor] missing ichorCNA resources -> skip CNA"; exit 1; }

one(){
  local sid="$1" bam="$2"
  local out="$OD/$sid"; [ -s "$out/${sid}.cna.seg" ] && return 0
  mkdir -p "$out"
  local lb="$ST/$(basename "$bam")"
  rsync -a --append-verify --inplace "$bam" "$lb" 2>/dev/null && rsync -a "${bam}.bai" "${lb}.bai" 2>/dev/null || { echo "[grail-ichor] stage fail $sid"; return 1; }
  "$ENVB/readCounter" --window 1000000 --quality 20 --chromosome "$CHRS" "$lb" > "$out/${sid}.wig" 2>/dev/null || { echo "[grail-ichor] readCounter fail $sid"; rm -f "$lb" "$lb.bai"; return 1; }
  "$ENVB/Rscript" "$RUN" --id "$sid" --WIG "$out/${sid}.wig" --gcWig "$GC" --mapWig "$MAPW" \
     ${CENT:+--centromere "$CENT"} --ploidy "c(2)" --normal "c(0.5,0.6,0.7,0.8,0.9)" --maxCN 5 \
     --includeHOMD False --chrs "paste0('chr',c(1:22))" --chrTrain "paste0('chr',c(1:22))" \
     --estimateNormal True --estimatePloidy True --estimateScPrevalence False --scStates "c()" \
     --outDir "$out" > "$out/ichor.log" 2>&1 || echo "[grail-ichor] ichorCNA fail $sid"
  rm -f "$lb" "$lb.bai"
  echo "[grail-ichor] done $sid"
}
export -f one; export ENVB SAM OD ST CHRS GC MAPW RUN CENT

awk -F'\t' 'NR>1 && $2!=""{print $1"\t"$2}' "$MAN" > "$OD/_work.tsv"
echo "[grail-ichor] $(wc -l < "$OD/_work.tsv") samples (P=$P)"
taskset -c 36-47 nice -n 19 ionice -c3 xargs -a "$OD/_work.tsv" -P "$P" -L1 bash -c 'one "$1" "$2"' _
echo "[grail-ichor] DONE -> $OD"
