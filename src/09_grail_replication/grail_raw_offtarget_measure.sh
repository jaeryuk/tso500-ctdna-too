#!/usr/bin/env bash
# Measure off-target coverage + fragment sizes on the RAW analysis1 GRAIL cfDNA BAM,
# directly comparable to the analysis2 pilot (which had: true off-target 0.02x, median 0 rd/Mb; 0% cfDNA frags).
set -uo pipefail
BAM=${1:-/data/grail_EGAD00001005302/bam_analysis1/EGAF00002547166/MRL0001CHraw.bam}
OUT=/home/jrkim/TSO_TFBS/project/results/auto_plan/grail/offprobe/raw_measure
mkdir -p "$OUT"
SAM=/usr/bin/samtools
RC=/home/jrkim/.conda/envs/cfse_cna/bin/readCounter
PY=/home/jrkim/.conda/envs/cfse/bin/python
PANEL=/home/jrkim/TSO_TFBS/project/results/auto_plan/grail/panel/grail_panel_exons.merged.bed

echo "[raw-measure] BAM=$BAM ($(du -h "$BAM" 2>/dev/null | cut -f1))"
echo "[raw-measure] flagstat (dup rate, mapped) ..."
$SAM flagstat -@ 8 "$BAM" | tee "$OUT/flagstat.txt"

echo "[raw-measure] readCounter 1Mb wig (q20, incl + excl dup) ..."
# with duplicates kept (quality 20 only):
$RC --window 1000000 --quality 20 \
  --chromosome "chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX,chrY" \
  "$BAM" > "$OUT/raw_q20.wig" 2>/dev/null

echo "[raw-measure] TLEN (insert-size) distribution from a 3M-read sample ..."
$SAM view -@8 -f 0x2 -F 0x900 "$BAM" 2>/dev/null | head -3000000 \
  | awk '{t=$9; if(t<0)t=-t; if(t>0&&t<2000) print t}' > "$OUT/tlen_sample.txt"

$PY - "$OUT/raw_q20.wig" "$PANEL" "$OUT/tlen_sample.txt" "$OUT" <<'PY'
import sys, numpy as np, collections
wig, panelbed, tlenf, outd = sys.argv[1:5]
panel=collections.defaultdict(list)
for ln in open(panelbed):
    c,s,e=ln.split("\t")[:3]; panel[c].append((int(s),int(e)))
def bin_on_panel(c,b):
    lo,hi=b*1000000,(b+1)*1000000
    for s,e in panel.get(c,[]):
        if s<hi and e>lo: return True
    return False
chrom=None;b=0;on=[];off=[]
for ln in open(wig):
    if ln.startswith("fixedStep"):
        d=dict(kv.split("=") for kv in ln.split()[1:]); chrom=d["chrom"]; b=0; continue
    if ln[0].isdigit() or ln[0]=='-':
        v=float(ln); (on if bin_on_panel(chrom,b) else off).append(v); b+=1
on=np.array(on); off=np.array(off)
tl=np.loadtxt(tlenf) if __import__('os').path.getsize(tlenf)>0 else np.array([0])
res=[]
res.append("=== RAW analysis1 MRL0001 cfDNA — off-target + fragment feasibility ===")
res.append(f"PANEL 1Mb bins:     n={len(on):4d}  reads={int(on.sum()):>10d}  median={np.median(on):.0f}")
res.append(f"NON-PANEL bins:     n={len(off):4d}  reads={int(off.sum()):>10d}  median={np.median(off):.0f}  mean={off.mean():.1f}")
res.append(f"TRUE off-target coverage (non-panel): {off.sum()*150/3.0e9:.4f}x   %bins>=50rd: {100*np.mean(off>=50):.1f}%   %bins>=100: {100*np.mean(off>=100):.1f}%")
res.append(f"  (analysis2 pilot was: 0.02x, median 0/Mb, 5.1% bins>=50  => ichorCNA needs ~0.05-0.1x, near-uniform)")
res.append(f"FRAGMENT insert size (n={len(tl)}): median={np.median(tl):.0f}  %in 100-180bp(cfDNA)={100*np.mean((tl>=100)&(tl<=180)):.1f}%  %in 30-180={100*np.mean((tl>=30)&(tl<=180)):.1f}%  %~167+/-15={100*np.mean((tl>=152)&(tl<=182)):.1f}%")
res.append(f"  (analysis2 pilot was: median ~320bp, 0.0% in cfDNA range)")
txt="\n".join(res); print(txt)
open(f"{outd}/SUMMARY.txt","w").write(txt+"\n")
PY
echo "[raw-measure] DONE -> $OUT/SUMMARY.txt"
