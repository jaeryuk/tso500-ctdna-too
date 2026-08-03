#!/usr/bin/env bash
# Background driver: regenerate the RC-cohort figures the user asked for (2026-07-17).
# Runs every step even if one fails (|| true + explicit OK/FAIL log), so one broken figure never
# silently blocks the rest. Each step is idempotent.
set -uo pipefail
cd /home/jrkim/TSO_TFBS/project
PY=/home/jrkim/.conda/envs/cfse/bin/python
R=/home/jrkim/.conda/envs/cfse_r/bin/Rscript
ND=scripts/auto/nc_readiness
PLOT=results/plot
log(){ echo "[$(date '+%F %T')] $*"; }
run(){ local n="$1"; shift; log "RUN  $n"; if "$@" >>"$LOGD/$n.log" 2>&1; then log "OK   $n"; else log "FAIL $n (see $LOGD/$n.log)"; fi; }
LOGD=logs/rc_figs; mkdir -p "$LOGD"

log "===== 1. CNA concordance data @ TFCUT>0.03 (rc_figures_extract: rc_table cohort, rcv{1,2}_cna TF) ====="
# back up the pre-change CSVs once
for f in $PLOT/cna_arm_concordance_byversion.csv $PLOT/depth_focal_cna_concordance_byversion.csv; do
  [ -f "$f" ] && [ ! -f "$f.pre003.bak" ] && cp -p "$f" "$f.pre003.bak" && log "backed up $(basename "$f")"
done
TFCUT=0.03 run rc_figures_extract $PY $ND/rc_figures_extract.py

log "===== 2. render the two CNA figures (subheading erased; amp/del now a shape LEGEND) ====="
for f in Fig_depth_focal_cna_concordance_byversion.png Fig_cna_arm_concordance_byversion.png; do
  [ -f "$PLOT/$f" ] && [ ! -f "$PLOT/${f%.png}.pre003.bak.png" ] && cp -p "$PLOT/$f" "$PLOT/${f%.png}.pre003.bak.png"
done
run fig_depth_focal $R $ND/fig_depth_focal_byversion_ggpubr.R
run fig_cna_arm     $R $ND/fig_cna_arm_byversion_ggpubr.R

log "===== 3. re-render the already-RC figures that depend on the refreshed CSVs ====="
run fig_e1_expr_bycancer  $R $ND/fig_e1_expr_bycancer_ggpubr.R
run fig_e1_expr_byversion $R $ND/fig_e1_expr_byversion_ggpubr.R

log "===== 4. SHAPE ablation figure — ONLY if the strict benchmark has produced real per-repeat data ====="
if [ -f results/rule_conformant/shape_ablation_strict_perrepeat.tsv ]; then
  run fig_shape_ablation_strict $PY $ND/fig_shape_ablation_strict.py
else
  log "SKIP fig_shape_ablation_strict — strict benchmark still running; rerun this driver when it lands"
fi

log "===== 5. SHAPE bias-control figure (RC strict, +lenent arms only) — ONLY if strict bias-control landed ====="
if [ -f results/rule_conformant/biascontrol_strict_perrepeat.tsv ]; then
  run fig_rc_biascontrol_strict $R $ND/fig_rc_biascontrol_strict_ggpubr.R
else
  log "SKIP fig_rc_biascontrol_strict — strict bias-control still running; rerun this driver when it lands"
fi

log "ALL_RC_FIGS_DRIVER_DONE"
