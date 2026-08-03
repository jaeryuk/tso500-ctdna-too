#!/usr/bin/env bash
# Phase 7 (autonomous, LAST, fail-soft) — GRAIL external validation, all metrics.
# Steps (each fail-soft; a failure logs and continues so the primary result is never blocked):
#   1. resume somatic calling over 198 cfDNA (grail_run_all.sh: 90 matched + 108 cfDNA-only tumor-only)
#   2. build GRAIL sample manifest + cancer-type labels (lung/breast/prostate + non-cancer controls)
#   3. genome-wide CNA via ichorCNA off-probe on GRAIL cfDNA BAMs (cfse_cna: readCounter + ichorCNA)
#   4. build GRAIL feature matrices (mutation profile/signature from calls; CNA; E1/depth/SHAPE from BAMs)
#   5. external validation: TSO500-dev models -> GRAIL, shared-column harmonization, report overlap classes
set -uo pipefail
cd /home/jrkim/TSO_TFBS/project
PROJ=$PWD; PY=/home/jrkim/.conda/envs/cfse/bin/python
GD=results/grail_deepsomatic; L=results/auto_plan/logs; mkdir -p "$L" results/auto_plan/grail
log(){ echo "[grail $(date '+%F %T')] $*"; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

log "1. resume somatic calling (198 cfDNA)"
( cd "$GD" && bash scripts/grail_run_all.sh "${GRAIL_P:-8}" ) >> "$L/grail_calling.log" 2>&1 || log "calling step incomplete"

log "2. GRAIL manifest + labels"
"$PY" scripts/auto/grail_manifest.py >> "$L/grail_manifest.log" 2>&1 || log "manifest step issue"

log "3. ichorCNA off-probe on GRAIL cfDNA BAMs"
bash scripts/auto/grail_ichorcna.sh 6 >> "$L/grail_ichorcna.log" 2>&1 || log "ichorCNA step incomplete"

log "4. build GRAIL feature matrices (all 6 classes)"
"$PY" scripts/auto/grail_build_features.py >> "$L/grail_features.log" 2>&1 || log "feature build incomplete"

log "5. external validation (TSO500 dev -> GRAIL)"
"$PY" scripts/auto/grail_external.py >> "$L/grail_external.log" 2>&1 || log "external validation incomplete"

log "GRAIL phase finished (see results/auto_plan/grail/)"
