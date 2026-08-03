#!/usr/bin/env bash
# Source this before running any stage script:  `source env.sh`
# The pipeline scripts were developed as a flat module namespace and import shared
# helpers by bare name (e.g. `import ncr_common`, `from rc_cohort import rc_table`).
# All such shared modules live in src/common/, so putting it on PYTHONPATH keeps the
# reorganized (numbered-stage) tree runnable.
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="${_here}/src/common:${PYTHONPATH}"
echo "PYTHONPATH += ${_here}/src/common"
# NOTE: data locations (PROJ, results/, reference FASTA, BAM dirs) are set at the top
# of each script and must be pointed at your own environment before running.
