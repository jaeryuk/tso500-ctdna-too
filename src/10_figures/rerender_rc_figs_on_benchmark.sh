#!/usr/bin/env bash
# Re-render every RC figure that reads results/rule_conformant/benchmark/{v1,v2}_oof.npz, gated on the
# nested benchmark (rc_benchmark_nested_v2.py) finishing and actually REWRITING that store.
#
# Why gated: the current store is dated 2026-07-08, i.e. it predates the SHAPE_nep300 switch. Every
# figure below is therefore correct-cohort (RC: v1 1093/K=10, v2 796/K=8) but stale-SHAPE. Re-rendering
# before the benchmark lands is a no-op -- same inputs, byte-identical output.
#
# Usage: rerender_rc_figs_on_benchmark.sh [PID]     (default: the running rc_benchmark_nested_v2.py)
# Out:   results/rule_conformant/logs/rerender_rc_figs.log
set -uo pipefail
P=/home/jrkim/TSO_TFBS/project
ND=$P/scripts/auto/nc_readiness
PY=/home/jrkim/.conda/envs/cfse/bin/python
R=/home/jrkim/.conda/envs/cfse_r/bin/Rscript
BEN=$P/results/rule_conformant/benchmark
PID=${1:-$(pgrep -f rc_benchmark_nested_v2.py | head -1)}

say(){ echo "[$(date '+%F %T')] $*"; }

# mtime of the store BEFORE waiting -- we require it to change, not just for the pid to vanish
before_v1=$(stat -c %Y "$BEN/v1_oof.npz" 2>/dev/null || echo 0)
before_v2=$(stat -c %Y "$BEN/v2_oof.npz" 2>/dev/null || echo 0)
say "waiting on nested benchmark PID=${PID:-<none>}; store mtimes before: v1=$before_v1 v2=$before_v2"

if [ -n "${PID:-}" ]; then
  while kill -0 "$PID" 2>/dev/null; do sleep 300; done
  say "benchmark PID $PID exited"
else
  say "no benchmark process found -- proceeding to the freshness check"
fi

after_v1=$(stat -c %Y "$BEN/v1_oof.npz" 2>/dev/null || echo 0)
after_v2=$(stat -c %Y "$BEN/v2_oof.npz" 2>/dev/null || echo 0)
if [ "$after_v1" -le "$before_v1" ] || [ "$after_v2" -le "$before_v2" ]; then
  say "ABORT: benchmark exited but {v1,v2}_oof.npz were NOT rewritten (v1 $before_v1->$after_v1, v2 $before_v2->$after_v2)."
  say "       It likely died before the write. Figures left untouched -- investigate before re-running."
  exit 1
fi
say "store refreshed (v1 $before_v1->$after_v1, v2 $before_v2->$after_v2) -- re-rendering"

# sanity: the RC cohort shape must still hold, else the figures would silently change meaning
"$PY" - <<'PYEOF' || { say "ABORT: RC cohort shape check failed"; exit 1; }
import numpy as np, sys
B="/home/jrkim/TSO_TFBS/project/results/rule_conformant/benchmark"
want={"v1":(1093,10),"v2":(796,8)}
for c,(n,k) in want.items():
    z=np.load(f"{B}/{c}_oof.npz",allow_pickle=True)
    got=(len(z["y"]), len(z["classes"]))
    print(f"  {c}: n={got[0]} K={got[1]} (want {n}/{k})")
    if got!=(n,k): sys.exit(f"RC shape mismatch for {c}: {got} != {(n,k)}")
print("  RC cohort shape OK")
PYEOF

say "--- extractors ---"
nice -n 10 "$PY" "$ND/rc_top1_extract.py"            || say "FAIL rc_top1_extract"
nice -n 10 "$PY" "$ND/rc_top1_percancer_extract.py"  || say "FAIL rc_top1_percancer_extract"
nice -n 10 "$PY" "$ND/rc_tfbin_extract.py"           || say "FAIL rc_tfbin_extract"

say "--- figures ---"
for s in fig_rc_top1_box_ggpubr.R fig_rc_top1_percancer_heatmap_ggpubr.R fig_rc_top1_tfbin_ggpubr.R \
         p_perf_vs_tfbin_ggpubr.R p_shapley_vs_tfbin_ggpubr.R; do
  nice -n 10 "$R" "$ND/$s" 2>&1 | grep -v Fontconfig || say "FAIL $s"
done
say "RERENDER_RC_FIGS_DONE"
