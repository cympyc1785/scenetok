#!/usr/bin/env bash
# Independent watcher for the LagerNVS-orig train3 run. Runs in its own screen so
# it survives across the harness turn boundary. Writes status to STATUS file.
set +e
LOG=/tmp/lagernvs_orig_dl3dv_1k.log
STATUS=/tmp/lagernvs_orig_dl3dv_1k.status
last_eval=0
while true; do
  sleep 300
  if ! screen -ls 2>/dev/null | grep -q train3; then
    { echo "STATUS=DIED $(date)"; tail -25 "$LOG"; } > "$STATUS"; break
  fi
  if grep -qi "traceback" "$LOG" 2>/dev/null; then
    { echo "STATUS=ERROR $(date)"; grep -iA3 traceback "$LOG" | tail -20; } > "$STATUS"; break
  fi
  n=$(grep -c "Evaluation metrics" "$LOG" 2>/dev/null)
  iter=$(grep -oE "Iter [0-9]+ loss [0-9.]+" "$LOG" 2>/dev/null | tail -1)
  { echo "STATUS=RUNNING $(date)"; echo "$iter"; echo "evals_done=$n";
    grep -B1 -A4 "Evaluation metrics" "$LOG" 2>/dev/null | tail -30; } > "$STATUS"
done
