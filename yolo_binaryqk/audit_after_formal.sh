#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/../.venv/bin/python"
LOG_DIR="$ROOT/logs/formal-plan"
FORMAL_UNIT="${1:-yolo11-binaryqk-formal.service}"
PERSISTENT_FORMAL_UNIT="yolo11-binaryqk-formal-persistent.service"

# Wait for a valid completion marker.  If the transient service exits before
# that boundary, hand execution to the restart-enabled persistent unit.  The
# runner itself reuses only strict, completed artifacts, so recovery cannot
# promote a partial run.
while ! grep -q 'FORMAL PLAN COMPLETE' "$LOG_DIR/runner.log"; do
    if systemctl --user is-active --quiet "$FORMAL_UNIT"; then
        sleep 30
        continue
    fi

    if [[ "$FORMAL_UNIT" != "$PERSISTENT_FORMAL_UNIT" ]]; then
        printf '[%s] transient formal service ended; handing off to persistent runner\n' "$(date '+%F %T')" \
            | tee -a "$LOG_DIR/runner.log"
        FORMAL_UNIT="$PERSISTENT_FORMAL_UNIT"
    else
        printf '[%s] persistent formal service inactive; requesting restart\n' "$(date '+%F %T')" \
            | tee -a "$LOG_DIR/runner.log"
    fi
    systemctl --user start "$FORMAL_UNIT"
    sleep 30
done

"$PY" -u -m binary_attention.cli report > "$LOG_DIR/final-report.log" 2>&1
"$PY" -u -m binary_attention.cli audit > "$LOG_DIR/final-audit.json" 2>&1
printf '[%s] FINAL ARTIFACT AUDIT PASSED\n' "$(date '+%F %T')" \
    | tee -a "$LOG_DIR/runner.log"
