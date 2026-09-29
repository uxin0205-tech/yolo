#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_REQUESTED="${P2_PYTHON:-}"
if [[ -n "$PYTHON_REQUESTED" ]]; then
    PYTHON_FOUND="$(command -v -- "$PYTHON_REQUESTED" || true)"
else
    PYTHON_FOUND="$(command -v python || command -v python3 || true)"
fi
if [[ -z "$PYTHON_FOUND" ]]; then
    echo "Python interpreter not found: ${PYTHON_REQUESTED:-python or python3}" >&2
    exit 1
fi
PYTHON_BIN="$(readlink -f -- "$PYTHON_FOUND")"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter is not executable: $PYTHON_BIN" >&2
    exit 1
fi
ARTIFACTS="$ROOT/p2_study/artifacts"
LOG="$ARTIFACTS/logs/controller.log"
PID_FILE="$ARTIFACTS/controller.pid"
SERVICE="yolo-p2-study"
CONFIG="$ROOT/p2_study/config.yaml"

mkdir -p "$ARTIFACTS/logs"

launch() {
    local resume_flag="${1:-}"
    local command=("$PYTHON_BIN" -m p2_study.run --config "$CONFIG")
    if [[ "$resume_flag" == "--resume" ]]; then
        command+=(--resume)
    fi
    if command -v systemd-run >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
        systemctl --user reset-failed "$SERVICE.service" >/dev/null 2>&1 || true
        systemd-run --user --unit="$SERVICE" --collect \
            --property="WorkingDirectory=$ROOT" \
            --property="StandardOutput=append:$LOG" \
            --property="StandardError=append:$LOG" \
            --setenv=PYTHONUNBUFFERED=1 "${command[@]}"
        echo "Started $SERVICE.service; Codex/terminal can now be closed."
    else
        (
            cd "$ROOT"
            nohup env PYTHONUNBUFFERED=1 "${command[@]}" >>"$LOG" 2>&1 &
            echo $! >"$PID_FILE"
        )
        echo "Started PID $(<"$PID_FILE") with nohup; Codex/terminal can now be closed."
    fi
}

case "${1:-}" in
    start)
        [[ ! -e "$ARTIFACTS/state.json" ]] || { echo "state.json exists; use resume"; exit 1; }
        launch
        ;;
    resume)
        launch --resume
        ;;
    status)
        systemctl --user status "$SERVICE.service" --no-pager 2>/dev/null || true
        if [[ -f "$ARTIFACTS/state.json" ]]; then
            "$PYTHON_BIN" -m json.tool "$ARTIFACTS/state.json"
        else
            echo "No state.json yet."
        fi
        ;;
    logs)
        touch "$LOG"
        tail -n 100 -f "$LOG"
        ;;
    stop)
        systemctl --user stop "$SERVICE.service" >/dev/null 2>&1 || true
        if [[ -f "$PID_FILE" ]]; then
            kill "$(<"$PID_FILE")" 2>/dev/null || true
            rm -f "$PID_FILE"
        fi
        echo "Controller stopped; state and checkpoints were preserved."
        ;;
    *)
        echo "Usage: $0 {start|resume|status|logs|stop}"
        exit 2
        ;;
esac
