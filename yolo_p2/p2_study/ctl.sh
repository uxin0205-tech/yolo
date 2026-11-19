#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/uxin/yolo/yolo_p2/.worktrees/p2-study"
PYTHON="/home/uxin/yolo/.venv/bin/python"
ARTIFACTS="$ROOT/p2_study/artifacts"
LOG="$ARTIFACTS/logs/controller.log"
PID_FILE="$ARTIFACTS/controller.pid"
SERVICE="yolo-p2-study"
CONFIG="$ROOT/p2_study/config.yaml"

mkdir -p "$ARTIFACTS/logs"

launch() {
    local resume_flag="${1:-}"
    local command=("$PYTHON" -m p2_study.run --config "$CONFIG")
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
            "$PYTHON" -m json.tool "$ARTIFACTS/state.json"
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
