#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/../.venv/bin/python"
RUN="$ROOT/artifacts/runs/T0/full/ebba4065-47f7-4eee-9ac4-0a324e8eee5c"

while ! "$PY" - "$RUN/status.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    status = json.loads(path.read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if status.get("completed") is True and status.get("valid_for_research") is True else 1)
PY
do
    sleep 30
done

"$PY" -u -m binary_attention.ema_repair \
    --run "$RUN" \
    --source-weights "$ROOT/../original/weight/yolo11m.pt" \
    > "$ROOT/logs/formal-plan/T0-ema-repair.log" 2>&1
