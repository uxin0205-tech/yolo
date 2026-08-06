#!/usr/bin/env bash
set -Eeuo pipefail

# Persistent sequential executor for the complete original T/N matrix. Every
# formal run uses independent FP-source attention-only fine-tuning; binary
# variants enable QAT while T0 remains the FP control. The selected T3/T4/N4
# artifacts are materialized for the original analysis chain.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/../.venv/bin/python"
DATA_MANIFEST="$ROOT/data/coco_full.txt"
DATA_YAML="$ROOT/data/coco-full.yaml"
SOURCE_WEIGHTS="$ROOT/../original/weight/yolo11m.pt"
LOG_DIR="$ROOT/logs/formal-plan"
mkdir -p "$LOG_DIR"

# A failed systemd attempt can otherwise leave a child trainer alive while a
# restart starts the same matrix again.  The descriptor is held for the whole
# runner lifetime and makes every retry single-owner.
exec 9>"$LOG_DIR/.formal-plan.lock"
if ! flock -n 9; then
    printf '[%s] another formal runner owns the matrix lock; exiting\n' "$(date '+%F %T')" \
        | tee -a "$LOG_DIR/runner.log"
    exit 0
fi

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_DIR/runner.log"
}

artifact_name() {
    local value="$1"
    value="${value//\//-}"
    value="${value//+/-}"
    printf '%s' "$value"
}

latest_valid_checkpoint() {
    local artifact="$1"
    local stage="$2"
    local expected_epochs="$3"
    local expected_config_hash="$4"
    "$PY" - "$ROOT" "$artifact" "$stage" "$expected_epochs" "$expected_config_hash" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
artifact = sys.argv[2]
stage = sys.argv[3]
expected_epochs = int(sys.argv[4])
expected_config_hash = sys.argv[5]
base = root / "artifacts" / "runs" / artifact / stage
valid = []
for run in base.iterdir() if base.exists() else ():
    status_path = run / "status.json"
    resolved_path = run / "resolved_config.json"
    training_path = run / "training_args.json"
    checkpoint = run / "checkpoints" / "strict-last.pt"
    if not status_path.exists() or not resolved_path.exists() or not training_path.exists() or not checkpoint.exists():
        continue
    try:
        status = json.loads(status_path.read_text())
        resolved = json.loads(resolved_path.read_text())
        training = json.loads(training_path.read_text())
    except Exception:
        continue
    contract = resolved.get("quantization_contract") or {}
    contract_ok = (
        (resolved.get("qk_mode") != "scaled_sign" or contract.get("qk_scale") == "mean_abs_channel_token_per_sample_head")
        and (resolved.get("v_bits") != 8 or contract.get("v8_scale") == "max_abs_token_per_sample_head_channel")
        and (resolved.get("p_bits") != 8 or contract.get("p8_scale") == "static_unsigned_1_over_255")
        and (resolved.get("bias_type") != "dense_2d" or (
            contract.get("bias_parameterization") == "full_2d_relative_position"
            and contract.get("bias_initialization") == "truncated_normal_std_0.02"
        ))
    )
    if (
        status.get("valid_for_research") is True
        and status.get("completed") is True
        and status.get("stage") == stage
        and status.get("epochs") == expected_epochs
        and training.get("epochs") == expected_epochs
        and training.get("trainable_scope") == "attention_only"
        and training.get("checkpoint_weight_source") == "epoch_ema"
        and training.get("metrics_weight_source") == "epoch_ema"
        and resolved.get("plan_name") == "YOLO11 BinaryAttention complete 10-epoch attention-only QAT plan"
        and resolved.get("config_hash") == expected_config_hash
        and contract_ok
    ):
        valid.append((checkpoint.stat().st_mtime_ns, checkpoint))
if valid:
    print(max(valid)[1])
PY
}

expected_config_hash() {
    local variant="$1"
    shift
    "$PY" - "$variant" "$@" <<'PY'
import argparse
import sys

from binary_attention.variants.definitions import (
    KD_INHERITING_VARIANTS,
    NON_KD_BIAS_VARIANTS,
    T6_CANDIDATES,
    get_variant,
    materialize_non_kd_bias_variant,
    materialize_t6_candidate,
    materialize_t7_variant,
)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--kd-components")
parser.add_argument("--base-variant")
parser.add_argument("--bias-type")
parser.add_argument("--magnitude-mode")
args = parser.parse_args(sys.argv[2:])
variant_id = sys.argv[1]
components = tuple(filter(None, (args.kd_components or "").split("+")))
if variant_id in T6_CANDIDATES or variant_id == "T6":
    variant = materialize_t6_candidate(
        variant_id, base_variant=args.base_variant, components=components,
    )
elif variant_id in KD_INHERITING_VARIANTS:
    variant = materialize_t7_variant(
        variant_id,
        base_variant=args.base_variant,
        kd_components=components,
        bias_type=args.bias_type,
    )
elif variant_id in NON_KD_BIAS_VARIANTS:
    variant = materialize_non_kd_bias_variant(
        variant_id,
        bias_type=args.bias_type,
        magnitude_mode=args.magnitude_mode,
        parent_variant=args.base_variant,
    )
else:
    variant = get_variant(variant_id)
print(variant.config_hash)
PY
}

require_checkpoint() {
    local artifact="$1"
    local stage="$2"
    local expected_epochs="$3"
    local expected_config_hash="$4"
    local checkpoint
    checkpoint="$(latest_valid_checkpoint "$artifact" "$stage" "$expected_epochs" "$expected_config_hash")"
    if [[ -z "$checkpoint" ]]; then
        log "ERROR: no valid strict checkpoint for $artifact" >&2
        exit 1
    fi
    printf '%s' "$checkpoint"
}

ensure_formal_log() {
    local run_dir="$1"
    local artifact="$2"
    local target="$run_dir/logs/formal.log"
    local source="$LOG_DIR/${artifact}.log"
    if [[ -s "$target" ]]; then
        return 0
    fi
    if [[ -s "$source" ]]; then
        cp "$source" "$target"
        return 0
    fi
    return 1
}

run_cli() {
    local variant="$1"
    local stage="$2"
    local expected_epochs="$3"
    shift 3
    local artifact
    artifact="$(artifact_name "$variant")"
    local expected_hash
    expected_hash="$(expected_config_hash "$variant" "$@")"
    log "START $variant stage=$stage epochs=$expected_epochs: independent FP-source attention-only fine-tuning $*"
    local command=(
        "$PY" -u -m binary_attention.cli run
        --variant "$variant"
        --stage "$stage"
        --data-manifest "$DATA_MANIFEST"
        --data "$DATA_YAML"
        --source-weights "$SOURCE_WEIGHTS"
        --device 0
        --execute
    )
    if [[ "$#" -gt 0 ]]; then
        command+=("$@")
    fi
    "${command[@]}" 2>&1 | tee "$LOG_DIR/${artifact}.log"
    "$PY" -u -m binary_attention.cli report > "$LOG_DIR/report-after-${artifact}.log" 2>&1
    local checkpoint
    local run_dir
    checkpoint="$(require_checkpoint "$artifact" "$stage" "$expected_epochs" "$expected_hash")"
    run_dir="$(dirname -- "$(dirname -- "$checkpoint")")"
    cp "$LOG_DIR/${artifact}.log" "$run_dir/logs/formal.log"
    log "DONE $variant stage=$stage checkpoint=$checkpoint"
}

run_if_needed() {
    local variant="$1"
    local stage="$2"
    local expected_epochs="$3"
    shift 3
    local artifact
    artifact="$(artifact_name "$variant")"
    local expected_hash
    expected_hash="$(expected_config_hash "$variant" "$@")"
    local checkpoint
    checkpoint="$(latest_valid_checkpoint "$artifact" "$stage" "$expected_epochs" "$expected_hash")"
    if [[ -n "$checkpoint" ]]; then
        local run_dir
        run_dir="$(dirname -- "$(dirname -- "$checkpoint")")"
        if ensure_formal_log "$run_dir" "$artifact"; then
            log "REUSE $variant stage=$stage checkpoint=$checkpoint"
            printf '%s' "$checkpoint"
            return
        fi
        log "REBUILD $variant: valid checkpoint has no recoverable formal log"
    fi
    run_cli "$variant" "$stage" "$expected_epochs" "$@" >/dev/null
    require_checkpoint "$artifact" "$stage" "$expected_epochs" "$expected_hash"
}

latest_valid_validation() {
    local artifact="$1"
    "$PY" - "$ROOT" "$artifact" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
artifact = sys.argv[2]
base = root / "artifacts" / "runs" / artifact / "validation"
valid = []
for run in base.iterdir() if base.exists() else ():
    try:
        status = json.loads((run / "status.json").read_text())
        resolved = json.loads((run / "resolved_config.json").read_text())
    except Exception:
        continue
    checkpoint = run / "checkpoints" / "strict-validation.pt"
    if (
        checkpoint.exists()
        and status.get("completed") is True
        and status.get("valid_for_research") is True
        and status.get("stage") == "validation"
        and resolved.get("id") == "E2-DUAL"
        and resolved.get("plan_name") == "YOLO11 BinaryAttention complete 10-epoch attention-only QAT plan"
    ):
        valid.append((checkpoint.stat().st_mtime_ns, run))
if valid:
    print(max(valid)[1])
PY
}

run_e2_validation_if_needed() {
    local run_dir
    run_dir="$(latest_valid_validation E2-DUAL)"
    if [[ -n "$run_dir" ]]; then
        if ensure_formal_log "$run_dir" E2-DUAL; then
            log "REUSE E2-DUAL validation run=$run_dir"
            return
        fi
        log "REBUILD E2-DUAL: valid validation has no recoverable formal log"
    fi
    log "START E2-DUAL zero-training matched dual-basis validation"
    "$PY" -u -m binary_attention.cli run \
        --variant E2-DUAL \
        --stage validation \
        --data-manifest "$DATA_MANIFEST" \
        --data "$DATA_YAML" \
        --source-weights "$SOURCE_WEIGHTS" \
        --device 0 \
        --execute 2>&1 | tee "$LOG_DIR/E2-DUAL.log"
    run_dir="$(latest_valid_validation E2-DUAL)"
    if [[ -z "$run_dir" ]]; then
        log "ERROR: E2-DUAL validation did not produce a valid artifact" >&2
        exit 1
    fi
    cp "$LOG_DIR/E2-DUAL.log" "$run_dir/logs/formal.log"
    log "DONE E2-DUAL validation run=$run_dir"
}

best_variant_artifact() {
    local stage="$1"
    shift
    "$PY" - "$ROOT" "$stage" "$@" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
stage = sys.argv[2]
artifacts = sys.argv[3:]
best = None
for artifact in artifacts:
    base = root / "artifacts" / "runs" / artifact.replace("/", "-").replace("+", "-") / stage
    latest = None
    for run in base.iterdir() if base.exists() else ():
        status_path = run / "status.json"
        resolved_path = run / "resolved_config.json"
        training_path = run / "training_args.json"
        metrics_path = run / "validation_metrics.json"
        checkpoint = run / "checkpoints" / "strict-last.pt"
        if not all(path.exists() for path in (status_path, resolved_path, training_path, metrics_path, checkpoint)):
            continue
        try:
            status = json.loads(status_path.read_text())
            resolved = json.loads(resolved_path.read_text())
            training = json.loads(training_path.read_text())
            metrics = json.loads(metrics_path.read_text())
            score = float(metrics["mAP50_95"])
        except Exception:
            continue
        expected_epochs = 10
        if (
            status.get("valid_for_research") is not True
            or status.get("completed") is not True
            or status.get("stage") != stage
            or training.get("epochs") != expected_epochs
            or training.get("trainable_scope") != "attention_only"
            or training.get("checkpoint_weight_source") != "epoch_ema"
            or training.get("metrics_weight_source") != "epoch_ema"
            or resolved.get("plan_name") != "YOLO11 BinaryAttention complete 10-epoch attention-only QAT plan"
        ):
            continue
        candidate = (score, run.stat().st_mtime_ns)
        if latest is None or candidate > latest:
            latest = candidate
    if latest is not None:
        candidate = (latest[0], latest[1], artifact)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
if best is None:
    raise SystemExit("no valid candidate with numeric mAP50_95")
print(best[2])
PY
}

best_t6_components() {
    "$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = {
    "T6-O": ("positional",),
    "T6-F": ("feature",),
    "T6-A": ("attention",),
    "T6-O/F": ("positional", "feature"),
    "T6-O/A": ("positional", "attention"),
    "T6-F/A": ("feature", "attention"),
}
best = None
for artifact, components in candidates.items():
    base = root / "artifacts" / "runs" / artifact.replace("/", "-") / "full"
    for run in base.iterdir() if base.exists() else ():
        try:
            status = json.loads((run / "status.json").read_text())
            training = json.loads((run / "training_args.json").read_text())
            score = float(json.loads((run / "validation_metrics.json").read_text())["mAP50_95"])
        except Exception:
            continue
        if (
            status.get("valid_for_research") is True
            and status.get("completed") is True
            and status.get("stage") == "full"
            and training.get("epochs") == 10
            and training.get("trainable_scope") == "attention_only"
        ):
            candidate = (score, run.stat().st_mtime_ns, components)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
if best is None:
    raise SystemExit("no valid T6 candidate with numeric mAP50_95")
print("+".join(best[2]))
PY
}

FORMAL_EPOCHS=10
FORMAL_STAGE=full
REPORT_DIR="$ROOT/artifacts/reports"

log "CORRECTED ATTENTION-ONLY QAT MATRIX START; T0/T1 reuse, continuation starts at T2"
run_e2_validation_if_needed

# T0/T1 were completed before this continuation.  Their strict EMA artifacts
# are reused; the resumed formal matrix starts at T2.
run_if_needed T0 "$FORMAL_STAGE" "$FORMAL_EPOCHS" >/dev/null
run_if_needed T1 "$FORMAL_STAGE" "$FORMAL_EPOCHS" >/dev/null
run_if_needed T2 "$FORMAL_STAGE" "$FORMAL_EPOCHS" >/dev/null

# Corrected T3/T4/T5 = original N1/N2/N3, with no KD.
for candidate in T3 T4 T5; do
    run_if_needed "$candidate" "$FORMAL_STAGE" "$FORMAL_EPOCHS" >/dev/null
done

# T6 is applied to the best completed T1--T5 branch.
best_t1_t5="$(best_variant_artifact "$FORMAL_STAGE" T1 T2 T3 T4 T5)"
log "SELECT best T1-T5 base=$best_t1_t5 for T6"
for candidate in T6-O T6-F T6-A 'T6-O/F' 'T6-O/A' 'T6-F/A'; do
    run_if_needed "$candidate" "$FORMAL_STAGE" "$FORMAL_EPOCHS" \
        --base-variant "$best_t1_t5" >/dev/null
done

selected_kd="$(best_t6_components)"
log "SELECT T6 components=$selected_kd by mAP50_95"
run_if_needed T6 "$FORMAL_STAGE" "$FORMAL_EPOCHS" \
    --base-variant "$best_t1_t5" --kd-components "$selected_kd" >/dev/null

for candidate in T7-D T7-R; do
    run_if_needed "$candidate" "$FORMAL_STAGE" "$FORMAL_EPOCHS" \
        --base-variant "$best_t1_t5" --kd-components "$selected_kd" >/dev/null
done
best_t7_bias_artifact="$(best_variant_artifact "$FORMAL_STAGE" T7-D T7-R)"
case "$best_t7_bias_artifact" in
    T7-D) selected_bias="dense_2d" ;;
    T7-R) selected_bias="decomposed_2d" ;;
    *) log "ERROR: unsupported selected T7 artifact=$best_t7_bias_artifact" >&2; exit 1 ;;
esac
log "SELECT best T7 bias=$best_t7_bias_artifact type=$selected_bias"

for candidate in T7-P T7-V T7-PV; do
    run_if_needed "$candidate" "$FORMAL_STAGE" "$FORMAL_EPOCHS" \
        --base-variant "$best_t1_t5" --kd-components "$selected_kd" --bias-type "$selected_bias" >/dev/null
done

# N4 parent selection considers T1--T5 and all completed T7 variants.  N4
# itself is always non-KD; only the selected bias provenance is inherited.
n4_parent="$(best_variant_artifact "$FORMAL_STAGE" T1 T2 T3 T4 T5 T7-D T7-R T7-P T7-V T7-PV)"
case "$n4_parent" in
    T7-D) n4_bias="dense_2d" ;;
    T7-R) n4_bias="decomposed_2d" ;;
    T7-P|T7-V|T7-PV) n4_bias="$selected_bias" ;;
    T1|T2|T3|T4|T5) n4_bias="none" ;;
    *) log "ERROR: unsupported N4 parent=$n4_parent" >&2; exit 1 ;;
esac
log "SELECT N4 parent=$n4_parent bias=$n4_bias; N4 KD disabled"
for candidate in N4-FP N4-I8 N4-I4; do
    run_if_needed "$candidate" "$FORMAL_STAGE" "$FORMAL_EPOCHS" \
        --base-variant "$n4_parent" --bias-type "$n4_bias" >/dev/null
done
best_n4_artifact="$(best_variant_artifact "$FORMAL_STAGE" N4-FP N4-I8 N4-I4)"
case "$best_n4_artifact" in
    N4-FP) selected_magnitude_mode="fp" ;;
    N4-I8) selected_magnitude_mode="int8" ;;
    N4-I4) selected_magnitude_mode="int4" ;;
    *) log "ERROR: unsupported selected N4 artifact=$best_n4_artifact" >&2; exit 1 ;;
esac
run_if_needed N4-PV "$FORMAL_STAGE" "$FORMAL_EPOCHS" \
    --base-variant "$n4_parent" --bias-type "$n4_bias" \
    --magnitude-mode "$selected_magnitude_mode" >/dev/null

final_candidates=(T7-D T7-R T7-P T7-V T7-PV N4-FP N4-I8 N4-I4 N4-PV)
selected_variant="$(best_variant_artifact "$FORMAL_STAGE" "${final_candidates[@]}")"
mkdir -p "$REPORT_DIR"
"$PY" - "$ROOT" "$selected_variant" "$best_t1_t5" "$selected_kd" "$best_t7_bias_artifact" "$selected_bias" "$n4_parent" "$best_n4_artifact" "$selected_magnitude_mode" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
selected, best_base, selected_kd, best_t7, selected_bias, n4_parent, best_n4, magnitude = sys.argv[2:]
candidates = ("T7-D", "T7-R", "T7-P", "T7-V", "T7-PV", "N4-FP", "N4-I8", "N4-I4", "N4-PV")
scores = {}
for variant in candidates:
    base = root / "artifacts" / "runs" / variant.replace("/", "-") / "full"
    values = []
    for run in base.iterdir() if base.exists() else ():
        try:
            status = json.loads((run / "status.json").read_text())
            training = json.loads((run / "training_args.json").read_text())
            metrics = json.loads((run / "validation_metrics.json").read_text())
            if (
                status.get("completed") is True
                and status.get("valid_for_research") is True
                and status.get("stage") == "full"
                and training.get("epochs") == 10
                and training.get("trainable_scope") == "attention_only"
                and training.get("checkpoint_weight_source") == "epoch_ema"
                and training.get("metrics_weight_source") == "epoch_ema"
            ):
                values.append((float(metrics["mAP50_95"]), run.stat().st_mtime_ns))
        except Exception:
            continue
    if values:
        scores[variant] = max(values, key=lambda item: (item[0], item[1]))[0]
payload = {
    "selection_metric": "mAP50_95",
    "selection_stage": "full",
    "epochs": 10,
    "trainable_scope": "attention_only",
    "selected_variant": selected,
    "selected_mAP50_95": scores.get(selected),
    "candidate_mAP50_95": scores,
    "selected_t6_base_variant": best_base,
    "selected_t6_components": selected_kd.split("+"),
    "selected_t7_artifact": best_t7,
    "selected_t7_bias_type": selected_bias,
    "selected_n4_parent": n4_parent,
    "selected_n4_artifact": best_n4,
    "selected_n4_magnitude_bits": {"fp": None, "int8": 8, "int4": 4}[magnitude],
    "t6_kd_target_family": "T1-T5",
    "n4_kd": False,
    "n4_non_kd_parent_family": "best T7+T1-T5, no KD",
    "initialization": "independent full-precision source checkpoint",
}
path = root / "artifacts" / "reports" / "paper_qat_selection.json"
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
print(path)
PY
"$PY" -u -m binary_attention.cli report > "$LOG_DIR/final-report.log" 2>&1
"$PY" -u -m binary_attention.cli audit > "$LOG_DIR/final-audit.json" 2>&1
log "CORRECTED MATRIX COMPLETE: selected=$selected_variant"
