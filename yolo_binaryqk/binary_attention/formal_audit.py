"""Fail-closed audit for the 10-epoch attention-only BinaryAttention matrix."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

from .variants.definitions import get_variant, quantization_contract, variant_from_resolved_config


PLAN_NAME = "YOLO11 BinaryAttention complete 10-epoch attention-only QAT plan"
LEGACY_PLAN_NAME = "YOLO11 BinaryAttention simplified full-COCO plan"
E_VARIANTS = ("E0", "E1-S", "E1", "E2-DUAL")
FORMAL_VARIANTS = (
    "T0", "T1", "T2",
    "T3", "T4", "T5",
    "T6-O", "T6-F", "T6-A", "T6-O/F", "T6-O/A", "T6-F/A", "T6",
    "T7-D", "T7-R", "T7-P", "T7-V", "T7-PV",
    "N4-FP", "N4-I8", "N4-I4", "N4-PV",
)
SELECTION_CANDIDATES = (
    "T7-D", "T7-R", "T7-P", "T7-V", "T7-PV",
    "N4-FP", "N4-I8", "N4-I4", "N4-PV",
)
T6_CANDIDATES = ("T6-O", "T6-F", "T6-A", "T6-O/F", "T6-O/A", "T6-F/A")
T7_CANDIDATES = ("T7-D", "T7-R", "T7-P", "T7-V", "T7-PV")
N4_CANDIDATES = ("N4-FP", "N4-I8", "N4-I4")
EXPECTED_EPOCHS = {"validation": 0, "full": 10}
VARIANT_FIELDS = (
    "id", "attention_type", "qk_mode", "use_qat", "use_distillation",
    "distillation_type", "kd_components", "bias_type", "p_bits", "v_bits",
    "magnitude_bits", "num_binary_qk", "num_softmax", "num_pv", "base_variant",
    "kd_target_family",
)


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_rows(path: Path) -> list:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_name(variant: str) -> str:
    return variant.replace("/", "-").replace("+", "-")


def _run_score(run: Path | None) -> float | None:
    if run is None:
        return None
    value = _read(run / "validation_metrics.json").get("mAP50_95")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _actual_scores(root: Path, variants: tuple[str, ...]) -> dict[str, float]:
    scores = {}
    for variant in variants:
        score = _run_score(_latest_valid_run(root, variant, "full"))
        if score is not None:
            scores[variant] = score
    return scores


def _latest_valid_run(root: Path, variant: str, stage: str) -> Path | None:
    base = root / "artifacts" / "runs" / _artifact_name(variant) / stage
    candidates = []
    for run in base.iterdir() if base.exists() else ():
        status = _read(run / "status.json")
        resolved = _read(run / "resolved_config.json")
        accepted_plans = {PLAN_NAME}
        if variant in E_VARIANTS:
            accepted_plans.add(LEGACY_PLAN_NAME)
        if (
            status.get("completed") is True
            and status.get("valid_for_research") is True
            and resolved.get("plan_name") in accepted_plans
            and (resolved.get("plan_name") == PLAN_NAME or stage == "validation")
            and resolved.get("id") == variant
        ):
            candidates.append((run.stat().st_mtime_ns, run))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _check_run(root: Path, variant: str, stage: str) -> tuple[Path | None, list[str]]:
    run = _latest_valid_run(root, variant, stage)
    if run is None:
        return None, [f"{variant}: no completed valid {stage} artifact"]

    status = _read(run / "status.json")
    resolved = _read(run / "resolved_config.json")
    training = _read(run / "training_args.json")
    architecture = _read(run / "architecture_manifest.json")
    checkpoint_manifest = _read(run / "checkpoint_manifest.json")
    diagnostics = _read(run / "attention_diagnostics.json")
    parameter_deltas = _read(run / "parameter_delta_diagnostics.json")
    environment = _read(run / "environment.json")
    checkpoint_name = "strict-validation.pt" if stage == "validation" else "strict-last.pt"
    required = (
        "experiment_report.md",
        "environment.json",
        "resolved_config.json",
        "training_args.json",
        "architecture_manifest.json",
        "checkpoint_manifest.json",
        "validation_metrics.json",
        "attention_diagnostics.json",
        "training_curves.csv",
        "logs",
        "checkpoints",
    )
    missing = [name for name in required if not (run / name).exists()]
    checkpoint = run / "checkpoints" / checkpoint_name
    if not checkpoint.exists():
        missing.append(f"checkpoints/{checkpoint_name}")
    if status.get("checkpoint_reload_verified") is not True:
        missing.append("status.checkpoint_reload_verified=true")
    if checkpoint_manifest.get("variant_id") != variant:
        missing.append("checkpoint_manifest.variant_id")
    if architecture.get("variant_id") != variant:
        missing.append("architecture_manifest.variant_id")
    if diagnostics.get("variant_id") != variant:
        missing.append("attention_diagnostics.variant_id")
    if not isinstance(environment.get("git_revision"), str) or len(environment.get("git_revision", "")) != 40:
        missing.append("environment.git_revision")
    if environment.get("cuda_available") is not True:
        missing.append("environment.cuda_available=true")
    if environment.get("gpu") in (None, "", "unavailable"):
        missing.append("environment.gpu")
    if not environment.get("torch") or not environment.get("ultralytics"):
        missing.append("environment framework versions")
    metrics = _read(run / "validation_metrics.json")
    score = metrics.get("mAP50_95")
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        missing.append("validation_metrics.mAP50_95")
    if status.get("stage") != stage:
        missing.append(f"status.stage={stage}")
    expected_manifest = str((root / "data" / "coco_full.txt").resolve())
    if status.get("data_manifest") != expected_manifest or training.get("data_manifest") != expected_manifest:
        missing.append("full COCO data_manifest provenance")
    if status.get("epochs") != EXPECTED_EPOCHS[stage] or training.get("epochs") != EXPECTED_EPOCHS[stage]:
        missing.append(f"epochs={EXPECTED_EPOCHS[stage]}")

    try:
        expected_variant = variant_from_resolved_config(resolved)
    except ValueError:
        missing.append("resolved_config.kd_components")
        expected_variant = get_variant(variant)
    for key in VARIANT_FIELDS:
        actual = resolved.get(key)
        expected = getattr(expected_variant, key)
        if isinstance(expected, tuple):
            actual, expected = tuple(actual or ()), expected
        if actual != expected:
            missing.append(f"resolved_config.{key}={expected}")
    expected_hash = expected_variant.config_hash
    if resolved.get("config_hash") != expected_hash:
        missing.append("resolved_config.config_hash")
    if checkpoint_manifest.get("variant_config_hash") != expected_hash:
        missing.append("checkpoint_manifest.variant_config_hash")
    if architecture.get("variant_config_hash") != expected_hash:
        missing.append("architecture_manifest.variant_config_hash")
    expected_cost = expected_variant.theoretical_cost
    if architecture.get("theoretical_cost") != expected_cost:
        missing.append("architecture_manifest.theoretical_cost")
    if diagnostics.get("theoretical_cost") != expected_cost:
        missing.append("attention_diagnostics.theoretical_cost")
    diagnostic_modules = diagnostics.get("modules")
    if not isinstance(diagnostic_modules, list):
        diagnostic_modules = []
        missing.append("attention_diagnostics.modules")
    if expected_variant.qk_mode == "fp":
        if diagnostic_modules or checkpoint_manifest.get("attention_module_count") != 0:
            missing.append("FP control contains binary attention diagnostics")
    else:
        if not diagnostic_modules:
            missing.append("binary attention diagnostics are empty")
        if checkpoint_manifest.get("attention_module_count") != len(diagnostic_modules):
            missing.append("checkpoint/diagnostic attention module counts disagree")
        if architecture.get("attention_module_count") != len(diagnostic_modules):
            missing.append("architecture/diagnostic attention module counts disagree")
        if checkpoint_manifest.get("binary_forward_count", 0) <= 0:
            missing.append("checkpoint binary_forward_count")
        for module in diagnostic_modules:
            forwards = module.get("binary_forward_count")
            if not isinstance(forwards, int) or forwards <= 0:
                missing.append("diagnostic binary_forward_count")
                continue
            for counter, cost_key in (
                ("binary_qk_count", "binary_qk"),
                ("softmax_count", "softmax"),
                ("pv_count", "pv"),
            ):
                if module.get(counter) != forwards * expected_cost[cost_key]:
                    missing.append(f"diagnostic {counter} does not match theoretical cost")
            if module.get("score_finite") is not True or module.get("probability_finite") is not True:
                missing.append("non-finite binary attention score/probability")
    expected_fake_quant = bool(expected_variant.p_bits or expected_variant.v_bits or expected_variant.magnitude_bits)
    if diagnostics.get("fake_quantization") is not expected_fake_quant:
        missing.append("attention_diagnostics.fake_quantization")
    if diagnostics.get("hardware_speed_claim") is not False:
        missing.append("attention_diagnostics.hardware_speed_claim=false")
    if stage == "full":
        if expected_variant.qk_mode != "fp":
            expected_contract = quantization_contract(expected_variant)
            if resolved.get("quantization_contract") != expected_contract:
                missing.append("resolved_config.quantization_contract")
            if architecture.get("quantization_contract") != expected_contract:
                missing.append("architecture_manifest.quantization_contract")
            if checkpoint_manifest.get("quantization_contract") != expected_contract:
                missing.append("checkpoint_manifest.quantization_contract")
        if not (run / "parameter_delta_diagnostics.json").exists():
            missing.append("parameter_delta_diagnostics.json")
        if parameter_deltas.get("variant_id") != variant or parameter_deltas.get("passed") is not True:
            missing.append("attention-only parameter delta proof")
        if parameter_deltas.get("max_frozen_parameter_delta") != 0.0:
            missing.append("frozen parameter tensors changed from FP source")
        if parameter_deltas.get("max_frozen_non_attention_bn_buffer_delta") != 0.0:
            missing.append("frozen non-attention BN buffers changed from FP source")
        if not isinstance(parameter_deltas.get("changed_source_attention_tensor_count"), int) or parameter_deltas.get("changed_source_attention_tensor_count") <= 0:
            missing.append("no source-initialized Attention tensor changed")
        formal_log = run / "logs" / "formal.log"
        if not formal_log.exists() or formal_log.stat().st_size == 0:
            missing.append("logs/formal.log")
        expected_profile = {
            "batch": 128,
            "requested_batch": 128,
            "micro_batch": 16,
            "effective_batch": 128,
            "gradient_accumulation_steps": 8,
            "imgsz": 640,
            "seed": 0,
            "deterministic": True,
            "amp": True,
            "workers": 8,
            "cache": "disk",
            "optimizer": "AdamW",
            "lr0": 5e-5,
            "lrf": 0.1,
            "weight_decay": 0.02,
            "cos_lr": True,
            "warmup_epochs": 0.0,
        }
        for key, expected in expected_profile.items():
            if training.get(key) != expected:
                missing.append(f"training_args.{key}={expected}")
        if status.get("finetune_checkpoint") != status.get("source_weights"):
            missing.append("FP-source fine-tune provenance")
        if status.get("checkpoint_weight_source") != "epoch_ema" or training.get("checkpoint_weight_source") != "epoch_ema":
            missing.append("strict checkpoint must use epoch EMA weights")
        if status.get("metrics_weight_source") != "epoch_ema" or training.get("metrics_weight_source") != "epoch_ema":
            missing.append("metrics/checkpoint EMA provenance")
        if status.get("ema_checkpoint_epoch") != 10 or training.get("ema_checkpoint_epoch") != 10:
            missing.append("epoch-last EMA checkpoint provenance")
        if status.get("trainable_scope") != "attention_only" or training.get("trainable_scope") != "attention_only":
            missing.append("attention-only trainable scope")
        names = training.get("trainable_parameter_names")
        if not isinstance(names, list) or not names or any(".attn." not in name for name in names):
            missing.append("trainable parameters restricted to .attn. modules")
        trainable_count = training.get("trainable_parameter_count")
        frozen_count = training.get("frozen_parameter_count")
        if not isinstance(trainable_count, int) or trainable_count <= 0:
            missing.append("positive trainable_parameter_count")
        if not isinstance(frozen_count, int) or frozen_count <= trainable_count:
            missing.append("majority non-attention parameters frozen")
        if not isinstance(training.get("frozen_non_attention_batchnorm_count"), int) or training.get("frozen_non_attention_batchnorm_count") <= 0:
            missing.append("non-attention BatchNorm statistics frozen")
        if architecture.get("trainable_scope") != "attention_only":
            missing.append("architecture_manifest.trainable_scope=attention_only")
        if isinstance(names, list) and sorted(names) != sorted(architecture.get("trainable_parameter_names") or []):
            missing.append("training/architecture trainable parameter names disagree")
        if trainable_count != architecture.get("trainable_parameter_count"):
            missing.append("training/architecture trainable parameter counts disagree")
        if variant != "T0" and resolved.get("use_qat") is not True:
            missing.append("QAT enabled")
    if variant == "E2-DUAL":
        if resolved.get("qk_mode") != "dual" or resolved.get("use_qat") is not False:
            missing.append("zero-training dual-basis configuration")
        if (resolved.get("num_binary_qk"), resolved.get("num_softmax"), resolved.get("num_pv")) != (2, 1, 1):
            missing.append("matched dual-basis theoretical cost 2/1/1")
    if variant in {"T3", "T4", "T5"}:
        if resolved.get("use_distillation") is not False or tuple(resolved.get("kd_components") or ()):
            missing.append(f"{variant} must be non-KD")
    if variant in set(T6_CANDIDATES) | {"T6"} | set(T7_CANDIDATES):
        components = tuple(resolved.get("kd_components") or ())
        allowed = {"positional", "feature", "attention"}
        if not components or not set(components).issubset(allowed):
            missing.append(f"{variant} selected KD components")
        if resolved.get("kd_target_family") != "T1-T5":
            missing.append(f"{variant} KD target family T1-T5")
    if variant == "T7-PV":
        if resolved.get("use_qat") is not True:
            missing.append("T7-PV QAT enabled")
        if resolved.get("use_distillation") is not True:
            missing.append("T7-PV KD enabled")
        if resolved.get("p_bits") != 8 or resolved.get("v_bits") != 8:
            missing.append("T7-PV P8/V8 fake quantization")
    return run, [f"{variant}: {item}" for item in missing]


def audit_formal_plan(root: Path) -> dict:
    """Return a machine-readable audit; ``ok`` is false for any missing proof."""

    errors: list[str] = []
    runs: dict[str, str] = {}
    manifest_path = root / "data" / "coco_full.txt"
    manifest_meta = _read(root / "data" / "coco_full.json")
    try:
        manifest_count = sum(1 for _line in manifest_path.open())
        manifest_hash = _sha256(manifest_path)
    except OSError:
        manifest_count, manifest_hash = 0, "unavailable"
    if (
        manifest_count != 118287
        or manifest_meta.get("count") != 118287
        or manifest_meta.get("seed") != 0
        or manifest_meta.get("sha256") != manifest_hash
    ):
        errors.append("data/coco_full manifest integrity/count/seed")
    train_images = root.parent / "coco2017" / "images" / "train2017"
    val_images = root.parent / "coco2017" / "images" / "val2017"
    if not train_images.exists() or not val_images.exists():
        errors.append("COCO2017 train2017/val2017 image roots")
    for variant in E_VARIANTS:
        run, run_errors = _check_run(root, variant, "validation")
        errors.extend(run_errors)
        if run is not None:
            runs[variant] = str(run.relative_to(root))
    for variant in FORMAL_VARIANTS:
        run, run_errors = _check_run(root, variant, "full")
        errors.extend(run_errors)
        if run is not None:
            runs[variant] = str(run.relative_to(root))

    source_hashes = {
        _read(root / relative / "checkpoint_manifest.json").get("source_checkpoint_hash")
        for relative in runs.values()
    }
    if len(source_hashes) != 1 or None in source_hashes or "unavailable" in source_hashes:
        errors.append("all artifacts must share one available FP source checkpoint hash")

    reports = root / "artifacts" / "reports"
    selection = _read(reports / "paper_qat_selection.json")
    selected_variant = selection.get("selected_variant")
    if selection.get("selection_metric") != "mAP50_95":
        errors.append("reports/paper_qat_selection.json selection_metric")
    if selection.get("selection_stage") != "full":
        errors.append("reports/paper_qat_selection.json selection_stage")
    if selection.get("epochs") != 10:
        errors.append("reports/paper_qat_selection.json epochs")
    if selection.get("trainable_scope") != "attention_only":
        errors.append("reports/paper_qat_selection.json trainable_scope")
    if selection.get("initialization") != "independent full-precision source checkpoint":
        errors.append("reports/paper_qat_selection.json initialization")

    actual_final_scores = _actual_scores(root, SELECTION_CANDIDATES)
    if selected_variant not in SELECTION_CANDIDATES:
        errors.append("reports/paper_qat_selection.json selected_variant")
    else:
        candidate_scores = selection.get("candidate_mAP50_95")
        if not isinstance(candidate_scores, dict) or not all(
            isinstance(candidate_scores.get(variant), (int, float))
            and math.isfinite(candidate_scores[variant])
            for variant in SELECTION_CANDIDATES
        ):
            errors.append("reports/paper_qat_selection.json candidate_mAP50_95")
        else:
            if any(candidate_scores.get(variant) != actual_final_scores.get(variant) for variant in SELECTION_CANDIDATES):
                errors.append("reports/paper_qat_selection.json scores disagree with run artifacts")
            selected_score = candidate_scores.get(selected_variant)
            if selection.get("selected_mAP50_95") != selected_score:
                errors.append("reports/paper_qat_selection.json selected variant/score disagree")
            if selected_score != max(candidate_scores[variant] for variant in SELECTION_CANDIDATES):
                errors.append("reports/paper_qat_selection.json selected score is not maximum")

    selected_t6_components = tuple(selection.get("selected_t6_components") or ())
    selected_t6_base = selection.get("selected_t6_base_variant")
    selected_t6_run = _latest_valid_run(root, "T6", "full")
    actual_t6_components = tuple(_read(selected_t6_run / "resolved_config.json").get("kd_components") or ()) if selected_t6_run else ()
    actual_t6_base = _read(selected_t6_run / "resolved_config.json").get("base_variant") if selected_t6_run else None
    if not selected_t6_components or selected_t6_components != actual_t6_components or selected_t6_base != actual_t6_base:
        errors.append("reports/paper_qat_selection.json selected T6 base/components")
    t6_scores = _actual_scores(root, T6_CANDIDATES)
    component_scores = {}
    for variant, score in t6_scores.items():
        resolved = _read(_latest_valid_run(root, variant, "full") / "resolved_config.json")
        component_scores[tuple(resolved.get("kd_components") or ())] = score
    if selected_t6_components not in component_scores or (
        component_scores and component_scores.get(selected_t6_components) != max(component_scores.values())
    ):
        errors.append("reports/paper_qat_selection.json T6 selection is not maximum")

    for key, candidates in (
        ("selected_t7_artifact", ("T7-D", "T7-R")),
        ("selected_n4_artifact", N4_CANDIDATES),
    ):
        chosen = selection.get(key)
        scores = _actual_scores(root, candidates)
        if chosen not in candidates or chosen not in scores or (scores and scores[chosen] != max(scores.values())):
            errors.append(f"reports/paper_qat_selection.json {key}")

    selected_t7 = selection.get("selected_t7_artifact")
    selected_n4 = selection.get("selected_n4_artifact")
    selected_t7_run = _latest_valid_run(root, selected_t7, "full") if selected_t7 in ("T7-D", "T7-R") else None
    selected_n4_run = _latest_valid_run(root, selected_n4, "full") if selected_n4 in N4_CANDIDATES else None
    selected_bias = _read(selected_t7_run / "resolved_config.json").get("bias_type") if selected_t7_run else None
    selected_magnitude = _read(selected_n4_run / "resolved_config.json").get("magnitude_bits") if selected_n4_run else None
    if selection.get("selected_t7_bias_type") != selected_bias:
        errors.append("reports/paper_qat_selection.json selected_t7_bias_type")
    if selection.get("selected_n4_magnitude_bits") != selected_magnitude:
        errors.append("reports/paper_qat_selection.json selected_n4_magnitude_bits")

    for variant in T7_CANDIDATES:
        run = _latest_valid_run(root, variant, "full")
        resolved = _read(run / "resolved_config.json") if run else {}
        if tuple(resolved.get("kd_components") or ()) != selected_t6_components:
            errors.append(f"{variant}: does not inherit selected T6 KD components")
        if variant in {"T7-P", "T7-V", "T7-PV"} and resolved.get("bias_type") != selected_bias:
            errors.append(f"{variant}: does not inherit selected T7 bias")

    for variant in ("N4-FP", "N4-I8", "N4-I4", "N4-PV"):
        run = _latest_valid_run(root, variant, "full")
        resolved = _read(run / "resolved_config.json") if run else {}
        if resolved.get("use_distillation") is not False or tuple(resolved.get("kd_components") or ()):
            errors.append(f"{variant}: N4 must be non-KD")
        if resolved.get("kd_target_family") is not None:
            errors.append(f"{variant}: N4 unexpectedly carries T3 KD target family")
        if resolved.get("bias_type") not in {"none", "dense_2d", "decomposed_2d"}:
            errors.append(f"{variant}: invalid non-KD bias")
    n4_pv_run = _latest_valid_run(root, "N4-PV", "full")
    n4_pv_resolved = _read(n4_pv_run / "resolved_config.json") if n4_pv_run else {}
    if n4_pv_resolved.get("magnitude_bits") != selected_magnitude:
        errors.append("N4-PV: does not inherit selected N4 magnitude mode")
    if selection.get("t6_kd_target_family") != "T1-T5":
        errors.append("reports/paper_qat_selection.json T6 KD target family")
    if selection.get("n4_kd") is not False or selection.get("n4_non_kd_parent_family") != "best T7+T1-T5, no KD":
        errors.append("reports/paper_qat_selection.json N4 non-KD provenance")

    required_reports = (
        "binary_attention_final_report.md",
        "binary_attention_summary.json",
        "binary_attention_summary.csv",
        "run_reports_index.md",
        "paper_qat_selection.json",
    )
    for name in required_reports:
        if not (reports / name).exists() or (reports / name).stat().st_size == 0:
            errors.append(f"reports/{name}")
    final_report = reports / "binary_attention_final_report.md"
    try:
        final_text = final_report.read_text()
    except OSError:
        final_text = ""
    for required_text in (
        "10-epoch attention-only adaptation",
        "300-epoch full-model",
        "E2-DUAL",
        "mean_abs_channel_token_per_sample_head",
        "epoch-last EMA",
        "fake quantization",
        "T6",
        "N4",
    ):
        if required_text not in final_text:
            errors.append(f"final report missing disclosure: {required_text}")
    required_figures = (
        "training_loss_map.png", "variant_accuracy.png", "t3_kd_comparison.png",
        "t4_comparison.png", "t5_comparison.png", "n_series_comparison.png",
    )
    for name in required_figures:
        path = reports / "figures" / name
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"reports/figures/{name}")
    summary_rows = _read_rows(reports / "binary_attention_summary.json")
    if len(summary_rows) != len(E_VARIANTS) + len(FORMAL_VARIANTS):
        errors.append("summary must contain exactly one canonical row for each of 26 variants")
    summary_variants = {str(row.get("variant")) for row in summary_rows if isinstance(row, dict)}
    summary_stages = {
        (str(row.get("variant")), str(row.get("stage")))
        for row in summary_rows if isinstance(row, dict)
    }
    if len(summary_stages) != len(summary_rows):
        errors.append("summary contains duplicate variant/stage rows")
    expected = set(E_VARIANTS) | set(FORMAL_VARIANTS)
    for variant in sorted(expected - summary_variants):
        errors.append(f"summary missing variant {variant}")
    for variant in FORMAL_VARIANTS:
        if (variant, "full") not in summary_stages:
            errors.append(f"summary missing full stage {variant}")
    expected_stages = {variant: "validation" for variant in E_VARIANTS}
    expected_stages.update({variant: "full" for variant in FORMAL_VARIANTS})
    for variant, stage in expected_stages.items():
        expected_run = runs.get(variant)
        matching = [
            row for row in summary_rows
            if isinstance(row, dict)
            and row.get("variant") == variant
            and row.get("stage") == stage
            and row.get("run") == expected_run
        ]
        if not matching:
            errors.append(f"summary does not reference audited run {variant}")
            continue
        run_path = root / expected_run if expected_run else None
        expected_score = _run_score(run_path)
        if matching[-1].get("mAP50_95") != expected_score:
            errors.append(f"summary mAP50_95 disagrees with audited run {variant}")

    return {
        "ok": not errors,
        "plan_name": PLAN_NAME,
        "expected_variant_count": len(expected),
        "expected_research_run_count": len(E_VARIANTS) + len(FORMAL_VARIANTS),
        "verified_variant_count": len(runs),
        "selected_best_variant": selected_variant,
        "runs": runs,
        "errors": errors,
    }
