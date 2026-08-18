"""Evaluation backend that materializes the evaluated variant and provenance."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone

import torch
import ultralytics

from .artifacts import _git_revision, sha256_file
from .config import VariantConfig
from .evaluation import (
    BDCNDiagnosticsCollector,
    EvaluationRequest,
    UltralyticsEvaluationBackend,
    collect_row_sum_max_error,
    standardize_metrics,
    write_standard_result,
)
from .integration import convert_yolo26_model, fixed_scale_modules


class PWLFinalEvaluationBackend(UltralyticsEvaluationBackend):
    """Save Float/Bit-True conversion before validation and report that checkpoint."""

    @staticmethod
    def _write_manifest(request: EvaluationRequest, variant: VariantConfig) -> None:
        run = request.run_dir
        run.mkdir(parents=True, exist_ok=False)
        ultra_path = ultralytics.__path__[0]
        payload = {
            "schema_version": 2,
            "run_id": request.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kind": "full-coco-evaluation",
            "variant": variant.to_dict(),
            "parent": {
                "path": str(request.parent_checkpoint.resolve()),
                "sha256": sha256_file(request.parent_checkpoint),
            },
            "dataset": {"path": request.recipe.data, "yaml_sha256": sha256_file(request.recipe.data)},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "git_revision": _git_revision(),
                "ultralytics_version": ultralytics.__version__,
                "ultralytics_source_path": ultra_path,
                "ultralytics_git_revision": _git_revision(ultra_path),
            },
        }
        (run / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def evaluate_variant(self, request: EvaluationRequest):
        if request.variant_path is None:
            raise ValueError("variant evaluation requires variant_path")
        variant = VariantConfig.from_yaml(request.variant_path)
        self._write_manifest(request, variant)
        model = self._make_model(request.parent_checkpoint)
        convert_yolo26_model(model.model, variant)
        if fixed_scale_modules(model.model):
            raise RuntimeError("final workflow requires calibrated PoT coefficients in its parent")
        checkpoint = request.run_dir / "checkpoints" / "evaluated-variant.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(checkpoint))
        with BDCNDiagnosticsCollector(model.model) as collector:
            metrics = standardize_metrics(model.val(**self._validation_args(request)))
        return write_standard_result(
            request.run_dir,
            metrics,
            checkpoint_path=checkpoint,
            profile_path=None,
            row_sum_max_error=collect_row_sum_max_error(model.model),
            bdcn_diagnostics=collector.result(),
        )
