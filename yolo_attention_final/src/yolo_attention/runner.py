"""有明確 gate 的訓練啟動器；建立 request 不產生 side effect。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from .artifacts import ArtifactStore
from .config import VariantConfig
from .run_config import TrainingRecipe
from .training import make_trainer


@dataclass(frozen=True)
class TrainingRequest:
    variant: VariantConfig
    training: TrainingRecipe
    artifacts_root: Path
    run_id: str
    project_root: Path

    @classmethod
    def from_files(
        cls,
        *,
        variant_path: str | Path,
        training_path: str | Path,
        artifacts_root: str | Path,
        run_id: str,
    ) -> TrainingRequest:
        variant_path = Path(variant_path).resolve()
        training_path = Path(training_path).resolve()
        common_root = Path.cwd().resolve()
        return cls(
            variant=VariantConfig.from_yaml(variant_path),
            training=TrainingRecipe.from_yaml(training_path),
            artifacts_root=Path(artifacts_root).resolve(),
            run_id=run_id,
            project_root=common_root,
        )

    def summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "variant": self.variant.name,
            "basis": self.variant.basis.value,
            "stage": self.training.stage,
            "epochs": self.training.epochs,
            "weights": self.training.weights,
            "data": self.training.data,
            "artifacts_root": str(self.artifacts_root),
            "will_execute": False,
        }


def launch_training(
    request: TrainingRequest,
    *,
    model_factory: Callable[[str], Any] = YOLO,
) -> Any:
    """建立 immutable provenance，再把控制權交給 Ultralytics。"""

    run = ArtifactStore(request.artifacts_root).create_run(
        request.run_id,
        request.variant,
        request.training,
    )
    args = request.training.to_ultralytics_args()
    data = Path(args["data"])
    if not data.is_absolute():
        args["data"] = str((request.project_root / data).resolve())
    args.update(project=str(run), name="ultralytics", exist_ok=True)
    model = model_factory(request.training.weights)
    result = model.train(
        trainer=make_trainer(
            request.variant, stage=request.training.stage, layer_lrs=request.training.layer_lrs
        ),
        **args,
    )
    marker = {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "requested_epochs": request.training.epochs,
        "patience": request.training.patience,
    }
    (run / "training-complete.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
