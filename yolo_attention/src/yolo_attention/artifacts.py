"""Immutable run directory creation and provenance capture."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics

from .config import VariantConfig
from .run_config import TrainingRecipe


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def create_run(
        self,
        run_id: str,
        variant: VariantConfig,
        training: TrainingRecipe,
    ) -> Path:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", run_id):
            raise ValueError("run_id must use lowercase letters, digits, dot, underscore, or dash")
        run = self.root / run_id
        run.mkdir(parents=True, exist_ok=False)
        for name in ("checkpoints", "metrics", "profiles", "exports", "logs"):
            (run / name).mkdir()
        variant.to_yaml(run / "variant.yaml")
        training.to_yaml(run / "training.yaml")
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "variant": variant.to_dict(),
            "training": training.to_dict(),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "git_revision": _git_revision(),
            },
        }
        (run / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return run
