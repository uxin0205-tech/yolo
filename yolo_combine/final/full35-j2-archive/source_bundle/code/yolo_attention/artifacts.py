"""建立包含 reproducibility provenance 的 immutable run。"""

from __future__ import annotations

import hashlib
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(path or Path.cwd()), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _dataset_provenance(recipe: TrainingRecipe) -> dict[str, str | None]:
    path = Path(recipe.data).resolve()
    return {"path": str(path), "yaml_sha256": sha256_file(path) if path.is_file() else None}


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def create_run(self, run_id: str, variant: VariantConfig, training: TrainingRecipe) -> Path:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", run_id):
            raise ValueError("invalid run_id")
        run = self.root / run_id
        run.mkdir(parents=True, exist_ok=False)
        for name in ("checkpoints", "metrics", "profiles", "exports", "logs"):
            (run / name).mkdir()
        variant.to_yaml(run / "variant.yaml")
        training.to_yaml(run / "training.yaml")
        ultra_path = Path(ultralytics.__file__).resolve().parent
        weights = Path(training.weights).resolve()
        manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "variant": variant.to_dict(),
            "training": training.to_dict(),
            "dataset": _dataset_provenance(training),
            "parent": {"path": str(weights), "sha256": sha256_file(weights)},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "git_revision": _git_revision(),
                "ultralytics_version": ultralytics.__version__,
                "ultralytics_source_path": str(ultra_path),
                "ultralytics_git_revision": _git_revision(ultra_path),
            },
        }
        (run / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return run
