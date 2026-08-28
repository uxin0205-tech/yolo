from __future__ import annotations

from pathlib import Path

import yaml

from achitechure_2.config import PROJECT_ROOT

REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
ASSET_ROOT = Path("/home/uxin/yolo/original/pose")
CANONICAL_ROOT = ASSET_ROOT / "derived/bbat5-v1"
PORTABLE_SNAPSHOT = Path(
    "yolo_achitechure/achitechure_2/artifacts/datasets/bbat5-v1"
)


def test_repository_uses_canonical_bbat5_and_declares_portable_snapshot() -> None:
    registry_path = REPOSITORY_ROOT / "configs/datasets/bbat5-v1.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))

    assert Path(registry["root"]) == CANONICAL_ROOT
    assert registry["policy"]["publication_root"] == str(ASSET_ROOT)
    assert registry["policy"]["project_dataset_copies"] == "forbidden"
    assert Path(registry["policy"]["superseded_snapshot"]) == PORTABLE_SNAPSHOT
    assert (REPOSITORY_ROOT / PORTABLE_SNAPSHOT / "github-dataset").is_dir()


def test_active_bbat5_configs_never_use_artifacts_datasets() -> None:
    active_configs = [
        *[
            path
            for path in sorted((PROJECT_ROOT / "configs/data").glob("bbat5-*.yaml"))
            if "screen-20" not in path.name
        ],
        *sorted((REPOSITORY_ROOT / "yolo_combine/configs/data").glob("bbt5-*.yaml")),
        *sorted((REPOSITORY_ROOT / "yolo_combine/variants").glob("*/variant.yaml")),
    ]

    assert active_configs
    for path in active_configs:
        text = path.read_text(encoding="utf-8")
        assert "artifacts/datasets" not in text
        assert str(ASSET_ROOT) in text


def test_yolo_combine_runtime_views_are_cache_only() -> None:
    source_path = REPOSITORY_ROOT / "yolo_combine/src/yolo_combine/variants.py"
    source = source_path.read_text(encoding="utf-8")

    assert 'return self.run_root / "datasets" / "bbat5-v1-runtime"' in source
