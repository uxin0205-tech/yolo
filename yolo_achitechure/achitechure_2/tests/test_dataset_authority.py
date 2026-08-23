from __future__ import annotations

from pathlib import Path

import yaml

from achitechure_2.config import PROJECT_ROOT

REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
ASSET_ROOT = Path("/home/uxin/yolo/original/pose")
CANONICAL_ROOT = ASSET_ROOT / "derived/bbat5-v1"


def test_repository_has_one_bbat5_asset_root() -> None:
    registry_path = REPOSITORY_ROOT / "configs/datasets/bbat5-v1.yaml"
    registry = yaml.safe_load(
        registry_path.read_text(encoding="utf-8")
    )

    assert Path(registry["root"]) == CANONICAL_ROOT
    assert registry["policy"]["publication_root"] == str(ASSET_ROOT)
    assert registry["policy"]["project_dataset_copies"] == "forbidden"
    assert not (PROJECT_ROOT / "artifacts/datasets/bbat5-v1").exists()


def test_active_bbat5_configs_never_use_artifacts_datasets() -> None:
    active_configs = [
        *sorted((PROJECT_ROOT / "configs/data").glob("bbat5-*.yaml")),
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

    assert 'run_root / "cache-views" / "bbat5-v1"' in source
