from __future__ import annotations

from pathlib import Path

from masf_yolo.runtime import pipeline_identity, verify_environment


def test_environment_verification_matches_all_pinned_local_assets() -> None:
    config_path = Path("configs/static-phase1.yaml").resolve()

    manifest = verify_environment(config_path, require_cuda=False)

    assert manifest.ultralytics == "8.4.90"
    assert manifest.faster_coco_eval == "1.7.2"
    assert manifest.source_weights_hash == "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"
    assert manifest.official_model_yaml_hash == "43d8a7c86acc77282ddf4966d6526e091e5b064c140ed4a6e4c0b68ecbc3784c"
    assert manifest.official_default_yaml_hash == "f6e19ab7228826341cc6370bcc8c78d10478d2bd604e25a13d9975b5efc9b410"


def test_pipeline_identity_changes_with_any_research_input_hash() -> None:
    first = pipeline_identity("config-a", "data-a", "environment-a")

    assert first == pipeline_identity("config-a", "data-a", "environment-a")
    assert first != pipeline_identity("config-b", "data-a", "environment-a")
    assert first != pipeline_identity("config-a", "data-b", "environment-a")
    assert len(first) == 12
