from __future__ import annotations

import yaml

from activation_lab.training import ActivationManifest, ActivationSite, load_manifest


def test_inspected_manifest_can_round_trip_through_contract_yaml(tmp_path) -> None:
    manifest = ActivationManifest(
        model_id="model-with-source-hash",
        model_source_sha256="a" * 64,
        reviewed=False,
        sites=(ActivationSite("model.0.act", "early", cost_weight=1024.0),),
    )
    path = tmp_path / "activation-manifest.yaml"
    path.write_text(
        yaml.safe_dump(manifest.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    assert load_manifest(path) == manifest
