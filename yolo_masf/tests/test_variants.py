from __future__ import annotations

from pathlib import Path

import pytest

from masf_yolo.variants import (
    CORE_VARIANTS,
    EVALUATED_MODELS,
    PRIORITY_VARIANTS,
    SELECTION_CANDIDATES,
    TRAINED_VARIANTS,
    VARIANTS,
    get_variant,
    load_priority_manifest,
)


def test_phase1_variant_contracts_are_locked() -> None:
    assert tuple(VARIANTS) == ("B1", "M7", "M0", "M1", "M2", "M3")
    assert get_variant("B1").p2_slot == "identity"
    assert get_variant("M0").kernel_branches == (3, 5, 7, 9)
    assert get_variant("M7").kernel_branches == (3, 5, 7)
    assert get_variant("M1").kernel_branches == (3, 5)
    assert get_variant("M2").processed_ratio == 0.5
    assert get_variant("M3").processed_ratio == 0.25
    assert all(variant.p3_slot == "identity" for variant in VARIANTS.values())


def test_variant_hash_is_stable_and_unique() -> None:
    first = {key: value.config_hash for key, value in VARIANTS.items()}
    second = {key: get_variant(key).config_hash for key in VARIANTS}

    assert first == second
    assert len(set(first.values())) == 6
    assert all(len(value) == 64 for value in first.values())


def test_unknown_variant_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported variant"):
        get_variant("S0")


def test_m7_is_priority_full_channel_357_variant() -> None:
    m7 = get_variant("M7")

    assert m7.kernel_branches == (3, 5, 7)
    assert m7.processed_ratio == 1.0
    assert m7.p2_slot == "mfam"
    assert m7.config_hash not in {get_variant("M0").config_hash, get_variant("M1").config_hash}
    assert CORE_VARIANTS == ("B1", "M0", "M1", "M2", "M3")
    assert PRIORITY_VARIANTS == ("M7",)
    assert TRAINED_VARIANTS == ("B1", "M7", "M0", "M1", "M2", "M3")
    assert EVALUATED_MODELS == ("B0", "B1", "M7", "M0", "M1", "M2", "M3")
    assert SELECTION_CANDIDATES == ("M2", "M3")


def test_m7_priority_manifest_is_strict_and_matches_registry(tmp_path) -> None:
    manifest = load_priority_manifest(Path("configs/m7-priority.yaml"))

    assert manifest.variant_id == "M7"
    assert manifest.kernels == (3, 5, 7)
    assert manifest.processed_ratio == 1.0
    assert manifest.smoke_epochs == 3
    assert manifest.formal_epochs == 100
    assert manifest.priority_before == "M0"
    assert len(manifest.manifest_hash) == 64

    invalid = tmp_path / "m7.yaml"
    invalid.write_text(Path("configs/m7-priority.yaml").read_text() + "unknown: true\n")
    with pytest.raises(ValueError, match="unknown M7 priority keys"):
        load_priority_manifest(invalid)


def test_static_phase1_config_bytes_remain_locked() -> None:
    import hashlib

    digest = hashlib.sha256(Path("configs/static-phase1.yaml").read_bytes()).hexdigest()
    assert digest == "c5cb4c82e063702a657e82e61db7f8b4f1884ee911506cff3d6dd73022294802"
