from __future__ import annotations

from pathlib import Path

import pytest
import masf_yolo.variants as variant_registry

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
    assert tuple(VARIANTS) == (
        "B1", "M7", "M0", "M1", "M2", "M3", "P3M", "SP2", "SP2M2", "SP2M3"
    )
    assert get_variant("B1").p2_slot == "identity"
    assert get_variant("M0").kernel_branches == (3, 5, 7, 9)
    assert get_variant("M7").kernel_branches == (3, 5, 7)
    assert get_variant("M1").kernel_branches == (3, 5)
    assert get_variant("M2").processed_ratio == 0.5
    assert get_variant("M3").processed_ratio == 0.25
    assert get_variant("P3M").p2_slot == "identity"
    assert get_variant("P3M").p3_slot == "mfam"
    assert get_variant("P3M").kernel_branches == (3, 5, 7)
    assert get_variant("SP2").p2_slot == "identity"
    assert all(
        variant.p3_slot == "identity"
        for name, variant in VARIANTS.items()
        if name != "P3M"
    )


def test_variant_hash_is_stable_and_unique() -> None:
    first = {key: value.config_hash for key, value in VARIANTS.items()}
    second = {key: get_variant(key).config_hash for key in VARIANTS}

    assert first == second
    assert len(set(first.values())) == 10
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
    assert TRAINED_VARIANTS == ("B1", "M7", "M0", "M1", "M2", "M3", "P3M", "SP2")
    assert EVALUATED_MODELS == (
        "B0", "B1", "M7", "M0", "M1", "M2", "M3", "P3M", "SP2", "SP2P"
    )
    assert SELECTION_CANDIDATES == ("M2", "M3")


def test_sp2p_internal_variants_are_static_partial_selective_models() -> None:
    assert hasattr(variant_registry, "sp2p_variant_id")
    assert hasattr(variant_registry, "is_selective_variant")
    sp2p_variant_id = variant_registry.sp2p_variant_id
    is_selective_variant = variant_registry.is_selective_variant

    assert sp2p_variant_id("M2") == "SP2M2"
    assert sp2p_variant_id("M3") == "SP2M3"
    with pytest.raises(ValueError, match="M2 or M3"):
        sp2p_variant_id("M1")

    for parent, ratio in (("M2", 0.5), ("M3", 0.25)):
        definition = get_variant(sp2p_variant_id(parent))
        assert definition.kernel_branches == (3, 5)
        assert definition.processed_ratio == ratio
        assert definition.p2_slot == "partial_mfam"
        assert is_selective_variant(definition.variant_id)


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
    assert digest == "c62ae5f4f6d2eb3c2837989fd01c9c8e1567a7a0401f612744dda242bafecbed"
