from __future__ import annotations

import pytest

from masf_yolo.variants import VARIANTS, get_variant


def test_phase1_variant_contracts_are_locked() -> None:
    assert tuple(VARIANTS) == ("B1", "M0", "M1", "M2", "M3")
    assert get_variant("B1").p2_slot == "identity"
    assert get_variant("M0").kernel_branches == (3, 5, 7, 9)
    assert get_variant("M1").kernel_branches == (3, 5)
    assert get_variant("M2").processed_ratio == 0.5
    assert get_variant("M3").processed_ratio == 0.25
    assert all(variant.p3_slot == "identity" for variant in VARIANTS.values())


def test_variant_hash_is_stable_and_unique() -> None:
    first = {key: value.config_hash for key, value in VARIANTS.items()}
    second = {key: get_variant(key).config_hash for key in VARIANTS}

    assert first == second
    assert len(set(first.values())) == 5
    assert all(len(value) == 64 for value in first.values())


def test_unknown_variant_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported variant"):
        get_variant("S0")
