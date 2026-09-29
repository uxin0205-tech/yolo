from __future__ import annotations

import pytest

from achitechure_1.training import (
    RUNTIME_MIN_AVAILABLE_RAM_BYTES,
    RUNTIME_MIN_FREE_VRAM_BYTES,
    _assert_resource_floor,
)


def test_runtime_resource_floor_accepts_safe_snapshot() -> None:
    assert RUNTIME_MIN_AVAILABLE_RAM_BYTES == 1 << 29
    _assert_resource_floor(
        {
            "ram_available_bytes": RUNTIME_MIN_AVAILABLE_RAM_BYTES,
            "cuda_available": True,
            "vram_free_bytes": RUNTIME_MIN_FREE_VRAM_BYTES,
        }
    )


@pytest.mark.parametrize(
    ("snapshot", "message"),
    (
        (
            {
                "ram_available_bytes": RUNTIME_MIN_AVAILABLE_RAM_BYTES - 1,
                "cuda_available": False,
            },
            "RAM",
        ),
        (
            {
                "ram_available_bytes": RUNTIME_MIN_AVAILABLE_RAM_BYTES,
                "cuda_available": True,
                "vram_free_bytes": RUNTIME_MIN_FREE_VRAM_BYTES - 1,
            },
            "VRAM",
        ),
    ),
)
def test_runtime_resource_floor_fails_closed(snapshot: dict[str, object], message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        _assert_resource_floor(snapshot)
