from __future__ import annotations

import pytest
import torch

from yolo_quantize import SD4Encoding


def test_sd4_decodes_all_sixteen_bit_patterns_and_both_zero_codes() -> None:
    encoding = SD4Encoding()
    codes = torch.arange(16)

    hardware = encoding.decode(codes, normalized=False)
    normalized = encoding.decode(codes, normalized=True)

    assert hardware.tolist() == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        0,
        -1,
        -2,
        -4,
        -8,
        -16,
        -32,
        -64,
        0,
    ]
    assert torch.equal(normalized, hardware / 64)
    assert encoding.positive_zero_code == 0b0111
    assert encoding.negative_zero_code == 0b1111
    assert encoding.canonical_zero_code == encoding.positive_zero_code


def test_sd4_encode_uses_canonical_zero_and_rejects_non_codebook_values() -> None:
    encoding = SD4Encoding()
    values = torch.tensor(
        [0.0, 1 / 64, -1 / 64, 1.0, -1.0],
        dtype=torch.float32,
    )

    codes = encoding.encode(values, normalized=True)

    assert codes.tolist() == [7, 0, 8, 6, 14]
    with pytest.raises(ValueError, match="not exact SD4"):
        encoding.encode(torch.tensor([0.3]), normalized=True)


def test_sd4_pack_unpack_roundtrip_preserves_nibbles_and_odd_count() -> None:
    encoding = SD4Encoding()
    codes = torch.arange(16)

    payload = encoding.pack(codes)
    restored = encoding.unpack(payload, count=16)
    odd_payload = encoding.pack(codes[:5])
    odd_restored = encoding.unpack(odd_payload, count=5)

    assert payload == bytes.fromhex("0123456789abcdef")
    assert torch.equal(restored, codes)
    assert odd_payload == bytes.fromhex("012347")
    assert torch.equal(odd_restored, codes[:5])
