"""Bit-true SD4 nibble mapping and deterministic packing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import Tensor


class SD4Encoding:
    """Encode the thesis SD4 value set without losing duplicate-zero identity."""

    positive_zero_code: ClassVar[int] = 0b0111
    negative_zero_code: ClassVar[int] = 0b1111
    canonical_zero_code: ClassVar[int] = positive_zero_code
    _HARDWARE_VALUES: ClassVar[tuple[int, ...]] = (
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
    )

    def code_map(self) -> Mapping[int, int]:
        """Return the immutable 4-bit-code to hardware-value contract."""

        return dict(enumerate(self._HARDWARE_VALUES))

    def decode(self, codes: Tensor, *, normalized: bool) -> Tensor:
        """Decode integer nibbles to hardware or normalized training values."""

        integer_codes = self._validated_codes(codes)
        table = torch.tensor(
            self._HARDWARE_VALUES,
            dtype=torch.float32,
            device=integer_codes.device,
        )
        decoded = table[integer_codes]
        return decoded / 64.0 if normalized else decoded

    def encode(self, values: Tensor, *, normalized: bool) -> Tensor:
        """Encode exact SD4 values, using positive zero as canonical zero."""

        if not values.dtype.is_floating_point and values.dtype != torch.bool:
            candidates = values.to(torch.float32)
        elif values.dtype == torch.bool:
            raise TypeError("SD4 values must be numeric, not boolean")
        else:
            candidates = values
        if not bool(torch.isfinite(candidates).all()):
            raise ValueError("SD4 values must be finite")
        table = torch.tensor(
            self._HARDWARE_VALUES,
            dtype=candidates.dtype,
            device=candidates.device,
        )
        if normalized:
            table = table / 64.0
        distances = (candidates[..., None] - table).abs()
        minimum, codes = distances.min(dim=-1)
        tolerance = torch.finfo(candidates.dtype).eps * 8
        if not bool((minimum <= tolerance).all()):
            raise ValueError("one or more values are not exact SD4 codebook values")
        return codes.to(torch.int64)

    def pack(self, codes: Tensor) -> bytes:
        """Pack two codes per byte, first code in the high nibble."""

        flat = self._validated_codes(codes).reshape(-1).cpu()
        if flat.numel() % 2:
            flat = torch.cat(
                (
                    flat,
                    flat.new_tensor([self.canonical_zero_code]),
                )
            )
        packed = (flat[0::2] << 4) | flat[1::2]
        return bytes(packed.tolist())

    def unpack(self, payload: bytes, *, count: int) -> Tensor:
        """Unpack high-nibble-first bytes and remove canonical odd padding."""

        if count < 0 or count > len(payload) * 2:
            raise ValueError("SD4 unpack count exceeds payload capacity")
        packed = torch.tensor(list(payload), dtype=torch.int64)
        codes = torch.empty(len(payload) * 2, dtype=torch.int64)
        codes[0::2] = packed >> 4
        codes[1::2] = packed & 0x0F
        if (
            count % 2
            and count < codes.numel()
            and int(codes[count].item()) != self.canonical_zero_code
        ):
            raise ValueError("odd SD4 payload does not use canonical zero padding")
        return codes[:count]

    @staticmethod
    def _validated_codes(codes: Tensor) -> Tensor:
        if codes.dtype == torch.bool or codes.dtype.is_floating_point:
            raise TypeError("SD4 codes must use an integer tensor dtype")
        integer_codes = codes.to(torch.int64)
        if not bool(((0 <= integer_codes) & (integer_codes <= 15)).all()):
            raise ValueError("SD4 codes must be 4-bit integers in [0,15]")
        return integer_codes
