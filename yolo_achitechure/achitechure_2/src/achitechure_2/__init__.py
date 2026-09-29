"""yolo_combine winner 的 C0–C3 簡化、資料與驗證工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .lite_c3k2 import KernelMode, LiteC3k2, LiteC3k2Config

__all__ = ["KernelMode", "LiteC3k2", "LiteC3k2Config"]
__version__ = "2.3.0"


def __getattr__(name: str) -> Any:
    """延後載入 torch，讓 GPU queue 能先驗證 grant 再公開 CUDA。"""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .lite_c3k2 import KernelMode, LiteC3k2, LiteC3k2Config

    exports = {
        "KernelMode": KernelMode,
        "LiteC3k2": LiteC3k2,
        "LiteC3k2Config": LiteC3k2Config,
    }
    value = exports[name]
    globals()[name] = value
    return value
