"""Final bundle 共用路徑與雜湊工具。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = BUNDLE_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def load_models() -> list[dict[str, Any]]:
    """讀取交付包中的候選 registry。"""

    return json.loads((BUNDLE_ROOT / "models.json").read_text(encoding="utf-8"))["models"]


def file_sha256(path: Path) -> str:
    """以串流方式計算檔案 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """以 replace 寫入 JSON，避免留下半成品。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
