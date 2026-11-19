from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_SRC = ROOT.parents[1] / "yolo_attention_final" / "src"

for path in (ROOT / "src", PARENT_SRC):
    sys.path.insert(0, str(path))
