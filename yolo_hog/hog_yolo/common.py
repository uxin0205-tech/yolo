"""來源、parent 與研究設定；改變時明確失敗。"""

import json
from pathlib import Path

from binaryqk.run import write_json
from yolo_attention.stage3_checkpoint import sha256

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "yolo_hog"
ARTIFACTS = PROJECT / "artifacts/stage5"
REVIEW = ARTIFACTS / "reference-review.json"


def read(path):
    return json.loads(Path(path).read_text())


def review():
    data = read(REVIEW)
    if not data.get("reviewed_before_implementation"):
        raise ValueError("optimize review required before implementation")
    for path, digest in data["sources_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"optimize reference changed: {path}")
    for key in ("parent", "parent_handoff", "ema_source_resume"):
        digest_key = (
            "ema_source_sha256" if key == "ema_source_resume" else key + "_sha256"
        )
        if sha256(data[key]) != data[digest_key]:
            raise ValueError(f"HOG source changed: {key}")
    return data


def source_guard(output, create=False):
    review()
    files = [
        p
        for name in (
            "yolo_hog/hog_yolo",
            "yolo_binaryqk/binaryqk",
            "yolo_attention/src/yolo_attention",
        )
        for p in (ROOT / name).rglob("*.py")
    ]
    files += [REVIEW, ARTIFACTS / "cpu-tests.json", ARTIFACTS / "coverage.json"]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    path = Path(output) / "source.json"
    if create:
        if (
            not read(ARTIFACTS / "cpu-tests.json")["passed"]
            or not read(ARTIFACTS / "coverage.json")["passed"]
        ):
            raise ValueError("CPU tests and coverage required")
        write_json(path, hashes)
    elif hashes != read(path):
        raise ValueError("Stage5 source/protocol changed after initialization")


def emit(output, event):
    import time

    write_json(Path(output) / "progress.json", {"time_unix": time.time(), **event})
