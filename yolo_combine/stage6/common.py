"""固定父模型、參考來源與第六階段的輸出位置。"""

import json
import time
from pathlib import Path

from binaryqk.run import write_json
from yolo_attention.stage3_checkpoint import sha256

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "yolo_combine"
ARTIFACTS = PROJECT / "artifacts/stage6"
REVIEW = ARTIFACTS / "reference-review.json"


def read(path):
    return json.loads(Path(path).read_text())


def review():
    data = read(REVIEW)
    if not data["reviewed_before_implementation"]:
        raise ValueError("optimize review required before implementation")
    for path, digest in data["sources_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"reference changed: {path}")
    for key in ("detect_parent", "pose_parent", "detect_handoff", "pose_handoff"):
        if sha256(data[key]) != data[key + "_sha256"]:
            raise ValueError(f"parent/handoff changed: {key}")
    return data


def event(output, **data):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"time_unix": time.time(), **data}
    write_json(output / "progress.json", payload)
    with (output / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def source_guard(output, create=False):
    review()
    files = list(ARTIFACTS.glob("*review.json"))
    for location in (Path(output), Path(output).parent):
        protocol_path = location / "protocol.json"
        if protocol_path.exists():
            protocol = read(protocol_path)
            if not protocol.get("reviewed_before_implementation"):
                raise ValueError("local protocol lacks pre-implementation review")
            for name, digest in protocol["sources_sha256"].items():
                if sha256(ROOT / name) != digest:
                    raise ValueError(f"local reference changed: {name}")
            files.append(protocol_path)
    for directory in ("stage6", "src/yolo_combine"):
        files.extend((PROJECT / directory).rglob("*.py"))
    for directory in ("yolo_attention/src/yolo_attention", "yolo_binaryqk/binaryqk"):
        files.extend((ROOT / directory).rglob("*.py"))
    files.extend([ROOT / "yolo_masf/masf_yolo/stage4/evaluate.py"])
    files.extend([PROJECT / "run_stage6.sh"])
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    path = Path(output) / "source.json"
    if create:
        if path.exists():
            raise ValueError("cannot replace source lock")
        write_json(path, hashes)
    elif read(path) != hashes:
        raise ValueError("Stage6 code changed after run initialization")
