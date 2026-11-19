"""固定來源與原子化進度；不修改 optimize 或已完成階段。"""

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "yolo_activation/stage7"
ARTIFACTS = ROOT / "yolo_activation/artifacts/stage7"
CANDIDATES = ("qsilu_pq", "hardswish", "poly_shift", "poly_quality")
REGIONS = (
    "backbone_early",
    "backbone_late",
    "neck",
    "attention",
    "detect_head",
    "pose_head",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def protocol(verify=False):
    result = read(HERE / "protocol.json")
    if (
        not result["reviewed_before_implementation"]
        or tuple(result["candidates"]) != CANDIDATES
    ):
        raise ValueError("缺少事前 optimize 核對或候選改變")
    if verify:
        for name, digest in result["sources_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"參考來源改變：{name}")
        if sha256(result["parent"]["checkpoint"]) != result["parent"]["sha256"]:
            raise ValueError("Stage6 parent SHA 改變")
    return result


def event(output, **data):
    output = Path(output)
    payload = {"time_unix": time.time(), **data}
    write_json(output / "progress.json", payload)
    with (output / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def source_guard(root, create=False):
    protocol(verify=True)
    files = [
        HERE / "protocol.json",
        HERE / "preflight.json",
        ROOT / "yolo_activation/run_stage7.sh",
    ]
    for directory in (
        "yolo_activation/stage7",
        "yolo_activation/src/activation_lab",
        "yolo_combine/stage6",
        "yolo_combine/src/yolo_combine",
        "yolo_attention/src/yolo_attention",
        "yolo_binaryqk/binaryqk",
        "yolo_masf/masf_yolo/stage4",
    ):
        files.extend((ROOT / directory).rglob("*.py"))
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))}
    path = Path(root) / "source.json"
    if create and not path.exists():
        write_json(path, hashes)
    elif read(path) != hashes:
        raise ValueError("執行期間來源鎖改變，禁止帶新配方續跑")


def baseline():
    return protocol()["parent"]["metrics"]


def passed(values):
    return all(values[k] >= v - 0.015 for k, v in baseline().items())


def score(values):
    coco = ("coco_AP", "person_AP", "sports_ball_AP", "baseball_bat_AP")
    return 0.5 * (
        sum(values[k] for k in coco) / 4
        + sum(values[k] for k in baseline() if k not in coco) / 6
    )


def rank(values):
    return passed(values), score(values)


def recipe(name, extension=False):
    result = protocol()["training"].copy()
    result["name"] = name
    if extension:
        result.update(epochs=5, loss_horizon=5)
        result["lrs"] = {k: v * 0.5 for k, v in result["lrs"].items()}
    return result
