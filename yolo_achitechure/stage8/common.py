"""固定 parent、參考來源與本階段預算。"""

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "yolo_achitechure/stage8"
ARTIFACTS = ROOT / "yolo_achitechure/artifacts/stage8"
CANDIDATES = ("rep17", "rep20", "rep17_20", "rep17_20_silu", "acb17_20")


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


def protocol(verify=False):
    value = read(HERE / "protocol.json")
    if (
        not value["reviewed_before_implementation"]
        or tuple(value["candidates"]) != CANDIDATES
    ):
        raise ValueError("未核對參考來源或候選改變")
    if verify:
        for path, digest in value["sources_sha256"].items():
            if sha256(ROOT / path) != digest:
                raise ValueError(f"參考來源改變：{path}")
        if sha256(value["parent"]["checkpoint"]) != value["parent"]["sha256"]:
            raise ValueError("qSiLU 共同 parent SHA 改變")
    return value


def source_guard(root, create=False):
    protocol(verify=True)
    files = [
        HERE / "cpu-contracts.json",
        HERE / "reference-review.json",
        HERE / "protocol.json",
        ROOT / "yolo_achitechure/run_stage8.sh",
    ]
    for folder in (
        "yolo_achitechure/stage8",
        "yolo_activation/stage7",
        "yolo_activation/src/activation_lab",
        "yolo_combine/stage6",
        "yolo_combine/src/yolo_combine",
        "yolo_attention/src/yolo_attention",
        "yolo_binaryqk/binaryqk",
        "yolo_masf/masf_yolo/stage4",
        "yolo_p2/ultralytics",
    ):
        files.extend((ROOT / folder).rglob("*.py"))
    values = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))}
    path = Path(root) / "source.json"
    if create and not path.exists():
        write_json(path, values)
    elif read(path) != values:
        raise ValueError("來源鎖改變，禁止帶新實作續跑")


def event(output, **data):
    output = Path(output)
    payload = {"time_unix": time.time(), **data}
    write_json(output / "progress.json", payload)
    with (output / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def baseline():
    return protocol()["parent"]["metrics"]


def passed(values):
    return all(
        values[k] >= v - protocol()["selection"]["relative_drop_limit"]
        for k, v in baseline().items()
    )


def score(values):
    coco = ("coco_AP", "person_AP", "sports_ball_AP", "baseball_bat_AP")
    return 0.5 * (
        sum(values[k] for k in coco) / 4
        + sum(values[k] for k in baseline() if k not in coco) / 6
    )


def rank(values):
    return passed(values), score(values)


def recipe(name, extension=False):
    value = protocol()["training"].copy()
    value["operators"] = protocol()["candidate_specs"][name]["operators"]
    value["name"] = name + ("-low-lr" if extension else "")
    if extension:
        value["lrs"] = {k: v * 0.5 for k, v in value["lrs"].items()}
    return value


def require_gpu_authorized():
    value = protocol()
    if value.get("gpu_launch_enabled") is not True:
        raise RuntimeError("GPU 啟動已停止：" + value.get("hold_reason", "尚未固定可執行配方"))
