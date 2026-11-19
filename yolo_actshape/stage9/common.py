"""來源鎖、選型與階段內預算；不更動上游研究。"""

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "yolo_actshape"
RUN = HERE / "artifacts/shape-20261108-v1"
PARENT_ROOT = ROOT / "yolo_achitechure/result/stage8-retrain"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def protocol():
    return read(HERE / "protocol.json")


def guard(create=False):
    p = protocol()
    if not p["reviewed_before_implementation"] or not p["gpu_authorized"]:
        raise ValueError("未固定參考與授權")
    if digest(ROOT / p["parent"]["checkpoint"]) != p["parent"]["sha256"]:
        raise ValueError("最終 qSiLU Rep parent SHA 改變")
    upstream = read(ROOT / "yolo_achitechure/artifacts/stage8/repconv-variants-20261107/source.json")
    for name, sha in upstream.items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"上游鎖定程式改變：{name}")
    files = sorted((HERE / "stage9").glob("*.py")) + [
        HERE / "protocol.json", HERE / "reference-review.json", HERE / "run_stage9.sh"
    ]
    values = {**upstream, **{str(f.relative_to(ROOT)): digest(f) for f in files}}
    for name, sha in p["sources_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"參考來源改變：{name}")
        values[name] = sha
    path = RUN / "source.json"
    if create and not path.exists():
        write(path, values)
    elif read(path) != values:
        raise ValueError("本輪來源改變；不可混用程式續跑")


def event(output, **values):
    payload = {"time_unix": time.time(), **values}
    write(Path(output) / "progress.json", payload)
    with (Path(output) / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def score(metrics):
    keys = list(protocol()["parent"]["metrics"])
    return 0.5 * (sum(metrics[k] for k in keys[:4]) / 4 + sum(metrics[k] for k in keys[4:]) / 6)


def rank(metrics):
    base = protocol()["parent"]["metrics"]
    return (all(metrics[k] >= v - 0.001 for k, v in base.items()), score(metrics))


def shape_for(name):
    return protocol()["candidates"][name]["shape"]
