"""擴充實驗獨立來源鎖，不改動正在執行的初輪。"""

from contextlib import contextmanager
from pathlib import Path

from stage9 import common as base
from stage9.common import ROOT, HERE, read, write, digest, event, score, rank

RUN = HERE / "artifacts/shape-expanded-20261108-v1"
CONFIG = HERE / "extension"
BASE_RUN = base.RUN


def protocol():
    return read(CONFIG / "protocol.json")


def guard(create=False):
    base.guard()
    p, original = protocol(), base.protocol()
    if not p["gpu_authorized"] or not p["reviewed_before_implementation"]:
        raise ValueError("擴充尚未完成事前核對")
    for name in ("parent", "seed", "training", "calibration", "preprocessing", "hardware", "evaluator"):
        if p[name] != original[name]:
            raise ValueError(f"既有引擎介面設定不符：{name}")
    if digest(BASE_RUN / "source.json") != p["dependency"]["source_sha256"]:
        raise ValueError("初輪來源鎖變動")
    values = dict(read(BASE_RUN / "source.json"))
    files = sorted((HERE / "stage9_extension").glob("*.py")) + [
        CONFIG / "protocol.json", CONFIG / "reference-review.json", HERE / "run_extension.sh"]
    values.update({str(f.relative_to(ROOT)): digest(f) for f in files})
    for name, expected in p["sources_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"參考來源改變：{name}")
        values[name] = expected
    values[str((BASE_RUN / "source.json").relative_to(ROOT))] = digest(BASE_RUN / "source.json")
    path = RUN / "source.json"
    if create and not path.exists():
        write(path, values)
    elif read(path) != values:
        raise ValueError("擴充來源改變，拒絕混用程式續跑")


@contextmanager
def training_run():
    """只在獨立 worker 內指定引擎的輸出根目錄與來源驗證；結束時還原。"""
    from stage9 import training
    original = training.RUN, training.guard
    training.RUN, training.guard = RUN, guard
    try:
        yield training
    finally:
        training.RUN, training.guard = original


def stopped():
    return any(p.exists() for p in (RUN / "STOP", BASE_RUN / "STOP", HERE / "STOP"))
