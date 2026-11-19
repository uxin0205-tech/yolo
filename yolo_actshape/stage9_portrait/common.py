"""直向候選的獨立來源鎖與具還原保證的引擎接線。"""

import copy
from contextlib import contextmanager

from stage9.common import ROOT, HERE, read, write, digest
from stage9_extension import common as previous

CONFIG = HERE / "portrait"
RUN = HERE / "artifacts/shape-portrait-20261112-v1"
RESULT = HERE / "result/stage9-portrait"
NAME = "rect768_512"


def protocol():
    return read(CONFIG / "protocol.json")


def guard(create=False):
    previous.guard()
    p, old = protocol(), previous.protocol()
    if not p["gpu_authorized"] or not p["reviewed_before_implementation"]:
        raise ValueError("本次尚未授權或未完成事前查核")
    for key in ("parent", "seed", "training", "calibration", "preprocessing", "hardware", "evaluator"):
        if p[key] != old[key]:
            raise ValueError(f"同配置契約改變：{key}")
    if p["candidates"] != {NAME: {"shape": [768, 512], "train": True}}:
        raise ValueError("只授權 768×512 單一候選")
    if p["training"]["epochs"] + p["training"]["low_lr_epochs"] != 20:
        raise ValueError("本次總預算必須為 20E")
    if digest(previous.RUN / "source.json") != p["dependency"]["source_sha256"]:
        raise ValueError("既有來源鎖改變")
    if read(previous.RUN / "status.json")["status"] != "complete":
        raise ValueError("前一輪尚未完成")
    values = dict(read(previous.RUN / "source.json"))
    files = sorted((HERE / "stage9_portrait").glob("*.py")) + [
        CONFIG / "protocol.json", CONFIG / "reference-review.json", CONFIG / "run_extension.sh",
        HERE / "run_portrait.sh", previous.RUN / "source.json"]
    values.update({str(path.relative_to(ROOT)): digest(path) for path in files})
    for name, expected in p["sources_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"本次參考來源改變：{name}")
        values[name] = expected
    path = RUN / "source.json"
    if create and not path.exists():
        write(path, values)
    elif read(path) != values:
        raise ValueError("新候選程式改變，不得混用來源續跑")


def stopped():
    return any(path.exists() for path in (RUN / "STOP", HERE / "STOP", previous.BASE_RUN / "STOP"))


@contextmanager
def training_run(calibration_recovery=False):
    from stage9 import training
    saved = training.RUN, training.guard, training.protocol
    p = protocol()
    if calibration_recovery:
        p = copy.deepcopy(p)
        p["calibration"].update({key: p["calibration_recovery"][key] for key in ("macros", "confirm", "attempts")})
    training.RUN, training.guard, training.protocol = RUN, guard, lambda: p
    try:
        yield training
    finally:
        training.RUN, training.guard, training.protocol = saved


@contextmanager
def queue_engine(state):
    """獨立程序內重用既有 supervisor；只改候選來源、輸出及啟動入口。"""
    from stage9_extension import runner
    replacements = dict(RUN=RUN, HERE=CONFIG, protocol=protocol, guard=guard,
                        stopped=stopped, state=state, training_run=training_run)
    saved = {key: getattr(runner, key) for key in replacements}
    for key, value in replacements.items():
        setattr(runner, key, value)
    try:
        yield runner
    finally:
        for key, value in saved.items():
            setattr(runner, key, value)
