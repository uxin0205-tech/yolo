"""沿用前階段 evaluator；完整 D1 部署路徑評估。"""

import gc
from pathlib import Path

import torch
from binaryqk.evaluate import evaluate as detect_evaluate
from masf_yolo.stage4.evaluate import evaluate_pose

from .common import event, review, write_json
from .data import dataset
from .models import manifest_for, view


def vector(detect, pose):
    d, p = detect["official"], pose["metrics"]
    return {
        "coco_AP": d["AP50_95"],
        "person_AP": d["classes"]["person"]["AP50_95"],
        "sports_ball_AP": d["classes"]["sports_ball"]["AP50_95"],
        "baseball_bat_AP": d["classes"]["baseball_bat"]["AP50_95"],
        **{
            k: p[k]
            for k in (
                "box_AP50_95",
                "pose_AP50_95",
                "ball_box_AP50_95",
                "ball_pose_AP50_95",
                "bat_box_AP50_95",
                "bat_pose_AP50_95",
            )
        },
    }


def expected():
    data = review()
    d, p = data["detect_expected"], data["pose_expected"]
    return {
        "coco_AP": d["AP50_95"],
        "person_AP": d["person_AP50_95"],
        "sports_ball_AP": d["sports_ball_AP50_95"],
        "baseball_bat_AP": d["baseball_bat_AP50_95"],
        **{
            k: p[k]
            for k in (
                "box_AP50_95",
                "pose_AP50_95",
                "ball_box_AP50_95",
                "ball_pose_AP50_95",
                "bat_box_AP50_95",
                "bat_pose_AP50_95",
            )
        },
    }


COCO_KEYS = ("coco_AP", "person_AP", "sports_ball_AP", "baseball_bat_AP")


def accepted(values):
    base = expected()
    return all(
        values[k] >= value - (0.005 if k in COCO_KEYS else 0.02)
        for k, value in base.items()
    )


def score(values):
    return (values["box_AP50_95"] + values["pose_AP50_95"]) / 2


def pair(model, output, progress_root=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    callback = lambda e: event(progress_root or output, **e)
    manifest_for(model)
    model.eval()
    ds, data, _ = dataset("detect", "val", 8)
    det = detect_evaluate(
        view(model, "detect"), ds, data, output / "detect", progress=callback
    )
    write_json(output / "detect.json", det)
    del ds
    ds, data, _ = dataset("pose", "val", 8)
    pose = evaluate_pose(
        view(model, "pose"), ds, data, output / "pose", progress=callback
    )
    write_json(output / "pose.json", pose)
    result = {
        "detect": det,
        "pose": pose,
        "metrics": vector(det, pose),
        "accepted": accepted(vector(det, pose)),
    }
    write_json(output / "metrics.json", result)
    manifest_for(model)
    del ds
    gc.collect()
    torch.cuda.empty_cache()
    return result
