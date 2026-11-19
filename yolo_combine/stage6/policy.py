"""融合前／後的驗收分開；不同階段的 LR 直接取參考配方。"""

import copy
import math

from .common import review
from .evaluate import COCO_KEYS, accepted, expected, score


def recipe_for(name, extension=0):
    protocol = review()["protocol"]
    recipe = copy.deepcopy(protocol["phases"][name])
    recipe.update(name=name, extension=extension)
    if extension:
        factor = protocol["extensions"][name][extension - 1]
        recipe["lrs"] = {k: v * factor for k, v in recipe["lrs"].items()}
        recipe.update(epochs=20, loss_horizon=20, extension_lr_factor=factor)
    return recipe


def eligible(metrics):
    base = expected()
    return all(metrics[k] >= base[k] - 0.005 for k in COCO_KEYS)


def rank(metrics, recipe):
    if recipe["scope"] == "pose_full":
        return (score(metrics),)
    return (accepted(metrics), selection_score(metrics, recipe))


def selection_score(metrics, recipe):
    if recipe.get("selection") == "balanced_recovery":
        coco = sum(metrics[k] for k in COCO_KEYS) / len(COCO_KEYS)
        pose = [metrics[k] for k in expected() if k not in COCO_KEYS]
        return 0.5 * (coco + sum(pose) / len(pose))
    return score(metrics)


def safety(metrics, parent, recipe, epoch):
    if "recovery_safety" in recipe:
        limits = recipe["recovery_safety"]
        original = expected()
        if any(
            metrics[k] < parent[k] - limits["coco_from_parent"]
            or metrics[k] < original[k] - limits["coco_from_original"]
            for k in COCO_KEYS
        ):
            return "恢復候選 COCO 較起點進一步退化"
        if any(
            metrics[k] < parent[k] - limits["pose_from_parent"]
            for k in original
            if k not in COCO_KEYS
        ):
            return "恢復候選 BBAT 較起點進一步退化"
        return None
    limits = review()["protocol"]["training_safety"]
    if recipe["scope"] == "head":
        if any(metrics[k] != parent[k] for k in COCO_KEYS):
            return "固定共享層與 Detect 的完整 AP 改變"
        if "head_recovery_max_pose_drop" in recipe and any(
            metrics[k] < parent[k] - recipe["head_recovery_max_pose_drop"]
            for k in expected()
            if k not in COCO_KEYS
        ):
            return "Pose head 恢復較起點退化超過安全線"
        # optimize J0/H1/M0 use head-only plateau. The -0.03 BBAT
        # stop belongs to full_pose.py, which also adapts the shared trunk.
        return None
    if recipe["task"] == "joint":
        cutoff = (
            limits["coco_catastrophic_drop"]
            if epoch < 5
            else limits["coco_drop_from_epoch5"]
        )
        if any(metrics[k] < expected()[k] - cutoff for k in COCO_KEYS):
            return "COCO 聯合訓練安全線"
        pose_drop = limits["pose_joint_drop"]
    else:
        # 完整 Pose 分支的 COCO 是診斷，不可誤作原 Detect anchor 退化。
        pose_drop = limits["pose_adaptation_drop"]
    if any(metrics[k] < parent[k] - pose_drop for k in parent if k not in COCO_KEYS):
        return "Pose 較該段起點嚴重退化"
    return None


class Plateau:
    def __init__(self, recipe):
        self.policy = {
            k: recipe[k] for k in ("patience", "min_delta", "plateau_half_after")
        }
        self.best = None
        self.stale = 0
        self.reductions = 0

    @property
    def multiplier(self):
        return 0.5**self.reductions

    def observe(self, value):
        if not math.isfinite(value):
            raise ValueError("nonfinite plateau score")
        improved = self.best is None or value > self.best + self.policy["min_delta"]
        if improved:
            self.best, self.stale = value, 0
        else:
            self.stale += 1
        if self.policy["plateau_half_after"] == self.stale and not self.reductions:
            self.reductions = 1
        return bool(self.policy["patience"] and self.stale >= self.policy["patience"])

    def state_dict(self):
        return {
            "policy": self.policy,
            "best": self.best,
            "stale": self.stale,
            "reductions": self.reductions,
        }

    def load_state_dict(self, data):
        if data["policy"] != self.policy:
            raise ValueError("resume plateau policy differs")
        if data["best"] is not None and not math.isfinite(data["best"]):
            raise ValueError("invalid plateau best")
        if data["stale"] < 0 or data["reductions"] not in (0, 1):
            raise ValueError("invalid plateau state")
        self.best, self.stale, self.reductions = (
            data["best"],
            data["stale"],
            data["reductions"],
        )
