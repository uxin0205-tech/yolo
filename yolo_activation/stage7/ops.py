"""沿用 optimize macro 尺度；16×16 Detect、16 Pose，參考 batch64。"""

from binaryqk.run import loader
from stage6.ops import (
    MacroEngine,
    ProbeOptimizer,
    ProgressiveLoss,
    fixed_hashes,
    gradient_groups,
    optimizer_for,
    restore_rng,
    rng_state,
)


def loss_for_recipe(model, recipe):
    from activation_lab.training.full35 import _fp32_bbox_iou
    from ultralytics.utils import loss

    if not getattr(loss.bbox_iou, "_stage7_fp32", False):
        loss.bbox_iou = _fp32_bbox_iou(loss.bbox_iou)
        loss.bbox_iou._stage7_fp32 = True
    return ProgressiveLoss(model, recipe)


def macros(detect_ds, pose_ds, epoch_seed):
    pi = iter(loader(pose_ds, 16, epoch_seed))
    cycle, group = 0, []

    def pose_batch():
        nonlocal pi, cycle
        try:
            return next(pi)
        except StopIteration:
            cycle += 1
            pi = iter(loader(pose_ds, 16, epoch_seed + 100000 * cycle))
            return next(pi)

    for db in loader(detect_ds, 16, epoch_seed):
        group.append(db)
        if len(group) == 16:
            yield tuple(group), (pose_batch(),)
            group = []
    if group:
        yield tuple(group), (pose_batch(),)


__all__ = [
    "MacroEngine",
    "ProbeOptimizer",
    "fixed_hashes",
    "gradient_groups",
    "loss_for_recipe",
    "macros",
    "optimizer_for",
    "restore_rng",
    "rng_state",
]
