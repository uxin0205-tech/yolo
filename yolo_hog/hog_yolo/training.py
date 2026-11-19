"""單一 HOG 候選；重用完整 epoch、固定 EMA 與 AMP replay 引擎。"""

import math

import torch
from binaryqk.protocol import COMMON
from yolo_attention.stage3_kd_training import metric_vector

from . import checkpoint
from .common import REVIEW, read, review, sha256
from .models import configure, criterion, parts, role


class HOGStudy:
    codec = checkpoint
    verify = staticmethod(review)
    configure = staticmethod(configure)
    initialize_criterion = staticmethod(criterion)

    def __init__(self, calibration_path):
        self.calibration_path = calibration_path
        self.calibration = read(calibration_path)
        data = review()
        if (
            not self.calibration["passed"]
            or self.calibration["parent_sha256"] != data["parent_sha256"]
        ):
            raise ValueError("calibration must match current parent")
        self.baseline = read(data["parent_metrics"])
        self.mu = 0.0

    def recipe(self, phase):
        if phase != "hog-adaptation":
            raise ValueError(phase)
        p = review()["protocol"]
        return {
            **COMMON,
            "version": 7,
            "phase": phase,
            "epochs": p["epochs"],
            "minimum_epochs": p["minimum_epochs"],
            "patience": p["patience"],
            "seed": p["seed"],
            "scope": "neck-head-hog-fixed-attention",
            "batch": p["batch"],
            "logical_batch": p["logical_batch"],
            "reference_batch": p["reference_batch"],
            "steps_per_epoch": p["steps_per_epoch"],
            "betas": [0.948, 0.999],
            "weight_decay": 0.00027,
            "group_lrs": p["group_lrs"],
            "lr0": 1e-5,
            "warmup_epochs": 1,
            "warmup_bias_lr": 0.0,
            "linear_final_fraction": 0.5,
            "scheduler": "warmup-cosine",
            "smoke_steps": 2,
            "bn": "all BN running statistics frozen",
            "ema": "inherited age, only trainable parameters averaged",
            "normalization": "B2/S1/integer PWL/D1 unchanged; all Attention parameters frozen",
            "loss": "native terminal E2E .1/.9 batch-sum + mu HOG batch-sum;64/actual macro",
            "selection": p["selection"],
            "reference_review_sha256": sha256(REVIEW),
            "hog": {
                "mu": self.calibration["mu"],
                "calibration_sha256": sha256(self.calibration_path),
                "schedule": p["mu_schedule"],
            },
        }

    def build(self, variant, phase, parent, calibration):
        if (
            variant != "H2"
            or phase != "hog-adaptation"
            or str(parent) != review()["parent"]
        ):
            raise ValueError("unexpected HOG lineage")
        return checkpoint.build(self.calibration_path)

    @staticmethod
    def mutable(model):
        return set()

    @staticmethod
    def initialize_state(model, ema, optimizer, scaler, recipe):
        ema.updates = review()["ema_initial_updates"]

    @staticmethod
    def optimizer(model, recipe):
        modules, groups = dict(model.named_modules()), {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            r = role(name)
            decay = (
                parameter.ndim >= 2
                and not name.endswith(".bias")
                and not isinstance(
                    modules[name.rsplit(".", 1)[0]],
                    torch.nn.modules.batchnorm._BatchNorm,
                )
            )
            key = (r, decay)
            g = groups.setdefault(
                key,
                {
                    "params": [],
                    "role": r,
                    "lr": recipe["group_lrs"][r],
                    "group_name": r + ("/decay" if decay else "/no_decay"),
                    "weight_decay": recipe["weight_decay"] if decay else 0.0,
                },
            )
            g["params"].append(parameter)
        return torch.optim.AdamW(list(groups.values()), betas=tuple(recipe["betas"]))

    @staticmethod
    def gradient_names(model):
        return ["model.16.cv2.conv.weight", "model.23.one2one_cv2.0.0.conv.weight"]

    def set_lr(self, optimizer, recipe, epoch, position, step, smoke):
        total = recipe["epochs"] * recipe["steps_per_epoch"]
        current = epoch * recipe["steps_per_epoch"] + step + 1
        factor = (
            1.0
            if smoke
            else (0.5 + 0.5 * 0.5 * (1 + math.cos(math.pi * current / total)))
            * min(1.0, 0.1 + 0.9 * current / recipe["steps_per_epoch"])
        )
        self.mu = self.calibration["mu"] * (
            1.0
            if smoke
            else (
                (step + 1) / recipe["steps_per_epoch"]
                if epoch == 0
                else 1.0
                if epoch < 8
                else 0.0
            )
        )
        for group in optimizer.param_groups:
            group["lr"] = recipe["group_lrs"][group["role"]] * factor
        return {
            **{g["group_name"]: g["lr"] for g in optimizer.param_groups},
            "hog_mu": self.mu,
        }

    def loss(self, model, batch):
        total, _, _, _, items = parts(model, batch, self.mu)
        return total, items

    @staticmethod
    def should_stop(recipe, completed, best_epoch, history):
        return (
            completed >= recipe["minimum_epochs"]
            and completed - best_epoch >= recipe["patience"]
        )

    @staticmethod
    def accept_candidate(report, history):
        return True

    def guard_metrics(self, report, history, smoke):
        if smoke:
            return
        values, base = metric_vector(report), metric_vector(self.baseline)
        if not history and any(abs(values[k] - base[k]) > 1e-8 for k in base):
            raise ValueError("HOG initial parent metrics changed")
        if any(values[k] < base[k] - 0.02 for k in base):
            raise ValueError("HOG AP safety stop; completed epoch retained")
