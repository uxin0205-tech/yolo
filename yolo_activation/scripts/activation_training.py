#!/usr/bin/env python3
"""Audit, compile, or dry-run the SIPA-BCSP training architecture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activation_lab.training import (
    RegionRule,
    StaticPolicy,
    TrainingArchitecture,
    load_manifest,
    load_observations,
    load_yaml_mapping,
)


def _write(payload: dict, output: Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class _ToyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.early = _ToyBlock()
        self.deep = _ToyBlock()
        self.neck = _ToyBlock()
        self.head = _ToyBlock()
        self.output_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(self.neck(self.deep(self.early(x))))
        return self.output_sigmoid(x)


def _toy_dry_run(architecture: TrainingArchitecture) -> dict:
    torch.manual_seed(0)
    model = _ToyDetector().eval()
    rules = tuple(
        RegionRule(pattern=rf"^{region}\.act$", region=region)
        for region in ("early", "deep", "neck", "head")
    )
    manifest = architecture.inspect(
        model,
        model_id="toy-detector",
        region_rules=rules,
    ).approve()
    policy = StaticPolicy(
        policy_id="uniform--poly_shift", default_activation="poly_shift"
    )
    applied = architecture.apply(model, manifest, policy)
    sample = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        output = applied.model(sample)
    original_still_silu = all(
        isinstance(model.get_submodule(site.module_path), nn.SiLU)
        for site in manifest.sites
    )
    return {
        "scope": "toy wiring dry-run; no dataset, checkpoint, AP, or hardware latency",
        "manifest": {
            "model_id": manifest.model_id,
            "reviewed": manifest.reviewed,
            "regions": manifest.regions,
            "sites": [site.module_path for site in manifest.sites],
        },
        "initial_plan": architecture.plan(manifest).to_dict(),
        "applied_policy": {
            "policy_id": policy.policy_id,
            "changed_paths": applied.changed_paths,
            "original_model_unchanged": original_still_silu,
            "output_shape": list(output.shape),
            "output_finite": bool(torch.isfinite(output).all()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIPA-BCSP activation 訓練架構工具")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "training/configs/pipeline.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-delivery", help="稽核未來交付資料夾契約")
    audit.add_argument("--delivery", type=Path, required=True)
    audit.add_argument("--no-check-files", action="store_true")
    audit.add_argument("--output", type=Path)

    plan = subparsers.add_parser("compile-plan", help="只產生下一個必要實驗 stage")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--observations", type=Path)
    plan.add_argument("--output", type=Path)

    dry_run = subparsers.add_parser(
        "toy-dry-run", help="以合成 PyTorch model 驗證 wiring"
    )
    dry_run.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    architecture = TrainingArchitecture.from_yaml(str(args.config))
    if args.command == "audit-delivery":
        delivery = load_yaml_mapping(args.delivery)
        audit = architecture.audit_delivery(
            delivery,
            check_files=not args.no_check_files,
        )
        _write(
            {
                "delivery_id": audit.delivery_id,
                "ok": audit.ok,
                "errors": audit.errors,
                "warnings": audit.warnings,
            },
            args.output,
        )
        return 0 if audit.ok else 2
    if args.command == "compile-plan":
        manifest = load_manifest(args.manifest)
        observations = (
            load_observations(args.observations)
            if args.observations is not None
            else ()
        )
        plan = architecture.plan(manifest, observations)
        _write(plan.to_dict(), args.output)
        return 0 if plan.next_stage != "blocked" else 2
    if args.command == "toy-dry-run":
        _write(_toy_dry_run(architecture), args.output)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
