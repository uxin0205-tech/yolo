#!/usr/bin/env python3
"""Audit a variant workspace without loading a model or using CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_combine.variants import VariantWorkspace

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variant",
        type=Path,
        help="Path to variants/full35 or variants/partial75",
    )
    parser.add_argument("--verify-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.variant if args.variant.is_absolute() else PROJECT_ROOT / args.variant
    workspace = VariantWorkspace.load(path)
    audit = workspace.audit(verify_hashes=args.verify_hashes)
    payload = {
        "architecture": workspace.architecture,
        "role": workspace.role,
        "root": str(workspace.root),
        "run_root": str(workspace.run_root),
        "source_bundle": str(workspace.source_bundle),
        "datasets": {name: str(value) for name, value in workspace.datasets.items()},
        "documents": {name: str(value) for name, value in workspace.documents.items()},
        "audit": {
            "ok": audit.ok,
            "missing_paths": list(audit.missing_paths),
            "hash_mismatches": list(audit.hash_mismatches),
            "hashes_verified": args.verify_hashes,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not audit.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
