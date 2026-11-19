"""只驗證受保護的 YOLO11 歷史權重，不能啟動舊實驗矩陣。"""

import contextlib
import io
import json
from pathlib import Path

import torch

from .model import sha256


def verify_history(root):
    # 最小相容層留在原 package 名稱，以保存舊 state-dict 的建構契約。
    from binary_attention.model import build_student
    from binary_attention.variants.definitions import variant_from_resolved_config
    from binary_attention.verification.checkpoint import strict_reload

    root = Path(root)
    manifest = json.loads((root / "artifacts/final_weights/manifest.json").read_text())
    rows = []
    for row in manifest["weights"]:
        path = root / row["archived_weight"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"historical checkpoint SHA mismatch: {path}")
        config = json.loads((path.parent / "resolved_config.json").read_text())
        variant = variant_from_resolved_config(config)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_student(root / row["model_yaml"], variant)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        contract = payload["manifest"]
        if contract["variant_id"] != variant.id or contract["variant_config_hash"] != variant.config_hash:
            raise ValueError("historical variant/config mismatch")
        if int(contract["experiment_schema_version"]) >= 3:
            strict_reload(path, model, variant)
        else:
            model.load_state_dict(payload["state_dict"], strict=True)
        rows.append({"variant": variant.id, "path": row["archived_weight"],
                     "sha256": row["sha256"], "strict_reload": True})
        print(f"historical strict reload {len(rows)}/{len(manifest['weights'])}", flush=True)
    return rows
