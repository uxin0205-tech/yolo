#!/usr/bin/env python3
"""以既有正式結果檢查 MASF 與 BinaryQK 的精度回歸訊號。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

YOLO_ROOT = Path(__file__).resolve().parents[3]
MASF_RESULTS = (
    YOLO_ROOT / "yolo_masf" / "clean_bbt5_study" / "results" / "comparison_mean_std.csv"
)
MASF_PER_SEED_RESULTS = (
    YOLO_ROOT / "yolo_masf" / "clean_bbt5_study" / "results" / "per_seed_metrics.csv"
)
BINARYQK_RESULTS = (
    YOLO_ROOT / "yolo_binaryqk" / "artifacts" / "reports" / "coco_area_metrics.csv"
)
YOLO26_ATTENTION_RESULTS = YOLO_ROOT / "yolo_attention" / "reports" / "comparison.csv"
YOLO26_MASF_RESULTS = (
    YOLO_ROOT
    / "yolo_achitechure"
    / "achitechure_1"
    / "final"
    / "reports"
    / "full35-partial75-ap.csv"
)
MATERIAL_GAIN = 0.001

MASF_VARIANTS = {
    "P2-PaperFormula-Clean",
    "P2-Lite35-Clean",
    "P2-Lite35-F7-Clean",
    "P2-Partial50-Clean",
    "P2-Partial25-Clean",
    "P3-M7-Clean",
    "P3-Lite35-Clean",
    "P3-Lite35-F7-Clean",
    "P3-Partial50-Clean",
    "P3-Partial25-Clean",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到正式結果檔：{path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def unique_row(rows: list[dict[str, str]], **filters: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(column) == expected for column, expected in filters.items())
    ]
    if len(matches) != 1:
        conditions = ", ".join(f"{key}={value}" for key, value in filters.items())
        raise ValueError(f"預期唯一一列（{conditions}），實際找到 {len(matches)} 列")
    return matches[0]


def main() -> int:
    masf_rows = read_csv(MASF_RESULTS)
    baseline_row = unique_row(
        masf_rows,
        split="val",
        comparison_tier="strict_fair",
        experiment="B0-Clean",
    )
    baseline_map = float(baseline_row["map50_95_mean"])
    masf_candidates = [
        row
        for row in masf_rows
        if row["split"] == "val"
        and row["comparison_tier"] == "strict_fair"
        and row["experiment"] in MASF_VARIANTS
    ]
    if {row["experiment"] for row in masf_candidates} != MASF_VARIANTS:
        missing = sorted(MASF_VARIANTS - {row["experiment"] for row in masf_candidates})
        raise ValueError(f"MASF 正式變體不完整：{missing}")
    best_masf = max(masf_candidates, key=lambda row: float(row["map50_95_mean"]))
    best_masf_map = float(best_masf["map50_95_mean"])
    masf_delta = best_masf_map - baseline_map
    p2_direct_map = float(
        unique_row(
            masf_rows,
            split="val",
            comparison_tier="strict_fair",
            experiment="P2-Direct-Clean",
        )["map50_95_mean"]
    )
    p2_paper_map = float(
        unique_row(
            masf_rows,
            split="val",
            comparison_tier="strict_fair",
            experiment="P2-PaperFormula-Clean",
        )["map50_95_mean"]
    )
    p3_partial25_map = float(
        unique_row(
            masf_rows,
            split="val",
            comparison_tier="strict_fair",
            experiment="P3-Partial25-Clean",
        )["map50_95_mean"]
    )
    masf_per_seed_rows = read_csv(MASF_PER_SEED_RESULTS)
    p3_partial25_seed_deltas = {
        seed: float(
            unique_row(
                masf_per_seed_rows,
                split="val",
                experiment="P3-Partial25-Clean",
                seed=str(seed),
            )["map50_95"]
        )
        - float(
            unique_row(
                masf_per_seed_rows,
                split="val",
                experiment="B0-Clean",
                seed=str(seed),
            )["map50_95"]
        )
        for seed in (42, 43)
    }

    binary_rows = read_csv(BINARYQK_RESULTS)
    binary_map = {
        variant: float(unique_row(binary_rows, variant=variant)["coco_mAP50_95"])
        for variant in ("E0", "E1-S", "E1", "E2-DUAL", "T0", "T1", "T2", "T4", "T6-F/A")
    }
    direct_sign_delta = binary_map["E1-S"] - binary_map["E0"]

    yolo26_rows = read_csv(YOLO26_ATTENTION_RESULTS)
    yolo26_map = {
        run: float(unique_row(yolo26_rows, run=run)["map50_95"])
        for run in ("b26-fp", "h-scr", "w-dir", "n0-shift", "a-final")
    }
    yolo26_final_delta = yolo26_map["a-final"] - yolo26_map["b26-fp"]

    current_masf_rows = read_csv(YOLO26_MASF_RESULTS)

    def current_masf_row(
        model_id: str,
        evaluator: str,
        scope: str,
    ) -> dict[str, str]:
        return unique_row(
            current_masf_rows,
            model_id=model_id,
            dataset="coco2017-val",
            evaluator=evaluator,
            scope=scope,
        )

    current_internal_map = {
        model_id: float(
            current_masf_row(model_id, "ultralytics-internal", "overall")["ap50_95"]
        )
        for model_id in ("a0", "full35-a2", "partial75-a2")
    }
    best_current_masf_id = max(
        ("full35-a2", "partial75-a2"),
        key=current_internal_map.__getitem__,
    )
    current_masf_delta = (
        current_internal_map[best_current_masf_id] - current_internal_map["a0"]
    )
    canonical_metrics = {
        model_id: {
            "overall": current_masf_row(model_id, "canonical-coco-api", "overall"),
            "sports_ball": current_masf_row(
                model_id, "canonical-coco-api", "sports_ball"
            ),
            "baseball_bat": current_masf_row(
                model_id, "canonical-coco-api", "baseball_bat"
            ),
        }
        for model_id in ("a0", "full35-a2", "partial75-a2")
    }

    failures: list[str] = []
    if masf_delta < 0.0:
        failures.append(
            "最佳含 MASF 的 strict-fair 架構未超過 B0（不等同 MASF 單獨因果）："
            f"{best_masf['experiment']}={best_masf_map:.6f}，"
            f"B0-Clean={baseline_map:.6f}，delta={masf_delta:+.6f}"
        )
    if direct_sign_delta < -0.01:
        failures.append(
            "BinaryQK 直接 sign-only 替換下降超過 0.01："
            f"E1-S={binary_map['E1-S']:.6f}，E0={binary_map['E0']:.6f}，"
            f"delta={direct_sign_delta:+.6f}"
        )
    if yolo26_final_delta < -0.01:
        failures.append(
            "YOLO26 BinaryQK A-FINAL 仍下降超過 0.01："
            f"A-FINAL={yolo26_map['a-final']:.6f}，"
            f"B26-FP={yolo26_map['b26-fp']:.6f}，delta={yolo26_final_delta:+.6f}"
        )
    if current_masf_delta < MATERIAL_GAIN:
        failures.append(
            "現行 YOLO26 P3 MASF 未達 +0.001 material-gain gate："
            f"{best_current_masf_id}={current_internal_map[best_current_masf_id]:.9f}，"
            f"A0={current_internal_map['a0']:.9f}，delta={current_masf_delta:+.9f}"
        )

    for failure in failures:
        print(f"[FAIL] {failure}")

    print("[INFO] MASF 因果拆解（validation mean；使用正確 family control）")
    print(
        f"  B0 → P2-Direct       : {p2_direct_map - baseline_map:+.6f}（P2 graph/head 效應）"
    )
    print(
        f"  P2-Direct → P2-MFAM : {p2_paper_map - p2_direct_map:+.6f}（MFAM 邊際效應）"
    )
    print(f"  B0 → P2-MFAM        : {p2_paper_map - baseline_map:+.6f}（兩種效應合計）")
    print(f"  B0 → P3-Partial25   : {p3_partial25_map - baseline_map:+.6f}")
    for seed, delta in p3_partial25_seed_deltas.items():
        print(f"    seed {seed}: {delta:+.6f}")

    print("[INFO] BinaryQK 恢復階梯（相對各自 FP control）")
    for variant in ("E1", "E2-DUAL"):
        print(f"  {variant:9s}: {binary_map[variant] - binary_map['E0']:+.6f}")
    for variant in ("T1", "T2", "T4", "T6-F/A"):
        print(f"  {variant:9s}: {binary_map[variant] - binary_map['T0']:+.6f}")
    print("[INFO] YOLO26 BinaryQK 恢復階梯（相對 B26-FP）")
    for run in ("h-scr", "w-dir", "n0-shift", "a-final"):
        print(f"  {run:9s}: {yolo26_map[run] - yolo26_map['b26-fp']:+.6f}")
    print("[INFO] 現行 YOLO26 P3 MASF（相對 A0；canonical COCO API）")
    baseline = canonical_metrics["a0"]
    for model_id in ("full35-a2", "partial75-a2"):
        metrics = canonical_metrics[model_id]
        print(
            f"  {model_id:12s}: overall "
            f"{float(metrics['overall']['ap50_95']) - float(baseline['overall']['ap50_95']):+.6f}，"
            f"AP_S {float(metrics['overall']['ap_s']) - float(baseline['overall']['ap_s']):+.6f}，"
            "sports_ball "
            f"{float(metrics['sports_ball']['ap50_95']) - float(baseline['sports_ball']['ap50_95']):+.6f}，"
            "baseball_bat "
            f"{float(metrics['baseball_bat']['ap50_95']) - float(baseline['baseball_bat']['ap50_95']):+.6f}"
        )

    if failures:
        print(f"[RED] 偵測到 {len(failures)} 個目前精度回歸訊號")
        return 1
    print("[GREEN] 目前正式結果未觸發精度回歸門檻")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(2) from error
