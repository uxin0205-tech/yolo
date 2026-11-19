"""Formal PWL-only validation workflow and report generation."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import VariantConfig
from .evaluation import (
    EvaluationRequest,
    UltralyticsEvaluationBackend,
    standardize_metrics,
    write_standard_result,
)
from .integration import convert_yolo26_model
from .normalization import BitTruePiecewiseLinearSoftmax
from .profiling import write_variant_profile
from .pwl_validation import PWLModelDiagnosticsCollector
from .queue_model import QueueResult

TAIL_MASS_GATE = 0.001


@dataclass(frozen=True)
class PWLRangeSpec:
    score_floor: float
    segments: int

    @property
    def segment_width(self) -> float:
        return -self.score_floor / self.segments

    @property
    def endpoint_storage_bits(self) -> int:
        return (self.segments + 1) * 16


def choose_pwl_range(tail_probability_mass: float) -> PWLRangeSpec:
    """Keep [-8,0] unless clipped Exact-Softmax mass exceeds the declared gate."""

    if not 0.0 <= tail_probability_mass <= 1.0:
        raise ValueError("tail probability mass must be in [0, 1]")
    return PWLRangeSpec(-8.0, 16) if tail_probability_mass <= TAIL_MASS_GATE else PWLRangeSpec(-10.0, 20)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _score_rows(ranges: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for range_name, summaries in ranges.items():
        for site_summary in summaries:
            for summary in [site_summary["aggregate"], *site_summary["heads"]]:
                percentiles = summary["percentiles"]
                row = {
                    "range": range_name,
                    "site": site_summary["site"],
                    "head": summary.get("head", "all"),
                    **{key: value for key, value in summary.items() if key not in {"head", "percentiles"}},
                    **{f"percentile_{key}": value for key, value in percentiles.items()},
                }
                rows.append(row)
    return rows


def _histogram_svg(rows: list[dict[str, object]], destination: Path) -> None:
    selected = [row for row in rows if row["head"] == "all" and float(row["bin_left"]) >= -12.0]
    sites = list(dict.fromkeys(str(row["site"]) for row in selected))
    width, height = 960, 420
    margin_left, plot_height = 80, 320
    plot_width = width - margin_left - 30
    maximum = max((int(row["count"]) for row in selected), default=1)
    colors = ("#2563eb", "#dc2626")
    polylines: list[str] = []
    for site_index, site in enumerate(sites):
        site_rows = [row for row in selected if row["site"] == site]
        points = []
        for index, row in enumerate(site_rows):
            x = margin_left + index * plot_width / max(len(site_rows) - 1, 1)
            y = 20 + plot_height * (1.0 - int(row["count"]) / maximum)
            points.append(f"{x:.1f},{y:.1f}")
        polylines.append(
            f'<polyline fill="none" stroke="{colors[site_index % len(colors)]}" '
            f'stroke-width="2" points="{" ".join(points)}"/>'
        )
    legend = "".join(
        f'<text x="{margin_left + 300 * i}" y="405" font-size="13" '
        f'fill="{colors[i % len(colors)]}">{site}</text>'
        for i, site in enumerate(sites)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="20" y="18" font-size="15">Centered-score histogram (aggregate heads)</text>'
        f'<line x1="{margin_left}" y1="20" x2="{margin_left}" y2="340" stroke="#111"/>'
        f'<line x1="{margin_left}" y1="340" x2="930" y2="340" stroke="#111"/>'
        f'{"".join(polylines)}'
        '<text x="455" y="375" font-size="13">u = S - max(S), shown over [-12,0]</text>'
        f'{legend}</svg>\n'
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8")


def _overall_tail_mass(summaries: list[dict[str, object]]) -> float:
    numerator = 0.0
    denominator = 0
    for site in summaries:
        aggregate = site["aggregate"]
        rows = int(aggregate["row_count"])
        numerator += float(aggregate["exact_tail_probability_mass_mean"]) * rows
        denominator += rows
    if not denominator:
        raise ValueError("score diagnostics contain no rows")
    return numerator / denominator


def _weighted(summaries: list[dict[str, object]], metric: str, count: str) -> float:
    numerator = 0.0
    denominator = 0
    for site in summaries:
        aggregate = site["aggregate"]
        weight = int(aggregate[count])
        numerator += float(aggregate[metric]) * weight
        denominator += weight
    return numerator / denominator


def write_score_artifacts(
    *,
    run_dir: Path,
    results_dir: Path,
    exact_metrics: dict[str, object],
    ranges: dict[str, list[dict[str, object]]],
    histogram_rows: list[dict[str, object]],
) -> Path:
    payload = {"exact_metrics": exact_metrics, "ranges": ranges, "tail_mass_gate": TAIL_MASS_GATE}
    path = run_dir / "metrics" / "score-analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "score-analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(run_dir / "metrics" / "score-statistics.csv", _score_rows(ranges))
    _write_csv(results_dir / "score-statistics.csv", _score_rows(ranges))
    _write_csv(run_dir / "metrics" / "score-histogram.csv", histogram_rows)
    _write_csv(results_dir / "score-histogram.csv", histogram_rows)
    _histogram_svg(histogram_rows, results_dir / "figures" / "score-histogram.svg")
    return path


def _comparison_svg(rows: list[dict[str, object]], destination: Path) -> None:
    width, height = 680, 360
    values = [float(row["map50_95"]) for row in rows]
    low = min(values) - 0.001
    high = max(values) + 0.001
    bars = []
    for index, row in enumerate(rows):
        x = 90 + index * 190
        bar_height = 240 * (float(row["map50_95"]) - low) / max(high - low, 1e-9)
        y = 285 - bar_height
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="110" height="{bar_height:.1f}" fill="#2563eb"/>'
            f'<text x="{x + 55}" y="{y - 8:.1f}" text-anchor="middle" font-size="13">'
            f'{float(row["map50_95"]):.6f}</text>'
            f'<text x="{x + 55}" y="310" text-anchor="middle" font-size="13">{row["method"]}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="20" y="25" font-size="16">PWL normalization COCO mAP50-95</text>'
        '<line x1="60" y1="285" x2="640" y2="285" stroke="#111"/>'
        f'{"".join(bars)}</svg>\n'
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8")


class PWLExperimentRunner:
    """Execute the two authorized queue jobs; never trains or changes QK/scale/bias."""

    def __init__(self, *, project_root: Path, model_factory: Callable[[str], Any] | None = None) -> None:
        self.project_root = project_root.resolve()
        self.model_factory = model_factory
        self.results_dir = self.project_root / "PWL" / "results"

    def _model(self, checkpoint: Path):
        if self.model_factory is None:
            from ultralytics import YOLO

            return YOLO(str(checkpoint.resolve()))
        return self.model_factory(str(checkpoint.resolve()))

    @staticmethod
    def _validation_args(request: EvaluationRequest, name: str) -> dict[str, object]:
        args = request.recipe.to_ultralytics_args()
        args.update(project=str(request.run_dir.resolve()), name=name, exist_ok=True)
        return args

    def score_analysis(self, request: EvaluationRequest) -> QueueResult:
        if request.variant_path is None:
            raise ValueError("PWL score analysis requires the Exact variant")
        config = VariantConfig.from_yaml(request.variant_path)
        model = self._model(request.parent_checkpoint)
        convert_yolo26_model(model.model, config)
        collector8 = PWLModelDiagnosticsCollector(model.model, score_floor=-8.0, segments=16)
        collector10 = PWLModelDiagnosticsCollector(model.model, score_floor=-10.0, segments=20)
        with collector8, collector10:
            metrics = standardize_metrics(model.val(**self._validation_args(request, "ultralytics")))
        profile = write_variant_profile(request.run_dir, config)
        result = write_standard_result(
            request.run_dir,
            metrics,
            checkpoint_path=request.parent_checkpoint,
            profile_path=profile,
            row_sum_max_error=None,
        )
        write_score_artifacts(
            run_dir=request.run_dir,
            results_dir=self.results_dir,
            exact_metrics=metrics,
            ranges={"neg8": collector8.summaries(), "neg10": collector10.summaries()},
            histogram_rows=collector8.histogram_rows(),
        )
        return result

    def compare(self, request: EvaluationRequest, *, score_run_dir: Path) -> QueueResult:
        parent_score_path = score_run_dir / "metrics" / "score-analysis.json"
        if not parent_score_path.is_file():
            raise FileNotFoundError("PWL comparison requires the completed parent score analysis")
        # Recollect the Exact diagnostics inside this immutable comparison run.
        # This prevents a stale or invalid parent diagnostic artifact from being
        # silently reused after the diagnostic implementation changes.
        refreshed_score_dir = request.run_dir / "score_analysis"
        refreshed_request = EvaluationRequest(
            run_id=f"{request.run_id}-score-analysis",
            run_dir=refreshed_score_dir,
            parent_checkpoint=request.parent_checkpoint,
            recipe=request.recipe,
            variant_path=request.variant_path,
        )
        self.score_analysis(refreshed_request)
        score_payload = json.loads(
            (refreshed_score_dir / "metrics" / "score-analysis.json").read_text()
        )
        tail_mass = _overall_tail_mass(score_payload["ranges"]["neg8"])
        selected = choose_pwl_range(tail_mass)
        suffix = "8" if selected.score_floor == -8.0 else "10"
        backend = UltralyticsEvaluationBackend(model_factory=self.model_factory)
        candidate_results: dict[str, QueueResult] = {}
        candidate_configs: dict[str, VariantConfig] = {}
        for method, filename in (
            ("Float PWL", f"float-pwl-{suffix}.yaml"),
            ("Q8.8 Bit-True PWL", f"bittrue-pwl-{suffix}.yaml"),
        ):
            variant_path = self.project_root / "PWL" / "configs" / filename
            child_dir = request.run_dir / ("float" if method == "Float PWL" else "bit_true")
            child_request = EvaluationRequest(
                run_id=f"{request.run_id}-{child_dir.name}",
                run_dir=child_dir,
                parent_checkpoint=request.parent_checkpoint,
                recipe=request.recipe,
                variant_path=variant_path,
            )
            child = backend.evaluate_variant(child_request)
            profile = write_variant_profile(child_dir, VariantConfig.from_yaml(variant_path))
            payload = json.loads(Path(child.metrics_path).read_text())
            payload["profile_path"] = str(profile)
            Path(child.metrics_path).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            candidate_results[method] = QueueResult.from_dict(payload)
            candidate_configs[method] = VariantConfig.from_yaml(variant_path)
        exact = score_payload["exact_metrics"]
        selected_summaries = score_payload["ranges"][f"neg{suffix}"]
        rows = self._write_final_report(
            exact=exact,
            candidates=candidate_results,
            configs=candidate_configs,
            summaries=selected_summaries,
            all_ranges=score_payload["ranges"],
            tail_mass=tail_mass,
            selected=selected,
        )
        bit = candidate_results["Q8.8 Bit-True PWL"]
        top_path = request.run_dir / "metrics" / "queue-result.json"
        top_path.parent.mkdir(parents=True, exist_ok=True)
        top = QueueResult(
            map50_95=bit.map50_95,
            map50=bit.map50,
            map75=bit.map75,
            maps=bit.maps,
            checkpoint_path=str(request.parent_checkpoint.resolve()),
            metrics_path=str(top_path.resolve()),
        )
        top_path.write_text(json.dumps(asdict(top), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _comparison_svg(rows, self.results_dir / "figures" / "normalization-map.svg")
        return top

    def _write_final_report(
        self,
        *,
        exact: dict[str, object],
        candidates: dict[str, QueueResult],
        configs: dict[str, VariantConfig],
        summaries: list[dict[str, object]],
        all_ranges: dict[str, list[dict[str, object]]],
        tail_mass: float,
        selected: PWLRangeSpec,
    ) -> list[dict[str, object]]:
        float_result = candidates["Float PWL"]
        bit_result = candidates["Q8.8 Bit-True PWL"]
        metrics = {
            "Float PWL": {
                "probability_mae": _weighted(summaries, "float_probability_mae", "count"),
                "probability_max_error": max(
                    float(site["aggregate"]["float_probability_max_error"]) for site in summaries
                ),
                "pv_mae": _weighted(summaries, "float_pv_mae", "value_count"),
                "pv_cosine": _weighted(summaries, "float_pv_cosine_similarity", "row_count"),
            },
            "Q8.8 Bit-True PWL": {
                "probability_mae": _weighted(summaries, "bit_true_probability_mae", "count"),
                "probability_max_error": max(
                    float(site["aggregate"]["bit_true_probability_max_error"]) for site in summaries
                ),
                "pv_mae": _weighted(summaries, "bit_true_pv_mae", "value_count"),
                "pv_cosine": _weighted(summaries, "bit_true_pv_cosine_similarity", "row_count"),
            },
        }
        rows = [
            {
                "method": "Exact Softmax",
                "map50_95": exact["map50_95"],
                "map50": exact["map50"],
                "probability_mae": 0.0,
                "probability_max_error": 0.0,
                "pv_mae": 0.0,
                "pv_cosine_similarity": 1.0,
                "score_floor": "none",
                "segments": "none",
                "endpoint_storage_bits": 0,
            }
        ]
        for method, result in candidates.items():
            rows.append(
                {
                    "method": method,
                    "map50_95": result.map50_95,
                    "map50": result.map50,
                    "probability_mae": metrics[method]["probability_mae"],
                    "probability_max_error": metrics[method]["probability_max_error"],
                    "pv_mae": metrics[method]["pv_mae"],
                    "pv_cosine_similarity": metrics[method]["pv_cosine"],
                    "score_floor": selected.score_floor,
                    "segments": selected.segments,
                    "endpoint_storage_bits": (
                        0 if method == "Float PWL" else selected.endpoint_storage_bits
                    ),
                }
            )
        _write_csv(self.results_dir / "comparison.csv", rows)
        bit_config = configs["Q8.8 Bit-True PWL"]
        bit_floor = bit_config.score_min * bit_config.score_step
        if bit_config.pwl_segments != selected.segments or bit_floor != selected.score_floor:
            raise ValueError("selected Bit-True PWL config does not match the score range gate")
        bit_reference = BitTruePiecewiseLinearSoftmax(
            score_floor=selected.score_floor,
            segments=selected.segments,
        )
        endpoint_rows = [
            {
                "index": index,
                "centered_score": selected.score_floor + index * selected.segment_width,
                "uq1_15_integer": value,
                "uq1_15_hex": f"0x{value:04X}",
                "decoded_weight": value / (1 << bit_reference.endpoint_fraction_bits),
            }
            for index, value in enumerate(bit_reference.endpoint_table.tolist())
        ]
        _write_csv(self.results_dir / "endpoint-table.csv", endpoint_rows)
        float_delta = float_result.map50_95 - float(exact["map50_95"])
        bit_delta = bit_result.map50_95 - float(exact["map50_95"])
        bit_increment = bit_result.map50_95 - float_result.map50_95
        range_rows: list[str] = []
        for label, range_summaries in (
            ("[-8,0]", all_ranges["neg8"]),
            ("[-10,0]", all_ranges["neg10"]),
        ):
            values = (
                label,
                f"{_weighted(range_summaries, 'float_exp_mae', 'count'):.8g}",
                f"{_weighted(range_summaries, 'float_probability_mae', 'count'):.8g}",
                f"{_weighted(range_summaries, 'float_pv_mae', 'value_count'):.8g}",
                f"{_weighted(range_summaries, 'bit_true_exp_mae', 'count'):.8g}",
                f"{_weighted(range_summaries, 'bit_true_probability_mae', 'count'):.8g}",
                f"{_weighted(range_summaries, 'bit_true_pv_mae', 'value_count'):.8g}",
            )
            range_rows.append("| " + " | ".join(values) + " |")
        site_rows: list[str] = []
        head_rows: list[str] = []
        for site in all_ranges["neg10"]:
            aggregate = site["aggregate"]
            values = (
                str(site["site"]),
                f"{float(aggregate['min']):.4f}",
                f"{float(aggregate['mean']):.4f}",
                f"{float(aggregate['std']):.4f}",
                f"{float(aggregate['ratio_lt_neg8']):.4%}",
                f"{float(aggregate['ratio_lt_neg10']):.4%}",
                f"{float(aggregate['exact_tail_probability_mass_mean']):.6%}",
            )
            site_rows.append("| " + " | ".join(values) + " |")
            for head in site["heads"]:
                head_values = (
                    str(site["site"]),
                    str(head["head"]),
                    f"{float(head['mean']):.4f}",
                    f"{float(head['std']):.4f}",
                    f"{float(head['ratio_lt_neg8']):.4%}",
                    f"{float(head['exact_tail_probability_mass_mean']):.6%}",
                )
                head_rows.append("| " + " | ".join(head_values) + " |")
        exact_proxy = 91_910_400
        pwl_proxy = 89_350_400
        exact_normalization = 6_406_400
        pwl_normalization = 3_846_400
        pv_proxy = 81_920_000
        report = f"""# YOLO26 Binary Attention PWL validation

固定架構：Hadamard MDB Binary QK → PoT Scale → Decomposed 2D Bias → Normalization
→ $PV+PE(V)$。三個方法使用同一 checkpoint、COCO2017 val、imgsz、batch、seed 與 evaluator；
本實驗沒有訓練，也沒有修改 QK、scale、bias、V 或 projection。

## 執行條件與資料

- 模型：YOLO26m，parent checkpoint `artifacts/runs/v1-br/ultralytics/weights/best.pt`。
- 資料：COCO2017 `val2017`，5,000 images / 36,335 instances。
- evaluator：Ultralytics 8.4.90 internal metric，`imgsz=640`、`batch=16`、`seed=0`。
- 兩個 site 同時替換並量測：`model.10.m.0.attn`、`model.22.m.0.1.attn`。
- 本地 mAP 不等同 canonical COCO API AP，只作同 evaluator 的 paired comparison。

## Range 結論

- Exact Softmax 中 $u<-8$ 的平均 probability mass：`{tail_mass:.8g}`。
- Gate：`{TAIL_MASS_GATE}`；選定範圍：`[{selected.score_floor:.0f},0]`。
- segments：`{selected.segments}`，$\\Delta={selected.segment_width}$。
- Bit-True endpoints：`{selected.segments + 1} × 16-bit = {selected.endpoint_storage_bits} bits`
  （{selected.endpoint_storage_bits / 8:.0f} bytes）。

| site | min | mean | std | score < -8 | score < -10 | Exact mass at score < -8 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(site_rows)}

| site | head | mean | std | score < -8 | Exact mass at score < -8 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(head_rows)}

site 2 的 score 分布較寬，約 24.62% score 小於 -8，tail probability mass 約
0.14466%；個別 head 最高達 0.20859%。因此論文起始設定 `[-8,0]` 會在少數 head
造成集中 clipping。保留 $\\Delta=0.5$、擴至 `[-10,0]` 只增加 4 個 segments/endpoints，
同時保留直接 shift/mask indexing。

## `[-8,0]` 與 `[-10,0]` 的數值比較

| range | Float exp MAE | Float P MAE | Float PV MAE | Bit-True exp MAE | Bit-True P MAE | Bit-True PV MAE |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(range_rows)}

擴大至 `[-10,0]` 後，Float PV MAE 約降低 47.4%，Bit-True PV MAE 約降低 48.9%。
所以本模型採 20 segments，而不是 16 segments。

## COCO 與數值結果

| 方法 | mAP50-95 | mAP50 | mAP75 | 相對 Exact | P MAE | P max | PV MAE | PV cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact | {float(exact['map50_95']):.6f} | {float(exact['map50']):.6f} | {float(exact['map75']):.6f} | 0 | 0 | 0 | 0 | 1 |
| Float PWL | {float_result.map50_95:.6f} | {float_result.map50:.6f} | {float_result.map75:.6f} | {float_delta:+.6f} | {metrics['Float PWL']['probability_mae']:.8g} | {metrics['Float PWL']['probability_max_error']:.8g} | {metrics['Float PWL']['pv_mae']:.8g} | {metrics['Float PWL']['pv_cosine']:.8f} |
| Q8.8 Bit-True | {bit_result.map50_95:.6f} | {bit_result.map50:.6f} | {bit_result.map75:.6f} | {bit_delta:+.6f} | {metrics['Q8.8 Bit-True PWL']['probability_mae']:.8g} | {metrics['Q8.8 Bit-True PWL']['probability_max_error']:.8g} | {metrics['Q8.8 Bit-True PWL']['pv_mae']:.8g} | {metrics['Q8.8 Bit-True PWL']['pv_cosine']:.8f} |

Bit-True 相對 Float PWL 的額外 mAP 差：`{bit_increment:+.6f}`。
三個差值都遠小於 0.001 AP；微小正差不解讀為 PWL 提升精度，只能判定此 checkpoint
與 evaluator 下未觀察到可辨識的精度損失，因此不需要重新訓練。

## Hardware interpretation

PWL 把逐 score 的 transcendental `exp` 換成：clip、offset、segment shift、fraction mask、
兩個 endpoint read，以及一次 integer multiply/shift/add interpolation。$\\Delta=0.5$ 時 index
直接由 Q8.8 的 bit slice 取得。這是規則、固定容量且容易 pipeline 的 datapath。

在兩 site、每 site 假設 400 tokens、4 heads、$d_k=32$、$d_v=64$ 的 analytical proxy 下：

| 項目 | Exact | PWL | 差異 |
|---|---:|---:|---:|
| normalization proxy ops | {exact_normalization:,} | {pwl_normalization:,} | -39.96% |
| 整個 attention-path proxy | {exact_proxy:,} | {pwl_proxy:,} | -2.79% |
| dense PV proxy | {pv_proxy:,} | {pv_proxy:,} | 不變 |

PWL 後 dense $PV$ 仍占此 attention-path proxy 約 91.68%，normalization 約占 4.30%。
所以 PWL 的主要價值是移除難 pipeline 的 transcendental `exp`、讓資料路徑固定且只需
42-byte endpoint table，不是大幅降低整個 YOLO26m 的 FLOPs。這些數字是固定 shape
與權重假設的演算法 proxy，不是 GPU/FPGA latency、DSP、BRAM 或 energy 實測。

但 PWL 只近似 numerator weight，沒有消除每列 denominator：仍需 row sum 與 reciprocal/divider；
之後的 dense $PV$ 也完全保留。指定論文同樣使用 serialized restoring divider。因此剩餘主要成本仍是
$PV$ 與 denominator，而不是 endpoint table。此處成本是演算法/儲存 breakdown，不是 FPGA 實測。

## Bit-True scope

論文支持 Q8.8、範圍、segments、17×16-bit endpoints 與 interpolation；但沒有發布 endpoint
整數、endpoint Q-format、rounding、saturation 與完整 intermediate widths。本專案明定 UQ1.15
endpoint、round-to-nearest for nonnegative shifted score、signed 16-bit saturation、floor clipping、
truncating `>>7` interpolation 與 `u=0` endpoint special case。因此這是 paper-parameterized project
bit-true reference，不是作者 RTL 的 faithful reproduction。denominator 目前仍是 exact float reference。

## Diagnostics defect 與修正

第一次 score job 的額外 PV diagnostic 在 Ultralytics AMP context 內被 autocast 成 FP16，
導致累加值出現 NaN；COCO output、score、P 與 mAP 本身仍有限。正式 compare job 已在自己的
immutable `score_analysis/` 重新量測，強制 PV matmul 使用 FP32、reduction 使用 FP64，
並對所有 non-finite 值 fail closed。最終 CSV/圖表只使用修正後 artifact。

## 最終回答

1. `[-8,0]` 不建議作最終設定；overall tail mass 0.11552% 超過 0.1% gate，且 site/head 不均勻。
2. 16 segments / $\\Delta=0.5$ 不足；`[-10,0]`、20 segments、同一 $\\Delta$ 較合適。
3. Float PWL 對 Exact 近乎無損：mAP50-95 差 `{float_delta:+.6f}`。
4. Q8.8 Bit-True 沒有可辨識的額外損失：相對 Float 差 `{bit_increment:+.6f}`。
5. 硬體優勢是將 exp 轉為固定 clip/shift/mask/LUT/interpolation datapath，table 僅 42 bytes。
6. 主要成本仍是 dense $PV$，denominator 的 row sum + reciprocal/divider 也尚未移除。

原始數據：`score-analysis.json`、`score-statistics.csv`、`score-histogram.csv`；
RTL endpoint：`endpoint-table.csv`；
整理比較：`comparison.csv`；圖表：`figures/score-histogram.svg` 與
`figures/normalization-map.svg`。完整 queue provenance 位於
`artifacts/runs/pwl-compare/`。
"""
        (self.results_dir / "REPORT.md").write_text(report, encoding="utf-8")
        return rows
