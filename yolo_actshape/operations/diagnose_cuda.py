"""隔離 D1 數值與 CUDA 同步診斷；不訓練任務模型，不修改正式來源鎖。"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for folder in ("yolo_actshape", "yolo_attention/src"):
    sys.path.insert(0, str(ROOT / folder))

import torch
from stage9.common import write, digest
from yolo_attention.config import ReciprocalConfig
from yolo_attention.normalization import CompletePWLSoftmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1931)
    args.output.mkdir(parents=True, exist_ok=True)
    norm = CompletePWLSoftmax(ReciprocalConfig()).cuda().train()
    reference = CompletePWLSoftmax(ReciprocalConfig()).cpu().eval()
    scores_cpu = torch.randn(1, 4, 37, 512) * 2
    with torch.no_grad():
        gpu = norm(scores_cpu.cuda()).cpu()
        cpu = reference(scores_cpu)
    if not torch.equal(gpu, cpu):
        raise ValueError("CPU/GPU D1 probability 不逐位一致")
    for bad in (torch.full((1, 4, 1, 512), float("nan"), device="cuda"),
                torch.zeros((1, 4, 1, 1025), device="cuda")):
        try:
            norm(bad)
        except ValueError:
            pass
        else:
            raise ValueError("D1 非法輸入未拒絕")
    torch.cuda.reset_peak_memory_stats()
    start, row_error = time.time(), 0.
    for iteration in range(1, args.iterations + 1):
        scores = (torch.randn(8, 4, 512, 512, device="cuda") * 2).requires_grad_()
        probability = norm(scores)
        loss = probability.square().sum()
        loss.backward()
        if not bool(torch.isfinite(scores.grad).all()) or float(scores.grad.abs().max()) == 0.:
            raise ValueError("D1 反向梯度異常")
        if not bool(torch.isfinite(probability).all()) or not bool(((probability >= 0) & (probability <= 1)).all()):
            raise ValueError("D1 probability 異常")
        row_error = max(row_error, float((probability.detach().sum(-1) - 1).abs().max()))
        if row_error > .005:
            raise ValueError("D1 row-sum 誤差異常；不使用浮點除法掩蓋")
        # The original native fault surfaced at a scalar GPU→CPU sync.
        torch.cuda.synchronize()
        if iteration == 1 or iteration % max(10, args.iterations // 100) == 0:
            write(args.output / "progress.json", {"iteration": iteration, "iterations": args.iterations,
                  "time_unix": time.time(), "seconds": time.time() - start, "max_row_sum_error": row_error})
        del scores, probability, loss
    result = {"status": "passed", "iterations": args.iterations, "score_shape": [8, 4, 512, 512],
              "seconds": time.time() - start, "cpu_gpu_probability_bitwise_equal": True,
              "invalid_inputs_rejected": True, "finite_nonzero_gradients": True,
              "max_row_sum_error": row_error, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "CUDA_LAUNCH_BLOCKING": os.getenv("CUDA_LAUNCH_BLOCKING", "0"),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "script_sha256": digest(__file__), "task_weights_saved": False,
              "limitation": "短時間合成D1診斷；未重現間歇性原生故障，不是完整任務AP或長訓穩定性保證"}
    write(args.output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
