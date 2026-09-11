#!/usr/bin/env python3
"""完整雙任務 CPU 等價檢查；不變更資料、checkpoint 或正式訓練入口。"""
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from yolo_optimize import runtime
from yolo_optimize.fixed_scale import install_fixed_early_return
from yolo_optimize.qk_diagnostic import SITES
from yolo_combine.graph_materialize import build_graph_validation_models
from verify_rep17_integration import compare
import torch


def main():
    torch.set_num_threads(2)
    torch.manual_seed(8)
    output = ROOT / 'artifacts/direction1-20260908/fixed-scale-equivalence'
    output.mkdir(exist_ok=False)
    checkpoint = runtime.FINAL_ROOT / 'weights/combined/inference/best_joint.pt'
    source, model, _, _ = runtime.load_model(runtime.load_config(output.parent), checkpoint, torch.device('cpu'))
    model.eval()
    x = torch.randn(1, 3, 160, 160)
    checks = []
    for kind in ('float', 'bittrue'):
        models = build_graph_validation_models(model, source, kind=kind)
        for task, graph in (('detect', models.detect), ('pose', models.pose)):
            graph.eval()
            with torch.no_grad():
                before = graph(x)
                state = {key: value.clone() for key, value in graph.state_dict().items()}
                scores = dict(graph.named_modules())
                for path in SITES.values():
                    install_fixed_early_return(scores[path])
                after = graph(x)
            compare(before, after, exact=True)
            compare(state, graph.state_dict(), exact=True)
            checks.append({'backend': kind, 'task': task, 'output_exact': True, 'state_exact': True})
    runtime.write_json(output / 'summary.json', {'status': 'passed', 'device': 'cpu', 'imgsz': 160,
        'checks': checks, 'checkpoint_sha256': runtime.sha256(checkpoint),
        'full_dataset_ap_validated': False, 'hardware_latency_measured': False})
    print('JOB_DONE: Float／BitTrue 雙任務輸出與 state 完全相等', flush=True)


if __name__ == '__main__':
    main()
