"""方向 1 兩候選的獨立匯出與完整驗證；不以小幅差異宣稱方法成功。"""
import json
import subprocess
import sys
from common import ROOT, setup, prepare_coco, write_json, sha256
import torch
from ultralytics import YOLO
from export_model import export
from continue_a0 import validate


def main():
    output = ROOT / 'artifacts/direction1-candidate-verification-v1'
    output.mkdir(parents=True, exist_ok=False)
    setup()
    torch.set_num_threads(4)
    data = prepare_coco()
    image_list = ROOT / 'artifacts/late-e8-inference-verification/person-first8.txt'
    images = image_list.read_text().splitlines()
    assert len(images) == 8
    reports = []
    for name, run in [('control-e8', 'masf-head-control-v1'), ('masf-e8', 'masf-task-bridge-v1')]:
        source = ROOT / 'artifacts' / run / 'epoch-08-resume.pt'
        original = json.loads((source.parent / 'summary.json').read_text())
        assert original['status'] == 'complete'
        expected = next(e['ema'] for e in original['epochs'] if e['epoch'] == 8)
        weights = export(source, output / f'{name}-bittrue.pt')
        model = YOLO(str(weights)).model
        metrics = validate(model, 'bittrue', data, output / f'{name}-full-coco')
        deltas = {k: metrics[k] - expected[k] for k in expected}
        assert all(abs(v) <= 1e-8 for v in deltas.values()), deltas
        subprocess.run([sys.executable, str(ROOT / 'scripts/infer.py'), '--weights', str(weights),
                        '--source', str(image_list), '--name', f'direction1-{name}-first8', '--conf', '.25'], check=True)
        reports.append({'name': name, 'run': run, 'epoch': 8, 'state': 'ema',
                        'snapshot': str(source), 'snapshot_sha256': sha256(source),
                        'inference': str(weights), 'inference_sha256': sha256(weights),
                        'full_coco_images': 5000, 'metrics': metrics, 'export_ap_delta': deltas,
                        'inference_images': 8, 'inference_imgsz': 640, 'confidence': .25,
                        'qualitative_review_done': False, 'hardware_latency_measured': False,
                        'method_accuracy_gain_accepted': False})
        write_json(output / 'summary.json', {'status': 'running', 'candidates': reports})
        del model
    write_json(output / 'summary.json', {'status': 'passed', 'candidates': reports,
               'candidate_rule': 'control best overall EMA; MASF bridge best overall EMA; both from completed E6-E10, not matched-budget method proof',
               'validation_set_used_for_selection': True, 'independent_test_evaluated': False,
               'deployment_choice_made': False, 'combine_started': False})
    print('ALL_DONE: 方向 1 兩候選全 COCO 匯出重驗與固定 8 張推論通過', flush=True)


if __name__ == '__main__':
    main()
