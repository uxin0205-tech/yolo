"""CPU-only 融合來源稽核，不組裝、不訓練、不改動任何來源權重。"""
import dataclasses
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'studies/pre-fusion-full35-b100/scripts'))
from common import setup, sha256, SOURCE
sys.path.insert(0, '/home/uxin/yolo/yolo_combine/src')
import torch
import yaml
from yolo_combine.fusion_model import audit_task_pair
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model


def main():
    destination = HERE / 'artifacts/source-audit-v1.json'
    if destination.exists():
        raise FileExistsError(destination)
    setup()
    torch.set_num_threads(4)
    configuration = Path('/home/uxin/yolo/yolo_combine/variants/full35/configs/joint.yaml')
    config = yaml.safe_load(configuration.read_text())
    pose_path = Path(config['source']['pose_checkpoint'])
    assert pose_path.is_file()
    payload = torch.load(pose_path, map_location='cpu', weights_only=False)
    pose = (payload.get('ema') or payload['model']).float().eval()
    convert_yolo26_model(pose, VariantConfig.from_yaml(SOURCE / 'configs/attention/bittrue-pwl-final.yaml'))
    summary_path = ROOT / 'studies/pre-fusion-full35-b100/artifacts/direction1-candidate-verification-v1/summary.json'
    verified = json.loads(summary_path.read_text())
    assert verified['status'] == 'passed'
    results = []
    for candidate in verified['candidates']:
        source = Path(candidate['inference'])
        assert sha256(source) == candidate['inference_sha256']
        model = torch.load(source, map_location='cpu', weights_only=False)['model'].float().eval()
        audit = audit_task_pair(model, pose)
        results.append({'candidate': candidate['name'], 'source': str(source),
                        'sha256': candidate['inference_sha256'], 'audit': dataclasses.asdict(audit),
                        'detect_layer16_type': type(model.model[16]).__name__,
                        'pose_layer16_type': type(pose.model[16]).__name__})
        del model
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'audit_complete', 'gpu_used': False, 'training_started': False,
              'assembly_attempted': False, 'candidate_selected': None,
              'pose_source': str(pose_path), 'pose_sha256': sha256(pose_path),
              'normalization_comparison': 'pose converted in memory to same Bit-True PWL backend',
              'config_sha256': sha256(configuration), 'candidates': results}
    with destination.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'audit_complete', 'candidates': [
        {'name': r['candidate'], 'compatible': r['audit']['compatible'],
         'difference_count': len(r['audit']['differences']),
         'difference_fields': [d.split(':', 1)[0] for d in r['audit']['differences']]}
        for r in results]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
