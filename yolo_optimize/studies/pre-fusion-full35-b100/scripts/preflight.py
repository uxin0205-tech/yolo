"""CPU 稽核 B100，禁止默默改成 retained A2。"""
from common import ROOT, SOURCE, registry, sha256, write_json, prepare_coco
import torch
from ultralytics import YOLO
from achitechure_1.model import inspect_yolo26_graph
from dataclasses import asdict


def main():
    record = registry()
    report = {'parent_id': record['id'], 'historical_gate': record['gate'], 'models': {}}
    states = {}
    for backend in ('float', 'bittrue'):
        path = SOURCE / record[backend]
        digest = sha256(path)
        if digest != record[backend + '_sha256']:
            raise ValueError(f'{backend} SHA256 不符')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = YOLO(str(path)).model.float()
        graph = inspect_yolo26_graph(model)
        if graph.p3_index != 16 or graph.detect_inputs != (16, 19, 22):
            raise ValueError('非預期 B100 graph')
        state = model.state_dict()
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError('權重包含非有限值')
        states[backend] = state
        attention = {name: {'config': asdict(m.config),
                    'buffers': {k: v.tolist() for k,v in m.named_buffers() if v.numel() <= 32}}
                    for name,m in model.named_modules()
                    if m.__class__.__name__ == 'HardwareFriendlyAttention'}
        report['models'][backend] = {'path':str(path), 'sha256':digest, 'graph':asdict(graph),
            'checkpoint_keys': list(checkpoint), 'epoch': checkpoint.get('epoch'),
            'updates': checkpoint.get('updates'), 'has_optimizer':checkpoint.get('optimizer') is not None,
            'has_ema': checkpoint.get('ema') is not None, 'parameters':sum(p.numel() for p in model.parameters()),
            'attention':attention, 'masf_alpha':float(model.model[16].p3_masf.alpha.detach())}
    left,right=states['float'],states['bittrue']
    changed=[k for k in left if k not in right or not torch.equal(left[k],right[k])]
    extra=sorted(set(right)-set(left))
    report['float_vs_bittrue_state']={'changed':changed,'extra':extra}
    report['data_yaml']=str(prepare_coco())
    report['status']='passed'
    report['training_performed']=False
    write_json(ROOT/'artifacts/preflight.json', report)
    print('PREFLIGHT_DONE', {'changed_tensors':len(changed),'extra':extra,
          'float_epoch':report['models']['float']['epoch']}, flush=True)


if __name__=='__main__': main()
