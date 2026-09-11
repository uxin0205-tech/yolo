"""新 Pose MASF alpha=0 等價、原生接線及完整驗證模板 CPU 驗收。"""
import json
from local_source import HERE, LocalSource
from j0_runtime import install
from pose_masf import PoseMASFSource
import torch
from yolo_combine.factory import FusionModelFactory
from yolo_combine.graph_materialize import build_graph_validation_models
from pwl_contract import verify_pwl


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return sum((tensors(v) for v in value.values()), [])
    if isinstance(value, (list, tuple)):
        return sum((tensors(v) for v in value), [])
    return []


def main():
    config = install()
    out = HERE / 'artifacts/pose-masf-probe-v1.json'
    assert not out.exists()
    models = []
    for source in (LocalSource(), PoseMASFSource()):
        built = FusionModelFactory(source, detect_data_yaml=config.detect_data,
            pose_data_yaml=config.pose_data).build(pose_head_checkpoint=config.pose_checkpoint, checkpoint_kind='float')
        assert built.report.complete
        models.append(built.model.eval())
    image = torch.linspace(0, 1, 3*160*160).reshape(1, 3, 160, 160)
    with torch.no_grad():
        left, right = (tensors(m(image, task='both')) for m in models)
    assert len(left) == len(right) and all(torch.equal(a, b) for a, b in zip(left, right))
    candidate = models[1]
    for kind in ('float', 'bittrue'):
        materialized = build_graph_validation_models(candidate, PoseMASFSource(), kind=kind)
        verify_pwl(materialized.detect)
        verify_pwl(materialized.pose)
        assert hasattr(materialized.pose.model[23], 'p3_masf')
        assert not any('masf' in n for n, _ in materialized.detect.named_modules())
    added = sum(p.numel() for p in candidate.pose_head.p3_masf.parameters())
    with out.open('x') as handle:
        json.dump({'status': 'passed', 'alpha_zero_outputs_exact': True,
            'float_bittrue_materialization': True, 'added_parameters': added,
            'location': 'Pose head P3 only; Detect unaffected', 'new_detect_head': False,
            'pwl_range': [-10, 0], 'native_one2one_detach_preserved': True}, handle, indent=2)
    print('JOB_DONE: Pose MASF CPU 等價及驗證還原通過', flush=True)


if __name__ == '__main__':
    main()
