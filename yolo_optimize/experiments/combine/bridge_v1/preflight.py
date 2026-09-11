"""Bridge 融合與 alpha=0／真正旁路等價的 CPU 契約。"""
import copy
import json
from safe_source import HERE, BridgeSource, initialize
import torch
from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.graph_materialize import build_graph_validation_models
from verify_qk_challenger import tensors
from pwl_contract import verify_pwl


def exact(a, b):
    a, b = tensors(a), tensors(b)
    assert len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def main():
    initialize()
    torch.set_num_threads(4)
    out = HERE / 'artifacts/preflight-v1.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists()
    source = BridgeSource()
    source.verify_manifest()
    source.verify_environment()
    pair = source.build_task_models()
    image = torch.rand(1, 3, 160, 160, generator=torch.Generator().manual_seed(20261003))
    with torch.inference_mode():
        expected = pair.detect(image)
    shared, report = assemble_graph_shared_model(pair.detect, pair.pose)
    assert report.complete and report.audit.compatible
    counter = [0]
    hook = shared.graph.model[0].register_forward_hook(lambda *args: counter.__setitem__(0, counter[0]+1))
    with torch.inference_mode():
        on = shared(image, task='both')
    hook.remove()
    assert counter[0] == 1
    exact(on['detect'], expected)
    alpha = shared.detect_head.p3_masf.alpha.detach().clone()
    with torch.no_grad():
        shared.detect_head.p3_masf.alpha.zero_()
    with torch.inference_mode():
        off = shared(image, task='both')
    exact(on['pose'], off['pose'])
    context = shared.detect_head.p3_masf
    shared.detect_head.p3_masf = torch.nn.Identity()
    with torch.inference_mode():
        bypass = shared(image, task='both')
    exact(off, bypass)
    shared.detect_head.p3_masf = context
    with torch.no_grad():
        context.alpha.copy_(alpha)
    for kind in ('float', 'bittrue'):
        materialized = build_graph_validation_models(shared, source, kind=kind)
        assert materialized.detect_report.complete and materialized.pose_report.complete
        verify_pwl(materialized.detect)
        verify_pwl(materialized.pose)
        assert hasattr(materialized.detect.model[23], 'p3_masf')
        assert not any('masf' in n for n, _ in materialized.pose.named_modules())
    with out.open('x') as handle:
        json.dump({'status': 'passed', 'source': source.provenance(), 'assembly': report.as_dict(),
            'alpha': float(alpha), 'detect_initialization_exact': True, 'shared_trunk_calls': counter[0],
            'alpha_zero_equals_real_bypass': True, 'pose_alpha_toggle_exact': True,
            'float_bittrue_materialization_complete': True, 'pwl_range': [-10, 0]}, handle, indent=2)
    print('JOB_DONE: Bridge 融合、alpha=0／旁路、Pose 不變、PWL 契約通過。', flush=True)


if __name__ == '__main__':
    main()
