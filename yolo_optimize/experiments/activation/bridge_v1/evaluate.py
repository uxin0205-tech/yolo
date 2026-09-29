"""P3 bridge 的獨立 activation 起點／零樣本驗證，不沿用舊 combine 權重。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'combine/bridge_v1'))
from safe_source import BridgeSource, initialize, SOURCE, sha256
sys.path.insert(0,'/home/uxin/yolo/yolo_activation/src')
from activation_lab.activations import build_activation
import torch
import json
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from pwl_contract import verify_pwl
from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.validation import JointValidator, ValidationSettings
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.metrics import GATE_METRICS
import yolo_combine.validation as validation
from validate import InternalValidator
from common import prepare_coco

START = ROOT/'combine/bridge_v1/artifacts/fusion/j3-pose-head-recovery-v1/inference/best_pose.pt'
SHA = 'd5b2083b2b736ded790578a2779d223d82413ce0c6cbac1468601105c094f9a6'
ARMS = ('silu','qsilu_pq','hardswish','poly_shift')


def replace_activations(model, arm):
    paths = [n for n,m in model.named_modules(remove_duplicate=False) if isinstance(m,torch.nn.SiLU)]
    assert paths
    if arm != 'silu':
        for n in paths: model.set_submodule(n,build_activation(arm))
        assert not any(isinstance(m,torch.nn.SiLU) for m in model.modules())
    return paths


class Source(BridgeSource):
    def __init__(self,*args,activation='silu',**kwargs):
        super().__init__(*args,**kwargs)
        assert activation in ARMS
        self.activation = activation
        assert sha256(START) == SHA
        self.record['detect_expected_metrics'] = {k:torch.load(START,map_location='cpu',weights_only=True)['metadata']['metrics'][k]
            for k in self.record['detect_expected_metrics']}

    def provenance(self,kind='float'):
        return {**super().provenance(kind),'source_kind':'optimized_P3_bridge_activation',
            'activation_start':str(START),'activation_start_sha256':SHA,'activation':self.activation,
            'source_is_not_accepted_best_joint':True}

    def build_task_models(self,kind='float',*,pose_head_checkpoint=None):
        pair = super().build_task_models('float',pose_head_checkpoint=pose_head_checkpoint)
        assert sha256(START) == SHA
        full = torch.load(START,map_location='cpu',weights_only=True)['state_dict']
        for task,model in [('detect',pair.detect),('pose',pair.pose)]:
            state = {n:full['graph.model.23.'+task+'_head.'+n[len('model.23.'):]]
                if n.startswith('model.23.') else full['graph.'+n] for n in model.state_dict()}
            model.load_state_dict(state,strict=True)
            convert_yolo26_model(model,VariantConfig.from_yaml(SOURCE/f'configs/attention/{kind}-pwl-final.yaml'))
            verify_pwl(model)
            replace_activations(model,self.activation)
        return pair


def main():
    initialize(); torch.set_num_threads(4)
    if '--preflight' in sys.argv:
        output = {}
        x = torch.rand(1,3,160,160,generator=torch.Generator().manual_seed(1))
        reference = None
        for arm in ARMS:
            pair = Source(activation=arm).build_task_models()
            graph,report = assemble_graph_shared_model(pair.detect,pair.pose)
            assert report.complete
            state = graph.state_dict()
            if reference is None: reference = {k:v.clone() for k,v in state.items()}
            assert all(torch.equal(v,reference[k]) for k,v in state.items())
            with torch.inference_mode():
                pred = graph.eval()(x,task='both')
            from verify_qk_challenger import tensors
            assert all(torch.isfinite(t).all() for t in tensors(pred))
            output[arm] = {'parameters_unchanged':True,'finite_both_heads':True,
                'activation_references':sum(type(m).__name__ in ('SiLU','ShiftPiecewiseQuadraticSiLU','Hardswish','IntegralPolynomialActivation')
                    for n,m in graph.named_modules(remove_duplicate=False))}
        HERE.joinpath('artifacts').mkdir(exist_ok=True)
        with (HERE/'artifacts/preflight-v1.json').open('x') as f: json.dump(output,f,indent=2)
        print('PASS activation CPU preflight',flush=True)
        return
    assert (HERE/'artifacts/preflight-v1.json').exists()
    out = HERE/'artifacts/zero-shot-v1'; out.mkdir(exist_ok=False)
    coco = prepare_coco()
    view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml',HERE/'artifacts/datasets/bbat5-v1-runtime')
    validation.DetectionValidator = InternalValidator
    results = {}
    for arm in ARMS:
        source = Source(activation=arm)
        pair = source.build_task_models()
        graph,report = assemble_graph_shared_model(pair.detect,pair.pose)
        assert report.complete
        validator = JointValidator(source,detect_data_yaml=coco,pose_data_yaml=view.yaml,output_root=out/arm,
            settings=ValidationSettings(imgsz=640,detect_batch_size=32,pose_batch_size=16,detect_workers=4,
                pose_workers=4,device='0',plots=False,save_coco_json=False))
        results[arm] = {}
        for kind in ('float','bittrue'):
            result = validator.validate(graph.eval(),epoch=0,kind=kind)
            assert len(InternalValidator.last_instance.dataloader.dataset) == 5000
            results[arm][kind] = dict(result.metrics)
            del result
            InternalValidator.last_instance = None
        if arm == 'silu':
            baseline = torch.load(START,map_location='cpu',weights_only=True)['metadata']['metrics']
            assert all(abs(results[arm]['bittrue'][k]-baseline[k]) < 1e-8 for k in GATE_METRICS)
        with (out/(arm+'.json')).open('x') as f: json.dump(results[arm],f,indent=2)
    with (out/'summary.json').open('x') as f:
        json.dump({'status':'completed','source':str(START),'sha256':SHA,'results':results,
            'delta_vs_silu':{arm:{k:results[arm]['bittrue'][k]-results['silu']['bittrue'][k] for k in GATE_METRICS} for arm in ARMS},
            'note':'activation 相對比較不代表已通過原融合 gate；尚未短訓練或 KD。'},f,indent=2)
    print('JOB_DONE: activation 四臂 Float／Bit-True 全量驗證完成',flush=True)


if __name__ == '__main__': main()
