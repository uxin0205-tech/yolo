"""獨立重建 qSiLU E2 與 FP-QK teacher 候選；只驗證、不假稱已訓練 teacher。"""
import copy
import json
import torch
from evaluate import HERE, ROOT, Source, initialize, sha256, SOURCE
from yolo_attention.binary_basis import BinaryScore
from yolo_attention.config import BasisKind, VariantConfig
from yolo_attention.integration import convert_yolo26_model
from pwl_contract import verify_pwl
from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.validation import JointValidator, ValidationSettings
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.metrics import GATE_METRICS
import yolo_combine.validation as validation
from validate import InternalValidator
from common import prepare_coco

SELECTED = HERE/'artifacts/runs/qsilu_pq-short-e10-seed1-v1/inference/best_joint.pt'
SELECTED_SHA = '1bbbbe0f87cb853827af1bb625e7066178e3fb2c7e14c410d997efb24e94834a'


class SelectedSource(Source):
    def __init__(self,*args,teacher=False,**kwargs):
        super().__init__(*args,activation='qsilu_pq',**kwargs)
        self.teacher = teacher
        assert sha256(SELECTED) == SELECTED_SHA

    def provenance(self,kind='float'):
        return {**super().provenance(kind),'selected':str(SELECTED),'selected_sha256':SELECTED_SHA,
            'score_override':'FP qTk/sqrt(d), existing relative bias retained' if self.teacher else 'unchanged BinaryQK',
            'teacher_candidate_not_fp_trained':self.teacher}

    def build_task_models(self,kind='float',*,pose_head_checkpoint=None):
        pair = super().build_task_models('float',pose_head_checkpoint=pose_head_checkpoint)
        full = torch.load(SELECTED,map_location='cpu',weights_only=True)['state_dict']
        for task,model in [('detect',pair.detect),('pose',pair.pose)]:
            state = {n:full['graph.model.23.'+task+'_head.'+n[len('model.23.'):]]
                if n.startswith('model.23.') else full['graph.'+n] for n in model.state_dict()}
            model.load_state_dict(state,strict=True)
            convert_yolo26_model(model,VariantConfig.from_yaml(SOURCE/f'configs/attention/{kind}-pwl-final.yaml'))
            if self.teacher:
                count = 0
                for m in model.modules():
                    if type(m).__name__ == 'HardwareFriendlyAttention':
                        assert m.progressive is None
                        # 只覆寫 score 計算；不是官方 P0（其契約不允許相對 bias）。
                        # 保留學生已學的相對 bias 與 PWL，明確記錄 teacher override。
                        m.score = BinaryScore(num_heads=m.num_heads,basis=BasisKind.FP,
                            scale_mode=m.score.scale_mode,use_ste=False)
                        count += 1
                assert count == 2
            verify_pwl(model)
        return pair


def main():
    initialize(); torch.set_num_threads(4)
    out = HERE/'artifacts/selected-and-teacher-probe-v1';out.mkdir(exist_ok=False)
    coco = prepare_coco()
    view = prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml',HERE/'artifacts/datasets/bbat5-v1-runtime')
    validation.DetectionValidator = InternalValidator
    results = {}
    for teacher in (False,True):
        label = 'fp-qk-teacher-candidate' if teacher else 'selected-binary-student'
        source = SelectedSource(teacher=teacher)
        pair = source.build_task_models()
        model,report = assemble_graph_shared_model(pair.detect,pair.pose)
        assert report.complete
        if teacher:
            for m in model.modules():
                if type(m).__name__ == 'HardwareFriendlyAttention':
                    q=torch.randn(1,m.num_heads,m.key_dim,9); k=torch.randn_like(q)
                    torch.testing.assert_close(m.score(q,k),(q*(m.key_dim**-.5)).transpose(-2,-1)@k,rtol=0,atol=0)
        validator = JointValidator(source,detect_data_yaml=coco,pose_data_yaml=view.yaml,output_root=out/label,
            settings=ValidationSettings(imgsz=640,detect_batch_size=32,pose_batch_size=16,detect_workers=4,
                pose_workers=4,device='0',plots=False,save_coco_json=False))
        results[label] = {}
        for kind in ('float','bittrue'):
            result=validator.validate(model.eval(),epoch=0,kind=kind)
            results[label][kind]=dict(result.metrics)
            assert len(InternalValidator.last_instance.dataloader.dataset)==5000
            if not teacher and kind=='bittrue':
                expected=torch.load(SELECTED,map_location='cpu',weights_only=True)['metadata']['metrics']
                assert all(abs(result.metrics[k]-expected[k])<1e-8 for k in GATE_METRICS)
            del result;InternalValidator.last_instance=None
        with (out/(label+'.json')).open('x') as f:json.dump(results[label],f,indent=2)
    with (out/'summary.json').open('x') as f:
        json.dump({'status':'completed','results':results,'teacher_not_fp_trained':True,
            'note':'teacher 僅為切換成真正 FP-QK 的候選；是否適合 KD 必須依任務優勢與梯度再判斷。'},f,indent=2)
    print('JOB_DONE: qSiLU 匯出重建與 FP-QK joint teacher 候選驗證完成',flush=True)


if __name__ == '__main__': main()
