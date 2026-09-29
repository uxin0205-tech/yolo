"""固定 E2 權重的 Pose P3 MASF 推論比較；不是新增 Pose MASF 訓練。"""
import argparse
import math
from pose_candidate import *
from yolo_combine.validation import JointValidator,ValidationSettings
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.metrics import GATE_METRICS
from yolo_combine.resume import save_inference_weights
import yolo_combine.validation as validation
from validate import InternalValidator
from common import prepare_coco

CASES=('baseline','pose_masf','pose_alpha_zero')

def evaluate(case):
    out=HERE/'artifacts/comparison-v1'/case
    result_path=out/'summary.json'
    if result_path.exists():
        result=json.loads(result_path.read_text());assert result['status']=='passed';return result
    out.mkdir(parents=True,exist_ok=False)
    model,source=build(enabled=case!='baseline',zero_alpha=case=='pose_alpha_zero')
    validation.DetectionValidator=InternalValidator
    old_pose=validation.PoseValidator
    class CountPose(old_pose):
        last_instance=None
        def init_metrics(self,model):
            super().init_metrics(model);type(self).last_instance=self
    validation.PoseValidator=CountPose
    try:
        view=prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml',HERE/'artifacts/datasets/bbat5-v1-runtime')
        validator=JointValidator(source,detect_data_yaml=prepare_coco(),pose_data_yaml=view.yaml,
            output_root=out/'validation',settings=ValidationSettings(imgsz=640,detect_batch_size=32,
            pose_batch_size=16,detect_workers=4,pose_workers=4,device='0',plots=False,save_coco_json=False))
        result=validator.validate(model.eval(),epoch=1,kind='bittrue')
        assert len(InternalValidator.last_instance.dataloader.dataset)==5000
        assert len(CountPose.last_instance.dataloader.dataset)==683
        metrics=dict(result.metrics);assert all(math.isfinite(v) for v in metrics.values())
        expected=payload()['metadata']['metrics']
        invariant=GATE_METRICS if case!='pose_masf' else [k for k in GATE_METRICS if k.startswith('coco/')]
        assert all(abs(metrics[k]-expected[k])<1e-8 for k in invariant),'同權重對照或 Detect 隔離未重現'
        checkpoint=CHECKPOINT
        if case=='pose_masf':
            checkpoint=out/'pose-masf-transferred-inference.pt'
            save_inference_weights(checkpoint,model=model,metadata={'metrics':metrics,'epoch':1,
                'parent_sha256':sha256(CHECKPOINT),'pose_masf_pose_training_performed':False,
                'not_selected_production_model':True,'source_module':'pose_candidate.PoseSource(enabled=True)'})
        summary={'status':'passed','case':case,'metrics':metrics,'parameters':sum(p.numel() for p in model.parameters()),
            'checkpoint':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,'sha256':sha256(checkpoint),
            'parent_sha256':sha256(CHECKPOINT),'pose_masf_pose_training_performed':False,
            'coco_val_images':5000,'bbat_val_images':683,'dataset_split_unchanged':True,'pwl':[-10,0,20],
            'case_checkpoint_note':'移接候選' if case=='pose_masf' else '既有基準；alpha_zero 的等價檢查不另存重複權重'}
        save(result_path,summary);return summary
    finally:validation.PoseValidator=old_pose

def report():
    rows={case:json.loads((HERE/'artifacts/comparison-v1'/case/'summary.json').read_text()) for case in CASES}
    assert all(row['status']=='passed' for row in rows.values())
    base=rows['baseline']['metrics'];candidate=rows['pose_masf']['metrics']
    result={'status':'completed','cases':rows,'delta_pose_masf_vs_baseline':{k:candidate[k]-base[k] for k in GATE_METRICS},
        'training_performed':False,'default_model_replaced':False,
        'limitation':'MASF 從 Detect 複製後接到 Pose P3，未用 BBAT5 對新分支訓練；結果只代表移接與開關效果。'}
    save(HERE/'artifacts/summary-v1.json',result)
    lines=['# Pose P3 MASF 優先比較結果','',result['limitation'],'',
        '| AP50–95 | 原 Pose | Pose＋MASF | 差值 |','| --- | ---: | ---: | ---: |']
    for key in GATE_METRICS:lines.append(f'| {key} | {base[key]:.6f} | {candidate[key]:.6f} | {candidate[key]-base[key]:+.6f} |')
    lines+=['','同一 native QK E2 起點；Detect MASF、Attention 與資料皆不變。未自動升版。',
        '',f"參數：{rows['baseline']['parameters']:,} → {rows['pose_masf']['parameters']:,}；實際候選權重 bytes：{rows['pose_masf']['checkpoint_bytes']:,}。",
        '', 'MAC／FLOPs、延遲、能耗與目標硬體表現尚未量測，不以驗證耗時冒充推論 benchmark。',
        '', '若要判断 Pose MASF 的訓練收益，下一階段需固定共享特徵與 Detect，作 Pose head 等預算配對訓練；先校準 bridge 的任務梯度，再進行正式比較。']
    with (HERE/'RESULTS.md').open('x') as stream:stream.write('\n'.join(lines)+'\n')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=[*CASES,'report']);args=parser.parse_args()
    initialize();torch.set_num_threads(4)
    assert json.loads((HERE/'artifacts/preflight-v1.json').read_text())['status']=='passed'
    if args.case=='report':report()
    else:evaluate(args.case)
    print('JOB_DONE '+args.case,flush=True)

if __name__=='__main__':main()
