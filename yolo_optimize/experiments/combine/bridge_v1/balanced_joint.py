"""報告實驗4啟發：固定任務權重校準，最終驗收與早期安全分離。"""
from dataclasses import replace
import json
import sys
import torch
import merge_joint as base
import runtime
from safe_source import HERE, sha256

J0_SHA = '42c48372c451e87112e63399acc064608088322711d1f311af9c2cb51c3a2c19'


def training_safety(epoch_count, coco_delta, pose_delta):
    if any(v < -.05 for v in coco_delta.values()):
        return 'catastrophic_coco_drop'
    if any(v < -.08 for v in pose_delta.values()):
        return 'catastrophic_pose_drop'
    if epoch_count >= 5 and any(v < -.02 for v in coco_delta.values()):
        return 'coco_not_recovered_after_grace'
    return None


def install():
    assert sha256(base.J0) == J0_SHA
    calibration = json.loads((HERE/'artifacts/task-weight-calibration-v1/summary.json').read_text())
    assert calibration['status'] == 'passed' and calibration['selected_pose_weight'] == .045
    config, stage = base.install()
    config = replace(config, pose_weight=.045, maximum_map_drop=.02)
    config._validate()
    assert config.preflight().ready
    return config, stage


class Session(runtime.Session):
    def __init__(self, *args, **kwargs):
        kwargs['run_name'] = {'j1-smoke-v1':'balanced-j1-smoke-v1',
            'j1-bridge-v1':'balanced-j1-v2'}[kwargs['run_name']]
        super().__init__(*args, **kwargs)
        payload = torch.load(base.J0, map_location='cpu', weights_only=True)
        self.pose_start = payload['metadata']['metrics']

    def _resolved_config(self):
        result = super()._resolved_config()
        result['balanced_joint_policy'] = {'pose_weight':.045, 'weight_calibrated_train_only':True,
            'final_coco_drop':.005, 'final_other_ap_drop':.02,
            'training_catastrophic_coco_drop':.05, 'training_catastrophic_pose_drop_from_start':.08,
            'training_coco_drop_at_epoch5_onward':.02,
            'source_sha256':J0_SHA, 'source':str(base.J0), 'not_pdf_exact_reproduction':True}
        return result

    def _save_selected(self, labels, **kwargs):
        # 直接使用正式保存，刻意不執行舊 runtime 每輪0.005立即停止；由下方新政策判定。
        saved = runtime.formal.FormalJointTrainingSession._save_selected(self, labels, **kwargs)
        metrics = kwargs['metrics']
        baseline = self.config.load_baseline()
        coco_delta = {k:metrics[k]-v for k,v in baseline.items() if k.startswith('coco/')}
        pose_delta = {k:metrics[k]-self.pose_start[k] for k in baseline if k.startswith('bbat/')}
        epoch_count = kwargs['progress'].next_epoch
        reason = training_safety(epoch_count, coco_delta, pose_delta)
        if reason:
            with (self.run_dir/'safety-stop.json').open('x') as f:
                json.dump({'status':'SAFETY_STOP','reason':reason,'epochs_completed':epoch_count,
                    'coco_delta':coco_delta,'pose_delta_from_start':pose_delta,
                    'saved':{k:str(v) for k,v in saved.items()}},f,indent=2)
            raise runtime.SafetyStop(reason)
        return saved


if __name__ == '__main__':
    # 單純門檻回歸：允許報告中的早期暫降，不接受持續或災難性退化。
    assert training_safety(1, {'person':-.03},{'pose':-.01}) is None
    assert training_safety(5, {'person':-.03},{'pose':-.01}) == 'coco_not_recovered_after_grace'
    assert training_safety(1, {'person':-.051},{'pose':0}) == 'catastrophic_coco_drop'
    assert training_safety(1, {'person':0},{'pose':-.081}) == 'catastrophic_pose_drop'
    assert training_safety(5, {'person':-.004},{'pose':0}) is None
    base.train_joint.install = base.smoke_joint.install = install
    base.train_joint.Session = base.smoke_joint.Session = Session
    base.smoke_joint.AdaptedBridgeSource = base.JointSource
    if '--smoke' in sys.argv:
        base.smoke_joint.main()
    else:
        assert json.loads((HERE/'artifacts/fusion/balanced-j1-smoke-v1/summary.json').read_text())['status']=='passed'
        base.train_joint.main()
