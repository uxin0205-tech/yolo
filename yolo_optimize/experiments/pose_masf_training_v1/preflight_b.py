"""B 組 CPU 契約檢查；不啟動 CUDA、不拿合成梯度當成實際精度。"""
import copy
import json
from pathlib import Path
import torch
from training_b import *
from yolo_combine.graph_materialize import build_graph_validation_models

def equal(a,b):
    aa,bb=tensors(a),tensors(b)
    assert len(aa)==len(bb)
    for x,y in zip(aa,bb): torch.testing.assert_close(x,y,rtol=0,atol=0)

def main():
    initialize(); torch.set_num_threads(4); torch.manual_seed(1)
    assert not torch.cuda.is_initialized()
    cfg=json.loads(CONFIG.read_text())
    out=HERE/'artifacts/cpu-preflight-v1'
    assert not (out/'summary.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    model,source=build(); base,_=parent.build()
    model.eval(); x=torch.rand(1,3,160,160)
    with torch.no_grad():
        equal(base(x,task='detect'),model(x,task='detect'))
        equal(base(x,task='pose'),model(x,task='pose'))
    for kind in ('float','bittrue'):
        mats=build_graph_validation_models(model,source,kind=kind)
        assert type(mats.pose.model[-1]) is PoseTrainingMASF
        assert mats.pose.model[-1].bridge_coefficient==1
        for m in (mats.detect,mats.pose):verify_pwl(m)
        assert all(torch.equal(v,mats.pose.model[-1].p3_masf.state_dict()[n]) for n,v in model.pose_head.p3_masf.state_dict().items())
    del mats,base
    guard=FrozenGuard(model); set_training(model)
    bundle=state_bundle(model,cfg,torch.device('cpu'))
    optimizer=bundle['optimizer']; scheduler=bundle['scheduler']
    names=[n for g in optimizer.param_groups for n in g['param_names']]
    assert set(names)==guard.active and len(names)==len(set(names))
    for g in optimizer.param_groups:
        if g['group_name'].endswith('/no_decay'):assert g['weight_decay']==0
    feature_shapes=[(2,c,s,s) for c,s in ((256,8),(512,4),(512,2))]
    gradients=[]
    for step in range(2):
        scheduler.prepare_step(); optimizer.zero_grad(set_to_none=True)
        features=[torch.randn(shape,requires_grad=True) for shape in feature_shapes]
        # 只取部署 one2one 合成目標，確認 bridge 真正回到 MASF、不回傳 trunk。
        outputs=model.pose_head(features)['one2one']
        objective=sum(t.square().mean() for k in ('boxes','scores','kpts') for t in tensors(outputs[k]) if t.requires_grad)
        objective.backward()
        assert all(f.grad is None for f in features)
        alpha=model.pose_head.p3_masf.alpha
        grad=[p.grad for n,p in model.pose_head.p3_masf.named_parameters() if n!='alpha' and p.grad is not None]
        row={'step':step+1,'alpha':float(alpha.detach()),'alpha_gradient':float(alpha.grad),
             'context_gradient_max':max(float(g.abs().max()) for g in grad)}
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        gradients.append(row)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10)
        optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.advance(); bundle['ema'].update(model)
        guard.check(model); guard.check(bundle['ema'].ema)
    assert gradients[0]['alpha']==0 and gradients[0]['alpha_gradient']!=0 and gradients[0]['context_gradient_max']==0
    assert gradients[1]['alpha']!=0 and gradients[1]['context_gradient_max']>0
    # 不建立另一套資料切分；核對完整資料數下的分组數學。
    sizes=[16]*372+[12]
    groups=list(batches_of(sizes,8))
    assert len(groups)==47 and sum(map(sum,groups))==5964 and sum(groups[-1])==76
    snapshot=out/'cpu-test-resume.pt'
    saved=save_training_snapshot(snapshot,model=model,**bundle,progress=TrainingProgress('pose_b',0,2,0),
        resolved_config=cfg,provenance=source.provenance(),loader_state={'synthetic_test_only':True},best_state={})
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():model.pose_head.p3_masf.alpha.add_(1)
    restored=load_training_snapshot(snapshot,model=model,**bundle)
    assert restored.progress.global_macro_step==2
    assert all(torch.equal(dict(model.named_parameters())[n],v) for n,v in before.items())
    guard.check(model);guard.check(bundle['ema'].ema)
    assert not torch.cuda.is_initialized()
    save(out/'summary.json',{'status':'passed','gpu_used':False,'accuracy_evaluated':False,
        'parent_sha256':sha256(parent.CHECKPOINT),'initial_detect_and_pose_exact_parent':True,
        'materialization_float_bittrue':True,'beta':1,'one2one_trunk_detached':True,
        'optimizer_trainable_names_exact':True,'frozen_live_and_ema_exact_after_2_steps':True,
        'gradients':gradients,'accumulation':{'microbatches':373,'updates':47,'images':5964,'tail_images':76},
        'full_resume_roundtrip':True,'cpu_snapshot_sha256':saved.sha256,
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters':sum(p.numel() for p in model.parameters()),
        'limitation':'真實 YOLO head 的合成 one2one 目標；完整 BBAT loss／AMP／GPU 記憶體須由排隊 smoke 驗證'})
    print('PASS B 組 CPU：零閘等價／隔離／梯度啟動／累積／完整續訓 roundtrip',flush=True)

if __name__=='__main__':main()
