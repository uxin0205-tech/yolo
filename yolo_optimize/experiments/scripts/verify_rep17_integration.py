#!/usr/bin/env python3
"""真實 Full35 CPU 整圖與完整 snapshot 重建；合成更新不作 AP 證據。"""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from yolo_optimize import runtime
from yolo_optimize.repconv import evaluation_copy
from yolo_optimize.training import _build, _state_digest, validation_boundary
from yolo_combine.resume import save_training_snapshot,load_training_snapshot,TrainingProgress
from yolo_combine.graph_materialize import build_graph_validation_models
import torch

def compare(a,b,exact=False):
    if isinstance(a,torch.Tensor):
        if exact: assert torch.equal(a,b)
        else: torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-4)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: compare(a[key],b[key],exact)
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): compare(x,y,exact)
    else: assert a==b

def main():
    torch.set_num_threads(2)
    root=ROOT/'artifacts/direction1-20260908/rep17-integration-cpu'
    root.mkdir(exist_ok=False)
    config=runtime.load_config(root.parent)
    data=runtime.prepare_data(config,root.parent/'datasets')
    def build(variant):
        return _build(config,runtime.FINAL_ROOT/'weights/combined/inference/best_joint.pt',
            runtime.FINAL_ROOT/'weights/combined/full-resume/best_joint.pt',data,32,variant,'cpu',ema_age_mode='parent')
    native=build('native')
    state=build('rep17')
    assert state.metadata['parent_initial_state_sha256']==native.metadata['base_state_sha256']
    assert state.model.contract()['reparameterization']['layer']==17
    assert not state.model.base.graph.model[17].conv2.bn.training
    native.model.base.eval(); state.model.base.eval()
    x=torch.randn(1,3,160,160)
    with torch.no_grad():
        ref=native.model.base(x)
        out=state.model.base(x)
    compare(ref,out,exact=True)
    digest=_state_digest(state.model)
    rng=torch.get_rng_state().clone()
    folded=evaluation_copy(state.model.base)
    assert torch.equal(rng,torch.get_rng_state()) and digest==_state_digest(state.model)
    with torch.no_grad(): compare(out,folded(x))
    with validation_boundary(state.model.base,native.model.base):
        for kind in ('float','bittrue'):
            a=build_graph_validation_models(native.model.base,state.source,kind=kind)
            b=build_graph_validation_models(folded,state.source,kind=kind)
            with torch.no_grad():
                compare(a.detect(x),b.detect(x)); compare(a.pose(x),b.pose(x))
            del a,b
    print('CPU 整圖兩任務 Float／BitTrue parity 通過',flush=True)
    del native,folded
    z=torch.randn(2,256,17,17)
    def step(s):
        s.training_mode(); s.scheduler.prepare_step()
        s.model.base.graph.model[17](z).square().mean().backward()
        s.optimizer.step(); s.optimizer.zero_grad(set_to_none=True)
        s.ema.update(s.model); s.scheduler.advance()
        s.guard.assert_unchanged(s.model.base)
    step(state)
    branch=state.model.base.graph.model[17].conv2.bn.weight
    assert bool(branch.ne(0).any())
    path=root/'synthetic-full-resume.pt'
    save_training_snapshot(path,model=state.model,ema=state.ema,optimizer=state.optimizer,
        scheduler=state.scheduler,scaler=state.scaler,criteria=state.router,
        progress=TrainingProgress(stage='recovery',next_epoch=0,global_macro_step=1,joint_epochs_completed=0),
        resolved_config=state.metadata,provenance={'synthetic_only':True},loader_state={},best_state={})
    restored=build('rep17')
    load_training_snapshot(path,model=restored.model,ema=restored.ema,optimizer=restored.optimizer,
        scheduler=restored.scheduler,scaler=restored.scaler,criteria=restored.router,restore_rng=True)
    compare(state.model.state_dict(),restored.model.state_dict(),exact=True)
    compare(state.ema.ema.state_dict(),restored.ema.ema.state_dict(),exact=True)
    step(state); step(restored)
    compare(state.model.state_dict(),restored.model.state_dict(),exact=True)
    compare(state.ema.ema.state_dict(),restored.ema.ema.state_dict(),exact=True)
    assert state.ema.updates==restored.ema.updates==26599
    runtime.write_json(root/'summary.json',{'status':'passed','device':'cpu','imgsz':160,
        'initial_full_graph_exact':True,'float_bittrue_two_task_parity':True,
        'full_snapshot_restored_next_update_exact':True,'new_branch_nonzero_update':True,
        'synthetic_only':True,'accuracy_validated':False,'deployable_weights':False})
    print('JOB_DONE: CPU snapshot 恢復後下一步與未中斷路徑逐 tensor 相等',flush=True)

if __name__=='__main__': main()
