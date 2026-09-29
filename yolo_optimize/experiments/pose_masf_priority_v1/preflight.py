"""只用 CPU 驗證 Pose MASF 隔離、零閘等價、重建與梯度；不是精度結果。"""
from pose_candidate import *
from yolo_combine.graph_materialize import build_graph_validation_models

def equal(a,b):
    aa,bb=tensors(a),tensors(b);assert len(aa)==len(bb)
    for x,y in zip(aa,bb):torch.testing.assert_close(x,y,rtol=0,atol=0)

def main():
    initialize();torch.set_num_threads(4);torch.manual_seed(50900913)
    base,_=build();candidate,source=build(True)
    x=torch.rand(1,3,160,160)
    with torch.no_grad():
        equal(base(x,task='detect'),candidate(x,task='detect'))
        expected=base(x,task='pose');active=candidate(x,task='pose')
        assert any(not torch.equal(a,b) for a,b in zip(tensors(expected),tensors(active)))
        alpha=candidate.pose_head.p3_masf.alpha.detach().clone()
        candidate.pose_head.p3_masf.alpha.zero_()
        equal(expected,candidate(x,task='pose'))
        candidate.pose_head.p3_masf.alpha.copy_(alpha)
    for kind in ('float','bittrue'):
        mats=build_graph_validation_models(candidate,source,kind=kind)
        for model in (mats.detect,mats.pose):verify_pwl(model)
        assert hasattr(mats.pose.model[-1],'p3_masf')
        assert type(mats.pose.model[-1]) is PoseP3BridgeMASF
        assert all(torch.equal(v,mats.pose.model[-1].p3_masf.state_dict()[n]) for n,v in candidate.pose_head.p3_masf.state_dict().items())
    head=copy.deepcopy(candidate.pose_head).train().requires_grad_(True)
    for m in head.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
    channels=[block[0].conv.in_channels for block in head.cv2]
    features=[torch.randn(2,c,size,size,requires_grad=True) for c,size in zip(channels,(8,4,2))]
    one=head(features)['one2one']
    assert all(k in one for k in ('boxes','scores','kpts'))
    objective=sum(t.square().mean() for k in ('boxes','scores','kpts') for t in tensors(one[k]) if t.requires_grad)
    objective.backward()
    grad=head.p3_masf.alpha.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)>0
    assert all(v.grad is None for v in features),'one2one bridge 不應回傳共享 trunk'
    added=sum(p.numel() for p in candidate.pose_head.p3_masf.parameters())
    save(HERE/'artifacts/preflight-v1.json',{'status':'passed','gpu_used':False,'accuracy_evaluation':False,
        'parent_inference_sha256':sha256(CHECKPOINT),'detect_output_exact_unchanged':True,
        'pose_alpha_zero_exact_baseline':True,'pose_enabled_changes_output':True,
        'float_bittrue_materialization':True,'pwl':[-10,0,20],
        'one2one_masf_alpha_gradient':float(grad),'one2one_trunk_detached':True,
        'parameters_baseline':sum(p.numel() for p in base.parameters()),'additional_pose_masf_parameters':added,
        'pose_alpha':float(alpha),'bridge_coefficient':float(head.bridge_coefficient),
        'bridge_coefficient_needs_real_loss_calibration_before_training':True,
        'pose_masf_pose_training_performed':False})
    print('PASS Pose MASF CPU 隔離、零閘等價、Float/BitTrue 重建與 one2one 梯度通過；未啟動 GPU。',flush=True)

if __name__=='__main__':main()
