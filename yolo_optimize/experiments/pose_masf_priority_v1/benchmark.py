"""同口徑雙 head 模型延遲與有限運算集合的 MAC 估算；不是上板量測。"""
import gc
import math
import platform
import statistics
import time
from pose_candidate import *
from yolo_combine.graph_materialize import build_graph_validation_models

def describe(values):
    values=sorted(values)
    return {'samples':len(values),'median_ms':statistics.median(values),
        'p95_ms':values[math.ceil(.95*len(values))-1],'mean_ms':statistics.mean(values)}

def operation_count(model,x):
    counts={'conv':0,'linear':0,'qk_av':0};handles=[]
    def hook(module,inputs,output):
        if isinstance(module,torch.nn.Conv2d):
            counts['conv']+=output.numel()*(module.in_channels//module.groups)*math.prod(module.kernel_size)
        elif isinstance(module,torch.nn.Linear):counts['linear']+=output.numel()*module.in_features
        else:
            b,c,h,w=inputs[0].shape;n=h*w
            counts['qk_av']+=b*module.num_heads*n*n*(module.key_dim+module.head_dim)
    for m in model.modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)) or type(m).__name__=='NativePWLAttention':handles.append(m.register_forward_hook(hook))
    with torch.inference_mode():model(x,task='both')
    for handle in handles:handle.remove()
    total=sum(counts.values())
    return {'mac':total,'flops_at_2_per_mac':2*total,'components_mac':counts,
        'excluded':'BN、qSiLU/PWL、逐元素加乘、decode/top-k、資料搬移；不是完整硬體指令計數'}

def main():
    initialize();torch.set_num_threads(4);torch.manual_seed(50900913)
    assert json.loads((HERE/'artifacts/summary-v1.json').read_text())['status']=='completed'
    root=HERE/'artifacts/benchmark-v1.json';assert not root.exists()
    rows={}
    cpu_name=platform.processor()
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):cpu_name=line.split(':',1)[1].strip();break
    for enabled in (False,True):
        original,source=build(enabled)
        mats=build_graph_validation_models(original,source,kind='bittrue')
        model,report=assemble_graph_shared_model(mats.detect,mats.pose);assert report.complete
        model.eval().requires_grad_(False)
        del original,mats,source;gc.collect()
        x=torch.rand(1,3,640,640)
        ops=operation_count(model,x)
        cpu=[]
        with torch.inference_mode():
            for _ in range(5):model(x,task='both')
            for _ in range(20):
                start=time.perf_counter();model(x,task='both');cpu.append((time.perf_counter()-start)*1000)
        torch.cuda.empty_cache();model=model.cuda();x=x.cuda()
        # DualHeadPrediction 不是 Detect，原 DetectionModel._apply 不會搬移兩個 head 的非 buffer 快取。
        for head in (model.detect_head,model.pose_head):
            for key in ('stride','anchors','strides'):
                setattr(head,key,getattr(head,key).to(x.device))
            head.shape=None
        gpu=[]
        with torch.inference_mode():
            for _ in range(20):model(x,task='both')
            assert all(getattr(head,key).device==x.device for head in (model.detect_head,model.pose_head) for key in ('stride','anchors','strides'))
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            for _ in range(100):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();model(x,task='both');end.record();end.synchronize();gpu.append(start.elapsed_time(end))
        name='pose_masf' if enabled else 'baseline'
        rows[name]={'cpu':describe(cpu),'gpu':describe(gpu),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'operation_estimate':ops}
        del model,x;gc.collect();torch.cuda.empty_cache()
    result={'status':'completed','hardware':{'cpu':cpu_name,'gpu':torch.cuda.get_device_name(0)},
        'torch_version':str(torch.__version__),'cuda_version':torch.version.cuda,'cpu_threads':4,
        'batch':1,'imgsz':640,'dtype':'float32','normalizer':'BitTrue PWL [-10,0],20 segments',
        'task':'both; shared trunk once; both heads; existing native postprocess',
        'input':'seeded synthetic tensor; same shape/distribution; no new dataset split',
        'timing_excludes':'模型載入、影像讀取、resize/letterbox、H2D/D2H、可視化',
        'order':['baseline','pose_masf'],'repeated_independent_sessions':False,
        'rows':rows,'target_hardware_latency':None,'energy_per_frame':None,
        'limitations':'單機單次順序量測，不宣稱小差異顯著；MAC 只涵蓋列出的運算。'}
    save(root,result);print('JOB_DONE benchmark',flush=True)

if __name__=='__main__':main()
