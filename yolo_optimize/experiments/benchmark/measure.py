"""受控推論成本量測：安全 checkpoint、固定合成輸入、CPU／CUDA／NVML。"""
import argparse,copy,ctypes,ctypes.util,gc,hashlib,importlib,json,os,platform,resource,statistics,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/activation/bridge_v1'))
from verify_selected import initialize
from evaluate import Source
from safe_source import ALLOWED as BASE_ALLOWED
from common import SOURCE,sha256
from yolo_attention.config import VariantConfig
from yolo_attention.integration import convert_yolo26_model
from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.graph_materialize import build_graph_validation_models
from pwl_contract import verify_pwl

EXTRA={
 'torch.nn.modules.loss':'BCEWithLogitsLoss',
 'ultralytics.nn.modules.conv':'RepConv',
 'ultralytics.nn.modules.block':'Attention C3k',
 'ultralytics.nn.modules.head':'Detect',
 'ultralytics.utils.tal':'TaskAlignedAssigner',
 'ultralytics.utils.loss':'v8DetectionLoss E2ELoss BboxLoss',
 'qk_challenger':'TrainableBinaryScore',
 'yolo_optimize.hog':'HOGAuxiliary',
 'masf_p2':'P2MASFC3k2',
 'masf_p3':'P3MASFDetect',
 'numpy._core.multiarray':'_reconstruct',
 'numpy':'ndarray dtype',
}

def safe_payload(row):
    p=Path(row['checkpoint']);assert sha256(p)==row['sha256']
    if row['loader']=='state_dict':return torch.load(p,map_location='cpu',weights_only=True)
    allowed={}
    for table in (BASE_ALLOWED,EXTRA):
        for module,names in table.items():
            for name in names.split():
                allowed[module+'.'+name]=(module,name)
    required=set(torch.serialization.get_unsafe_globals_in_checkpoint(p))
    assert required <= set(allowed), 'Unapproved pickle globals: '+repr(required-set(allowed))
    classes=[getattr(importlib.import_module(allowed[n][0]),allowed[n][1]) for n in sorted(required)]
    classes += [set,np.dtypes.UInt32DType,np.dtypes.Float64DType,np.dtypes.Int64DType]
    with torch.serialization.safe_globals(classes):return torch.load(p,map_location='cpu',weights_only=True)

def build(row):
    payload=safe_payload(row)
    info={}
    if row['loader']=='pickle':
        model=copy.deepcopy(payload.get(row.get('state','ema')) or payload['model']).float().eval()
        info['prefold_registered_params']=sum(p.numel() for p in model.parameters())
        if hasattr(model,'hog_aux'):
            info['removed_hog_params']=sum(p.numel() for p in model.hog_aux.parameters());del model.hog_aux
        if hasattr(model,'criterion'):del model.criterion
        if type(model.model[17]).__name__=='RepConv':
            from rep17 import folded
            sample=torch.rand(1,3,160,160)
            from verify_qk_challenger import tensors
            with torch.inference_mode():
                before=tensors(model(sample));model=folded(model);after=tensors(model(sample))
                assert len(before)==len(after)
                errors=[float((a-b).abs().max()) for a,b in zip(before,after)]
                for a,b in zip(before,after):torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
                info['rep_fold_cpu160_max_abs']=max(errors)
        from qk_challenger import remove
        remove(model)
        if row['backend']=='bittrue' and row.get('state')=='ema':
            convert_yolo26_model(model,VariantConfig.from_yaml(SOURCE/'configs/attention/bittrue-pwl-final.yaml'))
    else:
        if row.get('template')=='old':
            from compare_old_combine import OldSource
            source=OldSource()
        else:source=Source(activation=row['activation'])
        pair=source.build_task_models('float')
        graph,report=assemble_graph_shared_model(pair.detect,pair.pose);assert report.complete
        graph.load_state_dict(payload['state_dict'],strict=True)
        if row.get('zero_shot'):
            # activation is a non-state property, already installed by Source.
            info['zero_shot_activation']=row['activation']
        pair=build_graph_validation_models(graph,source,kind='bittrue')
        if row['task']=='pose':
            model=pair.pose
            model.end2end=row.get('route')!='one2many'
        else:
            model,report=assemble_graph_shared_model(pair.detect,pair.pose);assert report.complete
        del pair,graph
    model.eval().requires_grad_(False)
    if row['backend']=='bittrue':
        if row['id'] in ('a0','b100'):
            info['pwl_contract']=[{'site':n,'class':type(m).__name__,'range':[m.score_floor,0.],'segments':m.segments}
                for n,m in model.named_modules() if type(m).__name__ in ('PiecewiseLinearSoftmax','BitTruePiecewiseLinearSoftmax')]
            assert len(info['pwl_contract'])==2
        else:info['pwl_contract']=verify_pwl(model)
    del payload;gc.collect()
    return model,info

def forward(model,x,row):
    return model(x,task='both') if row['task']=='both' else model(x)

def finite(tree):
    if torch.is_tensor(tree):return bool(torch.isfinite(tree).all())
    if isinstance(tree,dict):return all(finite(v) for v in tree.values())
    if isinstance(tree,(list,tuple)):return all(finite(v) for v in tree)
    return True

class CostCounter(TorchDispatchMode):
    def __init__(self):super().__init__();self.float_mm_mac=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        name=str(func)
        if name in ('aten.mm.default','aten.bmm.default','aten.addmm.default'):
            a,b=args[-2:] if name=='aten.addmm.default' else args[:2]
            if a.is_floating_point() and b.is_floating_point():
                self.float_mm_mac+=a.numel()*b.shape[-1]
        return func(*args,**(kwargs or {}))

def costs(model,x,row):
    # MAC 稽核使用 no_grad；latency 仍使用 inference_mode。
    from accounting import count
    return count(model,x,row)

def distribution(samples):
    return {'n':len(samples),'median_ms':statistics.median(samples),'mean_ms':statistics.mean(samples),
            'p90_ms':float(np.percentile(samples,90)),'min_ms':min(samples),'max_ms':max(samples),'samples_ms':samples}

class Energy:
    def __init__(self):
        self.lib=ctypes.CDLL(ctypes.util.find_library('nvidia-ml'))
        assert self.lib.nvmlInit_v2()==0
        self.handle=ctypes.c_void_p();assert self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))==0
    def read(self):
        value=ctypes.c_ulonglong();status=self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle,ctypes.byref(value))
        if status:raise RuntimeError('NVML energy status '+str(status))
        return value.value
    def close(self):self.lib.nvmlShutdown()

def measure(row,quick=False):
    initialize();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(50900912)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,info=build(row)
    x=torch.rand(1,3,640,640,generator=torch.Generator().manual_seed(50900912))
    result={'case':row,'build':info,'registered_params':sum(p.numel() for p in model.parameters()),
        'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'buffer_bytes':sum(b.numel()*b.element_size() for b in model.buffers()),
        'checkpoint_bytes':Path(row['checkpoint']).stat().st_size,'torch_version':torch.__version__,
        'input':'synthetic uniform FP32 NCHW 1x3x640x640; seed50900912',
        'cpu_threads':4,'mode':'eager eval inference_mode, no fuse, no autocast, no compile, TF32 disabled',
        'target_hardware_latency_ms':None,'target_energy_j_per_frame':None,
        'target_status':'not measured: target board/equipment not specified or attached'}
    result['cost']=costs(model,x,row)
    with torch.inference_mode():
        for _ in range(2):out=forward(model,x,row)
        assert finite(out);del out
        samples=[]
        for _ in range(3 if quick else 10):
            start=time.perf_counter();out=forward(model,x,row);samples.append((time.perf_counter()-start)*1000);del out
        result['cpu_latency']=distribution(samples)
        result['cpu_process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        model=model.cuda();x=x.cuda()
        # 融合 wrapper 不會替嵌套 head 移動普通屬性的 decode cache。
        from ultralytics.nn.modules.head import Detect
        for module in model.modules():
            if isinstance(module,Detect):
                module.shape=None
                for name in ('stride','strides','anchors'):
                    value=getattr(module,name,None)
                    if torch.is_tensor(value):setattr(module,name,value.to(x.device))
        torch.cuda.synchronize()
        for _ in range(3 if quick else 10):out=forward(model,x,row)
        torch.cuda.synchronize();assert finite(out);del out
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        result['gpu_resident_allocated_bytes']=torch.cuda.memory_allocated()
        samples=[];walls=[]
        for _ in range(5 if quick else 50):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            t=time.perf_counter();start.record();out=forward(model,x,row);end.record();end.synchronize()
            samples.append(start.elapsed_time(end));walls.append((time.perf_counter()-t)*1000);del out
        result['gpu_latency']=distribution(samples);result['gpu_sync_wall_latency']=distribution(walls)
        result['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        result['gpu_name']=torch.cuda.get_device_name();result['cuda_version']=torch.version.cuda
        try:
            meter=Energy();blocks=[]
            for _ in range(1 if quick else 3):
                torch.cuda.synchronize();e0=meter.read();t=time.perf_counter();n=0
                while n<10 or time.perf_counter()-t<(0.5 if quick else 2.0):
                    out=forward(model,x,row);torch.cuda.synchronize();del out;n+=1
                elapsed=time.perf_counter()-t;e1=meter.read();assert e1>=e0
                blocks.append({'frames':n,'seconds':elapsed,'energy_mj':e1-e0,'j_per_frame':(e1-e0)/1000/n})
            meter.close();result['gpu_energy']={'scope':'whole GPU telemetry, includes desktop/idle; not system or target-board energy','blocks':blocks,
                'median_j_per_frame':statistics.median(b['j_per_frame'] for b in blocks)}
        except (RuntimeError,OSError,AssertionError) as exc:result['gpu_energy']={'status':'unavailable','reason':str(exc)}
    assert sha256(row['checkpoint'])==row['sha256']
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--quick',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
    row=next(r for r in json.loads((HERE/'manifest.json').read_text())['cases'] if r['id']==a.case)
    result=measure(row,a.quick);result['status']='passed';result['quick']=a.quick
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('JOB_DONE '+a.case,flush=True)

if __name__=='__main__':main()
