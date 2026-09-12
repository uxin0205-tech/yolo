"""把逐階段 AP 與受控推論成本組成可追溯報告；不改原始量測。"""
import csv,io,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tools'));from restructure_layout import edit
OUT=ROOT/'reports/performance'
AP=['coco/box/map50_95','coco/person/box/map50_95','coco/ball/box/map50_95','coco/bat/box/map50_95',
    'bbat/box/map50_95','bbat/pose/map50_95','bbat/ball/box/map50_95','bbat/ball/pose/map50_95',
    'bbat/bat/box/map50_95','bbat/bat/pose/map50_95']
NAMES={'fp':'原生 FP Detect','a0':'BinaryQK A0','b100':'B100 shared MASF','hog_control_e3':'native control E3',
'hog_e3':'HOG E3（部署移除輔助頭）','rep_control_e4':'Conv control E4','rep_folded_e4':'RepConv17 E4（已折疊）',
'p3_control_e5':'P3 control E5','p3_shared_e5':'P3 shared E5','p3_fork_e5':'P3 fork E5',
'p3_control_e8':'P3 control E8','p3_bridge_e8':'P3 bridge E8','p2_control_e5':'P2 control E5','p2_e5':'P2 MASF E5',
'old_joint':'舊 combine best_joint','j3_joint':'新 bridge J3','pose_recovery':'Pose head 恢復',
'silu_e9':'SiLU E9','qsilu_e2':'qSiLU E2','dual_kd_e4':'雙教師 KD E4','head_kd_e2':'Pose-head KD E2',
'mixed_keypoint':'原框＋KD 關鍵點','hardswish_zero':'Hardswish zero-shot','poly_shift_zero':'PolyShift zero-shot',
'qsilu_pose_one2one':'qSiLU Pose one2one','qsilu_pose_one2many':'qSiLU Pose one2many core'}

def fmt(value,places=3):return '—' if value is None else f'{value:.{places}f}'
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)+'\n'

def section(stage,rows):
    s='## '+stage+'\n\n所有 AP 為既有完整驗證，範圍 0–1；— 表示此模型沒有該任務或此配對未驗證，不表示 0。Detect-only 的 COCO ball／bat 與 BBAT 二類 AP 不可混用。\n\n'
    s+=table(['案例／工作量','COCO','person','COCO ball','COCO bat','BBAT box','BBAT Pose'],
        [[r['label']+' / '+r['task'],*[fmt(r.get(k),6) for k in AP[:6]]] for r in rows])
    s+=table(['案例','BBAT ball box','ball Pose','bat box','bat Pose'],[[r['label'],*[fmt(r.get(k),6) for k in AP[6:]]] for r in rows])
    s+='### Params、Model size、MAC／FLOPs\n\nTensor MB 是本次 FP32 重建後參數＋buffers 的 payload，非已打包的低位元模型。來源檔 MB 是實際 checkpoint 大小；標記「snapshot」者含訓練狀態，不宜與推論檔直接比較。MAC／FLOPs 只包含下述可精確計數的 subtotal，二值位元乘積另列。\n\n'
    s+=table(['案例','Params M','Tensor MB','來源檔 MB／類型','MAC G subtotal','FP FLOPs G subtotal','Binary bit-products M'],
       [[r['label'],fmt(r['params']/1e6),fmt(r['tensor_bytes']/1e6),fmt(r['checkpoint_bytes']/1e6)+' / '+r['checkpoint_kind'],fmt(r['mac']/1e9),fmt(r['float_flops']/1e9),fmt(r['binary_bit_products']/1e6)] for r in rows])
    s+='### Peak memory、CPU／GPU latency、target latency、energy/frame\n\nB1／640／FP32、CPU 4 threads。延遲為 median；GPU 是 CUDA events 的模型核心時間。含模型內 decode／top-k，但不含外部 NMS、影像讀取與 H2D。GPU peak 為暖機後的 PyTorch allocated tensor 高水位，並非整張卡總使用量。\n\n'
    s+=table(['案例','GPU peak MiB','CPU ms','GPU ms','target ms','GPU J/frame','target J/frame'],
       [[r['label'],fmt(r['gpu_peak_bytes']/2**20,1),fmt(r['cpu_ms']),fmt(r['gpu_ms']),'未量測',fmt(r['gpu_energy_j'],4),'未量測'] for r in rows])
    s+='GPU energy/frame 是 NVML 整卡能量差，含桌面／idle 活動；不能視為純模型、整機或目標板能耗。target 欄位因未指定／連接目標設備而保留缺值。\n\n'
    return s

def main():
    config=json.loads((HERE/'manifest.json').read_text());cases=config['cases']
    assert json.loads((HERE/'artifacts/comparison-v1/summary.json').read_text())['status']=='completed'
    rows=[]
    for case in cases:
        primary=HERE/'artifacts/comparison-v1'/(case['id']+'.json')
        corrected=HERE/'artifacts/isolated-recheck-v1'/(case['id']+'.json')
        path=corrected if corrected.exists() else primary
        d=json.loads(path.read_text());assert d['status']=='passed' and not d['quick']
        costpath=HERE/'artifacts/accounting-v2'/(case['id']+'.json');a=json.loads(costpath.read_text())
        assert d['case']['sha256']==a['checkpoint_sha256']==case['sha256']
        c=a['cost'];e=d.get('gpu_energy',{})
        rows.append(dict(id=case['id'],stage=case['stage'],label=NAMES[case['id']],task=case['task'],
            **{k:case['metrics'].get(k) for k in AP},params=d['registered_params'],
            tensor_bytes=d['parameter_bytes']+d['buffer_bytes'],checkpoint_bytes=d['checkpoint_bytes'],
            checkpoint_kind='snapshot' if case.get('checkpoint_is_training_snapshot') else 'inference/source',
            mac=c['dense_mac_subtotal'],float_flops=c['float_flops_subtotal'],integer_ops=c['integer_ops_subtotal'],
            binary_bit_products=c['binary_qk_bit_products'],cpu_ms=d['cpu_latency']['median_ms'],
            cpu_p90_ms=d['cpu_latency']['p90_ms'],gpu_ms=d['gpu_latency']['median_ms'],gpu_p90_ms=d['gpu_latency']['p90_ms'],
            gpu_wall_ms=d['gpu_sync_wall_latency']['median_ms'],gpu_peak_bytes=d['gpu_peak_allocated_bytes'],
            gpu_reserved_bytes=d['gpu_peak_reserved_bytes'],cpu_peak_rss_bytes=d['cpu_process_peak_rss_bytes'],
            gpu_energy_j=e.get('median_j_per_frame'),target_latency_ms=None,target_energy_j=None,
            target_status='未指定或未連接設備，未量測',measurement=str(path.relative_to(ROOT)),
            accounting=str(costpath.relative_to(ROOT)),checkpoint_sha256=case['sha256'],metrics_source=case['metrics_source']))
    OUT.mkdir(exist_ok=True,parents=True)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    edit(OUT/'comparison.csv',stream.getvalue());edit(OUT/'comparison.json',json.dumps(rows,ensure_ascii=False,indent=2))
    text='''# 全階段 AP50–95 與部署成本比較

更新：2026-09-12。26 組代表／配對，補入 AP50–95、Params、Model size、MAC／FLOPs、Peak memory、CPU latency、GPU latency、target hardware latency、energy/frame。這不是對全部 364 個本機模型逐一 benchmark，也沒有新增訓練或重測 AP。

## 量測口徑與證據強度

CPU 是 Intel Core Ultra 9 285K，GPU 是 RTX 5090；同一 PyTorch 環境、B1、640×640、FP32、固定合成輸入、CPU 4 threads，無 autocast／compile／額外 fuse，TF32 關閉。模型保持原始硬體友善 BinaryQK／PWL；qSiLU 是分段多項式軟體實作，不因此假稱 GPU 融合 kernel 或板端加速。

AP 重用各 checkpoint 的原完整 COCO val5000／BBAT5 v1 val683 結果。合成輸入只作算量／latency 測試，不更改資料 assignment，也不能取代真實影片 latency 或目視驗收。不同 task 的時間不能直接當演算法加速比：共享雙 head 每 frame 執行 Detect＋Pose，Detect-only／Pose-only 只做一個任務。

九項都列欄位，但 target latency／target energy 未量測：沒有指定／連接目標板、bitstream／runtime 與量測設備，不能拿 RTX 5090 遙測填 FPGA 欄位。也沒有整機插座功耗、FPGA rail power 或 CPU energy 的證據。

### 成本公式

Conv MAC = B×Hout×Wout×Cout×(Cin/groups)×Kh×Kw；矩陣乘法 MAC = batch×M×K×N。FLOPs subtotal = 2×浮點 MAC subtotal；整數 MAC／operations 與二值位元乘積另外記錄，不把 XNOR/popcount 算成普通浮點 FMA。

表內 MAC subtotal 為 Conv＋浮點／整數矩陣乘法，沒有計入 BN、activation、PWL／reciprocal、pooling、decode／top-k、整數 reductions、NMS、記憶體流量。它是明確範圍的運算估算，不是完整全算子 FLOPs，也不是硬體 cycle 數。相同 MAC 不代表相同 latency；qSiLU 的額外算子主要就在此 subtotal 之外。

Model size 同時提供實際來源檔 bytes 與 FP32 tensor payload；snapshot 包含 optimizer／EMA 等，不能用檔案大數倍推論部署模型也大數倍。BinaryQK 只二值化 attention 的 Q/K 運算，不是整個 YOLO 權重已壓成 1 bit。Params 計入仍註冊的雙分支權重，即使推論只執行其中一條。

### 時間、記憶體與能量

CPU 暖機 2 次＋10 次量測，GPU 暖機 10 次＋50 次量測；median／P90／全部樣本在 JSON，另存 GPU synchronized wall time。端到端 pipeline 的影像解碼、resize、H2D、外部 NMS 不在 core latency。one2many 比較尤其不能省略這項限制。

GPU peak allocated／reserved 在暖機後重置；主表使用 allocated，包括常駐模型與輸入，不等於 nvidia-smi 所見全卡顯存。CPU RSS 是 worker 全生命週期高水位，含載入與模型副本，只供記憶體稽核，不與 GPU activation peak 相加。

Energy/frame = (NVML end_mJ−start_mJ)/1000/frames。每案例 3 個至少 2 秒、至少 10 frames 區段取 median，不使用 TDP×latency。整卡 telemetry 包含 idle／桌面工作，沒有扣 idle、沒有整機功耗，也沒有獨占 GPU 或鎖定時脈；小幅差異應視為本次觀察，不作節能定論。CPU／GPU 都只有單機一次順序測試，非多輪隨機排序的統計保證。

### 量測修正

融合模型的 head decode cache 在 CPU→GPU 切換時未跟隨普通 module buffer，已於量測工具移動 cache 並清除 shape，短測通過，沒有改權重。

初版 dispatcher 在 inference_mode 下漏算矩陣乘法，原結果保留但不作 MAC 正文依據；報告統一使用 accounting-v2 的 no_grad CPU 計數，已通過已知尺寸浮點／整數矩陣與分組卷積的 3 項測試。latency／energy 仍在 inference_mode 下量測。

CPU 算量稽核曾與部分時延量測重疊。依工作時間保守圈定 j3_joint、pose_recovery、silu_e9、qsilu_e2，這 4 組另以無並行稽核的 isolated-recheck-v1 補測，其他成功量測不重跑。所有原紀錄保留；CSV 明列採用的來源。

## 原始證據

[精確比較 CSV](comparison.csv)／[比較 JSON](comparison.json)／[來源與原始量測](../../experiments/benchmark/README.md)。完整逐次 samples、每個能量區段與 checkpoint SHA 均可追溯。

'''
    stages=list(dict.fromkeys(r['stage'] for r in rows))
    for i,stage in enumerate(stages,1):
        selected=[r for r in rows if r['stage']==stage];body=section(stage,selected)
        edit(OUT/f'stage-{i}.md','# '+stage+'：精度與部署成本\n\n[完整量測口徑](README.md)／[精確 CSV](comparison.csv)。\n\n'+body)
        text+=body
    text+='''## 如何解讀，不應得出的結論

BinaryQK 在這個軟體 bool-tiled reference 上不保證比 FP GPU 快；硬體友善性必須由目標實作驗證。A0／B100／FP 是歷史模型鏈，AP 差不是只改單一開關的完整重訓因果消融。

HOG 是 training-only，部署移除後成本應與同骨架接近；這不能解讀成 HOG 精度有效。RepConv 折疊消除多分支訓練成本，不代表參數或 latency 必然優於原 Conv。P2 MASF context 的空間面積增加使該模組 MAC 約為 P3 四倍，不是整網四倍。

KD 的 teacher 不進部署模型，所以同骨架 KD／非 KD 的 Params 通常相同；小幅 latency／energy 差不能歸因於 KD 算子增減。Hardswish／PolyShift 是既有 zero-shot，不與微調 10 輪的 qSiLU 作同訓練預算精度比較。

one2many 欄位是 Pose core，不含額外 class-aware NMS；本輪沒有宣稱完整 pipeline 變快。bat Pose 增益仍伴隨 ball Pose 下降，不能只因成本相近便採用。固定 8-scale selector、區域 KD、PTQ／QAT、target reciprocal 等未實施方案沒有可量測正式模型，不能編造其 Params 或 latency。

### 正式 target 補量流程

先指定板卡／時脈／runtime、權重量化格式與實際導出圖，再做同資料輸入的輸出等價性；分別量 model core 與完整 pipeline 的 median／P90。功耗需說明量測 rail 或插座、採樣率、idle 是否扣除，以及 E/frame 的積分區間。沒有這些資訊時，target 欄位維持未量測。

官方依據：[PyTorch peak memory](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html)、[NVML 累計 GPU 能量](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)。本報告的實測依據是本機輸出 JSON，而非網頁上的其他模型 benchmark。
'''
    edit(OUT/'README.md',text)
    index={stage:i for i,stage in enumerate(stages,1)}
    placements={'experiments/studies/pre-fusion-full35-b100/README.md':['BinaryQK','HOG','RepConv','MASF'],
      'experiments/combine/pose-masf/RESULTS.md':['MASF'],'experiments/combine/bridge_v1/BBAT_RECOVERY_RESULTS.md':['融合'],
      'experiments/activation/bridge_v1/RESULTS.md':['Activation'],'experiments/kd/README.md':['KD'],
      'experiments/inference/README.md':['推論']}
    for rel,parts in placements.items():
        p=ROOT/rel;text=p.read_text();marker='\n## AP 與部署成本補充（2026-09-12）\n'
        text=text.split(marker)[0]+marker+'\n'
        for stage in parts:
            link=os.path.relpath(OUT/f'stage-{index[stage]}.md',p.parent)
            text+='- ['+stage+'：九項指標與比較]('+link+')\n'
        text+='\n包含 AP50–95、Params、Model size、MAC／FLOPs、Peak memory、CPU／GPU latency、target latency 及 energy/frame。target 未量測明記缺值；不將 core-only 時間當完整 pipeline。\n'
        edit(p,text)
    p=ROOT/'reports/final/README.md';s=p.read_text().split('\n## 19. 全階段精度與部署成本')[0]
    s+='\n## 19. 全階段精度與部署成本\n\n本次另完成 26 組已訓模型的 CPU／GPU 受控比較，未重新訓練；AP 沿用完整驗證，權重 SHA 不變。\n\n'
    s+='[詳細九項比較、算式與所有限制](../performance/README.md)／[精確 CSV](../performance/comparison.csv)。\n\n'
    for stage in stages:s+='- ['+stage+'：精度、大小、算量、峰值記憶體、延遲與能耗](../performance/stage-'+str(index[stage])+'.md)\n'
    s+='\n目標硬體未指定／連接，因此 target latency 與 target energy/frame 仍未量測。GPU 能量為 NVML 整卡遙測，不是整機或 FPGA 能耗。MAC／FLOPs 為明確算子範圍的 subtotal；請勿刪去這些限制再引用數字。\n'
    edit(p,s)
    print('REPORT_DONE',len(rows),len(stages))

if __name__=='__main__':main()
