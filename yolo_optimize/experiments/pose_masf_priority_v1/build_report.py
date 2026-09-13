"""從已驗證 JSON 重建唯一完整報告／CSV；文件更新一律經 apply_patch。"""
import csv
import io
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from tools.restructure_layout import edit

def main():
    raw=json.loads((HERE/'artifacts/summary-v1.json').read_text())
    bench=json.loads((HERE/'artifacts/benchmark-v1.json').read_text())
    diagnostic=json.loads((HERE/'artifacts/residual-diagnostic-v1.json').read_text())
    audit=json.loads((HERE/'artifacts/final-audit-v1.json').read_text())
    assert audit['status']=='passed' and audit['baseline_matches_pinned_parent']
    a=raw['cases']['baseline'];b=raw['cases']['pose_masf'];x=bench['rows']['baseline'];y=bench['rows']['pose_masf']
    metrics=[('COCO overall 框','coco/box/map50_95'),('COCO person 框','coco/person/box/map50_95'),
        ('BBAT overall 框','bbat/box/map50_95'),('BBAT overall 關鍵點','bbat/pose/map50_95'),
        ('ball 框','bbat/ball/box/map50_95'),('ball 關鍵點','bbat/ball/pose/map50_95'),
        ('bat 框','bbat/bat/box/map50_95'),('bat 關鍵點','bbat/bat/pose/map50_95')]
    out=io.StringIO();writer=csv.writer(out);writer.writerow(['指標','原Pose_AP比例','Pose_MASF_AP比例','差值比例','差值百分點'])
    for label,key in metrics:
        old=a['metrics'][key];new=b['metrics'][key];writer.writerow([label,f'{old:.12f}',f'{new:.12f}',f'{new-old:.12f}',f'{100*(new-old):.9f}'])
    edit(HERE/'metrics.csv',out.getvalue())
    costs=[('Params',a['parameters'],b['parameters'],'個','完整共享雙 head、未 fuse 的 state'),
        ('Model size',a['checkpoint_bytes'],b['checkpoint_bytes'],'bytes','實際 inference-only 檔，含 metadata／序列化差異'),
        ('MAC',x['operation_estimate']['mac'],y['operation_estimate']['mac'],'MAC','估算：Conv／Linear／QK／AV，其他運算排除'),
        ('FLOPs',x['operation_estimate']['flops_at_2_per_mac'],y['operation_estimate']['flops_at_2_per_mac'],'FLOP','估算：2 FLOPs/MAC'),
        ('Peak memory',x['gpu_peak_allocated_bytes'],y['gpu_peak_allocated_bytes'],'bytes','GPU peak allocated；不是訓練顯存'),
        ('CPU latency',x['cpu']['median_ms'],y['cpu']['median_ms'],'ms','batch 1, 640, FP32, 4 threads；中位數'),
        ('GPU latency',x['gpu']['median_ms'],y['gpu']['median_ms'],'ms','RTX 5090；batch 1, 640, FP32；中位數'),
        ('target hardware latency',None,None,'ms','專用目標板未量測'),('energy/frame',None,None,'J/frame','未量測，不以 TDP 代替')]
    out=io.StringIO();writer=csv.writer(out);writer.writerow(['指標','原Pose','Pose_MASF','單位','口徑'])
    writer.writerows(costs);edit(HERE/'costs.csv',out.getvalue())
    lines=['# 原 Pose 與 Pose P3 MASF：完整比較與分析','',
        '日期：2026-09-13。狀態：推論比較、全量殘差診斷、本機成本量測、權重重建與來源稽核均完成；其他 Attention 訓練保持暫停。','',
        '## 1. 結論：直接移接沒有整體提升，先不取代原 Pose','',
        '在同一份 native QK E2 權重上，將 Detect 端已訓練的 MASF 複製到 Pose P3，ball 關鍵點 AP 小幅上升、bat 關鍵點 AP 小幅下降，整體 Pose AP 幾乎持平而略降。COCO 完全不變。新增了參數與推論成本，現階段沒有足夠收益支持直接採用此候選。','',
        '**這次完成的是移接／開關的推論比較，不是 Pose MASF 專項訓練。** 結果不能用來斷言 MASF 在 Pose 訓練後也無效；沒有自動替換正式模型。','',
        '## 2. 「原 Pose」是哪一個？','',
        '本報告的「原 Pose」是剛完成兩個回合並暫停的 native QK E2 融合模型內、尚未額外加入 Pose MASF 的 head。它不是舊 combine 的獨立 Pose 模型，也不是另一個重新訓練的 BinaryQK 對照。','',
        '兩者皆為 YOLO26M 共享 backbone／neck＋COCO80 Detect head＋BBAT5 Pose head；Attention 是原生 QK＋PWL [-10,0]／20 段，保留 qSiLU 與既有 Detect P3 bridge MASF。唯一差異是 Pose P3 的新 MASF 分支。','',
        f"固定 parent：`{a['checkpoint']}`。SHA-256：`{a['sha256']}`。原完整 E2 續訓檔與 optimizer／scheduler／scaler／RNG 均保留，可從 E3 接續。",'',
        '## 3. 接線：改的是 Pose head 入口，不是整個 Neck','',
        '原接法：','',
        '```text',
        'layer16：p3_raw ──┬─> layer17 → P4 → layer20 → P5',
        '                 ├─> Detect MASF → p3_det ─> Detect([p3_det, p4_raw, p5_raw])',
        '                 └────────────────────────> Pose([p3_raw, p4_raw, p5_raw])',
        '```','', '新候選：','', '```text',
        'layer16：p3_raw ──┬─> layer17 → P4 → layer20 → P5',
        '                 ├─> Detect MASF → p3_det ──> Detect([p3_det, p4_raw, p5_raw])',
        '                 └─> Pose MASF   → p3_pose ─> Pose([p3_pose, p4_raw, p5_raw])',
        '```','',
        '新 Pose MASF 是獨立參數副本，不是可變參數共用，也沒有共用一次 MASF 輸出的計算快取。Pose 的框與關鍵點都能受影響；沒有增加 Detect head，P3→P4→P5 的共享路徑維持原狀。實作見 [pose_candidate.py](pose_candidate.py)。','',
        '## 4. 精度結果','',
        '以下 AP50–95 以百分比表示，差值為「百分點」，不是相對百分比。機器精度原值在 [metrics.csv](metrics.csv)。','',
        '| 指標 | 原 Pose AP (%) | Pose＋MASF AP (%) | 差值（百分點） |',
        '| --- | ---: | ---: | ---: |']
    for label,key in metrics:
        old=a['metrics'][key];new=b['metrics'][key]
        lines.append(f'| {label} | {100*old:.4f} | {100*new:.4f} | {100*(new-old):+.4f} |')
    lines+=['',
        '所有 COCO 指標保持一致；Pose α=0 的全部輸出指標也重現原基準（容差 1e-8）。原 Pose 與候選各用 COCO val 5,000 張、BBAT5 v1 val 683 張完整驗證，未重切、未抽樣、未改 labels。','',
        'BBAT5 Pose 驗證共 932 個標註實例（ball 393、bat 539）；資料來源固定為 canonical BBAT5 v1 的 Pose Task View。COCO80 head 沒有被拿去直接解讀 BBAT 的類別 0／1。','',
        '這些是同一驗證集上的確定性結果，不是多 seed 統計顯著性或獨立 test 表現。ball／bat 關鍵點差異約 ±0.05 個百分點，不能包裝成普遍精度提升。','',
        '## 5. 為什麼幾乎沒有整體改善？','',
        '### 5.1 不是 MASF 沒接上','',
        'CPU 驗證確認：Pose α 開啟會改變輸出，α=0 精確還原；新分支經 Float／BitTrue 重建後仍存在。再對全部 683 張 BBAT 驗證影像量測實際 P3 特徵，排除 warmup：','',
        f"- α = {diagnostic['alpha']:.9f}。",
        f"- 全資料聚合的 `||MASF(P3)−P3||₂ / ||P3||₂` = {diagnostic['relative_residual_l2_global']*100:.4f}%。",
        f"- 逐圖中位數 = {diagnostic['relative_residual_l2_median']*100:.4f}%，P95 = {diagnostic['relative_residual_l2_p95']*100:.4f}%，最大值 = {diagnostic['relative_residual_l2_max']*100:.4f}%。",'',
        '因此分支確實產生了約 2% 量級的特徵擾動，不是關閉或接線失效。這個量測比只看 α 更有依據，但特徵 L2 變動大小本身不能證明 AP 變好或變差的因果。殘差 hook 未改變既有候選的全部 BBAT 指標。','',
        '### 5.2 Detect 學到的補充特徵不等於 Pose 最需要的補充特徵','',
        '本次沒有更新新 Pose MASF 或 Pose head；它們沒有共同適應新增的上下文特徵。已知事實是 MASF 參數來自 Detect 分支、現在直接餵進原 Pose head。推論是：這種任務／輸入分布不匹配，可能使不同類別、框與關鍵點的效果方向不同；未量測任務梯度對齊，不能宣稱已證明唯一原因。','',
        '令 `F(x)=Project(DW3(x)+DW5(x))`，則 `y=x+αF(x)`。對可微分的 Pose 訓練損失作局部展開：','',
        '```text',
        'ΔL_pose ≈ α · (∇x L_pose)ᵀF(x) + ½α²F(x)ᵀH_poseF(x)',
        '```','',
        '是否改善，取決於補充特徵與 Pose 損失梯度的方向；存在一個 MASF 模組不會自動保證內積為負。這是對訓練 loss 的推導，不是把不光滑的 AP 當作可微損失。沒有據此在 val 上搜尋 α。','',
        '### 5.3 整體 AP 會掩蓋相反方向','',
        'ball 關鍵點略升、bat 關鍵點略降，兩者在 overall 關鍵點 AP 中大致抵消。框 AP 也沒有回升。故不能只挑 ball 關鍵點一項，宣稱整個 Pose 或 ball/bat 偵測都改善。','',
        '### 5.4 未來訓練還要避免 one2one 梯度被 detach','',
        '直接在 Pose.forward 外層加 MASF，再沿用原生 one2one detach，會使部署分支不直接監督 MASF。新候選已備妥 P3 bridge：先 detach 共享 P3，再送入 MASF，以受控比例傳回 MASF 梯度；共享輸入不接收 one2one 梯度。合成 boxes／scores／kpts 目標的 α 梯度有限且非零。','',
        '目前 bridge 比例沿用 Detect 的 0.012076444778011642，只證明接線可訓練；推論時不使用它。正式 Pose 訓練前需要以 BBAT 真實 loss 校準，不應直接宣稱此比例已適合 Pose。','',
        '## 6. 成本：實測與估算分開看','',
        '| 項目 | 原 Pose 接法 | Pose＋MASF | 口徑 |',
        '| --- | ---: | ---: | --- |',
        f"| Params | {a['parameters']:,} | {b['parameters']:,} | 完整共享雙 head，增加 75,777（0.286%） |",
        f"| Model size | {a['checkpoint_bytes']/2**20:.3f} MiB | {b['checkpoint_bytes']/2**20:.3f} MiB | 實際 inference-only 檔，包含序列化差異 |",
        f"| MAC（估算） | {x['operation_estimate']['mac']/1e9:.6f} G | {y['operation_estimate']['mac']/1e9:.6f} G | Conv／Linear／QK／AV |",
        f"| FLOPs（估算） | {x['operation_estimate']['flops_at_2_per_mac']/1e9:.6f} G | {y['operation_estimate']['flops_at_2_per_mac']/1e9:.6f} G | 2 FLOPs/MAC |",
        f"| Peak memory | {x['gpu_peak_allocated_bytes']/2**20:.3f} MiB | {y['gpu_peak_allocated_bytes']/2**20:.3f} MiB | GPU allocated，非訓練記憶體 |",
        f"| CPU latency | {x['cpu']['median_ms']:.3f} ms | {y['cpu']['median_ms']:.3f} ms | 模型推論中位數，4 threads |",
        f"| GPU latency | {x['gpu']['median_ms']:.3f} ms | {y['gpu']['median_ms']:.3f} ms | RTX 5090 模型推論中位數 |",
        '| target hardware latency | 未量測 | 未量測 | 尚無專用目標板實測 |',
        '| energy/frame | 未量測 | 未量測 | 不用 TDP 或訓練功率代替 |','',
        f"量測：{bench['hardware']['cpu']}、{bench['hardware']['gpu']}，PyTorch {bench['torch_version']}，batch 1、640×640、FP32、BitTrue PWL，`task=both` 一次共享 trunk＋兩個 head。CPU warmup 5、正式 20 次；GPU warmup 20、正式 100 次，CUDA events 計時。",'',
        f"CPU P95：{x['cpu']['p95_ms']:.3f} → {y['cpu']['p95_ms']:.3f} ms；GPU P95：{x['gpu']['p95_ms']:.3f} → {y['gpu']['p95_ms']:.3f} ms。本次中位數稍慢，但 P95 並非同方向；只有單次依序量測，不能把約 2% 的差距當成統計顯著。",'',
        '使用固定種子的合成輸入、相同形狀與分布，但兩組不是逐像素相同的 tensor。沒有 TensorRT、torch.compile 或專門部署裁剪；不含讀檔、resize、letterbox、H2D/D2H、繪圖，也不是影片端到端 FPS。原生 head 在此路徑仍保留實際執行的分支，不冒稱最小化部署 graph。','',
        'MAC 估算排除 BN、qSiLU/PWL、逐元素加乘、decode/top-k 及資料搬移。新增 MASF 的 Conv MAC 可獨立推導：','',
        '```text',
        'ΔMAC = H×W×(9C + 25C + C²)',
        '     = 80×80×(9×256 + 25×256 + 256²)',
        '     = 475,136,000 MAC ≈ 0.475136 GMAC',
        'ΔFLOPs ≈ 0.950272 GFLOPs（僅上述乘加）',
        'ΔParams = C² + 34C + 6C + 1 = 75,777',
        '```','',
        '若看到驗證 terminal 印出約 21.275M 參數／72.5 GFLOPs，那是抽出的 Pose-only、已 fuse 的 validator 模型與其 profiler 口徑，不是本表完整共享 Detect＋Pose 模型；不要混在同一欄直接比較。','',
        '### 量測中修正的問題','',
        'CPU→GPU benchmark 初次失敗：外層最後一層是 DualHeadPrediction，而原 DetectionModel._apply 只處理 Detect 的非 buffer 快取，導致 anchors／strides 留在 CPU。只在 benchmark 中顯式移動兩個 head 的 stride、anchors、strides，清除 shape 快取並驗證 GPU 裝置後重跑成功。精度驗證使用各自 materialized task model，未受此問題影響，也未因修復重跑。此修正沒有擴散到外部套件；其他直接搬移完整雙 head 模型的工具仍須注意相同情境。','',
        '## 7. 已完成哪些驗證？','',
        '1. CPU：Detect 隔離、Pose α=0 等價、α 開啟生效、Float／BitTrue 重建、one2one MASF 梯度。',
        '2. GPU：三種接法各自完整 COCO／BBAT 驗證，基準重現與 alpha-zero 全指標還原。',
        '3. 診斷：全部 683 張實際 P3 特徵殘差，排除 warmup；附帶核對 hook 未改變 BBAT 指標。',
        '4. 本機成本：相同輸入尺寸／雙 head 路徑的 CPU／GPU 延遲與峰值 allocated memory；有限運算集合的 MAC 估算。',
        '5. 交付：候選 export 的 state 與重建模型逐項精確一致，重新載入後 CPU 輸出一致；三組權重 SHA、原 E2 full resume、PWL 與新增 MAC 公式均通過核對。','',
        '## 8. 下一步建議，以及本次刻意沒做的事','',
        '目前保留原 Pose 接法，將 Pose MASF 保留為研究候選。若要驗證真正的訓練收益，建議最小的下一個實驗：固定共享 backbone／neck、Detect 與其 BN，從相同 E2 起點比較 Pose head 無新增 MASF／有新增 MASF，各 5 epoch、warmup 1，同資料、同 seed、同更新預算；先以 train batch 真實梯度校準 bridge，再鎖定 LR 與係數。','',
        'α=0 只證明可以回退。訓練時 `∂L/∂θ_MASF` 含 α，故若從 0 初始化，context 的梯度初始為零，應先確認 α 梯度能啟動，不把「可關閉」當成「一定會自動學到最佳開關」。不得在這 683 張 val 上反覆試 α 再將最好結果冒稱泛化提升。','',
        '此後續配對訓練未開始；沒有額外重跑 BinaryQK 對照，沒有恢復 native_qk E3 或 scale_bias，沒有第二輪優化、GitHub 發布或檔案刪除。','',
        '## 9. 產物與重現入口','',
        '- [原始精度與差值 JSON](artifacts/summary-v1.json)、[精度 CSV](metrics.csv)、[成本 CSV](costs.csv)。',
        '- [本機量測](artifacts/benchmark-v1.json)、[全量殘差診斷](artifacts/residual-diagnostic-v1.json)、[最終稽核](artifacts/final-audit-v1.json)。',
        '- [固定來源](source-pin.json)、[固定 E2 推論權重](artifacts/parent/native-qk-e2-inference.pt)。',
        '- [Pose MASF 候選權重](artifacts/comparison-v1/pose_masf/pose-masf-transferred-inference.pt)。',
        f"- 候選 SHA-256：`{b['sha256']}`。它是 inference-only，不含新的訓練 optimizer，也不是單靠官方 YOLO(path) 可直接辨識的任意 pickle。",'',
        '用 [pose_candidate.py](pose_candidate.py) 的 `build(enabled=True)` 重建相同架構，再以 `torch.load(..., weights_only=True)` 取得 state_dict 並 strict=True 載入。PWL 評估以既有 graph materialization 切到 bittrue；不能用只有原 Pose head 的模板硬載新增 MASF 參數。CPU→GPU 若已有 head 快取，參考 benchmark.py 的搬移處理。','',
        '報告由 [build_report.py](build_report.py) 依已完成 JSON 重建；GPU 比較入口仍會跳過已完成項目。所有測試、cache、權重與 provenance 保留；本次沒有清除產物。']
    lines += ['', '## 後續決定（2026-09-13）', '', '以上保留移接比較結案時的實測與配對建議。使用者後續取消 A 組加訓；僅 B 組程式與 CPU 檢查已完成，GPU 與等待佇列未啟動。現行設定、完整架構圖及歸因限制見 [B 組專項](../pose_masf_training_v1/README.md)。本次依明確指示發布現有圖與分析，commit 名稱為 `5090 Done 0913`。']
    edit(HERE/'RESULTS.md','\n'.join(lines)+'\n')
    print('已重建完整 RESULTS.md、metrics.csv、costs.csv。')

if __name__=='__main__':main()
