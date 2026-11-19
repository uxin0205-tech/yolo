# 2026-11-02：第七階段 Activation 比較與 GPU queue

## 任務與範圍

使用者指定開始下一階段，要求比較不同 activation 並安排 GPU 訓練驗證。本輪限定 Stage7，RepConv 仍待指定；交給 RepConv 的共同 parent 必須是已驗證的 qSiLU。全程單一主代理。

## 變更內容與原因

先唯讀核對 optimize activation bridge 的實際程式、配置、parent、10 輪結果及上游 J3；來源 SHA 已保存於 `yolo_activation/stage7/protocol.json`。另外查閱舊 Full35 PolyQuality 的配置與完整比較報告，將其加入本輪比較。Hardswish 舊零樣本掉分較大，但短恢復能挽回不少，因此四組均排相同 10 輪機會；不重跑 SiLU control 或 ReLU。

共同 parent 固定 Stage6 E1，SHA `e9b9fd7ccc88f10bbd170835138ebde5c7810db0726424496e2cac2aa462ffe7`。新增 Stage7 自包含 checkpoint、activation 類別／係數／alias／call-sites manifest 與嚴格重載。Stage6 manifest 不改寫，先驗證當前 activation 再以淺層 task view 的 metadata 驗證原 Stage3 binary／scale／bias／PWL／reciprocal 契約，禁止 normalization fallback。

沿用 optimize AdamW、低 LR、warmup1/cosine、shared BN running 固定且 affine 可訓練、head BN 可更新、fresh optimizer／EMA0、Detect16×16／Pose16／reference64；Attention 所有 tensors 凍結，保留 B2/S1/D1。Pose 權重依同一 train-only 梯度校準規則選定。CIoU 用 optimize 的 FP32 wrapper，模型仍 AMP。四組合計最多 40 輪，qSiLU 條件式降 LR 恢復最多 5 輪。每輪完整 FP32 AP，最佳權重與 resume 分開。

Queue 提供來源／配置鎖、完成 epoch resume、5000／683 完整 evaluator、12 個區域敏感度、最終重載完整 AP、比較 CSV／圖及獨立 qSiLU handoff。背景每 600 秒監測，worker 每 5 秒輪詢完成／錯誤，資源衝突時只終止本輪 worker 並等待空閒，不處理他人程序。正式結果尚待 GPU 執行。

清理前核對外部依賴與 GPU 程序。直接刪除 226 個舊 checkpoint／大型 predictions，釋放 26.137058918 GiB；保留四份量化等外部仍用的 activation best_joint 權重及引用小檔，SHA 核對一致。release 唯一模型、報告、配置、共用程式保留。沒有建立 archive／backup，沒有修改 optimize 或正式資料。[清理清單](../../yolo_activation/stage7/cleanup-20261102.json)。

## 驗證方式與結果

四種 activation 的範圍、分段、導數、尾端與 FP32／FP16／BF16 有限值檢查通過。qSiLU、PolyShift signed16/Q10 全 65,536 codes 已與浮點參考比較；結果見 `numerics.json`，不是整個任務模型 bittrue 或硬體量測。

四組 CPU 160×160：187 路徑／30 組 alias、26,453,936 parameters；單任務與雙任務對應輸出相同，strict checkpoint 重載相同；activation fallback、錯誤 qSiLU profile、改寫 reciprocal table 均被拒絕。選擇性 neck policy 正確保留其他區域。macro 單元檢查確認 515 個樣本組成 256／256／3、不丟最後尾批，Pose 週期與尾批正確。optimize 參考 source SHA、共同 parent SHA 及 inherited optimizer betas／WD 已核對。

## 困難與解法

首次 CPU 重載比較失敗，原因為 parent manifest 內整數 class-name keys 在 JSON sidecar 變為字串；改以 JSON canonical form 比對 sidecar，payload 仍保持原型別重建，四候選完整重載通過。未修改權重或上游 manifest。

本機沒有 `python` 短指令，兩個檔案生成輔助段落未執行；改用專案 `.venv/bin/python` 重做並檢查 imports／CPU 契約。未啟動 GPU 或實驗，不影響數據。

## 未解事項或風險

尚待 GPU 零樣本、校準、AMP smoke、正式恢復與最終評估；不把 CPU 檢查稱作精度結果。單 seed，未做 RTL/HLS／板端量測，Conv/QKV/PV/projection 與 activation 任務參考仍為浮點。Stage6 原四項目標未達保持記錄。矩形與二次冪 feature shape 分析仍在最後階段，不在本輪更換輸入尺寸或 token 上限。

## GPU 啟動修正與再驗證

第一個零樣本 worker 在首批前失敗：forward hook 回傳 context.pop() 的字串，意外取代 Tensor 輸出。Queue 自動停止，尚無完整 AP、訓練或 optimizer 更新。改為明確回傳 None，四候選再驗證診斷開關前後完整輸出逐值相同，觀察到 163 個部署位置及 2 個 Attention sites。source SHA 前後、錯誤原因與刪除清單見 `diagnostics-fix-20261102.json`；失敗 queue 已直接刪除重建，沒有封存。

Ruff、shell syntax、imports、macro 尾批覆蓋與獨立圖表重畫通過。未修改 optimize／canonical operator。修正後重新鎖定來源並啟動背景 GPU queue，等待正式驗證與首組訓練實測。

## 四組完整零樣本結果

COCO 5000／BBAT5 v1 683，FP32 E2E；以下尚未恢復訓練。

| Activation | COCO AP | BBAT box AP | BBAT pose AP | 最差十項 delta | 相對門檻 |
|---|---:|---:|---:|---:|---|
| hardswish | 0.414813 | 0.492820 | 0.838370 | -0.138553 | 未通過 |
| poly_quality | 0.512459 | 0.607368 | 0.899057 | -0.005620 | 通過 |
| poly_shift | 0.509990 | 0.595784 | 0.893998 | -0.021040 | 未通過 |
| qsilu_pq | 0.509268 | 0.599448 | 0.899132 | -0.012813 | 通過 |

四組都完成 163 個部署 activation 位置與兩個 Attention sites 的真實輸入診斷，D1 row-sum 誤差在原誤差範圍內，沒有改用浮點除法。依事前 protocol 仍給四組相同短恢復機會；此表不作最終選型。

## qSiLU 正式長訓啟動

2026-11-02 00:46（Asia/Taipei）已啟動 10 輪 qSiLU 恢復，supervisor PID 2862525、worker PID 2863261。GPU 為 RTX5090 32 GiB，未發現其他 compute process。

梯度校準從初始 weight0.25 得到 median ratio 3.453361，最終 Pose weight 0.04，確認 median ratio 0.976021，在 0.8～1.25 範圍內。全部 calibration 為零更新，每 macro 還原 parent。

2 個真實 AMP macro smoke 通過，558 個參數張量更新；frozen tensors 與 EMA 一致，峰值 allocated 16.3774 GiB、reserved 16.4980 GiB。Smoke 權重丟棄，正式長訓從乾淨候選 E0 重新載入、fresh optimizer／EMA0。首步 Detect256＋Pose16，loss／梯度有限、AMP retry0，共享層與兩個 head 均有梯度。

正式 queue 已接手每 600 秒監測；第 1 輪共 463 個 macro。四組全部跑完前不把零樣本表當最終比較，Stage8 不啟動。

首 10 步已完成、AMP retry0。去除首步啟動成本後約 5.60 秒／macro，估計每輪含驗證約 44.5 分鐘。qSiLU 本組暫估 11 月 2 日 08:00～09:00 完成；整個 queue 暫估 24～36 小時（11 月 3 日 01:00～13:00），其他候選時間尚未實測，不含讓出 GPU 的等待。穩定長訓後結束前景 working，背景 supervisor 持續。

## 14:56 進度核對

使用者詢問目前狀況，讀取 supervisor、實際 GPU process、全部完成 epoch 與選定 checkpoint SHA，沒有修改執行中的來源鎖、配方、權重或 queue。qSiLU／Hardswish 均完成 10 輪；PolyShift 正在第 5／10 輪，PolyQuality 待接續。GPU 為本輪唯一 compute process，約 23.0 GiB 顯存，最近 600 秒監測正常。

| 指標 AP50–95 | E1 parent | qSiLU E8 | Hardswish E8 |
|---|---:|---:|---:|
| coco_AP | 0.512676 | 0.513951 | 0.498474 |
| person_AP | 0.641255 | 0.643234 | 0.629305 |
| sports_ball_AP | 0.515161 | 0.514662 | 0.495165 |
| baseball_bat_AP | 0.470841 | 0.469036 | 0.426266 |
| box_AP50_95 | 0.609202 | 0.613904 | 0.609630 |
| pose_AP50_95 | 0.899850 | 0.900762 | 0.905294 |
| ball_box_AP50_95 | 0.493883 | 0.497430 | 0.504761 |
| ball_pose_AP50_95 | 0.863474 | 0.864731 | 0.878103 |
| bat_box_AP50_95 | 0.724521 | 0.730378 | 0.714499 |
| bat_pose_AP50_95 | 0.936227 | 0.936792 | 0.932484 |

qSiLU 通過十項 activation 相對門檻，依 balanced selector 保留 E8；COCO、BBAT overall box／pose 與 BBAT 各類指標均高於 E1，但 COCO sports_ball／baseball_bat 分別略低 0.000498／0.001805，不能宣稱每項皆改善。Hardswish Pose 提升，但 COCO sports_ball 下降 0.019995、baseball_bat 下降 0.044575，未通過相對門檻。兩組各有少量 AMP overflow 自動降 scale 重試（qSiLU 合計 2、Hardswish 合計 3），無未處理異常，frozen state 每輪均通過。

PolyShift 已完成 4 輪，目前依相同 selector 的最佳是 E3；每輪約 38.7 分鐘，暫估今日 18:00～18:30 完成。PolyQuality 之後約需 6～7 小時，整輪連同 12 個區域 sensitivity／重載 AP 暫估 11 月 3 日 01:00～03:00 完成，依後續實測及 GPU 可用性更新。qSiLU 已過相對門檻，現行規則不觸發額外 5 輪。

困難與解法：本次狀態檢查無新困難；未解事項是其餘候選、敏感度及最終重載尚未完成，並不把本次 qSiLU 相對改善視為原獨立模型目標全過。Stage8 未啟動。詳細十項數據見 `yolo_activation/stage7/status-20261102-1456.json`。

## 17:28 進度核對

使用者再次詢問狀態。Supervisor PID 2862525 與 PolyShift worker PID 2893259 正常，單一 GPU compute process；最新進度由第 9 輪 step330 持續推進，未發現停滯。已完成 8 輪完整驗證，依既定十項 balanced selector 目前選 PolyShift E8。

PolyShift E8：COCO 0.514738、BBAT box 0.611086、BBAT pose 0.899852，通過 activation 相對門檻；綜合 score 0.645560，仍低於 qSiLU E8 的 0.646277。E7 的 COCO 單項曾達 0.515209，但不是目前綜合 selector 選定權重，沒有將不同 epoch 的單項最佳拼成一組。

qSiLU／Hardswish 各 10 輪完成；後者仍有 COCO 球／球棒門檻未過。PolyQuality 尚待校準及訓練。PolyShift 仍預估今日 18:00～18:30 完成，整個 Stage7 含敏感度及重載驗證維持 11 月 3 日 01:00～03:00 估計，需依 PolyQuality 實測更新。

本次只讀取進度、驗證結果與程序，補充中文紀錄；沒有改動執行來源鎖、配方、模型或 GPU queue。困難：無新問題。未解事項：其餘訓練與最終驗證未完成，`result/stage7/` 尚未產生符合當前流程；Stage8 未啟動。機器可讀快照見 `yolo_activation/stage7/status-20261102-1728.json`。

## 22:44 進度核對

使用者詢問目前狀況。讀取 supervisor／GPU 程序／四組訓練結果與逐輪歷史，queue 正常；目前唯一 GPU compute process 是 PolyQuality worker 2916020，顯存約 23.0 GiB。qSiLU、Hardswish、PolyShift 各完成 10 輪；PolyQuality 已完成 6 輪完整驗證，正在第 7／10 輪。來源鎖、配方與 queue 均未更動。

| 候選 | 已完成輪數 | 目前選定 epoch | COCO AP | BBAT box AP | BBAT pose AP | 綜合 score | 相對門檻 |
|---|---:|---:|---:|---:|---:|---:|---|
| qsilu_pq | 10 | 8 | 0.513951 | 0.613904 | 0.900762 | 0.646277 | 通過 |
| hardswish | 10 | 8 | 0.498474 | 0.609630 | 0.905294 | 0.634882 | 未通過 |
| poly_shift | 10 | 10 | 0.514996 | 0.613729 | 0.901011 | 0.646895 | 通過 |
| poly_quality | 6 | 6 | 0.514525 | 0.613434 | 0.900274 | 0.646257 | 通過 |

PolyShift 在 E10 成為目前比較綜合 score 最高者，0.646895 略高於 qSiLU E8 的 0.646277；這是本輪單 seed、十項選型規則下的小幅差異，不是每項都較好。qSiLU 的 BBAT 球框／球關鍵點仍高於 PolyShift。依使用者要求，比較 winner 與 RepConv 主線分開，後者仍固定 qSiLU，不因比較排名改變而自動替換。

PolyQuality 仍在訓練，未解事項包含剩餘輪數、12 個區域敏感度、四組重載完整 AP 及交付。按實測每輪約 38.6 分鐘，訓練暫估 11 月 3 日 00:47 左右完成；整個 Stage7 仍估 01:00～03:00。Stage8 未啟動。

困難與解法：本次無新問題，四組目前沒有訓練安全線違反。PolyQuality 已完成六輪共 3 次 AMP overflow 均由既有降 scale 重試處理，最新進度 retry0，沒有未處理錯誤。機器可讀快照見 `yolo_activation/stage7/status-20261102-2244.json`。

## Activation 硬體資源與實作難度核對

使用者詢問哪個 activation 較耗硬體資源、較難實作。重新唯讀核對 canonical `activation_lab/activations.py`、`hardware.py`、Stage7 `numerics.json`，並查閱 PyTorch Hardswish 正式公式與 AMD UG1399 運算單元共享說明。沒有改動程式、來源鎖、訓練配置或 queue。

在相同輸入／輸出位寬、相同誤差目標與吞吐量下，依目前直接算式推估：PolyQuality 通常最複雜，PolyShift 次之，qSiLU 與 Hardswish 屬較低成本；這不是 RTL/HLS 合成的面積／功耗排序。qSiLU 與 Hardswish 的實際先後仍取決於固定係數實現與 pipeline。

| 候選 | 目前算式資料乘法次數 | 其他硬體負擔 | 本輪定點證據 |
|---|---:|---|---|
| Hardswish | 1 | 加常數、截斷、1/6 固定縮放；係數不是單一二次冪 | 尚無本輪整數 emulator |
| qSiLU | 1 次共用平方 | 四段二次式、比較／選擇、dyadic 係數 shift-add；可共用運算，不需照 Python 全分支平行實現 | signed16/Q10 全碼 reference，最大誤差約 0.001453 |
| PolyShift | 4 | u²／u³／u⁴／u⁵；固定係數可 shift-add，中間值與累加須防溢位／消去誤差 | signed16/Q10 全碼 reference，最大誤差約 0.002175 |
| PolyQuality | 4 | 同樣五次多項式，加上一般固定係數量化與常數乘法實現；固定係數也可量化成 shift-add，但需另外選位寬與驗證 | 尚無本輪整數 emulator |

PolyShift 的 Shift 指固定係數可移位加減，不代表 u 的冪次不用資料乘法。Hardswish 除以固定 6 可用固定乘數／倒數係數及明確捨入處理，不能把它當成一次右移，也不需要直接配置一般動態除法器。PolyQuality 的 threshold109/16 本身可精確二進位表示，主要難點是由 threshold 導出的非 dyadic 多項式係數與高次中間運算，不是比較常數是否為二次冪。

運算次數不等於 DSP 數量：多個乘法可以共用單元以降低面積，但可能增加每筆資料週期或降低吞吐量；反之展開並行會增加運算單元／暫存器。Python reference 採 int64 容器不代表硬體必須全用64位，還需逐個乘積／累加推導必要位寬。沒有把4對1次資料乘法說成整個YOLO面積少75%或實測4倍加速。

目前 qSiLU 精度已恢復，資料乘法少、係數固定且已有局部定點驗證，因此維持指定 qSiLU RepConv 主線的理由充分。PolyShift 綜合 AP 略高，仍保留為精度／成本比較；最終硬體優劣需在相同位寬／吞吐量下做合成。PWL exp＋LUT256 reciprocal 為候選共用契約，這份比較只算 activation，不把它重複算成某候選特有成本。

來源：[本機 canonical 實作](../../yolo_activation/src/activation_lab/activations.py)、[成本與定點 reference](../../yolo_activation/src/activation_lab/hardware.py)、[PyTorch Hardswish](https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardswish.html)、[AMD UG1399 運算單元共享](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Limiting-the-Number-of-Operators)。

驗證：公式／乘法鏈／係數與既有全碼誤差逐項核對；未啟動新實驗。困難：無。限制：未做 RTL/HLS、板端延遲、面積或功耗量測；以上為本輪實作的結構性推估。

## 2026-11-03 00:23 進度核對

使用者詢問目前狀況。Supervisor 與單一 PolyQuality GPU worker 均正常，沒有未處理錯誤。四組共 40 輪已完成 39 輪完整驗證，目前最後一組 PolyQuality 正在第 10／10 輪，約 step180～190／463，最新 AMP retry0；未改動來源鎖、配方或 queue。

PolyQuality 已完成九輪，目前選 E7：COCO 0.515324、BBAT box 0.615526、BBAT pose 0.901004，相對門檻通過。其綜合 score 0.646868515，與 PolyShift E10 的 0.646894878 僅差 0.000026363，不能把單 seed 的這種微小差距宣稱為穩定優勢。qSiLU 接續主線要求保持。

按當前每輪耗時，最後一輪加完整 AP 預估 00:48～00:50 結束，之後 queue 接續區域敏感度與四候選重載驗證；整個 Stage7 仍估 01:00～03:00 完成，Stage8 未啟動。困難：無新問題。未解事項：最後一輪結果、敏感度、重載 AP 與正式交付仍待完成。快照見 `yolo_activation/stage7/status-20261103-0023.json`。

## Queue 完成

比較選定 `poly_quality`；qSiLU 狀態：已通過 activation 相對門檻，可作下一階段共同 parent。完整結果見 [Stage7 結果](../../yolo_activation/result/stage7/README.md)。所有完成候選經重載完整 AP 核對；每個 worker 與 600 秒監測記錄保留於 queue。未啟動 Stage8。
