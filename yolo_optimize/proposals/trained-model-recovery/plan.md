# 已訓練 YOLO26M joint 模型的驗證與最小 recovery 流程

日期：2026-09-08。範圍：現有 COCO80 Detect＋BBAT5 ball/bat Pose。所有模型執行步驟均是**未來取得 GPU 授權後**的計畫；本次僅讀程式、文字配置、日誌與檔案 metadata。

> 最新修正：接續來源已固定為 `yolo_combine/final/full35`，不預設其中各部份已 train 好。先依[final 接續清單](<final-readiness.md>)完成基準確認／必要原生 loss 修復，得到 `P-READY` 後，才進行本頁 W/M/Q/R 新優化。`training_ready=false`；本頁的 5–10 epochs 是條件式比較窗口，不代表足以完成基準訓練。

## 1. 先釐清結論邊界

「已經訓練過，但看起來不好」不代表必須重訓，也不直接證明某個模組設計錯誤。這一輪的目標是：從使用者實際使用的 checkpoint 出發，找出可重現的錯誤，驗證一個改動是否比同起點、同訓練量的對照更好。

目前未取得具體視覺案例，依診斷流程尚停在問題重現的前置階段。以下列的是檢查項目與條件式實驗，不是已成立的根因。舊的「clean FP／無 MASF → J0 → 後續消融」適合另一個從源頭分離因果的研究問題，不再作為本輪 recovery 的必要前提。

## 2. P0：固定真正的基準

1. 使用者已選定來源 `yolo_combine/final/full35`；P0先以J3 `best_joint`為候選，品質尚待確認。視覺案例／實際推論設定仍待補，但不再以重新指定專案為前置。`best_detect` 的Pose未適應，不能直接作task=both的共同parent。
2. 留存 checkpoint hash、EMA/live 選擇、程式版本、resolved config、資料 registry、backend、task、class names、head branch、輸入尺寸、letterbox 與輸出座標規則。檔名相同不代表權重狀態相同。
3. 診斷回放用部署所用的 inference 權重；新訓練的兩臂都從同一份已訓練參數開始。優先匹配部署使用的 EMA，確認載入規則後建立相同新 EMA。新 optimizer/scheduler 是 **warm-start**，不是 exact resume。真正續跑原訓練才使用相應 full-resume 全狀態。
4. 原始 checkpoint 不覆寫。所有結果按 parent hash／arm／seed 記錄；先核對已做過的消融是否同 parent、同設定，避免換名字重跑已有證據。
5. 原生loss修復另立B分支，詳見最新清單。只有通過基準驗收的權重才命名P-READY；如果P0已合格則P-READY=P0，否則是經驗證的修復產物。下列新優化必須同P-READY起跑，不預設未修好的P0也適合直接加模組。

## 3. 不訓練的第一道驗證：重現、分解、歸因

這裡「不訓練」不代表本次可直接使用 GPU；推論與模型驗證也需後續授權。

### 3.1 視覺問題清單

對使用者提供的完整案例固定 frame ID／時間戳，保存原始預測與並排渲染。只看漂亮截圖不算通過。按現有標註能力記錄：

| 問題 | 能量化的項目 | 限制 |
| --- | --- | --- |
| 漏檢 | 有 GT 時按固定 IoU 的 recall、FN 數，附尺寸分組 | 無 GT 的影片只能列人工確認案例，不能報 AP |
| 誤檢／重複框 | FP/image、同 GT 重複匹配數、類別混淆 | 閾值下降可能同時補漏檢與增加誤檢 |
| 框偏移／尺寸不穩 | 與 GT 的 IoU、中心與尺寸誤差 | 單張框中心變異不代表影片抖動 |
| 影片閃爍／抖動 | 有可靠軌跡／GT 時，計算可見期間漏幀及去除真實運動後的誤差 | 沒有對應 ID／GT 時以固定完整片段作人工比較，不造數字 |
| Pose 點不準 | 沿既有可見性與點序定義，附正規化點誤差及原有 Pose 指標 | ball/bat 的 2 點不是人體骨架；不得自行換點序或增加骨長限制 |

不得自行建立新 BBAT5 split、抽樣或增改標註。若需新人工標註，另請使用者授權；使用者案例是開發用案例，不冒稱獨立 test set。

### 3.2 同一權重先排除設定差异

- 先重播使用者原始設定，再以同一權重檢查 confidence `0.10 / 0.25 / 0.40` 的敏感度；這只是診斷，不預設改成最低閾值。只有保存了足夠低閾值、未過早截斷的原始預測，才能靠離線過濾完成比較。
- 核對 EMA/live、Float/Bit-True、Detect/Pose 的 class mapping、letterbox 還原、keypoint 座標、task=both，以及 one-to-one／one-to-many 的實際輸出分支。
- 原推論程式預設 `imgsz=640, conf=0.25, iou=0.70, max_det=300`，**這不是使用者實際指令的證據**。`end2end` 分支存在時，不能假定調 NMS IoU 一定影響輸出。
- 可增加現有 J2、best_detect、best_pose 的相同案例回放；這能定位版本／selector 的取捨，但不是單一模組的嚴格因果消融。
- 现有 Float 與 Bit-True 都可能保留同一 BinaryQK 算法。兩者差異只能歸為所比較 backend 的近似，不能標成「純 FP-QK 對 BinaryQK 掉點」。

### 3.3 正式評估與視覺設定分開

完整跑既有 COCO val 與 canonical BBAT5 val，不重切資料。第一份報告維持現有 formal evaluator 的配置和指標名稱，作為與歷史結果可比的主表；若另加 canonical COCO API 指標，獨立列欄，不混算提升。

COCO bbox AP 使用 IoU 0.50:0.95、分數排序及其 maxDets 規則；不是固定 `conf=0.25` 的準確率。評估匯出閾值可暫定 0.001，但必須先核對原 evaluator 契約；API 的 AP maxDets=100 與推論匯出 cap=300 是不同環節，均需記錄。參考 [COCOeval 原始碼](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)。

需要細分 Detection 錯誤時可用 [TIDE](https://github.com/dbolya/tide) 的分類、定位、重複、背景與漏檢等錯誤分類；它不直接驗證 Pose 點誤差或影片抖動。

## 4. 訓練共同契約：只比較一個改動

每對新優化 control/candidate 必須同 P-READY、同 seed、同資料順序、同 augmentation、同有效 batch、同 optimizer/scheduler、同更新次數與 EMA 初始化。新模組初始化用独立 RNG，避免它改變 control 的資料亂數。共同參數逐一驗證相同，初始 identity 模組驗證數值等價。基準修復收益獨立報告，不歸功於新方法。

第一輪用 seed=0 篩選；有正向訊號且有後續預算才補 paired seeds=1、2。不能因單 seed +0.001 就宣稱統計顯著。以固定末輪 checkpoint 作配對主比較；best checkpoint 另表呈現，不能各自挑不同訓練量再當等預算結果。

共同超參數見 [hyperparameters.md](<hyperparameters.md>)。主線先保留既有 MASF 位置、BinaryQK policy、固定硬體係數，從已訓練權重恢復，而不是強制回到未量化起點。

## 5. 優先順序：不是全部都跑

| 順序 | 開啟條件 | 必要工作 | 停止條件 |
| --- | --- | --- | --- |
| V0 | 所有情況 | P0 重現、設定核對、完整驗證 | 若是使用方式問題，先處理該問題，不開訓練大矩陣 |
| B | 基準品質尚不可接受且有可重現修復目標 | 只用原生loss補訓／修復，驗收P-READY | 未通過就回查，不進HOG／MASF／QK新優化 |
| W | 定位／局部形狀問題有證據，且未有等價失敗消融 | `W-CTRL10 / W-HOG10` 各 10 epochs | HOG 無增益則停止，不疊其他 loss |
| M | 有 shared MASF 依賴／跨任務干擾的可測證據 | 必要 bridge，再做 Detect-only 對照 | bridge 無法恢復就保留舊位置，不硬搬 |
| Q | 確認剩餘損失與 QK／量化有關 | 先核對舊消融與固定 policy，再建立合法 challenger | 尚無 gradient／hardware 契約就不啟動 QK 訓練 |
| R（選配） | 尚需改善跨尺度 Neck，且有預算 | layer17 RepConv 兩臂各 5 epochs | 無指標／部署收益則保留原 Conv |

W 是取得可接受P-READY後的一個低推論成本訓練候選，**不是基準修復的第一必跑，也不是預先認定 HOG 必定能解決目前的問題**。若 V0 明確指向其他原因，可跳過 W，優先對應分支。同一輪先選一對，不把 W/M/Q/R 全列成必跑費用。

## 6. W：已訓練模型上的 HOG 輔助訓練

原本：

```text
raw P3 ─> 既有 shared MASF ─┬─> 後续 P4／P5
                           └─> Detect + Pose 原有輸出
```

本輪候選（部署不增加 HOG head）：

```text
raw P3 ───────────────┬─> 既有 shared MASF ─> 原有 Detect／Pose 路徑
                     │
                     └─> HOG head：1×1 Conv(C3→9) ─> HOG loss
                                                    ↑
                        同一張完成增強的訓練圖 ─> 方向直方圖 target

推論：移除 HOG head／target／loss，原來的主架構與位置維持不變。
```

HOG head 要主幹學會局部邊緣方向，不是第三個物件類別頭，也不是把 HOG 向量直接融合到部署輸出。反向梯度透過 live raw P3 訓練 Neck；既有 Detect/Pose loss 仍在。

- `W-CTRL10`：相同已驗收 P-READY，10 epochs native loss；輔助 head 可用獨立 RNG 同樣建構但不啟用 loss。
- `W-HOG10`：同上，僅增加 raw P3 HOG loss。保留既有 MASF／QK 設定；freeze backbone、attention、現有 MASF，只訓練 Neck 與兩個 task heads，candidate 另訓練 HOG head。
- 新 ID 有意區別舊 `F2-PRE-HOG9`：本輪是 trained-parent warm recovery，不是重跑舊 J0。
- 規格提案：stride 8／cell 8 pixels、9 個 unsigned bins、同一張完成幾何與色彩增強的 image 產生 target；RGB luma 權重 0.299/0.587/0.114，梯度方向在 [0,π)，以梯度幅值加權並在相鄰 bins 線性分配。每 cell 以總幅值正規化為分布，用 soft-target cross entropy。只用有效 GT box mask 內且梯度能量非零的 cells；無有效 cell 時 aux=0，但原生負樣本 loss 不取消。
- `C3` 由實際 raw P3 shape 確認，現有提案為 256；不得硬編碼後忽略不匹配。目標與 P3 cell 的原點、padding、GT mask、兩任務 box 座標必須有單元測試。
- 這不是聲稱完全複製論文的 HOG 配方，也不是已證明的新方法。只在 HOG 勝 native control 後，才按需要補同預算 `W-LUMA10`，分辨收益來自 HOG 還是一般辅助監督；未補這一組不能主張 HOG 特有創新。

## 7. M：不能把已訓練的 shared MASF 直接搬走

依目前 graph，原有 shared seam 會影響後面的主 Neck，也會影響兩種 task；這是資料流事實，不等於它必然有害。

目前（以語意節點表示，實際 child owner 由 graph adapter 核對）：

```text
layer16：raw P3 → shared MASF → p3_shared ─┬─> layer17 → P4 → layer20 → P5
                                         │
                                         ├─> Detect 的 P3 輸入
                                         └─> Pose   的 P3 輸入
layer19：p4_shared ──────────────────────────> Detect／Pose 的 P4 輸入
layer22：p5_shared ──────────────────────────> Detect／Pose 的 P5 輸入
```

希望測試的 Detect-only 架構：

```text
layer16：p3_raw ────────┬─> layer17 → P4 → layer20 → P5
                       │
                       ├─> MASF → p3_det ───┐
                       └─> p3_pose ──────────┼──────────┐
layer19：p4_raw ─────────────────────────────┤          │
layer22：p5_raw ─────────────────────────────┤          │
                                            ↓          ↓
                      Detect([p3_det,p4_raw,p5_raw])    Pose([p3_raw,p4_raw,p5_raw])
```

圖中的 Pose 也接 P4/P5；只有 Detect 的 P3 經新 MASF。實作 owner 應在 Detect 的 P3 入口，不能只因名稱叫 P3 就放在雙頭共用節點。最終 top-k 競爭仍可能改變其他尺度的輸出，所以 feature 隔離不等於所有 detections 相互獨立。

執行順序：

1. 可先用 P-READY 做暫時性的 shared gate as-is／effective gate=0 診斷，兩者不訓練。瞬間移除若掉點，表示目前模型依賴它，不能據此說新位置永遠不好；也不能據此直接搬模組。
2. 若確實要測位置，先建立 **no-MASF bridge**：`BR-KEEP5 / BR-OFF5` 同 P-READY 各 5 epochs，只更新 Neck／Detect／Pose heads，其他參數及固定 QK 係數凍結；唯一處理差別是舊 shared MASF 的輸出殘差有效係數是否為 0。不可改寫原 checkpoint。
3. bridge 候選必須相對 BR-KEEP5、P-READY 及 P0 通過保護指標／視覺非劣門檻，才成為共同 parent `P-BRIDGE`；若失敗就停止 relocation，保留原圖。
4. `M-CTRL5 / M-DET5` 同 P-BRIDGE，各 5 epochs。兩臂都只放行 Detect P3 的 box/class predictors（實際 one-to-many 和 one-to-one owner）；候選另加 Detect-only MASF。raw P3 producer、P4/P5 與 Pose 凍結。
5. 可按逐 tensor 記錄的 mapping 複用舊 MASF context 權重，但新殘差 **effective gate 必須初始為 0**，不以 sigmoid(0)=0.5 冒充 identity。context 在 gate=0 的第一步沒有梯度是預期現象，需檢查 gate 是否能學到非零。
6. 初始兩臂輸出要數值等價；gate on/off 不可改 raw P3、P4、P5 或 Pose 的中間輸出。通過後才測正式 AP 與部署成本。

若沒有已驗證可用的 no-MASF parent，這條分支的最小預算是 **bridge 兩臂＋位置兩臂，共四個 5-epoch runs**，不能宣稱只需兩個 runs。它是條件式工作，非所有模型都先做。

## 8. Q：BinaryQK 的 recovery 與硬體邊界

先利用既有訓練和消融結果核對 parent、site、scale、backend、fixed coefficients、head scope、評估契約；若歷史結果不是同起點就只能當線索。現有 backend 比較不能代替真正 FP-QK／BinaryQK 對照。

本輪 W/M/R 先維持現有 BinaryQK 和固定硬體係數。若更新了上游／下游權重，仍需複驗 fixed policy 的效果，不默默重新校準 Q/K scale。8 個 static scale、dynamic image-dependent scale、STE 或新 basis 都是不同處理，不能一起加入。

重要限制：現有 `_joint_config_impl.py` 明確以 `Q/K STE challenger is excluded from the baseline` 拒絕 baseline 的 `qk_ste=true`；現有硬體契約也會凍結 Q/K 等參數。所以「把開關改 true 再 fine-tune」不是可執行的正式計畫。

QK 訓練須另建立、驗證明確的 challenger 契約，不繞過 assertion。先做 trainable parameter 白名單、Q/K live gradient、fixed coefficient hash、數值／resume／export parity 測試，再申請對應訓練。蒸餾另需可比較且真的較好的 teacher；既有 detached `last_scores` 不能直接拿來當 student 可回傳梯度的蒸餾節點。未滿足前只保留預備超參數，不把 KD 或 QK-ste 排成現在立即必跑工作。

## 9. R：選配 RepConv

只先測 P3→P4 的 layer17 3×3 stride-2 Conv，`R-CTRL5 / R-175` 同已訓練 parent 各 5 epochs。複製原 3×3+BN、新 1×1 branch 的 BN gamma/beta=0；stride=2 不加 identity branch。初始化與 fusion 都須驗證數值等價。

RepConv 不是保證降低 fused 3×3 MACs 的方法；優點候選是訓練參數化，只有實測才知道本模型有沒有收益。FP fusion 等價不代表 INT8 自動等價。layer20、全面替換與多方法疊加皆不是首輪必做。

## 10. 接受、回退與部署複驗

先寫好 gate 再看結果，避免事後挑有利標準。以下是**新提議的工程門檻，不是已驗證的最優門檻或統計顯著性**：

- 正式八項沿歷史 schema：COCO overall/person、BBAT box overall/ball/bat、Pose overall/ball/bat。每項相對 matched control 不低超過 0.001；候選同時不能相對 P-READY 或 P0 超過此退化門檻。既有 joint selector 分數原樣另列，不重新發明加權。
- 指標提升路徑：COCO overall 或既有 joint selector 至少 +0.001，並通過八項保護。視覺提升路徑：八項非劣，且事先選定的主要錯誤明確減少；有足夠已標註事件時暫用錯誤數相對下降至少 10%，事件很少則報原始個數，不用百分比製造精確感。
- 無 GT 影片只能以相同完整案例的盲化並排人工確認視覺改善，不能說已量化 AP／時序誤差；使用者未確認前，僅標候選，不能升格部署 winner。
- 10 epochs 在 epoch 0/5/10 完整驗證；5 epochs 在 0/5 完整驗證。出現 NaN/Inf 立即停止；10-epoch 分支若中期任一保護指標相對 P0 掉超過 0.005，暫停調查，不自動追加 epochs。提前停掉的 arm 記為安全淘汰，不冒稱等預算成功比較。
- 有效候選再補 paired seeds 1/2，報每個 seed 的差值、平均及變異；若不補，只報單 seed 工程結果。
- 部署前對同一候選逐步驗證：去掉 HOG 分支／完成 RepConv fusion → 原定精度與固定 QK policy → 既有允許的部署量化流程。每一步都看固定案例、完整八項及部署差值，不把浮點訓練增益直接當最終硬體增益。
- 固定 QK 的 scale／coefficients 不在一般量化校準中被暗改；需要改動則另立版本與審核。硬體 latency／memory 在實際目標設備上量測，無設備不聲稱加速。

視覺案例與 validation 都參與模型選擇，不能再當未見測試集；本輪不自行新增 test split。失敗保留 P0／前一個已驗收版本，不自動把多個失敗改動合併。

## 11. 資料入口與交付物

- COCO80：`/home/uxin/yolo/coco2017.yaml`。
- BBAT5 Pose：`/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`。
- BBAT5 ball/bat Detect 若相應流程需要：同資料根目錄 `configs/detect.yaml`；不取代 joint COCO80 Detect。
- registry：`/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。資料 assignment、來源、影像與 labels 一律保持；不以 raw/historical dataset 或另切 split 規避路徑問題。

未來每組實驗需留下 manifest、resolved hyperparameters、checkpoint lineage、trainable parameter 清單、paired seed／update trace、完整 validation JSON/CSV、同場景預測與渲染、gate 結論、中文工作紀錄。這次只交付規畫與文字核對，沒有建立訓練結果或驗證圖片。
