# 2026-08-27：Full35 combined inference 權重大小診斷

## 任務與範圍

依使用者詢問，唯讀比較：

- `/home/uxin/yolo/original/weight/yolo11m.pt`
- `/home/uxin/yolo/yolo_combine/final/full35/weights/combined/inference/best_joint.pt`

目標是確認 combined inference 為何比官方 YOLO11m 權重大。本次沒有修改、轉換或刪除
任何 checkpoint，也沒有啟動訓練或驗證資料集；COCO2017 與 BBAT5 v1 的影像、標註及
split 均未改動。

## 變更內容與原因

- 沒有修改程式或權重。
- 新增本診斷紀錄，保存實際 byte、參數量、dtype 與 checkpoint payload 證據，
  避免只用檔名或檔案管理器顯示值推測。

## 驗證方式與結果

### 檔案與架構比較

| 項目 | 官方 YOLO11m | Full35 combined inference |
| --- | ---: | ---: |
| 檔案大小 | 40,684,120 bytes（40.68 MB／38.80 MiB） | 106,825,541 bytes（106.83 MB／101.88 MiB） |
| 參數量 | 20,114,688 | 26,529,701 |
| state elements | 20,160,746 | 26,605,138 |
| 主要儲存精度 | FP16 checkpoint | FP32 state dict |
| 任務 head | Detect | shared trunk＋Detect＋Pose26 |
| 實際檔案倍率 | 1× | 2.6257× |

- 本機 `yolo11m.yaml` 建模得到 20,114,688 parameters、20,160,746 state elements。
  若以 FP16 保存，tensor 估算為 40,321,492 bytes，與實際 40,684,120-byte checkpoint
  高度一致；ZIP pickle metadata 也包含 `HalfStorage`。
- `combined/inference/best_joint.pt` 可用 `torch.load(..., weights_only=True)`
  安全載入，頂層只有 schema、checkpoint kind、contract、source、單一
  `state_dict` 與 metadata；沒有 optimizer，也沒有 live＋EMA 雙份模型。
- combined 的單一 state dict 有 1,238 tensors、26,605,138 elements，其中
  FP32 tensors 占 106,419,820 bytes，全部 tensor 合計 106,421,270 bytes；
  檔案其餘約 0.40 MB 是 ZIP／pickle／metadata 開銷。
- 參數量倍率為 1.3189×，FP32 相對 FP16 每個浮點數為 2×；
  `1.3189 × 2 ≈ 2.6378`，與實際檔案倍率 2.6257× 一致。
- 若只把 combined 浮點 tensors 改存 FP16，未計格式微小開銷的 tensor 估算約
  53.21 MB；仍會比 YOLO11m 大，因為它另外包含 Pose26 head 與融合結構。

## 困難與解法

- standalone 舊式 Ultralytics checkpoint 需要 `weights_only=False` 才能直接還原
  pickled model object；此方式可能執行 pickle 程式碼，安全審查拒絕。沒有繞過限制，
  改用本機 YAML 建模、精確 ZIP 大小及靜態 pickle storage metadata 比對。
- combined checkpoint 使用安全、state-dict-only 格式，因此可以
  `weights_only=True` 直接取得精確 tensor 統計。

## 未解事項或風險

- 本次只診斷大小，沒有把 combined 權重轉為 FP16。若要建立約 53 MB 的 FP16
  inference artifact，必須新增獨立檔案並重新執行 strict-load、CPU／GPU forward
  與八項 Float／Bit-True accuracy gate，不能直接覆寫目前 accepted checkpoint。
- 若只需要 Detect，可另用 standalone Detect 權重，但它不是 shared
  Detect＋Pose 融合模型，功能與輸出契約不同。
