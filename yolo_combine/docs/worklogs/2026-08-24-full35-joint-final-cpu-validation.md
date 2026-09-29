# 2026-08-24 Full35 JOINT 最終 CPU 驗收工作紀錄

## 變更內容與原因

在完成 Full35 shared-trunk／dual-head 實作後，完整測試發現三個舊介面相容性問題：

1. `variants/full35/` 新增了 standalone baseline 與真實資料 CPU smoke 入口，但
   `variants/partial75/` 尚無同名待命入口，破壞兩個實驗資料夾的相同相對介面。
2. 新的 `joint.yaml` 已成為正式執行設定，但精簡後的 `training.yaml` 缺少既有工具仍會
   讀取的官方 reference、歷史 P1 與 P1/P2/P3 欄位。
3. 精簡後的 `fusion.yaml` 缺少既有 1:1／2:1 ratio 與 `0.08` 精度 gate 欄位。

本次補回上述相容層，並維持以下邊界：

- `joint.yaml` 仍是新融合流程唯一可執行設定來源。
- Full35 維持 `enabled: true`；Partial75 維持 `enabled: false`，只建立同介面入口，沒有
  執行 Partial75 實驗。
- `training.yaml` 明確區分官方 COCO reference、歷史 basic-split P1 與 canonical 新流程，
  不把歷史資料升格為新訓練入口。
- `fusion.yaml` 保留 1:1 診斷配置並明確選定 2:1；精度 gate 擴成同一 evaluator 下的
  八項 mAP50-95，任一項下降超過 0.08 即失敗。

## 驗證方式與結果

所有命令均使用 `CUDA_VISIBLE_DEVICES=-1`，未配置或使用 GPU。

1. 三個原先失敗的相容性測試：`3 passed in 0.98s`。
2. 完整 pytest：`87 passed, 3 skipped in 108.76s`。
3. 三個 skipped 均為明確要求 CUDA 的 integration test：
   - `tests/test_materialize.py::test_real_full35_float_to_bittrue_mapping`
   - `tests/test_models.py` 的 CUDA model integration
   - `tests/test_training.py` 的 CUDA training integration
4. 先前已另以 CPU 完成真實 Full35 graph materialization、真實 COCO／canonical BBAT5
   兩個 2:1 macro-step、gradient routing、BN freeze、EMA、criterion epoch schedule 與
   exact tiled XNOR 驗證；證據保存於
   `variants/full35/artifacts/formal-joint-cpu-smoke.json`。

## 遇到的困難及解法

1. 困難：新正式設定與舊檢查工具同時存在，若直接刪除舊欄位會破壞相容性。
   解法：將 `joint.yaml` 固定為執行真相來源，`training.yaml`／`fusion.yaml` 作清楚標記的
   相容與稽核層，避免兩份設定都能驅動訓練。
2. 困難：workspace sandbox 在更新既有檔時回報
   `bwrap: loopback: Failed RTM_NEWADDR`。
   解法：先將既有檔移至 `/tmp` 保留，再用 `apply_patch` 重建；沒有刪除資料。
3. 困難：CUDA integration test 在本次「不干擾 GPU」限制下不可執行。
   解法：明確 skip，並保留 preflight／正式空卡 memory gate；未把 CPU smoke 誤稱為
   640 精度或 GPU throughput 驗證。

## 未解事項與風險

1. 正式 canonical Full35 Pose26 P1→P2→P3 checkpoint 仍未產生或指定。
2. 同 evaluator 的 Float／Bit-True 八項 standalone baseline 仍未產生。
3. RTX5090 上 Detect physical batch 128 的完整 AMP memory profile 尚未執行；exact tiled
   XNOR 已消除舊 47.48 GiB 單一 reduction workspace，但不代表整體模型一定放得下。
4. 正式 640 training、validation 與 inference 數值仍待 GPU 空閒後執行。
5. TensorBoard dependency 目前不存在，遵守禁止升級／安裝依賴而未安裝；JSONL、CSV、
   PNG logging 不受影響。

原始 BBAT5、COCO、checkpoint、權威 Full35 bundle與既有 GPU 工作均未修改。Partial75
沒有執行。除上述明確風險外，無其他未解測試失敗。
