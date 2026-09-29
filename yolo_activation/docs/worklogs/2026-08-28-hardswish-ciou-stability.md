# 2026-08-28 Hardswish CIoU Backward 數值穩定修復

## 任務與範圍

在 Full35 activation serial queue 中，qSiLU short recovery 完成後，Hardswish recovery 因
non-finite gradient 使 runner 中止。本次依可重現診斷流程定位並修復，接續原 queue；資料仍為完整
COCO2017 與不可變 Canonical BBAT5 v1，沒有重新抽樣、切分或改動影像與標註。

## 變更內容與原因

1. qSiLU 10-epoch short recovery 已完成並通過八項 hard gate；final BitTrue 為 COCO
   `0.495273`、BBAT box `0.625751`、BBAT pose `0.901986`，最差 accepted-baseline delta
   `-0.008635`。
2. Hardswish physical microbatch 32 兩次都在 epoch 0 相同 deterministic 邊界失敗：最後成功為
   macro 225，下一個 macro 經 16 次 AMP scale retry 後仍產生全域 NaN gradient。
3. 曾以 physical microbatch 16 驗證假設，但在 macro 67 後更早失敗，故撤回此 workaround，正式
   Hardswish 恢復 matched physical microbatch 32。
4. loss-boundary probe 證明 `raw_total` 與 components 在 backward 前仍有限；只在已知 macro 開啟
   autograd anomaly 後，定位為 Detect one-to-many CIoU 的 `bbox_iou` aspect-ratio 項
   `w/h`，其 `DivBackward0` 對零高度 predicted box 產生 NaN。
5. activation recovery context 現在只將 `bbox_iou`／CIoU 算術提升為 FP32，模型其餘 forward、
   backward 與 GradScaler 仍維持 AMP。公式、epsilon、資料、activation、loss 權重、logical batch、
   optimizer step 與 gate 全部不變；resolved config 明記 `ciou_loss_precision: fp32` 與
   `ciou_formula_changed: false`。

## 驗證方式與結果

- 最小 red repro：FP16 predicted box `[0,0,1,0]` 對 target `[0,0,1,1]` 的原 CIoU forward
  有限，但 predicted height gradients 為 NaN。
- 回歸測試：同公式經 `_fp32_bbox_iou` 計算後，輸出為 FP32且所有回傳梯度有限；測試先紅後綠。
- Hardswish matched microbatch regression 確認 queue 命令恢復 `--detect-microbatch 32`。
- 完整 suite：`53 passed`；只有 4 個既有 ONNX legacy exporter deprecation warnings。
- 正式 Hardswish 修復 run 已通過原失敗邊界，到 macro 241 時 overflow 仍為 0，runner 持續運作。
- Hardswish epoch 0 已完整完成並進入 epoch 1；BitTrue 為 COCO `0.471712`、BBAT box `0.596477`、BBAT pose `0.890355`。八項最差為 BBAT bat box，相對 accepted baseline `-0.033779`，相對同 epoch SiLU 八項平均 `-0.018255`；目前 accuracy gate 未過，但依預註冊固定跑滿 10 epochs。
- Ruff、格式與 `git diff --check` 通過。

## 困難與解法

- 單看最終 gradient error 會誤判為 AMP scale 或 activation 本身；以 deterministic replay、loss
  finite probe、極小退化 box repro 與只在觸發 macro 開啟 anomaly，逐步縮小到 CIoU backward。
- `apply_patch` 修改既有檔案仍被環境 bwrap loopback 錯誤阻擋；先保留失敗證據，再使用唯一命中的
  精確 Perl 替換，並在每批後執行 formatter、lint、targeted/full tests。
- 無資料或標註困難；Canonical BBAT5 v1 契約未變。

## 未解事項與風險

- 已完成的 SiLU control 與 qSiLU run 使用原 FP16 CIoU；它們未觸發 non-finite gradient，但與後續
  FP32-CIoU recovery 存在 loss precision 差異。最終比較須明確揭露，並在資源允許時補 matched
  FP32-CIoU SiLU control／finalist runs。
- Hardswish 尚須完成 10 epochs 與 final BitTrue gate；目前只證明原 crash 已修復。
- 兩份 mb32、ㄧ份 mb16 失敗 run 與兩份 debug run 均移入 `artifacts/runs/full35/quarantine/`，可還原、
  未刪除；暫時 probe scripts 已刪除。

## Hardswish 最終結果與 poly_shift OOM 接續修復

- Hardswish 已完成 10 epochs；epoch 6–8 的 checkpoint selection gate 曾通過，epoch 8 最差
  delta 為 `-0.014471`，但 queue 預註冊的 final-epoch BitTrue gate 最終為 `-0.016884`，因此
  Hardswish 正式判定為 gate fail。queue 不以中途最佳值替代 final-epoch 契約，也不延長或重調參。
- 下一個 uniform `poly_shift` 使用 physical Detect microbatch 32 時，在首個 forward 的
  `IntegralPolynomialActivation` 四次冪展開式穩定重現 CUDA OOM；原 queue 與獨立 debug command
  都在約 13 秒內以相同堆疊失敗，30.03 GiB 為該程序自己的 PyTorch allocation，排除前一 job
  顯存殘留為主因。
- 僅將 physical Detect microbatch 改為 16、logical Detect batch 維持 128 後，診斷 run 已連續
  完成 macro 0–11，最後 overflow 0、joint loss 有限，證明記憶體峰值是假設的根因。正式 queue
  因此為 uniform `poly_shift` 加上 `detect_physical_microbatch: 16`；`poly_quality` 共用完全相同的
  polynomial tensor kernel，也以同一 memory-safe physical16 契約預防相同 OOM。
- 新增 queue regression，先確認 physical32 時失敗，再確認修復後命令為
  `--detect-microbatch 16`；完整 suite 為 `55 passed`，仍只有 4 個既有 ONNX deprecation warnings，
  `git diff --check` 通過。
- 失敗正式 run、physical32 red repro 與 physical16 green probe 均已移入可復原
  `artifacts/runs/full35/quarantine/`。資料仍為完整 COCO2017 與 Canonical BBAT5 v1，未重切、抽樣或
  修改標註。

### 本次困難、解法與風險

- 困難：`apply_patch` 仍因 bwrap loopback 錯誤無法更新既有測試、YAML 與工作紀錄；解法是先保存
  失敗輸出，再使用唯一命中的精確 Perl 替換，之後執行 targeted/full tests 與 diff check。
- 風險：physical16 與 Hardswish physical32 的 BN microbatch 統計不完全 matched；short screen
  必須揭露此差異，進入 finalist 時應補同 physical16 的 SiLU matched control。
- 預防性處理：`poly_quality` regression 在修復前同樣抓到 physical32，修復後與 `poly_shift` 都固定
  physical16；logical batch、epoch、seed、資料與 optimizer 均不變。
