# 2026-09-09：MASF 零殘差診斷完成，接續 BR-OFF5

## 診斷結果與決策

`masf-off-parent-diagnostic` 完整 Float／BitTrue 評分完成，supervisor exit0。原 alpha=0.1106591076，只將記憶體 alpha 設為 0，其他 state 不變。BitTrue joint=0.7103965838，原 BEST=0.7111747390。

| 必要 AP | MASF off | 相對原 BEST |
| --- | ---: | ---: |
| COCO Box | 0.4986330499 | +0.0006107135 |
| Person Box | 0.6204020552 | +0.0000205668 |
| BBAT Box | 0.6296191240 | −0.0004163978 |
| BBAT Pose | 0.9016643449 | −0.0020528292 |
| Ball Box | 0.5065150198 | −0.0009220163 |
| Ball Pose | 0.8554442050 | −0.0044645102 |
| Bat Box | 0.7527232282 | +0.0000892208 |
| Bat Pose | 0.9478844847 | +0.0003588517 |

目前 Pose 尤其 Ball Pose 依賴原 shared MASF；不能直接將 zero-gate 圖升格為可搬位 parent，也不能因此推論新位置必然失敗。最大退化仍在工程中期 −0.005 以內，按原計畫執行一個短程 BR-OFF5 恢復，檢驗是否能得到合法 bridge。

## 接線與驗證

trainer／CLI 增加 `masf_off`：從原 BEST、原 criterion 載入後才設定 alpha=0；原 parent digest 單獨記錄。凍結所有原 MASF 參數、QK 係數与 shared BN stats，沿 native Neck／heads scope，不新增模組；snapshot wrapper contract 明記 MASF 介入，inference state 保留 alpha=0。沒有 RepConv、HOG、MASF 移位或 BinaryQK 變更。

相關 training contract tests：23 passed，1.61 秒。真實 `masf-off-gpu-smoke` 於 01:46:30–01:46:53 完成，JOB_DONE／exit0／passed；固定 state／BN／EMA 檢查通過，只有一個成功 optimizer macro，更新丟棄。這是 correctness，不是 bridge AP 證據。

## 正式 BR-OFF5

新 run `artifacts/direction1-20260908/masf-off-parent-bridge`。原 BEST 為共同 parent，既有 native5 為 BR-KEEP 比較來源，僅比較共同 epoch；parent、fresh AdamW、原生 criterion continuation、seed、augmentation、physical32／logical128、Pose16、Neck1e-5／heads2.5e-5、warmup1、horizon10、parent EMA age26597 相同。唯一模型介入是原 shared MASF alpha=0；最多 5 epochs，任一 EMA 必要 AP 比原 PSEL 下降超過 0.005 保存後停止。不能改用較差的 off 零步分數作 gate 以掩蓋原 BEST 的退化。

候選需相對 keep 對照与原 BEST 滿足既定保護門檻，才允許進入 Detect-only 位置比較；失敗就停止 relocation，保留原 shared 圖，繼續 BinaryQK 前置診斷。不是停止所有計畫，也不自動追加未有證據的 epochs。

## 監測、困難與未解事項

正常只作最多 600 秒 blocking shell wait，不讀 log／GPU／指標；結束事件才分析。保持同一監測規則跨工作接續。正式 bridge 尚未取得 AP，無法預先保證恢復。困難：無。canonical dataset、原 final 與 BEST 唯讀，所有舊產物保留，無刪除、commit 或 push。
