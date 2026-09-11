# 推論處理研究：固定框分支與 KD 關鍵點重組

## 結論（2026-09-11）

已完成 CPU 相容性檢查、獨立匯出重載，以及 Float／BitTrue 的完整 COCO 5,000 張與 BBAT5 683 張驗證；沒有新增訓練。只換 KD E2 關鍵點分支可以精確保住原框，但不能保留 KD E2 的 Pose AP 增益，因此不取代目前起始 qSiLU E2 模型。下一步優先處理推論候選路徑與顯示門檻，不因本次負結果立即擴大訓練。

研究採 primary-source 查核與實際驗證，由主代理完成。既有 5 epoch Pose-head KD 已完成，沒有合格 best_joint；共享層試驗尚未執行。

## 目前與本次實測架構

```text
目前起始模型：
shared Backbone／Neck（固定） ──┬─> Detect head（P3 bridge MASF）→ COCO80
                              └─> Pose head ─┬─> 原框／分類 ─┐
                                             └─> 原關鍵點 ──┤
                                                            ↓
                                                   同一候選索引選取

本次候選：
shared Backbone／Neck（不變） ─┬─> 原 Detect head（不變）→ COCO80
                              └─> Pose head ─┬─> 原框／分類 ───┐
                                             └─> KD E2 關鍵點 ┤
                                                              ↓
                                                     同一候選索引選取
```

保留所有共享權重、Detect head、Pose cv2／cv3 及 one2one 對應分支，只置換 Pose cv4、cv4_kpts、cv4_sigma 與其 one2one 對應分支，共 96 個 state 項目（包含 BN buffers，不是 96 層）。PWL 維持 `[-10, 0]`、20 段；BinaryQK、qSiLU、MASF 位置不變。

没有增加參數或算子，也不是兩個模型同時推論。CPU 檢查的未 fuse Pose 模型是 23,531,948 個參數；這不是整個雙頭模型參數量，也不是硬體延遲測量。兩種後端在相同合成輸入上，候選的框／分數精確等於原模型，原始關鍵點精確等於 KD E2。

## 完整驗證结果

以下全部是相同 BitTrue 路徑的 AP50–95。原模型指 optimize 裡的 qSiLU E2 起始學生，不是歷史 combine。

| 指標 | 原模型 | 完整 KD E2 | 本次分支組合 |
| --- | ---: | ---: | ---: |
| COCO 框 | 0.503885 | 0.503885 | 0.503885 |
| person 框 | 0.625887 | 0.625887 | 0.625887 |
| BBAT 框 | 0.618008 | 0.613915 | 0.618008 |
| BBAT Pose | 0.891329 | 0.892962 | 0.890906 |
| ball 框 | 0.505192 | 0.501879 | 0.505192 |
| ball Pose | 0.859649 | 0.861716 | 0.859649 |
| bat 框 | 0.730823 | 0.725951 | 0.730823 |
| bat Pose | 0.923008 | 0.924207 | 0.922163 |

本次候選的所有框 AP 與 COCO／person AP 精確不變，並非只四捨五入相等；BBAT Pose 下降 0.000422555，bat Pose 下降 0.000845110。這是很小的差異，不能聲稱統計顯著，也沒有改善證據。完整 Float 結果另存，不拿 Float 對 BitTrue 差值當模型收益。

結果來源：[CPU 檢查與來源 hash](<artifacts/branch-v2/probe.json>)、[兩後端完整指標](<artifacts/branch-v2/metrics.json>)、[KD 五輪結果](<../../kd/pose_focus_v1/artifacts/direct-kd-result-v1.json>)。候選保留在 `artifacts/branch-v2/candidate.pt`，是 weights_only state-dict 研究匯出，需用本專案架構重建，不能當一般原生 YOLO pickle 直接載入；未標為 best_joint。

## 為什麼不能把兩份最好的指標直接合起來

對同一位置 i，固定共享特徵 F 產生框 b_i、分類分數 s_i、關鍵點 k_i。實際輸出不是所有 k_i，而是由分類分數排序／選取後的集合 `(b_i, s_i, k_i)`。本機 Pose26 的關鍵點座標以 anchor 和 stride 解碼，不是以預測框為座標原點；但候選選取與分數排序仍共用框／分類分支。

KD E2 的評估使用 `s_KD` 排序；本次重組使用 `s_原` 排序。因此 `AP(k_KD, s_KD)` 不等於 `AP(k_KD, s_原)`。完整驗證已顯示 KD 的收益不能只歸因於獨立關鍵點輸出。這尚未區分分類校準、候選選取與匹配各占多少，也不證明關鍵點分支訓練本身無效。沒有完整原生訓練對照，不能把全部變化歸因為 KD。

直接依據：本機 Pose26 分支與解碼（本機／歷史參照：`/home/uxin/yolo/yolo_combine/.venv/lib/python3.12/site-packages/ultralytics/nn/modules/head.py:666`；未隨本次報告發布）、[本次可重現檢查](<probe.py>)。

## 現在推論可以怎麼做：必要順序

1. **先保留原 qSiLU E2 為預設模型。** 本次候選不升版；若任務只看關鍵點，可把完整 KD E2 作為取捨候選，但必須連同 ball／bat 框下降一起呈現。
2. **比較既有 one2one 與 one2many 推論路徑。** 目前重建模型實測 `end2end=True`，走 one2one，後處理沒有 IoU NMS。下一個必要實驗是同一份原權重、相同資料與 imgsz，切到 one2many 加 class-aware NMS，比較 BBAT 框／Pose、ball／bat 與延遲；若沒有收益就保留 one2one。這項尚未執行。必須在 fuse 前切換，本機 fuse 會移除 one2many；不能從已移除分支的部署模型硬切回去。
3. **再處理實際顯示門檻。** ball 和 bat 可各用一個固定 conf 門檻，不需要每張圖片重算 scale。低門檻偏召回，高門檻偏精確率；門檻應依錯誤案例與 PR 曲線決定，現在沒有證據指定最佳數字。正式 AP 比較維持相同低 conf 設定，不用提高 conf 掩蓋漏檢或框不準。顯示門檻不會修正已預測錯誤的框座標。
4. **核對實際推論前後處理。** 使用相同 letterbox／RGB／除以 255、座標反轉換及 ball=0、bat=1 類別映射；框和關鍵點都要回到原圖尺度。這是待查清單，不是已發現有 bug。需要搭配使用者實際推論入口和代表性錯誤畫面，才能判斷視覺問題是否來自部署流程。

只在前述路徑仍不足時，才考慮更大 imgsz／局部高解析或影片追蹤。這些會增加計算或引入時間狀態，不列為目前硬體友善主方案；暫不採 TTA、雙模型 ensemble 或新增 Detect head。

本機 NMS 實作在 end2end 模式直接以 conf 篩選後 return，`iou` 不參與 suppression；所以現在直接調 `iou` 不能改善此路徑。本機 NMS（本機／歷史參照：`/home/uxin/yolo/yolo_combine/.venv/lib/python3.12/site-packages/ultralytics/utils/nms.py:68`；未隨本次報告發布）。官方說明區分 one-to-many + NMS 與 one-to-one NMS-free，但目前網頁參數示例與本機 8.4.90 自訂 nn.Module 驗證入口不完全相同，應以本機 `end2end` 實作為準，不盲抄 `nms=False`。[YOLO26 官方文件](https://docs.ultralytics.com/models/yolo26/)。一般 conf、iou、imgsz 推論參數用途見[官方 Predict 文件](https://docs.ultralytics.com/modes/predict/)。

## 驗證契約與限制

COCO80 使用 `/home/uxin/yolo/coco2017.yaml` 的完整 val。BBAT 使用 registry `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`、dataset ID `bbat5-v1`，正式 Pose YAML `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`、val 683 張；Pose 的框輸出同時提供 ball／bat box 指標，不拿 COCO80 head 冒充 BBAT 二類 Detect。runtime View 只重現 canonical assignment／labels，未重切、抽樣或修改資料。沒有獨立 test split，因此這是驗證集開發結果，不代表未見資料的最終泛化證明。

imgsz 640、Detect batch 32、Pose batch 16、workers 4、FP32 驗證；CPU 合成測試 160×160 只驗證接線，不冒充精度測試。GPU 驗證使用既有 600 秒 child.wait monitor，兩後端完整工作約 140 秒完成。沒有 optimizer、backward 或新訓練。

首次 v1 檢查因 BitTrue 額外 PWL 常數表不在 Float state 中而失敗，尚未進行 GPU 驗證；已改成只載入 Pose head，保留正確後端常數。v2 通過 CPU、匯出 roundtrip 及完整驗證。失敗目錄與事件保留，未覆寫正常結果。尚未驗證 one2many 路徑、ONNX／板端部署、實際影片或新資料；下一階段不把本次候選當成已提升精度的模型。
