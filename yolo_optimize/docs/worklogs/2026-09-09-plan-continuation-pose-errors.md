# 2026-09-09：恢復完整計畫，補齊同圖錯誤驗證

## 範圍與原因

上一階段完成 queue 不等於整個優化目標達成。使用者要求按原計畫繼續，本次重新逐項閱讀 master plan、optimizer policy 及原生退化報告。精度恢復仍未完成；沒有把現有等價效率修正替代增準目標，也沒有將所有條件式候選變成必跑長訓。

原計畫要求同案例的漏檢、誤檢、框與點位比較，先前尚缺可追溯全量逐圖資料。依 diagnosing-bugs 技能建立可重現回饋：`scripts/audit_pose_errors.py` 對原 BEST 與 native E5 EMA 各跑完整 canonical BBAT5 val683，保留原官方 AP evaluator，另以 conf=0.001／0.25 重算匹配。框 IoU 與點位 OKS 分別記錄，不互相冒充。原資料、來源權重與 split 不變；不是 person-only 或第二輪。

## 實際執行與結果

GPU 工作以 blocking supervisor 執行，`child.wait(timeout=600)`；正常期間不讀 log 或查 GPU，退出才分析，exit 0。兩個模型各 683 張且影像 ID 唯一；原 BEST Ball Box AP=0.5074370361873287，E5=0.5023759175814532，精確重現既有 delta −0.0050611186058755475。這是重新推論的退化回饋，不是只讀舊數字判斷。

| conf=0.25 指標 | 原 BEST | Native E5 |
| --- | ---: | ---: |
| Ball Box IoU50 TP／FP／FN | 348／90／45 | 349／86／44 |
| Ball Box IoU75 TP／FP／FN | 240／198／153 | 240／195／153 |
| Ball Pose OKS75 TP／FP／FN | 354／84／39 | 353／82／40 |
| Bat Box IoU50 TP／FP／FN | 500／96／39 | 496／88／43 |
| Bat Pose OKS75 TP／FP／FN | 506／90／33 | 502／82／37 |

低門檻 conf=0.001，Ball Box IoU50 TP 372→373、IoU75 TP 264→265。這些結果不支持「E5 單純多漏球」的解釋；全 AP 的置信排序與其餘 IoU 門檻仍需進一步拆解。也不能說 E5 更好：其 AP 安全線依然失敗，bat 的固定門檻召回下降。固定 conf 的計數與跨信心／IoU 的 AP 回答不同問題。

## 新產物

- `scripts/audit_pose_errors.py`：保留正式 AP、逐影像 GT／prediction 原圖座標與固定門檻錯誤。
- `scripts/render_pose_error_report.py`：由上述結果產生 self-contained HTML＋SVG，不修改原影像、不再推論。
- `artifacts/direction1-20260909/pose-error-audit/summary.json`：全量 AP 與重現 verdict。
- 同目錄 `parent.json`、`native_e5.json`：683 張逐圖資料；`case-manifest.json`：全量聚合及案例選擇。
- 同目錄 `comparison.html`：12 個同圖 GT／原 BEST／E5 對照，退化、改善、共同失敗各最多 4 個。

案例排序預先使用 conf=0.25 下 box50／pose75 的 FP+FN 差與共同錯誤數，兩種 matching 獨立，計數可能重疊，不是新評分。案例只是 validation 開發診斷，不作新 training subset。相似來源影格／增強版本不當成獨立證據。manifest 中影像路徑解析 runtime symlink 後可落到 historical raw storage；驗證入口仍是 canonical runtime View，不是使用 raw 的歷史 train split。未改任何 assignment。

## 困難、限制與接續

困難：直接圖片檢視工具因 bwrap loopback 權限失敗；改為產生可開啟的原生 HTML／SVG 對照，沒有冒稱已人工看過圖片或完成瀏覽器渲染檢查。其他困難：無。

已有逐图資料，可繼續區分高 IoU 框定位與 confidence ranking，然後依原 B／Q／O 的門檻決定必要修復或加訓；不能僅因無 GPU 工作就宣告整個目標完成。這次沒有新增 optimizer／LR／BN 訓練變因，也沒有重跑先前已做的 head／Neck／BN 單點介入。使用者原始失敗影片尚未提供，因此目前案例不能替代真實應用驗收。

未解決：精度回升、原始應用場景效果、最終部署驗證。所有新 GPU 工作仍須 blocking monitor，600 秒最大 kernel wait；遇 ERROR／STALLED／JOB_DONE 才查看必要資訊。沒有刪除、commit、push、資料改版或啟動子代理。
