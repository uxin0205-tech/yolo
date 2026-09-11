# 第一輪證據總表與尚未完成的邊界

更新：2026-09-09。目的仍是 Detect＋Pose 精度恢復且硬體可實作；不是只讓實驗成功退出。**目前沒有新的精度候選通過验收，仍使用原 J3 BEST。**

## 可核對的結果

[逐 epoch CSV](<results/epoch-comparison.csv>) 包含 6 組訓練、25 個完成 epoch、50 筆 EMA／live 八項 AP。每筆列出來源 summary、相對原 BEST 的差值；EMA 另列相同 epoch native 差值。不把提前停止的分支冒稱同完整預算比較，不以 live 選 BEST。

[證據索引](<results/evidence-index.json>) 已核對每個完成 epoch 的 inference／完整 checkpoint 存在、8 項 AP 有限、parent 路徑、physical32、warmup1。原 BEST SHA256 實際重算一致：`d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c`。此檔案檢查不等於重載每份完整 snapshot 或重新驗證每個 AP；對應實測與恢復測試見各工作紀錄。

| 原計畫項目 | 實際證據／結果 | 狀態與後續 |
| --- | --- | --- |
| BEST 比較 | J3 joint／pose 與條件式 J2 全 Float／BitTrue 重驗 | 完成。J2 ball 較好但 person／bat 退化，保留 J3；last 無新增優先證據，不盲目全掃 |
| Batch128 方法 | physical32×4、64×2 fit；128×1 OOM；32和64吞吐差小，保留歷史physical32 | 完成。logical128，不宣稱physical128可用 |
| Native／HOG | native5；HOG E1–E4 安全停止，patience4不是本次停止原因 | 未接受。HOG不追加LUMA／seed1/2，不把run-local best升格 |
| RepConv17 | 5epochs、零初始化／融合／reload／snapshot測試 | 未接受。不追加layer20或全替換；原BEST無RepConv需融合 |
| MASF bridge | alpha0診斷＋4epoch BR-OFF；同起點 native為保留對照 | 未接受。依原gate停止relocation，不直接把shared模組移位 |
| BinaryQK單點 | site10／22 FP-dot介入，完整驗證均下降 | 診斷完成，非純FP teacher。QAT／KD的合法梯度、teacher、resume／export契約仍未建立，不冒稱訓練已做 |
| 固定scale | 原16個slots；early-return局部adapter，CPU雙backend整圖等價與完整BitTrue8AP不變 | 工程驗證完成。未安裝成正式export流程，未測硬體加速；不算增準 |
| 梯度投影 | 既有新parent樣本負cosine比例高，但correction median約1.22%，低於2%條件 | 未觸發。稀疏量測有侷限，不稱唯一根因 |
| Scale codebook | 歷史dynamic與PoT差很小；沒有符合本輪新group/kernel的提名證據 | 不重跑舊消融、不自加每圖selector |
| B-HEAD修復 | 全5epochs、live/EMA雙驗證、各head子分支moments/更新及共享凍結稽核 | 減輕退化但未接受；不沿失敗E5直接加訓 |
| BN／LR排除 | Quarter LR、BN-only、heads E5 BN-only | 僅部分恢復，未接受，不宣稱唯一根因已找到 |
| MuSGD O前置 | 相同訓練trace、16＋16＋16更新校準；Detect no_decay比值0.268<0.5 | recipe拒絕；不啟動O-M20／O-A20，不私加group-specific LR。初版48macros另記為被修正的前置成本 |
| 視覺案例 | 完整683張逐圖資料、12張GT／原BEST／E5原生HTML/SVG | 開發對照已產生；未收到使用者失敗影片，未作人工盲化驗收或獨立test |
| 最終組合 | 沒有可疊加的已驗收增準winner | 未建立新組合。不能把失敗模組拼起來當完成 |
| 部署量化 | 量化專案明示延期，恢復需重新確認；原attention export只輸出2site state | 尚未授權恢復，未做新PTQ／QAT、全模型export／整數部署或設備latency／energy |
| GPU監測 | 所有後續GPU子工作以最多600秒阻塞等待，終止後才處理 | 執行期間遵守；現無active GPU job。monitor不具心跳式STALLED判定，不宣稱已驗證自動偵測活鎖 |
| 第二輪／person-only | 使用者明確暫不納入 | 保持排除 |

## 原本、嘗試與目前保留的架構

```text
原始／目前保留：
layer16：raw P3 → shared MASF → p3_shared ─┬─> layer17 Conv → P4 → layer20 Conv → P5
                                         ├─> Detect([p3_shared,P4,P5])
                                         └─> Pose  ([p3_shared,P4,P5])

HOG試驗：raw P3 → HOG loss（只在訓練，未接受，不加進原BEST）
RepConv試驗：只layer17，多分支→可融合3×3（未接受）
MASF目標：raw P3 → Detect-only MASF → p3_det（bridge未過，未直接搬移）
```

本次真正可保留的等價程式改善，是固定scale非校準路徑跳過無用dynamic reduction。參數、16個固定係數與圖的數學輸出不變；checkpoint不會自動携帶Python實例方法替換，日後部署入口須明確安裝並重新驗證。

## 接續需要的決策

現有第一輪預設候選已完成或因事前gate停止；這不表示精度恢復目標達成。继续追加同樣訓練沒有已證實收益。BinaryQK STE／KD 若要真正進入訓練，原計畫要求另建合法challenger、可回傳梯度、teacher與恢復／匯出契約，再申請相應訓練；不能把「繼續監測」解讀成繞過baseline guard的權限。

部署亦不是目前可自動啟動的下一項：`yolo_quantize/README.md` 與 `docs/reports/2026-09-08-quantization-phase-handoff.md` 明確記錄使用者／老師決定延後，恢復需重新確認；其V36 parent亦不同於本輪PSEL，不能搬用已有PTQ收益。需要新的階段決策後再建立queue。現在不捏造GPU工作保持忙碌，也不標記整體目標complete。

## 保留與發布邊界

只做唯讀盤點，未刪除。以下都是 Keep，不提出刪除要求：

| ID | 精確路徑 | 盤點大小 | 相依性／恢復風險 | 決策 |
| --- | --- | ---: | --- | --- |
| K01 | `/home/uxin/yolo/yolo_optimize/experiments/artifacts/direction1-20260908` | 約39G | 全部訓練快照、失敗證據、配對trace與資料View；重建費時，部分狀態不可重造 | 保留 |
| K02 | `/home/uxin/yolo/yolo_optimize/experiments/artifacts/direction1-20260909` | 約17M | 逐圖對照、排序／scope因果診斷；本報告有引用 | 保留 |

所有來源資料／權重及不相關dirty worktree均保留。未獲commit／push／清理授權，這些步驟不執行。小型結果表可由[generator](<../../experiments/scripts/build_round1_evidence.py>)重建。
