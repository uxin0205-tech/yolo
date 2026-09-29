# Bridge E8 相對原模型的表現與先前方案

## 核對目的與方式

回應使用者詢問 bridge 是否優於原先、先前採用什麼方案。只讀原模型 registry、attention YAML、graph graft 程式、完整 COCO baseline 與 E8 EMA summary，使用 CPU 算術計算差值；未啟動 GPU、載入 checkpoint 或恢復 queue。

## 結果

同口徑 BitTrue COCO AP50–95：原 Full35-B100 overall=0.5035890014、person=0.6241112368；bridge E8=0.5082119552／0.6276641271，分別增加 0.462295／0.355289 個百分點。此為整輪恢復、微調與結構調整的結果，不能全歸因 bridge。

bridge 相對同 native P3 MASF、無 bridge 的 E8：overall +0.005117、person +0.012210 個百分點；相對無 MASF Head control E8：overall -0.005514、person -0.003533 個百分點。變動很小，不宣稱穩定優勢或達到原 +0.001 AP 增準門檻。

原 B100 COCO ball AP=0.5162344722、bat AP=0.4694824179；bridge=0.5131480895／0.4806641861，故並非每個類別都改善。這些是 COCO 類別指標，不能冒稱 BBAT5 baseline 比較。

## 原方案與架構

原起點精確 ID 為 `full35-b-f10`，B 階段、train_fraction=1.0，registry 的既有 gate 為 rollback；使用者指定其為研究起點並不等於它已被原專案驗收。不是 `full35-a2` retained checkpoint，也不是未修改的原生 YOLO26M。

YOLO26M Detect，保留 layer16／19／22 的 P3／P4／P5（stride 8／16／32）；兩個 HardwareFriendlyAttention 位於 model.10.m.0.attn 與 model.22.m.0.1.attn。attention 設定為 Hadamard basis、decomposed_2d bias、power_of_two scale、BitTrue PWL；score_min=-80、step=0.125，實際下限 -10、20 段。原 Full35 MASF 在 layer16 C3k2 後處理全通道，DW3／DW5＋1×1 混合，以殘差輸出同時供 P3 Detect 與下游 P4／P5；不是 Partial75 的部分通道版本，也沒有新增 P2 head 或 Pose head。

立即在 bridge 之前的 native fork 已是 P3 Detect-only MASF；bridge 只改其 one-to-one 訓練梯度路徑，推論架構不再改。完整圖示與成本仍見 `combine/pose-masf/ARCHITECTURE.md`。

## 困難與限制

困難：無。沒有新測 BBAT5 原 B100 對照、硬體延遲或統計顯著性；不將不同來源／split 的結果混算。保留停止狀態及所有舊資料，無刪除、commit 或 push。
