# 2026-08-29：產出activation預選老師版報告與圖像

## 任務與範圍

依使用者要求解釋0.3797／0.097等指標，產出可拿給老師看的activation預先選擇圖與報告，並把SD4納入。此次只使用既有activation-smoke-v2結果，不重新執行activation實驗、weight實驗、validation或訓練。

## 變更內容與原因

1. 新增可重現繪圖腳本scripts/render_activation_preselection.py，直接讀取activation-smoke-v2.json計算每格worst raw NRMSE與minimum TopK overlap。
2. 產出PNG、SVG與單頁PDF老師版圖表；同時顯示A3至A8四個非SiLU候選曲線、A5數值解釋、A8預選與SD4分流。
3. 新增activation預選中文報告，明確說明NRMSE不是mAP、TopK overlap的Top-300意義，以及29／300與208／300的直觀解讀。
4. 新增machine-readable預選契約，固定三條A8主線、A6／A7探索、停止新增SiLU／Hardswish／ReLU與SD4範圍。
5. 將SD4拆成W-SD4主線與A-SD4探索支線。W-SD4配對三條A8 policy；A-SD4必須和LSQ+ A4做同parent、同boundary、同budget公平比較。
6. pyproject新增matplotlib 3.11.1，固定圖表重現依賴。
7. 同步README與工作紀錄索引。

## 驗證方式與結果

- 繪圖腳本先通過py_compile，再從原始JSON成功生成PNG、SVG、PDF。
- PNG尺寸2471乘1361、RGBA；PDF為1頁；SVG為向量格式。
- 使用Noto Sans CJK TC字型，經縮小base64預覽視覺檢查，中文無缺字，圖例、callout、三個說明框與SD4文字未裁切。
- A5 poly_quality數據從JSON重新計算為worst raw NRMSE 0.3797268482、minimum TopK overlap 0.0966666667，約29／300。
- A8 poly_quality為0.0816442637／0.6933333333，約208／300。
- 最終檔案與SHA-256已寫入報告。
- pytest -q：34 passed。
- ruff check src tests scripts：All checks passed；ruff format --check：17 files already formatted。
- YAML與數據一致性驗證通過：30／30 smoke cells為passed；A5、A8數值與原始JSON完全一致；三條主線與W-SD4／A-SD4狀態正確。
- 產物驗證通過：PNG為2471乘1361 RGBA、SVG可解析、PDF為1頁，README／報告／工作紀錄連結皆存在。
- 將腳本重跑到tmp後，PNG、SVG、PDF三種格式均與正式成品逐位元相同；SVG／PDF使用固定metadata，SVG另固定元素ID hash salt，消除生成時間與隨機ID差異。
- 沒有修改Full35、yolo_activation、BBAT5、checkpoint或外部Git狀態。

## 困難與解法

- 困難：第一版產生腳本時，多行字串的換行被外層產生器解讀，造成IndentationError。
- 解法：改用raw source模板並先執行py_compile，之後才重新輸出圖檔。
- 困難：ruff初次檢查發現一個未使用變數與一個字典迭代寫法。
- 解法：移除未使用變數並改用items迭代，之後lint、format與py_compile皆通過。
- 困難：內建view_image因bwrap loopback RTM_NEWADDR無法讀取圖片。
- 解法：在tmp建立縮小JPEG，再以唯讀base64內容進行視覺檢查。
- 困難：apply_patch也因相同bwrap RTM_NEWADDR環境錯誤無法使用。
- 解法：只在可寫子專案內使用帶唯一出現次數檢查的精確字串替換，並於寫入後重新驗證雜湊與內容。
- 困難：圖表第一版副標題與第一個panel距離太近，A8區域字樣靠近圖例。
- 解法：增加上方留白並移除重複band文字，重新輸出及再檢視。
- 困難：使用者所稱activation適用SD4可能同時指W-SD4配對與activation本身用SD4。
- 解法：在報告與圖中分成W-SD4主線、A-SD4探索支線，避免錯誤宣稱已有A-SD4結果。

## 未解事項或風險

- 本圖是小樣本no-training proxy，不是正式mAP圖。
- A-SD4尚未實作；是否適合必須由activation boundary分布與公平A4對照決定。
- PDF只是一頁圖表；完整論證以Markdown報告為準。
- 尚無完整validation、QAT、weight SD4或板上硬體結果。
- 完成本報告後依使用者指示停回重新規劃，不自動啟動下一階段。
