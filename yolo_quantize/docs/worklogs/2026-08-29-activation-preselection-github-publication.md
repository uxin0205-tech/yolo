# 2026-08-29：GitHub發布activation預選證據包

## 任務與範圍

依使用者要求，把已完成的activation預選報告與老師版PNG／PDF／SVG整理成可在GitHub獨立閱讀及重建的先行公開包，commit subject固定為`5090 Profile 0829`。本次只發布activation-output smoke與預選證據，不納入後續PTQ、W6、QAT、正式訓練、weights或其他工作樹變更。

## 變更內容與原因

1. 從最新`origin/main`建立隔離sparse worktree，避免主工作樹大量既有dirty／untracked內容混入commit。
2. 收錄30格source JSON／CSV、兩份machine-readable契約、完整smoke與預選報告、圖表生成器及三種老師版成品。
3. 擴充主要報告，補上模型checkpoint、資料、校正／probe、比較基準、執行環境、graph範圍、已量測／未量測分界與驗證紀錄。
4. 新增子專案與各公開資料夾README，說明閱讀順序、用途、重建方式、Git邊界與限制。
5. 更新根README及根工作紀錄索引，維持全專案可追溯入口。
6. 新增發布manifest，固定公開檔案範圍、SHA-256與明確排除項目。
7. 對固定CRLF的source CSV與Matplotlib生成SVG加入精確`.gitattributes`，保留checksummed bytes並避免Git把生成內容的行尾格式誤當人工文字錯誤。

## 驗證方式與結果

- source JSON／CSV、繪圖腳本與預選契約SHA-256均和報告固定值一致。
- JSON為30／30 passed、0 failed；CSV為30列且policy IDs完全相同。
- 兩份YAML可safe-load；checkpoint、Canonical BBAT5 v1、assignment unchanged、A8主線與A-SD4未實作狀態均符合契約。
- A5 poly_quality重新計算為0.3797268482／0.0966666667；A8為0.0816442637／0.6933333333。
- 從source JSON重建PNG、SVG、PDF，三者和正式成品逐位元相同。
- PNG為2471×1361 RGBA、SVG可解析、PDF為1頁。
- 新增Markdown相對連結全部存在；publication manifest列出的21個檔案大小與SHA-256全部一致。
- 公開包沒有`.pt`、`.pth`、ONNX、engine、cache或超過100 MB單檔。
- 繪圖腳本in-memory compile通過；Ruff回報`All checks passed!`與`1 file already formatted`；`git diff --check`通過。
- 沒有啟動GPU、模型validation、QAT或訓練。

## 困難與解法

- 困難：主工作樹相對`origin/main`分歧且有大量其他專案變更，不能直接提交。
- 解法：先fetch最新remote，再由`origin/main`建立獨立branch與sparse worktree，只複製授權範圍。
- 困難：第一次完整worktree checkout包含超過11萬檔案，工具輸出中止時留下不完整checkout。
- 解法：不在不完整狀態寫入，改用Git sparse-checkout收斂到根文件與`docs/`，恢復乾淨基線後才組裝公開包。
- 困難：內建`apply_patch`仍受`bwrap: loopback: Failed RTM_NEWADDR`阻擋。
- 解法：先保留apply_patch失敗，再於隔離worktree使用帶唯一marker／不存在性檢查的受控UTF-8寫入，並於最後以hash、diff與重建驗證。
- 困難：第一次Markdown連結檢查正確指出README已連到尚未建立的`PUBLICATION_MANIFEST.yaml`。
- 解法：先建立manifest，再重跑完整重建、schema、hash與連結檢查；最終全部通過。
- 困難：首次staged `git diff --check`把source CSV的CRLF及Matplotlib SVG path行尾空白列為trailing whitespace；直接清洗會破壞固定SHA-256及逐位元重建。
- 解法：在根`.gitattributes`只把這兩個精確生成檔設為`-text -diff`，保留原始bytes；人工維護文字仍完整接受`git diff --check`。

## 未解事項或風險

- 30格結果是每task兩張calibration與一張probe的no-training proxy，不是完整mAP。
- A-SD4、W-SD4、QAT、bit-true export與板上硬體結果均未包含或宣稱完成。
- 這個先行包能重建老師版圖表；raw GPU smoke runner與外部Full35／yolo_activation bundle不在此次commit範圍。
- GPU依使用者指示保持不使用；本次收尾與驗證全部是CPU或靜態檔案檢查。
