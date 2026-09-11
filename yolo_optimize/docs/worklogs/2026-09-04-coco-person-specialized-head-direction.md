# 2026-09-04 COCO person-only 專用 head 方向整理

## 變更內容與原因

- 唯讀稽核COCO2017 person labels、官方instance JSON、現行YOLO26 Detect head與Full35既有predictions。
- 建立研究報告
  [`2026-09-04-coco-person-only-structural-specialization.md`](<../research/2026-09-04-coco-person-only-structural-specialization.md>)，
  整理person尺度、密度、crowd、head成本與第一手行人偵測研究。
- 建立[`OPT-COCO-PERSON-SPECIALIZED-HEAD`](<../../optimizations/coco-person-specialized-head/README.md>)資料夾，
  包含[最小計畫](<../../optimizations/coco-person-specialized-head/plan.md>)與
  [H0/H1/H2終端架構圖](<../../optimizations/coco-person-specialized-head/architecture-report.md>)。
- 主線固定為`H0 Detect80(c3=256) → H1 Detect1(c3=256) → H2 Detect1-Lite64(c3=64)`。
- 修正研究草稿中殘留的舊P2-HCS首輪敘述；P2-HCS、centerness、keypoint/KD與crowd loss全部降為
  有對應誤差證據後的獨立方向。
- 明定person Runtime Dataset View保留全部118,287張train images及54,172張person-negative images，
  只移除非class-0 labels；禁止在完整COCO80 labels直接使用`single_cls=True`。

## 驗證方式與結果

- 研究稽核得到train person-positive／negative為64,115／54,172，YOLO person instances為257,252；
  val為2,693／2,307與10,777 instances。
- COCO val non-crowd persons依annotation area約39.97%／34.55%／25.48% small/medium/large；
  P3/P4/P5均有保留理由。
- 以既有bit-true predictions及COCO API重算：standalone person AP 0.643591，Full35 J3 0.636740；
  AP_S/M/L差為-0.003344／-0.007119／-0.006800，確認joint回歸不是small-only。
- 解析計數：H0→H1 two-branch只省121,818 params／0.6795264 GFLOPs；H0→H2省1,000,410 params／
  4.8531456 GFLOPs，fused差2.4265728 GFLOPs。這些只算解析成本，不冒稱target latency。
- 核對現行`Detect`已有`cls_channels`參數及parser seam；H2可顯式設64，不需把bbox tower或P3/P4/P5一起改。
- 全目錄38份Markdown的本地連結與code-fence稽核通過：missing links 0、unbalanced fences 0；尾隨空白0。
- 重新以獨立算式assert H0→H1／H1→H2／H0→H2參數差、two-branch/fused GFLOPs與person AP_S/M/L差，
  全部通過；並核對`head.py`與`tasks.py`的`cls_channels` constructor/parser seam。
- `/home/uxin/yolo/.venv/bin/python scripts/audit_repconv_seams.py`通過；
  `scripts/audit_accuracy_regressions.py`仍依設計exit 1並列出既有4個MASF/BinaryQK紅燈，沒有新增項目。
- 沒有建立Runtime View、沒有訓練／validation、沒有修改COCO或BBAT5資料、production model與checkpoint。

## 困難與解法

- 困難：根COCO目錄缺train instance JSON。解法：唯讀使用既有P2研究保存的同split官方annotation並記digest，
  不複製、不重建資料。
- 困難：背景研究成檔時仍有三段舊P2-HCS首選文字。解法：主代理完整讀取後，統一改成H1/H2首輪並保留
  P2-HCS為條件式證據。
- 困難：sandbox偶發`bwrap: loopback: Failed RTM_NEWADDR`。解法：使用受管的network-disabled／approved
  workspace命令與`apply_patch`，沒有繞過資料或權限邊界。
- 其餘無。

## 未解事項與風險

- H2的person AP、三seed變異、真實latency與訓練穩定性尚未實驗。
- H2 hidden shape無法像H1一樣epoch-0精確等價；首輪固定初始化，失敗不能只替H2加KD。
- COCO `iscrowd` ignore語意尚未接入YOLO loss；首輪H1/H2保持一致，另案處理。
- head winner整合Full35後仍須重做BBAT5 gates、shared-gradient screen、BinaryQK/PTQ calibration。
