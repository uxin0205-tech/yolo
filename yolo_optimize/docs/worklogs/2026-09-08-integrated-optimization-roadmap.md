# 2026-09-08 三類優化統整、實驗順序與新增方向

## 變更與原因

- 統一為架構（MASF／局部RepConv）、訓練（HOG／條件式梯度投影）、BinaryQK恢復三條主線，新增 optimizations/integrated-roadmap/README.md 與 plan.md。
- 依使用者最新指示暫緩person-only，不把H0/H1/H2放入本輪先決，不建立person Runtime View或更改COCO80 head。
- 把HOG安排在共同J0之後重跑J1–J3；整合版作用raw P3，代號F2-PRE-HOG9，保留舊post-MASF提案並明確區分。MASF仍放主訓練後做兩臂recovery。
- RepConv由舊四臂預設縮成條件式R0/R1；R2/R3僅有證據才增加。它是增準候選，不預設融合後比原3×3省MACs。
- 修正Float backend與FP-QK parent的混淆、新parent site screen不能沿用舊V1-BR結果、historical gap不能跨lineage相減、單任務W-DIR LR不能直接當joint配方。
- 補齊舊計畫沒有確定epoch數的recovery：本整合版RepConv／MASF首版5epochs明確標為新提案，未宣稱已有實測成功。joint QAT則依J3 role量級定義待驗證配方。
- 新增 docs/research/2026-09-08-additional-optimization-priorities.md，排序Q/K trainability、fixed-scale冗餘、prediction誤差分解、crowd ignore與条件式量化敏感層等方向。
- 根README、優化索引與既有六方向的入口加上本次有效順序；保留原始研究，不移除歷史數據。

## 子代理分工

依使用者指定使用 gpt-5.6-luna、reasoning_effort=max：

- roadmap_inventory：機械摘錄舊arm／parent／gate與來源路徑，輸出 source-inventory.md。
- roadmap_doc_sync：按主代理給定的確切規格同步入口／更新註記，檢查文件格式與連結。

主代理負責分析原始碼、比較契約、實驗順序與新增方向判定。子代理未獲准自行選模型winner、修改假說或啟動GPU。

## 驗證方法與結果

- 唯讀核對 Full35 RELEASE_STATUS.json、README、gate-deltas.csv、joint.yaml、Float variant與stage_policy.py等。
- 確認Float變體仍為hadamard／power_of_two；確認qkv.q、qkv.k、score.gamma有hardware_frozen規則；確認shared BN stats目前eval；確認fixed coefficient路徑先算後丟dynamic magnitude。
- 閱讀COCO API、TIDE、MaskFeat、PyTorch BatchNorm2d與QARepVGG的一手來源。這些來源提供機理或語意依據，不作本地增準證明。
- 文件驗證：最終本地連結／code fences／行尾空白與最新使用者範圍檢查由子代理與主代理完成，精確結果於本紀錄後續附錄登錄。
- 本次GPU工作數：0。沒有nvidia-smi、torch import、模型forward/backward、training、AP validation、calibration、kernel benchmark或checkpoint載入／修改。
- 本次新模型AP結果：無。所有方法仍為proposed；person-only為deferred_by_user。

## 困難與解法

- 預設sandbox讀取／更新因bwrap loopback初始化失敗；改用經審核的唯讀exec與interactive apply_patch，不更改安全設定。
- 舊文件在HOG時段與MASF最後recovery之間形成順序矛盾；採raw-P3整合版並新命名，保存舊post-MASF版，避免假稱同一實驗。
- 「Float」名稱與真正FP-QK含義不同；總計畫明列P-SRC未驗證可直接取用，不把現有binary-trained checkpoint改名當乾淨parent。
- 部分官方網頁初次開啟失敗，改以作者頁／arXiv或原始碼核對。
- 文件檢查首次呼叫 `python` 時環境無此指令；改用可用的 `python3` 標準函式庫完成，不安裝套件、不載入ML套件。
- 其餘：無。

## 未解事項與風險

- 合法no-MASF、FP-QK source是否可直接取得及其建構成本尚未做checkpoint級驗證，先列前置，不假定存在。
- 所有新增scope／recovery長度／HOG target細節必須在實作前通過數值安全與resume測試。
- Q/K frozen是已觀察的訓練限制，不是已證明的AP掉點根因；開放更新須符合最終投影硬體契約。
- 改圖／seed／evaluator後不能沿用舊scale／baseline數值；BBAT5沒有test split，不得自行創造。
- 目前工作樹有大量使用者既有未追蹤內容與上層更改；本次未commit、push、刪除、整理checkpoint或變更資料。

## 最終文件驗證附錄

- 本次新增5份文件：整合README、plan、source-inventory、新方向研究、中央工作紀錄；同步22份既有Markdown入口／方向文件。僅文件變更，未修改模型、訓練程式、資料或權重。
- 主代理用 `python3` 標準函式庫唯讀檢查27份文件的本地Markdown連結、fenced blocks與行尾空白，第一輪355個本地連結，錯誤0。全部修正後最終重檢：27份、354個本地連結、錯誤0，並確認檔尾換行；連結數減1來自GAP-09刪除重複參照。行號片段是摘錄快照，不當成穩定的章節錨點。
- doc_sync逐檔檢查22份：相對連結錯誤0、fences錯誤0、行尾空白0、缺少檔尾換行0；18/18方向文件具2026-09-08覆寫通知與新plan入口。
- source-inventory保留55條來源摘錄／問題紀錄，10個GAP已有規劃裁決或文件修正。歷史內容不回寫為新結果；person-only為暫緩。
- 四個正式資料入口存在性檢查全部通過：COCO80 YAML、BBAT5 Pose／Detect YAML與registry；未讀寫影像或labels。BinaryQK README失效相對V1-BR引用已改成既有絕對路徑；只查存在性，未載入checkpoint。
- 人工複核根README的舊RepConv雙點首輪、舊V1-BR兩次validation、舊Full35 alpha-off診斷摘要，已由doc_sync分別改為本輪條件式單點、新parent screens、歷史依賴診斷，避免新舊順序混用；其最終單檔14個本地連結與格式檢查通過。
- 根README保留資料集規範、研究與中文工作紀錄入口，兩個索引已同步。工作樹原本的未追蹤文件狀態保留；無commit／push／刪除。
- 驗證限制：只證明文件引用與規格整合，沒有模型数值parity、精度、梯度或硬體收益驗證。本次GPU工作數仍為0。

返回[工作紀錄索引](<README.md>)或[總計畫](<../../proposals/integrated-roadmap/README.md>)。
