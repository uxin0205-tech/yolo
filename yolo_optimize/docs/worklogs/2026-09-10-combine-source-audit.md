# Combine 來源相容性稽核與停用計畫

## 內容與原因

上一回合已完成方向1實驗與獨立驗證，屬實質進展。本回合在使用者尚未選定MASF候選前，只推進不依赖該選擇的CPU來源稽核，建立獨立 `combine/`，不啟動融合或其他GPU工作。

## 驗證與結果

`CUDA_VISIBLE_DEVICES=-1 .../python combine/audit_sources.py` 正常退出0。兩候選來源SHA符合已驗收manifest；原Pose來源由原joint.yaml定位並保存SHA。對齊記憶體中的PWL backend後，原fusion audit仍回傳兩候選皆不相容，且均只有layer16的4類差異：型別、參數數量、參數signature與buffer signature。

這是來源架構不相容，不是工具失敗；沒有嘗試繞过audit。新增README及停用的plan.json，`selected_detect=null`、`training_enabled=false`。原stage policy對Detect-only MASF的分組及BN規則需要額外適配，尚未實作。

## 困難與解法

舊Pose在shared layer16帶MASF，新Detect不再相同。保留原Pose基準，後續需建立符合新trunk的Pose head適配並重驗，而不是把原Pose AP當作換trunk後的AP。先等待使用者決定是否保留Detect-only MASF。其餘CPU稽核困難：無。

## 未解事項與風險

### 自動接續暫停稽核

方向1交付回合、上一個CPU來源稽核回合與本次接續，均尚未收到是否保留MASF的選擇。本次只重讀 `combine/plan.json`，確認 `selected_detect=null`、`training_enabled=false`，未重跑驗證、未查GPU或啟動監測。已完成不依賴選擇的來源稽核及獨立目錄準備；依使用者「有問題先停下詢問」要求，不擅自決定融合架構。將整體目標標記為受阻，等待使用者選擇後恢復；不是完成或取消後續計畫。新增執行困難：无。

候選選擇尚未回覆；未建立可執行GPU queue、不宣稱融合已開始或會自動接續。舊0.08 gate過寬，不能直接用作本輪COCO保護承諾；新指標門檻及適配驗收仍待明定。activation及方向2未處理。沒有刪除、改資料、改舊權重、提交或上傳。
