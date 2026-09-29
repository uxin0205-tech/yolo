# 2026-09-11：優先比較舊 combine 的 ball／bat 精度

使用者補充：目前 ball／bat 相對舊 combine 的精度下降不少，要求分析並研究能否再提升，可以等目前工作完成再處理。

接續順序更新為：保持目前 J3 正常執行 → 完成 J1／J2／J3 與 MASF 開關驗證 → 以相同 canonical BBAT5 資料與驗證口徑比較舊 combine → 根據差距做必要診斷與針對性補訓 → 融合驗收後才 activation／方向 2。

必須分開列出 ball／bat box AP 與 keypoint AP，不能把單獨 Pose baseline、舊融合模型、新融合模型混為同一基準。優先檢查資料／evaluator／checkpoint 選擇是否可比，再判斷共享特徵、MASF 路徑與訓練配置的影響；不先預設單一原因。

目前只記錄使用者新要求，沒有中斷訓練、額外讀正常 log 或啟動比較實驗。驗證結果：尚未執行本項。困難：無。未解：待 J3 完成後確認舊 combine 來源與同口徑差值，再設計最少必要對照。
