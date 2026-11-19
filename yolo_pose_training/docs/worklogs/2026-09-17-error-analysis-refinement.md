# 2026-09-17 誤差診斷與雙解析度精修

## 變更與原因

使用者同意先分析誤差，再从新P3最佳權重做640與960精修。新增analyze_pose_errors.py、run_refinement.py及configs/refinement兩份YAML，已啟動GPU。兩組皆從runs/combine_recipe_p3_640/weights/best.pt初始化，不相互接續、不使用last。初始化SHA256記錄於artifacts/refinement/provenance.json。

資料仍為bbat5-v1，registry=/home/uxin/yolo/configs/datasets/bbat5-v1.yaml，canonical Pose YAML=/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml；formal train／val 5964／683，全程不抽樣、重新切分或改影像標註。基線診斷再次檢查全部val labels摘要等於初始dataset-audit。

## 已完成的基線診斷

對683張val影像分別以640／960推論。固定conf0.25、同類別IoU≥0.5，依prediction信心一對一匹配。這是操作點診斷，不是AP，亦不與validator最佳F1操作點Precision／Recall混用。幾何採rect=False、batch8、FP32；正式AP評估仍沿用原validator設定。

640下，ball393個GT，漏檢57、額外預測54；bat539個GT，漏檢52、額外預測61。ball未匹配中47個沒有同類別IoU≥0.1預測、7個IoU落在0.1–0.5、3個匹配衝突；bat有45個低重疊、2個框不準、5個有其他類別高重疊預測。這些是操作性分類，不能斷定所有原因都是模型分類錯誤。

640可成功匹配且GT可見的keypoint，ball中位誤差1.264px、P90=8.724px；bat中位1.910px、P90=4.911px。ball只有第0點可見，第1點不計；bat逐點誤差分開保存。全體GT可見點的PCK@0.1框對角線（漏檢算失敗）為ball80.15%、bat88.63%。不可只用成功匹配點的誤差來宣稱整体定位品質。

極小球分組以投影至640後sqrt(box area)<16定義，固定分組供兩解析度比較。640匹配159/189，960為169/189；較大球分組sqrt area≥96，640匹配72/77、960為60/77。因此高解析度有小球改善但大球下降的取捨，不宣稱全面提升。

逐物件表格另保存box中心誤差、IoU、尺寸與GT crop Laplacian variance。清晰度量受尺寸與紋理干擾，不能宣稱是動態模糊；沒有遮擋／動態模糊GT，不做無依據分類。完整資料在artifacts/refinement/baseline-errors/。

## 精修設定

| 項目 | 640候選 | 960候選 |
| --- | --- | --- |
| 名稱 | refine_p3_640_lr0001 | refine_p3_960_lr0001 |
| physical batch | 32 | 16 |
| nbs | 128 | 128 |
| epoch上限／patience | 40／15 | 40／15 |
| optimizer／lr0 | MuSGD／0.0001 | MuSGD／0.0001 |
| lrf／warmup | 0.2／1輪 | 0.2／1輪 |
| scale／translate | 0.2／0.05 | 0.2／0.05 |

兩組全解凍、mosaic0、degrees0、無翻轉；momentum0.948、WD0.00027、warmup_bias_lr0不變。暖身後accumulation約4／8，effective batch128，但BN及microbatch loss不等价，不能把差異全歸因於解析度。

選擇低LR＋弱augmentation以保守精修；本輪未加入分組LR，避免首次實驗同時引入自訂optimizer行為。沒有新架構、沒有重訓前面階段，也沒有無限延長輪數。

## 排程與完成後比較

tmux=bbat5_refinement_20260917；狀態artifacts/refinement/status.json；主日誌logs/refinement_queue.log。沿用每600秒GPU監測與每5秒子程序檢查，OOM停止保留checkpoint。640完成後從共同初始權重啟動960；之後兩組各做640／960正式AP評估與相同操作點誤差分析。

推薦規則：同解析度平均AP提高，且ball／bat各自Box／Pose AP的任一項相對原綜合最佳退步不超過0.005（0.5個百分點）。另外列出平均、Pose、ball各自最高候選；保留全部權重，僅產生建議，不替換既有final。這個容許值是本次保守選模決策，不是統計顯著門檻。

## 驗證方式與結果

一對一匹配測試確認重複預測不可重複配對、錯類別不可匹配、空預測能回報；類別保護測試確認0.01退步遭拒而0.003可通過。腳本語法檢查與兩份YAML官方get_cfg驗證通過。基線640／960推論完整完成，輸出共683張／932物件的診斷，所有JSON無NaN。候選GPU啟動檢查另追加。

## 困難與解法

Ultralytics對half=False發出棄用警告，當前8.4.90仍可正常執行，不影響本次結果；固定版本保留明確FP32設定。影像的遮擋與模糊無真值，改為描述可量測的低重疊與清晰度，不宣稱因果。其他困難：無。

## 未解事項與風險

兩組精修與最終比較尚未完成，不保證提升。單seed、formal val反覆選模；沒有獨立test。匹配成功樣本在不同模型間可能不同，須搭配全體PCK与漏檢判斷。背景監測不代表代理交付後持续在線調參。

## GPU啟動驗證

640精修已完成2輪，實際args確認batch32、nbs128、lr0=1e-4、epochs40、patience15、來源為新P3 best.pt且resume=False。已完成輪次CSV數值有限，best／last／last-resumable已產生。GPU使用率81%，19353 MiB，訓練日誌約17.6 GB，沒有OOM。960候選及後續AP／像素誤差比較已排隊。

訓練流程展示：依使用者要求再次讀取640／960實際YAML及官方accumulate程式，確認兩分支共同P3 best起點、重新建立optimizer、全解凍、batch32／16與nbs128；暖身後約累積4／8次。狀態已進入960分支。此次僅展示流程，不變更訓練。困難：無；完整結果仍待比較。

## 完成結果查詢

2026-09-17 18:28:36兩組精修、AP評估與誤差診斷全部完成，GPU使用率0%。640精修實跑16輪、最佳第1輪；960實跑20輪、最佳第5輪。640評估：原P3 Box／Pose=0.63445／0.91701，640精修=0.62998／0.91292，未改善。960評估：新960精修=0.62838／0.91147，比舊Stage4=0.62433／0.90507高，但bat Pose由0.94598降為0.93132，下降1.466個百分點，未通過0.5百分點保護，因此保守推薦仍保留舊960模型。640首選仍是原P3。固定操作點下球漏檢原P3@960為55/393、新960精修為45/393；不能與AP或不同閾值混用。全部checkpoint與報告保留，此次沒有啟動其他訓練或替換權重。困難：無；單seed與驗證選模限制不變。
