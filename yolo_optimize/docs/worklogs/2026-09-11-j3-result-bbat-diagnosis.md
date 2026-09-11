# J3 完成與新舊 combine 的 BBAT 差距診斷

## 變更與原因

J3 完成 17 epoch／7,871 macro step，patience 5 正常停止，無 best_joint；best_pose 是第 12 epoch，以 box／pose 平均選模，不是已通過正式驗收。
依使用者追加要求，activation 與方向 2 延後，先用舊正式融合 checkpoint 作同口徑比較。

| AP50–95 | 舊 combine J3 best_joint | 新 bridge J3 best_pose | 新減舊 |
| --- | ---: | ---: | ---: |
| COCO | 0.498022 | 0.504242 | +0.006220 |
| person | 0.620381 | 0.626983 | +0.006601 |
| BBAT box | 0.630036 | 0.603892 | -0.026144 |
| BBAT pose | 0.903717 | 0.886071 | -0.017647 |
| ball box | 0.507437 | 0.486159 | -0.021278 |
| ball pose | 0.859909 | 0.855779 | -0.004130 |
| bat box | 0.752634 | 0.721624 | -0.031010 |
| bat pose | 0.947526 | 0.916362 | -0.031163 |

## 診斷方式

使用 diagnosing-bugs 技能的可重複差異驗證；主代理執行，不啟用子代理。
先用舊 `analysis/SUMMARY.json` 與新 `summary.json` 重現 BBAT 退化，再進行 GPU 完整同口徑驗證。
新建 `combine/bridge_v1/compare_old_combine.py`：固定 SHA256、安全 weights_only state dict 載入；新 Float／Bit-True／α=0、舊 Bit-True／α=0，均用 COCO 5,000 張及 canonical BBAT5 v1 的 683 張。
不覆写來源權重、不改 split、不改已完成結果。CPU preflight 先確認嚴格 state mapping、PWL [-10,0]／20 段及新 Pose 對 α 切換不變。

## 待檢驗假設（不是已確定原因）

1. Pose head 對新共享特徵仍未充分適應：固定共享 trunk 僅補訓 Pose head 若回升，支持此因素，且應不改 COCO。
2. 共享 BN affine 固定及低 LR 限制適應：與舊配置不同，後續只在有必要時做單變因試驗。
3. 舊 shared MASF 與新 Detect-only MASF 的路徑不同：同權重 α 消融能量化推論依賴，但不能代表重新訓練的因果收益。
4. 評估／PWL 口徑差異：舊 final config 也已是 score_step 0.125 × score_min -80 = -10，20 段；仍須以實際模型與完整驗證確認。

## 驗證結果、困難與風險

已完成：既有結果差異重現，四項 BBAT 指標比舊正式融合低超過 0.02；J3 停止 CSV stale_epochs=5。
CPU preflight 已通過：兩種架構 strict state load、Float／Bit-True materialization 與 PWL 契約正常；新 α=0.0171254873，舊 α=0.1106591076。數值大小不能直接比較增益，因位置與權重不同。
初次 CPU 執行遇到同名 preflight 模組匯入衝突，改用既有 tensor 展平函式與本檔 equality 檢查後成功；未動 GPU 或權重。
GPU 全量比較已啟動：`old-combine-comparison-v1`，PID 690952，UTC 2026-09-10T19:04:48，使用 600 秒 blocking monitor。尚未完成 GPU 同口徑重驗、後續受控補訓。
困難：舊模型的 MASF 在共享 trunk，新模型在 Detect head；不能直接以相同模型骨架載入兩份 state dict。使用各自正確架構與 strict load，不允許靜默漏載。
風險：目前單 seed、重複使用 validation 作開發決策；不能保證補訓回升或聲稱測試集泛化改善。

## 完整重驗與接續

五個案例已完成，新舊八項 Bit-True 指標均重現。新 Float／Bit-True 最大 AP 差約 0.00013，不能解釋 BBAT 差距。舊 MASF on−off：ball box +0.000922、ball pose +0.004465、bat box -0.000089、bat pose -0.000359；新 Pose 六项 AP 對 α 開關完全不變。因此「舊 MASF 直接提供 bat 的 0.031 優勢」不受這次推論消融支持；不能排除其訓練歷史影響。

診斷工具首輪在舊 COCO 重現斷言停止，六項舊 BBAT 當時已完全一致。原因是新建舊 Detect head 未呼叫 YOLO initialize_weights，BN eps 留在 1e-5 而非 1e-3；eps 不在 state_dict，strict load 不會偵測。僅修正診斷骨架初始化後，舊八項 AP 全部重現，支持此診斷工具錯誤的原因。原訓練使用既有完整 head，未受這次新建工具錯誤影響。修正後 monitor `old-combine-comparison-v2` 成功；前三個完成的新模型案例直接重用，舊錯誤輸出保留於原子目錄，正確版本帶 bn-eps-fixed 後綴。

下一個最小試驗：`recover_pose_head.py` 從 J3 E12 安全載入，固定完整共享 trunk／Detect／MASF，只訓練 Pose head，AdamW LR 2e-5、warmup 1、最多 10 epoch、patience 4、physical Pose batch 16、原 seed／增強／資料不變。Pose-only EMA 保留非 Pose state 精確不變；每輪全量 COCO 必須等於起點，六項 BBAT 對起點下降超過 0.02 即保存停止。正式驗收仍使用原基準，不因訓練結束自動採用。
共用 smoke 的 native loss horizon 改為實際 stage.epochs；原 40-epoch 試驗邏輯不變，新試驗會以 10 epoch 真實 loss 檢查更新。

真實 smoke 已通過兩個 batch／32 張訓練影像、Pose head 更新、非 Pose live／EMA state 完全不變及硬體契約。正式 `j3-pose-head-recovery-v1` 已於 UTC 2026-09-10T19:13:22 啟動，PID 696316，使用 600 秒 blocking monitor；未重新訓練先前正常工作。結果待完成，不預先宣稱精度提升。

## 本輪完成結果

Pose head 恢復實際完成 9 epoch、patience 4 平台停止，最佳 E5：box 0.608773、pose 0.886747、ball box 0.492201、ball pose 0.854418、bat box 0.725345、bat pose 0.919075；COCO 0.504242／person 0.626983 完全不變。CPU 安全匯出比較確認 827 個非 Pose 張量逐位一致、360 個 Pose 張量更新。仍有五項 BBAT 未過最初正式 gate，未升格。

接續 `calibrate_pose_bn.py` 只用完整 canonical train 5,964 張、關閉隨機增強，以 physical128 前向重估 48 層 Pose head BN buffers，沒有 optimizer／參數更新。JOB_DONE 後全量驗證：box +0.000196／pose +0.000516、bat box -0.000052，非主要改善，候選不採用。峰值 allocated memory 8,872,495,104 bytes；不等於 physical128 完整反向訓練可行。所有參數與其他 buffers 精確不變，COCO 不變。困難：無；未修改 canonical 來源或 split。

已整理 [完整結果與下一步](<../../experiments/combine/bridge_v1/BBAT_RECOVERY_RESULTS.md>)，同步根層／combine／bridge 入口與 plan。當前沒有執行中的 GPU job；沒有 activation／方向 2 queue。下一個尚未執行方向是共享 affine／Neck 的受控適應，而非繼續延長相同 head-only 訓練。主要未解風險：剩餘 BBAT 差距、單 seed、共享特徵因素尚未經單變因確認。
