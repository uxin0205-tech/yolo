# 2026-09-09：HOG 結果、ball／bat 覆蓋與 BN 診斷

## 結論與同回合比較

HOG 在 E4 依 `patience=4` 停止，正常退出，沒有 GPU 錯誤。原生完成 E5，但 HOG 因果比較只用共同 E1–E4。E1 的 HOG 關閉，四項 EMA AP 完全相同；開啟後 overall／person 沒有額外收益，本版不採用、不延長或拼進正式模型。

下表為 HOG 減原生對照的 internal AP50–95，單位是 0–1 AP：

| 回合 | overall | person | sports ball | baseball bat |
| --- | ---: | ---: | ---: | ---: |
| E1（HOG 關） | 0 | 0 | 0 | 0 |
| E2（漸開） | -0.001192184 | -0.000643805 | -0.006545131 | +0.009783385 |
| E3 | -0.000319275 | -0.000635798 | -0.004428375 | +0.002486720 |
| E4 | -0.000828527 | -0.000573541 | -0.002725028 | -0.006755113 |

HOG E4 的四項 AP 為 0.505058131／0.626156171／0.510523202／0.476215964。bat 在 E2／E3 有單類正向差異，但沒有維持至 E4，且當時 overall／person 都較差，不能挑 bat 峰值稱整體成功。來源是 `artifacts/prefusion-hog-control-v1/summary.json` 與 `artifacts/prefusion-hog-hog-v1/summary.json`。每回合完整 train118287、925 次更新及 COCO5000 EMA／live 驗證；非 warmup 回合含驗證約 630–632 秒。

## ball／bat 監督覆蓋

新增 `probe_hog_coverage.py`，在 CPU 固定檢查 seed 20260919 的前 1024 張增強後 train 影像，不更新 optimizer、不改資料、不依 AP 選樣。

| 類別 | GT 數 | 自身框無有效 P3 cell | 比例 |
| --- | ---: | ---: | ---: |
| 全部 80 類 | 7002 | 196 | 2.80% |
| person | 2103 | 52 | 2.47% |
| sports ball | 59 | 13 | 22.03% |
| baseball bat | 38 | 1 | 2.63% |

零有效 cell 都是框內沒有 cell 中心，不是能量門檻額外排除。ball 的 59 個框有 23 個短邊小於 8 像素；bat 是 38 個中 1 個。這支持 hard cell-center mask 對小球有直接監督盲區，不支持將同樣問題泛化到所有球棒。

這是固定 prefix，不是全資料估計；自身框無有效 cell 不代表完全沒有梯度，重疊框與原生 Detect loss 仍可能監督該位置。球棒框內的背景／人體可能影響 HOG，但尚未證實，不能直接歸因。證據為 `artifacts/hog-class-coverage-v1/summary.json`。不立即掃 P2、bins、μ；MASF 後續保留 ball／bat 觀察。

## 只恢復 head BN 統計的反事實

新增 `probe_head_bn.py`，對原生 E1 EMA 只在記憶體恢復共同起點的 head BN mean／variance／counter，斷言所有已學參數與其餘 state 不變，再完整驗證 COCO5000。

| 指標 | 原生 E1 | 恢復 head BN | 差值 |
| --- | ---: | ---: | ---: |
| overall | 0.505415519 | 0.502121334 | -0.003294184 |
| person | 0.626813434 | 0.625193620 | -0.001619814 |
| sports ball | 0.514542884 | 0.514858173 | +0.000315289 |
| baseball bat | 0.472786671 | 0.471148972 | -0.001637698 |

結果不支持「搬回統計就能恢復」，不採用。這不排除另一條從頭固定 BN 訓練的路徑，但目前不追加該輪。證據為 `artifacts/native-e1-head-bn-probe-v1/summary.json`，沒有寫回原 checkpoint。

## 後續、困難與風險

原生 Neck＋head 本身有 overall 回退；後續單點 RepConv 使用更小、兩組一致的訓練範圍，先驗證初始化、梯度與部署折疊，不以退化的完整 Neck 配方當唯一對照。保留 A0、B100、late E8；探索 parent 不等於正式 winner。

實驗新增執行困難：無。文件補丁首次封裝格式錯誤，未套用即被工具拒絕，重新分開套用。未解事項：BinaryQK 缺口仍在，RepConv／MASF 尚待融合前實測；ball／bat 是觀察項，overall／person 仍為主要 gate。
