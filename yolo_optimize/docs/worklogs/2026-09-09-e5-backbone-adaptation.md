# 2026-09-09：E5 分析與 Backbone 後段適應實驗

## 結果與判斷

control／QK E5 均正常完成完整 COCO80 訓練與 val5000。EMA BitTrue internal AP50–95 如下，值為 0–1：

| 權重 | overall | person |
| --- | ---: | ---: |
| FP | 0.518019276 | 0.630794912 |
| A0 | 0.506738574 | 0.626805274 |
| control E5 | 0.507063915 | 0.626509338 |
| QK E5 | 0.506623691 | 0.626800401 |

QK E5 相對 A0 為 -0.000114883／-0.000004873，已接近起點；相對 control 則為 -0.000440224／+0.000291063，沒有達到兩指標不退超過 0.001 且其中一項改善至少 0.001 的暫時驗收標準。control E4→E5 overall 為 0.506375619→0.507063915，person 為 0.625995235→0.626509338。QK E4→E5 overall 為 0.506568893→0.506623691，person 為 0.626457361→0.626800401。不能稱已補回相對 FP 的約 0.0113 overall 缺口。

目前證據支持兩個仍待區分的解釋：短訓仍在適應、或只調 Q/K 與 head 的範圍不足。依使用者允許修改 Backbone 的授權，以同一 QK E5 權重分成 narrow 延長與 late 解凍兩組，同時測試這兩種解釋；不從兩個不同歷史 parent 比較訓練範圍。

## 原本與本次候選

```text
相同起點：QK E5（沒有 MASF）
                 ├─ narrow：Q/K + layer23 Detect head → E6–E8
                 └─ late：以上範圍 + layer8 C3k2
                                      layer9 SPPF
                                      layer10 C2PSA 周邊
                                      layer22 C3k2／attention 周邊 → E6–E8

兩組推論圖不變：
layer8 → layer9 → layer10 ──> Neck → layer16 P3 ──┬─> layer17 → layer19 P4
                                               │                 └─> layer20 → layer22 P5
                                               └─> Detect([P3,P4,P5])
```

late 額外調整 Conv／BN affine、V、PE、輸出投影與 FFN；score gamma、fixed PoT、PWL buffers、位置 bias 都固定，所有非 head 的 BN running statistics 仍固定。固定的早期 Backbone 不變；本次是解凍而非更換拓撲，不增加推論運算或每圖 scale selector。layer22 屬 Neck，不稱為 Backbone。

## 超參數與續訓契約

兩組共用 `a0-qk-recovery-v3/epoch-05-resume.pt`，原 Q/K LR 5e-7、head LR 2.5e-5、AdamW (0.948,0.999)、weight decay 0.00027、clip10、AMP FP16、physical32×4、完整 train118287／val5000，criterion／cosine 仍沿原 10-epoch horizon。既有 optimizer／scaler／EMA 4625 updates 完整載入，不重啟原 warmup。

late 新增參數群 base LR 2e-6，再乘現有 cosine；僅新群在第一個解凍 epoch 做 0.1→1 的 1-epoch LR ramp，新增 moments 從零開始，舊群 moments 保留。這是「解凍＋必要的新參數 optimizer 初始化與 LR ramp」的範圍策略對照，不宣稱能把新增群 LR 與 scope 的效果單獨分離。非 head BN statistics 固定，可避免同步改 BN 統計策略。

先各跑一個真實 128 張 macro，檢查同 parent SHA、相同資料 trace、首更新前 loss 一致、舊群 step4626／新群 step1、EMA4626、criterion updates5、所有新增參數的有限非空 gradient，以及各新增 layer 的首步相對權重變化在 (0,0.001) 內。此比例是初步安全檢查，不代表長期 AP 一定穩定。smoke 的更新丟棄，正式仍從共同 E5 起跑。

先各訓至 E8，再比較完整 AP 趨勢。patience6 保留前序歷史，只以 overall／person 決策；相對 A0 任一項退超過 0.005 停止分析。未達標不堆 HOG／RepConv／MASF；有改善才進一步決定長訓或後續模組。

## 變更與驗證

CPU 已核對 A0 的 layer8 C3k2、9 SPPF、10 C2PSA、22 C3k2、23 Detect 及完整參數名稱。擴充 `continue_a0.py` 的完整 E5 邊界與受限 scope，新增 `run_scope_pair.py` 在校準檢查通過後串行訓練。保留所有舊 script hash／checkpoint／報告，來源檔案不改。

## 困難與解法

原 optimizer 不含新增參數，不能直接用更寬 optimizer load_state_dict；先載入原群狀態並驗證全部 step，之後 add_param_group。新參數尚無 Adam moments，因此另列初始化與 ramp，不稱 exact uninterrupted 全參數續訓。資料流沿用成對新 epoch 邊界 seed 的明確限制，沒有重切或抽樣。

## 未解事項與風險

GPU 校準與 E8 結果尚待執行，不保證此 LR 或解凍範圍有效；若失敗，先分析必要資訊並安全調整，不覆蓋失敗產物。更換 Backbone 結構、全 Backbone 解凍與新增模組尚未執行；單 seed、反覆 val 選模不能取代最終獨立場景驗證。

## GPU 校準完成與正式啟動

AST 語法檢查通過。2026-09-09 13:59（Asia/Taipei）narrow／late 真實 128 張 macro 均正常完成；首更新前 loss_sum 都為 223.10570526123047，資料 trace 都為 `af4bf5e634c046e5980b977ee4bdfb895021cbe2a212a8998c92699eac921096`。共同 E5 snapshot SHA256 為 `2f9c85f05c5c83f285b9900171979a05f42ef7bc8938f27bd8d0b32bc77cc734`。

late 額外解凍 4,870,912 個參數，layer8／9／10／22 首步相對權重變化依序約 1.47e-6／9.17e-7／1.33e-7／4.66e-7，均通過非零且小於 0.001 的初步門檻。optimizer 舊群 step4626、新群 step1、EMA4626、criterion updates5 等斷言全部通過；固定參數／非 head BN state 檢查通過。證據為 `studies/pre-fusion-full35-b100/artifacts/scope-proof-v1.json`。校準無 OOM 或其他新增困難。

13:59:18 啟動正式 `a0-scope-narrow-v1` E6–E8，queue 接續 `a0-scope-late-v1`；monitor session35337，每次 shell wait 最多 600 秒，正常不讀 log 或額外查 GPU。正式 AP 尚待完成事件；尚無通過驗收的 winner。
