# 2026-09-09：HOG／MASF 加看 COCO ball／bat

## 使用者要求與範圍

使用者新增要求：HOG 與 MASF 可以查看 BAT／BALL。本次依正在進行的融合前 COCO2017 研究解讀，加入 `sports ball` 與 `baseball bat` 的輔助分析；不自動擴回融合後 Detect／Pose 或 BBAT5 訓練。COCO 的 sports ball 包含多種運動球，不等於棒球專用指標。

overall／person 仍是主要選模與安全停止指標，ball／bat 先作觀察項，不自行恢復先前已取消的硬性 gate；不因少數類別的波動中斷正在執行的 HOG。若使用者之後另指定優先級，再明確更新門檻，而非事後挑有利類別。

## 已完成的唯讀核對

目前 `validate()` 已逐 epoch 保存四項 internal AP50–95，因此無須中斷、改訓練程式或重跑 HOG。此時只讀已完成的共同起點及原生對照，沒有讀執行中的 HOG log。

| 權重 | overall | person | sports ball | baseball bat |
| --- | ---: | ---: | ---: | ---: |
| late E8 共同起點 | 0.508153635 | 0.627520517 | 0.513563550 | 0.480437770 |
| 原生對照 E5 | 0.506643659 | 0.628015206 | 0.512091952 | 0.474966430 |

原生 E5 相對起點 ball -0.001471598、bat -0.005471340。bat 在 E1–E5 約 0.47027–0.48436，存在明顯跨回合波動；這不是 HOG 的結果，不能歸因 HOG。必須等 HOG 完成後，按照共同 epoch／共同選模規則比較，不能各挑不同回合的最好單類 AP 再拼成一組模型成績。

## 後續分析

HOG 完成後一起列 overall／person／ball／bat 與起點、同回合原生對照的差值。若 ball／bat 有值得追查的差異，優先檢查小球／細長棒在 stride8 hard cell-center mask 下的有效監督覆蓋，以及固定案例中的漏檢與框定位；不用單一 AP 直接宣稱形狀先驗有效。

MASF 後續比較也保留這兩類觀察，分清直接改動 P3 Detect 分支的效果與 shared P3 影響 P4／P5 的效果。只在證據支持時追加必要驗證，不另建 person-only 或棒球 split。

## 驗證、困難與未解事項

四項指標已從完成的 JSON 核對，來源為 `a0-scope-late-v1/summary.json` 與 `prefusion-hog-control-v1/summary.json`。本次沒有修改模型或訓練超參數，沒有新增困難。HOG 正式結果與 MASF 實驗尚待完成；維持既有 queue 與每次最多600秒的 shell 事件監測。
