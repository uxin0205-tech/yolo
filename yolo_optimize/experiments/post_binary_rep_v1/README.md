# BinaryQK 重訓完成後的 RepConv 三組

## 最新有效順序

依使用者最新要求，本資料夾取代舊的「RepConv 各自從原生 QK 的 MASF B5 開始」排程。原資料夾不刪除，已完成的 scale/bias 結果仍是本次上游。

```text
原生 QK E2
   ↓ Pose MASF 專項 5E
MASF B5（Detect／Pose 各自 P3 MASF）
   ↓ 換成修復梯度的 BinaryQK，再做 scale／bias／Attention／head 適應 5E
BinaryQK scale_bias E5（共同固定起點）
   ├─ Rep17：layer17 3×3 → RepConv
   ├─ Rep20：layer20 3×3 → RepConv
   └─ Rep17＋20：兩層同時替換

GPU 順序：舊 Rep17 在完整回合存檔／驗證後收束
          → 新 Rep17 smoke／5E／驗證
          → 新 Rep20 smoke／5E／驗證
          → 新 Rep17＋20 smoke／5E／驗證／三組報告
```

三組從同一 BinaryQK E5 開始，不把 Rep17 訓練結果再接 Rep20，避免混淆單層與雙層的作用。保留舊原生 QK 結果作補充，不追加未改架構加訓、不自動選為正式模型。

## scale／bias 那組到底改了什麼

B5 本身是**原生浮點 QK＋PWL**，不是已開 BinaryQK。因此這不是「只調 B5 的 scale/bias」，而是切換 Attention score 算法後的適應訓練。

| 部分 | MASF B5 | scale_bias E5 |
| --- | --- | --- |
| Q/K 相似度 | 原生 QᵀK/√d | sign Q/K 與 Hadamard 後 sign Q/K，兩條 binary dot 相加 |
| 反向 | 浮點點積 | 前向 exact XNOR/popcount，反向 dot surrogate＋clipped sign STE |
| scale | 1/√d 常數 | 每 site 4 heads×2 basis；兩 site 共 16 個可學常數 |
| bias | 原生該 score 無這組相對 bias | x/y 分解相對位移表，signed16 m/1024 |
| PWL | [-10,0]、20 段 | 保留 |
| MASF／qSiLU | Detect／Pose P3 MASF、qSiLU | 保留；MASF 固定 |

令 Z0 為 sign Q/K 的 signed binary dot，Z1 為 Hadamard 變換後的 signed binary dot，單一 head 的分數是：

```text
score = (m0 × Z0 + m1 × Z1 + bx[Δx] + by[Δy]) / 1024
m0、m1：固定 unsigned 整數，1～1024
bx、by：固定 signed16 整數，依相對位置索引
score → row-max／clamp → PWL[-10,0] → 正規化 → 與 V 加權
```

每張圖片使用同一批常數，不重新估計 scale，不需要逐圖選 8 個 scale；16 個值是兩個 site 各 4 heads、各兩個 basis 的係數。E5 的實際整數碼如下，每格除以 1024 就是尺度：

| site | head0 (m0,m1) | head1 | head2 | head3 |
| --- | --- | --- | --- | --- |
| layer10 attention | (173,188) | (191,186) | (206,214) | (237,237) |
| layer22 attention | (129,204) | (233,242) | (228,242) | (192,220) |

例如 layer10 head0 使用 173/1024 與 188/1024。訓練保留 float master，用 STE 讓前向採量化常數而反向可更新；推論只需固定整數乘加／位移。這不等於單一 power-of-two shift，也不是已完成板端 latency／energy 驗證。bias 必須依 key 的相對位置不同；整排相同的常數 bias 會在 row-max／softmax 中抵消。

原生 B5 的 QKV／PE／輸出投影權重保留後轉入相容 Attention；新 scale 與相對 bias 初始化參考歷史已選 BinaryQK，再由本輪學習。Q/K 原先經 bool/popcount 的梯度中斷，這次 score 前向與代理反向分開，並以真實雙任務 smoke 檢查兩個 site 的 Q、K、scale、x/y bias 各自梯度與參數更新。非零梯度只證明路徑可訓練，不保證 AP 會提高。

訓練 5 epoch、warmup 1；AdamW，head LR 5e-6、Attention／bias 1e-5、scale 2e-4。開放 Attention（不只是 scale/bias）與兩個 head，固定其餘共享參數、MASF 及 BN running。Detect physical16×16 microbatches、Pose physical16，每個 macro 權重 Detect1／Pose0.25；COCO 全量為 epoch 主時鐘。不是 Pose-only batch128。

[上游完整結果](<../post_masf_hardware_v1/artifacts/scale_bias/train/RESULTS.md>)：COCO +0.3034 pp、person +0.1582 pp，但 bat Pose -0.5228 pp，未過全面保護門檻。依最新授權可作後續研究起點，**不代表已經認定它全面勝出**。比較同時包含換結構與訓練，不能解釋成 scale/bias 的獨立因果效果。

## RepConv 結構與硬體目的

```text
layer16 p3_raw ──┬─> Detect P3 MASF → Detect P3
                ├─> Pose P3 MASF   → Pose P3
                └─> layer17 [Conv 或 RepConv] /2
                     → concat(layer13) → layer19 p4_raw
                          ├─> Detect／Pose P4
                          └─> layer20 [Conv 或 RepConv] /2
                               → concat(layer10) → layer22 p5_raw
                                    └─> Detect／Pose P5

訓練 RepConv：Act(BN3(Conv3(x)) + BN1(Conv1(x)))
部署：        Act(合併後單一 3×3 Conv(x))
```

挑 layer17／20 因為它們是 P3→P4、P4→P5 的獨立 stride2 3×3 下採樣 Conv，可局部重參數化，不新增 Detect head、不全面替换 backbone。它們不直接增強 P3 head 輸入，不能保證 ball 增準。stride2 不加 identity 分支。

640 輸入，Rep17 增加訓練參數 66,048、Rep20 增加 263,168；双層共 329,216。各增加訓練 Conv MAC 0.1048576G，雙層 0.2097152G；fold 後額外 Conv MAC 為 0。實際 latency／peak memory 不由此推定。

## 新三組訓練與驗證

沿用上游的 5 epoch、warmup 1 與雙任務資料配置。Rep LR 1e-5、head LR 5e-6，固定所有 Attention 參數（包括新 scale/bias）、兩套 MASF、其餘共享參數與 BN running。Attention 參數固定不等於切斷其對上游 RepConv 的代理反向傳播。

[CPU 檢查](<artifacts/preflight-v1.json>)：三組皆 strict 載入 E5；參數與整數碼一致；初始輸出與 E5 等價；非零 Rep 分支仍可 fold；Float／BitTrue 模板均保留已訓 BinaryQK。GPU smoke 會確認每個新支路有非零梯度且確實更新，接著才訓練。

逐回合完整 checkpoint 先保存，再跑 COCO val5000 與 BBAT5 v1 val683，最後 E5 獨立重載 BitTrue／Float。報告同時列 MASF B5、BinaryQK E5、新 Rep E5，不掩蓋上游 bat 下降。參考採用門檻同時保護兩個起點的八項 AP，並要求相對 BinaryQK 起點有最小改善；即使通過也不自動替換正式模型。

## 來源、排程與保存

起點：`../post_masf_hardware_v1/artifacts/scale_bias/train/epochs/e5/inference.pt`。SHA-256：`c8dcaf439b68c77d45a4f213972e57d51912028b558418511badee55fafbf681`。

- [config.json](<config.json>)、[queue-plan.json](<queue-plan.json>)：設定與來源 hash。
- [transition.json](<transition.json>)：取消舊排程、保留舊回合的依據。
- `artifacts/queue-v1/state.json`：新 queue 即時狀態；`transition-events.jsonl`：舊回合收束事件。
- `artifacts/<arm>/train/`：各組完整結果與權重。
- `RESULTS.md`：全部完成後自動彙整三組。

舊回合收束期間每 60 秒只看存檔／驗證邊界；新 GPU 正常工作維持 600 秒監測，不週期讀 log。錯誤保留產物並停止相依工作，供主代理修復；背景程式不能自行推理或保證喚醒已結束對話的模型。

[scale／bias 硬體核對：固定常數、實際運算與尚未完成的定點邊界](<SCALE-BIAS-HARDWARE.md>)。

## 2026-09-14：全部完成

三組已完成，queue 於台北時間 04:49:57 正常結束。[完整狀態、八項 AP、成本邊界與權重入口](<STATUS-20260914.md>)。三組均未通過採用閘，不自動替換正式模型，不追加訓練。
