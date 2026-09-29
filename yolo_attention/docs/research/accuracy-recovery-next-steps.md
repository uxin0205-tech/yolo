# 準確率恢復候選：保留但暫不執行

日期：2026-08-15

這份筆記保存目前討論過的下一輪準確率恢復方法。它不是已核准 workflow，也沒有啟動 GPU、建立 queue jobs 或宣稱實驗有效；正式執行前必須再由使用者決定並更新 `plan.md`。

## 現有證據

- B26-FP：mAP50-95 0.517998。
- W-DIR：0.507457，26/40 early stop，優於 W-PROG 的 0.502866。
- Dynamic scale 比 PoT scale 高約 0.000506。
- Dense 2D bias 比 decomposed 2D bias 高約 0.000219，但後者位於 tie band 且硬體介面較簡單。
- N0-SHIFT 為 0.506746，高於 N1-SHIFT 的 0.506357；目前 PMP recovery 沒有改善 normalization。
- D1-SHARED seed 0/1 相差 0.025084，codebook-only recovery 尚不穩定。

## 建議優先順序

1. 從 W-DIR best checkpoint 以較低 learning rate 做 10–20 epochs full-model recovery，增加 patience，而不是直接無條件增加原 schedule。
2. Recovery 期間先保留 Hadamard MDB、dynamic scale、exact Softmax 與 dense 2D bias；收斂後才做 PoT scale、SHIFT normalization 與 decomposed bias 的 accuracy/cost gate。
3. 保留 N0 normalization parent 參與 final selection，不強迫使用目前會掉點的 N1 PMP child。
4. 比較固定零 threshold 與 per-Attention/per-head learnable Q/K threshold：

   $$
   Q_b=\operatorname{sign}(Q-\tau_Q),\qquad
   K_b=\operatorname{sign}(K-\tau_K).
   $$

5. 若允許額外 training regularizer，再獨立 ablate bit-balance 或 score-ranking consistency；不可把這類新 trick 的收益誤算成 Hadamard MDB 本身。
6. 暫不把 D1 learned codebook 當主要準確率恢復路徑，先處理 seed stability。

## 建議的最小下一輪

```text
現有 W-DIR best
  ↓
低 LR full-model recovery（10–20 ep）
  ↓
Dense 2D bias 聯合訓練
  ↓
zero threshold vs per-head threshold
  ↓
選 accuracy parent
  ↓
PoT scale / SHIFT / decomposed bias zero-train hardware gate
```

Optional quantization 仍位於 A-FINAL 之後，這份筆記不啟動 PTQ、QAT、ternary、SD4 或 INT5/6/8。
