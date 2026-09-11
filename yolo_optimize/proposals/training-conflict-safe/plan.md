# Detect 優先衝突安全訓練：最小實驗計畫

> 2026-09-08 本輪維持COCO80+BBAT5，person-only不是前置；同一stage負cosine≥20%且correction median≥0.02才開；與HOG不疊，parent/順序以總計畫為準。詳見[整合總計畫](<../integrated-roadmap/plan.md>)。

本計畫只回答一個問題：在相同 person Detect＋BBAT5 Pose joint recipe 下，移除 Pose 對 Detect 的當次
反向 shared-gradient 分量，能否補回 person／joint 精度而不傷 Pose？背景證據見[方向說明](<README.md>)，
公式與完整資料流見[架構圖報告](<architecture-report.md>)。

## 一、在整體流程的位置

~~~text
COCO person task view + H1/H2 head 決選
                    │
                    ▼
凍結 standalone person Detect winner 與 standalone Pose parent
                    │
                    ▼
重新跑少量 macro-step 的共享梯度 screen
                    │
         ┌──────────┴──────────┐
         │衝突訊號太弱         │衝突仍存在
         ▼                     ▼
   本方向不啟動          G0-MATCH / G1-APC-DETECT
                                 │
                                 ▼
                         凍結最終 BASE-FP winner
                                 │
                                 ▼
                   MASF／RepConv／BinaryQK／PTQ／export
~~~

它不是 final checkpoint 後的 post-process。projection 會改 training updates，所以必須在 FP 架構與任務
定義確定後重跑 joint training；任何後續量化／calibration都要使用本方向決選後的 winner。

## 二、唯一可接受的 parent 與資料

- 兩臂使用相同、immutable 的 person Detect parent與 Pose parent，記 SHA-256、程式 commit、config digest。
- COCO person Runtime Dataset View 必須沿用原 COCO2017 train／val image manifests，只保留 class 0 labels，
  並保留所有 person-negative images。
- BBAT5 Pose只使用不可變的 `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`；不得重切、
  抽樣、修改影像或標註。
- 若最後決定不做 person-only head，允許沿用 COCO80，但所有結果名稱必須是 Detect-priority，不得冒稱
  person-gradient projection。

## 三、Phase 0：先確認值得開完整訓練

在最終 head 與新 baseline 程式上，固定取少量、可重播的 J1/J2/J3 macro steps，只記原生 `g_detect`、
`g_pose`：

| 訊號 | 啟動條件 |
|---|---|
| 負 cosine event rate | 任一 active joint stage `>=20%` |
| 負事件 correction ratio | 中位數 `>=0.02`，其中 `correction_ratio≈|cosine|` |
| shared scope | ownership manifest、numel與 stage policy 完整一致 |
| finite safety | loss、grad、AMP unscale後 buffer全部 finite |

若所有 stage 都低於任一衝突門檻，本方向不啟動：新 person head 已改變訓練幾何，歷史 Full35 的
32–41% 不能拿來強迫開實驗。門檻只用 training diagnostics，不用 validation挑 projection設定。

## 四、Phase 1：工程 gate

正式插入點是現行
[`MacroStepEngine.run()`](<../../../yolo_combine/src/yolo_combine/joint_loss.py>)中兩任務 backward
完成、`scaler.unscale_(optimizer)`與 finite check之後，`clip_grad_norm_()`之前。

需要的最小實作：

1. config feature flag，預設 `false`；關閉時走原生 path。
2. 每 stage 由 parameter ownership 建立 deterministic shared manifest，不以「剛好有 grad」猜 shared scope。
3. 保存 Detect 的 scaled snapshot；unscale後以 FP32 buffer還原 `g_d`，由 joint gradient減出 `g_p`。
4. 跨完整 shared vector計一個 global dot與 `||g_d||²`；首輪不能改成 per-layer projection。
5. `dot<0` 才回寫 `g_d+g_p_safe`；task-specific head gradients完全不回寫。
6. projection後沿用一次 global clip、一次 optimizer step、一次 scaler update與一次 EMA update。

先通過下列測試，才可啟動 GPU training：

- 合成向量：負 dot、正 dot、正交、`||g_d||=0`、missing gradient、NaN、Inf。
- 負 dot 時 `g_d` bitwise／tolerance不變，`g_d·g_p_safe≈0`；正 dot 時整個 update與 baseline等價。
- Detect／Pose head每個 tensor的 gradient，在 feature flag on/off與正／負 dot case都不被投影。
- AMP overflow時整個 macro-step一起 skip；不能殘留單一 task gradient或錯誤 scaler state。
- `projection_enabled=false` 的 loss、clip input、optimizer state與 parameter update符合既有 deterministic tolerance。
- save/resume、EMA、early-stop deep-copy、Float／Bit-True materialization與 export schema不變。

任一 gate失敗即修實作，不開始正式 training。

## 五、Phase 2：唯一首輪矩陣

| Arm | Joint recipe | Active shared update | 其他新機制 | 新 jobs |
|---|---|---|---|---:|
| `G0-MATCH` | 與決選後 baseline完全相同 | `g_d+g_p` | 無 | 1 |
| `G1-APC-DETECT` | 完全相同 | `g_d+g_p_safe` | 無 | 1 |

兩臂必須相同：parent、seed、sampler manifests與順序、physical／logical batch、gradient accumulation、
task weights、optimizer、LR、scheduler、warmup、stage length、patience、BN、augmentation、AMP、clip norm、
validation與 selector。現有歷史 J3不能取代 `G0-MATCH`。

每個 macro-step或固定 cadence新增：

~~~text
pre_cosine
post_cosine
projection_triggered
projection_rate_by_stage
detect_shared_norm
pose_shared_norm
correction_ratio = ||g_p_safe - g_p|| / (||g_p|| + epsilon)
missing_gradient_count
amp_overflow_retries
projection_wall_time_ms
peak_training_vram
~~~

## 六、seed-0 gate與停止條件

`G1 - G0` 必須同時符合：

- canonical COCO person AP與 joint score都 `>= +0.001`。
- person AP、ball pose與其餘既有 BBAT5 metrics任一項都不得低於 G0 超過 `0.001`。
- person AP或 ball pose至少一項改善 `>= +0.002`，否則只算幾何變漂亮，沒有目標效益。
- 所有觸發事件的 post dot在事前 tolerance內接近 0；非觸發事件與 G0等價。
- Float／Bit-True差、NaN/Inf、資料 digest、checkpoint lineage全部過 gate。
- training wall time與 peak VRAM增量在實作前登錄的預算內；不得偷偷降 batch換取通過。

任一失敗：標為 `rejected`，停止本方向。不調 projection strength、不改 per-layer、不加 GradNorm、teacher、
HOG或新 LR來救同一 arm。

只有 seed 0全過，才補 seeds 1、2的 paired `G0/G1`；以三組 paired mean/std套用相同 gate。三 seeds前最多
只能標 `provisional`，不能標 `validated`。

## 七、失敗後唯一允許的下一個判斷

若數值 probe證明 post conflict已消失，但 person／Pose遺忘仍在，先結束本方向，再另立
`Lineage-Preserving Anchor Bridge`：用 frozen standalone Detect/Pose teachers做短暫 object-masked feature
anchor。它不能和 G1疊加成第三 arm。

Sobel/HOG companion loss也永遠是另一個方向；若未來測，先比較相同 temporary side head的 luminance control
與 edge target，且 export前完全裁除。本計畫不執行它。

## 八、結果契約

每個 arm保存：

- parent、data、code、config與 shared manifest digests。
- seed、sampler順序、optimizer-step數、trainable names、AMP與 BN policy。
- best／last checkpoint digests與完整 resume state。
- canonical person AP50-95／AP50／AP75／AP_S/M/L、AR、negative-image false positives。
- BBAT5 ball／bat box與pose全部正式指標、joint score、Float／Bit-True parity。
- projection diagnostics、wall time、peak VRAM與失敗理由。

摘要寫入本方向未來的 `results.md`；大型 checkpoint與原始 artifacts留在正式實驗目錄。

返回[方向說明](<README.md>)或[優化方向索引](<../README.md>)。
