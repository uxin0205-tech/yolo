# V36 Full-Coverage qSiLU Weight Search

## 目的

承接已通過雙指標門檻的 V35 qSiLU+A8 mixed checkpoint，對 Full35 的 148 個 deployment Conv/Linear weight path 做逐層、逐格式分析與分階段 PTQ。每個 path 最終至少使用 W8；特殊格式與非二次冪 uniform bits 只在 CPU sensitivity 與 GPU task metrics 都支持時逐層升級。

## 固定契約

- activation 固定 V35 的 qSiLU+A8；不重新搜尋 activation，也不啟用已排除的 poly_quality。
- parent 是 V35 QAT epoch 3 的 hash-pinned deployment/full-resume pair；舊 V30 archive 不在新 queue 血緣內。
- canonical COCO 與 BBAT5 v1 不切分、不抽樣、不改 assignment；validation 同時保留 COCO Person、COCO80、BBAT5 detect 與 pose 的 mAP50、mAP50-95 八項指標。
- 總 mAP50 最差下降上限 0.015；mAP50-95 最差下降上限 0.04。CPU NRMSE 只排序，不直接宣稱 task accuracy。
- 全部 148 paths 以十個 region defaults W8 覆蓋；path route 可逐層指定 LS-SD4、三元、W6/W5/W7/W4。

## 執行順序

1. CPU-only profile：deployment view 的 W8/W7/W6/W5/W4、optimal Fixed-SD4、exact-scaled ternary、TWN-v3 filterwise、Paper-TWN v2，共 148×9。
2. Special PTQ：保留 V35 的 11-path policy，按 backbone→neck→head 各自測試 CPU 排名前段的 LS-SD4 與三元 cohort；只有 dual gate green/recover 才可鎖定。
3. Uniform PTQ：在已鎖定 special paths 之外，按相同順序比較 W6、W5、W7、W4；W8 是 fallback，所有候選仍是完整 148-path policy。
4. Final PTQ：把每區選出的 route 合併成一個完整 policy，再做一次八指標驗證。
5. Continuation QAT：從 V35 full-resume checkpoint warm-start，reset 新 path quantizer，最多 3 epochs、patience 5、AdamW 0.1× J3 role LR、scale-only epoch 0、epoch 1 起 full fake quant、batch 128/microbatch 16、pose 16、無新增雜訊。此階段不重跑 baseline sham，使用 V35 external sham/reference 並在報告中明確標示 unpaired continuation。

## 失敗與續跑

每一階段以 immutable YAML、SHA-256、JSON report 與 `execution-status.json` 留痕。queue 在既有 report 上 resume，不覆寫 drifted artifact；GPU 被其他 process 使用時每 600 秒等待。任何 candidate error 會停止 queue 並寫入 error，不會自行跳過失敗證據。
