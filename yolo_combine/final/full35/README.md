# Full35 final 交付包

依固定實驗selector，目前預設是已通過八項 hard gate、且joint score超越J2的
**J3 best_joint**（Bit-True `0.711174738975`）。
實際部署權重尚待使用者看完COCO person AP後選定；所有候選與J2 rollback均保留，
本次沒有因報告更新而更換任何權重。

## 最直接的檔案

- 部署／推論：`weights/combined/inference/best_joint.pt`
- 從最佳點續訓：`weights/combined/full-resume/best_joint.pt`
- J3 結束點／exact state：`weights/combined/full-resume/last.pt`
- 完整程式碼快照：`code/project/`
- Full35／XNOR 必要來源：`source_bundle/`
- Float 與 Bit-True 正式 AP：`outputs/validation/accepted-best-joint/`
- 完整訓練 CSV／JSONL／圖：`outputs/training/logs/`
- 機器可讀狀態：`RELEASE_STATUS.json`
- 完整中文結論、person AP候選CSV與八張圖：`analysis/FINAL_ANALYSIS.md`
- 全檔案 SHA256：`MANIFEST.json`、`CHECKSUMS.sha256`

## 環境

必須使用 Python 3.12、PyTorch 2.11.0+cu128、Ultralytics 8.4.90；不要自行升級。`source_bundle` 是從原 721MB 研究 bundle 裁出的 Full35 必要子集，保留 Float／Bit-True A2 權重與完整 Python 模組，不含 Partial75 或其他淘汰候選。

## 驗證完整性

```bash
python verify.py
```

## 推論

```bash
python run.py infer \
  --checkpoint weights/combined/inference/best_joint.pt \
  --source /path/to/image.jpg \
  --task both --device 0 \
  --output-json outputs/inference.json
```

`--task` 可用 `detect`、`pose`、`both`。`both` 只抽取一次 shared backbone/neck feature，再分別執行 Detect 與 Pose26 head。

## 正式驗證

```bash
python run.py validate \
  --checkpoint weights/combined/inference/best_joint.pt \
  --backend both --device 0 --name final-recheck
```

驗證依賴本機 canonical COCO2017 與 BBAT5 v1；資料本體刻意不複製進 final。固定路徑與 YAML snapshot 在 `configs/`，正式 BBAT5 仍只能是 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`。

## 續訓與回退

J3已完成並依patience5正常停止：

- `combined/full-resume/best_joint.pt`：global epoch58的正式最佳點；
- `combined/full-resume/last.pt`：global epoch63的stage-complete終點；
- `rollback/j2/`：升格前J2四種selector的full-resume與inference版本。

`combined/*/best_detect.pt`明確沿用J2，因J3沒有刷新Detect selector。若後續跑seed1、
MuSGD或其他tuning，必須另開run；stage切換不是exact resume。

J3使用physical32×4維持logical Detect128；不可把它稱為單次physical batch128。

## 權重角色

- `combined/inference/`：只有 EMA/live state 與 model contract，不可 exact resume。
- `combined/full-resume/`：含 live/EMA、optimizer、scheduler、AMP scaler、RNG、loader、criterion progressive state。
- `rollback/j2/`：升格前J2完整回退權重。
- `standalone/`：Detect A2 與 Pose P3 回退／比較來源，不是融合推論入口。

## 目前正式結果

- shared 參數：26,529,701；兩個獨立模型合計 45,580,762；減少 41.796%。
- Bit-True COCO overall mAP50-95：0.498022
- Bit-True COCO person AP50-95：0.620381
- Bit-True BBAT box mAP50-95：0.630036
- Bit-True BBAT pose mAP50-95：0.903717
- 八項 gate：全部通過；最差 delta 為 ball pose -0.016354，仍高於 -0.08 下限。

## COCO person AP候選

| 候選 | person AP50-95 | AP50 | AP75 | precision | recall | 八項gate |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| Standalone Detect | 0.626186 | 0.841473 | 0.684441 | 0.795257 | 0.769484 | 不適用 |
| Shared best_detect | 0.626057 | 0.841464 | 0.684228 | 0.797245 | 0.768767 | 未通過 |
| J2 best_joint | 0.618511 | 0.837604 | 0.676728 | 0.799476 | 0.765334 | 通過 |
| J3 best_joint | 0.620381 | 0.839410 | 0.679218 | 0.805898 | 0.763199 | 通過 |
| J3 best_pose | 0.620513 | 0.839392 | 0.678979 | 0.804254 | 0.765241 | 通過 |
| J3 last | 0.620398 | 0.839157 | 0.678712 | 0.806627 | 0.762921 | 通過 |

`Standalone Detect`只有Detect任務；`Shared best_detect`的Pose尚未適應完成，因此不能作
`task=both`正式答案。兩個head都要時，目前預設仍是`J3 best_joint`，但`J3 best_pose`
的person AP50-95與recall略高。差距很小且目前只有seed0，最後部署選擇保留給使用者。
完整數值、joint score、epoch、provenance與兩張person專圖見`analysis/FINAL_ANALYSIS.md`。

## 邊界與風險

- 目前是 seed0 正式結果；第二 seed 尚未執行，因此跨 seed 結論仍屬 provisional。
- J3已依實驗selector升格；實際部署checkpoint仍待使用者依person AP最後決定。
- 尚未做 FPGA、HLS、真實 latency／energy 驗收。
- Ultralytics checkpoint 及衍生部署的 AGPL／商用授權需在非研究交付前另行確認。
