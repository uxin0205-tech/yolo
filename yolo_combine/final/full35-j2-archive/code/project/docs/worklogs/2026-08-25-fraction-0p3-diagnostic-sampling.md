# 2026-08-25 `fraction=0.3` 診斷抽樣

## 目的與決策

使用者授權「能先用隨機 `fraction=0.3` 驗證的部分就使用」。本次將它限制為訓練前期
的非正式診斷工具，不能改變 canonical `bbat5-v1`、正式驗證口徑或 preflight gate。

| 工作 | 是否可用0.3 | 原因 |
| --- | --- | --- |
| Pose trainer／loss／checkpoint／短期收斂方向 | 可以 | 可縮短每 epoch 的 train exposure；仍須標為 diagnostic |
| 診斷模型的 Pose AP/mAP | train用0.3、val用1.0 | 允許看趨勢，但不得作正式 baseline |
| 正式 P1→P2→P3 | 不可以 | P3必須由5,964張canonical train完成 |
| 八項 standalone baseline與0.08 hard gate | 不可以 | 必須同口徑、全量 validation |
| COCO／BBAT5 full validation | 不可以 | 抽樣 validation 會改變 AP/mAP 定義與不確定性 |
| batch128 VRAM gate | 不可以 | fraction只減少epoch長度，不減少單一physical batch峰值 |
| 正式 JOINT preflight | 不可以 | 仍須 canonical P3與八項 baseline |

## 為何沒有直接使用 Ultralytics 原生 `fraction`

唯讀查核 Ultralytics 8.4.90 的 `BaseDataset.get_img_files()` 後確認：它先排序 image
paths，再取 `im_files[: round(len(im_files) * fraction)]`。因此原生 `fraction=0.3` 是檔名
前綴，不是隨機抽樣。BBAT5 的 `.rf.` brightness siblings 與來源命名存在結構，直接取前綴
可能使來源與 ball/bat 比例偏斜。

本次改用一個集中式 `DiagnosticSamplingPolicy`：

1. seed固定為0。
2. 以 `.rf.` 前的 source group 為抽樣單位，避免同來源的亮度版本被拆開。
3. 先隨機抽30%的groups，再把選定清單排序並寫入manifest；之後每次載入都使用同一份
   清單，不在每個epoch重抽。
4. Ultralytics端改傳 `trainer_fraction=1.0`，避免對已選清單再做一次sorted-prefix截斷。
5. train與val的影像／標註只建立symlink，cache寫在診斷View，不碰canonical source或正式
   runtime View的cache。
6. validation固定使用canonical 683張全量資料。

## 實作內容

- 新增 `src/yolo_combine/diagnostic_sampling.py`：
  - 固定seed、source-group抽樣。
  - 建立可重建的symlink-only train／val View。
  - 保存selected list SHA256、source runtime manifest SHA256、影像／group／class／empty
    counts。
  - 建立 `diagnostic-run.json` marker與checkpoint marker discovery。
- 更新 Pose CLI：任何明確的 `--fraction < 1` 都走隔離的
  `artifacts/pose/diagnostic/`；JSON輸出同時記錄requested fraction、真正傳給trainer的
  fraction、validation fraction與formal eligibility。
- 正式 fraction=1.0 Pose transition若收到診斷 checkpoint會fail-fast。
- JOINT preflight若設定到帶 marker 的診斷 Pose checkpoint，也會增加正式 blocker。
- Full35／Partial75設定、README與runbook已同步。Partial75只有程式能力，沒有materialize
  View或執行，符合「使用者要求時才跑」的決策。

## Full35 seed0 真實抽樣結果

來源：canonical `bbat5-v1` formal train/val，未重切、未改標註。

| 項目 | 全量 train | 30% diagnostic |
| --- | ---: | ---: |
| source groups | 2,320 | 696（30.000%） |
| images | 5,964 | 1,777（29.795%） |
| instances | 8,134 | 2,448 |
| ball | 3,220 | 991 |
| bat | 4,914 | 1,457 |
| empty labels | 38 | 9 |

全量ball instance比例為39.586%，診斷View為40.482%，差0.896個百分點。validation仍為
683/683。固定選樣SHA256：

```text
edebcbb5475dd4835540becfc228237ee783bbb1532cf71f62665ef3b1e1e8cc
```

正式產物：

```text
variants/full35/artifacts/datasets/diagnostic/pose/f0p3-seed0/
  manifest.json
  pose-diagnostic.yaml
  train-selected.txt
  train/{images,labels}/     # 1,777 + 1,777 symlinks
  val/{images,labels}/       # 683 + 683 symlinks
  train/labels.cache
  val/labels.cache
```

未複製任何影像或標註；共4,920個data symlinks。`find ... -type f` 對train/val資料目錄
結果為0。

## 驗證方式與結果

全程設定 `CUDA_VISIBLE_DEVICES=-1`，沒有建立模型、沒有啟動訓練、沒有配置GPU。

1. 單元測試：

   ```text
   tests/test_diagnostic_sampling.py
   tests/test_joint_config.py
   tests/test_pose_stages.py
   12 passed in 0.07s
   ```

   已涵蓋固定seed、group不可拆、idempotence、policy drift fail-closed、marker discovery、
   preflight拒絕與fraction邊界。

   完整repository測試另以`nice -n 19`、idle I/O、CPU 22–23與CUDA隱藏執行：

   ```text
   99 passed, 3 skipped in 12.40s
   ```

   3項skip都是既有的CUDA-only integration tests。

2. Ultralytics 8.4.90真實train loader：成功載入1,777張、9 backgrounds、0 corrupt；
   `nc=2`、`kpt_shape=[2,3]`，抽樣sample的visibility只包含0與2。
3. 真實validation loader：成功載入683張、0 corrupt，證實診斷YAML沒有對val套0.3。
4. cache分別建立在診斷View的`train/labels.cache`與`val/labels.cache`，未覆寫formal
   `bbat5-v1-runtime` cache。
5. 執行前主機有約99 GiB available RAM；此工作只增加symlink、cache與小型metadata。
6. 測試後GPU compute process仍只有既有MambaPose PID 3170826（21,280 MiB），本專案沒有
   CUDA process；available RAM約99 GiB。
7. `variants/full35/run.py pose --help`已確認會明示fraction<1是固定seed、group-aware、
   full validation且不得作正式證據。
8. 最終重跑`variants/full35/joint.py preflight`仍只列出原本兩個blocker：canonical P3
   checkpoint與同口徑八項baseline；30%功能沒有放寬或額外污染正式設定。

## 使用方式

GPU空閒後的最小短跑：

```bash
.venv/bin/python variants/full35/run.py pose --stage p1 --device 0 \
  --fraction 0.3 --epochs 2
```

這仍使用stage指定的physical batch；若batch128本身OOM，fraction0.3不會解決。若要串接
診斷P2/P3，後續stage也必須明確保留`--fraction 0.3`。正式鏈必須移除fraction override，
從canonical全量P1重新開始。

## 困難與解法

1. 困難：內建patch wrapper建立network namespace時發生
   `bwrap: loopback: Failed RTM_NEWADDR`，無法更新既有檔案。
   解法：新增檔案仍使用內建`apply_patch`；更新既有檔案改呼叫同一個workspace
   `apply_patch` executable，仍以逐段patch套用，沒有用腳本整檔覆寫。
2. 困難：最初的manifest-only設計會讓30% loader與100% loader共用
   `bbat5-v1-runtime/train/labels.cache`，造成往後互相重建。
   解法：改成symlink-only diagnostic runtime View，train/val cache完全隔離。先前兩版
   metadata已移至`/tmp/yolo-combine-f0p3-seed0-pre-stats`與
   `/tmp/yolo-combine-f0p3-seed0-manifest-only`作可恢復備份；它們都不含影像或標註。

## 未解事項與風險

- 本次沒有GPU，因此沒有執行30% Pose training，也沒有新增AP/mAP；真正數值要等GPU空閒。
- 診斷AP/mAP雖使用完整val，train exposure仍只有約30%，不能與正式P3或standalone
  baseline互換。
- 目前沒有把0.3接進正式JOINT session。其preflight尚缺canonical P3與八項baseline，且
  JOINT需要對COCO與BBAT5分別定義、記錄抽樣policy；在沒有這兩個正式輸入前直接加入會
  混淆實驗契約。正式前置完成後若需要JOINT短跑，應另開明確的diagnostic entrypoint。
- physical batch128仍須在GPU空閒時單獨profile；system RAM充足不代表VRAM可行。

除上述已記錄事項外，無其他未解困難。
