# 2026-08-26 Full35 JOINT v2 保守重訓

## 目標

保留既有 `full35-joint-adamw-seed0` 全部checkpoint、validation、log與圖表，在不同
run目錄建立新版Full35融合訓練。新版先讓Pose head適應Detect初始化的shared trunk，
再逐步解凍neck、後段backbone、MASF；attention延後到可選J3。Partial75使用同一套
程式與stage政策，但維持獨立設定／artifacts且不在本輪啟動。

## 變更原因

舊run的24次J1 gradient抽樣顯示Pose／Detect shared-gradient norm平均比約19.69倍，
平均cosine約0.0498，24次中7次為負。這表示主要問題不是長期正面衝突，而是Pose梯度
量級壓過Detect。舊J1同時開neck、MASF與兩heads，且neck/head峰值LR分別為2e-4／
1e-3；舊J2又提高trainable scope。COCO精度在warmup接近峰值LR時明顯惡化、LR下降後
部分恢復，符合overshoot而非不可逆結構破壞。

舊run在停止前已完整保存global epoch 13：

- `checkpoints/last.pt`：396,715,549 bytes，mtime 2026-08-26 16:11:14 CST。
- `checkpoints/best_joint.pt`：396,740,443 bytes，mtime 2026-08-26 14:42:43 CST。
- `logs/validation.csv`：68,256 bytes，mtime 2026-08-26 16:11:13 CST。
- 舊run目錄未刪除、未改名、未覆寫；停止只放棄尚未完成的下一個partial epoch。

## 接受的v2政策

| stage | task mode | max epochs / patience | trainable scope | peak LR |
| --- | --- | --- | --- | --- |
| J0 | Pose only | 8 / 0 | 只開Pose head；Detect head與shared trunk全凍結 | Pose head 2e-4 |
| J1 | Detect + Pose | 20 / 8 | neck與兩heads；backbone、MASF、attention凍結 | neck7.5e-5、heads2e-4 |
| J2 | Detect + Pose | 80 / 17 | backbone layers9+、neck、MASF、兩heads；attention凍結 | backbone1.5e-5、neck7.5e-5、MASF1.5e-4、heads2e-4 |
| J3 | Detect + Pose | 20 / 5 | full low-LR refinement，開attention可微部分 | backbone3.8e-6、neck1.9e-5、MASF3.8e-5、attention5e-7、heads5e-5 |

J0/J1/J2各warmup 1 epoch；J3保留3 epochs。J2在stale 8時只做一次LR×0.5，
stale 17早停；optimizer不在run中途改種，仍使用AdamW，weight decay 0.00027、
betas=(0.948,0.999)。J3保留在程式與resolved config，但正式預設
`enable_j3: false`；須先看J2八項gate與梯度證據，再明確傳`--enable-j3`。

Joint task weights改成Detect1.0／Pose0.25。J0的macro只有Pose task active，active
weight會重新正規化，所以Pose gradient不會再被0.25縮小。J1之後仍每macro累積
2個logical Detect128與1個Pose16；Detect logical128由physical64×2執行，因此每macro
是4×Detect64加1×Pose16，依序forward/backward後只step一次。

## 程式變更

- `stage_policy.py`：加入J0與task mode；零LR role真正`requires_grad=False`；
  inactive Detect-head BN在J0維持eval。J1不調MASF，J2延後到layer9+與MASF，
  attention只在J3開放；Q/K與`score.gamma`的硬體契約仍永久凍結。
- `joint_loss.py`：macro可只啟用一種task；inactive task不forward、不產gradient；
  loss以active weight sum正規化。criterion progressive counter可按task選擇更新。
- `_joint_trainer_impl.py`／`joint_trainer.py`：新增`PoseEpochRunner`，J0每epoch
  只走一次完整Pose loader；joint epoch loss改為實際macro joint loss平均。
- `early_stop.py`：新增可state-dict round-trip的stage-local early stopper。
- `_formal_training_impl.py`：依task mode選runner與steps/epoch；Detect/Pose criterion
  使用各自max epoch；J1/J3 early-stop state與`stage_complete`寫入exact-resume
  checkpoint；J2沿用resume-safe plateau state。
- `joint_config.py`與兩個variant YAML：鎖定預設`[j0,j1,j2]`、warmup1、
  weight1.0/0.25，並把所有J0–J3 policy寫入resolved config。
- `scripts/run_full35_v2_seed0.py`：新增不覆寫舊run的preflight→GPU idle
  gate→J0/J1/J2→best_joint雙backend final validation queue。

未升級或變更Ultralytics、PyTorch與其他dependency；資料仍只使用COCO2017與
canonical `bbat5-v1`，fraction=1，沒有重切、抽樣或改label。

## Full35／Partial75隔離

兩者共用上表政策與程式碼，並都使用logical Detect128／physical64、Pose16。
Full35保持`enabled: true`且preflight `ready=true`。Partial75保持
`enabled: false`，尚缺自己的Pose P3與八項baseline；Full35 checkpoint、metrics、
cache與結論不得移植成Partial75結果。

## 驗證

所有CPU驗證均使用`CUDA_VISIBLE_DEVICES=''`與低CPU優先序，沒有占用GPU：

- targeted stage/loss/runner/config/early-stop：17 passed。
- 完整test suite：119 passed、3 skipped；3項只因需要CUDA而刻意skip，0 failed。
- Full35 `joint.py preflight`：`ready=true`、0 blocker。
- Partial75 preflight：如預期`ready=false`，只有未啟用、缺Partial75 Pose P3、
  缺Partial75八項baseline三個blocker。
- v2 queue `--plan`確認新run/evaluation/state路徑均與舊run分離。
- GPU idle gate先觀察到舊PID與free 5,827 MiB／util 100%，維持waiting；舊service
  停止後GPU為free 31,663 MiB、util 0%、無compute process。

## 困難與解法

1. 舊run仍在GPU上執行，不能邊改邊覆寫。解法：所有新版路徑加v2名稱；先完成CPU
   regression與preflight，再啟動等待式queue，最後只停止舊service。舊artifact完整保留。
2. 把LR設0本身不足以保證J0只訓練Pose head，因舊policy仍會把neck/head設為
   `requires_grad=True`且BN設train。解法：trainability與BN mode都由stage role LR及
   task mode明確決定，並加入gradient routing與BN測試。
3. J0不能沿用要求兩任務都存在的macro algebra。解法：只對active task計算weight sum，
   且只更新Pose criterion counter；Detect forward次數與影像計數固定為0。

## 正式啟動與首epoch實證

新版user service是`yolo-full35-v2-seed0.service`。16:11:49 CST先觀察舊run
PID 3748532而拒絕啟動；16:12:49與16:13:50連續兩次確認GPU無compute process、
free 31,663 MiB、util 0%，隨即以PID 3847922啟動新版訓練。queue durable state位於
`variants/full35/artifacts/queue/full35-v2-seed0/queue-status.json`。

J0 global epoch 0已完整完成：

- 373 Pose macro、5,964 Pose images、1.0 canonical dataset pass；最後一批12張。
- Detect batches/images/loss均為0；gradient presence固定為
  `shared=false, detect_head=false, pose_head=true`。
- GPU訓練時約used 4,240 MiB、util 66%、52°C；未發生OOM。
- 初始AMP scale自動復原：2個macro需要retry，最大12次，之後穩定在scale8；
  所有retry都在optimizer step前重做，沒有未處理的NaN/Inf或silent step。
- J0 epoch平均Pose loss 16.20382。這是P3 Pose head適應Detect trunk的第一個epoch，
  不能與獨立P3同trunk loss直接比較。
- Bit-True COCO overall/person mAP50-95為0.506555/0.626057，對baseline delta只有
  +0.000154/-0.000129，符合Detect/shared完全未更新。
- Bit-True BBAT box/pose mAP50-95為0.358228/0.705828；六個BBAT gate暫時失敗。
  這是適應階段第一個epoch的預期中間狀態，必須看完整8 epochs趨勢，不能提前宣稱
  融合成功或失敗。
- `best_detect.pt`、`best_pose.pt`與`last.pt`均已原子落盤（約250 MB）；
  因hard gate尚未通過，尚未建立`best_joint.pt`，這是fail-closed selector的正確行為。

## 未解事項與風險

- 新版實際mAP與梯度比例仍須GPU正式run產生，不能由CPU測試推論精度已改善。
- physical64×2只維持logical Detect exposure128，不等同physical128的head BN與
  microbatch內loss normalization；baseline/gate仍以既有同口徑validator比較。
- J1 early stop與J2 plateau監控unconditional joint score；八項0.08 hard gate仍是
  checkpoint可接受性的獨立veto，平均分數不能掩蓋單項失敗。
- J3不是消失，而是刻意不自動執行。只有J2完成、八項gate與gradient evidence支持時
  才啟用，避免用full unfreeze掩蓋資料routing或loss scale問題。
- Partial75沒有正式精度結果；除非使用者再次明確要求，不會配置GPU或建立run。
