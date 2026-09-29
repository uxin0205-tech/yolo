# 2026-08-24 正式前置工作的 CPU 可行性查核

## 查核目的

確認 canonical Pose P1→P2→P3與八項standalone baseline在沒有GPU時，屬於程式不可執行，
還是僅因效能而不適合執行。

## 查核方式與結果

1. 在 `CUDA_VISIBLE_DEVICES=-1` 下唯讀檢查目前固定版本Ultralytics 8.4.90的
   `MuSGD`、`muon_update`與Newton–Schulz實作。它使用一般PyTorch tensor operation，
   沒有發現硬編碼CUDA device或CUDA-only kernel，因此程式層面可在CPU運作。
2. Pose、baseline與joint CLI的device皆可指定為`cpu`；preflight目前停止是因缺少正式
   Pose P3 checkpoint與八項baseline路徑，不是因為沒有CUDA。
3. 主機RAM查詢結果：總量125 GiB、當下available約77 GiB、swap 8 GiB。這只說明主機
   容量，不構成Full35 physical batch128的CPU memory gate。
4. 已有CPU smoke只在imgsz64、physical batch1完成兩個macro，證明graph/loss契約，不能
   外推為imgsz640、batch128正式訓練可行。

## 結論

- CPU現在適合：preflight、資料／hash／cache稽核、模型相容性、單步或小型smoke、loss與
  gradient routing、checkpoint round-trip，以及少量影像推論。
- standalone validation理論上可用CPU，但Full35 Float與Bit-True跑完整COCO／BBAT5會很慢；
  而且正式Pose P3尚不存在，所以目前無法產生完整八項matching baseline。
- P1 17＋P2 22＋P3最多100、imgsz640、physical batch128的正式Pose訓練，雖無程式硬性
  禁止CPU，實務上不建議：wall time可能非常長、CPU模式不使用正式GPU AMP口徑，且
  batch128主機RAM峰值尚未驗證。
- 因最終工作本來就要在RTX5090驗證與部署，應等待GPU空閒後先跑一個正式640 memory
  profile，再執行P1→P2→P3。不要用CPU長訓練產生不同口徑的正式基準。

## 困難、解法與未解風險

困難：沒有正式640 CPU或GPU profile，不能用64×64 smoke推測速度與記憶體。
解法：只做程式與資源唯讀查核，對formal run維持fail-fast，不啟動長時間CPU工作。

未解風險：GPU空卡時physical batch128是否可容納仍待驗證。除此之外，無新增程式blocker。
