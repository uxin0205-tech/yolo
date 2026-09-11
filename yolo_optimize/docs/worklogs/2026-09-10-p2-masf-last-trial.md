# 最後一個 MASF 方向：P2 特徵增強與 [-10,0] PWL 契約

## 使用者要求與狀態

使用者追加最後一個實驗：MASF 放在 P2，但不增加 Detect head；若仍不好，放棄 MASF，接續融合等後續工作。此指示解除之前「融合是否保留 MASF」的選擇阻礙，改由本次實驗結果決策。先完成此追加實驗，combine／activation／方向2仍未啟動。

使用者另明確要求 Softmax PWL 使用 [-10,0]。CPU 實際載入 P2 訓練模型與已驗證 control Bit-True 匯出，兩個 attention site 均為 [-10,0]、20段、每段寬0.5，PWL可訓練參數數量0。類別建構子的預設是[-8,0]，但正式 YAML 的 score_min=-80、score_step=0.125 已覆蓋成[-10,0]，不需要重跑舊訓練。

新增 `pwl_contract.py`，在 P2 模型建立與兩組 smoke 匯出後檢查實際模組；來源 YAML 與舊權重未修改。

## 架構

CPU 實測160輸入時layer2輸出為[1,256,40,40]，即P2 stride4。layer2改為繼承原C3k2的P2MASFC3k2，保留既有參數key，先執行原forward再加MASF。

```text
layer2：P2_raw → MASF → 原 layer3／Backbone／Neck
                              ├→ P3 layer16 ─┐
                              ├→ P4 layer19 ─┤→ 原 Detect
                              └→ P5 layer22 ─┘
```

Detect仍只有一個，from=[16,19,22]，strides=[8,16,32]，沒有P2 prediction head。P2是共享上游，會影響全部三尺度，不宣稱像P3 Detect-only一樣隔離P4／P5。

MASF使用DW3／DW5相加及1×1 projection，256通道，fresh seed20260930；alpha從0開始，限制[-0.25,0.25]。不把P3已訓練context直接當作P2context。增加75,777參數；640輸入下估計僅卷積增加1,900,544,000 MAC／image，不含BN／activation／加法／記憶體流量，不是實測硬體延遲。

## 配對設計與超參數

兩組同起點為已完整驗收的無MASF control E8 EMA，SHA256=3aeeaec2b379d465ceb5aee6051e5c11772e856497676ec9c87f2cb511a0c06c。先各5個完整epoch，若結果值得延長才再判斷，不繼續掃其他MASF位置。控制組也須新跑，因本次起點／訓練scope與先前對照不同；不是重跑原本正常的同一job。

兩組皆訓練所有原Detect head分支；P2組再加入MASF。其餘參數與全BN統計固定，attention前向仍為精確BinaryQK，訓練使用既有surrogate讓上游P2梯度通過；PWL表格與attention score／bias不更新。原生one-to-one detach保留，不同時加P3梯度橋接。

AdamW betas=(0.948,0.999)、eps=1e-8、wd=0.00027、clip10；head/context LR=1e-5、alpha=1e-4；fresh optimizer、warmup1、10-epoch cosine horizon，先跑5epoch。EMA起始權重同parent，age14800。physical32×accumulation4=logical128，imgsz640，完整COCO train118287，每epoch925步，EMA/live各完整驗證5000張。pair seed20260930，mosaic/mixup/cutmix/copy_paste=0、fliplr0.5。

安全線為overall或person相對parent低於-0.005時停下分析。候選增準門檻為同epoch的overall/person皆不差於對照、至少一項+0.001，且兩項皆不低於parent。先以完整同預算結果判斷；若無足夠收益，依使用者指示放棄MASF並接無MASF融合。不把單一最佳epoch的小幅浮動直接當成功。

## 驗證與啟動結果

P2 preflight已通過：alpha0整圖前向及原參數完全相同；alpha0.01時MASF context／alpha梯度非零；Detect數量及三尺度輸入不變；初始完整COCO5000的四項AP差值皆0。

兩組128張smoke均通過，first macro trace相同，loss均215.72183990478516、head梯度norm均274.50619429811275。P2的alpha梯度非零、初始context梯度0符合零gate設計；首相對更新約5.19e-6。固定參數／EMA state、Bit-True匯出CPU160重載、兩site [-10,0] PWL均通過。

queue已於UTC 2026-09-09 18:34:56啟動正式control，之後自動接P2組；session92065。每個子job使用600秒blocking monitor，正常不讀log或GPU，錯誤／完成時才處理。狀態見 `artifacts/masf-p2-queue-v1-state.json`。此queue目前只包含P2配對，不把尚未適配的融合訓練硬接進去；配對結束後由結果決策並接續融合適配。

新增檔案為masf_p2.py、train_masf_p2.py、probe_masf_p2.py、run_masf_p2.py、pwl_contract.py，均位於study scripts。原方向1結果及其驗證報告保留，現在是追加P2前的歷史階段，不代表全部追加工作已完成。

## 困難、解法與未解事項

本次CPU／GPU preflight及smoke困難：無；physical32可完成這次真實更新，不是僅估算。P2正式AP、收斂與實際硬體成本尚未完成。較高解析度可能增加計算與activation流量；不能只因沒有新增head就宣稱沒有硬體成本。此實驗不保證精度一定提高。
