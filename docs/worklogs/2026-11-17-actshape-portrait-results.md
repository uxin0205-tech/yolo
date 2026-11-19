# 2026-11-17：768×512 訓練完成與結果核對

## 任務與完成狀態

使用者詢問目前狀況。本次核對先前授權的 H×W＝768×512 同配置訓練，確認已於 **2026-11-13 00:07:28（Asia/Taipei）** 完成 10E＋半 LR 10E。兩段均以 budget 正常結束，各 job 只啟動一次，沒有 SIGSEGV 重試、資源暫停或補充校準。連同前輪共完成 120E；本研究 supervisor／worker 已結束，沒有啟動下一階段。

## 結果與比較

最佳模型為 adapt10 第 5 輪，後續半 LR 10 輪沒有超過此最佳。下表全部為 AP50–95 百分比，各列來自同一份 checkpoint。

| 模型／H×W | COCO | person | COCO 球棒 | BBAT box | Pose |
|---|---:|---:|---:|---:|---:|
| 原 qSiLU Rep17＋20／rect640 | 51.4347 | 64.2796 | 46.9576 | 61.4371 | 90.1389 |
| 512×768 最佳 | 51.9632 | 64.5865 | 45.3351 | 62.4501 | 89.9026 |
| 768×512 最佳 | 50.3517 | 62.9266 | 44.6974 | 60.5912 | 89.5456 |

768×512 的十項 AP 均低於 512×768。與原主線比較，除 BBAT ball box 提高 0.3759 個百分點，其餘九項超過允許的下降範圍；COCO 下降 1.0831、person 下降 1.3530、COCO 球棒下降 2.2603、Pose 下降 0.5933 個百分點。正式主線保持原 qSiLU Rep17＋20／rect640，不以候選取代。

兩方向的分析 Conv＋P.V 同為 45.508165632 GMAC，沒有計算量節省。直向的有效影像縮放較小、平均補邊較多（COCO 41.43%、BBAT5 52.63%），是退化的可能因素，未做額外因果分離。GPU 中位時間 29.4187 ms 對橫向 29.6809 ms，屬單次軟體量測，不宣稱穩定硬體加速。

## 驗證方式與結果

已逐項核對兩段實際 training recipe 與先前 512×768 完全一致，只有形狀與各自量測出的校準值不同。Pose weight 分別為 0.08、0.11，均由原規則直接通過，未啟用 16+8 補充；parent、seed、資料、evaluator 與部署契約保持。

- 兩段各 10 輪，frozen tensor 保護均通過；最佳完整重載的十項 AP 最大誤差為 0。
- 466 個來源檔案 SHA、372 個交接檔案的大小及 SHA、CSV 四列指標皆相符。
- 42 項 GPU replay 已通過，state 不變，算子契約通過。
- 既有獨立 CPU 可攜驗證 bitwise equal，錯誤 SiLU 會被拒絕；本次沒有重跑已通過的 CPU 模型 smoke 或 GPU AP。
- 新候選最佳 SHA：`c589ab0e0590937e496553d24a74b3c236adf9c854dbfbe15d911becb129b7cc`。
- 原正式 handoff SHA 仍為 `2454548104ed64768eb9ecdb99d44804ee9d8e1e371e2a508cf052c0911fdd35`。

完整核驗見 [final-audit.json](../../yolo_actshape/result/stage9-portrait/final-audit.json)。目前 GPU 由其他專案程序使用；本研究已釋放，未操作或中斷外部工作。

## 整理與清理

完成成果在 [result/stage9-portrait](../../yolo_actshape/result/stage9-portrait/README.md)，候選可使用 [獨立交接包](../../yolo_actshape/result/stage9-portrait/handoff/README.md)。自動清理已移除兩份 optimizer resume，共 **681,214,190 bytes（0.63443 GiB）**；本次逐項核對確實不存在，最佳 E5 權重保留。沒有新增刪除或 archive／backup。

補充正式報告的最佳輪次、方向比較、硬體界線與清理數量，更新根層／子專案 README、計畫及 AGENTS 的完成狀態，新增此工作紀錄與索引。原始 protocol、實驗來源鎖、指標與權重保持。

## 困難與未解事項

本次困難：無。先前訓練入口已預先核對 optimize；此次只查核已完成結果與文件，不新增配方或訓練。單 seed、validation 選型、無相同方法 control 加訓的限制保持；未做整網整數量化、RTL／HLS 或板端量測。過去其他 run 的 CUDA 原生崩潰根因仍未確認，本候選無該錯誤，不能由此宣稱驅動根因已修復。
