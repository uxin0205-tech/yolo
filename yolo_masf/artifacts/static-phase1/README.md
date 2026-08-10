# Static Phase 1 產物索引

Pipeline `cb7e0392cc32` 的 34 個階段已完成且 final audit 通過。

| 路徑 | 內容 |
|---|---|
| `dataset/` | 固定 split、COCO JSON、統計與 manifest |
| `runs/` | Ultralytics 正式 native best/last、CSV、圖表 |
| `training/` | resolved args、run manifest、canonical checkpoint |
| `evaluation/{val,test}/` | 十個模型的預測與 COCO 指標 |
| `profiles/` | RTX 5090 FP16 latency、GFLOPs、參數與 operator |
| `references/` | B0 權重 task/classes/stride/hash 驗證 |
| `stages/`、`state.json` | DAG、hash 與續跑依據 |
| `selection.json` / `final_audit.json` | M2/M3 選擇與終局稽核 |
| `report.md` | 機器證據重建的總報告 |

正式對應：B1=`runs/b1-b`+`training/b1_b`；M7=`runs/m7`+`training/formal_m7`；M0–M3/P3M 同名 formal 目錄；SP2=`runs/sp2-b`+`training/sp2_b`；SP2P=`runs/sp2p-b`+`training/sp2p_b`。A 是凍結 10 epochs、B 是全解凍 90 epochs，兩者都是正式證據。

`smoke_runs/`、`preflight/`、`m7_gate/` 的 manifests 可保留，但大型 checkpoint 可依白名單刪除。`evaluation_cpu/` 是早期 CPU 證據，不與正式 GPU 結果混用。
