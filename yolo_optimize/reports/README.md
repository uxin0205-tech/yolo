# 正式報告與模型索引

| 入口 | 用途 |
| --- | --- |
| [final](<final/README.md>) | 唯一全階段詳細總報告，含數值 CSV、run／checkpoint 清單 |
| [direction1](<direction1/README.md>) | 融合前方向 1 的深入結果與比較 |
| [checkpoints](<checkpoints/README.md>) | 現行預設、父模型、對照與未升版候選的新路徑和 SHA |
| [publication](<publication/README.md>) | 從 BinaryQK 起的閱讀順序與 GitHub 發布稽核 |

原日期型資料夾已實際改名為以上分類，日期保留在報告內容與工作紀錄，不再增加同一總報告的另一份「最終版」。原始量測留在 [experiments](<../experiments/README.md>)；本機大封存留在 archives（本機／歷史參照：`../archives/README.md`；未隨本次報告發布）。

此處不放模型 bytes。報告、CSV／JSON 與必要研究程式可以發布；checkpoint 透過清單追溯。更新結論須引用相同 parent、資料版本、backend 與 evaluator，不改寫失敗成成功。

[performance：九項精度／部署成本比較](<performance/README.md>)，含各階段分表與完整 CSV。
