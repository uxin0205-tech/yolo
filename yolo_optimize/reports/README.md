# 正式報告與模型索引

[先讀：目前實際使用的模型、架構圖與白話說明](<current-model/README.md>)

| 入口 | 用途 |
| --- | --- |
| [professor-overview](<professor-overview/README.md>) | 教授版完整研究報告：研究問題、架構演進、對照實驗、失敗嘗試、Accuracy–Cost 與下一步 |
| [current-model](<current-model/README.md>) | 目前選定模型、元件位置、同權重開關診斷與 Attention 重訓說明 |
| [final](<final/README.md>) | 唯一全階段詳細原始總報告，含數值 CSV、run／checkpoint 清單 |
| [direction1](<direction1/README.md>) | 融合前方向 1 的深入結果與比較 |
| [checkpoints](<checkpoints/README.md>) | 現行預設、父模型、對照與未升版候選的新路徑和 SHA |
| [publication](<publication/README.md>) | 從 BinaryQK 起的閱讀順序與 GitHub 發布稽核 |
| [performance](<performance/README.md>) | 九項精度／部署成本比較，含各階段分表與完整 CSV |

`professor-overview` 是重新編寫的研究敘事；`final` 則是原始、完整的研究帳本。兩者用途不同，不互相覆蓋。

原日期型資料夾已實際改名為以上分類，日期保留在報告內容與工作紀錄，不再增加同一總報告的另一份「最終版」。原始量測留在 [experiments](<../experiments/README.md>)；本機大封存留在 archives（本機檔案：`../archives/README.md`；本次未上傳）。

此處不放模型 bytes。報告、CSV／JSON 與必要研究程式可以發布；checkpoint 透過清單追溯。更新結論須引用相同 parent、資料版本、backend 與 evaluator，不改寫失敗成成功。
