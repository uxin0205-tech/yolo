# COCO2017 資料

此目錄保存重新訓練與正式驗證所需的 COCO2017 labels、annotation JSON 與影像連結。

| 分割 | 影像 | YOLO label 檔 | 物件標註 | 類別 |
| --- | ---: | ---: | ---: | ---: |
| train2017 | 118,287 | 117,266 | 860,001 | 80 |
| val2017 | 5,000 | 4,952 | 36,781 | 80 |

沒有物件的影像不一定有 `.txt` label，因此 label 檔數少於影像數。

```text
data/
├── README.md
├── coco2017/
│   ├── images -> <COCO2017_ROOT>/images
│   ├── labels/train2017/
│   ├── labels/val2017/
│   ├── train2017.txt
│   ├── val2017.txt
│   └── test-dev2017.txt
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

`labels/*.cache` 是 Ultralytics 自動建立的索引，可安全刪除並在下次訓練重建。此資料目錄除本 README 外不提交 Git。
