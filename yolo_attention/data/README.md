# data

`coco2017.yaml` 已提供完整 80-class names 與標準 train/val 相對路徑。目前使用已驗證的本機資料根目錄：

~~~yaml
path: /home/uxin/yolo/coco2017
~~~

改成真實資料根目錄。預期結構：

~~~text
coco/
├── images/
│   ├── train2017/
│   └── val2017/
├── labels/
│   ├── train2017/
│   └── val2017/
└── annotations/instances_val2017.json
~~~

檢查結果為 118,287 張 train images、5,000 張 val images，並有對應的 Ultralytics YOLO labels。沒有物件的圖片可以沒有 label text file，因此 label 數量不必和 image 數量完全相等。

COCO 圖片、annotations 副本與 cache 不提交 Git。若資料搬移，必須同步修改 YAML 的 `path`。smoke subset 與正式 val2017 必須使用不同 run ID。
