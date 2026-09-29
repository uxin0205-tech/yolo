# COCO2017 data

The plan uses the complete repository-root `coco2017/images/train2017` and
`coco2017/images/val2017` folders.  It does not run 5k/30k subsets or micro /
pilot training.

After the official images and YOLO labels are present, create one immutable
full manifest:

```bash
cd yolo_binaryqk
../.venv/bin/python -m binary_attention.cli make-manifest \
  --coco ../coco2017 --output data/coco_full.txt --seed 0
```

The command writes `data/coco_full.json` with the list hash and count.  The
dataset YAML is `data/coco-full.yaml`; no data or weights are downloaded by the
experiment package.
