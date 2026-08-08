# BBT5 detect baseline

`../pose_dataset/bbt5.v1i.yolov8` is exported as YOLO pose data, but this work item uses only bounding-box detection. The preparation script keeps the original dataset untouched, links the real `.jpg` files, and converts each 11-field pose row to its first 5 detection fields.

From this directory, prepare the local detect view:

```bash
../../.venv/bin/python prepare_dataset.py
```

The generated `dataset/` is intentionally ignored by Git. It is a symlinked view of the source images plus generated detect labels, so rerunning the command is safe and does not copy the 438 MB image set.

## Recommended baseline

Use the normal YOLO11m detection checkpoint. The available `../../original/weight/yolo11m.pt` is a detect model; Ultralytics will create the two-class detection head from `data.yaml` and transfer compatible COCO weights:

```bash
../../.venv/bin/yolo detect train \
  model=../../original/weight/yolo11m.pt \
  data=data.yaml epochs=100 imgsz=640 batch=-1 device=0 \
  project=runs name=yolo11m-bbt5-detect
```

Validate a trained checkpoint with:

```bash
../../.venv/bin/yolo detect val \
  model=runs/yolo11m-bbt5-detect/weights/best.pt \
  data=data.yaml device=0
```

## Optional pose-to-detect transfer

`../../pose_dataset/weight/yolo11m_bat.pt` is a `PoseModel` with a `Pose` head. It cannot be passed directly to `yolo detect train`: its checkpoint contains a keypoint branch and its task is pose. A detect model must be instantiated first.

The helper below builds a two-class detect model, transfers all matching backbone/neck/detection parameters, drops the pose-only keypoint branch, and saves a new initializer:

```bash
../../.venv/bin/python transfer_pose_to_detect.py
```

Then train it as a detect model:

```bash
../../.venv/bin/yolo detect train \
  model=weights/yolo11m_bat_detect_init.pt \
  data=data.yaml epochs=100 imgsz=640 batch=-1 device=0 \
  project=runs name=yolo11m-bbt5-detect-from-pose
```

This is a useful transfer experiment, but it should be reported separately from the standard detect baseline because the initializer has already seen this BBT5 dataset during pose training. Do not overwrite either source checkpoint.
