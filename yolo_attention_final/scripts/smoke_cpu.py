"""Small CPU construction/forward smoke for the fixed YOLO26m architecture."""

from yolo_attention.cli import main

raise SystemExit(main(["smoke", "--model", "yolo26m.yaml", "--imgsz", "64"]))
