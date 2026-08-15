"""Thin wrapper for CPU model-construction smoke checks."""

from yolo_attention.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["smoke", *__import__("sys").argv[1:]]))
