"""Thin wrapper for the installed yolo26-attention CLI."""

from yolo_attention.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *__import__("sys").argv[1:]]))
