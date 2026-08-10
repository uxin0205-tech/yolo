"""Leakage-safe dataset indexing, grouping, splitting, and export."""

from .labels import Box, parse_yolo_label

__all__ = ["Box", "parse_yolo_label"]
