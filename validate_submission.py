#!/usr/bin/env python3
"""Validate PAN 2026 Subtask 1 prediction JSONL format."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PAN Subtask 1 output format.")
    parser.add_argument("input_file", help="Original input dataset.jsonl")
    parser.add_argument("prediction_file", help="Prediction JSONL produced by predict.py")
    return parser.parse_args()


def load_input_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "id" not in obj or "text" not in obj:
                raise ValueError(f"{path}:{line_no} must contain 'id' and 'text'.")
            ids.append(str(obj["id"]))
    return ids


def load_prediction_ids(path: Path) -> Dict[str, float]:
    predictions: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if set(obj) != {"id", "label"}:
                raise ValueError(f"{path}:{line_no} must contain exactly 'id' and 'label'.")
            label = obj["label"]
            if not isinstance(label, (int, float)) or isinstance(label, bool):
                raise ValueError(f"{path}:{line_no} label must be numeric.")
            label = float(label)
            if math.isnan(label) or label < 0.0 or label > 1.0:
                raise ValueError(f"{path}:{line_no} label must be in [0.0, 1.0].")
            doc_id = str(obj["id"])
            if doc_id in predictions:
                raise ValueError(f"{path}:{line_no} duplicate id: {doc_id}")
            predictions[doc_id] = label
    return predictions


def main() -> int:
    args = parse_args()
    input_ids = load_input_ids(Path(args.input_file))
    predictions = load_prediction_ids(Path(args.prediction_file))

    input_set: Set[str] = set(input_ids)
    prediction_set: Set[str] = set(predictions)
    missing = input_set - prediction_set
    extra = prediction_set - input_set

    if len(input_ids) != len(input_set):
        raise ValueError("Input file contains duplicate ids.")
    if missing or extra:
        raise ValueError(
            f"ID mismatch: missing={len(missing)}, extra={len(extra)}."
        )

    print(
        f"OK: {len(predictions)} predictions, all ids match, labels are valid probabilities."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
