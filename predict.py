#!/usr/bin/env python3
"""PAN 2026 Subtask 1 inference entrypoint.

Usage expected by TIRA/PAN:
    python3 predict.py /absolute/path/to/dataset.jsonl /absolute/path/to/output_dir

The input JSONL must contain at least "id" and "text". The script writes exactly
one JSONL prediction file to the output directory, with one {"id", "label"} line
per input item. Labels are confidence scores in [0.0, 1.0], where higher means
AI-written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Record = Tuple[str, str]

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DEBERTA_DIR = APP_DIR / "models" / "deberta"
DEFAULT_NGRAM_MODEL = APP_DIR / "models" / "ngram_pipeline.joblib"
DEFAULT_OUTFILE = "prediction.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PAN 2026 Subtask 1 AI-authorship detector."
    )
    parser.add_argument("input_file", help="Path to input dataset.jsonl")
    parser.add_argument("output_dir", help="Directory where prediction JSONL is written")
    parser.add_argument(
        "--outfile-name",
        default=os.environ.get("OUTFILE_NAME", DEFAULT_OUTFILE),
        help=f"Output JSONL filename. Default: {DEFAULT_OUTFILE}",
    )
    parser.add_argument(
        "--deberta-dir",
        default=os.environ.get("DEBERTA_MODEL_DIR", str(DEFAULT_DEBERTA_DIR)),
        help="Local DeBERTa checkpoint directory.",
    )
    parser.add_argument(
        "--ngram-model",
        default=os.environ.get("NGRAM_MODEL_PATH", str(DEFAULT_NGRAM_MODEL)),
        help="Optional serialized sklearn/joblib N-gram pipeline.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "16")),
        help="Transformer inference batch size.",
    )
    return parser.parse_args()


def load_records(input_file: str) -> List[Record]:
    records: List[Record] = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            if "id" not in obj or "text" not in obj:
                raise ValueError(f"Line {line_no} must contain both 'id' and 'text'.")

            text = obj["text"]
            if text is None:
                text = ""
            records.append((str(obj["id"]), str(text)))

    return records


def batched(items: Sequence[Record], batch_size: int) -> Iterable[Sequence[Record]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def clip_probability(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def score_deberta(records: Sequence[Record], model_dir: Path, batch_size: int) -> Dict[str, float]:
    if not model_dir.exists():
        raise FileNotFoundError(
            f"DeBERTa model directory not found: {model_dir}. "
            "Build the Docker image with the trained checkpoint included."
        )

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    scores: Dict[str, float] = {}
    with torch.inference_mode():
        for batch in batched(records, batch_size):
            ids = [item[0] for item in batch]
            texts = [item[1] for item in batch]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            for doc_id, prob in zip(ids, probs):
                scores[doc_id] = clip_probability(float(prob))

    return scores


def score_ngram(records: Sequence[Record], model_path: Path) -> Dict[str, float]:
    import joblib

    model = joblib.load(model_path)
    ids = [item[0] for item in records]
    texts = [item[1] for item in records]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)[:, 1]
    elif isinstance(model, dict) and {"vectorizer", "classifier"} <= set(model):
        features = model["vectorizer"].transform(texts)
        probs = model["classifier"].predict_proba(features)[:, 1]
    else:
        raise TypeError(
            "N-gram model must be a sklearn Pipeline or a dict with "
            "'vectorizer' and 'classifier'."
        )

    return {doc_id: clip_probability(float(prob)) for doc_id, prob in zip(ids, probs)}


def blend_scores(
    records: Sequence[Record],
    deberta_scores: Dict[str, float],
    ngram_scores: Dict[str, float] | None,
) -> List[Dict[str, float | str]]:
    results: List[Dict[str, float | str]] = []

    use_ngram = bool(ngram_scores)
    # Validation showed the serialized N-gram model should dominate when present.
    ngram_weight = 0.9 if use_ngram else 0.0
    deberta_weight = 0.1 if use_ngram else 1.0

    for doc_id, _ in records:
        score = deberta_weight * deberta_scores[doc_id]
        total_weight = deberta_weight
        if use_ngram and ngram_scores is not None:
            score += ngram_weight * ngram_scores[doc_id]
            total_weight += ngram_weight
        results.append({"id": doc_id, "label": clip_probability(score / total_weight)})

    return results


def write_predictions(output_dir: str, outfile_name: str, predictions: Sequence[Dict[str, float | str]]) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_path = output_path / outfile_name

    with open(prediction_path, "w", encoding="utf-8") as out:
        for item in predictions:
            json.dump(item, out, ensure_ascii=False)
            out.write("\n")

    return prediction_path


def main() -> int:
    args = parse_args()

    records = load_records(args.input_file)
    print(f"Loaded {len(records)} input texts.", file=sys.stderr)

    deberta_scores = score_deberta(records, Path(args.deberta_dir), args.batch_size)

    ngram_scores = None
    ngram_path = Path(args.ngram_model)
    if ngram_path.exists():
        print(f"Loading optional N-gram model: {ngram_path}", file=sys.stderr)
        ngram_scores = score_ngram(records, ngram_path)
    else:
        print("Optional N-gram model not found; using DeBERTa only.", file=sys.stderr)

    predictions = blend_scores(records, deberta_scores, ngram_scores)
    prediction_path = write_predictions(args.output_dir, args.outfile_name, predictions)
    print(f"Wrote predictions to {prediction_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
