"""Store and score class probabilities produced elsewhere (e.g. on a cloud GPU).

Transformer models are trained and run on Kaggle; only their class
probabilities for the validation and test splits come back. They are stored as
``predictions/<owner__name>/<model>.npz`` and scored locally by the same
``score_model`` as every other model, so all results stay comparable and can be
recomputed on a laptop without a GPU or torch.

Each file records a hash of the dataset it was made from. Scoring refuses files
made from a different dataset (rebuilt after new issues arrived), because the
splits would no longer match.

Usage:
    python -m triage_lens.predictions astral-sh/uv pandas-dev/pandas
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from triage_lens.dataset import CLASSES, REPOS
from triage_lens.evaluate import (
    load_dataset,
    load_results,
    print_tables,
    save_result,
    score_model,
)

SCORED_SPLITS = ("validation", "test")


class PredictionMismatch(ValueError):
    pass


def file_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_predictions(
    path: Path,
    probs_by_split: dict[str, tuple[Sequence[int], np.ndarray]],
    meta: dict[str, Any],
) -> Path:
    """Write ``{split: (issue numbers, probabilities)}`` and metadata to one npz file."""
    arrays: dict[str, np.ndarray] = {}
    for split, (numbers, probs) in probs_by_split.items():
        probs = np.asarray(probs, dtype=np.float32)
        if probs.shape != (len(numbers), len(CLASSES)):
            raise ValueError(f"{split}: expected shape ({len(numbers)}, 4), got {probs.shape}")
        arrays[f"{split}_numbers"] = np.asarray(numbers, dtype=np.int64)
        arrays[f"{split}_probs"] = probs
    meta = {**meta, "classes": list(CLASSES)}
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first: an interrupted run never leaves a broken npz.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    tmp.replace(path)
    return path


def load_predictions(path: Path) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    with np.load(path) as data:  # closes the file: Windows keeps open files locked
        meta = json.loads(str(data["meta"]))
        by_split = {
            split: {
                int(n): p
                for n, p in zip(data[f"{split}_numbers"], data[f"{split}_probs"], strict=True)
            }
            for split in SCORED_SPLITS
            if f"{split}_numbers" in data
        }
    if meta.get("classes") != list(CLASSES):
        raise PredictionMismatch(f"{path.name}: class order {meta.get('classes')} != {CLASSES}")
    return by_split, meta


def aligned(probs: dict[int, np.ndarray], records: Sequence[dict[str, Any]]) -> np.ndarray:
    """Probabilities in the order of ``records``; every issue must be present."""
    missing = [r["number"] for r in records if r["number"] not in probs]
    if missing:
        raise PredictionMismatch(f"{len(missing)} issues have no prediction, e.g. #{missing[0]}")
    return np.stack([probs[r["number"]] for r in records])


def score_file(path: Path, dataset_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Scores of one predictions file and its metadata."""
    by_split, meta = load_predictions(path)
    expected = file_hash(dataset_path)
    if meta.get("dataset_sha1") != expected:
        raise PredictionMismatch(
            f"{path.name} was made from a different dataset "
            f"({meta.get('dataset_sha1')} != {expected}); rerun the model on the current one"
        )
    splits = load_dataset(dataset_path)

    def predict_proba(records):
        if not records:
            return np.zeros((0, len(CLASSES)))
        return aligned(by_split.get(records[0]["split"], {}), records)

    return score_model(path.stem, predict_proba, splits), meta


def run(
    repo: str, data_dir: Path, predictions_dir: Path, results_dir: Path
) -> dict[str, dict[str, Any]]:
    stem = repo.replace("/", "__")
    dataset_path = data_dir / f"{stem}.dataset.jsonl"
    files = sorted((predictions_dir / stem).glob("*.npz"))
    if not files:
        raise SystemExit(f"No predictions for {repo} in {predictions_dir / stem}")

    for path in files:
        scores, meta = score_file(path, dataset_path)
        save_result(results_dir, repo, path.stem, scores, meta)
    table = load_results(results_dir, repo)
    print_tables(repo, table)
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score stored model predictions.")
    parser.add_argument("repos", nargs="+", choices=sorted(REPOS), help="repositories")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--predictions", type=Path, default=Path("predictions"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    for repo in args.repos:
        run(repo, args.data, args.predictions, args.results)


if __name__ == "__main__":
    main()
