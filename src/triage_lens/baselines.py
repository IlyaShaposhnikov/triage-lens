"""Baselines for issue-type classification.

* ``majority`` — always the most frequent class of the training split.
* ``trust_form`` — the form the author picked; issues filed without a form get
  the majority class. This is what a repository gets today for free, so a model
  is only useful where it beats this baseline.
* ``tfidf_logreg`` — TF-IDF word uni- and bigrams + logistic regression with
  balanced class weights. The regularisation strength C is chosen on the
  validation split by macro-F1. Trains in seconds on a CPU and sets the bar the
  transformer models have to clear.

Usage:
    python -m triage_lens.baselines astral-sh/uv pandas-dev/pandas
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from triage_lens.dataset import CLASSES, REPOS
from triage_lens.evaluate import (
    TABLE_HEADER,
    evaluate,
    format_row,
    load_dataset,
    macro_f1,
    save_result,
)

C_GRID = (0.25, 1.0, 4.0, 16.0)
SEED = 42


def majority_class(train: Sequence[dict[str, Any]]) -> str:
    return Counter(r["label"] for r in train).most_common(1)[0][0]


def predict_majority(train, records) -> list[str]:
    label = majority_class(train)
    return [label] * len(records)


def predict_trust_form(train, records) -> list[str]:
    fallback = majority_class(train)
    return [r["form"] if r["form"] in CLASSES else fallback for r in records]


def make_tfidf_logreg(c: float) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    max_features=200_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=SEED),
            ),
        ]
    )


def fit_tfidf_logreg(train, validation) -> tuple[Pipeline, dict[str, Any]]:
    """Pick C on validation by macro-F1, return the model trained with it."""
    x_train, y_train = [r["text"] for r in train], [r["label"] for r in train]
    x_val, y_val = [r["text"] for r in validation], [r["label"] for r in validation]
    best = None
    search = {}
    for c in C_GRID:
        model = make_tfidf_logreg(c).fit(x_train, y_train)
        score = macro_f1(y_val, model.predict(x_val))
        search[str(c)] = round(score, 4)
        if best is None or score > best[0]:
            best = (score, c, model)
    _, c, model = best
    return model, {"C": c, "validation_macro_f1_by_C": search}


def run(repo: str, data_dir: Path, results_dir: Path) -> dict[str, dict[str, Any]]:
    stem = repo.replace("/", "__")
    splits = load_dataset(data_dir / f"{stem}.dataset.jsonl")
    train, validation, test = splits["train"], splits["validation"], splits["test"]

    model, params = fit_tfidf_logreg(train, validation)
    predictors = {
        "majority": lambda records: predict_majority(train, records),
        "trust_form": lambda records: predict_trust_form(train, records),
        "tfidf_logreg": lambda records: list(model.predict([r["text"] for r in records])),
    }

    summary = {}
    print(f"\n=== {repo} (test: {len(test)} issues)")
    print(TABLE_HEADER)
    for name, predict in predictors.items():
        scores = {
            "validation": evaluate(validation, predict(validation), bootstrap=False),
            "test": evaluate(test, predict(test)),
        }
        save_result(results_dir, repo, name, scores, params if name == "tfidf_logreg" else None)
        summary[name] = scores["test"]
        print(format_row(name, scores["test"]))
    print(
        f"tfidf_logreg: C={params['C']} (validation macro-F1 by C: "
        f"{params['validation_macro_f1_by_C']})"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the baselines.")
    parser.add_argument("repos", nargs="+", choices=sorted(REPOS), help="repositories")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    for repo in args.repos:
        run(repo, args.data, args.results)


if __name__ == "__main__":
    main()
