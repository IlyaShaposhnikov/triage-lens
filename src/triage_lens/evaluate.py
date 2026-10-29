"""Shared evaluation for every issue-type classifier in the project.

All models are scored the same way on the same split:

* macro-F1 (each class weighs the same, so rare classes such as pandas questions
  or uv docs count as much as bugs), with a bootstrap 95% confidence interval;
* accuracy, per-class precision / recall / F1 with support, confusion matrix;
* the *misfiled* view. Among issues filed through a form, a misfiled issue is one
  where the maintainer's label differs from the form the author picked. A model
  "flags" an issue when its prediction differs from the form. Reported:

  - ``caught``: share of misfiled issues the model flags;
  - ``fixed``: share of misfiled issues where the model predicts the right label;
  - ``flag_precision``: share of flags that point at a truly misfiled issue;
  - ``false_alarm_rate``: share of correctly filed issues the model flags.

Results are saved as JSON under results/<owner__name>/<model>.json.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from triage_lens.dataset import CLASSES

BOOTSTRAP_SAMPLES = 1000


def load_dataset(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Records of a dataset file grouped by split."""
    splits: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                splits.setdefault(record["split"], []).append(record)
    return splits


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=list(CLASSES), average="macro", zero_division=0))


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile interval of macro-F1 over resampled test sets."""
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = [
        macro_f1(true[idx], pred[idx])
        for idx in (rng.integers(0, len(true), len(true)) for _ in range(samples))
    ]
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def misfiled_report(records: Sequence[dict[str, Any]], y_pred: Sequence[str]) -> dict[str, Any]:
    """How well predictions catch issues filed through the wrong form."""
    rows = [
        (r["form"], r["label"], pred)
        for r, pred in zip(records, y_pred, strict=True)
        if r["form"] in CLASSES
    ]
    misfiled = [(form, label, pred) for form, label, pred in rows if form != label]
    correct = [(form, label, pred) for form, label, pred in rows if form == label]
    flagged_misfiled = sum(pred != form for form, _, pred in misfiled)
    flagged_correct = sum(pred != form for form, _, pred in correct)
    flags = flagged_misfiled + flagged_correct

    def share(a: int, b: int) -> float | None:
        return a / b if b else None

    return {
        "filed_via_form": len(rows),
        "misfiled": len(misfiled),
        "caught": share(flagged_misfiled, len(misfiled)),
        "fixed": share(sum(pred == label for _, label, pred in misfiled), len(misfiled)),
        "flag_precision": share(flagged_misfiled, flags),
        "false_alarm_rate": share(flagged_correct, len(correct)),
    }


def evaluate(
    records: Sequence[dict[str, Any]], y_pred: Sequence[str], bootstrap: bool = True
) -> dict[str, Any]:
    y_true = [r["label"] for r in records]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASSES), zero_division=0
    )
    result: dict[str, Any] = {
        "n": len(y_true),
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "per_class": {
            cls: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(CLASSES)
        },
        "confusion": {
            "labels": list(CLASSES),  # rows: true label, columns: prediction
            "matrix": confusion_matrix(y_true, y_pred, labels=list(CLASSES)).tolist(),
        },
        "misfiled": misfiled_report(records, y_pred),
    }
    if bootstrap:
        result["macro_f1_ci"] = list(bootstrap_ci(y_true, y_pred))
    return result


def save_result(
    results_dir: Path,
    repo: str,
    model: str,
    scores: dict[str, dict[str, Any]],
    params: dict[str, Any] | None = None,
) -> Path:
    """Write the scores of one model (per split) to results/<owner__name>/<model>.json."""
    path = results_dir / repo.replace("/", "__") / f"{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"repo": repo, "model": model, "params": params or {}, "scores": scores}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def format_row(model: str, result: dict[str, Any]) -> str:
    """One line of the summary table printed by the training scripts."""

    def pct(value: float | None) -> str:
        return "   -  " if value is None else f"{100 * value:5.1f}%"

    ci = result.get("macro_f1_ci")
    ci_text = f"[{ci[0]:.3f}-{ci[1]:.3f}]" if ci else " " * 13
    m = result["misfiled"]
    return (
        f"{model:<16}{result['macro_f1']:>7.3f} {ci_text}  {pct(result['accuracy'])}"
        f"   {pct(m['caught'])}  {pct(m['fixed'])}  {pct(m['false_alarm_rate'])}"
    )


TABLE_HEADER = (
    f"{'model':<16}{'macroF1':>7} {'95% CI':<13}  {'acc':>6}"
    f"   {'caught':>6}  {'fixed':>6}  {'false alarms':>6}"
)
