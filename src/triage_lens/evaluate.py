"""Shared evaluation for every issue-type classifier in the project.

A model is any function that maps records to class probabilities: an array of
shape (n, 4) with columns in ``CLASSES`` order. All models are scored the same
way, by :func:`score_model`, on the same time-based split.

Two views are reported.

**Standalone** — the model replaces the issue form:

* macro-F1 (every class weighs the same) and accuracy, each with a bootstrap 95%
  confidence interval. Both are shown because macro-F1 is dominated by tiny
  classes (15 uv docs issues, 73 pandas questions in test);
* per-class precision / recall / F1 with support, and the confusion matrix;
* the paired difference to the "trust the form" baseline: both are scored on the
  same resampled test sets, so the interval is for the difference itself.

**Form + flag** — how the model would actually be used. The form the author
picked stays the default; the model raises a flag when it is confident the form
is wrong, i.e. when ``1 - P(form class) >= threshold``, and suggests its best
other class. For issues filed without a form the model's own prediction is used.

* ``average_precision`` of the flag score: threshold-free quality of ranking
  misfiled issues above correctly filed ones;
* the threshold is chosen on validation as the one that catches the most
  misfiled issues while at least ``TARGET_FLAG_PRECISION`` of flags are right,
  then applied unchanged to test;
* at that threshold: ``caught`` (share of misfiled issues flagged), ``fixed``
  (flagged with the right suggestion), ``flag_precision``, ``false_alarm_rate``
  (share of correctly filed issues flagged), and the accuracy / macro-F1 of the
  combined system with its paired difference to the form alone.

The combined result is also split into two subsets, because they answer
different questions and mixing them overstates what flagging achieves:

* ``with_form`` — issues filed through a form: the gain comes only from flags;
* ``no_form`` — issues filed without a form (or through a form outside the four
  classes): the model's own prediction against the majority-class fallback.

Results are saved as JSON under results/<owner__name>/<model>.json.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from triage_lens.dataset import CLASSES

BOOTSTRAP_SAMPLES = 1000
TARGET_FLAG_PRECISION = 0.5
MIN_VALIDATION_FLAGS = 5  # a precision measured on fewer flags is too noisy to trust
CLASS_INDEX = {cls: i for i, cls in enumerate(CLASSES)}

Records = Sequence[dict[str, Any]]
PredictProba = Callable[[Records], np.ndarray]


# --------------------------------------------------------------------------- data


def load_dataset(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Records of a dataset file grouped by split."""
    splits: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                splits.setdefault(record["split"], []).append(record)
    return splits


def majority_class(train: Records) -> str:
    return Counter(r["label"] for r in train).most_common(1)[0][0]


def one_hot(labels: Sequence[str]) -> np.ndarray:
    probs = np.zeros((len(labels), len(CLASSES)))
    probs[np.arange(len(labels)), [CLASS_INDEX[label] for label in labels]] = 1.0
    return probs


def form_predictions(records: Records, fallback: str) -> list[str]:
    """The "trust the form" prediction; issues without a form get ``fallback``."""
    return [r["form"] if r["form"] in CLASSES else fallback for r in records]


def to_labels(probs: np.ndarray) -> list[str]:
    return [CLASSES[i] for i in np.asarray(probs).argmax(axis=1)]


# --------------------------------------------------------------------------- metrics


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Macro-F1 over the four classes (a class absent from both sides scores 0).

    Computed from a confusion matrix with numpy: the bootstrap calls it tens of
    thousands of times, where sklearn's f1_score is too slow. A test checks it
    against sklearn.
    """
    true = np.array([CLASS_INDEX[y] for y in y_true])
    pred = np.array([CLASS_INDEX[y] for y in y_pred])
    k = len(CLASSES)
    cm = np.bincount(true * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)  # 2*tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


METRICS = {"macro_f1": macro_f1, "accuracy": accuracy}


def _resamples(n: int, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        yield rng.integers(0, n, n)


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Callable = macro_f1,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile interval of a metric over resampled test sets."""
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    scores = [metric(true[idx], pred[idx]) for idx in _resamples(len(true), samples, seed)]
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def paired_bootstrap(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    metric: Callable = macro_f1,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> dict[str, float]:
    """Difference metric(a) - metric(b) on the same resamples, with a 95% interval.

    ``p_a_better`` is the share of resamples where a scores strictly higher.
    """
    true, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    diffs = np.array(
        [
            metric(true[idx], a[idx]) - metric(true[idx], b[idx])
            for idx in _resamples(len(true), samples, seed)
        ]
    )
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": metric(true, a) - metric(true, b),
        "ci": [float(low), float(high)],
        "p_a_better": float(np.mean(diffs > 0)),
    }


def standalone_report(records: Records, y_pred: Sequence[str]) -> dict[str, Any]:
    y_true = [r["label"] for r in records]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASSES), zero_division=0
    )
    return {
        "n": len(y_true),
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
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
    }


# --------------------------------------------------------------------------- form + flag


def flag_scores(records: Records, probs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Per issue: confidence that its form is wrong, and the suggested other class.

    Issues without a form get score NaN and their own top class as suggestion.
    """
    probs = np.asarray(probs, dtype=float)
    scores = np.full(len(records), np.nan)
    suggestions = []
    for i, record in enumerate(records):
        row = probs[i]
        if record["form"] in CLASSES:
            form = CLASS_INDEX[record["form"]]
            scores[i] = 1.0 - row[form]
            others = row.copy()
            others[form] = -1.0
            suggestions.append(CLASSES[int(others.argmax())])
        else:
            suggestions.append(CLASSES[int(row.argmax())])
    return scores, suggestions


def _flag_counts(records: Records, scores: np.ndarray, threshold: float) -> dict[str, int]:
    filed = [i for i, r in enumerate(records) if r["form"] in CLASSES]
    misfiled = {i for i in filed if records[i]["form"] != records[i]["label"]}
    flagged = {i for i in filed if scores[i] >= threshold}
    return {
        "filed": len(filed),
        "misfiled": len(misfiled),
        "flags": len(flagged),
        "true_flags": len(flagged & misfiled),
        "false_flags": len(flagged - misfiled),
    }


def choose_threshold(
    records: Records,
    probs: np.ndarray,
    target_precision: float = TARGET_FLAG_PRECISION,
    min_flags: int = MIN_VALIDATION_FLAGS,
) -> float | None:
    """Lowest threshold whose flags reach the target precision (most misfiled caught).

    Returns None when no threshold does, meaning the model should never flag.
    """
    scores, _ = flag_scores(records, probs)
    candidates = sorted({float(s) for s in scores if not np.isnan(s) and s > 0})
    for threshold in candidates:  # ascending: lower threshold = more flags
        counts = _flag_counts(records, scores, threshold)
        if counts["flags"] >= min_flags and counts["true_flags"] / counts["flags"] >= (
            target_precision
        ):
            return threshold
    return None


def combined_predictions(
    records: Records, probs: np.ndarray, threshold: float | None
) -> tuple[list[str], np.ndarray]:
    """Form by default, the model's suggestion where it flags, the model where no form."""
    scores, suggestions = flag_scores(records, probs)
    preds = []
    for i, record in enumerate(records):
        if record["form"] not in CLASSES:
            preds.append(suggestions[i])
        elif threshold is not None and scores[i] >= threshold:
            preds.append(suggestions[i])
        else:
            preds.append(record["form"])
    return preds, scores


def flag_report(records: Records, probs: np.ndarray, threshold: float | None) -> dict[str, Any]:
    preds, scores = combined_predictions(records, probs, threshold)
    filed = [i for i, r in enumerate(records) if r["form"] in CLASSES]
    is_misfiled = np.array([records[i]["form"] != records[i]["label"] for i in filed])
    filed_scores = scores[filed]

    def share(a: int, b: int) -> float | None:
        return a / b if b else None

    counts = _flag_counts(records, scores, threshold if threshold is not None else np.inf)
    fixed = sum(
        1
        for i in filed
        if records[i]["form"] != records[i]["label"] and preds[i] == records[i]["label"]
    )
    y_true = [r["label"] for r in records]
    return {
        "threshold": threshold,
        "average_precision": (
            float(average_precision_score(is_misfiled, filed_scores))
            if is_misfiled.any() and not is_misfiled.all()
            else None
        ),
        **counts,
        "caught": share(counts["true_flags"], counts["misfiled"]),
        "fixed": share(fixed, counts["misfiled"]),
        "flag_precision": share(counts["true_flags"], counts["flags"]),
        "false_alarm_rate": share(counts["false_flags"], counts["filed"] - counts["misfiled"]),
        "combined": {
            **_scores(y_true, preds),
            "subsets": {
                name: {"n": len(idx), **_scores([y_true[i] for i in idx], [preds[i] for i in idx])}
                for name, idx in subset_indices(records).items()
            },
        },
    }, preds


def subset_indices(records: Records) -> dict[str, list[int]]:
    """Positions of issues filed through one of the four forms, and of the rest."""
    with_form = [i for i, r in enumerate(records) if r["form"] in CLASSES]
    no_form = [i for i, r in enumerate(records) if r["form"] not in CLASSES]
    return {"with_form": with_form, "no_form": no_form}


def _scores(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float | None]:
    if not y_true:
        return {"accuracy": None, "macro_f1": None}
    return {"accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred)}


# --------------------------------------------------------------------------- orchestration


def evaluate(
    records: Records,
    probs: np.ndarray,
    threshold: float | None,
    reference: Sequence[str] | None = None,
    bootstrap: bool = True,
) -> dict[str, Any]:
    """Both views for one split. ``reference``: "trust the form" predictions."""
    y_true = [r["label"] for r in records]
    y_pred = to_labels(probs)
    result = standalone_report(records, y_pred)
    result["flagging"], combined = flag_report(records, probs, threshold)
    if bootstrap:
        for name, metric in METRICS.items():
            result[f"{name}_ci"] = list(bootstrap_ci(y_true, y_pred, metric))
    if bootstrap and reference is not None:
        result["vs_form"] = {
            name: paired_bootstrap(y_true, y_pred, reference, metric)
            for name, metric in METRICS.items()
        }
        result["flagging"]["combined"]["vs_form"] = {
            name: paired_bootstrap(y_true, combined, reference, metric)
            for name, metric in METRICS.items()
        }
        for subset, idx in subset_indices(records).items():
            if not idx:
                continue
            true = [y_true[i] for i in idx]
            result["flagging"]["combined"]["subsets"][subset]["vs_form"] = {
                name: paired_bootstrap(
                    true, [combined[i] for i in idx], [reference[i] for i in idx], metric
                )
                for name, metric in METRICS.items()
            }
    return result


def score_model(
    name: str,
    predict_proba: PredictProba,
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Score a model on validation and test; the flag threshold comes from validation."""
    validation, test = splits["validation"], splits["test"]
    val_probs, test_probs = predict_proba(validation), predict_proba(test)
    threshold = choose_threshold(validation, val_probs)
    reference = form_predictions(test, majority_class(splits["train"]))
    return {
        "validation": evaluate(validation, val_probs, threshold, bootstrap=False),
        "test": evaluate(test, test_probs, threshold, reference=reference),
    }


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


# --------------------------------------------------------------------------- printing


def _pct(value: float | None) -> str:
    return "    -  " if value is None else f"{100 * value:6.1f}%"


def _diff(entry: dict[str, Any] | None, scale: float = 100) -> str:
    if not entry:
        return " " * 22
    lo, hi = entry["ci"]
    return f"{scale * entry['diff']:+6.1f} [{scale * lo:+5.1f},{scale * hi:+5.1f}]"


STANDALONE_HEADER = (
    f"{'model':<16}{'macroF1':>8} {'95% CI':<14}{'acc':>8}   {'acc vs form (pp)':<22}"
)
FLAG_HEADER = (
    f"{'model':<16}{'AP':>6} {'thr':>5}  {'caught':>7} {'fixed':>7} {'prec':>7}"
    f" {'false al.':>9}   {'with form: acc vs form (pp)':<28}  {'no form: acc vs majority (pp)'}"
)


def format_standalone(model: str, result: dict[str, Any]) -> str:
    ci = result.get("macro_f1_ci")
    ci_text = f"[{ci[0]:.3f}-{ci[1]:.3f}]" if ci else " " * 13
    return (
        f"{model:<16}{result['macro_f1']:>8.3f} {ci_text:<14}{_pct(result['accuracy'])}   "
        f"{_diff(result.get('vs_form', {}).get('accuracy'))}"
    )


def format_flagging(model: str, result: dict[str, Any]) -> str:
    f = result["flagging"]
    ap = "   -  " if f["average_precision"] is None else f"{f['average_precision']:6.3f}"
    thr = "  -  " if f["threshold"] is None else f"{f['threshold']:5.2f}"
    return (
        f"{model:<16}{ap} {thr}  {_pct(f['caught'])} {_pct(f['fixed'])} "
        f"{_pct(f['flag_precision'])} {_pct(f['false_alarm_rate']):>9}   "
        f"{_subset_cell(f, 'with_form'):<28}  {_subset_cell(f, 'no_form')}"
    )


def _subset_cell(flagging: dict[str, Any], subset: str) -> str:
    entry = flagging["combined"]["subsets"][subset]
    if not entry["n"]:
        return "-"
    return f"{_pct(entry['accuracy'])} {_diff(entry.get('vs_form', {}).get('accuracy'))}"


def print_tables(repo: str, results: dict[str, dict[str, Any]]) -> None:
    """Summary of test results of several models for one repository."""
    test = next(iter(results.values()))
    f = test["flagging"]
    no_form = f["combined"]["subsets"]["no_form"]["n"]
    print(
        f"\n=== {repo}: test {test['n']} issues: {f['filed']} filed via a form "
        f"({f['misfiled']} misfiled), {no_form} without one"
    )
    print("\nStandalone (model replaces the form):")
    print(STANDALONE_HEADER)
    for model, result in results.items():
        print(format_standalone(model, result))
    print(f"\nForm + flag (target flag precision {TARGET_FLAG_PRECISION:.0%} on validation):")
    print(FLAG_HEADER)
    for model, result in results.items():
        print(format_flagging(model, result))
