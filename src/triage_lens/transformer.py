"""Transformer models for issue-type classification, meant to run on a cloud GPU.

Two commands, both writing class probabilities for the validation and test
splits to ``predictions/<owner__name>/<model>.npz`` (see ``predictions.py``);
scoring happens locally with ``python -m triage_lens.predictions``.

``finetune`` — fine-tunes a pretrained encoder (DistilBERT by default) end to end:

* class-weighted cross-entropy (balanced weights, as for the linear models);
* evaluation on validation after every epoch, early stopping on macro-F1 and the
  best epoch restored at the end, so test is never used for any choice;
* dynamic padding, fp16 on GPU, a fixed seed.

``zeroshot`` — no training: an NLI model scores the hypothesis "This GitHub issue
is {a bug report / a feature request / a usage question / a documentation
problem}." Texts are cut to ``ZS_MAX_CHARS`` to bound the cost.

Usage (Kaggle notebook; ``--limit 200 --epochs 1`` makes a one-minute smoke run):
    python -m triage_lens.transformer finetune astral-sh/uv pandas-dev/pandas --data DIR
    python -m triage_lens.transformer zeroshot astral-sh/uv pandas-dev/pandas --data DIR
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from triage_lens.dataset import CLASSES, REPOS
from triage_lens.embeddings import INPUTS, model_slug
from triage_lens.evaluate import accuracy, load_dataset, macro_f1
from triage_lens.predictions import file_hash, save_predictions

DEFAULT_FINETUNE_MODEL = "distilbert/distilbert-base-uncased"
DEFAULT_ZEROSHOT_MODEL = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This GitHub issue is {}."
ZS_LABELS = {
    "bug": "a bug report",
    "feature": "a feature request",
    "question": "a usage question",
    "docs": "a documentation problem",
}
ZS_MAX_CHARS = 2000
SEED = 42

log = logging.getLogger("triage_lens.transformer")


# --------------------------------------------------------------------------- helpers


def balanced_class_weights(labels: Sequence[str]) -> np.ndarray:
    """n / (k * count) per class, like scikit-learn's class_weight="balanced"."""
    counts = Counter(labels)
    missing = [cls for cls in CLASSES if counts[cls] == 0]
    if missing:
        raise ValueError(f"no training examples for {missing}")
    return np.array([len(labels) / (len(CLASSES) * counts[cls]) for cls in CLASSES])


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def limited(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """First ``limit`` records of each class, for smoke runs that still see every class."""
    if not limit:
        return records
    per_class = max(1, limit // len(CLASSES))
    seen: Counter = Counter()
    kept = []
    for record in records:
        if seen[record["label"]] < per_class:
            seen[record["label"]] += 1
            kept.append(record)
    return kept


def zeroshot_probs(results: Sequence[dict[str, Any]]) -> np.ndarray:
    """Pipeline outputs (labels sorted by score) -> probabilities in CLASSES order."""
    by_text = {text: cls for cls, text in ZS_LABELS.items()}
    probs = np.zeros((len(results), len(CLASSES)))
    for i, result in enumerate(results):
        for label, score in zip(result["labels"], result["scores"], strict=True):
            probs[i, CLASSES.index(by_text[label])] = score
    return probs


def model_name_for(kind: str, model: str, input_kind: str) -> str:
    suffix = "" if input_kind == "full" else f"_{input_kind}"
    return f"{kind}_{model_slug(model)}{suffix}"


def load_splits(repo: str, data_dir: Path, limit: int | None):
    path = data_dir / f"{repo.replace('/', '__')}.dataset.jsonl"
    splits = load_dataset(path)
    return {name: limited(records, limit) for name, records in splits.items()}, path


def write(
    out_dir: Path,
    repo: str,
    name: str,
    dataset_path: Path,
    splits: dict[str, list[dict[str, Any]]],
    probs: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> Path:
    path = out_dir / repo.replace("/", "__") / f"{name}.npz"
    payload = {split: ([r["number"] for r in splits[split]], probs[split]) for split in probs}
    meta = {**meta, "dataset_sha1": file_hash(dataset_path)}
    save_predictions(path, payload, meta)
    for split, p in probs.items():
        y = [r["label"] for r in splits[split]]
        pred = [CLASSES[i] for i in p.argmax(axis=1)]
        log.info(
            "%s %s: macro-F1 %.3f, accuracy %.3f", name, split, macro_f1(y, pred), accuracy(y, pred)
        )
    log.info("saved %s", path)
    return path


# --------------------------------------------------------------------------- fine-tuning


def finetune(
    repo: str,
    data_dir: Path,
    out_dir: Path,
    model_name: str = DEFAULT_FINETUNE_MODEL,
    input_kind: str = "full",
    max_length: int = 512,
    epochs: int = 5,
    learning_rate: float = 3e-5,
    batch_size: int = 16,
    patience: int = 2,
    limit: int | None = None,
    work_dir: Path = Path("checkpoints"),
) -> Path:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(SEED)
    splits, dataset_path = load_splits(repo, data_dir, limit)
    to_text = INPUTS[input_kind]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class Encoded(torch.utils.data.Dataset):
        def __init__(self, records):
            self.enc = tokenizer(
                [to_text(r) for r in records], truncation=True, max_length=max_length
            )
            self.labels = [CLASSES.index(r["label"]) for r in records]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {key: values[i] for key, values in self.enc.items()}
            item["labels"] = self.labels[i]
            return item

    data = {name: Encoded(splits[name]) for name in ("train", "validation", "test")}
    weights = torch.tensor(
        balanced_class_weights([r["label"] for r in splits["train"]]), dtype=torch.float
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=weights.to(outputs.logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = [CLASSES[i] for i in np.argmax(logits, axis=1)]
        true = [CLASSES[i] for i in labels]
        return {"macro_f1": macro_f1(true, pred), "accuracy": accuracy(true, pred)}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CLASSES),
        id2label=dict(enumerate(CLASSES)),
        label2id={cls: i for i, cls in enumerate(CLASSES)},
    )
    steps_per_epoch = math.ceil(len(data["train"]) / batch_size)
    name = model_name_for("ft", model_name, input_kind)
    args = TrainingArguments(
        output_dir=str(work_dir / repo.replace("/", "__") / name),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=int(0.1 * steps_per_epoch * epochs),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started

    probs = {
        split: softmax(trainer.predict(data[split]).predictions) for split in ("validation", "test")
    }
    history = [
        {k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items()}
        for entry in trainer.state.log_history
        if "eval_macro_f1" in entry
    ]
    meta = {
        "kind": "finetune",
        "base_model": model_name,
        "input": input_kind,
        "max_length": max_length,
        "epochs_max": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "class_weights": [round(float(w), 4) for w in weights],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "epochs_history": history,
        "train_seconds": round(train_seconds),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "limit": limit,
    }
    return write(out_dir, repo, name, dataset_path, splits, probs, meta)


# --------------------------------------------------------------------------- zero-shot


def zeroshot(
    repo: str,
    data_dir: Path,
    out_dir: Path,
    model_name: str = DEFAULT_ZEROSHOT_MODEL,
    input_kind: str = "full",
    batch_size: int = 16,
    limit: int | None = None,
) -> Path:
    import torch
    from transformers import pipeline

    splits, dataset_path = load_splits(repo, data_dir, limit)
    to_text = INPUTS[input_kind]
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", model=model_name, device=device)
    if device == 0:
        classifier.model.half()

    probs = {}
    started = time.perf_counter()
    for split in ("validation", "test"):
        texts = [to_text(r)[:ZS_MAX_CHARS] for r in splits[split]]
        results = classifier(
            texts,
            candidate_labels=list(ZS_LABELS.values()),
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
            batch_size=batch_size,
        )
        probs[split] = zeroshot_probs(results if isinstance(results, list) else [results])
    meta = {
        "kind": "zeroshot",
        "base_model": model_name,
        "input": input_kind,
        "hypothesis_template": HYPOTHESIS_TEMPLATE,
        "labels": ZS_LABELS,
        "max_chars": ZS_MAX_CHARS,
        "seconds": round(time.perf_counter() - started),
        "device": torch.cuda.get_device_name(0) if device == 0 else "cpu",
        "limit": limit,
    }
    name = model_name_for("zs", model_name, input_kind)
    return write(out_dir, repo, name, dataset_path, splits, probs, meta)


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tuned and zero-shot transformers.")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("finetune", "zeroshot"):
        p = sub.add_parser(command)
        p.add_argument("repos", nargs="+", choices=sorted(REPOS))
        p.add_argument("--data", type=Path, default=Path("data/processed"))
        p.add_argument("--out", type=Path, default=Path("predictions"))
        p.add_argument("--input", choices=sorted(INPUTS), default="full")
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--limit", type=int, default=None, help="examples per split (smoke run)")
    ft = sub.choices["finetune"]
    ft.add_argument("--model", default=DEFAULT_FINETUNE_MODEL)
    ft.add_argument("--epochs", type=int, default=5)
    ft.add_argument("--lr", type=float, default=3e-5)
    ft.add_argument("--max-length", type=int, default=512)
    ft.add_argument("--work-dir", type=Path, default=Path("checkpoints"))
    sub.choices["zeroshot"].add_argument("--model", default=DEFAULT_ZEROSHOT_MODEL)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for repo in args.repos:
        if args.command == "finetune":
            finetune(
                repo,
                args.data,
                args.out,
                args.model,
                args.input,
                args.max_length,
                args.epochs,
                args.lr,
                args.batch_size,
                limit=args.limit,
                work_dir=args.work_dir,
            )
        else:
            zeroshot(repo, args.data, args.out, args.model, args.input, args.batch_size, args.limit)
    log.info("done: %s", json.dumps([str(p) for p in sorted(args.out.rglob("*.npz"))]))


if __name__ == "__main__":
    main()
