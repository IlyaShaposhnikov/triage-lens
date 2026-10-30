"""Sentence embeddings + logistic regression for issue-type classification.

Each issue text is turned into a fixed-size vector by a pretrained
sentence-transformers model (no fine-tuning), and a logistic regression with
balanced class weights is trained on top. C is chosen on validation by
macro-F1, as for the TF-IDF baseline, and the model is scored by the shared
``score_model``, so the results line up with the baselines.

Encoding is the slow part (minutes on a CPU), so vectors are cached per
repository and model in ``data/embeddings``. The cache is keyed by issue number
and a hash of the text: a later run encodes only new or changed issues. The same
vectors are reused later for duplicate search.

``--input prose`` embeds the title and the author's prose without code blocks,
REPL lines and tracebacks (see ``dataset.prose_text``). The encoder reads only
the first 256 tokens, and in pandas bug reports the code comes first.

Usage:
    pip install -e ".[embeddings]"
    python -m triage_lens.embeddings astral-sh/uv pandas-dev/pandas
    python -m triage_lens.embeddings astral-sh/uv pandas-dev/pandas --input prose
    python -m triage_lens.embeddings astral-sh/uv --model sentence-transformers/all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from triage_lens.dataset import CLASSES, REPOS, prose_text
from triage_lens.evaluate import load_dataset, macro_f1, print_tables, save_result, score_model

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
C_GRID = (0.25, 1.0, 4.0, 16.0)
SEED = 42

Encoder = Callable[[list[str]], np.ndarray]
INPUTS: dict[str, Callable[[dict[str, Any]], str]] = {
    "full": lambda record: record["text"],
    "prose": prose_text,
}
log = logging.getLogger("triage_lens.embeddings")


# --------------------------------------------------------------------------- encoding


def model_slug(model_name: str) -> str:
    """Short file-safe name: "sentence-transformers/all-MiniLM-L6-v2" -> "all-minilm-l6-v2"."""
    return model_name.rsplit("/", 1)[-1].lower()


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_sentence_transformer(model_name: str, batch_size: int) -> Encoder:
    """Encoder backed by sentence-transformers (imported lazily: it pulls in torch)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    log.info("%s: max sequence length %d tokens", model_name, model.max_seq_length)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    return encode


def load_cache(path: Path) -> dict[tuple[int, str], np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path)
    return {
        (int(n), str(h)): v
        for n, h, v in zip(data["numbers"], data["hashes"], data["vectors"], strict=True)
    }


def save_cache(path: Path, cache: dict[tuple[int, str], np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(cache)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        tmp,
        numbers=np.array([n for n, _ in keys]),
        hashes=np.array([h for _, h in keys]),
        vectors=np.stack([cache[k] for k in keys]).astype(np.float32),
    )
    tmp.replace(path)


def embed_records(
    records: Sequence[dict[str, Any]],
    encode: Encoder,
    cache_path: Path,
    to_text: Callable[[dict[str, Any]], str] = INPUTS["full"],
) -> np.ndarray:
    """Vectors for ``records`` in order, encoding only what the cache lacks."""
    cache = load_cache(cache_path)
    texts = [to_text(r) for r in records]
    keys = [(r["number"], text_hash(text)) for r, text in zip(records, texts, strict=True)]
    missing = {key: text for key, text in zip(keys, texts, strict=True) if key not in cache}
    if missing:
        log.info("encoding %d texts (%d cached)", len(missing), len(keys) - len(missing))
        started = time.perf_counter()
        vectors = encode(list(missing.values()))
        log.info("encoded in %.0fs", time.perf_counter() - started)
        cache.update(zip(missing, np.asarray(vectors), strict=True))
        save_cache(cache_path, cache)
    else:
        log.info("all %d texts found in cache", len(keys))
    return np.stack([cache[key] for key in keys])


# --------------------------------------------------------------------------- classifier


def fit_logreg(
    x_train: np.ndarray, y_train: Sequence[str], x_val: np.ndarray, y_val: Sequence[str]
) -> tuple[LogisticRegression, dict[str, Any]]:
    """Pick C on validation by macro-F1, return the model trained with it."""
    best, search = None, {}
    for c in C_GRID:
        model = LogisticRegression(
            C=c, class_weight="balanced", max_iter=3000, random_state=SEED
        ).fit(x_train, y_train)
        score = macro_f1(y_val, model.predict(x_val))
        search[str(c)] = round(score, 4)
        if best is None or score > best[0]:
            best = (score, c, model)
    _, c, model = best
    return model, {"C": c, "validation_macro_f1_by_C": search}


def proba_in_class_order(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    order = [list(model.classes_).index(cls) for cls in CLASSES]
    return model.predict_proba(x)[:, order]


# --------------------------------------------------------------------------- CLI


def load_other_results(results_dir: Path, repo: str, names: Sequence[str]) -> dict[str, Any]:
    """Test scores of models trained earlier, to print them next to the new one."""
    found = {}
    for name in names:
        path = results_dir / repo.replace("/", "__") / f"{name}.json"
        if path.exists():
            found[name] = json.loads(path.read_text(encoding="utf-8"))["scores"]["test"]
    return found


def run(
    repo: str,
    model_name: str,
    encode: Encoder,
    data_dir: Path,
    cache_dir: Path,
    results_dir: Path,
    input_kind: str = "full",
) -> dict[str, Any]:
    stem = repo.replace("/", "__")
    slug = model_slug(model_name)
    suffix = "" if input_kind == "full" else f"_{input_kind}"
    splits = load_dataset(data_dir / f"{stem}.dataset.jsonl")
    records = [r for split in ("train", "validation", "test") for r in splits[split]]
    vectors = embed_records(
        records, encode, cache_dir / f"{stem}.{slug}{suffix}.npz", INPUTS[input_kind]
    )

    by_number = {r["number"]: v for r, v in zip(records, vectors, strict=True)}

    def matrix(part):
        return np.stack([by_number[r["number"]] for r in part])

    train, validation = splits["train"], splits["validation"]
    classifier, params = fit_logreg(
        matrix(train),
        [r["label"] for r in train],
        matrix(validation),
        [r["label"] for r in validation],
    )
    params = {"encoder": model_name, "input": input_kind, "dim": int(vectors.shape[1]), **params}

    name = f"emb_{slug}{suffix}"
    scores = score_model(name, lambda part: proba_in_class_order(classifier, matrix(part)), splits)
    save_result(results_dir, repo, name, scores, params)

    others = ["trust_form", "tfidf_logreg"] + ([f"emb_{slug}"] if suffix else [])
    table = load_other_results(results_dir, repo, others)
    table[name] = scores["test"]
    print_tables(repo, table)
    print(
        f"\n{name}: C={params['C']} (validation macro-F1 by C: "
        f"{params['validation_macro_f1_by_C']})"
    )
    return scores


def main(argv: list[str] | None = None, encoder_factory=load_sentence_transformer) -> None:
    parser = argparse.ArgumentParser(description="Sentence embeddings + logistic regression.")
    parser.add_argument("repos", nargs="+", choices=sorted(REPOS), help="repositories")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model")
    parser.add_argument("--input", choices=sorted(INPUTS), default="full", help="text to embed")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    encode = encoder_factory(args.model, args.batch_size)
    for repo in args.repos:
        run(repo, args.model, encode, args.data, args.cache, args.results, args.input)


if __name__ == "__main__":
    main()
