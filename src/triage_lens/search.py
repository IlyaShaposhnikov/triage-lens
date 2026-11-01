"""Duplicate search: find the original of a new issue among earlier issues.

Ground truth is the reviewed duplicate -> original pairs (``duplicates.py``,
precision checked in ``review.py``). Each duplicate is a query; the index holds
every issue of the repository created **before** it, as at the moment the
duplicate was filed. Only the original counts as a hit; other duplicates of the
same original do not. Pairs whose original is not an issue, is missing, or was
created after the duplicate are dropped and counted.

Methods, all with no training and no tuning on the evaluation pairs:

* ``bm25`` — keyword search: Okapi BM25 (k1=1.2, b=0.75) over word tokens,
  implemented on a sparse matrix so scoring the whole corpus is one product;
* ``dense_exact`` — cosine similarity of sentence embeddings, exact (numpy);
* ``dense_chroma`` — the same vectors served by a Chroma vector database with an
  HNSW index and a ``created < query`` metadata filter: what a deployed system
  would run. Comparing it with ``dense_exact`` shows what approximation costs;
* ``hybrid`` — Reciprocal Rank Fusion of the BM25 and dense rankings (k=60).

Metrics: recall@1/5/10 (the original is among the first k results) and MRR over
the top 100, with bootstrap 95% intervals and paired differences to BM25.

Texts are the cleaned issue texts used for classification (``dataset.cleaned``),
so embeddings already cached for the classification dataset are reused.

Usage:
    pip install -e ".[embeddings,search]"
    python -m triage_lens.search astral-sh/uv pandas-dev/pandas
    python -m triage_lens.search astral-sh/uv --backend numpy   # without Chroma
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from triage_lens.collect import load_latest
from triage_lens.dataset import REPOS, cleaned
from triage_lens.embeddings import (
    DEFAULT_MODEL,
    Encoder,
    embed_records,
    load_sentence_transformer,
    model_slug,
)
from triage_lens.evaluate import BOOTSTRAP_SAMPLES

TOP_N = 100
RECALL_AT = (1, 5, 10)
RRF_K = 60
TOKEN_PATTERN = r"(?u)\b\w\w+\b"

log = logging.getLogger("triage_lens.search")


# --------------------------------------------------------------------------- corpus


def timestamp(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


class Corpus:
    """All issues of a repository, ordered by creation time."""

    def __init__(self, repo: str, issues: dict[int, dict[str, Any]]):
        ordered = sorted(issues.values(), key=lambda i: (i["created_at"], i["number"]))
        self.numbers = np.array([i["number"] for i in ordered])
        self.created = np.array([timestamp(i["created_at"]) for i in ordered])
        self.texts = [cleaned(repo, i)[2] for i in ordered]
        self.position = {int(n): k for k, n in enumerate(self.numbers)}

    def records(self) -> list[dict[str, Any]]:
        return [
            {"number": int(n), "text": t} for n, t in zip(self.numbers, self.texts, strict=True)
        ]

    def before(self, position: int) -> np.ndarray:
        """Mask of issues created strictly before the issue at ``position``."""
        return self.created < self.created[position]


def evaluation_pairs(
    pairs: Sequence[dict[str, Any]], corpus: Corpus
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """(duplicate, original) pairs usable as queries, and counts of dropped ones."""
    kept, dropped = [], {"original_not_issue": 0, "missing": 0, "original_later": 0}
    for pair in pairs:
        dup, orig = pair["duplicate"], pair["original"]
        if not pair["original_is_issue"]:
            dropped["original_not_issue"] += 1
        elif dup not in corpus.position or orig not in corpus.position:
            dropped["missing"] += 1
        elif corpus.created[corpus.position[orig]] >= corpus.created[corpus.position[dup]]:
            dropped["original_later"] += 1
        else:
            kept.append((dup, orig))
    return kept, dropped


# --------------------------------------------------------------------------- retrieval


def top_k(scores: np.ndarray, mask: np.ndarray, k: int = TOP_N) -> np.ndarray:
    """Positions of the ``k`` best scores among allowed positions, best first."""
    allowed = np.flatnonzero(mask)
    if allowed.size == 0:
        return allowed
    s = scores[allowed]
    k = min(k, allowed.size)
    best = np.argpartition(-s, k - 1)[:k]
    return allowed[best[np.argsort(-s[best], kind="stable")]]


class BM25:
    """Okapi BM25 with document weights precomputed in a sparse matrix."""

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75):
        self.vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN, dtype=np.float32)
        tf = self.vectorizer.fit_transform(texts).tocsr()
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        avg_length = lengths.mean() if n_docs else 1.0
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        saturation = tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * lengths[rows] / avg_length))
        self.weights = csr_matrix((saturation * idf[tf.indices], tf.indices, tf.indptr), tf.shape)

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        q.data[:] = 1.0  # each query term counts once
        return np.asarray((self.weights @ q.T).todense()).ravel()


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K, n: int = TOP_N) -> list[int]:
    """Reciprocal Rank Fusion: sum of 1 / (k + rank) over the input rankings."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda item: (-fused[item], item))[:n]


class ChromaIndex:
    """Vectors in a persistent Chroma collection, queried with a creation-time filter.

    Vectors are L2-normalised, so Chroma's default L2 distance ranks exactly like
    cosine similarity.
    """

    def __init__(self, path: Path, name: str, numbers, vectors: np.ndarray, created):
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SystemExit('Chroma is not installed: pip install -e ".[search]"') from exc
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(name)
        if self.collection.count() != len(numbers):
            log.info("building Chroma collection %s (%d vectors)", name, len(numbers))
            self.client.delete_collection(name)
            self.collection = self.client.create_collection(name)
            batch = self.client.get_max_batch_size()
            for start in range(0, len(numbers), batch):
                end = start + batch
                self.collection.add(
                    ids=[str(int(n)) for n in numbers[start:end]],
                    embeddings=vectors[start:end].tolist(),
                    metadatas=[{"created": int(t)} for t in created[start:end]],
                )

    def search(self, vector: np.ndarray, before: int, k: int = TOP_N) -> list[int]:
        result = self.collection.query(
            query_embeddings=[vector.tolist()],
            n_results=k,
            where={"created": {"$lt": int(before)}},
            include=["distances"],
        )
        return [int(i) for i in result["ids"][0]]


# --------------------------------------------------------------------------- metrics


def rank_of(target: int, ranking: Sequence[int]) -> int | None:
    for rank, item in enumerate(ranking, start=1):
        if item == target:
            return rank
    return None


def _as_array(ranks: Sequence[int | None]) -> np.ndarray:
    """Ranks as integers, 0 meaning "not in the top 100"."""
    return np.array([r if r is not None else 0 for r in ranks], dtype=int)


def _metrics(found: np.ndarray) -> dict[str, float]:
    hit = found > 0
    out = {f"recall@{k}": float(np.mean(hit & (found <= k))) for k in RECALL_AT}
    out["mrr"] = float(np.mean(np.where(hit, 1.0 / np.maximum(found, 1), 0.0)))
    return out


def metrics(ranks: Sequence[int | None]) -> dict[str, float]:
    return _metrics(_as_array(ranks))


def bootstrap(
    ranks: dict[str, list[int | None]],
    reference: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """95% intervals per method and metric, and paired differences to ``reference``.

    Every resample draws the same queries for all methods, so the interval of a
    difference reflects the paired design.
    """
    arrays = {method: _as_array(values) for method, values in ranks.items()}
    n = len(next(iter(arrays.values())))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[dict[str, float]]] = {method: [] for method in arrays}
    for _ in range(samples):
        idx = rng.integers(0, n, n)
        for method, found in arrays.items():
            draws[method].append(_metrics(found[idx]))

    reference_point = _metrics(arrays[reference])
    report = {}
    for method, found in arrays.items():
        point = _metrics(found)
        entry: dict[str, Any] = {"n": n, **point, "ci": {}, f"vs_{reference}": {}}
        for name in point:
            values = np.array([d[name] for d in draws[method]])
            diffs = values - np.array([d[name] for d in draws[reference]])
            entry["ci"][name] = np.percentile(values, [2.5, 97.5]).tolist()
            entry[f"vs_{reference}"][name] = {
                "diff": point[name] - reference_point[name],
                "ci": np.percentile(diffs, [2.5, 97.5]).tolist(),
            }
        report[method] = entry
    return report


# --------------------------------------------------------------------------- run


def evaluate_search(
    corpus: Corpus,
    pairs: Sequence[tuple[int, int]],
    vectors: np.ndarray,
    chroma: Callable[[np.ndarray, int], list[int]] | None = None,
) -> dict[str, list[int | None]]:
    """Rank of the original for every query and method (None: not in the top 100)."""
    bm25 = BM25(corpus.texts)
    ranks: dict[str, list[int | None]] = {"bm25": [], "dense_exact": [], "hybrid": []}
    if chroma is not None:
        ranks["dense_chroma"] = []
    for dup, orig in pairs:
        pos = corpus.position[dup]
        mask = corpus.before(pos)
        keyword = corpus.numbers[top_k(bm25.scores(corpus.texts[pos]), mask)].tolist()
        exact = corpus.numbers[top_k(vectors @ vectors[pos], mask)].tolist()
        dense = exact
        if chroma is not None:
            dense = chroma(vectors[pos], int(corpus.created[pos]))
            ranks["dense_chroma"].append(rank_of(orig, dense))
        ranks["bm25"].append(rank_of(orig, keyword))
        ranks["dense_exact"].append(rank_of(orig, exact))
        ranks["hybrid"].append(rank_of(orig, rrf([keyword, dense])))
    return ranks


def print_table(repo: str, n_corpus: int, report: dict[str, Any], dropped: dict[str, int]) -> None:
    n = next(iter(report.values()))["n"]
    print(f"\n=== {repo}: {n} queries over {n_corpus} issues; dropped pairs: {dropped}")
    print(
        f"{'method':<14}"
        + "".join(f"{m:>22}" for m in ("recall@1", "recall@5", "recall@10"))
        + f"{'MRR':>8}   recall@10 vs bm25 (pp)"
    )
    for method, r in report.items():
        cells = "".join(
            f"{100 * r[m]:>7.1f}% [{100 * r['ci'][m][0]:4.1f}-{100 * r['ci'][m][1]:4.1f}]"
            for m in ("recall@1", "recall@5", "recall@10")
        )
        d = r["vs_bm25"]["recall@10"]
        diff = f"{100 * d['diff']:+5.1f} [{100 * d['ci'][0]:+5.1f},{100 * d['ci'][1]:+5.1f}]"
        print(f"{method:<14}{cells}{r['mrr']:>8.3f}   {diff}")


def run(
    repo: str,
    encode: Encoder,
    model_name: str,
    raw_dir: Path,
    processed_dir: Path,
    cache_dir: Path,
    results_dir: Path,
    backend: str,
    chroma_dir: Path,
) -> dict[str, Any]:
    stem = repo.replace("/", "__")
    corpus = Corpus(repo, load_latest(raw_dir / f"{stem}.jsonl"))
    with (processed_dir / f"{stem}.duplicate_pairs.jsonl").open(encoding="utf-8") as f:
        all_pairs = [json.loads(line) for line in f if line.strip()]
    pairs, dropped = evaluation_pairs(all_pairs, corpus)

    slug = model_slug(model_name)
    vectors = embed_records(corpus.records(), encode, cache_dir / f"{stem}.{slug}.npz")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)

    chroma = None
    if backend == "chroma":
        index = ChromaIndex(
            chroma_dir, f"{stem}__{slug}"[:63], corpus.numbers, vectors, corpus.created
        )
        chroma = index.search
    ranks = evaluate_search(corpus, pairs, vectors, chroma)
    report = bootstrap(ranks, reference="bm25")
    print_table(repo, len(corpus.numbers), report, dropped)

    path = results_dir / stem / "search.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo": repo,
        "encoder": model_name,
        "backend": backend,
        "corpus_size": len(corpus.numbers),
        "dropped_pairs": dropped,
        "methods": report,
        "ranks": {
            "pairs": [list(p) for p in pairs],
            **{method: values for method, values in ranks.items()},
        },
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return report


def main(argv: list[str] | None = None, encoder_factory=load_sentence_transformer) -> None:
    parser = argparse.ArgumentParser(description="Evaluate duplicate search.")
    parser.add_argument("repos", nargs="+", choices=sorted(REPOS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backend", choices=("chroma", "numpy"), default="chroma")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--chroma", type=Path, default=Path("data/chroma"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--rebuild-chroma", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.rebuild_chroma and args.chroma.exists():
        shutil.rmtree(args.chroma)
    encode = encoder_factory(args.model, args.batch_size)
    for repo in args.repos:
        run(
            repo,
            encode,
            args.model,
            args.raw,
            args.processed,
            args.cache,
            args.results,
            args.backend,
            args.chroma,
        )


if __name__ == "__main__":
    main()
