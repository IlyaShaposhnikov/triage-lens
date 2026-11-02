"""Build and publish the artifacts the demo app needs.

The demo runs on Streamlit Community Cloud, where ``data/`` is not available.
Everything it needs is built here and published as a Hugging Face dataset:

* ``<owner__name>/issues.jsonl.gz`` — every issue of the repository: number,
  title, state, labels, creation time, URL and a short snippet of the cleaned text;
* ``<owner__name>/vectors.<encoder>.f16.npy`` — the search vectors in the same
  order, float16 (half the size; ranking is unaffected at this precision);
* ``<owner__name>/classifier.joblib`` — TF-IDF + logistic regression with the C
  chosen on validation, refitted on train + validation. It is the model the demo
  uses for the issue type: on test it was within a few points of fine-tuned
  DistilBERT and needs no GPU or torch;
* ``manifest.json`` — encoder, versions, sizes and headline metrics;
* ``README.md`` — the dataset card.

Vectors are read from the embedding cache written by ``triage_lens.search``; run
it with the same ``--model`` first. The classifier file is a pickle, so the app
pins the scikit-learn version recorded in the manifest.

Usage:
    python -m triage_lens.demo_data build astral-sh/uv pandas-dev/pandas
    python -m triage_lens.demo_data push --repo-id USER/triage-lens-demo
"""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from triage_lens.baselines import make_logreg, make_vectorizer
from triage_lens.collect import load_latest
from triage_lens.dataset import CLASSES, REPOS
from triage_lens.embeddings import load_cache, model_slug, text_hash
from triage_lens.evaluate import load_dataset
from triage_lens.predictions import file_hash
from triage_lens.search import Corpus

DEMO_ENCODER = "BAAI/bge-small-en-v1.5"
SNIPPET_CHARS = 280
MAX_LABELS = 8


class MissingVectors(RuntimeError):
    pass


def issue_rows(corpus: Corpus, issues: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Display data for every issue, in corpus (creation time) order."""
    rows = []
    for number, text in zip(corpus.numbers, corpus.texts, strict=True):
        issue = issues[int(number)]
        body = text.split("\n\n", 1)[1] if "\n\n" in text else ""
        rows.append(
            {
                "number": int(number),
                "title": issue["title"],
                "state": issue["state"],
                "state_reason": issue.get("state_reason"),
                "labels": issue["labels"][:MAX_LABELS],
                "created_at": issue["created_at"],
                "url": issue["url"],
                "snippet": " ".join(body.split())[:SNIPPET_CHARS],
            }
        )
    return rows


def corpus_vectors(corpus: Corpus, cache_path: Path) -> np.ndarray:
    """Cached vectors in corpus order, L2-normalised, float16."""
    cache = load_cache(cache_path)
    keys = [(int(n), text_hash(t)) for n, t in zip(corpus.numbers, corpus.texts, strict=True)]
    missing = [number for number, key in zip(corpus.numbers, keys, strict=True) if key not in cache]
    if missing:
        raise MissingVectors(
            f"{len(missing)} issues have no cached vector (e.g. #{missing[0]}) in {cache_path}; "
            "run `python -m triage_lens.search ... --model` with this encoder first"
        )
    vectors = np.stack([cache[key] for key in keys]).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    return vectors.astype(np.float16)


def fit_demo_classifier(splits: dict[str, list[dict[str, Any]]], c: float) -> Pipeline:
    """TF-IDF + LogReg with the validated C, refitted on train + validation."""
    records = splits["train"] + splits["validation"]
    model = Pipeline([("tfidf", make_vectorizer()), ("clf", make_logreg(c))])
    return model.fit([r["text"] for r in records], [r["label"] for r in records])


def headline_metrics(results_dir: Path, stem: str, slug: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tfidf = results_dir / stem / "tfidf_logreg.json"
    if tfidf.exists():
        test = json.loads(tfidf.read_text(encoding="utf-8"))["scores"]["test"]
        out["classifier_test"] = {"accuracy": test["accuracy"], "macro_f1": test["macro_f1"]}
    search = results_dir / stem / f"search_{slug}.json"
    if search.exists():
        dense = json.loads(search.read_text(encoding="utf-8"))["methods"]["dense_exact"]
        out["search"] = {k: dense[k] for k in ("recall@1", "recall@5", "recall@10", "mrr")}
    return out


def build_repo(
    repo: str,
    raw_dir: Path,
    processed_dir: Path,
    cache_dir: Path,
    results_dir: Path,
    out_dir: Path,
    encoder: str,
) -> dict[str, Any]:
    stem = repo.replace("/", "__")
    slug = model_slug(encoder)
    issues = load_latest(raw_dir / f"{stem}.jsonl")
    corpus = Corpus(repo, issues)
    target = out_dir / stem
    target.mkdir(parents=True, exist_ok=True)

    with gzip.open(target / "issues.jsonl.gz", "wt", encoding="utf-8") as f:
        for row in issue_rows(corpus, issues):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    vectors = corpus_vectors(corpus, cache_dir / f"{stem}.{slug}.npz")
    np.save(target / f"vectors.{slug}.f16.npy", vectors)

    tfidf_result = json.loads((results_dir / stem / "tfidf_logreg.json").read_text("utf-8"))
    c = tfidf_result["params"]["C"]
    dataset_path = processed_dir / f"{stem}.dataset.jsonl"
    classifier = fit_demo_classifier(load_dataset(dataset_path), c)
    joblib.dump(classifier, target / "classifier.joblib", compress=3)

    return {
        "issues": len(corpus.numbers),
        "vectors_file": f"{stem}/vectors.{slug}.f16.npy",
        "vector_dim": int(vectors.shape[1]),
        "classifier": {"model": "tfidf_logreg", "C": c, "trained_on": "train+validation"},
        "dataset_sha1": file_hash(dataset_path),
        "newest_issue": max(i["created_at"] for i in issues.values()),
        "metrics": headline_metrics(results_dir, stem, slug),
    }


DATASET_CARD = """---
license: other
pretty_name: triage-lens demo artifacts
task_categories: [text-classification, sentence-similarity]
language: [en]
---

# triage-lens demo artifacts

Search vectors, issue metadata and a small classifier used by the
[triage-lens](https://github.com/IlyaShaposhnikov/triage-lens) demo, which
predicts the type of a new GitHub issue and finds earlier issues that may be
duplicates of it.

| Repository | Issues | Newest issue |
|---|---|---|
{rows}

Per repository:

* `issues.jsonl.gz` — number, title, state, labels, creation time, URL and the
  first {snippet} characters of the issue text with issue-form boilerplate removed;
* `vectors.{slug}.f16.npy` — `{encoder}` embeddings of the cleaned issue texts,
  L2-normalised, float16, in the same order as `issues.jsonl.gz`;
* `classifier.joblib` — TF-IDF + logistic regression predicting bug / feature /
  question / docs (scikit-learn {sklearn_version}; load only with that version).

Issues were collected from the public GitHub API. Their text belongs to its
authors and is subject to GitHub's terms; this dataset only redistributes short
snippets for search. See the project README for how everything was built and
evaluated.
"""


def write_card(out_dir: Path, manifest: dict[str, Any]) -> None:
    rows = "\n".join(
        f"| [{repo}](https://github.com/{repo}) | {info['issues']:,} "
        f"| {info['newest_issue'][:10]} |"
        for repo, info in manifest["repos"].items()
    )
    card = DATASET_CARD.format(
        rows=rows,
        snippet=SNIPPET_CHARS,
        slug=model_slug(manifest["encoder"]),
        encoder=manifest["encoder"],
        sklearn_version=manifest["scikit_learn"],
    )
    (out_dir / "README.md").write_text(card, encoding="utf-8")


def cmd_build(args) -> None:
    manifest: dict[str, Any] = {
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "encoder": args.encoder,
        "classes": list(CLASSES),
        "scikit_learn": sklearn.__version__,
        "repos": {},
    }
    for repo in args.repos:
        manifest["repos"][repo] = build_repo(
            repo, args.raw, args.processed, args.cache, args.results, args.out, args.encoder
        )
        info = manifest["repos"][repo]
        print(
            f"{repo}: {info['issues']} issues, vectors {info['vector_dim']}-d, "
            f"classifier C={info['classifier']['C']}"
        )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_card(args.out, manifest)
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"built {args.out} ({size / 1e6:.1f} MB)")


def cmd_push(args) -> None:
    from huggingface_hub import HfApi

    if not (args.out / "manifest.json").exists():
        raise SystemExit(f"Nothing to push: run `build` first ({args.out} has no manifest.json)")
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(args.out),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.message,
    )
    print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build and publish demo artifacts.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("repos", nargs="+", choices=sorted(REPOS))
    b.add_argument("--encoder", default=DEMO_ENCODER)
    b.add_argument("--raw", type=Path, default=Path("data/raw"))
    b.add_argument("--processed", type=Path, default=Path("data/processed"))
    b.add_argument("--cache", type=Path, default=Path("data/embeddings"))
    b.add_argument("--results", type=Path, default=Path("results"))
    b.add_argument("--out", type=Path, default=Path("build/demo"))
    p = sub.add_parser("push")
    p.add_argument("--repo-id", required=True, help="USER/NAME of the Hugging Face dataset")
    p.add_argument("--out", type=Path, default=Path("build/demo"))
    p.add_argument("--message", default="Update demo artifacts")
    args = parser.parse_args(argv)
    {"build": cmd_build, "push": cmd_push}[args.command](args)


if __name__ == "__main__":
    main()
