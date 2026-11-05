"""Incremental refresh of the published demo index, meant for a scheduled workflow.

The workflow runner has none of the local ``data/``, so the refresh starts from
the published Hugging Face dataset instead:

1. download the dataset (issues, vectors, classifier, manifest);
2. ask GitHub for every issue created **or changed** since the last refresh
   (a small overlap guards against clock edges);
3. update changed issues in place (new state, labels, title) and append new
   ones; only issues whose cleaned text changed are encoded again;
4. check that the result loads and answers a query, then publish it.

The classifier is not retrained: its evaluation belongs to the fixed time split,
and a type model does not go stale in a month the way a search index does.

Usage (a GitHub token with read access to public repositories is enough):
    export GITHUB_TOKEN=...
    python -m triage_lens.refresh --repo-id USER/triage-lens-demo           # dry run
    python -m triage_lens.refresh --repo-id USER/triage-lens-demo --push    # publish
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import requests

from triage_lens.app_core import DemoAssets
from triage_lens.collect import PAGE_SIZE, make_session, post_graphql, to_record
from triage_lens.dataset import cleaned
from triage_lens.demo_data import issue_row, write_card
from triage_lens.embeddings import Encoder, load_sentence_transformer

OVERLAP = timedelta(days=1)

log = logging.getLogger("triage_lens.refresh")


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def fetch_updated(
    session: requests.Session,
    repo: str,
    since: str,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Every issue of ``repo`` created or updated at or after ``since``."""
    owner, name = repo.split("/")
    records, cursor = [], None
    while True:
        variables = {
            "owner": owner,
            "name": name,
            "first": PAGE_SIZE,
            "after": cursor,
            "since": since,
        }
        data = post_graphql(session, variables, sleep)
        if data.get("repository") is None:
            raise RuntimeError(f"Repository {repo} not found or not accessible")
        issues = data["repository"]["issues"]
        records += [to_record(node) for node in issues["nodes"]]
        if not issues["pageInfo"]["hasNextPage"]:
            return records
        cursor = issues["pageInfo"]["endCursor"]


def load_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def merge(
    repo: str,
    rows: list[dict[str, Any]],
    vectors: np.ndarray,
    updates: list[dict[str, Any]],
    encode: Encoder,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, int]]:
    """Apply fetched issues to the index; encode only new or edited texts.

    Changed issues keep their position, new ones are appended, so rows and
    vectors stay aligned.
    """
    position = {row["number"]: i for i, row in enumerate(rows)}
    rows = list(rows)
    to_encode: list[tuple[int, str]] = []  # (row position, text)
    counts = {"new": 0, "changed": 0, "re_encoded": 0}

    for issue in updates:
        text = cleaned(repo, issue)[2]
        row = issue_row(issue, text)
        pos = position.get(row["number"])
        if pos is None:
            pos = len(rows)
            position[row["number"]] = pos
            rows.append(row)
            counts["new"] += 1
            to_encode.append((pos, text))
            continue
        if rows[pos].get("text_sha1") != row["text_sha1"]:
            to_encode.append((pos, text))
        rows[pos] = row
        counts["changed"] += 1

    new_vectors = np.zeros((len(rows), vectors.shape[1]), dtype=np.float32)
    new_vectors[: len(vectors)] = vectors.astype(np.float32)
    if to_encode:
        encoded = np.asarray(encode([text for _, text in to_encode]), dtype=np.float32)
        encoded /= np.linalg.norm(encoded, axis=1, keepdims=True).clip(min=1e-12)
        for (pos, _), vector in zip(to_encode, encoded, strict=True):
            new_vectors[pos] = vector
    counts["re_encoded"] = len(to_encode) - counts["new"]
    return rows, new_vectors, counts


def refresh_repo(
    root: Path,
    repo: str,
    manifest: dict[str, Any],
    session: requests.Session,
    encode: Encoder,
    now: datetime,
) -> dict[str, int]:
    info = manifest["repos"][repo]
    folder = root / repo.replace("/", "__")
    watermark = info.get("updated_through") or info["newest_issue"]
    since = iso(parse_iso(watermark) - OVERLAP)

    updates = fetch_updated(session, repo, since)
    log.info("%s: %d issues created or updated since %s", repo, len(updates), since)

    rows = load_rows(folder / "issues.jsonl.gz")
    vectors = np.load(root / info["vectors_file"])
    rows, vectors, counts = merge(repo, rows, vectors, updates, encode)

    save_rows(folder / "issues.jsonl.gz", rows)
    np.save(root / info["vectors_file"], vectors.astype(np.float16))
    info["issues"] = len(rows)
    info["newest_issue"] = max(row["created_at"] for row in rows)
    newest_update = max([u["updated_at"] for u in updates], default=watermark)
    info["updated_through"] = max(watermark, newest_update)
    info["refreshed_at"] = iso(now)
    log.info(
        "%s: %d new, %d changed (%d re-encoded); %d issues in total",
        repo,
        counts["new"],
        counts["changed"],
        counts["re_encoded"],
        len(rows),
    )
    return counts


def validate(root: Path, encode: Encoder) -> None:
    """The refreshed index must load and answer a query, or nothing is published."""
    assets = DemoAssets(root)
    for repo, data in assets.repos.items():
        sample = data.issues[-1]
        results = assets.similar(repo, sample["title"], sample["snippet"], encode, k=3)
        if not results:
            raise RuntimeError(f"{repo}: the refreshed index returned no results")
        assets.classify(repo, sample["title"], sample["snippet"])


def refresh(
    root: Path,
    session: requests.Session,
    encode: Encoder,
    now: datetime | None = None,
) -> dict[str, dict[str, int]]:
    """Refresh every repository of a downloaded dataset in place."""
    now = now or datetime.now(UTC)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {
        repo: refresh_repo(root, repo, manifest, session, encode, now) for repo in manifest["repos"]
    }
    manifest["refreshed_at"] = iso(now)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_card(root, manifest)  # its table shows issue counts and the newest issue
    validate(root, encode)
    return summary


def download(repo_id: str, target: Path) -> Path:
    """A writable copy of the published dataset in ``target``."""
    from huggingface_hub import snapshot_download

    source = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".cache", ".gitattributes"))
    return target


def push(repo_id: str, root: Path, summary: dict[str, dict[str, int]]) -> None:
    from huggingface_hub import HfApi

    totals = ", ".join(
        f"{repo}: +{c['new']} new, {c['changed']} changed" for repo, c in summary.items()
    )
    HfApi().upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Refresh index ({totals})",
    )
    log.info("pushed to https://huggingface.co/datasets/%s", repo_id)


def main(argv: list[str] | None = None, encoder_factory=load_sentence_transformer) -> None:
    parser = argparse.ArgumentParser(description="Refresh the published demo index.")
    parser.add_argument("--repo-id", required=True, help="USER/NAME of the Hugging Face dataset")
    parser.add_argument("--dir", type=Path, default=Path("build/refresh"))
    parser.add_argument("--push", action="store_true", help="publish the result (default: dry run)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set the GITHUB_TOKEN environment variable first.")

    root = download(args.repo_id, args.dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    encode = encoder_factory(manifest["encoder"], args.batch_size)
    summary = refresh(root, make_session(token), encode)
    if args.push:
        push(args.repo_id, root, summary)
    else:
        log.info("dry run: not pushed (use --push); result in %s", root)


if __name__ == "__main__":
    main()
