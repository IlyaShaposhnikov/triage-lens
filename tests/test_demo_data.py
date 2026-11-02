import gzip
import json
import sys
import types

import joblib
import numpy as np
import pytest

from triage_lens import demo_data as dd
from triage_lens.embeddings import save_cache, text_hash
from triage_lens.search import Corpus

WORDS = {"bug": "crash error", "feature": "add option", "question": "how do", "docs": "typo docs"}


def raw_issue(number, label, created):
    return {
        "number": number,
        "title": f"{WORDS[label]} {number}",
        "body": f"{WORDS[label]} details for issue {number}",
        "labels": [label] + [f"extra{i}" for i in range(10)],
        "state": "closed",
        "state_reason": "completed",
        "created_at": f"2025-01-{created:02d}T00:00:00Z",
        "updated_at": "2025-02-01T00:00:00Z",
        "author_type": "User",
        "url": f"https://github.com/astral-sh/uv/issues/{number}",
    }


@pytest.fixture
def workspace(tmp_path):
    raw, processed, cache, results = (tmp_path / d for d in ("raw", "proc", "emb", "res"))
    for d in (raw, processed, cache, results / "astral-sh__uv"):
        d.mkdir(parents=True)
    issues = [raw_issue(n, list(WORDS)[n % 4], n) for n in range(1, 25)]
    (raw / "astral-sh__uv.jsonl").write_text("\n".join(json.dumps(i) for i in issues) + "\n")

    records = []
    for i, issue in enumerate(issues):
        split = "train" if i < 16 else "validation" if i < 20 else "test"
        label = issue["labels"][0]
        records.append(
            {
                "number": issue["number"],
                "text": issue["title"],
                "label": label,
                "form": None,
                "split": split,
            }
        )
    (processed / "astral-sh__uv.dataset.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    (results / "astral-sh__uv" / "tfidf_logreg.json").write_text(
        json.dumps({"params": {"C": 4.0}, "scores": {"test": {"accuracy": 0.7, "macro_f1": 0.6}}})
    )
    (results / "astral-sh__uv" / "search_bge-small-en-v1.5.json").write_text(
        json.dumps(
            {
                "methods": {
                    "dense_exact": {"recall@1": 0.2, "recall@5": 0.4, "recall@10": 0.5, "mrr": 0.3}
                }
            }
        )
    )
    corpus = Corpus("astral-sh/uv", {i["number"]: i for i in issues})
    rng = np.random.default_rng(0)
    save_cache(
        cache / "astral-sh__uv.bge-small-en-v1.5.npz",
        {
            (int(n), text_hash(t)): rng.normal(size=8) * 3
            for n, t in zip(corpus.numbers, corpus.texts, strict=True)
        },
    )
    return tmp_path


def build_args(ws):
    return [
        "build",
        "astral-sh/uv",
        "--raw",
        str(ws / "raw"),
        "--processed",
        str(ws / "proc"),
        "--cache",
        str(ws / "emb"),
        "--results",
        str(ws / "res"),
        "--out",
        str(ws / "out"),
    ]


def test_build_writes_all_artifacts(workspace, capsys):
    dd.main(build_args(workspace))
    out = workspace / "out"

    with gzip.open(out / "astral-sh__uv" / "issues.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 24 and rows[0]["number"] == 1
    assert len(rows[0]["labels"]) == dd.MAX_LABELS
    assert rows[0]["snippet"].startswith("add option details")

    vectors = np.load(out / "astral-sh__uv" / "vectors.bge-small-en-v1.5.f16.npy")
    assert vectors.dtype == np.float16 and vectors.shape == (24, 8)
    np.testing.assert_allclose(np.linalg.norm(vectors.astype(np.float32), axis=1), 1, atol=1e-3)

    classifier = joblib.load(out / "astral-sh__uv" / "classifier.joblib")
    assert list(classifier.predict(["typo docs", "crash error"])) == ["docs", "bug"]

    manifest = json.loads((out / "manifest.json").read_text())
    info = manifest["repos"]["astral-sh/uv"]
    assert manifest["encoder"] == dd.DEMO_ENCODER and info["classifier"]["C"] == 4.0
    assert info["metrics"]["search"]["recall@10"] == 0.5
    card = (out / "README.md").read_text()
    assert "astral-sh/uv" in card and "license: other" in card
    assert "24 issues" in capsys.readouterr().out


def test_build_refuses_missing_vectors(workspace):
    (workspace / "emb" / "astral-sh__uv.bge-small-en-v1.5.npz").unlink()
    with pytest.raises(dd.MissingVectors, match="run `python -m triage_lens.search"):
        dd.main(build_args(workspace))


def test_push_uploads_folder(workspace, monkeypatch, capsys):
    dd.main(build_args(workspace))
    calls = []

    class FakeApi:
        def create_repo(self, repo_id, repo_type, exist_ok):
            calls.append(("create", repo_id, repo_type))

        def upload_folder(self, folder_path, repo_id, repo_type, commit_message):
            calls.append(("upload", repo_id, folder_path.endswith("out")))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    dd.main(["push", "--repo-id", "me/triage-lens-demo", "--out", str(workspace / "out")])
    assert calls == [
        ("create", "me/triage-lens-demo", "dataset"),
        ("upload", "me/triage-lens-demo", True),
    ]
    assert "datasets/me/triage-lens-demo" in capsys.readouterr().out


def test_push_without_build(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=object))
    with pytest.raises(SystemExit, match="run `build` first"):
        dd.main(["push", "--repo-id", "me/x", "--out", str(tmp_path)])
