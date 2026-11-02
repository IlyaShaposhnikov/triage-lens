import gzip
import json
import sys
import types

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.pipeline import Pipeline

from triage_lens import app_core as ac
from triage_lens.baselines import make_logreg, make_vectorizer
from triage_lens.dataset import CLASSES

WORDS = {"bug": "crash error", "feature": "add option", "question": "how do", "docs": "typo docs"}
VOCAB = ["crash", "add", "how", "typo", "lock"]


def bag(text):
    v = np.array([float(w in text.lower()) for w in VOCAB]) + 0.01
    return v / np.linalg.norm(v)


def make_assets(root, sklearn_version=sklearn.__version__):
    folder = root / "astral-sh__uv"
    folder.mkdir(parents=True)
    issues = [
        {
            "number": 1,
            "title": "crash on lock",
            "state": "open",
            "state_reason": None,
            "labels": ["bug"],
            "created_at": "2025-01-01T00:00:00Z",
            "url": "u/1",
            "snippet": "s",
        },
        {
            "number": 2,
            "title": "add option",
            "state": "closed",
            "state_reason": "completed",
            "labels": [],
            "created_at": "2025-01-02T00:00:00Z",
            "url": "u/2",
            "snippet": "",
        },
        {
            "number": 3,
            "title": "typo docs",
            "state": "closed",
            "state_reason": "duplicate",
            "labels": [],
            "created_at": "2025-01-03T00:00:00Z",
            "url": "u/3",
            "snippet": "",
        },
    ]
    with gzip.open(folder / "issues.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(i) for i in issues) + "\n")
    np.save(
        folder / "vectors.fake.f16.npy",
        np.stack([bag(i["title"]) for i in issues]).astype(np.float16),
    )
    texts = [f"{w} {n}" for label, w in WORDS.items() for n in range(5)]
    labels = [label for label in WORDS for _ in range(5)]
    model = Pipeline([("tfidf", make_vectorizer()), ("clf", make_logreg(4.0))])
    model.set_params(tfidf__min_df=1)
    joblib.dump(model.fit(texts, labels), folder / "classifier.joblib")
    manifest = {
        "encoder": "fake/encoder",
        "scikit_learn": sklearn_version,
        "classes": list(CLASSES),
        "repos": {
            "astral-sh/uv": {
                "issues": 3,
                "vectors_file": "astral-sh__uv/vectors.fake.f16.npy",
                "newest_issue": "2025-01-03T00:00:00Z",
                "metrics": {},
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def fake_encode(texts):
    return np.stack([bag(t) * 3 for t in texts])  # not normalised on purpose


def test_classify_returns_probabilities_in_class_order(tmp_path):
    assets = ac.DemoAssets(make_assets(tmp_path))
    probs = assets.classify("astral-sh/uv", "typo docs", "")
    assert list(probs) == list(CLASSES)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert max(probs, key=probs.get) == "docs"


def test_similar_ranks_by_cosine(tmp_path):
    assets = ac.DemoAssets(make_assets(tmp_path))
    results = assets.similar("astral-sh/uv", "uv lock crash", "", fake_encode, k=2)
    assert results[0]["number"] == 1 and len(results) == 2
    assert results[0]["similarity"] > results[1]["similarity"]
    assert results[0]["similarity"] <= 1.0 + 1e-3
    assert len(assets.similar("astral-sh/uv", "x", "", fake_encode, k=10)) == 3


def test_query_text_removes_form_boilerplate():
    body = "### Summary\n\nIt crashes.\n\n### Platform\n\nmacOS"
    assert ac.query_text("astral-sh/uv", "BUG: crash", body) == "crash\n\nIt crashes."


def test_version_mismatch_warns(tmp_path):
    with pytest.warns(UserWarning, match="scikit-learn 0.1"):
        ac.DemoAssets(make_assets(tmp_path, sklearn_version="0.1"))


def test_mismatched_vectors_are_rejected(tmp_path):
    root = make_assets(tmp_path)
    np.save(root / "astral-sh__uv" / "vectors.fake.f16.npy", np.zeros((2, 5), np.float16))
    with pytest.raises(ValueError, match="2 vectors for 3 issues"):
        ac.DemoAssets(root)


def test_from_hub_downloads_snapshot(tmp_path, monkeypatch):
    root = make_assets(tmp_path)
    calls = []

    def snapshot_download(repo_id, repo_type, revision):
        calls.append((repo_id, repo_type, revision))
        return str(root)

    hub = types.SimpleNamespace(snapshot_download=snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assets = ac.DemoAssets.from_hub("me/demo")
    assert calls == [("me/demo", "dataset", None)]
    assert assets.encoder_name == "fake/encoder"


def test_plain_snippet_drops_images_html_and_markup():
    snippet = "Fails ![shot](https://x/y.png) see [docs](http://d) <img src=a> `uv` *b*"
    assert ac.plain_snippet(snippet) == "Fails see docs \\`uv\\` \\*b\\*"
    assert ac.plain_snippet("") == ""
