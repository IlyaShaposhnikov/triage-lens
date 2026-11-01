import json
import sys
import types

import numpy as np
import pytest

from triage_lens import search as se


def issue(number, title, created, body=""):
    return {
        "number": number,
        "title": title,
        "body": body,
        "labels": [],
        "created_at": f"2025-01-{created:02d}T00:00:00Z",
        "updated_at": "2025-02-01T00:00:00Z",
    }


ISSUES = {
    1: issue(1, "uv lock hangs on private index", 1),
    2: issue(2, "add --frozen flag to uv tree", 2),
    3: issue(3, "documentation typo in install page", 3),
    4: issue(4, "uv lock hangs forever with private index", 4),  # duplicate of 1
    5: issue(5, "uv tree should support --frozen", 5),  # duplicate of 2
    6: issue(6, "unrelated question about python versions", 6),
}


@pytest.fixture
def corpus():
    return se.Corpus("astral-sh/uv", ISSUES)


def test_corpus_is_time_ordered(corpus):
    assert corpus.numbers.tolist() == [1, 2, 3, 4, 5, 6]
    assert corpus.before(corpus.position[4]).tolist() == [True, True, True, False, False, False]
    assert corpus.records()[0] == {"number": 1, "text": "uv lock hangs on private index"}


def test_evaluation_pairs_drops_unusable(corpus):
    pairs = [
        {"duplicate": 4, "original": 1, "original_is_issue": True},
        {"duplicate": 1, "original": 4, "original_is_issue": True},  # original is later
        {"duplicate": 5, "original": 99, "original_is_issue": True},  # missing
        {"duplicate": 5, "original": 2, "original_is_issue": False},
    ]
    kept, dropped = se.evaluation_pairs(pairs, corpus)
    assert kept == [(4, 1)]
    assert dropped == {"original_not_issue": 1, "missing": 1, "original_later": 1}


def test_top_k_respects_mask_and_order():
    scores = np.array([0.1, 0.9, 0.5, 0.95, 0.3])
    mask = np.array([True, True, True, False, True])
    assert se.top_k(scores, mask, k=3).tolist() == [1, 2, 4]
    assert se.top_k(scores, np.zeros(5, bool)).tolist() == []


def test_bm25_prefers_rare_shared_terms(corpus):
    bm25 = se.BM25(corpus.texts)
    scores = bm25.scores("uv lock hangs forever with private index")
    assert int(np.argmax(scores[:3])) == 0  # issue 1 among the earlier issues
    # "private" (rare) outweighs "uv" (frequent)
    assert bm25.scores("private")[0] > bm25.scores("uv")[0]
    assert bm25.scores("zzz").sum() == 0


def test_rrf():
    assert se.rrf([[1, 2, 3], [3, 1, 4]], k=60) == [1, 3, 2, 4]
    assert se.rrf([[5]], n=1) == [5]


def test_rank_of_and_metrics():
    assert se.rank_of(7, [3, 7, 9]) == 2 and se.rank_of(8, [3, 7]) is None
    m = se.metrics([1, 3, None, 12])
    assert m["recall@1"] == 0.25 and m["recall@5"] == 0.5 and m["recall@10"] == 0.5
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 12) / 4)


def test_bootstrap_paired_difference():
    ranks = {"bm25": [None] * 20 + [1] * 20, "dense_exact": [1] * 40}
    report = se.bootstrap(ranks, "bm25", samples=200)
    d = report["dense_exact"]["vs_bm25"]["recall@1"]
    assert d["diff"] == pytest.approx(0.5) and d["ci"][0] > 0
    assert report["bm25"]["vs_bm25"]["recall@1"]["ci"] == [0.0, 0.0]


def keyword_vectors(corpus):
    """Bag-of-words vectors over the corpus vocabulary, L2-normalised."""
    vocab = sorted({w for t in corpus.texts for w in t.lower().split()})
    rows = np.array([[float(w in t.lower().split()) for w in vocab] for t in corpus.texts])
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_evaluate_search_finds_originals(corpus):
    vectors = keyword_vectors(corpus)
    ranks = se.evaluate_search(corpus, [(4, 1), (5, 2)], vectors)
    assert ranks == {"bm25": [1, 1], "dense_exact": [1, 1], "hybrid": [1, 1]}


def test_evaluate_search_never_returns_later_issues(corpus):
    vectors = keyword_vectors(corpus)
    seen = []

    def chroma(vector, before):
        seen.append(before)
        return [1, 2]

    ranks = se.evaluate_search(corpus, [(4, 1)], vectors, chroma)
    assert ranks["dense_chroma"] == [1]
    assert seen == [int(corpus.created[corpus.position[4]])]


class FakeEncoder:
    def __init__(self, corpus):
        self.lookup = dict(zip(corpus.texts, keyword_vectors(corpus), strict=True))

    def __call__(self, texts):
        return np.stack([self.lookup[t] for t in texts])


def write_inputs(tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    (raw / "astral-sh__uv.jsonl").write_text(
        "\n".join(json.dumps(i) for i in ISSUES.values()) + "\n"
    )
    pairs = [
        {"duplicate": 4, "original": 1, "source": "comment", "original_is_issue": True},
        {"duplicate": 5, "original": 2, "source": "comment", "original_is_issue": True},
    ]
    (processed / "astral-sh__uv.duplicate_pairs.jsonl").write_text(
        "\n".join(json.dumps(p) for p in pairs) + "\n"
    )
    return raw, processed


def test_main_with_numpy_backend(tmp_path, capsys):
    raw, processed = write_inputs(tmp_path)
    corpus = se.Corpus("astral-sh/uv", ISSUES)
    args = [
        "astral-sh/uv",
        "--backend",
        "numpy",
        "--raw",
        str(raw),
        "--processed",
        str(processed),
        "--cache",
        str(tmp_path / "emb"),
        "--results",
        str(tmp_path / "res"),
    ]
    se.main(args, encoder_factory=lambda name, batch: FakeEncoder(corpus))

    saved = json.loads(
        (tmp_path / "res" / "astral-sh__uv" / "search_all-minilm-l6-v2.json").read_text()
    )
    assert saved["methods"]["hybrid"]["recall@1"] == 1.0
    assert saved["ranks"]["pairs"] == [[4, 1], [5, 2]]
    assert "dense_chroma" not in saved["methods"]
    assert "2 queries over 6 issues" in capsys.readouterr().out


class FakeCollection:
    def __init__(self):
        self.items = {}

    def count(self):
        return len(self.items)

    def add(self, ids, embeddings, metadatas):
        for i, e, m in zip(ids, embeddings, metadatas, strict=True):
            self.items[i] = (np.array(e), m["created"])

    def query(self, query_embeddings, n_results, where, include):
        q = np.array(query_embeddings[0])
        limit = where["created"]["$lt"]
        found = [(float(np.sum((e - q) ** 2)), i) for i, (e, c) in self.items.items() if c < limit]
        return {"ids": [[i for _, i in sorted(found)[:n_results]]]}


class FakeClient:
    collections = {}

    def __init__(self, path):
        pass

    def get_or_create_collection(self, name):
        return self.collections.setdefault(name, FakeCollection())

    def create_collection(self, name):
        self.collections[name] = FakeCollection()
        return self.collections[name]

    def delete_collection(self, name):
        self.collections.pop(name, None)

    def get_max_batch_size(self):
        return 2


def test_main_with_chroma_backend(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", types.SimpleNamespace(PersistentClient=FakeClient))
    raw, processed = write_inputs(tmp_path)
    corpus = se.Corpus("astral-sh/uv", ISSUES)
    args = [
        "astral-sh/uv",
        "--raw",
        str(raw),
        "--processed",
        str(processed),
        "--cache",
        str(tmp_path / "emb"),
        "--results",
        str(tmp_path / "res"),
        "--chroma",
        str(tmp_path / "chroma"),
    ]
    se.main(args, encoder_factory=lambda name, batch: FakeEncoder(corpus))

    saved = json.loads(
        (tmp_path / "res" / "astral-sh__uv" / "search_all-minilm-l6-v2.json").read_text()
    )
    assert saved["methods"]["dense_chroma"]["recall@1"] == 1.0
    assert FakeClient.collections["astral-sh__uv__all-minilm-l6-v2"].count() == 6
