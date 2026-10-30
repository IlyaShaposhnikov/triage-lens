import json

import numpy as np

from triage_lens import embeddings as em
from triage_lens.evaluate import to_labels

KEYWORDS = {
    "bug": "crash",
    "feature": "add",
    "question": "how",
    "docs": "typo",
}


class FakeEncoder:
    """Bag-of-keywords vectors; records every batch it is asked to encode."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        rows = [[float(word in text) for word in KEYWORDS.values()] + [0.1] for text in texts]
        return np.array(rows)


def rec(number, text, label="bug", form=None, split="train"):
    return {
        "number": number,
        "title": text,
        "body": "```\ncode\n```",
        "text": text,
        "label": label,
        "form": form,
        "split": split,
    }


def test_model_slug():
    assert em.model_slug("sentence-transformers/all-MiniLM-L6-v2") == "all-minilm-l6-v2"
    assert em.model_slug("local-model") == "local-model"


def test_embed_records_uses_and_updates_cache(tmp_path):
    cache = tmp_path / "c.npz"
    encode = FakeEncoder()
    first = em.embed_records([rec(1, "crash"), rec(2, "how")], encode, cache)
    assert first.shape == (2, 5) and len(encode.calls) == 1

    # Same issues again: nothing to encode. Issue 2 edited and issue 3 new: only those.
    em.embed_records([rec(1, "crash"), rec(2, "how")], encode, cache)
    assert len(encode.calls) == 1
    again = em.embed_records([rec(3, "typo"), rec(1, "crash"), rec(2, "how to add")], encode, cache)
    assert encode.calls[-1] == ["typo", "how to add"]
    np.testing.assert_allclose(again[1], first[0])
    assert len(em.load_cache(cache)) == 4  # the old version of issue 2 stays harmlessly


def test_fit_logreg_and_class_order():
    x = np.eye(4).repeat(5, axis=0) + 0.01
    y = [cls for cls in ("bug", "feature", "question", "docs") for _ in range(5)]
    model, params = em.fit_logreg(x, y, x, y)
    assert params["C"] in em.C_GRID
    probs = em.proba_in_class_order(model, np.eye(4))
    assert to_labels(probs) == ["bug", "feature", "question", "docs"]


def test_main_end_to_end(tmp_path, capsys):
    lines = []
    number = 0
    for split, n in (("train", 8), ("validation", 3), ("test", 3)):
        for label, word in KEYWORDS.items():
            for i in range(n):
                number += 1
                lines.append(rec(number, f"{word} text {i}", label, form=label, split=split))
    data = tmp_path / "data"
    data.mkdir()
    (data / "astral-sh__uv.dataset.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )
    results = tmp_path / "results"
    (results / "astral-sh__uv").mkdir(parents=True)
    encode = FakeEncoder()

    def factory(model_name, batch_size):
        assert batch_size == 32
        return encode

    args = ["astral-sh/uv", "--data", str(data), "--cache", str(tmp_path / "emb")]
    em.main([*args, "--results", str(results)], encoder_factory=factory)

    saved = json.loads((results / "astral-sh__uv" / "emb_all-minilm-l6-v2.json").read_text())
    assert saved["params"]["encoder"] == em.DEFAULT_MODEL and saved["params"]["dim"] == 5
    assert saved["scores"]["test"]["accuracy"] == 1.0
    assert (tmp_path / "emb" / "astral-sh__uv.all-minilm-l6-v2.npz").exists()
    assert "emb_all-minilm-l6-v2" in capsys.readouterr().out

    # A second run reads every vector from the cache.
    em.main([*args, "--results", str(results)], encoder_factory=factory)
    assert len(encode.calls) == 1


def test_embed_records_with_prose_input(tmp_path):
    encode = FakeEncoder()
    em.embed_records([rec(1, "crash")], encode, tmp_path / "c.npz", em.INPUTS["prose"])
    assert encode.calls == [["crash"]]  # the code block in the body is gone


def test_prose_run_is_saved_under_its_own_name(tmp_path, capsys):
    lines, number = [], 0
    for split, n in (("train", 8), ("validation", 3), ("test", 3)):
        for label, word in KEYWORDS.items():
            for i in range(n):
                number += 1
                lines.append(rec(number, f"{word} text {i}", label, form=label, split=split))
    data = tmp_path / "data"
    data.mkdir()
    (data / "astral-sh__uv.dataset.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )
    args = ["astral-sh/uv", "--data", str(data), "--cache", str(tmp_path / "emb")]
    args += ["--results", str(tmp_path / "results"), "--input", "prose"]
    em.main(args, encoder_factory=lambda name, batch: FakeEncoder())

    saved = tmp_path / "results" / "astral-sh__uv" / "emb_all-minilm-l6-v2_prose.json"
    assert json.loads(saved.read_text())["params"]["input"] == "prose"
    assert (tmp_path / "emb" / "astral-sh__uv.all-minilm-l6-v2_prose.npz").exists()
    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if line.startswith("model"))
    row = next(line for line in out.splitlines() if line.startswith("emb_all-minilm-l6-v2_prose"))
    assert row.index("%") > len("emb_all-minilm-l6-v2_prose")  # columns stay aligned
    assert header.index("macroF1") >= len("emb_all-minilm-l6-v2_prose")
