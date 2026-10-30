import json

import numpy as np
import pytest

from triage_lens import predictions as pr
from triage_lens import transformer as tr
from triage_lens.dataset import CLASSES
from triage_lens.evaluate import one_hot


def rec(number, label, split, form=None):
    return {"number": number, "label": label, "form": form, "split": split, "text": "t"}


def write_dataset(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path):
    records = [rec(n, CLASSES[n % 4], "train") for n in range(8)]
    records += [rec(10 + n, CLASSES[n % 4], "validation", CLASSES[n % 4]) for n in range(4)]
    records += [rec(20 + n, CLASSES[n % 4], "test", "bug") for n in range(4)]
    data = tmp_path / "data"
    data.mkdir()
    return write_dataset(data / "astral-sh__uv.dataset.jsonl", records), records


def perfect_predictions(records, sha1):
    by_split = {}
    for split in ("validation", "test"):
        part = [r for r in records if r["split"] == split]
        by_split[split] = ([r["number"] for r in part], one_hot([r["label"] for r in part]))
    return by_split, {"kind": "test", "dataset_sha1": sha1}


def test_save_and_load_round_trip(tmp_path):
    probs = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
    path = pr.save_predictions(tmp_path / "m.npz", {"test": ([5, 6], probs)}, {"a": 1})
    by_split, meta = pr.load_predictions(path)
    assert meta["a"] == 1 and meta["classes"] == list(CLASSES)
    np.testing.assert_allclose(by_split["test"][6], probs[1], rtol=1e-6)


def test_save_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match="expected shape"):
        pr.save_predictions(tmp_path / "m.npz", {"test": ([1, 2], np.zeros((2, 3)))}, {})


def test_aligned_reports_missing_issue():
    with pytest.raises(pr.PredictionMismatch, match="#7"):
        pr.aligned({1: np.zeros(4)}, [{"number": 1}, {"number": 7}])


def test_score_file_scores_like_any_model(dataset, tmp_path):
    path, records = dataset
    payload, meta = perfect_predictions(records, pr.file_hash(path))
    npz = pr.save_predictions(tmp_path / "p" / "m.npz", payload, meta)
    scores = pr.score_file(npz, path)
    assert scores["test"]["accuracy"] == 1.0
    assert scores["test"]["flagging"]["misfiled"] == 3  # test forms are all "bug"


def test_score_file_refuses_another_dataset(dataset, tmp_path):
    path, records = dataset
    payload, meta = perfect_predictions(records, "0" * 40)
    npz = pr.save_predictions(tmp_path / "p" / "m.npz", payload, meta)
    with pytest.raises(pr.PredictionMismatch, match="different dataset"):
        pr.score_file(npz, path)


def test_main_scores_every_file(dataset, tmp_path, capsys):
    path, records = dataset
    payload, meta = perfect_predictions(records, pr.file_hash(path))
    pr.save_predictions(tmp_path / "pred" / "astral-sh__uv" / "ft_x.npz", payload, meta)
    results = tmp_path / "results"
    pr.main(
        [
            "astral-sh/uv",
            "--data",
            str(path.parent),
            "--predictions",
            str(tmp_path / "pred"),
            "--results",
            str(results),
        ]
    )
    saved = json.loads((results / "astral-sh__uv" / "ft_x.json").read_text())
    assert saved["params"]["kind"] == "test"
    assert "ft_x" in capsys.readouterr().out


# --------------------------------------------------------------------------- transformer helpers


def test_balanced_class_weights():
    weights = tr.balanced_class_weights(["bug"] * 6 + ["feature", "question", "docs"] * 2)
    np.testing.assert_allclose(weights, [12 / 24, 12 / 8, 12 / 8, 12 / 8])
    with pytest.raises(ValueError, match="docs"):
        tr.balanced_class_weights(["bug", "feature", "question"])


def test_softmax_rows_sum_to_one():
    probs = tr.softmax(np.array([[1000.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]]))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 0] == pytest.approx(1.0) and probs[1].argmax() == 3


def test_limited_keeps_every_class():
    records = [rec(n, "bug", "train") for n in range(10)] + [rec(99, "docs", "train")]
    kept = tr.limited(records, 8)
    assert [r["label"] for r in kept] == ["bug", "bug", "docs"]
    assert tr.limited(records, None) is records


def test_zeroshot_probs_maps_labels_back():
    result = {
        "labels": [
            "a usage question",
            "a bug report",
            "a documentation problem",
            "a feature request",
        ],
        "scores": [0.6, 0.2, 0.15, 0.05],
    }
    np.testing.assert_allclose(tr.zeroshot_probs([result]), [[0.2, 0.05, 0.6, 0.15]])


def test_model_names():
    assert tr.model_name_for("ft", "distilbert/distilbert-base-uncased", "full") == (
        "ft_distilbert-base-uncased"
    )
    assert tr.model_name_for("zs", "facebook/bart-large-mnli", "prose") == (
        "zs_bart-large-mnli_prose"
    )


def test_cli_parses_without_torch(monkeypatch):
    calls = []
    monkeypatch.setattr(tr, "finetune", lambda *args, **kwargs: calls.append((args, kwargs)))
    tr.main(["finetune", "astral-sh/uv", "--limit", "40", "--epochs", "1"])
    ((args, kwargs),) = calls
    assert args[0] == "astral-sh/uv" and args[6] == 1 and kwargs["limit"] == 40
