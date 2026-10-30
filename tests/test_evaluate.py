import json

import numpy as np
import pytest

from triage_lens import baselines as bl
from triage_lens import evaluate as ev

# Column order of probability arrays: bug, feature, question, docs.
BUG, FEATURE, QUESTION, DOCS = np.eye(4)


def rec(label, form=None, text="", split="test"):
    return {"label": label, "form": form, "text": text, "split": split}


def probs(*rows):
    return np.array(rows, dtype=float)


# --------------------------------------------------------------------------- helpers


def test_one_hot_and_to_labels_round_trip():
    labels = ["docs", "bug", "question"]
    assert ev.to_labels(ev.one_hot(labels)) == labels


def test_form_predictions_fall_back_without_form():
    records = [rec("bug", form="question"), rec("bug"), rec("bug", form="other")]
    assert ev.form_predictions(records, "bug") == ["question", "bug", "bug"]


def test_standalone_report():
    records = [rec("bug"), rec("bug"), rec("question"), rec("docs")]
    result = ev.standalone_report(records, ["bug", "bug", "question", "docs"])
    assert result["accuracy"] == 1.0
    # "feature" has no examples and no predictions: it scores 0 and lowers macro-F1.
    assert result["macro_f1"] == pytest.approx(0.75)
    assert result["per_class"]["bug"]["support"] == 2
    assert result["confusion"]["matrix"][0] == [2, 0, 0, 0]


# --------------------------------------------------------------------------- bootstrap


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(1)
    classes = ["bug", "feature", "question", "docs"]
    y_true = list(rng.choice(classes[:3], 200))  # "docs" absent in truth
    y_pred = list(rng.choice(classes, 200))
    expected = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    assert ev.macro_f1(y_true, y_pred) == pytest.approx(expected)


def test_bootstrap_ci_brackets_the_score_and_is_reproducible():
    y_true = ["bug"] * 40 + ["question"] * 30 + ["feature"] * 20 + ["docs"] * 10
    y_pred = y_true[:]
    for i in range(0, 100, 7):
        y_pred[i] = "bug"
    for metric in (ev.macro_f1, ev.accuracy):
        low, high = ev.bootstrap_ci(y_true, y_pred, metric, samples=200)
        assert low <= metric(y_true, y_pred) <= high
        assert (low, high) == ev.bootstrap_ci(y_true, y_pred, metric, samples=200)


def test_paired_bootstrap():
    y_true = ["bug"] * 50 + ["question"] * 50
    same = ev.paired_bootstrap(y_true, y_true, y_true, ev.accuracy, samples=100)
    assert same == {"diff": 0.0, "ci": [0.0, 0.0], "p_a_better": 0.0}

    worse = ["bug"] * 100
    better = ev.paired_bootstrap(y_true, y_true, worse, ev.accuracy, samples=100)
    assert better["diff"] == pytest.approx(0.5)
    assert better["ci"][0] > 0 and better["p_a_better"] == 1.0


# --------------------------------------------------------------------------- form + flag


def test_flag_scores_and_suggestions():
    records = [rec("question", form="bug"), rec("bug")]
    p = probs([0.2, 0.1, 0.6, 0.1], [0.1, 0.7, 0.1, 0.1])
    scores, suggestions = ev.flag_scores(records, p)
    assert scores[0] == pytest.approx(0.8)
    assert np.isnan(scores[1])
    assert suggestions == ["question", "feature"]


def flag_case():
    """Ten issues filed as bug; the four misfiled ones get the highest scores."""
    records = [rec("question", form="bug")] * 3 + [rec("feature", form="bug")]
    records += [rec("bug", form="bug")] * 6
    rows = [
        [0.1, 0.0, 0.9, 0.0],  # misfiled question, score 0.9
        [0.2, 0.0, 0.8, 0.0],  # misfiled question, score 0.8
        [0.6, 0.0, 0.4, 0.0],  # misfiled question, score 0.4
        [0.3, 0.0, 0.7, 0.0],  # misfiled feature, suggested question, score 0.7
        [0.4, 0.0, 0.6, 0.0],  # correct, score 0.6
        [0.5, 0.0, 0.5, 0.0],  # correct, score 0.5
    ] + [[0.9, 0.0, 0.1, 0.0]] * 4  # correct, score 0.1
    return records, probs(*rows)


def test_choose_threshold_picks_lowest_threshold_with_target_precision():
    records, p = flag_case()
    # Flags at 0.4 and above: 4 misfiled + 2 correct -> precision 4/6 >= 0.5.
    # Flags at 0.1 and above: 4 of 10 -> 0.4, below target.
    # At 0.7 and above all three flags are misfiled issues: precision 1.0.
    assert ev.choose_threshold(records, p, min_flags=1) == pytest.approx(0.4)
    assert ev.choose_threshold(records, p, target_precision=0.9, min_flags=1) == pytest.approx(0.7)
    assert ev.choose_threshold(records, p, min_flags=20) is None


def test_one_hot_form_model_never_flags():
    records, _ = flag_case()
    assert ev.choose_threshold(records, ev.one_hot([r["form"] for r in records])) is None


def test_flag_report_at_threshold():
    records, p = flag_case()
    report, preds = ev.flag_report(records, p, threshold=0.5)
    assert report["flags"] == 5 and report["true_flags"] == 3 and report["false_flags"] == 2
    assert report["caught"] == pytest.approx(3 / 4)
    assert report["fixed"] == pytest.approx(2 / 4)  # the feature one gets "question"
    assert report["flag_precision"] == pytest.approx(3 / 5)
    assert report["false_alarm_rate"] == pytest.approx(2 / 6)
    assert report["average_precision"] > 0.8
    assert report["combined"]["accuracy"] == pytest.approx(6 / 10)
    assert preds[2] == "bug"  # score 0.4 is below the threshold: the form stands


def test_flag_report_without_threshold_keeps_the_form():
    records, p = flag_case()
    report, preds = ev.flag_report(records, p, threshold=None)
    assert report["flags"] == 0 and report["caught"] == 0.0 and report["flag_precision"] is None
    assert preds == ["bug"] * 10


def test_combined_uses_model_for_issues_without_form():
    preds, _ = ev.combined_predictions([rec("docs")], probs(DOCS), threshold=None)
    assert preds == ["docs"]


def test_score_model_end_to_end():
    records, p = flag_case()
    splits = {
        "train": [rec("bug")],
        "validation": records,
        "test": records,
    }
    scores = ev.score_model("m", lambda recs: p, splits)
    test = scores["test"]
    assert scores["validation"]["flagging"]["threshold"] == pytest.approx(0.4)
    assert test["flagging"]["threshold"] == pytest.approx(0.4)
    assert set(test["vs_form"]) == {"macro_f1", "accuracy"}
    assert "vs_form" in test["flagging"]["combined"]
    assert "vs_form" not in scores["validation"]


def test_save_result_and_load_dataset(tmp_path):
    path = ev.save_result(tmp_path, "o/r", "m", {"test": {"macro_f1": 0.5}}, {"C": 1})
    assert path == tmp_path / "o__r" / "m.json"
    assert json.loads(path.read_text())["params"] == {"C": 1}

    data = tmp_path / "d.jsonl"
    data.write_text(
        "\n".join(json.dumps(r) for r in [rec("bug", split="train"), rec("bug")]) + "\n"
    )
    assert sorted(ev.load_dataset(data)) == ["test", "train"]


# --------------------------------------------------------------------------- baselines


def test_majority_and_trust_form_probabilities():
    train = [rec("bug"), rec("bug"), rec("question")]
    records = [rec("question", form="question"), rec("bug"), rec("bug", form="other")]
    assert ev.to_labels(bl.proba_majority(train, records)) == ["bug", "bug", "bug"]
    assert ev.to_labels(bl.proba_trust_form(train, records)) == ["question", "bug", "bug"]


WORDS = {
    "bug": "crash traceback error fails panic",
    "feature": "please add support option request",
    "question": "how do i can you explain why",
    "docs": "documentation typo docs page link",
}


def test_tfidf_logreg_learns_and_returns_columns_in_class_order():
    train = [
        rec(label, text=f"{text} sample {i}") for label, text in WORDS.items() for i in range(10)
    ]
    validation = [rec(label, text=text) for label, text in WORDS.items()]
    model, params = bl.fit_tfidf_logreg(train, validation)
    assert params["C"] in bl.C_GRID
    p = bl.proba_in_class_order(model, [rec("x", text=WORDS["docs"]), rec("x", text=WORDS["bug"])])
    assert p.shape == (2, 4)
    assert ev.to_labels(p) == ["docs", "bug"]


def test_run_writes_results_and_prints_tables(tmp_path, capsys):
    lines = []
    for split, n in (("train", 6), ("validation", 3), ("test", 3)):
        for label, text in WORDS.items():
            lines += [rec(label, form=label, text=f"{text} {i}", split=split) for i in range(n)]
    data = tmp_path / "data"
    data.mkdir()
    (data / "astral-sh__uv.dataset.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )
    bl.main(["astral-sh/uv", "--data", str(data), "--results", str(tmp_path / "results")])

    saved = sorted(p.name for p in (tmp_path / "results" / "astral-sh__uv").iterdir())
    assert saved == ["majority.json", "tfidf_logreg.json", "trust_form.json"]
    out = capsys.readouterr().out
    assert "Standalone" in out and "Form + flag" in out and "tfidf_logreg" in out
