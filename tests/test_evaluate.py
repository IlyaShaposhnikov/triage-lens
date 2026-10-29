import json

import pytest

from triage_lens import baselines as bl
from triage_lens import evaluate as ev


def rec(label, form=None, text="", split="test"):
    return {"label": label, "form": form, "text": text, "split": split}


def test_evaluate_perfect_and_per_class():
    records = [rec("bug"), rec("bug"), rec("question"), rec("docs")]
    result = ev.evaluate(records, ["bug", "bug", "question", "docs"], bootstrap=False)
    assert result["accuracy"] == 1.0
    # "feature" has no examples and no predictions: it scores 0 and lowers macro-F1.
    assert result["macro_f1"] == pytest.approx(0.75)
    assert result["per_class"]["bug"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 2,
    }
    assert result["confusion"]["matrix"][0] == [2, 0, 0, 0]


def test_misfiled_report():
    records = [
        rec("question", form="bug"),  # misfiled, model fixes it
        rec("question", form="bug"),  # misfiled, model flags it with the wrong class
        rec("feature", form="bug"),  # misfiled, model misses it
        rec("bug", form="bug"),  # correct, model agrees
        rec("bug", form="bug"),  # correct, false alarm
        rec("bug", form=None),  # no form: ignored
        rec("bug", form="other"),  # pandas PERF/BUILD forms: ignored
    ]
    preds = ["question", "docs", "bug", "bug", "question", "bug", "bug"]
    report = ev.misfiled_report(records, preds)
    assert report == {
        "filed_via_form": 5,
        "misfiled": 3,
        "caught": pytest.approx(2 / 3),
        "fixed": pytest.approx(1 / 3),
        "flag_precision": pytest.approx(2 / 3),
        "false_alarm_rate": pytest.approx(1 / 2),
    }


def test_misfiled_report_without_forms():
    report = ev.misfiled_report([rec("bug")], ["bug"])
    assert report["misfiled"] == 0 and report["caught"] is None


def test_bootstrap_ci_brackets_the_score_and_is_reproducible():
    y_true = ["bug"] * 40 + ["question"] * 30 + ["feature"] * 20 + ["docs"] * 10
    y_pred = y_true[:]
    for i in range(0, 100, 7):
        y_pred[i] = "bug"
    low, high = ev.bootstrap_ci(y_true, y_pred, samples=200)
    assert low <= ev.macro_f1(y_true, y_pred) <= high
    assert (low, high) == ev.bootstrap_ci(y_true, y_pred, samples=200)


def test_save_result_and_load_dataset(tmp_path):
    path = ev.save_result(tmp_path, "o/r", "m", {"test": {"macro_f1": 0.5}}, {"C": 1})
    assert path == tmp_path / "o__r" / "m.json"
    assert json.loads(path.read_text())["params"] == {"C": 1}

    data = tmp_path / "d.jsonl"
    data.write_text(
        "\n".join(json.dumps(r) for r in [rec("bug", split="train"), rec("bug")]) + "\n"
    )
    splits = ev.load_dataset(data)
    assert sorted(splits) == ["test", "train"]


def test_format_row_handles_missing_values():
    result = ev.evaluate([rec("bug")], ["bug"], bootstrap=False)
    row = ev.format_row("majority", result)
    assert row.startswith("majority") and "-" in row


# --------------------------------------------------------------------------- baselines


def test_majority_and_trust_form():
    train = [rec("bug"), rec("bug"), rec("question")]
    records = [rec("question", form="question"), rec("bug", form=None), rec("bug", form="other")]
    assert bl.predict_majority(train, records) == ["bug", "bug", "bug"]
    assert bl.predict_trust_form(train, records) == ["question", "bug", "bug"]


def test_tfidf_logreg_learns_separable_classes():
    words = {
        "bug": "crash traceback error fails panic",
        "feature": "please add support option request",
        "question": "how do i can you explain why",
        "docs": "documentation typo docs page link",
    }
    train = [
        rec(label, text=f"{text} sample {i}") for label, text in words.items() for i in range(10)
    ]
    validation = [rec(label, text=text) for label, text in words.items()]
    model, params = bl.fit_tfidf_logreg(train, validation)
    assert params["C"] in bl.C_GRID
    assert list(model.predict([words["docs"], words["bug"]])) == ["docs", "bug"]


def test_run_writes_results(tmp_path, capsys):
    words = {
        "bug": "crash error",
        "feature": "add option",
        "question": "how do",
        "docs": "typo docs",
    }
    lines = []
    for split, n in (("train", 6), ("validation", 2), ("test", 2)):
        for label, text in words.items():
            lines += [rec(label, form=label, text=f"{text} {i}", split=split) for i in range(n)]
    data = tmp_path / "data"
    data.mkdir()
    (data / "astral-sh__uv.dataset.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )
    bl.main(["astral-sh/uv", "--data", str(data), "--results", str(tmp_path / "results")])

    saved = sorted(p.name for p in (tmp_path / "results" / "astral-sh__uv").iterdir())
    assert saved == ["majority.json", "tfidf_logreg.json", "trust_form.json"]
    assert "tfidf_logreg" in capsys.readouterr().out
