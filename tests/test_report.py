import json

import pytest

from triage_lens import report as rp


def classifier_result(accuracy):
    return {
        "scores": {
            "test": {
                "accuracy": accuracy,
                "accuracy_ci": [accuracy - 0.02, accuracy + 0.02],
                "macro_f1": 0.5,
                "flagging": {"combined": {"subsets": {"no_form": {"accuracy": 0.6}}}},
            }
        }
    }


def search_result(dense_ranks):
    method = {
        "recall@1": 0.1,
        "recall@5": 0.2,
        "recall@10": 0.3,
        "mrr": 0.15,
        "ci": {"recall@10": [0.25, 0.35]},
    }
    return {
        "methods": {"bm25": method, "dense_exact": method},
        "ranks": {"pairs": [[2, 1], [4, 3], [6, 5]], "dense_exact": dense_ranks},
    }


def test_paired_encoders():
    better, worse = search_result([1, 2, 3]), search_result([None, 2, 20])
    diff = rp.paired_encoders(better, worse, samples=200)
    assert diff["recall@10"]["diff"] == pytest.approx(2 / 3)
    assert diff["mrr"]["diff"] == pytest.approx((1 + 1 / 3 - 1 / 20) / 3)
    other = search_result([1, 2, 3])
    other["ranks"]["pairs"] = [[9, 8]] * 3
    with pytest.raises(ValueError, match="different queries"):
        rp.paired_encoders(better, other)


def test_report_renders_tables(tmp_path):
    folder = tmp_path / "astral-sh__uv"
    folder.mkdir()
    (folder / "tfidf_logreg.json").write_text(json.dumps(classifier_result(0.69)))
    (folder / "trust_form.json").write_text(json.dumps(classifier_result(0.775)))
    (folder / "search_bge-small-en-v1.5.json").write_text(json.dumps(search_result([1, 2, 3])))
    (folder / "search_all-minilm-l6-v2.json").write_text(json.dumps(search_result([1, None, 9])))
    (folder / "duplicate_pairs_precision.json").write_text(
        json.dumps({"estimated_precision": 0.93, "ci": [0.76, 0.97]})
    )
    text = rp.report(tmp_path)
    assert "## astral-sh/uv" in text and "pandas" not in text
    assert "| TF-IDF + logistic regression | 69.0% [67.0–71.0] | 0.500 | 60.0% |" in text
    assert text.index("Trust the issue form") < text.index("TF-IDF")
    assert text.count("BM25 (keywords)") == 1
    assert "bge-small-en-v1.5 − all-minilm-l6-v2" in text
    assert "precision 93.0% [76.0–97.0]" in text


def test_main_writes_utf8_file(tmp_path, capsys):
    folder = tmp_path / "astral-sh__uv"
    folder.mkdir()
    (folder / "tfidf_logreg.json").write_text(json.dumps(classifier_result(0.69)))
    out = tmp_path / "summary.md"
    rp.main(["--results", str(tmp_path), "--out", str(out)])
    assert "67.0–71.0" in out.read_text(encoding="utf-8")
    assert "wrote" in capsys.readouterr().out
