import json
from collections import Counter

import pytest

from triage_lens import review as rv


def pair(dup, orig, source, is_issue=True):
    return {"duplicate": dup, "original": orig, "source": source, "original_is_issue": is_issue}


def test_stratified_sample_takes_each_rule_and_skips_non_issues():
    pairs = [pair(n, 1, "comment") for n in range(10, 40)]
    pairs += [pair(n, 1, "event") for n in range(50, 53)]
    pairs += [pair(99, 2, "single_ref", is_issue=False)]
    sample = rv.stratified_sample(pairs, per_source=5)
    assert Counter(p["source"] for p in sample) == {"comment": 5, "event": 3}
    assert sample == rv.stratified_sample(pairs, per_source=5)  # reproducible


def test_mentioning_comments():
    thread = {
        "comments": [
            {"author": "a", "body": "Duplicate of #123"},
            {"author": "b", "body": "see #1234 and issues/12"},
            {"author": "c", "body": "same as https://github.com/o/r/issues/123"},
        ]
    }
    assert [c["author"] for c in rv.mentioning_comments(thread, 123)] == ["a", "c"]


def test_review_item_and_page_escaping():
    issues = {
        n: {
            "title": f"T{n} </script>",
            "body": "x" * 2000,
            "labels": ["bug"],
            "state": "closed",
            "state_reason": "duplicate",
            "url": f"https://github.com/o/r/issues/{n}",
        }
        for n in (5, 7)
    }
    threads = {5: {"comments": [{"author": "m", "body": "dup of #7"}]}}
    item = rv.review_item("o/r", pair(5, 7, "comment"), issues, threads)
    assert item["id"] == "o/r#5->7"
    assert len(item["duplicate"]["body"]) == rv.BODY_CHARS
    assert item["evidence"] == [{"author": "m", "body": "dup of #7"}]
    page = rv.render_page([item])
    assert "T5 <\\/script>" in page and "__DATA__" not in page


def test_wilson():
    assert rv.wilson(0, 0) is None
    lo, hi = rv.wilson(10, 10)
    assert hi == 1.0 and 0.65 < lo < 0.75
    lo, hi = rv.wilson(5, 10)
    assert lo < 0.5 < hi


def test_precision_report_weights_rules_by_size():
    sources = {f"o/r#{i}->1": "comment" for i in range(10)}
    sources |= {f"o/r#{i}->1": "single_ref" for i in range(10, 20)}
    verdicts = {f"o/r#{i}->1": "yes" for i in range(10)}  # comment: 10/10
    verdicts |= {f"o/r#{i}->1": ("yes" if i < 15 else "no") for i in range(10, 20)}  # 5/10
    verdicts["o/r#19->1"] = "unsure"  # single_ref becomes 5/9
    population = {"o/r": Counter({"comment": 300, "single_ref": 100})}
    report = rv.precision_report(verdicts, sources, population)["o/r"]
    assert report["rules"]["comment"]["precision"] == 1.0
    assert report["rules"]["single_ref"]["reviewed"] == 9
    assert report["rules"]["single_ref"]["unsure"] == 1
    assert report["estimated_precision"] == pytest.approx(0.75 * 1.0 + 0.25 * 5 / 9)
    assert report["rules"]["event"]["precision"] is None


def test_sample_and_score_end_to_end(tmp_path, capsys):
    raw, processed, out = tmp_path / "raw", tmp_path / "processed", tmp_path / "ann"
    raw.mkdir(), processed.mkdir()
    issues = [
        {
            "number": n,
            "title": f"t{n}",
            "body": "b",
            "labels": [],
            "state": "closed",
            "state_reason": None,
            "updated_at": "2025",
            "url": f"u/{n}",
        }
        for n in range(1, 30)
    ]
    (raw / "o__r.jsonl").write_text("\n".join(json.dumps(i) for i in issues) + "\n")
    (raw / "o__r.threads.jsonl").write_text(
        json.dumps(
            {
                "number": 10,
                "issue_updated_at": "2025",
                "events": [],
                "comments": [{"author": "m", "body": "Duplicate of #1", "created_at": ""}],
            }
        )
        + "\n"
    )
    pairs = [pair(n, 1, "comment") for n in range(10, 25)] + [pair(25, 2, "weak_comment")]
    (processed / "o__r.duplicate_pairs.jsonl").write_text(
        "\n".join(json.dumps(p) for p in pairs) + "\n"
    )
    rv.main(
        [
            "sample",
            "o/r",
            "--raw",
            str(raw),
            "--processed",
            str(processed),
            "--out",
            str(out),
            "--per-source",
            "4",
        ]
    )
    sample = json.loads((out / "duplicate_review.sample.json").read_text())
    assert len(sample["items"]) == 5
    assert (out / "duplicate_review.html").exists()

    verdicts = {it["id"]: "yes" for it in sample["items"]}
    (out / "v.json").write_text(json.dumps({"version": 1, "verdicts": verdicts}))
    rv.main(
        [
            "score",
            str(out / "v.json"),
            "--sample",
            str(out / "duplicate_review.sample.json"),
            "--results",
            str(tmp_path / "results"),
        ]
    )
    saved = json.loads(
        (tmp_path / "results" / "o__r" / "duplicate_pairs_precision.json").read_text()
    )
    assert saved["estimated_precision"] == 1.0
    assert "all pairs" in capsys.readouterr().out


def test_score_rejects_foreign_verdicts(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps({"population": {}, "items": []}))
    (tmp_path / "v.json").write_text(json.dumps({"verdicts": {"x#1->2": "yes"}}))
    with pytest.raises(SystemExit, match="not in the sample"):
        rv.main(["score", str(tmp_path / "v.json"), "--sample", str(tmp_path / "s.json")])


def test_weighted_interval_is_not_degenerate_when_the_big_rule_is_perfect():
    # The largest rule has 10/10 correct: a plug-in interval would be [~1, 1].
    rules = {
        "comment": {"pairs": 831, "reviewed": 10, "yes": 10},
        "weak_comment": {"pairs": 63, "reviewed": 11, "yes": 10},
        "event": {"pairs": 0, "reviewed": 0, "yes": 0},
    }
    estimate, (low, high) = rv.weighted_precision(rules)
    assert estimate == pytest.approx(831 / 894 + 63 / 894 * 10 / 11)
    assert low < 0.9 and high <= 1.0
    assert rv.weighted_precision({"event": {"pairs": 5, "reviewed": 0, "yes": 0}}) == (None, None)
