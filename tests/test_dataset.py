import json

import pytest

from triage_lens import dataset as ds

UV = ds.REPOS["astral-sh/uv"]
PANDAS = ds.REPOS["pandas-dev/pandas"]

# Bodies below are rendered the way GitHub issue forms render them.
UV_BUG_FORM = """### Summary

Running `uv sync` fails on a workspace with a path dependency.

```
error: Failed to build `foo @ file:///tmp/foo`
```

### Platform

macOS 14 arm64

### Version

uv 0.5.20 (1c17662b3 2025-01-15)

### Python version

Python 3.12.6"""

UV_FEATURE_FORM = """### Summary

Add a `--frozen` flag to `uv tree`.

### Example

_No response_"""

UV_QUESTION_FORM = """### Question

How do I pin the Python version for a single project?

### Platform

_No response_

### Version

_No response_"""

PANDAS_BUG_FORM = """### Pandas version checks

- [X] I have checked that this issue has not already been reported.

- [X] I have confirmed this bug exists on the [latest version](https://pandas.pydata.org) of pandas.

- [ ] I have confirmed this bug exists on the [main branch](https://pandas.pydata.org) of pandas.


### Reproducible Example

```python
import pandas as pd
# build a small frame
df = pd.DataFrame(range(5))
```

### Issue Description

The sum is wrong.

### Expected Behavior

It should be 10.

### Installed Versions

<details>

INSTALLED VERSIONS
------------------
commit                : 0691c5cf90477d3503834d983f69350f250a6ff7
python                : 3.11.4

</details>
"""

PANDAS_QUESTION_FORM = """### Research

- [X] I have searched the [[pandas] tag](https://stackoverflow.com) for similar questions.

- [X] I have asked my usage related question on [StackOverflow](https://stackoverflow.com).


### Link to question on StackOverflow

https://stackoverflow.com/questions/123/how-to-pivot

### Question about pandas

How do I pivot a long table to wide?"""

PANDAS_OLD_TEMPLATE = """#### Code Sample, a copy-pastable example if possible

```python
df.groupby("a").sum()
```
#### Problem description

The index is lost.

<!-- Please add the output of pd.show_versions() below -->
#### Output of ``pd.show_versions()``

<details>
INSTALLED VERSIONS
------------------
pandas: 0.23.0
</details>"""


# --------------------------------------------------------------------------- cleaning


def test_uv_bug_form_keeps_summary_and_log_only():
    clean = ds.clean_body(UV_BUG_FORM, UV.metadata_sections)
    assert clean == (
        "Running `uv sync` fails on a workspace with a path dependency.\n\n"
        "```\nerror: Failed to build `foo @ file:///tmp/foo`\n```"
    )


def test_uv_feature_form_drops_empty_section():
    assert ds.clean_body(UV_FEATURE_FORM, UV.metadata_sections) == (
        "Add a `--frozen` flag to `uv tree`."
    )


def test_uv_question_form():
    assert ds.clean_body(UV_QUESTION_FORM, UV.metadata_sections) == (
        "How do I pin the Python version for a single project?"
    )


def test_pandas_bug_form():
    clean = ds.clean_body(PANDAS_BUG_FORM, PANDAS.metadata_sections)
    assert "# build a small frame" in clean  # a Python comment, not a heading
    assert "The sum is wrong." in clean and "It should be 10." in clean
    for leak in ("I have checked", "INSTALLED VERSIONS", "Reproducible Example", "###"):
        assert leak not in clean


def test_pandas_question_form_drops_stackoverflow_link():
    assert ds.clean_body(PANDAS_QUESTION_FORM, PANDAS.metadata_sections) == (
        "How do I pivot a long table to wide?"
    )


def test_pandas_old_markdown_template():
    clean = ds.clean_body(PANDAS_OLD_TEMPLATE, PANDAS.metadata_sections)
    assert clean == '```python\ndf.groupby("a").sum()\n```\n\nThe index is lost.'


def test_details_with_author_logs_are_kept_without_tags():
    body = "It crashes.\n\n<details><summary>Verbose log</summary>\n\nDEBUG resolving\n</details>"
    assert ds.clean_body(body, UV.metadata_sections) == "It crashes.\n\nDEBUG resolving"


def test_plain_issue_is_left_alone():
    body = "Line one\r\n\r\n\r\n\r\nLine two  "
    assert ds.clean_body(body, UV.metadata_sections) == "Line one\n\nLine two"


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("BUG: read_csv drops a column", "read_csv drops a column"),
        ("ENH/API: add a keyword", "add a keyword"),
        ("DOC - fix a typo", "fix a typo"),
        ("[BUG]: crash on import", "crash on import"),
        ("Bug: uv lock hangs", "uv lock hangs"),
        ("QST:How to merge", "How to merge"),
        ("Debug output is too noisy", "Debug output is too noisy"),
        ("API changes needed for groupby", "API changes needed for groupby"),
        ("CSV: read fails", "CSV: read fails"),
    ],
)
def test_clean_title(title, expected):
    assert ds.clean_title(title) == expected


# --------------------------------------------------------------------------- forms


def test_detect_form_uv():
    def form(body):
        return ds.detect_form("astral-sh/uv", {"title": "t", "body": body})

    assert form(UV_BUG_FORM) == "bug"
    assert form(UV_FEATURE_FORM) == "feature"
    assert form(UV_QUESTION_FORM) == "question"
    assert form("No form at all") is None


def test_detect_form_pandas():
    def form(title):
        return ds.detect_form("pandas-dev/pandas", {"title": title, "body": ""})

    assert form("BUG: wrong sum") == "bug"
    assert form("qst: how to") == "question"
    assert form("PERF: slow merge") == "other"
    assert form("wrong sum") is None


# --------------------------------------------------------------------------- dataset


def issue(number, labels, created="2025-01-01T00:00:00Z", author_type="User", **extra):
    item = {
        "number": number,
        "title": "BUG: something",
        "body": "text",
        "labels": labels,
        "author_type": author_type,
        "created_at": created,
        "updated_at": created,
        "url": f"https://github.com/o/r/issues/{number}",
    }
    item.update(extra)
    return item


def test_target_class():
    assert ds.target_class(issue(1, ["Bug", "Indexing"]), PANDAS) == "bug"
    assert ds.target_class(issue(1, ["Bug", "Docs"]), PANDAS) is None  # ambiguous
    assert ds.target_class(issue(1, ["Indexing"]), PANDAS) is None
    assert ds.target_class(issue(1, ["Bug", "Needs Triage"]), PANDAS) is None  # unreviewed


def test_assign_splits_is_time_ordered():
    records = [{"number": n, "created_at": f"2025-01-{n:02d}"} for n in range(20, 0, -1)]
    ds.assign_splits(records)
    splits = [r["split"] for r in records]
    assert splits == ["train"] * 14 + ["validation"] * 3 + ["test"] * 3
    assert [r["number"] for r in records] == list(range(1, 21))


def test_build_filters_and_records_form():
    issues = {
        1: issue(1, ["Bug"], "2025-01-01T00:00:00Z"),
        2: issue(2, ["Usage Question"], "2025-01-02T00:00:00Z"),  # filed as BUG
        3: issue(3, ["Bug"], author_type="Bot"),
        4: issue(4, ["Bug", "Needs Triage"]),
        5: issue(5, []),
    }
    records, dropped = ds.build("pandas-dev/pandas", issues)
    assert [r["number"] for r in records] == [1, 2]
    assert records[1]["label"] == "question" and records[1]["form"] == "bug"
    assert records[0]["text"] == "something\n\ntext"
    assert dropped == {"bot": 1, "no single reviewed class": 2}


def test_main_writes_dataset(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    lines = [issue(n, ["bug"], f"2025-01-{n:02d}T00:00:00Z") for n in range(1, 11)]
    (raw / "astral-sh__uv.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    ds.main(["astral-sh/uv", "--raw", str(raw), "--out", str(tmp_path / "out")])

    written = (tmp_path / "out" / "astral-sh__uv.dataset.jsonl").read_text().splitlines()
    assert len(written) == 10
    assert "astral-sh/uv: 10 issues kept" in capsys.readouterr().out


# --------------------------------------------------------------------------- prose


def test_prose_text_drops_code_and_keeps_description():
    body = ds.clean_body(PANDAS_BUG_FORM, PANDAS.metadata_sections)
    prose = ds.prose_text({"title": "Wrong sum", "body": body})
    assert prose == "Wrong sum\n\nThe sum is wrong.\n\nIt should be 10."


def test_prose_text_drops_repl_lines_and_tracebacks_outside_fences():
    body = (
        "Calling `df.sum()` fails:\n"
        ">>> df.sum()\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "Is this expected?"
    )
    assert ds.prose_text({"title": "t", "body": body}) == (
        "t\n\nCalling `df.sum()` fails:\nIs this expected?"
    )


def test_prose_text_of_code_only_issue_is_the_title():
    assert ds.prose_text({"title": "t", "body": "```\nx = 1\n```"}) == "t"
