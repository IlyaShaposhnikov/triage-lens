"""Build the issue-type classification dataset from collected issues.

Steps for each repository:

1. Keep the newest version of every issue, written by a human (not a bot).
2. Target = the maintainer's final label, mapped to one of four classes
   (bug / feature / question / docs). Issues with none or several of these
   classes are dropped, and so are issues whose label nobody has reviewed yet
   (pandas: still marked "Needs Triage", so the label is just the form default).
3. Remove every trace of the issue form from the text: title prefixes such as
   "BUG:", form section headings, checkboxes, metadata sections (platform,
   versions, ``pd.show_versions()`` output) and HTML comments. Issue forms set
   the label automatically, so these traces would let a model read the label
   off the template instead of learning from what the author wrote.
4. Record which form the author picked (``form``). It is never a model input;
   it serves as the "trust the form" baseline and marks misfiled issues, where
   the maintainer's label differs from the author's choice.
5. Split by creation time within each repository: oldest 70% train, next 15%
   validation, newest 15% test.

Usage:
    python -m triage_lens.dataset astral-sh/uv pandas-dev/pandas
    python -m triage_lens.dataset astral-sh/uv --examples 3   # show cleaned samples

Output: data/processed/astral-sh__uv.dataset.jsonl, one issue per line.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triage_lens.collect import load_latest

CLASSES = ("bug", "feature", "question", "docs")
SPLITS = (("train", 0.70), ("validation", 0.15), ("test", 0.15))


@dataclass(frozen=True)
class RepoConfig:
    labels: dict[str, str]  # project label -> class
    unreviewed_labels: frozenset[str]  # label still set by the form, not by a person
    metadata_sections: frozenset[str]  # form sections removed with their content


REPOS = {
    "astral-sh/uv": RepoConfig(
        labels={
            "bug": "bug",
            "enhancement": "feature",
            "question": "question",
            "documentation": "docs",
        },
        unreviewed_labels=frozenset(),
        metadata_sections=frozenset({"platform", "version", "python version"}),
    ),
    "pandas-dev/pandas": RepoConfig(
        labels={
            "Bug": "bug",
            "Enhancement": "feature",
            "Usage Question": "question",
            "Docs": "docs",
        },
        unreviewed_labels=frozenset({"Needs Triage"}),
        metadata_sections=frozenset(
            {
                "installed versions",
                "output of pd.show_versions()",
                "output of ``pd.show_versions()``",
                "link to question on stackoverflow",
                "pandas version checks",
                "research",
                "installation check",
                "platform",
                "installation method",
                "pandas version",
                "python version",
                "feature type",
            }
        ),
    ),
}

# Prefixes authors put in titles, by convention or because a form adds them.
TITLE_PREFIX = re.compile(
    r"^\s*(?:\[?(?:bug|enh|doc|docs|qst|perf|build|vote|api|regr|tst|cln|ci|depr|compat|err"
    r"|typ|ref|wip|pdep|feat|feature|question)\]?\s*[/,&+]?\s*)+[:\-]\s*",
    re.IGNORECASE,
)
PANDAS_FORM_PREFIX = re.compile(r"^\s*(BUG|ENH|DOC|QST|PERF|BUILD|VOTE)\s*:", re.IGNORECASE)
PANDAS_FORMS = {"BUG": "bug", "ENH": "feature", "DOC": "docs", "QST": "question"}

HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
DETAILS = re.compile(r"<details>.*?</details>", re.DOTALL | re.IGNORECASE)
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
CHECKBOX = re.compile(r"^\s*[-*+]\s+\[[ xX]\]\s")
FENCE = re.compile(r"^\s*(```|~~~)")
FORM_LEFTOVERS = {"_no response_", "replace this line with the output of pd.show_versions()"}


# --------------------------------------------------------------------------- forms


def section_headings(body: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"^###\s+(.+?)\s*$", body, re.MULTILINE)}


def detect_form(repo: str, issue: dict[str, Any]) -> str | None:
    """Which issue form the author picked, judged from the raw issue."""
    if repo == "pandas-dev/pandas":
        match = PANDAS_FORM_PREFIX.match(issue["title"])
        if not match:
            return None
        return PANDAS_FORMS.get(match.group(1).upper(), "other")
    if repo == "astral-sh/uv":
        headings = section_headings(issue["body"])
        if {"Question", "Platform"} <= headings:
            return "question"
        if {"Summary", "Python version"} <= headings:
            return "bug"
        if {"Summary", "Example"} <= headings:
            return "feature"
    return None


# --------------------------------------------------------------------------- cleaning


def clean_title(title: str) -> str:
    return TITLE_PREFIX.sub("", title, count=1).strip()


def _remove_versions_dump(match: re.Match[str]) -> str:
    block = match.group(0)
    upper = block.upper()
    if "INSTALLED VERSIONS" in upper or "PD.SHOW_VERSIONS()" in upper:
        return ""
    # Other collapsible blocks hold the author's logs: keep them, drop the tags.
    return re.sub(r"</?details>|<summary>.*?</summary>", "", block, flags=re.IGNORECASE | re.DOTALL)


def clean_body(body: str, metadata_sections: frozenset[str]) -> str:
    """Remove issue-form traces and keep what the author wrote."""
    text = body.replace("\r\n", "\n")
    text = HTML_COMMENT.sub("", text)
    text = DETAILS.sub(_remove_versions_dump, text)

    kept: list[str] = []
    in_fence = skipping = False
    for line in text.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
            if not skipping:
                kept.append(line)
            continue
        if not in_fence:
            heading = HEADING.match(line)
            if heading:
                name = heading.group(1).strip().rstrip(":").lower()
                skipping = name in metadata_sections
                continue  # headings are dropped either way
        if skipping:
            continue
        if not in_fence and (CHECKBOX.match(line) or line.strip().lower() in FORM_LEFTOVERS):
            continue
        kept.append(line)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# --------------------------------------------------------------------------- dataset


def target_class(issue: dict[str, Any], config: RepoConfig) -> str | None:
    """The single class of the maintainer's label, or None if absent or ambiguous."""
    if config.unreviewed_labels & set(issue["labels"]):
        return None
    classes = {config.labels[label] for label in issue["labels"] if label in config.labels}
    return classes.pop() if len(classes) == 1 else None


def assign_splits(records: list[dict[str, Any]]) -> None:
    """Time-based split: oldest issues train, newest test."""
    records.sort(key=lambda r: (r["created_at"], r["number"]))
    n = len(records)
    bounds, start = [], 0
    for name, share in SPLITS[:-1]:
        end = start + int(n * share)
        bounds.append((name, start, end))
        start = end
    bounds.append((SPLITS[-1][0], start, n))
    for name, lo, hi in bounds:
        for record in records[lo:hi]:
            record["split"] = name


def build(repo: str, issues: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    config = REPOS[repo]
    records, dropped = [], Counter()
    for issue in issues.values():
        if issue["author_type"] == "Bot":
            dropped["bot"] += 1
            continue
        label = target_class(issue, config)
        if label is None:
            dropped["no single reviewed class"] += 1
            continue
        title = clean_title(issue["title"])
        body = clean_body(issue["body"], config.metadata_sections)
        records.append(
            {
                "repo": repo,
                "number": issue["number"],
                "title": title,
                "body": body,
                "text": f"{title}\n\n{body}".strip(),
                "label": label,
                "form": detect_form(repo, issue),
                "created_at": issue["created_at"],
                "url": issue["url"],
            }
        )
    assign_splits(records)
    return records, dropped


# --------------------------------------------------------------------------- CLI


def report(repo: str, records: list[dict[str, Any]], dropped: Counter) -> None:
    print(f"\n=== {repo}: {len(records)} issues kept; dropped: {dict(dropped)}")
    print(f"{'split':<11}{'n':>6}  {'from':<11}{'to':<11}" + "".join(f"{c:>9}" for c in CLASSES))
    for name, _ in SPLITS:
        part = [r for r in records if r["split"] == name]
        if not part:
            continue
        counts = Counter(r["label"] for r in part)
        print(
            f"{name:<11}{len(part):>6}  {part[0]['created_at'][:10]:<11}"
            f"{part[-1]['created_at'][:10]:<11}" + "".join(f"{counts[c]:>9}" for c in CLASSES)
        )
    for name, _ in SPLITS:
        part = [r for r in records if r["split"] == name and r["form"] in CLASSES]
        misfiled = sum(r["form"] != r["label"] for r in part)
        if part:
            print(
                f"{name:<11} filed via a form: {len(part)}, "
                f"misfiled: {misfiled} ({100 * misfiled / len(part):.1f}%)"
            )


def show_examples(records: list[dict[str, Any]], raw: dict[int, dict[str, Any]], n: int) -> None:
    rng = random.Random(0)
    with_form = [r for r in records if r["form"]]
    for record in rng.sample(with_form, min(n, len(with_form))):
        original = raw[record["number"]]
        print(f"\n----- #{record['number']} label={record['label']} form={record['form']}")
        print(f"RAW TITLE:   {original['title']}")
        print(f"RAW BODY:\n{original['body'][:1200]}")
        print(f"\nCLEAN TEXT:\n{record['text'][:1200]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the issue-type classification dataset.")
    parser.add_argument("repos", nargs="+", choices=sorted(REPOS), help="repositories")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--examples", type=int, default=0, help="print N cleaned samples")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for repo in args.repos:
        stem = repo.replace("/", "__")
        issues = load_latest(args.raw / f"{stem}.jsonl")
        if not issues:
            raise SystemExit(f"No issues for {repo} in {args.raw}; run triage_lens.collect first.")
        records, dropped = build(repo, issues)
        with (args.out / f"{stem}.dataset.jsonl").open("w", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        report(repo, records, dropped)
        if args.examples:
            show_examples(records, issues, args.examples)


if __name__ == "__main__":
    main()
