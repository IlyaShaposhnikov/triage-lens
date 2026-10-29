"""Find "duplicate -> original" issue pairs for evaluating duplicate search.

GitHub keeps no reliable machine-readable link between a duplicate and its
original. Maintainers close issues with the "duplicate" reason or label, but the
original is usually named only in a comment, in free form: "Duplicate of #123",
"same as https://github.com/.../issues/123", "covered by #123". The structured
"marked as duplicate" timeline event exists for a small share of issues, and its
canonical/duplicate fields come back swapped, so it only tells which other issue
is involved.

This module therefore works in two steps:

1. For every *candidate* issue (closed as duplicate, or carrying the project's
   duplicate label) download its duplicate events and recent comments.
2. Extract the original with ordered rules, recording which rule fired:
   ``event`` > ``comment`` (strong wording) > ``weak_comment`` (see / follow /
   tracked in) > ``single_ref`` (the thread mentions exactly one other issue).

Usage:
    export GITHUB_TOKEN=github_pat_...
    python -m triage_lens.duplicates astral-sh/uv pandas-dev/pandas

Output for each repository:
    data/raw/astral-sh__uv.threads.jsonl                 events and comments per candidate
    data/processed/astral-sh__uv.duplicate_pairs.jsonl   one pair per line
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NamedTuple

import requests

from triage_lens.collect import load_latest, make_session, post_graphql

BATCH_SIZE = 10
COMMENTS_PER_ISSUE = 50

# Labels each project uses for duplicate reports. pandas also has a "duplicated"
# label, but it is about the DataFrame.duplicated method, not duplicate reports.
DUPLICATE_LABELS = {
    "astral-sh/uv": {"duplicate"},
    "pandas-dev/pandas": {"Duplicate Report"},
}
DEFAULT_DUPLICATE_LABELS = {"duplicate"}
# Old names a repository was reachable under; links to them still appear in comments.
REPO_ALIASES = {
    "pandas-dev/pandas": {"pydata/pandas"},
}

STRONG = re.compile(
    r"\bdup(?:licate[sd]?|e)\b|\bsame (?:as|issue)\b|\bcovered (?:by|in)\b"
    r"|\bin favou?r of\b|\bsuperseded by\b",
    re.IGNORECASE,
)
WEAK = re.compile(r"\btrack(?:ed|ing)? (?:in|by|at)\b|\bfollow(?:ing)?\b|\bsee\b", re.IGNORECASE)
SHORT_REF = re.compile(r"(?<![\w/#&])#(\d+)\b")
URL_REF = re.compile(r"github\.com/([\w.-]+/[\w.-]+)/(?:issues|pull)/(\d+)", re.IGNORECASE)
CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
QUOTE_LINE = re.compile(r"^\s*>.*$", re.MULTILINE)

THREAD_FRAGMENTS = """
fragment Thread on Issue {
  number
  timelineItems(first: 10,
                itemTypes: [MARKED_AS_DUPLICATE_EVENT, UNMARKED_AS_DUPLICATE_EVENT]) {
    nodes {
      __typename
      ... on MarkedAsDuplicateEvent { canonical { ...Ref } duplicate { ...Ref } }
      ... on UnmarkedAsDuplicateEvent { canonical { ...Ref } duplicate { ...Ref } }
    }
  }
  comments(last: COMMENTS) { nodes { author { login } body createdAt } }
}

fragment Ref on IssueOrPullRequest {
  ... on Issue { number url }
  ... on PullRequest { number url }
}
""".replace("COMMENTS", str(COMMENTS_PER_ISSUE))

log = logging.getLogger("triage_lens.duplicates")


class Ref(NamedTuple):
    pos: int
    number: int


# --------------------------------------------------------------------------- download


def is_candidate(issue: dict[str, Any], labels: set[str]) -> bool:
    return issue["state_reason"] == "duplicate" or bool(labels & set(issue["labels"]))


def threads_query(numbers: Iterable[int]) -> str:
    """One query that fetches several issues at once through aliases."""
    fields = "\n".join(f"    i{n}: issue(number: {n}) {{ ...Thread }}" for n in numbers)
    return (
        "query($owner: String!, $name: String!) {\n"
        "  rateLimit { remaining }\n"
        "  repository(owner: $owner, name: $name) {\n"
        f"{fields}\n"
        "  }\n"
        "}\n" + THREAD_FRAGMENTS
    )


def to_thread(number: int, node: dict[str, Any] | None, issue_updated_at: str) -> dict[str, Any]:
    if node is None:  # transferred or deleted since the issue was collected
        return {"number": number, "issue_updated_at": issue_updated_at, "missing": True}
    return {
        "number": number,
        "issue_updated_at": issue_updated_at,
        "events": node["timelineItems"]["nodes"],
        "comments": [
            {
                "author": (comment.get("author") or {}).get("login"),
                "body": comment.get("body") or "",
                "created_at": comment["createdAt"],
            }
            for comment in node["comments"]["nodes"]
        ],
    }


def load_threads(path: Path) -> dict[int, dict[str, Any]]:
    threads: dict[int, dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    thread = json.loads(line)
                    threads[thread["number"]] = thread  # later lines are newer
    return threads


def fetch_threads(
    session: requests.Session,
    repo: str,
    candidates: dict[int, dict[str, Any]],
    path: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[int, dict[str, Any]]:
    """Download threads of candidates that are new or changed since the last run."""
    threads = load_threads(path)
    todo = sorted(
        number
        for number, issue in candidates.items()
        if number not in threads or threads[number]["issue_updated_at"] < issue["updated_at"]
    )
    log.info("%s: %d candidates, %d threads to download", repo, len(candidates), len(todo))
    owner, name = repo.split("/")
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start : start + BATCH_SIZE]
            data = post_graphql(
                session, {"owner": owner, "name": name}, sleep, query=threads_query(batch)
            )
            repository = data.get("repository") or {}
            for number in batch:
                thread = to_thread(
                    number, repository.get(f"i{number}"), candidates[number]["updated_at"]
                )
                out.write(json.dumps(thread, ensure_ascii=False) + "\n")
                threads[number] = thread
            out.flush()
            log.info(
                "%s: threads %d/%d, rate limit left %s",
                repo,
                min(start + BATCH_SIZE, len(todo)),
                len(todo),
                data.get("rateLimit", {}).get("remaining", "?"),
            )
    return threads


# --------------------------------------------------------------------------- extraction


def find_refs(text: str, repo_names: set[str]) -> list[Ref]:
    """Issue references (``#123`` or issue/PR URLs of this repository) in order."""
    refs = [Ref(m.start(), int(m.group(1))) for m in SHORT_REF.finditer(text)]
    refs += [
        Ref(m.start(), int(m.group(2)))
        for m in URL_REF.finditer(text)
        if m.group(1).lower() in repo_names
    ]
    return sorted(refs)


def clean_comment(body: str) -> str:
    """Drop code blocks (tracebacks) and quoted lines (they repeat other comments)."""
    return QUOTE_LINE.sub("", CODE_BLOCK.sub("", body))


def ref_near_keyword(text: str, keyword: re.Pattern[str], refs: list[Ref]) -> int | None:
    """The reference that follows a keyword, or the closest one before it.

    "take a look at #41162. This is a duplicate of #40849" -> 40849.
    """
    matches = list(keyword.finditer(text))
    if not matches or not refs:
        return None
    for match in matches:
        after = [ref for ref in refs if ref.pos >= match.end()]
        if after:
            return after[0].number
    before = [ref for ref in refs if ref.pos < matches[0].start()]
    return before[-1].number if before else None


def event_target(issue_number: int, events: list[dict[str, Any]], repo_names: set[str]):
    """Issue linked by a "marked as duplicate" event, ignoring the swapped field names."""
    targets: dict[int, None] = {}
    for event in events:
        pair = [event.get("canonical"), event.get("duplicate")]
        others = [ref for ref in pair if ref and ref["number"] != issue_number]
        if len(others) != 1 or not _in_repo(others[0]["url"], repo_names):
            continue
        if event["__typename"] == "MarkedAsDuplicateEvent":
            targets[others[0]["number"]] = None
        else:
            targets.pop(others[0]["number"], None)
    return next(iter(targets), None)


def _in_repo(url: str, repo_names: set[str]) -> bool:
    match = URL_REF.search(url)
    return bool(match) and match.group(1).lower() in repo_names


def extract_original(
    issue_number: int, thread: dict[str, Any], repo_names: set[str]
) -> tuple[int, str] | None:
    """(original issue number, rule that found it), or None if nothing was found."""
    target = event_target(issue_number, thread.get("events", []), repo_names)
    if target is not None:
        return target, "event"

    comments = []
    for comment in thread.get("comments", []):
        text = clean_comment(comment["body"])
        refs = [ref for ref in find_refs(text, repo_names) if ref.number != issue_number]
        comments.append((text, refs))

    # The closing comment is usually the last one, so newer comments go first.
    for keyword, source in ((STRONG, "comment"), (WEAK, "weak_comment")):
        for text, refs in reversed(comments):
            target = ref_near_keyword(text, keyword, refs)
            if target is not None:
                return target, source

    mentioned = {ref.number for _, refs in comments for ref in refs}
    if len(mentioned) == 1:
        return mentioned.pop(), "single_ref"
    return None


def build_pairs(
    repo: str, issues: dict[int, dict[str, Any]], threads: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], Counter]:
    repo_names = {repo.lower()} | {alias.lower() for alias in REPO_ALIASES.get(repo, set())}
    pairs, stats = [], Counter()
    for number in sorted(threads):
        thread = threads[number]
        if thread.get("missing"):
            stats["missing"] += 1
            continue
        found = extract_original(number, thread, repo_names)
        if found is None:
            stats["unresolved"] += 1
            continue
        original, source = found
        stats[source] += 1
        pairs.append(
            {
                "duplicate": number,
                "original": original,
                "source": source,
                # False when the reference points to a pull request or a deleted issue.
                "original_is_issue": original in issues,
            }
        )
    return pairs, stats


# --------------------------------------------------------------------------- CLI


def process(
    session: requests.Session, repo: str, raw_dir: Path, out_dir: Path
) -> list[dict[str, Any]]:
    stem = repo.replace("/", "__")
    issues = load_latest(raw_dir / f"{stem}.jsonl")
    if not issues:
        raise SystemExit(f"No issues for {repo} in {raw_dir}; run triage_lens.collect first.")
    labels = DUPLICATE_LABELS.get(repo, DEFAULT_DUPLICATE_LABELS)
    candidates = {n: issue for n, issue in issues.items() if is_candidate(issue, labels)}

    threads = fetch_threads(session, repo, candidates, raw_dir / f"{stem}.threads.jsonl")
    threads = {n: t for n, t in threads.items() if n in candidates}
    pairs, stats = build_pairs(repo, issues, threads)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.duplicate_pairs.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for pair in pairs:
            out.write(json.dumps(pair) + "\n")

    not_issue = sum(1 for pair in pairs if not pair["original_is_issue"])
    log.info(
        "%s: %d candidates -> %d pairs (%s); %d point to a PR or deleted issue",
        repo,
        len(candidates),
        len(pairs),
        ", ".join(f"{key} {value}" for key, value in stats.most_common()),
        not_issue,
    )
    return pairs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract duplicate -> original issue pairs.")
    parser.add_argument("repos", nargs="+", help="repositories as owner/name")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"), help="collected issues")
    parser.add_argument("--out", type=Path, default=Path("data/processed"), help="pairs output")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set the GITHUB_TOKEN environment variable first.")

    session = make_session(token)
    for repo in args.repos:
        process(session, repo, args.raw, args.out)


if __name__ == "__main__":
    main()
