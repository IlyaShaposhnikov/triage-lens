"""Download the issues of a GitHub repository into a JSON Lines file.

The collector uses the GitHub GraphQL API, which returns issues only (no pull
requests) and pages through them with an opaque cursor. Issues are read in order
of last update; the cursor is saved after every page, so an interrupted run
resumes where it stopped. Rate limits are respected: the collector waits for the
limit to reset instead of failing.

Besides the issue itself, each record keeps the issues it was marked as a
duplicate of (``Duplicate of #N``), taken from the issue timeline.

Usage:
    export GITHUB_TOKEN=github_pat_...
    python -m triage_lens.collect astral-sh/uv pandas-dev/pandas
    python -m triage_lens.collect astral-sh/uv --max-pages 2   # quick trial run

Output for each repository (``owner__name`` stem):
    data/raw/astral-sh__uv.jsonl        one issue version per line
    data/raw/astral-sh__uv.state.json   resume cursor and refresh point

Once a run reaches the most recently updated issue, the next run fetches only
issues created or changed since then (new labels, closed, marked as duplicate).
A changed issue is appended as a new line; the newest version of each issue
number wins when the dataset is built.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

GRAPHQL_URL = "https://api.github.com/graphql"
PAGE_SIZE = 50
MAX_ATTEMPTS = 6
TIMEOUT_S = 60
# A refresh starts a little before the last seen update, so issues updated in the
# same second are not lost; repeats are skipped by (number, updated_at).
REFRESH_OVERLAP = timedelta(minutes=5)

QUERY = """
query($owner: String!, $name: String!, $first: Int!, $after: String, $since: DateTime) {
  rateLimit { remaining }
  repository(owner: $owner, name: $name) {
    issues(first: $first, after: $after,
           orderBy: {field: UPDATED_AT, direction: ASC},
           filterBy: {since: $since}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        body
        url
        state
        stateReason(enableDuplicate: true)
        createdAt
        updatedAt
        closedAt
        authorAssociation
        author { login __typename }
        labels(first: 20) { nodes { name } }
        comments { totalCount }
        timelineItems(first: 20,
                      itemTypes: [MARKED_AS_DUPLICATE_EVENT, UNMARKED_AS_DUPLICATE_EVENT]) {
          nodes {
            __typename
            ... on MarkedAsDuplicateEvent {
              canonical { ...Ref }
              duplicate { ...Ref }
            }
            ... on UnmarkedAsDuplicateEvent {
              canonical { ...Ref }
              duplicate { ...Ref }
            }
          }
        }
      }
    }
  }
}

fragment Ref on IssueOrPullRequest {
  ... on Issue { number url }
  ... on PullRequest { number url }
}
"""

log = logging.getLogger("triage_lens.collect")


class GraphQLError(RuntimeError):
    pass


def make_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "User-Agent": "triage-lens",
        }
    )
    return session


def rate_limit_delay(resp: requests.Response, now: float | None = None) -> float | None:
    """Seconds to wait if the response is a rate-limit rejection, otherwise None.

    GitHub signals the secondary limit with a Retry-After header and the primary
    hourly limit with X-RateLimit-Remaining: 0 and a reset timestamp. GraphQL may
    report the primary limit as HTTP 200 with a RATE_LIMITED error in the body.
    Any other 403 is a real permission error and must not be retried.
    """
    limited = resp.status_code in (403, 429) or _has_rate_limited_error(resp)
    if not limited:
        return None
    if "Retry-After" in resp.headers:
        return float(resp.headers["Retry-After"])
    if resp.headers.get("X-RateLimit-Remaining") == "0" or _has_rate_limited_error(resp):
        now = time.time() if now is None else now
        reset = float(resp.headers.get("X-RateLimit-Reset", now + 60))
        return max(reset - now, 0.0) + 1.0
    return None


def _has_rate_limited_error(resp: requests.Response) -> bool:
    if resp.status_code != 200:
        return False
    try:
        errors = resp.json().get("errors") or []
    except ValueError:
        return False
    return any(error.get("type") == "RATE_LIMITED" for error in errors)


def post_graphql(
    session: requests.Session,
    variables: dict[str, Any],
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run the issues query, waiting out rate limits and retrying transient errors."""
    payload = {"query": QUERY, "variables": variables}
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = session.post(GRAPHQL_URL, json=payload, timeout=TIMEOUT_S)
        except (requests.ConnectionError, requests.Timeout) as exc:
            delay = 2.0**attempt
            log.warning("%s, retrying in %.0fs", type(exc).__name__, delay)
            sleep(delay)
            continue

        delay = rate_limit_delay(resp)
        if delay is None and resp.status_code >= 500:
            delay = 2.0**attempt  # heavy GraphQL queries sometimes time out with 502
        if delay is not None:
            log.warning("HTTP %d, waiting %.0fs", resp.status_code, delay)
            sleep(delay)
            continue

        resp.raise_for_status()
        body = resp.json()
        if body.get("errors"):
            messages = "; ".join(error.get("message", "?") for error in body["errors"])
            if not body.get("data"):
                raise GraphQLError(messages)
            # Partial errors, e.g. a duplicate pointing into a deleted repository.
            log.warning("GraphQL returned partial errors: %s", messages)
        return body["data"]
    raise GraphQLError(f"Giving up after {MAX_ATTEMPTS} attempts")


def duplicate_targets(issue_number: int, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Issues this one is currently marked as a duplicate of.

    Timeline events are replayed in order: a mark adds the canonical issue, an
    unmark removes it. Events where this issue is the canonical one are ignored.
    """
    targets: dict[str, dict[str, Any]] = {}
    for event in events:
        canonical, duplicate = event.get("canonical"), event.get("duplicate")
        if not canonical or not duplicate or duplicate.get("number") != issue_number:
            continue
        if event["__typename"] == "MarkedAsDuplicateEvent":
            targets[canonical["url"]] = {"number": canonical["number"], "url": canonical["url"]}
        else:
            targets.pop(canonical["url"], None)
    return list(targets.values())


def to_record(node: dict[str, Any]) -> dict[str, Any]:
    """Flatten a GraphQL issue node into the record saved to disk."""
    author = node.get("author") or {}
    reason = node.get("stateReason")
    return {
        "number": node["number"],
        "title": node["title"],
        "body": node.get("body") or "",
        "labels": [label["name"] for label in node["labels"]["nodes"]],
        "state": node["state"].lower(),
        "state_reason": reason.lower() if reason else None,
        "created_at": node["createdAt"],
        "updated_at": node["updatedAt"],
        "closed_at": node.get("closedAt"),
        "author": author.get("login"),
        "author_type": author.get("__typename"),
        "author_association": node.get("authorAssociation"),
        "comments": node["comments"]["totalCount"],
        "duplicate_of": duplicate_targets(node["number"], node["timelineItems"]["nodes"]),
        "url": node["url"],
    }


def load_state(path: Path) -> dict[str, Any]:
    """Resume state; files from the earlier REST-based collector are ignored."""
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("version") == 2:
            return state
    return {"version": 2, "since": None, "cursor": None, "newest": None}


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Write the state atomically so a crash never leaves a half-written file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_seen_versions(path: Path) -> set[tuple[int, str]]:
    """(number, updated_at) pairs already saved, used to skip exact repeats."""
    if not path.exists():
        return set()
    seen = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                seen.add((record["number"], record["updated_at"]))
    return seen


def refresh_point(newest: str) -> str:
    """Timestamp a later run starts from: the newest update minus a safety overlap."""
    moment = datetime.fromisoformat(newest.replace("Z", "+00:00")) - REFRESH_OVERLAP
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def collect(
    session: requests.Session,
    repo: str,
    out_dir: Path,
    max_pages: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Download issues of ``repo`` into ``out_dir``; return how many records were added."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = repo.replace("/", "__")
    data_path = out_dir / f"{stem}.jsonl"
    state_path = out_dir / f"{stem}.state.json"
    owner, name = repo.split("/")

    state = load_state(state_path)
    seen = load_seen_versions(data_path)
    numbers = {number for number, _ in seen}
    log.info(
        "%s: %s, %d issues already saved",
        repo,
        f"updates since {state['since']}" if state["since"] else "full download",
        len(numbers),
    )

    added = pages_fetched = 0
    with data_path.open("a", encoding="utf-8") as out:
        while max_pages is None or pages_fetched < max_pages:
            variables = {
                "owner": owner,
                "name": name,
                "first": PAGE_SIZE,
                "after": state["cursor"],
                "since": state["since"],
            }
            data = post_graphql(session, variables, sleep)
            if data.get("repository") is None:
                raise GraphQLError(f"Repository {repo} not found or not accessible")
            issues = data["repository"]["issues"]

            new_on_page = 0
            for node in issues["nodes"]:
                key = (node["number"], node["updatedAt"])
                if key in seen:
                    continue
                out.write(json.dumps(to_record(node), ensure_ascii=False) + "\n")
                seen.add(key)
                numbers.add(node["number"])
                new_on_page += 1
                state["newest"] = max(state["newest"] or "", node["updatedAt"])
            out.flush()
            added += new_on_page
            pages_fetched += 1

            page_info = issues["pageInfo"]
            finished = not page_info["hasNextPage"]
            if finished:
                # Next run: only issues updated since the newest one seen.
                if state["newest"]:
                    state["since"] = refresh_point(state["newest"])
                state["cursor"] = None
            else:
                state["cursor"] = page_info["endCursor"]
            save_state(state_path, state)

            log.info(
                "%s: up to %s, +%d records (%d issues), rate limit left %s",
                repo,
                state["newest"],
                new_on_page,
                len(numbers),
                data.get("rateLimit", {}).get("remaining", "?"),
            )
            if finished:
                log.info("%s: reached the most recently updated issues", repo)
                break
    return added


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download GitHub issues to JSON Lines.")
    parser.add_argument("repos", nargs="+", help="repositories as owner/name")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="output directory")
    parser.add_argument("--max-pages", type=int, default=None, help="stop after N pages")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set the GITHUB_TOKEN environment variable first.")

    session = make_session(token)
    for repo in args.repos:
        added = collect(session, repo, args.out, max_pages=args.max_pages)
        log.info("%s: done, %d new records", repo, added)


if __name__ == "__main__":
    main()
