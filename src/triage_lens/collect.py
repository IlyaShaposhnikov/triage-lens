"""Download the issues of a GitHub repository into a JSON Lines file.

The collector walks a repository's issues in order of last update, one page of
100 at a time, and saves its position after every page, so an interrupted run
resumes where it stopped. Pull requests, which the GitHub issues endpoint
returns mixed in with issues, are skipped. Rate limits are respected: the
collector waits for the limit to reset instead of failing.

Usage:
    export GITHUB_TOKEN=github_pat_...
    python -m triage_lens.collect astral-sh/uv pandas-dev/pandas
    python -m triage_lens.collect astral-sh/uv --max-pages 2   # quick trial run

Output for each repository (``owner__name`` stem):
    data/raw/astral-sh__uv.jsonl        one issue version per line
    data/raw/astral-sh__uv.state.json   cursor: last seen update time

Running the command again continues from the saved cursor, so a later run
fetches both new issues and issues changed since (new labels, closed). A changed
issue is appended as a new line; the newest version of each issue number wins
when the dataset is built.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

API_URL = "https://api.github.com"
PER_PAGE = 100
MAX_ATTEMPTS = 6
TIMEOUT_S = 30

log = logging.getLogger("triage_lens.collect")


def make_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "triage-lens",
        }
    )
    return session


def rate_limit_delay(resp: requests.Response, now: float | None = None) -> float | None:
    """Seconds to wait if the response is a rate-limit rejection, otherwise None.

    GitHub signals two kinds of limits with HTTP 403 or 429: the secondary limit
    comes with a Retry-After header, the primary hourly limit with
    X-RateLimit-Remaining: 0 and a reset timestamp. Any other 403 is a real
    permission error and must not be retried.
    """
    if resp.status_code not in (403, 429):
        return None
    if "Retry-After" in resp.headers:
        return float(resp.headers["Retry-After"])
    if resp.headers.get("X-RateLimit-Remaining") == "0":
        now = time.time() if now is None else now
        reset = float(resp.headers.get("X-RateLimit-Reset", now))
        return max(reset - now, 0.0) + 1.0
    return None


def get_with_retry(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    sleep: Callable[[float], None] = time.sleep,
) -> requests.Response:
    """GET a URL, waiting out rate limits and retrying network and 5xx errors."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = session.get(url, params=params, timeout=TIMEOUT_S)
        except (requests.ConnectionError, requests.Timeout) as exc:
            delay = 2.0**attempt
            log.warning("%s, retrying in %.0fs", type(exc).__name__, delay)
            sleep(delay)
            continue

        delay = rate_limit_delay(resp)
        if delay is None and resp.status_code >= 500:
            delay = 2.0**attempt
        if delay is None:
            resp.raise_for_status()
            return resp
        log.warning("HTTP %d, waiting %.0fs", resp.status_code, delay)
        sleep(delay)
    raise RuntimeError(f"Giving up on {url} after {MAX_ATTEMPTS} attempts")


def to_record(item: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields the project uses from a GitHub issue object."""
    user = item.get("user") or {}
    return {
        "number": item["number"],
        "title": item["title"],
        "body": item.get("body") or "",
        "labels": [label["name"] for label in item.get("labels", [])],
        "state": item["state"],
        "state_reason": item.get("state_reason"),
        "created_at": item["created_at"],
        "updated_at": item["updated_at"],
        "closed_at": item.get("closed_at"),
        "author": user.get("login"),
        "author_type": user.get("type"),
        "author_association": item.get("author_association"),
        "comments": item.get("comments", 0),
        "url": item["html_url"],
    }


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if "since" in state:
            return state
    return {"since": None, "page": 1}


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


def next_cursor(since: str | None, page: int, items: list[dict[str, Any]]) -> tuple[str, int]:
    """Where the next request starts after a full page.

    Normally the cursor moves to the newest ``updated_at`` on the page and the
    page number resets to 1. If every item shares the current cursor timestamp
    (more than a page of issues updated in the same second), the cursor cannot
    move, so the page number is advanced instead.
    """
    newest = max(item["updated_at"] for item in items)
    if newest != since:
        return newest, 1
    return since, page + 1


def collect(
    session: requests.Session,
    repo: str,
    out_dir: Path,
    max_pages: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Download issues of ``repo`` into ``out_dir``; return how many records were added.

    GitHub refuses page numbers beyond the first 10,000 results, so issues are
    walked in order of last update and ``since`` (updated at or after) serves as
    the cursor. Each request asks for the first pages after the cursor.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = repo.replace("/", "__")
    data_path = out_dir / f"{stem}.jsonl"
    state_path = out_dir / f"{stem}.state.json"

    state = load_state(state_path)
    since, page = state["since"], state["page"]
    seen = load_seen_versions(data_path)
    numbers = {number for number, _ in seen}
    url = f"{API_URL}/repos/{repo}/issues"
    base = {"state": "all", "sort": "updated", "direction": "asc", "per_page": PER_PAGE}
    log.info(
        "%s: starting from %s, %d issues already saved",
        repo,
        since or "the beginning",
        len(numbers),
    )

    added = pages_fetched = 0
    with data_path.open("a", encoding="utf-8") as out:
        while max_pages is None or pages_fetched < max_pages:
            params = {**base, "page": page}
            if since:
                params["since"] = since
            resp = get_with_retry(session, url, params, sleep)
            items = resp.json()

            new_on_page = 0
            for item in items:
                key = (item["number"], item["updated_at"])
                if "pull_request" in item or key in seen:
                    continue
                out.write(json.dumps(to_record(item), ensure_ascii=False) + "\n")
                seen.add(key)
                numbers.add(item["number"])
                new_on_page += 1
            out.flush()
            added += new_on_page
            pages_fetched += 1

            last = len(items) < PER_PAGE
            if items:
                since, page = next_cursor(since, page, items)
            if last:
                page = 1  # next run re-reads from the cursor to catch new updates
            save_state(state_path, {"since": since, "page": page})
            log.info(
                "%s: up to %s, +%d records (%d issues), rate limit left %s",
                repo,
                since,
                new_on_page,
                len(numbers),
                resp.headers.get("X-RateLimit-Remaining", "?"),
            )
            if last:
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
        log.info("%s: done, %d new issues", repo, added)


if __name__ == "__main__":
    main()
