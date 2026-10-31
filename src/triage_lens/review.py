"""Manual review of extracted duplicate -> original pairs.

The pairs in ``data/processed/*.duplicate_pairs.jsonl`` were extracted by rules
(``duplicates.py``), so before they serve as ground truth for duplicate search,
their precision is measured by hand on a stratified sample: the same number of
pairs from every extraction rule of every repository.

``sample`` writes ``annotations/duplicate_review.html``: a self-contained page
with both issues of each pair side by side, the comments that name the original,
and three buttons (yes / no / unsure). Verdicts are kept in the browser while
you work and exported with "Download verdicts" as
``annotations/duplicate_verdicts.json``.

``score`` reads the verdicts and reports precision per rule with a Wilson 95%
interval, plus the estimated precision of all pairs of a repository (each rule
weighted by its share of the pairs). Unsure verdicts are left out of the counts.

Usage:
    python -m triage_lens.review sample astral-sh/uv pandas-dev/pandas
    python -m triage_lens.review score annotations/duplicate_verdicts.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from triage_lens.collect import load_latest
from triage_lens.duplicates import load_threads

SOURCES = ("event", "comment", "weak_comment", "single_ref")
PER_SOURCE = 12
SEED = 0
BODY_CHARS = 900
COMMENT_CHARS = 500
Z95 = 1.96


def pair_id(repo: str, pair: dict[str, Any]) -> str:
    return f"{repo}#{pair['duplicate']}->{pair['original']}"


# --------------------------------------------------------------------------- sampling


def load_pairs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_sample(
    pairs: list[dict[str, Any]], per_source: int = PER_SOURCE, seed: int = SEED
) -> list[dict[str, Any]]:
    """Up to ``per_source`` pairs from every rule; only pairs whose original is an issue."""
    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        if pair["original_is_issue"]:
            by_source[pair["source"]].append(pair)
    sample = []
    for source in SOURCES:
        group = sorted(by_source[source], key=lambda p: p["duplicate"])
        sample += rng.sample(group, min(per_source, len(group)))
    return sample


def mentioning_comments(thread: dict[str, Any], number: int) -> list[dict[str, Any]]:
    """Comments of a thread that refer to issue ``number`` (as #N or a URL)."""
    pattern = re.compile(rf"(?<![\w/#&])#{number}\b|/(?:issues|pull)/{number}\b")
    return [c for c in thread.get("comments", []) if pattern.search(c["body"])]


def review_item(
    repo: str,
    pair: dict[str, Any],
    issues: dict[int, dict[str, Any]],
    threads: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    def issue_view(number: int) -> dict[str, Any]:
        issue = issues[number]
        return {
            "number": number,
            "title": issue["title"],
            "body": issue["body"][:BODY_CHARS],
            "labels": issue["labels"],
            "state": f"{issue['state']} ({issue['state_reason'] or '-'})",
            "url": issue["url"],
        }

    comments = mentioning_comments(threads.get(pair["duplicate"], {}), pair["original"])
    return {
        "id": pair_id(repo, pair),
        "repo": repo,
        "source": pair["source"],
        "duplicate": issue_view(pair["duplicate"]),
        "original": issue_view(pair["original"]),
        "evidence": [
            {"author": c["author"], "body": c["body"][:COMMENT_CHARS]} for c in comments[-3:]
        ],
    }


# --------------------------------------------------------------------------- page

PAGE_TEMPLATE = Path(__file__).with_name("review_page.html")


def render_page(items: list[dict[str, Any]]) -> str:
    # "</" inside the JSON would end the <script> element early.
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    return PAGE_TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", data)


# --------------------------------------------------------------------------- scoring


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def precision_report(
    verdicts: dict[str, str], sources: dict[str, str], population: dict[str, Counter]
) -> dict[str, dict[str, Any]]:
    """Per repository: precision per rule and the weighted estimate for all its pairs.

    ``sources`` maps a pair id to its rule; ``population[repo][source]`` is the number
    of pairs of that rule (original is an issue), used as the rule's weight.
    """
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for item_id, verdict in verdicts.items():
        repo = item_id.split("#", 1)[0]
        counts[repo][sources[item_id]][verdict] += 1

    report = {}
    for repo, by_source in counts.items():
        total = sum(population[repo].values())
        rules, estimate, variance = {}, 0.0, 0.0
        for source in SOURCES:
            c = by_source.get(source, Counter())
            n = c["yes"] + c["no"]
            size = population[repo][source]
            rules[source] = {
                "pairs": size,
                "reviewed": n,
                "unsure": c["unsure"],
                "yes": c["yes"],
                "precision": c["yes"] / n if n else None,
                "ci": wilson(c["yes"], n),
            }
            if n and total:
                weight, p = size / total, c["yes"] / n
                estimate += weight * p
                variance += weight * weight * p * (1 - p) / n
        reviewed_weight = sum(population[repo][s] for s in SOURCES if rules[s]["reviewed"]) / max(
            total, 1
        )
        overall = estimate / reviewed_weight if reviewed_weight else None
        half = Z95 * math.sqrt(variance) / reviewed_weight if reviewed_weight else None
        report[repo] = {
            "rules": rules,
            "estimated_precision": overall,
            "ci": [max(0.0, overall - half), min(1.0, overall + half)] if overall else None,
        }
    return report


# --------------------------------------------------------------------------- CLI


def cmd_sample(args) -> None:
    items, population = [], {}
    for repo in args.repos:
        stem = repo.replace("/", "__")
        pairs = load_pairs(args.processed / f"{stem}.duplicate_pairs.jsonl")
        issues = load_latest(args.raw / f"{stem}.jsonl")
        threads = load_threads(args.raw / f"{stem}.threads.jsonl")
        population[repo] = dict(Counter(p["source"] for p in pairs if p["original_is_issue"]))
        sample = stratified_sample(pairs, args.per_source)
        items += [review_item(repo, pair, issues, threads) for pair in sample]
        print(f"{repo}: {len(sample)} pairs sampled from {population[repo]}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "duplicate_review.html").write_text(render_page(items), encoding="utf-8")
    sample_file = args.out / "duplicate_review.sample.json"
    sample_file.write_text(
        json.dumps(
            {
                "seed": SEED,
                "per_source": args.per_source,
                "population": population,
                "items": [{"id": it["id"], "source": it["source"]} for it in items],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"open {args.out / 'duplicate_review.html'} in a browser; {len(items)} pairs")


def cmd_score(args) -> None:
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    sources = {item["id"]: item["source"] for item in sample["items"]}
    verdicts = json.loads(args.verdicts.read_text(encoding="utf-8"))["verdicts"]
    unknown = set(verdicts) - set(sources)
    if unknown:
        raise SystemExit(f"verdicts for pairs not in the sample, e.g. {sorted(unknown)[0]}")
    population = {repo: Counter(sizes) for repo, sizes in sample["population"].items()}
    report = precision_report(verdicts, sources, population)

    for repo, r in report.items():
        print(f"\n=== {repo}: {sum(population[repo].values())} pairs")
        print(f"{'rule':<14}{'pairs':>6}{'reviewed':>9}{'unsure':>7}{'precision':>11}  95% CI")
        for source, s in r["rules"].items():
            prec = "-" if s["precision"] is None else f"{s['precision']:.0%}"
            ci = "" if s["ci"] is None else f"[{s['ci'][0]:.0%}-{s['ci'][1]:.0%}]"
            print(f"{source:<14}{s['pairs']:>6}{s['reviewed']:>9}{s['unsure']:>7}{prec:>11}  {ci}")
        if r["estimated_precision"] is not None:
            lo, hi = r["ci"]
            print(
                f"all pairs (weighted by rule): {r['estimated_precision']:.0%} [{lo:.0%}-{hi:.0%}]"
            )
        path = args.results / repo.replace("/", "__") / "duplicate_pairs_precision.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(r, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manual review of duplicate pairs.")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="build the review page")
    s.add_argument("repos", nargs="+")
    s.add_argument("--per-source", type=int, default=PER_SOURCE)
    s.add_argument("--raw", type=Path, default=Path("data/raw"))
    s.add_argument("--processed", type=Path, default=Path("data/processed"))
    s.add_argument("--out", type=Path, default=Path("annotations"))
    c = sub.add_parser("score", help="precision from exported verdicts")
    c.add_argument("verdicts", type=Path)
    c.add_argument("--sample", type=Path, default=Path("annotations/duplicate_review.sample.json"))
    c.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    {"sample": cmd_sample, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    main()
