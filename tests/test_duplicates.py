import json

import requests

from triage_lens import duplicates as d

UV = {"astral-sh/uv"}
PANDAS = {"pandas-dev/pandas", "pydata/pandas"}


def thread(*bodies, events=()):
    return {
        "number": 0,
        "issue_updated_at": "2025-01-01T00:00:00Z",
        "events": list(events),
        "comments": [{"author": "m", "body": b, "created_at": "2025"} for b in bodies],
    }


def uv_ref(number):
    return {"number": number, "url": f"https://github.com/astral-sh/uv/issues/{number}"}


# --------------------------------------------------------------------- real comments


def test_duplicate_of_url():
    # astral-sh/uv#14959
    t = thread("Duplicate of https://github.com/astral-sh/uv/issues/13878")
    assert d.extract_original(14959, t, UV) == (13878, "comment")


def test_same_as_url():
    # astral-sh/uv#15696
    t = thread("CC @Gankra", "This is the same as https://github.com/astral-sh/uv/issues/10709")
    assert d.extract_original(15696, t, UV) == (10709, "comment")


def test_weak_wording_suggest_following():
    # astral-sh/uv#12403
    t = thread(
        "Generally no, because the virtual environment will point to a Python elsewhere.",
        "I suggest following https://github.com/astral-sh/uv/issues/11746.",
    )
    assert d.extract_original(12403, t, UV) == (11746, "weak_comment")


def test_question_confirmed_in_next_comment():
    # astral-sh/uv#16803
    t = thread(
        "Is this a duplicate of https://github.com/astral-sh/uv/issues/14992 ?",
        "Oops, yes. I searched for 'cooldown' as a keyword and missed that.",
    )
    assert d.extract_original(16803, t, UV) == (14992, "comment")


def test_short_reference():
    # pandas-dev/pandas#38498
    assert d.extract_original(38498, thread("duplicate of #38424"), PANDAS) == (38424, "comment")


def test_reference_after_keyword_wins_over_earlier_one():
    # pandas-dev/pandas#41660
    t = thread(
        "take a look at #41162 . this is duplicate of #40849. in my opinion ...",
        "Closing as duplicate, the other issue covers this",
    )
    assert d.extract_original(41660, t, PANDAS) == (40849, "comment")


def test_covered_by():
    # pandas-dev/pandas#28625
    t = thread("I'll close this since I think it is covered by #22824")
    assert d.extract_original(28625, t, PANDAS) == (22824, "comment")


def test_old_repository_name_in_url():
    # pandas-dev/pandas#13118
    t = thread("And it is a duplicate of https://github.com/pydata/pandas/issues/13098 Thanks")
    assert d.extract_original(13118, t, PANDAS) == (13098, "comment")


def test_single_reference_fallback():
    # pandas-dev/pandas#18322: "a duplicate of that" refers to the xref above it
    # The keyword comment has no reference, so the single-reference rule decides.
    t = thread("xref #18301", "Yes, this is a duplicate of that. See also the whatsnew notes")
    assert d.extract_original(18322, t, PANDAS) == (18301, "single_ref")


def test_no_reference_is_unresolved():
    # pandas-dev/pandas#45553
    assert d.extract_original(45553, thread("closing as duplicate"), PANDAS) is None


def test_ambiguous_references_without_keyword_are_unresolved():
    assert d.extract_original(5, thread("related: #1", "also #2"), UV) is None


def test_single_ref_rule():
    assert d.extract_original(5, thread("Thanks!", "Fixed upstream in #7"), UV) == (
        7,
        "single_ref",
    )


# ------------------------------------------------------------------------ edge cases


def test_event_with_swapped_fields():
    # astral-sh/uv#14959 as GitHub returns it: canonical and duplicate are reversed.
    event = {
        "__typename": "MarkedAsDuplicateEvent",
        "canonical": uv_ref(14959),
        "duplicate": uv_ref(13878),
    }
    assert d.extract_original(14959, thread(events=[event]), UV) == (13878, "event")


def test_unmarked_event_cancels_mark():
    mark = {"__typename": "MarkedAsDuplicateEvent", "canonical": uv_ref(1), "duplicate": uv_ref(2)}
    unmark = dict(mark, __typename="UnmarkedAsDuplicateEvent")
    assert d.event_target(2, [mark, unmark], UV) is None


def test_event_into_other_repository_is_ignored():
    other = {"number": 9, "url": "https://github.com/pypa/pip/issues/9"}
    event = {"__typename": "MarkedAsDuplicateEvent", "canonical": other, "duplicate": uv_ref(2)}
    assert d.event_target(2, [event], UV) is None


def test_references_in_code_quotes_and_other_repos_are_ignored():
    body = (
        "> quoting an earlier comment that mentions #11\n"
        "```\nTraceback ... see #12\n```\n"
        "pip has the same issue in https://github.com/pypa/pip/issues/13"
    )
    assert d.find_refs(d.clean_comment(body), UV) == []


def test_self_reference_and_html_entities_are_ignored():
    assert d.extract_original(5, thread("duplicate of #5, &#39;quoted&#39;"), UV) is None


def test_drop_duplicates_is_not_a_keyword():
    assert d.extract_original(5, thread("df.drop_duplicates() differs from #6 and #7"), UV) is None


# --------------------------------------------------------------------------- pipeline


def test_is_candidate():
    issue = {"state_reason": "completed", "labels": ["Duplicate Report"]}
    assert d.is_candidate(issue, {"Duplicate Report"})
    assert not d.is_candidate(dict(issue, labels=["duplicated"]), {"Duplicate Report"})
    assert d.is_candidate({"state_reason": "duplicate", "labels": []}, {"x"})


def test_threads_query_aliases_each_issue():
    query = d.threads_query([3, 17])
    assert "i3: issue(number: 3) { ...Thread }" in query
    assert "i17: issue(number: 17) { ...Thread }" in query
    assert "fragment Thread on Issue" in query


def test_build_pairs_counts_sources_and_flags_non_issues():
    issues = {1: {}, 10: {}, 20: {}}
    threads = {
        10: thread("Duplicate of #1"),
        20: thread("duplicate of #99"),  # #99 is a PR or deleted issue
        30: {"number": 30, "missing": True},
        40: thread("closing as duplicate"),
    }
    pairs, stats = d.build_pairs("astral-sh/uv", issues, threads)
    assert pairs == [
        {"duplicate": 10, "original": 1, "source": "comment", "original_is_issue": True},
        {"duplicate": 20, "original": 99, "source": "comment", "original_is_issue": False},
    ]
    assert stats == {"comment": 2, "missing": 1, "unresolved": 1}


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.queries = []

    def post(self, url, json=None, timeout=None):
        self.queries.append(json["query"])
        return self.responses.pop(0)


def graphql_response(repository):
    resp = requests.Response()
    resp.status_code = 200
    resp._content = json.dumps(
        {"data": {"rateLimit": {"remaining": 1}, "repository": repository}}
    ).encode()
    return resp


def node(*bodies):
    return {
        "timelineItems": {"nodes": []},
        "comments": {"nodes": [{"author": None, "body": b, "createdAt": "2025"} for b in bodies]},
    }


def test_fetch_threads_downloads_only_new_or_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "BATCH_SIZE", 2)
    path = tmp_path / "threads.jsonl"
    candidates = {
        1: {"updated_at": "2025-01-01"},
        2: {"updated_at": "2025-01-01"},
        3: {"updated_at": "2025-01-01"},
    }
    first = FakeSession(
        [
            graphql_response({"i1": node("Duplicate of #9"), "i2": None}),
            graphql_response({"i3": node()}),
        ]
    )
    threads = d.fetch_threads(first, "o/r", candidates, path, sleep=lambda s: None)
    assert len(first.queries) == 2
    assert threads[1]["comments"][0]["body"] == "Duplicate of #9"
    assert threads[2]["missing"] is True

    # Only issue 3 changed since: one request for one issue.
    candidates[3] = {"updated_at": "2025-02-01"}
    second = FakeSession([graphql_response({"i3": node("same as #1")})])
    threads = d.fetch_threads(second, "o/r", candidates, path, sleep=lambda s: None)
    assert len(second.queries) == 1 and "i3: issue" in second.queries[0]
    assert "i1: issue" not in second.queries[0]
    assert threads[3]["comments"][0]["body"] == "same as #1"
