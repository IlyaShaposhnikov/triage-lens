import json

import pytest
import requests

from triage_lens import collect as c


def make_response(status=200, payload=None, headers=None):
    resp = requests.Response()
    resp.status_code = status
    resp._content = json.dumps(payload if payload is not None else {}).encode()
    resp.headers.update(headers or {})
    return resp


def ref(number, repo="o/r"):
    return {"number": number, "url": f"https://github.com/{repo}/issues/{number}"}


def mark(canonical, duplicate, typename="MarkedAsDuplicateEvent"):
    return {"__typename": typename, "canonical": canonical, "duplicate": duplicate}


def node(number, updated="2025-01-01T00:00:00Z", events=(), **extra):
    item = {
        "number": number,
        "title": f"Issue {number}",
        "body": None,
        "url": f"https://github.com/o/r/issues/{number}",
        "state": "CLOSED",
        "stateReason": "DUPLICATE",
        "createdAt": "2025-01-01T00:00:00Z",
        "updatedAt": updated,
        "closedAt": None,
        "authorAssociation": "NONE",
        "author": {"login": "alice", "__typename": "User"},
        "labels": {"nodes": [{"name": "bug"}]},
        "comments": {"totalCount": 2},
        "timelineItems": {"nodes": list(events)},
    }
    item.update(extra)
    return item


def page(nodes, cursor=None, remaining=4999):
    return make_response(
        200,
        {
            "data": {
                "rateLimit": {"remaining": remaining},
                "repository": {
                    "issues": {
                        "pageInfo": {"hasNextPage": cursor is not None, "endCursor": cursor},
                        "nodes": nodes,
                    }
                },
            }
        },
    )


class FakeSession:
    """Serves a fixed list of responses and records the GraphQL variables sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append(json["variables"])
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def no_sleep(seconds):
    pass


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_state(path):
    return json.loads((path / "o__r.state.json").read_text())


def test_to_record_flattens_node():
    record = c.to_record(node(7))
    assert record["number"] == 7
    assert record["body"] == ""
    assert record["labels"] == ["bug"]
    assert record["state"] == "closed"
    assert record["state_reason"] == "duplicate"
    assert record["author"] == "alice"
    assert record["author_type"] == "User"
    assert record["comments"] == 2
    assert record["duplicate_of"] == []


def test_to_record_handles_deleted_author():
    assert c.to_record(node(8, author=None))["author"] is None


def test_duplicate_targets_replays_marks_and_unmarks():
    events = [
        mark(ref(1), ref(10)),
        mark(ref(2), ref(10)),
        mark(ref(1), ref(10), "UnmarkedAsDuplicateEvent"),
        mark(ref(10), ref(99)),  # this issue is the canonical one: ignored
        mark(None, ref(10)),  # canonical not accessible: ignored
    ]
    assert c.duplicate_targets(10, events) == [ref(2)]


def test_rate_limit_delay():
    assert c.rate_limit_delay(make_response(200, {"data": {}})) is None
    assert c.rate_limit_delay(make_response(403)) is None  # real permission error
    assert c.rate_limit_delay(make_response(429, headers={"Retry-After": "30"})) == 30
    primary = make_response(
        403, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1100"}
    )
    assert c.rate_limit_delay(primary, now=1000) == 101
    in_body = make_response(
        200, {"errors": [{"type": "RATE_LIMITED"}]}, {"X-RateLimit-Reset": "1010"}
    )
    assert c.rate_limit_delay(in_body, now=1000) == 11


def test_post_graphql_waits_out_errors():
    sleeps = []
    session = FakeSession(
        [
            requests.ConnectionError(),
            make_response(502),
            make_response(429, headers={"Retry-After": "5"}),
            page([node(1)]),
        ]
    )
    data = c.post_graphql(session, {}, sleep=sleeps.append)
    assert data["repository"]["issues"]["nodes"][0]["number"] == 1
    assert sleeps == [1.0, 2.0, 5.0]


def test_post_graphql_raises_on_permission_and_query_errors():
    with pytest.raises(requests.HTTPError):
        c.post_graphql(FakeSession([make_response(403)]), {}, sleep=no_sleep)
    bad_query = make_response(200, {"errors": [{"message": "Field 'x' doesn't exist"}]})
    with pytest.raises(c.GraphQLError, match="doesn't exist"):
        c.post_graphql(FakeSession([bad_query]), {}, sleep=no_sleep)


def test_post_graphql_accepts_partial_errors():
    resp = page([node(1)])
    body = json.loads(resp._content)
    body["errors"] = [{"message": "Could not resolve to a node"}]
    resp._content = json.dumps(body).encode()
    data = c.post_graphql(FakeSession([resp]), {}, sleep=no_sleep)
    assert data["repository"]["issues"]["nodes"]


def test_collect_follows_cursor_and_sets_refresh_point(tmp_path):
    session = FakeSession(
        [
            page([node(1, "2025-01-01T00:00:00Z"), node(2, "2025-01-02T00:00:00Z")], "c1"),
            page([node(3, "2025-01-03T12:00:00Z", events=[mark(ref(1), ref(3))])]),
        ]
    )
    added = c.collect(session, "o/r", tmp_path, sleep=no_sleep)

    records = read_jsonl(tmp_path / "o__r.jsonl")
    assert added == 3
    assert [r["number"] for r in records] == [1, 2, 3]
    assert records[-1]["duplicate_of"] == [ref(1)]
    assert [(call["after"], call["since"]) for call in session.calls] == [
        (None, None),
        ("c1", None),
    ]
    state = read_state(tmp_path)
    assert state["cursor"] is None
    assert state["since"] == "2025-01-03T11:55:00Z"  # newest update minus overlap


def test_collect_resumes_from_saved_cursor(tmp_path):
    c.collect(FakeSession([page([node(1)], "c1")]), "o/r", tmp_path, max_pages=1, sleep=no_sleep)
    assert read_state(tmp_path)["cursor"] == "c1"

    second = FakeSession([page([node(2)])])
    c.collect(second, "o/r", tmp_path, sleep=no_sleep)
    assert second.calls[0]["after"] == "c1"
    assert [r["number"] for r in read_jsonl(tmp_path / "o__r.jsonl")] == [1, 2]


def test_refresh_appends_changed_issues_and_skips_repeats(tmp_path):
    c.collect(
        FakeSession([page([node(1, "2025-01-01T00:00:00Z")])]), "o/r", tmp_path, sleep=no_sleep
    )

    # Issue 1 comes back unchanged (overlap), issue 2 was relabelled since.
    refresh = FakeSession(
        [page([node(1, "2025-01-01T00:00:00Z"), node(2, "2025-02-01T00:00:00Z")])]
    )
    added = c.collect(refresh, "o/r", tmp_path, sleep=no_sleep)

    assert refresh.calls[0]["since"] == "2024-12-31T23:55:00Z"
    assert added == 1
    assert [r["number"] for r in read_jsonl(tmp_path / "o__r.jsonl")] == [1, 2]


def test_collect_ignores_state_from_rest_version(tmp_path):
    (tmp_path / "o__r.state.json").write_text('{"since": "2025-01-01T00:00:00Z", "page": 3}')
    session = FakeSession([page([node(1)])])
    c.collect(session, "o/r", tmp_path, sleep=no_sleep)
    assert (session.calls[0]["after"], session.calls[0]["since"]) == (None, None)


def test_collect_reports_missing_repository(tmp_path):
    missing = make_response(200, {"data": {"repository": None}})
    with pytest.raises(c.GraphQLError, match="not found"):
        c.collect(FakeSession([missing]), "o/r", tmp_path, sleep=no_sleep)
