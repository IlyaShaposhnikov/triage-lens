import json

import pytest
import requests

from triage_lens import collect as c


def make_response(status=200, payload=None, headers=None):
    resp = requests.Response()
    resp.status_code = status
    resp._content = json.dumps(payload if payload is not None else []).encode()
    resp.headers.update(headers or {})
    return resp


def issue(number, **extra):
    item = {
        "number": number,
        "title": f"Issue {number}",
        "body": None,
        "labels": [{"name": "bug"}],
        "state": "open",
        "state_reason": None,
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-02T00:00:00Z",
        "closed_at": None,
        "user": {"login": "alice", "type": "User"},
        "author_association": "NONE",
        "comments": 0,
        "html_url": f"https://github.com/o/r/issues/{number}",
    }
    item.update(extra)
    return item


class FakeSession:
    """Serves a fixed list of responses and records the requested params."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def ts(minute):
    return f"2025-01-01T00:{minute:02d}:00Z"


def full_page(first_number, updated):
    """A full page of PER_PAGE issues, all updated at ``updated``."""
    return [issue(first_number + i, updated_at=updated) for i in range(c.PER_PAGE)]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_state(path):
    return json.loads((path / "o__r.state.json").read_text())


def test_to_record_keeps_fields_and_normalises_empty_body():
    record = c.to_record(issue(7))
    assert record["number"] == 7
    assert record["body"] == ""
    assert record["labels"] == ["bug"]
    assert record["author"] == "alice"
    assert record["author_type"] == "User"


def test_to_record_handles_missing_user():
    record = c.to_record(issue(8, user=None))
    assert record["author"] is None


def test_rate_limit_delay():
    assert c.rate_limit_delay(make_response(200)) is None
    assert c.rate_limit_delay(make_response(403)) is None  # real permission error
    assert c.rate_limit_delay(make_response(429, headers={"Retry-After": "30"})) == 30
    primary = make_response(
        403, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1100"}
    )
    assert c.rate_limit_delay(primary, now=1000) == 101


def test_get_with_retry_waits_out_errors(monkeypatch):
    sleeps = []
    session = FakeSession(
        [
            requests.ConnectionError(),
            make_response(502),
            make_response(429, headers={"Retry-After": "5"}),
            make_response(200, [issue(1)]),
        ]
    )
    resp = c.get_with_retry(session, "url", {}, sleep=sleeps.append)
    assert resp.status_code == 200
    assert sleeps == [1.0, 2.0, 5.0]


def test_get_with_retry_raises_on_permission_error():
    session = FakeSession([make_response(403)])
    with pytest.raises(requests.HTTPError):
        c.get_with_retry(session, "url", {}, sleep=lambda s: None)


def test_next_cursor_moves_to_newest_update():
    assert c.next_cursor(None, 1, [issue(1, updated_at=ts(1)), issue(2, updated_at=ts(2))]) == (
        ts(2),
        1,
    )


def test_next_cursor_advances_page_when_timestamp_is_stuck():
    assert c.next_cursor(ts(5), 1, full_page(1, ts(5))) == (ts(5), 2)


def test_collect_walks_by_cursor_and_skips_pull_requests(tmp_path):
    page_one = full_page(1, ts(1))
    page_one[-1] = issue(100, updated_at=ts(2), pull_request={})
    session = FakeSession(
        [
            make_response(200, page_one),
            # The cursor is inclusive, so the boundary update comes back again.
            make_response(
                200, [issue(100, updated_at=ts(2), pull_request={}), issue(101, updated_at=ts(3))]
            ),
        ]
    )
    added = c.collect(session, "o/r", tmp_path, sleep=lambda s: None)

    numbers = [r["number"] for r in read_jsonl(tmp_path / "o__r.jsonl")]
    assert added == 100
    assert 100 not in numbers and numbers[-1] == 101
    assert "since" not in session.calls[0]
    assert session.calls[1]["since"] == ts(2) and session.calls[1]["page"] == 1
    assert read_state(tmp_path) == {"since": ts(3), "page": 1}


def test_collect_handles_a_full_page_with_one_timestamp(tmp_path):
    session = FakeSession(
        [
            make_response(200, full_page(1, ts(1))),
            make_response(200, full_page(101, ts(1))),
            make_response(200, [issue(201, updated_at=ts(2))]),
        ]
    )
    c.collect(session, "o/r", tmp_path, sleep=lambda s: None)

    assert [(call.get("since"), call["page"]) for call in session.calls] == [
        (None, 1),
        (ts(1), 1),
        (ts(1), 2),
    ]
    assert len(read_jsonl(tmp_path / "o__r.jsonl")) == 201


def test_collect_resumes_and_appends_updated_versions(tmp_path):
    first = FakeSession([make_response(200, full_page(1, ts(1)))])
    c.collect(first, "o/r", tmp_path, max_pages=1, sleep=lambda s: None)
    assert read_state(tmp_path) == {"since": ts(1), "page": 1}

    # Issue 1 was relabelled after the first run: its new version is appended,
    # while an unchanged repeat of issue 2 is skipped.
    second = FakeSession(
        [make_response(200, [issue(2, updated_at=ts(1)), issue(1, updated_at=ts(4))])]
    )
    added = c.collect(second, "o/r", tmp_path, sleep=lambda s: None)

    assert second.calls[0]["since"] == ts(1)
    assert added == 1
    records = read_jsonl(tmp_path / "o__r.jsonl")
    assert len(records) == 101
    assert (records[-1]["number"], records[-1]["updated_at"]) == (1, ts(4))


def test_collect_ignores_state_from_page_based_version(tmp_path):
    (tmp_path / "o__r.state.json").write_text('{"next_page": 100}')
    session = FakeSession([make_response(200, [issue(1)])])
    c.collect(session, "o/r", tmp_path, sleep=lambda s: None)
    assert session.calls[0] == {
        "state": "all",
        "sort": "updated",
        "direction": "asc",
        "per_page": c.PER_PAGE,
        "page": 1,
    }
