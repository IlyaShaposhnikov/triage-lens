import json
import sys
import types
from datetime import UTC, datetime

import numpy as np
import pytest
from test_app_core import bag, fake_encode, make_assets
from test_collect import FakeSession, node, page

from triage_lens import refresh as r
from triage_lens.app_core import DemoAssets
from triage_lens.demo_data import issue_row

REPO = "astral-sh/uv"
NOW = datetime(2025, 2, 1, tzinfo=UTC)


class CountingEncoder:
    def __init__(self):
        self.texts = []

    def __call__(self, texts):
        self.texts += texts
        return fake_encode(texts)


def fetched(number, title, updated="2025-01-20T00:00:00Z", created="2025-01-01T00:00:00Z", **extra):
    return node(number, updated, title=title, createdAt=created, **extra)


def test_fetch_updated_follows_cursor_and_passes_since():
    session = FakeSession([page([node(1)], cursor="c1"), page([node(2)])])
    records = r.fetch_updated(session, REPO, "2025-01-01T00:00:00Z", sleep=lambda s: None)
    assert [x["number"] for x in records] == [1, 2]
    assert session.calls[0]["since"] == "2025-01-01T00:00:00Z"
    assert session.calls[0]["after"] is None and session.calls[1]["after"] == "c1"
    assert session.calls[0]["owner"] == "astral-sh" and session.calls[0]["name"] == "uv"


def test_merge_updates_in_place_and_encodes_only_changed_text():
    old = {
        "number": 1,
        "title": "crash on lock",
        "body": "",
        "labels": [],
        "state": "open",
        "created_at": "2025-01-01T00:00:00Z",
        "url": "u/1",
    }
    rows = [
        issue_row(old, "crash on lock"),
        issue_row({**old, "number": 2, "title": "add option"}, "add option"),
    ]
    vectors = np.stack([bag("crash on lock"), bag("add option")]).astype(np.float16)
    updates = [
        {**old, "state": "closed", "labels": ["bug"]},  # same text: metadata only
        {**old, "number": 2, "title": "typo docs"},  # edited text
        {**old, "number": 3, "title": "how do", "created_at": "2025-01-05T00:00:00Z"},  # new
    ]
    encoder = CountingEncoder()
    new_rows, new_vectors, counts = r.merge(REPO, rows, vectors, updates, encoder)

    assert [row["number"] for row in new_rows] == [1, 2, 3]
    assert new_rows[0]["state"] == "closed" and new_rows[0]["labels"] == ["bug"]
    assert encoder.texts == ["typo docs", "how do"]
    assert counts == {"new": 1, "changed": 2, "re_encoded": 1}
    assert new_vectors.shape == (3, len(bag("x")))
    np.testing.assert_allclose(new_vectors[0], vectors[0].astype(np.float32))
    np.testing.assert_allclose(new_vectors[1], bag("typo docs"), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(new_vectors, axis=1), 1.0, atol=1e-3)


def test_merge_reencodes_rows_without_hash():
    row = {"number": 1, "title": "crash", "snippet": ""}  # published before text_sha1 existed
    issue = {
        "number": 1,
        "title": "crash",
        "body": "",
        "labels": [],
        "state": "open",
        "created_at": "2025-01-01T00:00:00Z",
        "url": "u/1",
    }
    encoder = CountingEncoder()
    _, _, counts = r.merge(REPO, [row], np.zeros((1, 5), np.float16), [issue], encoder)
    assert encoder.texts == ["crash"] and counts["re_encoded"] == 1


def test_refresh_updates_files_and_manifest(tmp_path):
    root = make_assets(tmp_path)
    session = FakeSession(
        [
            page(
                [
                    fetched(1, "crash on lock", stateReason="COMPLETED"),
                    fetched(
                        4,
                        "how do I lock",
                        created="2025-01-10T00:00:00Z",
                        updated="2025-01-11T00:00:00Z",
                        state="OPEN",
                        stateReason=None,
                    ),
                ]
            )
        ]
    )
    summary = r.refresh(root, session, fake_encode, now=NOW)

    assert summary == {REPO: {"new": 1, "changed": 1, "re_encoded": 1}}
    # The published rows had no hash yet, so #1 is encoded again once.
    assert session.calls[0]["since"] == "2025-01-02T00:00:00Z"  # newest_issue minus a day

    manifest = json.loads((root / "manifest.json").read_text())
    info = manifest["repos"][REPO]
    assert info["issues"] == 4
    assert info["newest_issue"] == "2025-01-10T00:00:00Z"
    assert info["updated_through"] == "2025-01-20T00:00:00Z"
    assert info["refreshed_at"] == manifest["refreshed_at"] == "2025-02-01T00:00:00Z"
    assert "| 4 | 2025-01-10 |" in (root / "README.md").read_text()  # dataset card

    assets = DemoAssets(root)
    issues = assets.repos[REPO].issues
    assert issues[0]["state_reason"] == "completed" and issues[-1]["number"] == 4
    assert assets.similar(REPO, "how do I lock", "", fake_encode, k=1)[0]["number"] == 4


def test_refresh_uses_updated_through_watermark(tmp_path):
    root = make_assets(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["repos"][REPO]["updated_through"] = "2025-01-30T12:00:00Z"
    (root / "manifest.json").write_text(json.dumps(manifest))
    session = FakeSession([page([])])
    summary = r.refresh(root, session, fake_encode, now=NOW)
    assert session.calls[0]["since"] == "2025-01-29T12:00:00Z"
    assert summary[REPO] == {"new": 0, "changed": 0, "re_encoded": 0}
    info = json.loads((root / "manifest.json").read_text())["repos"][REPO]
    assert info["updated_through"] == "2025-01-30T12:00:00Z"  # never moves backwards


def test_refresh_fails_on_missing_repository(tmp_path):
    root = make_assets(tmp_path)
    session = FakeSession([page([])])
    session.responses[0]._content = json.dumps({"data": {"repository": None}}).encode()
    with pytest.raises(RuntimeError, match="not found"):
        r.refresh(root, session, fake_encode, now=NOW)


def test_main_dry_run_does_not_push(tmp_path, monkeypatch):
    published = make_assets(tmp_path / "hub")
    for path in published.rglob("*"):
        if path.is_file():
            path.chmod(0o444)  # the Hugging Face cache on Linux runners is read-only
    uploads = []
    hub = types.SimpleNamespace(
        snapshot_download=lambda repo_id, repo_type: str(published),
        HfApi=lambda: types.SimpleNamespace(upload_folder=lambda **kw: uploads.append(kw)),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(r, "make_session", lambda token: FakeSession([page([]), page([])]))
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    work = tmp_path / "work"

    r.main(["--repo-id", "me/demo", "--dir", str(work)], encoder_factory=lambda m, b: fake_encode)
    assert uploads == [] and (work / "manifest.json").exists()
    assert all(p.stat().st_mode & 0o200 for p in work.rglob("*") if p.is_file())
    assert "refreshed_at" not in json.loads((published / "manifest.json").read_text())

    r.main(
        ["--repo-id", "me/demo", "--dir", str(work), "--push"],
        encoder_factory=lambda m, b: fake_encode,
    )
    assert len(uploads) == 1 and uploads[0]["repo_id"] == "me/demo"
    assert "+0 new" in uploads[0]["commit_message"]


def test_main_requires_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        r.main(["--repo-id", "me/demo"])
