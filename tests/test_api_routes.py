"""HTTP-level tests for the FastAPI app.

The engine global is stubbed directly — TestClient is used without its
context manager so the real lifespan (which tries to load the user's index)
never runs.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import main
from api.models import SearchResponse, SourceResult


class StubEngine:
    def __init__(self, response: SearchResponse | None = None, image_path: str | None = None):
        self.response = response or SearchResponse(answer="", sources=[], query_ms=1)
        self.image_path = image_path
        self.requests: list = []

    async def search(self, req):
        self.requests.append(req)
        return self.response

    def get_image_path(self, att_id: int):
        return self.image_path


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def clean_engine(monkeypatch):
    monkeypatch.setattr(main, "engine", None)
    monkeypatch.setattr(main, "engine_error", None)


class TestHealth:
    def test_reports_reason_when_degraded(self, client, monkeypatch):
        monkeypatch.setattr(main, "engine_error", "index missing")
        body = client.get("/health").json()
        assert body == {"ok": False, "error": "index missing"}

    def test_default_reason(self, client):
        assert client.get("/health").json()["error"] == "Search engine not initialized"

    def test_ok_when_engine_loaded(self, client, monkeypatch):
        monkeypatch.setattr(main, "engine", StubEngine())
        assert client.get("/health").json() == {"ok": True}


class TestSearchRoute:
    def test_503_without_engine(self, client):
        r = client.post("/search", json={"query": "hello"})
        assert r.status_code == 503

    def test_returns_engine_response(self, client, monkeypatch):
        stub = StubEngine(SearchResponse(
            answer="an answer",
            sources=[SourceResult(source="notes", contact_names=[], score=1.0,
                                  snippet="a note", date_start="2026-01-01T00:00:00",
                                  date_end="2026-01-01T00:00:00")],
            query_ms=7,
        ))
        monkeypatch.setattr(main, "engine", stub)
        body = client.post("/search", json={"query": "hello", "top_k": 3}).json()
        assert body["answer"] == "an answer"
        assert body["sources"][0]["snippet"] == "a note"
        assert body["query_ms"] == 7
        assert stub.requests[0].top_k == 3

    def test_intent_is_forwarded(self, client, monkeypatch):
        stub = StubEngine()
        monkeypatch.setattr(main, "engine", stub)
        client.post("/search", json={
            "query": "photos from sam",
            "intent": {"topic": "photos", "contacts": ["sam"],
                       "must_have_attachment": True, "query_type": "standard"},
        })
        intent = stub.requests[0].intent
        assert intent.contacts == ["sam"]
        assert intent.must_have_attachment is True

    def test_rejects_empty_query(self, client, monkeypatch):
        monkeypatch.setattr(main, "engine", StubEngine())
        assert client.post("/search", json={"query": ""}).status_code == 422

    def test_rejects_out_of_range_top_k(self, client, monkeypatch):
        monkeypatch.setattr(main, "engine", StubEngine())
        assert client.post("/search", json={"query": "hi", "top_k": 99}).status_code == 422


class TestAttachmentRoute:
    def test_browser_requests_rejected(self, client, monkeypatch, tmp_path):
        img = tmp_path / "a.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 8)
        monkeypatch.setattr(main, "engine", StubEngine(image_path=str(img)))
        r = client.get("/attachment/1", headers={"Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 403

    def test_503_without_engine(self, client):
        assert client.get("/attachment/1").status_code == 503

    def test_404_for_unknown_id(self, client, monkeypatch):
        monkeypatch.setattr(main, "engine", StubEngine(image_path=None))
        assert client.get("/attachment/1").status_code == 404

    def test_404_when_file_missing_on_disk(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "engine", StubEngine(image_path=str(tmp_path / "gone.jpg")))
        assert client.get("/attachment/1").status_code == 404

    def test_serves_jpeg_with_cache_header(self, client, monkeypatch, tmp_path):
        img = tmp_path / "a.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0jpegbytes")
        monkeypatch.setattr(main, "engine", StubEngine(image_path=str(img)))
        r = client.get("/attachment/1")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert "max-age=86400" in r.headers["cache-control"]
        assert r.content == b"\xff\xd8\xff\xe0jpegbytes"

    def test_unknown_suffix_falls_back_to_octet_stream(self, client, monkeypatch, tmp_path):
        blob = tmp_path / "a.bin"
        blob.write_bytes(b"whatever")
        monkeypatch.setattr(main, "engine", StubEngine(image_path=str(blob)))
        r = client.get("/attachment/1")
        assert r.headers["content-type"] == "application/octet-stream"


class TestContactPhotoRoute:
    key = "a" * 16

    def test_browser_requests_rejected(self, client):
        r = client.get(f"/contact-photo/{self.key}", headers={"Sec-Fetch-Site": "same-origin"})
        assert r.status_code == 403

    def test_invalid_key_rejected(self, client):
        assert client.get("/contact-photo/../../etc/passwd").status_code in (400, 404)
        assert client.get("/contact-photo/NOTHEX").status_code == 400

    def test_missing_photo_404(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "CONTACTS_DIR", tmp_path)
        assert client.get(f"/contact-photo/{self.key}").status_code == 404

    def test_png_detected_from_magic_bytes(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "CONTACTS_DIR", tmp_path)
        (tmp_path / f"{self.key}.bin").write_bytes(b"\x89PNG\r\n\x1a\n" + b"rest")
        r = client.get(f"/contact-photo/{self.key}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"

    def test_jpeg_default(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "CONTACTS_DIR", tmp_path)
        (tmp_path / f"{self.key}.bin").write_bytes(b"\xff\xd8\xff\xe1jpeg")
        r = client.get(f"/contact-photo/{self.key}")
        assert r.headers["content-type"] == "image/jpeg"
