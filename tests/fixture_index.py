"""Builds a tiny on-disk index (metadata.db + FAISS) so SearchEngine can be
exercised end-to-end without the real 68k-chunk index or any network call.

Embeddings come from FakeEmbedder: a deterministic bag-of-words hash, so
cosine similarity tracks literal token overlap. That keeps retrieval
assertions stable without loading sentence-transformers.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import zlib
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np

EMBED_DIM = 384
CLIP_DIM = 512

# Anchored to real utcnow so fixtures stay aligned with the recency boost
# (_parse_age_days measures against utcnow) no matter when tests run.
NOW = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0)


def iso_days_ago(days: float, now: dt.datetime = NOW) -> str:
    return (now - dt.timedelta(days=days)).replace(microsecond=0).isoformat()


def message(sender: str, text: str, date_iso: str, is_from_me: bool = False) -> dict:
    return {
        "sender": sender,
        "is_from_me": is_from_me,
        "text": text,
        "date_iso": date_iso,
        "contact_key": None,
        "known": True,
        "is_best_match": False,
    }


def chunk(
    chunk_id: str,
    source: str,
    contacts: list[str],
    text: str,
    *,
    start_days_ago: float,
    end_days_ago: float | None = None,
    subject: str | None = None,
    chat_title: str | None = None,
    messages: list[dict] | None = None,
    now: dt.datetime = NOW,
) -> dict:
    date_start = iso_days_ago(start_days_ago, now)
    date_end = iso_days_ago(end_days_ago if end_days_ago is not None else start_days_ago, now)
    if messages is None and contacts:
        messages = [message(contacts[0], text, date_end)]
    return {
        "chunk_id": chunk_id,
        "source": source,
        "contact_names": contacts,
        "date_start": date_start,
        "date_end": date_end,
        "text": text,
        "messages": messages or [],
        "subject": subject,
        "chat_title": chat_title,
    }


def _hash_vector(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype="float32")
    tokens = [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if t]
    for tok in tokens:
        vec[zlib.crc32(tok.encode()) % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        vec[0] = 1.0
        norm = 1.0
    return vec / norm


class FakeEmbedder:
    """Drop-in for indexer.embedder.Embedder with no model download."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    def embed_batch(self, texts: list[str], show_progress_bar: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype="float32")
        return np.vstack([_hash_vector(t, EMBED_DIM) for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return _hash_vector(text, EMBED_DIM)


class FakeClipEmbedder:
    def embed_text_one(self, text: str) -> np.ndarray:
        return _hash_vector(text, CLIP_DIM)


class FakeLLM:
    """Stands in for AsyncOpenAI. Records calls; returns `content` or raises."""

    def __init__(self, content: str = "", error: Exception | None = None):
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)
        self._content = content
        self._error = error

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


def _init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, source TEXT, contact_names TEXT,
            date_start TEXT, date_end TEXT, text TEXT, row_ids TEXT,
            messages TEXT, subject TEXT, chat_title TEXT)"""
    )
    conn.execute("CREATE INDEX idx_chunks_source ON chunks(source)")
    conn.execute("CREATE INDEX idx_chunks_date_end ON chunks(date_end)")
    conn.execute(
        'CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text, '
        'tokenize = "unicode61 remove_diacritics 2")'
    )
    conn.execute(
        """CREATE TABLE images (
            attachment_id INTEGER PRIMARY KEY, path TEXT NOT NULL, mime TEXT,
            date_iso TEXT, sender_name TEXT, sender_known INTEGER, contact_key TEXT,
            chat_title TEXT, chat_id INTEGER, is_from_me INTEGER, msg_text TEXT)"""
    )
    conn.execute(
        """CREATE TABLE contact_personas (
            name TEXT PRIMARY KEY, style_summary TEXT, avg_msg_length REAL,
            emoji_frequency REAL, response_style TEXT, top_topics TEXT,
            first_message_iso TEXT, last_message_iso TEXT, total_messages INTEGER,
            sample_hash TEXT)"""
    )
    conn.execute(
        """CREATE TABLE contact_summaries (
            name TEXT PRIMARY KEY, first_message_iso TEXT, last_message_iso TEXT,
            total_chunks INTEGER, last_30d_chunks INTEGER, n_clusters_sampled INTEGER,
            summary TEXT, sample_hash TEXT)"""
    )
    return conn


def build_fixture(
    data_dir: Path,
    chunks: list[dict],
    *,
    images: list[dict] | None = None,
    personas: list[dict] | None = None,
    summaries: list[dict] | None = None,
) -> Path:
    """Write metadata.db + index.faiss + id_map.json into `data_dir`."""
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = _init_db(data_dir / "metadata.db")
    for c in chunks:
        conn.execute(
            "INSERT INTO chunks (chunk_id, source, contact_names, date_start, date_end, "
            "text, row_ids, messages, subject, chat_title) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                c["chunk_id"], c["source"],
                json.dumps(c["contact_names"], ensure_ascii=False),
                c["date_start"], c["date_end"], c["text"], "[]",
                json.dumps(c["messages"]), c["subject"], c["chat_title"],
            ),
        )
        conn.execute(
            "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
            (c["chunk_id"], c["text"]),
        )
    for img in images or []:
        conn.execute(
            "INSERT INTO images (attachment_id, path, mime, date_iso, sender_name, "
            "sender_known, contact_key, chat_title, chat_id, is_from_me, msg_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                img["attachment_id"], img["path"], img.get("mime", "image/jpeg"),
                img["date_iso"], img.get("sender_name"), 1, None,
                img.get("chat_title"), None, 0, img.get("msg_text", ""),
            ),
        )
    for p in personas or []:
        conn.execute(
            "INSERT INTO contact_personas (name, style_summary, avg_msg_length, "
            "emoji_frequency, response_style, top_topics, first_message_iso, "
            "last_message_iso, total_messages, sample_hash) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                p["name"], p.get("style_summary", ""), p.get("avg_msg_length", 10.0),
                p.get("emoji_frequency", 0.0), p.get("response_style", "quick"),
                json.dumps(p.get("top_topics", [])), p.get("first_message_iso", ""),
                p.get("last_message_iso", ""), p.get("total_messages", 100), "hash",
            ),
        )
    for s in summaries or []:
        conn.execute(
            "INSERT INTO contact_summaries (name, first_message_iso, last_message_iso, "
            "total_chunks, last_30d_chunks, n_clusters_sampled, summary, sample_hash) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                s["name"], s.get("first_message_iso", ""), s.get("last_message_iso", ""),
                s.get("total_chunks", 10), s.get("last_30d_chunks", 1), 1,
                s.get("summary", ""), "hash",
            ),
        )
    conn.commit()
    conn.close()

    embedder = FakeEmbedder()
    vectors = embedder.embed_batch([c["text"] for c in chunks], show_progress_bar=False)
    index = faiss.IndexFlatIP(EMBED_DIM)
    if len(vectors):
        index.add(vectors)
    faiss.write_index(index, str(data_dir / "index.faiss"))
    (data_dir / "id_map.json").write_text(json.dumps([c["chunk_id"] for c in chunks]))
    return data_dir


def build_image_index(data_dir: Path, images: list[dict]) -> None:
    """Optional CLIP-side index. Callers must also stub engine.clip_embedder."""
    vectors = np.vstack([_hash_vector(i.get("msg_text", ""), CLIP_DIM) for i in images])
    index = faiss.IndexFlatIP(CLIP_DIM)
    index.add(vectors)
    faiss.write_index(index, str(data_dir / "images.faiss"))
    (data_dir / "image_id_map.json").write_text(
        json.dumps([i["attachment_id"] for i in images])
    )
