"""Search pipeline: hybrid (FAISS + FTS5) retrieval → strict-prompt synthesis."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from openai import AsyncOpenAI

from . import hyperspell, temporal
from .models import ChunkMessage, QueryIntent, SearchRequest, SearchResponse, SourceResult
from .prompts import COMPARE_FOCUSED_PROMPT, CONTACT_FOCUSED_PROMPT, SYSTEM_PROMPT, TEMPORAL_LABEL_SYSTEM_PROMPT
from .query_analysis import (  # noqa: F401 -- re-exported for tests/other modules
    RECENCY_BOOST,
    RECENCY_HALFLIFE_DAYS,
    TEMPORAL_MIN_RECENT_CHUNKS,
    TEMPORAL_RECENT_DAYS,
    TEMPORAL_RECENT_FLOOR,
    _ANSWER_STRIP_CHARS,
    _QUERY_TOKEN_RE,
    _adaptive_recent_split,
    _best_message_index,
    _build_fts_match,
    _classify_query_type,
    _display_name,
    _filter_for_target_contacts,
    _infer_sources_from_query,
    _label,
    _looks_like_non_answer,
    _looks_like_question,
    _normalize_name,
    _parse_age_days,
    _persona_topics,
    _query_tokens,
    _recency_multiplier,
    _strip_invented_quote_marks,
    _strip_query_tokens,
    _validate_no_invented_quotes,
)

INDEXER_DIR = Path(__file__).resolve().parent.parent / "indexer"
sys.path.insert(0, str(INDEXER_DIR))
from embedder import Embedder  # noqa: E402
from persona_builder import escape_like, filter_topic_labels  # noqa: E402

DATA_DIR = INDEXER_DIR / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
META_DB_PATH = DATA_DIR / "metadata.db"
ID_MAP_PATH = DATA_DIR / "id_map.json"
IMAGE_INDEX_PATH = DATA_DIR / "images.faiss"
IMAGE_ID_MAP_PATH = DATA_DIR / "image_id_map.json"

# Local OpenAI-compatible server (Ollama) by default — same pattern as
# indexer/persona_builder.py.
LLM_BASE_URL = os.getenv("SEMSE_LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("SEMSE_LLM_MODEL", "qwen2.5:14b")
# How long Ollama keeps the model resident after the last request (the 14b
# model holds ~9 GB while loaded). Passed per-request; ignored by non-Ollama
# servers. Lower it (e.g. "2m") on memory-pressured machines.
LLM_KEEP_ALIVE = os.getenv("SEMSE_LLM_KEEP_ALIVE", "5m")
_LLM_EXTRA_BODY = {"keep_alive": LLM_KEEP_ALIVE}

RRF_K = 60  # standard reciprocal rank fusion constant
FUSE_FETCH = 24  # how many candidates each retriever pulls before fusion

TEMPORAL_SIDE_SAMPLE = 300   # workload cap per side (embedding + clustering)
TEMPORAL_FETCH_LIMIT = 1200  # newest chunks pulled per contact before splitting

# Tokens we never accept as a contact candidate in the fallback scan, even if
# they happen to appear in the contact index. Keeps the heuristic from
# attaching meaning to verbs/articles/etc that share spelling with nicknames.
_CONTACT_QUERY_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "of", "in", "on", "at", "to",
    "for", "with", "and", "or", "but", "if", "then", "this", "that", "it",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "she", "they",
    "what", "who", "when", "where", "why", "how", "any", "some", "all",
    "about", "from", "by", "as", "tell", "show", "find", "get", "got",
    "say", "said", "ask", "asked", "think", "thought", "like", "likes",
    "liked", "prefer", "prefers", "preferred", "want", "wants", "wanted",
    "claude", "chatgpt", "gpt", "openai", "anthropic", "lockheed", "amazon",
    "google", "microsoft", "stanford", "harvard", "mit", "berkeley",
    "mail", "email", "text", "texts", "message", "messages", "picture",
    "photo", "image", "pic",
}


class SearchEngine:
    def __init__(self) -> None:
        if not INDEX_PATH.exists() or not META_DB_PATH.exists():
            raise FileNotFoundError(
                f"Index or metadata not found. Run `python indexer/build_index.py` first.\n"
                f"  expected: {INDEX_PATH}\n  expected: {META_DB_PATH}"
            )
        self.index = faiss.read_index(str(INDEX_PATH))
        self.id_map: list[str] = json.loads(ID_MAP_PATH.read_text())
        self.embedder = Embedder()
        self._db_path = str(META_DB_PATH)
        self._openai = AsyncOpenAI(
            base_url=LLM_BASE_URL, api_key=os.getenv("OPENAI_API_KEY") or "ollama"
        )
        # Image index is optional — if it doesn't exist, image search is skipped.
        self.image_index = None
        self.image_id_map: list[int] = []
        self.clip_embedder = None
        if IMAGE_INDEX_PATH.exists() and IMAGE_ID_MAP_PATH.exists():
            self.image_index = faiss.read_index(str(IMAGE_INDEX_PATH))
            self.image_id_map = json.loads(IMAGE_ID_MAP_PATH.read_text())
        self._contact_summaries: dict[str, dict] = self._load_contact_summaries()
        self._contact_norm_index: dict[str, set[str]] = self._build_contact_norm_index()

    def _build_contact_norm_index(self) -> dict[str, set[str]]:
        """Map normalized name token → set of canonical contact names that
        contain that token. Used to fuzzy-resolve an LLM-extracted contact
        ("jerry") to canonical names from the index ("Jerry Yan").
        """
        index: dict[str, set[str]] = {}
        with self._open_db() as conn:
            try:
                rows = conn.execute("SELECT DISTINCT contact_names FROM chunks").fetchall()
            except sqlite3.OperationalError:
                return index
        seen_canonical: set[str] = set()
        for r in rows:
            try:
                names = json.loads(r["contact_names"])
            except (TypeError, ValueError):
                continue
            for canonical in names:
                if not isinstance(canonical, str) or canonical in seen_canonical:
                    continue
                seen_canonical.add(canonical)
                # Index by each whitespace-separated token AND by the full
                # normalized name so multi-word LLM extractions still match.
                full_norm = _normalize_name(canonical)
                if full_norm:
                    index.setdefault(full_norm, set()).add(canonical)
                for tok in full_norm.split():
                    if len(tok) >= 2:
                        index.setdefault(tok, set()).add(canonical)
        return index

    def _extract_contacts_from_query(self, query: str) -> set[str]:
        """Heuristic fallback when the LLM misses a contact name that's in
        the query verbatim. We scan tokens (length >= 2) against the
        normalized contact index. Stopwords are skipped so common words
        don't accidentally resolve to a contact named e.g. "and"."""
        if not query or not self._contact_norm_index:
            return set()
        tokens = _QUERY_TOKEN_RE.findall(query.lower())
        # Filter out generic English stopwords + verbs that often appear
        # alongside contact names. (Don't filter on length alone — short
        # nicknames like "M" or "IL" exist in the user's index.)
        tokens = [t for t in tokens if t not in _CONTACT_QUERY_STOPWORDS]
        out: set[str] = set()
        for t in tokens:
            if t in self._contact_norm_index:
                candidates = self._contact_norm_index[t]
                if len(candidates) == 1:
                    out.update(candidates)
                elif len(candidates) <= 4:
                    # Ambiguous token ("ruthvik" → two contacts): take the one
                    # the user actually messages, by chunk volume, when the
                    # gap is decisive (>3x). A common first name shared by
                    # many active contacts still gets dropped.
                    ranked = sorted(
                        candidates,
                        key=lambda n: self._contact_summaries.get(n, {}).get(
                            "total_chunks", 0
                        ),
                        reverse=True,
                    )
                    top = self._contact_summaries.get(ranked[0], {}).get("total_chunks", 0)
                    second = self._contact_summaries.get(ranked[1], {}).get("total_chunks", 0)
                    if top > 0 and top >= 3 * max(second, 1):
                        out.add(ranked[0])
        return out

    def _contact_name_tokens(self, contact_filter: set[str]) -> set[str]:
        """Name tokens of the filtered contacts that are DISTINCTIVE — i.e.
        map to few contacts in the index. Shared surname-like tokens
        ("robotics" in dozens of contact names) stay out so a query about
        the robotics TOPIC isn't gutted when a "X Robotics" contact matches."""
        toks: set[str] = set()
        for name in contact_filter:
            for tok in _normalize_name(name).split():
                if len(tok) < 2:
                    continue
                candidates = self._contact_norm_index.get(tok, set())
                if candidates and len(candidates) <= 4:
                    toks.add(tok)
        return toks

    def _resolve_contacts(self, extracted: list[str]) -> set[str]:
        """Given LLM-extracted contact strings, return the set of canonical
        contact names from the index they could refer to. Multi-token strings
        require all tokens to point to the same canonical name (intersection)."""
        resolved: set[str] = set()
        for raw in extracted:
            norm = _normalize_name(raw)
            if not norm:
                continue
            # Exact normalized hit first
            if norm in self._contact_norm_index:
                resolved.update(self._contact_norm_index[norm])
                continue
            tokens = [t for t in norm.split() if len(t) >= 2]
            if not tokens:
                continue
            # Intersect per-token candidate sets so "jerry yan" prefers
            # contacts that match BOTH tokens; if intersection is empty,
            # fall back to the union (best-effort soft match).
            per_token = [self._contact_norm_index.get(t, set()) for t in tokens]
            intersect = set.intersection(*per_token) if per_token else set()
            if intersect:
                resolved.update(intersect)
            else:
                for s in per_token:
                    resolved.update(s)
        return resolved

    def _load_contact_summaries(self) -> dict[str, dict]:
        with self._open_db() as conn:
            try:
                rows = conn.execute(
                    "SELECT name, summary, total_chunks, last_30d_chunks, "
                    "first_message_iso, last_message_iso FROM contact_summaries"
                ).fetchall()
            except sqlite3.OperationalError:
                return {}
        out: dict[str, dict] = {}
        for r in rows:
            out[r["name"]] = {
                "summary": r["summary"],
                "total_chunks": r["total_chunks"],
                "last_30d_chunks": r["last_30d_chunks"],
                "first": (r["first_message_iso"] or "")[:10],
                "last": (r["last_message_iso"] or "")[:10],
            }
        return out

    def _load_persona(self, name: str) -> dict | None:
        """Load a contact's persona row from contact_personas, or None if missing."""
        with self._open_db() as conn:
            try:
                row = conn.execute(
                    "SELECT * FROM contact_personas WHERE name=?", (name,)
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        return dict(row) if row else None

    def _recent_chunks_for_contacts(
        self,
        contact_filter: set[str],
        limit: int,
        before_iso: str | None = None,
    ) -> list[SourceResult]:
        """Fetch the most recent chunks where any target contact appears.
        Optionally filtered to chunks before `before_iso`."""
        if not contact_filter:
            return []
        # Same escaped raw-unicode pattern as the indexer (contact_names is
        # stored with ensure_ascii=False, so match the literal name).
        like_args = [f'%"{escape_like(n)}"%' for n in contact_filter]
        conditions = " OR ".join("contact_names LIKE ? ESCAPE '\\'" for _ in like_args)
        params: list = list(like_args)
        extra = ""
        if before_iso:
            extra = " AND date_end < ?"
            params.append(before_iso)
        with self._open_db() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE ({conditions}){extra} "
                f"ORDER BY date_end DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        out: list[SourceResult] = []
        for r in rows:
            messages = [ChunkMessage(**m) for m in json.loads(r["messages"] or "[]")]
            out.append(
                SourceResult(
                    source=r["source"],
                    contact_names=json.loads(r["contact_names"]),
                    date_start=r["date_start"] or "",
                    date_end=r["date_end"] or "",
                    score=0.5,
                    messages=messages,
                    subject=r["subject"],
                    chat_title=r["chat_title"],
                    snippet=(r["text"] or "")[:280] if not messages else "",
                )
            )
        return out

    async def _synthesize_style(
        self,
        contact_filter: set[str],
        query: str,
        top_k: int,
    ) -> tuple[str, list[SourceResult]] | None:
        """Return (answer, sources) for a style query, or None to fall through."""
        personas = [self._load_persona(n) for n in contact_filter]
        personas = [p for p in personas if p]
        if not personas:
            return None

        parts: list[str] = []
        for p in personas:
            topics = _persona_topics(p)
            top4 = " · ".join(t["topic"] for t in topics[:4])
            style_sum = p.get("style_summary") or f"{p['name']} communicates concisely."
            if top4:
                parts.append(f"{style_sum}\n\nThey mainly talk about: {top4}.")
            else:
                parts.append(style_sum)
        answer = "\n\n".join(parts)

        # Fetch 3 recent chunks as supporting sources.
        sources = await asyncio.to_thread(
            self._recent_chunks_for_contacts, contact_filter, min(3, top_k)
        )
        return answer, sources

    async def _synthesize_affinity(
        self,
        contact_filter: set[str],
        query: str,
        top_k: int,
    ) -> tuple[str, list[SourceResult]] | None:
        """Return (answer, sources) for an affinity query, or None to fall through."""
        personas = [self._load_persona(n) for n in contact_filter]
        personas = [p for p in personas if p]
        if not personas:
            return None

        all_sources: list[SourceResult] = []
        answer_parts: list[str] = []
        for p in personas:
            canonical = p["name"]
            name = _display_name(canonical)
            topics = [t for t in _persona_topics(p) if isinstance(t.get("score"), (int, float))]
            if not topics:
                continue
            topic_list = ", ".join(
                f"{t['topic']} ({round(t['score'] * 100):.0f}%)"
                for t in topics[:6]
            )
            prompt = (
                f"Rewrite this as one natural sentence about what {name} talks about most. "
                f"Start the sentence with the name \"{name}\" and always refer to them by "
                f"that name — never as \"someone\", \"they\", or \"this person\".\n"
                f"{name}'s top topics: {topic_list}\n\nOutput: just the sentence, no preamble."
            )
            fallback_answer = f"{name} mainly discusses: {topic_list}."
            try:
                resp = await self._openai.chat.completions.create(
                extra_body=_LLM_EXTRA_BODY,
                    model=LLM_MODEL,
                    temperature=0.0,
                    max_tokens=80,
                    messages=[{"role": "user", "content": prompt}],
                )
                part = (resp.choices[0].message.content or "").strip()
                # The rewrite must actually name the contact; a nameless
                # "someone talks about…" is worse than the plain topic list.
                first_name = name.split()[0].lower() if name.split() else ""
                if part and first_name and first_name in part.lower():
                    answer_parts.append(part)
                else:
                    answer_parts.append(fallback_answer)
            except Exception:
                answer_parts.append(fallback_answer)

            # One recent chunk per top topic as sources (up to 4).
            all_sources.extend(
                await asyncio.to_thread(
                    self._recent_chunks_for_contacts, {canonical}, min(4, top_k)
                )
            )

        if not answer_parts:
            return None
        return "\n\n".join(answer_parts), all_sources[:top_k]

    async def _synthesize_temporal(
        self,
        contact_filter: set[str],
        query: str,
        top_k: int,
    ) -> tuple[str, list[SourceResult]] | None:
        """Return (answer, sources) for a temporal/change query, or None to fall through.

        Pipeline (≤2 LLM calls): adaptive recent/older split → K-means per side →
        one LLM call labels every cluster with a concrete noun-phrase topic →
        one LLM call compares the two labeled topic lists across date ranges.
        """
        from sklearn.cluster import KMeans

        chunks = await asyncio.to_thread(
            self._recent_chunks_for_contacts, contact_filter, TEMPORAL_FETCH_LIMIT
        )
        recent_chunks, older_chunks = _adaptive_recent_split(chunks)
        if not recent_chunks or not older_chunks:
            return None

        recent_sample = recent_chunks[:TEMPORAL_SIDE_SAMPLE]
        # Older side: sample evenly across the whole history, not just its
        # newest slice, so long-ago topics still register.
        stride = max(1, len(older_chunks) // TEMPORAL_SIDE_SAMPLE)
        older_sample = older_chunks[::stride][:TEMPORAL_SIDE_SAMPLE]

        def _chunk_text(c: SourceResult) -> str:
            if c.messages:
                return " ".join(m.text for m in c.messages if m.text)[:400]
            return (c.snippet or "")[:400]

        def _cluster_reps(side: list[SourceResult], k: int = 4) -> list[str]:
            """Centroid-nearest representative text per K-means cluster."""
            texts = [t for t in (_chunk_text(c) for c in side) if len(t) > 10]
            if len(texts) < 8:
                return sorted(texts, key=len, reverse=True)[:k]
            vecs = self.embedder.embed_batch(
                texts, show_progress_bar=False
            ).astype("float32")
            n_clust = min(k, len(texts))
            km = KMeans(n_clusters=n_clust, n_init=4, random_state=42)
            labels = km.fit_predict(vecs)
            reps: list[str] = []
            for cid in range(n_clust):
                members = np.where(labels == cid)[0].tolist()
                if not members:
                    continue
                centroid = km.cluster_centers_[cid]
                sims = vecs[members] @ centroid / (np.linalg.norm(centroid) or 1.0)
                reps.append(texts[members[int(np.argmax(sims))]][:250])
            return reps

        recent_reps = await asyncio.to_thread(_cluster_reps, recent_sample)
        older_reps = await asyncio.to_thread(_cluster_reps, older_sample)
        if not recent_reps or not older_reps:
            return None

        # LLM call 1: label all clusters (both sides) as concrete noun phrases,
        # persona-builder style. Junk labels are filtered the same way persona
        # topics are (filter_topic_labels).
        lines = [f"R{i + 1}: {t}" for i, t in enumerate(recent_reps)]
        lines += [f"E{i + 1}: {t}" for i, t in enumerate(older_reps)]
        label_map: dict[str, str] = {}
        try:
            resp = await self._openai.chat.completions.create(
                extra_body=_LLM_EXTRA_BODY,
                model=LLM_MODEL,
                temperature=0.0,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": TEMPORAL_LABEL_SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(lines)},
                ],
            )
            for line in (resp.choices[0].message.content or "").splitlines():
                m = re.match(r"([RE]\d+)\s*:\s*(.+)", line.strip())
                if m:
                    label_map[m.group(1)] = m.group(2).strip()
        except Exception:
            label_map = {}

        def _labels(prefix: str, reps: list[str]) -> list[str]:
            # Fall back to a truncated rep text when the LLM missed a cluster.
            raw = [
                label_map.get(f"{prefix}{i + 1}") or reps[i][:60]
                for i in range(len(reps))
            ]
            kept = filter_topic_labels([{"topic": lbl} for lbl in raw])
            return [t["topic"] for t in kept]

        recent_topics = _labels("R", recent_reps)
        older_topics = _labels("E", older_reps)
        if not recent_topics or not older_topics:
            return None

        def _range(side: list[SourceResult]) -> str:
            # side is newest-first: last element = earliest date.
            start = (side[-1].date_end or "")[:10] or "?"
            end = (side[0].date_end or "")[:10] or "?"
            return f"{start} to {end}"

        names = ", ".join(_display_name(n) for n in sorted(contact_filter))
        # LLM call 2: compare the two labeled topic lists across date ranges.
        prompt = (
            f"Topics {names} talked about earlier ({_range(older_sample)}): "
            f"{', '.join(older_topics)}.\n"
            f"Topics {names} talks about recently ({_range(recent_sample)}): "
            f"{', '.join(recent_topics)}.\n\n"
            f"In ONE sentence, describe how {names}'s focus has shifted between "
            f"the two periods, naming the concrete topics. If the focus hasn't "
            f"meaningfully changed, phrase it as \"{names}'s focus has stayed "
            f"on …\" naming the stable topics — never a bare \"no meaningful "
            f"shift\". No preamble."
        )
        try:
            resp = await self._openai.chat.completions.create(
                extra_body=_LLM_EXTRA_BODY,
                model=LLM_MODEL,
                temperature=0.0,
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = (resp.choices[0].message.content or "").strip()
        except Exception:
            answer = ""
        if not answer or _looks_like_non_answer(answer):
            return None

        # Interleave 3 recent + 3 older as sources.
        sources: list[SourceResult] = []
        for i in range(max(len(recent_chunks[:3]), len(older_chunks[:3]))):
            if i < len(recent_chunks[:3]):
                sources.append(recent_chunks[i])
            if i < len(older_chunks[:3]):
                sources.append(older_chunks[i])
        return answer, sources[:top_k]

    async def _synthesize_compare(
        self,
        query: str,
        contact_filter: set[str],
        embed_query: str,
        top_k: int,
        t0: float,
    ) -> SearchResponse | None:
        """Compare mode: fetch per-contact chunks, interleave, send to LLM."""
        # Fetch per-contact chunks via the contact-similarity reranker path.
        per_contact: dict[str, list[SourceResult]] = {}
        for name in contact_filter:
            chunks = await asyncio.to_thread(
                self._recent_chunks_for_contacts, {name}, top_k * 2
            )
            # Re-rank by similarity to embed_query using contact's own lines.
            if chunks:
                chunks = await asyncio.to_thread(
                    self._rerank_by_contact_similarity, chunks, {name}, embed_query, top_k
                )
            per_contact[name] = chunks

        if not any(per_contact.values()):
            return None

        # Interleave: [X1, Y1, X2, Y2, …]
        interleaved: list[SourceResult] = []
        names_sorted = sorted(per_contact.keys())
        max_len = max(len(v) for v in per_contact.values())
        for i in range(max_len):
            for n in names_sorted:
                if i < len(per_contact[n]):
                    interleaved.append(per_contact[n][i])
        interleaved = interleaved[:top_k]

        # Build context — only contact's own lines, prefixed by speaker.
        blocks: list[str] = []
        for s in interleaved:
            if not s.messages:
                continue
            contact_lines = _filter_for_target_contacts(s.messages, contact_filter)
            if not contact_lines:
                continue
            date = (s.date_start or "")[:10]
            header = f"[{_label(s.source)} · {date}]"
            body = "\n".join(f"{m.sender}: {m.text}" for m in contact_lines)
            blocks.append(f"{header}\n{body}")
        if not blocks:
            return None

        context = "\n\n".join(blocks)
        try:
            resp = await self._openai.chat.completions.create(
                extra_body=_LLM_EXTRA_BODY,
                model=LLM_MODEL,
                temperature=0.0,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": COMPARE_FOCUSED_PROMPT},
                    {"role": "user", "content": f"Query: {query}\n\n{context}"},
                ],
            )
            answer = (resp.choices[0].message.content or "").strip()
        except Exception:
            answer = ""
        if not answer or _looks_like_non_answer(answer):
            return None
        ok, bad = _validate_no_invented_quotes(answer, context)
        if not ok:
            answer = _strip_invented_quote_marks(answer, context)
            if not answer.strip(_ANSWER_STRIP_CHARS):
                return None
        return SearchResponse(
            answer=answer,
            sources=interleaved,
            query_ms=int((time.perf_counter() - t0) * 1000),
        )

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- Retrievers (each returns ordered list of chunk_ids) --

    def _dense_search(self, query: str, k: int) -> list[str]:
        vec = self.embedder.embed_one(query).astype("float32").reshape(1, -1)
        _scores, idxs = self.index.search(vec, k)
        return [self.id_map[i] for i in idxs[0].tolist() if 0 <= i < len(self.id_map)]

    def _ensure_clip(self):
        if self.clip_embedder is None:
            from clip_embedder import ClipEmbedder
            self.clip_embedder = ClipEmbedder.shared()
        return self.clip_embedder

    def _image_search(self, query: str, k: int) -> list[SourceResult]:
        if self.image_index is None or not self.image_id_map:
            return []
        clip = self._ensure_clip()
        vec = clip.embed_text_one(query).astype("float32").reshape(1, -1)
        scores, idxs = self.image_index.search(vec, k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()
        att_ids = [self.image_id_map[i] for i in idxs if 0 <= i < len(self.image_id_map)]
        if not att_ids:
            return []
        placeholders = ",".join("?" * len(att_ids))
        with self._open_db() as conn:
            rows = conn.execute(
                f"SELECT * FROM images WHERE attachment_id IN ({placeholders})", att_ids
            ).fetchall()
        by_id = {r["attachment_id"]: r for r in rows}
        out: list[SourceResult] = []
        for att_id, score in zip(att_ids, scores):
            r = by_id.get(att_id)
            if not r:
                continue
            sender = r["sender_name"] or "Unknown"
            chat_title = r["chat_title"]
            out.append(
                SourceResult(
                    source="image",
                    contact_names=[sender],
                    date_start=r["date_iso"] or "",
                    date_end=r["date_iso"] or "",
                    score=float(score),
                    messages=[],
                    subject=None,
                    chat_title=chat_title,
                    snippet=(r["msg_text"] or "")[:280],
                    image_url=f"/attachment/{att_id}",
                    image_caption=r["msg_text"] or None,
                    attachment_id=att_id,
                )
            )
        return out

    def get_image_path(self, attachment_id: int) -> str | None:
        with self._open_db() as conn:
            row = conn.execute(
                "SELECT path FROM images WHERE attachment_id = ?", (attachment_id,)
            ).fetchone()
        return row["path"] if row else None

    def _fts_search(
        self, query: str, k: int, source_filter: set[str] | None = None
    ) -> list[str]:
        match = _build_fts_match(query)
        if not match:
            return []
        with self._open_db() as conn:
            if source_filter:
                # Restrict at query time: a small source (208 notes) never
                # survives a global top-k against 57k iMessage chunks.
                placeholders = ",".join("?" * len(source_filter))
                rows = conn.execute(
                    "SELECT f.chunk_id FROM chunks_fts f "
                    "JOIN chunks c ON c.chunk_id = f.chunk_id "
                    f"WHERE chunks_fts MATCH ? AND c.source IN ({placeholders}) "
                    "ORDER BY bm25(chunks_fts) LIMIT ?",
                    (match, *source_filter, k),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
                    "ORDER BY bm25(chunks_fts) LIMIT ?",
                    (match, k),
                ).fetchall()
        return [r["chunk_id"] for r in rows]

    def _date_search(self, after_iso: str | None, before_iso: str | None, k: int) -> list[str]:
        """Return chunks whose date_start falls in the window, ordered by recency.
        Acts as a third retriever when a temporal phrase narrows the search."""
        if not after_iso and not before_iso:
            return []
        with self._open_db() as conn:
            params: list = []
            clauses: list[str] = []
            # Overlap semantics, matching _hydrate's date filter.
            if after_iso:
                clauses.append("date_end >= ?")
                params.append(after_iso)
            if before_iso:
                clauses.append("date_start <= ?")
                params.append(before_iso)
            params.append(k)
            sql = (
                "SELECT chunk_id FROM chunks WHERE " + " AND ".join(clauses) +
                " ORDER BY date_start DESC LIMIT ?"
            )
            rows = conn.execute(sql, params).fetchall()
        return [r["chunk_id"] for r in rows]

    def _apply_recency_boost(self, fused: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Re-sort fused (chunk_id, score) pairs with a mild time-decay boost so
        recent chunks win ties over equally-similar years-old ones. Chunks with
        a missing/unparseable date_end keep their raw score (no boost)."""
        if not fused:
            return fused
        chunk_ids = [cid for cid, _ in fused]
        placeholders = ",".join("?" * len(chunk_ids))
        with self._open_db() as conn:
            rows = conn.execute(
                f"SELECT chunk_id, date_end FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        date_by_id = {r["chunk_id"]: r["date_end"] for r in rows}
        boosted = [
            (cid, score * _recency_multiplier(_parse_age_days(date_by_id.get(cid))))
            for cid, score in fused
        ]
        return sorted(boosted, key=lambda x: x[1], reverse=True)

    @staticmethod
    def _rrf(*ranked_lists: list[str], k: int = RRF_K) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for ranking in ranked_lists:
            for rank, chunk_id in enumerate(ranking):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _hydrate(
        self,
        ranked: list[tuple[str, float]],
        top_k: int,
        date_range: tuple[str | None, str | None] | None = None,
        query: str = "",
        source_filter: set[str] | None = None,
        contact_filter: set[str] | None = None,
    ) -> list[SourceResult]:
        """Hydrate fused chunk_ids into SourceResults. Filters (date, source,
        contact) drop rows post-fetch — we over-fetch to compensate so the
        final list still hits top_k when filters are active."""
        if not ranked:
            return []
        has_filter = bool(date_range) or source_filter is not None or contact_filter is not None
        fetch_n = min(len(ranked), top_k * 8 if has_filter else top_k)
        chunk_ids = [cid for cid, _ in ranked[:fetch_n]]
        placeholders = ",".join("?" * len(chunk_ids))
        with self._open_db() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
            ).fetchall()
        by_id = {r["chunk_id"]: r for r in rows}
        out: list[SourceResult] = []
        after_iso, before_iso = date_range or (None, None)
        q_tokens = _query_tokens(query) if query else set()
        for chunk_id, score in ranked[:fetch_n]:
            r = by_id.get(chunk_id)
            if not r:
                continue
            if source_filter is not None and r["source"] not in source_filter:
                continue
            # Overlap, not containment: a chunk spanning a month (calendar
            # groups) must survive a one-day window that falls inside it.
            if after_iso and (r["date_end"] or "") < after_iso:
                continue
            if before_iso and (r["date_start"] or "9999") > before_iso:
                continue
            if contact_filter is not None:
                try:
                    chunk_contacts = set(json.loads(r["contact_names"] or "[]"))
                except (TypeError, ValueError):
                    chunk_contacts = set()
                if not (chunk_contacts & contact_filter):
                    continue
            messages_json = r["messages"] or "[]"
            messages = [ChunkMessage(**m) for m in json.loads(messages_json)]
            # Tag the message that best matches the query so the UI can anchor
            # the preview around it instead of always showing the chunk's tail.
            if messages and q_tokens:
                best = _best_message_index(messages, q_tokens)
                messages[best].is_best_match = True
            snippet = r["text"] or ""
            out.append(
                SourceResult(
                    source=r["source"],
                    contact_names=json.loads(r["contact_names"]),
                    date_start=r["date_start"] or "",
                    date_end=r["date_end"] or "",
                    score=float(score),
                    messages=messages,
                    subject=r["subject"],
                    chat_title=r["chat_title"],
                    snippet=snippet[:280] if not messages else "",
                )
            )
            if len(out) >= top_k:
                break
        return out

    def _rerank_by_contact_similarity(
        self,
        sources: list[SourceResult],
        target_contacts: set[str],
        query: str,
        top_k: int,
    ) -> list[SourceResult]:
        """For each source, embed only the contact's lines and score against
        the query. Sort descending by that score, then truncate to top_k.

        Solves the failure mode where a chunk dominated by the USER talking
        about a topic outranks a chunk where the CONTACT actually spoke on
        it. We retain the contact_summaries semantics (chunks where the
        contact never spoke are already filtered out earlier).
        """
        contact_texts: list[str] = []
        for s in sources:
            lines = [
                m.text for m in (s.messages or [])
                if m.sender in target_contacts and (m.text or "").strip()
            ]
            contact_texts.append("\n".join(lines))
        # If a source has no contact lines at all, give it a low fallback
        # score — it shouldn't surface above sources where the contact
        # actually spoke.
        non_empty_idx = [i for i, t in enumerate(contact_texts) if t.strip()]
        if not non_empty_idx:
            return sources[:top_k]
        valid_texts = [contact_texts[i] for i in non_empty_idx]
        query_vec = self.embedder.embed_one(query).astype("float32")
        text_vecs = self.embedder.embed_batch(
            valid_texts, show_progress_bar=False
        ).astype("float32")
        scores: list[float] = [-1.0] * len(sources)
        for vi, src_i in enumerate(non_empty_idx):
            scores[src_i] = float(np.dot(query_vec, text_vecs[vi]))
        ordered = sorted(range(len(sources)), key=lambda i: -scores[i])
        return [sources[i] for i in ordered[:top_k]]

    # -- Public --

    async def search(self, req: SearchRequest) -> SearchResponse:
        t0 = time.perf_counter()
        intent = req.intent or QueryIntent()
        if req.intent is not None:
            print(
                f"[search] query=\"{req.query}\" intent: topic=\"{intent.topic}\" "
                f"sources={intent.sources} contacts={intent.contacts} "
                f"attachment={intent.must_have_attachment} "
                f"query_type={intent.query_type}",
                file=sys.stderr,
            )

        # 1) Temporal stripping — independent of intent. We still want
        #    "yesterday" / "last week" detection on top of source filters.
        cleaned_query, trange = temporal.parse(req.query)
        # 2) The on-device router gives us the semantic core. Prefer it,
        #    fall back to the temporal-stripped query, then the raw query.
        topic = (intent.topic or "").strip()
        embed_query = topic or (cleaned_query if cleaned_query.strip() else req.query)
        date_filter = trange.to_iso_range() if trange else None

        # 3) Translate intent into retrieval-time filters.
        source_filter: set[str] | None = set(intent.sources) if intent.sources else None
        if source_filter is None:
            source_filter = _infer_sources_from_query(req.query)
        contact_filter: set[str] | None = None
        if intent.contacts:
            resolved = self._resolve_contacts(intent.contacts)
            if resolved:
                contact_filter = resolved
        # Fallback: the on-device LLM sometimes misses contact names that
        # are clearly in the query. Tokenize the raw query and pick up any
        # tokens that map to a known canonical contact name.
        fallback = self._extract_contacts_from_query(req.query)
        if fallback:
            if contact_filter is None:
                contact_filter = fallback
            else:
                contact_filter = contact_filter | fallback
            if not intent.contacts or not self._resolve_contacts(intent.contacts):
                print(
                    f"[search] fallback contact extraction added: {sorted(fallback)}",
                    file=sys.stderr,
                )
        # A contact name inside the embed text pollutes topic similarity
        # ("ruthvik cad" drifts toward chunks that merely mention ruthvik).
        # The contact constraint is enforced by metadata filtering, so the
        # dense/FTS query should carry only the topical remainder.
        if contact_filter:
            embed_query = _strip_query_tokens(
                embed_query, self._contact_name_tokens(contact_filter)
            )

        # Early-return paths for persona-based query types (style/affinity/temporal).
        # These bypass the full retrieval pipeline when we have richer per-contact
        # data than what FAISS/FTS can surface.
        # Prefer the on-device hint from QueryRouter when available (non-default),
        # fall back to keyword classifier. This avoids an extra round-trip cost.
        # Runs BEFORE the compare gate so "how do alex and sam talk" routes to
        # style, not compare.
        if intent and intent.query_type and intent.query_type != "standard":
            query_type = intent.query_type
        else:
            query_type = _classify_query_type(req.query, has_contact=bool(contact_filter))
        if query_type in ("style", "affinity", "temporal") and contact_filter:
            handler = {
                "style": self._synthesize_style,
                "affinity": self._synthesize_affinity,
                "temporal": self._synthesize_temporal,
            }[query_type]
            result = await handler(contact_filter, req.query, req.top_k)
            if result is not None:
                answer, sources = result
                return SearchResponse(
                    answer=answer,
                    sources=sources,
                    query_ms=int((time.perf_counter() - t0) * 1000),
                )
            # result is None → persona data not available, fall through to standard pipeline.

        # Task 3: Cross-contact comparison — two+ contacts + "both"/"vs"/"compare".
        # Bare "and" is deliberately NOT a trigger — it would hijack any
        # two-contact query containing the word.
        if (
            contact_filter is not None
            and len(contact_filter) >= 2
            and re.search(r"\b(both|vs|versus|compare)\b", req.query, re.I)
        ):
            result = await self._synthesize_compare(
                req.query, contact_filter, embed_query, req.top_k, t0
            )
            if result is not None:
                return result

        # If the user explicitly asked for an image, treat image as the only
        # relevant source — text excerpts about a photo aren't what they want.
        if intent.must_have_attachment:
            source_filter = {"image"}

        # Decide which retrievers to launch. Skip image if the user excluded
        # it; skip text retrievers when image is the only allowed source.
        run_text = source_filter is None or bool(source_filter - {"image"})
        run_image = source_filter is None or "image" in source_filter

        # Over-fetch when any post-fetch filter (source OR contact) will drop
        # candidates, so fusion still has enough survivors to fill top_k.
        fetch_mult = 4 if (source_filter or contact_filter) else 1
        # A source filter needs a much deeper dense pool: the post-fetch
        # filter throws away every other source, and small sources (a few
        # hundred chunks) rarely crack a global top-96 in a 68k index.
        dense_k = FUSE_FETCH * (32 if source_filter else fetch_mult)
        dense_task = asyncio.create_task(asyncio.to_thread(self._dense_search, embed_query, dense_k)) if run_text else None
        fts_task = asyncio.create_task(asyncio.to_thread(self._fts_search, embed_query, FUSE_FETCH * fetch_mult, source_filter)) if run_text else None
        image_task = asyncio.create_task(asyncio.to_thread(self._image_search, embed_query, max(4, req.top_k))) if run_image else None
        # Skip the remote call entirely (zero network traffic) when no key is set.
        remote_task = (
            asyncio.create_task(hyperspell.search(req.query, top_k=req.top_k))
            if os.getenv("HYPERSPELL_API_KEY")
            else None
        )
        date_task = None
        if date_filter and run_text:
            after_iso, before_iso = date_filter
            date_task = asyncio.create_task(
                asyncio.to_thread(self._date_search, after_iso, before_iso, FUSE_FETCH)
            )

        tasks: list = [t for t in (dense_task, fts_task, image_task, remote_task, date_task) if t is not None]
        results_list = await asyncio.gather(*tasks)
        results = dict(zip([t for t in (dense_task, fts_task, image_task, remote_task, date_task) if t is not None], results_list))
        dense_ids = results.get(dense_task, [])
        fts_ids = results.get(fts_task, [])
        image_results = results.get(image_task, [])
        remote_results = results.get(remote_task, [])
        date_ids = results.get(date_task, [])

        if date_ids:
            fused = self._rrf(dense_ids, fts_ids, date_ids)
        else:
            fused = self._rrf(dense_ids, fts_ids)

        # Recency-weighted re-rank of the fused scores. Skipped when an explicit
        # temporal range is in play — the date filter already handles time.
        if fused and date_filter is None:
            fused = await asyncio.to_thread(self._apply_recency_boost, fused)

        # When contact-filtered, over-hydrate so we have enough material to
        # re-rank by contact-only similarity below; otherwise the top-K cut
        # before re-ranking can already discard the chunks where the contact
        # actually spoke about the topic.
        hydrate_k = req.top_k * 4 if contact_filter is not None else req.top_k
        local = self._hydrate(
            fused,
            hydrate_k,
            date_range=date_filter,
            query=embed_query,
            source_filter=source_filter,
            contact_filter=contact_filter,
        )
        # Re-rank by similarity of the contact's OWN lines to the query. The
        # default FAISS/FTS ranking scores the whole chunk, which lets a
        # chunk dominated by the USER's monologue about a topic outrank a
        # chunk where the contact actually spoke on it. We re-score using
        # only contact-spoken text, then truncate to top_k.
        if contact_filter is not None and local:
            local = await asyncio.to_thread(
                self._rerank_by_contact_similarity,
                local, contact_filter, embed_query, req.top_k,
            )
        if date_filter:
            after_iso, before_iso = date_filter
            image_results = [
                r for r in image_results
                if (not after_iso or (r.date_start or "") >= after_iso)
                and (not before_iso or (r.date_end or "") <= before_iso)
            ]
        if contact_filter is not None:
            image_results = [
                r for r in image_results
                if any(name in contact_filter for name in (r.contact_names or []))
            ]

        # Fallback: if hard filters wiped everything out, retry without
        # contact/source filters so the user still sees results rather than
        # a confused empty state. We log so misroutes are visible.
        if not local and not image_results and (source_filter or contact_filter):
            print(
                f"[search] filtered to empty (sources={source_filter}, contacts={contact_filter}); "
                f"retrying unfiltered",
                file=sys.stderr,
            )
            local = self._hydrate(fused, req.top_k, date_range=date_filter, query=embed_query)

        merged = self._merge_results(local, image_results, remote_results, max_total=req.top_k)
        answer = await self._synthesize(req.query, merged, trange, target_contacts=contact_filter)
        return SearchResponse(
            answer=answer,
            sources=merged,
            query_ms=int((time.perf_counter() - t0) * 1000),
        )

    @staticmethod
    def _merge_results(
        local: list[SourceResult],
        images: list[SourceResult],
        remote: list[SourceResult],
        max_total: int,
    ) -> list[SourceResult]:
        # Interleave text + image roughly 2:1 so images get represented but don't
        # dominate. Then append any remote leftovers.
        out: list[SourceResult] = []
        ti, ii = 0, 0
        while len(out) < max_total and (ti < len(local) or ii < len(images)):
            for _ in range(2):
                if ti < len(local) and len(out) < max_total:
                    out.append(local[ti]); ti += 1
            if ii < len(images) and len(out) < max_total:
                out.append(images[ii]); ii += 1
        for r in remote:
            if len(out) >= max_total:
                break
            out.append(r)
        return out[:max_total]


    async def _synthesize(
        self,
        query: str,
        sources: list[SourceResult],
        trange: temporal.TemporalRange | None = None,
        target_contacts: set[str] | None = None,
    ) -> str:
        """Synthesize an answer from retrieved excerpts.

        When `target_contacts` is non-empty, we route to a separate path that
        feeds the LLM ONLY the contact's own lines and uses a focused prompt.
        Past attempts kept the user's words in via labels and lost — the LLM
        consistently leaked user opinions back as the contact's. The simpler
        the input, the safer the output.
        """
        # No API-key gate: the default LLM is a local keyless server, and every
        # call site degrades to "" on connection errors.
        if not sources:
            return ""
        if not _looks_like_question(query):
            return ""

        if target_contacts:
            return await self._synthesize_contact_focused(
                query, sources, trange, target_contacts
            )

        excerpts: list[str] = []
        for s in sources:
            who = ", ".join(s.contact_names) or "Unknown"
            date = (s.date_start or "")[:10]
            if s.messages:
                lines: list[str] = []
                for m in s.messages:
                    tag = "" if m.known or m.is_from_me else " (new contact, not in address book)"
                    lines.append(f"{m.sender}{tag}: {m.text}")
                full_text = "\n".join(lines)
            else:
                full_text = s.snippet
            excerpts.append(
                f"[SOURCE: {_label(s.source)} | CONTACT: {who} | DATE: {date}]\n{full_text}\n---"
            )
        context = "\n".join(excerpts)

        # Inject relationship summaries for any contacts that appear in the
        # excerpts. This grounds the LLM in who the user actually knows
        # (cluster-sampled summaries built at index time).
        contact_block = ""
        if self._contact_summaries:
            unique_contacts: list[str] = []
            for s in sources:
                for name in s.contact_names:
                    if name in self._contact_summaries and name not in unique_contacts:
                        unique_contacts.append(name)
            if unique_contacts:
                lines = []
                for name in unique_contacts[:6]:
                    cs = self._contact_summaries[name]
                    lines.append(
                        f"- {name}: {cs['summary']}  "
                        f"(known since {cs['first']}, {cs['total_chunks']} chunks total, "
                        f"{cs['last_30d_chunks']} in last 30 days)"
                    )
                contact_block = "Known contacts in this excerpt set:\n" + "\n".join(lines) + "\n\n"
        # Pass current date and any detected temporal range to the LLM so it can
        # reason about "yesterday" / "last week" against the excerpt dates.
        import datetime as _dt
        today = _dt.date.today().isoformat()
        date_note = f"Today is {today}."

        if trange:
            after = trange.after.date().isoformat() if trange.after else "?"
            before = trange.before.date().isoformat() if trange.before else "?"
            date_note += (
                f" The user's query refers to '{trange.label}' which means dates between "
                f"{after} and {before}. Only use excerpts in that range; ignore any older "
                "matches even if they contain similar words."
            )
        try:
            resp = await self._openai.chat.completions.create(
                extra_body=_LLM_EXTRA_BODY,
                model=LLM_MODEL,
                temperature=0.0,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{date_note}\n\n{contact_block}Query: {query}\n\nExcerpts:\n{context}"},
                ],
            )
            answer = (resp.choices[0].message.content or "").strip()
            # Defensive: if the model ignored instructions and returned a non-answer
            # placeholder, normalize back to an actual empty string.
            if _looks_like_non_answer(answer):
                return ""
            return answer
        except Exception:
            return ""

    async def _synthesize_contact_focused(
        self,
        query: str,
        sources: list[SourceResult],
        trange: temporal.TemporalRange | None,
        target_contacts: set[str],
    ) -> str:
        """Strict path: the LLM sees ONLY the targeted contacts' own messages.

        No user lines, no other speakers, no [EVIDENCE]/[CONTEXT] labels —
        the format is unmistakable: every line is a message the contact
        sent. Pattern/tone inference is still encouraged by the prompt.
        """
        # Gather only the contact's lines across all sources, with light
        # source headers so the model knows messages come from different
        # threads/dates.
        blocks: list[str] = []
        total_lines = 0
        for s in sources:
            if not s.messages:
                continue
            contact_lines = _filter_for_target_contacts(s.messages, target_contacts)
            if not contact_lines:
                continue
            date = (s.date_start or "")[:10]
            chat = s.chat_title or ""
            header_bits = [_label(s.source), date]
            if chat:
                header_bits.append(chat)
            header = " · ".join(b for b in header_bits if b)
            body = "\n".join(f"{m.sender}: {m.text}" for m in contact_lines)
            blocks.append(f"[{header}]\n{body}")
            total_lines += len(contact_lines)
        if total_lines == 0:
            # No contact-spoken lines in any excerpt — nothing to synthesize.
            return ""

        names = ", ".join(sorted(target_contacts))
        context = "\n\n".join(blocks)

        # Deliberately omit contact_summary here. The summary is the only
        # other place where user-phrased content could reach the LLM in
        # this path; keeping it out makes the input strictly the contact's
        # own words, which is what the quote validator also checks against.

        import datetime as _dt
        today = _dt.date.today().isoformat()
        date_note = f"Today is {today}."
        if trange:
            after = trange.after.date().isoformat() if trange.after else "?"
            before = trange.before.date().isoformat() if trange.before else "?"
            date_note += (
                f" The query refers to '{trange.label}' (between {after} and {before}). "
                f"Only use excerpts in that range."
            )

        user_content = (
            f"{date_note}\n\n"
            f"Query: {query}\n\n"
            f"Messages from {names} (the only person/people whose words you can see):\n"
            f"{context}"
        )
        try:
            resp = await self._openai.chat.completions.create(
                extra_body=_LLM_EXTRA_BODY,
                model=LLM_MODEL,
                temperature=0.0,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": CONTACT_FOCUSED_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            answer = (resp.choices[0].message.content or "").strip()
            if _looks_like_non_answer(answer):
                return ""
            # Quote integrity check: if the LLM put text in double quotes,
            # that text must appear verbatim in the messages we sent.
            # When the LLM paraphrases-in-quotes (e.g. "claude swore" when
            # the source has "did claude swear??"), we'd rather salvage the
            # paraphrase than return nothing — strip the offending quote
            # marks, leaving the paraphrase inline. The user still gets a
            # useful answer plus the underlying source cards.
            ok, bad = _validate_no_invented_quotes(answer, context)
            if not ok:
                print(
                    f"[synthesis] stripping invented quote {bad!r} from answer",
                    file=sys.stderr,
                )
                answer = _strip_invented_quote_marks(answer, context)
                # If after stripping the answer is empty or only punctuation,
                # there was nothing of substance beyond the fake quote.
                if not answer.strip(_ANSWER_STRIP_CHARS):
                    return ""
            return answer
        except Exception:
            return ""

