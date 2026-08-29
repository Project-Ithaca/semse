"""Pure query-understanding helpers: classification, name/token normalization,
source inference, FTS query building, quote validation, and recency scoring.

None of these touch the database, the index, or the LLM — they're kept
separate from SearchEngine so they can be unit-tested in isolation.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .models import ChunkMessage

INDEXER_DIR = Path(__file__).resolve().parent.parent / "indexer"
sys.path.insert(0, str(INDEXER_DIR))
from persona_builder import filter_topic_labels  # noqa: E402

# -- Query classification (style / affinity / temporal / standard) --

# Strip punctuation for pattern matching in _classify_query_type.
_STRIP_PUNC = re.compile(r"[^\w\s]")

_QUESTION_LEAD_WORDS = {
    "what", "who", "where", "when", "why", "how", "does", "do", "did",
    "is", "are", "was", "were", "has", "have", "had", "can", "could",
    "should", "will", "would", "tell", "explain", "summarize", "list",
    "show", "find", "any", "which",
}


def _looks_like_question(query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    if q.endswith("?"):
        return True
    first = q.split(maxsplit=1)[0] if q else ""
    # Strip contractions: "what's" → "what", "who're" → "who", etc.
    bare = re.sub(r"['’](?:s|re|ve|d|ll|m)$", "", first)
    if first in _QUESTION_LEAD_WORDS or bare in _QUESTION_LEAD_WORDS:
        return True
    return False


def _classify_query_type(query: str, has_contact: bool) -> str:
    """Returns 'style' | 'affinity' | 'temporal' | 'standard'.

    Fast keyword matching only. No LLM. Runs before the search pipeline.
    All patterns require has_contact=True because persona data is per-contact.
    """
    if not has_contact:
        return "standard"
    q = _STRIP_PUNC.sub(" ", query).lower()

    # Style: how does this person communicate / write / talk?
    if ("how does" in q or "how do" in q) and any(
        w in q for w in ("talk", "write", "communicate")
    ):
        return "style"
    if "like to talk to" in q or "like to text" in q:
        return "style"
    # "what's X like" / "what is X like" at end of query. _STRIP_PUNC has
    # already replaced the apostrophe with a space ("what's" → "what s"),
    # so match the stripped forms, never the apostrophe one.
    if re.search(r"what(?:s|\s+s|\s+is)\s+\w[\w\s]*\blike\b\s*$", q):
        return "style"

    # Affinity: what topics / interests / things they care about
    if "care about" in q or "cares about" in q:
        return "affinity"
    if "what topics" in q:
        return "affinity"
    if ("what is" in q or "what does" in q) and " into " in q:
        return "affinity"
    if ("what does" in q or "what do" in q) and "talk about" in q:
        return "affinity"

    # Temporal: how someone has changed over time
    if "been thinking" in q and "recent" in q:
        return "temporal"
    if ("how has" in q or "how have" in q) and "changed" in q:
        return "temporal"
    if "used to talk" in q:
        return "temporal"
    if "different now" in q:
        return "temporal"

    return "standard"


# -- Explicit source inference ("in my notes", "on whatsapp") --

# Explicit source mentions in the query → hard source filter. Patterns are
# deliberately phrasal ("my notes", not bare "notes") so ordinary words don't
# hijack the search.
_SOURCE_PHRASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:my|in|from)\s+notes?\b|\bnotes? about\b", re.I), "notes"),
    (re.compile(r"\bwhats\s*app\b|\bwhatsapp\b", re.I), "whatsapp"),
    (re.compile(r"\b(?:call|called|calls|phone call|facetime)\b", re.I), "calls"),
    (re.compile(r"\b(?:website|websites|browse|browsed|browsing|visited\s+site)\b", re.I), "browsing"),
    (re.compile(r"\b(?:email|emails|mail)\b", re.I), "mail"),
    (re.compile(r"\b(?:calendar|event|events|meeting|appointment|schedule)\b", re.I), "calendar"),
    (re.compile(r"\breminders?\b|\btodo list\b|\bto-do\b", re.I), "reminders"),
]


def _infer_sources_from_query(query: str) -> set[str] | None:
    """Detect explicit source mentions ("in my notes", "on whatsapp") and
    return a source filter, or None when the query names no source."""
    hits = {tag for pat, tag in _SOURCE_PHRASES if pat.search(query)}
    return hits or None


# -- FTS5 query building --

_FTS_TOKEN_SAFE = re.compile(r"[A-Za-z0-9]+")

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "of", "in", "on", "at", "to",
    "for", "with", "and", "or", "but", "if", "then", "this", "that", "it",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "she", "they",
    "what", "who", "when", "where", "why", "how", "any", "some", "all",
    "about", "from", "by", "as",
}


def _build_fts_match(query: str) -> str:
    """Tokenize a free-text query into an FTS5 MATCH expression with prefix tokens.

    Uses OR so we maximize recall (a chunk mentioning *either* term is a candidate);
    RRF will surface the strongest matches when fused with dense retrieval.
    Stripped of stop-words and short tokens to avoid noise.
    """
    tokens = [t.lower() for t in _FTS_TOKEN_SAFE.findall(query) if len(t) >= 2]
    tokens = [t for t in tokens if t not in _STOP]
    if not tokens:
        return ""
    return " OR ".join(f"{t}*" for t in tokens)


# -- Name / token normalization --

_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")
_NAME_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def _query_tokens(query: str) -> set[str]:
    return {t.lower() for t in _QUERY_TOKEN_RE.findall(query)}


def _normalize_name(name: str) -> str:
    """Lowercase + strip punctuation. Used to fuzzy-resolve LLM-extracted
    contact names against the index's canonical names."""
    if not name:
        return ""
    return _NAME_NORM_RE.sub("", name.lower()).strip()


def _strip_query_tokens(text: str, tokens: set[str]) -> str:
    """Remove whitespace-separated words whose normalized form is in `tokens`.
    Returns the original text when stripping would leave nothing — a query
    that is ONLY a contact name still needs something to embed."""
    if not text or not tokens:
        return text
    kept = [w for w in text.split() if _normalize_name(w) not in tokens]
    stripped = " ".join(kept)
    return stripped if stripped.strip() else text


def _display_name(name: str) -> str:
    """Capitalize all-lowercase saved contact names ("vedo" → "Vedo") for
    answers; names with existing capitals are left untouched."""
    return " ".join(w.capitalize() if w.islower() else w for w in (name or "").split())


def _best_message_index(messages: list[ChunkMessage], q_tokens: set[str]) -> int:
    """Pick the message with the highest query-token overlap. Tie-breaks to the
    longest message (more substantive)."""
    if not messages or not q_tokens:
        return 0
    best_i, best_score, best_len = 0, -1, -1
    for i, m in enumerate(messages):
        toks = {t.lower() for t in _QUERY_TOKEN_RE.findall(m.text or "")}
        score = len(toks & q_tokens)
        text_len = len(m.text or "")
        if score > best_score or (score == best_score and text_len > best_len):
            best_i, best_score, best_len = i, score, text_len
    return best_i


def _filter_for_target_contacts(
    messages: list[ChunkMessage],
    target: set[str],
) -> list[ChunkMessage]:
    """Keep ONLY messages spoken by target contacts. No surrounding context.

    Previous versions kept one preceding line for context, but the LLM kept
    paraphrasing that preceding line as the contact's view. With user lines
    gone entirely, there's nothing for it to mistakenly attribute. The
    contact's tone, vocabulary, and topical pattern is still fully present
    in their own messages."""
    if not messages or not target:
        return messages
    return [m for m in messages if m.sender in target]


# -- Quote validation (defends against LLM-invented verbatim quotes) --

# Chars stripped when checking whether an answer has any substance left
# after invented quotes are removed. Includes curly quotes.
_ANSWER_STRIP_CHARS = " .,;:!?\"'“”‘’()[]"

# Only double-quote pairs count as a quotation for validation. Single quotes
# / apostrophes are too ambiguous in English (possessives, contractions) to
# treat as quote delimiters without false positives.
_QUOTE_RE = re.compile(r'["“”]([^"“”\n]{2,120})["“”]')
_NORMALIZE_QUOTE_RE = re.compile(r"\s+")


def _normalize_for_compare(s: str) -> str:
    # Lowercase, collapse whitespace, strip trailing/leading punctuation
    # that the LLM might add when embedding a quote in its own sentence.
    cleaned = _NORMALIZE_QUOTE_RE.sub(" ", s.lower()).strip()
    return cleaned.strip(".,;:!?…\"'“”‘’ ")


def _strip_invented_quote_marks(answer: str, allowed_text: str) -> str:
    """For each double-quoted phrase in `answer`, if it's not verbatim in
    `allowed_text`, remove the surrounding quote marks (keep the text).
    Real verbatim quotes are preserved."""
    haystack = _normalize_for_compare(allowed_text)

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        norm = _normalize_for_compare(inner)
        if norm and norm in haystack:
            return m.group(0)  # real quote — keep as-is
        return inner  # strip the quote marks, keep the content

    return _QUOTE_RE.sub(repl, answer)


def _validate_no_invented_quotes(answer: str, allowed_text: str) -> tuple[bool, str | None]:
    """Every quoted phrase in `answer` must appear (case-insensitively, with
    whitespace collapsed, and ignoring surrounding punctuation) inside
    `allowed_text`. Returns (ok, offending_quote).

    Defense against gpt-4o-mini fabricating "alarm said 'so peak'" when those
    words were never in the excerpts. A single fabricated quote rejects the
    whole answer — the user gets nothing instead of misinformation.
    """
    haystack = _normalize_for_compare(allowed_text)
    for m in _QUOTE_RE.finditer(answer):
        quoted = m.group(1).strip()
        if len(quoted) < 3:
            continue
        norm = _normalize_for_compare(quoted)
        if not norm:
            continue
        if norm not in haystack:
            return False, quoted
    return True, None


_NON_ANSWER_PATTERNS = (
    "empty",
    "no answer",
    "no relevant",
    "i cannot",
    "i can't",
    "i don't have",
    "(nothing)",
    "n/a",
)


def _looks_like_non_answer(text: str) -> bool:
    if not text:
        return True
    t = text.strip().lower().strip(".\"'() ")
    if len(t) < 30 and any(p in t for p in _NON_ANSWER_PATTERNS):
        return True
    return False


# -- Recency-weighted ranking --

# Recency-weighted ranking: after RRF fusion, each fused score is multiplied by
# 1 + RECENCY_BOOST * exp(-age_days / RECENCY_HALFLIFE_DAYS). Recent content
# wins ties decisively; a strong old exact match still surfaces (max uplift is
# only 1.35x). Skipped when an explicit temporal range already filters dates.
RECENCY_BOOST = 0.35
RECENCY_HALFLIFE_DAYS = 270.0


def _parse_age_days(date_iso: str | None, now: "object" = None) -> float | None:
    """Age in days of a naive-UTC ISO date string relative to `now` (naive-UTC
    datetime; defaults to utcnow). Returns None when the date is missing or
    unparseable — callers treat that as "no boost"."""
    import datetime as _dt

    if not date_iso:
        return None
    try:
        d = _dt.datetime.fromisoformat(str(date_iso).replace("Z", "").strip())
    except (ValueError, TypeError):
        return None
    if d.tzinfo is not None:
        d = d.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    ref = now if now is not None else _dt.datetime.utcnow()
    return (ref - d).total_seconds() / 86400.0


def _recency_multiplier(age_days: float | None) -> float:
    """Mild exponential time-decay boost. None (unknown date) → no boost;
    future-dated chunks are clamped to age 0 (max boost)."""
    import math

    if age_days is None:
        return 1.0
    age = max(0.0, age_days)
    return 1.0 + RECENCY_BOOST * math.exp(-age / RECENCY_HALFLIFE_DAYS)


# -- Temporal ("how has X changed") recent/older split --

# Temporal ("how has X changed") split tuning: recent = last N days, widened to
# the newest fraction of the contact's chunks when the window is too thin.
TEMPORAL_RECENT_DAYS = 90
TEMPORAL_MIN_RECENT_CHUNKS = 15
TEMPORAL_RECENT_FRACTION = 0.25
TEMPORAL_RECENT_FLOOR = 10


def _adaptive_recent_split(chunks: list, now: "object" = None) -> tuple[list, list]:
    """Split chunks (sorted newest-first by date_end) into (recent, older).

    Recent = last TEMPORAL_RECENT_DAYS days; when that yields fewer than
    TEMPORAL_MIN_RECENT_CHUNKS, widen to the newest TEMPORAL_RECENT_FRACTION
    of the contact's chunks (at least TEMPORAL_RECENT_FLOOR). A contact with
    very few chunks total may end up with an empty `older` side — callers
    treat that as "not enough history to compare"."""
    if not chunks:
        return [], []
    cut = 0
    for c in chunks:
        age = _parse_age_days(getattr(c, "date_end", None), now=now)
        if age is not None and age <= TEMPORAL_RECENT_DAYS:
            cut += 1
        else:
            break
    if cut < TEMPORAL_MIN_RECENT_CHUNKS:
        frac = int(len(chunks) * TEMPORAL_RECENT_FRACTION)
        cut = min(len(chunks), max(cut, TEMPORAL_RECENT_FLOOR, frac))
    return chunks[:cut], chunks[cut:]


# -- Persona helpers --


def _persona_topics(persona: dict) -> list[dict]:
    """Parse a persona row's top_topics JSON; malformed rows must not 500 the
    endpoint, so bad JSON and non-dict/topicless entries are skipped. Junk
    labels (pronouns, question words, bare verbs) written by older persona
    builds are filtered here so they never surface in answers."""
    try:
        raw = json.loads(persona.get("top_topics") or "[]")
    except (TypeError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    valid = [t for t in raw if isinstance(t, dict) and t.get("topic")]
    return filter_topic_labels(valid)


# -- Display helpers --


def _label(source: str) -> str:
    return {"imessage": "iMessage", "mail": "Mail", "hyperspell": "Hyperspell"}.get(source, source.title())
