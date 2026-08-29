"""Per-contact persona builder.

Captures HOW a contact communicates (style, vocabulary, emoji use, topic breadth),
distinct from contact_summaries which captures the user's relationship with them.

Algorithm per contact:
  1. Extract all individual messages from that contact (is_from_me=False) from chunks.
  2. Compute style stats: avg length, emoji frequency, response_style bucket.
  3. K-means cluster messages into min(8, max(2, n//10)) topic buckets; label each via LLM.
  4. Generate 1-sentence style summary via LLM from 5 representative messages.
  5. Cache by sample_hash; skip LLM if unchanged.

LLM calls go to a local OpenAI-compatible server (Ollama) by default — see
SEMSE_LLM_BASE_URL / SEMSE_LLM_MODEL. Reruns are nearly free (sample_hash skip).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

MIN_MESSAGES_FOR_PERSONA = 10   # skip contacts with fewer individual messages


# Read at call time, not import time, so load_dotenv() in __main__ blocks
# still applies overrides.
def llm_base_url() -> str:
    return os.getenv("SEMSE_LLM_BASE_URL", "http://localhost:11434/v1")


def llm_model() -> str:
    return os.getenv("SEMSE_LLM_MODEL", "qwen2.5:14b")

# Unicode ranges covering the most common emoji codepoints — no new dep needed.
# Deliberately narrow: broad ranges like U+24C2–U+1F251 swallow all CJK text.
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"    # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"    # transport & map
    "\U0001F1E6-\U0001F1FF"    # regional indicators (flags)
    "\U0001F900-\U0001F9FF"    # supplemental symbols & pictographs
    "\U0001FA70-\U0001FAFF"    # symbols & pictographs extended-A
    "\U00002702-\U000027B0"    # dingbats
    "\U00002600-\U000026FF"    # misc symbols (sun, heart suits, etc.)
    "\U0000FE0F"               # variation selector
    "\U00002764"               # heavy black heart
    "]+"
)


def make_llm_client(async_client: bool = False):
    """OpenAI-compatible client pointed at a local server (Ollama) by default.

    Set SEMSE_LLM_PREFER_OPENAI=1 (with OPENAI_API_KEY, and SEMSE_LLM_BASE_URL
    unset) to route to OpenAI instead.
    """
    from openai import AsyncOpenAI, OpenAI

    explicit_local = "SEMSE_LLM_BASE_URL" in os.environ
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and not explicit_local and os.getenv("SEMSE_LLM_PREFER_OPENAI"):
        kwargs = {"api_key": openai_key}
    else:
        kwargs = {"base_url": llm_base_url(), "api_key": openai_key or "ollama"}
    return AsyncOpenAI(**kwargs) if async_client else OpenAI(**kwargs)


def check_llm_available() -> str | None:
    """Return an error string if the LLM endpoint is unreachable, else None."""
    try:
        client = make_llm_client()
        client.models.list()
        return None
    except Exception as e:
        return (
            f"LLM endpoint unreachable at {llm_base_url()} ({e}). "
            "Start it with: ollama serve  (and: ollama pull " + llm_model() + ")"
        )


def escape_like(value: str) -> str:
    """Escape %, _ and \\ for use in a LIKE pattern with ESCAPE '\\'."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# Conservative deny-list for topic labels: pronouns, question words, and bare
# common verbs carry no topical content. Kept narrow on purpose — ambiguous
# noun/verb words ("request", "plans") stay allowed.
_JUNK_LABEL_DENY = {
    # pronouns / determiners
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your",
    "yours", "he", "him", "his", "she", "her", "hers", "they", "them",
    "their", "theirs", "it", "its", "this", "that", "these", "those",
    "someone", "something", "anything", "nothing", "everyone", "everything",
    # question words
    "who", "whom", "whose", "what", "which", "when", "where", "why", "how",
    # bare/auxiliary verbs and common verb forms
    "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "done", "have", "has", "had",
    "say", "says", "said", "saying", "ask", "asks", "asked",
    "tell", "tells", "told", "talk", "talks", "talked", "talking",
    "think", "thinks", "thought", "want", "wants", "wanted",
    "know", "knows", "knew", "get", "gets", "got", "go", "goes",
    "went", "going", "see", "sees", "saw", "make", "makes", "made",
    "score", "scored", "can", "could", "will", "would", "should",
    "may", "might", "must",
    # filler
    "yes", "no", "ok", "okay", "yeah", "nah", "lol", "stuff", "thing",
    "things", "misc", "unknown", "none", "name", "names",
}

_LABEL_WORD_RE = re.compile(r"[a-z]+")


def is_junk_topic_label(label: str) -> bool:
    """True for labels that carry no topical content: too short, letterless,
    or made up entirely of pronouns/question-words/bare-verbs."""
    text = (label or "").strip()
    if len(text) < 3:
        return True
    words = _LABEL_WORD_RE.findall(text.lower())
    if not words:
        return True
    return all(w in _JUNK_LABEL_DENY for w in words)


def filter_topic_labels(topics: list[dict]) -> list[dict]:
    """Drop junk-labeled topics and case-insensitive duplicates, preserving
    order. Each entry must be a dict with a string 'topic' key."""
    seen: set[str] = set()
    out: list[dict] = []
    for t in topics:
        label = str(t.get("topic") or "").strip()
        if is_junk_topic_label(label):
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


@dataclass
class ContactPersona:
    name: str
    style_summary: str
    avg_msg_length: float
    emoji_frequency: float
    response_style: str     # "brief" | "medium" | "verbose"
    top_topics: str         # JSON: [{topic: str, score: float}, ...]
    first_message_iso: str
    last_message_iso: str
    total_messages: int
    sample_hash: str


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contact_personas (
            name TEXT PRIMARY KEY,
            style_summary TEXT,
            avg_msg_length REAL,
            emoji_frequency REAL,
            response_style TEXT,
            top_topics TEXT,
            first_message_iso TEXT,
            last_message_iso TEXT,
            total_messages INTEGER,
            sample_hash TEXT
        )
    """)
    conn.commit()


def _existing_hash(conn: sqlite3.Connection, name: str) -> str | None:
    row = conn.execute(
        "SELECT sample_hash FROM contact_personas WHERE name=?", (name,)
    ).fetchone()
    return row[0] if row else None


def _save(conn: sqlite3.Connection, p: ContactPersona) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO contact_personas "
        "(name, style_summary, avg_msg_length, emoji_frequency, response_style, "
        " top_topics, first_message_iso, last_message_iso, total_messages, sample_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            p.name, p.style_summary, p.avg_msg_length, p.emoji_frequency,
            p.response_style, p.top_topics, p.first_message_iso, p.last_message_iso,
            p.total_messages, p.sample_hash,
        ),
    )
    conn.commit()


def _extract_messages(conn: sqlite3.Connection, name: str) -> list[dict]:
    """Return all individual messages from this contact across all their chunks.

    Deduplicates by (text, date_iso) since messages appear in multiple overlapping
    sliding windows. Returns sorted by date_iso ascending.
    """
    # Match the full JSON-quoted name so "Al" can't match "Alice Smith".
    like = f'%"{escape_like(name)}"%'

    rows = conn.execute(
        "SELECT messages FROM chunks WHERE source='imessage' "
        "AND contact_names LIKE ? ESCAPE '\\'",
        (like,),
    ).fetchall()

    seen: set[str] = set()
    messages: list[dict] = []
    for (messages_json,) in rows:
        try:
            msgs = json.loads(messages_json or "[]")
        except json.JSONDecodeError:
            continue
        for m in msgs:
            if m.get("is_from_me"):
                continue
            # In group chats a chunk mentions several people — keep only THIS
            # contact's messages, or their persona absorbs everyone else's style.
            if m.get("sender") != name:
                continue
            text = (m.get("text") or "").strip()
            if len(text) < 3:
                continue
            key = f"{text}|{m.get('date_iso', '')}"
            if key in seen:
                continue
            seen.add(key)
            messages.append(m)

    messages.sort(key=lambda m: m.get("date_iso") or "")
    return messages


def _emoji_count(text: str) -> int:
    return sum(len(match.group()) for match in _EMOJI_RE.finditer(text))


def _compute_style_stats(messages: list[dict]) -> tuple[float, float, str]:
    """Returns (avg_msg_length, emoji_frequency, response_style)."""
    texts = [m.get("text") or "" for m in messages]
    lengths = [len(t) for t in texts]
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0

    total_chars = sum(lengths)
    total_emojis = sum(_emoji_count(t) for t in texts)
    emoji_freq = (total_emojis / total_chars * 100) if total_chars > 0 else 0.0

    if avg_len < 60:
        style = "brief"
    elif avg_len <= 200:
        style = "medium"
    else:
        style = "verbose"

    return round(avg_len, 1), round(emoji_freq, 2), style


def _compute_sample_hash(messages: list[dict], name: str) -> str:
    # Hash first 200 messages (stable across reruns with same data).
    sample_texts = [m.get("text") or "" for m in messages[:200]]
    payload = name + "\n" + "\n".join(sample_texts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _cluster_and_label(
    messages: list[dict],
    embedder,
    client,
    name: str,
) -> tuple[list[dict], list[str], bool]:
    """Cluster messages, label via LLM, return (top_topics, style_samples, llm_ok).

    top_topics: [{topic: str, score: float}, ...] sorted by score desc.
    style_samples: up to 5 diverse message texts for the style summary call.
    llm_ok: False when the labeling call failed (caller should not cache).
    """
    texts = [m.get("text") or "" for m in messages]
    n = len(texts)

    # ~1 cluster per 10 messages so even 10-39-msg contacts get real topics.
    n_clusters = min(8, max(2, n // 10))

    # Too few messages for meaningful clustering — decide BEFORE paying for embeds.
    if n < 6:
        rep_texts = sorted(texts, key=len, reverse=True)[:5]
        return [{"topic": "general", "score": 1.0}], rep_texts, True

    print(f"  {name}: embedding {n} messages, {n_clusters} clusters…", file=sys.stderr)
    vecs = embedder.embed_batch(texts).astype("float32")

    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=42)
    labels = km.fit_predict(vecs)

    # Build clusters; drop those with < 3 messages per spec.
    clusters: list[tuple[int, list[int], str]] = []
    for cid in range(n_clusters):
        members = np.where(labels == cid)[0].tolist()
        if len(members) < 3:
            continue
        centroid = km.cluster_centers_[cid]
        member_vecs = vecs[members]
        sims = member_vecs @ centroid / (np.linalg.norm(centroid) or 1.0)
        best_local = int(np.argmax(sims))
        rep = texts[members[best_local]][:300]
        clusters.append((cid, members, rep))

    if not clusters:
        rep_texts = sorted(texts, key=len, reverse=True)[:5]
        return [{"topic": "general", "score": 1.0}], rep_texts, True

    # Topic label LLM call — one call for all clusters.
    reps_block = "\n".join(f"Cluster {i + 1}: {c[2]}" for i, c in enumerate(clusters))
    raw_labels = ""
    llm_ok = True
    try:
        resp = client.chat.completions.create(
            model=llm_model(),
            temperature=0.0,
            max_tokens=120,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You receive one representative message per topic cluster from "
                        "one person's texts. Output exactly one line per cluster: the "
                        "cluster number, a colon, then a concrete 2-3 word noun-phrase "
                        "topic label naming the SUBJECT MATTER of the message.\n"
                        "Rules: every label must be a noun phrase. Never output a "
                        "pronoun, a question word, or a bare verb phrase as a label.\n"
                        "Good labels: 'robotics competition', 'college applications', "
                        "'weekend dinner plans'.\n"
                        "Bad labels: 'who' (question word, not a subject), "
                        "'scored them' (verb phrase with no subject).\n"
                        "No preamble. No extras.\n"
                        "Example:\n1: weekend plans\n2: work stress\n3: family updates"
                    ),
                },
                {"role": "user", "content": reps_block},
            ],
        )
        raw_labels = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  {name}: topic labeling failed ({e})", file=sys.stderr)
        llm_ok = False

    label_map: dict[int, str] = {}
    for line in raw_labels.splitlines():
        m = re.match(r"(\d+):\s*(.+)", line.strip())
        if m:
            label_map[int(m.group(1))] = m.group(2).strip()

    top_topics: list[dict] = []
    for i, (_, members, _) in enumerate(clusters):
        label = label_map.get(i + 1, f"topic {i + 1}")
        score = round(len(members) / n, 3)
        top_topics.append({"topic": label, "score": score})

    top_topics.sort(key=lambda t: t["score"], reverse=True)
    top_topics = filter_topic_labels(top_topics)[:8]

    # Style sample: one recent-ish message per cluster (up to 5 clusters).
    style_msgs: list[str] = []
    for _, members, _ in clusters[:5]:
        # Prefer recent messages (higher indices = later dates after sort).
        for idx in sorted(members, reverse=True)[:3]:
            t = texts[idx]
            if len(t) > 20:
                style_msgs.append(t[:300])
                break

    return top_topics, style_msgs[:5], llm_ok


def _style_summary(rep_messages: list[str], name: str, client) -> tuple[str, bool]:
    """One-sentence description of HOW this person writes. Returns (text, llm_ok)."""
    if not rep_messages:
        return f"{name} communicates via short messages.", True
    sample_block = "\n\n".join(f"• {m}" for m in rep_messages)
    try:
        resp = client.chat.completions.create(
            model=llm_model(),
            temperature=0.0,
            max_tokens=80,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Describe in ONE sentence how this person writes — "
                        "their tone, length, vocabulary, humor, emoji use. "
                        "Not what they say, how they say it. No preamble."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Contact: {name}\n\nSample messages:\n{sample_block}",
                },
            ],
        )
        return (resp.choices[0].message.content or "").strip().strip("\"'"), True
    except Exception as e:
        print(f"  {name}: style summary failed ({e})", file=sys.stderr)
        return f"{name} communicates concisely.", False


def build(metadata_db: Path) -> int:
    """Build or refresh contact personas. Returns number of personas written."""
    err = check_llm_available()
    if err:
        raise SystemExit(f"persona_builder: {err}")
    from embedder import Embedder
    from contacts import ContactResolver

    client = make_llm_client()
    embedder = Embedder()

    conn = sqlite3.connect(metadata_db)
    init_db(conn)

    resolver = ContactResolver()
    resolver.load()
    contacts_by_name: dict[str, object] = {}
    for c in (*resolver._by_phone.values(), *resolver._by_email.values()):
        if c.name not in contacts_by_name:
            contacts_by_name[c.name] = c

    print(f"Building personas for {len(contacts_by_name)} contacts…", file=sys.stderr)

    written, skipped, too_few = 0, 0, 0
    for name in sorted(contacts_by_name):
        messages = _extract_messages(conn, name)
        if len(messages) < MIN_MESSAGES_FOR_PERSONA:
            too_few += 1
            continue

        sample_hash = _compute_sample_hash(messages, name)
        if _existing_hash(conn, name) == sample_hash:
            skipped += 1
            continue

        avg_len, emoji_freq, response_style = _compute_style_stats(messages)

        top_topics, rep_messages, cluster_ok = _cluster_and_label(
            messages, embedder, client, name
        )
        style_sum, style_ok = _style_summary(rep_messages, name, client)
        # A failed LLM call must not be cached as done — blank the hash so the
        # next run retries instead of permanently keeping the fallback text.
        if not (cluster_ok and style_ok):
            sample_hash = ""

        dates = [m.get("date_iso") for m in messages if m.get("date_iso")]
        first_iso = min(dates) if dates else ""
        last_iso = max(dates) if dates else ""

        persona = ContactPersona(
            name=name,
            style_summary=style_sum,
            avg_msg_length=avg_len,
            emoji_frequency=emoji_freq,
            response_style=response_style,
            top_topics=json.dumps(top_topics),
            first_message_iso=first_iso,
            last_message_iso=last_iso,
            total_messages=len(messages),
            sample_hash=sample_hash,
        )
        _save(conn, persona)
        written += 1

        print(
            f"  [{response_style}, {len(messages)} msgs, {len(top_topics)} topics] "
            f"{name}: {style_sum}",
            file=sys.stderr,
        )

    conn.close()
    print(
        f"\npersona_builder: wrote {written}, cached {skipped}, "
        f"{too_few} contacts had fewer than {MIN_MESSAGES_FOR_PERSONA} messages",
        file=sys.stderr,
    )
    return written


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / "api" / ".env")
    load_dotenv(Path(__file__).parent / ".env")
    db = Path(__file__).parent / "data" / "metadata.db"
    if not db.exists():
        raise SystemExit(f"persona_builder: {db} not found — run build_index.py first")
    build(db)
