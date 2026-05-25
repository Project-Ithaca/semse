"""Per-contact persona builder.

Captures HOW a contact communicates (style, vocabulary, emoji use, topic breadth),
distinct from contact_summaries which captures the user's relationship with them.

Algorithm per contact:
  1. Extract all individual messages from that contact (is_from_me=False) from chunks.
  2. Compute style stats: avg length, emoji frequency, response_style bucket.
  3. K-means cluster messages into min(8, n//20) topic buckets; label each via LLM.
  4. Generate 1-sentence style summary via LLM from 5 representative messages.
  5. Cache by sample_hash; skip LLM if unchanged.

Cost: ~2 LLM calls per contact × ~150 tokens each × ~50 contacts ≈ 3¢ total.
Reruns are nearly free (sample_hash skip).
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

OPENAI_MODEL = "gpt-4o-mini"
MIN_MESSAGES_FOR_PERSONA = 10   # skip contacts with fewer individual messages

# Unicode ranges covering the most common emoji codepoints — no new dep needed.
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"    # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"    # transport & map
    "\U0001F1E0-\U0001F1FF"    # flags
    "\U00002702-\U000027B0"    # dingbats
    "\U000024C2-\U0001F251"    # enclosed chars
    "]+"
)


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
    quoted = json.dumps(name)           # → "Alice Smith"
    like = f'%{quoted[1:-1]}%'          # crude LIKE; same pattern as contact_summaries

    rows = conn.execute(
        "SELECT messages FROM chunks WHERE source='imessage' AND contact_names LIKE ?",
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
) -> tuple[list[dict], list[str]]:
    """Cluster messages, label via LLM, return (top_topics, style_sample_messages).

    top_topics: [{topic: str, score: float}, ...] sorted by score desc.
    style_sample_messages: up to 5 diverse message texts for the style summary call.
    """
    texts = [m.get("text") or "" for m in messages]
    n = len(texts)

    n_clusters = max(1, min(8, n // 20))

    print(f"  {name}: embedding {n} messages, {n_clusters} clusters…", file=sys.stderr)
    vecs = embedder.embed_batch(texts).astype("float32")

    # Too few messages for meaningful clustering.
    if n_clusters == 1 or n < 6:
        rep_texts = sorted(texts, key=len, reverse=True)[:5]
        return [{"topic": "general", "score": 1.0}], rep_texts

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
        return [{"topic": "general", "score": 1.0}], rep_texts

    # Topic label LLM call — one call for all clusters.
    reps_block = "\n".join(f"Cluster {i + 1}: {c[2]}" for i, c in enumerate(clusters))
    raw_labels = ""
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,
            max_tokens=120,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You receive one representative message per topic cluster. "
                        "Output exactly one line per cluster: the cluster number, a colon, "
                        "then a 2-3 word topic label. No preamble. No extras.\n"
                        "Example:\n1: weekend plans\n2: work stress\n3: family updates"
                    ),
                },
                {"role": "user", "content": reps_block},
            ],
        )
        raw_labels = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  {name}: topic labeling failed ({e})", file=sys.stderr)

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
    top_topics = top_topics[:8]

    # Style sample: one recent-ish message per cluster (up to 5 clusters).
    style_msgs: list[str] = []
    for _, members, _ in clusters[:5]:
        # Prefer recent messages (higher indices = later dates after sort).
        for idx in sorted(members, reverse=True)[:3]:
            t = texts[idx]
            if len(t) > 20:
                style_msgs.append(t[:300])
                break

    return top_topics, style_msgs[:5]


def _style_summary(rep_messages: list[str], name: str, client) -> str:
    """One-sentence description of HOW this person writes."""
    if not rep_messages:
        return f"{name} communicates via short messages."
    sample_block = "\n\n".join(f"• {m}" for m in rep_messages)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
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
        return (resp.choices[0].message.content or "").strip().strip("\"'")
    except Exception as e:
        print(f"  {name}: style summary failed ({e})", file=sys.stderr)
        return f"{name} communicates concisely."


def build(metadata_db: Path, openai_api_key: str | None = None) -> int:
    """Build or refresh contact personas. Returns number of personas written."""
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("persona_builder: OPENAI_API_KEY not set; skipping", file=sys.stderr)
        return 0
    from openai import OpenAI
    from embedder import Embedder
    from contacts import ContactResolver

    client = OpenAI(api_key=api_key)
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

        top_topics, rep_messages = _cluster_and_label(messages, embedder, client, name)
        style_sum = _style_summary(rep_messages, name, client)

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

        topics_preview = " · ".join(t["topic"] for t in top_topics[:4])
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
    base = Path(__file__).parent / "data"
    n = build(base / "metadata.db")
    sys.exit(0 if n >= 0 else 1)
