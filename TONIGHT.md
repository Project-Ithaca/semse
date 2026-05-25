# TONIGHT.md — Semse Overnight Session

## How to work

Work through the tasks below **in order**. For each task:

1. **Plan first.** Before writing any code, think through the approach. Read the relevant existing files. State your plan as a short comment block at the top of any new file you create, or as a brief preamble before edits.
2. **Implement.** Follow the exact style of the surrounding code. No new dependencies unless absolutely necessary — check `indexer/requirements.txt` first.
3. **Verify.** Run any quick smoke tests possible (the `if __name__ == "__main__"` blocks in existing files are good models — replicate that pattern). If a task has acceptance criteria, run them.
4. **Review.** Run `/codex:review` on your changes before committing.
5. **Commit.** Use a clear conventional-commit-style message: `feat(indexer): add persona builder` etc.
6. **Move on.** Do not ask the user questions. Make reasonable decisions and document them in code comments.

If a task is blocked (e.g. requires data that doesn't exist yet), skip it, leave a TODO comment in the relevant file, and continue to the next task. Don't get stuck.

---

## Vision context (read CLAUDE.md first, but here's the short version)

Semse is a Spotlight-style semantic search over personal communication history (iMessage, Mail, image attachments). The user wants it to feel less like a search engine and more like a memory assistant that **knows the people in their life**. Tonight's work pushes the project in that direction: build per-contact personas, support style/topic/temporal queries, and make incremental updates trivial.

The codebase is already mature. Don't refactor existing patterns. Extend them.

---

## Task 1 — Contact Persona Builder

**Goal:** Per-contact persona richer than the existing `contact_summaries`. Captures *how* someone communicates, not just *what* they talk about.

**Create:** `indexer/persona_builder.py`

**Schema** — add this table to `_init_metadata_db()` in `build_index.py`:

```sql
CREATE TABLE IF NOT EXISTS contact_personas (
    name TEXT PRIMARY KEY,
    style_summary TEXT,           -- 1-2 sentence prose: how they write
    avg_msg_length REAL,          -- avg chars per message
    emoji_frequency REAL,         -- emojis per 100 chars
    response_style TEXT,          -- "brief" | "medium" | "verbose"
    top_topics TEXT,              -- JSON array of {topic: str, score: float} pairs, top 8
    first_message_iso TEXT,
    last_message_iso TEXT,
    total_messages INTEGER,
    sample_hash TEXT              -- skip LLM rerun if unchanged
);
```

**Algorithm:**

1. For each contact present in `contact_summaries`, pull all their **individual messages** (not chunks) from the `messages` JSON column of `chunks` table where `is_from_me = false` and `sender == name`.
2. Compute style stats from raw message text:
   - `avg_msg_length`: mean character count per message
   - `emoji_frequency`: count emoji chars (use a unicode range check, no new dep needed) divided by total chars × 100
   - `response_style`: bucketed from `avg_msg_length` — `<60: brief`, `60-200: medium`, `>200: verbose`
3. Top topics:
   - Embed all their messages (use the existing `Embedder` from `embedder.py`)
   - K-means cluster into `min(8, n_messages // 20)` clusters (require at least 3 messages per cluster, drop singletons)
   - For each cluster, pick the message closest to the centroid as representative
   - Send the representatives to `gpt-4o-mini` with a prompt asking for a 2-3 word topic label per cluster
   - Score each topic by % of messages in that cluster
4. Style summary:
   - Sample 5 representative messages (across topic clusters, prefer recent)
   - Send to `gpt-4o-mini`: "Describe in ONE sentence how this person writes — their tone, length, vocabulary, humor, emoji use. Not what they say, how they say it. No preamble."
5. Skip pattern: same `sample_hash` mechanism as `contact_summaries.py` — hash the inputs, skip LLM call if unchanged.

**Wiring:**

- Add `--personas` flag to `build_index.py` argparse
- When `--personas` is passed, run after `--summaries`
- When `--summaries` is passed alone, do not auto-run personas (keep them independent)

**Cost target:** Under 5¢ total across all contacts using gpt-4o-mini. The sample_hash skip is what keeps this cheap on reruns.

**Acceptance:**

```bash
python indexer/build_index.py --sources imessage --update --summaries --personas
```

Should print a sample for 2-3 contacts in the format:

```
[brief, 142 msgs, 6 topics] Jerry Yan: writes in short bursts, often technical, rarely uses emoji
[verbose, 891 msgs, 8 topics] Sarah: long thoughtful messages with frequent humor and emoji, often venting or processing
```

---

## Task 2 — New Synthesis Routes

**Goal:** Detect three new query types and route them to specialized synthesis paths instead of the generic excerpt-based answer.

**Edit:** `api/search.py`

**Step 2a — Query classifier**

Add a function (before the `SearchEngine` class):

```python
def _classify_query_type(query: str, has_contact: bool) -> str:
    """Returns: 'style' | 'affinity' | 'temporal' | 'standard'
    
    Fast keyword matching only. No LLM. Runs before the search pipeline."""
```

Triggers (case-insensitive, after stripping punctuation):

- **style**: `"how does {X} talk"`, `"how does {X} write"`, `"how does {X} communicate"`, `"what's {X} like to talk to"`, `"what is {X} like"` — requires `has_contact = True`
- **affinity**: `"what does {X} care about"`, `"what topics does {X}"`, `"what is {X} into"`, `"what does {X} talk about"` — requires `has_contact = True`
- **temporal**: `"what has {X} been thinking about recently"`, `"how has {X} changed"`, `"what did {X} used to talk about"`, `"how is {X} different now"` — requires `has_contact = True`
- **standard**: everything else

The `has_contact` flag comes from whether `contact_filter` is non-empty after the existing `_resolve_contacts` + fallback extraction.

**Step 2b — Style synthesis**

In `SearchEngine.search()`, after `contact_filter` is resolved, check `_classify_query_type`. If `style` and `contact_filter` is non-empty:

1. Load the contact's persona from `contact_personas` (write a helper `_load_persona(name) -> dict | None`)
2. If no persona exists for any of the target contacts, fall through to standard pipeline
3. Build the answer directly from the persona — **no LLM call**:
   ```
   {style_summary}
   
   They mainly talk about: {top 4 topics joined with ·}.
   ```
4. For sources, fetch 3 of their most recent chunks where they spoke (use the existing rerank-by-contact-similarity path with a generic query embedding like their first topic label)
5. Return early with this answer + sources

**Step 2c — Affinity synthesis**

Similar pattern. Answer is generated from persona topics with a tiny gpt-4o-mini prose pass:

```
prompt: "Rewrite this as one natural sentence about what someone talks about most:
{name} top topics: {topic_list_with_scores}

Output: just the sentence, no preamble."
```

Sources: one representative chunk per top topic (3-4 sources).

**Step 2d — Temporal synthesis**

1. Pull all chunks for the target contact, sorted by date
2. Split: chunks within last 30 days vs prior
3. For each half: embed all messages, find the top 3 topic centroids (mini k-means or just nearest-neighbor density)
4. LLM call: "Recent topics: [A, B, C]. Earlier topics: [D, E, F]. In one sentence, describe the shift in what this person talks about. If there's no meaningful shift, say so plainly."
5. Sources: 3 recent + 3 older chunks, interleaved

**Acceptance:**

```bash
curl -s localhost:8000/search -X POST -H "Content-Type: application/json" \
  -d '{"query": "how does jerry talk"}' | jq '.answer'
curl -s localhost:8000/search -X POST -H "Content-Type: application/json" \
  -d '{"query": "what topics does sarah care about"}' | jq '.answer'
curl -s localhost:8000/search -X POST -H "Content-Type: application/json" \
  -d '{"query": "how has alex changed recently"}' | jq '.answer'
```

All three should return non-empty, non-generic answers when those contacts exist in the index.

---

## Task 3 — Cross-Contact Comparison

**Goal:** Queries like *"what do Sarah and Alex both think about the trip"* return a comparative answer.

**Edit:** `api/search.py`

**Detection:** When `contact_filter` has **2 or more** resolved canonical names AND the query contains `"both"`, `"and"`, `"vs"`, `"versus"`, or `"compare"`, branch into compare mode.

**Pipeline:**

1. For each target contact, run the existing rerank-by-contact-similarity path with the topic-only embed query (this gives N chunks per contact, contact-spoken lines only)
2. Interleave: [X1, Y1, X2, Y2, X3, Y3] up to `top_k`
3. Synthesis prompt (new constant `COMPARE_FOCUSED_PROMPT`):
   ```
   You are comparing what two or more people said about a topic, based on excerpts of THEIR OWN messages (the user's words are not shown).
   
   In 1-2 sentences:
   - Note where they agree and where they disagree on the topic
   - Use the speaker label that prefixes each line; only attribute what's prefixed with that person's name
   - If one of them barely spoke on the topic, say so plainly: "{name} didn't say much about this."
   
   No preamble. No quotation marks unless the quoted phrase is verbatim in the messages.
   ```
4. The existing `_validate_no_invented_quotes` check still applies — call it on the result

**Acceptance:** A two-contact query with `"both"` in it returns a comparative answer with both names mentioned.

---

## Task 4 — Auto-Update Script

**Create:** `scripts/update_index.sh`

```bash
#!/usr/bin/env bash
# Incremental update of the Semse index.
# To run nightly, add to crontab: 0 2 * * * /full/path/to/semse/scripts/update_index.sh
set -e
cd "$(dirname "$0")/.."
if [ -f indexer/.env ]; then
  export $(grep -v '^#' indexer/.env | xargs)
fi
python indexer/build_index.py --sources imessage mail --update --summaries --personas
```

`chmod +x scripts/update_index.sh` after creating it.

---

## Task 5 — Swift Query Router Hints (only if time)

**Edit:** `app-mac/Sources/Memory/QueryRouter.swift`

Add an optional `queryType` field to `QueryIntent` and `QueryIntentWire` matching the new types (`style`, `affinity`, `temporal`, `standard`). The backend already classifies in Task 2, so this is purely a latency win — the on-device model can pre-signal.

Add to the `routerInstructions` prompt block:

```
queryType: when the query is asking about HOW a person communicates → "style".
When asking what topics a person cares about → "affinity". When asking how a
person has changed over time → "temporal". Default → "standard".
```

In the backend `api/models.py` `QueryIntent` model, add `query_type: str = "standard"` and have `search.py` prefer this hint over the keyword classifier when present.

**Skip this task if any of Tasks 1-3 are still buggy** — Swift requires a rebuild and is the lowest-leverage piece tonight.

---

## Anti-goals for tonight

Do **not**:

- Modify `~/Library/Messages/chat.db` directly under any circumstances
- Refactor `chunker.py`, `parse_imessage.py`, or `parse_mail.py` — those are working
- Add a chat interface, auth, or any frontend framework
- Change embedding models or vector dimensions
- Touch the FAISS index format
- Delete or recreate `metadata.db` (this would lose the indexed data — `--update` is additive)
- Add web dependencies (no React Query, SWR, axios, etc.)

## Stop conditions

Stop the session and write a `MORNING.md` summary file in the repo root if any of these happen:

- Three consecutive task failures
- The build_index command fails to run at all (something is fundamentally broken)
- You discover that the schema assumptions in Task 1 don't hold (e.g. `messages` JSON column is structured differently than expected)

In `MORNING.md` write: what worked, what didn't, what I should look at first when I wake up.
