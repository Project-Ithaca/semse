# Semse — CLAUDE.md

## Vision

A Spotlight-style semantic search over your entire personal communication history
(iMessage, Apple Mail, image attachments). One search bar, natural language queries,
synthesized answer + cited source cards in ~2 seconds. Fully local: embeddings,
FAISS search, and LLM synthesis all run on-device. No data leaves the machine.

## Architecture

```
PHASE 1 — Indexing (run once, then incremental via --update)
  indexer/build_index.py → parse iMessage/Mail/attachments → chunk → embed
  (sentence-transformers, local) → FAISS index + metadata.db
  Optional post-passes: --summaries (contact relationship summaries),
  --personas (per-contact communication-style personas). Both use the local LLM.

PHASE 2 — Runtime
  FastAPI server (127.0.0.1:8000) — embeds query, searches FAISS + FTS5,
  routes special query types (style/affinity/temporal/compare), synthesizes
  answers via a local LLM (Ollama, OpenAI-compatible endpoint)
  Swift menu-bar app (app-mac/, macOS 26+) — the Spotlight UI. Global hotkey
  Ctrl+Option+Space. On-device FoundationModels pre-classifies query intent
  (topic, contacts, sources, query_type) and sends it as a hint.
```

## Tech stack (do not substitute)

**Backend / indexer (Python 3.11, venv at `.venv/`):**
- `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) for text, `clip-ViT-B-32` for images
- `faiss-cpu` — FlatIP under 100k chunks, IVFFlat above
- `sqlite3` stdlib — metadata store (+ FTS5), and for reading chat.db / Envelope Index
- `fastapi` + `uvicorn`
- `openai` SDK pointed at a **local** OpenAI-compatible endpoint (Ollama) — see LLM section
- `scikit-learn` — K-means for topic clustering
- `pytest` — tests in `tests/`

**App (Swift, `app-mac/`):** SwiftPM executable, macOS 26.0+, no external deps.
AppKit + SwiftUI + FoundationModels + Carbon hotkey.

## Local LLM

All synthesis/persona/summary calls go through an OpenAI-compatible client
configured by env vars (read at call time):

- `SEMSE_LLM_BASE_URL` — default `http://localhost:11434/v1` (Ollama)
- `SEMSE_LLM_MODEL` — default `qwen2.5:14b`
- Escape hatch: `SEMSE_LLM_PREFER_OPENAI=1` + `OPENAI_API_KEY` routes to OpenAI.

Setup: `brew install ollama && ollama pull qwen2.5:14b && ollama serve`.
The API degrades gracefully when the LLM is down: sources still return, answer is empty.

## Key runtime behavior

- `POST /search` `{query, top_k (1-50), intent?}` → `{answer, sources, query_ms}`
- Query routing in `api/search.py`: the Swift-supplied `intent.query_type` hint wins;
  otherwise `_classify_query_type` keyword matching. Routes:
  - `style` ("how does X talk") → persona-based answer, no LLM call
  - `affinity` ("what does X care about") → persona topics + one small LLM call
  - `temporal` ("how has X changed") → recent-vs-older topic shift via K-means + LLM
  - compare (2+ contacts + both/vs/versus/compare) → interleaved per-contact excerpts
  - `standard` → dense FAISS + FTS5 + optional CLIP image search, RRF fusion, synthesis
- Quote validation: `_validate_no_invented_quotes` strips LLM-invented quotes.
- `/attachment/{id}` and `/contact-photo/{key}` reject any request carrying a
  `Sec-Fetch-Site` header (browsers) — only the native client may read images.
- The server boots degraded (health reports the reason) when `indexer/data/` is missing.

## Conventions

- `contact_names` in metadata.db is JSON with `ensure_ascii=False`; match contacts with
  the escaped full-quoted-name LIKE pattern via `persona_builder.escape_like` + `ESCAPE '\'`.
- All date strings are naive-UTC ISO (no `+00:00` suffix); comparisons are lexicographic.
- LLM failures must never be cached as success (personas blank the sample_hash on failure).
- Progress bars stay off the API request path (`embed_batch(..., show_progress_bar=False)`).

## Strict DO NOT rules

- Do not write to `~/Library/Messages/chat.db` — copy it first, read only
- Do not add auth, a chat interface, routing, or any web frontend — the UI is the Swift app
- Do not change embedding models or vector dimensions
- Do not delete or recreate `metadata.db` (`--update` is additive)
- Do not add cloud LLM calls by default — local-first is a product decision
- Do not source `.env` files from shell scripts; Python loads them via python-dotenv

## Commands

```bash
# Full index build (needs Full Disk Access for the host app)
.venv/bin/python indexer/build_index.py --sources imessage mail --summaries --personas

# Incremental (also what scripts/update_index.sh runs nightly)
.venv/bin/python indexer/build_index.py --sources imessage mail --update --summaries --personas

# API
.venv/bin/uvicorn api.main:app --port 8000

# App
cd app-mac && swift run   # menu-bar icon; Ctrl+Option+Space opens the panel

# Tests
.venv/bin/python -m pytest tests/ -q
```
