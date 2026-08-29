# Agent Instructions

Read and follow [`CLAUDE.md`](./CLAUDE.md) as the canonical project instructions. Translate Claude-specific tool names to Codex equivalents. This bridge exists so Claude Code and Codex share one project context file.

## Ongoing

Updated: 2026-08-29T23:55:00Z by claude session

Done:
- Full functional pass: env (.venv py3.11), 30+ bug fixes across indexer/api/app, real index built — 68,449 chunks across 9 sources (imessage 57k, browsing 8.5k, mail 2k, whatsapp, notes, calendar, reminders, calls; images optional) (05a7a14…d4c8a9a)
- Fully local LLM: Ollama qwen2.5:14b default after 7b-vs-14b A/B, per-request keep_alive=5m for memory control; 234 personas regenerated with improved topic-label prompt (bb19b2f)
- Semse.app in /Applications: self-boots ollama+API, click menu-bar icon or Ctrl+Option+Space, movable panel with saved position; Spotlight-parity quick actions (app launcher, calculator, file search via NSMetadataQuery, url/web rows) (9c82988)
- Search quality: recency boost (halflife 270d), adaptive temporal shift, named affinity answers, source inference from phrasing ("in my notes" → notes), source-restricted FTS + deep dense pool, ambiguous-contact resolution by volume, schedule-query fixes (typo-tolerant "tomorrow", overlap date windows, relative-date prompt guard) (901c718)
- api/search.py split → prompts.py + query_analysis.py; docs (CLAUDE.md/README) rewritten to match reality (a4815e4)
- 112 pytest tests green; scripts/{start,stop}_semse.command + make_app.sh; nightly update_index.sh covers all sources
- Test expansion (39afae6): 116 → 180 pytest. `tests/fixture_index.py` builds a synthetic metadata.db + FAISS index in a tmpdir with a deterministic hash embedder and a fake LLM (no model load, no Ollama, no dependency on the real index); `tests/conftest.py` exposes it as the `make_engine` fixture. New `tests/test_search_engine.py` (45 SearchEngine integration tests: standard/style/affinity/temporal/compare gates, source + contact filters, date-window overlap, image route, RRF/merge helpers) and `tests/test_api_routes.py` (19 HTTP tests over /health, /search, /attachment, /contact-photo)
- Swift test target added (`app-mac/Tests/MemoryTests`, run with `swift test`) — 21 tests covering Recency
- "Past" collapsing: results older than 14 days (dated by date_end) collapse behind one clickable "Past · N older results" row; click, or Return while it is selected, expands. The top `Recency.alwaysVisibleCount` (3) results stay visible regardless of age — without that pin the panel showed zero cards on almost every real query. Files: `app-mac/Sources/Memory/Recency.swift`, `Views/PastDisclosureRow.swift`; SearchView holds the recent/past split in state and the flat selection index now includes the toggle row

In flight: nothing — all lanes closed.

Blocked: nothing.

Next:
1. `/Applications/Semse.app` has NOT been rebuilt with the Past change — run `bash scripts/make_app.sh` (it re-signs the bundle, which may reset its Full Disk Access grant).
2. The Past row has not been verified visually in the running panel — only by unit test and by bucketing live /search responses by age.
3. `formatDate()` in `SourceCard.swift` parses naive-UTC stamps in the LOCAL timezone (`Recency.swift` parses them correctly as UTC), so displayed dates can be off by the UTC offset.
4. Refinements parked: exclude subscribed holiday calendars from indexing (watermark clamp in _read_cutoffs is a workaround, build_index.py:213); persona "person names"-style generic labels; affinity/compare prompts could name speakers more consistently; Chromium history only records last-visit-per-URL; Notes edits create stale duplicate chunks until full rebuild.
5. Possible sources later: Photos library, Voice Memos (needs local transcription), Stickies.
6. Known env facts: FDA is granted per claude-code version bundle (re-grant after Claude updates); bash-guard hook blocks any command text containing ".env".
