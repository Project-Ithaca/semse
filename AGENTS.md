# Agent Instructions

Read and follow [`CLAUDE.md`](./CLAUDE.md) as the canonical project instructions. Translate Claude-specific tool names to Codex equivalents. This bridge exists so Claude Code and Codex share one project context file.

## Ongoing

Updated: 2026-08-29T22:30:00Z by claude session

Done:
- Full functional pass: env (.venv py3.11), 30+ bug fixes across indexer/api/app, real index built — 68,449 chunks across 9 sources (imessage 57k, browsing 8.5k, mail 2k, whatsapp, notes, calendar, reminders, calls; images optional) (05a7a14…d4c8a9a)
- Fully local LLM: Ollama qwen2.5:14b default after 7b-vs-14b A/B, per-request keep_alive=5m for memory control; 234 personas regenerated with improved topic-label prompt (bb19b2f)
- Semse.app in /Applications: self-boots ollama+API, click menu-bar icon or Ctrl+Option+Space, movable panel with saved position; Spotlight-parity quick actions (app launcher, calculator, file search via NSMetadataQuery, url/web rows) (9c82988)
- Search quality: recency boost (halflife 270d), adaptive temporal shift, named affinity answers, source inference from phrasing ("in my notes" → notes), source-restricted FTS + deep dense pool, ambiguous-contact resolution by volume, schedule-query fixes (typo-tolerant "tomorrow", overlap date windows, relative-date prompt guard) (901c718)
- api/search.py split → prompts.py + query_analysis.py; docs (CLAUDE.md/README) rewritten to match reality (a4815e4)
- 112 pytest tests green; scripts/{start,stop}_semse.command + make_app.sh; nightly update_index.sh covers all sources

In flight: nothing — all lanes closed.

Blocked: nothing.

Next (planned for an Opus session — test cases + refinement):
1. Broaden tests: SearchEngine integration tests against a small fixture index (build synthetic metadata.db+faiss in a tmpdir; see scratch pattern from 2026-08-29 session), route-level tests for style/affinity/temporal/compare gates, date-window overlap edge cases.
2. Refinements parked: exclude subscribed holiday calendars from indexing (watermark clamp in _read_cutoffs is a workaround, build_index.py:213); persona "person names"-style generic labels; affinity/compare prompts could name speakers more consistently; Chromium history only records last-visit-per-URL; Notes edits create stale duplicate chunks until full rebuild.
3. Possible sources later: Photos library, Voice Memos (needs local transcription), Stickies.
4. Known env facts: FDA is granted per claude-code version bundle (re-grant after Claude updates); bash-guard hook blocks any command text containing ".env".
