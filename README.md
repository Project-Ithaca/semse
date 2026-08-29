<img width="801" height="658" alt="image" src="https://github.com/user-attachments/assets/2014fda5-4f55-437e-a06f-b159da178996" />


with Semse you can easily search your conversations with natural language! no more silly "More results will be shown once messages finishes indexing," having to remmber the exact words of a conversation, or opening search across eight different apps.

ask questions about a contact—"what events did my girlfriend mention wanting to attend this month?"—or specify filters using language like "a year or so ago."

then get accurate results:

<img width="749" height="170" alt="image" src="https://github.com/user-attachments/assets/6781cf3d-e1f8-48c9-a463-41bacc6f1cb3" />

everything runs locally: embeddings, search, and the answer-writing LLM. nothing leaves your Mac.

## Setup

Requires macOS 26+ (for the app), Python 3.11, and Full Disk Access granted to your terminal (to read the iMessage/Mail databases).

```bash
# 1. Python env
python3.11 -m venv .venv
.venv/bin/pip install -r indexer/requirements.txt

# 2. Local LLM (answers are written by a model running on your Mac)
brew install ollama
ollama pull qwen2.5:14b
ollama serve   # keep running (or: brew services start ollama)

# 3. Build the index (~20-30 min first time)
.venv/bin/python indexer/build_index.py --sources imessage mail --summaries --personas

# 4. Build the app once
cd app-mac && swift build -c release && cd ..
```

## Run it

Double-click **`scripts/start_semse.command`** (drag it to your Dock for one-click access).
It starts the local LLM, the API, and the menu-bar app — then press **Ctrl+Option+Space** to search.
`scripts/stop_semse.command` shuts everything down.

Nightly index refresh: add `scripts/update_index.sh` to cron (see the comment in the script).

## Try these

- `how does [name] talk` — communication-style summary (instant, no LLM)
- `what does [name] care about` — their top topics
- `how has [name] changed recently` — recent vs. older topic shift
- `what do [name1] and [name2] both think about [topic]` — comparison
- `has anyone recommended a ramen place` — regular semantic search

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```
