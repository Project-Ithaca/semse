#!/usr/bin/env bash
# One-click Semse launcher. Double-click this file (or drag it to the Dock).
# Starts Ollama, the API, and the menu-bar app — skipping anything already up.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

echo "── Starting Semse ──"

if ! curl -s --max-time 2 http://localhost:11434/api/version >/dev/null; then
  echo "starting local LLM (ollama)…"
  nohup ollama serve >/dev/null 2>&1 &
  for _ in $(seq 1 20); do
    curl -s --max-time 1 http://localhost:11434/api/version >/dev/null && break
    sleep 0.5
  done
fi
echo "✓ LLM running"

if ! curl -s --max-time 2 http://localhost:8000/health >/dev/null; then
  echo "starting search API…"
  mkdir -p indexer/data
  nohup .venv/bin/uvicorn api.main:app --port 8000 >> indexer/data/api.log 2>&1 &
  for _ in $(seq 1 40); do
    curl -s --max-time 1 http://localhost:8000/health >/dev/null && break
    sleep 0.5
  done
fi
echo "✓ API running"

if ! pgrep -f "release/Memory" >/dev/null; then
  echo "starting menu-bar app…"
  nohup app-mac/.build/release/Memory >/dev/null 2>&1 &
  sleep 1
fi
echo "✓ App running"

echo ""
echo "Semse is ready — press Ctrl+Option+Space to search."
echo "(You can close this window.)"
