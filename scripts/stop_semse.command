#!/usr/bin/env bash
# Stops everything start_semse.command started.
pkill -f "release/Memory" 2>/dev/null && echo "✓ app stopped" || echo "app was not running"
pkill -f "uvicorn api.main" 2>/dev/null && echo "✓ API stopped" || echo "API was not running"
pkill -f "ollama serve" 2>/dev/null && echo "✓ LLM stopped" || echo "LLM was not running"
