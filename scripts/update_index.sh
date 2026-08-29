#!/usr/bin/env bash
# Incremental update of the Semse index.
# To run nightly, add to crontab (output appends to indexer/data/update.log):
#   0 2 * * * /full/path/to/semse/scripts/update_index.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
# No shell-level .env sourcing: build_index.py loads api/.env and indexer/.env
# itself via python-dotenv, and `source` would execute arbitrary file content.
mkdir -p indexer/data
{
  echo "=== update_index $(date '+%Y-%m-%d %H:%M:%S') ==="
  .venv/bin/python indexer/build_index.py --sources imessage mail --update --summaries --personas
} >> indexer/data/update.log 2>&1
