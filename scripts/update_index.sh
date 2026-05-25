#!/usr/bin/env bash
# Incremental update of the Semse index.
# To run nightly, add to crontab: 0 2 * * * /full/path/to/semse/scripts/update_index.sh
set -e
cd "$(dirname "$0")/.."
if [ -f indexer/.env ]; then
  export $(grep -v '^#' indexer/.env | xargs)
fi
python indexer/build_index.py --sources imessage mail --update --summaries --personas
