#!/usr/bin/env bash
# One-click Semse launcher. Double-click this file (or drag it to the Dock).
# Semse.app boots its own backend (ollama + API); this just opens the app.
set -euo pipefail

if [ -d /Applications/Semse.app ]; then
  open /Applications/Semse.app
  echo "Semse launched — click the menu-bar icon or press Ctrl+Option+Space."
  echo "(You can close this window.)"
else
  echo "/Applications/Semse.app not found."
  echo "Build it first:  scripts/make_app.sh"
  exit 1
fi
