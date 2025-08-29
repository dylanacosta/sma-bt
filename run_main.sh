#!/bin/zsh

# Auto-activate venv and run console script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/bin/activate" || { echo "Virtualenv not found. Run: python3 -m venv ."; exit 1; }
cd "$SCRIPT_DIR" || exit 1
python main.py



