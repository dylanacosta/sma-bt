#!/bin/zsh

# Double-clickable launcher: activates venv and runs Streamlit app

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Create venv if missing
if [ ! -d "$SCRIPT_DIR/bin" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv .
fi

# Activate venv
source "$SCRIPT_DIR/bin/activate"

# Install dependencies if streamlit not present
if ! command -v streamlit >/dev/null 2>&1; then
    echo "[setup] Installing Python dependencies..."
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt
    else
        pip install streamlit pandas numpy yfinance
    fi
fi

echo "[run] Launching Streamlit app..."
echo "" | streamlit run app.py


