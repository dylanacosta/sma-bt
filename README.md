# SMA Strategy App

A Streamlit app for exploring a Simple Moving Average (SMA) trading strategy. Coded with LLMs.

## Prerequisites
- Python 3.10+

## Setup
```bash
python3 -m venv .
source ./bin/activate
pip install -r requirements.txt
```

## Run
- Quick start (script):
```bash
chmod +x ./run_app.sh
./run_app.sh
```
- Double-clickable (macOS):
  - Double-click `launch.command` in Finder. First time: right-click → Open.

## Notes
- App entrypoint is `app.py`.
- `run_app.sh` and `launch.command` are path-agnostic; they compute their directory at runtime and do not contain user-specific paths.
