from pathlib import Path

def get_root() -> Path:
    return Path(__file__).resolve().parents[1]
