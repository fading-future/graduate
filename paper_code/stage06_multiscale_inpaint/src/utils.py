import os
from pathlib import Path
import csv
from typing import Dict


def get_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_porosity_map(csv_path: str) -> Dict[str, float]:
    por_map: Dict[str, float] = {}
    if not csv_path or not os.path.exists(csv_path):
        return por_map
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            fname = row[0].strip()
            try:
                por = float(row[2])
                por_map[fname] = por
            except ValueError:
                continue
    return por_map
