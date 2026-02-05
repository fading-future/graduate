from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os
import pandas as pd


def compute_porosity(volume, pore_value: int = 1) -> float:
    # volume assumed binary
    return float((volume == pore_value).mean())


@dataclass
class PorosityLookup:
    file_col: str
    porosity_col: str
    index_cols: Optional[Tuple[str, str, str]]
    table: Dict

    def get(self, file_path: str, idx: Optional[Tuple[int, int, int]] = None) -> Optional[float]:
        key = os.path.basename(file_path)
        if self.index_cols and idx is not None:
            z, y, x = idx
            return self.table.get((key, z, y, x))
        return self.table.get(key)


def _detect_columns(df: pd.DataFrame):
    file_candidates = ["file", "filename", "path", "npy", "sample", "name"]
    porosity_candidates = ["porosity", "phi", "bulk_porosity", "bulk_por", "por"]
    index_candidates = [
        ("z", "y", "x"),
        ("i", "j", "k"),
        ("z_idx", "y_idx", "x_idx"),
        ("iz", "iy", "ix"),
    ]

    file_col = next((c for c in file_candidates if c in df.columns), None)
    porosity_col = next((c for c in porosity_candidates if c in df.columns), None)
    index_cols = next((t for t in index_candidates if all(c in df.columns for c in t)), None)
    return file_col, porosity_col, index_cols


def load_porosity_csv(csv_path: str) -> PorosityLookup:
    df = pd.read_csv(csv_path)
    file_col, porosity_col, index_cols = _detect_columns(df)
    if file_col is None or porosity_col is None:
        raise ValueError(
            f"Cannot detect file/porosity columns in {csv_path}. "
            f"Columns found: {df.columns.tolist()}"
        )

    table = {}
    if index_cols:
        for _, row in df.iterrows():
            key = os.path.basename(str(row[file_col]))
            z, y, x = (int(row[index_cols[0]]), int(row[index_cols[1]]), int(row[index_cols[2]]))
            table[(key, z, y, x)] = float(row[porosity_col])
    else:
        for _, row in df.iterrows():
            key = os.path.basename(str(row[file_col]))
            table[key] = float(row[porosity_col])

    return PorosityLookup(file_col=file_col, porosity_col=porosity_col, index_cols=index_cols, table=table)
