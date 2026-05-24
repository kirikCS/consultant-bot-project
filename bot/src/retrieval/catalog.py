from __future__ import annotations

from pathlib import Path

import pandas as pd


class Catalog:
    """In-memory lookup table for service payloads, keyed by FAISS/BM25 row index."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.set_index("row", drop=False)

    @classmethod
    def load(cls, indices_dir: str) -> "Catalog":
        return cls(pd.read_parquet(Path(indices_dir) / "meta.parquet"))

    def get(self, row: int) -> dict:
        rec = self._df.loc[row]
        return {
            "row": int(rec["row"]),
            "name": str(rec["name"]),
            "category": str(rec["category"]),
            "group": str(rec["group"]),
            "price": str(rec["price"]),
            "id": str(rec["id"]),
            "text": str(rec["text"]),
        }

    def many(self, rows: list[int]) -> list[dict]:
        return [self.get(r) for r in rows]
