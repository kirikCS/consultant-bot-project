"""Reciprocal Rank Fusion: объединение нескольких ранжированных списков ретривера в один."""
from __future__ import annotations


def rrf_merge(
    *ranked_lists: list[tuple[int, float]],
    k: int = 60,
    top: int = 10,
) -> list[int]:
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (row, _) in enumerate(lst):
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank + 1)

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [row for row, _ in ordered[:top]]
