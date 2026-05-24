from __future__ import annotations


def rrf_merge(
    dense: list[tuple[int, float]],
    sparse: list[tuple[int, float]],
    *,
    k: int = 60,
    top: int = 10,
) -> list[int]:
    """Reciprocal Rank Fusion of two ranked lists; returns merged top-N row indices."""
    scores: dict[int, float] = {}
    for rank, (row, _) in enumerate(dense):
        scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank + 1)
    for rank, (row, _) in enumerate(sparse):
        scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank + 1)

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [row for row, _ in ordered[:top]]
