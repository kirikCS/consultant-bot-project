"""Сборка retrieval-индексов: FAISS (dense), BM25 (sparse), TF-IDF (char-n-gram) + parquet с payload каталога.

Запускается один раз при первом старте бота (или вручную при пересборке).
Fast-path: если FAISS+BM25+meta уже на диске, а TF-IDF отсутствует — собирает
только TF-IDF без повторного эмбеддинга 5,8K услуг.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import orjson
import pandas as pd
from rank_bm25 import BM25Okapi

from src.config import settings
from src.retrieval.embedder import EmbedderClient
from src.retrieval.tfidf_store import build_and_save as build_tfidf
from src.retrieval.tokenize import tokenize

log = logging.getLogger(__name__)


def _searchable_text(svc: dict) -> str:
    return (
        f"{svc.get('name', '')}. "
        f"Категория: {svc.get('category', '')}. "
        f"Группа: {svc.get('group', '')}. "
        f"Цена: {svc.get('price', '')} руб."
    )


def _load_services(path: str) -> list[dict]:
    raw = Path(path).read_bytes()
    parsed = orjson.loads(raw)
    if isinstance(parsed, list) and parsed and "json" in parsed[0]:
        items = parsed[0]["json"].get("data", [])
    elif isinstance(parsed, dict) and "data" in parsed:
        items = parsed["data"]
    elif isinstance(parsed, list):
        items = parsed
    else:
        raise RuntimeError("unrecognized med_services.json shape")
    return [s for s in items if isinstance(s, dict) and s.get("name")]


async def _embed_all(embedder: EmbedderClient, texts: list[str], batch: int = 16) -> np.ndarray:
    vectors: list[np.ndarray] = []
    total = len(texts)
    for start in range(0, total, batch):
        chunk = texts[start : start + batch]
        vecs = await embedder.embed_batch(chunk)
        vectors.append(vecs)
        log.info("embedded %d / %d", min(start + batch, total), total)
    return np.vstack(vectors).astype(np.float32)


def _build_faiss(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


_ARTIFACTS = ("faiss.index", "bm25.pkl", "meta.parquet", "tfidf.pkl")


def artifacts_exist() -> bool:
    d = Path(settings.indices_dir)
    return all((d / f).exists() for f in _ARTIFACTS)


async def build(force: bool = False) -> None:
    out = Path(settings.indices_dir)
    out.mkdir(parents=True, exist_ok=True)

    if artifacts_exist() and not force:
        log.info("Indices already present at %s, skipping build", out)
        return

    existing_core = (
        (out / "faiss.index").exists()
        and (out / "bm25.pkl").exists()
        and (out / "meta.parquet").exists()
    )
    if existing_core and not (out / "tfidf.pkl").exists() and not force:
        log.info("Core indices present; building only TF-IDF char-3-5 index")
        meta = pd.read_parquet(out / "meta.parquet")
        texts = meta["text"].astype(str).tolist()
        vocab, nnz = build_tfidf(texts, str(out))
        log.info("Wrote TF-IDF index: vocab=%d, nnz=%d", vocab, nnz)
        return

    services = _load_services(settings.med_services_path)
    log.info("Loaded %d services", len(services))

    texts = [_searchable_text(s) for s in services]

    embedder = EmbedderClient()
    try:
        vectors = await _embed_all(embedder, texts)
    finally:
        await embedder.aclose()

    index = _build_faiss(vectors)
    faiss.write_index(index, str(out / "faiss.index"))
    log.info("Wrote FAISS index: %d vectors, dim=%d", index.ntotal, vectors.shape[1])

    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(out / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Wrote BM25 index")

    meta = pd.DataFrame(
        {
            "row": list(range(len(services))),
            "name": [s.get("name", "") for s in services],
            "category": [s.get("category", "") for s in services],
            "group": [s.get("group", "") for s in services],
            "price": [s.get("price", "") for s in services],
            "id": [s.get("id", "") for s in services],
            "text": texts,
        }
    )
    meta.to_parquet(out / "meta.parquet", index=False)
    log.info("Wrote meta.parquet: %d rows", len(meta))

    vocab, nnz = build_tfidf(texts, str(out))
    log.info("Wrote TF-IDF index: vocab=%d, nnz=%d", vocab, nnz)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(build(force=os.environ.get("REBUILD_INDEX") == "1"))


if __name__ == "__main__":
    main()
