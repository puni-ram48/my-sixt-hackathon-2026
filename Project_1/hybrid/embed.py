"""Stage 2B: local sentence embeddings (fastembed + BAAI/bge-small-en-v1.5, ONNX on CPU).

Why this model: 384 dimensions, ~67 MB (quantized ONNX from Hugging Face: Qdrant/bge-small-en-v1.5-onnx-Q), about 1,000 short texts per second on a laptop CPU, no torch.
The project's LLM endpoint has no /embeddings route, so embeddings must be local.

If fastembed is not installed, get_embedder() returns None and the pipeline falls back to BM25 only.
Clause embeddings are cached on disk per document (data/cache/emb/), so re-runs are instant.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading

import numpy as np

from solution.data import DATA

log = logging.getLogger("hybrid.embed")
MODEL = os.environ.get("HYBRID_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CACHE = DATA / "cache" / "emb"


class Embedder:
    def __init__(self, model: str = MODEL):
        from fastembed import TextEmbedding  # imported here so the package is optional

        self.name = model
        self._model = TextEmbedding(model)
        self._queries: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()  # one ONNX session shared by worker threads

    @staticmethod
    def _norm(m: np.ndarray) -> np.ndarray:
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    def passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        key = hashlib.sha1((self.name + "\x00" + "\x01".join(texts)).encode("utf-8")).hexdigest()
        path = CACHE / f"{key}.npy"
        if path.exists():
            return np.load(path)
        with self._lock:
            m = self._norm(np.array(list(self._model.passage_embed(texts)), dtype=np.float32))
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, m)
        return m

    def queries(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._queries]
            if missing:
                for t, v in zip(missing, self._model.query_embed(missing)):
                    self._queries[t] = self._norm(np.asarray(v, dtype=np.float32)[None])[0]
            return np.stack([self._queries[t] for t in texts])


_embedder: Embedder | None = None
_tried = False


_init_lock = threading.Lock()


def get_embedder() -> Embedder | None:
    global _embedder, _tried
    with _init_lock:  # worker threads must not see "tried" before loading has finished
        if not _tried:
            try:
                _embedder = Embedder()
            except Exception as e:  # not installed, or model download failed
                log.warning("embeddings unavailable, falling back to BM25 only: %s", e)
            _tried = True
    return _embedder
