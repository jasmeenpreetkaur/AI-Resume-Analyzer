"""Offline stand-in for sentence-transformers, used ONLY by tests/test_pipeline.py
(placed on PYTHONPATH of the subprocesses it launches).

Hashes word unigrams/bigrams into 384 dimensions and L2-normalises, so
cosine similarity tracks word overlap. Deterministic; no network.
"""
import hashlib
import re

import numpy as np


class SentenceTransformer:

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.dim = 384

    def _vec(self, text):
        words = re.findall(r"[a-z0-9+#]+", str(text).lower())
        grams = words + [a + "_" + b for a, b in zip(words, words[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 20) % 2 else -0.4
        return v

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=False, **kw):
        if isinstance(texts, str):
            texts = [texts]
        out = np.vstack([self._vec(t) for t in texts]).astype(np.float32)
        if normalize_embeddings:
            n = np.linalg.norm(out, axis=1, keepdims=True)
            n[n == 0] = 1
            out = out / n
        return out
