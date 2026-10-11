"""
======================================================================
AI RESUME ANALYZER - EMBEDDING STORE (offline pipeline)
======================================================================

An on-disk cache of text embeddings with ONE vector per distinct text,
keyed by ``sha1(model name + text)``.

Step 7 needs every resume and job embedded to retrieve semantic
candidates; Step 9 needs the same vectors to compute pairwise similarity.
Without a shared cache each step would encode everything again (minutes
on a CPU for a real job dataset). Keying by text - not by row order or
row count - means a step that reads the same text in a different order,
or a subset of it, still gets cache hits, and a changed text is simply
re-encoded.

The live app keeps its own fingerprinted job-embedding cache
(``job_ranker.load_or_build_embeddings``); this store is for the offline
steps only and lives in ``data/cache/offline/``.
======================================================================
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STORE_DIR = PROJECT_ROOT / "data" / "cache" / "offline"

ENCODE_CHUNK = 2048


def text_key(model_name: str, text: str) -> str:

    digest = hashlib.sha1()

    digest.update(model_name.encode("utf-8", "ignore"))
    digest.update(b"\x1f")
    digest.update(str(text).encode("utf-8", "ignore"))

    return digest.hexdigest()


class EmbeddingStore:
    """``store.embed(texts, encode)`` -> float32 matrix, one row per text."""

    def __init__(
        self,
        model_name: str,
        directory: Optional[Path] = None,
    ) -> None:

        self.model_name = model_name

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)

        self.path = Path(directory or DEFAULT_STORE_DIR) / f"embeddings_{safe}.npz"

        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------

    def _load(self) -> tuple:
        """``(keys list, vectors)``; empty when absent or unreadable."""

        if not self.path.is_file():
            return [], np.empty((0, 0), dtype=np.float32)

        try:

            with np.load(self.path, allow_pickle=False) as stored:

                keys = [str(key) for key in stored["keys"]]
                vectors = np.asarray(stored["vectors"], dtype=np.float32)

            if vectors.ndim != 2 or vectors.shape[0] != len(keys):
                raise ValueError("keys and vectors do not line up")

            return keys, vectors

        except Exception as exc:  # corrupt store -> start again
            logger.warning("Ignoring unreadable embedding store: %s", exc)

            return [], np.empty((0, 0), dtype=np.float32)

    def _save(self, keys: List[str], vectors: np.ndarray) -> None:

        self.path.parent.mkdir(parents=True, exist_ok=True)

        handle, temporary = tempfile.mkstemp(
            dir=self.path.parent, suffix=".npz.tmp"
        )

        os.close(handle)

        try:

            with open(temporary, "wb") as file:
                np.savez(file, keys=np.asarray(keys), vectors=vectors)

            os.replace(temporary, self.path)

        finally:

            if os.path.exists(temporary):
                os.unlink(temporary)

    # ------------------------------------------------------------------

    def embed(
        self,
        texts: Sequence[str],
        encode: Callable[[List[str]], np.ndarray],
        *,
        batch_size: int = ENCODE_CHUNK,
        progress: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """
        Embeddings for ``texts`` (any order, duplicates allowed). Only
        texts not already stored are passed to ``encode``.
        """

        texts = [str(text) for text in texts]

        wanted = [text_key(self.model_name, text) for text in texts]

        keys, vectors = self._load()

        position = {key: row for row, key in enumerate(keys)}

        first_text = {}

        for key, text in zip(wanted, texts):

            if key not in position:
                first_text.setdefault(key, text)

        missing = list(first_text)

        self.hits += len(texts) - sum(1 for key in wanted if key in first_text)
        self.misses += len(missing)

        if missing:

            logger.info(
                "Encoding %d new text(s) (%d already stored)",
                len(missing),
                len(position),
            )

            fresh = []

            batch_size = max(1, int(batch_size))
            for start in range(0, len(missing), batch_size):

                chunk = [
                    first_text[key]
                    for key in missing[start:start + batch_size]
                ]

                encoded = np.asarray(encode(chunk), dtype=np.float32)

                if encoded.ndim != 2 or encoded.shape[0] != len(chunk):
                    raise ValueError(
                        f"encode() returned shape {encoded.shape} for "
                        f"{len(chunk)} texts."
                    )

                fresh.append(encoded)
                if progress:
                    progress(min(1.0, (start + len(chunk)) / max(1, len(missing))))

            fresh_matrix = np.vstack(fresh)

            if vectors.size and vectors.shape[1] != fresh_matrix.shape[1]:
                raise ValueError(
                    f"New embeddings have {fresh_matrix.shape[1]} dimensions "
                    f"but {self.path} holds {vectors.shape[1]}; the model "
                    f"behind '{self.model_name}' changed. Delete that file "
                    "and re-run."
                )

            vectors = (
                np.vstack([vectors, fresh_matrix])
                if vectors.size
                else fresh_matrix
            )

            for key in missing:
                position[key] = len(keys)
                keys.append(key)

            self._save(keys, vectors)

        if not texts:
            return np.empty((0, vectors.shape[1] if vectors.size else 0),
                            dtype=np.float32)

        return vectors[[position[key] for key in wanted]]
