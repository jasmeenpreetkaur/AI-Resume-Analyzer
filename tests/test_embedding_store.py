from __future__ import annotations

import numpy as np
import pytest

from embedding_store import EmbeddingStore
import candidate_retrieval as cr


def deterministic_encoder(texts):
    return np.asarray(
        [[len(text), sum(text.encode("utf-8")) % 11 + 1, 1.0] for text in texts],
        dtype=np.float32,
    )


def test_store_encodes_unique_texts_once_and_reuses_across_order(tmp_path):
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return deterministic_encoder(texts)

    store = EmbeddingStore("encoder-v1", tmp_path)
    first = store.embed(["alpha", "beta", "alpha"], encode)
    assert first.shape == (3, 3)
    assert calls == [["alpha", "beta"]]

    second = store.embed(["beta", "alpha"], encode)
    assert calls == [["alpha", "beta"]]
    np.testing.assert_allclose(second, first[[1, 0]])
    assert store.path.is_file()


def test_store_cache_is_scoped_by_model_name(tmp_path):
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return deterministic_encoder(texts)

    EmbeddingStore("model-a", tmp_path).embed(["same text"], encode)
    EmbeddingStore("model-b", tmp_path).embed(["same text"], encode)
    assert len(calls) == 2


def test_shared_cache_wrapper_returns_normalized_vectors_and_reuses(tmp_path):
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return deterministic_encoder(texts)

    first = cr.encode_texts_cached(
        ["alpha", "beta", "alpha"], encode=encode,
        model_name="test-encoder", cache_dir=tmp_path,
    )
    second = cr.encode_texts_cached(
        ["beta", "alpha"], encode=encode,
        model_name="test-encoder", cache_dir=tmp_path,
    )

    np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.ones(3))
    np.testing.assert_allclose(second, first[[1, 0]])
    assert len(calls) == 1


def test_store_rejects_encoder_shape_mismatch(tmp_path):
    store = EmbeddingStore("bad", tmp_path)
    with pytest.raises(ValueError, match=r"encode\(\) returned shape"):
        store.embed(["one", "two"], lambda texts: np.ones((1, 3), dtype=np.float32))
