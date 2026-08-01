import threading
import time

import numpy as np
import pytest

from embedding_batcher import (
    EmbeddingBatchOutput,
    EmbeddingBatchError,
    EmbeddingBatchRejected,
    EmbeddingBatchTimeout,
    ImageEmbeddingBatcher,
)


def test_batch_metadata_is_returned_with_each_embedding():
    batcher = ImageEmbeddingBatcher(
        lambda images: EmbeddingBatchOutput(
            np.ones((len(images), 2), dtype=np.float32),
            {"fingerprint": "epoch-a", "dimension": 2},
        ),
        max_wait_ms=0,
    )

    vector, metadata = batcher.embed_one_with_metadata("frame")

    assert batcher.stop()
    np.testing.assert_allclose(vector, [1.0, 1.0])
    assert metadata == {"fingerprint": "epoch-a", "dimension": 2}


def test_eight_concurrent_channels_are_encoded_in_one_microbatch():
    calls = []
    gate = threading.Barrier(9)

    def embed_many(images):
        calls.append(list(images))
        return np.asarray([[float(value), 1.0] for value in images], dtype=np.float32)

    batcher = ImageEmbeddingBatcher(
        embed_many,
        max_batch_size=8,
        max_wait_ms=250,
        queue_capacity=16,
    )
    results = {}

    def submit(channel):
        gate.wait()
        results[channel] = batcher.embed_one(channel)

    threads = [
        threading.Thread(target=submit, args=(channel,))
        for channel in range(8)
    ]
    for thread in threads:
        thread.start()
    gate.wait()
    for thread in threads:
        thread.join(timeout=2)
    batcher.stop()

    assert all(not thread.is_alive() for thread in threads)
    assert len(calls) == 1
    assert sorted(calls[0]) == list(range(8))
    assert sorted(results) == list(range(8))
    for channel, embedding in results.items():
        np.testing.assert_allclose(embedding, [float(channel), 1.0])
    status = batcher.status()
    assert status["counters"]["completed_total"] == 8
    assert status["largest_batch_size"] == 8


def test_single_channel_is_not_lost_when_batch_wait_expires():
    calls = []

    def embed_many(images):
        calls.append(list(images))
        return np.ones((len(images), 3), dtype=np.float32)

    batcher = ImageEmbeddingBatcher(
        embed_many,
        max_batch_size=8,
        max_wait_ms=15,
    )
    started = time.monotonic()
    result = batcher.embed_one("one")
    elapsed = time.monotonic() - started
    batcher.stop()

    np.testing.assert_allclose(result, [1.0, 1.0, 1.0])
    assert calls == [["one"]]
    assert elapsed >= 0.01


def test_batch_failure_is_returned_to_every_caller():
    gate = threading.Barrier(3)
    errors = []

    def embed_many(_images):
        raise RuntimeError("encoder unavailable")

    batcher = ImageEmbeddingBatcher(
        embed_many,
        max_batch_size=2,
        max_wait_ms=100,
    )

    def submit(value):
        gate.wait()
        try:
            batcher.embed_one(value)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=submit, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    gate.wait()
    for thread in threads:
        thread.join(timeout=2)
    status = batcher.status()
    batcher.stop()

    assert len(errors) == 2
    assert all(isinstance(error, EmbeddingBatchError) for error in errors)
    assert status["counters"]["failed_total"] == 2
    assert "encoder unavailable" in status["last_error"]


def test_queue_rejection_and_request_timeout_are_explicit():
    release = threading.Event()

    def embed_many(images):
        release.wait(timeout=1)
        return np.ones((len(images), 2), dtype=np.float32)

    batcher = ImageEmbeddingBatcher(
        embed_many,
        max_batch_size=1,
        max_wait_ms=0,
        queue_capacity=1,
        request_timeout_sec=0.02,
    )

    with pytest.raises(EmbeddingBatchTimeout):
        batcher.embed_one("slow")

    release.set()
    assert batcher.drain(timeout_sec=1)
    batcher.stop()
    with pytest.raises(EmbeddingBatchRejected):
        batcher.embed_one("after-stop")
