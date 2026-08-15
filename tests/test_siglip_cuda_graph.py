from contextlib import nullcontext
from types import SimpleNamespace

from siglip_cuda_graph import SiglipCudaGraphRunner


class _FakeTensor:
    shape = (1, 3, 224, 224)
    dtype = "float16"
    device = "cuda:1"
    is_cuda = True

    def __init__(self):
        self.copies = 0

    def copy_(self, _other):
        self.copies += 1
        return self


class _FakeStream:
    def wait_stream(self, _other):
        return None


class _FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class _FakeCuda:
    def __init__(self, *, fail_capture=False):
        self.fail_capture = fail_capture

    def is_available(self):
        return True

    def current_stream(self, _device):
        return _FakeStream()

    def Stream(self, *, device):
        assert device == "cuda:1"
        return _FakeStream()

    def stream(self, _stream):
        return nullcontext()

    def synchronize(self, _device):
        return None

    def CUDAGraph(self):
        return _FakeGraph()

    def graph(self, _graph):
        if self.fail_capture:
            raise RuntimeError("capture unavailable")
        return nullcontext()


class _FakeTorch:
    def __init__(self, *, fail_capture=False):
        self.cuda = _FakeCuda(fail_capture=fail_capture)

    def empty_like(self, _value):
        return _FakeTensor()

    def inference_mode(self):
        return nullcontext()


def test_fixed_batch_one_is_captured_once_and_replayed():
    runner = SiglipCudaGraphRunner(_FakeTorch(), warmup_steps=2)
    forward_calls = []

    def forward(**_kwargs):
        output = SimpleNamespace(call=len(forward_calls))
        forward_calls.append(output)
        return output

    inputs = {"pixel_values": _FakeTensor()}
    first, first_replayed = runner.run(
        forward,
        inputs,
        generation="generation-a",
        device="cuda:1",
    )
    second, second_replayed = runner.run(
        forward,
        inputs,
        generation="generation-a",
        device="cuda:1",
    )

    assert first_replayed is True
    assert second_replayed is True
    assert first is second
    assert len(forward_calls) == 3  # two warmups plus one captured call
    status = runner.status()
    assert status["state"] == "captured"
    assert status["counters"]["captures_total"] == 1
    assert status["counters"]["replays_total"] == 2


def test_failed_equivalence_rejects_graph_and_uses_eager_generation_fallback():
    runner = SiglipCudaGraphRunner(_FakeTorch(), warmup_steps=1)
    calls = []

    def forward(**_kwargs):
        calls.append(True)
        return len(calls)

    inputs = {"pixel_values": _FakeTensor()}
    output, replayed = runner.run(
        forward,
        inputs,
        generation="generation-a",
        device="cuda:1",
        validator=lambda _eager, _graph: {"ok": False, "cosine": 0.5},
    )

    assert replayed is False
    assert output == 3  # warmup, captured call, then eager fallback
    status = runner.status()
    assert status["state"] == "fallback"
    assert status["counters"]["failures_total"] == 1
    assert "failed eager equivalence" in status["last_error"]


def test_capture_failure_falls_back_once_per_generation():
    runner = SiglipCudaGraphRunner(
        _FakeTorch(fail_capture=True),
        warmup_steps=1,
    )
    calls = []

    def forward(**_kwargs):
        calls.append(True)
        return len(calls)

    inputs = {"pixel_values": _FakeTensor()}
    first, first_replayed = runner.run(
        forward,
        inputs,
        generation="generation-a",
        device="cuda:1",
    )
    second, second_replayed = runner.run(
        forward,
        inputs,
        generation="generation-a",
        device="cuda:1",
    )

    assert first_replayed is False
    assert second_replayed is False
    assert first == 2  # warmup, then eager fallback
    assert second == 3  # no second capture attempt in the failed generation
    status = runner.status()
    assert status["state"] == "fallback"
    assert status["counters"]["failures_total"] == 1
    assert status["counters"]["eager_total"] == 2
    assert status["fallback_reasons"]["generation_fallback"] == 1


def test_non_unit_batch_stays_on_eager_path():
    runner = SiglipCudaGraphRunner(_FakeTorch())
    pixels = _FakeTensor()
    pixels.shape = (2, 3, 224, 224)

    output, replayed = runner.run(
        lambda **_kwargs: "eager",
        {"pixel_values": pixels},
        generation="generation-a",
        device="cuda:1",
    )

    assert output == "eager"
    assert replayed is False
    status = runner.status()
    assert status["counters"]["eager_total"] == 1
    assert status["fallback_reasons"]["batch_not_one"] == 1
