"""Fixed-shape CUDA Graph execution for the latency-sensitive SigLIP path.

The live probe/archive contract normally submits one 224px frame at a time.
Replaying that immutable vision graph avoids Python/CUDA launch overhead without
changing preprocessing, model weights, pooled embeddings, or patch outputs.
Unsupported shapes and any capture failure fall back to the ordinary eager
forward for the rest of the current model generation.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


@dataclass
class _CudaGraphEntry:
    generation: str
    signature: tuple[Any, ...]
    static_inputs: dict[str, Any]
    output: Any
    graph: Any


class SiglipCudaGraphRunner:
    """Replay a captured batch-one SigLIP vision forward when it is safe."""

    def __init__(
        self,
        torch_module: Any,
        *,
        enabled: bool = True,
        warmup_steps: int = 3,
    ) -> None:
        self._torch = torch_module
        self.enabled = bool(enabled)
        self.warmup_steps = max(1, min(10, int(warmup_steps)))
        self._lock = threading.RLock()
        self._entry: Optional[_CudaGraphEntry] = None
        self._failed_generation = ""
        self._state = "disabled" if not self.enabled else "idle"
        self._last_error: Optional[str] = None
        self._last_validation: dict[str, Any] = {}
        self._last_capture_ms = 0.0
        self._counters: Counter[str] = Counter()
        self._fallback_reasons: Counter[str] = Counter()

    def clear(self, generation: str = "") -> None:
        """Release graph-owned tensors when the encoder generation changes."""

        with self._lock:
            self._entry = None
            self._failed_generation = ""
            self._state = "disabled" if not self.enabled else "idle"
            self._last_error = None
            self._last_validation = {}
            self._counters["resets_total"] += 1
            if generation:
                self._counters["generations_total"] += 1

    def run(
        self,
        forward: Callable[..., Any],
        model_inputs: Mapping[str, Any],
        *,
        generation: str,
        device: str,
        validator: Optional[Callable[[Any, Any], Mapping[str, Any]]] = None,
    ) -> tuple[Any, bool]:
        """Return ``(vision_output, replayed)`` with eager fail-safe behavior."""

        with self._lock:
            reason, signature = self._eligibility(model_inputs, device=device)
            if reason is not None or signature is None:
                return self._eager(
                    forward,
                    model_inputs,
                    reason=reason or "unsupported",
                ), False

            entry = self._entry
            if (
                entry is not None
                and entry.generation == generation
                and entry.signature == signature
            ):
                try:
                    return self._replay(entry, model_inputs), True
                except BaseException as exc:
                    self._mark_failed(generation, exc, reason="replay_failed")
                    return self._eager(
                        forward,
                        model_inputs,
                        reason="replay_failed",
                    ), False

            if self._failed_generation == generation:
                return self._eager(
                    forward,
                    model_inputs,
                    reason="generation_fallback",
                ), False

            capture_started = time.perf_counter()
            try:
                entry, validation = self._capture(
                    forward,
                    model_inputs,
                    generation=generation,
                    signature=signature,
                    validator=validator,
                )
                self._entry = entry
                self._state = "captured"
                self._last_error = None
                self._last_validation = dict(validation or {})
                self._last_capture_ms = (
                    time.perf_counter() - capture_started
                ) * 1000.0
                self._counters["captures_total"] += 1
                return self._replay(entry, model_inputs), True
            except BaseException as exc:
                self._mark_failed(generation, exc, reason="capture_failed")
                return self._eager(
                    forward,
                    model_inputs,
                    reason="capture_failed",
                ), False

    def status(self) -> dict[str, Any]:
        with self._lock:
            entry = self._entry
            return {
                "enabled": self.enabled,
                "state": self._state,
                "captured": bool(entry is not None),
                "generation": entry.generation if entry is not None else None,
                "warmup_steps": self.warmup_steps,
                "last_capture_ms": round(self._last_capture_ms, 3),
                "last_error": self._last_error,
                "validation": dict(self._last_validation),
                "counters": dict(sorted(self._counters.items())),
                "fallback_reasons": dict(sorted(self._fallback_reasons.items())),
            }

    def _eligibility(
        self,
        model_inputs: Mapping[str, Any],
        *,
        device: str,
    ) -> tuple[Optional[str], Optional[tuple[Any, ...]]]:
        if not self.enabled:
            return "disabled", None
        cuda = getattr(self._torch, "cuda", None)
        if (
            cuda is None
            or not str(device or "").lower().startswith("cuda")
            or not bool(cuda.is_available())
        ):
            return "cuda_unavailable", None
        # The pinned 224px SigLIP2 processor emits only pixel_values. Keeping
        # the first production seam this narrow prevents accidentally graphing
        # a dynamic NaFlex/spatial-shape contract.
        if tuple(sorted(model_inputs)) != ("pixel_values",):
            return "dynamic_inputs", None
        pixels = model_inputs.get("pixel_values")
        shape = tuple(getattr(pixels, "shape", ()))
        if not bool(getattr(pixels, "is_cuda", False)):
            return "input_not_cuda", None
        if len(shape) != 4 or int(shape[0]) != 1:
            return "batch_not_one", None
        signature = (
            "pixel_values",
            shape,
            str(getattr(pixels, "dtype", "")),
            str(getattr(pixels, "device", "")),
        )
        return None, signature

    def _capture(
        self,
        forward: Callable[..., Any],
        model_inputs: Mapping[str, Any],
        *,
        generation: str,
        signature: tuple[Any, ...],
        validator: Optional[Callable[[Any, Any], Mapping[str, Any]]],
    ) -> tuple[_CudaGraphEntry, Mapping[str, Any]]:
        torch_module = self._torch
        pixels = model_inputs["pixel_values"]
        static_inputs = {"pixel_values": torch_module.empty_like(pixels)}
        static_inputs["pixel_values"].copy_(pixels)
        cuda = torch_module.cuda
        current_stream = cuda.current_stream(pixels.device)
        warmup_stream = cuda.Stream(device=pixels.device)
        warmup_stream.wait_stream(current_stream)
        with cuda.stream(warmup_stream), torch_module.inference_mode():
            output = None
            for _ in range(self.warmup_steps):
                output = forward(**static_inputs, return_dict=True)
        current_stream.wait_stream(warmup_stream)
        cuda.synchronize(pixels.device)

        graph = cuda.CUDAGraph()
        with cuda.graph(graph), torch_module.inference_mode():
            graph_output = forward(**static_inputs, return_dict=True)
        graph.replay()
        cuda.synchronize(pixels.device)
        validation: Mapping[str, Any] = {}
        if validator is not None:
            validation = dict(validator(output, graph_output))
            if not bool(validation.get("ok")):
                raise RuntimeError("captured SigLIP output failed eager equivalence")
        return _CudaGraphEntry(
            generation=str(generation or ""),
            signature=signature,
            static_inputs=static_inputs,
            output=graph_output,
            graph=graph,
        ), validation

    def _replay(
        self,
        entry: _CudaGraphEntry,
        model_inputs: Mapping[str, Any],
    ) -> Any:
        entry.static_inputs["pixel_values"].copy_(model_inputs["pixel_values"])
        entry.graph.replay()
        self._counters["replays_total"] += 1
        return entry.output

    def _eager(
        self,
        forward: Callable[..., Any],
        model_inputs: Mapping[str, Any],
        *,
        reason: str,
    ) -> Any:
        self._counters["eager_total"] += 1
        self._fallback_reasons[str(reason or "unspecified")] += 1
        return forward(**model_inputs, return_dict=True)

    def _mark_failed(
        self,
        generation: str,
        exc: BaseException,
        *,
        reason: str,
    ) -> None:
        self._entry = None
        self._failed_generation = str(generation or "")
        self._state = "fallback"
        self._last_error = f"{type(exc).__name__}: {exc}"[:500]
        self._counters["failures_total"] += 1
        self._fallback_reasons[reason] += 1
