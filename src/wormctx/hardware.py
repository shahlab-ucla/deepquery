"""Portable, privacy-preserving accelerator qualification.

The functions in this module deliberately avoid hostnames, user names, network
addresses, environment variables, filesystem paths, and subprocess-based GPU
inspection.  PyTorch is the sole accelerator probe so the same code can run in
local, container, batch, and interactive environments.
"""

from __future__ import annotations

import importlib
import json
import math
import platform
import re
import time
from typing import Any, Literal


GIB = 1024**3
RTX_3090_CLASS_VRAM_GIB = 24.0
MAX_MATRIX_SIZE = 4096
MAX_WARMUP_ITERATIONS = 20
MAX_TIMED_ITERATIONS = 100

_AUTO_TORCH = object()
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]+")


class HardwareQualificationError(RuntimeError):
    """Raised when a requested benchmark has no permitted execution backend."""


def _safe_label(value: object, *, limit: int = 160) -> str | None:
    if value is None:
        return None
    label = _CONTROL_CHARACTERS.sub(" ", str(value)).strip()
    return label[:limit] or None


def _load_torch(torch_module: Any) -> tuple[Any | None, str | None]:
    if torch_module is not _AUTO_TORCH:
        if torch_module is None:
            return None, "pytorch_not_available"
        return torch_module, None
    try:
        return importlib.import_module("torch"), None
    except ModuleNotFoundError:
        return None, "pytorch_not_installed"
    except Exception:
        # Import failures can contain machine-specific library paths.  Return a
        # stable code instead of serializing the exception text.
        return None, "pytorch_import_failed"


def _runtime_profile(platform_module: Any) -> dict[str, str]:
    """Collect only portable runtime fields, never machine identity fields."""
    raw_system = _safe_label(platform_module.system(), limit=32) or "Unknown"
    normalized = {
        "darwin": "macOS",
        "linux": "Linux",
        "windows": "Windows",
    }.get(raw_system.casefold(), "Other")
    return {
        "python_version": _safe_label(platform_module.python_version(), limit=32)
        or "unknown",
        "python_implementation": _safe_label(
            platform_module.python_implementation(), limit=32
        )
        or "unknown",
        "os_family": normalized,
    }


def detect_hardware(
    *,
    torch_module: Any = _AUTO_TORCH,
    platform_module: Any = platform,
    device_index: int = 0,
) -> dict[str, Any]:
    """Return a sanitized, JSON-serializable runtime and accelerator profile."""
    if not isinstance(device_index, int) or isinstance(device_index, bool):
        raise TypeError("device_index must be an integer")
    if device_index < 0:
        raise ValueError("device_index must be non-negative")

    torch, import_status = _load_torch(torch_module)
    profile: dict[str, Any] = {
        "runtime": _runtime_profile(platform_module),
        "framework": {
            "name": "pytorch",
            "available": torch is not None,
            "version": None,
            "cuda_build_version": None,
            "status": import_status or "available",
        },
        "accelerator": {
            "backend": None,
            "available": False,
            "device_index": None,
            "device_count": 0,
            "name": None,
            "compute_capability": None,
            "total_vram_bytes": None,
            "total_vram_gib": None,
        },
    }
    if torch is None:
        return _assert_json_serializable(profile)

    profile["framework"]["version"] = _safe_label(
        getattr(torch, "__version__", None), limit=64
    )
    torch_version = getattr(torch, "version", None)
    profile["framework"]["cuda_build_version"] = _safe_label(
        getattr(torch_version, "cuda", None), limit=64
    )
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        profile["framework"]["status"] = "cuda_not_available"
        return _assert_json_serializable(profile)

    device_count = int(cuda.device_count())
    profile["accelerator"]["device_count"] = max(device_count, 0)
    if device_index >= device_count:
        profile["framework"]["status"] = "cuda_device_index_out_of_range"
        return _assert_json_serializable(profile)

    properties = cuda.get_device_properties(device_index)
    capability = cuda.get_device_capability(device_index)
    if len(capability) != 2:
        raise HardwareQualificationError("invalid CUDA compute capability")
    total_vram_bytes = int(properties.total_memory)
    accelerator = profile["accelerator"]
    accelerator.update(
        {
            "backend": "cuda",
            "available": True,
            "device_index": device_index,
            "device_count": device_count,
            "name": _safe_label(properties.name),
            "compute_capability": {
                "major": int(capability[0]),
                "minor": int(capability[1]),
            },
            "total_vram_bytes": total_vram_bytes,
            "total_vram_gib": round(total_vram_bytes / GIB, 3),
        }
    )
    profile["framework"]["status"] = "cuda_available"
    return _assert_json_serializable(profile)


def evaluate_accelerator(
    profile: dict[str, Any],
    *,
    minimum_vram_gib: float = RTX_3090_CLASS_VRAM_GIB,
    allow_cpu_fallback: bool = False,
) -> dict[str, Any]:
    """Evaluate a generic VRAM requirement and select a permitted backend.

    A CPU backend is selected only when ``allow_cpu_fallback`` is explicitly
    true.  The result distinguishes satisfying the accelerator requirement from
    merely permitting a CPU fallback.
    """
    minimum = float(minimum_vram_gib)
    if not math.isfinite(minimum) or minimum < 0:
        raise ValueError("minimum_vram_gib must be finite and non-negative")
    minimum_bytes = math.ceil(minimum * GIB)

    accelerator = profile.get("accelerator", {})
    available = accelerator.get("backend") == "cuda" and bool(
        accelerator.get("available")
    )
    total_vram = accelerator.get("total_vram_bytes")
    meets_requirement = (
        available
        and isinstance(total_vram, int)
        and not isinstance(total_vram, bool)
        and total_vram >= minimum_bytes
    )

    if meets_requirement:
        status = "qualified"
        backend: Literal["cuda", "cpu"] | None = "cuda"
        reason = "cuda_vram_requirement_met"
    elif allow_cpu_fallback:
        status = "cpu_fallback"
        backend = "cpu"
        reason = (
            "cuda_vram_requirement_not_met"
            if available
            else "cuda_not_available"
        )
    else:
        status = "rejected"
        backend = None
        reason = (
            "cuda_vram_requirement_not_met"
            if available
            else "cuda_not_available"
        )

    return _assert_json_serializable(
        {
            "status": status,
            "execution_permitted": backend is not None,
            "execution_backend": backend,
            "accelerator_requirement_met": meets_requirement,
            "minimum_vram_bytes": minimum_bytes,
            "minimum_vram_gib": minimum,
            "cpu_fallback_permitted": bool(allow_cpu_fallback),
            "reason": reason,
        }
    )


def _bounded_integer(
    name: str,
    value: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def run_bounded_matmul_benchmark(
    *,
    torch_module: Any = _AUTO_TORCH,
    backend: Literal["auto", "cuda", "cpu"] = "auto",
    device_index: int = 0,
    matrix_size: int = 1024,
    warmup_iterations: int = 2,
    timed_iterations: int = 5,
    seed: int = 2025,
    allow_cpu_fallback: bool = False,
    timer: Any = time.perf_counter,
) -> dict[str, Any]:
    """Run a deterministic, resource-bounded float32 matrix workload.

    Synchronization occurs immediately before and after the timed region.  CUDA
    peak allocation is reset after warmup and recorded after the timed region.
    Timing is observational; the random workload is fixed by ``seed``.
    """
    matrix_size = _bounded_integer(
        "matrix_size", matrix_size, minimum=2, maximum=MAX_MATRIX_SIZE
    )
    warmup_iterations = _bounded_integer(
        "warmup_iterations",
        warmup_iterations,
        minimum=0,
        maximum=MAX_WARMUP_ITERATIONS,
    )
    timed_iterations = _bounded_integer(
        "timed_iterations",
        timed_iterations,
        minimum=1,
        maximum=MAX_TIMED_ITERATIONS,
    )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if backend not in {"auto", "cuda", "cpu"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'cpu'")

    torch, _ = _load_torch(torch_module)
    if torch is None:
        raise HardwareQualificationError("PyTorch is required for the benchmark")
    cuda_available = bool(torch.cuda.is_available())

    requested_backend = backend
    if backend == "auto":
        backend = "cuda" if cuda_available else "cpu"
    if backend == "cuda" and not cuda_available:
        raise HardwareQualificationError("CUDA was requested but is not available")
    if backend == "cpu" and not allow_cpu_fallback:
        raise HardwareQualificationError("CPU fallback was not explicitly permitted")

    if not isinstance(device_index, int) or isinstance(device_index, bool):
        raise TypeError("device_index must be an integer")
    if device_index < 0:
        raise ValueError("device_index must be non-negative")
    if backend == "cuda" and device_index >= int(torch.cuda.device_count()):
        raise HardwareQualificationError("CUDA device index is out of range")

    torch.manual_seed(seed)
    if backend == "cuda" and hasattr(torch.cuda, "manual_seed_all"):
        torch.cuda.manual_seed_all(seed)
    device = f"cuda:{device_index}" if backend == "cuda" else "cpu"
    left = torch.randn(
        (matrix_size, matrix_size), device=device, dtype=torch.float32
    )
    right = torch.randn(
        (matrix_size, matrix_size), device=device, dtype=torch.float32
    )

    result = None
    for _ in range(warmup_iterations):
        result = left @ right
    if backend == "cuda":
        torch.cuda.synchronize(device_index)
        torch.cuda.reset_peak_memory_stats(device_index)

    if backend == "cuda":
        torch.cuda.synchronize(device_index)
    started = float(timer())
    for _ in range(timed_iterations):
        result = left @ right
    if backend == "cuda":
        torch.cuda.synchronize(device_index)
    elapsed_seconds = max(float(timer()) - started, 0.0)

    if result is None:  # pragma: no cover - timed_iterations is validated positive
        raise HardwareQualificationError("benchmark produced no result")
    checksum = float(result[0, 0].item())
    if not math.isfinite(checksum):
        checksum_value: float | None = None
    else:
        checksum_value = checksum
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device_index))
        if backend == "cuda"
        else None
    )

    return _assert_json_serializable(
        {
            "status": "completed",
            "backend": backend,
            "requested_backend": requested_backend,
            "device_index": device_index if backend == "cuda" else None,
            "dtype": "float32",
            "matrix_size": matrix_size,
            "warmup_iterations": warmup_iterations,
            "timed_iterations": timed_iterations,
            "deterministic_seed": seed,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "milliseconds_per_iteration": round(
                (elapsed_seconds * 1000) / timed_iterations, 6
            ),
            "peak_allocated_vram_bytes": peak_bytes,
            "result_sample": checksum_value,
        }
    )


def create_accelerator_receipt(
    *,
    minimum_vram_gib: float = RTX_3090_CLASS_VRAM_GIB,
    allow_cpu_fallback: bool = False,
    include_benchmark: bool = False,
    torch_module: Any = _AUTO_TORCH,
    platform_module: Any = platform,
    device_index: int = 0,
    matrix_size: int = 1024,
    warmup_iterations: int = 2,
    timed_iterations: int = 5,
    seed: int = 2025,
    timer: Any = time.perf_counter,
) -> dict[str, Any]:
    """Build a sanitized qualification receipt and optionally benchmark it."""
    profile = detect_hardware(
        torch_module=torch_module,
        platform_module=platform_module,
        device_index=device_index,
    )
    qualification = evaluate_accelerator(
        profile,
        minimum_vram_gib=minimum_vram_gib,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    receipt: dict[str, Any] = {
        "schema_version": "deepquery.accelerator-qualification.v1",
        **profile,
        "qualification": qualification,
        "benchmark": None,
    }
    if include_benchmark:
        execution_backend = qualification["execution_backend"]
        if execution_backend is None:
            receipt["benchmark"] = {
                "status": "not_run",
                "reason": "qualification_rejected",
            }
        else:
            receipt["benchmark"] = run_bounded_matmul_benchmark(
                torch_module=torch_module,
                backend=execution_backend,
                device_index=device_index,
                matrix_size=matrix_size,
                warmup_iterations=warmup_iterations,
                timed_iterations=timed_iterations,
                seed=seed,
                allow_cpu_fallback=allow_cpu_fallback,
                timer=timer,
            )
    return _assert_json_serializable(receipt)


def _assert_json_serializable(payload: dict[str, Any]) -> dict[str, Any]:
    """Fail closed if a future field cannot be encoded as strict JSON."""
    json.dumps(payload, allow_nan=False, sort_keys=True)
    return payload
