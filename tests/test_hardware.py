from __future__ import annotations

import json
import unittest

from wormctx.hardware import (
    GIB,
    MAX_MATRIX_SIZE,
    HardwareQualificationError,
    create_accelerator_receipt,
    detect_hardware,
    evaluate_accelerator,
    run_bounded_matmul_benchmark,
)


class _FakePlatform:
    @staticmethod
    def system() -> str:
        return "Linux"

    @staticmethod
    def python_version() -> str:
        return "3.11.9"

    @staticmethod
    def python_implementation() -> str:
        return "CPython"

    @staticmethod
    def node() -> str:
        raise AssertionError("machine identity must not be queried")


class _FakeProperties:
    name = "NVIDIA GeForce RTX 3090"
    total_memory = 24 * GIB


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _FakeTensor:
    def __init__(self, value: float) -> None:
        self.value = value

    def __matmul__(self, other: object) -> _FakeTensor:
        assert isinstance(other, _FakeTensor)
        return _FakeTensor(self.value + other.value)

    def __getitem__(self, key: object) -> _FakeScalar:
        del key
        return _FakeScalar(self.value)


class _FakeCuda:
    def __init__(self, *, available: bool) -> None:
        self.available = available
        self.synchronize_calls = 0
        self.reset_calls = 0
        self.manual_seed_calls: list[int] = []

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1 if self.available else 0

    def get_device_properties(self, device_index: int) -> _FakeProperties:
        assert device_index == 0
        return _FakeProperties()

    def get_device_capability(self, device_index: int) -> tuple[int, int]:
        assert device_index == 0
        return 8, 6

    def synchronize(self, device_index: int) -> None:
        assert device_index == 0
        self.synchronize_calls += 1

    def reset_peak_memory_stats(self, device_index: int) -> None:
        assert device_index == 0
        self.reset_calls += 1

    def max_memory_allocated(self, device_index: int) -> int:
        assert device_index == 0
        return 96 * 1024**2

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)


class _FakeVersion:
    cuda = "12.4"


class _FakeTorch:
    __version__ = "2.6.0"
    version = _FakeVersion()
    float32 = object()

    def __init__(self, *, cuda_available: bool) -> None:
        self.cuda = _FakeCuda(available=cuda_available)
        self.manual_seed_calls: list[int] = []
        self.randn_calls: list[tuple[tuple[int, int], str, object]] = []

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)

    def randn(
        self,
        shape: tuple[int, int],
        *,
        device: str,
        dtype: object,
    ) -> _FakeTensor:
        self.randn_calls.append((shape, device, dtype))
        return _FakeTensor(float(len(self.randn_calls)))


class _StepTimer:
    def __init__(self) -> None:
        self.values = iter((10.0, 10.25))

    def __call__(self) -> float:
        return next(self.values)


class HardwareTests(unittest.TestCase):
    def test_detects_sanitized_3090_class_profile(self) -> None:
        fake_torch = _FakeTorch(cuda_available=True)
        profile = detect_hardware(
            torch_module=fake_torch,
            platform_module=_FakePlatform,
        )

        self.assertEqual(
            profile["runtime"],
            {
                "python_version": "3.11.9",
                "python_implementation": "CPython",
                "os_family": "Linux",
            },
        )
        self.assertEqual(profile["framework"]["cuda_build_version"], "12.4")
        self.assertEqual(profile["accelerator"]["name"], "NVIDIA GeForce RTX 3090")
        self.assertEqual(profile["accelerator"]["compute_capability"], {"major": 8, "minor": 6})
        self.assertEqual(profile["accelerator"]["total_vram_bytes"], 24 * GIB)
        json.dumps(profile, allow_nan=False)

    def test_24_gib_gpu_meets_3090_class_requirement(self) -> None:
        profile = detect_hardware(
            torch_module=_FakeTorch(cuda_available=True),
            platform_module=_FakePlatform,
        )
        qualification = evaluate_accelerator(profile, minimum_vram_gib=24.0)

        self.assertTrue(qualification["accelerator_requirement_met"])
        self.assertTrue(qualification["execution_permitted"])
        self.assertEqual(qualification["execution_backend"], "cuda")

    def test_cpu_fallback_is_never_implicit(self) -> None:
        profile = detect_hardware(
            torch_module=_FakeTorch(cuda_available=False),
            platform_module=_FakePlatform,
        )
        rejected = evaluate_accelerator(profile, minimum_vram_gib=24.0)
        permitted = evaluate_accelerator(
            profile,
            minimum_vram_gib=24.0,
            allow_cpu_fallback=True,
        )

        self.assertEqual(rejected["status"], "rejected")
        self.assertFalse(rejected["execution_permitted"])
        self.assertIsNone(rejected["execution_backend"])
        self.assertEqual(permitted["status"], "cpu_fallback")
        self.assertTrue(permitted["execution_permitted"])
        self.assertEqual(permitted["execution_backend"], "cpu")
        self.assertFalse(permitted["accelerator_requirement_met"])

    def test_gpu_benchmark_is_bounded_seeded_synchronized_and_measured(self) -> None:
        fake_torch = _FakeTorch(cuda_available=True)
        result = run_bounded_matmul_benchmark(
            torch_module=fake_torch,
            matrix_size=128,
            warmup_iterations=1,
            timed_iterations=2,
            seed=7,
            timer=_StepTimer(),
        )

        self.assertEqual(fake_torch.manual_seed_calls, [7])
        self.assertEqual(fake_torch.cuda.manual_seed_calls, [7])
        self.assertEqual(len(fake_torch.randn_calls), 2)
        self.assertEqual(fake_torch.randn_calls[0][0], (128, 128))
        self.assertEqual(fake_torch.randn_calls[0][1], "cuda:0")
        self.assertEqual(fake_torch.cuda.reset_calls, 1)
        self.assertEqual(fake_torch.cuda.synchronize_calls, 3)
        self.assertEqual(result["elapsed_seconds"], 0.25)
        self.assertEqual(result["milliseconds_per_iteration"], 125.0)
        self.assertEqual(result["peak_allocated_vram_bytes"], 96 * 1024**2)
        self.assertEqual(result["result_sample"], 3.0)

    def test_cpu_benchmark_requires_explicit_permission(self) -> None:
        fake_torch = _FakeTorch(cuda_available=False)
        with self.assertRaisesRegex(
            HardwareQualificationError, "CPU fallback"
        ):
            run_bounded_matmul_benchmark(
                torch_module=fake_torch,
                matrix_size=32,
                warmup_iterations=0,
                timed_iterations=1,
                timer=_StepTimer(),
            )

        result = run_bounded_matmul_benchmark(
            torch_module=fake_torch,
            matrix_size=32,
            warmup_iterations=0,
            timed_iterations=1,
            allow_cpu_fallback=True,
            timer=_StepTimer(),
        )
        self.assertEqual(result["backend"], "cpu")
        self.assertIsNone(result["peak_allocated_vram_bytes"])

    def test_benchmark_limits_are_enforced_before_allocation(self) -> None:
        fake_torch = _FakeTorch(cuda_available=True)
        with self.assertRaisesRegex(ValueError, "matrix_size"):
            run_bounded_matmul_benchmark(
                torch_module=fake_torch,
                matrix_size=MAX_MATRIX_SIZE + 1,
            )
        self.assertEqual(fake_torch.randn_calls, [])

    def test_receipt_rejects_benchmark_without_permitted_backend(self) -> None:
        receipt = create_accelerator_receipt(
            torch_module=_FakeTorch(cuda_available=False),
            platform_module=_FakePlatform,
            include_benchmark=True,
        )

        self.assertEqual(receipt["qualification"]["status"], "rejected")
        self.assertEqual(
            receipt["benchmark"],
            {"status": "not_run", "reason": "qualification_rejected"},
        )
        serialized = json.dumps(receipt, allow_nan=False, sort_keys=True)
        for forbidden in (
            "hostname",
            "username",
            "ip_address",
            "working_directory",
        ):
            self.assertNotIn(forbidden, serialized.casefold())

    def test_missing_pytorch_is_reported_without_raw_error_text(self) -> None:
        profile = detect_hardware(
            torch_module=None,
            platform_module=_FakePlatform,
        )

        self.assertFalse(profile["framework"]["available"])
        self.assertEqual(profile["framework"]["status"], "pytorch_not_available")
        self.assertFalse(profile["accelerator"]["available"])


if __name__ == "__main__":
    unittest.main()
