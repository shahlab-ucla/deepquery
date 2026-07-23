"""Fail-closed, provider-neutral execution contracts for portable experiments.

The public manifest is intentionally data-only.  It names an explicitly trusted
Python module and an argument vector; it cannot contain a shell command.  All
filesystem arguments are declared separately, byte-bound, and expanded beneath
caller-supplied roots during preflight.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BINDING_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
_EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$")
_MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_PLACEHOLDER_PATTERN = re.compile(
    r"^\{(?P<kind>config|input|output):(?P<name>[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*)\}$"
)


class ExecutionError(RuntimeError):
    """Base class for execution-contract failures."""


class PreflightError(ExecutionError):
    """Raised when a manifest cannot be executed safely on the current host."""


class ReceiptExistsError(ExecutionError):
    """Raised rather than overwriting a prior execution receipt."""


class ExecutionFailed(ExecutionError):
    """Raised after a failed subprocess or failed output verification."""

    def __init__(self, message: str, outcome: ExecutionOutcome | None = None) -> None:
        super().__init__(message)
        self.outcome = outcome


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_sha256(value: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("sha256 must be a lowercase, 64-character hexadecimal digest")
    return value


def _validate_binding_name(value: str) -> str:
    if not _BINDING_NAME_PATTERN.fullmatch(value):
        raise ValueError("binding names must be descriptive lowercase identifiers")
    return value


def _validate_relative_path(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("paths must be nonempty relative POSIX paths")
    candidate = PurePosixPath(value)
    windows_candidate = PureWindowsPath(value)
    if (
        candidate.is_absolute()
        or windows_candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or any(":" in part for part in candidate.parts)
    ):
        raise ValueError("paths must be normalized relative POSIX paths without traversal")
    return value


class FileBinding(_StrictModel):
    """A named, immutable byte binding beneath either config_root or input_root."""

    name: str
    path: str
    sha256: str

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _validate_binding_name(value)

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        return _validate_sha256(value)


class OutputBinding(_StrictModel):
    """A named file or directory that the experiment is required to create."""

    name: str
    path: str
    kind: Literal["file", "directory"] = "file"

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _validate_binding_name(value)

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        return _validate_relative_path(value)


class PythonModuleEntrypoint(_StrictModel):
    """A Python ``-m`` target and an ordered, non-shell argument vector."""

    module: str
    argv: tuple[str, ...] = ()

    @field_validator("module")
    @classmethod
    def valid_module(cls, value: str) -> str:
        if not _MODULE_PATTERN.fullmatch(value):
            raise ValueError("entrypoint must be a dotted Python module name")
        return value

    @field_validator("argv")
    @classmethod
    def valid_argv(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) > 512:
            raise ValueError("entrypoint argv exceeds the 512-item safety limit")
        for value in values:
            if not value or "\x00" in value or "\r" in value or "\n" in value:
                raise ValueError("argv items must be nonempty and contain no control separators")
            if _PLACEHOLDER_PATTERN.fullmatch(value):
                continue
            if "{" in value or "}" in value:
                raise ValueError("unknown argv placeholder")
            posix = PurePosixPath(value)
            windows = PureWindowsPath(value)
            if posix.is_absolute() or windows.is_absolute():
                raise ValueError("absolute argv paths are forbidden; use a declared placeholder")
            if ".." in posix.parts or ".." in windows.parts:
                raise ValueError("argv path traversal is forbidden")
        return values


class ResourceRequest(_StrictModel):
    """Generic resources required before an experiment may start."""

    cpu_cores: int = Field(default=1, ge=1, le=4096)
    ram_gib: float = Field(default=0.25, gt=0, le=65536)
    gpus: int = Field(default=0, ge=0, le=64)
    min_vram_gib: float = Field(default=0.0, ge=0, le=1024)

    @model_validator(mode="after")
    def consistent_gpu_request(self) -> ResourceRequest:
        if self.gpus == 0 and self.min_vram_gib != 0:
            raise ValueError("min_vram_gib must be zero when no GPUs are requested")
        if self.gpus > 0 and self.min_vram_gib <= 0:
            raise ValueError("GPU requests must state a positive per-GPU minimum VRAM")
        return self


class ResourceAvailability(_StrictModel):
    """Host measurements supplied by discovery or a scheduler integration."""

    cpu_cores: int = Field(ge=0)
    ram_gib: float = Field(ge=0)
    gpu_vram_gib: tuple[float, ...] = ()


class ExecutionManifest(_StrictModel):
    """Portable, deterministic description of one experiment invocation."""

    execution_manifest_schema_version: Literal["1.0"] = "1.0"
    experiment_id: str
    entrypoint: PythonModuleEntrypoint
    configs: tuple[FileBinding, ...] = ()
    inputs: tuple[FileBinding, ...] = ()
    outputs: tuple[OutputBinding, ...] = Field(min_length=1)
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    required_environment: tuple[str, ...] = ()
    timeout_seconds: int = Field(default=86400, ge=1, le=604800)

    @field_validator("experiment_id")
    @classmethod
    def descriptive_experiment_id(cls, value: str) -> str:
        if not _EXPERIMENT_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "experiment_id must be a descriptive, multi-token lowercase kebab-case name"
            )
        if re.fullmatch(r"(?:es|s)-?\d+[a-z]?", value):
            raise ValueError("opaque alphanumeric experiment identifiers are forbidden")
        return value

    @field_validator("required_environment")
    @classmethod
    def valid_environment_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("required environment-variable names must be unique")
        for value in values:
            if not _ENVIRONMENT_NAME_PATTERN.fullmatch(value):
                raise ValueError("environment requirements must contain names, never assignments")
        return values

    @model_validator(mode="after")
    def internally_consistent(self) -> ExecutionManifest:
        for label, bindings in (
            ("config", self.configs),
            ("input", self.inputs),
            ("output", self.outputs),
        ):
            names = [item.name for item in bindings]
            paths = [item.path for item in bindings]
            if len(names) != len(set(names)) or len(paths) != len(set(paths)):
                raise ValueError(f"{label} binding names and paths must be unique")
        if any(PurePosixPath(item.path).parts[0] == "receipts" for item in self.outputs):
            raise ValueError("the receipts namespace is reserved for execution authority")
        output_paths = [PurePosixPath(item.path) for item in self.outputs]
        for index, left in enumerate(output_paths):
            for right in output_paths[index + 1 :]:
                if left in right.parents or right in left.parents:
                    raise ValueError("declared output paths must not overlap")

        available = {
            "config": {item.name for item in self.configs},
            "input": {item.name for item in self.inputs},
            "output": {item.name for item in self.outputs},
        }
        for argument in self.entrypoint.argv:
            match = _PLACEHOLDER_PATTERN.fullmatch(argument)
            if match and match.group("name") not in available[match.group("kind")]:
                raise ValueError(f"argv refers to an undeclared {match.group('kind')} binding")
        return self

    def canonical_bytes(self) -> bytes:
        """Return a stable encoding with unordered declarations normalized by identity."""

        payload = self.model_dump(mode="json")
        for field in ("configs", "inputs", "outputs"):
            payload[field] = sorted(
                payload[field],
                key=lambda item: (item["name"], item["path"]),
            )
        payload["required_environment"] = sorted(payload["required_environment"])
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class ExecutionRoots(_StrictModel):
    """Explicit host roots; these values never enter an execution receipt."""

    config_root: Path
    input_root: Path
    output_root: Path

    @field_validator("config_root", "input_root", "output_root")
    @classmethod
    def absolute_roots_only(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("execution roots must be explicit absolute paths")
        return value


class VerifiedInput(_StrictModel):
    kind: Literal["config", "input"]
    name: str
    path: str
    sha256: str

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _validate_binding_name(value)

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        return _validate_sha256(value)


class ProducedOutput(_StrictModel):
    name: str
    path: str
    kind: Literal["file", "directory"]
    sha256: str
    byte_size: int = Field(ge=0)
    file_count: int = Field(ge=1)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _validate_binding_name(value)

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        return _validate_sha256(value)


class ExecutionReceipt(_StrictModel):
    """Sanitized record containing no host identity, roots, argv, or environment values."""

    execution_receipt_schema_version: Literal["1.0"] = "1.0"
    manifest_sha256: str
    experiment_id: str
    entrypoint_module: str
    status: Literal["dry_run", "succeeded", "failed"]
    started_at: datetime
    finished_at: datetime
    elapsed_seconds: float = Field(ge=0)
    exit_code: int | None
    resources: ResourceRequest
    required_environment: tuple[str, ...]
    verified_inputs: tuple[VerifiedInput, ...]
    declared_outputs: tuple[OutputBinding, ...]
    produced_outputs: tuple[ProducedOutput, ...]

    @field_validator("manifest_sha256")
    @classmethod
    def valid_manifest_sha256(cls, value: str) -> str:
        return _validate_sha256(value)

    @model_validator(mode="after")
    def valid_status(self) -> ExecutionReceipt:
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("execution receipt timestamps must include timezones")
        if self.status == "dry_run" and self.exit_code is not None:
            raise ValueError("dry-run receipts cannot report an exit code")
        if self.status == "succeeded" and self.exit_code != 0:
            raise ValueError("successful receipts must report exit code zero")
        if self.status != "succeeded" and self.produced_outputs:
            raise ValueError("only successful receipts can bind produced outputs")
        return self


@dataclass(frozen=True)
class PreparedExecution:
    """In-memory preflight result.  Absolute paths are intentionally not serializable."""

    manifest_sha256: str
    command: tuple[str, ...]
    verified_inputs: tuple[VerifiedInput, ...]
    output_paths: tuple[tuple[OutputBinding, Path], ...]
    receipt_path: Path
    resolved_output_root: Path


@dataclass(frozen=True)
class ExecutionOutcome:
    """Minimal caller result; the referenced on-disk receipt is the authority."""

    status: Literal["dry_run", "succeeded", "failed"]
    manifest_sha256: str
    returncode: int | None
    receipt_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_closure(path: Path) -> tuple[str, int, int]:
    records: list[dict[str, str | int]] = []
    total_bytes = 0
    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise PreflightError("declared output directory contains a symbolic link")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise PreflightError("declared output directory contains a special entry")
        relative = candidate.relative_to(path).as_posix()
        size = candidate.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": relative,
                "sha256": _sha256_file(candidate),
                "byte_size": size,
            }
        )
    if not records:
        raise PreflightError("declared output directory contains no regular files")
    payload = json.dumps(
        records,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest(), total_bytes, len(records)


def _existing_directory(root: Path, label: str) -> Path:
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise PreflightError(f"{label} does not exist") from exc
    if not resolved.is_dir():
        raise PreflightError(f"{label} is not a directory")
    return resolved


def _resolve_beneath(
    root: Path,
    relative_path: str,
    *,
    must_exist: bool,
    regular_file: bool = False,
) -> Path:
    pure_path = PurePosixPath(_validate_relative_path(relative_path))
    candidate = root.joinpath(*pure_path.parts)
    cursor = root
    for part in pure_path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PreflightError(f"symlinked path components are forbidden: {relative_path}")
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise PreflightError(f"declared path does not exist: {relative_path}") from exc
    if not resolved.is_relative_to(root):
        raise PreflightError(f"declared path escapes its root: {relative_path}")
    if regular_file and not resolved.is_file():
        raise PreflightError(f"declared input is not a regular file: {relative_path}")
    return resolved


def _discover_ram_gib() -> float:
    if os.name == "nt":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.length = ctypes.sizeof(_MemoryStatus)
        try:
            success = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        except (AttributeError, OSError):
            return 0.0
        return status.available_physical / (1024**3) if success else 0.0

    sysconf = getattr(os, "sysconf", None)
    if not callable(sysconf):
        return 0.0
    try:
        page_size = sysconf("SC_PAGE_SIZE")
        page_count = sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return 0.0
    return float(page_size * page_count) / (1024**3)


def _discover_gpu_vram_gib() -> tuple[float, ...]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            shell=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if result.returncode != 0:
        return ()
    values: list[float] = []
    for line in result.stdout.splitlines():
        try:
            values.append(float(line.strip()) / 1024)
        except ValueError:
            return ()
    return tuple(values)


def discover_resources() -> ResourceAvailability:
    """Measure resources without depending on a provider-specific scheduler API."""

    return ResourceAvailability(
        cpu_cores=os.cpu_count() or 0,
        ram_gib=_discover_ram_gib(),
        gpu_vram_gib=_discover_gpu_vram_gib(),
    )


def _verify_resource_request(
    request: ResourceRequest,
    availability: ResourceAvailability,
) -> None:
    if availability.cpu_cores < request.cpu_cores:
        raise PreflightError("requested CPU cores are unavailable")
    if availability.ram_gib < request.ram_gib:
        raise PreflightError("requested RAM is unavailable or could not be measured")
    qualifying_gpus = sum(
        available >= request.min_vram_gib for available in availability.gpu_vram_gib
    )
    if qualifying_gpus < request.gpus:
        raise PreflightError("requested GPU count or per-GPU VRAM is unavailable")


def _verified_binding(
    kind: Literal["config", "input"],
    binding: FileBinding,
    path: Path,
) -> VerifiedInput:
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != binding.sha256:
        raise PreflightError(
            f"{kind} binding failed sha256 verification: {binding.name}"
        )
    return VerifiedInput(
        kind=kind,
        name=binding.name,
        path=binding.path,
        sha256=observed_sha256,
    )


def _receipt_relative_path(manifest: ExecutionManifest, *, dry_run: bool) -> str:
    suffix = "dry-run" if dry_run else "execution"
    return (
        f"receipts/{manifest.experiment_id}/"
        f"{manifest.canonical_sha256()}.{suffix}.json"
    )


def preflight_execution(
    manifest: ExecutionManifest,
    roots: ExecutionRoots,
    *,
    trusted_modules: Collection[str],
    environment: Mapping[str, str] | None = None,
    availability: ResourceAvailability | None = None,
    python_executable: str | Path | None = None,
    dry_run: bool = False,
) -> PreparedExecution:
    """Resolve and verify an execution, raising before any experiment code starts."""

    if not trusted_modules or manifest.entrypoint.module not in set(trusted_modules):
        raise PreflightError("entrypoint module is not present in the explicit trust set")

    effective_environment = os.environ if environment is None else environment
    if any(
        name not in effective_environment
        or not isinstance(effective_environment[name], str)
        or not effective_environment[name]
        for name in manifest.required_environment
    ):
        raise PreflightError("one or more named environment requirements are unsatisfied")

    config_root = _existing_directory(roots.config_root, "config_root")
    input_root = _existing_directory(roots.input_root, "input_root")
    output_root = _existing_directory(roots.output_root, "output_root")
    measured = availability if availability is not None else discover_resources()
    _verify_resource_request(manifest.resources, measured)

    executable = Path(sys.executable if python_executable is None else python_executable)
    try:
        resolved_executable = executable.resolve(strict=True)
    except OSError as exc:
        raise PreflightError("Python executable does not exist") from exc
    if not resolved_executable.is_file():
        raise PreflightError("Python executable is not a regular file")

    resolved_bindings: dict[tuple[str, str], Path] = {}
    verified_inputs: list[VerifiedInput] = []
    for binding in manifest.configs:
        resolved = _resolve_beneath(
            config_root,
            binding.path,
            must_exist=True,
            regular_file=True,
        )
        resolved_bindings[("config", binding.name)] = resolved
        verified_inputs.append(_verified_binding("config", binding, resolved))
    for binding in manifest.inputs:
        resolved = _resolve_beneath(
            input_root,
            binding.path,
            must_exist=True,
            regular_file=True,
        )
        resolved_bindings[("input", binding.name)] = resolved
        verified_inputs.append(_verified_binding("input", binding, resolved))

    output_paths: list[tuple[OutputBinding, Path]] = []
    for output_binding in manifest.outputs:
        resolved = _resolve_beneath(
            output_root,
            output_binding.path,
            must_exist=False,
        )
        if resolved.exists():
            raise PreflightError(
                f"declared output already exists: {output_binding.path}"
            )
        resolved_bindings[("output", output_binding.name)] = resolved
        output_paths.append((output_binding, resolved))

    receipt_path = _resolve_beneath(
        output_root,
        _receipt_relative_path(manifest, dry_run=dry_run),
        must_exist=False,
    )
    if receipt_path.exists():
        raise ReceiptExistsError("execution receipt already exists and will not be overwritten")

    expanded_argv: list[str] = []
    for argument in manifest.entrypoint.argv:
        match = _PLACEHOLDER_PATTERN.fullmatch(argument)
        if match:
            expanded_argv.append(
                str(resolved_bindings[(match.group("kind"), match.group("name"))])
            )
        else:
            expanded_argv.append(argument)

    return PreparedExecution(
        manifest_sha256=manifest.canonical_sha256(),
        command=(
            str(resolved_executable),
            "-m",
            manifest.entrypoint.module,
            *expanded_argv,
        ),
        verified_inputs=tuple(
            sorted(verified_inputs, key=lambda item: (item.kind, item.name, item.path))
        ),
        output_paths=tuple(output_paths),
        receipt_path=receipt_path,
        resolved_output_root=output_root,
    )


def _write_receipt_once(path: Path, receipt: ExecutionReceipt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            receipt.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise ReceiptExistsError(
            "execution receipt already exists and will not be overwritten"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _receipt(
    manifest: ExecutionManifest,
    prepared: PreparedExecution,
    *,
    status: Literal["dry_run", "succeeded", "failed"],
    started_at: datetime,
    finished_at: datetime,
    elapsed_seconds: float,
    exit_code: int | None,
    produced_outputs: tuple[ProducedOutput, ...] = (),
) -> ExecutionReceipt:
    return ExecutionReceipt(
        manifest_sha256=prepared.manifest_sha256,
        experiment_id=manifest.experiment_id,
        entrypoint_module=manifest.entrypoint.module,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=max(0.0, elapsed_seconds),
        exit_code=exit_code,
        resources=manifest.resources,
        required_environment=tuple(sorted(manifest.required_environment)),
        verified_inputs=prepared.verified_inputs,
        declared_outputs=tuple(
            sorted(manifest.outputs, key=lambda item: (item.name, item.path))
        ),
        produced_outputs=produced_outputs,
    )


def execute_manifest(
    manifest: ExecutionManifest,
    roots: ExecutionRoots,
    *,
    trusted_modules: Collection[str],
    environment: Mapping[str, str] | None = None,
    availability: ResourceAvailability | None = None,
    python_executable: str | Path | None = None,
    dry_run: bool = False,
) -> ExecutionOutcome:
    """Preflight and execute a manifest without invoking a shell.

    The receipt is created atomically and never overwritten.  It omits argv,
    absolute paths, host identity, subprocess output, and environment values.
    """

    effective_environment = os.environ if environment is None else environment
    prepared = preflight_execution(
        manifest,
        roots,
        trusted_modules=trusted_modules,
        environment=effective_environment,
        availability=availability,
        python_executable=python_executable,
        dry_run=dry_run,
    )
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()

    if dry_run:
        finished_at = datetime.now(timezone.utc)
        receipt = _receipt(
            manifest,
            prepared,
            status="dry_run",
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=time.monotonic() - started_monotonic,
            exit_code=None,
        )
        _write_receipt_once(prepared.receipt_path, receipt)
        return ExecutionOutcome(
            status="dry_run",
            manifest_sha256=prepared.manifest_sha256,
            returncode=None,
            receipt_path=prepared.receipt_path,
        )

    for _, output_path in prepared.output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    returncode: int | None = None
    try:
        result = subprocess.run(
            list(prepared.command),
            check=False,
            shell=False,
            cwd=prepared.resolved_output_root,
            env=dict(effective_environment),
            timeout=manifest.timeout_seconds,
        )
        returncode = result.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        finished_at = datetime.now(timezone.utc)
        receipt = _receipt(
            manifest,
            prepared,
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=time.monotonic() - started_monotonic,
            exit_code=None,
        )
        _write_receipt_once(prepared.receipt_path, receipt)
        outcome = ExecutionOutcome(
            status="failed",
            manifest_sha256=prepared.manifest_sha256,
            returncode=None,
            receipt_path=prepared.receipt_path,
        )
        raise ExecutionFailed("Python module execution could not start", outcome) from exc

    if returncode != 0:
        finished_at = datetime.now(timezone.utc)
        receipt = _receipt(
            manifest,
            prepared,
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=time.monotonic() - started_monotonic,
            exit_code=returncode,
        )
        _write_receipt_once(prepared.receipt_path, receipt)
        outcome = ExecutionOutcome(
            status="failed",
            manifest_sha256=prepared.manifest_sha256,
            returncode=returncode,
            receipt_path=prepared.receipt_path,
        )
        raise ExecutionFailed(
            f"Python module exited unsuccessfully with code {returncode}",
            outcome,
        )

    try:
        produced: list[ProducedOutput] = []
        for binding, _ in prepared.output_paths:
            resolved = _resolve_beneath(
                prepared.resolved_output_root,
                binding.path,
                must_exist=True,
                regular_file=binding.kind == "file",
            )
            if binding.kind == "directory":
                if not resolved.is_dir():
                    raise PreflightError("declared directory output is not a directory")
                digest, byte_size, file_count = _directory_closure(resolved)
            else:
                digest = _sha256_file(resolved)
                byte_size = resolved.stat().st_size
                file_count = 1
            produced.append(
                ProducedOutput(
                    name=binding.name,
                    path=binding.path,
                    kind=binding.kind,
                    sha256=digest,
                    byte_size=byte_size,
                    file_count=file_count,
                )
            )
    except (OSError, PreflightError) as exc:
        finished_at = datetime.now(timezone.utc)
        receipt = _receipt(
            manifest,
            prepared,
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=time.monotonic() - started_monotonic,
            exit_code=0,
        )
        _write_receipt_once(prepared.receipt_path, receipt)
        outcome = ExecutionOutcome(
            status="failed",
            manifest_sha256=prepared.manifest_sha256,
            returncode=0,
            receipt_path=prepared.receipt_path,
        )
        raise ExecutionFailed("declared output verification failed", outcome) from exc

    finished_at = datetime.now(timezone.utc)
    receipt = _receipt(
        manifest,
        prepared,
        status="succeeded",
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=time.monotonic() - started_monotonic,
        exit_code=0,
        produced_outputs=tuple(
            sorted(produced, key=lambda item: (item.name, item.path))
        ),
    )
    _write_receipt_once(prepared.receipt_path, receipt)
    return ExecutionOutcome(
        status="succeeded",
        manifest_sha256=prepared.manifest_sha256,
        returncode=0,
        receipt_path=prepared.receipt_path,
    )


def verify_execution_outputs(
    receipt_path: str | Path,
    output_root: str | Path,
) -> dict[str, str | int]:
    """Replay a successful receipt's checksum closure against retrieved outputs."""

    receipt_source = Path(receipt_path)
    receipt = ExecutionReceipt.model_validate_json(receipt_source.read_bytes())
    if receipt.status != "succeeded":
        raise PreflightError("only a successful execution receipt can verify outputs")

    declared = {
        (item.name, item.path, item.kind)
        for item in receipt.declared_outputs
    }
    produced = {
        (item.name, item.path, item.kind)
        for item in receipt.produced_outputs
    }
    if declared != produced:
        raise PreflightError("receipt output declarations and closure differ")

    root = _existing_directory(Path(output_root), "output_root")
    total_bytes = 0
    for expected in receipt.produced_outputs:
        resolved = _resolve_beneath(
            root,
            expected.path,
            must_exist=True,
            regular_file=expected.kind == "file",
        )
        if expected.kind == "directory":
            if not resolved.is_dir():
                raise PreflightError("declared directory output is not a directory")
            digest, byte_size, file_count = _directory_closure(resolved)
        else:
            digest = _sha256_file(resolved)
            byte_size = resolved.stat().st_size
            file_count = 1
        if (
            digest != expected.sha256
            or byte_size != expected.byte_size
            or file_count != expected.file_count
        ):
            raise PreflightError(f"output closure differs from receipt: {expected.name}")
        total_bytes += byte_size

    return {
        "status": "execution_outputs_verified",
        "experiment_id": receipt.experiment_id,
        "manifest_sha256": receipt.manifest_sha256,
        "receipt_sha256": _sha256_file(receipt_source),
        "output_count": len(receipt.produced_outputs),
        "total_output_bytes": total_bytes,
    }
