"""Small reproducibility and artifact-integrity helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)


def write_json(path: str | Path, payload: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload) + b"\n")


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    atomic_write_bytes(path, payload)


def write_text(path: str | Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def runtime_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import pydantic

        manifest["pydantic_version"] = pydantic.__version__
    except ImportError:
        manifest["pydantic_version"] = None
    try:
        import numpy

        manifest["numpy_version"] = numpy.__version__
    except ImportError:
        manifest["numpy_version"] = None
    try:
        import torch

        manifest["torch_version"] = torch.__version__
        manifest["torch_cuda_version"] = torch.version.cuda
        manifest["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["cuda_device"] = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            manifest["cuda_total_memory_bytes"] = properties.total_memory
            manifest["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    except ImportError:
        manifest["torch_version"] = None
        manifest["cuda_available"] = False
    return manifest


def git_manifest(repository: str | Path) -> dict[str, Any]:
    root = Path(repository)
    result: dict[str, Any] = {
        "repository": str(root.resolve()),
        "provenance": "git_cli",
    }
    for name, args in {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--short"],
    }.items():
        try:
            completed = subprocess.run(
                args,
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            result[name] = completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            result[name] = None
    if result.get("commit") is None:
        revision_path = root / "SOURCE_REVISION"
        try:
            exported_revision = revision_path.read_text(encoding="ascii").strip()
        except OSError:
            exported_revision = ""
        if re.fullmatch(r"[0-9a-f]{40}", exported_revision):
            result.update(
                {
                    "commit": exported_revision,
                    "branch": None,
                    "status": "archive_without_git_metadata",
                    "provenance": "git_archive_export_subst",
                }
            )
    return result


def write_checksums(root: str | Path) -> list[dict[str, str]]:
    directory = Path(root)
    excluded = {"checksums.sha256", "SUCCESS"}
    entries: list[dict[str, str]] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative in excluded or path.name.endswith(".tmp"):
            continue
        entries.append({"path": relative, "sha256": sha256_file(path)})
    lines = "".join(f"{item['sha256']}  {item['path']}\n" for item in entries)
    write_text(directory / "checksums.sha256", lines)
    return entries


def verify_checksums(root: str | Path) -> list[str]:
    directory = Path(root)
    receipt = directory / "checksums.sha256"
    failures: list[str] = []
    for line in receipt.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = directory / relative
        if not target.is_file():
            failures.append(f"missing:{relative}")
        elif sha256_file(target) != digest:
            failures.append(f"mismatch:{relative}")
    return failures
