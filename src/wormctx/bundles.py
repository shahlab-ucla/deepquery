"""Create and verify content-addressed source bundles without copying a worktree."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


BUNDLE_RECEIPT_VERSION = "deepquery-source-bundle-1.0"


class SourceBundleReceipt(BaseModel):
    """Portable metadata required to verify a source archive before extraction."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["deepquery-source-bundle-1.0"] = BUNDLE_RECEIPT_VERSION
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    archive: str = Field(pattern=r"^[A-Za-z0-9._-]+[.]tar[.]gz$")
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_bytes: int = Field(gt=0)
    archive_prefix: str = Field(pattern=r"^[A-Za-z0-9._-]+/$")
    member_count: int = Field(gt=0)


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        summary = detail[-1] if detail else "git command failed"
        raise ValueError(summary)
    return completed.stdout.strip()


def _resolve_revision(repository: Path, revision: str) -> str:
    if not revision or revision.startswith("-") or "\x00" in revision:
        raise ValueError("revision must be a non-option Git revision")
    resolved = _git(
        repository,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{revision}^{{commit}}",
    )
    if len(resolved) != 40 or any(character not in "0123456789abcdef" for character in resolved):
        raise ValueError("resolved revision is not a full lowercase commit digest")
    return resolved


def _require_clean_worktree(repository: Path) -> None:
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError("source bundle requires a clean worktree")


def inspect_source_archive(archive: str | Path, *, expected_prefix: str | None = None) -> int:
    """Reject traversal, links, devices, and files outside the declared archive prefix."""

    source = Path(archive)
    count = 0
    with tarfile.open(source, mode="r:gz") as bundle:
        for member in bundle.getmembers():
            count += 1
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError("source archive contains an unsafe member path")
            if not pure.parts or pure.parts[0] in ("", "."):
                raise ValueError("source archive contains an invalid member path")
            if expected_prefix is not None and pure.parts[0] != expected_prefix.rstrip("/"):
                raise ValueError("source archive member lies outside its declared prefix")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError("source archive contains a link or special filesystem entry")
    if count == 0:
        raise ValueError("source archive is empty")
    return count


def _embedded_revision(archive: Path, prefix: str) -> str:
    member_name = f"{prefix}SOURCE_REVISION"
    with tarfile.open(archive, mode="r:gz") as bundle:
        try:
            member = bundle.getmember(member_name)
        except KeyError as exc:
            raise ValueError("source archive lacks SOURCE_REVISION") from exc
        stream = bundle.extractfile(member)
        if stream is None:
            raise ValueError("SOURCE_REVISION is not a regular archive member")
        return stream.read().decode("ascii").strip()


def create_source_bundle(
    repository: str | Path,
    output_directory: str | Path,
    *,
    revision: str = "HEAD",
    require_clean: bool = True,
) -> SourceBundleReceipt:
    """Archive a committed revision and write a portable digest receipt."""

    root = Path(repository).resolve()
    if _git(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError("repository is not a Git worktree")
    commit = _resolve_revision(root, revision)
    if require_clean:
        _require_clean_worktree(root)

    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    short_revision = commit[:12]
    archive_name = f"deepquery-{short_revision}.tar.gz"
    archive_path = destination / archive_name
    receipt_path = destination / f"{archive_name}.receipt.json"
    if archive_path.exists() or receipt_path.exists():
        raise FileExistsError("source bundle destination already contains this revision")

    prefix = f"deepquery-{short_revision}/"
    _git(
        root,
        "archive",
        "--format=tar.gz",
        f"--prefix={prefix}",
        f"--output={archive_path}",
        commit,
    )
    member_count = inspect_source_archive(archive_path, expected_prefix=prefix)
    if _embedded_revision(archive_path, prefix) != commit:
        raise ValueError("embedded SOURCE_REVISION does not match the archived commit")

    receipt = SourceBundleReceipt(
        revision=commit,
        archive=archive_name,
        archive_sha256=_sha256(archive_path),
        archive_bytes=archive_path.stat().st_size,
        archive_prefix=prefix,
        member_count=member_count,
    )
    with receipt_path.open("xb") as stream:
        stream.write(_canonical_json(receipt.model_dump(mode="json")))
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def verify_source_bundle(
    archive: str | Path,
    receipt: str | Path,
) -> SourceBundleReceipt:
    """Verify a bundle receipt and its safe archive structure before extraction."""

    archive_path = Path(archive)
    receipt_path = Path(receipt)
    parsed = SourceBundleReceipt.model_validate_json(receipt_path.read_bytes())
    if archive_path.name != parsed.archive:
        raise ValueError("archive filename differs from its receipt")
    if archive_path.stat().st_size != parsed.archive_bytes:
        raise ValueError("archive byte count differs from its receipt")
    if _sha256(archive_path) != parsed.archive_sha256:
        raise ValueError("archive digest differs from its receipt")
    count = inspect_source_archive(archive_path, expected_prefix=parsed.archive_prefix)
    if count != parsed.member_count:
        raise ValueError("archive member count differs from its receipt")
    if _embedded_revision(archive_path, parsed.archive_prefix) != parsed.revision:
        raise ValueError("embedded SOURCE_REVISION differs from its receipt")
    return parsed
