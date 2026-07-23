from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from wormctx.bundles import (
    create_source_bundle,
    inspect_source_archive,
    verify_source_bundle,
)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "DeepQuery Test")
    _git(repository, "config", "user.email", "deepquery-test@example.org")
    (repository / ".gitattributes").write_text(
        "SOURCE_REVISION export-subst\n",
        encoding="ascii",
    )
    (repository / "SOURCE_REVISION").write_text("$Format:%H$\n", encoding="ascii")
    (repository / "payload.txt").write_text("bounded fixture\n", encoding="ascii")
    _git(repository, "add", ".gitattributes", "SOURCE_REVISION", "payload.txt")
    _git(repository, "commit", "-m", "fixture")
    return repository, _git(repository, "rev-parse", "HEAD")


def test_create_and_verify_source_bundle(tmp_path: Path) -> None:
    repository, revision = _repository(tmp_path)
    output = tmp_path / "bundles"

    receipt = create_source_bundle(repository, output)
    archive = output / receipt.archive
    receipt_path = output / f"{receipt.archive}.receipt.json"

    assert receipt.revision == revision
    assert receipt.archive_prefix == f"deepquery-{revision[:12]}/"
    assert receipt.member_count == 4
    assert receipt_path.is_file()
    assert verify_source_bundle(archive, receipt_path) == receipt
    serialized = json.loads(receipt_path.read_text(encoding="ascii"))
    assert str(tmp_path) not in json.dumps(serialized)

    with tarfile.open(archive, mode="r:gz") as bundle:
        embedded = bundle.extractfile(f"{receipt.archive_prefix}SOURCE_REVISION")
        assert embedded is not None
        assert embedded.read().decode("ascii").strip() == revision


def test_bundle_creation_rejects_dirty_worktree(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)
    (repository / "payload.txt").write_text("changed\n", encoding="ascii")

    with pytest.raises(ValueError, match="clean worktree"):
        create_source_bundle(repository, tmp_path / "bundles")


def test_bundle_creation_is_write_once(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)
    output = tmp_path / "bundles"
    create_source_bundle(repository, output)

    with pytest.raises(FileExistsError, match="already contains"):
        create_source_bundle(repository, output)


def test_archive_inspection_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        payload = b"unsafe\n"
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe member path"):
        inspect_source_archive(archive)


def test_verification_rejects_tampering(tmp_path: Path) -> None:
    repository, _ = _repository(tmp_path)
    output = tmp_path / "bundles"
    receipt = create_source_bundle(repository, output)
    archive = output / receipt.archive
    archive.write_bytes(archive.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="byte count"):
        verify_source_bundle(archive, output / f"{receipt.archive}.receipt.json")
