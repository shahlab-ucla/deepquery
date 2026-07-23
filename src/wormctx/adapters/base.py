"""Source adapter interfaces; adapters parse local snapshots and never fetch."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Protocol

from ..models import (
    ArtifactReceipt,
    ContextualObservation,
    DatasetProfile,
    ReleaseManifest,
)


class Severity(str, Enum):
    warning = "warning"
    error = "error"


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    record_locator: str | None = None


@dataclass
class ValidationReport:
    source_id: str
    snapshot_id: str | None = None
    adapter_version: str | None = None
    schema_version: str = "0.1.0"
    valid: bool = True
    checked_records: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity is Severity.error:
            self.valid = False


@dataclass(frozen=True)
class SnapshotContext:
    """A parser receives a verified release lock and its exact artifact receipts."""

    manifest: ReleaseManifest
    receipts: tuple[ArtifactReceipt, ...]
    root: Path

    @property
    def snapshot_id(self) -> str:
        manifest_hashes = {item.manifest_sha256 for item in self.receipts}
        if len(manifest_hashes) != 1:
            raise ValueError("snapshot receipts do not share one manifest hash")
        return next(iter(manifest_hashes))

    def artifact_path(self, artifact_id: str) -> Path:
        matching = [item for item in self.receipts if item.artifact_id == artifact_id]
        if len(matching) != 1:
            raise KeyError(f"expected one verified receipt for {artifact_id!r}")
        path = self.resolve_receipt_path(matching[0])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def resolve_receipt_path(self, receipt: ArtifactReceipt) -> Path:
        """Resolve a receipt only against this snapshot's active raw root.

        Portable receipts use the exact digest-derived locator.  Legacy
        receipts retain their absolute path for compatibility, but it must be
        the canonical content-addressed path under this context's root.  This
        deliberately prevents a copied legacy receipt from silently reading
        bytes at its old host location.
        """

        root = self.root.resolve()
        expected_locator = (
            f"blobs/sha256/{receipt.sha256[:2]}/{receipt.sha256}"
        )
        expected = (root / Path(*expected_locator.split("/"))).resolve()
        if not expected.is_relative_to(root):
            raise ValueError("content-addressed blob path escapes the snapshot root")

        if receipt.artifact_receipt_schema_version == "2.0":
            if receipt.blob_locator != expected_locator:
                raise ValueError("artifact receipt has a noncanonical blob locator")
            candidate = (
                root / Path(*receipt.blob_locator.split("/"))
            ).resolve()
        else:
            assert receipt.blob_path is not None
            legacy = Path(receipt.blob_path)
            if not legacy.is_absolute():
                raise ValueError("legacy blob_path is not absolute on this host")
            candidate = legacy.resolve()

        if not candidate.is_relative_to(root):
            raise ValueError("artifact receipt points outside the active blob store")
        if candidate != expected:
            raise ValueError("artifact receipt has a noncanonical blob path")
        return candidate


class SourceAdapter(Protocol):
    source_id: str
    adapter_version: str

    def validate_raw(self, snapshot: SnapshotContext) -> ValidationReport:
        """Validate source-specific format, headers, counts, and references."""
        ...

    def iter_observations(self, snapshot: SnapshotContext) -> Iterator[ContextualObservation]:
        """Deterministically normalize a local snapshot."""
        ...

    def dataset_profiles(self, snapshot: SnapshotContext) -> Iterator[DatasetProfile]:
        """Emit source-level coverage metadata for the transport map."""
        ...
