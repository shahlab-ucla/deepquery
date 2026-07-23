"""Release catalogs, immutable artifact retrieval, and content-addressed receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import ValidationError

from .io import read_json, write_json
from .models import (
    ArtifactReceipt,
    ReleaseManifest,
    RightsStatus,
    SourceCatalog,
)
from .adapters.base import SnapshotContext


class ManifestError(RuntimeError):
    pass


PORTABLE_RECEIPT_SCHEMA_VERSION = "2.0"
LEGACY_RECEIPT_SCHEMA_VERSION = "1.1"


def load_release_manifest(path: str | Path) -> ReleaseManifest:
    return ReleaseManifest.model_validate(read_json(path))


def list_catalog(path: str | Path) -> list[dict[str, Any]]:
    catalog = SourceCatalog.model_validate(read_json(path))
    return [item.model_dump(mode="json", exclude_none=True) for item in catalog.sources]


def catalog_plan(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for source in list_catalog(path):
        artifacts = source.get("artifacts", [])
        rows.append(
            {
                "source_id": source.get("source_id"),
                "release": source.get("release"),
                "enabled": bool(source.get("enabled", False)),
                "immutable": not bool(source.get("release_is_mutable", True)),
                "rights_status": source.get("rights_status", "missing"),
                "inventory_complete": bool(source.get("inventory_complete", False)),
                "implementation_status": source.get("implementation_status", "missing"),
                "artifact_count": len(artifacts),
                "enabled_artifact_count": sum(
                    1 for artifact in artifacts if artifact.get("enabled", False)
                ),
                "adapter": source.get("adapter"),
            }
        )
    return rows


def fetch_release(
    manifest_path: str | Path,
    raw_root: str | Path,
    *,
    timeout_seconds: float = 60.0,
    max_bytes: int = 5 * 1024**3,
) -> list[ArtifactReceipt]:
    manifest_file = Path(manifest_path).resolve()
    manifest = load_release_manifest(manifest_file)
    _preflight_release(manifest)
    if manifest.release_is_mutable:
        raise ManifestError("refusing mutable release; resolve and lock it before fetching")
    if manifest.rights_status is not RightsStatus.cleared:
        raise ManifestError(
            f"refusing source with rights_status={manifest.rights_status.value}; review first"
        )

    manifest_sha256 = _sha256_file(manifest_file)
    receipts: list[ArtifactReceipt] = []
    for artifact in manifest.artifacts:
        if not artifact.enabled:
            continue
        receipt = _fetch_artifact(
            manifest=manifest,
            artifact=artifact,
            manifest_directory=manifest_file.parent,
            manifest_sha256=manifest_sha256,
            raw_root=Path(raw_root),
            timeout_seconds=timeout_seconds,
            max_bytes=max_bytes,
        )
        receipts.append(receipt)
    receipt_root = (
        Path(raw_root)
        / "receipts"
        / manifest.source_id
        / manifest.release
        / manifest_sha256
    )
    _write_release_receipts(
        receipts=receipts,
        receipt_root=receipt_root,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        schema_version=PORTABLE_RECEIPT_SCHEMA_VERSION,
    )
    return receipts


def load_snapshot_context(
    manifest_path: str | Path, raw_root: str | Path
) -> SnapshotContext:
    """Reconstruct and verify the immutable bridge from fetch to normalization."""
    snapshot, _ = load_snapshot_context_with_binding(manifest_path, raw_root)
    return snapshot


def load_snapshot_context_with_binding(
    manifest_path: str | Path, raw_root: str | Path
) -> tuple[SnapshotContext, dict[str, Any]]:
    """Verify a snapshot and return its exact, portable provenance binding."""
    return _load_snapshot_context_with_binding(
        manifest_path, raw_root, allow_relocated_legacy=False
    )


def _load_snapshot_context_with_binding(
    manifest_path: str | Path,
    raw_root: str | Path,
    *,
    allow_relocated_legacy: bool,
) -> tuple[SnapshotContext, dict[str, Any]]:
    """Internal loader with a migration-only legacy relocation mode.

    Normal callers never enable relocation.  The migration path still verifies
    the complete legacy receipt chain and locked metadata, but derives the blob
    from the explicitly supplied source root after validating the stale path's
    canonical content-addressed suffix.
    """

    manifest_file = Path(manifest_path).resolve()
    manifest = load_release_manifest(manifest_file)
    _preflight_release(manifest)
    manifest_sha256 = _sha256_file(manifest_file)
    receipt_root = (
        Path(raw_root)
        / "receipts"
        / manifest.source_id
        / manifest.release
        / manifest_sha256
    )
    release_receipt_path = receipt_root / "release.receipt.json"
    if not release_receipt_path.is_file():
        raise ManifestError(f"release completion receipt is missing: {release_receipt_path}")
    release_receipt = read_json(release_receipt_path)
    release_schema_version = _validate_release_receipt_header(
        release_receipt, manifest, manifest_sha256
    )
    release_items = release_receipt.get("artifact_receipts")
    if not isinstance(release_items, list):
        raise ManifestError("release receipt artifact_receipts must be a list")
    release_by_file: dict[str, dict[str, Any]] = {}
    for item in release_items:
        if not isinstance(item, dict) or set(item) != {
            "artifact_id",
            "artifact_spec_sha256",
            "sha256",
            "byte_size",
            "blob_uri",
            "receipt_file",
            "receipt_sha256",
        }:
            raise ManifestError("release receipt contains a malformed artifact tuple")
        receipt_file = item.get("receipt_file")
        if not isinstance(receipt_file, str) or Path(receipt_file).name != receipt_file:
            raise ManifestError("release receipt contains an unsafe receipt filename")
        if receipt_file in release_by_file:
            raise ManifestError("release receipt contains duplicate receipt filenames")
        release_by_file[receipt_file] = item

    enabled_specs = {
        item.artifact_id: item for item in manifest.artifacts if item.enabled
    }
    receipts: list[ArtifactReceipt] = []
    portable_artifacts: list[dict[str, Any]] = []
    blob_root = (Path(raw_root) / "blobs").resolve()
    artifact_paths = sorted(
        path
        for path in receipt_root.glob("*.receipt.json")
        if path.name != "release.receipt.json"
    )
    if {path.name for path in artifact_paths} != set(release_by_file):
        raise ManifestError(
            "artifact receipt files do not exactly match the release receipt tuple"
    )
    for path in artifact_paths:
        release_item = release_by_file[path.name]
        try:
            receipt = ArtifactReceipt.model_validate(read_json(path))
        except ValidationError as exc:
            raise ManifestError(f"invalid artifact receipt: {path}: {exc}") from exc
        if release_schema_version == PORTABLE_RECEIPT_SCHEMA_VERSION:
            if receipt.artifact_receipt_schema_version != PORTABLE_RECEIPT_SCHEMA_VERSION:
                raise ManifestError(
                    f"portable release contains a nonportable artifact receipt: {path}"
                )
        elif receipt.artifact_receipt_schema_version is not None:
            raise ManifestError(
                f"legacy release contains a versioned artifact receipt: {path}"
            )
        artifact = enabled_specs.get(receipt.artifact_id)
        if artifact is None:
            raise ManifestError(f"receipt has no enabled locked artifact: {receipt.artifact_id}")
        _verify_artifact_receipt(
            receipt=receipt,
            artifact=artifact,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            manifest_directory=manifest_file.parent,
            receipt_path=path,
            allow_relocated_local_source=allow_relocated_legacy,
        )
        try:
            if (
                allow_relocated_legacy
                and receipt.artifact_receipt_schema_version is None
            ):
                _validate_legacy_blob_path_suffix(receipt)
                blob = _canonical_blob_path(Path(raw_root), receipt.sha256)
            else:
                resolution_context = SnapshotContext(
                    manifest=manifest,
                    receipts=(receipt,),
                    root=Path(raw_root).resolve(),
                )
                blob = resolution_context.resolve_receipt_path(receipt)
        except ValueError as exc:
            raise ManifestError(f"invalid artifact blob location in {path}: {exc}") from exc
        if not blob.is_relative_to(blob_root):
            raise ManifestError(f"artifact receipt points outside the blob store: {path}")
        if (
            not blob.is_file()
            or blob.stat().st_size != receipt.byte_size
            or _sha256_file(blob) != receipt.sha256
        ):
            raise ManifestError(f"artifact blob failed verification: {blob}")
        if receipt.blob_uri != f"urn:sha256:{receipt.sha256}":
            raise ManifestError(f"artifact receipt has an invalid blob URI: {path}")
        expected_blob = _canonical_blob_path(Path(raw_root), receipt.sha256)
        if blob != expected_blob:
            raise ManifestError(f"artifact receipt has a noncanonical blob path: {path}")
        if _sha256_file(path) != release_item["receipt_sha256"]:
            raise ManifestError(
                f"artifact receipt file hash does not match release receipt: {path}"
            )
        expected_release_item = {
            "artifact_id": receipt.artifact_id,
            "artifact_spec_sha256": receipt.artifact_spec_sha256,
            "sha256": receipt.sha256,
            "byte_size": receipt.byte_size,
            "blob_uri": receipt.blob_uri,
            "receipt_file": path.name,
            "receipt_sha256": _sha256_file(path),
        }
        if release_item != expected_release_item:
            raise ManifestError(
                f"artifact receipt values do not match the release receipt tuple: {path}"
            )
        receipts.append(receipt)
        portable_artifacts.append(
            {
                "artifact_id": receipt.artifact_id,
                "filename": receipt.filename,
                "artifact_sha256": receipt.sha256,
                "expected_sha256": receipt.expected_sha256.lower(),
                "artifact_spec_sha256": receipt.artifact_spec_sha256,
                "byte_size": receipt.byte_size,
                "blob_uri": receipt.blob_uri,
                "artifact_receipt_sha256": release_item["receipt_sha256"],
            }
        )
    expected = sorted(enabled_specs)
    actual = sorted(item.artifact_id for item in receipts)
    if actual != expected:
        raise ManifestError(
            f"snapshot artifact mismatch: expected={expected}, receipts={actual}"
        )
    snapshot = SnapshotContext(
        manifest=manifest,
        receipts=tuple(sorted(receipts, key=lambda item: item.artifact_id)),
        root=Path(raw_root).resolve(),
    )
    binding = {
        "source_id": manifest.source_id,
        "source_release": manifest.release,
        "manifest_sha256": manifest_sha256,
        "release_receipt_sha256": _sha256_file(release_receipt_path),
        "adapter": manifest.adapter,
        "adapter_version": manifest.adapter_version,
        "artifact_receipts": sorted(
            portable_artifacts, key=lambda item: item["artifact_id"]
        ),
    }
    return snapshot, binding


def _validate_release_receipt_header(
    release_receipt: Any, manifest: ReleaseManifest, manifest_sha256: str
) -> str:
    if not isinstance(release_receipt, dict):
        raise ManifestError("release receipt must be a JSON object")
    expected_keys = {
        "receipt_schema_version",
        "source_id",
        "release",
        "manifest_sha256",
        "completed_at",
        "artifact_receipts",
    }
    if set(release_receipt) != expected_keys:
        raise ManifestError("release receipt has unexpected or missing fields")
    schema_version = release_receipt.get("receipt_schema_version")
    if schema_version not in {
        LEGACY_RECEIPT_SCHEMA_VERSION,
        PORTABLE_RECEIPT_SCHEMA_VERSION,
    }:
        raise ManifestError(
            f"unsupported release receipt schema version: {schema_version!r}"
        )
    expected_values = {
        "source_id": manifest.source_id,
        "release": manifest.release,
        "manifest_sha256": manifest_sha256,
    }
    for field, expected in expected_values.items():
        if release_receipt.get(field) != expected:
            raise ManifestError(f"release receipt {field} does not match the manifest")
    completed_at = release_receipt.get("completed_at")
    if not isinstance(completed_at, str):
        raise ManifestError("release receipt completed_at must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError("release receipt completed_at is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ManifestError("release receipt completed_at must include a timezone")
    return schema_version


def _verify_artifact_receipt(
    *,
    receipt: ArtifactReceipt,
    artifact: Any,
    manifest: ReleaseManifest,
    manifest_sha256: str,
    manifest_directory: Path,
    receipt_path: Path,
    allow_relocated_local_source: bool,
) -> None:
    """Require receipt identity and lock-derived values to match exactly."""
    locked_values = {
        "source_id": manifest.source_id,
        "release": manifest.release,
        "artifact_id": artifact.artifact_id,
        "manifest_sha256": manifest_sha256,
        "artifact_spec_sha256": _canonical_model_hash(artifact),
        "adapter": manifest.adapter,
        "adapter_version": manifest.adapter_version,
        "role": artifact.role,
        "filename": artifact.filename,
        "requested_url": artifact.url,
        "expected_sha256": artifact.expected_sha256,
        "rights_status": artifact.rights_status,
        "license_uri": str(artifact.license_uri),
    }
    actual_values = {
        **{field: getattr(receipt, field) for field in locked_values if field != "license_uri"},
        "license_uri": str(receipt.license_uri),
    }
    mismatches = [
        field for field, expected in locked_values.items() if actual_values[field] != expected
    ]
    if mismatches:
        raise ManifestError(
            f"artifact receipt does not match locked ArtifactSpec ({', '.join(mismatches)}): "
            f"{receipt_path}"
        )
    if receipt.sha256.lower() != artifact.expected_sha256.lower():
        raise ManifestError(f"artifact receipt checksum differs from lock: {receipt_path}")
    if artifact.expected_size is not None and receipt.byte_size != artifact.expected_size:
        raise ManifestError(f"artifact receipt size differs from lock: {receipt_path}")
    if not receipt.redirect_chain:
        raise ManifestError(f"artifact receipt redirect chain is empty: {receipt_path}")
    if receipt.redirect_chain[0] != receipt.requested_url:
        raise ManifestError(f"artifact receipt redirect chain has the wrong origin: {receipt_path}")
    parsed = urlparse(artifact.url)
    if parsed.scheme == "":
        expected_final_url = (manifest_directory / artifact.url).resolve().as_uri()
        portable_final_url = artifact.url
        allowed_final_urls = {portable_final_url}
        if receipt.artifact_receipt_schema_version is None:
            allowed_final_urls = {expected_final_url}
        elif receipt.final_url == expected_final_url:
            # Compatibility with the short-lived pre-portability v2 form.  It
            # remains loadable only at its original manifest root and can be
            # normalized with the explicit migration command.
            allowed_final_urls.add(expected_final_url)
        if allow_relocated_local_source and urlparse(receipt.final_url).scheme == "file":
            # Migration never reads this URI.  The manifest lock, receipt
            # chain, canonical blob suffix, byte count, and digest remain the
            # authority while a stale host-specific acquisition URI is
            # normalized in the destination receipt.
            allowed_final_urls.add(receipt.final_url)
        if (
            receipt.final_url not in allowed_final_urls
            or receipt.redirect_chain != [artifact.url]
        ):
            raise ManifestError(
                f"local artifact receipt has an invalid source path: {receipt_path}"
            )
        if receipt.response_status is not None:
            raise ManifestError(f"local artifact receipt cannot have HTTP status: {receipt_path}")
    else:
        if receipt.redirect_chain[-1] != receipt.final_url:
            raise ManifestError(
                f"artifact receipt redirect chain has the wrong destination: {receipt_path}"
            )
        for recorded_url in receipt.redirect_chain:
            recorded = urlparse(recorded_url)
            if (
                recorded.scheme != "https"
                or recorded.hostname not in set(manifest.allowed_hosts)
            ):
                raise ManifestError(
                    f"artifact receipt records an insecure or unapproved redirect: {receipt_path}"
                )
        if receipt.response_status is None or not 200 <= receipt.response_status < 300:
            raise ManifestError(
                f"remote artifact receipt lacks a successful HTTP status: {receipt_path}"
            )


def _fetch_artifact(
    *,
    manifest: ReleaseManifest,
    artifact: Any,
    manifest_directory: Path,
    manifest_sha256: str,
    raw_root: Path,
    timeout_seconds: float,
    max_bytes: int,
) -> ArtifactReceipt:
    parsed = urlparse(artifact.url)
    requested_url = artifact.url
    response_headers: dict[str, str] = {}
    response_status: int | None = None
    response_content_type: str | None = artifact.media_type
    final_url = requested_url
    redirect_chain = [requested_url]

    if parsed.scheme == "":
        source_path = (manifest_directory / artifact.url).resolve()
        if not source_path.is_relative_to(manifest_directory):
            raise ManifestError("relative local artifact escapes the manifest directory")
        if not source_path.is_file():
            raise ManifestError(f"local artifact does not exist: {source_path}")
        source: BinaryIO = source_path.open("rb")
        # A v2 receipt binds the verified bytes, not the checkout location of
        # a relative local fixture.  Keep the locked relative source locator so
        # moving the manifest and raw tree together does not leak or bind an
        # absolute host path.
        final_url = artifact.url
    elif parsed.scheme == "https":
        if not parsed.hostname or parsed.hostname not in set(manifest.allowed_hosts):
            raise ManifestError(f"remote host is not allow-listed: {parsed.hostname!r}")
        headers = {
            "User-Agent": "wormctx/0.1 (+https://w3id.org/wormctx/)",
            **artifact.acquisition.non_secret_headers,
        }
        if artifact.acquisition.accept:
            headers["Accept"] = artifact.acquisition.accept
        request = Request(
            artifact.url,
            headers=headers,
            method=artifact.acquisition.method,
        )
        redirect_handler = _AllowlistedRedirectHandler(
            set(manifest.allowed_hosts), redirect_chain
        )
        opener = build_opener(redirect_handler)
        response = opener.open(request, timeout=timeout_seconds)  # noqa: S310 - validated hops
        source = response
        final_url = response.geturl()
        final_parts = urlparse(final_url)
        final_host = final_parts.hostname
        if final_parts.scheme != "https" or final_host not in set(manifest.allowed_hosts):
            response.close()
            raise ManifestError(
                "redirected to an insecure or non-allow-listed destination: "
                f"{final_url!r}"
            )
        response_headers = {key.lower(): value for key, value in response.headers.items()}
        response_status = response.getcode()
        response_content_type = response_headers.get("content-type", artifact.media_type)
    else:
        raise ManifestError(f"unsupported or insecure URL scheme: {parsed.scheme!r}")

    raw_root.mkdir(parents=True, exist_ok=True)
    staging = raw_root / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_size = 0
    temporary_path: Path | None = None
    try:
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=staging) as destination:
                temporary_path = Path(destination.name)
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    byte_size += len(chunk)
                    if byte_size > max_bytes:
                        raise ManifestError(
                            f"artifact exceeded configured byte limit ({max_bytes} bytes)"
                        )
                    digest.update(chunk)
                    destination.write(chunk)
        finally:
            source.close()
    except Exception:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        raise

    sha256 = digest.hexdigest()
    if artifact.expected_size is not None and byte_size != artifact.expected_size:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        raise ManifestError(
            f"size mismatch for {artifact.artifact_id}: expected {artifact.expected_size}, "
            f"received {byte_size}"
        )
    if artifact.expected_sha256 and sha256.lower() != artifact.expected_sha256.lower():
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        raise ManifestError(
            f"checksum mismatch for {artifact.artifact_id}: expected "
            f"{artifact.expected_sha256}, received {sha256}"
        )

    blob = raw_root / "blobs" / "sha256" / sha256[:2] / sha256
    blob.parent.mkdir(parents=True, exist_ok=True)
    if blob.exists():
        existing_hash = _sha256_file(blob)
        if blob.stat().st_size != byte_size or existing_hash != sha256:
            if temporary_path:
                temporary_path.unlink(missing_ok=True)
            raise ManifestError(f"existing content-addressed blob failed integrity check: {blob}")
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
    else:
        assert temporary_path is not None
        os.replace(temporary_path, blob)

    receipt = ArtifactReceipt(
        artifact_receipt_schema_version=PORTABLE_RECEIPT_SCHEMA_VERSION,
        source_id=manifest.source_id,
        release=manifest.release,
        artifact_id=artifact.artifact_id,
        manifest_sha256=manifest_sha256,
        artifact_spec_sha256=_canonical_model_hash(artifact),
        adapter=manifest.adapter,
        adapter_version=manifest.adapter_version,
        role=artifact.role,
        filename=artifact.filename,
        requested_url=requested_url,
        final_url=final_url,
        redirect_chain=redirect_chain,
        fetched_at=datetime.now(timezone.utc),
        sha256=sha256,
        expected_sha256=artifact.expected_sha256,
        byte_size=byte_size,
        blob_uri=f"urn:sha256:{sha256}",
        blob_locator=_canonical_blob_locator(sha256),
        rights_status=artifact.rights_status,
        license_uri=artifact.license_uri,
        response_status=response_status,
        response_content_type=response_content_type,
        response_etag=response_headers.get("etag"),
        response_last_modified=response_headers.get("last-modified"),
    )
    return receipt


def migrate_snapshot_receipts(
    manifest_path: str | Path,
    source_raw_root: str | Path,
    destination_raw_root: str | Path,
) -> dict[str, Any]:
    """Re-emit a fully verified snapshot with portable version 2.0 receipts.

    Migration is source-to-fresh-target and never mutates the source.  A copied
    legacy snapshot may have an absolute path from another OS; only migration
    can tolerate that stale prefix, and only after the legacy release chain,
    locked metadata, canonical content-addressed suffix, and bytes under the
    explicitly supplied source root have all been verified.
    """

    manifest_file = Path(manifest_path).resolve()
    source_root = Path(source_raw_root).resolve()
    destination_root = Path(destination_raw_root).resolve()
    if source_root == destination_root:
        raise ManifestError("migration destination must differ from the source root")
    if destination_root.is_relative_to(source_root) or source_root.is_relative_to(
        destination_root
    ):
        raise ManifestError("migration source and destination roots cannot be nested")
    if destination_root.exists():
        raise ManifestError("migration destination must not already exist")

    manifest = load_release_manifest(manifest_file)
    manifest_sha256 = _sha256_file(manifest_file)
    source_release_path = (
        source_root
        / "receipts"
        / manifest.source_id
        / manifest.release
        / manifest_sha256
        / "release.receipt.json"
    )
    source_release = read_json(source_release_path)
    source_schema_version = _validate_release_receipt_header(
        source_release, manifest, manifest_sha256
    )
    snapshot, source_binding = _load_snapshot_context_with_binding(
        manifest_file,
        source_root,
        allow_relocated_legacy=True,
    )

    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.receipt-migration-",
            dir=destination_root.parent,
        )
    ).resolve()
    try:
        migrated_receipts: list[ArtifactReceipt] = []
        enabled_specs = {
            item.artifact_id: item for item in manifest.artifacts if item.enabled
        }
        for receipt in snapshot.receipts:
            source_blob = _canonical_blob_path(source_root, receipt.sha256)
            _verify_blob_bytes(source_blob, receipt)
            destination_blob = _canonical_blob_path(staging_root, receipt.sha256)
            destination_blob.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_blob, destination_blob)
            _verify_blob_bytes(destination_blob, receipt)

            payload = receipt.model_dump(mode="json", exclude_none=True)
            payload.pop("blob_path", None)
            payload["artifact_receipt_schema_version"] = (
                PORTABLE_RECEIPT_SCHEMA_VERSION
            )
            payload["blob_locator"] = _canonical_blob_locator(receipt.sha256)
            artifact = enabled_specs[receipt.artifact_id]
            if urlparse(artifact.url).scheme == "":
                payload["final_url"] = artifact.url
            migrated_receipts.append(ArtifactReceipt.model_validate(payload))

        destination_receipt_root = (
            staging_root
            / "receipts"
            / manifest.source_id
            / manifest.release
            / manifest_sha256
        )
        _write_release_receipts(
            receipts=migrated_receipts,
            receipt_root=destination_receipt_root,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            schema_version=PORTABLE_RECEIPT_SCHEMA_VERSION,
        )
        _, staged_binding = load_snapshot_context_with_binding(
            manifest_file, staging_root
        )
        _publish_directory_no_replace(staging_root, destination_root)
        # Publishing is one same-filesystem directory rename.  The fully
        # verified, path-free binding therefore remains identical without a
        # post-publication failure window that could leave a reported error
        # after a valid target has already appeared.
        destination_binding = staged_binding
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    return {
        "status": "migrated_portable_receipts",
        "source_receipt_schema_version": source_schema_version,
        "destination_receipt_schema_version": PORTABLE_RECEIPT_SCHEMA_VERSION,
        "source_raw_root": str(source_root),
        "destination_raw_root": str(destination_root),
        "manifest_sha256": manifest_sha256,
        "artifact_count": len(snapshot.receipts),
        "source_release_receipt_sha256": source_binding["release_receipt_sha256"],
        "destination_release_receipt_sha256": destination_binding[
            "release_receipt_sha256"
        ],
        "destination_binding": destination_binding,
    }


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a staged directory without replacing any target.

    Windows already gives ``os.rename`` no-replace behavior.  Linux requires
    ``renameat2(RENAME_NOREPLACE)``; refusing an unavailable primitive is safer
    than silently weakening the fresh-destination contract.
    """

    if os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise ManifestError("migration destination appeared during staging") from exc
        return
    if not sys.platform.startswith("linux"):
        raise ManifestError(
            "atomic no-replace directory publication is unsupported on this platform"
        )

    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ManifestError("Linux renameat2 is unavailable; refusing unsafe publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ManifestError("migration destination appeared during staging")
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise ManifestError(
            "atomic no-replace directory publication is unavailable on this filesystem"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _write_release_receipts(
    *,
    receipts: list[ArtifactReceipt],
    receipt_root: Path,
    manifest: ReleaseManifest,
    manifest_sha256: str,
    schema_version: str,
) -> None:
    if schema_version != PORTABLE_RECEIPT_SCHEMA_VERSION:
        raise ManifestError("new release receipts must use portable schema version 2.0")
    release_artifacts = []
    for receipt in sorted(receipts, key=lambda item: item.artifact_id):
        if receipt.artifact_receipt_schema_version != schema_version:
            raise ManifestError("artifact and release receipt schema versions differ")
        receipt_path = receipt_root / (
            f"{receipt.artifact_id}.{receipt.sha256[:12]}.receipt.json"
        )
        write_json(receipt_path, receipt.model_dump(mode="json", exclude_none=True))
        release_artifacts.append(
            {
                "artifact_id": receipt.artifact_id,
                "artifact_spec_sha256": receipt.artifact_spec_sha256,
                "sha256": receipt.sha256,
                "byte_size": receipt.byte_size,
                "blob_uri": receipt.blob_uri,
                "receipt_file": receipt_path.name,
                "receipt_sha256": _sha256_file(receipt_path),
            }
        )
    write_json(
        receipt_root / "release.receipt.json",
        {
            "receipt_schema_version": schema_version,
            "source_id": manifest.source_id,
            "release": manifest.release,
            "manifest_sha256": manifest_sha256,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "artifact_receipts": release_artifacts,
        },
    )


def _canonical_blob_locator(sha256: str) -> str:
    return f"blobs/sha256/{sha256[:2]}/{sha256}"


def _canonical_blob_path(raw_root: Path, sha256: str) -> Path:
    root = Path(raw_root).resolve()
    candidate = (root / Path(*_canonical_blob_locator(sha256).split("/"))).resolve()
    if not candidate.is_relative_to(root):
        raise ManifestError("content-addressed blob path escapes the raw root")
    return candidate


def _validate_legacy_blob_path_suffix(receipt: ArtifactReceipt) -> None:
    """Validate an old absolute path without trusting its host-specific prefix."""

    assert receipt.blob_path is not None
    value = receipt.blob_path
    if "\x00" in value or not value:
        raise ManifestError("legacy blob_path is empty or contains NUL")
    normalized = value.replace("\\", "/")
    is_posix_absolute = normalized.startswith("/")
    is_windows_absolute = re.match(r"^[A-Za-z]:/", normalized) is not None
    if not (is_posix_absolute or is_windows_absolute):
        raise ManifestError("legacy blob_path is not an absolute host path")
    parts = normalized.split("/")
    if any(part in {".", ".."} for part in parts):
        raise ManifestError("legacy blob_path contains path traversal")
    expected_suffix = _canonical_blob_locator(receipt.sha256).split("/")
    if parts[-len(expected_suffix) :] != expected_suffix:
        raise ManifestError("legacy blob_path has a noncanonical digest suffix")


def _verify_blob_bytes(path: Path, receipt: ArtifactReceipt) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != receipt.byte_size
        or _sha256_file(path) != receipt.sha256
    ):
        raise ManifestError(f"artifact blob failed verification: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_model_hash(model: Any) -> str:
    payload = json.dumps(
        model.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _preflight_release(manifest: ReleaseManifest) -> None:
    if manifest.manifest_schema_version != "1.0":
        raise ManifestError("unsupported release manifest schema version")
    if manifest.release_is_mutable:
        raise ManifestError("refusing mutable release; resolve and lock it before fetching")
    if manifest.release.lower() in {"latest", "current", "unresolved"}:
        raise ManifestError("refusing a mutable release alias")
    if manifest.rights_status is not RightsStatus.cleared:
        raise ManifestError(
            f"refusing source with rights_status={manifest.rights_status.value}; review first"
        )
    enabled = [artifact for artifact in manifest.artifacts if artifact.enabled]
    if not enabled:
        raise ManifestError("release manifest has no enabled artifacts")
    if len(manifest.allowed_hosts) != len(set(manifest.allowed_hosts)):
        raise ManifestError("allowed_hosts contains duplicates")
    for artifact in enabled:
        if artifact.rights_status is not RightsStatus.cleared:
            raise ManifestError(
                f"artifact {artifact.artifact_id} is not rights-cleared"
            )
        if artifact.acquisition.method != "GET":
            raise ManifestError(
                f"artifact {artifact.artifact_id} uses unsupported acquisition method "
                f"{artifact.acquisition.method}; materialize the query result separately"
            )
        if (
            artifact.acquisition.parameters
            or artifact.acquisition.body_sha256
            or artifact.acquisition.pagination_state
        ):
            raise ManifestError(
                f"artifact {artifact.artifact_id} has an acquisition query that the "
                "byte fetcher does not execute; materialize and lock its result first"
            )
        parsed = urlparse(artifact.url)
        if parsed.scheme == "https" and parsed.hostname not in set(manifest.allowed_hosts):
            raise ManifestError(
                f"artifact {artifact.artifact_id} host is not allow-listed: "
                f"{parsed.hostname!r}"
            )
        if parsed.scheme not in {"", "https"}:
            raise ManifestError(
                f"artifact {artifact.artifact_id} has unsupported URL scheme: "
                f"{parsed.scheme!r}"
            )


class _AllowlistedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str], redirect_chain: list[str]) -> None:
        super().__init__()
        self.allowed_hosts = allowed_hosts
        self.redirect_chain = redirect_chain

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Any:
        parsed = urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts:
            raise ManifestError(
                f"refusing redirect to insecure or non-allow-listed destination: {newurl!r}"
            )
        self.redirect_chain.append(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def copy_snapshot_view(
    snapshot: SnapshotContext, target: str | Path, *, overwrite: bool = False
) -> None:
    """Create a readable view using paths resolved from the active snapshot root."""
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)
    for receipt in snapshot.receipts:
        destination = target_path / receipt.artifact_id
        if destination.exists() and not overwrite:
            raise ManifestError(f"snapshot target already exists: {destination}")
        shutil.copyfile(snapshot.resolve_receipt_path(receipt), destination)
