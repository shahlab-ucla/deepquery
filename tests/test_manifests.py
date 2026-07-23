from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from wormctx.manifests import (
    ManifestError,
    _AllowlistedRedirectHandler,
    copy_snapshot_view,
    fetch_release,
    load_release_manifest,
    load_snapshot_context,
    load_snapshot_context_with_binding,
    migrate_snapshot_receipts,
    _publish_directory_no_replace,
)
from wormctx.models import ArtifactReceipt


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class _FakeHttpResponse:
    """Minimal offline stand-in for the object returned by ``opener.open``."""

    def __init__(
        self,
        payload: bytes,
        final_url: str,
        status: int = 200,
        headers: dict | None = None,
    ) -> None:
        self._buffer = payload
        self._final_url = final_url
        self._status = status
        self.headers = headers or {"content-type": "application/octet-stream"}
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        chunk, self._buffer = self._buffer, b""
        return chunk

    def geturl(self) -> str:
        return self._final_url

    def getcode(self) -> int:
        return self._status

    def close(self) -> None:
        self.closed = True


def _remote_manifest_data(digest: str, byte_size: int, *, url: str, host: str) -> dict:
    data = json.loads((FIXTURES / "local_release.json").read_text(encoding="utf-8"))
    data["allowed_hosts"] = [host]
    data["artifacts"][0].update(
        {
            "url": url,
            "filename": "file.txt",
            "expected_sha256": digest,
            "expected_size": byte_size,
        }
    )
    return data


def _receipt_paths(raw: Path, receipt: ArtifactReceipt) -> tuple[Path, Path]:
    receipt_root = (
        raw
        / "receipts"
        / "wormctx-local-fixture"
        / "fixture-1"
        / receipt.manifest_sha256
    )
    artifact = next(
        path
        for path in receipt_root.glob("*.receipt.json")
        if path.name != "release.receipt.json"
    )
    return artifact, receipt_root / "release.receipt.json"


def _rewrite_receipt_hash(artifact_path: Path, release_path: Path) -> None:
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["artifact_receipts"][0]["receipt_sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    release_path.write_text(json.dumps(release), encoding="utf-8")


def _downgrade_to_legacy(
    raw: Path,
    receipt: ArtifactReceipt,
    blob_path: str | None = None,
    final_url: str | None = None,
) -> None:
    artifact_path, release_path = _receipt_paths(raw, receipt)
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    locator = value.pop("blob_locator")
    value.pop("artifact_receipt_schema_version")
    value["blob_path"] = blob_path or str((raw / Path(*locator.split("/"))).resolve())
    value["final_url"] = final_url or (FIXTURES / "source_bytes.txt").resolve().as_uri()
    artifact_path.write_text(json.dumps(value), encoding="utf-8")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["receipt_schema_version"] = "1.1"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    _rewrite_receipt_hash(artifact_path, release_path)


class ManifestTests(unittest.TestCase):
    def test_local_fixture_fetches_to_content_addressed_blob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipts = fetch_release(FIXTURES / "local_release.json", directory)
            self.assertEqual(len(receipts), 1)
            receipt = receipts[0]
            self.assertEqual(
                receipt.sha256,
                "f11034caaddd8c1579ecca29815067aacb1a09430ea36c833812ff269dc3b35a",
            )
            self.assertEqual(receipt.artifact_receipt_schema_version, "2.0")
            self.assertIsNone(receipt.blob_path)
            self.assertEqual(
                receipt.blob_locator,
                f"blobs/sha256/{receipt.sha256[:2]}/{receipt.sha256}",
            )
            snapshot = load_snapshot_context(FIXTURES / "local_release.json", directory)
            self.assertTrue(snapshot.artifact_path("source-bytes").is_file())
            receipt_file = (
                Path(directory)
                / "receipts"
                / "wormctx-local-fixture"
                / "fixture-1"
                / receipt.manifest_sha256
                / f"source-bytes.{receipt.sha256[:12]}.receipt.json"
            )
            self.assertTrue(receipt_file.is_file())
            self.assertEqual(receipt.blob_uri, f"urn:sha256:{receipt.sha256}")
            release_path = receipt_file.parent / "release.receipt.json"
            self.assertTrue(release_path.is_file())
            release = json.loads(release_path.read_text(encoding="utf-8"))
            self.assertEqual(release["receipt_schema_version"], "2.0")

    def test_mutable_manifest_is_refused(self) -> None:
        data = json.loads((FIXTURES / "local_release.json").read_text(encoding="utf-8"))
        data["release_is_mutable"] = True
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            source = Path(directory) / "source_bytes.txt"
            source.write_bytes((FIXTURES / "source_bytes.txt").read_bytes())
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ManifestError):
                fetch_release(manifest, Path(directory) / "raw")

    def test_rights_gate_is_enforced(self) -> None:
        data = json.loads((FIXTURES / "local_release.json").read_text(encoding="utf-8"))
        data["rights_status"] = "review_required"
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            source = Path(directory) / "source_bytes.txt"
            source.write_bytes((FIXTURES / "source_bytes.txt").read_bytes())
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ManifestError):
                fetch_release(manifest, Path(directory) / "raw")

    def test_manifest_schema_loads(self) -> None:
        manifest = load_release_manifest(FIXTURES / "local_release.json")
        self.assertEqual(manifest.release, "fixture-1")

    def test_path_unsafe_source_id_is_rejected(self) -> None:
        data = json.loads((FIXTURES / "local_release.json").read_text(encoding="utf-8"))
        data["source_id"] = "../escape"
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_release_manifest(manifest)

    def test_release_alias_is_rejected_even_if_claimed_immutable(self) -> None:
        data = json.loads((FIXTURES / "local_release.json").read_text(encoding="utf-8"))
        data["release"] = "latest"
        data["release_is_mutable"] = False
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_release_manifest(manifest)

    def test_artifact_level_rights_gate_is_enforced(self) -> None:
        data = json.loads((FIXTURES / "local_release.json").read_text(encoding="utf-8"))
        data["artifacts"][0]["rights_status"] = "review_required"
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            source = Path(directory) / "source_bytes.txt"
            source.write_bytes((FIXTURES / "source_bytes.txt").read_bytes())
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ManifestError):
                fetch_release(manifest, Path(directory) / "raw")

    def test_redirect_handler_rejects_each_bad_hop_before_following(self) -> None:
        handler = _AllowlistedRedirectHandler({"allowed.example"}, [])
        for url in ("http://allowed.example/file", "https://evil.example/file"):
            with self.subTest(url=url), self.assertRaises(ManifestError):
                handler.redirect_request(None, None, 302, "Found", {}, url)

    def test_relative_local_path_escape_is_rejected(self) -> None:
        data = json.loads((FIXTURES / "local_release.json").read_text(encoding="utf-8"))
        data["artifacts"][0]["url"] = "../outside.txt"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "manifest"
            manifest_dir.mkdir()
            (root / "outside.txt").write_text("synthetic source bytes\n", encoding="utf-8")
            manifest = manifest_dir / "manifest.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "escapes"):
                fetch_release(manifest, root / "raw")

    def test_failed_release_does_not_write_partial_receipts(self) -> None:
        data = json.loads((FIXTURES / "local_release.json").read_text(encoding="utf-8"))
        second = dict(data["artifacts"][0])
        second.update(
            {
                "artifact_id": "missing-bytes",
                "url": "missing.txt",
                "filename": "missing.txt",
                "expected_sha256": "0" * 64,
            }
        )
        data["artifacts"].append(second)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            (root / "source_bytes.txt").write_bytes(
                (FIXTURES / "source_bytes.txt").read_bytes()
            )
            manifest.write_text(json.dumps(data), encoding="utf-8")
            raw = root / "raw"
            with self.assertRaises(ManifestError):
                fetch_release(manifest, raw)
            self.assertFalse((raw / "receipts").exists())

    def test_missing_expected_hash_is_rejected(self) -> None:
        data = json.loads((FIXTURES / "local_release.json").read_text(encoding="utf-8"))
        del data["artifacts"][0]["expected_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_release_manifest(manifest)

    def test_snapshot_receipt_rejects_path_traversal_locator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            receipts = fetch_release(FIXTURES / "local_release.json", raw)
            artifact_receipt, release_path = _receipt_paths(raw, receipts[0])
            value = json.loads(artifact_receipt.read_text(encoding="utf-8"))
            value["blob_locator"] = "../source_bytes.txt"
            artifact_receipt.write_text(json.dumps(value), encoding="utf-8")
            _rewrite_receipt_hash(artifact_receipt, release_path)
            with self.assertRaisesRegex(ManifestError, "blob locator"):
                load_snapshot_context(FIXTURES / "local_release.json", raw)

    def test_snapshot_receipt_must_match_locked_artifact_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            receipts = fetch_release(FIXTURES / "local_release.json", raw)
            receipt_dir = (
                raw
                / "receipts"
                / "wormctx-local-fixture"
                / "fixture-1"
                / receipts[0].manifest_sha256
            )
            artifact_receipt = next(
                path
                for path in receipt_dir.glob("*.receipt.json")
                if path.name != "release.receipt.json"
            )
            value = json.loads(artifact_receipt.read_text(encoding="utf-8"))
            value["role"] = "tampered-role"
            artifact_receipt.write_text(json.dumps(value), encoding="utf-8")
            release_path = receipt_dir / "release.receipt.json"
            release = json.loads(release_path.read_text(encoding="utf-8"))
            release["artifact_receipts"][0]["receipt_sha256"] = hashlib.sha256(
                artifact_receipt.read_bytes()
            ).hexdigest()
            release_path.write_text(json.dumps(release), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "locked ArtifactSpec"):
                load_snapshot_context(FIXTURES / "local_release.json", raw)

    def test_release_receipt_tuple_must_match_artifact_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            receipts = fetch_release(FIXTURES / "local_release.json", raw)
            release_path = (
                raw
                / "receipts"
                / "wormctx-local-fixture"
                / "fixture-1"
                / receipts[0].manifest_sha256
                / "release.receipt.json"
            )
            release = json.loads(release_path.read_text(encoding="utf-8"))
            release["artifact_receipts"][0]["byte_size"] += 1
            release_path.write_text(json.dumps(release), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "release receipt tuple"):
                load_snapshot_context(FIXTURES / "local_release.json", raw)

    # --- Content-integrity and remote-fetch safety (the core safety surface) ---

    def test_checksum_drift_at_fetch_is_rejected(self) -> None:
        """A local file whose bytes diverge from the lock must be refused."""
        data = json.loads((FIXTURES / "local_release.json").read_text(encoding="utf-8"))
        data["artifacts"][0]["expected_sha256"] = "a" * 64  # well-formed but wrong
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source_bytes.txt").write_bytes(
                (FIXTURES / "source_bytes.txt").read_bytes()
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "checksum mismatch"):
                fetch_release(manifest, root / "raw")
            self.assertFalse((root / "raw" / "blobs").exists())

    def test_stored_blob_corruption_is_detected_on_snapshot_load(self) -> None:
        """A content-addressed blob mutated after fetch must fail verification."""
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            receipts = fetch_release(FIXTURES / "local_release.json", raw)
            assert receipts[0].blob_locator is not None
            blob = raw / Path(*receipts[0].blob_locator.split("/"))
            blob.write_bytes(b"corrupted bytes that do not match the receipt")
            with self.assertRaisesRegex(ManifestError, "blob failed verification"):
                load_snapshot_context(FIXTURES / "local_release.json", raw)

    def test_portable_snapshot_survives_cross_root_move(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original-raw"
            receipts = fetch_release(FIXTURES / "local_release.json", original)
            _, original_binding = load_snapshot_context_with_binding(
                FIXTURES / "local_release.json", original
            )
            moved = root / "moved-raw"
            shutil.move(original, moved)

            snapshot, moved_binding = load_snapshot_context_with_binding(
                FIXTURES / "local_release.json", moved
            )
            artifact = snapshot.artifact_path(receipts[0].artifact_id)
            self.assertTrue(artifact.is_relative_to(moved.resolve()))
            self.assertEqual(moved_binding, original_binding)

            view = root / "view"
            copy_snapshot_view(snapshot, view)
            self.assertEqual(
                (view / receipts[0].artifact_id).read_bytes(),
                artifact.read_bytes(),
            )

    def test_portable_snapshot_survives_manifest_and_raw_tree_move(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "windows-checkout"
            fixture_copy = original / "fixtures"
            fixture_copy.mkdir(parents=True)
            shutil.copyfile(FIXTURES / "local_release.json", fixture_copy / "release.json")
            shutil.copyfile(FIXTURES / "source_bytes.txt", fixture_copy / "source_bytes.txt")
            manifest = fixture_copy / "release.json"
            raw = original / "raw"
            receipts = fetch_release(manifest, raw)
            receipt_path, _ = _receipt_paths(raw, receipts[0])
            receipt_value = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt_value["final_url"], "source_bytes.txt")
            self.assertNotIn(str(original), receipt_path.read_text(encoding="utf-8"))

            moved = root / "linux-checkout"
            shutil.move(original, moved)
            snapshot = load_snapshot_context(
                moved / "fixtures" / "release.json", moved / "raw"
            )
            self.assertEqual(
                snapshot.artifact_path(receipts[0].artifact_id).read_bytes(),
                (moved / "fixtures" / "source_bytes.txt").read_bytes(),
            )

    def test_legacy_receipt_loads_only_at_its_original_canonical_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "legacy-raw"
            receipts = fetch_release(FIXTURES / "local_release.json", original)
            _downgrade_to_legacy(original, receipts[0])
            snapshot = load_snapshot_context(FIXTURES / "local_release.json", original)
            self.assertTrue(snapshot.artifact_path(receipts[0].artifact_id).is_file())

            copied = root / "copied-legacy-raw"
            shutil.copytree(original, copied)
            with self.assertRaisesRegex(ManifestError, "active blob store"):
                load_snapshot_context(FIXTURES / "local_release.json", copied)

    def test_migration_recovers_copied_cross_os_legacy_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "copied-windows-snapshot"
            receipts = fetch_release(FIXTURES / "local_release.json", source)
            digest = receipts[0].sha256
            stale_windows_path = (
                rf"D:\archived\raw\blobs\sha256\{digest[:2]}\{digest}"
            )
            _downgrade_to_legacy(
                source,
                receipts[0],
                stale_windows_path,
                "file:///D:/archived/fixture/source_bytes.txt",
            )
            with self.assertRaisesRegex(
                ManifestError, "invalid source path|active blob store|not absolute"
            ):
                load_snapshot_context(FIXTURES / "local_release.json", source)

            source_receipt, source_release = _receipt_paths(source, receipts[0])
            source_receipt_bytes = source_receipt.read_bytes()
            source_release_bytes = source_release.read_bytes()
            destination = root / "portable-snapshot"
            result = migrate_snapshot_receipts(
                FIXTURES / "local_release.json", source, destination
            )
            self.assertEqual(result["source_receipt_schema_version"], "1.1")
            self.assertEqual(result["destination_receipt_schema_version"], "2.0")
            self.assertEqual(source_receipt.read_bytes(), source_receipt_bytes)
            self.assertEqual(source_release.read_bytes(), source_release_bytes)

            snapshot = load_snapshot_context(
                FIXTURES / "local_release.json", destination
            )
            self.assertEqual(
                snapshot.artifact_path(receipts[0].artifact_id).read_bytes(),
                (FIXTURES / "source_bytes.txt").read_bytes(),
            )
            migrated_receipt, migrated_release = _receipt_paths(
                destination, receipts[0]
            )
            migrated = json.loads(migrated_receipt.read_text(encoding="utf-8"))
            self.assertEqual(migrated["artifact_receipt_schema_version"], "2.0")
            self.assertNotIn("blob_path", migrated)
            self.assertEqual(migrated["final_url"], "source_bytes.txt")
            self.assertEqual(
                json.loads(migrated_release.read_text(encoding="utf-8"))[
                    "receipt_schema_version"
                ],
                "2.0",
            )

    def test_migration_rejects_legacy_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy-raw"
            receipts = fetch_release(FIXTURES / "local_release.json", source)
            digest = receipts[0].sha256
            unsafe_path = (
                rf"D:\archived\raw\blobs\sha256\{digest[:2]}\..\{digest}"
            )
            _downgrade_to_legacy(source, receipts[0], unsafe_path)
            destination = root / "destination"
            with self.assertRaisesRegex(ManifestError, "path traversal"):
                migrate_snapshot_receipts(
                    FIXTURES / "local_release.json", source, destination
                )
            self.assertFalse(destination.exists())

    def test_atomic_publication_never_replaces_an_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "verified-stage"
            destination = root / "appeared-destination"
            source.mkdir()
            destination.mkdir()
            (source / "source-marker").write_text("source", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "destination appeared"):
                _publish_directory_no_replace(source, destination)
            self.assertTrue((source / "source-marker").is_file())
            self.assertEqual(list(destination.iterdir()), [])

    def test_migration_rejects_noncanonical_legacy_digest_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy-raw"
            receipts = fetch_release(FIXTURES / "local_release.json", source)
            digest = receipts[0].sha256
            wrong_prefix = "00" if digest[:2] != "00" else "ff"
            bad_path = rf"D:\archived\raw\blobs\sha256\{wrong_prefix}\{digest}"
            _downgrade_to_legacy(source, receipts[0], bad_path)
            destination = root / "destination"
            with self.assertRaisesRegex(ManifestError, "noncanonical digest suffix"):
                migrate_snapshot_receipts(
                    FIXTURES / "local_release.json", source, destination
                )
            self.assertFalse(destination.exists())

    def test_migration_rejects_tampered_blob_without_publishing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy-raw"
            receipts = fetch_release(FIXTURES / "local_release.json", source)
            _downgrade_to_legacy(source, receipts[0])
            digest = receipts[0].sha256
            blob = source / "blobs" / "sha256" / digest[:2] / digest
            blob.write_bytes(b"tampered")
            destination = root / "destination"
            with self.assertRaisesRegex(ManifestError, "blob failed verification"):
                migrate_snapshot_receipts(
                    FIXTURES / "local_release.json", source, destination
                )
            self.assertFalse(destination.exists())

    def test_size_limit_is_enforced_during_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ManifestError, "byte limit"):
                fetch_release(
                    FIXTURES / "local_release.json",
                    Path(directory) / "raw",
                    max_bytes=1,
                )

    def test_http_scheme_url_is_rejected_by_preflight(self) -> None:
        data = _remote_manifest_data(
            "a" * 64, 1, url="http://example.org/file.txt", host="example.org"
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "unsupported URL scheme"):
                fetch_release(manifest, Path(directory) / "raw")

    def test_https_host_not_allow_listed_is_rejected_by_preflight(self) -> None:
        data = _remote_manifest_data(
            "a" * 64, 1, url="https://evil.example/file.txt", host="good.example"
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "not allow-listed"):
                fetch_release(manifest, Path(directory) / "raw")

    def test_https_branch_fetches_via_opener_and_records_provenance(self) -> None:
        payload = b"synthetic remote artifact bytes\n"
        digest = hashlib.sha256(payload).hexdigest()
        data = _remote_manifest_data(
            digest, len(payload), url="https://good.example/file.txt", host="good.example"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            response = _FakeHttpResponse(payload, "https://good.example/file.txt")
            with patch("wormctx.manifests.build_opener") as build_opener:
                build_opener.return_value.open.return_value = response
                receipts = fetch_release(manifest, root / "raw")
            self.assertEqual(len(receipts), 1)
            receipt = receipts[0]
            self.assertEqual(receipt.sha256, digest)
            self.assertEqual(receipt.response_status, 200)
            self.assertEqual(receipt.final_url, "https://good.example/file.txt")
            self.assertEqual(receipt.redirect_chain, ["https://good.example/file.txt"])
            self.assertTrue(response.closed)
            snapshot = load_snapshot_context(manifest, root / "raw")
            self.assertTrue(snapshot.artifact_path(receipt.artifact_id).is_file())

    def test_https_post_redirect_destination_is_revalidated(self) -> None:
        payload = b"synthetic remote artifact bytes\n"
        digest = hashlib.sha256(payload).hexdigest()
        data = _remote_manifest_data(
            digest, len(payload), url="https://good.example/file.txt", host="good.example"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            # Opener resolves to a final URL whose host is not allow-listed.
            response = _FakeHttpResponse(payload, "https://evil.example/file.txt")
            with patch("wormctx.manifests.build_opener") as build_opener:
                build_opener.return_value.open.return_value = response
                with self.assertRaisesRegex(ManifestError, "non-allow-listed destination"):
                    fetch_release(manifest, root / "raw")
            self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
