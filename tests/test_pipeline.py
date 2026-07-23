from __future__ import annotations

import hashlib
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wormctx.io import iter_jsonl, load_observations, read_json, write_jsonl
from wormctx.manifests import fetch_release
from wormctx.pipeline import (
    GRAPH_PRODUCT_FILES,
    build_graph,
    build_transport_map,
    normalize_snapshot,
    verify_build_receipt,
    verify_normalization_receipt,
    write_build_receipt,
    _publish_staged_files,
)


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_fetch_to_normalize_provenance_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "observations.jsonl"
            source.write_bytes(
                (ROOT / "data" / "examples" / "observations.jsonl").read_bytes()
            )
            payload = source.read_bytes()
            manifest_data = json.loads(
                (ROOT / "tests" / "fixtures" / "local_release.json").read_text(
                    encoding="utf-8"
                )
            )
            artifact = manifest_data["artifacts"][0]
            artifact.update(
                {
                    "url": "observations.jsonl",
                    "filename": "observations.jsonl",
                    "role": "canonical_observation_fixture",
                    "media_type": "application/x-ndjson",
                    "expected_sha256": hashlib.sha256(payload).hexdigest(),
                    "expected_size": len(payload),
                }
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
            raw = root / "raw"
            fetched = fetch_release(manifest, raw)
            output = root / "normalized" / "observations.jsonl"
            result = normalize_snapshot(manifest, raw, output)
            observations = load_observations(output)
            self.assertEqual(len(observations), 5)
            self.assertTrue(
                all(
                    item.provenance.source_id == "wormctx-local-fixture"
                    and item.provenance.checksum_sha256 == fetched[0].sha256
                    for item in observations
                )
            )
            normalization_receipt = read_json(result["normalization_receipt"])
            self.assertEqual(
                normalization_receipt["manifest_sha256"], fetched[0].manifest_sha256
            )
            self.assertEqual(
                normalization_receipt["output_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            binding = verify_normalization_receipt(
                output, result["normalization_receipt"]
            )
            self.assertEqual(binding["status"], "byte_bound_only")
            self.assertEqual(binding["manifest_sha256"], fetched[0].manifest_sha256)
            full_binding = verify_normalization_receipt(
                output,
                result["normalization_receipt"],
                manifest_path=manifest,
                raw_root=raw,
            )
            self.assertEqual(full_binding["status"], "full_provenance_verified")
            self.assertEqual(
                full_binding["normalization_receipt_sha256"],
                hashlib.sha256(
                    Path(result["normalization_receipt"]).read_bytes()
                ).hexdigest(),
            )

            tampered_receipt = root / "tampered-normalization-receipt.json"
            tampered = read_json(result["normalization_receipt"])
            tampered["release_receipt_sha256"] = "0" * 64
            tampered_receipt.write_text(json.dumps(tampered), encoding="utf-8")
            byte_binding = verify_normalization_receipt(output, tampered_receipt)
            self.assertEqual(byte_binding["status"], "byte_bound_only")
            with self.assertRaisesRegex(ValueError, "fully verified source snapshot"):
                verify_normalization_receipt(
                    output,
                    tampered_receipt,
                    manifest_path=manifest,
                    raw_root=raw,
                )

            unknown_field_receipt = root / "unknown-field-receipt.json"
            tampered = read_json(result["normalization_receipt"])
            tampered["unrecognized"] = True
            unknown_field_receipt.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_normalization_receipt(output, unknown_field_receipt)

            build_output = root / "build"
            build_summary = build_graph(output, build_output)
            build_summary["input_binding"] = full_binding
            build_receipt = write_build_receipt(build_output, build_summary)
            normalization_bindings = build_receipt["bindings"]["normalization"]
            self.assertEqual(len(normalization_bindings), 1)
            self.assertEqual(
                normalization_bindings[0]["status"], "full_provenance_verified"
            )
            self.assertEqual(
                normalization_bindings[0]["manifest_sha256"],
                fetched[0].manifest_sha256,
            )
            self.assertEqual(
                build_receipt["bindings"]["inputs"]["observations_sha256"],
                [hashlib.sha256(output.read_bytes()).hexdigest()],
            )
            build_verification = verify_build_receipt(
                build_output,
                local_inputs={"observations_sha256": [output]},
                normalization_receipt_path=result["normalization_receipt"],
                manifest_path=manifest,
                raw_root=raw,
            )
            self.assertEqual(
                build_verification["normalization"]["status"],
                "full_provenance_verified",
            )

            # A self-consistent but forged normalization receipt is only byte
            # bound. Full verification must deterministically replay the adapter.
            forged_rows = output.read_text(encoding="utf-8").splitlines()
            forged_first = json.loads(forged_rows[0])
            forged_first["record_origin"] = "extracted"
            forged_rows[0] = json.dumps(
                forged_first, sort_keys=True, ensure_ascii=False
            )
            output.write_text("\n".join(forged_rows) + "\n", encoding="utf-8")
            forged_receipt = read_json(result["normalization_receipt"])
            forged_receipt["output_sha256"] = hashlib.sha256(
                output.read_bytes()
            ).hexdigest()
            forged_receipt_path = root / "forged-normalization-receipt.json"
            forged_receipt_path.write_text(
                json.dumps(forged_receipt), encoding="utf-8"
            )
            self.assertEqual(
                verify_normalization_receipt(output, forged_receipt_path)["status"],
                "byte_bound_only",
            )
            with self.assertRaisesRegex(ValueError, "deterministic replay"):
                verify_normalization_receipt(
                    output,
                    forged_receipt_path,
                    manifest_path=manifest,
                    raw_root=raw,
                )

    def test_end_to_end_demo_products(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "stale-unrelated.txt").write_text("stale", encoding="utf-8")
            graph = build_graph(
                ROOT / "data" / "examples" / "observations.jsonl", output
            )
            self.assertFalse((output / "stale-unrelated.txt").exists())
            (output / "injected-between-stages.txt").write_text(
                "stale", encoding="utf-8"
            )
            transport = build_transport_map(
                ROOT / "data" / "examples" / "observations.jsonl",
                ROOT / "data" / "examples" / "query_context.json",
                ROOT / "config" / "transport_policy.json",
                output,
                carry_forward_files=graph["produced_files"],
            )
            self.assertFalse((output / "injected-between-stages.txt").exists())
            summary = {
                "graph": graph,
                "transport": transport,
                "input_binding": {
                    "status": "unbound_bundled_synthetic_fixture",
                    "warning": "synthetic fixture",
                },
            }
            receipt = write_build_receipt(output, summary)
            self.assertEqual(graph["observation_count"], 5)
            self.assertEqual(graph["kgx_edge_count"], 0)
            self.assertEqual(
                graph["kgx_projection_policy"]["mode"],
                "standard_strict_skip_blocked",
            )
            self.assertEqual(transport["score_count"], 5)
            self.assertEqual(transport["classification_counts"]["direct"], 1)
            for filename in (
                "validated_observations.jsonl",
                "contextual_nodes.tsv",
                "contextual_edges.tsv",
                "kgx_nodes.tsv",
                "kgx_edges.tsv",
                "kgx_context_sidecar.jsonl",
                "coverage_long.csv",
                "coverage_cube.csv",
                "transport_scores.csv",
                "transport_scores.json",
                "transport_map.html",
                "build_receipt.json",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            self.assertGreater(len(receipt["files"]), 0)
            receipted_paths = {item["path"] for item in receipt["files"]}
            self.assertNotIn("stale-unrelated.txt", receipted_paths)
            self.assertEqual(
                receipted_paths,
                set(graph["produced_files"]) | set(transport["produced_files"]),
            )
            self.assertEqual(
                receipt["bindings"]["inputs"]["query_sha256"],
                [hashlib.sha256(
                    (ROOT / "data" / "examples" / "query_context.json").read_bytes()
                ).hexdigest()],
            )
            self.assertEqual(
                receipt["bindings"]["inputs"]["policy_sha256"],
                [hashlib.sha256(
                    (ROOT / "config" / "transport_policy.json").read_bytes()
                ).hexdigest()],
            )
            self.assertTrue(receipt["bindings"]["schema_sha256"])
            self.assertTrue(receipt["bindings"]["mapping_sha256"])
            self.assertEqual(len(receipt["bindings"]["code_sha256"]), 64)
            verification = verify_build_receipt(
                output,
                local_inputs={
                    "observations_sha256": [
                        ROOT / "data" / "examples" / "observations.jsonl"
                    ],
                    "query_sha256": [
                        ROOT / "data" / "examples" / "query_context.json"
                    ],
                    "policy_sha256": [ROOT / "config" / "transport_policy.json"],
                },
            )
            self.assertEqual(
                verification["status"], "authoritative_build_snapshot_verified"
            )
            self.assertEqual(verification["unchecked_inputs"], [])

    def test_demo_receipt_can_record_an_explicit_unbound_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            graph = build_graph(
                ROOT / "data" / "examples" / "observations.jsonl", output
            )
            summary = {
                "graph": graph,
                "input_binding": {
                    "status": "unbound_bundled_synthetic_fixture",
                    "warning": "synthetic fixture",
                },
            }
            receipt = write_build_receipt(output, summary)
            self.assertEqual(
                receipt["bindings"]["normalization"][0]["status"],
                "unbound_bundled_synthetic_fixture",
            )

    def test_graph_build_keeps_canonical_unmapped_claim_and_blocks_kgx_only(self) -> None:
        row = copy.deepcopy(
            next(iter_jsonl(ROOT / "data" / "examples" / "observations.jsonl"))
        )
        row["predicate"] = "RO:0000000"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "unmapped.jsonl"
            write_jsonl(source, [row])
            output = root / "graph"
            summary = build_graph(source, output)
            self.assertEqual(summary["observation_count"], 1)
            self.assertEqual(summary["kgx_edge_count"], 0)
            validated = load_observations(output / "validated_observations.jsonl")
            self.assertEqual(validated[0].predicate, "RO:0000000")
            sidecar = list(iter_jsonl(output / "kgx_context_sidecar.jsonl"))
            self.assertEqual(
                sidecar[0]["projection_status"], "blocked_unmapped_predicate"
            )

    def test_build_receipt_refuses_directory_scan_without_declared_products(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "unrelated.txt").write_text("unrelated", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no explicit produced_files"):
                write_build_receipt(output, {"observation_count": 0})

    def test_build_receipt_rejects_snapshot_and_binding_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "build"
            graph = build_graph(
                ROOT / "data" / "examples" / "observations.jsonl", output
            )
            summary = {
                "graph": graph,
                "input_binding": {
                    "status": "unbound_bundled_synthetic_fixture",
                    "warning": "synthetic fixture",
                },
            }
            write_build_receipt(output, summary)
            receipt_path = output / "build_receipt.json"
            original_receipt = receipt_path.read_bytes()

            (output / "unrelated.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or extra"):
                verify_build_receipt(output)
            (output / "unrelated.txt").unlink()

            product = output / "kgx_nodes.tsv"
            original_product = product.read_bytes()
            product.write_bytes(original_product + b"tampered")
            with self.assertRaisesRegex(ValueError, "byte verification"):
                verify_build_receipt(output)
            product.write_bytes(original_product)

            product.unlink()
            with self.assertRaisesRegex(ValueError, "missing or extra"):
                verify_build_receipt(output)
            product.write_bytes(original_product)

            tampered_receipt = read_json(receipt_path)
            tampered_receipt["unexpected"] = True
            receipt_path.write_text(json.dumps(tampered_receipt), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_build_receipt(output)
            receipt_path.write_bytes(original_receipt)

            tampered_receipt = read_json(receipt_path)
            tampered_receipt["bindings"]["code_sha256"] = "0" * 64
            receipt_path.write_text(json.dumps(tampered_receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "code hash"):
                verify_build_receipt(output)
            receipt_path.write_bytes(original_receipt)

            # The per-file code binding must be independently tamper-evident even
            # when the aggregate code_sha256 is left untouched.
            tampered_receipt = read_json(receipt_path)
            some_file = sorted(tampered_receipt["bindings"]["code_file_sha256"])[0]
            tampered_receipt["bindings"]["code_file_sha256"][some_file] = "0" * 64
            receipt_path.write_text(json.dumps(tampered_receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "code-file hashes"):
                verify_build_receipt(output)
            receipt_path.write_bytes(original_receipt)

            wrong_input = output.parent / "wrong-observations.jsonl"
            wrong_input.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input binding"):
                verify_build_receipt(
                    output,
                    local_inputs={"observations_sha256": [wrong_input]},
                )

    def test_directory_publish_restores_previous_snapshot_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "published"
            output.mkdir()
            (output / "previous.txt").write_text("previous", encoding="utf-8")
            staging = parent / "staging"
            staging.mkdir()
            for filename in GRAPH_PRODUCT_FILES:
                (staging / filename).write_text(filename, encoding="utf-8")
            real_replace = os.replace

            def replace_with_publish_failure(source: object, destination: object) -> None:
                if Path(source).resolve() == staging.resolve():
                    raise OSError("synthetic publish failure")
                real_replace(source, destination)

            with patch(
                "wormctx.pipeline.os.replace", side_effect=replace_with_publish_failure
            ), self.assertRaisesRegex(OSError, "synthetic publish failure"):
                _publish_staged_files(
                    staging, output, sorted(GRAPH_PRODUCT_FILES)
                )
            self.assertEqual(
                (output / "previous.txt").read_text(encoding="utf-8"), "previous"
            )
            self.assertFalse((output / "kgx_nodes.tsv").exists())


if __name__ == "__main__":
    unittest.main()
