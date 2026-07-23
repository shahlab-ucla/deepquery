from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from pydantic import ValidationError

from wormctx.poc import qtl_heldout_prediction as hp


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/natural_variation/heldout_strain_prediction/config/heldout_prediction.json"
HANDOFF = ROOT / "experiments/natural_variation/pangenome_state_qualification/config/state_input_handoff.json"
FULL_KERNEL_ROOT = (
    ROOT
    / "build"
    / "kernel-results"
    / "ws283-full-kernels-20260722T181148Z-e283182e53f6"
    / "bundle"
)


def _synthetic_dataset() -> hp.PredictionDataset:
    rng = np.random.default_rng(20260721)
    sample_ids = tuple(f"strain_{index:03d}" for index in range(hp.SAMPLE_COUNT))
    groups = tuple(f"ancestry_{index % 5}" for index in range(hp.SAMPLE_COUNT))
    genome = rng.normal(size=(hp.SAMPLE_COUNT, 8))
    excluding = genome[:, :5]
    haplotypes = np.zeros((hp.SAMPLE_COUNT, 3), dtype=np.float64)
    haplotypes[np.arange(hp.SAMPLE_COUNT), np.arange(hp.SAMPLE_COUNT) % 3] = 1.0
    whole_kernel = genome @ genome.T / genome.shape[1] + np.eye(hp.SAMPLE_COUNT) * 0.01
    excluding_kernel = (
        excluding @ excluding.T / excluding.shape[1] + np.eye(hp.SAMPLE_COUNT) * 0.01
    )
    group_effect = np.asarray([int(value.rsplit("_", 1)[1]) for value in groups]) * 0.1
    base = 0.7 * genome[:, 0] + 0.5 * haplotypes[:, 1] + group_effect
    phenotypes = {
        trait: base * (1.0 + index * 0.2) + rng.normal(scale=0.2, size=hp.SAMPLE_COUNT)
        for index, trait in enumerate(hp.TRAITS)
    }
    return hp.PredictionDataset(
        sample_ids=sample_ids,
        population_groups=groups,
        phenotypes=phenotypes,
        whole_genome_kernel=whole_kernel,
        genome_excluding_chrv_kernel=excluding_kernel,
        chrv_haplotype_features=haplotypes,
    )


def _synthetic_receipt(dataset: hp.PredictionDataset) -> hp.PredictionInputReceipt:
    hashes = hp.dataset_hashes(dataset)
    return hp.PredictionInputReceipt(
        schema_version=hp.INPUT_RECEIPT_VERSION,
        mode="synthetic_fixture",
        sample_count=209,
        **hashes,
        genotype_feature_mask_scope=hp.FEATURE_MASK_SCOPE,
        legacy_analysis_id=None,
        legacy_input_paths=[],
        reported_source_sample_count=None,
        reported_aligned_finite_count=None,
    )


def _write_asset(root: Path, root_id: str, relative_path: str, role: str, data: bytes) -> dict:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "role": role,
        "root_id": root_id,
        "relative_path": relative_path,
        "bytes": len(data),
        "sha256": hp._sha256(path),
    }


def _json_bytes(payload: dict) -> bytes:
    return hp._canonical_bytes(payload)


def _real_fixture(
    root: Path, dataset: hp.PredictionDataset
) -> tuple[hp.HeldoutPredictionManifest, hp.PredictionInputReceipt, dict[str, Path]]:
    roots = {
        "modern_parent": root / "modern",
        "parametric_null_run": root / "parametric_null",
        "regional_state": root / "regional_state",
        "full_kernels": root / "kernels",
        "repository": root / "repository",
    }
    for path in roots.values():
        path.mkdir(parents=True)
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    modern = roots["modern_parent"]
    fam_bytes = "".join(
        f"{sample}\t{sample}\t0\t0\t0\t-9\n" for sample in dataset.sample_ids
    ).encode()
    modern_assets = [
        _write_asset(modern, "modern_parent", "genotype/abamectin_209_qc.bed", "bed", b"bed\n"),
        _write_asset(modern, "modern_parent", "genotype/abamectin_209_qc.bim", "bim", b"bim\n"),
        _write_asset(
            modern, "modern_parent", "genotype/abamectin_209_qc.fam", "fam", fam_bytes
        ),
    ]
    for trait in hp.TRAITS:
        values = np.asarray(dataset.phenotypes[trait], dtype=np.float64)
        content = "".join(
            f"{sample}\t{sample}\t{format(float(value), '.17g')}\n"
            for sample, value in zip(dataset.sample_ids, values, strict=True)
        ).encode()
        modern_assets.append(
            _write_asset(
                modern,
                "modern_parent",
                f"prepared/cohort/{hp.TRAIT_SLUGS[trait]}.phen",
                f"phenotype:{trait}",
                content,
            )
        )
    payload["modern_cohort"]["required_files"] = modern_assets

    parametric_null = roots["parametric_null_run"]
    rich_summary = {
        "schema_version": hp.PARAMETRIC_NULL_SUMMARY_VERSION,
        "analysis_id": "abamectin_qtl_ws283_parametric_polygenic_null_v1",
        "status": "preliminary_model_specific_null_calibration_complete",
        "manifest_sha256": "a" * 64,
        "validated": False,
        "biological_claims_permitted": False,
        "design": {"cells": 16, "replicates_per_cell": 100, "total_null_maps": 1600},
        "cells": [{"cell_id": f"cell_{index:02d}"} for index in range(16)],
    }
    rich_asset = _write_asset(
        parametric_null,
        "parametric_null_run",
        "summary/ws283_parametric_null/ws283_parametric_null_summary.json",
        "parametric_null_rich_summary",
        _json_bytes(rich_summary),
    )
    aggregation = {
        "schema_version": hp.PARAMETRIC_NULL_AGGREGATION_VERSION,
        "manifest_sha256": "a" * 64,
        "calibration_cells": 16,
        "replicates_per_cell": 100,
        "attempted_null_maps": 1600,
        "completed_null_maps": 1600,
        "validated": False,
        "biological_claims_permitted": False,
        "rich_summary": {"path": rich_asset["relative_path"], "sha256": rich_asset["sha256"]},
        "cells": [
            {
                "cell_id": f"cell_{index:02d}",
                "replicates_attempted": 100,
                "replicates_completed": 100,
            }
            for index in range(16)
        ],
    }
    parametric_null_gate = payload["upstream_gates"]["parametric_null_terminal"]
    parametric_null_gate["success_marker"] = _write_asset(
        parametric_null, "parametric_null_run", "SUCCESS", "parametric_null_terminal_success", b"SUCCESS\n"
    )
    parametric_null_gate["aggregation_summary"] = _write_asset(
        parametric_null,
        "parametric_null_run",
        "summary/parametric_null_aggregation.json",
        "parametric_null_aggregation_summary",
        _json_bytes(aggregation),
    )
    parametric_null_gate["rich_summary"] = rich_asset

    regional_state = roots["regional_state"]
    regions = []
    for region_id, start, end in (
        ("chrv_left", 1_747_612, 4_333_001),
        ("chrv_right", 13_606_517, 16_754_986),
    ):
        regions.append(
            {
                "region_id": region_id,
                "source_assembly": hp.WS276_ASSEMBLY,
                "target_assembly": hp.WS283_ASSEMBLY,
                "source_chromosome": 5,
                "target_chromosome": 5,
                "source_start": start,
                "source_end": end,
                "target_start": start,
                "target_end": end,
                "mapping_mode": "sequence_identical_coordinate_identity",
                "source_interval_sha256": str(start)[0] * 64,
                "target_interval_sha256": str(start)[0] * 64,
            }
        )
    coordinate = {
        "schema_version": hp.COORDINATE_RECEIPT_VERSION,
        "source_assembly": hp.WS276_ASSEMBLY,
        "target_assembly": hp.WS283_ASSEMBLY,
        "source_reference_fasta_sha256": "1" * 64,
        "target_reference_fasta_sha256": "1" * 64,
        "source_reference_fai_sha256": "2" * 64,
        "target_reference_fai_sha256": "2" * 64,
        "source_chromosome_v_length": 20_924_180,
        "target_chromosome_v_length": 20_924_180,
        "source_chromosome_v_sha256": "3" * 64,
        "target_chromosome_v_sha256": "3" * 64,
        "full_reference_byte_identical": True,
        "generated_before_state_ingestion": True,
        "phenotype_accessed": False,
        "association_results_accessed": False,
        "operator_outcome_blinding_asserted": False,
        "outcome_access_claim_scope": "qualification_execution_path_only",
        "regions": regions,
    }
    coordinate_asset = _write_asset(
        regional_state,
        "regional_state",
        "coordinate_identity.json",
        "coordinate_identity_receipt",
        _json_bytes(coordinate),
    )
    coordinate_path = regional_state / coordinate_asset["relative_path"]
    state_qualification = {
        "schema_version": hp.STATE_PANEL_RECEIPT_VERSION,
        "coordinate_receipt_sha256": hp._sha256(coordinate_path),
        "sample_count": 209,
        "block_count": 2,
        "eligible_block_count": 2,
        "blocks": [],
        "outcome_blind": True,
    }
    state_asset = _write_asset(
        regional_state,
        "regional_state",
        "state_panel_qualification.json",
        "state_panel_qualification_receipt",
        _json_bytes(state_qualification),
    )
    feature_path = regional_state / "chrv_haplotype_features.npy"
    np.save(
        feature_path,
        np.asarray(dataset.chrv_haplotype_features, dtype="<f8"),
        allow_pickle=False,
    )
    feature_asset = {
        "role": "state_feature_matrix",
        "root_id": "regional_state",
        "relative_path": feature_path.name,
        "bytes": feature_path.stat().st_size,
        "sha256": hp._sha256(feature_path),
    }
    hashes = hp.dataset_hashes(dataset)
    state_feature_receipt = {
        "schema_version": hp.STATE_FEATURE_RECEIPT_VERSION,
        "state_panel_schema_version": hp.STATE_PANEL_RECEIPT_VERSION,
        "state_panel_qualification_sha256": state_asset["sha256"],
        "coordinate_identity_receipt_sha256": coordinate_asset["sha256"],
        "sample_count": 209,
        "sample_order_sha256": hashes["sample_order_sha256"],
        "feature_matrix_sha256": hashes["chrv_haplotype_features_sha256"],
        "feature_mask_scope": hp.FEATURE_MASK_SCOPE,
        "phenotype_values_accessed": False,
        "outer_test_phenotypes_accessed": False,
        "biological_claims_permitted": False,
    }
    state_feature_asset = _write_asset(
        regional_state,
        "regional_state",
        "state_feature_receipt.json",
        "state_feature_receipt",
        _json_bytes(state_feature_receipt),
    )
    payload["upstream_gates"]["coordinate_identity"] = {
        "status": "qualified",
        "receipt_schema_version": hp.COORDINATE_RECEIPT_VERSION,
        "receipt": coordinate_asset,
    }
    payload["upstream_gates"]["state_panel"] = {
        "status": "qualified",
        "qualification_schema_version": hp.STATE_PANEL_RECEIPT_VERSION,
        "feature_receipt_schema_version": hp.STATE_FEATURE_RECEIPT_VERSION,
        "qualification_receipt": state_asset,
        "feature_receipt": state_feature_asset,
        "feature_matrix": feature_asset,
    }

    kernel_root = roots["full_kernels"]
    kernel_assets = []
    kernel_records = []
    for kernel_id, marker_count, values, role in (
        ("whole_genome", 373_279, dataset.whole_genome_kernel, "whole_genome_matrix"),
        (
            "genome_excluding_chrv",
            258_096,
            dataset.genome_excluding_chrv_kernel,
            "genome_excluding_chrv_matrix",
        ),
    ):
        path = kernel_root / f"{kernel_id}.npy"
        matrix = np.asarray(values, dtype="<f8")
        np.save(path, matrix, allow_pickle=False)
        asset = {
            "role": role,
            "root_id": "full_kernels",
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hp._sha256(path),
        }
        kernel_assets.append(asset)
        kernel_records.append(
            {
                "id": kernel_id,
                "relationship_markers": marker_count,
                "npy": {
                    "name": path.name,
                    "bytes": asset["bytes"],
                    "sha256": asset["sha256"],
                    "matrix_payload_sha256": hp._matrix_payload_sha256(matrix),
                    "dtype": "<f8",
                    "shape": [209, 209],
                    "finite": True,
                    "exactly_symmetric": True,
                },
            }
        )
    kernel_receipt = {
        "schema_version": hp.FULL_KERNEL_RECEIPT_VERSION,
        "samples": 209,
        "ordered_iid_sha256": hp._ordered_iid_sha256(dataset.sample_ids),
        "kernels": kernel_records,
        "biological_claims_permitted": False,
    }
    kernel_receipt_asset = _write_asset(
        kernel_root,
        "full_kernels",
        "qualification_receipt.json",
        "full_kernel_qualification_receipt",
        _json_bytes(kernel_receipt),
    )
    payload["upstream_gates"]["full_kernels"] = {
        "status": "qualified",
        "receipt_schema_version": hp.FULL_KERNEL_RECEIPT_VERSION,
        "qualification_receipt": kernel_receipt_asset,
        "whole_genome_matrix": kernel_assets[0],
        "genome_excluding_chrv_matrix": kernel_assets[1],
    }

    repository = roots["repository"]
    assignments = (
        "IID\tpopulation_group\n"
        + "".join(
            f"{sample}\t{group}\n"
            for sample, group in zip(dataset.sample_ids, dataset.population_groups, strict=True)
        )
    ).encode()
    assignment_asset = _write_asset(
        repository,
        "repository",
        "mappings/population_groups.tsv",
        "population_group_assignments",
        assignments,
    )
    group_sizes = {
        group: dataset.population_groups.count(group)
        for group in sorted(set(dataset.population_groups))
    }
    group_receipt = {
        "schema_version": hp.GROUP_RECEIPT_VERSION,
        "assignments_sha256": assignment_asset["sha256"],
        "sample_count": 209,
        "selected_k": len(group_sizes),
        "group_sizes": group_sizes,
        "ordered_iid_sha256": hp._ordered_iid_sha256(dataset.sample_ids),
        "eigenvalues_sha256": "a" * 64,
        "eigenvalue_count": 10,
        "eigenvalues_used_for_clustering": False,
        "axis_weighting": "equal_weight_after_per_pc_standardization",
        "eigenvalue_usage": "verified_provenance_only_not_used_for_clustering",
        "phenotype_paths_accepted": False,
        "phenotype_values_accessed": False,
        "outcome_access_claim_scope": "derivation_execution_path_only",
        "operator_outcome_blinding_asserted": False,
        "labels_are_external_ancestry_assignments": False,
    }
    group_receipt_asset = _write_asset(
        repository,
        "repository",
        "mappings/population_groups.receipt.json",
        "population_group_receipt",
        _json_bytes(group_receipt),
    )
    group_gate = payload["upstream_gates"]["population_groups"]
    group_gate["receipt"] = group_receipt_asset
    group_gate["assignments"] = assignment_asset
    group_gate["eigenvalues_sha256"] = group_receipt["eigenvalues_sha256"]
    payload["status"] = "qualified_for_file_backed_real_execution"

    manifest = hp.HeldoutPredictionManifest.model_validate(payload)
    receipt_payload = _synthetic_receipt(dataset).model_dump()
    receipt_payload.update(
        {
            "mode": "real_ws283",
            "reported_source_sample_count": 209,
            "reported_aligned_finite_count": 209,
        }
    )
    receipt = hp.PredictionInputReceipt.model_validate(receipt_payload)
    return manifest, receipt, roots


class HeldoutPredictionTests(unittest.TestCase):
    def test_manifest_freezes_modern_identity_models_and_legacy_refusal(self) -> None:
        manifest = hp.load_manifest(CONFIG)
        self.assertEqual(manifest.modern_cohort.sample_count, 209)
        self.assertEqual(manifest.modern_cohort.marker_count, 373_279)
        self.assertEqual(
            manifest.modern_cohort.non_chromosome_v_marker_count,
            373_279 - 115_183,
        )
        self.assertEqual([item.id for item in manifest.comparators], list(hp.MODEL_IDS))
        self.assertFalse(manifest.legacy_quarantine.accepted_as_heldout_prediction_input)
        self.assertFalse(manifest.biological_claims_permitted)
        self.assertEqual(manifest.upstream_gates.parametric_null_terminal.status, "qualified")
        self.assertEqual(manifest.upstream_gates.population_groups.status, "qualified")
        self.assertEqual(manifest.upstream_gates.coordinate_identity.status, "qualified")
        self.assertEqual(manifest.upstream_gates.state_panel.status, "missing")
        self.assertEqual(manifest.upstream_gates.full_kernels.status, "qualified")
        self.assertEqual(
            manifest.selection.genotype_feature_mask_scope,
            hp.FEATURE_MASK_SCOPE,
        )

        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["modern_cohort"]["sample_count"] = 203
        with self.assertRaises(ValidationError):
            hp.HeldoutPredictionManifest.model_validate(payload)

        stale = json.loads(CONFIG.read_text(encoding="utf-8"))
        stale["status"] = "blocked_on_coordinate_and_state_receipts"
        with self.assertRaisesRegex(ValidationError, "execution status differs"):
            hp.HeldoutPredictionManifest.model_validate(stale)

    @unittest.skip(
        "real kernel receipts are intentionally excluded from the public repository"
    )
    def test_real_manifest_pins_handoff_exact_qualified_full_kernels(self) -> None:
        manifest = hp.load_manifest(CONFIG)
        handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))["full_kernels"]
        gate = manifest.upstream_gates.full_kernels
        self.assertEqual(gate.status, "qualified")
        self.assertEqual(manifest.status, "blocked_on_state_receipts")

        handoff_assets = {
            item["role"]: item for item in handoff["qualified_output_assets"]
        }
        expected = [
            (
                gate.qualification_receipt,
                handoff["numerical_qualification_receipt"],
            ),
            (gate.whole_genome_matrix, handoff_assets["whole_genome_float64_npy"]),
            (
                gate.genome_excluding_chrv_matrix,
                handoff_assets["genome_excluding_chrv_float64_npy"],
            ),
        ]
        for asset, frozen in expected:
            self.assertIsNotNone(asset)
            assert asset is not None
            self.assertEqual(asset.model_dump(mode="json"), frozen)
            verified = hp._verify_frozen_file(
                asset, {"full_kernel_run": FULL_KERNEL_ROOT}
            )
            self.assertEqual(hp._sha256(verified), asset.sha256)

    def test_dataset_identity_group_splits_and_tamper_gate(self) -> None:
        manifest = hp.load_manifest(CONFIG)
        dataset = _synthetic_dataset()
        receipt = _synthetic_receipt(dataset)
        qualification = hp.qualify_dataset(manifest, dataset, receipt)
        self.assertEqual(qualification.sample_count, 209)
        self.assertEqual(qualification.population_group_count, 5)
        self.assertTrue(qualification.leakage_gates_passed)
        splits = hp.build_outer_splits(
            manifest, dataset.sample_ids, dataset.population_groups, qualification
        )
        self.assertEqual(len(splits.folds), 5)
        tested = [index for fold in splits.folds for index in fold.test_indices]
        self.assertEqual(sorted(tested), list(range(209)))
        self.assertTrue(all(not fold.outer_test_used_for_selection for fold in splits.folds))

        changed = receipt.model_copy(deep=True)
        changed.whole_genome_kernel_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "input identity"):
            hp.qualify_dataset(manifest, dataset, changed)

    def test_legacy_203_165_and_pheno_only_inputs_are_refused(self) -> None:
        dataset = _synthetic_dataset()
        for change in (
            {"legacy_analysis_id": hp.LEGACY_ANALYSIS_ID},
            {"legacy_input_paths": ["runs/e2_kinship/pheno_only.rel"]},
            {"reported_source_sample_count": 203},
            {"reported_aligned_finite_count": 165},
        ):
            payload = {**_synthetic_receipt(dataset).model_dump(), **change}
            receipt = hp.PredictionInputReceipt.model_validate(payload)
            with self.assertRaisesRegex(ValueError, "quarantined"):
                hp.refuse_legacy_audit(receipt)

    def test_fabricated_real_receipts_cannot_bypass_blocked_manifest(self) -> None:
        dataset = _synthetic_dataset()
        payload = _synthetic_receipt(dataset).model_dump()
        payload.update(
            {
                "mode": "real_ws283",
                "reported_source_sample_count": 209,
                "reported_aligned_finite_count": 209,
            }
        )
        receipt = hp.PredictionInputReceipt.model_validate(payload)
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary)
            (fake / "coordinate.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "receipt contract remains blocked"):
                hp.qualify_dataset(
                    hp.load_manifest(CONFIG),
                    dataset,
                    receipt,
                    upstream_roots={"regional_state": fake},
                )

        old_self_attestation = dict(payload)
        old_self_attestation["null_interpretation_verified"] = True
        with self.assertRaises(ValidationError):
            hp.PredictionInputReceipt.model_validate(old_self_attestation)

    def test_real_receipt_chain_is_file_backed_and_semantically_verified(self) -> None:
        dataset = _synthetic_dataset()
        with tempfile.TemporaryDirectory() as temporary:
            manifest, receipt, roots = _real_fixture(Path(temporary), dataset)
            qualification = hp.qualify_dataset(
                manifest, dataset, receipt, upstream_roots=roots
            )
            self.assertEqual(qualification.mode, "real_ws283")
            self.assertIn(
                "full_kernel_qualification_receipt",
                qualification.upstream_asset_sha256_by_role,
            )
            self.assertIn(
                "state_panel_qualification_receipt",
                qualification.upstream_asset_sha256_by_role,
            )

            coordinate = roots["regional_state"] / "coordinate_identity.json"
            coordinate.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "byte count differs|SHA-256 differs"):
                hp.qualify_dataset(manifest, dataset, receipt, upstream_roots=roots)

    def test_real_receipt_semantics_fail_even_when_tampered_bytes_are_rebound(self) -> None:
        dataset = _synthetic_dataset()
        with tempfile.TemporaryDirectory() as temporary:
            manifest, receipt, roots = _real_fixture(Path(temporary), dataset)
            coordinate = roots["regional_state"] / "coordinate_identity.json"
            payload = json.loads(coordinate.read_text(encoding="utf-8"))
            payload["phenotype_accessed"] = True
            coordinate.write_bytes(_json_bytes(payload))
            manifest_payload = manifest.model_dump(mode="json")
            asset = manifest_payload["upstream_gates"]["coordinate_identity"]["receipt"]
            asset["bytes"] = coordinate.stat().st_size
            asset["sha256"] = hp._sha256(coordinate)
            rebound = hp.HeldoutPredictionManifest.model_validate(manifest_payload)
            with self.assertRaisesRegex(ValueError, "coordinate receipt semantic field"):
                hp.qualify_dataset(rebound, dataset, receipt, upstream_roots=roots)

    def test_nested_selection_predictions_calibration_and_write_once_bundle(self) -> None:
        manifest = hp.load_manifest(CONFIG)
        dataset = _synthetic_dataset()
        receipt = _synthetic_receipt(dataset)
        qualification = hp.qualify_dataset(manifest, dataset, receipt)
        splits = hp.build_outer_splits(
            manifest, dataset.sample_ids, dataset.population_groups, qualification
        )
        self.assertEqual(len(hp._candidate_grid("whole_genome_gblup")), 5)
        self.assertEqual(
            len(hp._candidate_grid("combined_excluding_chrv_plus_haplotype")), 15
        )
        # Keep the synthetic execution quick while separately asserting the frozen grids above.
        with mock.patch.object(hp, "RIDGE_GRID", (1.0,)), mock.patch.object(
            hp, "COMBINED_WEIGHT_GRID", (0.5,)
        ):
            result = hp.evaluate_heldout_prediction(
                manifest, dataset, qualification, splits
            )
        self.assertEqual(len(result["predictions"]), 4 * 5 * 209)
        self.assertTrue(
            all(not item["outer_test_used_for_selection"] for item in result["selection"])
        )
        self.assertEqual(set(result["metrics"]), set(hp.TRAITS))
        for trait in hp.TRAITS:
            self.assertEqual(set(result["metrics"][trait]), set(hp.MODEL_IDS))
            for metrics in result["metrics"][trait].values():
                self.assertTrue(math.isfinite(metrics["population_group_macro_rmse"]))
                self.assertGreaterEqual(metrics["coverage_90"], 0.0)
                self.assertLessEqual(metrics["coverage_90"], 1.0)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "heldout_prediction-bundle"
            published = hp.publish_write_once_bundle(
                manifest, output, qualification, splits, result
            )
            self.assertTrue(published["terminal_success"])
            self.assertEqual(published["verified_payload_count"], 4)
            self.assertEqual((output / "SUCCESS").read_bytes(), b"SUCCESS\n")
            self.assertEqual(
                len((output / "predictions.jsonl").read_text().splitlines()),
                4 * 5 * 209,
            )
            with self.assertRaises(FileExistsError):
                hp.publish_write_once_bundle(
                    manifest, output, qualification, splits, result
                )
            (output / "summary.json").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum failed"):
                hp.verify_write_once_bundle(manifest, output)

    def test_evaluation_rejects_altered_arrays_and_swapped_qualification(self) -> None:
        manifest = hp.load_manifest(CONFIG)
        dataset = _synthetic_dataset()
        qualification = hp.qualify_dataset(manifest, dataset, _synthetic_receipt(dataset))
        splits = hp.build_outer_splits(
            manifest, dataset.sample_ids, dataset.population_groups, qualification
        )
        changed_kernel = np.array(dataset.whole_genome_kernel, copy=True)
        changed_kernel[0, 0] += 0.5
        altered = replace(dataset, whole_genome_kernel=changed_kernel)
        with self.assertRaisesRegex(ValueError, "dataset differs from its input qualification"):
            hp.evaluate_heldout_prediction(manifest, altered, qualification, splits)

        changed_phenotypes = {
            trait: np.array(values, copy=True) for trait, values in dataset.phenotypes.items()
        }
        changed_phenotypes[hp.TRAITS[0]][0] += 1.0
        second = replace(dataset, phenotypes=changed_phenotypes)
        second_qualification = hp.qualify_dataset(
            manifest, second, _synthetic_receipt(second)
        )
        with self.assertRaisesRegex(ValueError, "dataset differs from its input qualification"):
            hp.evaluate_heldout_prediction(manifest, dataset, second_qualification, splits)

    def test_evaluation_rejects_swapped_splits_receipt_hashes_and_leaky_folds(self) -> None:
        manifest = hp.load_manifest(CONFIG)
        dataset = _synthetic_dataset()
        qualification = hp.qualify_dataset(manifest, dataset, _synthetic_receipt(dataset))
        splits = hp.build_outer_splits(
            manifest, dataset.sample_ids, dataset.population_groups, qualification
        )

        rotated_groups = dataset.population_groups[1:] + dataset.population_groups[:1]
        regrouped = replace(dataset, population_groups=rotated_groups)
        regrouped_qualification = hp.qualify_dataset(
            manifest, regrouped, _synthetic_receipt(regrouped)
        )
        swapped_splits = hp.build_outer_splits(
            manifest,
            regrouped.sample_ids,
            regrouped.population_groups,
            regrouped_qualification,
        )
        with self.assertRaisesRegex(ValueError, "split receipt is not bound"):
            hp.evaluate_heldout_prediction(manifest, dataset, qualification, swapped_splits)

        wrong_hash = splits.model_copy(deep=True)
        wrong_hash.input_qualification_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "split receipt is not bound"):
            hp.evaluate_heldout_prediction(manifest, dataset, qualification, wrong_hash)

        leaky = splits.model_copy(deep=True)
        leaked_index = leaky.folds[0].test_indices[0]
        leaky.folds[0].train_indices.append(leaked_index)
        leaky.folds[0].train_sample_ids.append(dataset.sample_ids[leaked_index])
        with self.assertRaisesRegex(ValueError, "outer fold leaks"):
            hp.evaluate_heldout_prediction(manifest, dataset, qualification, leaky)

        invalid_payload = splits.model_dump(mode="json")
        invalid_payload["folds"][0]["train_indices"].append(leaked_index)
        invalid_payload["folds"][0]["train_sample_ids"].append(dataset.sample_ids[leaked_index])
        with self.assertRaises(ValidationError):
            hp.SplitReceipt.model_validate(invalid_payload)


if __name__ == "__main__":
    unittest.main()
