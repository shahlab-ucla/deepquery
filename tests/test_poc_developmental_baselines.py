from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wormctx.poc import developmental_baselines as baselines
from wormctx.poc import developmental_omix_stage_adapter as stage
from wormctx.poc.developmental_omix import _sha256, _write_bytes_atomic, _write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/developmental_genetics/gene_disjoint_outcome_prediction/config/gene_disjoint_baselines.json"


def _test_contract() -> baselines.BaselineContract:
    payload = baselines.load_contract(CONFIG).model_dump(mode="json")
    payload["analysis_id"] = "synthetic_gene_disjoint_baselines"
    payload["models"]["flat_ridge"]["alphas"] = [0.1, 1.0]
    payload["models"]["semantic_linear"]["alphas"] = [0.1, 1.0]
    payload["models"]["nonlinear_edge_free"].update(
        {
            "target_trainable_parameters": 81,
            "maximum_relative_parameter_delta": 0.2,
            "learning_rates": [0.01],
            "l2_penalties": [0.0],
            "seeds": [17],
            "epochs": 8,
        }
    )
    payload["evaluation"]["bootstrap_replicates"] = 20
    payload["evaluation"]["bootstrap_seed"] = 91
    return baselines.BaselineContract.model_validate(payload)


def _write_receipted_bundle(root: Path, payloads: dict[str, Any]) -> None:
    root.mkdir()
    for name, payload in payloads.items():
        _write_json_atomic(root / name, payload)
    lines = [f"{_sha256(root / name)}  {name}" for name in sorted(payloads)]
    _write_bytes_atomic(root / "SHA256SUMS.txt", ("\n".join(lines) + "\n").encode())
    _write_bytes_atomic(root / "SUCCESS", b"SUCCESS\n")


def _split_manifest(path: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    index = 1
    for partition, count in (("train", 410), ("validation", 137), ("sealed_test", 137)):
        for _ in range(count):
            gene_id = f"WBGene{index:08d}"
            entries.append(
                {
                    "gene_id": gene_id,
                    "public_name": f"GENE-{index}",
                    "partition": partition,
                    "embryo_count": 1,
                    "embryo_ids": [f"EMB-{index:04d}"],
                }
            )
            index += 1
    payload = {
        "schema_version": "wormctx-omix709-gene-split-manifest-1.0",
        "key": "wormbase_gene_id",
        "whole_gene": True,
        "claim_bearing_genes": entries,
        "development_only_exposed_pilot_genes": [
            {
                "gene_id": "WBGene99999999",
                "partition": "development_only_exposed_pilot",
                "embryo_ids": ["PILOT-1"],
            }
        ],
    }
    _write_json_atomic(path, payload)
    return payload


def _stage_bundle(root: Path, split_path: Path, contract: baselines.BaselineContract) -> None:
    input_cells = [f"C26-{index:03d}" for index in range(26)]
    endpoint_cells = [f"C200-{index:03d}" for index in range(200)]
    payloads = {
        "analysis_manifest.json": {
            "schema_version": stage.ADAPTER_ANALYSIS_VERSION,
            "contract_sha256": contract.stage_adapter.contract_sha256,
            "parent_split_manifest_sha256": _sha256(split_path),
            "model_launch_status": "eligible_for_preregistered_gene_disjoint_baselines",
            "biological_claims_permitted": False,
            "sealed_test_opened": False,
        },
        "lineage_manifest.json": {
            "schema_version": stage.LINEAGE_MANIFEST_VERSION,
            "edges_through_200": [],
        },
        "modality_mappings.json": {
            "schema_version": stage.MAPPING_MANIFEST_VERSION,
            "s4_sheet_4_accessed": False,
            "s4_workbook_parsed": False,
            "perturbation_values_decoded": False,
            "s4_outcome_values_decoded": False,
        },
        "overlap_status.json": {
            "schema_version": stage.OVERLAP_RECEIPT_VERSION,
            "embryo_level_disjointness_established": True,
        },
        "qualification.json": {
            "schema_version": stage.ADAPTER_QUALIFICATION_VERSION,
            "technical_status": "success",
            "qualification_status": "qualified_with_exclusions",
            "model_launch_status": "eligible_for_preregistered_gene_disjoint_baselines",
            "gates": {
                "pilot_embryo_overlap_closed": True,
                "sealed_test_opened": False,
            },
        },
        "raw_control_source_receipt.json": {
            "schema_version": stage.RAW_CONTROL_RECEIPT_VERSION,
            "archive": {"verified": True},
            "extracted": {"verified": True},
            "access": {
                "nonpermitted_value_columns_policy": "never_decode",
                "outcome_columns_decoded": False,
            },
            "lineage": {"exact_parentage_verified": True},
        },
        "stage_membership.json": {
            "schema_version": stage.STAGE_MANIFEST_VERSION,
            "raw_control_outcome_columns_decoded": False,
            "perturbation_measurement_values_read": False,
            "s4_outcome_values_read": False,
            "input_26": {"frontier": input_cells},
            "endpoint_200": {"frontier": endpoint_cells},
        },
    }
    _write_receipted_bundle(root, payloads)


def _observation(index: int) -> dict[str, Any]:
    value = ((index - 1) % 37 - 18) / 10.0
    flat_second = None if index % 11 == 0 else value**2
    semantic_third = None if index % 13 == 0 else value**3
    outcome = 0.55 * value + 0.18 * value**2 - 0.04 * value**3
    return {
        "embryo_id": f"EMB-{index:04d}",
        "wormbase_gene_id": f"WBGene{index:08d}",
        "flat_features": [value, flat_second],
        "semantic_features": [value, value**2, semantic_third],
        "outcome": outcome,
    }


def _prepared_bundle(
    root: Path,
    split: dict[str, Any],
    split_path: Path,
    stage_root: Path,
) -> Path:
    root.mkdir()
    artifacts = []
    entries_by_partition: dict[str, list[dict[str, Any]]] = {
        name: [] for name in baselines.PARTITION_ORDER
    }
    for item in split["claim_bearing_genes"]:
        entries_by_partition[item["partition"]].append(item)
    for partition in baselines.PARTITION_ORDER:
        observations = [
            _observation(int(item["gene_id"][-8:])) for item in entries_by_partition[partition]
        ]
        filename = f"{partition}.json"
        path = root / filename
        _write_json_atomic(
            path,
            {
                "schema_version": baselines.PARTITION_VERSION,
                "partition": partition,
                "observations": observations,
            },
        )
        artifacts.append(
            {
                "partition": partition,
                "filename": filename,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "genes": len(observations),
                "embryos": len(observations),
            }
        )
    manifest = root / "prepared_manifest.json"
    _write_json_atomic(
        manifest,
        {
            "schema_version": baselines.PREPARED_MANIFEST_VERSION,
            "analysis_id": "synthetic_prepared_developmental_baselines",
            "stage_adapter_qualification_sha256": _sha256(
                stage_root / "qualification.json"
            ),
            "split_manifest_sha256": _sha256(split_path),
            "source_stage_membership_sha256": _sha256(
                stage_root / "stage_membership.json"
            ),
            "partition_artifacts_are_separate": True,
            "sealed_test_artifact_not_read_during_selection": True,
            "feature_schema": {
                "flat_feature_names": ["prefix_mean", "prefix_quadratic"],
                "semantic_feature_names": [
                    "ABa:timing",
                    "ABp:timing",
                    "EMS:expression",
                ],
                "semantic_feature_cell_ids": ["ABa", "ABp", "EMS"],
                "outcome_id": "continuous_cnd1_200_cell_burden",
                "outcome_scale": "control_standardized",
                "control_standardized_expected_mean": 0.0,
            },
            "artifacts": artifacts,
        },
    )
    return manifest


def _fixture(tmp_path: Path) -> tuple[
    baselines.BaselineContract,
    Path,
    Path,
    Path,
]:
    contract = _test_contract()
    split_path = tmp_path / "split_manifest.json"
    split = _split_manifest(split_path)
    stage_root = tmp_path / "stage-adapter"
    _stage_bundle(stage_root, split_path, contract)
    prepared_manifest = _prepared_bundle(
        tmp_path / "prepared",
        split,
        split_path,
        stage_root,
    )
    return contract, stage_root, split_path, prepared_manifest


def test_contract_freezes_four_families_and_410_137_137_whole_gene_split() -> None:
    contract = baselines.load_contract(CONFIG)

    assert contract.models.family_order == baselines.FAMILY_ORDER
    assert (
        contract.split.train_genes,
        contract.split.validation_genes,
        contract.split.sealed_test_genes,
    ) == (410, 137, 137)
    assert contract.evaluation.sealed_test_openings == 1
    assert contract.evaluation.sealed_test_permitted_for_selection is False

    payload = contract.model_dump(mode="json")
    payload["split"]["sealed_test_genes"] = 136
    with pytest.raises(ValueError, match="410/137/137"):
        baselines.BaselineContract.model_validate(payload)


def test_gene_macro_metrics_calibration_and_clustered_bootstrap_are_deterministic() -> None:
    observed = [0.0, 2.0, 1.0, 3.0]
    predicted = [0.0, 1.0, 1.0, 4.0]
    genes = ["G1", "G1", "G2", "G2"]

    metrics = baselines.gene_macro_metrics(observed, predicted, genes)
    first = baselines.clustered_gene_bootstrap(
        observed,
        predicted,
        genes,
        replicates=30,
        seed=7,
    )
    second = baselines.clustered_gene_bootstrap(
        observed,
        predicted,
        genes,
        replicates=30,
        seed=7,
    )

    assert metrics["genes"] == 2
    assert metrics["gene_macro_rmse"] == pytest.approx(0.7071067811865476)
    assert metrics["gene_macro_mae"] == pytest.approx(0.5)
    assert first == second
    assert first["resampling_unit"] == "whole_gene_cluster"


def test_selection_does_not_open_or_hash_missing_sealed_artifact(tmp_path: Path) -> None:
    contract, stage_root, split_path, prepared_manifest = _fixture(tmp_path)
    sealed = prepared_manifest.parent / "sealed_test.json"
    sealed.unlink()
    selection = tmp_path / "selection"

    result = baselines.fit_select(
        contract,
        stage_root,
        split_path,
        prepared_manifest,
        selection,
    )

    assert result["fit_gene_counts"] == {"train": 410, "validation": 137}
    assert result["sealed_test_artifact_bytes_read"] == 0
    assert result["sealed_test_deserialized"] is False
    assert result["sealed_test_openings"] == 0
    assert baselines.verify_selection(selection)["verified"] is True


def test_four_family_selection_and_single_sealed_evaluation_are_receipted(
    tmp_path: Path,
) -> None:
    contract, stage_root, split_path, prepared_manifest = _fixture(tmp_path)
    selection = tmp_path / "selection"
    baselines.fit_select(
        contract,
        stage_root,
        split_path,
        prepared_manifest,
        selection,
    )
    selection_receipt = baselines.verify_selection(selection)
    prepared = baselines.load_prepared_manifest(prepared_manifest)
    sealed_artifact = next(
        item for item in prepared.artifacts if item.partition == "sealed_test"
    )
    gate_name = baselines.sealed_gate_name(
        _sha256(prepared_manifest),
        selection_receipt["selection_manifest_sha256"],
        sealed_artifact.sha256,
    )
    gate = tmp_path / gate_name
    evaluation = tmp_path / "sealed-evaluation"

    result = baselines.evaluate_sealed(
        contract,
        stage_root,
        split_path,
        prepared_manifest,
        selection,
        gate,
        evaluation,
        open_sealed_test_once=True,
    )

    assert result["sealed_test_genes"] == 137
    assert result["sealed_test_opening_ordinal"] == 1
    assert (gate / "SUCCESS").read_bytes() == b"SUCCESS\n"
    assert baselines.verify_evaluation(evaluation)["verified"] is True
    metrics = json.loads((evaluation / "test_metrics.json").read_text(encoding="utf-8"))
    bootstrap = json.loads(
        (evaluation / "test_bootstrap.json").read_text(encoding="utf-8")
    )
    assert set(metrics["families"]) == set(baselines.FAMILY_ORDER)
    assert set(bootstrap["families"]) == set(baselines.FAMILY_ORDER)
    nonlinear_state = json.loads(
        (selection / "model_states.json").read_text(encoding="utf-8")
    )["states"]["nonlinear_edge_free"]["model"]
    assert nonlinear_state["edge_access"] is False
    assert nonlinear_state["parameter_count"] == 81

    with pytest.raises(FileExistsError, match="already been opened"):
        baselines.evaluate_sealed(
            contract,
            stage_root,
            split_path,
            prepared_manifest,
            selection,
            gate,
            tmp_path / "second-evaluation",
            open_sealed_test_once=True,
        )


def test_sealed_evaluation_requires_explicit_opening_and_tamper_is_detected(
    tmp_path: Path,
) -> None:
    contract, stage_root, split_path, prepared_manifest = _fixture(tmp_path)
    selection = tmp_path / "selection"
    baselines.fit_select(
        contract,
        stage_root,
        split_path,
        prepared_manifest,
        selection,
    )
    with pytest.raises(PermissionError, match="explicit single sealed-test opening"):
        baselines.evaluate_sealed(
            contract,
            stage_root,
            split_path,
            prepared_manifest,
            selection,
            tmp_path / "unused-gate",
            tmp_path / "unused-evaluation",
            open_sealed_test_once=False,
        )

    target = selection / "validation_metrics.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum failed"):
        baselines.verify_selection(selection)
