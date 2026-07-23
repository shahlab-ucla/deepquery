from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wormctx.poc import qtl_haplotype_pav as hp


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/natural_variation/pangenome_state_qualification/config/pangenome_state_contract.json"
NULL_CONFIG = ROOT / "experiments/natural_variation/parametric_polygenic_null_calibration/config/parametric_null.json"
HANDOFF_CONFIG = ROOT / "experiments/natural_variation/pangenome_state_qualification/config/state_input_handoff.json"
INTERPRETATION_CONFIG = (
    ROOT / "artifacts/upstream/parametric_polygenic_null_calibration/interpretation.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_coordinate_receipt(tmp_path: Path, *, mismatch: bool = False) -> Path:
    regions = []
    for index, (region_id, (start, end)) in enumerate(hp.EXPECTED_REGIONS.items(), 1):
        sequence_hash = f"{index:x}" * 64
        regions.append(
            {
                "region_id": region_id,
                "source_assembly": hp.WS276_ASSEMBLY,
                "target_assembly": hp.WS283_ASSEMBLY,
                "source_chromosome": 5,
                "target_chromosome": 5,
                "source_start": start,
                "source_end": end,
                "target_start": start + (1 if mismatch and region_id == "chrv_left" else 0),
                "target_end": end,
                "mapping_mode": "sequence_identical_coordinate_identity",
                "source_interval_sha256": sequence_hash,
                "target_interval_sha256": sequence_hash,
            }
        )
    payload = {
        "schema_version": hp.COORDINATE_RECEIPT_VERSION,
        "source_assembly": hp.WS276_ASSEMBLY,
        "target_assembly": hp.WS283_ASSEMBLY,
        "source_reference_fasta_sha256": "a" * 64,
        "target_reference_fasta_sha256": "a" * 64,
        "source_reference_fai_sha256": "b" * 64,
        "target_reference_fai_sha256": "b" * 64,
        "source_chromosome_v_length": 20_924_180,
        "target_chromosome_v_length": 20_924_180,
        "source_chromosome_v_sha256": "c" * 64,
        "target_chromosome_v_sha256": "c" * 64,
        "full_reference_byte_identical": True,
        "generated_before_state_ingestion": True,
        "phenotype_accessed": False,
        "association_results_accessed": False,
        "operator_outcome_blinding_asserted": False,
        "outcome_access_claim_scope": "qualification_execution_path_only",
        "regions": regions,
    }
    path = tmp_path / "coordinate_identity.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _coordinate(tmp_path: Path) -> hp.CoordinateQualification:
    return hp.qualify_coordinate_identity(
        hp.load_manifest(CONFIG), _write_coordinate_receipt(tmp_path)
    )


def _write_null_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / hp.NULL_RUN_ID
    (root / "receipts").mkdir(parents=True)
    (root / "summary").mkdir()
    (root / "SUCCESS").write_bytes(b"SUCCESS\n")
    (root / "receipts/SOURCE_REVISION").write_text(
        hp.NULL_SOURCE_COMMIT + "\n", encoding="utf-8"
    )
    frozen = root / "receipts/frozen_parametric_null_contract.json"
    frozen.write_bytes(NULL_CONFIG.read_bytes())
    assert _sha256(frozen) == hp.load_manifest(CONFIG).null_dependency.frozen_contract_sha256
    cells = []
    for cell_id in hp._expected_cell_ids():
        boundary = cell_id in hp.EXPECTED_BOUNDARY_CELLS
        cells.append(
            {
                "cell_id": cell_id,
                "replicates_attempted": 100,
                "replicates_completed": 100,
                "nominal_bonferroni_fwer": {"estimate": 0.05},
                "anchor_boundary": boundary,
                "threshold_eligible": not boundary,
            }
        )
    aggregation = {
        "schema_version": "wormctx-abamectin-ws283-parametric-null-aggregation-1.0",
        "manifest_sha256": _sha256(frozen),
        "rich_summary": {"path": "summary.json", "sha256": "e" * 64},
        "calibration_cells": 16,
        "replicates_per_cell": 100,
        "attempted_null_maps": 1600,
        "completed_null_maps": 1600,
        "cells": cells,
        "validated": False,
        "biological_claims_permitted": False,
    }
    aggregation_path = root / "summary/parametric_null_aggregation.json"
    aggregation_path.write_text(json.dumps(aggregation), encoding="utf-8")
    interpretation = {
        "schema_version": hp.NULL_INTERPRETATION_VERSION,
        "null_run_id": hp.NULL_RUN_ID,
        "aggregation_sha256": _sha256(aggregation_path),
        "decision": "proceed_with_descriptive_haplotype_scaffold",
        "reviewed_before_haplotype_outcomes": True,
        "marker_null_not_treated_as_block_test_calibration": True,
        "causal_interpretation_permitted": False,
    }
    interpretation_path = tmp_path / "null_interpretation.json"
    interpretation_path.write_text(json.dumps(interpretation), encoding="utf-8")
    return root, interpretation_path


def _synthetic_dependency() -> hp.NullDependencyQualification:
    eligible = [
        cell for cell in hp._expected_cell_ids() if cell not in hp.EXPECTED_BOUNDARY_CELLS
    ]
    return hp.NullDependencyQualification(
        state="success_descriptive_scaffold_enabled",
        run_id=hp.NULL_RUN_ID,
        terminal_success=True,
        analysis_plan_permitted=True,
        scaffold_development_permitted=True,
        aggregation_sha256="f" * 64,
        interpretation_receipt_sha256=None,
        threshold_eligible_cells=eligible,
        boundary_cells=list(hp.EXPECTED_BOUNDARY_CELLS),
    )


def _blocks() -> list[hp.BlockDefinition]:
    return [
        hp.BlockDefinition(
            block_id="left_hap_block_01",
            region_id="chrv_left",
            chromosome=5,
            assembly=hp.WS283_ASSEMBLY,
            start=1_800_000,
            end=2_100_000,
            state_kind="haplotype",
            reference_state_id="hap_ref",
            definition_method="outcome_blind_pangenome_path_cluster",
            source_snapshot_sha256="1" * 64,
            phenotype_accessed_during_definition=False,
        ),
        hp.BlockDefinition(
            block_id="right_pav_locus_01",
            region_id="chrv_right",
            chromosome=5,
            assembly=hp.WS283_ASSEMBLY,
            start=15_900_000,
            end=16_100_000,
            state_kind="pav",
            reference_state_id="pav_present",
            definition_method="outcome_blind_presence_absence_locus",
            source_snapshot_sha256="2" * 64,
            phenotype_accessed_during_definition=False,
        ),
    ]


def _sample_order() -> list[str]:
    return [f"strain_{index:03d}" for index in range(209)]


def _calls(*, rare_haplotype: bool = False) -> list[hp.GenomicStateCall]:
    records: list[hp.GenomicStateCall] = []
    alt_start = 205 if rare_haplotype else 100
    unresolved_start = 209 if rare_haplotype else 204
    for index, sample in enumerate(_sample_order()):
        if index >= unresolved_start:
            hap_state = "unresolved"
            hap_state_id = "hap_unresolved"
            path_id = None
        elif index >= alt_start:
            hap_state = "alternate_path"
            hap_state_id = "hap_alt"
            path_id = "path_alt"
        else:
            hap_state = "reference_path"
            hap_state_id = "hap_ref"
            path_id = "path_ref"
        records.append(
            hp.HaplotypeStateCall(
                sample_id=sample,
                block_id="left_hap_block_01",
                region_id="chrv_left",
                assembly=hp.WS283_ASSEMBLY,
                state_id=hap_state_id,
                confidence=0.99,
                kind="haplotype",
                state=hap_state,
                path_id=path_id,
            )
        )
        pav_present = index < 120
        records.append(
            hp.PavStateCall(
                sample_id=sample,
                block_id="right_pav_locus_01",
                region_id="chrv_right",
                assembly=hp.WS283_ASSEMBLY,
                state_id="pav_present" if pav_present else "pav_absent",
                confidence=0.98,
                kind="pav",
                state="present" if pav_present else "absent",
                locus_id="pav_locus_01",
                copy_number=1 if pav_present else 0,
            )
        )
    return records


def test_frozen_manifest_regions_dependency_and_claim_boundary() -> None:
    manifest = hp.load_manifest(CONFIG)
    assert [(item.id, item.ws276_start, item.ws276_end) for item in manifest.regions] == [
        ("chrv_left", 1_747_612, 4_333_001),
        ("chrv_right", 13_606_517, 16_754_986),
    ]
    assert manifest.null_dependency.run_id == hp.NULL_RUN_ID
    assert manifest.null_dependency.marker_null_is_not_haplotype_test_calibration is True
    assert manifest.status == "blocked_on_real_pangenome_inputs"
    assert manifest.null_dependency.interpretation_receipt_status == "frozen"
    assert (
        manifest.null_dependency.interpretation_receipt_sha256
        == hp.NULL_INTERPRETATION_SHA256
    )
    assert len(manifest.null_dependency.interpretation_receipt_sha256) == 64
    assert not INTERPRETATION_CONFIG.exists()
    assert manifest.carrier_gate.minimum_carriers_per_state == 5
    assert manifest.multiplicity.threshold_formula == "0.05/eligible_block_count"
    assert manifest.biological_claims_permitted is False

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["regions"][0]["ws276_start"] += 1
    with pytest.raises(ValidationError, match="frozen envelope"):
        hp.HaplotypePavManifest.model_validate(payload)

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["status"] = "qualified_for_file_backed_descriptive_execution"
    payload["null_dependency"]["interpretation_receipt_status"] = "missing"
    payload["null_dependency"]["interpretation_receipt_relative_path"] = None
    payload["null_dependency"]["interpretation_receipt_sha256"] = None
    with pytest.raises(ValidationError, match="frozen interpretation receipt"):
        hp.HaplotypePavManifest.model_validate(payload)

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["real_execution_handoff"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        hp.HaplotypePavManifest.model_validate(payload)


def test_coordinate_identity_is_outcome_blind_and_fail_closed(tmp_path: Path) -> None:
    manifest = hp.load_manifest(CONFIG)
    qualification = hp.qualify_coordinate_identity(
        manifest, _write_coordinate_receipt(tmp_path)
    )
    assert qualification.qualified is True
    assert qualification.region_ids == ["chrv_left", "chrv_right"]

    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(ValidationError, match="coordinate identity"):
        hp.qualify_coordinate_identity(manifest, _write_coordinate_receipt(bad, mismatch=True))


def test_active_null_blocks_plan_until_success_and_interpretation(tmp_path: Path) -> None:
    manifest = hp.load_manifest(CONFIG)
    active = tmp_path / hp.NULL_RUN_ID
    active.mkdir()
    state = hp.qualify_null_dependency(manifest, active)
    assert state.state == "active_nonterminal"
    assert state.analysis_plan_permitted is False

    completed, interpretation = _write_null_fixture(tmp_path / "completed")
    waiting = hp.qualify_null_dependency(manifest, completed)
    assert waiting.state == "success_awaiting_interpretation"
    assert waiting.analysis_plan_permitted is False
    with pytest.raises(ValueError, match="interpretation receipt checksum mismatch"):
        hp.qualify_null_dependency(manifest, completed, interpretation)


def test_typed_state_panel_carrier_gates_and_missingness(tmp_path: Path) -> None:
    manifest = hp.load_manifest(CONFIG)
    qualification = hp.qualify_state_panel(
        manifest,
        _coordinate(tmp_path),
        _blocks(),
        _calls(),
        _sample_order(),
    )
    assert qualification.sample_count == 209
    assert qualification.eligible_block_count == 2
    haplotype = qualification.blocks[0]
    assert haplotype.callable_count == 204
    assert haplotype.unresolved_count == 5
    assert {item.semantic_state for item in haplotype.state_counts} == {
        "haplotype:reference_path",
        "haplotype:alternate_path",
    }
    pav = qualification.blocks[1]
    assert pav.callable_count == 209
    assert {item.semantic_state for item in pav.state_counts} == {
        "pav:present",
        "pav:absent",
    }

    rare = hp.qualify_state_panel(
        manifest,
        _coordinate(tmp_path),
        _blocks(),
        _calls(rare_haplotype=True),
        _sample_order(),
    )
    assert rare.eligible_block_count == 1
    assert rare.blocks[0].eligible is False
    assert "state_below_minimum_carriers:hap_alt" in rare.blocks[0].exclusion_reasons


def test_pav_copy_number_and_complete_call_gates_are_typed(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="copy number"):
        hp.PavStateCall(
            sample_id="strain_001",
            block_id="right_pav_locus_01",
            region_id="chrv_right",
            assembly=hp.WS283_ASSEMBLY,
            state_id="pav_absent",
            confidence=1.0,
            kind="pav",
            state="absent",
            locus_id="pav_locus_01",
            copy_number=1,
        )

    calls = _calls()
    calls.pop()
    with pytest.raises(ValueError, match="one call per sample"):
        hp.qualify_state_panel(
            hp.load_manifest(CONFIG),
            _coordinate(tmp_path),
            _blocks(),
            calls,
            _sample_order(),
        )


def test_analysis_plan_is_kinship_aware_cellwise_and_descriptive(tmp_path: Path) -> None:
    manifest = hp.load_manifest(CONFIG)
    coordinate = _coordinate(tmp_path)
    state_panel = hp.qualify_state_panel(
        manifest, coordinate, _blocks(), _calls(), _sample_order()
    )
    plan = hp.build_synthetic_analysis_plan(
        manifest, coordinate, _synthetic_dependency(), state_panel
    )

    assert plan.model_cell_count == 16
    assert plan.request_count == 32
    assert plan.block_bonferroni_alpha == pytest.approx(0.025)
    assert all(item.chromosome_excluded_from_kinship == 5 for item in plan.requests)
    assert all(item.descriptive_only for item in plan.requests)
    assert plan.execution_scope == "synthetic_fixture"
    assert plan.real_execution_permitted is False
    assert all(not item.causal_interpretation_permitted for item in plan.requests)
    boundary_requests = [
        item for item in plan.requests if item.cell_id in hp.EXPECTED_BOUNDARY_CELLS
    ]
    assert len(boundary_requests) == 4
    assert all(not item.null_threshold_eligible for item in boundary_requests)
    assert {
        item.kinship_resource for item in plan.requests
    } == {
        "build_full_marker_chr5_excluded_grm",
        "reuse_ldpruned_sensitivity_chr5_excluded_grm",
    }


def test_kinship_backend_interface_checks_shape_and_symmetry(tmp_path: Path) -> None:
    manifest = hp.load_manifest(CONFIG)
    coordinate = _coordinate(tmp_path)
    panel = hp.qualify_state_panel(manifest, coordinate, _blocks(), _calls(), _sample_order())
    request = hp.build_synthetic_analysis_plan(
        manifest, coordinate, _synthetic_dependency(), panel
    ).requests[0]
    request = hp.KinshipAwareAnalysisRequest.model_validate(
        {**request.model_dump(), "callable_samples": 3}
    )
    inputs = hp.KinshipAwareInputs(
        sample_ids=("a", "b", "c"),
        phenotype=(1.0, 2.0, 3.0),
        state_ids=("hap_ref", "hap_alt", "hap_ref"),
        kinship=((1.0, 0.1, 0.0), (0.1, 1.0, 0.2), (0.0, 0.2, 1.0)),
        covariates=((1.0,), (1.0,), (1.0,)),
    )
    with pytest.raises(ValueError, match="cannot be submitted to a real backend"):
        hp.validate_backend_inputs(request, inputs)
    hp.validate_backend_inputs(request, inputs, allow_synthetic_fixture=True)

    asymmetric = hp.KinshipAwareInputs(
        sample_ids=inputs.sample_ids,
        phenotype=inputs.phenotype,
        state_ids=inputs.state_ids,
        kinship=((1.0, 0.5, 0.0), (0.1, 1.0, 0.2), (0.0, 0.2, 1.0)),
        covariates=inputs.covariates,
    )
    with pytest.raises(ValueError, match="not symmetric"):
        hp.validate_backend_inputs(request, asymmetric, allow_synthetic_fixture=True)


@pytest.mark.skip(
    reason="requires private upstream receipts that are intentionally excluded"
)
def test_real_handoff_audit_is_hash_bound_and_reports_current_blockers(
    tmp_path: Path,
) -> None:
    manifest = hp.load_manifest(CONFIG)
    audit = hp.audit_real_execution_handoff(manifest, HANDOFF_CONFIG)
    assert audit.handoff_manifest_sha256 == hp.HANDOFF_SHA256
    assert audit.real_execution_ready is False
    assert audit.gate_statuses["full_kernels"] == "qualified"
    assert audit.gate_statuses["population_groups"] == "qualified"
    assert audit.gate_statuses["coordinate_identity"] == "qualified"
    assert "handoff_gate_not_qualified:state_panel" in audit.blockers
    assert "handoff_gate_not_qualified:coordinate_identity" not in audit.blockers
    assert "coordinate_identity_receipt_not_frozen_in_handoff" not in audit.blockers
    assert "null_interpretation_receipt_not_frozen" not in audit.blockers
    assert "regional_state_manifest_blocked_on_real_pangenome_inputs" in audit.blockers
    assert audit.null_interpretation_receipt_sha256 == hp.NULL_INTERPRETATION_SHA256
    assert audit.operator_outcome_blinding_asserted is False

    tampered = tmp_path / HANDOFF_CONFIG.name
    tampered.write_bytes(HANDOFF_CONFIG.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="handoff manifest checksum mismatch"):
        hp.audit_real_execution_handoff(manifest, tampered)

    isolated = tmp_path / "isolated/private-config" / HANDOFF_CONFIG.name
    isolated.parent.mkdir(parents=True)
    isolated.write_bytes(HANDOFF_CONFIG.read_bytes())
    missing_receipt = hp.audit_real_execution_handoff(manifest, isolated)
    assert "frozen_null_interpretation_receipt_missing_or_changed" in (
        missing_receipt.blockers
    )


@pytest.mark.skip(
    reason="requires private upstream receipts that are intentionally excluded"
)
def test_real_planner_cannot_be_released_by_in_memory_qualifications() -> None:
    manifest = hp.load_manifest(CONFIG)
    with pytest.raises(ValueError, match="real regional-state qualification execution remains blocked"):
        hp.build_analysis_plan(
            manifest,
            handoff_manifest_path=HANDOFF_CONFIG,
            handoff_roots={},
            null_run_root=ROOT / "not-consulted",
            interpretation_path=ROOT / "not-consulted.json",
        )
