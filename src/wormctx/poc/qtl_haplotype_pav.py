"""Governed chromosome-V haplotype/PAV qualification scaffold.

This module deliberately stops before fitting a biological association model.  It
freezes the two chromosome-V regions, qualifies WS276/WS283 coordinate identity,
validates typed haplotype and presence/absence calls, applies carrier-count gates,
and can emit non-executable synthetic requests for interface testing.  Real requests
are fail-closed: they require an exact frozen regional-state qualification and held-out prediction handoff, verified file bytes and
receipt semantics, and a separately frozen null-interpretation receipt.  Boolean
attestations or symbolic resource names are never sufficient to release execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .qtl import TRAITS, TRAIT_SLUGS


SCHEMA_VERSION = "wormctx-abamectin-ws283-chrv-haplotype-pav-scaffold-1.1"
COORDINATE_RECEIPT_VERSION = "wormctx-ws276-ws283-coordinate-identity-1.0"
NULL_INTERPRETATION_VERSION = "wormctx-ws283-null-interpretation-gate-1.0"
STATE_PANEL_VERSION = "wormctx-chrv-haplotype-pav-panel-1.0"
PLAN_VERSION = "wormctx-chrv-haplotype-pav-analysis-plan-1.1"
HANDOFF_AUDIT_VERSION = "wormctx-chrv-haplotype-pav-handoff-audit-1.0"
HANDOFF_SCHEMA_VERSION = "wormctx-abamectin-regional_state-heldout_prediction-real-input-handoff-1.1"
HANDOFF_ANALYSIS_ID = "abamectin_ws283_regional_state_heldout_prediction_real_input_handoff_20260721"
HANDOFF_RELATIVE_PATH = (
    "experiments/natural_variation/pangenome_state_qualification/config/state_input_handoff.json"
)
HANDOFF_SHA256 = "6cf173cbdfb3c112cbe517c11e3bfab3f467a8e50e469fb8426c1b7af95c67c6"
NULL_INTERPRETATION_RELATIVE_PATH = (
    "artifacts/upstream/parametric_polygenic_null_calibration/interpretation.json"
)
NULL_INTERPRETATION_SHA256 = (
    "aa933f7610a875feafebf44f108c29320785e86e714d33071a42108b07d92565"
)
COORDINATE_HANDOFF_RECEIPT_ROLE = "ws276_ws283_coordinate_identity_receipt"
REQUIRED_HANDOFF_GATES = (
    "coordinate_identity",
    "state_panel",
    "ldpruned_chr5_excluded_grm",
    "full_kernels",
    "population_groups",
)
REQUIRED_STATE_ASSET_ROLES = (
    "real_haplotype_pav_state_panel",
    "real_haplotype_pav_sample_order",
    "real_haplotype_pav_call_definition_receipt",
    "real_haplotype_pav_qualification_receipt",
)

WS276_ASSEMBLY = "PRJNA13758.WS276"
WS283_ASSEMBLY = "PRJNA13758.WS283_inferred_not_declared_in_vcf_header"
NULL_RUN_ID = "abamectin-ws283-parametric-null-20260721T161005Z-c01179f4a52e"
NULL_SOURCE_COMMIT = "c01179f4a52e97fcc6c0d1bc1df07e03fe2215c8"
EXPECTED_BOUNDARY_CELLS = ("ldpruned_pc0_norm_n", "ldpruned_pc10_norm_n")
RELATIONSHIP_KINDS = ("full", "ldpruned")
ENDPOINTS = ("pc0", "pc10")
EXPECTED_REGIONS = {
    "chrv_left": (1_747_612, 4_333_001),
    "chrv_right": (13_606_517, 16_754_986),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RegionContract(_StrictModel):
    id: Literal["chrv_left", "chrv_right"]
    chromosome: Literal[5]
    source_assembly: Literal[WS276_ASSEMBLY]
    target_assembly: Literal[WS283_ASSEMBLY]
    ws276_start: int = Field(gt=0)
    ws276_end: int = Field(gt=0)
    ws283_query_start: int = Field(gt=0)
    ws283_query_end: int = Field(gt=0)
    coordinate_identity_assumed: Literal[False]
    identity_receipt_required_before_state_ingestion: Literal[True]

    @model_validator(mode="after")
    def exact_region(self) -> "RegionContract":
        expected = EXPECTED_REGIONS[self.id]
        observed_source = (self.ws276_start, self.ws276_end)
        observed_target = (self.ws283_query_start, self.ws283_query_end)
        if observed_source != expected or observed_target != expected:
            raise ValueError(f"{self.id} coordinates differ from the frozen envelope")
        return self


class CoordinateGateContract(_StrictModel):
    mapping_mode: Literal["sequence_identical_coordinate_identity"]
    source_and_target_reference_sha256_required: Literal[True]
    chromosome_v_sha256_required: Literal[True]
    interval_sequence_sha256_required: Literal[True]
    numerical_coordinate_equality_required: Literal[True]
    outcome_blind_receipt_required: Literal[True]
    liftover_fallback_permitted_in_v1: Literal[False]


class NullDependencyContract(_StrictModel):
    run_id: Literal[NULL_RUN_ID]
    source_git_commit: Literal[NULL_SOURCE_COMMIT]
    frozen_contract_relative_path: Literal[
        "receipts/frozen_parametric_null_contract.json"
    ]
    frozen_contract_sha256: str
    terminal_success_required_before_analysis_plan: Literal[True]
    interpretation_receipt_required: Literal[True]
    interpretation_receipt_status: Literal["missing", "frozen"]
    interpretation_receipt_relative_path: (
        Literal[NULL_INTERPRETATION_RELATIVE_PATH] | None
    )
    interpretation_receipt_sha256: Literal[NULL_INTERPRETATION_SHA256] | None
    interpretation_receipt_must_be_frozen_in_manifest: Literal[True]
    expected_calibration_cells: Literal[16]
    expected_replicates_per_cell: Literal[100]
    expected_completed_maps: Literal[1600]
    expected_threshold_eligible_cells: Literal[14]
    expected_boundary_cells: list[str]
    scaffold_development_permitted_while_nonterminal: Literal[True]
    marker_null_is_not_haplotype_test_calibration: Literal[True]

    @field_validator("frozen_contract_sha256", "interpretation_receipt_sha256")
    @classmethod
    def valid_hash(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("frozen null receipt SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def exact_boundaries(self) -> "NullDependencyContract":
        if self.expected_boundary_cells != list(EXPECTED_BOUNDARY_CELLS):
            raise ValueError("parametric-null boundary cells differ from the frozen run")
        frozen = (
            self.interpretation_receipt_sha256 is not None
            and self.interpretation_receipt_relative_path is not None
        )
        if (self.interpretation_receipt_sha256 is None) != (
            self.interpretation_receipt_relative_path is None
        ):
            raise ValueError("null interpretation path and SHA-256 must be frozen together")
        if frozen != (self.interpretation_receipt_status == "frozen"):
            raise ValueError(
                "null interpretation status must match its frozen receipt SHA-256"
            )
        return self


class RealExecutionHandoffContract(_StrictModel):
    manifest_relative_path: Literal[HANDOFF_RELATIVE_PATH]
    manifest_sha256: Literal[HANDOFF_SHA256]
    schema_version: Literal[HANDOFF_SCHEMA_VERSION]
    analysis_id: Literal[HANDOFF_ANALYSIS_ID]
    required_qualified_gates: list[
        Literal[
            "coordinate_identity",
            "state_panel",
            "ldpruned_chr5_excluded_grm",
            "full_kernels",
            "population_groups",
        ]
    ]
    allowed_nonqualified_advisory_gates: list[Literal["phenotype_provenance"]]
    coordinate_receipt_role: Literal[COORDINATE_HANDOFF_RECEIPT_ROLE]
    required_state_asset_roles: list[str]
    runtime_root_mapping_required: Literal[True]
    every_execution_asset_byte_and_sha256_verified: Literal[True]
    every_execution_receipt_semantically_verified: Literal[True]
    symbolic_resource_names_are_execution_evidence: Literal[False]
    runtime_boolean_self_attestation_permitted: Literal[False]
    operator_outcome_blinding_asserted: Literal[False]
    outcome_access_claim_scope: Literal["derivation_execution_paths_only"]

    @model_validator(mode="after")
    def exact_handoff(self) -> "RealExecutionHandoffContract":
        if self.required_qualified_gates != list(REQUIRED_HANDOFF_GATES):
            raise ValueError("real execution gates differ from the frozen handoff contract")
        if self.allowed_nonqualified_advisory_gates != ["phenotype_provenance"]:
            raise ValueError("only raw phenotype provenance may remain advisory")
        if self.required_state_asset_roles != list(REQUIRED_STATE_ASSET_ROLES):
            raise ValueError("real state-panel assets differ from the frozen contract")
        return self


class StateContract(_StrictModel):
    target_assembly: Literal[WS283_ASSEMBLY]
    expected_samples: Literal[209]
    sample_order: Literal["baseline_parent_fam_order"]
    allowed_kinds: list[Literal["haplotype", "pav"]]
    one_call_per_sample_per_block: Literal[True]
    unresolved_is_missing_not_reference: Literal[True]
    absent_pav_is_explicit_not_missing: Literal[True]
    block_definitions_frozen_before_phenotype_access: Literal[True]

    @model_validator(mode="after")
    def exact_kinds(self) -> "StateContract":
        if self.allowed_kinds != ["haplotype", "pav"]:
            raise ValueError("state kinds must be ordered haplotype, PAV")
        return self


class CarrierGateContract(_StrictModel):
    minimum_carriers_per_state: Literal[5]
    minimum_noncarriers_per_state: Literal[5]
    maximum_unresolved_fraction: Literal[0.1]
    rare_state_policy: Literal["exclude_entire_block_without_collapsing_states"]
    require_reference_and_alternative: Literal[True]
    gate_evaluated_without_phenotypes: Literal[True]


class KinshipContract(_StrictModel):
    relationships: list[Literal["full", "ldpruned"]]
    endpoints: list[Literal["pc0", "pc10"]]
    traits: list[str]
    chromosome_excluded_from_kinship: Literal[5]
    primary_test: Literal["kinship_aware_block_omnibus"]
    secondary_state_contrasts: Literal[
        "within_block_holm_descriptive_only_after_primary"
    ]
    full_kinship_resource: Literal["build_full_marker_chr5_excluded_grm"]
    ldpruned_kinship_resource: Literal[
        "reuse_ldpruned_sensitivity_chr5_excluded_grm"
    ]
    model_selection_from_known_locus_permitted: Literal[False]
    pooled_null_thresholds_permitted: Literal[False]

    @model_validator(mode="after")
    def exact_grid(self) -> "KinshipContract":
        if self.relationships != list(RELATIONSHIP_KINDS):
            raise ValueError("relationship kinds differ from the frozen grid")
        if self.endpoints != list(ENDPOINTS):
            raise ValueError("PC endpoints differ from the frozen grid")
        if self.traits != list(TRAITS):
            raise ValueError("traits differ from the four frozen abamectin traits")
        return self


class MultiplicityContract(_StrictModel):
    family_alpha: Literal[0.05]
    primary_unit: Literal["outcome_blind_eligible_block"]
    family_scope: Literal[
        "within_trait_relationship_endpoint_across_both_frozen_regions"
    ]
    primary_correction: Literal["bonferroni_over_eligible_blocks"]
    threshold_formula: Literal["0.05/eligible_block_count"]
    state_contrasts_in_primary_family: Literal[False]
    no_cross_cell_pooling: Literal[True]


class ClaimContract(_StrictModel):
    permitted: list[str]
    prohibited: list[str]


class HaplotypePavManifest(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    analysis_id: Literal["abamectin_ws283_chrv_haplotype_pav_first_wave_v1"]
    classification: Literal["governed_descriptive_haplotype_pav_scaffold"]
    status: Literal[
        "blocked_on_frozen_interpretation_and_real_pangenome_inputs",
        "blocked_on_real_pangenome_inputs",
        "qualified_for_file_backed_descriptive_execution",
    ]
    validated: Literal[False]
    biological_claims_permitted: Literal[False]
    regions: list[RegionContract]
    coordinate_gate: CoordinateGateContract
    null_dependency: NullDependencyContract
    states: StateContract
    carrier_gate: CarrierGateContract
    kinship: KinshipContract
    multiplicity: MultiplicityContract
    real_execution_handoff: RealExecutionHandoffContract
    claims: ClaimContract

    @model_validator(mode="after")
    def exact_scaffold(self) -> "HaplotypePavManifest":
        if [region.id for region in self.regions] != ["chrv_left", "chrv_right"]:
            raise ValueError("regions must be ordered left then right")
        prohibited = " ".join(self.claims.prohibited).lower()
        for token in ("causal", "independent replication", "model selection"):
            if token not in prohibited:
                raise ValueError(f"claim boundary must prohibit {token}")
        if (
            self.status == "qualified_for_file_backed_descriptive_execution"
            and self.null_dependency.interpretation_receipt_status != "frozen"
        ):
            raise ValueError(
                "real execution cannot be qualified without a frozen interpretation receipt"
            )
        return self


def load_manifest(path: str | Path) -> HaplotypePavManifest:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"haplotype/PAV manifest is not a file: {source}")
    return HaplotypePavManifest.model_validate_json(source.read_text(encoding="utf-8"))


class RegionIdentityRecord(_StrictModel):
    region_id: Literal["chrv_left", "chrv_right"]
    source_assembly: Literal[WS276_ASSEMBLY]
    target_assembly: Literal[WS283_ASSEMBLY]
    source_chromosome: Literal[5]
    target_chromosome: Literal[5]
    source_start: int = Field(gt=0)
    source_end: int = Field(gt=0)
    target_start: int = Field(gt=0)
    target_end: int = Field(gt=0)
    mapping_mode: Literal["sequence_identical_coordinate_identity"]
    source_interval_sha256: str
    target_interval_sha256: str

    @field_validator("source_interval_sha256", "target_interval_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("interval SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def sequence_and_coordinate_identity(self) -> "RegionIdentityRecord":
        if (self.source_start, self.source_end) != (self.target_start, self.target_end):
            raise ValueError("v1 requires exact numerical coordinate identity")
        if self.source_interval_sha256 != self.target_interval_sha256:
            raise ValueError("v1 requires byte-identical interval sequence")
        return self


class CoordinateIdentityReceipt(_StrictModel):
    schema_version: Literal[COORDINATE_RECEIPT_VERSION]
    source_assembly: Literal[WS276_ASSEMBLY]
    target_assembly: Literal[WS283_ASSEMBLY]
    source_reference_fasta_sha256: str
    target_reference_fasta_sha256: str
    source_reference_fai_sha256: str
    target_reference_fai_sha256: str
    source_chromosome_v_length: Literal[20_924_180]
    target_chromosome_v_length: Literal[20_924_180]
    source_chromosome_v_sha256: str
    target_chromosome_v_sha256: str
    full_reference_byte_identical: Literal[True]
    generated_before_state_ingestion: Literal[True]
    phenotype_accessed: Literal[False]
    association_results_accessed: Literal[False]
    operator_outcome_blinding_asserted: Literal[False]
    outcome_access_claim_scope: Literal["qualification_execution_path_only"]
    regions: list[RegionIdentityRecord]

    @field_validator(
        "source_reference_fasta_sha256",
        "target_reference_fasta_sha256",
        "source_reference_fai_sha256",
        "target_reference_fai_sha256",
        "source_chromosome_v_sha256",
        "target_chromosome_v_sha256",
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("assembly SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def exact_identity_evidence(self) -> "CoordinateIdentityReceipt":
        if self.source_reference_fasta_sha256 != self.target_reference_fasta_sha256:
            raise ValueError("coordinate receipt references are not byte-identical")
        if self.source_reference_fai_sha256 != self.target_reference_fai_sha256:
            raise ValueError("coordinate receipt FASTA indexes are not identical")
        if self.source_chromosome_v_sha256 != self.target_chromosome_v_sha256:
            raise ValueError("coordinate receipt chromosome-V sequences are not identical")
        if [item.region_id for item in self.regions] != ["chrv_left", "chrv_right"]:
            raise ValueError("coordinate receipt regions differ from the frozen order")
        return self


class CoordinateQualification(_StrictModel):
    qualified: Literal[True]
    receipt_sha256: str
    source_assembly: Literal[WS276_ASSEMBLY]
    target_assembly: Literal[WS283_ASSEMBLY]
    region_ids: list[str]
    qualification_scope: Literal["synthetic_unbound_receipt", "frozen_real_handoff"]
    handoff_manifest_sha256: str | None

    @model_validator(mode="after")
    def binding_matches_scope(self) -> "CoordinateQualification":
        bound = self.handoff_manifest_sha256 is not None
        if bound != (self.qualification_scope == "frozen_real_handoff"):
            raise ValueError("coordinate handoff hash must match qualification scope")
        return self


def qualify_coordinate_identity(
    manifest: HaplotypePavManifest, receipt_path: str | Path
) -> CoordinateQualification:
    """Validate an unbound coordinate receipt for synthetic scaffold tests only.

    A receipt supplied by its own author is not execution evidence.  Real planning
    uses :func:`audit_real_execution_handoff` and requires the receipt itself to be
    a byte-pinned asset in the frozen handoff.
    """
    path = Path(receipt_path).resolve()
    receipt = CoordinateIdentityReceipt.model_validate_json(path.read_text(encoding="utf-8"))
    expected_regions = {region.id: region for region in manifest.regions}
    if [record.region_id for record in receipt.regions] != list(expected_regions):
        raise ValueError("coordinate receipt regions differ from the frozen order")
    for record in receipt.regions:
        contract = expected_regions[record.region_id]
        if (record.source_start, record.source_end) != (
            contract.ws276_start,
            contract.ws276_end,
        ):
            raise ValueError(f"{record.region_id} WS276 interval differs from the contract")
        if (record.target_start, record.target_end) != (
            contract.ws283_query_start,
            contract.ws283_query_end,
        ):
            raise ValueError(f"{record.region_id} WS283 interval differs from the contract")
    return CoordinateQualification(
        qualified=True,
        receipt_sha256=_sha256(path),
        source_assembly=receipt.source_assembly,
        target_assembly=receipt.target_assembly,
        region_ids=[record.region_id for record in receipt.regions],
        qualification_scope="synthetic_unbound_receipt",
        handoff_manifest_sha256=None,
    )


class NullAggregationCell(_StrictModel):
    cell_id: str
    replicates_attempted: Literal[100]
    replicates_completed: Literal[100]
    nominal_bonferroni_fwer: dict[str, Any]
    anchor_boundary: bool
    threshold_eligible: bool


class NullAggregation(_StrictModel):
    schema_version: Literal[
        "wormctx-abamectin-ws283-parametric-null-aggregation-1.0"
    ]
    manifest_sha256: str
    rich_summary: dict[str, str]
    calibration_cells: Literal[16]
    replicates_per_cell: Literal[100]
    attempted_null_maps: Literal[1600]
    completed_null_maps: Literal[1600]
    cells: list[NullAggregationCell]
    validated: Literal[False]
    biological_claims_permitted: Literal[False]


class NullInterpretationReceipt(_StrictModel):
    schema_version: Literal[NULL_INTERPRETATION_VERSION]
    null_run_id: Literal[NULL_RUN_ID]
    aggregation_sha256: str
    decision: Literal[
        "proceed_with_descriptive_haplotype_scaffold",
        "revise_model_before_haplotype_scaffold",
    ]
    reviewed_before_haplotype_outcomes: Literal[True]
    marker_null_not_treated_as_block_test_calibration: Literal[True]
    causal_interpretation_permitted: Literal[False]

    @field_validator("aggregation_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("aggregation SHA-256 must be lowercase hexadecimal")
        return value


class NullDependencyQualification(_StrictModel):
    state: Literal[
        "active_nonterminal",
        "success_awaiting_interpretation",
        "success_descriptive_scaffold_enabled",
        "success_revision_required",
    ]
    run_id: Literal[NULL_RUN_ID]
    terminal_success: bool
    analysis_plan_permitted: bool
    scaffold_development_permitted: Literal[True]
    aggregation_sha256: str | None
    interpretation_receipt_sha256: str | None
    threshold_eligible_cells: list[str]
    boundary_cells: list[str]


def _expected_cell_ids() -> list[str]:
    return [
        f"{relationship}_{endpoint}_{TRAIT_SLUGS[trait]}"
        for relationship in RELATIONSHIP_KINDS
        for endpoint in ENDPOINTS
        for trait in TRAITS
    ]


def qualify_null_dependency(
    manifest: HaplotypePavManifest,
    run_root: str | Path,
    interpretation_path: str | Path | None = None,
) -> NullDependencyQualification:
    root = Path(run_root).resolve()
    dependency = manifest.null_dependency
    if root.name != dependency.run_id:
        raise ValueError("parametric-null run directory name differs from the frozen run ID")
    if (root / "FAILURE").exists():
        raise ValueError("parametric-null dependency has a FAILURE marker")
    success = root / "SUCCESS"
    if not success.exists():
        return NullDependencyQualification(
            state="active_nonterminal",
            run_id=dependency.run_id,
            terminal_success=False,
            analysis_plan_permitted=False,
            scaffold_development_permitted=True,
            aggregation_sha256=None,
            interpretation_receipt_sha256=None,
            threshold_eligible_cells=[],
            boundary_cells=list(EXPECTED_BOUNDARY_CELLS),
        )
    if success.read_bytes() != b"SUCCESS\n":
        raise ValueError("parametric-null SUCCESS marker is not exact")
    source_revision = root / "receipts/SOURCE_REVISION"
    if source_revision.read_text(encoding="utf-8").strip() != dependency.source_git_commit:
        raise ValueError("parametric-null source revision differs from the frozen dependency")
    frozen_contract = root / dependency.frozen_contract_relative_path
    if _sha256(frozen_contract) != dependency.frozen_contract_sha256:
        raise ValueError("parametric-null frozen contract checksum mismatch")
    aggregation_path = root / "summary/parametric_null_aggregation.json"
    aggregation = NullAggregation.model_validate_json(
        aggregation_path.read_text(encoding="utf-8")
    )
    if aggregation.manifest_sha256 != dependency.frozen_contract_sha256:
        raise ValueError("parametric-null aggregation is bound to a different contract")
    if [cell.cell_id for cell in aggregation.cells] != _expected_cell_ids():
        raise ValueError("parametric-null aggregation cells differ from the frozen grid")
    boundary = [cell.cell_id for cell in aggregation.cells if cell.anchor_boundary]
    eligible = [cell.cell_id for cell in aggregation.cells if cell.threshold_eligible]
    if boundary != list(EXPECTED_BOUNDARY_CELLS):
        raise ValueError("parametric-null boundary cells differ from the expected stress tests")
    if len(eligible) != dependency.expected_threshold_eligible_cells:
        raise ValueError("parametric-null threshold-eligible cell count differs from contract")
    if any(cell.threshold_eligible for cell in aggregation.cells if cell.anchor_boundary):
        raise ValueError("a boundary cell cannot supply an empirical threshold")
    aggregation_sha256 = _sha256(aggregation_path)
    if interpretation_path is None:
        return NullDependencyQualification(
            state="success_awaiting_interpretation",
            run_id=dependency.run_id,
            terminal_success=True,
            analysis_plan_permitted=False,
            scaffold_development_permitted=True,
            aggregation_sha256=aggregation_sha256,
            interpretation_receipt_sha256=None,
            threshold_eligible_cells=eligible,
            boundary_cells=boundary,
        )
    interpretation_source = Path(interpretation_path).resolve()
    if dependency.interpretation_receipt_status != "frozen":
        raise ValueError(
            "an operator-supplied null interpretation cannot release real planning; "
            "freeze its exact SHA-256 in the manifest first"
        )
    if _sha256(interpretation_source) != dependency.interpretation_receipt_sha256:
        raise ValueError("null interpretation receipt checksum mismatch")
    interpretation = NullInterpretationReceipt.model_validate_json(
        interpretation_source.read_text(encoding="utf-8")
    )
    if interpretation.aggregation_sha256 != aggregation_sha256:
        raise ValueError("null interpretation is bound to a different aggregation")
    proceed = interpretation.decision == "proceed_with_descriptive_haplotype_scaffold"
    return NullDependencyQualification(
        state=(
            "success_descriptive_scaffold_enabled"
            if proceed
            else "success_revision_required"
        ),
        run_id=dependency.run_id,
        terminal_success=True,
        analysis_plan_permitted=proceed,
        scaffold_development_permitted=True,
        aggregation_sha256=aggregation_sha256,
        interpretation_receipt_sha256=_sha256(interpretation_source),
        threshold_eligible_cells=eligible,
        boundary_cells=boundary,
    )


class BlockDefinition(_StrictModel):
    block_id: str
    region_id: Literal["chrv_left", "chrv_right"]
    chromosome: Literal[5]
    assembly: Literal[WS283_ASSEMBLY]
    start: int = Field(gt=0)
    end: int = Field(gt=0)
    state_kind: Literal["haplotype", "pav"]
    reference_state_id: str
    definition_method: Literal[
        "outcome_blind_pangenome_path_cluster",
        "outcome_blind_presence_absence_locus",
    ]
    source_snapshot_sha256: str
    phenotype_accessed_during_definition: Literal[False]

    @field_validator("block_id", "reference_state_id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("block and state identifiers must be safe tokens")
        return value

    @field_validator("source_snapshot_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("block source SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def valid_interval_and_method(self) -> "BlockDefinition":
        if self.end < self.start:
            raise ValueError("block end precedes start")
        expected_method = (
            "outcome_blind_pangenome_path_cluster"
            if self.state_kind == "haplotype"
            else "outcome_blind_presence_absence_locus"
        )
        if self.definition_method != expected_method:
            raise ValueError("block definition method does not match state kind")
        return self


class _BaseStateCall(_StrictModel):
    sample_id: str
    block_id: str
    region_id: Literal["chrv_left", "chrv_right"]
    assembly: Literal[WS283_ASSEMBLY]
    state_id: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("sample_id", "block_id", "state_id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("sample, block, and state identifiers must be safe tokens")
        return value


class HaplotypeStateCall(_BaseStateCall):
    kind: Literal["haplotype"]
    state: Literal[
        "reference_path", "alternate_path", "structural_haplotype", "unresolved"
    ]
    path_id: str | None

    @model_validator(mode="after")
    def path_semantics(self) -> "HaplotypeStateCall":
        if self.state == "unresolved":
            if self.path_id is not None:
                raise ValueError("unresolved haplotype cannot have a path ID")
        elif self.path_id is None or not _SAFE_ID_RE.fullmatch(self.path_id):
            raise ValueError("resolved haplotype requires a safe path ID")
        return self


class PavStateCall(_BaseStateCall):
    kind: Literal["pav"]
    state: Literal["present", "absent", "multi_copy", "unresolved"]
    locus_id: str
    copy_number: int | None = Field(default=None, ge=0)

    @field_validator("locus_id")
    @classmethod
    def safe_locus(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("PAV locus ID must be a safe token")
        return value

    @model_validator(mode="after")
    def copy_number_semantics(self) -> "PavStateCall":
        expected = {
            "absent": lambda value: value == 0,
            "present": lambda value: value == 1,
            "multi_copy": lambda value: value is not None and value >= 2,
            "unresolved": lambda value: value is None,
        }
        if not expected[self.state](self.copy_number):
            raise ValueError("PAV state and copy number are inconsistent")
        return self


GenomicStateCall = Annotated[
    HaplotypeStateCall | PavStateCall, Field(discriminator="kind")
]
_STATE_ADAPTER = TypeAdapter(GenomicStateCall)


def load_state_calls(path: str | Path) -> list[GenomicStateCall]:
    source = Path(path).resolve()
    calls: list[GenomicStateCall] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            calls.append(_STATE_ADAPTER.validate_json(raw))
        except Exception as exc:
            raise ValueError(f"invalid state record at line {line_number}: {exc}") from exc
    return calls


class StateCount(_StrictModel):
    state_id: str
    semantic_state: str
    carriers: int = Field(ge=0)
    noncarriers: int = Field(ge=0)


class BlockQualification(_StrictModel):
    block_id: str
    region_id: str
    state_kind: Literal["haplotype", "pav"]
    reference_state_id: str
    sample_count: int
    callable_count: int
    unresolved_count: int
    unresolved_fraction: float
    state_counts: list[StateCount]
    eligible: bool
    exclusion_reasons: list[str]


class StatePanelQualification(_StrictModel):
    schema_version: Literal[STATE_PANEL_VERSION]
    coordinate_receipt_sha256: str
    sample_count: Literal[209]
    block_count: int = Field(gt=0)
    eligible_block_count: int = Field(ge=0)
    blocks: list[BlockQualification]
    outcome_blind: Literal[True]
    qualification_scope: Literal["synthetic_unbound_inputs", "frozen_real_handoff"]
    handoff_manifest_sha256: str | None
    state_panel_asset_sha256: str | None
    sample_order_asset_sha256: str | None
    definition_receipt_sha256: str | None

    @model_validator(mode="after")
    def binding_matches_scope(self) -> "StatePanelQualification":
        hashes = (
            self.handoff_manifest_sha256,
            self.state_panel_asset_sha256,
            self.sample_order_asset_sha256,
            self.definition_receipt_sha256,
        )
        bound = all(item is not None for item in hashes)
        if any(item is not None for item in hashes) and not bound:
            raise ValueError("real state-panel qualification requires every binding hash")
        if bound != (self.qualification_scope == "frozen_real_handoff"):
            raise ValueError("state-panel hashes must match qualification scope")
        return self


def _is_unresolved(call: GenomicStateCall) -> bool:
    return call.state == "unresolved"


def qualify_state_panel(
    manifest: HaplotypePavManifest,
    coordinate: CoordinateQualification,
    blocks: Sequence[BlockDefinition],
    calls: Sequence[GenomicStateCall],
    sample_order: Sequence[str],
) -> StatePanelQualification:
    if not coordinate.qualified:
        raise ValueError("coordinate identity must qualify before state ingestion")
    if len(sample_order) != manifest.states.expected_samples:
        raise ValueError("sample order must contain exactly 209 strains")
    if len(set(sample_order)) != len(sample_order):
        raise ValueError("sample order contains duplicate strain IDs")
    if any(not _SAFE_ID_RE.fullmatch(sample) for sample in sample_order):
        raise ValueError("sample order contains an unsafe strain ID")
    block_by_id = {block.block_id: block for block in blocks}
    if not block_by_id or len(block_by_id) != len(blocks):
        raise ValueError("block definitions must be nonempty and unique")
    region_by_id = {region.id: region for region in manifest.regions}
    for block in blocks:
        region = region_by_id[block.region_id]
        if block.start < region.ws283_query_start or block.end > region.ws283_query_end:
            raise ValueError(f"block {block.block_id} lies outside its frozen region")

    calls_by_block: dict[str, list[GenomicStateCall]] = {block.block_id: [] for block in blocks}
    for call in calls:
        if call.block_id not in block_by_id:
            raise ValueError(f"state call references unknown block {call.block_id}")
        block = block_by_id[call.block_id]
        if call.region_id != block.region_id or call.kind != block.state_kind:
            raise ValueError("state call region or kind differs from its block definition")
        calls_by_block[call.block_id].append(call)

    sample_set = set(sample_order)
    qualifications: list[BlockQualification] = []
    gate = manifest.carrier_gate
    for block in blocks:
        block_calls = calls_by_block[block.block_id]
        call_samples = [call.sample_id for call in block_calls]
        if len(call_samples) != len(set(call_samples)):
            raise ValueError(f"block {block.block_id} has duplicate sample calls")
        if set(call_samples) != sample_set or len(call_samples) != len(sample_order):
            raise ValueError(f"block {block.block_id} does not have one call per sample")
        semantic_by_state: dict[str, str] = {}
        counts: dict[str, int] = {}
        unresolved = 0
        for call in block_calls:
            if _is_unresolved(call):
                unresolved += 1
                continue
            semantic = f"{call.kind}:{call.state}"
            previous = semantic_by_state.setdefault(call.state_id, semantic)
            if previous != semantic:
                raise ValueError("one state ID maps to conflicting typed semantics")
            counts[call.state_id] = counts.get(call.state_id, 0) + 1
        callable_count = len(sample_order) - unresolved
        state_counts = [
            StateCount(
                state_id=state_id,
                semantic_state=semantic_by_state[state_id],
                carriers=count,
                noncarriers=callable_count - count,
            )
            for state_id, count in sorted(counts.items())
        ]
        reasons: list[str] = []
        unresolved_fraction = unresolved / len(sample_order)
        if unresolved_fraction > gate.maximum_unresolved_fraction:
            reasons.append("unresolved_fraction_above_gate")
        if block.reference_state_id not in counts:
            reasons.append("reference_state_absent")
        if len(counts) < 2:
            reasons.append("reference_or_alternative_missing")
        for count in state_counts:
            if count.carriers < gate.minimum_carriers_per_state:
                reasons.append(f"state_below_minimum_carriers:{count.state_id}")
            if count.noncarriers < gate.minimum_noncarriers_per_state:
                reasons.append(f"state_below_minimum_noncarriers:{count.state_id}")
        qualifications.append(
            BlockQualification(
                block_id=block.block_id,
                region_id=block.region_id,
                state_kind=block.state_kind,
                reference_state_id=block.reference_state_id,
                sample_count=len(sample_order),
                callable_count=callable_count,
                unresolved_count=unresolved,
                unresolved_fraction=unresolved_fraction,
                state_counts=state_counts,
                eligible=not reasons,
                exclusion_reasons=reasons,
            )
        )
    eligible_count = sum(block.eligible for block in qualifications)
    return StatePanelQualification(
        schema_version=STATE_PANEL_VERSION,
        coordinate_receipt_sha256=coordinate.receipt_sha256,
        sample_count=209,
        block_count=len(qualifications),
        eligible_block_count=eligible_count,
        blocks=qualifications,
        outcome_blind=True,
        qualification_scope="synthetic_unbound_inputs",
        handoff_manifest_sha256=None,
        state_panel_asset_sha256=None,
        sample_order_asset_sha256=None,
        definition_receipt_sha256=None,
    )


class KinshipAwareAnalysisRequest(_StrictModel):
    cell_id: str
    trait: str
    relationship_kind: Literal["full", "ldpruned"]
    endpoint: Literal["pc0", "pc10"]
    pc_count: Literal[0, 10]
    block_id: str
    region_id: str
    state_kind: Literal["haplotype", "pav"]
    reference_state_id: str
    state_ids: list[str]
    callable_samples: int
    chromosome_excluded_from_kinship: Literal[5]
    kinship_resource: str
    primary_test: Literal["kinship_aware_block_omnibus"]
    family_block_count: int = Field(gt=0)
    block_bonferroni_alpha: float = Field(gt=0.0, le=0.05)
    null_threshold_eligible: bool
    descriptive_only: Literal[True]
    causal_interpretation_permitted: Literal[False]
    execution_scope: Literal["synthetic_fixture", "frozen_real_handoff"]
    real_execution_permitted: bool
    handoff_manifest_sha256: str | None

    @model_validator(mode="after")
    def executable_only_when_bound(self) -> "KinshipAwareAnalysisRequest":
        bound = (
            self.execution_scope == "frozen_real_handoff"
            and self.handoff_manifest_sha256 == HANDOFF_SHA256
        )
        if self.real_execution_permitted != bound:
            raise ValueError("request execution permission differs from its frozen binding")
        return self


class HaplotypePavAnalysisPlan(_StrictModel):
    schema_version: Literal[PLAN_VERSION]
    null_run_id: Literal[NULL_RUN_ID]
    null_aggregation_sha256: str
    coordinate_receipt_sha256: str
    eligible_block_count: int = Field(gt=0)
    model_cell_count: Literal[16]
    request_count: int = Field(gt=0)
    block_bonferroni_alpha: float = Field(gt=0.0, le=0.05)
    requests: list[KinshipAwareAnalysisRequest]
    descriptive_only: Literal[True]
    biological_claims_permitted: Literal[False]
    execution_scope: Literal["synthetic_fixture", "frozen_real_handoff"]
    real_execution_permitted: bool
    handoff_manifest_sha256: str | None
    verified_asset_sha256_by_role: dict[str, str]

    @model_validator(mode="after")
    def executable_only_when_bound(self) -> "HaplotypePavAnalysisPlan":
        bound = (
            self.execution_scope == "frozen_real_handoff"
            and self.handoff_manifest_sha256 == HANDOFF_SHA256
            and bool(self.verified_asset_sha256_by_role)
        )
        if self.real_execution_permitted != bound:
            raise ValueError("plan execution permission differs from its frozen binding")
        if any(item.real_execution_permitted != self.real_execution_permitted for item in self.requests):
            raise ValueError("plan and request execution permissions differ")
        return self


class KinshipAwareBlockResult(_StrictModel):
    cell_id: str
    block_id: str
    backend_id: str
    converged: bool
    callable_samples: int = Field(gt=0)
    block_p_value: float = Field(gt=0.0, le=1.0)
    variance_component_boundary: bool
    state_contrast_p_values: dict[str, float]

    @field_validator("state_contrast_p_values")
    @classmethod
    def valid_contrast_p_values(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(item) or item <= 0.0 or item > 1.0 for item in value.values()):
            raise ValueError("state contrast p-values must be finite in (0, 1]")
        return value


@dataclass(frozen=True)
class KinshipAwareInputs:
    """Numerical payload supplied by a future backend implementation."""

    sample_ids: tuple[str, ...]
    phenotype: tuple[float, ...]
    state_ids: tuple[str, ...]
    kinship: tuple[tuple[float, ...], ...]
    covariates: tuple[tuple[float, ...], ...]


@runtime_checkable
class KinshipAwareBackend(Protocol):
    """Interface only; the first-wave scaffold does not select an implementation."""

    backend_id: str

    def fit_block(
        self,
        request: KinshipAwareAnalysisRequest,
        inputs: KinshipAwareInputs,
    ) -> KinshipAwareBlockResult: ...


def validate_backend_inputs(
    request: KinshipAwareAnalysisRequest,
    inputs: KinshipAwareInputs,
    *,
    allow_synthetic_fixture: bool = False,
) -> None:
    if not request.real_execution_permitted and not allow_synthetic_fixture:
        raise ValueError(
            "synthetic/unbound requests cannot be submitted to a real backend"
        )
    size = len(inputs.sample_ids)
    if size != request.callable_samples:
        raise ValueError("backend sample count differs from the qualified request")
    if len(set(inputs.sample_ids)) != size:
        raise ValueError("backend sample IDs are not unique")
    if len(inputs.phenotype) != size or len(inputs.state_ids) != size:
        raise ValueError("phenotype or state vector length differs from sample count")
    if len(inputs.kinship) != size or any(len(row) != size for row in inputs.kinship):
        raise ValueError("kinship matrix is not square in qualified sample order")
    if len(inputs.covariates) != size:
        raise ValueError("covariate row count differs from sample count")
    if any(not math.isfinite(value) for value in inputs.phenotype):
        raise ValueError("phenotype contains a nonfinite value")
    for row_index, row in enumerate(inputs.kinship):
        for column_index, value in enumerate(row):
            if not math.isfinite(value):
                raise ValueError("kinship contains a nonfinite value")
            if not math.isclose(
                value,
                inputs.kinship[column_index][row_index],
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError("kinship matrix is not symmetric")


class RealExecutionHandoffAudit(_StrictModel):
    schema_version: Literal[HANDOFF_AUDIT_VERSION]
    handoff_manifest_sha256: Literal[HANDOFF_SHA256]
    handoff_schema_version: Literal[HANDOFF_SCHEMA_VERSION]
    handoff_analysis_id: Literal[HANDOFF_ANALYSIS_ID]
    gate_statuses: dict[str, str]
    blockers: list[str]
    advisory_limitations: list[str]
    verified_asset_sha256_by_role: dict[str, str]
    null_interpretation_receipt_sha256: str | None
    coordinate_receipt_sha256: str | None
    state_panel_qualification_sha256: str | None
    semantic_receipts_verified: bool
    real_execution_ready: bool
    operator_outcome_blinding_asserted: Literal[False]
    biological_claims_permitted: Literal[False]

    @model_validator(mode="after")
    def readiness_is_fail_closed(self) -> "RealExecutionHandoffAudit":
        ready = (
            not self.blockers
            and bool(self.verified_asset_sha256_by_role)
            and self.null_interpretation_receipt_sha256
            == NULL_INTERPRETATION_SHA256
            and self.coordinate_receipt_sha256 is not None
            and self.state_panel_qualification_sha256 is not None
            and self.semantic_receipts_verified
        )
        if self.real_execution_ready != ready:
            raise ValueError("handoff readiness differs from its verified evidence")
        return self


def _asset_path_by_role(
    verified_assets: Sequence[Any], role: str
) -> Path:
    matches = [Path(item.resolved_path) for item in verified_assets if item.role == role]
    if len(matches) != 1:
        raise ValueError(f"frozen handoff must contain exactly one asset role: {role}")
    return matches[0]


def _semantic_verify_real_handoff(
    handoff: Any,
    verified_assets: Sequence[Any],
) -> tuple[str, str, dict[str, str]]:
    """Open the receipts needed by regional-state qualification; frozen hashes alone are not semantics."""

    from . import qtl_regional_state_prediction_gates as handoff_gates

    coordinate_path = _asset_path_by_role(
        verified_assets, COORDINATE_HANDOFF_RECEIPT_ROLE
    )
    coordinate = CoordinateIdentityReceipt.model_validate_json(
        coordinate_path.read_text(encoding="utf-8")
    )
    gate = handoff.coordinate_identity
    expected_reference_hashes = {
        WS276_ASSEMBLY: gate.source.fasta.sha256,
        WS283_ASSEMBLY: gate.target.fasta.sha256,
    }
    if (
        coordinate.source_reference_fasta_sha256
        != expected_reference_hashes[WS276_ASSEMBLY]
        or coordinate.target_reference_fasta_sha256
        != expected_reference_hashes[WS283_ASSEMBLY]
        or coordinate.source_chromosome_v_sha256
        != gate.source.chromosome_v_canonical_sha256
        or coordinate.target_chromosome_v_sha256
        != gate.target.chromosome_v_canonical_sha256
    ):
        raise ValueError("coordinate receipt differs from frozen handoff references")
    source_regions = {item.id: item for item in gate.source.intervals}
    target_regions = {item.id: item for item in gate.target.intervals}
    for record in coordinate.regions:
        source = source_regions[record.region_id]
        target = target_regions[record.region_id]
        if (
            record.source_interval_sha256 != source.canonical_sequence_sha256
            or record.target_interval_sha256 != target.canonical_sequence_sha256
        ):
            raise ValueError(
                f"coordinate receipt interval differs from handoff: {record.region_id}"
            )

    state_path = _asset_path_by_role(
        verified_assets, "real_haplotype_pav_qualification_receipt"
    )
    state = StatePanelQualification.model_validate_json(
        state_path.read_text(encoding="utf-8")
    )
    state_evidence = handoff.state_panel.qualified_panel_evidence
    if state_evidence is None:
        raise ValueError("qualified handoff lacks embedded state-panel evidence")
    if (
        state.coordinate_receipt_sha256 != _sha256(coordinate_path)
        or state.sample_count != state_evidence.samples
        or state.block_count != state_evidence.state_count
        or state.eligible_block_count < 1
    ):
        raise ValueError("state-panel receipt differs from its frozen handoff evidence")

    group_assets = {item.role: item for item in handoff.population_groups.output_assets}
    assignments = _asset_path_by_role(
        verified_assets, "phenotype_blind_population_groups"
    )
    group_receipt = _asset_path_by_role(
        verified_assets, "phenotype_blind_population_group_receipt"
    )
    handoff_gates.verify_population_group_bundle(
        handoff, assignments, group_receipt
    )
    if set(group_assets) != {
        "phenotype_blind_population_groups",
        "phenotype_blind_population_group_receipt",
    }:
        raise ValueError("population-group handoff inventory differs")

    kernel_receipt_path = _asset_path_by_role(
        verified_assets, "full_kernel_numerical_qualification_receipt"
    )
    kernel_receipt = json.loads(kernel_receipt_path.read_text(encoding="utf-8"))
    kernel_assets = {
        item.role: item for item in handoff.full_kernels.qualified_output_assets
    }
    expected_kernel_assets = {
        "whole_genome": kernel_assets["whole_genome_float64_npy"],
        "genome_excluding_chrv": kernel_assets["genome_excluding_chrv_float64_npy"],
    }
    if (
        kernel_receipt.get("schema_version")
        != handoff_gates.KERNEL_QUALIFICATION_VERSION
        or kernel_receipt.get("samples") != 209
        or kernel_receipt.get("ordered_iid_sha256")
        != handoff.ldpruned_chr5_excluded_grm.verification.ordered_iid_sha256
        or kernel_receipt.get("biological_claims_permitted") is not False
    ):
        raise ValueError("full-kernel qualification receipt semantics differ")
    kernels = kernel_receipt.get("kernels")
    if not isinstance(kernels, list) or [item.get("id") for item in kernels] != [
        "whole_genome",
        "genome_excluding_chrv",
    ]:
        raise ValueError("full-kernel receipt does not contain the frozen kernel pair")
    marker_counts = {item.id: item.marker_count for item in handoff.full_kernels.outputs}
    for item in kernels:
        asset = expected_kernel_assets[item["id"]]
        npy = item.get("npy")
        if (
            item.get("relationship_markers") != marker_counts[item["id"]]
            or not isinstance(npy, dict)
            or npy.get("sha256") != asset.sha256
            or npy.get("bytes") != asset.bytes
            or npy.get("shape") != [209, 209]
            or npy.get("dtype") != "<f8"
            or npy.get("finite") is not True
            or npy.get("exactly_symmetric") is not True
        ):
            raise ValueError(f"full-kernel receipt differs: {item['id']}")

    verified = {item.role: item.sha256 for item in verified_assets}
    return _sha256(coordinate_path), _sha256(state_path), dict(sorted(verified.items()))


def audit_real_execution_handoff(
    manifest: HaplotypePavManifest,
    handoff_manifest_path: str | Path,
    roots: Mapping[str, str | Path] | None = None,
) -> RealExecutionHandoffAudit:
    """Audit real regional-state qualification readiness from a pinned handoff; never accept self-attestation."""

    from . import qtl_regional_state_prediction_gates as handoff_gates

    source = Path(handoff_manifest_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"frozen regional-state qualification and held-out prediction handoff is missing: {source}")
    observed_handoff_sha256 = _sha256(source)
    if observed_handoff_sha256 != manifest.real_execution_handoff.manifest_sha256:
        raise ValueError("regional-state qualification and held-out prediction handoff manifest checksum mismatch")
    handoff = handoff_gates.load_handoff_manifest(source)
    if (
        handoff.schema_version != manifest.real_execution_handoff.schema_version
        or handoff.analysis_id != manifest.real_execution_handoff.analysis_id
    ):
        raise ValueError("regional-state qualification and held-out prediction handoff identity differs from the regional-state qualification contract")
    if handoff.null_resource_guard.run_id != manifest.null_dependency.run_id:
        raise ValueError("handoff and regional-state qualification manifest bind different parametric-null calibration runs")

    statuses = handoff_gates.gate_summary(handoff)
    blockers = [
        f"handoff_gate_not_qualified:{name}"
        for name in manifest.real_execution_handoff.required_qualified_gates
        if statuses.get(name) != "qualified"
    ]
    assets = handoff_gates.all_frozen_assets(handoff)
    asset_roles = {item.role for item in assets}
    if COORDINATE_HANDOFF_RECEIPT_ROLE not in asset_roles:
        blockers.append("coordinate_identity_receipt_not_frozen_in_handoff")
    if handoff.state_panel.status == "qualified":
        missing_state = [
            role
            for role in manifest.real_execution_handoff.required_state_asset_roles
            if role not in asset_roles
        ]
        blockers.extend(f"state_asset_not_frozen:{role}" for role in missing_state)
    interpretation_sha256: str | None = None
    if manifest.null_dependency.interpretation_receipt_status != "frozen":
        blockers.append("null_interpretation_receipt_not_frozen")
    else:
        relative_handoff = Path(
            manifest.real_execution_handoff.manifest_relative_path
        )
        repository_root = source
        for _ in relative_handoff.parts:
            repository_root = repository_root.parent
        interpretation_path = repository_root / Path(
            manifest.null_dependency.interpretation_receipt_relative_path
        )
        if (
            not interpretation_path.is_file()
            or _sha256(interpretation_path)
            != manifest.null_dependency.interpretation_receipt_sha256
        ):
            blockers.append("frozen_null_interpretation_receipt_missing_or_changed")
        else:
            interpretation = NullInterpretationReceipt.model_validate_json(
                interpretation_path.read_text(encoding="utf-8")
            )
            if (
                interpretation.aggregation_sha256
                != handoff.null_resource_guard.aggregation_sha256
                or interpretation.decision
                != "proceed_with_descriptive_haplotype_scaffold"
            ):
                blockers.append("frozen_null_interpretation_semantics_differ")
            else:
                interpretation_sha256 = _sha256(interpretation_path)
    if manifest.status == "blocked_on_real_pangenome_inputs":
        blockers.append("regional_state_manifest_blocked_on_real_pangenome_inputs")
    elif manifest.status != "qualified_for_file_backed_descriptive_execution":
        blockers.append("regional_state_manifest_execution_release_not_frozen")

    advisory = []
    if handoff.phenotype_provenance.status != "qualified":
        advisory.append(
            "raw_measurement_provenance_unresolved_derived_traits_descriptive_only"
        )

    verified_hashes: dict[str, str] = {}
    coordinate_sha256: str | None = None
    state_sha256: str | None = None
    semantics_verified = False
    if not blockers:
        if roots is None:
            blockers.append("governed_runtime_root_mapping_missing")
        else:
            verified_assets = handoff_gates.verify_frozen_assets(handoff, roots)
            coordinate_sha256, state_sha256, verified_hashes = (
                _semantic_verify_real_handoff(handoff, verified_assets)
            )
            semantics_verified = True

    ready = (
        not blockers
        and bool(verified_hashes)
        and interpretation_sha256 == NULL_INTERPRETATION_SHA256
        and coordinate_sha256 is not None
        and state_sha256 is not None
        and semantics_verified
    )
    return RealExecutionHandoffAudit(
        schema_version=HANDOFF_AUDIT_VERSION,
        handoff_manifest_sha256=observed_handoff_sha256,
        handoff_schema_version=handoff.schema_version,
        handoff_analysis_id=handoff.analysis_id,
        gate_statuses=statuses,
        blockers=blockers,
        advisory_limitations=advisory,
        verified_asset_sha256_by_role=verified_hashes,
        null_interpretation_receipt_sha256=interpretation_sha256,
        coordinate_receipt_sha256=coordinate_sha256,
        state_panel_qualification_sha256=state_sha256,
        semantic_receipts_verified=semantics_verified,
        real_execution_ready=ready,
        operator_outcome_blinding_asserted=False,
        biological_claims_permitted=False,
    )


def _assemble_analysis_plan(
    manifest: HaplotypePavManifest,
    coordinate: CoordinateQualification,
    null_dependency: NullDependencyQualification,
    state_panel: StatePanelQualification,
    *,
    execution_scope: Literal["synthetic_fixture", "frozen_real_handoff"],
    handoff_manifest_sha256: str | None,
    verified_asset_sha256_by_role: Mapping[str, str],
) -> HaplotypePavAnalysisPlan:
    if not null_dependency.analysis_plan_permitted:
        raise ValueError("active parametric-null dependency has not enabled analysis planning")
    if null_dependency.aggregation_sha256 is None:
        raise ValueError("null aggregation identity is missing")
    if coordinate.receipt_sha256 != state_panel.coordinate_receipt_sha256:
        raise ValueError("state panel and plan use different coordinate receipts")
    real = execution_scope == "frozen_real_handoff"
    if real:
        if (
            coordinate.qualification_scope != "frozen_real_handoff"
            or state_panel.qualification_scope != "frozen_real_handoff"
            or coordinate.handoff_manifest_sha256 != HANDOFF_SHA256
            or state_panel.handoff_manifest_sha256 != HANDOFF_SHA256
            or handoff_manifest_sha256 != HANDOFF_SHA256
            or not verified_asset_sha256_by_role
        ):
            raise ValueError("real analysis inputs are not bound to the frozen handoff")
        if (
            null_dependency.interpretation_receipt_sha256
            != manifest.null_dependency.interpretation_receipt_sha256
        ):
            raise ValueError("real plan uses an unfrozen null interpretation receipt")
    elif (
        coordinate.qualification_scope != "synthetic_unbound_receipt"
        or state_panel.qualification_scope != "synthetic_unbound_inputs"
        or handoff_manifest_sha256 is not None
        or verified_asset_sha256_by_role
    ):
        raise ValueError("synthetic plan cannot claim frozen real-input evidence")
    eligible_blocks = [block for block in state_panel.blocks if block.eligible]
    if not eligible_blocks:
        raise ValueError("no outcome-blind block passed the carrier-count gates")
    family_count = len(eligible_blocks)
    alpha = manifest.multiplicity.family_alpha / family_count
    threshold_eligible = set(null_dependency.threshold_eligible_cells)
    requests: list[KinshipAwareAnalysisRequest] = []
    for relationship in manifest.kinship.relationships:
        resource = (
            manifest.kinship.full_kinship_resource
            if relationship == "full"
            else manifest.kinship.ldpruned_kinship_resource
        )
        for endpoint in manifest.kinship.endpoints:
            pc_count: Literal[0, 10] = 0 if endpoint == "pc0" else 10
            for trait in manifest.kinship.traits:
                cell = f"{relationship}_{endpoint}_{TRAIT_SLUGS[trait]}"
                for block in eligible_blocks:
                    requests.append(
                        KinshipAwareAnalysisRequest(
                            cell_id=cell,
                            trait=trait,
                            relationship_kind=relationship,
                            endpoint=endpoint,
                            pc_count=pc_count,
                            block_id=block.block_id,
                            region_id=block.region_id,
                            state_kind=block.state_kind,
                            reference_state_id=block.reference_state_id,
                            state_ids=[item.state_id for item in block.state_counts],
                            callable_samples=block.callable_count,
                            chromosome_excluded_from_kinship=5,
                            kinship_resource=resource,
                            primary_test=manifest.kinship.primary_test,
                            family_block_count=family_count,
                            block_bonferroni_alpha=alpha,
                            null_threshold_eligible=cell in threshold_eligible,
                            descriptive_only=True,
                            causal_interpretation_permitted=False,
                            execution_scope=execution_scope,
                            real_execution_permitted=real,
                            handoff_manifest_sha256=handoff_manifest_sha256,
                        )
                    )
    expected_requests = 16 * family_count
    if len(requests) != expected_requests:
        raise ValueError("analysis request grid is incomplete")
    return HaplotypePavAnalysisPlan(
        schema_version=PLAN_VERSION,
        null_run_id=null_dependency.run_id,
        null_aggregation_sha256=null_dependency.aggregation_sha256,
        coordinate_receipt_sha256=coordinate.receipt_sha256,
        eligible_block_count=family_count,
        model_cell_count=16,
        request_count=len(requests),
        block_bonferroni_alpha=alpha,
        requests=requests,
        descriptive_only=True,
        biological_claims_permitted=False,
        execution_scope=execution_scope,
        real_execution_permitted=real,
        handoff_manifest_sha256=handoff_manifest_sha256,
        verified_asset_sha256_by_role=dict(
            sorted(verified_asset_sha256_by_role.items())
        ),
    )


def build_synthetic_analysis_plan(
    manifest: HaplotypePavManifest,
    coordinate: CoordinateQualification,
    null_dependency: NullDependencyQualification,
    state_panel: StatePanelQualification,
) -> HaplotypePavAnalysisPlan:
    """Build an explicitly non-executable plan for interface and synthetic tests."""

    return _assemble_analysis_plan(
        manifest,
        coordinate,
        null_dependency,
        state_panel,
        execution_scope="synthetic_fixture",
        handoff_manifest_sha256=None,
        verified_asset_sha256_by_role={},
    )


def build_analysis_plan(
    manifest: HaplotypePavManifest,
    *,
    handoff_manifest_path: str | Path,
    handoff_roots: Mapping[str, str | Path],
    null_run_root: str | Path,
    interpretation_path: str | Path,
) -> HaplotypePavAnalysisPlan:
    """Build the real plan only from file-backed, byte-pinned upstream evidence."""

    from . import qtl_regional_state_prediction_gates as handoff_gates

    audit = audit_real_execution_handoff(
        manifest, handoff_manifest_path, handoff_roots
    )
    if not audit.real_execution_ready:
        raise ValueError(
            "real regional-state qualification execution remains blocked: " + ",".join(audit.blockers)
        )
    dependency = qualify_null_dependency(
        manifest, null_run_root, interpretation_path
    )
    if not dependency.analysis_plan_permitted:
        raise ValueError("frozen null interpretation did not release descriptive planning")

    handoff = handoff_gates.load_handoff_manifest(handoff_manifest_path)
    verified_assets = handoff_gates.verify_frozen_assets(handoff, handoff_roots)
    coordinate_path = _asset_path_by_role(
        verified_assets, COORDINATE_HANDOFF_RECEIPT_ROLE
    )
    coordinate_receipt = CoordinateIdentityReceipt.model_validate_json(
        coordinate_path.read_text(encoding="utf-8")
    )
    coordinate = CoordinateQualification(
        qualified=True,
        receipt_sha256=_sha256(coordinate_path),
        source_assembly=coordinate_receipt.source_assembly,
        target_assembly=coordinate_receipt.target_assembly,
        region_ids=[item.region_id for item in coordinate_receipt.regions],
        qualification_scope="frozen_real_handoff",
        handoff_manifest_sha256=audit.handoff_manifest_sha256,
    )
    state_path = _asset_path_by_role(
        verified_assets, "real_haplotype_pav_qualification_receipt"
    )
    state_panel = StatePanelQualification.model_validate_json(
        state_path.read_text(encoding="utf-8")
    )
    if (
        state_panel.qualification_scope != "frozen_real_handoff"
        or state_panel.handoff_manifest_sha256 != audit.handoff_manifest_sha256
        or state_panel.coordinate_receipt_sha256 != audit.coordinate_receipt_sha256
        or state_panel.state_panel_asset_sha256
        != audit.verified_asset_sha256_by_role["real_haplotype_pav_state_panel"]
        or state_panel.sample_order_asset_sha256
        != audit.verified_asset_sha256_by_role["real_haplotype_pav_sample_order"]
        or state_panel.definition_receipt_sha256
        != audit.verified_asset_sha256_by_role[
            "real_haplotype_pav_call_definition_receipt"
        ]
    ):
        raise ValueError("state-panel qualification is not bound to its frozen files")
    return _assemble_analysis_plan(
        manifest,
        coordinate,
        dependency,
        state_panel,
        execution_scope="frozen_real_handoff",
        handoff_manifest_sha256=audit.handoff_manifest_sha256,
        verified_asset_sha256_by_role=audit.verified_asset_sha256_by_role,
    )
