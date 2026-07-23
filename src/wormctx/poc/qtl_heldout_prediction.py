"""Leakage-controlled held-out-strain prediction scaffold for WS283 abamectin data.

The scaffold evaluates five prespecified comparators under operational genotype-group-
blocked nested cross-validation. It supports synthetic qualification now and fails closed
for real data until all source identities and file-backed upstream gates are available. It
never accepts the quarantined 203-isolate/165-aligned legacy audit as an input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .qtl import TRAITS, TRAIT_SLUGS  # noqa: F401 - retained as part of the module API


SCHEMA_VERSION = "wormctx-abamectin-ws283-heldout-strain-prediction-1.1"
INPUT_RECEIPT_VERSION = "wormctx-abamectin-ws283-prediction-inputs-1.1"
QUALIFICATION_VERSION = "wormctx-abamectin-ws283-prediction-qualification-1.1"
SPLIT_VERSION = "wormctx-abamectin-ws283-prediction-splits-1.1"
SUMMARY_VERSION = "wormctx-abamectin-ws283-prediction-summary-1.0"
BUNDLE_VERSION = "wormctx-abamectin-ws283-prediction-bundle-1.0"

PARAMETRIC_NULL_AGGREGATION_VERSION = "wormctx-abamectin-ws283-parametric-null-aggregation-1.0"
PARAMETRIC_NULL_SUMMARY_VERSION = "wormctx-abamectin-ws283-parametric-polygenic-null-summary-1.0"
COORDINATE_RECEIPT_VERSION = "wormctx-ws276-ws283-coordinate-identity-1.0"
STATE_PANEL_RECEIPT_VERSION = "wormctx-chrv-haplotype-pav-panel-1.0"
STATE_FEATURE_RECEIPT_VERSION = "wormctx-abamectin-ws283-heldout_prediction-state-features-1.0"
FULL_KERNEL_RECEIPT_VERSION = "wormctx-abamectin-full-kernel-qualification-1.0"
GROUP_RECEIPT_VERSION = "wormctx-abamectin-phenotype-blind-groups-1.1"
FEATURE_MASK_SCOPE = "full_209_genotype_panel_transductive_no_outcome_values_accessed"
WS276_ASSEMBLY = "PRJNA13758.WS276"
WS283_ASSEMBLY = "PRJNA13758.WS283_inferred_not_declared_in_vcf_header"

SAMPLE_COUNT = 209
MODEL_IDS = (
    "training_mean",
    "whole_genome_gblup",
    "chrv_haplotype",
    "genome_excluding_chrv",
    "combined_excluding_chrv_plus_haplotype",
)
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
COMBINED_WEIGHT_GRID = (0.25, 0.5, 0.75)
INTERVAL_LEVELS = (0.5, 0.9)
LEGACY_ANALYSIS_ID = "caendr_abamectin_legacy_prediction_audit_20260717"
LEGACY_EXPECTED_IDS = 203
LEGACY_ALIGNED_FINITE = 165
NULL_RUN_ID = "abamectin-ws283-parametric-null-20260721T161005Z-c01179f4a52e"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    header = f"{array.ndim}:{','.join(str(item) for item in array.shape)}:<f8\n".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FrozenFile(_StrictModel):
    role: str
    root_id: str
    relative_path: str
    bytes: int = Field(gt=0)
    sha256: str

    @field_validator("role", "root_id")
    @classmethod
    def safe_token(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("frozen-file role and root ID must be safe tokens")
        return value

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("frozen-file path must be governed root-relative")
        return path.as_posix()

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("frozen-file SHA-256 must be lowercase hexadecimal")
        return value


class ModernCohortContract(_StrictModel):
    parent_run_id: Literal["abamectin-ws283-bd41637-20260717T200012Z"]
    parent_source_commit: Literal["bd41637f5ff2998333964cc954ee9f9f36a351d6"]
    sample_count: Literal[209]
    marker_count: Literal[373279]
    chromosome_v_marker_count: Literal[115183]
    non_chromosome_v_marker_count: Literal[258096]
    sample_order: Literal["baseline_parent_fam_order"]
    phenotype_condition: Literal["abamectin"]
    phenotype_traits: list[str]
    required_files: list[FrozenFile]

    @model_validator(mode="after")
    def exact_cohort(self) -> "ModernCohortContract":
        if self.phenotype_traits != list(TRAITS):
            raise ValueError("phenotype traits differ from the frozen four-trait panel")
        roles = [item.role for item in self.required_files]
        expected = ["bed", "bim", "fam", *[f"phenotype:{trait}" for trait in TRAITS]]
        if roles != expected:
            raise ValueError("modern cohort frozen files differ from the required order")
        return self


class LegacyQuarantineContract(_StrictModel):
    analysis_id: Literal[LEGACY_ANALYSIS_ID]
    relationship_prefix: Literal["pheno_only"]
    original_ids: Literal[203]
    abamectin_roster_overlap: Literal[188]
    finite_aligned_ids: Literal[165]
    accepted_as_heldout_prediction_input: Literal[False]
    refusal_required: Literal[True]


class ParametricNullTerminalContract(_StrictModel):
    status: Literal["qualified"]
    run_id: Literal[NULL_RUN_ID]
    success_marker: FrozenFile
    aggregation_summary: FrozenFile
    rich_summary: FrozenFile
    expected_calibration_cells: Literal[16]
    expected_replicates_per_cell: Literal[100]
    expected_completed_maps: Literal[1600]


class OptionalReceiptContract(_StrictModel):
    status: Literal["missing", "qualified"]
    receipt_schema_version: str
    receipt: FrozenFile | None

    @model_validator(mode="after")
    def asset_matches_status(self) -> "OptionalReceiptContract":
        if self.status == "qualified" and self.receipt is None:
            raise ValueError("qualified upstream receipt status requires a frozen receipt")
        if self.status == "missing" and self.receipt is not None:
            raise ValueError("missing upstream receipt status cannot contain a frozen receipt")
        return self


class StatePanelReceiptContract(_StrictModel):
    status: Literal["missing", "qualified"]
    qualification_schema_version: Literal[STATE_PANEL_RECEIPT_VERSION]
    feature_receipt_schema_version: Literal[STATE_FEATURE_RECEIPT_VERSION]
    qualification_receipt: FrozenFile | None
    feature_receipt: FrozenFile | None
    feature_matrix: FrozenFile | None

    @model_validator(mode="after")
    def assets_match_status(self) -> "StatePanelReceiptContract":
        assets = (
            self.qualification_receipt,
            self.feature_receipt,
            self.feature_matrix,
        )
        if self.status == "qualified" and any(item is None for item in assets):
            raise ValueError("qualified state-panel status requires both receipts and the matrix")
        if self.status == "missing" and any(item is not None for item in assets):
            raise ValueError("missing state-panel status cannot contain frozen assets")
        return self


class FullKernelReceiptContract(_StrictModel):
    status: Literal["missing", "qualified"]
    receipt_schema_version: Literal[FULL_KERNEL_RECEIPT_VERSION]
    qualification_receipt: FrozenFile | None
    whole_genome_matrix: FrozenFile | None
    genome_excluding_chrv_matrix: FrozenFile | None

    @model_validator(mode="after")
    def assets_match_status(self) -> "FullKernelReceiptContract":
        assets = (
            self.qualification_receipt,
            self.whole_genome_matrix,
            self.genome_excluding_chrv_matrix,
        )
        if self.status == "qualified" and any(item is None for item in assets):
            raise ValueError("qualified full-kernel status requires a receipt and both matrices")
        if self.status == "missing" and any(item is not None for item in assets):
            raise ValueError("missing full-kernel status cannot contain frozen assets")
        return self


class PopulationGroupReceiptContract(_StrictModel):
    status: Literal["qualified"]
    receipt_schema_version: str
    receipt: FrozenFile
    assignments: FrozenFile
    expected_samples: Literal[209]
    minimum_groups: Literal[5]
    eigenvalues_sha256: str
    eigenvalue_count: Literal[10]
    eigenvalues_used_for_clustering: Literal[False]
    axis_weighting: Literal["equal_weight_after_per_pc_standardization"]
    eigenvalue_usage: Literal["verified_provenance_only_not_used_for_clustering"]
    phenotype_paths_accepted: Literal[False]
    phenotype_values_accessed: Literal[False]
    outcome_access_claim_scope: Literal["derivation_execution_path_only"]
    operator_outcome_blinding_asserted: Literal[False]
    labels_are_external_ancestry_assignments: Literal[False]

    @field_validator("eigenvalues_sha256")
    @classmethod
    def valid_eigenvalue_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("population-group eigenvalue SHA-256 is malformed")
        return value


class UpstreamGateContract(_StrictModel):
    parametric_null_terminal: ParametricNullTerminalContract
    coordinate_identity: OptionalReceiptContract
    state_panel: StatePanelReceiptContract
    full_kernels: FullKernelReceiptContract
    population_groups: PopulationGroupReceiptContract
    every_real_asset_byte_and_sha256_verified: Literal[True]
    every_real_receipt_semantically_verified: Literal[True]
    synthetic_mode_may_bypass_real_upstream_inputs: Literal[True]
    synthetic_mode_biological_claims_permitted: Literal[False]

    @model_validator(mode="after")
    def exact_receipt_schemas(self) -> "UpstreamGateContract":
        if self.coordinate_identity.receipt_schema_version != COORDINATE_RECEIPT_VERSION:
            raise ValueError("coordinate receipt schema differs from the frozen held-out strain prediction contract")
        return self


class SplitContract(_StrictModel):
    method: Literal["leave_one_population_group_out_nested_cv"]
    expected_samples: Literal[209]
    minimum_outer_groups: Literal[5]
    minimum_group_size: Literal[5]
    outer_test_group_used_for_selection: Literal[False]
    inner_validation_unit: Literal["whole_population_group"]
    group_labels_derived_without_phenotypes: Literal[True]
    every_strain_tested_exactly_once: Literal[True]


class ComparatorContract(_StrictModel):
    id: Literal[
        "training_mean",
        "whole_genome_gblup",
        "chrv_haplotype",
        "genome_excluding_chrv",
        "combined_excluding_chrv_plus_haplotype",
    ]
    method: str
    chromosome_v_in_genome_kernel: bool | None
    hyperparameters: list[str]


class SelectionContract(_StrictModel):
    criterion: Literal["inner_group_macro_rmse"]
    ridge_grid: list[float]
    combined_haplotype_weight_grid: list[float]
    tie_break: Literal["lower_rmse_then_higher_ridge_then_lower_haplotype_weight"]
    preprocessing_uses_outer_training_only: Literal[True]
    outer_test_phenotypes_hidden_until_final_prediction: Literal[True]
    prediction_intervals_from_inner_validation_absolute_residuals: Literal[True]
    genotype_feature_mask_scope: Literal[FEATURE_MASK_SCOPE]
    transductive_genotype_panel_access_permitted: Literal[True]
    transductive_phenotype_access_permitted: Literal[False]

    @model_validator(mode="after")
    def exact_grids(self) -> "SelectionContract":
        if self.ridge_grid != list(RIDGE_GRID):
            raise ValueError("ridge grid differs from the frozen contract")
        if self.combined_haplotype_weight_grid != list(COMBINED_WEIGHT_GRID):
            raise ValueError("combined-weight grid differs from the frozen contract")
        return self


class EvaluationContract(_StrictModel):
    primary_metric: Literal["population_group_macro_rmse"]
    secondary_metrics: list[str]
    interval_levels: list[float]
    calibration_fit: Literal["heldout_observed_on_heldout_predicted_descriptive"]
    trait_pooling_permitted: Literal[False]
    group_pooling_for_primary_permitted: Literal[False]

    @model_validator(mode="after")
    def exact_evaluation(self) -> "EvaluationContract":
        if self.secondary_metrics != [
            "sample_rmse",
            "sample_mae",
            "pearson",
            "spearman",
            "calibration_intercept",
            "calibration_slope",
            "coverage_50",
            "coverage_90",
        ]:
            raise ValueError("secondary metrics differ from the frozen contract")
        if self.interval_levels != list(INTERVAL_LEVELS):
            raise ValueError("prediction interval levels differ from the frozen contract")
        return self


class ReceiptContract(_StrictModel):
    publication: Literal["atomic_write_once_directory"]
    existing_output_policy: Literal["refuse"]
    checksum_algorithm: Literal["sha256"]
    terminal_marker: Literal["SUCCESS"]
    required_payloads: list[str]

    @model_validator(mode="after")
    def exact_payloads(self) -> "ReceiptContract":
        if self.required_payloads != [
            "input_qualification.json",
            "outer_splits.json",
            "predictions.jsonl",
            "summary.json",
        ]:
            raise ValueError("write-once payload inventory differs from the frozen contract")
        return self


class ClaimContract(_StrictModel):
    permitted: list[str]
    prohibited: list[str]


class HeldoutPredictionManifest(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    analysis_id: Literal["abamectin_ws283_heldout_strain_prediction_heldout_prediction_v1"]
    classification: Literal["predeclared_heldout_strain_prediction_scaffold"]
    status: Literal[
        "blocked_on_coordinate_and_state_receipts",
        "blocked_on_state_receipts",
        "qualified_for_file_backed_real_execution",
    ]
    validated: Literal[False]
    biological_claims_permitted: Literal[False]
    modern_cohort: ModernCohortContract
    legacy_quarantine: LegacyQuarantineContract
    upstream_gates: UpstreamGateContract
    splits: SplitContract
    comparators: list[ComparatorContract]
    selection: SelectionContract
    evaluation: EvaluationContract
    receipts: ReceiptContract
    claims: ClaimContract

    @model_validator(mode="after")
    def exact_design(self) -> "HeldoutPredictionManifest":
        if [item.id for item in self.comparators] != list(MODEL_IDS):
            raise ValueError("comparators differ from the frozen five-model order")
        expected_roles = ["bed", "bim", "fam", *[f"phenotype:{trait}" for trait in TRAITS]]
        if [item.role for item in self.modern_cohort.required_files] != expected_roles:
            raise ValueError("modern cohort file roles differ from the frozen held-out strain prediction contract")
        upstream_roles = [
            self.upstream_gates.parametric_null_terminal.success_marker.role,
            self.upstream_gates.parametric_null_terminal.aggregation_summary.role,
            self.upstream_gates.parametric_null_terminal.rich_summary.role,
            self.upstream_gates.population_groups.receipt.role,
            self.upstream_gates.population_groups.assignments.role,
        ]
        if upstream_roles != [
            "parametric_null_terminal_success",
            "parametric_null_aggregation_summary",
            "parametric_null_rich_summary",
            "population_group_receipt",
            "population_group_assignments",
        ]:
            raise ValueError("qualified upstream asset roles differ from the frozen contract")
        optional_assets = [
            self.upstream_gates.coordinate_identity.receipt,
            self.upstream_gates.state_panel.qualification_receipt,
            self.upstream_gates.state_panel.feature_receipt,
            self.upstream_gates.state_panel.feature_matrix,
            self.upstream_gates.full_kernels.qualification_receipt,
            self.upstream_gates.full_kernels.whole_genome_matrix,
            self.upstream_gates.full_kernels.genome_excluding_chrv_matrix,
        ]
        all_assets = [
            *self.modern_cohort.required_files,
            self.upstream_gates.parametric_null_terminal.success_marker,
            self.upstream_gates.parametric_null_terminal.aggregation_summary,
            self.upstream_gates.parametric_null_terminal.rich_summary,
            self.upstream_gates.population_groups.receipt,
            self.upstream_gates.population_groups.assignments,
            *(item for item in optional_assets if item is not None),
        ]
        roles = [item.role for item in all_assets]
        if len(roles) != len(set(roles)):
            raise ValueError("real upstream asset roles must be globally unique")
        coordinate_status = self.upstream_gates.coordinate_identity.status
        state_status = self.upstream_gates.state_panel.status
        kernel_status = self.upstream_gates.full_kernels.status
        if (coordinate_status, state_status, kernel_status) == (
            "qualified",
            "qualified",
            "qualified",
        ):
            expected_status = "qualified_for_file_backed_real_execution"
        elif (coordinate_status, state_status, kernel_status) == (
            "qualified",
            "missing",
            "qualified",
        ):
            expected_status = "blocked_on_state_receipts"
        elif (coordinate_status, state_status, kernel_status) == (
            "missing",
            "missing",
            "qualified",
        ):
            expected_status = "blocked_on_coordinate_and_state_receipts"
        else:
            raise ValueError("upstream receipt gate combination has no frozen held-out strain prediction status")
        if self.status != expected_status:
            raise ValueError("manifest execution status differs from its upstream receipt gates")
        prohibited = " ".join(self.claims.prohibited).lower()
        for token in ("causal", "legacy", "independent replication", "inductive"):
            if token not in prohibited:
                raise ValueError(f"claim boundary must prohibit {token}")
        return self


def load_manifest(path: str | Path) -> HeldoutPredictionManifest:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"held-out prediction manifest is not a file: {source}")
    return HeldoutPredictionManifest.model_validate_json(source.read_text(encoding="utf-8"))


class PredictionInputReceipt(_StrictModel):
    schema_version: Literal[INPUT_RECEIPT_VERSION]
    mode: Literal["synthetic_fixture", "real_ws283"]
    sample_count: Literal[209]
    sample_order_sha256: str
    group_labels_sha256: str
    phenotype_sha256_by_trait: dict[str, str]
    whole_genome_kernel_sha256: str
    genome_excluding_chrv_kernel_sha256: str
    chrv_haplotype_features_sha256: str
    genotype_feature_mask_scope: Literal[FEATURE_MASK_SCOPE]
    legacy_analysis_id: str | None
    legacy_input_paths: list[str]
    reported_source_sample_count: int | None = Field(default=None, gt=0)
    reported_aligned_finite_count: int | None = Field(default=None, gt=0)

    @field_validator(
        "sample_order_sha256",
        "group_labels_sha256",
        "whole_genome_kernel_sha256",
        "genome_excluding_chrv_kernel_sha256",
        "chrv_haplotype_features_sha256",
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("input receipt SHA-256 must be lowercase hexadecimal")
        return value

    @field_validator("phenotype_sha256_by_trait")
    @classmethod
    def valid_trait_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if list(value) != list(TRAITS):
            raise ValueError("phenotype hashes differ from the ordered four-trait panel")
        if any(not _SHA256_RE.fullmatch(item) for item in value.values()):
            raise ValueError("phenotype SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def mode_gates(self) -> "PredictionInputReceipt":
        if self.mode == "real_ws283" and (
            self.reported_source_sample_count != SAMPLE_COUNT
            or self.reported_aligned_finite_count != SAMPLE_COUNT
        ):
            raise ValueError("real WS283 mode requires the exact 209-sample source identities")
        return self


@dataclass(frozen=True)
class PredictionDataset:
    sample_ids: tuple[str, ...]
    population_groups: tuple[str, ...]
    phenotypes: Mapping[str, np.ndarray]
    whole_genome_kernel: np.ndarray
    genome_excluding_chrv_kernel: np.ndarray
    chrv_haplotype_features: np.ndarray


class InputQualification(_StrictModel):
    schema_version: Literal[QUALIFICATION_VERSION]
    mode: Literal["synthetic_fixture", "real_ws283"]
    sample_count: Literal[209]
    population_group_count: int = Field(ge=5)
    population_group_sizes: dict[str, int]
    haplotype_feature_count: int = Field(gt=0)
    hashes: dict[str, Any]
    input_receipt_sha256: str
    upstream_asset_sha256_by_role: dict[str, str]
    genotype_feature_mask_scope: Literal[FEATURE_MASK_SCOPE]
    leakage_gates_passed: Literal[True]
    legacy_quarantine_passed: Literal[True]
    biological_claims_permitted: Literal[False]

    @field_validator("input_receipt_sha256")
    @classmethod
    def valid_receipt_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("input-receipt SHA-256 must be lowercase hexadecimal")
        return value

    @field_validator("upstream_asset_sha256_by_role")
    @classmethod
    def valid_upstream_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not _SHA256_RE.fullmatch(item) for item in value.values()):
            raise ValueError("upstream asset SHA-256 must be lowercase hexadecimal")
        return value


def refuse_legacy_audit(receipt: PredictionInputReceipt) -> None:
    path_tokens = [Path(item).name.casefold() for item in receipt.legacy_input_paths]
    if receipt.legacy_analysis_id == LEGACY_ANALYSIS_ID:
        raise ValueError("quarantined legacy prediction audit cannot enter held-out strain prediction")
    if any(token.startswith("pheno_only") for token in path_tokens):
        raise ValueError("quarantined pheno_only relationship input cannot enter held-out strain prediction")
    if receipt.reported_source_sample_count == LEGACY_EXPECTED_IDS:
        raise ValueError("quarantined 203-isolate cohort cannot enter held-out strain prediction")
    if receipt.reported_aligned_finite_count == LEGACY_ALIGNED_FINITE:
        raise ValueError("quarantined 165-aligned audit cannot enter held-out strain prediction")


def dataset_hashes(dataset: PredictionDataset) -> dict[str, Any]:
    return {
        "sample_order_sha256": _hash_strings(dataset.sample_ids),
        "group_labels_sha256": _hash_strings(
            [f"{sample}\t{group}" for sample, group in zip(
                dataset.sample_ids, dataset.population_groups, strict=True
            )]
        ),
        "phenotype_sha256_by_trait": {
            trait: _hash_array(np.asarray(dataset.phenotypes[trait], dtype=np.float64))
            for trait in TRAITS
        },
        "whole_genome_kernel_sha256": _hash_array(dataset.whole_genome_kernel),
        "genome_excluding_chrv_kernel_sha256": _hash_array(
            dataset.genome_excluding_chrv_kernel
        ),
        "chrv_haplotype_features_sha256": _hash_array(dataset.chrv_haplotype_features),
    }


def _model_sha256(value: BaseModel | Mapping[str, Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _matrix_payload_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _ordered_iid_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _verify_frozen_file(
    asset: FrozenFile,
    roots: Mapping[str, str | Path],
) -> Path:
    if asset.root_id not in roots:
        raise ValueError(f"missing governed root for upstream asset: {asset.root_id}")
    root = Path(roots[asset.root_id]).resolve()
    if not root.is_dir():
        raise ValueError(f"governed upstream root is not a directory: {asset.root_id}")
    path = (root / asset.relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"upstream asset escapes its governed root: {asset.role}") from error
    if not path.is_file():
        raise FileNotFoundError(f"upstream asset is missing: {asset.role}")
    if path.stat().st_size != asset.bytes:
        raise ValueError(f"upstream asset byte count differs: {asset.role}")
    if _sha256(path) != asset.sha256:
        raise ValueError(f"upstream asset SHA-256 differs: {asset.role}")
    return path


def _load_json_asset(
    asset: FrozenFile,
    roots: Mapping[str, str | Path],
) -> tuple[Path, dict[str, Any]]:
    path = _verify_frozen_file(asset, roots)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"upstream JSON receipt is invalid: {asset.role}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"upstream JSON receipt is not an object: {asset.role}")
    return path, payload


def _verify_modern_parent(
    manifest: HeldoutPredictionManifest,
    dataset: PredictionDataset,
    roots: Mapping[str, str | Path],
) -> dict[str, str]:
    paths = {
        asset.role: _verify_frozen_file(asset, roots)
        for asset in manifest.modern_cohort.required_files
    }
    fam_rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(paths["fam"].read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"modern parent FAM line {line_number} has fewer than two fields")
        fam_rows.append((fields[0], fields[1]))
    if len(fam_rows) != SAMPLE_COUNT or len(fam_rows) != len(set(fam_rows)):
        raise ValueError("modern parent FAM does not contain 209 unique FID/IID rows")
    if tuple(item[1] for item in fam_rows) != tuple(dataset.sample_ids):
        raise ValueError("prediction dataset sample order differs from the frozen parent FAM")

    for trait in TRAITS:
        observed_ids: list[tuple[str, str]] = []
        observed_values: list[float] = []
        role = f"phenotype:{trait}"
        for line_number, line in enumerate(paths[role].read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{trait} phenotype line {line_number} must have three fields")
            try:
                value = float(fields[2])
            except ValueError as error:
                raise ValueError(f"{trait} phenotype line {line_number} is not numeric") from error
            if not math.isfinite(value):
                raise ValueError(f"{trait} parent phenotype contains a nonfinite value")
            observed_ids.append((fields[0], fields[1]))
            observed_values.append(value)
        if observed_ids != fam_rows:
            raise ValueError(f"{trait} parent phenotype order differs from the frozen FAM")
        expected = np.asarray(dataset.phenotypes[trait], dtype=np.float64)
        if not np.array_equal(np.asarray(observed_values, dtype=np.float64), expected):
            raise ValueError(f"{trait} dataset values differ from the frozen parent phenotype")
    return {
        asset.role: asset.sha256 for asset in manifest.modern_cohort.required_files
    }


def _verify_parametric_null_terminal(
    gate: ParametricNullTerminalContract,
    roots: Mapping[str, str | Path],
) -> dict[str, str]:
    success = _verify_frozen_file(gate.success_marker, roots)
    if success.read_bytes() != b"SUCCESS\n":
        raise ValueError("parametric-null calibration terminal SUCCESS marker is not exact")
    _, aggregation = _load_json_asset(gate.aggregation_summary, roots)
    _, summary = _load_json_asset(gate.rich_summary, roots)
    expected_aggregation = {
        "schema_version": PARAMETRIC_NULL_AGGREGATION_VERSION,
        "calibration_cells": gate.expected_calibration_cells,
        "replicates_per_cell": gate.expected_replicates_per_cell,
        "attempted_null_maps": gate.expected_completed_maps,
        "completed_null_maps": gate.expected_completed_maps,
        "validated": False,
        "biological_claims_permitted": False,
    }
    for key, expected in expected_aggregation.items():
        if aggregation.get(key) != expected:
            raise ValueError(f"parametric-null calibration aggregation semantic field differs: {key}")
    cells = aggregation.get("cells")
    if not isinstance(cells, list) or len(cells) != gate.expected_calibration_cells:
        raise ValueError("parametric-null calibration aggregation does not contain the frozen 16-cell grid")
    if any(
        not isinstance(cell, dict)
        or cell.get("replicates_attempted") != gate.expected_replicates_per_cell
        or cell.get("replicates_completed") != gate.expected_replicates_per_cell
        for cell in cells
    ):
        raise ValueError("parametric-null calibration aggregation contains an incomplete calibration cell")
    rich_pointer = aggregation.get("rich_summary")
    if not isinstance(rich_pointer, dict) or rich_pointer.get("sha256") != gate.rich_summary.sha256:
        raise ValueError("parametric-null calibration aggregation is not bound to the frozen rich summary")
    expected_summary = {
        "schema_version": PARAMETRIC_NULL_SUMMARY_VERSION,
        "analysis_id": "abamectin_qtl_ws283_parametric_polygenic_null_v1",
        "status": "preliminary_model_specific_null_calibration_complete",
        "validated": False,
        "biological_claims_permitted": False,
        "manifest_sha256": aggregation.get("manifest_sha256"),
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"parametric-null calibration rich-summary semantic field differs: {key}")
    design = summary.get("design")
    if (
        not isinstance(design, dict)
        or design.get("cells") != gate.expected_calibration_cells
        or design.get("replicates_per_cell") != gate.expected_replicates_per_cell
        or design.get("total_null_maps") != gate.expected_completed_maps
        or len(summary.get("cells", [])) != gate.expected_calibration_cells
    ):
        raise ValueError("parametric-null calibration rich summary does not preserve the completed frozen design")
    return {
        gate.success_marker.role: gate.success_marker.sha256,
        gate.aggregation_summary.role: gate.aggregation_summary.sha256,
        gate.rich_summary.role: gate.rich_summary.sha256,
    }


def _verify_coordinate_receipt(
    gate: OptionalReceiptContract,
    roots: Mapping[str, str | Path],
) -> tuple[str, dict[str, str]]:
    if gate.status != "qualified" or gate.receipt is None:
        raise ValueError("real upstream gate remains blocked: coordinate_identity")
    path, receipt = _load_json_asset(gate.receipt, roots)
    expected = {
        "schema_version": COORDINATE_RECEIPT_VERSION,
        "source_assembly": WS276_ASSEMBLY,
        "target_assembly": WS283_ASSEMBLY,
        "generated_before_state_ingestion": True,
        "phenotype_accessed": False,
        "association_results_accessed": False,
        "full_reference_byte_identical": True,
        "source_chromosome_v_length": 20_924_180,
        "target_chromosome_v_length": 20_924_180,
        "operator_outcome_blinding_asserted": False,
        "outcome_access_claim_scope": "qualification_execution_path_only",
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"coordinate receipt semantic field differs: {key}")
    for key in (
        "source_reference_fasta_sha256",
        "target_reference_fasta_sha256",
        "source_reference_fai_sha256",
        "target_reference_fai_sha256",
        "source_chromosome_v_sha256",
        "target_chromosome_v_sha256",
    ):
        if not _SHA256_RE.fullmatch(str(receipt.get(key, ""))):
            raise ValueError(f"coordinate receipt lacks a valid {key}")
    for left, right in (
        ("source_reference_fasta_sha256", "target_reference_fasta_sha256"),
        ("source_reference_fai_sha256", "target_reference_fai_sha256"),
        ("source_chromosome_v_sha256", "target_chromosome_v_sha256"),
    ):
        if receipt[left] != receipt[right]:
            raise ValueError(f"coordinate receipt identity differs: {left}/{right}")
    expected_regions = {
        "chrv_left": (1_747_612, 4_333_001),
        "chrv_right": (13_606_517, 16_754_986),
    }
    regions = receipt.get("regions")
    if not isinstance(regions, list) or [
        item.get("region_id") for item in regions
    ] != list(expected_regions):
        raise ValueError("coordinate receipt regions differ from the frozen order")
    for region in regions:
        bounds = expected_regions[region["region_id"]]
        if (
            (region.get("source_start"), region.get("source_end")) != bounds
            or (region.get("target_start"), region.get("target_end")) != bounds
            or region.get("mapping_mode") != "sequence_identical_coordinate_identity"
            or region.get("source_assembly") != WS276_ASSEMBLY
            or region.get("target_assembly") != WS283_ASSEMBLY
            or region.get("source_chromosome") != 5
            or region.get("target_chromosome") != 5
            or region.get("source_interval_sha256") != region.get("target_interval_sha256")
            or not _SHA256_RE.fullmatch(str(region.get("source_interval_sha256", "")))
        ):
            raise ValueError(f"coordinate identity is incomplete for {region['region_id']}")
    return _sha256(path), {gate.receipt.role: gate.receipt.sha256}


def _verify_state_panel(
    gate: StatePanelReceiptContract,
    coordinate_sha256: str,
    dataset: PredictionDataset,
    hashes: Mapping[str, Any],
    roots: Mapping[str, str | Path],
) -> dict[str, str]:
    if gate.status != "qualified" or any(
        item is None
        for item in (gate.qualification_receipt, gate.feature_receipt, gate.feature_matrix)
    ):
        raise ValueError("real upstream gate remains blocked: state_panel")
    qualification_asset = gate.qualification_receipt
    feature_asset = gate.feature_receipt
    matrix_asset = gate.feature_matrix
    assert qualification_asset is not None
    assert feature_asset is not None
    assert matrix_asset is not None
    qualification_path, qualification = _load_json_asset(qualification_asset, roots)
    if (
        qualification.get("schema_version") != gate.qualification_schema_version
        or qualification.get("coordinate_receipt_sha256") != coordinate_sha256
        or qualification.get("sample_count") != SAMPLE_COUNT
        or not isinstance(qualification.get("block_count"), int)
        or qualification.get("block_count", 0) < 1
        or not isinstance(qualification.get("eligible_block_count"), int)
        or qualification.get("eligible_block_count", 0) < 1
        or qualification.get("outcome_blind") is not True
    ):
        raise ValueError("regional-state qualification state-panel qualification semantics differ")
    _, feature_receipt = _load_json_asset(feature_asset, roots)
    expected_feature_fields = {
        "schema_version": gate.feature_receipt_schema_version,
        "state_panel_schema_version": gate.qualification_schema_version,
        "state_panel_qualification_sha256": _sha256(qualification_path),
        "coordinate_identity_receipt_sha256": coordinate_sha256,
        "sample_count": SAMPLE_COUNT,
        "sample_order_sha256": hashes["sample_order_sha256"],
        "feature_matrix_sha256": hashes["chrv_haplotype_features_sha256"],
        "feature_mask_scope": FEATURE_MASK_SCOPE,
        "phenotype_values_accessed": False,
        "outer_test_phenotypes_accessed": False,
        "biological_claims_permitted": False,
    }
    for key, expected in expected_feature_fields.items():
        if feature_receipt.get(key) != expected:
            raise ValueError(f"regional-state qualification state-feature receipt semantic field differs: {key}")
    matrix_path = _verify_frozen_file(matrix_asset, roots)
    try:
        matrix = np.load(matrix_path, allow_pickle=False)
    except Exception as error:
        raise ValueError("regional-state qualification state-feature matrix is not a safe NPY array") from error
    expected_matrix = np.asarray(dataset.chrv_haplotype_features, dtype=np.float64)
    if (
        matrix.dtype != np.dtype("<f8")
        or matrix.shape != expected_matrix.shape
        or not np.array_equal(matrix, expected_matrix)
    ):
        raise ValueError("prediction haplotype features differ from the qualified regional-state qualification matrix")
    return {
        qualification_asset.role: qualification_asset.sha256,
        feature_asset.role: feature_asset.sha256,
        matrix_asset.role: matrix_asset.sha256,
    }


def _verify_full_kernels(
    gate: FullKernelReceiptContract,
    dataset: PredictionDataset,
    roots: Mapping[str, str | Path],
) -> tuple[str, dict[str, str]]:
    if gate.status != "qualified" or any(
        item is None
        for item in (
            gate.qualification_receipt,
            gate.whole_genome_matrix,
            gate.genome_excluding_chrv_matrix,
        )
    ):
        raise ValueError("real upstream gate remains blocked: full_kernels")
    receipt_asset = gate.qualification_receipt
    whole_asset = gate.whole_genome_matrix
    excluding_asset = gate.genome_excluding_chrv_matrix
    assert receipt_asset is not None and whole_asset is not None and excluding_asset is not None
    _, receipt = _load_json_asset(receipt_asset, roots)
    if (
        receipt.get("schema_version") != gate.receipt_schema_version
        or receipt.get("samples") != SAMPLE_COUNT
        or receipt.get("biological_claims_permitted") is not False
        or not _SHA256_RE.fullmatch(str(receipt.get("ordered_iid_sha256", "")))
    ):
        raise ValueError("full-kernel qualification semantics differ")
    expected = [
        ("whole_genome", 373_279, whole_asset, dataset.whole_genome_kernel),
        (
            "genome_excluding_chrv",
            258_096,
            excluding_asset,
            dataset.genome_excluding_chrv_kernel,
        ),
    ]
    kernels = receipt.get("kernels")
    if not isinstance(kernels, list) or [item.get("id") for item in kernels] != [
        item[0] for item in expected
    ]:
        raise ValueError("full-kernel receipt inventory differs from the frozen pair")
    for record, (kernel_id, marker_count, asset, values) in zip(kernels, expected, strict=True):
        matrix_path = _verify_frozen_file(asset, roots)
        try:
            stored = np.load(matrix_path, allow_pickle=False)
        except Exception as error:
            raise ValueError(f"{kernel_id} upstream matrix is not a safe NPY array") from error
        observed = np.asarray(values, dtype=np.float64)
        npy = record.get("npy")
        if (
            record.get("relationship_markers") != marker_count
            or not isinstance(npy, dict)
            or npy.get("sha256") != asset.sha256
            or npy.get("bytes") != asset.bytes
            or npy.get("shape") != [SAMPLE_COUNT, SAMPLE_COUNT]
            or npy.get("dtype") != "<f8"
            or npy.get("finite") is not True
            or npy.get("exactly_symmetric") is not True
            or npy.get("matrix_payload_sha256") != _matrix_payload_sha256(observed)
            or stored.shape != observed.shape
            or not np.array_equal(stored, observed)
        ):
            raise ValueError(f"{kernel_id} matrix differs from its full-kernel qualification")
    return str(receipt["ordered_iid_sha256"]), {
        receipt_asset.role: receipt_asset.sha256,
        whole_asset.role: whole_asset.sha256,
        excluding_asset.role: excluding_asset.sha256,
    }


def _verify_population_groups(
    gate: PopulationGroupReceiptContract,
    dataset: PredictionDataset,
    roots: Mapping[str, str | Path],
) -> tuple[str, dict[str, str]]:
    _, receipt = _load_json_asset(gate.receipt, roots)
    assignments_path = _verify_frozen_file(gate.assignments, roots)
    if (
        receipt.get("schema_version") != gate.receipt_schema_version
        or receipt.get("assignments_sha256") != gate.assignments.sha256
        or receipt.get("sample_count") != gate.expected_samples
        or receipt.get("eigenvalues_sha256") != gate.eigenvalues_sha256
        or receipt.get("eigenvalue_count") != gate.eigenvalue_count
        or receipt.get("eigenvalues_used_for_clustering")
        is not gate.eigenvalues_used_for_clustering
        or receipt.get("axis_weighting") != gate.axis_weighting
        or receipt.get("eigenvalue_usage") != gate.eigenvalue_usage
        or receipt.get("phenotype_paths_accepted") is not gate.phenotype_paths_accepted
        or receipt.get("phenotype_values_accessed") is not gate.phenotype_values_accessed
        or receipt.get("outcome_access_claim_scope") != gate.outcome_access_claim_scope
        or receipt.get("operator_outcome_blinding_asserted")
        is not gate.operator_outcome_blinding_asserted
        or receipt.get("labels_are_external_ancestry_assignments")
        is not gate.labels_are_external_ancestry_assignments
    ):
        raise ValueError("population-group receipt semantics differ")
    lines = assignments_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "IID\tpopulation_group":
        raise ValueError("population-group assignment header differs")
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        if len(fields) != 2 or any(not _SAFE_ID_RE.fullmatch(item) for item in fields):
            raise ValueError(f"population-group assignment line {line_number} is invalid")
        rows.append((fields[0], fields[1]))
    if len(rows) != gate.expected_samples or len({item[0] for item in rows}) != len(rows):
        raise ValueError("population-group assignments do not contain 209 unique strains")
    if rows != list(zip(dataset.sample_ids, dataset.population_groups, strict=True)):
        raise ValueError("prediction population groups differ from the frozen assignments")
    sizes = {
        group: dataset.population_groups.count(group)
        for group in sorted(set(dataset.population_groups))
    }
    if (
        len(sizes) < gate.minimum_groups
        or receipt.get("selected_k") != len(sizes)
        or receipt.get("group_sizes") != sizes
        or receipt.get("ordered_iid_sha256") != _ordered_iid_sha256(dataset.sample_ids)
    ):
        raise ValueError("population-group receipt counts or sample order differ")
    return str(receipt["ordered_iid_sha256"]), {
        gate.receipt.role: gate.receipt.sha256,
        gate.assignments.role: gate.assignments.sha256,
    }


def _verify_real_upstreams(
    manifest: HeldoutPredictionManifest,
    dataset: PredictionDataset,
    hashes: Mapping[str, Any],
    roots: Mapping[str, str | Path] | None,
) -> dict[str, str]:
    if roots is None:
        raise ValueError("real WS283 mode requires governed upstream root mappings")
    if manifest.status != "qualified_for_file_backed_real_execution":
        raise ValueError("real upstream receipt contract remains blocked by manifest status")
    gates = manifest.upstream_gates
    blocked = [
        name
        for name, status in (
            ("coordinate_identity", gates.coordinate_identity.status),
            ("state_panel", gates.state_panel.status),
            ("full_kernels", gates.full_kernels.status),
        )
        if status != "qualified"
    ]
    if blocked:
        raise ValueError(f"real upstream receipt contract remains blocked: {','.join(blocked)}")
    verified: dict[str, str] = {}
    verified.update(_verify_modern_parent(manifest, dataset, roots))
    verified.update(_verify_parametric_null_terminal(gates.parametric_null_terminal, roots))
    coordinate_sha256, assets = _verify_coordinate_receipt(gates.coordinate_identity, roots)
    verified.update(assets)
    verified.update(
        _verify_state_panel(gates.state_panel, coordinate_sha256, dataset, hashes, roots)
    )
    kernel_order_sha256, assets = _verify_full_kernels(gates.full_kernels, dataset, roots)
    verified.update(assets)
    group_order_sha256, assets = _verify_population_groups(gates.population_groups, dataset, roots)
    verified.update(assets)
    if kernel_order_sha256 != group_order_sha256:
        raise ValueError("full kernels and population groups use different ordered strains")
    return dict(sorted(verified.items()))


def _validate_kernel(name: str, values: np.ndarray, size: int) -> np.ndarray:
    kernel = np.asarray(values, dtype=np.float64)
    if kernel.shape != (size, size):
        raise ValueError(f"{name} must be a {size} by {size} matrix")
    if not np.all(np.isfinite(kernel)):
        raise ValueError(f"{name} contains nonfinite values")
    if not np.allclose(kernel, kernel.T, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} is not symmetric")
    eigenvalues = np.linalg.eigvalsh((kernel + kernel.T) / 2.0)
    tolerance = 1e-8 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError(f"{name} is not positive semidefinite")
    return kernel


def qualify_dataset(
    manifest: HeldoutPredictionManifest,
    dataset: PredictionDataset,
    receipt: PredictionInputReceipt,
    *,
    upstream_roots: Mapping[str, str | Path] | None = None,
) -> InputQualification:
    refuse_legacy_audit(receipt)
    if len(dataset.sample_ids) != SAMPLE_COUNT or len(set(dataset.sample_ids)) != SAMPLE_COUNT:
        raise ValueError("held-out strain prediction requires exactly 209 unique modern-cohort strains")
    if any(not _SAFE_ID_RE.fullmatch(item) for item in dataset.sample_ids):
        raise ValueError("sample identifiers contain an unsafe token")
    if len(dataset.population_groups) != SAMPLE_COUNT:
        raise ValueError("population-group vector differs from the 209-sample order")
    if any(not _SAFE_ID_RE.fullmatch(item) for item in dataset.population_groups):
        raise ValueError("population group contains an unsafe token")
    group_sizes = {
        group: dataset.population_groups.count(group)
        for group in sorted(set(dataset.population_groups))
    }
    if len(group_sizes) < manifest.splits.minimum_outer_groups:
        raise ValueError("too few ancestry/population groups for blocked outer evaluation")
    if min(group_sizes.values()) < manifest.splits.minimum_group_size:
        raise ValueError("a population group is smaller than the frozen gate")
    if list(dataset.phenotypes) != list(TRAITS):
        raise ValueError("dataset traits differ from the ordered four-trait panel")
    for trait in TRAITS:
        phenotype = np.asarray(dataset.phenotypes[trait], dtype=np.float64)
        if phenotype.shape != (SAMPLE_COUNT,) or not np.all(np.isfinite(phenotype)):
            raise ValueError(f"{trait} phenotype must have 209 finite values")
    _validate_kernel("whole-genome kernel", dataset.whole_genome_kernel, SAMPLE_COUNT)
    _validate_kernel(
        "genome-excluding-chromosome-V kernel",
        dataset.genome_excluding_chrv_kernel,
        SAMPLE_COUNT,
    )
    features = np.asarray(dataset.chrv_haplotype_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != SAMPLE_COUNT or features.shape[1] < 1:
        raise ValueError("chromosome-V haplotype feature matrix has invalid dimensions")
    if not np.all(np.isfinite(features)):
        raise ValueError("chromosome-V haplotype features contain nonfinite values")
    observed_hashes = dataset_hashes(dataset)
    expected_hashes = {
        "sample_order_sha256": receipt.sample_order_sha256,
        "group_labels_sha256": receipt.group_labels_sha256,
        "phenotype_sha256_by_trait": receipt.phenotype_sha256_by_trait,
        "whole_genome_kernel_sha256": receipt.whole_genome_kernel_sha256,
        "genome_excluding_chrv_kernel_sha256": (
            receipt.genome_excluding_chrv_kernel_sha256
        ),
        "chrv_haplotype_features_sha256": receipt.chrv_haplotype_features_sha256,
    }
    if observed_hashes != expected_hashes:
        raise ValueError("prediction input identity differs from its frozen receipt")
    upstream_hashes = (
        _verify_real_upstreams(manifest, dataset, observed_hashes, upstream_roots)
        if receipt.mode == "real_ws283"
        else {}
    )
    return InputQualification(
        schema_version=QUALIFICATION_VERSION,
        mode=receipt.mode,
        sample_count=209,
        population_group_count=len(group_sizes),
        population_group_sizes=group_sizes,
        haplotype_feature_count=int(features.shape[1]),
        hashes=observed_hashes,
        input_receipt_sha256=_model_sha256(receipt),
        upstream_asset_sha256_by_role=upstream_hashes,
        genotype_feature_mask_scope=receipt.genotype_feature_mask_scope,
        leakage_gates_passed=True,
        legacy_quarantine_passed=True,
        biological_claims_permitted=False,
    )


class OuterSplit(_StrictModel):
    fold_id: str
    heldout_group: str
    train_indices: list[int]
    test_indices: list[int]
    train_sample_ids: list[str]
    test_sample_ids: list[str]
    inner_validation_groups: list[str]
    outer_test_used_for_selection: Literal[False]


class SplitReceipt(_StrictModel):
    schema_version: Literal[SPLIT_VERSION]
    sample_count: Literal[209]
    group_count: int = Field(ge=5)
    sample_order_sha256: str
    group_labels_sha256: str
    input_qualification_sha256: str
    folds: list[OuterSplit]
    every_sample_tested_exactly_once: Literal[True]
    phenotype_values_used: Literal[False]

    @field_validator(
        "sample_order_sha256", "group_labels_sha256", "input_qualification_sha256"
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("split binding SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def valid_static_partition(self) -> "SplitReceipt":
        if len(self.folds) != self.group_count:
            raise ValueError("outer fold count differs from the declared group count")
        heldout_groups = [fold.heldout_group for fold in self.folds]
        if len(heldout_groups) != len(set(heldout_groups)):
            raise ValueError("outer folds contain a duplicate held-out group")
        fold_ids = [fold.fold_id for fold in self.folds]
        if len(fold_ids) != len(set(fold_ids)):
            raise ValueError("outer folds contain a duplicate fold ID")
        tested: list[int] = []
        tested_ids: list[str] = []
        universe = set(range(self.sample_count))
        for fold in self.folds:
            train = fold.train_indices
            test = fold.test_indices
            if (
                len(train) != len(set(train))
                or len(test) != len(set(test))
                or set(train) & set(test)
                or set(train) | set(test) != universe
                or any(index not in universe for index in [*train, *test])
            ):
                raise ValueError("outer fold indices are overlapping, duplicated, or incomplete")
            if len(fold.train_sample_ids) != len(train) or len(fold.test_sample_ids) != len(test):
                raise ValueError("outer fold sample IDs do not align with its indices")
            if (
                len(fold.train_sample_ids) != len(set(fold.train_sample_ids))
                or len(fold.test_sample_ids) != len(set(fold.test_sample_ids))
                or set(fold.train_sample_ids) & set(fold.test_sample_ids)
            ):
                raise ValueError("outer fold sample IDs overlap or contain duplicates")
            if fold.heldout_group in fold.inner_validation_groups:
                raise ValueError("held-out group appears in inner validation groups")
            if len(fold.inner_validation_groups) != len(set(fold.inner_validation_groups)):
                raise ValueError("inner validation group list contains duplicates")
            tested.extend(test)
            tested_ids.extend(fold.test_sample_ids)
        if sorted(tested) != list(range(self.sample_count)) or len(tested) != self.sample_count:
            raise ValueError("outer test folds do not cover every index exactly once")
        if len(tested_ids) != len(set(tested_ids)) or len(tested_ids) != self.sample_count:
            raise ValueError("outer test folds do not cover every sample ID exactly once")
        expected_inner = set(heldout_groups)
        for fold in self.folds:
            if set(fold.inner_validation_groups) != expected_inner - {fold.heldout_group}:
                raise ValueError("inner validation groups differ from the other outer groups")
        return self


def build_outer_splits(
    manifest: HeldoutPredictionManifest,
    sample_ids: Sequence[str],
    groups: Sequence[str],
    qualification: InputQualification,
) -> SplitReceipt:
    if len(sample_ids) != SAMPLE_COUNT or len(groups) != SAMPLE_COUNT:
        raise ValueError("outer split construction requires the exact 209-sample cohort")
    ordered_groups = sorted(set(groups))
    if len(ordered_groups) < manifest.splits.minimum_outer_groups:
        raise ValueError("outer split construction has too few groups")
    sample_hash = _hash_strings(sample_ids)
    group_hash = _hash_strings(
        [f"{sample}\t{group}" for sample, group in zip(sample_ids, groups, strict=True)]
    )
    if (
        qualification.sample_count != SAMPLE_COUNT
        or qualification.hashes.get("sample_order_sha256") != sample_hash
        or qualification.hashes.get("group_labels_sha256") != group_hash
    ):
        raise ValueError("outer split inputs differ from the input qualification")
    folds: list[OuterSplit] = []
    tested: list[int] = []
    for group in ordered_groups:
        test = [index for index, value in enumerate(groups) if value == group]
        train = [index for index, value in enumerate(groups) if value != group]
        inner_groups = [item for item in ordered_groups if item != group]
        if not test or len(inner_groups) < 2:
            raise ValueError("blocked split produced an empty or unnested fold")
        tested.extend(test)
        folds.append(
            OuterSplit(
                fold_id=f"outer_{len(folds) + 1:02d}_{group}",
                heldout_group=group,
                train_indices=train,
                test_indices=test,
                train_sample_ids=[sample_ids[index] for index in train],
                test_sample_ids=[sample_ids[index] for index in test],
                inner_validation_groups=inner_groups,
                outer_test_used_for_selection=False,
            )
        )
    if sorted(tested) != list(range(SAMPLE_COUNT)) or len(tested) != SAMPLE_COUNT:
        raise ValueError("outer test folds do not partition all 209 strains exactly once")
    return SplitReceipt(
        schema_version=SPLIT_VERSION,
        sample_count=209,
        group_count=len(ordered_groups),
        sample_order_sha256=sample_hash,
        group_labels_sha256=group_hash,
        input_qualification_sha256=_model_sha256(qualification),
        folds=folds,
        every_sample_tested_exactly_once=True,
        phenotype_values_used=False,
    )


def _haplotype_kernel(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    scale = np.std(values, axis=0, ddof=0)
    retained = scale > np.finfo(np.float64).eps
    if not np.any(retained):
        raise ValueError("all chromosome-V haplotype features are constant")
    # This mask is intentionally transductive: the fixed genotype panel for all
    # 209 strains determines which encoded columns exist. No outcome value or outer-
    # test phenotype enters the mask, and the resulting claim is not inductive
    # performance on unseen populations. Avoid estimating an additional global
    # centering/scaling transform from outer-test strains.
    encoded = values[:, retained]
    return encoded @ encoded.T / encoded.shape[1]


def _predict_kernel(
    kernel: np.ndarray,
    phenotype: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    ridge: float,
) -> np.ndarray:
    training_mean = float(np.mean(phenotype[train]))
    centered = phenotype[train] - training_mean
    system = kernel[np.ix_(train, train)] + ridge * np.eye(len(train))
    try:
        coefficients = np.linalg.solve(system, centered)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, centered, rcond=None)[0]
    return training_mean + kernel[np.ix_(test, train)] @ coefficients


def _candidate_grid(model_id: str) -> list[dict[str, float]]:
    if model_id == "training_mean":
        return [{}]
    if model_id == "combined_excluding_chrv_plus_haplotype":
        return [
            {"ridge": ridge, "haplotype_weight": weight}
            for ridge in RIDGE_GRID
            for weight in COMBINED_WEIGHT_GRID
        ]
    return [{"ridge": ridge} for ridge in RIDGE_GRID]


def _kernel_for_candidate(
    model_id: str,
    candidate: Mapping[str, float],
    whole: np.ndarray,
    excluding: np.ndarray,
    haplotype: np.ndarray,
) -> np.ndarray | None:
    if model_id == "training_mean":
        return None
    if model_id == "whole_genome_gblup":
        return whole
    if model_id == "chrv_haplotype":
        return haplotype
    if model_id == "genome_excluding_chrv":
        return excluding
    if model_id == "combined_excluding_chrv_plus_haplotype":
        weight = candidate["haplotype_weight"]
        return (1.0 - weight) * excluding + weight * haplotype
    raise ValueError(f"unknown comparator: {model_id}")


def _predict_candidate(
    model_id: str,
    candidate: Mapping[str, float],
    phenotype: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    whole: np.ndarray,
    excluding: np.ndarray,
    haplotype: np.ndarray,
) -> np.ndarray:
    if model_id == "training_mean":
        return np.full(len(test), float(np.mean(phenotype[train])), dtype=np.float64)
    kernel = _kernel_for_candidate(model_id, candidate, whole, excluding, haplotype)
    assert kernel is not None
    return _predict_kernel(kernel, phenotype, train, test, candidate["ridge"])


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def _select_candidate(
    model_id: str,
    phenotype: np.ndarray,
    groups: np.ndarray,
    outer_group: str,
    whole: np.ndarray,
    excluding: np.ndarray,
    haplotype: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, Any]], np.ndarray]:
    candidates = _candidate_grid(model_id)
    scores: list[dict[str, Any]] = []
    residuals_by_candidate: list[np.ndarray] = []
    inner_groups = [item for item in sorted(set(groups.tolist())) if item != outer_group]
    for candidate in candidates:
        group_rmse: list[float] = []
        residuals: list[float] = []
        for validation_group in inner_groups:
            validation = np.flatnonzero(groups == validation_group)
            training = np.flatnonzero(
                (groups != outer_group) & (groups != validation_group)
            )
            prediction = _predict_candidate(
                model_id,
                candidate,
                phenotype,
                training,
                validation,
                whole,
                excluding,
                haplotype,
            )
            group_rmse.append(_rmse(phenotype[validation], prediction))
            residuals.extend(float(item) for item in phenotype[validation] - prediction)
        score = float(math.fsum(group_rmse) / len(group_rmse))
        scores.append(
            {
                "candidate": dict(candidate),
                "inner_group_macro_rmse": score,
                "inner_group_rmse": group_rmse,
            }
        )
        residuals_by_candidate.append(np.asarray(residuals, dtype=np.float64))
    selected_index = min(
        range(len(candidates)),
        key=lambda index: (
            scores[index]["inner_group_macro_rmse"],
            -candidates[index].get("ridge", math.inf),
            candidates[index].get("haplotype_weight", -math.inf),
        ),
    )
    return candidates[selected_index], scores, residuals_by_candidate[selected_index]


def _absolute_residual_quantile(residuals: np.ndarray, level: float) -> float:
    values = np.sort(np.abs(np.asarray(residuals, dtype=np.float64)))
    if not len(values):
        raise ValueError("cannot calibrate a prediction interval without validation residuals")
    index = min(len(values) - 1, max(0, math.ceil(level * (len(values) + 1)) - 1))
    return float(values[index])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(
        np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    )
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(np.sum(first_centered * second_centered) / denominator, -1, 1))


def _calibration(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(predicted)), predicted])
    coefficients = np.linalg.lstsq(design, observed, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def _model_summary(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    observed = np.asarray([item["observed"] for item in records], dtype=np.float64)
    predicted = np.asarray([item["predicted"] for item in records], dtype=np.float64)
    group_rmse = []
    for group in sorted({item["heldout_group"] for item in records}):
        selected = [item for item in records if item["heldout_group"] == group]
        group_rmse.append(
            _rmse(
                np.asarray([item["observed"] for item in selected]),
                np.asarray([item["predicted"] for item in selected]),
            )
        )
    intercept, slope = _calibration(observed, predicted)
    return {
        "population_group_macro_rmse": float(math.fsum(group_rmse) / len(group_rmse)),
        "sample_rmse": _rmse(observed, predicted),
        "sample_mae": float(np.mean(np.abs(observed - predicted))),
        "pearson": _correlation(observed, predicted),
        "spearman": _correlation(_average_ranks(observed), _average_ranks(predicted)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "coverage_50": float(
            np.mean(
                [
                    item["interval_50_lower"]
                    <= item["observed"]
                    <= item["interval_50_upper"]
                    for item in records
                ]
            )
        ),
        "coverage_90": float(
            np.mean(
                [
                    item["interval_90_lower"]
                    <= item["observed"]
                    <= item["interval_90_upper"]
                    for item in records
                ]
            )
        ),
    }


def _expected_real_upstream_hashes(manifest: HeldoutPredictionManifest) -> dict[str, str]:
    gates = manifest.upstream_gates
    assets: list[FrozenFile] = [
        *manifest.modern_cohort.required_files,
        gates.parametric_null_terminal.success_marker,
        gates.parametric_null_terminal.aggregation_summary,
        gates.parametric_null_terminal.rich_summary,
        gates.population_groups.receipt,
        gates.population_groups.assignments,
    ]
    optional_assets = [
        gates.coordinate_identity.receipt,
        gates.state_panel.qualification_receipt,
        gates.state_panel.feature_receipt,
        gates.state_panel.feature_matrix,
        gates.full_kernels.qualification_receipt,
        gates.full_kernels.whole_genome_matrix,
        gates.full_kernels.genome_excluding_chrv_matrix,
    ]
    assets.extend(item for item in optional_assets if item is not None)
    roles = [item.role for item in assets]
    if len(roles) != len(set(roles)):
        raise ValueError("real upstream asset roles are not unique")
    return dict(sorted((item.role, item.sha256) for item in assets))


def _validate_evaluation_bindings(
    manifest: HeldoutPredictionManifest,
    dataset: PredictionDataset,
    qualification: InputQualification,
    splits: SplitReceipt,
) -> None:
    observed_hashes = dataset_hashes(dataset)
    if observed_hashes != qualification.hashes:
        raise ValueError("evaluation dataset differs from its input qualification")
    if qualification.genotype_feature_mask_scope != manifest.selection.genotype_feature_mask_scope:
        raise ValueError("evaluation feature-mask scope differs from the frozen contract")
    if qualification.mode == "real_ws283":
        if qualification.upstream_asset_sha256_by_role != _expected_real_upstream_hashes(manifest):
            raise ValueError("real input qualification upstream receipt hashes differ")
    elif qualification.upstream_asset_sha256_by_role:
        raise ValueError("synthetic input qualification cannot carry real upstream receipts")
    if (
        splits.sample_count != SAMPLE_COUNT
        or splits.sample_order_sha256 != observed_hashes["sample_order_sha256"]
        or splits.group_labels_sha256 != observed_hashes["group_labels_sha256"]
        or splits.input_qualification_sha256 != _model_sha256(qualification)
    ):
        raise ValueError("outer split receipt is not bound to this qualification and dataset")

    groups = tuple(dataset.population_groups)
    sample_ids = tuple(dataset.sample_ids)
    ordered_groups = sorted(set(groups))
    group_sizes = {group: groups.count(group) for group in ordered_groups}
    if (
        len(ordered_groups) != qualification.population_group_count
        or group_sizes != qualification.population_group_sizes
        or len(splits.folds) != len(ordered_groups)
        or splits.group_count != len(ordered_groups)
    ):
        raise ValueError("split groups differ from the qualified population-group inventory")
    tested: list[int] = []
    for fold_number, (fold, heldout_group) in enumerate(
        zip(splits.folds, ordered_groups, strict=True), 1
    ):
        expected_test = [index for index, group in enumerate(groups) if group == heldout_group]
        expected_train = [index for index, group in enumerate(groups) if group != heldout_group]
        expected_inner = [group for group in ordered_groups if group != heldout_group]
        if (
            fold.fold_id != f"outer_{fold_number:02d}_{heldout_group}"
            or fold.heldout_group != heldout_group
            or fold.train_indices != expected_train
            or fold.test_indices != expected_test
            or fold.train_sample_ids != [sample_ids[index] for index in expected_train]
            or fold.test_sample_ids != [sample_ids[index] for index in expected_test]
            or fold.inner_validation_groups != expected_inner
            or fold.outer_test_used_for_selection is not False
        ):
            raise ValueError(
                f"outer fold leaks or differs from the frozen grouping: {fold.fold_id}"
            )
        if any(groups[index] == heldout_group for index in fold.train_indices):
            raise ValueError(f"outer held-out group appears in training: {fold.fold_id}")
        if heldout_group in fold.inner_validation_groups:
            raise ValueError(f"outer held-out group appears in inner validation: {fold.fold_id}")
        tested.extend(fold.test_indices)
    if sorted(tested) != list(range(SAMPLE_COUNT)) or len(tested) != SAMPLE_COUNT:
        raise ValueError("outer split receipt does not test every strain exactly once")


def evaluate_heldout_prediction(
    manifest: HeldoutPredictionManifest,
    dataset: PredictionDataset,
    qualification: InputQualification,
    splits: SplitReceipt,
) -> dict[str, Any]:
    if qualification.sample_count != SAMPLE_COUNT or splits.sample_count != SAMPLE_COUNT:
        raise ValueError("qualified inputs and splits must retain all 209 strains")
    _validate_evaluation_bindings(manifest, dataset, qualification, splits)
    whole = np.asarray(dataset.whole_genome_kernel, dtype=np.float64)
    excluding = np.asarray(dataset.genome_excluding_chrv_kernel, dtype=np.float64)
    haplotype = _haplotype_kernel(dataset.chrv_haplotype_features)
    groups = np.asarray(dataset.population_groups, dtype=str)
    all_predictions: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for trait in TRAITS:
        phenotype = np.asarray(dataset.phenotypes[trait], dtype=np.float64)
        for fold in splits.folds:
            training = np.asarray(fold.train_indices, dtype=np.int64)
            testing = np.asarray(fold.test_indices, dtype=np.int64)
            for model_id in MODEL_IDS:
                selected, candidate_scores, validation_residuals = _select_candidate(
                    model_id,
                    phenotype,
                    groups,
                    fold.heldout_group,
                    whole,
                    excluding,
                    haplotype,
                )
                prediction = _predict_candidate(
                    model_id,
                    selected,
                    phenotype,
                    training,
                    testing,
                    whole,
                    excluding,
                    haplotype,
                )
                half_widths = {
                    level: _absolute_residual_quantile(validation_residuals, level)
                    for level in INTERVAL_LEVELS
                }
                selections.append(
                    {
                        "trait": trait,
                        "fold_id": fold.fold_id,
                        "heldout_group": fold.heldout_group,
                        "model_id": model_id,
                        "selected_hyperparameters": selected,
                        "candidate_scores": candidate_scores,
                        "selection_source": "inner_population_groups_only",
                        "outer_test_used_for_selection": False,
                        "validation_residual_count": int(len(validation_residuals)),
                        "interval_half_widths": {
                            "0.5": half_widths[0.5],
                            "0.9": half_widths[0.9],
                        },
                    }
                )
                for local_index, sample_index in enumerate(testing.tolist()):
                    value = float(prediction[local_index])
                    all_predictions.append(
                        {
                            "trait": trait,
                            "fold_id": fold.fold_id,
                            "heldout_group": fold.heldout_group,
                            "sample_id": dataset.sample_ids[sample_index],
                            "model_id": model_id,
                            "observed": float(phenotype[sample_index]),
                            "predicted": value,
                            "interval_50_lower": value - half_widths[0.5],
                            "interval_50_upper": value + half_widths[0.5],
                            "interval_90_lower": value - half_widths[0.9],
                            "interval_90_upper": value + half_widths[0.9],
                        }
                    )
    expected_predictions = len(TRAITS) * len(MODEL_IDS) * SAMPLE_COUNT
    if len(all_predictions) != expected_predictions:
        raise ValueError("held-out prediction grid is incomplete")
    trait_summaries: dict[str, Any] = {}
    for trait in TRAITS:
        trait_summaries[trait] = {}
        for model_id in MODEL_IDS:
            records = [
                item
                for item in all_predictions
                if item["trait"] == trait and item["model_id"] == model_id
            ]
            trait_summaries[trait][model_id] = _model_summary(records)
    return {
        "schema_version": SUMMARY_VERSION,
        "analysis_id": manifest.analysis_id,
        "mode": qualification.mode,
        "sample_count": SAMPLE_COUNT,
        "population_group_count": qualification.population_group_count,
        "traits": list(TRAITS),
        "models": list(MODEL_IDS),
        "selection": selections,
        "predictions": all_predictions,
        "metrics": trait_summaries,
        "leakage_control": {
            "outer_test_used_for_selection": False,
            "selection_source": "inner_population_groups_only",
            "group_labels_use_phenotypes": False,
            "features_use_phenotypes": False,
        },
        "claim_boundary": {
            "biological_claims_permitted": False,
            "causal_inference_permitted": False,
            "independent_replication_established": False,
            "legacy_audit_reused": False,
        },
    }


def _write_new_file(path: Path, content: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"write-once file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_write_once_bundle(
    manifest: HeldoutPredictionManifest,
    output: str | Path,
    qualification: InputQualification,
    splits: SplitReceipt,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"write-once output already exists: {destination}")
    payloads = {
        "input_qualification.json": _canonical_bytes(qualification.model_dump(mode="json")),
        "outer_splits.json": _canonical_bytes(splits.model_dump(mode="json")),
        "predictions.jsonl": b"".join(
            _canonical_bytes(item) for item in evaluation["predictions"]
        ),
        "summary.json": _canonical_bytes(
            {key: value for key, value in evaluation.items() if key != "predictions"}
        ),
    }
    if list(payloads) != manifest.receipts.required_payloads:
        raise ValueError("write-once payload inventory differs from the manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        for name, content in payloads.items():
            _write_new_file(stage / name, content)
        lines = [f"{_sha256(stage / name)}  ./{name}" for name in payloads]
        _write_new_file(stage / "SHA256SUMS.txt", ("\n".join(lines) + "\n").encode())
        _write_new_file(stage / "SUCCESS", b"SUCCESS\n")
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    verification = verify_write_once_bundle(manifest, destination)
    return {
        "schema_version": BUNDLE_VERSION,
        "output": str(destination),
        **verification,
    }


def verify_write_once_bundle(
    manifest: HeldoutPredictionManifest, output: str | Path
) -> dict[str, Any]:
    root = Path(output).resolve()
    expected = [*manifest.receipts.required_payloads, "SHA256SUMS.txt", "SUCCESS"]
    observed = sorted(item.name for item in root.iterdir() if item.is_file())
    if observed != sorted(expected):
        raise ValueError("write-once bundle inventory differs")
    if (root / "SUCCESS").read_bytes() != b"SUCCESS\n":
        raise ValueError("write-once bundle SUCCESS marker differs")
    entries: dict[str, str] = {}
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  \./([^/]+)", line)
        if match is None:
            raise ValueError("write-once checksum manifest is malformed")
        entries[match.group(2)] = match.group(1)
    if list(entries) != manifest.receipts.required_payloads:
        raise ValueError("write-once checksum coverage differs")
    for name, expected_hash in entries.items():
        if _sha256(root / name) != expected_hash:
            raise ValueError(f"write-once checksum failed: {name}")
    return {
        "verified_payload_count": len(entries),
        "checksum_manifest_sha256": _sha256(root / "SHA256SUMS.txt"),
        "terminal_success": True,
    }
