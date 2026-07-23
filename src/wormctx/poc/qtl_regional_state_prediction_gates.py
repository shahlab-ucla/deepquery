"""Governed real-input gates for the abamectin regional-state qualification and held-out prediction evidence streams.

The derivation interfaces in this module are outcome-free.  They record and verify
immutable input identities, refuse to infer assembly identity, and can derive operational
cross-validation strata from genotype principal components without accepting a phenotype
path or phenotype values.  They do not assert that the human operator lacked prior outcome
awareness, and they do not run association or prediction analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "wormctx-abamectin-regional_state-heldout_prediction-real-input-handoff-1.1"
GROUP_RECEIPT_VERSION = "wormctx-abamectin-phenotype-blind-groups-1.1"
KERNEL_MARKER_RECEIPT_VERSION = "wormctx-abamectin-full-kernel-markers-1.0"
KERNEL_QUALIFICATION_VERSION = "wormctx-abamectin-full-kernel-qualification-1.0"
NULL_RELEASE_RECEIPT_VERSION = "wormctx-abamectin-null-release-verification-1.0"
WS276_ASSEMBLY = "PRJNA13758.WS276"
WS283_ASSEMBLY = "PRJNA13758.WS283_inferred_not_declared_in_vcf_header"
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


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _ordered_id_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FrozenAsset(_StrictModel):
    role: str
    root_id: str
    relative_path: str
    bytes: int = Field(gt=0)
    sha256: str

    @field_validator("role", "root_id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("asset role/root_id must be a safe identifier")
        return value

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value.startswith("~"):
            raise ValueError("asset paths must be governed root-relative paths")
        if "\\" in value or not path.parts:
            raise ValueError("asset paths must use nonempty POSIX-relative syntax")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("asset SHA-256 must be lowercase hexadecimal")
        return value


class IntervalDigest(_StrictModel):
    id: Literal["chrv_left", "chrv_right"]
    chromosome: Literal[5]
    start: int = Field(gt=0)
    end: int = Field(gt=0)
    canonical_sequence_sha256: str
    canonicalization: Literal["uppercase_sequence_without_fasta_headers_or_whitespace"]

    @field_validator("canonical_sequence_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("sequence SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def frozen_interval(self) -> "IntervalDigest":
        if (self.start, self.end) != EXPECTED_REGIONS[self.id]:
            raise ValueError(f"{self.id} differs from the frozen coordinate envelope")
        return self


class ReferenceAcquisition(_StrictModel):
    url: str
    object_version_id: str
    observed_content_length: int = Field(gt=0)
    observed_etag: str
    expected_sha256: None
    sha256_must_be_computed_after_download: Literal[True]
    download_deferred_until_parametric_null_terminal: Literal[True]
    equal_content_length_is_identity_evidence: Literal[False]


class ReferenceEvidence(_StrictModel):
    assembly: Literal[WS276_ASSEMBLY, WS283_ASSEMBLY]
    status: Literal["qualified", "missing"]
    vcf_header_declares_reference_or_assembly: Literal[False]
    fasta: FrozenAsset | None
    fai: FrozenAsset | None
    chromosome_v_length: int | None = Field(default=None, gt=0)
    chromosome_v_canonical_sha256: str | None
    intervals: list[IntervalDigest]
    acquisition: ReferenceAcquisition | None

    @field_validator("chromosome_v_canonical_sha256")
    @classmethod
    def valid_optional_hash(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("chromosome-V SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def evidence_matches_status(self) -> "ReferenceEvidence":
        if self.status == "qualified":
            if None in (self.fasta, self.fai, self.chromosome_v_length):
                raise ValueError("qualified reference evidence requires FASTA, FAI, and chrV length")
            if self.chromosome_v_canonical_sha256 is None:
                raise ValueError("qualified reference evidence requires a chrV sequence hash")
            if [item.id for item in self.intervals] != ["chrv_left", "chrv_right"]:
                raise ValueError("qualified reference evidence requires both frozen intervals")
            if self.acquisition is not None:
                raise ValueError("qualified reference evidence cannot retain a pending acquisition")
        else:
            if any(
                item is not None
                for item in (
                    self.fasta,
                    self.fai,
                    self.chromosome_v_length,
                    self.chromosome_v_canonical_sha256,
                )
            ) or self.intervals:
                raise ValueError("missing reference evidence cannot contain sequence claims")
            if self.acquisition is None:
                raise ValueError("missing reference evidence requires an acquisition contract")
        return self


class CoordinateIdentityGate(_StrictModel):
    status: Literal["blocked", "qualified"]
    mapping_mode: Literal["sequence_identical_coordinate_identity"]
    source: ReferenceEvidence
    target: ReferenceEvidence
    qualification_receipt: FrozenAsset | None
    numerical_coordinate_equality_required: Literal[True]
    interval_sequence_identity_required: Literal[True]
    vcf_filename_or_release_date_is_identity_evidence: Literal[False]
    equal_fasta_byte_length_is_identity_evidence: Literal[False]
    assembly_identity_inferred: Literal[False]
    liftover_fallback_permitted_in_v1: Literal[False]

    @model_validator(mode="after")
    def identity_status(self) -> "CoordinateIdentityGate":
        if self.source.assembly != WS276_ASSEMBLY or self.target.assembly != WS283_ASSEMBLY:
            raise ValueError("coordinate gate source/target assemblies are reversed or unknown")
        ready = coordinate_identity_ready(self)
        if self.status == "qualified" and not ready:
            raise ValueError("coordinate identity cannot be qualified from incomplete or unequal evidence")
        if self.status == "blocked" and ready:
            raise ValueError("coordinate identity evidence is complete; status must be qualified")
        if self.status == "blocked" and self.qualification_receipt is not None:
            raise ValueError("blocked coordinate identity cannot contain a qualification receipt")
        return self


class StatePanelQualification(_StrictModel):
    schema_version: Literal["wormctx-abamectin-haplotype-pav-panel-qualification-1.0"]
    samples: Literal[209]
    ordered_iid_sha256: str
    state_count: int = Field(gt=0)
    complete_binary_or_categorical_calls: Literal[True]
    outcome_blind: Literal[True]

    @field_validator("ordered_iid_sha256")
    @classmethod
    def valid_order_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("state-panel order hash must be lowercase hexadecimal")
        return value


class StatePanelGate(_StrictModel):
    status: Literal["missing", "qualified"]
    expected_samples: Literal[209]
    exact_sample_order: Literal["baseline_parent_fam_order"]
    accepted_definition_methods_v1: list[
        Literal["outcome_blind_pangenome_path_cluster", "outcome_blind_presence_absence_locus"]
    ]
    observed_small_variant_vcf: FrozenAsset
    observed_small_variant_vcf_definition_method: Literal["beagle_imputed_small_variant_genotypes"]
    observed_small_variant_vcf_accepted_as_v1_haplotype_or_pav: Literal[False]
    divergent_region_bed_accepted_as_presence_absence_calls: Literal[False]
    qualified_panel_assets: list[FrozenAsset]
    qualified_panel_evidence: StatePanelQualification | None
    phenotype_accessed_for_block_definition: Literal[False]
    decision_required: Literal[
        "acquire_genuine_pangenome_or_pav_calls_or_version_an_explicit_v2_vcf_haplotype_contract"
    ]

    @model_validator(mode="after")
    def real_panel_required(self) -> "StatePanelGate":
        if self.accepted_definition_methods_v1 != [
            "outcome_blind_pangenome_path_cluster",
            "outcome_blind_presence_absence_locus",
        ]:
            raise ValueError("accepted regional-state qualification-v1 state methods differ from the frozen contract")
        expected_roles = [
            "real_haplotype_pav_state_panel",
            "real_haplotype_pav_sample_order",
            "real_haplotype_pav_call_definition_receipt",
            "real_haplotype_pav_qualification_receipt",
        ]
        if self.status == "qualified":
            if [item.role for item in self.qualified_panel_assets] != expected_roles:
                raise ValueError("qualified state panel requires the exact frozen asset set")
            if self.qualified_panel_evidence is None:
                raise ValueError("qualified state panel requires numerical qualification evidence")
        elif self.qualified_panel_assets or self.qualified_panel_evidence is not None:
            raise ValueError("missing state-panel status cannot contain qualified evidence")
        return self


class GrmVerification(_StrictModel):
    samples: Literal[209]
    relationship_markers: Literal[1080]
    triangular_float32_entries: Literal[21945]
    pairwise_marker_count_min: Literal[1080]
    pairwise_marker_count_max: Literal[1080]
    positive_semidefinite_within_float32_tolerance: Literal[True]
    minimum_eigenvalue: float
    psd_tolerance: float = Field(gt=0)
    fam_sha256: str
    ordered_iid_sha256: str

    @field_validator("fam_sha256", "ordered_iid_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("GRM evidence hash must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def valid_psd_evidence(self) -> "GrmVerification":
        if not math.isfinite(self.minimum_eigenvalue):
            raise ValueError("GRM minimum eigenvalue must be finite")
        if self.minimum_eigenvalue < -self.psd_tolerance:
            raise ValueError("GRM minimum eigenvalue exceeds its frozen PSD tolerance")
        return self


class QualifiedGrmGate(_StrictModel):
    status: Literal["qualified"]
    relationship_kind: Literal["ldpruned"]
    chromosome_excluded: Literal[5]
    files: list[FrozenAsset]
    verification: GrmVerification

    @model_validator(mode="after")
    def exact_files(self) -> "QualifiedGrmGate":
        if [item.role for item in self.files] != [
            "ldpruned_chr5_excluded_grm_bin",
            "ldpruned_chr5_excluded_grm_n_bin",
            "ldpruned_chr5_excluded_grm_id",
            "ldpruned_chr5_excluded_marker_list",
            "ldpruned_grm_verification_receipt",
        ]:
            raise ValueError("LD-pruned chrV-excluded GRM file inventory is incomplete")
        return self


class KernelOutputContract(_StrictModel):
    id: Literal["whole_genome", "genome_excluding_chrv"]
    marker_count: Literal[373279, 258096]
    preserve_gcta_grm_triple: Literal[True]
    emit_symmetric_float64_npy: Literal[True]


class FullKernelDerivationGate(_StrictModel):
    status: Literal[
        "deferred_until_parametric_null_terminal",
        "ready_for_remote_derivation",
        "qualified",
    ]
    input_files: list[FrozenAsset]
    sample_count: Literal[209]
    total_marker_count: Literal[373279]
    chromosome_v_marker_count: Literal[115183]
    non_chromosome_v_marker_count: Literal[258096]
    chromosome_v_code: Literal[5]
    marker_selection_rule: Literal["bim_column_2_in_file_order_partitioned_by_bim_column_1_equals_5"]
    outputs: list[KernelOutputContract]
    defer_compute_until_parametric_null_terminal_to_avoid_contention: Literal[True]
    required_verification: list[str]
    qualified_output_assets: list[FrozenAsset]
    numerical_qualification_receipt: FrozenAsset | None

    @model_validator(mode="after")
    def exact_derivation(self) -> "FullKernelDerivationGate":
        if [item.role for item in self.input_files] != [
            "modern_cohort_bed",
            "modern_cohort_bim",
            "modern_cohort_fam",
        ]:
            raise ValueError("full-kernel source must be the frozen parent BED/BIM/FAM")
        if [item.id for item in self.outputs] != ["whole_genome", "genome_excluding_chrv"]:
            raise ValueError("full-kernel outputs must be whole-genome then ex-chrV")
        required = {
            "sha256_and_bytes",
            "exact_fam_iid_order",
            "209_by_209_shape",
            "finite",
            "symmetric",
            "psd_with_recorded_tolerance",
            "pairwise_marker_counts",
            "marker_list_sha256",
        }
        if set(self.required_verification) != required:
            raise ValueError("full-kernel verification requirements are incomplete")
        expected_roles = [
            "whole_genome_grm_bin",
            "whole_genome_grm_n_bin",
            "whole_genome_grm_id",
            "whole_genome_float64_npy",
            "genome_excluding_chrv_grm_bin",
            "genome_excluding_chrv_grm_n_bin",
            "genome_excluding_chrv_grm_id",
            "genome_excluding_chrv_float64_npy",
            "full_kernel_marker_manifest",
            "full_kernel_run_manifest",
            "full_kernel_checksum_receipt",
        ]
        if self.status == "qualified":
            if [item.role for item in self.qualified_output_assets] != expected_roles:
                raise ValueError("qualified full kernels require the exact frozen output asset set")
            receipt = self.numerical_qualification_receipt
            if receipt is None or receipt.role != "full_kernel_numerical_qualification_receipt":
                raise ValueError("qualified full kernels require a numerical qualification receipt")
        elif self.qualified_output_assets or self.numerical_qualification_receipt is not None:
            raise ValueError("unqualified full kernels cannot claim frozen output evidence")
        return self


class GroupAlgorithm(_StrictModel):
    id: Literal["standardized_pc10_deterministic_farthest_first_kmeans_silhouette_v1"]
    pc_columns: list[str]
    standardization: Literal["column_mean_zero_population_sd_one"]
    initialization: Literal[
        "nearest_origin_then_farthest_first_with_lexicographic_iid_ties"
    ]
    assignment_tie_break: Literal["lowest_center_index"]
    candidate_k: list[int]
    minimum_group_size: Literal[5]
    selection: Literal["maximum_sample_silhouette_then_lower_k"]
    label_canonicalization: Literal["lexicographic_minimum_member_iid"]
    axis_weighting: Literal["equal_weight_after_per_pc_standardization"]
    eigenvalue_usage: Literal["verified_provenance_only_not_used_for_clustering"]

    @model_validator(mode="after")
    def exact_algorithm(self) -> "GroupAlgorithm":
        if self.pc_columns != [f"PC{index}" for index in range(1, 11)]:
            raise ValueError("population grouping must use exactly PC1-PC10")
        if self.candidate_k != list(range(5, 13)):
            raise ValueError("population-group candidate k grid differs from the frozen design")
        return self


class PopulationGroupGate(_StrictModel):
    status: Literal["derivable", "qualified"]
    expected_samples: Literal[209]
    source_files: list[FrozenAsset]
    source_ordered_iid_sha256: str
    pca_header_present: Literal[True]
    pca_dimensions: Literal[10]
    algorithm: GroupAlgorithm
    phenotype_paths_accepted_by_derivation_cli: Literal[False]
    phenotype_values_accessed: Literal[False]
    outcome_access_claim_scope: Literal["derivation_execution_path_only"]
    operator_outcome_blinding_asserted: Literal[False]
    labels_are_external_ancestry_assignments: Literal[False]
    output_assets: list[FrozenAsset]

    @field_validator("source_ordered_iid_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("population-group order hash must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def group_status(self) -> "PopulationGroupGate":
        if [item.role for item in self.source_files] != [
            "population_group_source_fam",
            "population_group_source_pca10",
            "population_group_source_eigenvalues",
        ]:
            raise ValueError("population-group source inventory is incomplete")
        expected_output_roles = [
            "phenotype_blind_population_groups",
            "phenotype_blind_population_group_receipt",
        ]
        if self.status == "qualified" and [
            item.role for item in self.output_assets
        ] != expected_output_roles:
            raise ValueError(
                "qualified population groups require the assignment and receipt assets"
            )
        if self.status == "derivable" and self.output_assets:
            raise ValueError("derivable population groups cannot claim frozen outputs")
        return self


class PhenotypeProvenanceGate(_StrictModel):
    status: Literal["blocked_on_raw_measurement_provenance", "qualified"]
    publication_doi: Literal["10.1371/journal.ppat.1009297"]
    source_repository_commit: Literal["c197efe22cd5175aeb66aab05d56765cf9702e7f"]
    observed_derived_strain_trait_tables: list[FrozenAsset]
    raw_well_plate_replicate_assets_present: Literal[False]
    phenotype_transform_receipt_present: Literal[False]
    missing_required_evidence: list[str]
    derived_strain_trait_tables_accepted_as_raw_measurements: Literal[False]

    @model_validator(mode="after")
    def raw_evidence_is_not_satisfied_by_tables(self) -> "PhenotypeProvenanceGate":
        required = {
            "raw_well_or_animal_measurements",
            "plate_and_replicate_identifiers",
            "exclusion_rules_and_excluded_observations",
            "trait_transformation_code_or_receipt",
            "measurement_uncertainty_or_replicate_dispersion",
        }
        if set(self.missing_required_evidence) != required:
            raise ValueError("raw phenotype provenance missing-evidence inventory is incomplete")
        if self.status == "qualified":
            raise ValueError("this frozen audit cannot qualify absent raw phenotype provenance")
        return self


class NullResourceGuard(_StrictModel):
    run_id: Literal["abamectin-ws283-parametric-null-20260721T161005Z-c01179f4a52e"]
    audit_observation: Literal["terminal_success_checksum_qualified"]
    active_run_must_not_be_duplicated: Literal[True]
    heavyweight_derivations_wait_for_terminal_marker: Literal[True]
    heavyweight_derivations_released: Literal[True]
    marker_null_is_not_block_test_calibration: Literal[True]
    terminal_ended_utc: Literal["2026-07-22T17:45:11Z"]
    success_sha256: str
    run_manifest_sha256: str
    aggregation_sha256: str
    rich_summary_sha256: str
    checksum_receipt_sha256: str

    @field_validator(
        "success_sha256",
        "run_manifest_sha256",
        "aggregation_sha256",
        "rich_summary_sha256",
        "checksum_receipt_sha256",
    )
    @classmethod
    def valid_terminal_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("terminal parametric-null calibration evidence hash must be lowercase hexadecimal")
        return value


class RealInputHandoffManifest(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    analysis_id: Literal["abamectin_ws283_regional_state_heldout_prediction_real_input_handoff_20260721"]
    classification: Literal[
        "outcome_independent_execution_path_partial_real_input_qualification"
    ]
    biological_claims_permitted: Literal[False]
    root_ids_are_non_sensitive_logical_names: Literal[True]
    coordinate_identity: CoordinateIdentityGate
    state_panel: StatePanelGate
    ldpruned_chr5_excluded_grm: QualifiedGrmGate
    full_kernels: FullKernelDerivationGate
    population_groups: PopulationGroupGate
    phenotype_provenance: PhenotypeProvenanceGate
    null_resource_guard: NullResourceGuard

    @model_validator(mode="after")
    def cross_gate_boundaries(self) -> "RealInputHandoffManifest":
        roles: list[str] = []
        roles.extend(item.role for item in self.ldpruned_chr5_excluded_grm.files)
        roles.extend(item.role for item in self.full_kernels.input_files)
        roles.extend(item.role for item in self.full_kernels.qualified_output_assets)
        if self.full_kernels.numerical_qualification_receipt is not None:
            roles.append(self.full_kernels.numerical_qualification_receipt.role)
        roles.extend(item.role for item in self.population_groups.source_files)
        roles.extend(item.role for item in self.population_groups.output_assets)
        roles.extend(item.role for item in self.phenotype_provenance.observed_derived_strain_trait_tables)
        roles.append(self.state_panel.observed_small_variant_vcf.role)
        roles.extend(item.role for item in self.state_panel.qualified_panel_assets)
        for reference in (
            self.coordinate_identity.source,
            self.coordinate_identity.target,
        ):
            if reference.fasta is not None:
                roles.append(reference.fasta.role)
            if reference.fai is not None:
                roles.append(reference.fai.role)
        if self.coordinate_identity.qualification_receipt is not None:
            roles.append(self.coordinate_identity.qualification_receipt.role)
        if len(roles) != len(set(roles)):
            raise ValueError("frozen asset roles must be globally unique")
        fam_assets = [
            item for item in self.full_kernels.input_files if item.role == "modern_cohort_fam"
        ]
        group_fam = self.population_groups.source_files[0]
        if len(fam_assets) != 1 or fam_assets[0].sha256 != group_fam.sha256:
            raise ValueError("kernel and grouping gates must bind the same 209-sample FAM")
        if (
            self.ldpruned_chr5_excluded_grm.verification.fam_sha256
            != fam_assets[0].sha256
        ):
            raise ValueError("LD-pruned GRM must bind the same 209-sample FAM")
        if (
            self.ldpruned_chr5_excluded_grm.verification.ordered_iid_sha256
            != self.population_groups.source_ordered_iid_sha256
        ):
            raise ValueError("GRM and grouping gates must bind the same ordered IIDs")
        return self


class VerifiedAsset(_StrictModel):
    role: str
    resolved_path: str
    bytes: int
    sha256: str
    qualified: Literal[True]


def coordinate_identity_ready(gate: CoordinateIdentityGate) -> bool:
    """Return true only for complete, matching interval evidence on both references."""

    if gate.source.status != "qualified" or gate.target.status != "qualified":
        return False
    receipt = gate.qualification_receipt
    if receipt is None or receipt.role != "ws276_ws283_coordinate_identity_receipt":
        return False
    if gate.source.fasta is None or gate.target.fasta is None:
        return False
    if gate.source.fai is None or gate.target.fai is None:
        return False
    if gate.source.fasta.sha256 != gate.target.fasta.sha256:
        return False
    if gate.source.fai.sha256 != gate.target.fai.sha256:
        return False
    if gate.source.chromosome_v_length != gate.target.chromosome_v_length:
        return False
    if (
        gate.source.chromosome_v_canonical_sha256
        != gate.target.chromosome_v_canonical_sha256
    ):
        return False
    source = {item.id: item for item in gate.source.intervals}
    target = {item.id: item for item in gate.target.intervals}
    if list(source) != ["chrv_left", "chrv_right"]:
        return False
    if list(target) != ["chrv_left", "chrv_right"]:
        return False
    for region_id, expected in EXPECTED_REGIONS.items():
        left = source[region_id]
        right = target[region_id]
        if (left.start, left.end) != expected or (right.start, right.end) != expected:
            return False
        if left.canonical_sequence_sha256 != right.canonical_sequence_sha256:
            return False
    return True


def load_handoff_manifest(path: str | Path) -> RealInputHandoffManifest:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"regional-state qualification and held-out prediction handoff manifest is not a file: {source}")
    return RealInputHandoffManifest.model_validate_json(source.read_text(encoding="utf-8"))


def all_frozen_assets(manifest: RealInputHandoffManifest) -> list[FrozenAsset]:
    assets = list(manifest.ldpruned_chr5_excluded_grm.files)
    assets.extend(manifest.full_kernels.input_files)
    assets.extend(manifest.full_kernels.qualified_output_assets)
    if manifest.full_kernels.numerical_qualification_receipt is not None:
        assets.append(manifest.full_kernels.numerical_qualification_receipt)
    assets.extend(manifest.population_groups.source_files)
    assets.extend(manifest.population_groups.output_assets)
    assets.extend(manifest.phenotype_provenance.observed_derived_strain_trait_tables)
    assets.append(manifest.state_panel.observed_small_variant_vcf)
    assets.extend(manifest.state_panel.qualified_panel_assets)
    for reference in (
        manifest.coordinate_identity.source,
        manifest.coordinate_identity.target,
    ):
        if reference.fasta is not None:
            assets.append(reference.fasta)
        if reference.fai is not None:
            assets.append(reference.fai)
    if manifest.coordinate_identity.qualification_receipt is not None:
        assets.append(manifest.coordinate_identity.qualification_receipt)
    return assets


def verify_frozen_asset(asset: FrozenAsset, roots: Mapping[str, str | Path]) -> VerifiedAsset:
    if asset.root_id not in roots:
        raise KeyError(f"no governed root supplied for {asset.root_id}")
    root = Path(roots[asset.root_id]).resolve()
    path = (root / PurePosixPath(asset.relative_path)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"asset escapes governed root: {asset.role}") from error
    if not path.is_file():
        raise FileNotFoundError(f"frozen asset is missing: {asset.role}: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != asset.bytes:
        raise ValueError(
            f"byte count mismatch for {asset.role}: expected {asset.bytes}, got {observed_bytes}"
        )
    observed_hash = _sha256(path)
    if observed_hash != asset.sha256:
        raise ValueError(f"SHA-256 mismatch for {asset.role}")
    return VerifiedAsset(
        role=asset.role,
        resolved_path=str(path),
        bytes=observed_bytes,
        sha256=observed_hash,
        qualified=True,
    )


def verify_frozen_assets(
    manifest: RealInputHandoffManifest,
    roots: Mapping[str, str | Path],
    roles: Sequence[str] | None = None,
) -> list[VerifiedAsset]:
    assets = all_frozen_assets(manifest)
    if roles is not None:
        requested = list(roles)
        index = {item.role: item for item in assets}
        unknown = [role for role in requested if role not in index]
        if unknown:
            raise ValueError(f"unknown frozen asset roles: {unknown}")
        assets = [index[role] for role in requested]
    return [verify_frozen_asset(asset, roots) for asset in assets]


def gate_summary(manifest: RealInputHandoffManifest) -> dict[str, str]:
    return {
        "coordinate_identity": manifest.coordinate_identity.status,
        "state_panel": manifest.state_panel.status,
        "ldpruned_chr5_excluded_grm": manifest.ldpruned_chr5_excluded_grm.status,
        "full_kernels": manifest.full_kernels.status,
        "population_groups": manifest.population_groups.status,
        "phenotype_provenance": manifest.phenotype_provenance.status,
    }


def _load_fam_iids(path: Path) -> list[str]:
    ids: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"FAM line {line_number} has fewer than two fields")
        ids.append(fields[1])
    if len(ids) != len(set(ids)):
        raise ValueError("FAM IIDs are not unique")
    return ids


def _load_pca10(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    expected_header = ["#FID", "IID", *[f"PC{index}" for index in range(1, 11)]]
    if not lines or lines[0].split() != expected_header:
        raise ValueError("PCA header must be #FID IID PC1 ... PC10")
    ids: list[str] = []
    rows: list[list[float]] = []
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split()
        if len(fields) != 12:
            raise ValueError(f"PCA line {line_number} must contain exactly 12 fields")
        ids.append(fields[1])
        try:
            rows.append([float(value) for value in fields[2:]])
        except ValueError as error:
            raise ValueError(f"PCA line {line_number} contains a nonnumeric PC") from error
    if len(ids) != len(set(ids)):
        raise ValueError("PCA IIDs are not unique")
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 10 or not np.all(np.isfinite(values)):
        raise ValueError("PCA matrix must be finite with exactly ten columns")
    return ids, values


def _load_pca_eigenvalues(path: Path) -> np.ndarray:
    """Load the exact ten positive, nonincreasing PCA eigenvalues."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 10:
        raise ValueError("PCA eigenvalue file must contain exactly ten values")
    try:
        values = np.asarray([float(line.strip()) for line in lines], dtype=np.float64)
    except ValueError as error:
        raise ValueError("PCA eigenvalue file contains a nonnumeric value") from error
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("PCA eigenvalues must be finite and positive")
    if np.any(values[1:] > values[:-1]):
        raise ValueError("PCA eigenvalues must be nonincreasing")
    return values


def _canonicalize_labels(assignments: np.ndarray, ids: Sequence[str]) -> np.ndarray:
    clusters = sorted(
        np.unique(assignments).tolist(),
        key=lambda group: min(ids[index] for index in np.where(assignments == group)[0]),
    )
    mapping = {old: new for new, old in enumerate(clusters)}
    return np.asarray([mapping[int(item)] for item in assignments], dtype=np.int64)


def _kmeans(values: np.ndarray, ids: Sequence[str], k: int) -> np.ndarray:
    norms = np.einsum("ij,ij->i", values, values)
    first = min(range(len(ids)), key=lambda index: (float(norms[index]), ids[index]))
    centers = [values[first].copy()]
    chosen = {first}
    while len(centers) < k:
        distances = np.stack(
            [np.einsum("ij,ij->i", values - center, values - center) for center in centers],
            axis=1,
        )
        nearest = np.min(distances, axis=1)
        remaining = [index for index in range(len(ids)) if index not in chosen]
        selected = min(remaining, key=lambda index: (-float(nearest[index]), ids[index]))
        chosen.add(selected)
        centers.append(values[selected].copy())
    center_matrix = np.stack(centers)
    previous: np.ndarray | None = None
    for _ in range(300):
        distances = np.sum((values[:, None, :] - center_matrix[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(distances, axis=1)
        if previous is not None and np.array_equal(assignments, previous):
            break
        if len(np.unique(assignments)) != k:
            raise ValueError(f"deterministic k-means produced an empty cluster for k={k}")
        previous = assignments.copy()
        center_matrix = np.stack(
            [values[assignments == group].mean(axis=0) for group in range(k)]
        )
    else:
        raise ValueError(f"deterministic k-means failed to converge for k={k}")
    return _canonicalize_labels(assignments, ids)


def _silhouette(values: np.ndarray, assignments: np.ndarray) -> float:
    distances = np.sqrt(np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2))
    scores: list[float] = []
    for index, group in enumerate(assignments):
        same = np.where(assignments == group)[0]
        same = same[same != index]
        if len(same) == 0:
            raise ValueError("silhouette is undefined for singleton groups")
        within = float(distances[index, same].mean())
        between = min(
            float(distances[index, assignments == other].mean())
            for other in np.unique(assignments)
            if other != group
        )
        denominator = max(within, between)
        scores.append(0.0 if denominator == 0.0 else (between - within) / denominator)
    return float(np.mean(scores))


def derive_groups_from_matrix(
    ids: Sequence[str],
    values: np.ndarray,
    *,
    candidate_k: Sequence[int] = tuple(range(5, 13)),
    minimum_group_size: int = 5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Derive deterministic groups from genotype PCs without phenotype inputs."""

    ids = list(ids)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(ids), 10):
        raise ValueError("group derivation requires one finite PC1-PC10 row per IID")
    if len(ids) != len(set(ids)) or not np.all(np.isfinite(values)):
        raise ValueError("group derivation requires unique IIDs and finite PCs")
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    if np.any(~np.isfinite(scales)) or np.any(scales <= np.finfo(np.float64).eps):
        raise ValueError("each PC must have nonzero finite population standard deviation")
    standardized = (values - means) / scales
    candidates: list[tuple[float, int, np.ndarray]] = []
    evaluations: list[dict[str, Any]] = []
    for k in candidate_k:
        if k < 2 or k > len(ids):
            raise ValueError("candidate k must be between two and the sample count")
        assignments = _kmeans(standardized, ids, k)
        counts = np.bincount(assignments, minlength=k)
        eligible = bool(np.min(counts) >= minimum_group_size)
        score = _silhouette(standardized, assignments) if eligible else None
        evaluations.append(
            {
                "k": int(k),
                "eligible": eligible,
                "minimum_group_size": int(np.min(counts)),
                "maximum_group_size": int(np.max(counts)),
                "silhouette": score,
            }
        )
        if score is not None:
            candidates.append((score, int(k), assignments))
    if not candidates:
        raise ValueError("no candidate grouping satisfies the minimum group size")
    best = min(candidates, key=lambda item: (-round(item[0], 12), item[1]))
    return best[2], evaluations


def derive_population_groups(
    manifest: RealInputHandoffManifest,
    fam_path: str | Path,
    pca_path: str | Path,
    eigenval_path: str | Path,
) -> tuple[bytes, dict[str, Any]]:
    """Validate frozen group inputs and return assignment TSV bytes plus a receipt."""

    fam = Path(fam_path).resolve()
    pca = Path(pca_path).resolve()
    eigenval = Path(eigenval_path).resolve()
    fam_asset, pca_asset, eigenval_asset = manifest.population_groups.source_files
    for path, asset in (
        (fam, fam_asset),
        (pca, pca_asset),
        (eigenval, eigenval_asset),
    ):
        if not path.is_file() or path.stat().st_size != asset.bytes or _sha256(path) != asset.sha256:
            raise ValueError(f"group source does not match frozen asset {asset.role}")
    fam_ids = _load_fam_iids(fam)
    pca_ids, values = _load_pca10(pca)
    eigenvalues = _load_pca_eigenvalues(eigenval)
    gate = manifest.population_groups
    if len(fam_ids) != gate.expected_samples or pca_ids != fam_ids:
        raise ValueError("PCA IIDs must exactly equal the 209-strain FAM order")
    order_hash = _ordered_id_sha256(fam_ids)
    if order_hash != gate.source_ordered_iid_sha256:
        raise ValueError("ordered IID hash differs from the frozen group contract")
    assignments, evaluations = derive_groups_from_matrix(
        fam_ids,
        values,
        candidate_k=gate.algorithm.candidate_k,
        minimum_group_size=gate.algorithm.minimum_group_size,
    )
    selected_k = int(np.max(assignments)) + 1
    labels = [f"POP{int(group) + 1:02d}" for group in assignments]
    assignment_bytes = (
        "IID\tpopulation_group\n"
        + "".join(f"{iid}\t{label}\n" for iid, label in zip(fam_ids, labels, strict=True))
    ).encode("utf-8")
    receipt = {
        "schema_version": GROUP_RECEIPT_VERSION,
        "algorithm_id": gate.algorithm.id,
        "fam_sha256": fam_asset.sha256,
        "pca10_sha256": pca_asset.sha256,
        "eigenvalues_sha256": eigenval_asset.sha256,
        "eigenvalue_count": int(len(eigenvalues)),
        "eigenvalues_used_for_clustering": False,
        "axis_weighting": gate.algorithm.axis_weighting,
        "eigenvalue_usage": gate.algorithm.eigenvalue_usage,
        "ordered_iid_sha256": order_hash,
        "sample_count": len(fam_ids),
        "selected_k": selected_k,
        "group_sizes": {
            f"POP{group + 1:02d}": int(np.sum(assignments == group))
            for group in range(selected_k)
        },
        "candidate_evaluations": evaluations,
        "assignments_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
        "phenotype_paths_accepted": False,
        "phenotype_values_accessed": False,
        "association_results_accessed": False,
        "outcome_access_claim_scope": gate.outcome_access_claim_scope,
        "operator_outcome_blinding_asserted": gate.operator_outcome_blinding_asserted,
        "labels_are_external_ancestry_assignments": (
            gate.labels_are_external_ancestry_assignments
        ),
        "biological_claims_permitted": False,
    }
    return assignment_bytes, receipt


def verify_population_group_bundle(
    manifest: RealInputHandoffManifest,
    assignments_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Verify a frozen genotype-PC grouping without reading phenotype data."""

    gate = manifest.population_groups
    if gate.status != "qualified":
        raise ValueError("population-group gate is not qualified")
    assignments = Path(assignments_path).resolve()
    receipt = Path(receipt_path).resolve()
    for path, asset in zip(
        (assignments, receipt), gate.output_assets, strict=True
    ):
        if (
            not path.is_file()
            or path.stat().st_size != asset.bytes
            or _sha256(path) != asset.sha256
        ):
            raise ValueError(f"population-group output differs from {asset.role}")

    assignment_bytes = assignments.read_bytes()
    try:
        assignment_text = assignment_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("population-group assignment file must be UTF-8") from error
    lines = assignment_text.splitlines()
    if not lines or lines[0] != "IID\tpopulation_group":
        raise ValueError("population-group assignment header is invalid")
    ids: list[str] = []
    labels: list[str] = []
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        if len(fields) != 2 or not _SAFE_ID_RE.fullmatch(fields[0]):
            raise ValueError(f"population-group assignment line {line_number} is invalid")
        if not re.fullmatch(r"POP[0-9]{2}", fields[1]):
            raise ValueError(f"population-group label on line {line_number} is invalid")
        ids.append(fields[0])
        labels.append(fields[1])
    if len(ids) != gate.expected_samples or len(ids) != len(set(ids)):
        raise ValueError("population-group assignments require 209 unique IIDs")
    if _ordered_id_sha256(ids) != gate.source_ordered_iid_sha256:
        raise ValueError("population-group assignments differ from the frozen IID order")

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("population-group receipt must be a JSON object")
    source_fam, source_pca, source_eigenvalues = gate.source_files
    expected_scalars = {
        "schema_version": GROUP_RECEIPT_VERSION,
        "algorithm_id": gate.algorithm.id,
        "fam_sha256": source_fam.sha256,
        "pca10_sha256": source_pca.sha256,
        "eigenvalues_sha256": source_eigenvalues.sha256,
        "eigenvalue_count": 10,
        "eigenvalues_used_for_clustering": False,
        "axis_weighting": gate.algorithm.axis_weighting,
        "eigenvalue_usage": gate.algorithm.eigenvalue_usage,
        "ordered_iid_sha256": gate.source_ordered_iid_sha256,
        "sample_count": gate.expected_samples,
        "assignments_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
        "phenotype_paths_accepted": False,
        "phenotype_values_accessed": False,
        "association_results_accessed": False,
        "outcome_access_claim_scope": gate.outcome_access_claim_scope,
        "operator_outcome_blinding_asserted": gate.operator_outcome_blinding_asserted,
        "labels_are_external_ancestry_assignments": (
            gate.labels_are_external_ancestry_assignments
        ),
        "biological_claims_permitted": False,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ValueError(f"population-group receipt field differs: {key}")
    selected_k = payload.get("selected_k")
    if not isinstance(selected_k, int) or selected_k not in gate.algorithm.candidate_k:
        raise ValueError("population-group receipt has an invalid selected k")
    expected_labels = [f"POP{index:02d}" for index in range(1, selected_k + 1)]
    observed_sizes = {label: labels.count(label) for label in sorted(set(labels))}
    if list(observed_sizes) != expected_labels:
        raise ValueError("population-group labels are not contiguous")
    if min(observed_sizes.values()) < gate.algorithm.minimum_group_size:
        raise ValueError("population-group assignment violates the minimum group size")
    if payload.get("group_sizes") != observed_sizes:
        raise ValueError("population-group receipt sizes differ from assignments")
    evaluations = payload.get("candidate_evaluations")
    if not isinstance(evaluations, list) or [
        item.get("k") if isinstance(item, dict) else None for item in evaluations
    ] != gate.algorithm.candidate_k:
        raise ValueError("population-group receipt differs from the frozen candidate grid")
    eligible_scores: list[tuple[float, int]] = []
    for item in evaluations:
        minimum_size = item.get("minimum_group_size")
        maximum_size = item.get("maximum_group_size")
        eligible = item.get("eligible")
        score = item.get("silhouette")
        if (
            not isinstance(minimum_size, int)
            or not isinstance(maximum_size, int)
            or minimum_size < 0
            or maximum_size < minimum_size
            or not isinstance(eligible, bool)
            or eligible != (minimum_size >= gate.algorithm.minimum_group_size)
        ):
            raise ValueError("population-group candidate eligibility is inconsistent")
        if eligible:
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
                or not -1.0 <= float(score) <= 1.0
            ):
                raise ValueError("eligible population-group candidate lacks a valid silhouette")
            eligible_scores.append((float(score), int(item["k"])))
        elif score is not None:
            raise ValueError("ineligible population-group candidate must not have a silhouette")
    if not eligible_scores:
        raise ValueError("population-group receipt has no eligible candidate")
    receipt_best_k = min(
        eligible_scores, key=lambda item: (-round(item[0], 12), item[1])
    )[1]
    if selected_k != receipt_best_k:
        raise ValueError("population-group receipt did not select the frozen optimum")
    return payload


def write_population_group_bundle(
    manifest: RealInputHandoffManifest,
    fam_path: str | Path,
    pca_path: str | Path,
    eigenval_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Atomically publish a write-once, genotype-PC population-group bundle."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"population-group output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        assignment_bytes, receipt = derive_population_groups(
            manifest, fam_path, pca_path, eigenval_path
        )
        assignment_path = temporary / "population_groups.tsv"
        receipt_path = temporary / "population_groups_receipt.json"
        assignment_path.write_bytes(assignment_bytes)
        receipt_path.write_bytes(_canonical_json(receipt))
        inventory = {
            "population_groups.tsv": _sha256(assignment_path),
            "population_groups_receipt.json": _sha256(receipt_path),
        }
        (temporary / "MANIFEST.json").write_bytes(_canonical_json(inventory))
        (temporary / "SUCCESS").write_text("qualified\n", encoding="utf-8")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def _kernel_source_paths(
    manifest: RealInputHandoffManifest,
    fam_path: str | Path,
    bim_path: str | Path,
) -> tuple[Path, Path, list[tuple[str, str]], list[tuple[int, str]]]:
    """Verify the exact frozen FAM/BIM inputs and return their ordered records."""

    fam = Path(fam_path).resolve()
    bim = Path(bim_path).resolve()
    by_role = {item.role: item for item in manifest.full_kernels.input_files}
    for path, role in ((fam, "modern_cohort_fam"), (bim, "modern_cohort_bim")):
        asset = by_role[role]
        if not path.is_file() or path.stat().st_size != asset.bytes or _sha256(path) != asset.sha256:
            raise ValueError(f"full-kernel source differs from frozen asset {role}")

    fam_rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(fam.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"FAM line {line_number} has fewer than two fields")
        fam_rows.append((fields[0], fields[1]))
    if len(fam_rows) != manifest.full_kernels.sample_count:
        raise ValueError("full-kernel FAM does not contain exactly 209 samples")
    if len(fam_rows) != len(set(fam_rows)) or len({iid for _, iid in fam_rows}) != len(fam_rows):
        raise ValueError("full-kernel FAM identifiers are not unique")
    if _ordered_id_sha256([iid for _, iid in fam_rows]) != (
        manifest.population_groups.source_ordered_iid_sha256
    ):
        raise ValueError("full-kernel FAM IID order differs from the frozen shared order")

    markers: list[tuple[int, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(bim.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"BIM line {line_number} has fewer than two fields")
        try:
            chromosome = int(fields[0])
        except ValueError as error:
            raise ValueError(f"BIM line {line_number} chromosome is not numeric") from error
        marker = fields[1]
        if chromosome not in range(1, 7) or not marker or marker in seen:
            raise ValueError(f"BIM line {line_number} has an invalid chromosome or marker ID")
        seen.add(marker)
        markers.append((chromosome, marker))
    gate = manifest.full_kernels
    if len(markers) != gate.total_marker_count:
        raise ValueError("BIM total marker count differs from the frozen contract")
    chromosome_v = sum(chromosome == gate.chromosome_v_code for chromosome, _ in markers)
    if chromosome_v != gate.chromosome_v_marker_count:
        raise ValueError("BIM chromosome-V marker count differs from the frozen contract")
    if len(markers) - chromosome_v != gate.non_chromosome_v_marker_count:
        raise ValueError("BIM non-chromosome-V marker count differs from the frozen contract")
    return fam, bim, fam_rows, markers


def prepare_full_kernel_markers(
    manifest_path: str | Path,
    fam_path: str | Path,
    bim_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Atomically emit the exact whole/ex-chrV marker lists used by GCTA."""

    source = Path(manifest_path).resolve()
    manifest = load_handoff_manifest(source)
    if manifest.full_kernels.status not in {"ready_for_remote_derivation", "qualified"}:
        raise ValueError("full-kernel manifest is not released for remote derivation")
    fam, bim, fam_rows, markers = _kernel_source_paths(manifest, fam_path, bim_path)
    gate = manifest.full_kernels
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"full-kernel marker output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        selections = {
            "whole_genome.snplist": [marker for _, marker in markers],
            "genome_excluding_chrv.snplist": [
                marker for chromosome, marker in markers if chromosome != gate.chromosome_v_code
            ],
            "chromosome_v.snplist": [
                marker for chromosome, marker in markers if chromosome == gate.chromosome_v_code
            ],
        }
        expected_counts = {
            "whole_genome.snplist": gate.total_marker_count,
            "genome_excluding_chrv.snplist": gate.non_chromosome_v_marker_count,
            "chromosome_v.snplist": gate.chromosome_v_marker_count,
        }
        marker_files: list[dict[str, Any]] = []
        for name, selected in selections.items():
            if len(selected) != expected_counts[name]:
                raise ValueError(f"derived marker count differs for {name}")
            destination = temporary / name
            destination.write_text("".join(f"{marker}\n" for marker in selected), encoding="utf-8")
            marker_files.append(
                {
                    "name": name,
                    "markers": len(selected),
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )
        receipt = {
            "schema_version": KERNEL_MARKER_RECEIPT_VERSION,
            "handoff_manifest_sha256": _sha256(source),
            "fam_sha256": _sha256(fam),
            "bim_sha256": _sha256(bim),
            "samples": len(fam_rows),
            "ordered_iid_sha256": _ordered_id_sha256([iid for _, iid in fam_rows]),
            "ordered_fid_iid_sha256": hashlib.sha256(
                "".join(f"{fid}\t{iid}\n" for fid, iid in fam_rows).encode("utf-8")
            ).hexdigest(),
            "selection_rule": gate.marker_selection_rule,
            "chromosome_v_code": gate.chromosome_v_code,
            "marker_files": marker_files,
        }
        (temporary / "marker_manifest.json").write_bytes(_canonical_json(receipt))
        inventory = {
            item["name"]: item["sha256"] for item in marker_files
        }
        inventory["marker_manifest.json"] = _sha256(temporary / "marker_manifest.json")
        (temporary / "MANIFEST.json").write_bytes(_canonical_json(inventory))
        (temporary / "SUCCESS").write_text("qualified\n", encoding="utf-8")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def verify_null_resource_release(
    manifest_path: str | Path,
    null_run_path: str | Path,
) -> dict[str, Any]:
    """Bind the heavyweight gate to the exact terminal parametric-null calibration evidence bundle."""

    source = Path(manifest_path).resolve()
    manifest = load_handoff_manifest(source)
    guard = manifest.null_resource_guard
    run = Path(null_run_path).resolve()
    if not run.is_dir() or run.name != guard.run_id:
        raise ValueError("parametric-null calibration release path does not identify the frozen null run")
    if (run / "FAILURE").exists() or (run / "SUCCESS").read_bytes() != b"SUCCESS\n":
        raise ValueError("parametric-null calibration release requires exactly the canonical SUCCESS marker")
    expected = {
        "SUCCESS": guard.success_sha256,
        "receipts/run_manifest.json": guard.run_manifest_sha256,
        "summary/parametric_null_aggregation.json": guard.aggregation_sha256,
        "summary/ws283_parametric_null/ws283_parametric_null_summary.json": (
            guard.rich_summary_sha256
        ),
        "receipts/SHA256SUMS.txt": guard.checksum_receipt_sha256,
    }
    observed: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        path = (run / PurePosixPath(relative)).resolve()
        try:
            path.relative_to(run)
        except ValueError as error:
            raise ValueError("parametric-null calibration release asset escapes the frozen run") from error
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"parametric-null calibration release evidence differs: {relative}")
        observed[relative] = expected_hash
    ended = (run / "receipts/ended_utc.txt").read_text(encoding="utf-8").strip()
    if ended != guard.terminal_ended_utc:
        raise ValueError("parametric-null calibration terminal timestamp differs from the frozen release")
    return {
        "schema_version": NULL_RELEASE_RECEIPT_VERSION,
        "handoff_manifest_sha256": _sha256(source),
        "run_id": guard.run_id,
        "terminal_ended_utc": ended,
        "files": observed,
        "terminal_success": True,
        "full_checksum_closure_must_be_verified_by_runner": True,
        "heavyweight_derivations_released": True,
    }


def _load_marker_manifest(
    manifest: RealInputHandoffManifest,
    manifest_path: Path,
    fam_path: Path,
    bim_path: Path,
    marker_root: Path,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    fam, bim, fam_rows, markers = _kernel_source_paths(manifest, fam_path, bim_path)
    receipt_path = marker_root / "marker_manifest.json"
    if not receipt_path.is_file() or (marker_root / "SUCCESS").read_bytes() != b"qualified\n":
        raise ValueError("full-kernel marker bundle is incomplete")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != KERNEL_MARKER_RECEIPT_VERSION:
        raise ValueError("full-kernel marker manifest schema differs")
    expected_fixed = {
        "handoff_manifest_sha256": _sha256(manifest_path),
        "fam_sha256": _sha256(fam),
        "bim_sha256": _sha256(bim),
        "samples": len(fam_rows),
        "ordered_iid_sha256": _ordered_id_sha256([iid for _, iid in fam_rows]),
        "selection_rule": manifest.full_kernels.marker_selection_rule,
        "chromosome_v_code": manifest.full_kernels.chromosome_v_code,
    }
    for key, value in expected_fixed.items():
        if payload.get(key) != value:
            raise ValueError(f"full-kernel marker manifest differs: {key}")
    expected_lists = {
        "whole_genome.snplist": [marker for _, marker in markers],
        "genome_excluding_chrv.snplist": [
            marker
            for chromosome, marker in markers
            if chromosome != manifest.full_kernels.chromosome_v_code
        ],
        "chromosome_v.snplist": [
            marker
            for chromosome, marker in markers
            if chromosome == manifest.full_kernels.chromosome_v_code
        ],
    }
    observed_files = payload.get("marker_files")
    if not isinstance(observed_files, list) or len(observed_files) != 3:
        raise ValueError("full-kernel marker manifest has the wrong file inventory")
    by_name = {
        item.get("name"): item for item in observed_files if isinstance(item, dict)
    }
    if set(by_name) != set(expected_lists):
        raise ValueError("full-kernel marker manifest names differ")
    for name, expected_markers in expected_lists.items():
        path = marker_root / name
        expected_bytes = "".join(f"{marker}\n" for marker in expected_markers).encode("utf-8")
        if not path.is_file() or path.read_bytes() != expected_bytes:
            raise ValueError(f"full-kernel marker list content differs: {name}")
        item = by_name[name]
        if (
            item.get("markers") != len(expected_markers)
            or item.get("bytes") != len(expected_bytes)
            or item.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"full-kernel marker list receipt differs: {name}")
    return payload, fam_rows


def _qualify_gcta_kernel(
    prefix: Path,
    fam_rows: Sequence[tuple[str, str]],
    expected_markers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    paths = {
        "grm_bin": Path(f"{prefix}.grm.bin"),
        "grm_n_bin": Path(f"{prefix}.grm.N.bin"),
        "grm_id": Path(f"{prefix}.grm.id"),
        "gcta_log": Path(f"{prefix}.log"),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"GCTA kernel output is incomplete: {prefix.name}")
    ids: list[tuple[str, str]] = []
    for line_number, line in enumerate(paths["grm_id"].read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"GRM ID line {line_number} must contain FID and IID")
        ids.append((fields[0], fields[1]))
    if ids != list(fam_rows):
        raise ValueError(f"GRM IDs/order differ from the frozen FAM: {prefix.name}")
    entries = len(fam_rows) * (len(fam_rows) + 1) // 2
    grm_values = np.fromfile(paths["grm_bin"], dtype="<f4")
    marker_counts = np.fromfile(paths["grm_n_bin"], dtype="<f4")
    if grm_values.shape != (entries,) or marker_counts.shape != (entries,):
        raise ValueError(f"GRM triangular binary size differs: {prefix.name}")
    if not np.all(np.isfinite(grm_values)) or not np.all(np.isfinite(marker_counts)):
        raise ValueError(f"GRM triangular binary contains non-finite values: {prefix.name}")
    if not np.all(np.isclose(marker_counts, expected_markers, rtol=0.0, atol=1e-3)):
        raise ValueError(f"GRM pairwise marker counts differ: {prefix.name}")
    matrix = np.empty((len(fam_rows), len(fam_rows)), dtype=np.float64)
    lower = np.tril_indices(len(fam_rows))
    matrix[lower] = grm_values
    matrix[(lower[1], lower[0])] = grm_values
    if matrix.shape != (209, 209) or not np.array_equal(matrix, matrix.T):
        raise ValueError(f"GRM matrix is not exactly symmetric 209x209: {prefix.name}")
    if not np.all(np.isfinite(matrix)) or np.any(np.diag(matrix) <= 0.0):
        raise ValueError(f"GRM matrix is non-finite or has non-positive diagonal: {prefix.name}")
    eigenvalues = np.linalg.eigvalsh(matrix)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = max(1e-5, spectral_scale * 1e-5)
    minimum = float(eigenvalues[0])
    if minimum < -tolerance:
        raise ValueError(f"GRM is not PSD within float32 tolerance: {prefix.name}")
    receipt = {
        "samples": len(fam_rows),
        "relationship_markers": expected_markers,
        "triangular_float32_entries": entries,
        "pairwise_marker_count_min": int(np.min(marker_counts)),
        "pairwise_marker_count_max": int(np.max(marker_counts)),
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "psd_tolerance": tolerance,
        "positive_semidefinite_within_float32_tolerance": True,
        "diagonal_min": float(np.min(np.diag(matrix))),
        "diagonal_max": float(np.max(np.diag(matrix))),
        "files": {
            role: {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for role, path in paths.items()
        },
    }
    return matrix, receipt


def qualify_full_kernels(
    manifest_path: str | Path,
    fam_path: str | Path,
    bim_path: str | Path,
    marker_dir: str | Path,
    grm_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Qualify two GCTA triples and atomically emit symmetric float64 NPY kernels."""

    source = Path(manifest_path).resolve()
    manifest = load_handoff_manifest(source)
    fam = Path(fam_path).resolve()
    bim = Path(bim_path).resolve()
    marker_root = Path(marker_dir).resolve()
    grm_root = Path(grm_dir).resolve()
    if not marker_root.is_dir() or not grm_root.is_dir():
        raise FileNotFoundError("marker or GCTA GRM root is missing")
    marker_receipt, fam_rows = _load_marker_manifest(
        manifest, source, fam, bim, marker_root
    )
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"full-kernel qualification output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        results: list[dict[str, Any]] = []
        for contract in manifest.full_kernels.outputs:
            prefix = grm_root / contract.id
            matrix, qualification = _qualify_gcta_kernel(
                prefix, fam_rows, contract.marker_count
            )
            matrix_path = temporary / f"{contract.id}.npy"
            np.save(matrix_path, matrix.astype("<f8", copy=False), allow_pickle=False)
            reloaded = np.load(matrix_path, allow_pickle=False)
            if (
                reloaded.shape != (209, 209)
                or reloaded.dtype != np.dtype("<f8")
                or not np.array_equal(reloaded, matrix)
                or not np.array_equal(reloaded, reloaded.T)
                or not np.all(np.isfinite(reloaded))
            ):
                raise ValueError(f"saved NPY kernel failed round-trip: {contract.id}")
            qualification.update(
                {
                    "id": contract.id,
                    "npy": {
                        "name": matrix_path.name,
                        "bytes": matrix_path.stat().st_size,
                        "sha256": _sha256(matrix_path),
                        "matrix_payload_sha256": hashlib.sha256(
                            reloaded.astype("<f8", copy=False).tobytes(order="C")
                        ).hexdigest(),
                        "dtype": reloaded.dtype.str,
                        "shape": list(reloaded.shape),
                        "finite": True,
                        "exactly_symmetric": True,
                    },
                }
            )
            results.append(qualification)
        receipt = {
            "schema_version": KERNEL_QUALIFICATION_VERSION,
            "handoff_manifest_sha256": _sha256(source),
            "marker_manifest_sha256": _sha256(marker_root / "marker_manifest.json"),
            "fam_sha256": _sha256(fam),
            "bim_sha256": _sha256(bim),
            "ordered_iid_sha256": _ordered_id_sha256([iid for _, iid in fam_rows]),
            "ordered_fid_iid_sha256": marker_receipt["ordered_fid_iid_sha256"],
            "samples": len(fam_rows),
            "kernels": results,
            "biological_claims_permitted": False,
        }
        receipt_path = temporary / "qualification_receipt.json"
        receipt_path.write_bytes(_canonical_json(receipt))
        inventory = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        (temporary / "MANIFEST.json").write_bytes(_canonical_json(inventory))
        (temporary / "SUCCESS").write_text("qualified\n", encoding="utf-8")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def _parse_roots(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("roots must be supplied as root_id=/absolute/path")
        root_id, path = value.split("=", 1)
        if not _SAFE_ID_RE.fullmatch(root_id) or root_id in roots:
            raise ValueError(f"invalid or duplicate root id: {root_id}")
        roots[root_id] = Path(path)
    return roots


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    verify = subparsers.add_parser("verify-assets")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--root", action="append", default=[], metavar="ID=PATH")
    verify.add_argument("--role", action="append", default=None)
    groups = subparsers.add_parser("derive-groups")
    groups.add_argument("--manifest", required=True)
    groups.add_argument("--fam", required=True)
    groups.add_argument("--pca", required=True)
    groups.add_argument("--eigenval", required=True)
    groups.add_argument("--output-dir", required=True)
    null_release = subparsers.add_parser("verify-null-release")
    null_release.add_argument("--manifest", required=True)
    null_release.add_argument("--null-run", required=True)
    markers = subparsers.add_parser("prepare-kernel-markers")
    markers.add_argument("--manifest", required=True)
    markers.add_argument("--fam", required=True)
    markers.add_argument("--bim", required=True)
    markers.add_argument("--output-dir", required=True)
    kernels = subparsers.add_parser("qualify-kernels")
    kernels.add_argument("--manifest", required=True)
    kernels.add_argument("--fam", required=True)
    kernels.add_argument("--bim", required=True)
    kernels.add_argument("--marker-dir", required=True)
    kernels.add_argument("--grm-dir", required=True)
    kernels.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = load_handoff_manifest(args.manifest)
    if args.command == "validate-manifest":
        print(json.dumps(gate_summary(manifest), sort_keys=True, indent=2))
    elif args.command == "verify-assets":
        results = verify_frozen_assets(manifest, _parse_roots(args.root), args.role)
        print(json.dumps([item.model_dump(mode="json") for item in results], sort_keys=True, indent=2))
    elif args.command == "derive-groups":
        receipt = write_population_group_bundle(
            manifest, args.fam, args.pca, args.eigenval, args.output_dir
        )
        print(json.dumps(receipt, sort_keys=True, indent=2))
    elif args.command == "verify-null-release":
        receipt = verify_null_resource_release(args.manifest, args.null_run)
        print(json.dumps(receipt, sort_keys=True, indent=2))
    elif args.command == "prepare-kernel-markers":
        receipt = prepare_full_kernel_markers(
            args.manifest, args.fam, args.bim, args.output_dir
        )
        print(json.dumps(receipt, sort_keys=True, indent=2))
    elif args.command == "qualify-kernels":
        receipt = qualify_full_kernels(
            args.manifest,
            args.fam,
            args.bim,
            args.marker_dir,
            args.grm_dir,
            args.output_dir,
        )
        print(json.dumps(receipt, sort_keys=True, indent=2))
    else:  # pragma: no cover - argparse enforces the subcommands
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
