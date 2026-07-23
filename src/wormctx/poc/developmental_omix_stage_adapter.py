"""Outcome-blind semantic-stage adapter for governed OMIX709 sources.

This module is deliberately narrower than a tensorizer.  It derives exact
26- and 200-cell semantic frontiers from identity/time-only raw reference
controls, freezes parent/child lineage edges, and records modality-local
identifier maps from the processed tables.  It never decodes raw-control
outcome columns, perturbation measurements, or any S4 outcome value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import developmental_omix_scaleup as scaleup
from .developmental_omix import (
    SHA256_RE,
    _canonical_bytes,
    _sha256,
    _write_bytes_atomic,
    _write_json_atomic,
)


SCHEMA_VERSION = "wormctx-omix709-semantic-stage-adapter-contract-1.1"
STAGE_MANIFEST_VERSION = "wormctx-omix709-semantic-stage-membership-1.0"
MAPPING_MANIFEST_VERSION = "wormctx-omix709-modality-local-mappings-1.0"
LINEAGE_MANIFEST_VERSION = "wormctx-omix709-semantic-lineage-1.0"
RAW_CONTROL_RECEIPT_VERSION = "wormctx-omix709-raw-control-lineage-source-receipt-1.0"
OVERLAP_RECEIPT_VERSION = "wormctx-omix709-pilot-overlap-receipt-1.0"
PILOT_NORMALIZED_BYTES = 62315263
PILOT_NORMALIZED_SHA256 = "51fcda2051bb0d7cff376581e29e80be20da013211b4c40cb16a2186b7039659"
PILOT_EMBRYOS = 251
OVERLAP_ATTESTATION_REQUEST_SHA256 = (
    "c7cf62c22fbeef0781ec9ea55d7cec0f001eac749052673e66d1a48744b43eaf"
)
ADAPTER_QUALIFICATION_VERSION = "wormctx-omix709-stage-adapter-qualification-1.0"
ADAPTER_ANALYSIS_VERSION = "wormctx-omix709-stage-adapter-analysis-manifest-1.0"
VERIFY_VERSION = "wormctx-omix709-stage-adapter-verification-1.0"

OUTPUT_FILES = {
    "analysis_manifest.json",
    "lineage_manifest.json",
    "modality_mappings.json",
    "overlap_status.json",
    "qualification.json",
    "raw_control_source_receipt.json",
    "stage_membership.json",
}

EXPECTED_MODALITIES = list(scaleup.MODALITY_IDS)
EXPECTED_QUALIFIED_S4 = ["1", "2", "3", "5", "6"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ParentQualificationContract(_StrictModel):
    manifest_filename: str
    manifest_sha256: str
    schema_version: str
    required_status: str
    required_model_launch_status: str

    @field_validator("manifest_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("parent manifest SHA-256 must be lowercase hexadecimal")
        return value


class RightsContract(_StrictModel):
    permitted_use: str
    operator_attestation_required: bool
    source_redistribution_permitted: bool
    public_training_permitted: bool
    commercial_use_permitted: bool
    attestation_text: str


class SupportContract(_StrictModel):
    minimum_reference_count: int = Field(gt=0)
    minimum_reference_fraction: float = Field(gt=0.0, le=1.0)
    missing_tokens: list[str]
    unsupported_value_policy: str
    perturbation_missing_value_policy: str
    structural_missing_policy: str
    downstream_imputation_policy: str

    @model_validator(mode="after")
    def exact_policy(self) -> "SupportContract":
        if self.unsupported_value_policy != "masked_not_zero":
            raise ValueError("unsupported reference cells must be masked, not zeroed")
        if self.perturbation_missing_value_policy != "explicit_mask_no_adapter_imputation":
            raise ValueError("the stage adapter may not impute perturbation values")
        if self.structural_missing_policy != "modality_local_absence_recorded":
            raise ValueError("structural absence must be modality-local and explicit")
        if self.downstream_imputation_policy != "fit_on_training_genes_only":
            raise ValueError("downstream imputation must be fit on training genes only")
        if len(self.missing_tokens) != len(set(self.missing_tokens)):
            raise ValueError("missing tokens must be unique")
        return self


class RawArchiveIdentity(_StrictModel):
    filename: str
    bytes: int = Field(gt=0)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("raw-control archive SHA-256 must be lowercase hexadecimal")
        return value


class RawExtractedInventoryIdentity(_StrictModel):
    directory_name: str
    member_filename_regex: str
    member_sequence_start: int = Field(ge=0)
    expected_file_count: int = Field(gt=0)
    expected_total_bytes: int = Field(gt=0)
    inventory_canonicalization: str
    inventory_sha256: str

    @field_validator("inventory_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("raw-control inventory SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def exact_inventory_policy(self) -> "RawExtractedInventoryIdentity":
        if self.directory_name != "Raw_data_Control":
            raise ValueError("raw-control extracted directory identity differs")
        if self.member_filename_regex != (
            r"ctr_emb([1-9]|[1-9][0-9]|10[0-5])_raw_data\.txt"
        ):
            raise ValueError("raw-control member filename grammar differs")
        if self.member_sequence_start != 1:
            raise ValueError("raw-control member sequence must start at one")
        if self.inventory_canonicalization != (
            "sha256_canonical_json_sorted_name_bytes_sha256"
        ):
            raise ValueError("raw-control inventory canonicalization differs")
        return self


class PermittedRawColumn(_StrictModel):
    semantic_role: str
    source_header: str
    zero_based_index: int = Field(ge=0)
    value_type: str


class RawIdentityTimeAccess(_StrictModel):
    header_sha256: str
    header_field_count: int = Field(gt=0)
    permitted_columns: list[PermittedRawColumn]
    nonpermitted_value_columns_policy: str
    relative_birth_time_anchor: str

    @field_validator("header_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("raw-control header SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def exact_access_policy(self) -> "RawIdentityTimeAccess":
        expected = [
            PermittedRawColumn(
                semantic_role="time",
                source_header="time",
                zero_based_index=1,
                value_type="nonnegative_integer",
            ),
            PermittedRawColumn(
                semantic_role="cell_id",
                source_header="cell_name",
                zero_based_index=2,
                value_type="ascii_semantic_cell_identifier",
            ),
        ]
        if self.header_field_count != 9 or self.permitted_columns != expected:
            raise ValueError("all and only raw-control time/cell-ID columns must be permitted")
        if self.nonpermitted_value_columns_policy != "never_decode":
            raise ValueError("raw-control outcome columns must never be decoded")
        if self.relative_birth_time_anchor != "min_last_ABa_ABp_plus_one_per_control":
            raise ValueError("raw-control relative-birth-time anchor differs")
        return self


class RawControlLineageSourceContract(_StrictModel):
    archive: RawArchiveIdentity
    extracted: RawExtractedInventoryIdentity
    identity_time_access: RawIdentityTimeAccess
    expected_union_cell_count: int = Field(gt=0)
    expected_binary_parent_count: int = Field(gt=0)
    expected_supported_division_events: int = Field(gt=0)
    expected_unsupported_division_events: int = Field(ge=0)
    expected_sibling_birth_time_disagreements: int = Field(ge=0)
    expected_input_event_count: int = Field(gt=0)
    expected_endpoint_event_count: int = Field(gt=0)


class LineageContract(_StrictModel):
    initial_frontier: list[str]
    virtual_root: str
    explicit_parent_by_child: dict[str, str]
    suffix_parent_characters: str
    suffix_parent_roots: list[str]
    daughter_suffix_pairs: list[str]
    daughter_count: int = Field(gt=0)
    unknown_identifier_policy: str
    event_order: str
    tie_breaker: str

    @model_validator(mode="after")
    def exact_policy(self) -> "LineageContract":
        if self.initial_frontier != ["AB", "P1"] or self.virtual_root != "P0":
            raise ValueError("the processed adapter must begin from the AB/P1 frontier")
        if self.daughter_count != 2:
            raise ValueError("the early embryonic lineage contract is binary")
        if self.suffix_parent_characters != "aplrdv":
            raise ValueError("lineage suffix grammar must include a/p, l/r, and d/v")
        if self.daughter_suffix_pairs != ["ap", "lr", "dv"]:
            raise ValueError("lineage daughter suffix pairs differ")
        if self.unknown_identifier_policy != "fail_closed":
            raise ValueError("unknown lineage identities must fail closed")
        if self.event_order != "raw_control_median_relative_daughter_birth_time":
            raise ValueError("stage events must use raw-control relative daughter birth times")
        if self.tie_breaker != "semantic_cell_id_codepoint_order":
            raise ValueError("stage timing ties require the frozen semantic-ID tie breaker")
        if len(set(self.initial_frontier)) != 2:
            raise ValueError("initial frontier identities must be unique")
        return self


class LandmarkContract(_StrictModel):
    input_cell_count: int = Field(gt=1)
    endpoint_cell_count: int = Field(gt=1)
    input_membership: str
    endpoint_membership: str
    input_feature_event_scope: str
    endpoint_primary_modality: str
    exact_frontier_required: bool

    @model_validator(mode="after")
    def exact_landmarks(self) -> "LandmarkContract":
        if (self.input_cell_count, self.endpoint_cell_count) != (26, 200):
            raise ValueError("this contract freezes the 26-cell input and 200-cell endpoint")
        if self.input_membership != "control_derived_exact_semantic_frontier":
            raise ValueError("input membership policy differs")
        if self.endpoint_membership != "control_derived_exact_semantic_frontier":
            raise ValueError("endpoint membership policy differs")
        if self.input_feature_event_scope != "cells_born_by_26_frontier_inclusive":
            raise ValueError("input event scope differs")
        if self.endpoint_primary_modality != "cnd1_gfp_expression":
            raise ValueError("the primary endpoint must remain continuous CND-1")
        if not self.exact_frontier_required:
            raise ValueError("both stage frontiers must contain exactly the requested count")
        return self


class ModalityPolicy(_StrictModel):
    id: str
    input_scope: str
    endpoint_eligible: bool


class SourceAccessContract(_StrictModel):
    reference_values_permitted: list[str]
    s3_access: str
    s4_access: str
    qualified_s4_sheets: list[str]
    excluded_s4_sheets: list[str]
    descriptive_s4_sheets: list[str]
    outcome_prevalence_inspection_permitted: bool

    @model_validator(mode="after")
    def outcome_blind(self) -> "SourceAccessContract":
        if self.reference_values_permitted != list(scaleup.MODALITY_IDS):
            raise ValueError("all and only S2 reference modalities must be declared")
        if self.s3_access != "identifiers_headers_and_semantic_row_labels_only":
            raise ValueError("S3 perturbation values may not be decoded")
        if self.s4_access != "qualified_headers_and_semantic_row_labels_only_no_values":
            raise ValueError("S4 outcome values may not be decoded")
        if self.qualified_s4_sheets != EXPECTED_QUALIFIED_S4:
            raise ValueError("qualified S4 sheet set differs")
        if self.excluded_s4_sheets != ["4"] or self.descriptive_s4_sheets != ["7", "8"]:
            raise ValueError("S4 sheet 4 must remain excluded and 7/8 descriptive-only")
        if self.outcome_prevalence_inspection_permitted:
            raise ValueError("stage qualification may not inspect outcome prevalence")
        return self


class OverlapContract(_StrictModel):
    receipt_schema_version: str
    receipt_optional_for_adapter_freeze: bool
    receipt_required_for_model_launch: bool
    scaleup_join_key: str
    pilot_join_key: str
    mapping_basis: str
    exposed_partition: str
    claim_bearing_overlap_permitted: bool


class StageAdapterContract(_StrictModel):
    schema_version: str
    analysis_id: str
    classification: str
    validated: bool
    biological_claims_permitted: bool
    parent: ParentQualificationContract
    rights: RightsContract
    support: SupportContract
    raw_control_lineage_source: RawControlLineageSourceContract
    lineage: LineageContract
    landmarks: LandmarkContract
    modalities: list[ModalityPolicy]
    source_access: SourceAccessContract
    pilot_overlap: OverlapContract

    @model_validator(mode="after")
    def exact_contract(self) -> "StageAdapterContract":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unknown semantic-stage adapter contract")
        if self.validated or self.biological_claims_permitted:
            raise ValueError("adapter qualification cannot assert validation or biology")
        if self.parent.schema_version != scaleup.QUALIFICATION_VERSION:
            raise ValueError("parent qualification schema differs")
        if self.parent.required_status != "qualified_with_exclusions":
            raise ValueError("parent qualification status differs")
        if self.parent.required_model_launch_status != (
            "blocked_pending_semantic_cell_stage_adapter_freeze"
        ):
            raise ValueError("parent launch status differs")
        if [item.id for item in self.modalities] != EXPECTED_MODALITIES:
            raise ValueError("modality order or identity differs")
        if any(
            (
                self.rights.source_redistribution_permitted,
                self.rights.public_training_permitted,
                self.rights.commercial_use_permitted,
            )
        ):
            raise ValueError("rights contract is broader than the governed source")
        if not self.rights.operator_attestation_required:
            raise ValueError("controlled-use attestation must be required")
        overlap = self.pilot_overlap
        if (
            overlap.receipt_schema_version != OVERLAP_RECEIPT_VERSION
            or not overlap.receipt_optional_for_adapter_freeze
            or not overlap.receipt_required_for_model_launch
            or overlap.scaleup_join_key != "embryo_id"
            or overlap.pilot_join_key != "normalized_embryo_id"
            or overlap.mapping_basis != "source_supported_exact_identity_receipt"
            or overlap.exposed_partition != "development_only_exposed_pilot"
            or overlap.claim_bearing_overlap_permitted
        ):
            raise ValueError("pilot-overlap policy differs from the frozen contract")
        return self


class OverlapPair(_StrictModel):
    normalized_embryo_id: str
    scaleup_embryo_id: str
    source_member: str


class PilotOverlapAssertions(_StrictModel):
    table_s3_uses_only_omix709_05_54_embryos: bool
    omix709_05_54_and_05_55_are_biological_embryo_disjoint: bool
    identifier_namespaces_are_complete_and_not_reused_across_archives: bool
    any_reused_embryos_are_exhaustively_listed_in_overlap_pairs: bool

    @model_validator(mode="after")
    def all_attested(self) -> "PilotOverlapAssertions":
        if (
            not self.table_s3_uses_only_omix709_05_54_embryos
            or not self.identifier_namespaces_are_complete_and_not_reused_across_archives
            or not self.any_reused_embryos_are_exhaustively_listed_in_overlap_pairs
        ):
            raise ValueError("source authority did not establish complete overlap evidence")
        return self


class PilotOverlapReceipt(_StrictModel):
    schema_version: str
    pilot_normalized_sha256: str
    pilot_normalized_bytes: int = Field(gt=0)
    pilot_embryos: int = Field(gt=0)
    scaleup_source_bundle_sha256: str
    attestation_request_sha256: str
    source_evidence_record_identifier: str = Field(min_length=1)
    source_evidence_record_bytes: int = Field(gt=0)
    source_evidence_record_sha256: str
    attestor_name: str = Field(min_length=1)
    attestor_role_and_source_authority: str = Field(min_length=1)
    attested_utc: str
    assertions: PilotOverlapAssertions
    mapping_basis: str
    mapping_complete: bool
    outcome_values_used_for_identity_decision: bool
    pairs: list[OverlapPair]

    @field_validator(
        "pilot_normalized_sha256",
        "scaleup_source_bundle_sha256",
        "attestation_request_sha256",
        "source_evidence_record_sha256",
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("overlap receipt hashes must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def exact_receipt(self) -> "PilotOverlapReceipt":
        if self.schema_version != OVERLAP_RECEIPT_VERSION:
            raise ValueError("unknown pilot-overlap receipt schema")
        if self.mapping_basis != "source_supported_exact_identity_receipt":
            raise ValueError("pilot overlap must use an exact source-supported mapping")
        if (
            self.pilot_normalized_sha256 != PILOT_NORMALIZED_SHA256
            or self.pilot_normalized_bytes != PILOT_NORMALIZED_BYTES
            or self.pilot_embryos != PILOT_EMBRYOS
        ):
            raise ValueError("pilot overlap receipt is bound to a different normalized pilot")
        if self.attestation_request_sha256 != OVERLAP_ATTESTATION_REQUEST_SHA256:
            raise ValueError("pilot overlap receipt is bound to a different attestation request")
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", self.attested_utc) is None:
            raise ValueError("pilot overlap attestation time must be UTC ISO-8601")
        if self.outcome_values_used_for_identity_decision:
            raise ValueError("pilot overlap identity decision may not use outcome values")
        if not self.mapping_complete:
            raise ValueError("pilot overlap receipt does not assert complete mapping")
        normalized = [item.normalized_embryo_id for item in self.pairs]
        scaleup_ids = [item.scaleup_embryo_id for item in self.pairs]
        if len(normalized) != len(set(normalized)) or len(scaleup_ids) != len(set(scaleup_ids)):
            raise ValueError("pilot overlap receipt is not one-to-one")
        disjoint = self.assertions.omix709_05_54_and_05_55_are_biological_embryo_disjoint
        if disjoint and self.pairs:
            raise ValueError("disjoint archives may not declare overlap pairs")
        if not disjoint and not self.pairs:
            raise ValueError("non-disjoint archives require exhaustive overlap pairs")
        return self


@dataclass(frozen=True)
class ReferenceRow:
    cell_id: str
    xlsx_row: int
    support: int


@dataclass(frozen=True)
class RawControlLineageData:
    relative_births_by_control: dict[str, dict[str, int]]
    source_receipt: dict[str, Any]


def load_contract(path: str | Path) -> StageAdapterContract:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"stage-adapter contract is not a file: {source}")
    return StageAdapterContract.model_validate_json(source.read_text(encoding="utf-8"))


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _dimension_rows(reference: str) -> int:
    return scaleup._dimension_shape(reference)[0]


def _nonempty_strings(values: Sequence[Any]) -> list[str]:
    return [str(value) for value in values if value is not None and str(value) != ""]


def _column_map(identifiers: Sequence[str]) -> list[dict[str, Any]]:
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("embryo identifiers are not unique within a modality")
    return [
        {
            "embryo_id": identifier,
            "zero_based_data_column": index,
            "xlsx_column": index + 2,
        }
        for index, identifier in enumerate(identifiers)
    ]


def _row_map(rows: Sequence[tuple[int, Any]]) -> list[dict[str, Any]]:
    result = [
        {"semantic_cell_id": str(value), "xlsx_row": row_number}
        for row_number, value in rows
        if value is not None and str(value) != ""
    ]
    identifiers = [item["semantic_cell_id"] for item in result]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("semantic-cell identifiers are not unique within a modality")
    return result


def _parse_reference_value(
    value: Any,
    missing_tokens: set[str],
    modality_id: str,
) -> float | tuple[float, float, float] | None:
    if value is None or (isinstance(value, str) and value.strip() in missing_tokens):
        return None
    if isinstance(value, bool):
        raise ValueError("boolean reference measurements are not permitted")
    if modality_id == "three_dimensional_position":
        if not isinstance(value, str):
            raise ValueError("three-dimensional reference positions must be coordinate triples")
        match = re.fullmatch(
            r"\(\s*([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\s*,\s*"
            r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\s*,\s*"
            r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))\s*\)",
            value.strip(),
        )
        if match is None:
            raise ValueError(
                f"three-dimensional reference position is malformed: {value!r}"
            )
        first, second, third = (float(item) for item in match.groups())
        coordinates = (first, second, third)
        if not all(math.isfinite(item) for item in coordinates):
            raise ValueError("three-dimensional reference positions must be finite")
        return coordinates
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reference measurement is neither numeric nor missing: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError("reference measurements must be finite")
    return parsed


def _verify_raw_control_inventory(
    archive_path: Path,
    raw_control_dir: Path,
    source: RawControlLineageSourceContract,
) -> tuple[list[tuple[Path, dict[str, Any]]], dict[str, Any]]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"raw-control archive is not a file: {archive_path}")
    if archive_path.name != source.archive.filename:
        raise ValueError("raw-control archive filename differs")
    archive_bytes = archive_path.stat().st_size
    if archive_bytes != source.archive.bytes:
        raise ValueError(
            "raw-control archive byte identity differs: "
            f"expected {source.archive.bytes}, observed {archive_bytes}"
        )
    archive_sha256 = _sha256(archive_path)
    if archive_sha256 != source.archive.sha256:
        raise ValueError("raw-control archive SHA-256 identity differs")

    if not raw_control_dir.is_dir():
        raise FileNotFoundError(
            f"raw-control extraction directory is missing: {raw_control_dir}"
        )
    if raw_control_dir.name != source.extracted.directory_name:
        raise ValueError("raw-control extraction directory name differs")
    members = sorted(raw_control_dir.iterdir(), key=lambda item: item.name)
    if any(not item.is_file() for item in members):
        raise ValueError("raw-control extraction contains a non-file member")
    if len(members) != source.extracted.expected_file_count:
        raise ValueError(
            "raw-control extracted file count differs: "
            f"expected {source.extracted.expected_file_count}, observed {len(members)}"
        )

    member_pattern = re.compile(source.extracted.member_filename_regex)
    observed_serials: list[int] = []
    for path in members:
        match = member_pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(
                f"raw-control extraction filename violates the frozen grammar: {path.name}"
            )
        observed_serials.append(int(match.group(1)))
    expected_serials = list(
        range(
            source.extracted.member_sequence_start,
            source.extracted.member_sequence_start + source.extracted.expected_file_count,
        )
    )
    if sorted(observed_serials) != expected_serials:
        raise ValueError("raw-control extracted member sequence differs")

    total_bytes = sum(path.stat().st_size for path in members)
    if total_bytes != source.extracted.expected_total_bytes:
        raise ValueError(
            "raw-control extracted byte count differs: "
            f"expected {source.extracted.expected_total_bytes}, observed {total_bytes}"
        )
    inventory: list[dict[str, Any]] = []
    bound_members: list[tuple[Path, dict[str, Any]]] = []
    for path in members:
        item = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        inventory.append(item)
        bound_members.append((path, item))
    inventory_sha256 = _stable_hash(inventory)
    if inventory_sha256 != source.extracted.inventory_sha256:
        raise ValueError("raw-control extracted inventory SHA-256 differs")
    receipt = {
        "schema_version": RAW_CONTROL_RECEIPT_VERSION,
        "archive": {
            "filename": archive_path.name,
            "bytes": archive_bytes,
            "sha256": archive_sha256,
            "verified": True,
        },
        "extracted": {
            "directory_name": raw_control_dir.name,
            "file_count": len(inventory),
            "total_bytes": total_bytes,
            "inventory_canonicalization": source.extracted.inventory_canonicalization,
            "inventory_sha256": inventory_sha256,
            "members": inventory,
            "verified": True,
        },
    }
    return bound_members, receipt


def _decode_raw_time(value: bytes, path: Path, line_number: int) -> int:
    try:
        text = value.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}:{line_number}: time is not ASCII") from exc
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", text) is None:
        raise ValueError(f"{path.name}:{line_number}: time is not a nonnegative integer")
    return int(text)


def _decode_raw_cell_id(value: bytes, path: Path, line_number: int) -> str:
    try:
        text = value.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}:{line_number}: cell ID is not ASCII") from exc
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,127}", text) is None:
        raise ValueError(f"{path.name}:{line_number}: invalid semantic cell identifier")
    return text


def _read_raw_control_lineage_source(
    archive_path: str | Path,
    raw_control_dir: str | Path,
    contract: StageAdapterContract,
) -> RawControlLineageData:
    """Read only the frozen time and cell-ID fields from exact raw controls.

    Files are opened in binary mode.  Each record is split only far enough to
    isolate fields 1 and 2; bytes in all outcome columns remain undecoded.
    """

    source = contract.raw_control_lineage_source
    members, receipt = _verify_raw_control_inventory(
        Path(archive_path).resolve(),
        Path(raw_control_dir).resolve(),
        source,
    )
    access = source.identity_time_access
    time_column, cell_column = access.permitted_columns
    relative_births: dict[str, dict[str, int]] = {}
    control_summaries: list[dict[str, Any]] = []
    total_identity_time_rows = 0
    for path, identity in members:
        births: dict[str, int] = {}
        last_times: dict[str, int] = {}
        seen_cell_times: set[tuple[int, str]] = set()
        row_count = 0
        with path.open("rb") as handle:
            raw_header = handle.readline()
            if not raw_header:
                raise ValueError(f"raw-control table is empty: {path.name}")
            header = raw_header.rstrip(b"\r\n")
            if hashlib.sha256(header).hexdigest() != access.header_sha256:
                raise ValueError(f"{path.name}: raw-control header identity differs")
            header_fields = header.split(b"\t")
            if len(header_fields) != access.header_field_count:
                raise ValueError(f"{path.name}: raw-control header field count differs")
            for column in (time_column, cell_column):
                if header_fields[column.zero_based_index] != column.source_header.encode("ascii"):
                    raise ValueError(f"{path.name}: permitted raw-control header differs")
            for line_number, raw_line in enumerate(handle, 2):
                line = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
                line = line[:-1] if line.endswith(b"\r") else line
                if not line:
                    raise ValueError(f"{path.name}:{line_number}: blank raw-control row")
                if line.count(b"\t") != access.header_field_count - 1:
                    raise ValueError(
                        f"{path.name}:{line_number}: raw-control field count differs"
                    )
                prefix = line.split(b"\t", 3)
                if len(prefix) != 4:
                    raise ValueError(
                        f"{path.name}:{line_number}: permitted raw-control columns are absent"
                    )
                time = _decode_raw_time(prefix[time_column.zero_based_index], path, line_number)
                cell_id = _decode_raw_cell_id(
                    prefix[cell_column.zero_based_index], path, line_number
                )
                key = (time, cell_id)
                if key in seen_cell_times:
                    raise ValueError(
                        f"{path.name}:{line_number}: duplicate cell/time identity: {cell_id}/{time}"
                    )
                seen_cell_times.add(key)
                births[cell_id] = min(time, births.get(cell_id, time))
                last_times[cell_id] = max(time, last_times.get(cell_id, time))
                row_count += 1
        if not row_count:
            raise ValueError(f"raw-control table contains no identity/time rows: {path.name}")
        if "ABa" not in last_times or "ABp" not in last_times:
            raise ValueError(f"{path.name}: relative-time anchor cells ABa/ABp are absent")
        anchor = min(last_times["ABa"], last_times["ABp"]) + 1
        relative_births[path.name] = {
            cell_id: time - anchor for cell_id, time in births.items()
        }
        control_summaries.append(
            {
                "name": path.name,
                "source_sha256": identity["sha256"],
                "identity_time_rows": row_count,
                "unique_cells": len(births),
                "relative_birth_time_anchor": anchor,
            }
        )
        total_identity_time_rows += row_count

    union_cells = set().union(*(set(item) for item in relative_births.values()))
    if len(union_cells) != source.expected_union_cell_count:
        raise ValueError(
            "raw-control semantic-cell union differs: "
            f"expected {source.expected_union_cell_count}, observed {len(union_cells)}"
        )
    receipt["access"] = {
        "header_sha256": access.header_sha256,
        "header_field_count": access.header_field_count,
        "permitted_columns": [item.model_dump(mode="json") for item in access.permitted_columns],
        "nonpermitted_value_columns_policy": access.nonpermitted_value_columns_policy,
        "relative_birth_time_anchor": access.relative_birth_time_anchor,
        "identity_time_rows_decoded": total_identity_time_rows,
        "raw_control_files_decoded": len(relative_births),
        "semantic_cell_union_count": len(union_cells),
        "outcome_columns_decoded": False,
        "control_summaries": control_summaries,
    }
    return RawControlLineageData(
        relative_births_by_control=relative_births,
        source_receipt=receipt,
    )


def _reference_rows(
    package: scaleup.XlsxPackage,
    sheet: str,
    dimension: str,
    support: SupportContract,
    modality_id: str,
) -> tuple[list[str], list[ReferenceRow]]:
    scan = package.scan_sheet(sheet, full_rows=range(1, _dimension_rows(dimension) + 1))
    header = scan.full_rows.get(2, [])
    if len(header) < 2:
        raise ValueError(f"S2/{sheet}: reference-control header is absent")
    controls = _nonempty_strings(header[1:])
    if len(controls) != len(set(controls)):
        raise ValueError(f"S2/{sheet}: duplicate control embryo identifiers")
    missing_tokens = set(support.missing_tokens)
    rows: list[ReferenceRow] = []
    seen: set[str] = set()
    for row_number in range(3, _dimension_rows(dimension) + 1):
        values = scan.full_rows.get(row_number, [])
        if not values or values[0] is None or str(values[0]) == "":
            continue
        cell_id = str(values[0])
        if cell_id in seen:
            raise ValueError(f"S2/{sheet}: duplicate semantic-cell identifier: {cell_id}")
        seen.add(cell_id)
        parsed = [
            item
            for item in (
                _parse_reference_value(value, missing_tokens, modality_id)
                for value in values[1 : len(controls) + 1]
            )
            if item is not None
        ]
        rows.append(
            ReferenceRow(
                cell_id=cell_id,
                xlsx_row=row_number,
                support=len(parsed),
            )
        )
    if not rows:
        raise ValueError(f"S2/{sheet}: no semantic-cell rows")
    return controls, rows


def _parent_of(cell_id: str, contract: LineageContract) -> str | None:
    explicit = contract.explicit_parent_by_child.get(cell_id)
    if explicit is not None:
        return explicit
    if cell_id and cell_id[-1] in set(contract.suffix_parent_characters):
        candidate = cell_id[:-1]
        if candidate and any(candidate.startswith(root) for root in contract.suffix_parent_roots):
            return candidate
    return None


def _derive_lineage(
    raw_controls: RawControlLineageData,
    contract: StageAdapterContract,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = contract.raw_control_lineage_source
    births_by_control = raw_controls.relative_births_by_control
    control_count = len(births_by_control)
    if control_count != source.extracted.expected_file_count:
        raise ValueError("raw-control count differs after identity/time parsing")
    catalog = set().union(*(set(item) for item in births_by_control.values()))
    catalog.update(contract.lineage.initial_frontier)
    children: dict[str, list[str]] = defaultdict(list)
    parent_by_child: dict[str, str] = {}
    unknown: list[str] = []
    for cell_id in sorted(catalog):
        parent = _parent_of(cell_id, contract.lineage)
        if parent is None:
            if cell_id not in {
                *contract.lineage.initial_frontier,
                contract.lineage.virtual_root,
            }:
                unknown.append(cell_id)
            continue
        parent_by_child[cell_id] = parent
        children[parent].append(cell_id)
    if unknown:
        raise ValueError("unknown semantic lineage identifier: " + ", ".join(unknown[:8]))
    missing_parents = sorted(
        {
            parent
            for parent in parent_by_child.values()
            if parent != contract.lineage.virtual_root and parent not in catalog
        }
    )
    if missing_parents:
        raise ValueError(
            "lineage parent is absent from the raw-control catalog: "
            + ", ".join(missing_parents[:8])
        )

    explicit_children: dict[str, set[str]] = defaultdict(set)
    for child, parent in contract.lineage.explicit_parent_by_child.items():
        explicit_children[parent].add(child)
    for parent in children:
        children[parent].sort()
        if len(children[parent]) != contract.lineage.daughter_count:
            raise ValueError(f"lineage parent does not have exactly two observed daughters: {parent}")
        if parent in explicit_children:
            if set(children[parent]) != explicit_children[parent]:
                raise ValueError(f"explicit daughter identities differ for lineage parent: {parent}")
        else:
            suffixes = {child[-1] for child in children[parent]}
            allowed_pairs = [set(pair) for pair in contract.lineage.daughter_suffix_pairs]
            if suffixes not in allowed_pairs:
                raise ValueError(f"daughter suffix pair differs for lineage parent: {parent}")
    if len(children) != source.expected_binary_parent_count:
        raise ValueError(
            "raw-control binary parent count differs: "
            f"expected {source.expected_binary_parent_count}, observed {len(children)}"
        )

    minimum_support = max(
        contract.support.minimum_reference_count,
        math.ceil(contract.support.minimum_reference_fraction * control_count),
    )
    events: list[tuple[float, str, int]] = []
    unsupported_events: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for parent, daughters in sorted(children.items()):
        if len(daughters) != 2 or parent == contract.lineage.virtual_root:
            continue
        per_control_times: list[int] = []
        for control_name, births in sorted(births_by_control.items()):
            if not all(daughter in births for daughter in daughters):
                continue
            first, second = (births[daughter] for daughter in daughters)
            if first != second:
                disagreements.append(
                    {
                        "control": control_name,
                        "parent": parent,
                        "children": daughters,
                        "relative_birth_times": [first, second],
                    }
                )
                continue
            per_control_times.append(first)
        if len(per_control_times) < minimum_support:
            unsupported_events.append({"parent": parent, "support": len(per_control_times)})
            continue
        events.append(
            (float(statistics.median(per_control_times)), parent, len(per_control_times))
        )
    if len(disagreements) != source.expected_sibling_birth_time_disagreements:
        raise ValueError(
            "raw-control sibling birth-time disagreement count differs: "
            f"expected {source.expected_sibling_birth_time_disagreements}, "
            f"observed {len(disagreements)}"
        )
    if len(events) != source.expected_supported_division_events:
        raise ValueError(
            "supported raw-control division-event count differs: "
            f"expected {source.expected_supported_division_events}, observed {len(events)}"
        )
    if len(unsupported_events) != source.expected_unsupported_division_events:
        raise ValueError(
            "unsupported raw-control division-event count differs: "
            f"expected {source.expected_unsupported_division_events}, "
            f"observed {len(unsupported_events)}"
        )
    events.sort(key=lambda item: (item[0], item[1]))

    targets = {
        contract.landmarks.input_cell_count: None,
        contract.landmarks.endpoint_cell_count: None,
    }
    frontier = set(contract.lineage.initial_frontier)
    born = set(frontier)
    completed: list[str] = []
    event_receipts: list[dict[str, Any]] = []
    for median, parent, support in events:
        if parent not in frontier:
            if parent in born:
                raise ValueError(f"division timing violates lineage order at {parent}")
            raise ValueError(f"division event precedes parent birth at {parent}")
        daughters = children[parent]
        frontier.remove(parent)
        frontier.update(daughters)
        born.update(daughters)
        completed.append(parent)
        event_receipts.append(
            {
                "event_index": len(event_receipts),
                "parent": parent,
                "children": daughters,
                "reference_median_relative_daughter_birth_time": median,
                "reference_support": support,
                "frontier_size_after_event": len(frontier),
            }
        )
        if len(frontier) in targets and targets[len(frontier)] is None:
            targets[len(frontier)] = {
                "frontier": sorted(frontier),
                "born_cells": sorted(born),
                "completed_division_cells": list(completed),
                "cutoff_event_index": len(event_receipts) - 1,
                "cutoff_reference_median_relative_daughter_birth_time": median,
            }
        if targets[contract.landmarks.endpoint_cell_count] is not None:
            break
    missing_targets = [size for size, value in targets.items() if value is None]
    if missing_targets:
        raise ValueError(f"reference lineage cannot reach exact stage frontiers: {missing_targets}")

    stage_26 = targets[contract.landmarks.input_cell_count]
    stage_200 = targets[contract.landmarks.endpoint_cell_count]
    assert stage_26 is not None and stage_200 is not None
    if len(stage_26["completed_division_cells"]) != source.expected_input_event_count:
        raise ValueError("raw-control input frontier event count differs")
    if len(stage_200["completed_division_cells"]) != source.expected_endpoint_event_count:
        raise ValueError("raw-control endpoint frontier event count differs")
    lineage_cells = set(stage_200["born_cells"]) | {contract.lineage.virtual_root}
    edges = [
        {"parent": parent, "child": child}
        for child, parent in sorted(parent_by_child.items())
        if child in lineage_cells and parent in lineage_cells
    ]
    lineage = {
        "schema_version": LINEAGE_MANIFEST_VERSION,
        "virtual_root": contract.lineage.virtual_root,
        "initial_frontier": contract.lineage.initial_frontier,
        "parent_rule": {
            "explicit_parent_by_child": contract.lineage.explicit_parent_by_child,
            "suffix_parent_characters": contract.lineage.suffix_parent_characters,
            "suffix_parent_roots": contract.lineage.suffix_parent_roots,
            "daughter_suffix_pairs": contract.lineage.daughter_suffix_pairs,
        },
        "event_order": contract.lineage.event_order,
        "tie_breaker": contract.lineage.tie_breaker,
        "minimum_reference_support": minimum_support,
        "cells_through_200": sorted(lineage_cells),
        "edges_through_200": edges,
        "events_through_200": event_receipts,
    }
    stages = {
        "schema_version": STAGE_MANIFEST_VERSION,
        "derivation_source": (
            "OMIX709-05-53.rar/Raw_data_Control/identity_time_only_relative_births"
        ),
        "raw_control_outcome_columns_decoded": False,
        "perturbation_measurement_values_read": False,
        "s4_outcome_values_read": False,
        "input_26": stage_26,
        "endpoint_200": stage_200,
    }
    source_receipt = dict(raw_controls.source_receipt)
    source_receipt["lineage"] = {
        "semantic_cell_union_count": len(catalog),
        "binary_parent_count": len(children),
        "supported_division_events": len(events),
        "unsupported_division_events": len(unsupported_events),
        "minimum_event_support": minimum_support,
        "sibling_birth_time_disagreements": len(disagreements),
        "unsupported_event_support": unsupported_events,
        "input_frontier_event_count": len(stage_26["completed_division_cells"]),
        "endpoint_frontier_event_count": len(stage_200["completed_division_cells"]),
        "input_frontier_cutoff_relative_birth_time": stage_26[
            "cutoff_reference_median_relative_daughter_birth_time"
        ],
        "endpoint_frontier_cutoff_relative_birth_time": stage_200[
            "cutoff_reference_median_relative_daughter_birth_time"
        ],
        "exact_parentage_verified": True,
    }
    return lineage, stages, source_receipt


def _scan_identifier_mappings(
    parent_manifest: scaleup.ScaleupQualificationManifest,
    source_dir: Path,
    contract: StageAdapterContract,
    stages: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, ReferenceRow]]]:
    source_by_name = {item.filename: item for item in parent_manifest.sources}
    for source in parent_manifest.sources:
        scaleup._verify_source(source_dir / source.filename, source)

    s2_contract = source_by_name["Table_S2.xlsx"]
    s2_dimensions = {item.name: item.dimension for item in s2_contract.sheets}
    policy_by_id = {item.id: item for item in contract.modalities}

    reference_by_modality: dict[str, dict[str, ReferenceRow]] = {}
    mappings: list[dict[str, Any]] = []
    s3_embryo_sets: list[set[str]] = []
    with (
        scaleup.XlsxPackage(source_dir / "Table_S2.xlsx") as s2,
        scaleup.XlsxPackage(source_dir / "Table_S3.xlsx") as s3,
    ):
        for modality in parent_manifest.modalities:
            controls, reference_rows = _reference_rows(
                s2,
                modality.s2_sheet,
                s2_dimensions[modality.s2_sheet],
                contract.support,
                modality.id,
            )
            if len(controls) != parent_manifest.expected_counts.reference_controls:
                raise ValueError(
                    f"S2/{modality.s2_sheet}: reference-control count differs from parent"
                )
            reference_by_modality[modality.id] = {
                item.cell_id: item for item in reference_rows
            }

            s3_scan = s3.scan_sheet(
                modality.s3_sheet,
                full_rows={1, 2},
                first_column_start_row=3,
            )
            s3_ids = _nonempty_strings(s3_scan.full_rows.get(2, [])[1:])
            s3_rows = _row_map(s3_scan.first_column)
            s3_embryo_sets.append(set(s3_ids))
            s3_row_ids = {item["semantic_cell_id"] for item in s3_rows}

            s4_payload: dict[str, Any]
            if modality.s4_sheet == "4":
                s4_payload = {
                    "sheet": "4",
                    "status": "excluded_no_access",
                    "workbook_parsed": False,
                    "header_decoded": False,
                    "semantic_rows_decoded": False,
                    "values_decoded": False,
                }
            else:
                s4_payload = {
                    "sheet": modality.s4_sheet,
                    "status": "qualified_identifier_join_inherited_from_parent",
                    "workbook_parsed": False,
                    "embryo_join_key": "embryo_id",
                    "semantic_row_join_key": "semantic_cell_id",
                    "values_decoded": False,
                }

            policy = policy_by_id[modality.id]
            if policy.input_scope == "completed_division_cells":
                input_cells = stages["input_26"]["completed_division_cells"]
            elif policy.input_scope == "cells_born_by_frontier":
                input_cells = stages["input_26"]["born_cells"]
            else:
                raise ValueError(f"unknown input scope for {modality.id}")
            endpoint_cells = stages["endpoint_200"]["frontier"] if policy.endpoint_eligible else []
            reference_index = reference_by_modality[modality.id]
            mappings.append(
                {
                    "id": modality.id,
                    "s2_sheet": modality.s2_sheet,
                    "s3_sheet": modality.s3_sheet,
                    "s3_embryo_columns": _column_map(s3_ids),
                    "s3_semantic_rows": s3_rows,
                    "s3_values_decoded": False,
                    "input_26": {
                        "scope": policy.input_scope,
                        "requested_cells": list(input_cells),
                        "present_s3_cells": sorted(set(input_cells) & s3_row_ids),
                        "structurally_absent_s3_cells": sorted(set(input_cells) - s3_row_ids),
                        "reference_support": {
                            cell: reference_index[cell].support
                            for cell in sorted(set(input_cells) & set(reference_index))
                        },
                    },
                    "endpoint_200": {
                        "eligible": policy.endpoint_eligible,
                        "requested_cells": list(endpoint_cells),
                        "present_s3_cells": sorted(set(endpoint_cells) & s3_row_ids),
                        "structurally_absent_s3_cells": sorted(set(endpoint_cells) - s3_row_ids),
                        "reference_support": {
                            cell: reference_index[cell].support
                            for cell in sorted(set(endpoint_cells) & set(reference_index))
                        },
                    },
                    "s4": s4_payload,
                }
            )
    if not s3_embryo_sets or any(item != s3_embryo_sets[0] for item in s3_embryo_sets[1:]):
        raise ValueError("S3 modality-local mappings do not contain the same embryo set")
    return {
        "schema_version": MAPPING_MANIFEST_VERSION,
        "mapping_rule": "modality_local_never_reuse_cross_sheet_position",
        "perturbation_values_decoded": False,
        "s4_outcome_values_decoded": False,
        "s4_workbook_parsed": False,
        "s4_sheet_4_accessed": False,
        "modalities": mappings,
    }, reference_by_modality


def _overlap_status(
    receipt_path: str | Path | None,
    evidence_record_path: str | Path | None,
    source_bundle_sha256: str,
    split_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    claim_ids = {
        embryo_id
        for gene in split_manifest["claim_bearing_genes"]
        for embryo_id in gene["embryo_ids"]
    }
    exposed_ids = {
        embryo_id
        for gene in split_manifest["development_only_exposed_pilot_genes"]
        for embryo_id in gene["embryo_ids"]
    }
    if receipt_path is None:
        if evidence_record_path is not None:
            raise ValueError("overlap evidence record was supplied without a receipt")
        return {
            "schema_version": OVERLAP_RECEIPT_VERSION,
            "receipt_supplied": False,
            "embryo_level_disjointness_established": False,
            "model_launch_gate": "blocked_pending_pilot_embryo_overlap_receipt",
            "gene_level_exclusion_retained": True,
        }
    path = Path(receipt_path).resolve()
    receipt = PilotOverlapReceipt.model_validate_json(path.read_text(encoding="utf-8"))
    if evidence_record_path is None:
        raise ValueError("pilot-overlap receipt requires its durable source evidence record")
    evidence = Path(evidence_record_path).resolve()
    if not evidence.is_file():
        raise FileNotFoundError(f"pilot-overlap evidence record is missing: {evidence}")
    if evidence.stat().st_size != receipt.source_evidence_record_bytes:
        raise ValueError("pilot-overlap evidence record byte identity differs")
    if _sha256(evidence) != receipt.source_evidence_record_sha256:
        raise ValueError("pilot-overlap evidence record SHA-256 differs")
    if receipt.scaleup_source_bundle_sha256 != source_bundle_sha256:
        raise ValueError("pilot-overlap receipt is bound to a different S1-S4 source bundle")
    mapped = {item.scaleup_embryo_id for item in receipt.pairs}
    unknown = sorted(mapped - claim_ids - exposed_ids)
    if unknown:
        raise ValueError(f"pilot-overlap receipt contains unknown scale-up embryos: {unknown[:5]}")
    claim_overlap = sorted(mapped & claim_ids)
    if claim_overlap:
        raise ValueError(
            "pilot embryo overlap reaches claim-bearing partitions: " + ", ".join(claim_overlap[:5])
        )
    return {
        "schema_version": OVERLAP_RECEIPT_VERSION,
        "receipt_supplied": True,
        "receipt_sha256": _sha256(path),
        "source_evidence_record_identifier": receipt.source_evidence_record_identifier,
        "source_evidence_record_bytes": receipt.source_evidence_record_bytes,
        "source_evidence_record_sha256": receipt.source_evidence_record_sha256,
        "pilot_normalized_sha256": receipt.pilot_normalized_sha256,
        "pilot_embryos": receipt.pilot_embryos,
        "mapped_scaleup_embryos": len(mapped),
        "mapped_development_only_embryos": len(mapped & exposed_ids),
        "claim_bearing_overlap_embryos": 0,
        "embryo_level_disjointness_established": True,
        "model_launch_gate": "closed",
        "gene_level_exclusion_retained": True,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_run_atomic(output_root: Path, payloads: Mapping[str, Any]) -> None:
    if output_root.exists():
        raise FileExistsError(f"stage-adapter output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for name in sorted(payloads):
            _write_json_atomic(temporary / name, payloads[name])
        lines = [f"{_sha256(temporary / name)}  {name}" for name in sorted(payloads)]
        _write_bytes_atomic(temporary / "SHA256SUMS.txt", ("\n".join(lines) + "\n").encode())
        _write_bytes_atomic(temporary / "SUCCESS", b"SUCCESS\n")
        os.replace(temporary, output_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def freeze_stage_adapter(
    contract_or_path: StageAdapterContract | str | Path,
    parent_manifest_path: str | Path,
    parent_qualification_root: str | Path,
    source_dir: str | Path,
    raw_control_archive: str | Path,
    raw_control_dir: str | Path,
    output_root: str | Path,
    *,
    academic_use_attested: bool,
    pilot_overlap_receipt: str | Path | None = None,
    pilot_overlap_evidence_record: str | Path | None = None,
) -> dict[str, Any]:
    contract = (
        contract_or_path
        if isinstance(contract_or_path, StageAdapterContract)
        else load_contract(contract_or_path)
    )
    if contract.rights.operator_attestation_required and not academic_use_attested:
        raise PermissionError(contract.rights.attestation_text)
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"stage-adapter output root already exists: {output}")
    parent_manifest_source = Path(parent_manifest_path).resolve()
    if parent_manifest_source.name != contract.parent.manifest_filename:
        raise ValueError("parent qualification manifest filename differs")
    if _sha256(parent_manifest_source) != contract.parent.manifest_sha256:
        raise ValueError("parent qualification manifest SHA-256 differs")
    parent_manifest = scaleup.load_manifest(parent_manifest_source)
    parent_root = Path(parent_qualification_root).resolve()
    scaleup.verify_qualification(parent_root, parent_manifest_source)
    parent_qualification = _read_json(parent_root / "qualification.json")
    if (
        parent_qualification.get("schema_version") != contract.parent.schema_version
        or parent_qualification.get("qualification_status") != contract.parent.required_status
        or parent_qualification.get("model_launch_status")
        != contract.parent.required_model_launch_status
    ):
        raise ValueError("parent qualification state differs from the adapter contract")
    if parent_qualification.get("outcome_prevalence_inspected") is not False:
        raise ValueError("parent qualification is not outcome-blind")
    if parent_qualification.get("gates", {}).get("s4_cnd1_status_excluded") is not True:
        raise ValueError("parent qualification does not preserve the S4 sheet 4 exclusion")

    parent_receipt = _read_json(parent_root / "source_receipt.json")
    if parent_receipt.get("academic_use_attested") is not True:
        raise ValueError("parent source receipt lacks controlled-use attestation")
    source_bundle_sha256 = str(parent_receipt.get("source_bundle_sha256", ""))
    if not SHA256_RE.fullmatch(source_bundle_sha256):
        raise ValueError("parent source-bundle receipt is malformed")
    split_manifest = _read_json(parent_root / "split_manifest.json")
    source_path = Path(source_dir).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"processed OMIX709 source directory is missing: {source_path}")

    if (
        contract.raw_control_lineage_source.extracted.expected_file_count
        != parent_manifest.expected_counts.reference_controls
    ):
        raise ValueError("raw-control file count differs from the qualified parent")
    raw_controls = _read_raw_control_lineage_source(
        raw_control_archive,
        raw_control_dir,
        contract,
    )
    lineage, stages, raw_control_receipt = _derive_lineage(raw_controls, contract)
    raw_control_receipt_sha256 = _stable_hash(raw_control_receipt)
    lineage["raw_control_source_receipt_sha256"] = raw_control_receipt_sha256
    stages["raw_control_source_receipt_sha256"] = raw_control_receipt_sha256
    mappings, _reference = _scan_identifier_mappings(
        parent_manifest,
        source_path,
        contract,
        stages,
    )
    overlap = _overlap_status(
        pilot_overlap_receipt,
        pilot_overlap_evidence_record,
        source_bundle_sha256,
        split_manifest,
    )
    overlap_closed = overlap["embryo_level_disjointness_established"] is True
    launch_status = (
        "eligible_for_preregistered_gene_disjoint_baselines"
        if overlap_closed
        else "blocked_pending_pilot_embryo_overlap_receipt"
    )
    qualification_status = (
        "qualified_with_exclusions"
        if overlap_closed
        else "stage_adapter_frozen_with_open_overlap_gate"
    )
    contract_path = (
        Path(contract_or_path).resolve()
        if isinstance(contract_or_path, (str, Path))
        else None
    )
    contract_hash = (
        _sha256(contract_path)
        if contract_path is not None
        else _stable_hash(contract.model_dump(mode="json"))
    )
    module_hash = _sha256(Path(__file__).resolve())
    stages.update(
        {
            "contract_sha256": contract_hash,
            "lineage_manifest_sha256": _stable_hash(lineage),
            "input_26_frontier_count": len(stages["input_26"]["frontier"]),
            "endpoint_200_frontier_count": len(stages["endpoint_200"]["frontier"]),
        }
    )
    analysis = {
        "schema_version": ADAPTER_ANALYSIS_VERSION,
        "analysis_id": contract.analysis_id,
        "classification": contract.classification,
        "contract_sha256": contract_hash,
        "adapter_module_sha256": module_hash,
        "parent_qualification_sha256": _sha256(parent_root / "qualification.json"),
        "parent_split_manifest_sha256": _sha256(parent_root / "split_manifest.json"),
        "source_bundle_sha256": source_bundle_sha256,
        "raw_control_archive_sha256": raw_control_receipt["archive"]["sha256"],
        "raw_control_inventory_sha256": raw_control_receipt["extracted"][
            "inventory_sha256"
        ],
        "raw_control_source_receipt_sha256": raw_control_receipt_sha256,
        "model_launch_status": launch_status,
        "biological_claims_permitted": False,
        "sealed_test_opened": False,
        "outcome_prevalence_inspected": False,
    }
    qualification = {
        "schema_version": ADAPTER_QUALIFICATION_VERSION,
        "technical_status": "success",
        "qualification_status": qualification_status,
        "model_launch_status": launch_status,
        "gates": {
            "parent_qualification_verified": True,
            "raw_control_archive_identity_verified": True,
            "raw_control_extracted_inventory_verified": True,
            "raw_control_identity_time_only_stage_derivation": True,
            "raw_control_outcome_columns_not_decoded": True,
            "reference_control_only_stage_derivation": True,
            "input_26_exact_frontier_frozen": True,
            "endpoint_200_exact_frontier_frozen": True,
            "parent_child_lineage_frozen": True,
            "modality_local_embryo_maps_frozen": True,
            "missingness_and_support_policy_frozen": True,
            "s3_perturbation_values_not_read": True,
            "s4_outcome_values_not_read": True,
            "s4_sheet_4_excluded_without_access": True,
            "pilot_embryo_overlap_closed": overlap_closed,
            "sealed_test_opened": False,
        },
        "counts": {
            "input_26_frontier_cells": len(stages["input_26"]["frontier"]),
            "input_26_born_cells": len(stages["input_26"]["born_cells"]),
            "input_26_completed_divisions": len(
                stages["input_26"]["completed_division_cells"]
            ),
            "endpoint_200_frontier_cells": len(stages["endpoint_200"]["frontier"]),
            "raw_control_files": raw_control_receipt["extracted"]["file_count"],
            "raw_control_semantic_cell_union": raw_control_receipt["lineage"][
                "semantic_cell_union_count"
            ],
            "raw_control_supported_division_events": raw_control_receipt["lineage"][
                "supported_division_events"
            ],
            "lineage_edges_through_200": len(lineage["edges_through_200"]),
            "measurement_embryos": parent_manifest.expected_counts.measurement_embryos,
            "modalities": len(mappings["modalities"]),
        },
        "outcome_prevalence_inspected": False,
        "scientific_status": "not_estimable_until_preregistered_model_run",
        "warnings": (
            []
            if overlap_closed
            else [
                {
                    "id": "pilot_embryo_overlap_receipt_missing",
                    "message": (
                        "Stage and mapping artifacts are frozen, but model launch remains "
                        "blocked until a complete source-supported pilot overlap receipt closes."
                    ),
                }
            ]
        ),
    }
    payloads = {
        "analysis_manifest.json": analysis,
        "lineage_manifest.json": lineage,
        "modality_mappings.json": mappings,
        "overlap_status.json": overlap,
        "qualification.json": qualification,
        "raw_control_source_receipt.json": raw_control_receipt,
        "stage_membership.json": stages,
    }
    _write_run_atomic(output, payloads)
    return qualification


def verify_stage_adapter(
    run_root: str | Path,
    contract: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"stage-adapter run root is missing: {root}")
    expected = OUTPUT_FILES | {"SHA256SUMS.txt", "SUCCESS"}
    observed = {item.name for item in root.iterdir()}
    if observed != expected:
        raise ValueError("stage-adapter output inventory differs")
    if (root / "SUCCESS").read_bytes() != b"SUCCESS\n":
        raise ValueError("stage-adapter SUCCESS marker is invalid")
    lines = (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    checked: set[str] = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None or match.group(2) not in OUTPUT_FILES or match.group(2) in checked:
            raise ValueError("stage-adapter checksum manifest is malformed")
        if _sha256(root / match.group(2)) != match.group(1):
            raise ValueError(f"stage-adapter checksum failed: {match.group(2)}")
        checked.add(match.group(2))
    if checked != OUTPUT_FILES:
        raise ValueError("stage-adapter checksum manifest coverage differs")
    qualification = _read_json(root / "qualification.json")
    if (
        qualification.get("schema_version") != ADAPTER_QUALIFICATION_VERSION
        or qualification.get("technical_status") != "success"
    ):
        raise ValueError("stage-adapter qualification identity differs")
    stages = _read_json(root / "stage_membership.json")
    if (
        len(stages.get("input_26", {}).get("frontier", [])) != 26
        or len(stages.get("endpoint_200", {}).get("frontier", [])) != 200
        or stages.get("raw_control_outcome_columns_decoded") is not False
        or stages.get("perturbation_measurement_values_read") is not False
        or stages.get("s4_outcome_values_read") is not False
    ):
        raise ValueError("stage membership or outcome-blindness receipt differs")
    raw_control = _read_json(root / "raw_control_source_receipt.json")
    if (
        raw_control.get("schema_version") != RAW_CONTROL_RECEIPT_VERSION
        or raw_control.get("archive", {}).get("verified") is not True
        or raw_control.get("extracted", {}).get("verified") is not True
        or raw_control.get("access", {}).get("nonpermitted_value_columns_policy")
        != "never_decode"
        or raw_control.get("access", {}).get("outcome_columns_decoded") is not False
        or raw_control.get("lineage", {}).get("exact_parentage_verified") is not True
    ):
        raise ValueError("raw-control source or outcome-blindness receipt differs")
    mappings = _read_json(root / "modality_mappings.json")
    if (
        mappings.get("s4_sheet_4_accessed") is not False
        or mappings.get("s4_workbook_parsed") is not False
        or mappings.get("perturbation_values_decoded") is not False
        or mappings.get("s4_outcome_values_decoded") is not False
    ):
        raise ValueError("identifier-only mapping receipt differs")
    if contract is not None:
        contract_path = Path(contract).resolve()
        expected_hash = _sha256(contract_path)
        frozen_contract = load_contract(contract_path)
        analysis = _read_json(root / "analysis_manifest.json")
        if analysis.get("contract_sha256") != expected_hash:
            raise ValueError("stage-adapter contract identity differs")
        raw_source = frozen_contract.raw_control_lineage_source
        if (
            raw_control.get("archive", {}).get("sha256") != raw_source.archive.sha256
            or raw_control.get("archive", {}).get("bytes") != raw_source.archive.bytes
            or raw_control.get("extracted", {}).get("file_count")
            != raw_source.extracted.expected_file_count
            or raw_control.get("extracted", {}).get("total_bytes")
            != raw_source.extracted.expected_total_bytes
            or raw_control.get("extracted", {}).get("inventory_sha256")
            != raw_source.extracted.inventory_sha256
        ):
            raise ValueError("raw-control receipt identity differs from contract")
    return {
        "schema_version": VERIFY_VERSION,
        "verified": True,
        "checked_files": len(checked),
        "qualification_status": qualification["qualification_status"],
        "model_launch_status": qualification["model_launch_status"],
        "qualification_sha256": _sha256(root / "qualification.json"),
    }


def _emit(payload: Any, output: str | Path | None = None) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(content)
    else:
        _write_bytes_atomic(Path(output).resolve(), content.encode("utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wormctx.poc.developmental_omix_stage_adapter",
        description="Freeze the outcome-blind OMIX709 semantic-cell stage adapter",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--contract", required=True)
    freeze.add_argument("--parent-manifest", required=True)
    freeze.add_argument("--parent-qualification-root", required=True)
    freeze.add_argument("--source-dir", required=True)
    freeze.add_argument("--raw-control-archive", required=True)
    freeze.add_argument("--raw-control-dir", required=True)
    freeze.add_argument("--output-root", required=True)
    freeze.add_argument("--pilot-overlap-receipt")
    freeze.add_argument("--pilot-overlap-evidence-record")
    freeze.add_argument(
        "--attest-controlled-noncommercial-academic-use",
        action="store_true",
    )
    verify = commands.add_parser("verify")
    verify.add_argument("--run-root", required=True)
    verify.add_argument("--contract")
    verify.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        payload = freeze_stage_adapter(
            args.contract,
            args.parent_manifest,
            args.parent_qualification_root,
            args.source_dir,
            args.raw_control_archive,
            args.raw_control_dir,
            args.output_root,
            academic_use_attested=args.attest_controlled_noncommercial_academic_use,
            pilot_overlap_receipt=args.pilot_overlap_receipt,
            pilot_overlap_evidence_record=args.pilot_overlap_evidence_record,
        )
        _emit(payload)
    else:
        _emit(verify_stage_adapter(args.run_root, args.contract), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
