"""Strict real-data developmental-genetics benchmark for OMIX709.

The module deliberately does not reuse :class:`BiologicalEpisode`: that
contract describes simulated episodes.  OMIX embryos retain source identity,
semantic cell names, whole-embryo split membership, and a separate claims
boundary.  The first executable lane uses the 105-control archive and the
146-embryo RNAi addendum; broader processed tables are frozen as a disabled
scale-up source rather than silently mixed into this analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "wormctx-omix709-developmental-robustness-1.0"
NORMALIZED_VERSION = "wormctx-omix709-normalized-embryos-1.0"
SUMMARY_VERSION = "wormctx-omix709-developmental-summary-1.0"
RAW_HEADER = [
    "",
    "time",
    "cell_name",
    "X",
    "Y",
    "Z",
    "size",
    "raw-expression",
    "blot-correction",
]
CONTROL_NAME = re.compile(r"^ctr_emb([1-9][0-9]*)_raw_data\.txt$")
ADDENDUM_NAME = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9.-]*)_add_emb([1-9][0-9]*)_raw_data\.txt$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceArtifact(_StrictModel):
    role: str
    filename: str
    url: str
    bytes: int = Field(gt=0)
    sha256: str
    expected_extracted_root: str
    expected_extracted_files: int = Field(gt=0)
    expected_extracted_bytes: int = Field(gt=0)

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("source SHA-256 must be 64 lowercase hexadecimal characters")
        return value


class SupplementalArtifact(_StrictModel):
    role: str
    filename: str
    url: str
    bytes: int = Field(gt=0)
    sha256: str
    enabled_in_primary_lane: bool

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("supplemental SHA-256 must be lowercase hexadecimal")
        return value


class RightsContract(_StrictModel):
    access_class: str
    permitted_use: str
    source_redistribution_permitted: bool
    commercial_use_permitted: bool
    public_training_permitted: bool
    operator_attestation_required: bool
    review_contact: str


class DataContract(_StrictModel):
    raw_header: list[str]
    reference_control_embryos: int = Field(gt=0)
    external_control_embryos: int = Field(gt=0)
    perturbation_embryos: int = Field(gt=0)
    perturbation_conditions: int = Field(gt=0)
    total_embryos: int = Field(gt=0)
    condition_counts: dict[str, int]
    time_interval_seconds: int = Field(gt=0)
    stage_landmarks: list[int]
    primary_prefix_stage: int = Field(gt=0)
    endpoint_stage: int = Field(gt=0)

    @model_validator(mode="after")
    def consistent_counts(self) -> "DataContract":
        if self.raw_header != RAW_HEADER:
            raise ValueError("raw OMIX header differs from the frozen contract")
        if self.total_embryos != (
            self.reference_control_embryos
            + self.external_control_embryos
            + self.perturbation_embryos
        ):
            raise ValueError("embryo counts do not sum to total_embryos")
        if self.condition_counts.get("CTR") != self.external_control_embryos:
            raise ValueError("CTR addendum count differs from external controls")
        perturbations = {key: value for key, value in self.condition_counts.items() if key != "CTR"}
        if len(perturbations) != self.perturbation_conditions:
            raise ValueError("perturbation condition count differs from the contract")
        if sum(perturbations.values()) != self.perturbation_embryos:
            raise ValueError("perturbation embryo counts differ from the contract")
        if self.primary_prefix_stage not in self.stage_landmarks:
            raise ValueError("primary prefix stage is not a frozen landmark")
        if self.endpoint_stage not in self.stage_landmarks:
            raise ValueError("endpoint stage is not a frozen landmark")
        if self.stage_landmarks != sorted(set(self.stage_landmarks)):
            raise ValueError("stage landmarks must be unique and sorted")
        return self


class SplitContract(_StrictModel):
    method: str
    indivisible_unit: str
    calibration_source: str
    external_controls_used_for_selection: bool
    gene_identity_features_permitted: bool
    target_representation: str
    fold_local_preprocessing: bool


class EvaluationContract(_StrictModel):
    seed: int
    alpha: float = Field(gt=0.0, lt=1.0)
    ridge_alpha: float = Field(gt=0.0)
    max_threads: int = Field(gt=0, le=4)
    minimum_external_controls: int = Field(gt=0)
    maximum_external_false_positives: int = Field(ge=0)
    minimum_supportive_targets: int = Field(gt=0)


class ClaimContract(_StrictModel):
    scientific_status_values: list[str]
    permitted: list[str]
    prohibited: list[str]


class NormalizedArtifactContract(_StrictModel):
    bytes: int | None = Field(default=None, gt=0)
    sha256: str | None = None

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("normalized SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def both_or_neither(self) -> "NormalizedArtifactContract":
        if (self.bytes is None) != (self.sha256 is None):
            raise ValueError("normalized bytes and SHA-256 must be frozen together")
        return self


class OmixDevelopmentalManifest(_StrictModel):
    schema_version: str
    analysis_id: str
    classification: str
    status: str
    validated: bool
    biological_claims_permitted: bool
    sources: list[SourceArtifact]
    supplemental_processed_bundle: list[SupplementalArtifact]
    rights: RightsContract
    data: DataContract
    split: SplitContract
    evaluation: EvaluationContract
    endpoints: list[dict[str, Any]]
    normalized_artifact: NormalizedArtifactContract
    claims: ClaimContract

    @model_validator(mode="after")
    def exact_lane_shape(self) -> "OmixDevelopmentalManifest":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unknown OMIX developmental schema version")
        if [item.role for item in self.sources] != ["reference_control", "addendum"]:
            raise ValueError("sources must be ordered reference_control, addendum")
        if self.split.method != "leave_one_perturbation_target_out":
            raise ValueError("primary split must hold out perturbation targets")
        if self.split.indivisible_unit != "whole_embryo":
            raise ValueError("whole embryo must be the indivisible unit")
        if self.split.calibration_source != "reference_controls_only":
            raise ValueError("calibration must use reference controls only")
        if self.split.external_controls_used_for_selection:
            raise ValueError("external controls cannot be used for selection")
        if self.split.gene_identity_features_permitted:
            raise ValueError("gene identity features are prohibited in the primary arm")
        if self.split.target_representation != "semantic_cell_id":
            raise ValueError("targets must use semantic cell identifiers")
        if self.validated or self.biological_claims_permitted:
            raise ValueError("this preliminary lane cannot claim validation or biology")
        if self.claims.scientific_status_values != [
            "supportive",
            "not_supportive",
            "not_estimable",
        ]:
            raise ValueError("scientific status vocabulary differs from the contract")
        if any(item.enabled_in_primary_lane for item in self.supplemental_processed_bundle):
            raise ValueError("supplemental processed tables cannot silently enter this lane")
        return self


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


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"output must not already exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_json_atomic(path: Path, payload: Any) -> None:
    _write_bytes_atomic(path, _canonical_bytes(payload))


def load_manifest(path: str | Path) -> OmixDevelopmentalManifest:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"OMIX developmental manifest is not a file: {source}")
    return OmixDevelopmentalManifest.model_validate_json(source.read_text(encoding="utf-8"))


def _manifest(value: OmixDevelopmentalManifest | str | Path | None) -> OmixDevelopmentalManifest | None:
    if value is None or isinstance(value, OmixDevelopmentalManifest):
        return value
    return load_manifest(value)


def verify_source_archive(
    path: str | Path,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source archive is not a regular file: {source}")
    observed_bytes = source.stat().st_size
    if observed_bytes != expected_bytes:
        raise ValueError(
            f"source archive byte identity failed: expected {expected_bytes}, observed {observed_bytes}"
        )
    observed_hash = _sha256(source)
    if observed_hash != expected_sha256.lower():
        raise ValueError(
            f"source archive SHA-256 identity failed: expected {expected_sha256.lower()}, "
            f"observed {observed_hash}"
        )
    return {
        "verified": True,
        "bytes": observed_bytes,
        "sha256": observed_hash,
        "filename": source.name,
    }


def verify_sources(
    manifest_or_path: OmixDevelopmentalManifest | str | Path,
    control_archive: str | Path,
    addendum_archive: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    manifest = _manifest(manifest_or_path)
    assert manifest is not None
    paths = {
        "reference_control": Path(control_archive),
        "addendum": Path(addendum_archive),
    }
    receipts: dict[str, dict[str, Any]] = {}
    for contract in manifest.sources:
        receipt = verify_source_archive(paths[contract.role], contract.bytes, contract.sha256)
        if receipt["filename"] != contract.filename:
            raise ValueError(f"{contract.role} archive filename differs from the frozen identity")
        receipts[contract.role] = receipt
    result = {
        "schema_version": "wormctx-omix709-source-verification-1.0",
        "verified": True,
        "sources": receipts,
    }
    if output is not None:
        _write_json_atomic(Path(output).resolve(), result)
    return result


def _finite_float(value: str, label: str, source: Path, line: int) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{source.name}:{line}: {label} is not numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{source.name}:{line}: {label} is not finite")
    return parsed


def _parse_time(value: str, source: Path, line: int) -> int:
    parsed = _finite_float(value, "time", source, line)
    if parsed < 0 or not parsed.is_integer():
        raise ValueError(f"{source.name}:{line}: time must be a nonnegative integer")
    return int(parsed)


def _parse_embryo_file(path: Path, stage_landmarks: Sequence[int]) -> dict[str, Any]:
    births: dict[str, dict[str, Any]] = {}
    last_time: dict[str, int] = {}
    seen_cell_times: set[tuple[int, str]] = set()
    seen_indices: set[int] = set()
    frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
    row_count = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty OMIX table: {path}") from exc
        if header != RAW_HEADER:
            raise ValueError(f"{path.name}: table header differs from the OMIX contract")
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(RAW_HEADER):
                raise ValueError(f"{path.name}:{line_number}: expected nine tab-separated fields")
            try:
                index = int(row[0])
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_number}: row index is not an integer") from exc
            if index < 0 or index in seen_indices:
                raise ValueError(f"{path.name}:{line_number}: duplicate or negative row index")
            seen_indices.add(index)
            time = _parse_time(row[1], path, line_number)
            cell_id = row[2].strip()
            if not cell_id or len(cell_id) > 128 or any(ord(char) < 32 for char in cell_id):
                raise ValueError(f"{path.name}:{line_number}: invalid semantic cell identifier")
            key = (time, cell_id)
            if key in seen_cell_times:
                raise ValueError(
                    f"{path.name}:{line_number}: duplicate semantic cell at one time: {cell_id}/{time}"
                )
            seen_cell_times.add(key)
            observation = {
                "time": time,
                "cell_id": cell_id,
                "x": _finite_float(row[3], "X", path, line_number),
                "y": _finite_float(row[4], "Y", path, line_number),
                "z": _finite_float(row[5], "Z", path, line_number),
                "size": _finite_float(row[6], "size", path, line_number),
                "raw_expression": _finite_float(row[7], "raw-expression", path, line_number),
                "corrected_expression": _finite_float(
                    row[8], "blot-correction", path, line_number
                ),
            }
            frames[time].append(observation)
            if cell_id not in births or time < births[cell_id]["time"]:
                births[cell_id] = dict(observation)
            last_time[cell_id] = max(time, last_time.get(cell_id, time))
            row_count += 1

    if row_count == 0:
        raise ValueError(f"OMIX table contains no observations: {path}")
    for cell_id, item in births.items():
        item["last_time"] = last_time[cell_id]

    stage_times: dict[str, int] = {}
    stage_observations: dict[str, list[dict[str, Any]]] = {}
    for stage in sorted(set(stage_landmarks)):
        matches = [time for time, values in frames.items() if len(values) >= stage]
        if matches:
            stage_time = min(matches)
            stage_times[str(stage)] = stage_time
            stage_observations[str(stage)] = sorted(
                (dict(item) for item in frames[stage_time]), key=lambda item: item["cell_id"]
            )

    if "ABa" in last_time and "ABp" in last_time:
        anchor_time = min(last_time["ABa"], last_time["ABp"]) + 1
    else:
        anchor_time = min(item["time"] for item in births.values())
    for item in births.values():
        item["relative_birth_time"] = item["time"] - anchor_time
        item["relative_last_time"] = item["last_time"] - anchor_time

    return {
        "rows": row_count,
        "unique_cells": len(births),
        "max_time": max(frames),
        "max_cells_per_frame": max(len(values) for values in frames.values()),
        "anchor_time": anchor_time,
        "stage_times": stage_times,
        "observations": sorted(births.values(), key=lambda item: item["cell_id"]),
        "stage_observations": stage_observations,
    }


def _inventory_files(root: Path, kind: str) -> list[tuple[Path, str, int]]:
    if not root.is_dir():
        raise FileNotFoundError(f"{kind} extraction root is not a directory: {root}")
    items: list[tuple[Path, str, int]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            raise ValueError(f"{kind} extraction root contains a non-file member: {path.name}")
        match = CONTROL_NAME.fullmatch(path.name) if kind == "control" else ADDENDUM_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"{kind} extraction filename violates the frozen grammar: {path.name}")
        condition = "REFERENCE_CTR" if kind == "control" else match.group(1)
        serial = int(match.group(1) if kind == "control" else match.group(2))
        items.append((path, condition, serial))
    if not items:
        raise ValueError(f"{kind} extraction root contains no embryo tables")
    return items


def _source_receipt_payload(source_receipts: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for role in sorted(source_receipts):
        item = source_receipts[role]
        if not isinstance(item, Mapping):
            raise ValueError("source receipt must map source role to a receipt object")
        if item.get("verified") is not True:
            raise ValueError(f"source receipt for {role} is not verified")
        observed_hash = str(item.get("sha256", "")).lower()
        observed_bytes = item.get("bytes")
        if not SHA256_RE.fullmatch(observed_hash) or not isinstance(observed_bytes, int):
            raise ValueError(f"source receipt for {role} has malformed identity fields")
        payload[role] = {
            "verified": True,
            "bytes": observed_bytes,
            "sha256": observed_hash,
            "filename": str(item.get("filename", "")),
        }
    if not payload:
        raise ValueError("at least one verified source receipt is required")
    return payload


def normalize_dataset(
    control_dir: str | Path,
    addendum_dir: str | Path,
    output_path: str | Path,
    source_receipts: Mapping[str, Any],
    manifest: OmixDevelopmentalManifest | str | Path | None = None,
) -> dict[str, Any]:
    """Normalize extracted OMIX tables into deterministic, compact embryo records.

    One birth observation per semantic cell is retained, plus stage snapshots.
    This preserves the developmental signal needed by the pilot without
    redistributing every repeated cell/time row.
    """

    contract = _manifest(manifest)
    stages = contract.data.stage_landmarks if contract is not None else [26, 200]
    controls = _inventory_files(Path(control_dir).resolve(), "control")
    addendum = _inventory_files(Path(addendum_dir).resolve(), "addendum")

    if contract is not None:
        expected = {item.role: item for item in contract.sources}
        if len(controls) != expected["reference_control"].expected_extracted_files:
            raise ValueError("reference-control extracted file count differs from the contract")
        if len(addendum) != expected["addendum"].expected_extracted_files:
            raise ValueError("addendum extracted file count differs from the contract")
        if sum(item[0].stat().st_size for item in controls) != expected["reference_control"].expected_extracted_bytes:
            raise ValueError("reference-control extracted byte count differs from the contract")
        if sum(item[0].stat().st_size for item in addendum) != expected["addendum"].expected_extracted_bytes:
            raise ValueError("addendum extracted byte count differs from the contract")

    records: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    identities: set[str] = set()
    serials: dict[str, set[int]] = defaultdict(set)
    for cohort, items in (("reference", controls), ("addendum", addendum)):
        for path, condition, serial in items:
            if serial in serials[condition]:
                raise ValueError(f"duplicate embryo serial for condition {condition}: {serial}")
            serials[condition].add(serial)
            if cohort == "reference":
                role = "reference_control"
                embryo_id = f"reference:ctr_emb{serial}"
                output_condition = "REFERENCE_CTR"
            elif condition == "CTR":
                role = "external_control"
                embryo_id = f"addendum:CTR_emb{serial}"
                output_condition = "CTR"
            else:
                role = "perturbation"
                embryo_id = f"addendum:{condition}_emb{serial}"
                output_condition = condition
            if embryo_id in identities:
                raise ValueError(f"duplicate whole-embryo identity: {embryo_id}")
            identities.add(embryo_id)
            parsed = _parse_embryo_file(path, stages)
            source_hash = _sha256(path)
            record = {
                "embryo_id": embryo_id,
                "condition": output_condition,
                "role": role,
                "source_member": path.name,
                "source_bytes": path.stat().st_size,
                "source_sha256": source_hash,
                **parsed,
            }
            records.append(record)
            inventory.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": source_hash}
            )

    records.sort(key=lambda item: item["embryo_id"])
    inventory.sort(key=lambda item: item["name"])
    payload = {
        "schema_version": NORMALIZED_VERSION,
        "source_receipts": _source_receipt_payload(source_receipts),
        "extracted_inventory_sha256": hashlib.sha256(_canonical_bytes(inventory)).hexdigest(),
        "embryos": records,
    }
    output = Path(output_path).resolve()
    _write_json_atomic(output, payload)
    counts = Counter(item["role"] for item in records)
    conditions = Counter(item["condition"] for item in records if item["role"] != "reference_control")
    return {
        "schema_version": "wormctx-omix709-normalization-receipt-1.0",
        "qualified": True,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "embryos": len(records),
        "roles": dict(sorted(counts.items())),
        "conditions": dict(sorted(conditions.items())),
        "extracted_inventory_sha256": payload["extracted_inventory_sha256"],
    }


def normalize(
    manifest_or_path: OmixDevelopmentalManifest | str | Path,
    control_dir: str | Path,
    addendum_dir: str | Path,
    output_path: str | Path,
    source_receipts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = _manifest(manifest_or_path)
    assert manifest is not None
    if source_receipts is None:
        source_receipts = {
            item.role: {
                "verified": True,
                "bytes": item.bytes,
                "sha256": item.sha256,
                "filename": item.filename,
            }
            for item in manifest.sources
        }
    return normalize_dataset(control_dir, addendum_dir, output_path, source_receipts, manifest)


def _load_normalized(path: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"normalized OMIX snapshot is not a file: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"normalized OMIX snapshot is not valid JSON: {source}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != NORMALIZED_VERSION:
        raise ValueError("normalized OMIX schema version is missing or wrong")
    records = payload.get("embryos")
    if not isinstance(records, list) or not records:
        raise ValueError("normalized OMIX snapshot has no embryo records")
    return source, payload, records


def _validate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    embryo_ids: set[str] = set()
    roles = Counter()
    conditions: set[str] = set()
    for record in records:
        embryo_id = record.get("embryo_id")
        condition = record.get("condition")
        role = record.get("role")
        observations = record.get("observations")
        if not isinstance(embryo_id, str) or not embryo_id or embryo_id in embryo_ids:
            raise ValueError("normalized records contain an invalid or duplicate embryo identity")
        embryo_ids.add(embryo_id)
        if role not in {"reference_control", "external_control", "perturbation"}:
            raise ValueError(f"{embryo_id}: unknown embryo role")
        if not isinstance(condition, str) or not condition:
            raise ValueError(f"{embryo_id}: missing condition")
        if role == "reference_control" and condition != "REFERENCE_CTR":
            raise ValueError(f"{embryo_id}: reference control condition is wrong")
        if role == "external_control" and condition != "CTR":
            raise ValueError(f"{embryo_id}: external control condition is wrong")
        if role == "perturbation" and condition in {"CTR", "REFERENCE_CTR"}:
            raise ValueError(f"{embryo_id}: perturbation has a control condition")
        if not isinstance(observations, list) or not observations:
            raise ValueError(f"{embryo_id}: embryo has no semantic-cell observations")
        seen_cells: set[str] = set()
        for observation in observations:
            if not isinstance(observation, Mapping):
                raise ValueError(f"{embryo_id}: malformed observation")
            cell_id = observation.get("cell_id")
            if not isinstance(cell_id, str) or not cell_id or cell_id in seen_cells:
                raise ValueError(f"{embryo_id}: duplicate or invalid semantic cell ID")
            seen_cells.add(cell_id)
            for field in (
                "time",
                "x",
                "y",
                "z",
                "size",
                "raw_expression",
                "corrected_expression",
            ):
                value = observation.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"{embryo_id}/{cell_id}: {field} is not finite")
        roles[role] += 1
        if role == "perturbation":
            conditions.add(condition)
    return {
        "total_embryos": len(records),
        "reference_control_embryos": roles["reference_control"],
        "external_control_embryos": roles["external_control"],
        "perturbation_embryos": roles["perturbation"],
        "perturbation_conditions": len(conditions),
    }


def qualify_dataset(
    normalized_path: str | Path,
    expected_counts: Mapping[str, int] | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    source, payload, records = _load_normalized(normalized_path)
    observed_hash = _sha256(source)
    if expected_sha256 is not None and observed_hash != expected_sha256.lower():
        raise ValueError("normalized snapshot SHA-256 identity failed")
    counts = _validate_records(records)
    if expected_counts is not None:
        for key, expected in expected_counts.items():
            if key not in counts or counts[key] != expected:
                raise ValueError(
                    f"normalized {key} count differs: expected {expected}, observed {counts.get(key)}"
                )
    source_receipts = payload.get("source_receipts")
    if not isinstance(source_receipts, Mapping):
        raise ValueError("normalized snapshot lacks source receipts")
    _source_receipt_payload(source_receipts)
    return {
        "schema_version": "wormctx-omix709-qualification-1.0",
        "qualified": True,
        "bytes": source.stat().st_size,
        "sha256": observed_hash,
        "counts": counts,
        "extracted_inventory_sha256": payload.get("extracted_inventory_sha256"),
    }


def qualify(
    manifest_or_path: OmixDevelopmentalManifest | str | Path,
    normalized_path: str | Path,
) -> dict[str, Any]:
    manifest = _manifest(manifest_or_path)
    assert manifest is not None
    expected_counts = {
        "total_embryos": manifest.data.total_embryos,
        "reference_control_embryos": manifest.data.reference_control_embryos,
        "external_control_embryos": manifest.data.external_control_embryos,
        "perturbation_embryos": manifest.data.perturbation_embryos,
        "perturbation_conditions": manifest.data.perturbation_conditions,
    }
    receipt = qualify_dataset(
        normalized_path,
        expected_counts=expected_counts,
        expected_sha256=manifest.normalized_artifact.sha256,
    )
    if (
        manifest.normalized_artifact.bytes is not None
        and receipt["bytes"] != manifest.normalized_artifact.bytes
    ):
        raise ValueError("normalized snapshot byte identity differs from the manifest")
    return receipt


def build_leave_one_condition_out_folds(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build deterministic whole-embryo, whole-perturbation outer folds."""

    _validate_records(records)
    perturbations = [item for item in records if item["role"] == "perturbation"]
    reference_ids = sorted(
        item["embryo_id"] for item in records if item["role"] == "reference_control"
    )
    conditions = sorted({str(item["condition"]) for item in perturbations})
    folds: list[dict[str, Any]] = []
    seen_test: set[str] = set()
    for condition in conditions:
        test_ids = sorted(
            item["embryo_id"] for item in perturbations if item["condition"] == condition
        )
        train_ids = reference_ids + sorted(
            item["embryo_id"] for item in perturbations if item["condition"] != condition
        )
        if not test_ids or set(train_ids).intersection(test_ids):
            raise ValueError("whole-condition split construction failed")
        if seen_test.intersection(test_ids):
            raise ValueError("an embryo appears in more than one outer test fold")
        seen_test.update(test_ids)
        folds.append(
            {
                "held_out_condition": condition,
                "train_embryo_ids": train_ids,
                "test_embryo_ids": test_ids,
            }
        )
    if seen_test != {item["embryo_id"] for item in perturbations}:
        raise ValueError("outer folds do not cover every perturbation embryo exactly once")
    return folds


def calibrate_control_threshold(
    control_scores: Mapping[str, float] | Sequence[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if isinstance(control_scores, Mapping):
        values = [float(control_scores[key]) for key in sorted(control_scores)]
    else:
        values = [float(item) for item in control_scores]
    if not values or any(not math.isfinite(item) for item in values):
        raise ValueError("control score set must be nonempty and finite")
    ordered = sorted(values)
    rank = min(len(ordered), max(1, math.ceil((len(ordered) + 1) * (1.0 - alpha))))
    return {
        "calibration_source": "reference_controls_only",
        "n_controls": len(ordered),
        "alpha": alpha,
        "threshold": ordered[rank - 1],
        "order_statistic_rank": rank,
        "quantile_method": "finite_sample_upper_order_statistic_ceil_(n+1)*(1-alpha)",
    }


def _prefix_observations(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    observations = list(record["observations"])
    stage_times = record.get("stage_times")
    if isinstance(stage_times, Mapping) and "26" in stage_times:
        cutoff = int(stage_times["26"])
        return [item for item in observations if int(item["time"]) <= cutoff]
    return observations


def build_primary_features(
    records: Sequence[Mapping[str, Any]],
    semantic_cell_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build target-blind, order-invariant semantic-cell prefix features."""

    if not records:
        raise ValueError("at least one embryo is required for primary features")
    _validate_records(records)
    ordered_records = sorted(records, key=lambda item: item["embryo_id"])
    if semantic_cell_ids is None:
        cells = sorted(
            {
                str(observation["cell_id"])
                for record in ordered_records
                for observation in _prefix_observations(record)
            }
        )
    else:
        cells = sorted(set(semantic_cell_ids))
    if not cells:
        raise ValueError("primary feature vocabulary contains no semantic cells")
    fields = (
        "relative_birth_time",
        "x",
        "y",
        "z",
        "size",
        "log_corrected_expression",
    )
    feature_names = [
        f"semantic_cell__{cell_id}__{field}" for cell_id in cells for field in fields
    ]
    matrix: list[list[float]] = []
    for record in ordered_records:
        by_cell = {str(item["cell_id"]): item for item in _prefix_observations(record)}
        anchor = float(record.get("anchor_time", 0.0))
        row: list[float] = []
        for cell_id in cells:
            item = by_cell.get(cell_id)
            if item is None:
                row.extend([0.0] * len(fields))
                continue
            relative = float(item.get("relative_birth_time", float(item["time"]) - anchor))
            corrected = max(0.0, float(item["corrected_expression"]))
            row.extend(
                [
                    relative,
                    float(item["x"]),
                    float(item["y"]),
                    float(item["z"]),
                    float(item["size"]),
                    math.log1p(corrected),
                ]
            )
        matrix.append(row)
    return {
        "embryo_ids": [item["embryo_id"] for item in ordered_records],
        "feature_names": feature_names,
        "matrix": matrix,
        "semantic_cell_ids": cells,
    }


def _robust_location_scale(values: Iterable[float], floor: float) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("reference values must be finite and nonempty")
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center)))
    scale = max(floor, 1.4826 * mad)
    return center, scale


def _timing_atlas(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    by_cell: dict[str, list[float]] = defaultdict(list)
    for record in records:
        anchor = float(record.get("anchor_time", 0.0))
        endpoint_time = None
        if isinstance(record.get("stage_times"), Mapping):
            endpoint_time = record["stage_times"].get("200")
        for item in record["observations"]:
            if endpoint_time is not None and float(item["time"]) > float(endpoint_time):
                continue
            relative = float(item.get("relative_birth_time", float(item["time"]) - anchor))
            by_cell[str(item["cell_id"])].append(relative)
    minimum = max(2, math.ceil(0.5 * len(records)))
    atlas: dict[str, dict[str, float]] = {}
    for cell_id, values in sorted(by_cell.items()):
        if len(values) < minimum:
            continue
        center, scale = _robust_location_scale(values, 1.0)
        atlas[cell_id] = {"center": center, "scale": scale, "support": len(values)}
    return atlas


def _timing_score(
    record: Mapping[str, Any], atlas: Mapping[str, Mapping[str, float]]
) -> dict[str, Any]:
    anchor = float(record.get("anchor_time", 0.0))
    deviations: list[dict[str, Any]] = []
    for item in record["observations"]:
        cell_id = str(item["cell_id"])
        reference = atlas.get(cell_id)
        if reference is None:
            continue
        relative = float(item.get("relative_birth_time", float(item["time"]) - anchor))
        signed = (relative - float(reference["center"])) / float(reference["scale"])
        deviations.append(
            {
                "cell_id": cell_id,
                "relative_birth_time": relative,
                "signed_z": signed,
                "absolute_z": abs(signed),
                "reference_support": int(reference["support"]),
            }
        )
    if not deviations:
        return {"score": None, "rms": None, "supported_cells": 0, "deviations": []}
    return {
        "score": max(item["absolute_z"] for item in deviations),
        "rms": math.sqrt(sum(item["signed_z"] ** 2 for item in deviations) / len(deviations)),
        "supported_cells": len(deviations),
        "deviations": sorted(
            deviations, key=lambda item: (item["relative_birth_time"], item["cell_id"])
        ),
    }


def _control_loo_scores(reference: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    if len(reference) < 3:
        return scores
    for excluded in reference:
        atlas = _timing_atlas(
            [item for item in reference if item["embryo_id"] != excluded["embryo_id"]]
        )
        result = _timing_score(excluded, atlas)
        if result["score"] is not None:
            scores[str(excluded["embryo_id"])] = float(result["score"])
    return scores


def _stage_expression(record: Mapping[str, Any], stage: int = 200) -> dict[str, float]:
    snapshots = record.get("stage_observations")
    if not isinstance(snapshots, Mapping) or not isinstance(snapshots.get(str(stage)), list):
        return {}
    raw = {
        str(item["cell_id"]): math.log1p(max(0.0, float(item["corrected_expression"])))
        for item in snapshots[str(stage)]
    }
    if not raw:
        return {}
    center = float(np.median(np.asarray(list(raw.values()), dtype=np.float64)))
    return {cell_id: value - center for cell_id, value in raw.items()}


def _expression_atlas(reference: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in reference:
        for cell_id, value in _stage_expression(record).items():
            values[cell_id].append(value)
    minimum = max(2, math.ceil(0.5 * len(reference)))
    result: dict[str, dict[str, float]] = {}
    for cell_id, observed in sorted(values.items()):
        if len(observed) < minimum:
            continue
        center, scale = _robust_location_scale(observed, 0.05)
        result[cell_id] = {"center": center, "scale": scale, "support": len(observed)}
    return result


def _expression_burden(
    record: Mapping[str, Any], atlas: Mapping[str, Mapping[str, float]]
) -> float | None:
    values: list[float] = []
    for cell_id, observed in _stage_expression(record).items():
        reference = atlas.get(cell_id)
        if reference is None:
            continue
        values.append((observed - float(reference["center"])) / float(reference["scale"]))
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _flat_features(record: Mapping[str, Any]) -> list[float]:
    observations = _prefix_observations(record)
    anchor = float(record.get("anchor_time", 0.0))
    timing = np.asarray(
        [float(item.get("relative_birth_time", float(item["time"]) - anchor)) for item in observations],
        dtype=np.float64,
    )
    size = np.asarray([float(item["size"]) for item in observations], dtype=np.float64)
    expression = np.asarray(
        [math.log1p(max(0.0, float(item["corrected_expression"]))) for item in observations],
        dtype=np.float64,
    )
    coords = np.asarray(
        [[float(item["x"]), float(item["y"]), float(item["z"])] for item in observations],
        dtype=np.float64,
    )
    return [
        float(len(observations)),
        float(np.mean(timing)),
        float(np.std(timing)),
        float(np.mean(size)),
        float(np.std(size)),
        float(np.mean(expression)),
        float(np.std(expression)),
        float(np.std(coords[:, 0])),
        float(np.std(coords[:, 1])),
        float(np.std(coords[:, 2])),
    ]


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean_x = np.mean(train_x, axis=0)
    scale_x = np.std(train_x, axis=0)
    scale_x[scale_x < 1e-12] = 1.0
    x = (train_x - mean_x) / scale_x
    z = (test_x - mean_x) / scale_x
    mean_y = float(np.mean(train_y))
    centered_y = train_y - mean_y
    if x.shape[1] > x.shape[0]:
        coefficients = x.T @ np.linalg.solve(
            x @ x.T + alpha * np.eye(x.shape[0], dtype=np.float64), centered_y
        )
    else:
        coefficients = np.linalg.solve(
            x.T @ x + alpha * np.eye(x.shape[1], dtype=np.float64), x.T @ centered_y
        )
    return mean_y + z @ coefficients


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _prediction_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mae": None, "rmse": None, "spearman": None, "target_macro_mae": None, "target_macro_rmse": None}
    observed = np.asarray([float(item["observed"]) for item in rows], dtype=np.float64)
    predicted = np.asarray([float(item[key]) for item in rows], dtype=np.float64)
    errors = predicted - observed
    target_mae: list[float] = []
    target_rmse: list[float] = []
    for condition in sorted({str(item["condition"]) for item in rows}):
        subset = [item for item in rows if item["condition"] == condition]
        delta = np.asarray([float(item[key]) - float(item["observed"]) for item in subset])
        target_mae.append(float(np.mean(np.abs(delta))))
        target_rmse.append(float(math.sqrt(float(np.mean(delta * delta)))))
    spearman: float | None = None
    if observed.size >= 3 and np.std(observed) > 0 and np.std(predicted) > 0:
        correlation = np.corrcoef(_average_ranks(observed), _average_ranks(predicted))[0, 1]
        if math.isfinite(float(correlation)):
            spearman = float(correlation)
    return {
        "n": len(rows),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(math.sqrt(float(np.mean(errors * errors)))),
        "spearman": spearman,
        "target_macro_mae": float(np.mean(target_mae)),
        "target_macro_rmse": float(np.mean(target_rmse)),
    }


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    content = b"".join(_canonical_bytes(dict(item)) for item in rows)
    _write_bytes_atomic(path, content)


def _write_checksums(run_root: Path) -> None:
    candidates = sorted(
        path
        for path in run_root.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS.txt", "SUCCESS", "FAILURE"}
        and ".tmp-" not in path.name
    )
    lines = [f"{_sha256(path)}  {path.relative_to(run_root).as_posix()}\n" for path in candidates]
    _write_bytes_atomic(run_root / "SHA256SUMS.txt", "".join(lines).encode("utf-8"))


def _model_predictions(
    records: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, float | None],
    folds: Sequence[Mapping[str, Any]],
    alpha: float,
    seed: int,
) -> list[dict[str, Any]]:
    reference_cells = sorted(
        {
            str(item["cell_id"])
            for record in records
            if record["role"] == "reference_control"
            for item in _prefix_observations(record)
        }
    )
    features = build_primary_features(records, semantic_cell_ids=reference_cells)
    feature_by_id = {
        embryo_id: np.asarray(row, dtype=np.float64)
        for embryo_id, row in zip(features["embryo_ids"], features["matrix"], strict=True)
    }
    flat_by_id = {
        str(record["embryo_id"]): np.asarray(_flat_features(record), dtype=np.float64)
        for record in records
    }
    condition_by_id = {str(item["embryo_id"]): str(item["condition"]) for item in records}
    predictions: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        train_ids = [
            item
            for item in fold["train_embryo_ids"]
            if outcomes.get(str(item)) is not None and str(item) in feature_by_id
        ]
        test_ids = [
            item
            for item in fold["test_embryo_ids"]
            if outcomes.get(str(item)) is not None and str(item) in feature_by_id
        ]
        if len(train_ids) < 3 or not test_ids:
            continue
        train_y = np.asarray([float(outcomes[str(item)]) for item in train_ids], dtype=np.float64)
        lineage_train = np.vstack([feature_by_id[str(item)] for item in train_ids])
        lineage_test = np.vstack([feature_by_id[str(item)] for item in test_ids])
        flat_train = np.vstack([flat_by_id[str(item)] for item in train_ids])
        flat_test = np.vstack([flat_by_id[str(item)] for item in test_ids])
        constant = np.full(len(test_ids), float(np.mean(train_y)), dtype=np.float64)
        flat = _ridge_predict(flat_train, train_y, flat_test, alpha)
        lineage = _ridge_predict(lineage_train, train_y, lineage_test, alpha)

        # A topology-negative control: independently permuting semantic-cell
        # blocks within each embryo preserves values and sample membership but
        # breaks canonical lineage correspondence.  It never informs fitting.
        block = 6
        cell_count = lineage_train.shape[1] // block
        rng = np.random.Generator(np.random.PCG64DXSM(seed + fold_index))
        shuffled_train = lineage_train.reshape(len(train_ids), cell_count, block).copy()
        shuffled_test = lineage_test.reshape(len(test_ids), cell_count, block).copy()
        for row in shuffled_train:
            row[:] = row[rng.permutation(cell_count)]
        for row in shuffled_test:
            row[:] = row[rng.permutation(cell_count)]
        shuffled = _ridge_predict(
            shuffled_train.reshape(len(train_ids), -1),
            train_y,
            shuffled_test.reshape(len(test_ids), -1),
            alpha,
        )
        for index, embryo_id in enumerate(test_ids):
            predictions.append(
                {
                    "embryo_id": str(embryo_id),
                    "condition": condition_by_id[str(embryo_id)],
                    "held_out_condition": str(fold["held_out_condition"]),
                    "observed": float(outcomes[str(embryo_id)]),
                    "constant_prediction": float(constant[index]),
                    "flat_prediction": float(flat[index]),
                    "lineage_prediction": float(lineage[index]),
                    "topology_shuffle_prediction": float(shuffled[index]),
                }
            )
    return sorted(predictions, key=lambda item: item["embryo_id"])


def run_analysis(
    normalized_path: str | Path,
    output_dir: str | Path,
    seed: int = 1729,
    manifest: OmixDevelopmentalManifest | str | Path | None = None,
    threads: int = 2,
) -> dict[str, Any]:
    """Run the frozen conventional developmental benchmark.

    Technical success records a valid execution, not a favorable scientific
    result.  Small fixtures without 200-cell snapshots therefore complete as
    ``not_estimable`` rather than failing engineering validation.
    """

    contract = _manifest(manifest)
    if threads < 1 or threads > (contract.evaluation.max_threads if contract else 4):
        raise ValueError("thread count is outside the developmental-lane contract")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    source, _, records = _load_normalized(normalized_path)
    counts = _validate_records(records)
    run_root = Path(output_dir).resolve()
    if run_root.exists():
        raise FileExistsError(f"run output must not already exist: {run_root}")
    run_root.mkdir(parents=True)
    (run_root / "scratch").mkdir()

    reference = [item for item in records if item["role"] == "reference_control"]
    external = [item for item in records if item["role"] == "external_control"]
    perturbations = [item for item in records if item["role"] == "perturbation"]
    timing_atlas = _timing_atlas(reference)
    loo_scores = _control_loo_scores(reference)
    alpha = contract.evaluation.alpha if contract else 0.05
    threshold = calibrate_control_threshold(loo_scores, alpha) if loo_scores else None

    deviations: list[dict[str, Any]] = []
    score_by_id: dict[str, float | None] = {}
    for record in records:
        result = _timing_score(record, timing_atlas)
        score_by_id[str(record["embryo_id"])] = result["score"]
        earliest = None
        crosses = False
        if threshold is not None and result["score"] is not None:
            crosses = float(result["score"]) >= float(threshold["threshold"])
            qualifying = [
                item
                for item in result["deviations"]
                if item["absolute_z"] >= float(threshold["threshold"])
            ]
            if qualifying:
                earliest = qualifying[0]
        deviations.append(
            {
                "embryo_id": record["embryo_id"],
                "condition": record["condition"],
                "role": record["role"],
                "timing_max_absolute_z": result["score"],
                "timing_rms_z": result["rms"],
                "supported_cells": result["supported_cells"],
                "crosses_control_threshold": crosses,
                "earliest_detected_deviation": earliest,
            }
        )

    expression_atlas = _expression_atlas(reference)
    outcomes = {
        str(record["embryo_id"]): _expression_burden(record, expression_atlas)
        for record in records
    }
    folds = build_leave_one_condition_out_folds(records)
    ridge_alpha = contract.evaluation.ridge_alpha if contract else 1.0
    predictions = _model_predictions(records, outcomes, folds, ridge_alpha, seed)
    metric_keys = {
        "constant": "constant_prediction",
        "flat": "flat_prediction",
        "lineage": "lineage_prediction",
        "topology_shuffle": "topology_shuffle_prediction",
    }
    metrics = {name: _prediction_metrics(predictions, key) for name, key in metric_keys.items()}

    external_crosses = sum(
        1
        for item in deviations
        if item["role"] == "external_control" and item["crosses_control_threshold"]
    )
    external_rate = external_crosses / len(external) if external else None
    target_detection: dict[str, float] = {}
    for condition in sorted({str(item["condition"]) for item in perturbations}):
        rows = [
            item
            for item in deviations
            if item["role"] == "perturbation" and item["condition"] == condition
        ]
        target_detection[condition] = (
            sum(bool(item["crosses_control_threshold"]) for item in rows) / len(rows)
        )
    target_macro_detection = (
        float(np.mean(list(target_detection.values()))) if target_detection else None
    )

    scientific_status = "not_estimable"
    scientific_gates: dict[str, bool | None] = {
        "external_control_gate": None,
        "perturbation_detection_gate": None,
        "lineage_over_constant_gate": None,
        "lineage_over_flat_gate": None,
        "lineage_over_topology_shuffle_gate": None,
    }
    expected_external = contract.evaluation.minimum_external_controls if contract else 2
    max_external_fp = contract.evaluation.maximum_external_false_positives if contract else 1
    minimum_targets = contract.evaluation.minimum_supportive_targets if contract else 5
    prediction_estimable = bool(
        predictions
        and metrics["lineage"]["target_macro_rmse"] is not None
        and metrics["constant"]["target_macro_rmse"] is not None
        and metrics["flat"]["target_macro_rmse"] is not None
        and metrics["topology_shuffle"]["target_macro_rmse"] is not None
    )
    detection_estimable = threshold is not None and len(external) >= expected_external
    if detection_estimable:
        scientific_gates["external_control_gate"] = external_crosses <= max_external_fp
        scientific_gates["perturbation_detection_gate"] = bool(
            target_macro_detection is not None
            and external_rate is not None
            and target_macro_detection > external_rate
            and sum(rate >= 0.5 for rate in target_detection.values()) >= minimum_targets
        )
    if prediction_estimable:
        lineage_rmse = float(metrics["lineage"]["target_macro_rmse"])
        scientific_gates["lineage_over_constant_gate"] = (
            lineage_rmse < float(metrics["constant"]["target_macro_rmse"])
        )
        scientific_gates["lineage_over_flat_gate"] = (
            lineage_rmse < float(metrics["flat"]["target_macro_rmse"])
        )
        scientific_gates["lineage_over_topology_shuffle_gate"] = (
            lineage_rmse < float(metrics["topology_shuffle"]["target_macro_rmse"])
        )
    if detection_estimable and prediction_estimable:
        scientific_status = (
            "supportive" if all(value is True for value in scientific_gates.values()) else "not_supportive"
        )

    summary = {
        "schema_version": SUMMARY_VERSION,
        "technical_status": "success",
        "scientific_status": scientific_status,
        "analysis_scope": "preliminary_real_developmental_genetics_feasibility",
        "validated": False,
        "biological_claims_permitted": False,
        "seed": seed,
        "threads": threads,
        "normalized_sha256": _sha256(source),
        "counts": counts,
        "split_method": "leave_one_perturbation_target_out",
        "whole_embryo_isolation": True,
        "gene_identity_features_permitted": False,
        "calibration": threshold,
        "timing_atlas_cells": len(timing_atlas),
        "expression_atlas_cells": len(expression_atlas),
        "external_control_false_positives": external_crosses,
        "external_control_false_positive_rate": external_rate,
        "target_detection_rates": target_detection,
        "target_macro_detection_rate": target_macro_detection,
        "prediction_metrics": metrics,
        "scientific_gates": scientific_gates,
        "interpretation": (
            "Earliest detected deviation is assay-relative and retrospective, not a causal event; "
            "CND-1 is a reporter rather than a complete terminal-fate label."
        ),
    }

    _write_json_atomic(run_root / "folds.json", list(folds))
    _write_jsonl_atomic(run_root / "deviations.jsonl", deviations)
    _write_jsonl_atomic(run_root / "predictions.jsonl", predictions)
    _write_json_atomic(
        run_root / "reference_atlas_receipt.json",
        {
            "calibration_source": "reference_controls_only",
            "reference_embryos": len(reference),
            "timing_cells": len(timing_atlas),
            "expression_cells": len(expression_atlas),
            "control_loo_scores": dict(sorted(loo_scores.items())),
        },
    )
    _write_json_atomic(run_root / "omix709_developmental_summary.json", summary)
    _write_checksums(run_root)
    _write_bytes_atomic(run_root / "SUCCESS", b"SUCCESS\n")
    return summary


def run(
    manifest_or_path: OmixDevelopmentalManifest | str | Path,
    normalized_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int | None = None,
    threads: int = 2,
) -> dict[str, Any]:
    contract = _manifest(manifest_or_path)
    assert contract is not None
    qualify(contract, normalized_path)
    return run_analysis(
        normalized_path,
        output_dir,
        seed=contract.evaluation.seed if seed is None else seed,
        manifest=contract,
        threads=threads,
    )


def verify_run(
    run_dir: str | Path,
    manifest: OmixDevelopmentalManifest | str | Path | None = None,
) -> dict[str, Any]:
    contract = _manifest(manifest)
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"developmental run root is not a directory: {root}")
    success = root / "SUCCESS"
    if success.read_bytes() != b"SUCCESS\n" or (root / "FAILURE").exists():
        raise ValueError("run terminal identity is not exactly one SUCCESS marker")
    scratch = root / "scratch"
    if not scratch.is_dir() or any(scratch.iterdir()):
        raise ValueError("run scratch directory is absent or not empty")
    checksum_path = root / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        raise ValueError("run checksum manifest is missing")
    checked = 0
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._/-]+)", line)
        if match is None:
            raise ValueError(f"checksum manifest line {line_number} is malformed")
        relative = Path(match.group(2))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checksum manifest contains an unsafe path")
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("checksum target escapes run root") from exc
        if not target.is_file() or _sha256(target) != match.group(1):
            raise ValueError(f"checksum identity failed for {relative.as_posix()}")
        checked += 1
    summary_path = root / "omix709_developmental_summary.json"
    if not summary_path.is_file():
        raise ValueError("run summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != SUMMARY_VERSION or summary.get("technical_status") != "success":
        raise ValueError("run summary identity or technical status is invalid")
    if summary.get("scientific_status") not in {
        "supportive",
        "not_supportive",
        "not_estimable",
    }:
        raise ValueError("run scientific status is outside the frozen vocabulary")
    if contract is not None:
        if summary.get("threads", 0) > contract.evaluation.max_threads:
            raise ValueError("run thread receipt exceeds the manifest cap")
        if summary.get("seed") != contract.evaluation.seed:
            raise ValueError("run seed differs from the manifest")
    return {
        "schema_version": "wormctx-omix709-run-verification-1.0",
        "verified": True,
        "checked_files": checked,
        "technical_status": summary["technical_status"],
        "scientific_status": summary["scientific_status"],
        "summary_sha256": _sha256(summary_path),
    }


def compare_summaries(
    reference_summary: str | Path,
    candidate_summary: str | Path,
    absolute_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Compare two qualified summaries without claiming cross-platform byte identity.

    Strings, booleans, nulls, integer counts, keys, and list shapes must match
    exactly.  Only finite floating-point values receive the declared absolute
    tolerance.  This captures last-bit BLAS differences while refusing a status,
    split, count, threshold, or biologically meaningful metric change.
    """

    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("summary comparison tolerance must be finite and nonnegative")
    reference_path = Path(reference_summary).resolve()
    candidate_path = Path(candidate_summary).resolve()
    for path in (reference_path, candidate_path):
        if not path.is_file():
            raise FileNotFoundError(f"summary comparison input is not a file: {path}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    mismatches: list[dict[str, Any]] = []
    float_differences: list[dict[str, Any]] = []
    numeric_comparisons = 0

    def compare(left: Any, right: Any, location: str) -> None:
        nonlocal numeric_comparisons
        if isinstance(left, dict):
            if not isinstance(right, dict) or set(left) != set(right):
                mismatches.append({"path": location, "reason": "object_keys_or_type"})
                return
            for key in sorted(left):
                compare(left[key], right[key], f"{location}.{key}")
            return
        if isinstance(left, list):
            if not isinstance(right, list) or len(left) != len(right):
                mismatches.append({"path": location, "reason": "list_length_or_type"})
                return
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
                compare(left_item, right_item, f"{location}[{index}]")
            return
        if isinstance(left, bool) or left is None or isinstance(left, (str, int)):
            if type(left) is not type(right) or left != right:
                mismatches.append(
                    {"path": location, "reason": "exact_value", "reference": left, "candidate": right}
                )
            return
        if isinstance(left, float):
            numeric_comparisons += 1
            if not isinstance(right, (int, float)) or isinstance(right, bool):
                mismatches.append({"path": location, "reason": "numeric_type"})
                return
            right_value = float(right)
            if not math.isfinite(left) or not math.isfinite(right_value):
                mismatches.append({"path": location, "reason": "nonfinite_numeric"})
                return
            difference = abs(left - right_value)
            if difference > 0.0:
                float_differences.append(
                    {
                        "path": location,
                        "reference": left,
                        "candidate": right_value,
                        "absolute_difference": difference,
                    }
                )
            if difference > absolute_tolerance:
                mismatches.append(
                    {
                        "path": location,
                        "reason": "absolute_tolerance",
                        "absolute_difference": difference,
                    }
                )
            return
        if type(left) is not type(right) or left != right:
            mismatches.append({"path": location, "reason": "unsupported_or_unequal_value"})

    compare(reference, candidate, "summary")
    max_difference = max(
        (item["absolute_difference"] for item in float_differences), default=0.0
    )
    return {
        "schema_version": "wormctx-omix709-cross-platform-summary-comparison-1.0",
        "equivalent_within_tolerance": not mismatches,
        "absolute_tolerance": absolute_tolerance,
        "reference_sha256": _sha256(reference_path),
        "candidate_sha256": _sha256(candidate_path),
        "byte_identical": reference_path.read_bytes() == candidate_path.read_bytes(),
        "numeric_comparisons": numeric_comparisons,
        "nonzero_float_differences": len(float_differences),
        "maximum_absolute_difference": max_difference,
        "mismatches": mismatches,
    }


def _read_receipt(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"receipt is not a file: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("receipt must be a JSON object")
    return payload


def _emit(payload: Any, output: str | Path | None = None) -> None:
    if output is None:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    else:
        _write_json_atomic(Path(output).resolve(), payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wormctx.poc.developmental_omix",
        description="Strict OMIX709 real developmental-genetics benchmark",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-sources", help="verify exact source archive bytes")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--control-archive", required=True)
    verify.add_argument("--addendum-archive", required=True)
    verify.add_argument("--output")

    normalize_command = commands.add_parser("normalize", help="normalize extracted embryo tables")
    normalize_command.add_argument("--manifest", required=True)
    normalize_command.add_argument("--control-archive", required=True)
    normalize_command.add_argument("--addendum-archive", required=True)
    normalize_command.add_argument("--control-root", required=True)
    normalize_command.add_argument("--addendum-root", required=True)
    normalize_command.add_argument("--output", required=True)
    normalize_command.add_argument("--receipt", required=True)

    qualify_command = commands.add_parser("qualify", help="qualify a normalized snapshot")
    qualify_command.add_argument("--manifest", required=True)
    qualify_command.add_argument("--normalized", required=True)
    qualify_command.add_argument("--normalization-receipt")
    qualify_command.add_argument("--output")

    run_command = commands.add_parser("run", help="run held-out-target developmental baselines")
    run_command.add_argument("--manifest", required=True)
    run_command.add_argument("--normalized", required=True)
    run_command.add_argument("--qualification")
    run_command.add_argument("--output-root", required=True)
    run_command.add_argument("--seed", type=int, default=1729)
    run_command.add_argument("--threads", type=int, default=2)

    verify_run_command = commands.add_parser("verify-run", help="verify a completed run")
    verify_run_command.add_argument("--manifest")
    verify_run_command.add_argument("--run-root", "--run", dest="run_root", required=True)
    verify_run_command.add_argument("--output")

    compare_command = commands.add_parser(
        "compare-runs", help="compare cross-platform summaries with exact structural fields"
    )
    compare_command.add_argument("--reference-summary", required=True)
    compare_command.add_argument("--candidate-summary", required=True)
    compare_command.add_argument("--absolute-tolerance", type=float, default=1e-12)
    compare_command.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-sources":
        payload = verify_sources(
            args.manifest, args.control_archive, args.addendum_archive, args.output
        )
        if args.output is None:
            _emit(payload)
    elif args.command == "normalize":
        sources = verify_sources(args.manifest, args.control_archive, args.addendum_archive)
        payload = normalize(
            args.manifest,
            args.control_root,
            args.addendum_root,
            args.output,
            sources["sources"],
        )
        _emit(payload, args.receipt)
    elif args.command == "qualify":
        payload = qualify(args.manifest, args.normalized)
        if args.normalization_receipt:
            normalization = _read_receipt(args.normalization_receipt)
            if (
                normalization.get("sha256") != payload["sha256"]
                or normalization.get("bytes") != payload["bytes"]
            ):
                raise ValueError("normalization receipt differs from the qualified snapshot")
        _emit(payload, args.output)
    elif args.command == "run":
        if args.qualification:
            qualification = _read_receipt(args.qualification)
            observed = qualify(args.manifest, args.normalized)
            if qualification.get("sha256") != observed["sha256"]:
                raise ValueError("qualification receipt differs from the normalized snapshot")
        payload = run(
            args.manifest,
            args.normalized,
            args.output_root,
            seed=args.seed,
            threads=args.threads,
        )
        _emit(payload)
    elif args.command == "verify-run":
        payload = verify_run(args.run_root, args.manifest)
        _emit(payload, args.output)
    elif args.command == "compare-runs":
        payload = compare_summaries(
            args.reference_summary, args.candidate_summary, args.absolute_tolerance
        )
        _emit(payload, args.output)
        if not payload["equivalent_within_tolerance"]:
            raise ValueError("run summaries differ beyond the cross-platform replay contract")
    else:  # pragma: no cover - argparse makes this unreachable
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
