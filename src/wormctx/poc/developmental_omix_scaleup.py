"""Fail-closed qualification for the processed OMIX709 S1--S4 scale-up.

The qualifier deliberately does not fit a model or summarize outcome
prevalence.  It verifies immutable workbook identity, records the workbook
schema, proves the permitted joins, excludes targets exposed by the earlier
251-embryo pilot, and freezes deterministic whole-gene partitions.

Only the Python standard library and the existing POC manifest helpers are
used.  XLSX files are inspected as ZIP/XML packages so the stage remains
portable to a minimally provisioned remote workstation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .developmental_omix import (
    SHA256_RE,
    _canonical_bytes,
    _sha256,
    _write_bytes_atomic,
    _write_json_atomic,
)


SCHEMA_VERSION = "wormctx-omix709-processed-scaleup-qualification-1.0"
QUALIFICATION_VERSION = "wormctx-omix709-processed-qualification-1.0"
SOURCE_RECEIPT_VERSION = "wormctx-omix709-processed-source-receipt-1.0"
WORKBOOK_SCHEMA_VERSION = "wormctx-omix709-workbook-schema-manifest-1.0"
SPLIT_VERSION = "wormctx-omix709-gene-split-manifest-1.0"
ANALYSIS_MANIFEST_VERSION = "wormctx-omix709-scaleup-analysis-manifest-1.0"
VERIFY_VERSION = "wormctx-omix709-processed-qualification-verification-1.0"

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
REL_ID = f"{{{DOC_REL_NS}}}id"
CELL_REF = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")
DIMENSION_REF = re.compile(r"^(?:[A-Z]+[1-9][0-9]*:)?([A-Z]+)([1-9][0-9]*)$")
WORMBASE_ID_FIELD = re.compile(
    r"^WBGene[0-9]{8}(?: \(WBGene[0-9]{8}\))?$"
)

EXPECTED_SOURCE_ROLES = [
    "gene_metadata",
    "reference_control_quantitative_measurements",
    "knockdown_quantitative_measurements",
    "knockdown_defect_calls_secondary_outcomes",
]
EXPECTED_SOURCE_FILENAMES = [
    "Table_S1.xlsx",
    "Table_S2.xlsx",
    "Table_S3.xlsx",
    "Table_S4.xlsx",
]
MODALITY_IDS = [
    "cell_cycle_length",
    "division_timing",
    "sibling_division_timing_difference",
    "cnd1_gfp_expression",
    "division_angle",
    "three_dimensional_position",
]
OUTPUT_FILES = {
    "analysis_manifest.json",
    "qualification.json",
    "source_receipt.json",
    "split_manifest.json",
    "workbook_schema_manifest.json",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SheetContract(_StrictModel):
    name: str
    dimension: str

    @field_validator("dimension")
    @classmethod
    def valid_dimension(cls, value: str) -> str:
        if not re.fullmatch(
            r"[A-Z]+[1-9][0-9]*(?::[A-Z]+[1-9][0-9]*)?", value
        ):
            raise ValueError("workbook dimension is not a canonical A1 range")
        return value


class SourceContract(_StrictModel):
    role: str
    filename: str
    url: str
    bytes: int = Field(gt=0)
    sha256: str
    sheets: list[SheetContract]

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("source SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def distinct_sheets(self) -> "SourceContract":
        names = [item.name for item in self.sheets]
        if not names or len(names) != len(set(names)):
            raise ValueError("source sheet names must be present and unique")
        return self


class RightsContract(_StrictModel):
    access_class: str
    permitted_use: str
    source_redistribution_permitted: bool
    commercial_use_permitted: bool
    public_training_permitted: bool
    operator_attestation_required: bool
    attestation_text: str
    review_contact: str


class ExpectedCounts(_StrictModel):
    reference_controls: int = Field(gt=0)
    measurement_embryos: int = Field(gt=0)
    measurement_genes: int = Field(gt=0)


class ModalityContract(_StrictModel):
    id: str
    s2_sheet: str
    s2_title: str
    s3_sheet: str
    s3_title: str
    s4_sheet: str
    s4_title: str
    s4_embryo_join: str
    s4_rows_joined_by_semantic_cell_id: bool
    s4_qualification: str


class OverlapContract(_StrictModel):
    exposed_public_names: list[str]
    exclude_from_claim_bearing: bool
    gene_identity_join_key: str
    public_name_used_as_join_key: bool
    observed_s1_s3_public_name_alias_mismatches: int = Field(ge=0)
    alias_mismatch_is_unambiguous_by_wormbase_gene_id: bool
    target_gene_overlap_exclusion_established: bool
    embryo_level_disjointness_established: bool
    embryo_level_disjointness_blocker: str

    @model_validator(mode="after")
    def distinct_exposed_targets(self) -> "OverlapContract":
        normalized = [item.upper() for item in self.exposed_public_names]
        if len(normalized) != len(set(normalized)):
            raise ValueError("exposed public names must be unique case-insensitively")
        return self


class SplitContract(_StrictModel):
    method: str
    salt: str
    key: str
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)
    whole_gene: bool
    assignment_precedes_outcome_value_access: bool

    @model_validator(mode="after")
    def exact_split(self) -> "SplitContract":
        if not math.isclose(
            self.train_fraction + self.validation_fraction + self.test_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("split fractions must sum to one")
        return self


class StageEncodingContract(_StrictModel):
    requested_landmarks: list[int]
    explicit_frame_snapshots: bool
    source_representation: str
    adapter_required: bool
    adapter_contract: str
    modeling_permitted_before_adapter_freeze: bool


class ScaleupQualificationManifest(_StrictModel):
    schema_version: str
    analysis_id: str
    classification: str
    status: str
    validated: bool
    biological_claims_permitted: bool
    sources: list[SourceContract]
    rights: RightsContract
    expected_counts: ExpectedCounts
    modalities: list[ModalityContract]
    overlap: OverlapContract
    split: SplitContract
    stage_encoding: StageEncodingContract
    anti_circularity: dict[str, Any]
    claims: dict[str, Any]

    @model_validator(mode="after")
    def exact_contract(self) -> "ScaleupQualificationManifest":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unknown processed-scaleup qualification schema")
        if [item.role for item in self.sources] != EXPECTED_SOURCE_ROLES:
            raise ValueError("processed sources have the wrong roles or order")
        if [item.filename for item in self.sources] != EXPECTED_SOURCE_FILENAMES:
            raise ValueError("processed sources have the wrong filenames or order")
        if [item.id for item in self.modalities] != MODALITY_IDS:
            raise ValueError("modality contracts have the wrong identities or order")
        for index, modality in enumerate(self.modalities, 1):
            if (modality.s2_sheet, modality.s3_sheet, modality.s4_sheet) != (
                str(index),
                str(index + 1),
                str(index),
            ):
                raise ValueError("modality sheet mapping differs from the frozen contract")
            if (
                modality.s4_embryo_join != "embryo_id"
                or not modality.s4_rows_joined_by_semantic_cell_id
            ):
                raise ValueError("S4 requires embryo-ID columns and semantic-cell row joins")
            expected_qualification = (
                "blocked_header_identity"
                if modality.id == "cnd1_gfp_expression"
                else "qualified"
            )
            if modality.s4_qualification != expected_qualification:
                raise ValueError("S4 modality qualification differs from the frozen exclusion")
        if self.split.method != "sha256_sorted_gene_holdout":
            raise ValueError("split method must be SHA-256 sorted gene holdout")
        if self.split.key != "wormbase_gene_id" or not self.split.whole_gene:
            raise ValueError("the indivisible split key must be the whole WormBase gene")
        if not self.split.assignment_precedes_outcome_value_access:
            raise ValueError("split assignment must precede outcome-value access")
        if not self.overlap.exclude_from_claim_bearing:
            raise ValueError("previously exposed targets must be excluded")
        if (
            self.overlap.gene_identity_join_key != "wormbase_gene_id"
            or self.overlap.public_name_used_as_join_key
            or not self.overlap.alias_mismatch_is_unambiguous_by_wormbase_gene_id
            or not self.overlap.target_gene_overlap_exclusion_established
            or self.overlap.embryo_level_disjointness_established
        ):
            raise ValueError("WormBase ID must be the sole unambiguous gene join key")
        if self.validated or self.biological_claims_permitted:
            raise ValueError("qualification cannot assert validation or biological claims")
        if (
            self.rights.source_redistribution_permitted
            or self.rights.commercial_use_permitted
            or self.rights.public_training_permitted
        ):
            raise ValueError("rights contract is broader than the governed source permits")
        if not self.rights.operator_attestation_required:
            raise ValueError("controlled academic-use attestation must be required")
        if (
            self.stage_encoding.explicit_frame_snapshots
            or not self.stage_encoding.adapter_required
            or self.stage_encoding.modeling_permitted_before_adapter_freeze
            or self.stage_encoding.requested_landmarks != [26, 200]
        ):
            raise ValueError("processed tables require a frozen semantic-cell stage adapter")
        if self.anti_circularity.get("outcome_prevalence_inspection_permitted") is not False:
            raise ValueError("qualification may not inspect outcome prevalence")
        if self.anti_circularity.get("s4_values_permitted_as_predictors") is not False:
            raise ValueError("S4 calls cannot enter primary model inputs")
        return self


@dataclass
class SheetScan:
    name: str
    xml_path: str
    state: str
    dimension: str | None = None
    row_elements: int = 0
    cell_elements: int = 0
    formula_cells: int = 0
    merged_ranges: list[str] = field(default_factory=list)
    full_rows: dict[int, list[Any]] = field(default_factory=dict)
    selected_rows: list[tuple[int, dict[int, Any]]] = field(default_factory=list)
    first_column: list[tuple[int, Any]] = field(default_factory=list)


def _tag(namespace: str, local: str) -> str:
    return f"{{{namespace}}}{local}"


def _column_index(reference: str) -> int:
    match = CELL_REF.fullmatch(reference)
    if match is None:
        raise ValueError(f"invalid XLSX cell reference: {reference!r}")
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def _dimension_shape(reference: str) -> tuple[int, int]:
    match = DIMENSION_REF.fullmatch(reference)
    if match is None:
        raise ValueError(f"invalid XLSX dimension: {reference!r}")
    column = 0
    for character in match.group(1):
        column = column * 26 + ord(character) - ord("A") + 1
    return int(match.group(2)), column


def _trim(values: list[Any]) -> list[Any]:
    while values and values[-1] is None:
        values.pop()
    return values


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _ordered_strings_hash(values: Sequence[str]) -> str:
    return _stable_hash(list(values))


class XlsxPackage:
    """Small, streaming XLSX reader for identity and schema qualification."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.archive = zipfile.ZipFile(path)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"source workbook is not a valid XLSX ZIP package: {path}") from exc
        self.defined_name_count = 0
        self._validate_package_members()
        self.sheet_refs = self._load_sheet_refs()
        self.shared_strings = self._load_shared_strings()

    def close(self) -> None:
        self.archive.close()

    def __enter__(self) -> "XlsxPackage":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _validate_package_members(self) -> None:
        names: set[str] = set()
        for info in self.archive.infolist():
            path = PurePosixPath(info.filename)
            if (
                info.filename in names
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in info.filename
            ):
                raise ValueError(f"unsafe or duplicate XLSX package member: {info.filename!r}")
            if info.flag_bits & 0x1:
                raise ValueError("encrypted XLSX package members are not permitted")
            names.add(info.filename)
        required = {
            "[Content_Types].xml",
            "_rels/.rels",
            "xl/workbook.xml",
            "xl/_rels/workbook.xml.rels",
        }
        if not required.issubset(names):
            raise ValueError("XLSX package is missing required workbook parts")
        corrupt = self.archive.testzip()
        if corrupt is not None:
            raise ValueError(f"XLSX package CRC validation failed: {corrupt}")

    def _load_sheet_refs(self) -> list[dict[str, str]]:
        relationships = ET.fromstring(self.archive.read("xl/_rels/workbook.xml.rels"))
        targets: dict[str, str] = {}
        for item in relationships.findall(_tag(PACKAGE_REL_NS, "Relationship")):
            if item.get("TargetMode") == "External":
                raise ValueError("external workbook relationships are not permitted")
            if item.get("Type", "").endswith("/worksheet"):
                relation_id = item.get("Id")
                target = item.get("Target")
                if relation_id and target:
                    targets[relation_id] = f"xl/{target.lstrip('/')}"
        workbook = ET.fromstring(self.archive.read("xl/workbook.xml"))
        defined_names = workbook.find(_tag(MAIN_NS, "definedNames"))
        self.defined_name_count = (
            0
            if defined_names is None
            else len(defined_names.findall(_tag(MAIN_NS, "definedName")))
        )
        sheets = workbook.find(_tag(MAIN_NS, "sheets"))
        if sheets is None:
            raise ValueError("XLSX workbook has no sheet catalog")
        result = []
        for sheet in sheets.findall(_tag(MAIN_NS, "sheet")):
            relation_id = sheet.get(REL_ID)
            name = sheet.get("name")
            if not relation_id or not name or relation_id not in targets:
                raise ValueError("XLSX sheet relationship cannot be resolved")
            result.append(
                {
                    "name": name,
                    "path": targets[relation_id],
                    "state": sheet.get("state", "visible"),
                }
            )
        if not result or len({item["name"] for item in result}) != len(result):
            raise ValueError("XLSX sheet names must be present and unique")
        return result

    def _load_shared_strings(self) -> list[str]:
        if "xl/sharedStrings.xml" not in self.archive.namelist():
            return []
        values: list[str] = []
        with self.archive.open("xl/sharedStrings.xml") as handle:
            for event, element in ET.iterparse(handle, events=("end",)):
                if element.tag == _tag(MAIN_NS, "si"):
                    values.append(
                        "".join(
                            item.text or "" for item in element.iter(_tag(MAIN_NS, "t"))
                        )
                    )
                    element.clear()
        return values

    def package_summary(self) -> dict[str, Any]:
        infos = self.archive.infolist()
        external_relationships = 0
        for info in infos:
            if not info.filename.endswith(".rels"):
                continue
            try:
                root = ET.fromstring(self.archive.read(info.filename))
            except ET.ParseError as exc:
                raise ValueError(f"invalid XLSX relationship XML: {info.filename}") from exc
            external_relationships += sum(
                item.get("TargetMode") == "External"
                for item in root.findall(_tag(PACKAGE_REL_NS, "Relationship"))
            )
        names = [item.filename for item in infos]
        return {
            "zip_members": len(infos),
            "compressed_bytes": sum(item.compress_size for item in infos),
            "uncompressed_bytes": sum(item.file_size for item in infos),
            "shared_strings": len(self.shared_strings),
            "defined_names": self.defined_name_count,
            "external_relationships": external_relationships,
            "macro_parts": sum(
                name.lower().endswith("vbaproject.bin") or "/vba" in name.lower()
                for name in names
            ),
        }

    def scan_sheet(
        self,
        name: str,
        *,
        full_rows: Iterable[int] = (),
        selected_columns: Iterable[int] = (),
        selected_start_row: int | None = None,
        first_column_start_row: int | None = None,
    ) -> SheetScan:
        reference = next((item for item in self.sheet_refs if item["name"] == name), None)
        if reference is None:
            raise ValueError(f"XLSX workbook is missing sheet {name!r}")
        full_row_set = set(full_rows)
        selected_column_set = set(selected_columns)
        scan = SheetScan(
            name=name,
            xml_path=reference["path"],
            state=reference["state"],
        )
        with self.archive.open(reference["path"]) as handle:
            for _event, element in ET.iterparse(handle, events=("end",)):
                if element.tag == _tag(MAIN_NS, "dimension"):
                    scan.dimension = element.get("ref")
                    element.clear()
                    continue
                if element.tag == _tag(MAIN_NS, "mergeCell"):
                    merged = element.get("ref")
                    if merged:
                        scan.merged_ranges.append(merged)
                    element.clear()
                    continue
                if element.tag != _tag(MAIN_NS, "row"):
                    continue
                row_number = int(element.get("r", scan.row_elements + 1))
                scan.row_elements += 1
                cells = element.findall(_tag(MAIN_NS, "c"))
                scan.cell_elements += len(cells)
                scan.formula_cells += sum(
                    cell.find(_tag(MAIN_NS, "f")) is not None for cell in cells
                )
                capture_full = row_number in full_row_set
                capture_selected = (
                    selected_start_row is not None
                    and row_number >= selected_start_row
                    and bool(selected_column_set)
                )
                capture_first = (
                    first_column_start_row is not None
                    and row_number >= first_column_start_row
                )
                if capture_full or capture_selected or capture_first:
                    decoded: dict[int, Any] = {}
                    next_column = 0
                    for cell in cells:
                        cell_reference = cell.get("r")
                        column = (
                            _column_index(cell_reference)
                            if cell_reference is not None
                            else next_column
                        )
                        next_column = column + 1
                        if (
                            capture_full
                            or (capture_selected and column in selected_column_set)
                            or (capture_first and column == 0)
                        ):
                            decoded[column] = self._cell_value(cell)
                    if capture_full:
                        width = max(decoded, default=-1) + 1
                        if scan.dimension:
                            width = max(width, _dimension_shape(scan.dimension)[1])
                        row_values = [None] * width
                        for column, value in decoded.items():
                            row_values[column] = value
                        scan.full_rows[row_number] = _trim(row_values)
                    if capture_selected:
                        scan.selected_rows.append(
                            (
                                row_number,
                                {
                                    column: decoded.get(column)
                                    for column in sorted(selected_column_set)
                                },
                            )
                        )
                    if capture_first:
                        scan.first_column.append((row_number, decoded.get(0)))
                element.clear()
        if scan.dimension is None:
            scan.dimension = "A1"
        scan.merged_ranges.sort()
        return scan

    def _cell_value(self, cell: ET.Element) -> Any:
        cell_type = cell.get("t")
        value = cell.find(_tag(MAIN_NS, "v"))
        raw = value.text if value is not None else None
        if cell_type == "s":
            if raw is None:
                return None
            index = int(raw)
            if index < 0 or index >= len(self.shared_strings):
                raise ValueError("XLSX shared-string index is out of range")
            return self.shared_strings[index]
        if cell_type == "inlineStr":
            inline = cell.find(_tag(MAIN_NS, "is"))
            if inline is None:
                return None
            return "".join(item.text or "" for item in inline.iter(_tag(MAIN_NS, "t")))
        if cell_type in {"str", "e"}:
            return raw
        if cell_type == "b":
            return raw == "1"
        if raw is None or raw == "":
            return None
        try:
            integer = int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
        return integer


def load_manifest(path: str | Path) -> ScaleupQualificationManifest:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"processed-scaleup manifest is not a file: {source}")
    return ScaleupQualificationManifest.model_validate_json(
        source.read_text(encoding="utf-8")
    )


def _manifest(
    value: ScaleupQualificationManifest | str | Path,
) -> ScaleupQualificationManifest:
    if isinstance(value, ScaleupQualificationManifest):
        return value
    return load_manifest(value)


def _verify_source(path: Path, contract: SourceContract) -> dict[str, Any]:
    if not path.is_file() or path.name != contract.filename:
        raise FileNotFoundError(f"required processed workbook is missing: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != contract.bytes:
        raise ValueError(
            f"{contract.filename}: byte identity failed: expected {contract.bytes}, "
            f"observed {observed_bytes}"
        )
    observed_hash = _sha256(path)
    if observed_hash != contract.sha256:
        raise ValueError(
            f"{contract.filename}: SHA-256 identity failed: expected {contract.sha256}, "
            f"observed {observed_hash}"
        )
    return {
        "role": contract.role,
        "filename": contract.filename,
        "bytes": observed_bytes,
        "sha256": observed_hash,
        "source_redistribution_permitted": False,
    }


def _sheet_manifest(scan: SheetScan) -> dict[str, Any]:
    first_column_values = [
        value
        for _row_number, value in scan.first_column
        if value is not None and str(value) != ""
    ]
    return {
        "name": scan.name,
        "xml_path": scan.xml_path,
        "state": scan.state,
        "dimension": scan.dimension,
        "row_elements": scan.row_elements,
        "cell_elements": scan.cell_elements,
        "formula_cells": scan.formula_cells,
        "merged_ranges": scan.merged_ranges,
        "captured_schema_row_sha256": {
            str(row_number): _stable_hash(values)
            for row_number, values in sorted(scan.full_rows.items())
        },
        "captured_first_column_values": len(first_column_values),
        "captured_first_column_order_sha256": (
            _stable_hash(first_column_values) if first_column_values else None
        ),
    }


def _as_nonempty_strings(values: Sequence[Any]) -> list[str]:
    result = []
    for value in values:
        if value is None or str(value) == "":
            continue
        result.append(str(value))
    return result


def _title(scan: SheetScan, row: int = 1) -> str:
    values = scan.full_rows.get(row, [])
    if not values or values[0] is None:
        return ""
    return str(values[0])


def _exact_sheet_contract(
    source: SourceContract,
    package: XlsxPackage,
    scans: Mapping[str, SheetScan],
) -> None:
    observed_names = [item["name"] for item in package.sheet_refs]
    expected_names = [item.name for item in source.sheets]
    if observed_names != expected_names:
        raise ValueError(
            f"{source.filename}: sheet names/order differ: "
            f"expected={expected_names}, observed={observed_names}"
        )
    expected_dimensions = {item.name: item.dimension for item in source.sheets}
    for name in expected_names:
        scan = scans[name]
        if scan.state != "visible":
            raise ValueError(f"{source.filename}/{name}: hidden sheets are not permitted")
        if scan.dimension != expected_dimensions[name]:
            raise ValueError(
                f"{source.filename}/{name}: dimension differs: "
                f"expected={expected_dimensions[name]}, observed={scan.dimension}"
            )
        if scan.formula_cells:
            raise ValueError(f"{source.filename}/{name}: formulas are not permitted")
        if scan.merged_ranges:
            raise ValueError(f"{source.filename}/{name}: merged cells are not permitted")


def _scan_sources(
    manifest: ScaleupQualificationManifest,
    source_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, SheetScan]],
    dict[str, dict[str, Any]],
]:
    receipts: list[dict[str, Any]] = []
    scans_by_file: dict[str, dict[str, SheetScan]] = {}
    packages: dict[str, dict[str, Any]] = {}
    for contract in manifest.sources:
        path = source_dir / contract.filename
        receipts.append(_verify_source(path, contract))
        with XlsxPackage(path) as package:
            package_summary = package.package_summary()
            if package_summary["external_relationships"]:
                raise ValueError(f"{contract.filename}: external relationships are prohibited")
            if package_summary["macro_parts"]:
                raise ValueError(f"{contract.filename}: macro parts are prohibited")
            if package_summary["defined_names"]:
                raise ValueError(f"{contract.filename}: defined names are prohibited")
            scans: dict[str, SheetScan] = {}
            for item in package.sheet_refs:
                name = item["name"]
                kwargs: dict[str, Any] = {}
                if contract.filename == "Table_S1.xlsx":
                    if name == "1":
                        kwargs = {
                            "full_rows": {1},
                            "selected_columns": {0, 1, 6, 8},
                            "selected_start_row": 2,
                        }
                    elif name == "2":
                        kwargs = {"full_rows": {1}}
                    else:
                        kwargs = {"full_rows": {1}}
                elif contract.filename == "Table_S2.xlsx":
                    kwargs = (
                        {"full_rows": {1, 2}, "first_column_start_row": 3}
                        if name in {"1", "2", "3", "4", "5", "6"}
                        else {"full_rows": {1}}
                    )
                elif contract.filename == "Table_S3.xlsx":
                    if name == "1":
                        kwargs = {
                            "full_rows": {1},
                            "selected_columns": {0, 1, 2, 4},
                            "selected_start_row": 2,
                        }
                    elif name in {"2", "3", "4", "5", "6", "7"}:
                        kwargs = {
                            "full_rows": {1, 2},
                            "first_column_start_row": 3,
                        }
                    elif name == "8":
                        kwargs = {
                            "full_rows": {1},
                            "selected_columns": {0, 1, 2},
                            "selected_start_row": 2,
                        }
                    else:
                        kwargs = {"full_rows": {1}}
                elif contract.filename == "Table_S4.xlsx":
                    if name in {"1", "2", "3", "4", "5", "6"}:
                        kwargs = {"full_rows": {1}, "first_column_start_row": 2}
                    elif name in {"7", "8"}:
                        kwargs = {"full_rows": {1, 2}}
                    else:
                        kwargs = {"full_rows": {1}}
                scans[name] = package.scan_sheet(name, **kwargs)
            _exact_sheet_contract(contract, package, scans)
            scans_by_file[contract.filename] = scans
            packages[contract.filename] = package_summary
    return receipts, scans_by_file, packages


def _metadata_rows(scan: SheetScan) -> list[dict[str, str]]:
    result = []
    for _row_number, values in scan.selected_rows:
        gene_id = values.get(0)
        public_name = values.get(1)
        embryo_id = values.get(2)
        treatment = values.get(4)
        if gene_id is None and public_name is None and embryo_id is None:
            continue
        result.append(
            {
                "gene_id": "" if gene_id is None else str(gene_id),
                "public_name": "" if public_name is None else str(public_name),
                "embryo_id": "" if embryo_id is None else str(embryo_id),
                "treatment": "" if treatment is None else str(treatment),
            }
        )
    return result


def _inventory_rows(scan: SheetScan) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for _row_number, values in scan.selected_rows:
        gene_id = "" if values.get(0) is None else str(values[0])
        if not gene_id:
            continue
        if gene_id in result:
            raise ValueError(f"Table_S1/1: duplicate WormBase gene ID: {gene_id}")
        result[gene_id] = {
            "public_name": "" if values.get(1) is None else str(values[1]),
            "selected": "" if values.get(6) is None else str(values[6]),
            "validation": "" if values.get(8) is None else str(values[8]),
        }
    return result


def _selected_metadata(scan: SheetScan, measurement_ids: set[str]) -> dict[str, dict[str, str]]:
    rows = _metadata_rows(scan)
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        embryo_id = row["embryo_id"]
        if embryo_id not in measurement_ids:
            continue
        if embryo_id in selected:
            raise ValueError(f"Table_S3/1: duplicate measurement embryo metadata: {embryo_id}")
        if not WORMBASE_ID_FIELD.fullmatch(row["gene_id"]):
            raise ValueError(f"Table_S3/1: invalid WormBase gene ID for {embryo_id}")
        if not row["public_name"]:
            raise ValueError(f"Table_S3/1: missing public name for {embryo_id}")
        selected[embryo_id] = row
    missing = sorted(measurement_ids - set(selected))
    if missing:
        raise ValueError(f"Table_S3/1: measurement embryo metadata are missing: {missing[:5]}")
    return selected


def _round_quota(total: int, fraction: float) -> int:
    return int(total * fraction + 0.5)


def _public_name_aliases(value: str) -> set[str]:
    """Return conservative exact aliases without repairing arbitrary IDs."""

    normalized = value.strip().upper()
    aliases = {normalized}
    if "(" in normalized and normalized.endswith(")"):
        prefix, parenthetical = normalized[:-1].split("(", 1)
        if prefix.strip():
            aliases.add(prefix.strip())
        for item in re.split(r"[,/]", parenthetical):
            if item.strip():
                aliases.add(item.strip())
    return aliases


def _build_split(
    manifest: ScaleupQualificationManifest,
    metadata: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_gene: dict[str, dict[str, Any]] = {}
    public_to_genes: dict[str, set[str]] = {}
    for embryo_id, row in metadata.items():
        gene_id = row["gene_id"]
        public_name = row["public_name"]
        item = by_gene.setdefault(
            gene_id,
            {"gene_id": gene_id, "public_names": set(), "embryo_ids": []},
        )
        item["public_names"].add(public_name)
        item["embryo_ids"].append(embryo_id)
        for alias in _public_name_aliases(public_name):
            public_to_genes.setdefault(alias, set()).add(gene_id)
    ambiguous_gene_names = {
        gene_id: sorted(item["public_names"])
        for gene_id, item in by_gene.items()
        if len(item["public_names"]) != 1
    }
    if ambiguous_gene_names:
        raise ValueError(f"measurement genes have ambiguous public names: {ambiguous_gene_names}")
    exposed_present: list[str] = []
    exposed_absent: list[str] = []
    exposed_gene_ids: set[str] = set()
    for public_name in manifest.overlap.exposed_public_names:
        genes = public_to_genes.get(public_name.upper(), set())
        if len(genes) > 1:
            raise ValueError(f"exposed public name maps to multiple WormBase IDs: {public_name}")
        if genes:
            exposed_present.append(public_name)
            exposed_gene_ids.update(genes)
        else:
            exposed_absent.append(public_name)

    def gene_entry(gene_id: str, partition: str) -> dict[str, Any]:
        item = by_gene[gene_id]
        public_name = next(iter(item["public_names"]))
        embryo_ids = sorted(item["embryo_ids"])
        digest = hashlib.sha256(
            f"{manifest.split.salt}|{gene_id}".encode("utf-8")
        ).hexdigest()
        return {
            "gene_id": gene_id,
            "public_name": public_name,
            "partition": partition,
            "partition_digest": digest,
            "embryo_count": len(embryo_ids),
            "embryo_ids": embryo_ids,
        }

    eligible = sorted(
        set(by_gene) - exposed_gene_ids,
        key=lambda gene_id: (
            hashlib.sha256(
                f"{manifest.split.salt}|{gene_id}".encode("utf-8")
            ).hexdigest(),
            gene_id,
        ),
    )
    train_count = _round_quota(len(eligible), manifest.split.train_fraction)
    validation_count = _round_quota(len(eligible), manifest.split.validation_fraction)
    if train_count + validation_count >= len(eligible):
        raise ValueError("claim-bearing gene count is too small for a nonempty sealed test")
    partitions = {
        **{gene_id: "train" for gene_id in eligible[:train_count]},
        **{
            gene_id: "validation"
            for gene_id in eligible[train_count : train_count + validation_count]
        },
        **{
            gene_id: "sealed_test"
            for gene_id in eligible[train_count + validation_count :]
        },
    }
    entries = [gene_entry(gene_id, partitions[gene_id]) for gene_id in eligible]
    development_only = [
        gene_entry(gene_id, "development_only_exposed_pilot")
        for gene_id in sorted(exposed_gene_ids)
    ]
    counts = {
        partition: {
            "genes": sum(item["partition"] == partition for item in entries),
            "embryos": sum(
                item["embryo_count"] for item in entries if item["partition"] == partition
            ),
        }
        for partition in ["train", "validation", "sealed_test"]
    }
    counts["development_only_exposed_pilot"] = {
        "genes": len(development_only),
        "embryos": sum(item["embryo_count"] for item in development_only),
    }
    split_manifest = {
        "schema_version": SPLIT_VERSION,
        "method": manifest.split.method,
        "salt": manifest.split.salt,
        "key": manifest.split.key,
        "whole_gene": manifest.split.whole_gene,
        "fractions": {
            "train": manifest.split.train_fraction,
            "validation": manifest.split.validation_fraction,
            "sealed_test": manifest.split.test_fraction,
        },
        "counts": counts,
        "claim_bearing_genes": entries,
        "development_only_exposed_pilot_genes": development_only,
    }
    overlap = {
        "exposed_public_names_declared": manifest.overlap.exposed_public_names,
        "present": exposed_present,
        "absent": exposed_absent,
        "wormbase_gene_ids": sorted(exposed_gene_ids),
        "genes": len(exposed_gene_ids),
        "embryos": sum(item["embryo_count"] for item in development_only),
        "gene_level_exclusion_frozen": True,
        "pilot_embryo_level_disjointness": "not_established_from_s1_s4_only",
    }
    return split_manifest, overlap


def _qualify_relations(
    manifest: ScaleupQualificationManifest,
    scans: Mapping[str, Mapping[str, SheetScan]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    s1 = scans["Table_S1.xlsx"]
    s2 = scans["Table_S2.xlsx"]
    s3 = scans["Table_S3.xlsx"]
    s4 = scans["Table_S4.xlsx"]

    inventory = _inventory_rows(s1["1"])
    control_ids_by_modality: dict[str, list[str]] = {}
    measurement_ids_by_modality: dict[str, list[str]] = {}
    modality_bindings = []
    for modality in manifest.modalities:
        control_scan = s2[modality.s2_sheet]
        measurement_scan = s3[modality.s3_sheet]
        defect_scan = s4[modality.s4_sheet]
        if _title(control_scan) != modality.s2_title:
            raise ValueError(f"Table_S2/{modality.s2_sheet}: modality title differs")
        if _title(measurement_scan) != modality.s3_title:
            raise ValueError(f"Table_S3/{modality.s3_sheet}: modality title differs")
        if _title(defect_scan) != modality.s4_title:
            raise ValueError(f"Table_S4/{modality.s4_sheet}: modality title differs")
        control_header = control_scan.full_rows.get(2, [])
        measurement_header = measurement_scan.full_rows.get(2, [])
        defect_header = defect_scan.full_rows.get(1, [])
        if not control_header or not measurement_header or not defect_header:
            raise ValueError(f"{modality.id}: source header is absent")
        control_ids = _as_nonempty_strings(control_header[1:])
        measurement_ids = _as_nonempty_strings(measurement_header[1:])
        defect_ids = _as_nonempty_strings(defect_header[1:])
        if len(control_ids) != len(set(control_ids)):
            raise ValueError(f"{modality.id}: duplicate control embryo IDs")
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError(f"{modality.id}: duplicate measurement embryo IDs")
        control_ids_by_modality[modality.id] = control_ids
        measurement_ids_by_modality[modality.id] = measurement_ids
        s3_rows = [
            str(value)
            for _row_number, value in measurement_scan.first_column
            if value is not None and str(value) != ""
        ]
        s4_rows = [
            str(value)
            for _row_number, value in defect_scan.first_column
            if value is not None and str(value) != ""
        ]
        if len(s3_rows) != len(set(s3_rows)) or len(s4_rows) != len(set(s4_rows)):
            raise ValueError(f"{modality.id}: semantic cell row IDs are not unique")
        if not set(s4_rows).issubset(set(s3_rows)):
            raise ValueError(f"{modality.id}: S4 semantic cells are not a subset of S3")
        s3_shape = _dimension_shape(measurement_scan.dimension or "A1")
        s4_shape = _dimension_shape(defect_scan.dimension or "A1")
        if s3_shape[1] != s4_shape[1] or s3_shape[1] != len(measurement_ids) + 1:
            raise ValueError(f"{modality.id}: S4 embryo width cannot bind to S3")
        if len(defect_ids) != len(measurement_ids):
            raise ValueError(f"{modality.id}: S4 embryo header length differs from S3")
        defect_counts = Counter(defect_ids)
        defect_duplicates = {
            identifier: count
            for identifier, count in sorted(defect_counts.items())
            if count > 1
        }
        defect_set = set(defect_ids)
        missing_from_s4 = sorted(set(measurement_ids) - defect_set)
        unexpected_in_s4 = sorted(defect_set - set(measurement_ids))
        if modality.s4_qualification == "qualified":
            if defect_duplicates or missing_from_s4 or unexpected_in_s4:
                raise ValueError(f"{modality.id}: qualified S4 embryo IDs differ from S3")
            s4_join_status = "qualified_embryo_id_join"
        else:
            if not (defect_duplicates or missing_from_s4 or unexpected_in_s4):
                raise ValueError(
                    f"{modality.id}: frozen S4 header exclusion no longer applies; review required"
                )
            s4_join_status = "blocked_header_identity"
        modality_bindings.append(
            {
                "id": modality.id,
                "control_embryos": len(control_ids),
                "measurement_embryos": len(measurement_ids),
                "control_header_order_sha256": _ordered_strings_hash(control_ids),
                "s3_header_order_sha256": _ordered_strings_hash(measurement_ids),
                "s3_semantic_cells": len(s3_rows),
                "s4_semantic_cells": len(s4_rows),
                "s4_rows_are_s3_subset": True,
                "s4_has_embryo_header": True,
                "s4_embryo_join": "embryo_id",
                "s4_header_order_sha256": _ordered_strings_hash(defect_ids),
                "s4_order_matches_s3": defect_ids == measurement_ids,
                "s4_unique_embryo_ids": len(defect_set),
                "s4_duplicate_embryo_ids": defect_duplicates,
                "s4_missing_s3_embryo_ids": missing_from_s4,
                "s4_unexpected_embryo_ids": unexpected_in_s4,
                "s4_join_status": s4_join_status,
                "s4_rows_joined_by_semantic_cell_id": True,
            }
        )

    reference_controls = control_ids_by_modality[MODALITY_IDS[0]]
    for modality_id, identifiers in control_ids_by_modality.items():
        if identifiers != reference_controls:
            raise ValueError(f"{modality_id}: control embryo order differs across modalities")
    if len(reference_controls) != manifest.expected_counts.reference_controls:
        raise ValueError("control embryo count differs from the frozen contract")

    reference_measurements = measurement_ids_by_modality[MODALITY_IDS[0]]
    reference_measurement_set = set(reference_measurements)
    order_matches: dict[str, bool] = {}
    for modality_id, identifiers in measurement_ids_by_modality.items():
        if set(identifiers) != reference_measurement_set:
            raise ValueError(f"{modality_id}: measurement embryo set differs across modalities")
        order_matches[modality_id] = identifiers == reference_measurements
    if len(reference_measurements) != manifest.expected_counts.measurement_embryos:
        raise ValueError("measurement embryo count differs from the frozen contract")
    if set(reference_controls) & reference_measurement_set:
        raise ValueError("control and knockdown embryo identities overlap")

    metadata = _selected_metadata(s3["1"], reference_measurement_set)
    treatments = {item["treatment"] for item in metadata.values()}
    if treatments != {"lineaging and cell lineage tracing"}:
        raise ValueError("measurement metadata contain an unexpected treatment class")
    measured_gene_ids = {item["gene_id"] for item in metadata.values()}
    if len(measured_gene_ids) != manifest.expected_counts.measurement_genes:
        raise ValueError("measurement gene count differs from the frozen contract")
    missing_inventory = sorted(measured_gene_ids - set(inventory))
    if missing_inventory:
        raise ValueError(f"measurement genes are absent from Table_S1: {missing_inventory[:5]}")
    aliases = []
    public_by_gene: dict[str, set[str]] = {}
    for item in metadata.values():
        public_by_gene.setdefault(item["gene_id"], set()).add(item["public_name"])
    for gene_id in sorted(measured_gene_ids):
        observed = sorted(public_by_gene[gene_id])
        inventory_name = inventory[gene_id]["public_name"]
        if {item.upper() for item in observed} != {inventory_name.upper()}:
            aliases.append(
                {
                    "gene_id": gene_id,
                    "s1_public_name": inventory_name,
                    "s3_public_names": observed,
                    "join_key_used": "wormbase_gene_id",
                }
            )
    if len(aliases) != manifest.overlap.observed_s1_s3_public_name_alias_mismatches:
        raise ValueError("S1/S3 public-name alias mismatch count differs from the contract")

    summary_rows = {
        str(values.get(2))
        for _row_number, values in s3["8"].selected_rows
        if values.get(2) is not None and str(values.get(2)) != ""
    }
    if summary_rows != reference_measurement_set:
        raise ValueError("Table_S3/8 embryo set differs from measurement matrices")

    split_manifest, overlap = _build_split(manifest, metadata)
    relation_report = {
        "reference_controls": len(reference_controls),
        "measurement_embryos": len(reference_measurements),
        "measurement_genes": len(measured_gene_ids),
        "s1_inventory_genes": len(inventory),
        "measurement_embryo_metadata_coverage": 1.0,
        "measurement_gene_inventory_coverage": 1.0,
        "modality_embryo_sets_identical": True,
        "modality_order_matches_primary": order_matches,
        "s3_summary_embryo_set_identical": True,
        "measurement_treatment_classes": sorted(treatments),
        "public_name_alias_mismatches": aliases,
        "primary_join_key": "wormbase_gene_id",
        "compound_wormbase_id_fields": sorted(
            gene_id for gene_id in measured_gene_ids if " (" in gene_id
        ),
        "wormbase_id_field_policy": "retain_the_exact_published_field_as_the_split_key",
    }
    binding_report = {
        "modalities": modality_bindings,
        "s4_binding_rule": (
            "join each qualified S4 modality to its corresponding S3 modality by explicit "
            "embryo ID; join rows by explicit semantic cell ID"
        ),
        "global_positional_binding_permitted": False,
        "blocked_modalities": [
            item["id"]
            for item in modality_bindings
            if item["s4_join_status"] == "blocked_header_identity"
        ],
    }
    return relation_report, binding_report, {
        "split_manifest": split_manifest,
        "overlap": overlap,
    }


def _write_run_atomic(output_root: Path, payloads: Mapping[str, Any]) -> None:
    if output_root.exists():
        raise FileExistsError(f"qualification output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for name in sorted(payloads):
            _write_json_atomic(temporary / name, payloads[name])
        checksum_lines = []
        for name in sorted(payloads):
            checksum_lines.append(f"{_sha256(temporary / name)}  {name}")
        _write_bytes_atomic(
            temporary / "SHA256SUMS.txt",
            ("\n".join(checksum_lines) + "\n").encode("utf-8"),
        )
        _write_bytes_atomic(temporary / "SUCCESS", b"SUCCESS\n")
        os.replace(temporary, output_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def freeze_qualification(
    manifest_or_path: ScaleupQualificationManifest | str | Path,
    source_dir: str | Path,
    output_root: str | Path,
    *,
    academic_use_attested: bool,
) -> dict[str, Any]:
    manifest = _manifest(manifest_or_path)
    if manifest.rights.operator_attestation_required and not academic_use_attested:
        raise PermissionError(manifest.rights.attestation_text)
    source_path = Path(source_dir).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"processed source directory is missing: {source_path}")
    output_path = Path(output_root).resolve()
    if output_path.exists():
        raise FileExistsError(f"qualification output root already exists: {output_path}")

    manifest_path = (
        Path(manifest_or_path).resolve()
        if isinstance(manifest_or_path, (str, Path))
        else None
    )
    manifest_sha256 = (
        _sha256(manifest_path)
        if manifest_path is not None
        else hashlib.sha256(_canonical_bytes(manifest.model_dump(mode="json"))).hexdigest()
    )
    qualifier_module_sha256 = _sha256(Path(__file__).resolve())
    receipts, scans, package_summaries = _scan_sources(manifest, source_path)
    relation_report, binding_report, relation_payloads = _qualify_relations(manifest, scans)
    split_manifest = relation_payloads["split_manifest"]
    overlap = relation_payloads["overlap"]

    schema_files = []
    for contract in manifest.sources:
        schema_files.append(
            {
                "filename": contract.filename,
                "role": contract.role,
                "package": package_summaries[contract.filename],
                "sheets": [
                    _sheet_manifest(scans[contract.filename][item.name])
                    for item in contract.sheets
                ],
            }
        )
    workbook_schema = {
        "schema_version": WORKBOOK_SCHEMA_VERSION,
        "source_manifest_sha256": manifest_sha256,
        "qualifier_module_sha256": qualifier_module_sha256,
        "files": schema_files,
        "relations": relation_report,
        "s4_bindings": binding_report,
        "outcome_prevalence_inspected": False,
    }
    source_receipt = {
        "schema_version": SOURCE_RECEIPT_VERSION,
        "source_manifest_sha256": manifest_sha256,
        "qualifier_module_sha256": qualifier_module_sha256,
        "academic_use_attested": academic_use_attested,
        "permitted_use": manifest.rights.permitted_use,
        "source_redistribution_permitted": False,
        "commercial_use_permitted": False,
        "public_training_permitted": False,
        "sources": receipts,
        "source_bundle_sha256": _stable_hash(receipts),
    }
    analysis_manifest = {
        "schema_version": ANALYSIS_MANIFEST_VERSION,
        "analysis_id": manifest.analysis_id,
        "classification": manifest.classification,
        "qualifier_module_sha256": qualifier_module_sha256,
        "validated": False,
        "biological_claims_permitted": False,
        "anti_circularity": manifest.anti_circularity,
        "claims": manifest.claims,
        "stage_encoding": manifest.stage_encoding.model_dump(mode="json"),
        "model_launch_status": "blocked_pending_semantic_cell_stage_adapter_freeze",
        "s4_secondary_outcome_exclusions": binding_report["blocked_modalities"],
        "continuous_s3_cnd1_endpoint_qualified": True,
        "split_manifest_sha256": _stable_hash(split_manifest),
        "workbook_schema_manifest_sha256": _stable_hash(workbook_schema),
    }
    gates = {
        "source_identity": True,
        "xlsx_package_integrity": True,
        "exact_sheet_schema": True,
        "no_formulas_or_external_relationships": True,
        "control_identity_consistency": True,
        "measurement_embryo_set_consistency": True,
        "wormbase_gene_join_coverage": True,
        "s4_embryo_id_binding_frozen_for_qualified_modalities": True,
        "s4_semantic_cell_row_binding_frozen": True,
        "s4_cnd1_status_header_identity": False,
        "s4_cnd1_status_excluded": True,
        "exposed_pilot_targets_excluded": True,
        "whole_gene_partitions_frozen": True,
        "outcome_prevalence_not_inspected": True,
        "semantic_cell_stage_adapter_frozen": False,
    }
    qualification = {
        "schema_version": QUALIFICATION_VERSION,
        "technical_status": "success",
        "qualification_status": "qualified_with_exclusions",
        "model_launch_status": "blocked_pending_semantic_cell_stage_adapter_freeze",
        "source_manifest_sha256": manifest_sha256,
        "qualifier_module_sha256": qualifier_module_sha256,
        "counts": {
            "reference_controls": relation_report["reference_controls"],
            "measurement_embryos": relation_report["measurement_embryos"],
            "measurement_genes": relation_report["measurement_genes"],
            "claim_bearing_embryos": sum(
                item["embryo_count"] for item in split_manifest["claim_bearing_genes"]
            ),
            "claim_bearing_genes": len(split_manifest["claim_bearing_genes"]),
            "development_only_exposed_embryos": overlap["embryos"],
            "development_only_exposed_genes": overlap["genes"],
            "qualified_s4_secondary_modalities": 5,
            "blocked_s4_secondary_modalities": 1,
        },
        "exposed_overlap": overlap,
        "gates": gates,
        "warnings": [
            {
                "id": "s4_cnd1_status_header_identity_blocked",
                "message": (
                    "The S4 CND-1 status header contains aliases and duplicate embryo IDs; "
                    "that secondary call sheet is excluded. Continuous CND-1 values in S3 "
                    "remain qualified."
                ),
            },
            {
                "id": "stage_landmarks_not_explicit_frames",
                "message": (
                    "The processed workbooks are cell-resolved matrices, not frame-resolved "
                    "26/200-cell snapshots; freeze a control-derived semantic-cell stage adapter "
                    "before model launch."
                ),
            },
            *(
                [
                    {
                        "id": "public_name_alias_mismatch",
                        "message": (
                            "One or more S1/S3 public-name labels differ; joins use the stable "
                            "WormBase gene ID; mismatches are recorded in the schema "
                            "manifest."
                        ),
                    }
                ]
                if relation_report["public_name_alias_mismatches"]
                else []
            ),
            *(
                [
                    {
                        "id": "compound_wormbase_id_field",
                        "message": (
                            "One published WormBase field contains an updated ID in parentheses; "
                            "the exact immutable field is retained as one whole-gene split key."
                        ),
                    }
                ]
                if relation_report["compound_wormbase_id_fields"]
                else []
            ),
        ],
        "outcome_prevalence_inspected": False,
        "scientific_status": "not_estimable_until_model_run",
    }
    payloads = {
        "analysis_manifest.json": analysis_manifest,
        "qualification.json": qualification,
        "source_receipt.json": source_receipt,
        "split_manifest.json": split_manifest,
        "workbook_schema_manifest.json": workbook_schema,
    }
    _write_run_atomic(output_path, payloads)
    return qualification


def verify_qualification(
    run_root: str | Path,
    manifest: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"qualification run root is missing: {root}")
    expected = OUTPUT_FILES | {"SHA256SUMS.txt", "SUCCESS"}
    observed = {item.name for item in root.iterdir()}
    if observed != expected:
        raise ValueError(
            f"qualification output inventory differs: expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )
    if (root / "SUCCESS").read_bytes() != b"SUCCESS\n":
        raise ValueError("qualification SUCCESS marker is invalid")
    lines = (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    if len(lines) != len(OUTPUT_FILES):
        raise ValueError("qualification checksum manifest has the wrong length")
    checked = 0
    checked_names: set[str] = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if (
            match is None
            or match.group(2) not in OUTPUT_FILES
            or match.group(2) in checked_names
        ):
            raise ValueError("qualification checksum manifest is malformed")
        if _sha256(root / match.group(2)) != match.group(1):
            raise ValueError(f"qualification checksum failed: {match.group(2)}")
        checked_names.add(match.group(2))
        checked += 1
    if checked_names != OUTPUT_FILES:
        raise ValueError("qualification checksum manifest does not cover the exact output set")
    qualification = json.loads((root / "qualification.json").read_text(encoding="utf-8"))
    if (
        qualification.get("schema_version") != QUALIFICATION_VERSION
        or qualification.get("technical_status") != "success"
        or qualification.get("qualification_status") != "qualified_with_exclusions"
    ):
        raise ValueError("qualification identity or technical status is invalid")
    if manifest is not None:
        expected_hash = _sha256(Path(manifest).resolve())
        if qualification.get("source_manifest_sha256") != expected_hash:
            raise ValueError("qualification source manifest differs")
    return {
        "schema_version": VERIFY_VERSION,
        "verified": True,
        "checked_files": checked,
        "qualification_status": qualification["qualification_status"],
        "model_launch_status": qualification["model_launch_status"],
        "qualification_sha256": _sha256(root / "qualification.json"),
    }


def _emit(payload: Any, output: str | Path | None = None) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(content)
    else:
        destination = Path(output).resolve()
        _write_bytes_atomic(destination, content.encode("utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wormctx.poc.developmental_omix_scaleup",
        description="Qualify and freeze the governed OMIX709 S1-S4 scale-up",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="qualify S1-S4 and freeze gene partitions")
    freeze.add_argument("--manifest", required=True)
    freeze.add_argument("--source-dir", required=True)
    freeze.add_argument("--output-root", required=True)
    freeze.add_argument(
        "--attest-controlled-noncommercial-academic-use",
        action="store_true",
        help="record the required controlled noncommercial academic-use attestation",
    )
    verify = commands.add_parser("verify", help="verify a completed qualification bundle")
    verify.add_argument("--run-root", required=True)
    verify.add_argument("--manifest")
    verify.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        payload = freeze_qualification(
            args.manifest,
            args.source_dir,
            args.output_root,
            academic_use_attested=args.attest_controlled_noncommercial_academic_use,
        )
        _emit(payload)
    elif args.command == "verify":
        payload = verify_qualification(args.run_root, args.manifest)
        _emit(payload, args.output)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
