from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import pytest

from wormctx.poc import developmental_omix_scaleup as scaleup


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/developmental_genetics/processed_data_qualification/config/processed_data_qualification.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _column_name(index: int) -> str:
    """Return an A1 column name for a zero-based column index."""

    value = index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _sheet_xml(dimension: str, rows: dict[int, dict[int, Any]]) -> bytes:
    encoded_rows = []
    for row_number, values in sorted(rows.items()):
        cells = []
        for column, value in sorted(values.items()):
            if value is None:
                continue
            reference = f"{_column_name(column)}{row_number}"
            cells.append(
                f'<c r="{reference}" t="inlineStr"><is><t>'
                f"{escape(str(value))}</t></is></c>"
            )
        encoded_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<worksheet xmlns="{scaleup.MAIN_NS}">'
        f'<dimension ref="{dimension}"/><sheetData>{"".join(encoded_rows)}</sheetData>'
        "</worksheet>"
    ).encode("utf-8")


def _write_member(archive: zipfile.ZipFile, name: str, content: str | bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, content.encode("utf-8") if isinstance(content, str) else content)


def _write_xlsx(
    path: Path,
    sheets: list[tuple[str, str, dict[int, dict[int, Any]]]],
) -> None:
    """Build the minimal standards-compliant XLSX package needed by the scanner."""

    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.'
        'spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{overrides}</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Relationships xmlns="{scaleup.PACKAGE_REL_NS}">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    sheet_catalog = "".join(
        f'<sheet name={quoteattr(name)} sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _dimension, _rows) in enumerate(sheets, 1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<workbook xmlns="{scaleup.MAIN_NS}" xmlns:r="{scaleup.DOC_REL_NS}">'
        f"<sheets>{sheet_catalog}</sheets></workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Relationships xmlns="{scaleup.PACKAGE_REL_NS}">'
        + "".join(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
            for index in range(1, len(sheets) + 1)
        )
        + "</Relationships>"
    )

    with zipfile.ZipFile(path, "w") as archive:
        _write_member(archive, "[Content_Types].xml", content_types)
        _write_member(archive, "_rels/.rels", root_rels)
        _write_member(archive, "xl/workbook.xml", workbook)
        _write_member(archive, "xl/_rels/workbook.xml.rels", workbook_rels)
        for index, (_name, dimension, rows) in enumerate(sheets, 1):
            _write_member(
                archive,
                f"xl/worksheets/sheet{index}.xml",
                _sheet_xml(dimension, rows),
            )


def _synthetic_bundle(
    tmp_path: Path,
    *,
    mismatched_modality_embryos: bool = False,
    s4_width_mismatch: bool = False,
    s4_bad_semantic_cell: bool = False,
) -> tuple[scaleup.ScaleupQualificationManifest, Path]:
    """Create a tiny S1--S4 bundle and a content-addressed reduced manifest."""

    base = scaleup.load_manifest(CONFIG)
    source_dir = tmp_path / "processed"
    source_dir.mkdir()

    controls = ["CTR-1", "CTR-2"]
    records = [
        ("WBGene00000001", "C30H7.2", "EMB-1"),
        ("WBGene00000002", "GENE-2", "EMB-2"),
        ("WBGene00000003", "GENE-3", "EMB-3"),
        ("WBGene00000004", "GENE-4", "EMB-4"),
        ("WBGene00000005", "GENE-5", "EMB-5"),
        ("WBGene00000006", "GENE-6", "EMB-6"),
    ]
    embryos = [item[2] for item in records]

    s1_inventory: dict[int, dict[int, Any]] = {
        1: {0: "WormBase gene ID", 1: "Public name", 6: "Selected", 8: "Validation"}
    }
    s3_metadata: dict[int, dict[int, Any]] = {
        1: {0: "WormBase gene ID", 1: "Public name", 2: "Embryo ID", 4: "Treatment"}
    }
    s3_summary: dict[int, dict[int, Any]] = {
        1: {0: "WormBase gene ID", 1: "Public name", 2: "Embryo ID"}
    }
    for row_number, (gene_id, public_name, embryo_id) in enumerate(records, 2):
        s1_inventory[row_number] = {
            0: gene_id,
            1: public_name,
            6: "yes",
            8: "yes",
        }
        s3_metadata[row_number] = {
            0: gene_id,
            1: public_name,
            2: embryo_id,
            4: "lineaging and cell lineage tracing",
        }
        s3_summary[row_number] = {0: gene_id, 1: public_name, 2: embryo_id}

    s1_sheets = [
        ("Legend", "A1", {1: {0: "Legend"}}),
        ("1", "A1:J7", s1_inventory),
        ("2", "A1", {1: {0: "Unused in qualification"}}),
    ]
    s2_sheets: list[tuple[str, str, dict[int, dict[int, Any]]]] = [
        ("Legend", "A1", {1: {0: "Legend"}})
    ]
    s3_sheets: list[tuple[str, str, dict[int, dict[int, Any]]]] = [
        ("Legend", "A1", {1: {0: "Legend"}}),
        ("1", "A1:F7", s3_metadata),
    ]
    s4_sheets: list[tuple[str, str, dict[int, dict[int, Any]]]] = [
        ("Legend", "A1", {1: {0: "Legend"}})
    ]
    for index, modality in enumerate(base.modalities):
        s2_sheets.append(
            (
                modality.s2_sheet,
                "A1:C3",
                {
                    1: {0: modality.s2_title},
                    2: {0: "Semantic cell", 1: controls[0], 2: controls[1]},
                    3: {0: "ABa", 1: "1.0", 2: "1.1"},
                },
            )
        )
        modality_embryos = list(embryos)
        if mismatched_modality_embryos and index == 1:
            modality_embryos[-1] = "EMB-DIFFERENT"
        s3_sheets.append(
            (
                modality.s3_sheet,
                "A1:G3",
                {
                    1: {0: modality.s3_title},
                    2: {
                        0: "Semantic cell",
                        **{
                            column: embryo_id
                            for column, embryo_id in enumerate(modality_embryos, 1)
                        },
                    },
                    3: {0: "ABa", **{column: "1.0" for column in range(1, 7)}},
                },
            )
        )
        width = "F" if s4_width_mismatch and index == 0 else "G"
        semantic_cell = "NOT-A-LINEAGE-CELL" if s4_bad_semantic_cell and index == 0 else "ABa"
        defect_embryos = list(modality_embryos)
        if modality.s4_qualification == "blocked_header_identity":
            defect_embryos[-1] = defect_embryos[-2]
        s4_sheets.append(
            (
                modality.s4_sheet,
                f"A1:{width}2",
                {
                    1: {
                        0: modality.s4_title,
                        **{
                            column: embryo_id
                            for column, embryo_id in enumerate(defect_embryos, 1)
                        },
                    },
                    2: {0: semantic_cell},
                },
            )
        )
    s3_sheets.append(("8", "A1:F7", s3_summary))
    s4_sheets.extend(
        [
            ("7", "A1", {1: {0: "Unused in qualification"}}),
            ("8", "A1", {1: {0: "Unused in qualification"}}),
        ]
    )

    sheets_by_filename = {
        "Table_S1.xlsx": s1_sheets,
        "Table_S2.xlsx": s2_sheets,
        "Table_S3.xlsx": s3_sheets,
        "Table_S4.xlsx": s4_sheets,
    }
    payload = base.model_dump(mode="json")
    payload["analysis_id"] = "synthetic_omix709_processed_scaleup_qualification"
    payload["expected_counts"] = {
        "reference_controls": len(controls),
        "measurement_embryos": len(embryos),
        "measurement_genes": len(records),
    }
    payload["overlap"]["exposed_public_names"] = ["C30H7.2"]
    payload["overlap"]["observed_s1_s3_public_name_alias_mismatches"] = 0

    for source in payload["sources"]:
        sheets = sheets_by_filename[source["filename"]]
        path = source_dir / source["filename"]
        _write_xlsx(path, sheets)
        source["bytes"] = path.stat().st_size
        source["sha256"] = _sha256(path)
        source["sheets"] = [
            {"name": name, "dimension": dimension}
            for name, dimension, _rows in sheets
        ]
    return scaleup.ScaleupQualificationManifest.model_validate(payload), source_dir


def _mutate_manifest(
    manifest: scaleup.ScaleupQualificationManifest,
    mutate: Any,
) -> scaleup.ScaleupQualificationManifest:
    payload = manifest.model_dump(mode="json")
    mutate(payload)
    return scaleup.ScaleupQualificationManifest.model_validate(payload)


def test_frozen_manifest_parses_and_rejects_broader_rights() -> None:
    manifest = scaleup.load_manifest(CONFIG)

    assert manifest.schema_version == scaleup.SCHEMA_VERSION
    assert [source.filename for source in manifest.sources] == scaleup.EXPECTED_SOURCE_FILENAMES
    assert (
        manifest.rights.permitted_use
        == "controlled_internal_noncommercial_academic_analysis_only"
    )
    assert manifest.rights.operator_attestation_required is True
    assert manifest.rights.source_redistribution_permitted is False
    assert manifest.rights.commercial_use_permitted is False
    assert manifest.rights.public_training_permitted is False

    for field, value in (
        ("source_redistribution_permitted", True),
        ("commercial_use_permitted", True),
        ("public_training_permitted", True),
        ("operator_attestation_required", False),
    ):
        payload = manifest.model_dump(mode="json")
        payload["rights"][field] = value
        with pytest.raises(ValueError, match="(rights contract|attestation must be required)"):
            scaleup.ScaleupQualificationManifest.model_validate(payload)


def test_reduced_s1_s4_bundle_qualifies_and_freezes_outputs(tmp_path: Path) -> None:
    manifest, source_dir = _synthetic_bundle(tmp_path)
    output = tmp_path / "qualification"

    result = scaleup.freeze_qualification(
        manifest,
        source_dir,
        output,
        academic_use_attested=True,
    )

    assert result["technical_status"] == "success"
    assert result["qualification_status"] == "qualified_with_exclusions"
    assert result["counts"] == {
        "reference_controls": 2,
        "measurement_embryos": 6,
        "measurement_genes": 6,
        "claim_bearing_embryos": 5,
        "claim_bearing_genes": 5,
        "development_only_exposed_embryos": 1,
        "development_only_exposed_genes": 1,
        "qualified_s4_secondary_modalities": 5,
        "blocked_s4_secondary_modalities": 1,
    }
    assert result["outcome_prevalence_inspected"] is False
    assert result["gates"]["whole_gene_partitions_frozen"] is True
    assert result["gates"]["s4_cnd1_status_excluded"] is True
    assert result["gates"]["semantic_cell_stage_adapter_frozen"] is False

    receipt = json.loads((output / "source_receipt.json").read_text(encoding="utf-8"))
    assert receipt["academic_use_attested"] is True
    assert receipt["source_redistribution_permitted"] is False
    assert receipt["commercial_use_permitted"] is False
    assert receipt["public_training_permitted"] is False
    assert all(
        item["sha256"] == _sha256(source_dir / item["filename"])
        for item in receipt["sources"]
    )

    verification = scaleup.verify_qualification(output)
    assert verification["verified"] is True
    assert verification["checked_files"] == len(scaleup.OUTPUT_FILES)


def test_missing_academic_use_attestation_fails_before_output(tmp_path: Path) -> None:
    manifest, source_dir = _synthetic_bundle(tmp_path)
    output = tmp_path / "qualification"

    with pytest.raises(PermissionError, match="controlled, noncommercial academic use"):
        scaleup.freeze_qualification(
            manifest,
            source_dir,
            output,
            academic_use_attested=False,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("bytes", lambda value: value + 1, "byte identity failed"),
        ("sha256", lambda _value: "0" * 64, "SHA-256 identity failed"),
    ],
)
def test_source_byte_and_hash_identity_fail_closed(
    tmp_path: Path,
    field: str,
    bad_value: Any,
    message: str,
) -> None:
    manifest, source_dir = _synthetic_bundle(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["sources"][0][field] = bad_value(payload["sources"][0][field])

    invalid = _mutate_manifest(manifest, mutate)
    with pytest.raises(ValueError, match=message):
        scaleup.freeze_qualification(
            invalid,
            source_dir,
            tmp_path / "qualification",
            academic_use_attested=True,
        )


def test_mismatched_modality_embryo_set_is_rejected(tmp_path: Path) -> None:
    manifest, source_dir = _synthetic_bundle(
        tmp_path,
        mismatched_modality_embryos=True,
    )

    with pytest.raises(ValueError, match="measurement embryo set differs across modalities"):
        scaleup.freeze_qualification(
            manifest,
            source_dir,
            tmp_path / "qualification",
            academic_use_attested=True,
        )


@pytest.mark.parametrize(
    ("bundle_options", "message"),
    [
        ({"s4_width_mismatch": True}, "S4 embryo width cannot bind to S3"),
        ({"s4_bad_semantic_cell": True}, "S4 semantic cells are not a subset of S3"),
    ],
)
def test_s4_width_and_semantic_row_contracts_fail_closed(
    tmp_path: Path,
    bundle_options: dict[str, bool],
    message: str,
) -> None:
    manifest, source_dir = _synthetic_bundle(tmp_path, **bundle_options)

    with pytest.raises(ValueError, match=message):
        scaleup.freeze_qualification(
            manifest,
            source_dir,
            tmp_path / "qualification",
            academic_use_attested=True,
        )


def test_whole_gene_split_is_deterministic_and_excludes_exposed_target() -> None:
    manifest = scaleup.load_manifest(CONFIG)
    metadata = {
        "EMB-1": {
            "gene_id": "WBGene00000001",
            "public_name": "C30H7.2",
            "embryo_id": "EMB-1",
        },
        "EMB-2B": {
            "gene_id": "WBGene00000002",
            "public_name": "GENE-2",
            "embryo_id": "EMB-2B",
        },
        "EMB-2A": {
            "gene_id": "WBGene00000002",
            "public_name": "GENE-2",
            "embryo_id": "EMB-2A",
        },
        **{
            f"EMB-{index}": {
                "gene_id": f"WBGene{index:08d}",
                "public_name": f"GENE-{index}",
                "embryo_id": f"EMB-{index}",
            }
            for index in range(3, 7)
        },
    }

    forward = scaleup._build_split(manifest, metadata)
    reverse = scaleup._build_split(manifest, dict(reversed(list(metadata.items()))))
    assert forward == reverse

    split, overlap = forward
    claim_gene_ids = {item["gene_id"] for item in split["claim_bearing_genes"]}
    exposed = split["development_only_exposed_pilot_genes"]
    assert "WBGene00000001" not in claim_gene_ids
    assert [item["gene_id"] for item in exposed] == ["WBGene00000001"]
    assert exposed[0]["partition"] == "development_only_exposed_pilot"
    assert overlap["present"] == ["C30H7.2"]
    assert split["counts"]["train"]["genes"] == 3
    assert split["counts"]["validation"]["genes"] == 1
    assert split["counts"]["sealed_test"]["genes"] == 1
    gene_two = next(
        item
        for item in split["claim_bearing_genes"]
        if item["gene_id"] == "WBGene00000002"
    )
    assert gene_two["embryo_ids"] == ["EMB-2A", "EMB-2B"]


def test_verifier_detects_payload_tampering_and_requires_checksum_coverage(
    tmp_path: Path,
) -> None:
    manifest, source_dir = _synthetic_bundle(tmp_path)
    output = tmp_path / "qualification"
    scaleup.freeze_qualification(
        manifest,
        source_dir,
        output,
        academic_use_attested=True,
    )

    qualification = output / "qualification.json"
    original = qualification.read_bytes()
    qualification.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="qualification checksum failed"):
        scaleup.verify_qualification(output)
    qualification.write_bytes(original)

    sums = output / "SHA256SUMS.txt"
    lines = sums.read_text(encoding="utf-8").splitlines()
    sums.write_text("\n".join([lines[0], *lines[:-1]]) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="(coverage|duplicate|inventory|manifest)"):
        scaleup.verify_qualification(output)
