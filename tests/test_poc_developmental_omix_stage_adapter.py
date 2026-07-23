from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import pytest

from wormctx.poc import developmental_omix_scaleup as scaleup
from wormctx.poc import developmental_omix_stage_adapter as adapter


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/developmental_genetics/semantic_stage_and_lineage_preparation/config/semantic_stage_adapter.json"
PARENT_CONFIG = ROOT / "experiments/developmental_genetics/processed_data_qualification/config/processed_data_qualification.json"
RAW_HEADER = b"\ttime\tcell_name\tX\tY\tZ\tsize\traw-expression\tblot-correction"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _column_name(index: int) -> str:
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


def _daughters(parent: str) -> tuple[str, str]:
    special = {
        "AB": ("ABa", "ABp"),
        "P1": ("EMS", "P2"),
        "EMS": ("MS", "E"),
        "P2": ("C", "P3"),
        "P3": ("D", "P4"),
        "P4": ("Z2", "Z3"),
    }
    return special.get(parent, (f"{parent}a", f"{parent}p"))


def _lineage_catalog() -> tuple[list[str], dict[str, int]]:
    """Return a deterministic binary fixture that reaches an exact 200-cell frontier."""

    frontier = ["AB", "P1"]
    cells = set(frontier)
    event_time: dict[str, int] = {}
    for event in range(1, 199):
        parent = frontier.pop(0)
        daughters = _daughters(parent)
        event_time[parent] = event
        frontier.extend(daughters)
        cells.update(daughters)
    assert len(frontier) == 200
    assert len(cells) == 398
    return sorted(cells), event_time


def _raw_inventory(raw_control_dir: Path) -> tuple[int, int, str]:
    inventory = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(raw_control_dir.iterdir(), key=lambda item: item.name)
    ]
    canonical = (
        json.dumps(inventory, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    return len(inventory), sum(item["bytes"] for item in inventory), hashlib.sha256(
        canonical
    ).hexdigest()


def _write_raw_controls(root: Path, controls: int = 2) -> tuple[Path, Path]:
    archive = root / "OMIX709-05-53.rar"
    archive.write_bytes(b"synthetic governed raw-control archive\n")
    raw_control_dir = root / "Raw_data_Control"
    raw_control_dir.mkdir()
    cells, event_time = _lineage_catalog()
    parent_by_child: dict[str, str] = {}
    frontier = ["AB", "P1"]
    for _event in range(1, 199):
        parent = frontier.pop(0)
        daughters = _daughters(parent)
        frontier.extend(daughters)
        parent_by_child.update({daughter: parent for daughter in daughters})
    for control in range(1, controls + 1):
        rows = [RAW_HEADER + b"\n"]
        for index, cell in enumerate(cells):
            time = 0 if cell in {"AB", "P1"} else event_time[parent_by_child[cell]]
            prefix = f"{index}\t{time}\t{cell}\t".encode("ascii")
            # Invalid UTF-8 in every nonpermitted field proves that the adapter
            # never text-decodes the outcome-column remainder.
            poison = b"\xff\t\xfe\t\xfd\t\xfc\t\xfb\t\xfa\n"
            rows.append(prefix + poison)
        (raw_control_dir / f"ctr_emb{control}_raw_data.txt").write_bytes(b"".join(rows))
    return archive, raw_control_dir


def _bind_raw_source_contract(
    contract_payload: dict[str, Any],
    archive: Path,
    raw_control_dir: Path,
    *,
    expected_union_cell_count: int = 398,
) -> None:
    contract_payload["support"]["minimum_reference_count"] = 1
    file_count, total_bytes, inventory_sha256 = _raw_inventory(raw_control_dir)
    source = contract_payload["raw_control_lineage_source"]
    source["archive"] = {
        "filename": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": _sha256(archive),
    }
    source["extracted"]["expected_file_count"] = file_count
    source["extracted"]["expected_total_bytes"] = total_bytes
    source["extracted"]["inventory_sha256"] = inventory_sha256
    source["expected_union_cell_count"] = expected_union_cell_count
    source["expected_binary_parent_count"] = 199
    source["expected_supported_division_events"] = 198
    source["expected_unsupported_division_events"] = 0
    source["expected_sibling_birth_time_disagreements"] = 0
    source["expected_input_event_count"] = 24
    source["expected_endpoint_event_count"] = 198


def _rebind_raw_source_contract(
    contract: adapter.StageAdapterContract,
    archive: Path,
    raw_control_dir: Path,
    *,
    expected_union_cell_count: int = 398,
) -> adapter.StageAdapterContract:
    payload = contract.model_dump(mode="json")
    _bind_raw_source_contract(
        payload,
        archive,
        raw_control_dir,
        expected_union_cell_count=expected_union_cell_count,
    )
    return adapter.StageAdapterContract.model_validate(payload)


def _bundle(
    tmp_path: Path,
    *,
    poison_division_timing_s2_catalog: bool = False,
) -> tuple[Path, Path, Path, Path, Path, adapter.StageAdapterContract]:
    base = scaleup.load_manifest(PARENT_CONFIG)
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
    cells, event_time = _lineage_catalog()

    s1_inventory: dict[int, dict[int, Any]] = {
        1: {0: "WormBase gene ID", 1: "Public name", 6: "Selected", 8: "Validation"}
    }
    s3_metadata: dict[int, dict[int, Any]] = {
        1: {0: "WormBase gene ID", 1: "Public name", 2: "Embryo ID", 4: "Treatment"}
    }
    s3_summary: dict[int, dict[int, Any]] = {
        1: {0: "WormBase gene ID", 1: "Public name", 2: "Embryo ID"}
    }
    for row, (gene_id, public_name, embryo_id) in enumerate(records, 2):
        s1_inventory[row] = {0: gene_id, 1: public_name, 6: "yes", 8: "yes"}
        s3_metadata[row] = {
            0: gene_id,
            1: public_name,
            2: embryo_id,
            4: "lineaging and cell lineage tracing",
        }
        s3_summary[row] = {0: gene_id, 1: public_name, 2: embryo_id}

    s1_sheets = [
        ("Legend", "A1", {1: {0: "Legend"}}),
        ("1", "A1:J7", s1_inventory),
        ("2", "A1", {1: {0: "Unused"}}),
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
        s2_rows: dict[int, dict[int, Any]] = {
            1: {0: modality.s2_title},
            2: {0: "Semantic cell", 1: controls[0], 2: controls[1]},
        }
        s3_order = list(reversed(embryos)) if index == 1 else list(embryos)
        s3_rows: dict[int, dict[int, Any]] = {
            1: {0: modality.s3_title},
            2: {
                0: "Semantic cell",
                **{column: embryo for column, embryo in enumerate(s3_order, 1)},
            },
        }
        for row, cell in enumerate(cells, 3):
            if (
                poison_division_timing_s2_catalog
                and modality.id == "division_timing"
                and cell in {"AB", "Caap"}
            ):
                # Reproduce the real-data failure mode: Table S2 is a measured-row
                # support table, not the authoritative daughter catalog.  The raw
                # control lineage must still be able to define the stage frontiers.
                s3_rows[row] = {
                    0: cell,
                    **{
                        column: "PERTURBATION-VALUE-NOT-FOR-ADAPTER"
                        for column in range(1, 7)
                    },
                }
                continue
            value = event_time.get(cell)
            s2_rows[row] = {0: cell}
            if modality.id == "three_dimensional_position":
                s2_rows[row].update({1: "(250, 75, 68)", 2: "(251.5, 74, 69)"})
            elif value is not None:
                s2_rows[row].update({1: value, 2: value + 0.25})
            elif modality.id != "division_timing":
                s2_rows[row].update({1: 1.0, 2: 1.25})
            s3_rows[row] = {
                0: cell,
                **{column: "PERTURBATION-VALUE-NOT-FOR-ADAPTER" for column in range(1, 7)},
            }
        s2_sheets.append((modality.s2_sheet, f"A1:C{len(cells) + 2}", s2_rows))
        s3_sheets.append((modality.s3_sheet, f"A1:G{len(cells) + 2}", s3_rows))

        s4_order = list(s3_order)
        if modality.s4_qualification == "blocked_header_identity":
            s4_order[-1] = s4_order[-2]
        s4_rows: dict[int, dict[int, Any]] = {
            1: {
                0: modality.s4_title,
                **{column: embryo for column, embryo in enumerate(s4_order, 1)},
            }
        }
        for row, cell in enumerate(cells, 2):
            s4_rows[row] = {
                0: cell,
                **{column: "OUTCOME-VALUE-NOT-FOR-ADAPTER" for column in range(1, 7)},
            }
        s4_sheets.append((modality.s4_sheet, f"A1:G{len(cells) + 1}", s4_rows))
    s3_sheets.append(("8", "A1:F7", s3_summary))
    s4_sheets.extend(
        [
            ("7", "A1", {1: {0: "Descriptive only"}}),
            ("8", "A1", {1: {0: "Descriptive only"}}),
        ]
    )

    sheets_by_file = {
        "Table_S1.xlsx": s1_sheets,
        "Table_S2.xlsx": s2_sheets,
        "Table_S3.xlsx": s3_sheets,
        "Table_S4.xlsx": s4_sheets,
    }
    parent_payload = base.model_dump(mode="json")
    parent_payload["analysis_id"] = "synthetic_stage_adapter_parent"
    parent_payload["expected_counts"] = {
        "reference_controls": len(controls),
        "measurement_embryos": len(embryos),
        "measurement_genes": len(records),
    }
    parent_payload["overlap"]["exposed_public_names"] = ["C30H7.2"]
    parent_payload["overlap"]["observed_s1_s3_public_name_alias_mismatches"] = 0
    for source in parent_payload["sources"]:
        sheets = sheets_by_file[source["filename"]]
        path = source_dir / source["filename"]
        _write_xlsx(path, sheets)
        source["bytes"] = path.stat().st_size
        source["sha256"] = _sha256(path)
        source["sheets"] = [
            {"name": name, "dimension": dimension} for name, dimension, _rows in sheets
        ]
    parent_manifest = tmp_path / "omix709_processed_scaleup_qualification_20260721.json"
    parent_manifest.write_text(
        json.dumps(parent_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parent_root = tmp_path / "parent-qualification"
    scaleup.freeze_qualification(
        parent_manifest,
        source_dir,
        parent_root,
        academic_use_attested=True,
    )

    contract_payload = adapter.load_contract(CONFIG).model_dump(mode="json")
    contract_payload["analysis_id"] = "synthetic_semantic_stage_adapter"
    contract_payload["parent"]["manifest_sha256"] = _sha256(parent_manifest)
    contract_payload["support"]["minimum_reference_count"] = 1
    raw_control_archive, raw_control_dir = _write_raw_controls(tmp_path)
    _bind_raw_source_contract(contract_payload, raw_control_archive, raw_control_dir)
    contract = adapter.StageAdapterContract.model_validate(contract_payload)
    return (
        source_dir,
        parent_manifest,
        parent_root,
        raw_control_archive,
        raw_control_dir,
        contract,
    )


def _overlap_receipt(
    parent_root: Path,
    *,
    scaleup_embryo_id: str = "EMB-1",
    evidence_record: Path | None = None,
) -> dict[str, Any]:
    source_receipt = json.loads(
        (parent_root / "source_receipt.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": adapter.OVERLAP_RECEIPT_VERSION,
        "pilot_normalized_sha256": adapter.PILOT_NORMALIZED_SHA256,
        "pilot_normalized_bytes": adapter.PILOT_NORMALIZED_BYTES,
        "pilot_embryos": adapter.PILOT_EMBRYOS,
        "scaleup_source_bundle_sha256": source_receipt["source_bundle_sha256"],
        "attestation_request_sha256": adapter.OVERLAP_ATTESTATION_REQUEST_SHA256,
        "source_evidence_record_identifier": (
            evidence_record.name if evidence_record is not None else "synthetic-curator-record-1"
        ),
        "source_evidence_record_bytes": (
            evidence_record.stat().st_size if evidence_record is not None else 1
        ),
        "source_evidence_record_sha256": (
            scaleup._sha256(evidence_record) if evidence_record is not None else "b" * 64
        ),
        "attestor_name": "Synthetic Source Curator",
        "attestor_role_and_source_authority": "OMIX709 source depositor",
        "attested_utc": "2026-07-22T20:00:00Z",
        "assertions": {
            "table_s3_uses_only_omix709_05_54_embryos": True,
            "omix709_05_54_and_05_55_are_biological_embryo_disjoint": False,
            "identifier_namespaces_are_complete_and_not_reused_across_archives": True,
            "any_reused_embryos_are_exhaustively_listed_in_overlap_pairs": True,
        },
        "mapping_basis": "source_supported_exact_identity_receipt",
        "mapping_complete": True,
        "outcome_values_used_for_identity_decision": False,
        "pairs": [
            {
                "normalized_embryo_id": "addendum:C30H7.2_emb1",
                "scaleup_embryo_id": scaleup_embryo_id,
                "source_member": "C30H7.2_emb1.txt",
            }
        ],
    }


def test_real_contract_freezes_outcome_blind_access_and_s4_exclusion() -> None:
    contract = adapter.load_contract(CONFIG)

    assert contract.landmarks.input_cell_count == 26
    assert contract.landmarks.endpoint_cell_count == 200
    assert contract.source_access.outcome_prevalence_inspection_permitted is False
    assert contract.source_access.excluded_s4_sheets == ["4"]
    assert contract.support.perturbation_missing_value_policy == (
        "explicit_mask_no_adapter_imputation"
    )
    assert contract.raw_control_lineage_source.archive.sha256 == (
        "cf85a12ec6b6eabf900e303a630e366f61e2a7aeff0b050feb38bb26a06c07a4"
    )
    assert contract.raw_control_lineage_source.extracted.expected_file_count == 105
    assert contract.raw_control_lineage_source.expected_union_cell_count == 794
    assert contract.raw_control_lineage_source.identity_time_access.nonpermitted_value_columns_policy == (
        "never_decode"
    )
    assert contract.lineage.suffix_parent_characters == "aplrdv"

    payload = contract.model_dump(mode="json")
    payload["source_access"]["s4_access"] = "all_values"
    with pytest.raises(ValueError, match="S4 outcome values may not be decoded"):
        adapter.StageAdapterContract.model_validate(payload)


def test_raw_control_lineage_derives_exact_26_and_200_frontiers(tmp_path: Path) -> None:
    payload = adapter.load_contract(CONFIG).model_dump(mode="json")
    archive, raw_control_dir = _write_raw_controls(tmp_path)
    _bind_raw_source_contract(payload, archive, raw_control_dir)
    contract = adapter.StageAdapterContract.model_validate(payload)
    raw_controls = adapter._read_raw_control_lineage_source(
        archive, raw_control_dir, contract
    )

    lineage, stages, receipt = adapter._derive_lineage(raw_controls, contract)

    assert len(stages["input_26"]["frontier"]) == 26
    assert len(stages["input_26"]["completed_division_cells"]) == 24
    assert len(stages["input_26"]["born_cells"]) == 50
    assert len(stages["endpoint_200"]["frontier"]) == 200
    assert len(stages["endpoint_200"]["completed_division_cells"]) == 198
    assert len(lineage["edges_through_200"]) == 398
    assert stages["raw_control_outcome_columns_decoded"] is False
    assert stages["perturbation_measurement_values_read"] is False
    assert stages["s4_outcome_values_read"] is False
    assert receipt["lineage"]["input_frontier_event_count"] == 24
    assert receipt["lineage"]["endpoint_frontier_event_count"] == 198
    assert receipt["access"]["outcome_columns_decoded"] is False


def test_freeze_without_overlap_receipt_is_frozen_but_model_blocked(tmp_path: Path) -> None:
    source_dir, parent_manifest, parent_root, archive, raw_dir, contract = _bundle(tmp_path)
    output = tmp_path / "stage-adapter"

    result = adapter.freeze_stage_adapter(
        contract,
        parent_manifest,
        parent_root,
        source_dir,
        archive,
        raw_dir,
        output,
        academic_use_attested=True,
    )

    assert result["qualification_status"] == "stage_adapter_frozen_with_open_overlap_gate"
    assert result["model_launch_status"] == "blocked_pending_pilot_embryo_overlap_receipt"
    assert result["gates"]["input_26_exact_frontier_frozen"] is True
    assert result["gates"]["endpoint_200_exact_frontier_frozen"] is True
    assert result["gates"]["raw_control_outcome_columns_not_decoded"] is True
    assert result["gates"]["pilot_embryo_overlap_closed"] is False
    assert result["gates"]["sealed_test_opened"] is False
    verification = adapter.verify_stage_adapter(output)
    assert verification["verified"] is True


def test_sparse_s2_division_rows_cannot_define_or_break_lineage(tmp_path: Path) -> None:
    source_dir, parent_manifest, parent_root, archive, raw_dir, contract = _bundle(
        tmp_path,
        poison_division_timing_s2_catalog=True,
    )
    output = tmp_path / "stage-adapter"

    result = adapter.freeze_stage_adapter(
        contract,
        parent_manifest,
        parent_root,
        source_dir,
        archive,
        raw_dir,
        output,
        academic_use_attested=True,
    )

    assert result["qualification_status"] == "stage_adapter_frozen_with_open_overlap_gate"
    lineage = json.loads((output / "lineage_manifest.json").read_text(encoding="utf-8"))
    mappings = json.loads((output / "modality_mappings.json").read_text(encoding="utf-8"))
    division = next(item for item in mappings["modalities"] if item["id"] == "division_timing")
    assert "Caap" in lineage["cells_through_200"]
    assert "AB" not in division["input_26"]["reference_support"]
    assert adapter.verify_stage_adapter(output)["verified"] is True


def test_complete_overlap_receipt_closes_model_launch_and_maps_each_modality(
    tmp_path: Path,
) -> None:
    source_dir, parent_manifest, parent_root, archive, raw_dir, contract = _bundle(tmp_path)
    evidence_path = tmp_path / "source-curator-record.txt"
    evidence_path.write_text("synthetic source-authorized evidence\n", encoding="utf-8")
    receipt_path = tmp_path / "pilot-overlap.json"
    receipt_path.write_text(
        json.dumps(
            _overlap_receipt(parent_root, evidence_record=evidence_path),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_receipt = json.loads(
        (parent_root / "source_receipt.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (parent_root / "split_manifest.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="requires its durable source evidence record"):
        adapter._overlap_status(
            receipt_path,
            None,
            source_receipt["source_bundle_sha256"],
            split_manifest,
        )
    tampered_evidence = tmp_path / "tampered-source-curator-record.txt"
    tampered_evidence.write_bytes(b"x" * evidence_path.stat().st_size)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        adapter._overlap_status(
            receipt_path,
            tampered_evidence,
            source_receipt["source_bundle_sha256"],
            split_manifest,
        )
    output = tmp_path / "stage-adapter"

    result = adapter.freeze_stage_adapter(
        contract,
        parent_manifest,
        parent_root,
        source_dir,
        archive,
        raw_dir,
        output,
        academic_use_attested=True,
        pilot_overlap_receipt=receipt_path,
        pilot_overlap_evidence_record=evidence_path,
    )

    assert result["qualification_status"] == "qualified_with_exclusions"
    assert result["model_launch_status"] == (
        "eligible_for_preregistered_gene_disjoint_baselines"
    )
    mappings = json.loads((output / "modality_mappings.json").read_text(encoding="utf-8"))
    division = next(item for item in mappings["modalities"] if item["id"] == "division_timing")
    cnd1 = next(item for item in mappings["modalities"] if item["id"] == "cnd1_gfp_expression")
    assert division["s3_embryo_columns"][0]["embryo_id"] == "EMB-6"
    assert cnd1["s3_embryo_columns"][0]["embryo_id"] == "EMB-1"
    assert cnd1["s4"] == {
        "sheet": "4",
        "status": "excluded_no_access",
        "workbook_parsed": False,
        "header_decoded": False,
        "semantic_rows_decoded": False,
        "values_decoded": False,
    }
    assert mappings["perturbation_values_decoded"] is False
    assert mappings["s4_outcome_values_decoded"] is False
    assert mappings["s4_workbook_parsed"] is False
    assert mappings["s4_sheet_4_accessed"] is False


def test_overlap_receipt_rejects_claim_bearing_embryo(tmp_path: Path) -> None:
    source_dir, parent_manifest, parent_root, archive, raw_dir, contract = _bundle(tmp_path)
    evidence_path = tmp_path / "source-curator-record.txt"
    evidence_path.write_text("synthetic source-authorized evidence\n", encoding="utf-8")
    receipt_path = tmp_path / "bad-overlap.json"
    receipt_path.write_text(
        json.dumps(
            _overlap_receipt(
                parent_root,
                scaleup_embryo_id="EMB-2",
                evidence_record=evidence_path,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap reaches claim-bearing partitions"):
        adapter.freeze_stage_adapter(
            contract,
            parent_manifest,
            parent_root,
            source_dir,
            archive,
            raw_dir,
            tmp_path / "stage-adapter",
            academic_use_attested=True,
            pilot_overlap_receipt=receipt_path,
            pilot_overlap_evidence_record=evidence_path,
        )


def test_overlap_receipt_requires_frozen_pilot_and_source_authority(tmp_path: Path) -> None:
    _, _, parent_root, _, _, _ = _bundle(tmp_path)
    valid = _overlap_receipt(parent_root)

    wrong_pilot = json.loads(json.dumps(valid))
    wrong_pilot["pilot_normalized_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="different normalized pilot"):
        adapter.PilotOverlapReceipt.model_validate(wrong_pilot)

    missing_authority = json.loads(json.dumps(valid))
    missing_authority.pop("source_evidence_record_sha256")
    with pytest.raises(ValueError, match="source_evidence_record_sha256"):
        adapter.PilotOverlapReceipt.model_validate(missing_authority)

    outcome_informed = json.loads(json.dumps(valid))
    outcome_informed["outcome_values_used_for_identity_decision"] = True
    with pytest.raises(ValueError, match="may not use outcome values"):
        adapter.PilotOverlapReceipt.model_validate(outcome_informed)

    incomplete = json.loads(json.dumps(valid))
    incomplete["assertions"][
        "any_reused_embryos_are_exhaustively_listed_in_overlap_pairs"
    ] = False
    with pytest.raises(ValueError, match="complete overlap evidence"):
        adapter.PilotOverlapReceipt.model_validate(incomplete)


def test_disjoint_attestation_requires_zero_pairs_and_non_disjoint_requires_pairs(
    tmp_path: Path,
) -> None:
    _, _, parent_root, _, _, _ = _bundle(tmp_path)
    disjoint = _overlap_receipt(parent_root)
    disjoint["assertions"][
        "omix709_05_54_and_05_55_are_biological_embryo_disjoint"
    ] = True
    with pytest.raises(ValueError, match="may not declare overlap pairs"):
        adapter.PilotOverlapReceipt.model_validate(disjoint)

    non_disjoint = _overlap_receipt(parent_root)
    non_disjoint["pairs"] = []
    with pytest.raises(ValueError, match="require exhaustive overlap pairs"):
        adapter.PilotOverlapReceipt.model_validate(non_disjoint)

    disjoint["pairs"] = []
    assert adapter.PilotOverlapReceipt.model_validate(disjoint).pairs == []


def test_missing_attestation_and_checksum_tamper_fail_closed(tmp_path: Path) -> None:
    source_dir, parent_manifest, parent_root, archive, raw_dir, contract = _bundle(tmp_path)
    with pytest.raises(PermissionError, match="controlled, noncommercial academic use"):
        adapter.freeze_stage_adapter(
            contract,
            parent_manifest,
            parent_root,
            source_dir,
            archive,
            raw_dir,
            tmp_path / "not-created",
            academic_use_attested=False,
        )

    output = tmp_path / "stage-adapter"
    adapter.freeze_stage_adapter(
        contract,
        parent_manifest,
        parent_root,
        source_dir,
        archive,
        raw_dir,
        output,
        academic_use_attested=True,
    )
    target = output / "stage_membership.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum failed"):
        adapter.verify_stage_adapter(output)


def test_missing_and_tampered_raw_control_files_fail_identity_gates(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    archive, raw_dir = _write_raw_controls(missing_root)
    payload = adapter.load_contract(CONFIG).model_dump(mode="json")
    _bind_raw_source_contract(payload, archive, raw_dir)
    contract = adapter.StageAdapterContract.model_validate(payload)
    (raw_dir / "ctr_emb2_raw_data.txt").unlink()
    with pytest.raises(ValueError, match="extracted file count differs"):
        adapter._read_raw_control_lineage_source(archive, raw_dir, contract)

    tampered_root = tmp_path / "tampered"
    tampered_root.mkdir()
    archive, raw_dir = _write_raw_controls(tampered_root)
    payload = adapter.load_contract(CONFIG).model_dump(mode="json")
    _bind_raw_source_contract(payload, archive, raw_dir)
    contract = adapter.StageAdapterContract.model_validate(payload)
    member = raw_dir / "ctr_emb1_raw_data.txt"
    original = member.read_bytes()
    member.write_bytes(original.replace(b"\xff", b"\xf0", 1))
    assert member.stat().st_size == len(original)
    with pytest.raises(ValueError, match="inventory SHA-256 differs"):
        adapter._read_raw_control_lineage_source(archive, raw_dir, contract)


def test_tampered_raw_control_archive_fails_identity_gate(tmp_path: Path) -> None:
    archive, raw_dir = _write_raw_controls(tmp_path)
    payload = adapter.load_contract(CONFIG).model_dump(mode="json")
    _bind_raw_source_contract(payload, archive, raw_dir)
    contract = adapter.StageAdapterContract.model_validate(payload)
    original = archive.read_bytes()
    archive.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="archive SHA-256 identity differs"):
        adapter._read_raw_control_lineage_source(archive, raw_dir, contract)


def test_wrong_raw_control_header_fails_after_inventory_binding(tmp_path: Path) -> None:
    archive, raw_dir = _write_raw_controls(tmp_path)
    member = raw_dir / "ctr_emb1_raw_data.txt"
    member.write_bytes(member.read_bytes().replace(b"\ttime\t", b"\twhen\t", 1))
    payload = adapter.load_contract(CONFIG).model_dump(mode="json")
    _bind_raw_source_contract(payload, archive, raw_dir)
    contract = adapter.StageAdapterContract.model_validate(payload)
    with pytest.raises(ValueError, match="raw-control header identity differs"):
        adapter._read_raw_control_lineage_source(archive, raw_dir, contract)


def test_unknown_raw_control_cell_identity_fails_closed(tmp_path: Path) -> None:
    archive, raw_dir = _write_raw_controls(tmp_path)
    for member in raw_dir.iterdir():
        member.write_bytes(member.read_bytes().replace(b"\tCa\t", b"\tQX\t", 1))
    payload = adapter.load_contract(CONFIG).model_dump(mode="json")
    _bind_raw_source_contract(payload, archive, raw_dir)
    contract = adapter.StageAdapterContract.model_validate(payload)
    raw_controls = adapter._read_raw_control_lineage_source(archive, raw_dir, contract)
    with pytest.raises(ValueError, match="unknown semantic lineage identifier: QX"):
        adapter._derive_lineage(raw_controls, contract)


def test_incomplete_raw_control_daughter_catalog_fails_closed(tmp_path: Path) -> None:
    archive, raw_dir = _write_raw_controls(tmp_path)
    cells, event_time = _lineage_catalog()
    terminal = next(cell for cell in reversed(cells) if cell not in event_time)
    target = f"\t{terminal}\t".encode("ascii")
    for member in raw_dir.iterdir():
        rows = member.read_bytes().splitlines(keepends=True)
        member.write_bytes(b"".join(row for row in rows if target not in row))
    payload = adapter.load_contract(CONFIG).model_dump(mode="json")
    _bind_raw_source_contract(
        payload,
        archive,
        raw_dir,
        expected_union_cell_count=397,
    )
    contract = adapter.StageAdapterContract.model_validate(payload)
    raw_controls = adapter._read_raw_control_lineage_source(archive, raw_dir, contract)
    with pytest.raises(ValueError, match="does not have exactly two observed daughters"):
        adapter._derive_lineage(raw_controls, contract)


def test_invalid_utf8_outcome_poison_is_never_decoded(tmp_path: Path) -> None:
    archive, raw_dir = _write_raw_controls(tmp_path)
    payload = adapter.load_contract(CONFIG).model_dump(mode="json")
    _bind_raw_source_contract(payload, archive, raw_dir)
    contract = adapter.StageAdapterContract.model_validate(payload)

    raw_controls = adapter._read_raw_control_lineage_source(archive, raw_dir, contract)

    assert raw_controls.source_receipt["access"]["outcome_columns_decoded"] is False
    assert raw_controls.source_receipt["access"]["nonpermitted_value_columns_policy"] == (
        "never_decode"
    )
