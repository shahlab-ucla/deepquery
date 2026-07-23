"""Outcome-blind readiness and topology-coordinate preparation for OMIX709.

This module deliberately stops before any perturbation measurement or endpoint
value is decoded.  It verifies the frozen developmental source qualification adapter and whole-gene split,
materializes graph coordinates that depend only on lineage/stage identity, and
emits an explicit blocker while the pilot/scale-up embryo-overlap gate is open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import developmental_baselines as baselines
from . import developmental_omix_stage_adapter as stage
from .developmental_omix import (
    SHA256_RE,
    _canonical_bytes,
    _sha256,
    _write_bytes_atomic,
    _write_json_atomic,
)


SCHEMA_VERSION = "wormctx-omix709-developmental-readiness-contract-1.0"
READINESS_VERSION = "wormctx-omix709-developmental-readiness-1.0"
TOPOLOGY_COORDINATE_VERSION = "wormctx-omix709-topology-coordinate-bundle-1.0"
EVIDENCE_ASSESSMENT_VERSION = "wormctx-omix709-overlap-evidence-assessment-1.0"
RETRIEVAL_REQUEST_VERSION = "wormctx-omix709-identity-only-retrieval-request-1.0"
VERIFY_VERSION = "wormctx-omix709-developmental-readiness-verification-1.0"

OUTPUT_FILES = {
    "readiness.json",
    "source_evidence_assessment.json",
    "topology_coordinates.json",
    "retrieval_request.json",
}
PARTITIONS = (
    "train",
    "validation",
    "sealed_test",
    "development_only_exposed_pilot",
)
REQUIRED_ASSERTIONS = [
    "table_s3_uses_only_omix709_05_54_embryos",
    "omix709_05_54_and_05_55_are_biological_embryo_disjoint",
    "identifier_namespaces_are_complete_and_not_reused_across_archives",
    "any_reused_embryos_are_exhaustively_listed_in_overlap_pairs",
]


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"JSON input is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must be an object: {source}")
    return payload


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_hash(value: Any, label: str) -> str:
    parsed = str(value).lower()
    if not SHA256_RE.fullmatch(parsed):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return parsed


def _verify_file_identity(
    path: str | Path,
    specification: Mapping[str, Any],
    *,
    filename_key: str,
    bytes_key: str,
    sha256_key: str,
    label: str,
) -> Path:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    if source.name != specification[filename_key]:
        raise ValueError(f"{label} filename differs")
    if source.stat().st_size != int(specification[bytes_key]):
        raise ValueError(f"{label} byte identity differs")
    if _sha256(source) != _require_hash(specification[sha256_key], f"{label} hash"):
        raise ValueError(f"{label} SHA-256 identity differs")
    return source


def load_contract(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unknown developmental readiness contract")
    if any(
        payload.get(name) is not False
        for name in (
            "validated",
            "biological_claims_permitted",
            "outcome_values_permitted",
            "outcome_prevalence_inspection_permitted",
            "sealed_test_access_permitted",
        )
    ):
        raise ValueError("readiness qualification must remain outcome-blind and nonclaiming")

    adapter = payload["stage_adapter"]
    expected = adapter["expected_counts"]
    if (
        adapter["required_open_qualification_status"]
        != "stage_adapter_frozen_with_open_overlap_gate"
        or adapter["required_open_model_launch_status"]
        != "blocked_pending_pilot_embryo_overlap_receipt"
        or expected
        != {
            "input_26_frontier_cells": 26,
            "input_26_born_cells": 50,
            "input_26_completed_divisions": 24,
            "endpoint_200_frontier_cells": 200,
            "lineage_cells_through_200": 399,
            "lineage_edges_through_200": 398,
            "measurement_embryos": 2075,
            "modalities": 6,
            "raw_control_files": 105,
            "raw_control_semantic_cell_union": 794,
            "raw_control_supported_division_events": 361,
        }
    ):
        raise ValueError("frozen developmental source qualification state or counts differ")
    for name in ("contract_sha256", "required_open_qualification_sha256"):
        _require_hash(adapter[name], f"stage adapter {name}")

    split = payload["split"]
    if (
        split["key"] != "wormbase_gene_id"
        or split["whole_gene"] is not True
        or split["exposed_partition_permitted_in_modeling"] is not False
        or split["partition_counts"]
        != {
            "train": {"genes": 410, "embryos": 1229},
            "validation": {"genes": 137, "embryos": 407},
            "sealed_test": {"genes": 137, "embryos": 398},
            "development_only_exposed_pilot": {"genes": 14, "embryos": 41},
        }
    ):
        raise ValueError("whole-gene split policy differs")
    _require_hash(split["manifest_sha256"], "split-manifest hash")

    baseline = payload["baseline"]
    if (
        baseline["real_fit_permitted_while_overlap_open"] is not False
        or baseline["required_stage_contract_sha256"] != adapter["contract_sha256"]
    ):
        raise ValueError("baseline launch policy differs")
    _require_hash(baseline["contract_sha256"], "baseline contract hash")

    overlap = payload["overlap"]
    if (
        overlap["required_receipt_schema_version"] != stage.OVERLAP_RECEIPT_VERSION
        or overlap["preferred_closure"]
        != "source_curator_attestation_with_hash_bound_durable_evidence"
        or overlap["public_records_establish_biological_embryo_disjointness"] is not False
        or overlap["gene_quarantine_alone_closes_current_contract"] is not False
        or overlap["required_assertions"] != REQUIRED_ASSERTIONS
    ):
        raise ValueError("pilot/scale-up overlap policy differs")
    _require_hash(overlap["attestation_request_sha256"], "attestation request hash")
    raw = overlap["raw_archive_supporting_evidence"]
    if (
        raw["omix_file_id"] != "OMIX709-05-54"
        or raw["filename"] != "OMIX709-05-54.rar"
        or int(raw["expected_bytes"]) != 677141156
        or raw["expected_sha256"]
        != "5e268e447c8f45fdd83823c029cfef6d67d389ad39013b66f9cb7191eb9a2030"
        or int(raw["expected_embryos"]) != 2075
        or raw["expected_extracted_root"] != "Raw_data_RNAi"
        or raw["comparison_omix_file_id"] != "OMIX709-05-55"
        or raw["comparison_filename"] != "OMIX709-05-55.rar"
        or int(raw["comparison_expected_bytes"]) != 51428713
        or raw["comparison_expected_sha256"]
        != "429f8888f3549738e66e28c7f2af440233982b96950350fd97b5d2b6d68d49c8"
        or int(raw["comparison_expected_embryos"]) != 146
        or raw["comparison_expected_extracted_root"] != "Raw_data_RNAi_add"
        or raw["permitted_access"]
        != "archive_identity_member_names_member_sizes_and_member_sha256_only"
        or raw["outcome_columns_permitted"] is not False
        or raw["no_exact_member_or_hash_match_interpretation"]
        != "supporting_evidence_only_not_contract_closure"
    ):
        raise ValueError("raw-archive supporting-evidence policy differs")

    preparation = payload["preparation"]
    if (
        preparation["topology_coordinates_may_be_built_while_overlap_open"] is not True
        or preparation["topology_coordinates_may_read_outcomes"] is not False
    ):
        raise ValueError("outcome-blind topology preparation policy differs")
    _require_hash(
        preparation["expected_topology_array_sha256"],
        "expected topology-array hash",
    )
    return payload


def _validate_split(
    split_manifest: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        split_manifest.get("schema_version")
        != "wormctx-omix709-gene-split-manifest-1.0"
        or split_manifest.get("key") != "wormbase_gene_id"
        or split_manifest.get("whole_gene") is not True
    ):
        raise ValueError("split manifest identity differs")
    by_partition: dict[str, list[Mapping[str, Any]]] = {
        name: [] for name in PARTITIONS
    }
    for row in split_manifest.get("claim_bearing_genes", []):
        partition = str(row.get("partition"))
        if partition not in by_partition or partition == "development_only_exposed_pilot":
            raise ValueError("claim-bearing split contains an invalid partition")
        by_partition[partition].append(row)
    for row in split_manifest.get("development_only_exposed_pilot_genes", []):
        if row.get("partition") != "development_only_exposed_pilot":
            raise ValueError("pilot-exposed gene has an invalid partition")
        by_partition["development_only_exposed_pilot"].append(row)

    gene_ids: set[str] = set()
    embryo_ids: set[str] = set()
    observed: dict[str, dict[str, int]] = {}
    for partition in PARTITIONS:
        rows = by_partition[partition]
        partition_embryos = 0
        for row in rows:
            gene_id = str(row.get("gene_id", ""))
            if not gene_id or gene_id in gene_ids:
                raise ValueError("split contains a missing or duplicate whole-gene key")
            gene_ids.add(gene_id)
            embryos = [str(item) for item in row.get("embryo_ids", [])]
            if not embryos or len(embryos) != len(set(embryos)):
                raise ValueError("split gene has missing or duplicate embryo identities")
            collision = set(embryos) & embryo_ids
            if collision:
                raise ValueError("embryo identities overlap split partitions")
            embryo_ids.update(embryos)
            partition_embryos += len(embryos)
        observed[partition] = {"genes": len(rows), "embryos": partition_embryos}
    if observed != contract["partition_counts"]:
        raise ValueError(f"split counts differ: {observed}")
    return {
        "partition_counts": observed,
        "total_genes": len(gene_ids),
        "total_embryos": len(embryo_ids),
        "gene_partitions_disjoint": True,
        "embryo_partitions_disjoint": True,
        "pilot_exposed_genes_permitted_in_modeling": False,
    }


def topology_coordinates_from_stage(
    lineage_manifest: Mapping[str, Any],
    stage_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct the topology identity arrays without importing Torch."""

    if lineage_manifest.get("schema_version") != stage.LINEAGE_MANIFEST_VERSION:
        raise ValueError("unexpected semantic-lineage manifest schema")
    if stage_manifest.get("schema_version") != stage.STAGE_MANIFEST_VERSION:
        raise ValueError("unexpected semantic-stage manifest schema")
    if (
        stage_manifest.get("raw_control_outcome_columns_decoded") is not False
        or stage_manifest.get("perturbation_measurement_values_read") is not False
        or stage_manifest.get("s4_outcome_values_read") is not False
    ):
        raise ValueError("topology construction requires outcome-blind stage artifacts")

    root = str(lineage_manifest["virtual_root"])
    cells = tuple(sorted(str(item) for item in lineage_manifest["cells_through_200"]))
    if not cells or len(cells) != len(set(cells)) or root not in cells:
        raise ValueError("lineage cell vocabulary is malformed")
    position = {cell: index for index, cell in enumerate(cells)}
    parent_by_child: dict[str, str] = {}
    for edge in lineage_manifest["edges_through_200"]:
        parent, child = str(edge["parent"]), str(edge["child"])
        if parent not in position or child not in position:
            raise ValueError("lineage edge references an unknown cell")
        if child == root or child in parent_by_child:
            raise ValueError("each nonroot cell must have exactly one parent")
        parent_by_child[child] = parent
    if set(parent_by_child) != set(cells) - {root}:
        raise ValueError("lineage does not connect every nonroot cell")

    depth_cache = {root: 0}

    def depth(cell: str, trail: frozenset[str] = frozenset()) -> int:
        if cell in depth_cache:
            return depth_cache[cell]
        if cell in trail:
            raise ValueError("lineage contains a cycle")
        value = depth(parent_by_child[cell], trail | {cell}) + 1
        depth_cache[cell] = value
        return value

    depths = tuple(depth(cell) for cell in cells)
    input_membership = stage_manifest.get("input_26", {})
    endpoint_membership = stage_manifest.get("endpoint_200", {})
    input_frontier = set(map(str, input_membership.get("frontier", [])))
    endpoint_frontier = set(map(str, endpoint_membership.get("frontier", [])))
    born_by_input = set(map(str, input_membership.get("born_cells", [])))
    born_by_endpoint = set(map(str, endpoint_membership.get("born_cells", [])))
    if len(input_frontier) != 26 or len(endpoint_frontier) != 200:
        raise ValueError("stage adapter must freeze exact 26- and 200-cell frontiers")
    if not input_frontier <= born_by_input or not endpoint_frontier <= born_by_endpoint:
        raise ValueError("stage frontier is not a subset of born cells")
    if born_by_endpoint | {root} != set(cells) or not born_by_input <= born_by_endpoint:
        raise ValueError("stage memberships do not match the lineage vocabulary")

    parent_indices = tuple(
        -1 if cell == root else position[parent_by_child[cell]] for cell in cells
    )
    stage_ids = tuple(0 if cell == root else 1 if cell in born_by_input else 2 for cell in cells)
    for child, parent in enumerate(parent_indices):
        if parent == -1:
            if cells[child] != root or depths[child] != 0:
                raise ValueError("topology root identity differs")
        elif parent < 0 or parent >= len(cells) or depths[parent] + 1 != depths[child]:
            raise ValueError("parent/depth topology invariant failed")
    if parent_indices.count(-1) != 1:
        raise ValueError("topology must contain exactly one root")

    array_payload = {
        "cell_ids": cells,
        "parent_index": parent_indices,
        "depth_id": depths,
        "stage_id": stage_ids,
    }
    arrays_sha256 = hashlib.sha256(
        json.dumps(array_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": TOPOLOGY_COORDINATE_VERSION,
        "source": "outcome_blind_semantic_stage_adapter",
        "cell_ids": list(cells),
        "semantic_id": list(range(len(cells))),
        "parent_index": list(parent_indices),
        "depth_id": list(depths),
        "stage_id": list(stage_ids),
        "born_by_input_mask": [cell in born_by_input for cell in cells],
        "endpoint_frontier_mask": [cell in endpoint_frontier for cell in cells],
        "receipt": {
            "cell_count_through_200": len(cells),
            "edge_count_through_200": len(parent_by_child),
            "input_frontier_count": len(input_frontier),
            "input_born_count": len(born_by_input),
            "endpoint_frontier_count": len(endpoint_frontier),
            "maximum_depth": max(depths),
            "arrays_sha256": arrays_sha256,
            "outcome_values_read": False,
        },
    }


def _write_bundle(output_root: Path, payloads: Mapping[str, Any]) -> None:
    if output_root.exists():
        raise FileExistsError(f"readiness output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    # Keep the staging name short enough for Windows' legacy path-length limit;
    # the atomic JSON writer adds its own UUID suffix to every filename.
    temporary = output_root.parent / f".dr-{uuid.uuid4().hex[:8]}"
    temporary.mkdir()
    try:
        for name in sorted(payloads):
            _write_json_atomic(temporary / name, payloads[name])
        lines = [f"{_sha256(temporary / name)}  {name}" for name in sorted(payloads)]
        _write_bytes_atomic(
            temporary / "SHA256SUMS.txt", ("\n".join(lines) + "\n").encode("utf-8")
        )
        _write_bytes_atomic(temporary / "SUCCESS", b"SUCCESS\n")
        os.replace(temporary, output_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def prepare_readiness(
    contract_path: str | Path,
    stage_contract_path: str | Path,
    stage_root: str | Path,
    split_manifest_path: str | Path,
    baseline_contract_path: str | Path,
    overlap_request_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    contract_source = Path(contract_path).resolve()
    contract = load_contract(contract_source)
    adapter_contract = _verify_file_identity(
        stage_contract_path,
        contract["stage_adapter"],
        filename_key="contract_filename",
        bytes_key="contract_bytes",
        sha256_key="contract_sha256",
        label="stage-adapter contract",
    )
    split_source = _verify_file_identity(
        split_manifest_path,
        contract["split"],
        filename_key="manifest_filename",
        bytes_key="manifest_bytes",
        sha256_key="manifest_sha256",
        label="split manifest",
    )
    baseline_source = _verify_file_identity(
        baseline_contract_path,
        contract["baseline"],
        filename_key="contract_filename",
        bytes_key="contract_bytes",
        sha256_key="contract_sha256",
        label="baseline contract",
    )
    request_source = _verify_file_identity(
        overlap_request_path,
        contract["overlap"],
        filename_key="attestation_request_filename",
        bytes_key="attestation_request_bytes",
        sha256_key="attestation_request_sha256",
        label="overlap attestation request",
    )

    baseline_contract = baselines.load_contract(baseline_source)
    if baseline_contract.stage_adapter.contract_sha256 != contract["stage_adapter"][
        "contract_sha256"
    ]:
        raise ValueError("gene-disjoint developmental prediction baseline is not bound to the frozen v1.1 adapter contract")

    root = Path(stage_root).resolve()
    verification = stage.verify_stage_adapter(root, adapter_contract)
    qualification_path = root / "qualification.json"
    if _sha256(qualification_path) != contract["stage_adapter"][
        "required_open_qualification_sha256"
    ]:
        raise ValueError("open stage-adapter qualification hash differs")
    qualification = _read_json(qualification_path)
    if (
        qualification.get("qualification_status")
        != contract["stage_adapter"]["required_open_qualification_status"]
        or qualification.get("model_launch_status")
        != contract["stage_adapter"]["required_open_model_launch_status"]
        or qualification.get("outcome_prevalence_inspected") is not False
        or qualification.get("gates", {}).get("pilot_embryo_overlap_closed") is not False
        or qualification.get("gates", {}).get("sealed_test_opened") is not False
    ):
        raise ValueError("stage-adapter open-gate state differs")
    for key, expected in contract["stage_adapter"]["expected_counts"].items():
        if key in qualification.get("counts", {}) and qualification["counts"][key] != expected:
            raise ValueError(f"stage-adapter count differs: {key}")

    split_manifest = _read_json(split_source)
    split_receipt = _validate_split(split_manifest, contract["split"])
    lineage = _read_json(root / "lineage_manifest.json")
    stages = _read_json(root / "stage_membership.json")
    topology = topology_coordinates_from_stage(lineage, stages)
    topology_receipt = topology["receipt"]
    expected_counts = contract["stage_adapter"]["expected_counts"]
    if (
        topology_receipt["cell_count_through_200"]
        != expected_counts["lineage_cells_through_200"]
        or topology_receipt["edge_count_through_200"]
        != expected_counts["lineage_edges_through_200"]
        or topology_receipt["input_born_count"]
        != expected_counts["input_26_born_cells"]
        or topology_receipt["arrays_sha256"]
        != contract["preparation"]["expected_topology_array_sha256"]
    ):
        raise ValueError("real outcome-blind topology coordinate identity differs")

    request = _read_json(request_source)
    requested = request.get("requested_assertions", {})
    if (
        request.get("status") != "awaiting_source_curator_attestation"
        or request.get("outcome_values_used_for_identity_decision") is not False
        or list(requested) != REQUIRED_ASSERTIONS
        or any(requested[name] is not None for name in REQUIRED_ASSERTIONS)
        or request.get("required_response", {}).get("overlap_pairs") != []
    ):
        raise ValueError("overlap request is not the unanswered outcome-blind template")

    evidence = {
        "schema_version": EVIDENCE_ASSESSMENT_VERSION,
        "decision": "insufficient_for_current_overlap_contract_closure",
        "established": [
            "OMIX709-05-54 and OMIX709-05-55 are distinct repository objects",
            "Raw_data_RNAi and Raw_data_RNAi_add are distinct source namespaces",
            "all 14 pilot targets present in scale-up are quarantined by whole gene",
        ],
        "not_established": [
            "biological embryo disjointness across the two archives",
            "absence of renamed or reprocessed biological duplicates",
            "complete one-to-one cross-archive biological identity mapping",
        ],
        "public_record_urls": [
            "https://ngdc.cncb.ac.cn/omix/release/OMIX709",
            "https://dulab.genetics.ac.cn/single-cell-phenomics/download.html",
            "https://doi.org/10.1016/j.cels.2022.07.001",
        ],
        "raw_archive_inventory_role": "supporting_evidence_only_not_contract_closure",
        "source_curator_response_required": True,
        "outcome_values_read": False,
        "sealed_test_opened": False,
    }
    retrieval = {
        "schema_version": RETRIEVAL_REQUEST_VERSION,
        "classification": "identity_only_no_measurement_decoding",
        "source": contract["overlap"]["raw_archive_supporting_evidence"],
        "required_products": [
            "archive_sha256",
            "complete_sorted_member_name_size_sha256_inventory",
            "cross_archive_exact_member_name_matches",
            "cross_archive_exact_member_sha256_matches",
            "identity_only_qualification_receipt",
        ],
        "acceptance_boundary": (
            "No exact match is supporting evidence only; it does not close the frozen "
            "biological-embryo overlap contract without source-authorized evidence."
        ),
        "controlled_noncommercial_academic_use_attestation_required": True,
        "outcome_columns_may_be_decoded": False,
    }
    readiness = {
        "schema_version": READINESS_VERSION,
        "analysis_id": contract["analysis_id"],
        "technical_status": "success",
        "execution_status": "blocked_pending_pilot_embryo_overlap_receipt",
        "next_modeling_step": "gene_disjoint_prediction_gene_disjoint_baseline_preparation_and_fit",
        "next_modeling_step_permitted": False,
        "safe_preparation_complete": True,
        "gates": {
            "stage_adapter_v11_identity_verified": True,
            "stage_adapter_outcome_blind": True,
            "stage_adapter_overlap_closed": False,
            "whole_gene_split_verified": True,
            "pilot_exposed_genes_quarantined": True,
            "gene_disjoint_prediction_baseline_rebound_to_stage_v11": True,
            "topology_coordinates_materialized": True,
            "outcome_values_read": False,
            "outcome_prevalence_inspected": False,
            "sealed_test_opened": False,
            "real_model_fit_permitted": False,
        },
        "counts": {
            **split_receipt["partition_counts"],
            "lineage_cells_through_200": topology_receipt["cell_count_through_200"],
            "lineage_edges_through_200": topology_receipt["edge_count_through_200"],
            "input_26_frontier_cells": topology_receipt["input_frontier_count"],
            "endpoint_200_frontier_cells": topology_receipt["endpoint_frontier_count"],
            "modalities": expected_counts["modalities"],
        },
        "bindings": {
            "readiness_contract_sha256": _sha256(contract_source),
            "stage_contract_sha256": _sha256(adapter_contract),
            "stage_qualification_sha256": _sha256(qualification_path),
            "stage_lineage_manifest_sha256": _sha256(root / "lineage_manifest.json"),
            "stage_membership_sha256": _sha256(root / "stage_membership.json"),
            "split_manifest_sha256": _sha256(split_source),
            "baseline_contract_sha256": _sha256(baseline_source),
            "overlap_attestation_request_sha256": _sha256(request_source),
            "topology_arrays_sha256": topology_receipt["arrays_sha256"],
        },
        "stage_verification": verification,
        "split_verification": split_receipt,
        "remaining_external_dependency": (
            "A source-authorized, hash-frozen response establishing complete biological "
            "embryo identity/disjointness, or an explicit preregistered contract amendment."
        ),
        "biological_claims_permitted": False,
    }
    _write_bundle(
        Path(output_root).resolve(),
        {
            "readiness.json": readiness,
            "source_evidence_assessment.json": evidence,
            "topology_coordinates.json": topology,
            "retrieval_request.json": retrieval,
        },
    )
    return readiness


def verify_readiness(
    output_root: str | Path, contract_path: str | Path | None = None
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"readiness output is missing: {root}")
    if {item.name for item in root.iterdir()} != OUTPUT_FILES | {"SHA256SUMS.txt", "SUCCESS"}:
        raise ValueError("readiness output inventory differs")
    if (root / "SUCCESS").read_bytes() != b"SUCCESS\n":
        raise ValueError("readiness SUCCESS marker differs")
    checked: set[str] = set()
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None or match.group(2) not in OUTPUT_FILES or match.group(2) in checked:
            raise ValueError("readiness checksum manifest is malformed")
        if _sha256(root / match.group(2)) != match.group(1):
            raise ValueError(f"readiness checksum failed: {match.group(2)}")
        checked.add(match.group(2))
    if checked != OUTPUT_FILES:
        raise ValueError("readiness checksum coverage differs")
    readiness = _read_json(root / "readiness.json")
    topology = _read_json(root / "topology_coordinates.json")
    evidence = _read_json(root / "source_evidence_assessment.json")
    if (
        readiness.get("schema_version") != READINESS_VERSION
        or readiness.get("execution_status")
        != "blocked_pending_pilot_embryo_overlap_receipt"
        or readiness.get("next_modeling_step_permitted") is not False
        or readiness.get("gates", {}).get("outcome_values_read") is not False
        or readiness.get("gates", {}).get("sealed_test_opened") is not False
        or evidence.get("decision") != "insufficient_for_current_overlap_contract_closure"
        or evidence.get("outcome_values_read") is not False
        or topology.get("receipt", {}).get("outcome_values_read") is not False
        or topology.get("receipt", {}).get("arrays_sha256")
        != readiness.get("bindings", {}).get("topology_arrays_sha256")
    ):
        raise ValueError("readiness outcome boundary or blocker state differs")
    if contract_path is not None:
        contract_source = Path(contract_path).resolve()
        load_contract(contract_source)
        if readiness["bindings"]["readiness_contract_sha256"] != _sha256(contract_source):
            raise ValueError("readiness contract binding differs")
    return {
        "schema_version": VERIFY_VERSION,
        "verified": True,
        "checked_files": len(checked),
        "execution_status": readiness["execution_status"],
        "topology_arrays_sha256": topology["receipt"]["arrays_sha256"],
        "outcome_values_read": False,
        "sealed_test_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wormctx.poc.developmental_readiness",
        description="Verify and prepare the outcome-blind OMIX709 developmental lane",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--contract", required=True)
    prepare.add_argument("--stage-contract", required=True)
    prepare.add_argument("--stage-root", required=True)
    prepare.add_argument("--split-manifest", required=True)
    prepare.add_argument("--baseline-contract", required=True)
    prepare.add_argument("--overlap-request", required=True)
    prepare.add_argument("--output-root", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--output-root", required=True)
    verify.add_argument("--contract")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        payload = prepare_readiness(
            args.contract,
            args.stage_contract,
            args.stage_root,
            args.split_manifest,
            args.baseline_contract,
            args.overlap_request,
            args.output_root,
        )
    else:
        payload = verify_readiness(args.output_root, args.contract)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
