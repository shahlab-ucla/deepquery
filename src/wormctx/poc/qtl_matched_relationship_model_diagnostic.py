"""matched relationship-model diagnostic paired matched-model diagnostic for the completed WS283 parametric null.

This lane never simulates a phenotype.  It reuses four predeclared sets of 100
parametric-null calibration phenotype files byte for byte, maps each with one whole-panel external GRM
using GCTA ``--mlma``, and pairs the resulting diagnostics with the existing
LOCO checkpoint for the same cell and replicate.
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
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import qtl_parametric_null as parametric_null
from .qtl_grm_sensitivity import _accepted_options, _qualify_log_text


SCHEMA_VERSION = "wormctx-abamectin-ws283-matched_relationship_model-matched-model-1.0"
SOURCE_VERSION = "wormctx-abamectin-ws283-matched_relationship_model-source-verification-1.0"
MAP_VERSION = "wormctx-abamectin-ws283-matched_relationship_model-matched-map-1.0"
SUMMARY_VERSION = "wormctx-abamectin-ws283-matched_relationship_model-matched-summary-1.0"
RUN_VERSION = "wormctx-abamectin-ws283-matched_relationship_model-run-1.0"

EXPECTED_CELL_IDS = (
    "full_pc0_mean_EXT",
    "full_pc10_mean_norm_EXT",
    "full_pc10_norm_n",
    "ldpruned_pc10_mean_TOF",
)
REPLICATES = tuple(range(1, 101))
SENTINELS = (1, 25, 50, 75, 100)
REQUIRED_DIAGNOSTICS = (
    "minimum_p",
    "per_chromosome_minimum_p",
    "bonferroni_marker_count",
    "lambda_gc",
    "lambda_gc_excluding_trait_intervals",
    "lambda_gc_excluding_all_intervals",
    "qq_quantiles",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKSUM_LINE_RE = re.compile(r"^([0-9a-f]{64})  \./([^\r\n]+)$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceEs0(_StrictModel):
    run_id: Literal["abamectin-ws283-parametric-null-20260721T161005Z-c01179f4a52e"]
    success_sha256: str
    frozen_contract_sha256: str
    run_manifest_sha256: str
    sha256sums_sha256: str
    parent_verification_sha256: str
    aggregation_sha256: str
    summary_sha256: str
    required_total_maps: Literal[1600]
    required_selected_phenotypes: Literal[400]
    required_selected_simulation_receipts: Literal[400]
    required_selected_loco_checkpoints: Literal[400]
    full_checksum_closure_required: Literal[True]

    @field_validator(
        "success_sha256",
        "frozen_contract_sha256",
        "run_manifest_sha256",
        "sha256sums_sha256",
        "parent_verification_sha256",
        "aggregation_sha256",
        "summary_sha256",
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("source identity must be a lowercase SHA-256")
        return value


class Cell(_StrictModel):
    id: str
    relationship_kind: Literal["full", "ldpruned"]
    endpoint: Literal["pc0", "pc10"]
    trait: Literal["mean.EXT", "mean.TOF", "mean.norm.EXT", "norm.n"]
    trait_slug: Literal["mean_EXT", "mean_TOF", "mean_norm_EXT", "norm_n"]
    global_grm_relative_prefix: Literal["grm/global/full", "grm/global/ldpruned"]
    qcovar_parent_role: Literal["calibration"] | None
    qcovar_relative_path: Literal["genotype/qcovars/pc10.qcovar"] | None

    @model_validator(mode="after")
    def coherent(self) -> "Cell":
        expected_id = f"{self.relationship_kind}_{self.endpoint}_{self.trait_slug}"
        if self.id != expected_id:
            raise ValueError("cell id is not the canonical kind/endpoint/trait id")
        expected_slug = self.trait.replace(".", "_")
        if self.trait_slug != expected_slug:
            raise ValueError("trait and trait slug disagree")
        expected_grm = f"grm/global/{self.relationship_kind}"
        if self.global_grm_relative_prefix != expected_grm:
            raise ValueError("cell does not use its matching global generating GRM")
        if self.endpoint == "pc10":
            if self.qcovar_parent_role != "calibration" or self.qcovar_relative_path is None:
                raise ValueError("PC10 cell must use the frozen calibration qcovar")
        elif self.qcovar_parent_role is not None or self.qcovar_relative_path is not None:
            raise ValueError("PC0 cell cannot use a qcovar")
        return self


class Execution(_StrictModel):
    source_replicates_per_cell: Literal[100]
    new_map_count: Literal[400]
    association_method: Literal["GCTA_MLMA_with_single_whole_panel_external_GRM"]
    comparison_method: Literal[
        "paired_by_exact_PARAMETRIC_NULL_cell_and_replicate_against_existing_LOCO_checkpoint"
    ]
    genotype_parent_role: Literal["baseline"]
    genotype_relative_prefix: Literal["genotype/abamectin_209_qc"]
    marker_count: Literal[373279]
    sample_count: Literal[209]
    maf: Literal[0.05]
    autosome_num: Literal[6]
    thread_count: Literal[16]
    smoke_replicates: list[int]
    sentinel_replicates: list[int]
    phenotypes_are_reused_byte_for_byte: Literal[True]
    phenotype_regeneration_permitted: Literal[False]
    pooled_cells_permitted: Literal[False]

    @model_validator(mode="after")
    def exact_sets(self) -> "Execution":
        if self.smoke_replicates != [1, 2]:
            raise ValueError("smoke replicates must be exactly 1 and 2")
        if self.sentinel_replicates != list(SENTINELS):
            raise ValueError("sentinel replicates differ from the frozen parametric-null calibration set")
        return self


class Tools(_StrictModel):
    gcta_absolute_path: Literal["/opt/deepquery/bin/gcta64"]
    gcta_version: Literal["1.94.1_2022-11-15"]
    gcta_sha256: str

    @field_validator("gcta_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("GCTA identity must be a lowercase SHA-256")
        return value


class Diagnostics(_StrictModel):
    alpha: Literal[0.05]
    marker_bonferroni_p: float = Field(gt=0.0, lt=1.0)
    qq_quantiles: list[float]
    preserve: list[str]
    paired_discordance_required: Literal[True]

    @model_validator(mode="after")
    def frozen_diagnostics(self) -> "Diagnostics":
        if self.qq_quantiles != [0.5, 0.9, 0.95, 0.99, 0.999]:
            raise ValueError("QQ quantiles differ from parametric-null calibration")
        if set(self.preserve) != {
            "nominal_bonferroni_fwer",
            "minimum_p_distribution",
            "per_chromosome_minimum_p_distribution",
            "qq_quantile_distribution",
            "lambda_gc",
            "lambda_gc_excluding_trait_intervals",
            "lambda_gc_excluding_all_intervals",
        }:
            raise ValueError("required matched relationship-model diagnostic diagnostics are incomplete")
        return self


class Claims(_StrictModel):
    permitted: list[str]
    prohibited: list[str]


class Manifest(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    analysis_id: Literal["abamectin_ws283_matched_relationship_model_matched_whole_grm_diagnostic_v1"]
    classification: Literal["preliminary_model_match_diagnostic_not_threshold_calibration"]
    status: Literal["frozen_executable"]
    validated: Literal[False]
    biological_claims_permitted: Literal[False]
    source_parametric_null: SourceEs0
    cells: list[Cell]
    execution: Execution
    tools: Tools
    diagnostics: Diagnostics
    claims: Claims

    @model_validator(mode="after")
    def exact_cells(self) -> "Manifest":
        if tuple(cell.id for cell in self.cells) != EXPECTED_CELL_IDS:
            raise ValueError("matched relationship-model diagnostic cells or their order differ from the frozen four-cell lane")
        if len({cell.id for cell in self.cells}) != len(self.cells):
            raise ValueError("matched relationship-model diagnostic cells must be unique")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, description: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    stage = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        stage.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(stage, path)
    except BaseException:
        if stage.exists():
            stage.unlink()
        raise


def load_manifest(path: str | Path) -> Manifest:
    source = Path(path).resolve()
    return Manifest.model_validate_json(source.read_text(encoding="utf-8"))


def _safe_relative(value: str) -> str:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe checksum path: {value!r}")
    return pure.as_posix()


def parse_checksum_closure(path: str | Path) -> dict[str, str]:
    source = Path(path).resolve()
    closure: dict[str, str] = {}
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        match = _CHECKSUM_LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed checksum line {number}")
        relative = _safe_relative(match.group(2))
        if relative in closure:
            raise ValueError(f"duplicate checksum path: {relative}")
        closure[relative] = match.group(1)
    if not closure:
        raise ValueError("checksum closure is empty")
    return closure


def _verify_file(path: Path, expected: str, description: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{description} is absent or not a regular file: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{description} SHA-256 mismatch: {path}")


def _cell(manifest: Manifest, cell_id: str) -> Cell:
    matches = [item for item in manifest.cells if item.id == cell_id]
    if len(matches) != 1:
        raise ValueError(f"cell is not in the frozen matched relationship-model diagnostic lane: {cell_id}")
    return matches[0]


def _selected_paths(cell: Cell, replicate: int) -> tuple[str, str, str]:
    tag = f"rep-{replicate:03d}"
    return (
        f"null/phenotypes/{cell.id}/{tag}.phen",
        f"receipts/simulations/{cell.id}/{tag}.json",
        f"receipts/checkpoints/{cell.id}/{tag}.json",
    )


def _verify_selected_tuple(
    manifest: Manifest,
    root: Path,
    closure: dict[str, str],
    cell: Cell,
    replicate: int,
) -> dict[str, Any]:
    phenotype_rel, simulation_rel, checkpoint_rel = _selected_paths(cell, replicate)
    for relative in (phenotype_rel, simulation_rel, checkpoint_rel):
        if relative not in closure:
            raise ValueError(f"parametric-null calibration checksum closure lacks selected input: {relative}")
        _verify_file(root / relative, closure[relative], "selected parametric-null calibration input")

    phenotype = (root / phenotype_rel).resolve()
    simulation = _load_object(root / simulation_rel, "parametric-null calibration simulation receipt")
    checkpoint = _load_object(root / checkpoint_rel, "parametric-null calibration LOCO checkpoint")
    expected_cell = {
        "id": cell.id,
        "relationship_kind": cell.relationship_kind,
        "endpoint": cell.endpoint,
        "trait": cell.trait,
        "trait_slug": cell.trait_slug,
    }
    if simulation.get("schema_version") != parametric_null.PHENOTYPE_VERSION:
        raise ValueError("selected simulation receipt has the wrong schema")
    if simulation.get("cell") != expected_cell or simulation.get("replicate") != replicate:
        raise ValueError("selected simulation receipt has the wrong cell/replicate")
    simulated_pheno = simulation.get("phenotype")
    if not isinstance(simulated_pheno, dict):
        raise ValueError("selected simulation receipt lacks phenotype identity")
    if (
        simulated_pheno.get("sha256") != closure[phenotype_rel]
        or simulated_pheno.get("bytes") != phenotype.stat().st_size
        or simulation.get("byte_replay_verified") is not True
        or simulation.get("byte_replay_sha256") != closure[phenotype_rel]
    ):
        raise ValueError("selected phenotype is not the byte-replayed parametric-null calibration realization")
    if checkpoint.get("schema_version") != parametric_null.MAP_VERSION:
        raise ValueError("selected LOCO checkpoint has the wrong schema")
    if checkpoint.get("cell") != expected_cell or checkpoint.get("replicate") != replicate:
        raise ValueError("selected LOCO checkpoint has the wrong cell/replicate")
    if checkpoint.get("qualified") is not True or checkpoint.get("observed") is not False:
        raise ValueError("selected LOCO checkpoint is not a qualified null map")
    checkpoint_pheno = checkpoint.get("phenotype")
    if not isinstance(checkpoint_pheno, dict) or checkpoint_pheno.get("sha256") != closure[phenotype_rel]:
        raise ValueError("LOCO checkpoint and simulation do not bind the same phenotype")
    diagnostics = checkpoint.get("diagnostics")
    if not isinstance(diagnostics, dict) or any(key not in diagnostics for key in REQUIRED_DIAGNOSTICS):
        raise ValueError("selected LOCO checkpoint lacks a required diagnostic")
    return {
        "phenotype": phenotype,
        "phenotype_relative_path": phenotype_rel,
        "phenotype_sha256": closure[phenotype_rel],
        "simulation_relative_path": simulation_rel,
        "simulation_sha256": closure[simulation_rel],
        "loco_checkpoint_relative_path": checkpoint_rel,
        "loco_checkpoint_sha256": closure[checkpoint_rel],
        "loco_checkpoint": checkpoint,
    }


def verify_parametric_null_source(
    manifest_path: str | Path,
    source_root: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_source)
    root = Path(source_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"parametric-null calibration source root is absent or unsafe: {root}")

    identities = {
        "SUCCESS": manifest.source_parametric_null.success_sha256,
        "receipts/frozen_parametric_null_contract.json": manifest.source_parametric_null.frozen_contract_sha256,
        "receipts/run_manifest.json": manifest.source_parametric_null.run_manifest_sha256,
        "receipts/SHA256SUMS.txt": manifest.source_parametric_null.sha256sums_sha256,
        "receipts/parent_verification.json": manifest.source_parametric_null.parent_verification_sha256,
        "summary/parametric_null_aggregation.json": manifest.source_parametric_null.aggregation_sha256,
        "summary/ws283_parametric_null/ws283_parametric_null_summary.json": manifest.source_parametric_null.summary_sha256,
    }
    for relative, expected in identities.items():
        _verify_file(root / relative, expected, f"frozen parametric-null calibration {relative}")
    if (root / "FAILURE").exists():
        raise ValueError("terminal parametric-null calibration source still has a FAILURE marker")

    closure = parse_checksum_closure(root / "receipts/SHA256SUMS.txt")
    checked = 0
    for relative, expected in closure.items():
        _verify_file(root / relative, expected, "parametric-null calibration checksum closure member")
        checked += 1

    source_manifest_path = root / "receipts/frozen_parametric_null_contract.json"
    parametric_null.load_manifest(source_manifest_path)
    run_manifest = _load_object(root / "receipts/run_manifest.json", "parametric-null calibration run manifest")
    if (
        run_manifest.get("completed_null_maps") != manifest.source_parametric_null.required_total_maps
        or run_manifest.get("null_map_checkpoints") != manifest.source_parametric_null.required_total_maps
        or run_manifest.get("all_cells_complete") is not True
        or run_manifest.get("pooled_thresholds_used") is not False
        or run_manifest.get("manifest_sha256") != manifest.source_parametric_null.frozen_contract_sha256
    ):
        raise ValueError("parametric-null calibration run manifest is not the required complete non-pooled release")

    path_receipts = {}
    for role in ("baseline", "calibration", "sensitivity"):
        relative = f"receipts/{role}_parent_run_path.txt"
        if relative not in closure:
            raise ValueError(f"parametric-null calibration closure lacks {role} parent path receipt")
        _verify_file(root / relative, closure[relative], f"parametric-null calibration {role} path receipt")
        value = (root / relative).read_text(encoding="utf-8").strip()
        parent = Path(value)
        if not parent.is_absolute() or not parent.is_dir():
            raise ValueError(f"parametric-null calibration {role} parent path is unavailable")
        path_receipts[role] = parent.resolve()

    # Reverify the three immutable parents through the original parametric-null calibration contract;
    # this prevents the new map calls from trusting path receipts alone.
    parent_verification = parametric_null.verify_parents(
        source_manifest_path,
        path_receipts["baseline"],
        path_receipts["calibration"],
        path_receipts["sensitivity"],
    )
    if parent_verification.get("all_three_immutable_parents_verified") is not True:
        raise ValueError("original parametric-null calibration parents did not requalify")

    selected_digest = hashlib.sha256()
    selected_count = 0
    for cell in manifest.cells:
        for replicate in REPLICATES:
            selected = _verify_selected_tuple(manifest, root, closure, cell, replicate)
            for key in (
                "phenotype_sha256",
                "simulation_sha256",
                "loco_checkpoint_sha256",
            ):
                selected_digest.update(bytes.fromhex(selected[key]))
            selected_count += 1
    if selected_count != manifest.execution.new_map_count:
        raise ValueError("selected parametric-null calibration tuple count differs from the frozen 400-map lane")

    for kind, expected_markers in (("full", 373279), ("ldpruned", 1370)):
        for suffix in ("grm.bin", "grm.N.bin", "grm.id", "log"):
            relative = f"grm/global/{kind}.{suffix}"
            if relative not in closure:
                raise ValueError(f"parametric-null calibration closure lacks global {kind} GRM member {suffix}")
            _verify_file(root / relative, closure[relative], f"global {kind} GRM member")
        parametric_null.verify_grm(
            source_manifest_path,
            path_receipts["baseline"] / "genotype/abamectin_209_qc.fam",
            root / f"grm/global/{kind}",
            kind,
            expected_markers,
        )

    receipt = {
        "schema_version": SOURCE_VERSION,
        "manifest_sha256": _sha256(manifest_source),
        "source_run_id": manifest.source_parametric_null.run_id,
        "source_root": str(root),
        "source_sha256sums_sha256": manifest.source_parametric_null.sha256sums_sha256,
        "full_checksum_entries_verified": checked,
        "selected_cells": list(EXPECTED_CELL_IDS),
        "selected_tuples_verified": selected_count,
        "selected_identity_digest_sha256": selected_digest.hexdigest(),
        "parent_paths": {key: str(value) for key, value in path_receipts.items()},
        "parent_verification": parent_verification,
        "global_generating_grms_verified": ["full", "ldpruned"],
        "phenotypes_regenerated": False,
        "qualified": True,
    }
    if output is not None:
        _write_json_atomic(Path(output).resolve(), receipt)
    return receipt


def _qualify_mlma_log(
    path: Path,
    manifest: Manifest,
    source_root: Path,
    cell: Cell,
    phenotype: Path,
    bfile_prefix: Path,
    qcovar: Path | None,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"GCTA log is missing: {path}")
    text = path.read_text(encoding="utf-8")
    warnings = _qualify_log_text(text, path)
    options = _accepted_options(text, path)
    expected = {
        "--mlma",
        "--bfile",
        "--grm",
        "--pheno",
        "--maf",
        "--autosome-num",
        "--thread-num",
        "--out",
    }
    if qcovar is not None:
        expected.add("--qcovar")
    if set(options) != expected or options.get("--mlma") is not None:
        raise ValueError("matched analysis accepted options differ from the frozen --mlma call")
    if "--mlma-loco" in options:
        raise ValueError("matched analysis unexpectedly used --mlma-loco")
    expected_paths = {
        "--bfile": bfile_prefix,
        "--grm": source_root / cell.global_grm_relative_prefix,
        "--pheno": phenotype,
    }
    if qcovar is not None:
        expected_paths["--qcovar"] = qcovar
    for option, expected_path in expected_paths.items():
        value = options.get(option)
        if value is None or Path(value).resolve() != expected_path.resolve():
            raise ValueError(f"matched analysis {option} differs from the verified input")
    if (
        options.get("--maf") != str(manifest.execution.maf)
        or options.get("--autosome-num") != str(manifest.execution.autosome_num)
        or options.get("--thread-num") != str(manifest.execution.thread_count)
    ):
        raise ValueError("matched analysis numeric options differ from the frozen contract")
    if "209 individuals are in common in these files." not in text:
        raise ValueError("matched analysis log lacks exact common-sample evidence")
    if qcovar is not None:
        if "10 quantitative covariate(s) of 209 individuals are included" not in text:
            raise ValueError("matched PC10 analysis lacks covariate inclusion evidence")
    elif "quantitative covariate(s)" in text:
        raise ValueError("matched PC0 analysis unexpectedly included qcovars")
    convergence = [line.strip() for line in text.splitlines() if "converg" in line.lower()]
    if not convergence or any("not converg" in line.lower() for line in convergence):
        raise ValueError("matched analysis lacks positive convergence evidence")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "accepted_options": options,
        "convergence_evidence": convergence,
        "warnings": warnings,
    }


def qualify_matched_map(
    manifest_path: str | Path,
    source_root: str | Path,
    cell_id: str,
    replicate: int,
    bim_path: str | Path,
    bfile_prefix: str | Path,
    qcovar_path: str | Path | None,
    map_path: str | Path,
    log_path: str | Path,
    output: str | Path,
    retained_map: str | Path | None = None,
) -> dict[str, Any]:
    if replicate not in REPLICATES:
        raise ValueError("replicate must be 1..100")
    manifest_source = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_source)
    cell = _cell(manifest, cell_id)
    root = Path(source_root).resolve()
    sums = root / "receipts/SHA256SUMS.txt"
    _verify_file(sums, manifest.source_parametric_null.sha256sums_sha256, "parametric-null calibration checksum receipt")
    closure = parse_checksum_closure(sums)
    selected = _verify_selected_tuple(manifest, root, closure, cell, replicate)
    phenotype = selected["phenotype"]
    bfile = Path(bfile_prefix).resolve()
    bim = Path(bim_path).resolve()
    qcovar = Path(qcovar_path).resolve() if qcovar_path is not None else None
    if (cell.endpoint == "pc10") != (qcovar is not None):
        raise ValueError("qcovar presence differs from the selected endpoint")

    source_contract = root / "receipts/frozen_parametric_null_contract.json"
    _verify_file(source_contract, manifest.source_parametric_null.frozen_contract_sha256, "parametric-null calibration contract")
    parametric_null_manifest = parametric_null.load_manifest(source_contract)
    result_map = Path(map_path).resolve()
    diagnostics = parametric_null.stream_map_diagnostics(result_map, bim, cell.trait, parametric_null_manifest)
    log = _qualify_mlma_log(
        Path(log_path).resolve(), manifest, root, cell, phenotype, bfile, qcovar
    )

    retained = None
    if retained_map is not None:
        destination = Path(retained_map).resolve()
        if destination.exists():
            raise FileExistsError(f"retained map already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        try:
            shutil.copyfile(result_map, stage)
            if _sha256(stage) != diagnostics["map_sha256"]:
                raise RuntimeError("retained map copy differs from its qualified source")
            os.replace(stage, destination)
        except BaseException:
            if stage.exists():
                stage.unlink()
            raise
        retained = {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }
    elif replicate in SENTINELS:
        raise ValueError("frozen sentinel replicate must retain its complete map")

    source_checkpoint = selected["loco_checkpoint"]
    receipt = {
        "schema_version": MAP_VERSION,
        "manifest_sha256": _sha256(manifest_source),
        "source_parametric_null": {
            "run_id": manifest.source_parametric_null.run_id,
            "root": str(root),
            "phenotype_relative_path": selected["phenotype_relative_path"],
            "phenotype_sha256": selected["phenotype_sha256"],
            "simulation_receipt_relative_path": selected["simulation_relative_path"],
            "simulation_receipt_sha256": selected["simulation_sha256"],
            "loco_checkpoint_relative_path": selected["loco_checkpoint_relative_path"],
            "loco_checkpoint_sha256": selected["loco_checkpoint_sha256"],
        },
        "cell": cell.model_dump(),
        "replicate": replicate,
        "phenotype": {
            "path": str(phenotype),
            "bytes": phenotype.stat().st_size,
            "sha256": _sha256(phenotype),
            "regenerated": False,
        },
        "association_method": manifest.execution.association_method,
        "association_log": log,
        "matched_diagnostics": diagnostics,
        "source_loco_diagnostics": {
            key: source_checkpoint["diagnostics"][key] for key in REQUIRED_DIAGNOSTICS
        },
        "retained_map": retained,
        "raw_map_may_be_removed_after_checkpoint": retained is None,
        "paired_by_exact_cell_replicate_and_phenotype_sha256": True,
        "qualified": True,
    }
    _write_json_atomic(Path(output).resolve(), receipt)
    return receipt


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    ordered = sorted(float(value) for value in values)

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "replicates": len(ordered),
        "minimum": ordered[0],
        "q025": quantile(0.025),
        "q05": quantile(0.05),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "q95": quantile(0.95),
        "q975": quantile(0.975),
        "maximum": ordered[-1],
    }


def _fwer(diagnostics: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    events = sum(item["minimum_p"]["p"] <= threshold for item in diagnostics)
    return {
        "threshold": threshold,
        "replicates": len(diagnostics),
        "events": events,
        "rate": events / len(diagnostics),
        "binomial_interval": parametric_null._binomial_interval(events, len(diagnostics)),
        "nominal_not_threshold_grade": True,
    }


def _metric_comparison(
    matched: Sequence[dict[str, Any]], source: Sequence[dict[str, Any]], key: str
) -> dict[str, Any]:
    matched_values = [float(item[key]) for item in matched]
    source_values = [float(item[key]) for item in source]
    differences = [new - old for new, old in zip(matched_values, source_values, strict=True)]
    return {
        "matched_whole_grm": _distribution(matched_values),
        "source_loco": _distribution(source_values),
        "paired_difference_matched_minus_loco": _distribution(differences),
        "matched_lower_than_loco": sum(new < old for new, old in zip(matched_values, source_values, strict=True)),
        "ties": sum(new == old for new, old in zip(matched_values, source_values, strict=True)),
        "matched_higher_than_loco": sum(new > old for new, old in zip(matched_values, source_values, strict=True)),
    }


def aggregate(
    manifest_path: str | Path,
    source_root: str | Path,
    checkpoint_root: str | Path,
    output_directory: str | Path,
    run_manifest_output: str | Path,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_source)
    root = Path(source_root).resolve()
    source_verification = verify_parametric_null_source(manifest_source, root)
    checkpoints = Path(checkpoint_root).resolve()
    output = Path(output_directory).resolve()
    run_output = Path(run_manifest_output).resolve()
    if output.exists() or run_output.exists():
        raise FileExistsError("aggregate output and run manifest must not already exist")

    closure = parse_checksum_closure(root / "receipts/SHA256SUMS.txt")
    cells_summary = []
    inventory = []
    for cell in manifest.cells:
        matched_rows: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        for replicate in REPLICATES:
            checkpoint = checkpoints / cell.id / f"rep-{replicate:03d}.json"
            if not checkpoint.is_file() or checkpoint.is_symlink():
                raise FileNotFoundError(f"matched checkpoint is missing: {checkpoint}")
            payload = _load_object(checkpoint, "matched checkpoint")
            selected = _verify_selected_tuple(manifest, root, closure, cell, replicate)
            if (
                payload.get("schema_version") != MAP_VERSION
                or payload.get("manifest_sha256") != _sha256(manifest_source)
                or payload.get("replicate") != replicate
                or payload.get("qualified") is not True
                or payload.get("cell", {}).get("id") != cell.id
                or payload.get("source_parametric_null", {}).get("phenotype_sha256")
                != selected["phenotype_sha256"]
                or payload.get("source_parametric_null", {}).get("loco_checkpoint_sha256")
                != selected["loco_checkpoint_sha256"]
            ):
                raise ValueError(f"matched checkpoint binding failed: {checkpoint}")
            matched = payload.get("matched_diagnostics")
            source = payload.get("source_loco_diagnostics")
            if not isinstance(matched, dict) or not isinstance(source, dict):
                raise ValueError(f"matched checkpoint lacks paired diagnostics: {checkpoint}")
            if any(key not in matched or key not in source for key in REQUIRED_DIAGNOSTICS):
                raise ValueError(f"matched checkpoint diagnostics are incomplete: {checkpoint}")
            matched_rows.append(matched)
            source_rows.append(source)
            inventory.append(
                {
                    "cell": cell.id,
                    "replicate": replicate,
                    "path": str(checkpoint),
                    "sha256": _sha256(checkpoint),
                }
            )

        threshold = manifest.diagnostics.marker_bonferroni_p
        matched_events = [row["minimum_p"]["p"] <= threshold for row in matched_rows]
        source_events = [row["minimum_p"]["p"] <= threshold for row in source_rows]
        discordance = {
            "both": sum(a and b for a, b in zip(matched_events, source_events, strict=True)),
            "matched_only": sum(a and not b for a, b in zip(matched_events, source_events, strict=True)),
            "loco_only": sum(not a and b for a, b in zip(matched_events, source_events, strict=True)),
            "neither": sum(not a and not b for a, b in zip(matched_events, source_events, strict=True)),
        }
        per_chromosome = []
        for chromosome in range(1, 7):
            matched_p = [row["per_chromosome_minimum_p"][chromosome - 1]["p"] for row in matched_rows]
            source_p = [row["per_chromosome_minimum_p"][chromosome - 1]["p"] for row in source_rows]
            per_chromosome.append(
                {
                    "chromosome": chromosome,
                    "matched_whole_grm": _distribution(matched_p),
                    "source_loco": _distribution(source_p),
                    "paired_log10_ratio_matched_over_loco": _distribution(
                        [math.log10(a / b) for a, b in zip(matched_p, source_p, strict=True)]
                    ),
                }
            )
        qq = []
        for index, quantile_value in enumerate(manifest.diagnostics.qq_quantiles):
            matched_q = [row["qq_quantiles"][index]["observed_neg_log10_p"] for row in matched_rows]
            source_q = [row["qq_quantiles"][index]["observed_neg_log10_p"] for row in source_rows]
            if any(row["qq_quantiles"][index]["quantile"] != quantile_value for row in matched_rows + source_rows):
                raise ValueError("checkpoint QQ quantile order differs from the frozen contract")
            qq.append(
                {
                    "quantile": quantile_value,
                    "matched_whole_grm": _distribution(matched_q),
                    "source_loco": _distribution(source_q),
                    "paired_difference_matched_minus_loco": _distribution(
                        [a - b for a, b in zip(matched_q, source_q, strict=True)]
                    ),
                }
            )
        cells_summary.append(
            {
                "id": cell.id,
                "relationship_kind": cell.relationship_kind,
                "endpoint": cell.endpoint,
                "trait": cell.trait,
                "replicates": 100,
                "nominal_bonferroni_fwer": {
                    "matched_whole_grm": _fwer(matched_rows, threshold),
                    "source_loco": _fwer(source_rows, threshold),
                    "paired_discordance": discordance,
                },
                "minimum_p": _metric_comparison(
                    [{"value": row["minimum_p"]["p"]} for row in matched_rows],
                    [{"value": row["minimum_p"]["p"]} for row in source_rows],
                    "value",
                ),
                "bonferroni_marker_count": _metric_comparison(
                    matched_rows, source_rows, "bonferroni_marker_count"
                ),
                "lambda_gc": _metric_comparison(matched_rows, source_rows, "lambda_gc"),
                "lambda_gc_excluding_trait_intervals": _metric_comparison(
                    matched_rows, source_rows, "lambda_gc_excluding_trait_intervals"
                ),
                "lambda_gc_excluding_all_intervals": _metric_comparison(
                    matched_rows, source_rows, "lambda_gc_excluding_all_intervals"
                ),
                "per_chromosome_minimum_p": per_chromosome,
                "qq_quantiles": qq,
            }
        )

    summary = {
        "schema_version": SUMMARY_VERSION,
        "analysis_id": manifest.analysis_id,
        "classification": manifest.classification,
        "status": "technical_complete_interpretation_pending",
        "validated": False,
        "biological_claims_permitted": False,
        "manifest_sha256": _sha256(manifest_source),
        "source_parametric_null": source_verification,
        "design": {
            "cells": list(EXPECTED_CELL_IDS),
            "replicates_per_cell": 100,
            "new_maps": 400,
            "paired_on": ["cell", "replicate", "phenotype_sha256"],
            "matched_model": manifest.execution.association_method,
            "comparator": "existing_PARAMETRIC_NULL_LOCO_checkpoint",
            "pooled_cells": False,
        },
        "cells": cells_summary,
        "claim_boundary": manifest.claims.model_dump(),
    }

    output.mkdir(parents=True)
    summary_json = output / "matched_relationship_model_matched_model_summary.json"
    summary_md = output / "matched_relationship_model_matched_model_summary.md"
    _write_json_atomic(summary_json, summary)
    lines = [
        "# matched relationship-model diagnostic matched-model diagnostic",
        "",
        "This is a paired model-mismatch diagnostic, not threshold-grade calibration.",
        "All 400 matched maps reused the exact retained parametric-null calibration phenotype bytes.",
        "",
        "| Cell | LOCO nominal FWER | Matched whole-GRM nominal FWER | LOCO-only | Matched-only |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in cells_summary:
        fwer = item["nominal_bonferroni_fwer"]
        lines.append(
            f"| `{item['id']}` | {fwer['source_loco']['rate']:.3f} | "
            f"{fwer['matched_whole_grm']['rate']:.3f} | "
            f"{fwer['paired_discordance']['loco_only']} | "
            f"{fwer['paired_discordance']['matched_only']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation must remain cell-specific. Correlated cells and shared random streams are not independent replications.",
            "",
        ]
    )
    summary_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    inventory = sorted(inventory, key=lambda item: (item["cell"], item["replicate"]))
    run_receipt = {
        "schema_version": RUN_VERSION,
        "analysis_id": manifest.analysis_id,
        "manifest_sha256": _sha256(manifest_source),
        "source_parametric_null_run_id": manifest.source_parametric_null.run_id,
        "source_parametric_null_sha256sums_sha256": manifest.source_parametric_null.sha256sums_sha256,
        "attempted_maps": 400,
        "completed_maps": 400,
        "phenotypes_regenerated": 0,
        "exact_source_phenotypes_reused": 400,
        "source_loco_checkpoints_paired": 400,
        "cells": list(EXPECTED_CELL_IDS),
        "pooled_cells": False,
        "summary_json": {"path": str(summary_json), "sha256": _sha256(summary_json)},
        "summary_markdown": {"path": str(summary_md), "sha256": _sha256(summary_md)},
        "checkpoint_inventory": inventory,
        "checkpoint_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "qualified": True,
    }
    _write_json_atomic(run_output, run_receipt)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-source")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--output", type=Path)

    qualify = commands.add_parser("qualify-map")
    qualify.add_argument("--manifest", type=Path, required=True)
    qualify.add_argument("--source-root", type=Path, required=True)
    qualify.add_argument("--cell", choices=EXPECTED_CELL_IDS, required=True)
    qualify.add_argument("--replicate", type=int, required=True)
    qualify.add_argument("--bim", type=Path, required=True)
    qualify.add_argument("--bfile-prefix", type=Path, required=True)
    qualify.add_argument("--qcovar", type=Path)
    qualify.add_argument("--map", type=Path, required=True)
    qualify.add_argument("--gcta-log", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    qualify.add_argument("--retained-map", type=Path)

    summarize = commands.add_parser("aggregate")
    summarize.add_argument("--manifest", type=Path, required=True)
    summarize.add_argument("--source-root", type=Path, required=True)
    summarize.add_argument("--checkpoint-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--run-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify-source":
            result = verify_parametric_null_source(args.manifest, args.source_root, args.output)
        elif args.command == "qualify-map":
            result = qualify_matched_map(
                args.manifest,
                args.source_root,
                args.cell,
                args.replicate,
                args.bim,
                args.bfile_prefix,
                args.qcovar,
                args.map,
                args.gcta_log,
                args.output,
                args.retained_map,
            )
        elif args.command == "aggregate":
            result = aggregate(
                args.manifest,
                args.source_root,
                args.checkpoint_root,
                args.output,
                args.run_manifest,
            )
        else:  # pragma: no cover
            raise ValueError(f"unsupported command: {args.command}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
