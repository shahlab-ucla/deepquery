"""Frozen WS283 LD-pruned leave-one-chromosome-out GRM sensitivity.

This lane binds the original 209-sample WS283 technical run and the completed
PC calibration as two immutable parents.  It constructs six dense GRMs from
the exact frozen LD-pruned marker set, excluding the chromosome being tested,
then compares PC0 and PC10 external-GRM MLMA maps with their matched inherited
full-marker MLMA-LOCO references.  It is a calibration sensitivity only.
"""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .qtl import MLMA_HEADER, TRAITS, TRAIT_SLUGS, _lambda_gc, _parse_mlma, load_qtl_manifest
from .qtl_calibration import (
    _deterministic_spearman,
    _in_intervals,
    _interval_summary,
    _qq_quantiles,
    _top_overlap,
)


SCHEMA_VERSION = "wormctx-abamectin-ws283-ldpruned-grm-sensitivity-1.0"
SUMMARY_VERSION = "wormctx-abamectin-ws283-ldpruned-grm-sensitivity-summary-1.0"
MODEL_IDS = ("ldgrm_pc0", "ldgrm_pc10")
REFERENCE_MODEL_IDS = ("pc0", "pc10")
CHROMOSOMES = (1, 2, 3, 4, 5, 6)
_SHA256 = frozenset("0123456789abcdef")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FrozenFile(_StrictModel):
    relative_path: str
    bytes: int = Field(ge=0)
    sha256: str

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("relative_path must use POSIX separators")
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("relative_path must remain below its parent run")
        return path.as_posix()

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in _SHA256 for character in value):
            raise ValueError("sha256 must be lowercase hexadecimal")
        return value


def _unique_paths(files: list[FrozenFile]) -> set[str]:
    paths = [item.relative_path for item in files]
    if len(paths) != len(set(paths)):
        raise ValueError("required file paths must be unique")
    return set(paths)


def _frozen_file(files: list[FrozenFile], relative_path: str) -> FrozenFile:
    matches = [item for item in files if item.relative_path == relative_path]
    if len(matches) != 1:
        raise ValueError(f"frozen file is not unique in the contract: {relative_path}")
    return matches[0]


class BaselineParentContract(_StrictModel):
    run_id: Literal["abamectin-ws283-bd41637-20260717T200012Z"]
    source_git_commit: Literal["bd41637f5ff2998333964cc954ee9f9f36a351d6"]
    post_qc_samples: Literal[209]
    post_qc_markers: Literal[373279]
    checksum_file_count: Literal[44]
    required_files: list[FrozenFile]

    @model_validator(mode="after")
    def exact_file_set(self) -> "BaselineParentContract":
        expected = {
            "SUCCESS",
            "receipts/SOURCE_REVISION",
            "receipts/frozen_analysis_manifest.json",
            "receipts/post_qc_counts.tsv",
            "receipts/SHA256SUMS.txt",
            "genotype/abamectin_209_qc.bed",
            "genotype/abamectin_209_qc.bim",
            "genotype/abamectin_209_qc.fam",
            *(f"prepared/cohort/{TRAIT_SLUGS[trait]}.phen" for trait in TRAITS),
            *(f"association/{TRAIT_SLUGS[trait]}.loco.mlma" for trait in TRAITS),
        }
        if _unique_paths(self.required_files) != expected:
            raise ValueError("baseline parent file set differs from the frozen contract")
        return self


class CalibrationParentContract(_StrictModel):
    run_id: Literal["abamectin-ws283-pc-calibration-20260720T215332Z-192322baefdf"]
    source_git_commit: Literal["192322baefdfa246a8422de17e5a380fa7c5d93f"]
    post_qc_samples: Literal[209]
    post_qc_markers: Literal[373279]
    ld_pruned_markers: Literal[1370]
    checksum_file_count: Literal[150]
    required_files: list[FrozenFile]

    @model_validator(mode="after")
    def exact_file_set(self) -> "CalibrationParentContract":
        expected = {
            "SUCCESS",
            "receipts/SOURCE_REVISION",
            "receipts/frozen_calibration_contract.json",
            "receipts/run_manifest.json",
            "receipts/pca_marker_counts.tsv",
            "receipts/SHA256SUMS.txt",
            "receipts/deployer-runner-exit-code.txt",
            "logs/checksums.verify.log",
            "genotype/ws283_ld_pruned.prune.in",
            "genotype/ws283_ld_pruned_pca10.eigenvec",
            "genotype/qcovars/pc10.qcovar",
            "summary/ld_clump_counts.tsv",
            "summary/ws283_pc_calibration/ws283_pc_calibration_summary.json",
            *(
                f"association/{model}/{TRAIT_SLUGS[trait]}.loco.mlma"
                for model in REFERENCE_MODEL_IDS
                for trait in TRAITS
            ),
        }
        if _unique_paths(self.required_files) != expected:
            raise ValueError("calibration parent file set differs from the frozen contract")
        return self


class ToolContract(_StrictModel):
    name: Literal["plink2", "gcta64"]
    version: str
    absolute_path: str
    sha256: str

    @field_validator("absolute_path")
    @classmethod
    def normalized_absolute_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("tool path must be normalized and absolute")
        return path.as_posix()

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in _SHA256 for character in value):
            raise ValueError("tool sha256 must be lowercase hexadecimal")
        return value


class ChromosomeContract(_StrictModel):
    chromosome: int
    association_markers: int = Field(gt=0)
    pruned_on_chromosome: int = Field(gt=0)
    relationship_markers: int = Field(gt=0)
    association_list_sha256: str
    relationship_list_sha256: str

    @field_validator("association_list_sha256", "relationship_list_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in _SHA256 for character in value):
            raise ValueError("marker-list sha256 must be lowercase hexadecimal")
        return value


class RelationshipContract(_StrictModel):
    source: Literal["calibration_parent_frozen_ld_prune_in"]
    method: Literal["gcta_make_grm_default_algorithm_0"]
    leave_one_chromosome_out: Literal[True]
    gcta_autosome_num: Literal[6]
    use_make_grm_inbred: Literal[False]
    maf_refilter: Literal[False]
    sample_order: Literal["baseline_parent_fam_order"]
    total_ld_pruned_markers: Literal[1370]
    chromosomes: list[ChromosomeContract]

    @model_validator(mode="after")
    def exact_chromosome_counts(self) -> "RelationshipContract":
        expected = {
            1: (
                31023,
                146,
                1224,
                "56efd95d1e8e71c04b192ae8a828154dbd0ec29ba7c142720bcf5465656dc350",
                "2e5594a005f1bbe7b4a19f361d38dbb25f6e6dfb7d6559e07bceca7390918cf0",
            ),
            2: (
                76840,
                400,
                970,
                "5d197e1dce2b396a2227e144cb69f102922fd2719ae9b26ee3a07aa57a32440d",
                "62fa3d015e3d3fee1bca70da91325f9e5a2b5d369c8c5d9ec098f4e631090291",
            ),
            3: (
                55886,
                242,
                1128,
                "fa14193fa9c871d92d8429e6f462b683508db0181cffcd37936a56eb2b0ec08f",
                "7f3aff20f94c33e14f09eaf3675d666eafe67c126a0403eb4cc8992e2f01106b",
            ),
            4: (
                53961,
                149,
                1221,
                "3927131bc0c7b813a1aa8862bde44a772d13191e169ea0d4dae09af3b3d3b31a",
                "9e80be4491620ab42add58263e1d39ef32ea7a468b512034501d3cf70cf5ad09",
            ),
            5: (
                115183,
                290,
                1080,
                "67f5169cd25982442ffe65a64cf3b96d200d28d98635fb1d09c875daf3e1dfd4",
                "61d7d87b08e25fd9f3eb63e368d524d93c2801109d74f2de27ee17b2de448ae5",
            ),
            6: (
                40386,
                143,
                1227,
                "4ed9409ca04b4994c284a3628b70715d1c8f5f5735f8f5d7215c873cf0b298eb",
                "88adafc69a496d035bd850918ac573063505ccd03d2e807192de0e12a4bc48ff",
            ),
        }
        observed = {
            item.chromosome: (
                item.association_markers,
                item.pruned_on_chromosome,
                item.relationship_markers,
                item.association_list_sha256,
                item.relationship_list_sha256,
            )
            for item in self.chromosomes
        }
        if observed != expected or len(self.chromosomes) != len(expected):
            raise ValueError("chromosome marker counts differ from the frozen contract")
        if sum(item.association_markers for item in self.chromosomes) != 373279:
            raise ValueError("association marker counts do not sum to 373279")
        if sum(item.pruned_on_chromosome for item in self.chromosomes) != 1370:
            raise ValueError("pruned marker counts do not sum to 1370")
        if any(
            item.relationship_markers != 1370 - item.pruned_on_chromosome
            for item in self.chromosomes
        ):
            raise ValueError("relationship counts are not chromosome complements")
        return self


class AssociationModel(_StrictModel):
    id: Literal["ldgrm_pc0", "ldgrm_pc10"]
    pc_count: Literal[0, 10]
    reference_model: Literal["pc0", "pc10"]
    qcovar_relative_path: Literal["genotype/qcovars/pc10.qcovar"] | None


class AssociationContract(_StrictModel):
    method: Literal["GCTA_MLMA_external_LD_pruned_LOCO_GRM"]
    gcta_autosome_num: Literal[6]
    thread_count: Literal[16]
    maf: Literal[0.05]
    chromosome_scan_count: Literal[48]
    assembled_map_count: Literal[8]
    models: list[AssociationModel]
    covariate_fitting: Literal["gcta_default_preadjustment"]
    model_selection_from_results_permitted: Literal[False]

    @model_validator(mode="after")
    def exact_models(self) -> "AssociationContract":
        if [item.id for item in self.models] != list(MODEL_IDS):
            raise ValueError(f"models must appear in order {list(MODEL_IDS)}")
        if [item.pc_count for item in self.models] != [0, 10]:
            raise ValueError("model PC endpoints must be [0, 10]")
        if [item.reference_model for item in self.models] != list(REFERENCE_MODEL_IDS):
            raise ValueError("models must use matched pc0 and pc10 references")
        if self.models[0].qcovar_relative_path is not None:
            raise ValueError("ldgrm_pc0 must omit qcovar")
        if self.models[1].qcovar_relative_path != "genotype/qcovars/pc10.qcovar":
            raise ValueError("ldgrm_pc10 must reuse the frozen PC10 qcovar")
        return self


class ClumpContract(_StrictModel):
    method: Literal["plink2_unphased"]
    p1: float
    p2: Literal[0.05]
    r2: Literal[0.2]
    kb: Literal[1000]


class DiagnosticContract(_StrictModel):
    alpha: Literal[0.05]
    marker_bonferroni_p: float
    historical_effective_test_p_reference_only: float
    qq_quantiles: list[float]
    lambda_exclusions: list[str]
    ld_clump: ClumpContract
    top_set_sizes: list[int]

    @model_validator(mode="after")
    def frozen_diagnostics(self) -> "DiagnosticContract":
        expected = 0.05 / 373279
        if not math.isclose(self.marker_bonferroni_p, expected, rel_tol=0.0, abs_tol=1e-20):
            raise ValueError("marker Bonferroni threshold differs from 0.05 / 373279")
        if not math.isclose(self.ld_clump.p1, expected, rel_tol=0.0, abs_tol=1e-20):
            raise ValueError("clump p1 must equal the marker Bonferroni threshold")
        if self.qq_quantiles != [0.5, 0.9, 0.95, 0.99, 0.999]:
            raise ValueError("QQ quantiles differ from the frozen contract")
        if self.lambda_exclusions != [
            "none",
            "trait_published_intervals",
            "all_published_intervals",
        ]:
            raise ValueError("lambda exclusions differ from the frozen contract")
        if self.top_set_sizes != [100, 1000]:
            raise ValueError("top-set sizes differ from the frozen contract")
        return self


class ClaimContract(_StrictModel):
    permitted: list[str]
    prohibited: list[str]


class GrmSensitivityManifest(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    analysis_id: Literal[
        "abamectin_qtl_ws283_caendr20250625_ldpruned_grm_sensitivity_v1"
    ]
    classification: Literal["exploratory_ws283_ldpruned_loco_grm_sensitivity"]
    status: Literal["predeclared_post_calibration_grm_sensitivity"]
    validated: Literal[False]
    biological_claims_permitted: Literal[False]
    baseline_parent: BaselineParentContract
    calibration_parent: CalibrationParentContract
    tools: list[ToolContract]
    relationship_matrix: RelationshipContract
    association: AssociationContract
    diagnostics: DiagnosticContract
    claims: ClaimContract

    @model_validator(mode="after")
    def exact_tools_and_claim_boundary(self) -> "GrmSensitivityManifest":
        if [tool.name for tool in self.tools] != ["plink2", "gcta64"]:
            raise ValueError("tools must be frozen in plink2, gcta64 order")
        prohibited = " ".join(self.claims.prohibited).lower()
        if "causal" not in prohibited or "selection" not in prohibited:
            raise ValueError("claim boundary must prohibit causal claims and model selection")
        return self


@dataclass(frozen=True)
class _BimMarker:
    chromosome: int
    snp: str
    bp: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_grm_manifest(path: str | Path) -> GrmSensitivityManifest:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GRM-sensitivity manifest is not a file: {source}")
    return GrmSensitivityManifest.model_validate_json(source.read_text(encoding="utf-8"))


def _path_below(root: Path, relative_path: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative_path).parts)
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes parent run: {relative_path}") from exc
    return resolved


def _verify_frozen_files(root: Path, files: list[FrozenFile]) -> list[dict[str, Any]]:
    verified = []
    for item in files:
        path = _path_below(root, item.relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen parent file: {item.relative_path}")
        observed_bytes = path.stat().st_size
        observed_hash = _sha256(path)
        if observed_bytes != item.bytes or observed_hash != item.sha256:
            raise ValueError(f"parent file identity mismatch: {item.relative_path}")
        verified.append(
            {
                "relative_path": item.relative_path,
                "bytes": observed_bytes,
                "sha256": observed_hash,
            }
        )
    return verified


def _verify_internal_checksums(
    root: Path, receipt_relative_path: str, expected_count: int
) -> dict[str, Any]:
    receipt = _path_below(root, receipt_relative_path)
    lines = receipt.read_text(encoding="utf-8").splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"{root.name} checksum receipt has the wrong entry count: "
            f"{len(lines)} != {expected_count}"
        )
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if "  ./" not in line:
            raise ValueError(f"malformed checksum receipt line {line_number}")
        expected_hash, relative = line.split("  ./", 1)
        if len(expected_hash) != 64 or any(character not in _SHA256 for character in expected_hash):
            raise ValueError(f"malformed checksum hash on line {line_number}")
        posix = PurePosixPath(relative)
        if posix.is_absolute() or not posix.parts or ".." in posix.parts or "\\" in relative:
            raise ValueError(f"unsafe checksum path on line {line_number}")
        normalized = posix.as_posix()
        if normalized in seen:
            raise ValueError(f"duplicate checksum path: {normalized}")
        seen.add(normalized)
        path = _path_below(root, normalized)
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"internal parent checksum mismatch: {normalized}")
        verified.append({"relative_path": normalized, "sha256": expected_hash})
    return {
        "receipt": receipt_relative_path,
        "receipt_sha256": _sha256(receipt),
        "verified_file_count": len(verified),
    }


def _read_fam(path: Path, expected_samples: int = 209) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"FAM line {line_number} must contain six fields")
        samples.append((fields[0], fields[1]))
    if len(samples) != expected_samples or len(samples) != len(set(samples)):
        raise ValueError(f"FAM must contain exactly {expected_samples} unique FID/IID pairs")
    return samples


def _read_bim(path: Path, expected_markers: int) -> list[_BimMarker]:
    markers: list[_BimMarker] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"BIM line {line_number} must contain six fields")
        try:
            chromosome = int(fields[0])
            bp = int(fields[3])
        except ValueError as exc:
            raise ValueError(f"BIM line {line_number} has non-numeric coordinates") from exc
        snp = fields[1]
        if chromosome not in CHROMOSOMES or bp <= 0 or not snp or snp in seen:
            raise ValueError(f"BIM line {line_number} has invalid or duplicate marker identity")
        seen.add(snp)
        markers.append(_BimMarker(chromosome, snp, bp))
    if len(markers) != expected_markers:
        raise ValueError(f"BIM marker count differs from {expected_markers}")
    return markers


def _read_snplist(path: Path, expected_markers: int) -> list[str]:
    markers = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(markers) != expected_markers or len(markers) != len(set(markers)):
        raise ValueError(f"marker list must contain exactly {expected_markers} unique IDs")
    if any(any(character.isspace() for character in marker) for marker in markers):
        raise ValueError("marker IDs cannot contain whitespace")
    return markers


def _verify_qcovar(path: Path, samples: list[tuple[str, str]]) -> None:
    observed: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 12:
            raise ValueError(f"PC10 qcovar line {line_number} must contain 12 fields")
        try:
            scores = [float(value) for value in fields[2:]]
        except ValueError as exc:
            raise ValueError(f"PC10 qcovar line {line_number} is not numeric") from exc
        if not all(math.isfinite(value) for value in scores):
            raise ValueError(f"PC10 qcovar line {line_number} contains non-finite scores")
        observed.append((fields[0], fields[1]))
    if observed != samples:
        raise ValueError("PC10 qcovar sample IDs/order differ from the baseline FAM")


def verify_parents(
    manifest_path: str | Path,
    baseline_parent: str | Path,
    calibration_parent: str | Path,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_grm_manifest(manifest_source)
    baseline = Path(baseline_parent).resolve()
    calibration = Path(calibration_parent).resolve()
    if not baseline.is_dir() or baseline.name != manifest.baseline_parent.run_id:
        raise ValueError("baseline parent directory differs from the frozen contract")
    if not calibration.is_dir() or calibration.name != manifest.calibration_parent.run_id:
        raise ValueError("calibration parent directory differs from the frozen contract")
    for root in (baseline, calibration):
        if (root / "FAILURE").exists():
            raise ValueError(f"successful parent contains a FAILURE marker: {root.name}")
        if (root / "SUCCESS").read_text(encoding="utf-8").strip() != "SUCCESS":
            raise ValueError(f"parent SUCCESS marker is malformed: {root.name}")

    baseline_verified = _verify_frozen_files(baseline, manifest.baseline_parent.required_files)
    calibration_verified = _verify_frozen_files(
        calibration, manifest.calibration_parent.required_files
    )
    for root, expected_revision in (
        (baseline, manifest.baseline_parent.source_git_commit),
        (calibration, manifest.calibration_parent.source_git_commit),
    ):
        revision = (root / "receipts/SOURCE_REVISION").read_text(encoding="utf-8").strip()
        if revision != expected_revision:
            raise ValueError(f"parent source revision mismatch: {root.name}")

    counts = dict(
        line.split("\t", 1)
        for line in (baseline / "receipts/post_qc_counts.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    if counts != {
        "post_qc_samples": "209",
        "post_qc_markers": "373279",
        "chromosomes": "1,2,3,4,5,6",
    }:
        raise ValueError("baseline post-QC counts differ from the frozen contract")
    pca_counts = dict(
        line.split("\t", 1)
        for line in (calibration / "receipts/pca_marker_counts.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    if pca_counts != {"source_markers": "373279", "pruned_markers": "1370"}:
        raise ValueError("calibration PCA marker counts differ from the frozen contract")

    baseline_internal = _verify_internal_checksums(
        baseline,
        "receipts/SHA256SUMS.txt",
        manifest.baseline_parent.checksum_file_count,
    )
    calibration_internal = _verify_internal_checksums(
        calibration,
        "receipts/SHA256SUMS.txt",
        manifest.calibration_parent.checksum_file_count,
    )
    if (
        calibration / "receipts/deployer-runner-exit-code.txt"
    ).read_text(encoding="utf-8") != "0\n":
        raise ValueError("calibration deployer exit receipt is not exactly zero")
    checksum_log = (
        calibration / "logs/checksums.verify.log"
    ).read_text(encoding="utf-8").splitlines()
    if (
        len(checksum_log) != manifest.calibration_parent.checksum_file_count
        or any(not line.endswith(": OK") for line in checksum_log)
    ):
        raise ValueError("calibration checksum verification log is not an all-OK receipt")
    fam = _read_fam(baseline / "genotype/abamectin_209_qc.fam")
    qcovar = calibration / "genotype/qcovars/pc10.qcovar"
    _verify_qcovar(qcovar, fam)
    prune = _read_snplist(
        calibration / "genotype/ws283_ld_pruned.prune.in",
        manifest.calibration_parent.ld_pruned_markers,
    )
    bim = _read_bim(
        baseline / "genotype/abamectin_209_qc.bim",
        manifest.baseline_parent.post_qc_markers,
    )
    if not set(prune).issubset({marker.snp for marker in bim}):
        raise ValueError("calibration prune set contains IDs absent from the baseline BIM")

    run_manifest = json.loads(
        (calibration / "receipts/run_manifest.json").read_text(encoding="utf-8")
    )
    if (
        run_manifest.get("source_markers") != 373279
        or run_manifest.get("samples") != 209
        or run_manifest.get("ld_pruned_pca_markers") != 1370
        or run_manifest.get("alternative_grm_performed") is not False
        or run_manifest.get("summary_sha256")
        != "d2cafd479c235b19ef4e7de820b78869e14e62b9eec3bc280d7b65e2b329c8a0"
    ):
        raise ValueError("calibration run manifest differs from the frozen decision parent")

    return {
        "verified": True,
        "manifest_sha256": _sha256(manifest_source),
        "baseline_parent": str(baseline),
        "baseline_parent_run_id": baseline.name,
        "baseline_verified_files": baseline_verified,
        "calibration_parent": str(calibration),
        "calibration_parent_run_id": calibration.name,
        "calibration_verified_files": calibration_verified,
        "baseline_internal_checksums": baseline_internal,
        "calibration_internal_checksums": calibration_internal,
        "samples": len(fam),
        "association_markers": len(bim),
        "ld_pruned_markers": len(prune),
        "pc10_qcovar_sha256": _sha256(qcovar),
    }


def prepare_marker_lists(
    manifest_path: str | Path,
    bim_path: str | Path,
    prune_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_grm_manifest(manifest_source)
    bim_source = Path(bim_path).resolve()
    prune_source = Path(prune_path).resolve()
    if not bim_source.is_file() or not prune_source.is_file():
        raise FileNotFoundError("baseline BIM or frozen prune list is missing")
    expected_bim = _frozen_file(
        manifest.baseline_parent.required_files,
        "genotype/abamectin_209_qc.bim",
    )
    expected_prune = _frozen_file(
        manifest.calibration_parent.required_files,
        "genotype/ws283_ld_pruned.prune.in",
    )
    if (
        bim_source.stat().st_size != expected_bim.bytes
        or _sha256(bim_source) != expected_bim.sha256
    ):
        raise ValueError("BIM identity differs from the frozen baseline parent")
    if (
        prune_source.stat().st_size != expected_prune.bytes
        or _sha256(prune_source) != expected_prune.sha256
    ):
        raise ValueError("prune-list identity differs from the frozen calibration parent")
    bim = _read_bim(bim_source, manifest.baseline_parent.post_qc_markers)
    prune_ids = _read_snplist(prune_source, manifest.calibration_parent.ld_pruned_markers)
    prune_set = set(prune_ids)
    if not prune_set.issubset({marker.snp for marker in bim}):
        unknown = sorted(prune_set - {marker.snp for marker in bim})[:5]
        raise ValueError(f"prune list contains marker IDs absent from BIM: {unknown}")
    pruned_in_bim_order = [marker for marker in bim if marker.snp in prune_set]
    by_chromosome = {
        chromosome: [marker for marker in bim if marker.chromosome == chromosome]
        for chromosome in CHROMOSOMES
    }
    pruned_by_chromosome = {
        chromosome: [
            marker for marker in pruned_in_bim_order if marker.chromosome == chromosome
        ]
        for chromosome in CHROMOSOMES
    }
    expected_by_chromosome = {
        item.chromosome: item for item in manifest.relationship_matrix.chromosomes
    }
    for chromosome in CHROMOSOMES:
        expected = expected_by_chromosome[chromosome]
        if len(by_chromosome[chromosome]) != expected.association_markers:
            raise ValueError(f"association marker count mismatch for chromosome {chromosome}")
        if len(pruned_by_chromosome[chromosome]) != expected.pruned_on_chromosome:
            raise ValueError(f"pruned marker count mismatch for chromosome {chromosome}")

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"marker-list output must not exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        (stage / "association").mkdir(parents=True)
        (stage / "relationship").mkdir()
        chromosome_receipts = []
        count_lines = [
            "chromosome\tassociation_markers\tpruned_on_chromosome\trelationship_markers"
        ]
        for chromosome in CHROMOSOMES:
            candidate_path = stage / "association" / f"candidate_chr{chromosome}.snplist"
            relationship_path = (
                stage / "relationship" / f"grm_exclude_chr{chromosome}.snplist"
            )
            candidates = [marker.snp for marker in by_chromosome[chromosome]]
            relationship = [
                marker.snp
                for marker in pruned_in_bim_order
                if marker.chromosome != chromosome
            ]
            expected = expected_by_chromosome[chromosome]
            if len(relationship) != expected.relationship_markers:
                raise ValueError(f"GRM complement count mismatch for chromosome {chromosome}")
            if set(candidates) & set(relationship):
                raise ValueError(f"candidate and GRM marker sets overlap for chromosome {chromosome}")
            candidate_path.write_text(
                "".join(f"{marker}\n" for marker in candidates),
                encoding="utf-8",
                newline="\n",
            )
            relationship_path.write_text(
                "".join(f"{marker}\n" for marker in relationship),
                encoding="utf-8",
                newline="\n",
            )
            candidate_hash = _sha256(candidate_path)
            relationship_hash = _sha256(relationship_path)
            if candidate_hash != expected.association_list_sha256:
                raise ValueError(
                    f"association marker-list identity mismatch for chromosome {chromosome}"
                )
            if relationship_hash != expected.relationship_list_sha256:
                raise ValueError(
                    f"GRM complement marker-list identity mismatch for chromosome {chromosome}"
                )
            count_lines.append(
                f"{chromosome}\t{len(candidates)}\t"
                f"{len(pruned_by_chromosome[chromosome])}\t{len(relationship)}"
            )
            chromosome_receipts.append(
                {
                    "chromosome": chromosome,
                    "association_markers": len(candidates),
                    "association_list": f"association/{candidate_path.name}",
                    "association_list_sha256": candidate_hash,
                    "pruned_on_chromosome": len(pruned_by_chromosome[chromosome]),
                    "relationship_markers": len(relationship),
                    "relationship_list": f"relationship/{relationship_path.name}",
                    "relationship_list_sha256": relationship_hash,
                }
            )
        counts_path = stage / "marker_counts.tsv"
        counts_path.write_text("\n".join(count_lines) + "\n", encoding="utf-8", newline="\n")
        receipt = {
            "schema_version": "wormctx-abamectin-ws283-grm-marker-lists-1.0",
            "manifest_sha256": _sha256(manifest_source),
            "baseline_bim_sha256": _sha256(bim_source),
            "frozen_prune_in_sha256": _sha256(prune_source),
            "association_markers": len(bim),
            "ld_pruned_markers": len(pruned_in_bim_order),
            "marker_order": "baseline_bim_order",
            "chromosomes": chromosome_receipts,
            "marker_counts_sha256": _sha256(counts_path),
        }
        _write_json(stage / "marker_manifest.json", receipt)
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return receipt


def _verify_marker_manifest(
    manifest: GrmSensitivityManifest,
    manifest_source: Path,
    marker_manifest_source: Path,
) -> dict[str, Any]:
    payload = json.loads(marker_manifest_source.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "manifest_sha256",
        "baseline_bim_sha256",
        "frozen_prune_in_sha256",
        "association_markers",
        "ld_pruned_markers",
        "marker_order",
        "chromosomes",
        "marker_counts_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("marker manifest fields differ from the frozen receipt schema")
    expected_bim = _frozen_file(
        manifest.baseline_parent.required_files,
        "genotype/abamectin_209_qc.bim",
    )
    expected_prune = _frozen_file(
        manifest.calibration_parent.required_files,
        "genotype/ws283_ld_pruned.prune.in",
    )
    if payload != {
        **payload,
        "schema_version": "wormctx-abamectin-ws283-grm-marker-lists-1.0",
        "manifest_sha256": _sha256(manifest_source),
        "baseline_bim_sha256": expected_bim.sha256,
        "frozen_prune_in_sha256": expected_prune.sha256,
        "association_markers": manifest.baseline_parent.post_qc_markers,
        "ld_pruned_markers": manifest.relationship_matrix.total_ld_pruned_markers,
        "marker_order": "baseline_bim_order",
    }:
        raise ValueError("marker manifest provenance differs from the frozen contract")

    rows = payload["chromosomes"]
    if not isinstance(rows, list) or len(rows) != len(CHROMOSOMES):
        raise ValueError("marker manifest must contain exactly six chromosome rows")
    row_keys = {
        "chromosome",
        "association_markers",
        "association_list",
        "association_list_sha256",
        "pruned_on_chromosome",
        "relationship_markers",
        "relationship_list",
        "relationship_list_sha256",
    }
    observed_chromosomes = [row.get("chromosome") for row in rows if isinstance(row, dict)]
    if observed_chromosomes != list(CHROMOSOMES):
        raise ValueError("marker manifest chromosome rows are missing, duplicated, or reordered")
    contract_by_chromosome = {
        item.chromosome: item for item in manifest.relationship_matrix.chromosomes
    }
    for row in rows:
        if set(row) != row_keys:
            raise ValueError("marker manifest chromosome fields differ from the receipt schema")
        chromosome = row["chromosome"]
        contract = contract_by_chromosome[chromosome]
        expected_row = {
            "chromosome": chromosome,
            "association_markers": contract.association_markers,
            "association_list": f"association/candidate_chr{chromosome}.snplist",
            "association_list_sha256": contract.association_list_sha256,
            "pruned_on_chromosome": contract.pruned_on_chromosome,
            "relationship_markers": contract.relationship_markers,
            "relationship_list": f"relationship/grm_exclude_chr{chromosome}.snplist",
            "relationship_list_sha256": contract.relationship_list_sha256,
        }
        if row != expected_row:
            raise ValueError(
                f"marker manifest row differs from the frozen contract for chromosome {chromosome}"
            )
        candidate_path = _path_below(marker_manifest_source.parent, row["association_list"])
        relationship_path = _path_below(
            marker_manifest_source.parent, row["relationship_list"]
        )
        candidates = _read_snplist(candidate_path, contract.association_markers)
        relationship = _read_snplist(relationship_path, contract.relationship_markers)
        if _sha256(candidate_path) != contract.association_list_sha256:
            raise ValueError(f"association marker list is altered for chromosome {chromosome}")
        if _sha256(relationship_path) != contract.relationship_list_sha256:
            raise ValueError(f"GRM complement marker list is altered for chromosome {chromosome}")
        if set(candidates) & set(relationship):
            raise ValueError(
                f"candidate and GRM complement lists overlap for chromosome {chromosome}"
            )

    counts_path = marker_manifest_source.parent / "marker_counts.tsv"
    if not counts_path.is_file() or _sha256(counts_path) != payload["marker_counts_sha256"]:
        raise ValueError("marker-count table identity differs from the marker manifest")
    expected_count_lines = [
        "chromosome\tassociation_markers\tpruned_on_chromosome\trelationship_markers",
        *(
            f"{item.chromosome}\t{item.association_markers}\t"
            f"{item.pruned_on_chromosome}\t{item.relationship_markers}"
            for item in manifest.relationship_matrix.chromosomes
        ),
    ]
    if counts_path.read_text(encoding="utf-8").splitlines() != expected_count_lines:
        raise ValueError("marker-count table differs from the frozen chromosome counts")
    return payload


def _read_float32(path: Path, expected_entries: int) -> list[float]:
    if not path.is_file() or path.stat().st_size != expected_entries * 4:
        raise ValueError(f"GRM binary size mismatch: {path}")
    values = array.array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, expected_entries)
    if len(values) != expected_entries or not all(math.isfinite(value) for value in values):
        raise ValueError(f"GRM binary contains the wrong number or non-finite values: {path}")
    return list(values)


def verify_grms(
    manifest_path: str | Path,
    fam_path: str | Path,
    marker_manifest_path: str | Path,
    grm_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_grm_manifest(manifest_source)
    fam_source = Path(fam_path).resolve()
    marker_manifest_source = Path(marker_manifest_path).resolve()
    root = Path(grm_root).resolve()
    if not root.is_dir() or not marker_manifest_source.is_file():
        raise FileNotFoundError("GRM root or marker manifest is missing")
    samples = _read_fam(fam_source, manifest.baseline_parent.post_qc_samples)
    marker_manifest = _verify_marker_manifest(
        manifest, manifest_source, marker_manifest_source
    )
    marker_counts = {
        int(item["chromosome"]): int(item["relationship_markers"])
        for item in marker_manifest["chromosomes"]
    }
    if set(marker_counts) != set(CHROMOSOMES):
        raise ValueError("marker manifest does not cover all six chromosomes")
    triangular_entries = len(samples) * (len(samples) + 1) // 2
    grms = []
    for chromosome in CHROMOSOMES:
        prefix = root / f"leave_chr{chromosome}_out"
        id_path = Path(f"{prefix}.grm.id")
        grm_path = Path(f"{prefix}.grm.bin")
        n_path = Path(f"{prefix}.grm.N.bin")
        ids = []
        for line_number, line in enumerate(id_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"GRM ID line {line_number} must contain FID and IID")
            ids.append((fields[0], fields[1]))
        if ids != samples:
            raise ValueError(f"GRM IDs/order differ from FAM for chromosome {chromosome}")
        grm_values = _read_float32(grm_path, triangular_entries)
        n_values = _read_float32(n_path, triangular_entries)
        relationship_markers = marker_counts[chromosome]
        if any(
            value <= 0
            or value > relationship_markers
            or not math.isclose(value, round(value), abs_tol=1e-3)
            for value in n_values
        ):
            raise ValueError(f"invalid GRM marker-count entries for chromosome {chromosome}")
        diagonal = [
            grm_values[(index + 1) * (index + 2) // 2 - 1]
            for index in range(len(samples))
        ]
        if any(value <= 0 for value in diagonal):
            raise ValueError(f"non-positive GRM diagonal for chromosome {chromosome}")
        matrix = np.empty((len(samples), len(samples)), dtype=np.float64)
        cursor = 0
        for row in range(len(samples)):
            for column in range(row + 1):
                value = grm_values[cursor]
                matrix[row, column] = value
                matrix[column, row] = value
                cursor += 1
        eigenvalues = np.linalg.eigvalsh(matrix)
        spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        psd_tolerance = max(1e-5, spectral_scale * 1e-5)
        minimum_eigenvalue = float(eigenvalues[0])
        maximum_eigenvalue = float(eigenvalues[-1])
        if minimum_eigenvalue < -psd_tolerance:
            raise ValueError(
                f"GRM is not positive semidefinite within float32 tolerance for "
                f"chromosome {chromosome}: {minimum_eigenvalue} < {-psd_tolerance}"
            )
        grms.append(
            {
                "excluded_chromosome": chromosome,
                "samples": len(samples),
                "relationship_markers": relationship_markers,
                "triangular_float32_entries": triangular_entries,
                "grm_bin_sha256": _sha256(grm_path),
                "grm_n_bin_sha256": _sha256(n_path),
                "grm_id_sha256": _sha256(id_path),
                "grm_min": min(grm_values),
                "grm_max": max(grm_values),
                "diagonal_min": min(diagonal),
                "diagonal_max": max(diagonal),
                "pairwise_marker_count_min": int(min(n_values)),
                "pairwise_marker_count_max": int(max(n_values)),
                "minimum_eigenvalue": minimum_eigenvalue,
                "maximum_eigenvalue": maximum_eigenvalue,
                "psd_tolerance": psd_tolerance,
                "positive_semidefinite_within_float32_tolerance": True,
            }
        )
    receipt = {
        "schema_version": "wormctx-abamectin-ws283-grm-verification-1.0",
        "manifest_sha256": _sha256(manifest_source),
        "fam_sha256": _sha256(fam_source),
        "marker_manifest_sha256": _sha256(marker_manifest_source),
        "samples": len(samples),
        "grms": grms,
    }
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"GRM verification output must not exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    _write_json(stage, receipt)
    os.replace(stage, destination)
    return receipt


def _accepted_options(log_text: str, log_path: Path) -> dict[str, str | None]:
    lines = log_text.splitlines()
    option_headers = {"Accepted options:", "Options:"}
    header_indexes = [
        index for index, line in enumerate(lines) if line.strip() in option_headers
    ]
    if len(header_indexes) != 1:
        raise ValueError(f"GCTA log lacks exactly one recognized options block: {log_path}")
    start = header_indexes[0] + 1
    options: dict[str, str | None] = {}
    option_seen = False
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            if option_seen:
                break
            continue
        if not stripped.startswith("--"):
            raise ValueError(f"malformed accepted option in {log_path}: {stripped}")
        option_seen = True
        fields = stripped.split(maxsplit=1)
        name = fields[0]
        if name in options:
            raise ValueError(f"duplicate accepted option in {log_path}: {name}")
        options[name] = fields[1] if len(fields) == 2 else None
    if not options:
        raise ValueError(f"GCTA Accepted options block is empty: {log_path}")
    return options


def _require_suffix(value: str | None, suffix: str, option: str, log_path: Path) -> None:
    if value is None or not value.replace("\\", "/").endswith(suffix):
        raise ValueError(f"{option} has the wrong accepted path in {log_path}")


def _qualify_log_text(log_text: str, log_path: Path) -> list[str]:
    lowered = log_text.lower()
    forbidden = (
        "error:",
        "failed to converge",
        "did not converge",
        "not converged",
    )
    hits = [phrase for phrase in forbidden if phrase in lowered]
    if hits:
        raise ValueError(f"GCTA log contains a failure indicator in {log_path}: {hits}")
    return [
        line.strip()
        for line in log_text.splitlines()
        if "warning" in line.lower() or "boundary" in line.lower()
    ]


def verify_runtime_logs(
    manifest_path: str | Path,
    marker_manifest_path: str | Path,
    grm_root: str | Path,
    chunks_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_grm_manifest(manifest_source)
    marker_manifest_source = Path(marker_manifest_path).resolve()
    grms = Path(grm_root).resolve()
    chunks = Path(chunks_root).resolve()
    if not grms.is_dir() or not chunks.is_dir() or not marker_manifest_source.is_file():
        raise FileNotFoundError("GRM, chunk, or marker-manifest input is missing")
    marker_manifest = _verify_marker_manifest(
        manifest, manifest_source, marker_manifest_source
    )
    marker_rows = {item["chromosome"]: item for item in marker_manifest["chromosomes"]}

    grm_receipts: list[dict[str, Any]] = []
    for chromosome in CHROMOSOMES:
        log_path = grms / f"leave_chr{chromosome}_out.log"
        if not log_path.is_file():
            raise FileNotFoundError(f"missing GRM construction log: {log_path}")
        text = log_path.read_text(encoding="utf-8")
        warnings = _qualify_log_text(text, log_path)
        options = _accepted_options(text, log_path)
        expected_names = {
            "--bfile",
            "--autosome-num",
            "--autosome",
            "--extract",
            "--make-grm",
            "--make-grm-alg",
            "--thread-num",
            "--out",
        }
        if set(options) != expected_names:
            raise ValueError(f"GRM accepted options differ from the contract: {log_path}")
        if options["--autosome"] is not None or options["--make-grm"] is not None:
            raise ValueError(f"flag-only GRM options have unexpected values: {log_path}")
        if (
            options["--autosome-num"] != "6"
            or options["--make-grm-alg"] != "0"
            or options["--thread-num"] != "16"
        ):
            raise ValueError(f"GRM numeric accepted options differ from the contract: {log_path}")
        _require_suffix(
            options["--bfile"],
            f"/{manifest.baseline_parent.run_id}/genotype/abamectin_209_qc",
            "--bfile",
            log_path,
        )
        expected_extract = str(
            marker_manifest_source.parent
            / marker_rows[chromosome]["relationship_list"]
        )
        if options["--extract"] != expected_extract:
            raise ValueError(f"--extract is not the exact complement list in {log_path}")
        expected_prefix = str(grms / f"leave_chr{chromosome}_out")
        if options["--out"] != expected_prefix:
            raise ValueError(f"--out is not the exact GRM prefix in {log_path}")
        marker_count = marker_rows[chromosome]["relationship_markers"]
        if f"{marker_count} SNPs" not in text:
            raise ValueError(f"GRM log lacks the expected marker count in {log_path}")
        if "209 individuals" not in text or "GRM" not in text:
            raise ValueError(f"GRM log lacks sample/computation evidence in {log_path}")
        grm_receipts.append(
            {
                "excluded_chromosome": chromosome,
                "relationship_markers": marker_count,
                "log_sha256": _sha256(log_path),
                "accepted_options": options,
                "warnings": warnings,
            }
        )

    scan_receipts: list[dict[str, Any]] = []
    for model_contract in manifest.association.models:
        for trait in TRAITS:
            trait_slug = TRAIT_SLUGS[trait]
            for chromosome in CHROMOSOMES:
                log_path = chunks / model_contract.id / trait_slug / f"chr{chromosome}.log"
                if not log_path.is_file():
                    raise FileNotFoundError(f"missing association log: {log_path}")
                text = log_path.read_text(encoding="utf-8")
                warnings = _qualify_log_text(text, log_path)
                options = _accepted_options(text, log_path)
                expected_names = {
                    "--mlma",
                    "--bfile",
                    "--extract",
                    "--grm",
                    "--pheno",
                    "--maf",
                    "--autosome-num",
                    "--thread-num",
                    "--out",
                }
                if model_contract.pc_count == 10:
                    expected_names.add("--qcovar")
                if set(options) != expected_names or options["--mlma"] is not None:
                    raise ValueError(
                        f"association accepted options differ from the contract: {log_path}"
                    )
                if (
                    options["--maf"] != "0.05"
                    or options["--autosome-num"] != "6"
                    or options["--thread-num"] != "16"
                ):
                    raise ValueError(
                        f"association numeric accepted options differ from the contract: {log_path}"
                    )
                _require_suffix(
                    options["--bfile"],
                    f"/{manifest.baseline_parent.run_id}/genotype/abamectin_209_qc",
                    "--bfile",
                    log_path,
                )
                _require_suffix(
                    options["--pheno"],
                    f"/{manifest.baseline_parent.run_id}/prepared/cohort/{trait_slug}.phen",
                    "--pheno",
                    log_path,
                )
                expected_extract = str(
                    marker_manifest_source.parent
                    / marker_rows[chromosome]["association_list"]
                )
                if options["--extract"] != expected_extract:
                    raise ValueError(f"--extract is not the exact candidate list in {log_path}")
                expected_grm = str(grms / f"leave_chr{chromosome}_out")
                if options["--grm"] != expected_grm:
                    raise ValueError(f"--grm is not the exact chromosome complement in {log_path}")
                expected_prefix = str(
                    chunks / model_contract.id / trait_slug / f"chr{chromosome}"
                )
                if options["--out"] != expected_prefix:
                    raise ValueError(f"--out is not the exact scan prefix in {log_path}")
                if model_contract.pc_count == 10:
                    _require_suffix(
                        options["--qcovar"],
                        f"/{manifest.calibration_parent.run_id}/genotype/qcovars/pc10.qcovar",
                        "--qcovar",
                        log_path,
                    )
                    if "10 quantitative covariate(s) of 209 individuals are included" not in text:
                        raise ValueError(f"PC10 covariate receipt is absent from {log_path}")
                elif "--qcovar" in text or "quantitative covariate(s)" in text:
                    raise ValueError(f"PC0 log unexpectedly contains a qcovar in {log_path}")
                marker_count = marker_rows[chromosome]["association_markers"]
                if f"Running association tests for {marker_count} SNPs" not in text:
                    raise ValueError(f"association marker-count receipt is absent from {log_path}")
                if text.count("Log-likelihood ratio converged.") != 1:
                    raise ValueError(f"association REML did not converge exactly once in {log_path}")
                if "209 individuals are in common in these files." not in text:
                    raise ValueError(f"common-sample receipt is absent from {log_path}")
                scan_receipts.append(
                    {
                        "model": model_contract.id,
                        "trait": trait,
                        "chromosome": chromosome,
                        "association_markers": marker_count,
                        "log_sha256": _sha256(log_path),
                        "accepted_options": options,
                        "reml_converged": True,
                        "warnings": warnings,
                    }
                )
    if len(grm_receipts) != 6 or len(scan_receipts) != manifest.association.chromosome_scan_count:
        raise ValueError("runtime log inventory differs from six GRMs and 48 scans")
    receipt = {
        "schema_version": "wormctx-abamectin-ws283-grm-runtime-qualification-1.0",
        "manifest_sha256": _sha256(manifest_source),
        "marker_manifest_sha256": _sha256(marker_manifest_source),
        "thread_count": manifest.association.thread_count,
        "all_reml_fits_converged": True,
        "grm_log_count": len(grm_receipts),
        "association_log_count": len(scan_receipts),
        "grms": grm_receipts,
        "association_scans": scan_receipts,
    }
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"runtime qualification output must not exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    _write_json(stage, receipt)
    os.replace(stage, destination)
    return receipt


def assemble_maps(
    manifest_path: str | Path,
    bim_path: str | Path,
    chunks_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_grm_manifest(manifest_source)
    bim_source = Path(bim_path).resolve()
    chunks = Path(chunks_root).resolve()
    if not chunks.is_dir() or not bim_source.is_file():
        raise FileNotFoundError("association chunk root or baseline BIM is missing")
    expected_bim_file = _frozen_file(
        manifest.baseline_parent.required_files,
        "genotype/abamectin_209_qc.bim",
    )
    if (
        bim_source.stat().st_size != expected_bim_file.bytes
        or _sha256(bim_source) != expected_bim_file.sha256
    ):
        raise ValueError("assembly BIM identity differs from the frozen baseline parent")
    bim = _read_bim(bim_source, manifest.baseline_parent.post_qc_markers)
    by_chromosome = {
        chromosome: [marker for marker in bim if marker.chromosome == chromosome]
        for chromosome in CHROMOSOMES
    }
    concatenated = [marker for chromosome in CHROMOSOMES for marker in by_chromosome[chromosome]]
    if concatenated != bim:
        raise ValueError("baseline BIM is not in canonical chromosome 1-6 block order")

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"assembled-map output must not exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    maps = []
    try:
        stage.mkdir()
        for model in MODEL_IDS:
            model_output = stage / model
            model_output.mkdir()
            for trait in TRAITS:
                trait_slug = TRAIT_SLUGS[trait]
                assembled_path = model_output / f"{trait_slug}.ldpruned_loco.mlma"
                chunk_receipts = []
                with assembled_path.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write("\t".join(MLMA_HEADER) + "\n")
                    for chromosome in CHROMOSOMES:
                        chunk = chunks / model / trait_slug / f"chr{chromosome}.mlma"
                        rows = _parse_mlma(chunk, {chromosome})
                        observed = [(row.chromosome, row.snp, row.bp) for row in rows]
                        expected = [
                            (marker.chromosome, marker.snp, marker.bp)
                            for marker in by_chromosome[chromosome]
                        ]
                        if observed != expected:
                            raise ValueError(
                                f"chunk marker identity/order mismatch for "
                                f"{model}/{trait}/chromosome {chromosome}"
                            )
                        with chunk.open("r", encoding="utf-8-sig") as source:
                            header = tuple(source.readline().split())
                            if header != MLMA_HEADER:
                                raise ValueError(f"chunk header mismatch: {chunk}")
                            copied = 0
                            for line_number, line in enumerate(source, 2):
                                fields = line.split()
                                if len(fields) != len(MLMA_HEADER):
                                    raise ValueError(
                                        f"chunk row {line_number} has the wrong field count: {chunk}"
                                    )
                                handle.write("\t".join(fields) + "\n")
                                copied += 1
                        if copied != len(expected):
                            raise ValueError(f"chunk copy count mismatch: {chunk}")
                        chunk_receipts.append(
                            {
                                "chromosome": chromosome,
                                "markers": copied,
                                "relative_path": (
                                    f"{model}/{trait_slug}/chr{chromosome}.mlma"
                                ),
                                "sha256": _sha256(chunk),
                            }
                        )
                assembled_rows = _parse_mlma(assembled_path, set(CHROMOSOMES))
                identity = [
                    (row.chromosome, row.snp, row.bp) for row in assembled_rows
                ]
                expected_identity = [
                    (marker.chromosome, marker.snp, marker.bp) for marker in bim
                ]
                if identity != expected_identity:
                    raise ValueError(f"assembled marker identity/order mismatch for {model}/{trait}")
                maps.append(
                    {
                        "model": model,
                        "trait": trait,
                        "relative_path": f"{model}/{assembled_path.name}",
                        "markers": len(assembled_rows),
                        "bytes": assembled_path.stat().st_size,
                        "sha256": _sha256(assembled_path),
                        "chunks": chunk_receipts,
                    }
                )
        if len(maps) != manifest.association.assembled_map_count:
            raise ValueError("assembled map count differs from the frozen contract")
        receipt = {
            "schema_version": "wormctx-abamectin-ws283-assembled-grm-maps-1.0",
            "manifest_sha256": _sha256(manifest_source),
            "baseline_bim_sha256": _sha256(bim_source),
            "marker_count_per_map": len(bim),
            "map_count": len(maps),
            "chunk_count": sum(len(item["chunks"]) for item in maps),
            "maps": maps,
        }
        if receipt["chunk_count"] != manifest.association.chromosome_scan_count:
            raise ValueError("association chunk count differs from the frozen contract")
        _write_json(stage / "assembled_map_manifest.json", receipt)
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return receipt


def _read_clump_counts(path: Path) -> dict[tuple[str, str], int]:
    if not path.is_file():
        raise FileNotFoundError(f"clump count table is not a file: {path}")
    counts: dict[tuple[str, str], int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["model", "trait", "index_clumps"]:
            raise ValueError("clump count header mismatch")
        for row in reader:
            key = (row["model"], row["trait"])
            if key in counts or key[0] not in MODEL_IDS or key[1] not in TRAITS:
                raise ValueError(f"invalid or duplicate clump row: {key}")
            value = int(row["index_clumps"])
            if value < 0:
                raise ValueError("clump counts cannot be negative")
            counts[key] = value
    expected = {(model, trait) for model in MODEL_IDS for trait in TRAITS}
    if set(counts) != expected:
        raise ValueError("clump table does not cover every GRM model/trait combination")
    return counts


def _read_reference_clump_counts(path: Path) -> dict[tuple[str, str], int]:
    if not path.is_file():
        raise FileNotFoundError(f"reference clump table is not a file: {path}")
    all_counts: dict[tuple[str, str], int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["model", "trait", "index_clumps"]:
            raise ValueError("reference clump count header mismatch")
        for row in reader:
            key = (row["model"], row["trait"])
            if key in all_counts:
                raise ValueError(f"duplicate reference clump row: {key}")
            value = int(row["index_clumps"])
            if value < 0:
                raise ValueError("reference clump counts cannot be negative")
            all_counts[key] = value
    expected_all = {
        (model, trait)
        for model in ("pc0", "pc3", "pc5", "pc10")
        for trait in TRAITS
    }
    if set(all_counts) != expected_all:
        raise ValueError("reference clump table differs from the completed calibration grid")
    return {
        (model, trait): all_counts[(model, trait)]
        for model in REFERENCE_MODEL_IDS
        for trait in TRAITS
    }


def _diagnostic_metrics(
    rows: list[Any],
    trait: str,
    all_intervals: list[Any],
    manifest: GrmSensitivityManifest,
    clump_count: int,
) -> dict[str, Any]:
    trait_intervals = [item for item in all_intervals if item.trait == trait]
    outside_trait = [row for row in rows if not _in_intervals(row, trait_intervals)]
    outside_all = [row for row in rows if not _in_intervals(row, all_intervals)]
    if not outside_trait or not outside_all:
        raise ValueError("published-interval exclusion removed every marker")
    return {
        "marker_count": len(rows),
        "lambda_gc": _lambda_gc(rows),
        "lambda_gc_excluding_trait_intervals": _lambda_gc(outside_trait),
        "lambda_gc_excluding_all_intervals": _lambda_gc(outside_all),
        "bonferroni_marker_count": sum(
            row.p <= manifest.diagnostics.marker_bonferroni_p for row in rows
        ),
        "historical_reference_marker_count": sum(
            row.p <= manifest.diagnostics.historical_effective_test_p_reference_only
            for row in rows
        ),
        "ld_clump_count": clump_count,
        "qq_quantiles": _qq_quantiles(rows, manifest.diagnostics.qq_quantiles),
        "published_intervals": _interval_summary(rows, trait_intervals),
    }


def _interval_rank_comparison(
    reference: list[dict[str, Any]], sensitivity: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    reference_by_id = {item["id"]: item for item in reference}
    sensitivity_by_id = {item["id"]: item for item in sensitivity}
    if set(reference_by_id) != set(sensitivity_by_id):
        raise ValueError("reference and sensitivity interval sets differ")
    comparisons = []
    for interval_id in reference_by_id:
        reference_item = reference_by_id[interval_id]
        sensitivity_item = sensitivity_by_id[interval_id]
        reference_lead = reference_item["lead_marker"]
        sensitivity_lead = sensitivity_item["lead_marker"]
        comparisons.append(
            {
                "id": interval_id,
                "chromosome": reference_item["chromosome"],
                "start": reference_item["start"],
                "end": reference_item["end"],
                "reference_lead": reference_lead,
                "ldpruned_grm_lead": sensitivity_lead,
                "global_rank_delta_ldpruned_minus_reference": None
                if reference_lead is None or sensitivity_lead is None
                else sensitivity_lead["global_rank"] - reference_lead["global_rank"],
            }
        )
    return comparisons


def _assert_matched_map_signature(
    reference_rows: list[Any],
    sensitivity_rows: list[Any],
    label: str,
) -> None:
    if len(reference_rows) != len(sensitivity_rows):
        raise ValueError(f"marker count differs from the matched reference for {label}")
    for index, (reference, sensitivity) in enumerate(
        zip(reference_rows, sensitivity_rows, strict=True), 1
    ):
        if (
            reference.chromosome != sensitivity.chromosome
            or reference.snp != sensitivity.snp
            or reference.bp != sensitivity.bp
            or reference.a1 != sensitivity.a1
            or reference.a2 != sensitivity.a2
            or not math.isclose(
                reference.frequency,
                sensitivity.frequency,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "marker/allele/frequency signature differs from the matched reference "
                f"for {label} at row {index}"
            )


def _map_signature(rows: list[Any]) -> list[tuple[int, str, int, str, str, float]]:
    return [
        (row.chromosome, row.snp, row.bp, row.a1, row.a2, row.frequency)
        for row in rows
    ]


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Abamectin WS283 LD-pruned LOCO-GRM sensitivity",
        "",
        "This is a predeclared post-calibration sensitivity. Biological claims are prohibited.",
        "",
        "| Endpoint | Trait | Reference lambda | New lambda | Reference lambda outside all intervals | New lambda outside all intervals | Delta outside all | New Bonferroni markers | New clumps | Rank Spearman |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in summary["models"]:
        for trait in model["traits"]:
            reference = trait["reference"]
            sensitivity = trait["ldpruned_grm"]
            delta = trait["paired_delta_ldpruned_minus_reference"]
            stability = trait["rank_stability_to_matched_reference"]
            lines.append(
                f"| {model['id']} | {trait['trait']} | {reference['lambda_gc']:.4f} | "
                f"{sensitivity['lambda_gc']:.4f} | "
                f"{reference['lambda_gc_excluding_all_intervals']:.4f} | "
                f"{sensitivity['lambda_gc_excluding_all_intervals']:.4f} | "
                f"{delta['lambda_gc_excluding_all_intervals']:+.4f} | "
                f"{sensitivity['bonferroni_marker_count']} | "
                f"{sensitivity['ld_clump_count']} | "
                f"{stability['deterministic_spearman_p_rank']:.4f} |"
            )
    lines.extend(["", "## Published-interval rank comparison", ""])
    lines.append(
        "| Endpoint | Trait | Interval | Reference lead/rank | LD-pruned-GRM lead/rank | Rank delta |"
    )
    lines.append("|---|---|---|---|---|---:|")
    for model in summary["models"]:
        for trait in model["traits"]:
            for interval in trait["published_interval_rank_comparison"]:
                reference = interval["reference_lead"]
                sensitivity = interval["ldpruned_grm_lead"]
                reference_label = (
                    "none"
                    if reference is None
                    else f"{reference['snp']} / {reference['global_rank']}"
                )
                sensitivity_label = (
                    "none"
                    if sensitivity is None
                    else f"{sensitivity['snp']} / {sensitivity['global_rank']}"
                )
                delta = interval["global_rank_delta_ldpruned_minus_reference"]
                lines.append(
                    f"| {model['id']} | {trait['trait']} | {interval['id']} | "
                    f"{reference_label} | {sensitivity_label} | "
                    f"{delta if delta is not None else 'n/a'} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Both PC endpoints and the GRM construction were fixed before these results. "
            "A favorable lambda, clump count, or known-interval rank cannot select a preferred model. "
            "PC10 is the inherited genome-wide PC endpoint, not a chromosome-specific PC-LOCO analysis. "
            "LD clumps are descriptive units rather than independent loci or haplotypes.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_grm_sensitivity(
    manifest_path: str | Path,
    baseline_manifest_path: str | Path,
    results_root: str | Path,
    calibration_parent: str | Path,
    clump_counts_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_grm_manifest(manifest_source)
    baseline_manifest_source = Path(baseline_manifest_path).resolve()
    baseline_manifest = load_qtl_manifest(baseline_manifest_source)
    results = Path(results_root).resolve()
    calibration = Path(calibration_parent).resolve()
    if not results.is_dir() or not calibration.is_dir():
        raise FileNotFoundError("assembled results root or calibration parent is missing")
    if results.name != "assembled" or results.parent.name != "association":
        raise ValueError("results root must be the run's association/assembled directory")
    run_root = results.parent.parent
    marker_manifest_source = run_root / "genotype/marker_lists/marker_manifest.json"
    grm_verification_source = run_root / "receipts/grm_verification.json"
    runtime_qualification_source = run_root / "receipts/runtime_qualification.json"
    assembled_manifest_source = results / "assembled_map_manifest.json"
    for evidence_path in (
        marker_manifest_source,
        grm_verification_source,
        runtime_qualification_source,
        assembled_manifest_source,
    ):
        if not evidence_path.is_file():
            raise FileNotFoundError(f"required qualification evidence is missing: {evidence_path}")
    _verify_marker_manifest(manifest, manifest_source, marker_manifest_source)
    marker_manifest_hash = _sha256(marker_manifest_source)
    manifest_hash = _sha256(manifest_source)
    grm_verification = json.loads(grm_verification_source.read_text(encoding="utf-8"))
    if (
        grm_verification.get("schema_version")
        != "wormctx-abamectin-ws283-grm-verification-1.0"
        or grm_verification.get("manifest_sha256") != manifest_hash
        or grm_verification.get("marker_manifest_sha256") != marker_manifest_hash
        or grm_verification.get("samples") != manifest.baseline_parent.post_qc_samples
        or len(grm_verification.get("grms", [])) != len(CHROMOSOMES)
        or [item.get("excluded_chromosome") for item in grm_verification.get("grms", [])]
        != list(CHROMOSOMES)
        or any(
            item.get("positive_semidefinite_within_float32_tolerance") is not True
            for item in grm_verification.get("grms", [])
        )
    ):
        raise ValueError("GRM verification receipt is not fully qualified")
    runtime_qualification = json.loads(
        runtime_qualification_source.read_text(encoding="utf-8")
    )
    if (
        runtime_qualification.get("schema_version")
        != "wormctx-abamectin-ws283-grm-runtime-qualification-1.0"
        or runtime_qualification.get("manifest_sha256") != manifest_hash
        or runtime_qualification.get("marker_manifest_sha256") != marker_manifest_hash
        or runtime_qualification.get("thread_count") != manifest.association.thread_count
        or runtime_qualification.get("grm_log_count") != len(CHROMOSOMES)
        or runtime_qualification.get("association_log_count")
        != manifest.association.chromosome_scan_count
        or runtime_qualification.get("all_reml_fits_converged") is not True
        or any(
            item.get("reml_converged") is not True
            for item in runtime_qualification.get("association_scans", [])
        )
    ):
        raise ValueError("runtime qualification receipt is not fully qualified")
    assembled_manifest = json.loads(
        assembled_manifest_source.read_text(encoding="utf-8")
    )
    expected_bim_hash = _frozen_file(
        manifest.baseline_parent.required_files,
        "genotype/abamectin_209_qc.bim",
    ).sha256
    if (
        assembled_manifest.get("schema_version")
        != "wormctx-abamectin-ws283-assembled-grm-maps-1.0"
        or assembled_manifest.get("manifest_sha256") != manifest_hash
        or assembled_manifest.get("baseline_bim_sha256") != expected_bim_hash
        or assembled_manifest.get("marker_count_per_map")
        != manifest.baseline_parent.post_qc_markers
        or assembled_manifest.get("map_count") != manifest.association.assembled_map_count
        or assembled_manifest.get("chunk_count")
        != manifest.association.chromosome_scan_count
        or len(assembled_manifest.get("maps", []))
        != manifest.association.assembled_map_count
    ):
        raise ValueError("assembled-map receipt differs from the frozen contract")
    clump_source = Path(clump_counts_path).resolve()
    sensitivity_clumps = _read_clump_counts(clump_source)
    reference_clump_source = calibration / "summary/ld_clump_counts.tsv"
    reference_clumps = _read_reference_clump_counts(reference_clump_source)
    all_intervals = list(baseline_manifest.published_intervals)

    common_signature: list[tuple[int, str, int, str, str, float]] | None = None
    models = []
    for model_contract in manifest.association.models:
        traits = []
        for trait in TRAITS:
            trait_slug = TRAIT_SLUGS[trait]
            sensitivity_path = (
                results / model_contract.id / f"{trait_slug}.ldpruned_loco.mlma"
            )
            reference_path = (
                calibration
                / "association"
                / model_contract.reference_model
                / f"{trait_slug}.loco.mlma"
            )
            sensitivity_rows = _parse_mlma(sensitivity_path, set(CHROMOSOMES))
            reference_rows = _parse_mlma(reference_path, set(CHROMOSOMES))
            _assert_matched_map_signature(
                reference_rows,
                sensitivity_rows,
                f"{model_contract.id}/{trait}",
            )
            sensitivity_signature = _map_signature(sensitivity_rows)
            if common_signature is None:
                common_signature = sensitivity_signature
            elif sensitivity_signature != common_signature:
                raise ValueError("marker/allele/frequency signature differs across assembled maps")
            if len(sensitivity_rows) != manifest.baseline_parent.post_qc_markers:
                raise ValueError(f"marker count differs for {model_contract.id}/{trait}")

            reference_metrics = _diagnostic_metrics(
                reference_rows,
                trait,
                all_intervals,
                manifest,
                reference_clumps[(model_contract.reference_model, trait)],
            )
            sensitivity_metrics = _diagnostic_metrics(
                sensitivity_rows,
                trait,
                all_intervals,
                manifest,
                sensitivity_clumps[(model_contract.id, trait)],
            )
            reference_metrics["result_sha256"] = _sha256(reference_path)
            sensitivity_metrics["result_sha256"] = _sha256(sensitivity_path)
            traits.append(
                {
                    "trait": trait,
                    "reference": reference_metrics,
                    "ldpruned_grm": sensitivity_metrics,
                    "paired_delta_ldpruned_minus_reference": {
                        "lambda_gc": sensitivity_metrics["lambda_gc"]
                        - reference_metrics["lambda_gc"],
                        "lambda_gc_excluding_trait_intervals": sensitivity_metrics[
                            "lambda_gc_excluding_trait_intervals"
                        ]
                        - reference_metrics["lambda_gc_excluding_trait_intervals"],
                        "lambda_gc_excluding_all_intervals": sensitivity_metrics[
                            "lambda_gc_excluding_all_intervals"
                        ]
                        - reference_metrics["lambda_gc_excluding_all_intervals"],
                        "bonferroni_marker_count": sensitivity_metrics[
                            "bonferroni_marker_count"
                        ]
                        - reference_metrics["bonferroni_marker_count"],
                        "ld_clump_count": sensitivity_metrics["ld_clump_count"]
                        - reference_metrics["ld_clump_count"],
                    },
                    "rank_stability_to_matched_reference": {
                        "deterministic_spearman_p_rank": _deterministic_spearman(
                            reference_rows, sensitivity_rows
                        ),
                        "top_sets": [
                            _top_overlap(reference_rows, sensitivity_rows, count)
                            for count in manifest.diagnostics.top_set_sizes
                        ],
                    },
                    "published_interval_rank_comparison": _interval_rank_comparison(
                        reference_metrics["published_intervals"],
                        sensitivity_metrics["published_intervals"],
                    ),
                }
            )
        models.append(
            {
                "id": model_contract.id,
                "pc_count": model_contract.pc_count,
                "reference_model": model_contract.reference_model,
                "traits": traits,
            }
        )

    summary: dict[str, Any] = {
        "schema_version": SUMMARY_VERSION,
        "analysis_id": manifest.analysis_id,
        "classification": manifest.classification,
        "status": manifest.status,
        "validated": False,
        "biological_claims_permitted": False,
        "manifest_sha256": _sha256(manifest_source),
        "baseline_manifest_sha256": _sha256(baseline_manifest_source),
        "calibration_parent": str(calibration),
        "calibration_summary_sha256": _sha256(
            calibration
            / "summary/ws283_pc_calibration/ws283_pc_calibration_summary.json"
        ),
        "sensitivity_clump_counts_sha256": _sha256(clump_source),
        "reference_clump_counts_sha256": _sha256(reference_clump_source),
        "qualification_evidence": {
            "marker_manifest_sha256": marker_manifest_hash,
            "grm_verification_sha256": _sha256(grm_verification_source),
            "runtime_qualification_sha256": _sha256(runtime_qualification_source),
            "assembled_map_manifest_sha256": _sha256(assembled_manifest_source),
            "all_six_grms_numerically_qualified": True,
            "all_48_scans_runtime_qualified": True,
            "all_48_reml_fits_converged": True,
            "all_eight_maps_match_reference_marker_allele_frequency_signatures": True,
        },
        "sample_count": manifest.baseline_parent.post_qc_samples,
        "marker_count": manifest.baseline_parent.post_qc_markers,
        "relationship_matrix": {
            "source_ld_pruned_markers": manifest.relationship_matrix.total_ld_pruned_markers,
            "leave_one_chromosome_out": True,
            "algorithm": manifest.relationship_matrix.method,
            "chromosomes": [
                item.model_dump(mode="json")
                for item in manifest.relationship_matrix.chromosomes
            ],
        },
        "models": models,
        "claims": manifest.claims.model_dump(mode="json"),
        "limitations": {
            "post_calibration_sensitivity_not_model_selection": True,
            "pc10_is_genomewide_not_pc_loco": True,
            "smallest_complement_grm_marker_count": min(
                item.relationship_markers
                for item in manifest.relationship_matrix.chromosomes
            ),
            "gcta_default_grm_algorithm_0_not_inbred_estimator": True,
            "haplotype_analysis_performed": False,
            "causal_or_mechanistic_interpretation_permitted": False,
        },
    }
    json.dumps(summary, allow_nan=False)
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"GRM-sensitivity summary output must not exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        _write_json(stage / "ws283_ldpruned_grm_sensitivity_summary.json", summary)
        (stage / "ws283_ldpruned_grm_sensitivity_summary.md").write_text(
            _markdown(summary), encoding="utf-8", newline="\n"
        )
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return summary


def _parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="command", required=True)

    verify = subcommands.add_parser("verify-parents")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--baseline-parent", type=Path, required=True)
    verify.add_argument("--calibration-parent", type=Path, required=True)

    markers = subcommands.add_parser("prepare-marker-lists")
    markers.add_argument("--manifest", type=Path, required=True)
    markers.add_argument("--bim", type=Path, required=True)
    markers.add_argument("--prune-in", type=Path, required=True)
    markers.add_argument("--output", type=Path, required=True)

    grms = subcommands.add_parser("verify-grms")
    grms.add_argument("--manifest", type=Path, required=True)
    grms.add_argument("--fam", type=Path, required=True)
    grms.add_argument("--marker-manifest", type=Path, required=True)
    grms.add_argument("--grm-root", type=Path, required=True)
    grms.add_argument("--output", type=Path, required=True)

    runtime = subcommands.add_parser("verify-runtime-logs")
    runtime.add_argument("--manifest", type=Path, required=True)
    runtime.add_argument("--marker-manifest", type=Path, required=True)
    runtime.add_argument("--grm-root", type=Path, required=True)
    runtime.add_argument("--chunks-root", type=Path, required=True)
    runtime.add_argument("--output", type=Path, required=True)

    assemble = subcommands.add_parser("assemble-maps")
    assemble.add_argument("--manifest", type=Path, required=True)
    assemble.add_argument("--bim", type=Path, required=True)
    assemble.add_argument("--chunks-root", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)

    summarize = subcommands.add_parser("summarize")
    summarize.add_argument("--manifest", type=Path, required=True)
    summarize.add_argument("--baseline-manifest", type=Path, required=True)
    summarize.add_argument("--results-root", type=Path, required=True)
    summarize.add_argument("--calibration-parent", type=Path, required=True)
    summarize.add_argument("--clump-counts", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    return command


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify-parents":
            result = verify_parents(
                args.manifest, args.baseline_parent, args.calibration_parent
            )
        elif args.command == "prepare-marker-lists":
            result = prepare_marker_lists(
                args.manifest, args.bim, args.prune_in, args.output
            )
        elif args.command == "verify-grms":
            result = verify_grms(
                args.manifest,
                args.fam,
                args.marker_manifest,
                args.grm_root,
                args.output,
            )
        elif args.command == "verify-runtime-logs":
            result = verify_runtime_logs(
                args.manifest,
                args.marker_manifest,
                args.grm_root,
                args.chunks_root,
                args.output,
            )
        elif args.command == "assemble-maps":
            result = assemble_maps(args.manifest, args.bim, args.chunks_root, args.output)
        elif args.command == "summarize":
            result = summarize_grm_sensitivity(
                args.manifest,
                args.baseline_manifest,
                args.results_root,
                args.calibration_parent,
                args.clump_counts,
                args.output,
            )
        else:  # pragma: no cover - argparse enforces this
            raise ValueError(f"unsupported command: {args.command}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
