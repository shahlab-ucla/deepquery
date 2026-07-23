"""Outcome-blind WS283 regional small-variant haplotype sensitivity.

This lane is deliberately distinct from the genuine regional-state qualification pangenome/PAV contract.  It
clusters LD-pruned small-variant genotype profiles inside two frozen chromosome-V
regions without accepting phenotype paths, then performs a descriptive kinship-aware
regional omnibus sensitivity with restricted residual-coordinate permutations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .caendr_compendium_203_panel import _kmeans, _silhouette
from .qtl import TRAITS, TRAIT_SLUGS
from .qtl_parametric_null import (
    _read_design_matrix,
    _read_ordered_phenotype,
    read_gcta_grm,
)


SCHEMA_VERSION = "wormctx-abamectin-ws283-chrv-vcf-haplotype-sensitivity-1.0"
PANEL_VERSION = "wormctx-ws283-chrv-vcf-haplotype-panel-1.0"
RESULT_VERSION = "wormctx-ws283-chrv-vcf-haplotype-sensitivity-result-1.0"
EXPECTED_REGIONS = {
    "chrv_left": (5, 1_747_612, 4_333_001),
    "chrv_right": (5, 13_606_517, 16_754_986),
}
RELATIONSHIPS = ("full_chr5_excluded", "ldpruned_chr5_excluded")
ENDPOINTS = ("pc0", "pc10")
_SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FileIdentity(_Strict):
    bytes: int = Field(gt=0)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("SHA-256 must be lowercase hexadecimal")
        return value


class CohortContract(_Strict):
    samples: Literal[209]
    sample_order: Literal["baseline_parent_fam_order"]
    fam_sha256: str
    ordered_iid_sha256: str

    @field_validator("fam_sha256", "ordered_iid_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("cohort SHA-256 must be lowercase hexadecimal")
        return value


class BfileContract(_Strict):
    bed: FileIdentity
    bim: FileIdentity
    fam: FileIdentity
    markers: Literal[373279]
    chromosome_v_markers: Literal[115183]


class UpstreamVcf(_Strict):
    relative_path: Literal["vcf/WI.20250625.impute.isotype.vcf.gz"]
    bytes: Literal[131250981]
    sha256: str
    role: Literal["provenance_only_not_read_by_this_runner"]

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("VCF SHA-256 must be lowercase hexadecimal")
        return value


class CoordinateReceipt(_Strict):
    relative_path: Literal[
        "artifacts/upstream/reference_coordinate_identity/qualification.json"
    ]
    sha256: str

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("coordinate receipt SHA-256 must be lowercase hexadecimal")
        return value


class SourceContract(_Strict):
    upstream_ws283_imputed_vcf: UpstreamVcf
    vcf_derived_qc_bfile: BfileContract
    phenotypes: dict[str, FileIdentity]
    pc10_qcovar: FileIdentity
    coordinate_identity_receipt: CoordinateReceipt

    @model_validator(mode="after")
    def exact_phenotypes(self) -> "SourceContract":
        if list(self.phenotypes) != [TRAIT_SLUGS[item] for item in TRAITS]:
            raise ValueError("phenotype identities differ from the frozen trait order")
        return self


class RegionContract(_Strict):
    id: Literal["chrv_left", "chrv_right"]
    chromosome: Literal[5]
    start: int = Field(gt=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def exact_interval(self) -> "RegionContract":
        if (self.chromosome, self.start, self.end) != EXPECTED_REGIONS[self.id]:
            raise ValueError("region differs from the frozen chromosome-V interval")
        return self


class DerivationContract(_Strict):
    phenotype_paths_accepted: Literal[False]
    outcome_blind: Literal[True]
    input_kind: Literal[
        "imputed_small_variant_genotypes_not_pangenome_paths_or_pav"
    ]
    plink2_maf: Literal[0.05]
    plink2_geno: Literal[0.05]
    ld_pruning: Literal["indep-pairwise 50 5 0.8"]
    regional_representation: Literal[
        "first_10_svd_scores_of_standardized_pruned_dosages"
    ]
    candidate_cluster_k: list[int]
    cluster_algorithm: Literal["deterministic_farthest_first_kmeans"]
    cluster_selection: Literal["maximum_sample_silhouette_then_lower_k"]
    cluster_label_canonicalization: Literal["lexicographic_minimum_member_iid"]
    minimum_carriers_per_state: Literal[5]
    missing_dosage_policy: Literal["marker_mean_imputation_recorded"]
    reference_state_policy: Literal[
        "largest_carrier_count_then_lexicographic_state"
    ]
    one_state_per_sample_per_region: Literal[True]

    @model_validator(mode="after")
    def exact_grid(self) -> "DerivationContract":
        if self.candidate_cluster_k != [2, 3, 4, 5, 6, 7, 8]:
            raise ValueError("candidate cluster grid differs from the frozen contract")
        return self


class RelationshipContract(_Strict):
    id: Literal["full_chr5_excluded", "ldpruned_chr5_excluded"]
    expected_markers: int = Field(gt=0)
    grm_bin_sha256: str
    grm_n_bin_sha256: str
    grm_id_sha256: str

    @field_validator("grm_bin_sha256", "grm_n_bin_sha256", "grm_id_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("GRM SHA-256 must be lowercase hexadecimal")
        return value


class AnalysisContract(_Strict):
    traits: list[str]
    endpoints: list[Literal["pc0", "pc10"]]
    cells: Literal[16]
    null_covariance: Literal["profile_REML_one_GRM_under_covariate_only_null"]
    test: Literal["fixed_covariance_GLS_regional_state_omnibus"]
    calibration: Literal[
        "Abney_2015_restricted_whitened_residual_coordinate_permutation"
    ]
    permutations_per_cell: Literal[2000]
    within_cell_family: Literal[
        "Westfall_Young_single_step_minP_across_two_regions"
    ]
    seed: Literal[20260723]
    reestimate_covariance_per_permutation: Literal[False]
    report_boundary_h2_without_replacing_endpoint: Literal[True]
    partial_results_evidentiary: Literal[False]

    @model_validator(mode="after")
    def exact_grid(self) -> "AnalysisContract":
        if self.traits != list(TRAITS) or self.endpoints != list(ENDPOINTS):
            raise ValueError("analysis cells differ from the frozen grid")
        return self


class ExecutionContract(_Strict):
    restricted_residual_bootstrap_terminal_success_required_before_heavy_launch: Literal[True]
    run_write_once: Literal[True]
    atomic_receipts: Literal[True]
    terminal_checksum_closure: Literal[True]
    thread_cap_after_restricted_residual_bootstrap: Literal[8]
    gpu_required: Literal[False]


class ClaimContract(_Strict):
    permitted: list[str]
    prohibited: list[str]


class Manifest(_Strict):
    schema_version: Literal[SCHEMA_VERSION]
    analysis_id: Literal["abamectin_ws283_chrv_vcf_haplotype_sensitivity_20260723"]
    classification: Literal[
        "predeclared_descriptive_small_variant_haplotype_sensitivity"
    ]
    status: Literal["frozen_before_state_derivation_and_outcome_analysis"]
    validated: Literal[False]
    biological_claims_permitted: Literal[False]
    cohort: CohortContract
    source: SourceContract
    regions: list[RegionContract]
    derivation: DerivationContract
    relationships: list[RelationshipContract]
    analysis: AnalysisContract
    execution: ExecutionContract
    claims: ClaimContract

    @model_validator(mode="after")
    def exact_contract(self) -> "Manifest":
        if [item.id for item in self.regions] != list(EXPECTED_REGIONS):
            raise ValueError("regions differ from the frozen order")
        if [item.id for item in self.relationships] != list(RELATIONSHIPS):
            raise ValueError("relationship controls differ from the frozen order")
        if [item.expected_markers for item in self.relationships] != [258096, 1080]:
            raise ValueError("relationship marker counts differ from the handoff")
        prohibited = " ".join(self.claims.prohibited).lower()
        for token in ("pangenome", "presence-absence", "causal", "published interval"):
            if token not in prohibited:
                raise ValueError(f"claim boundary must prohibit {token}")
        return self


def load_manifest(path: str | Path) -> Manifest:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"manifest is not a file: {source}")
    return Manifest.model_validate_json(source.read_text(encoding="utf-8"))


def _verify(path: Path, identity: FileIdentity) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"input is not a regular file: {path}")
    observed_bytes = path.stat().st_size
    observed_hash = _sha256(path)
    if observed_bytes != identity.bytes or observed_hash != identity.sha256:
        raise ValueError(f"input identity mismatch: {path}")
    return {"path": str(path), "bytes": observed_bytes, "sha256": observed_hash}


def _read_fam(path: Path, manifest: Manifest) -> list[tuple[str, str]]:
    _verify(path, manifest.source.vcf_derived_qc_bfile.fam)
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"FAM row {line_number} must contain six fields")
        rows.append((fields[0], fields[1]))
    if len(rows) != manifest.cohort.samples or len({iid for _, iid in rows}) != len(rows):
        raise ValueError("FAM does not contain the exact unique cohort")
    order_hash = hashlib.sha256(
        ("\n".join(iid for _, iid in rows) + "\n").encode("utf-8")
    ).hexdigest()
    if order_hash != manifest.cohort.ordered_iid_sha256:
        raise ValueError("FAM IID order differs from the frozen cohort")
    return rows


def _read_marker_list(path: Path) -> list[str]:
    markers = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not markers or any(not item for item in markers) or len(markers) != len(set(markers)):
        raise ValueError("pruned marker list is empty, duplicated, or malformed")
    return markers


def _read_plink_raw(path: Path, expected_iids: Sequence[str]) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) != len(expected_iids) + 1:
        raise ValueError("PLINK additive export row count differs from the cohort")
    header = lines[0].split()
    if len(header) < 8 or header[1] != "IID":
        raise ValueError("PLINK additive export header is malformed")
    marker_columns = header[6:]
    values = np.empty((len(expected_iids), len(marker_columns)), dtype=np.float64)
    for row_index, (line, expected_iid) in enumerate(
        zip(lines[1:], expected_iids, strict=True)
    ):
        fields = line.split()
        if len(fields) != len(header) or fields[1] != expected_iid:
            raise ValueError(f"PLINK additive export IID/order mismatch at row {row_index + 2}")
        for column_index, value in enumerate(fields[6:]):
            if value in {"NA", "nan", "."}:
                values[row_index, column_index] = np.nan
            else:
                try:
                    values[row_index, column_index] = float(value)
                except ValueError as error:
                    raise ValueError("PLINK additive export contains a nonnumeric dosage") from error
    if np.any(np.isinf(values)):
        raise ValueError("PLINK additive export contains infinite dosage")
    return marker_columns, values


@dataclass(frozen=True)
class RegionDerivation:
    region_id: str
    state_ids: tuple[str, ...]
    receipt: dict[str, Any]


def derive_region_states(
    region_id: str,
    iids: Sequence[str],
    dosage: np.ndarray,
    candidate_k: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
    minimum_carriers: int = 5,
) -> RegionDerivation:
    """Derive deterministic genotype-profile clusters without phenotype input."""

    if region_id not in EXPECTED_REGIONS:
        raise ValueError("unknown region")
    matrix = np.asarray(dosage, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(iids) or matrix.shape[1] < 2:
        raise ValueError("regional dosage matrix has the wrong shape")
    if len(iids) != len(set(iids)) or np.any(np.isinf(matrix)):
        raise ValueError("regional dosage cohort is invalid")
    missing = int(np.count_nonzero(np.isnan(matrix)))
    means = np.nanmean(matrix, axis=0)
    if np.any(~np.isfinite(means)):
        raise ValueError("a regional marker is missing in every sample")
    filled = np.where(np.isnan(matrix), means[None, :], matrix)
    standard_deviations = np.std(filled, axis=0, ddof=0)
    keep = standard_deviations > 1e-12
    if int(np.count_nonzero(keep)) < 2:
        raise ValueError("fewer than two nonconstant regional markers remain")
    standardized = (filled[:, keep] - np.mean(filled[:, keep], axis=0)) / standard_deviations[keep]
    u, singular_values, _ = np.linalg.svd(standardized, full_matrices=False)
    tolerance = max(standardized.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    components = min(10, rank)
    if components < 2:
        raise ValueError("regional genotype representation has rank below two")
    scores = u[:, :components] * singular_values[:components]
    candidates: list[dict[str, Any]] = []
    selected_assignments: np.ndarray | None = None
    selected_score = -math.inf
    selected_k = 0
    for k in candidate_k:
        try:
            assignments = _kmeans(scores, list(iids), int(k))
            counts = np.bincount(assignments, minlength=int(k))
            eligible = bool(np.min(counts) >= minimum_carriers)
            score = float(_silhouette(scores, assignments)) if eligible else None
            reason = None if eligible else "cluster_below_minimum_carriers"
        except ValueError as error:
            assignments = None
            counts = np.asarray([], dtype=int)
            eligible = False
            score = None
            reason = str(error)
        candidates.append(
            {
                "k": int(k),
                "eligible": eligible,
                "carrier_counts": [int(value) for value in counts],
                "sample_silhouette": score,
                "exclusion_reason": reason,
            }
        )
        if eligible and score is not None and (
            score > selected_score + 1e-15
            or (math.isclose(score, selected_score, abs_tol=1e-15) and int(k) < selected_k)
        ):
            selected_score = score
            selected_k = int(k)
            selected_assignments = assignments
    if selected_assignments is None:
        raise ValueError("no regional cluster solution passes the frozen carrier gate")
    state_ids = tuple(
        f"{region_id}:cluster_{int(label) + 1:02d}" for label in selected_assignments
    )
    counts = {state: state_ids.count(state) for state in sorted(set(state_ids))}
    receipt = {
        "region_id": region_id,
        "input_markers": int(matrix.shape[1]),
        "nonconstant_markers": int(np.count_nonzero(keep)),
        "missing_dosages_mean_imputed": missing,
        "svd_rank": rank,
        "svd_components": components,
        "selected_k": selected_k,
        "selected_sample_silhouette": selected_score,
        "state_carrier_counts": counts,
        "candidates": candidates,
    }
    return RegionDerivation(region_id=region_id, state_ids=state_ids, receipt=receipt)


def derive_panel(
    manifest_path: str | Path,
    fam_path: str | Path,
    coordinate_receipt_path: str | Path,
    left_raw_path: str | Path,
    left_markers_path: str | Path,
    right_raw_path: str | Path,
    right_markers_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_source)
    fam = Path(fam_path).resolve()
    rows = _read_fam(fam, manifest)
    iids = [iid for _, iid in rows]
    coordinate = Path(coordinate_receipt_path).resolve()
    if _sha256(coordinate) != manifest.source.coordinate_identity_receipt.sha256:
        raise ValueError("coordinate identity receipt differs from the frozen contract")
    inputs = {
        "chrv_left": (Path(left_raw_path).resolve(), Path(left_markers_path).resolve()),
        "chrv_right": (Path(right_raw_path).resolve(), Path(right_markers_path).resolve()),
    }
    derived: list[RegionDerivation] = []
    input_receipts: dict[str, Any] = {}
    for region_id, (raw_path, marker_path) in inputs.items():
        markers = _read_marker_list(marker_path)
        raw_markers, dosage = _read_plink_raw(raw_path, iids)
        if len(raw_markers) != len(markers):
            raise ValueError(f"{region_id} PLINK export and prune-list counts differ")
        item = derive_region_states(
            region_id,
            iids,
            dosage,
            manifest.derivation.candidate_cluster_k,
            manifest.derivation.minimum_carriers_per_state,
        )
        derived.append(item)
        input_receipts[region_id] = {
            "additive_export": {"bytes": raw_path.stat().st_size, "sha256": _sha256(raw_path)},
            "pruned_marker_list": {
                "bytes": marker_path.stat().st_size,
                "sha256": _sha256(marker_path),
                "markers": len(markers),
            },
        }
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite panel directory: {root}")
    root.mkdir(parents=True)
    panel_path = root / "regional_states.tsv"
    lines = ["FID\tIID\tregion_id\tstate_id"]
    for (fid, iid), index in zip(rows, range(len(rows)), strict=True):
        for item in derived:
            lines.append(f"{fid}\t{iid}\t{item.region_id}\t{item.state_ids[index]}")
    _write_new(panel_path, ("\n".join(lines) + "\n").encode("utf-8"))
    receipt = {
        "schema_version": PANEL_VERSION,
        "analysis_id": manifest.analysis_id,
        "classification": manifest.classification,
        "manifest_sha256": _sha256(manifest_source),
        "coordinate_identity_receipt_sha256": _sha256(coordinate),
        "fam_sha256": _sha256(fam),
        "ordered_iid_sha256": manifest.cohort.ordered_iid_sha256,
        "samples": len(rows),
        "regions": [item.receipt for item in derived],
        "inputs": input_receipts,
        "panel": {
            "relative_path": panel_path.name,
            "bytes": panel_path.stat().st_size,
            "sha256": _sha256(panel_path),
            "rows": len(rows) * len(derived),
        },
        "phenotype_paths_accepted": False,
        "phenotype_values_accessed": False,
        "pangenome_or_pav_claim_permitted": False,
        "qualified_for_descriptive_vcf_haplotype_sensitivity": True,
        "biological_claims_permitted": False,
    }
    receipt_path = root / "panel_qualification.json"
    _write_new(receipt_path, _canonical_json(receipt))
    return {**receipt, "receipt_sha256": _sha256(receipt_path)}


def _read_panel(
    path: Path, receipt_path: Path, manifest: Manifest, rows: Sequence[tuple[str, str]]
) -> dict[str, tuple[str, ...]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != PANEL_VERSION
        or receipt.get("analysis_id") != manifest.analysis_id
        or receipt.get("samples") != len(rows)
        or receipt.get("qualified_for_descriptive_vcf_haplotype_sensitivity") is not True
        or receipt.get("phenotype_values_accessed") is not False
        or receipt.get("panel", {}).get("sha256") != _sha256(path)
    ):
        raise ValueError("state-panel qualification receipt is invalid")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "FID\tIID\tregion_id\tstate_id":
        raise ValueError("state panel header is invalid")
    by_region: dict[str, list[str]] = {item: [] for item in EXPECTED_REGIONS}
    cursor = 1
    for fid, iid in rows:
        for region_id in EXPECTED_REGIONS:
            fields = lines[cursor].split("\t") if cursor < len(lines) else []
            cursor += 1
            if len(fields) != 4 or fields[:3] != [fid, iid, region_id]:
                raise ValueError("state panel order differs from FAM/region order")
            by_region[region_id].append(fields[3])
    if cursor != len(lines):
        raise ValueError("state panel contains unexpected rows")
    for states in by_region.values():
        counts = {state: states.count(state) for state in set(states)}
        if len(counts) < 2 or min(counts.values()) < manifest.derivation.minimum_carriers_per_state:
            raise ValueError("state panel no longer passes the carrier gate")
    return {key: tuple(value) for key, value in by_region.items()}


@dataclass(frozen=True)
class ProfileReml:
    h2: float
    sigma2: float
    variance_genetic: float
    variance_residual: float
    objective: float
    covariance: np.ndarray


def profile_reml(
    y: np.ndarray,
    design: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> ProfileReml:
    """Profile one-GRM REML over h2 using the GRM eigenbasis."""

    vector = np.asarray(y, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    n = len(vector)
    if (
        x.ndim != 2
        or x.shape[0] != n
        or vectors.shape != (n, n)
        or values.shape != (n,)
        or np.linalg.matrix_rank(x) != x.shape[1]
    ):
        raise ValueError("profile REML inputs are invalid")
    y_e = vectors.T @ vector
    x_e = vectors.T @ x
    residual_df = n - x.shape[1]

    def evaluate(h2: float) -> tuple[float, float]:
        diagonal = h2 * values + (1.0 - h2)
        if np.any(diagonal <= 0.0):
            return math.inf, math.nan
        weighted_x = x_e / diagonal[:, None]
        xtwx = x_e.T @ weighted_x
        sign, logdet_x = np.linalg.slogdet(xtwx)
        if sign <= 0.0:
            return math.inf, math.nan
        beta = np.linalg.solve(xtwx, weighted_x.T @ y_e)
        residual = y_e - x_e @ beta
        quadratic = float(np.sum(residual * residual / diagonal))
        if quadratic <= 0.0:
            return math.inf, math.nan
        sigma2 = quadratic / residual_df
        objective = (
            float(np.sum(np.log(diagonal)))
            + float(logdet_x)
            + residual_df * math.log(sigma2)
        )
        return objective, sigma2

    grid = np.linspace(0.0, 0.99, 100)
    objectives = np.asarray([evaluate(float(item))[0] for item in grid])
    best_index = int(np.argmin(objectives))
    best_h2 = float(grid[best_index])
    best_objective, best_sigma2 = evaluate(best_h2)
    if 0 < best_index < len(grid) - 1:
        left = float(grid[best_index - 1])
        right = float(grid[best_index + 1])
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        c = right - ratio * (right - left)
        d = left + ratio * (right - left)
        fc = evaluate(c)[0]
        fd = evaluate(d)[0]
        for _ in range(60):
            if fc <= fd:
                right, d, fd = d, c, fc
                c = right - ratio * (right - left)
                fc = evaluate(c)[0]
            else:
                left, c, fc = c, d, fd
                d = left + ratio * (right - left)
                fd = evaluate(d)[0]
        candidate = (left + right) / 2.0
        candidate_objective, candidate_sigma2 = evaluate(candidate)
        if candidate_objective < best_objective:
            best_h2 = candidate
            best_objective = candidate_objective
            best_sigma2 = candidate_sigma2
    covariance = best_sigma2 * (
        best_h2 * (vectors @ (values[:, None] * vectors.T))
        + (1.0 - best_h2) * np.eye(n)
    )
    covariance = (covariance + covariance.T) / 2.0
    if float(np.linalg.eigvalsh(covariance)[0]) <= 0.0:
        raise ValueError("profile REML covariance is not positive definite")
    return ProfileReml(
        h2=best_h2,
        sigma2=best_sigma2,
        variance_genetic=best_sigma2 * best_h2,
        variance_residual=best_sigma2 * (1.0 - best_h2),
        objective=best_objective,
        covariance=covariance,
    )


def _state_design(states: Sequence[str]) -> tuple[np.ndarray, str, dict[str, int]]:
    counts = {state: states.count(state) for state in sorted(set(states))}
    reference = sorted(counts, key=lambda state: (-counts[state], state))[0]
    alternatives = [state for state in sorted(counts) if state != reference]
    matrix = np.asarray(
        [[1.0 if state == alternative else 0.0 for alternative in alternatives] for state in states],
        dtype=np.float64,
    )
    return matrix, reference, counts


def restricted_minp_test(
    y: np.ndarray,
    design: np.ndarray,
    covariance: np.ndarray,
    region_states: dict[str, Sequence[str]],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Run fixed-covariance GLS block tests and Westfall-Young minP."""

    vector = np.asarray(y, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    omega = np.asarray(covariance, dtype=np.float64)
    n = len(vector)
    if permutations < 1 or omega.shape != (n, n) or x.shape[0] != n:
        raise ValueError("restricted minP inputs are invalid")
    lower = np.linalg.cholesky(omega)
    whitened_y = np.linalg.solve(lower, vector)
    whitened_x = np.linalg.solve(lower, x)
    q, _ = np.linalg.qr(whitened_x, mode="complete")
    rank_x = int(np.linalg.matrix_rank(whitened_x))
    if rank_x != x.shape[1]:
        raise ValueError("whitened fixed-effect design is rank deficient")
    residual_basis = q[:, rank_x:]
    xi = residual_basis.T @ whitened_y
    projectors: list[np.ndarray] = []
    region_metadata: list[dict[str, Any]] = []
    for region_id in EXPECTED_REGIONS:
        state_design, reference, counts = _state_design(list(region_states[region_id]))
        coordinates = residual_basis.T @ np.linalg.solve(lower, state_design)
        rank = int(np.linalg.matrix_rank(coordinates))
        if rank != state_design.shape[1] or rank < 1:
            raise ValueError(f"{region_id} state design is aliased after covariates")
        q_state, _ = np.linalg.qr(coordinates, mode="reduced")
        projectors.append(q_state[:, :rank])
        region_metadata.append(
            {
                "region_id": region_id,
                "reference_state": reference,
                "state_carrier_counts": counts,
                "degrees_of_freedom": rank,
            }
        )
    total = permutations + 1
    statistics = np.empty((total, len(projectors)), dtype=np.float64)

    def calculate(coordinates: np.ndarray) -> np.ndarray:
        return np.asarray(
            [float(np.sum((projector.T @ coordinates) ** 2)) for projector in projectors]
        )

    statistics[0] = calculate(xi)
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    for replicate in range(1, total):
        statistics[replicate] = calculate(xi[generator.permutation(len(xi))])
    empirical = np.empty_like(statistics)
    for column in range(statistics.shape[1]):
        sorted_values = np.sort(statistics[:, column])
        empirical[:, column] = (
            total - np.searchsorted(sorted_values, statistics[:, column], side="left")
        ) / total
    minimum_p = np.min(empirical, axis=1)
    observed_minimum = float(minimum_p[0])
    family_p = float(np.count_nonzero(minimum_p <= observed_minimum + 1e-15) / total)
    results = []
    for column, metadata in enumerate(region_metadata):
        results.append(
            {
                **metadata,
                "observed_statistic": float(statistics[0, column]),
                "marginal_empirical_p": float(empirical[0, column]),
            }
        )
    return {
        "permutations": permutations,
        "restricted_residual_rank": int(len(xi)),
        "seed": seed,
        "regions": results,
        "observed_minimum_marginal_p": observed_minimum,
        "westfall_young_familywise_p": family_p,
    }


def analyze(
    manifest_path: str | Path,
    fam_path: str | Path,
    panel_path: str | Path,
    panel_receipt_path: str | Path,
    phenotype_dir: str | Path,
    pc10_qcovar_path: str | Path,
    full_grm_prefix: str | Path,
    ldpruned_grm_prefix: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_source)
    fam = Path(fam_path).resolve()
    rows = _read_fam(fam, manifest)
    panel = Path(panel_path).resolve()
    panel_receipt = Path(panel_receipt_path).resolve()
    states = _read_panel(panel, panel_receipt, manifest, rows)
    phenotype_root = Path(phenotype_dir).resolve()
    qcovar = Path(pc10_qcovar_path).resolve()
    _verify(qcovar, manifest.source.pc10_qcovar)
    relationships = {}
    prefixes = {
        "full_chr5_excluded": Path(full_grm_prefix).resolve(),
        "ldpruned_chr5_excluded": Path(ldpruned_grm_prefix).resolve(),
    }
    for contract in manifest.relationships:
        relationships[contract.id] = read_gcta_grm(
            prefixes[contract.id],
            fam,
            contract.expected_markers,
            expected_samples=manifest.cohort.samples,
            expected_hashes={
                "grm_bin": contract.grm_bin_sha256,
                "grm_n_bin": contract.grm_n_bin_sha256,
                "grm_id": contract.grm_id_sha256,
            },
        )
    phenotypes: dict[str, np.ndarray] = {}
    phenotype_receipts: dict[str, Any] = {}
    for trait in TRAITS:
        slug = TRAIT_SLUGS[trait]
        path = phenotype_root / f"{slug}.phen"
        phenotype_receipts[slug] = _verify(path, manifest.source.phenotypes[slug])
        phenotypes[trait] = _read_ordered_phenotype(path, rows)
    cells: list[dict[str, Any]] = []
    for relationship_id in RELATIONSHIPS:
        grm = relationships[relationship_id]
        for endpoint in ENDPOINTS:
            design, terms, qcovar_hash = _read_design_matrix(
                rows, 0 if endpoint == "pc0" else 10, None if endpoint == "pc0" else qcovar
            )
            for trait in TRAITS:
                cell_id = f"{relationship_id}_{endpoint}_{TRAIT_SLUGS[trait]}"
                fitted = profile_reml(
                    phenotypes[trait], design, grm.eigenvalues, grm.eigenvectors
                )
                seed_digest = hashlib.sha256(
                    f"{manifest.analysis.seed}:{cell_id}".encode("utf-8")
                ).digest()
                seed = int.from_bytes(seed_digest[:8], "big", signed=False)
                test = restricted_minp_test(
                    phenotypes[trait],
                    design,
                    fitted.covariance,
                    states,
                    manifest.analysis.permutations_per_cell,
                    seed,
                )
                cells.append(
                    {
                        "cell_id": cell_id,
                        "relationship": relationship_id,
                        "endpoint": endpoint,
                        "trait": trait,
                        "design_terms": terms,
                        "qcovar_sha256": qcovar_hash,
                        "null_fit": {
                            "h2": fitted.h2,
                            "h2_boundary": bool(
                                math.isclose(fitted.h2, 0.0, abs_tol=1e-12)
                                or math.isclose(fitted.h2, 0.99, abs_tol=1e-12)
                            ),
                            "variance_genetic": fitted.variance_genetic,
                            "variance_residual": fitted.variance_residual,
                            "profile_reml_objective_up_to_constant": fitted.objective,
                        },
                        "test": test,
                    }
                )
    result = {
        "schema_version": RESULT_VERSION,
        "analysis_id": manifest.analysis_id,
        "classification": manifest.classification,
        "manifest_sha256": _sha256(manifest_source),
        "panel_sha256": _sha256(panel),
        "panel_receipt_sha256": _sha256(panel_receipt),
        "fam_sha256": _sha256(fam),
        "pc10_qcovar_sha256": _sha256(qcovar),
        "phenotypes": phenotype_receipts,
        "relationship_receipts": {
            key: value.receipt for key, value in relationships.items()
        },
        "cells": cells,
        "cell_count": len(cells),
        "permutations_per_cell": manifest.analysis.permutations_per_cell,
        "partial_results_evidentiary": False,
        "pangenome_or_pav_claim_permitted": False,
        "causal_interpretation_permitted": False,
        "biological_claims_permitted": False,
    }
    _write_new(Path(output_path), _canonical_json(result))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-contract")
    validate.add_argument("--manifest", type=Path, required=True)
    derive = commands.add_parser("derive-panel")
    derive.add_argument("--manifest", type=Path, required=True)
    derive.add_argument("--fam", type=Path, required=True)
    derive.add_argument("--coordinate-receipt", type=Path, required=True)
    derive.add_argument("--left-raw", type=Path, required=True)
    derive.add_argument("--left-markers", type=Path, required=True)
    derive.add_argument("--right-raw", type=Path, required=True)
    derive.add_argument("--right-markers", type=Path, required=True)
    derive.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("analyze")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--fam", type=Path, required=True)
    run.add_argument("--panel", type=Path, required=True)
    run.add_argument("--panel-receipt", type=Path, required=True)
    run.add_argument("--phenotype-dir", type=Path, required=True)
    run.add_argument("--pc10-qcovar", type=Path, required=True)
    run.add_argument("--full-grm-prefix", type=Path, required=True)
    run.add_argument("--ldpruned-grm-prefix", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-contract":
        manifest = load_manifest(args.manifest)
        print(json.dumps(manifest.model_dump(), sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "derive-panel":
        result = derive_panel(
            args.manifest,
            args.fam,
            args.coordinate_receipt,
            args.left_raw,
            args.left_markers,
            args.right_raw,
            args.right_markers,
            args.output_dir,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    result = analyze(
        args.manifest,
        args.fam,
        args.panel,
        args.panel_receipt,
        args.phenotype_dir,
        args.pc10_qcovar,
        args.full_grm_prefix,
        args.ldpruned_grm_prefix,
        args.output,
    )
    print(json.dumps({"cell_count": result["cell_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
