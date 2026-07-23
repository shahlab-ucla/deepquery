"""restricted-residual bootstrap covariance-preserving residual-space bootstrap for WS283 LOCO GWAS.

The implementation follows the MVNpermute construction described by Abney
(Genetic Epidemiology 2015, doi:10.1002/gepi.21893).  It does *not* permute
phenotype or genotype labels.  For each frozen PC10 trait it whitens the null
residual with the fitted 209-strain covariance, moves to the 198-dimensional
restricted residual space, permutes those exchangeable coordinates, and
recolors the residual before the exact GCTA ``--mlma-loco`` analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import uuid
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import qtl_parametric_null as parametric_null
from .qtl_grm_sensitivity import _accepted_options, _qualify_log_text


SCHEMA_VERSION = "wormctx-abamectin-ws283-restricted_residual_bootstrap-covariance-bootstrap-1.0"
TRANSFORM_VERSION = "wormctx-abamectin-ws283-restricted_residual_bootstrap-restricted-transform-1.0"
PHENOTYPE_VERSION = "wormctx-abamectin-ws283-restricted_residual_bootstrap-phenotype-1.0"
MAP_VERSION = "wormctx-abamectin-ws283-restricted_residual_bootstrap-map-1.0"
SUMMARY_VERSION = "wormctx-abamectin-ws283-restricted_residual_bootstrap-summary-1.0"
SOURCE_VERSION = "wormctx-abamectin-ws283-restricted_residual_bootstrap-source-verification-1.0"

TRAITS = ("mean.EXT", "mean.TOF", "mean.norm.EXT", "norm.n")
TRAIT_SLUGS = {
    "mean.EXT": "mean_EXT",
    "mean.TOF": "mean_TOF",
    "mean.norm.EXT": "mean_norm_EXT",
    "norm.n": "norm_n",
}
SLUG_TRAITS = {value: key for key, value in TRAIT_SLUGS.items()}
EXPECTED_ROLES = {
    "baseline_success",
    "baseline_checksums",
    "baseline_bed",
    "baseline_bim",
    "baseline_fam",
    "phenotype_mean_EXT",
    "phenotype_mean_TOF",
    "phenotype_mean_norm_EXT",
    "phenotype_norm_n",
    "calibration_success",
    "calibration_checksums",
    "pc10_qcovar",
    "parametric_null_success",
    "parametric_null_contract",
    "parametric_null_checksums",
    "full_grm_bin",
    "full_grm_n_bin",
    "full_grm_id",
    "fit_mean_EXT",
    "fit_mean_TOF",
    "fit_mean_norm_EXT",
    "fit_norm_n",
    "hsq_mean_EXT",
    "hsq_mean_TOF",
    "hsq_mean_norm_EXT",
    "hsq_norm_n",
    "matched_relationship_model_contract",
    "matched_relationship_model_result",
}
ROOT_IDS = ("baseline", "calibration", "parametric_null", "repository")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FrozenAsset(_StrictModel):
    role: str
    root_id: Literal["baseline", "calibration", "parametric_null", "repository"]
    relative_path: str
    bytes: int = Field(gt=0)
    sha256: str

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not value or "\\" in value:
            raise ValueError("asset path must be a safe POSIX-style relative path")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("asset SHA-256 must be lowercase hexadecimal")
        return value


class SourceRun(_StrictModel):
    role: Literal["baseline", "calibration", "parametric_null"]
    run_id: str
    terminal_marker: Literal["SUCCESS"]
    failure_marker_absent_required: Literal[True]


class TraitContract(_StrictModel):
    trait: Literal["mean.EXT", "mean.TOF", "mean.norm.EXT", "norm.n"]
    trait_slug: Literal["mean_EXT", "mean_TOF", "mean_norm_EXT", "norm_n"]
    phenotype_role: str
    fit_role: str
    hsq_role: str

    @model_validator(mode="after")
    def canonical_roles(self) -> "TraitContract":
        expected_slug = TRAIT_SLUGS[self.trait]
        if self.trait_slug != expected_slug:
            raise ValueError("trait slug is not canonical")
        if self.phenotype_role != f"phenotype_{expected_slug}":
            raise ValueError("trait phenotype role is not canonical")
        if self.fit_role != f"fit_{expected_slug}" or self.hsq_role != f"hsq_{expected_slug}":
            raise ValueError("trait fit roles are not canonical")
        return self


class BootstrapContract(_StrictModel):
    method: Literal["Abney_2015_MVNpermute_restricted_residual_space"]
    null_covariance: Literal["Omega=Vg*K_full_209+Ve*I"]
    transform_formula: Literal[
        "y_star=X*beta_GLS+L*U1*P*U1T*Linv*(y-X*beta_GLS)"
    ]
    cholesky_convention: Literal["Omega=L*LT_lower_triangular"]
    permutation_target: Literal["198_restricted_whitened_residual_coordinates"]
    phenotype_label_permutation_permitted: Literal[False]
    genotype_label_permutation_permitted: Literal[False]
    sign_flip_permitted: Literal[False]
    covariance_reestimated_per_replicate_by_gcta_loco: Literal[True]
    replicates_per_trait: Literal[2000]
    trait_count: Literal[4]
    total_maps: Literal[8000]
    random_generator: Literal["numpy.PCG64DXSM"]
    seed_derivation: Literal["SHA256_length_delimited_v1"]
    master_seed_sha256: str
    phenotype_float_format: Literal[".17g"]
    byte_replay_required: Literal[True]
    smoke_replicates: list[int]
    sentinel_replicates: list[int]

    @field_validator("master_seed_sha256")
    @classmethod
    def valid_seed(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("master seed must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def frozen_sets(self) -> "BootstrapContract":
        if self.smoke_replicates != [1, 2]:
            raise ValueError("smoke replicates must be exactly 1 and 2")
        if self.sentinel_replicates != [1, 250, 500, 1000, 1500, 2000]:
            raise ValueError("sentinel set differs from the frozen contract")
        return self


class MultiplicityContract(_StrictModel):
    overall_alpha: Literal[0.05]
    within_trait_statistic: Literal["minimum_p_across_373279_markers_and_six_chromosomes"]
    across_trait_method: Literal["Bonferroni_union_bound_four_predeclared_traits"]
    trait_alpha: Literal[0.0125]
    empirical_p_correction: Literal["plus_one"]
    empirical_p_formula: Literal["(1+count(null_min_p<=observed_min_p))/(B+1)"]
    pooled_trait_null_permitted: Literal[False]
    cross_trait_exchangeability_assumed: Literal[False]
    critical_rank: Literal[25]
    threshold_rule: Literal["observed_min_p_strictly_less_than_25th_sorted_null_min_p"]


class PrecisionPower(_StrictModel):
    p_value_resolution: float
    expected_tail_draws_at_trait_alpha: float
    monte_carlo_se_at_trait_alpha: float
    monte_carlo_se_at_nominal_005: float
    diagnostic_null_fwer: Literal[0.05]
    diagnostic_one_sided_alpha_per_trait: Literal[0.0125]
    diagnostic_critical_events: Literal[123]
    power_if_true_fwer_006: float
    power_if_true_fwer_0075: float
    alternative_effect_power_claims_permitted: Literal[False]
    production_threshold_claim_permitted: Literal[False]


class ExecutionContract(_StrictModel):
    association_method: Literal["GCTA_MLMA_LOCO"]
    endpoint: Literal["pc10"]
    sample_count: Literal[209]
    fixed_effect_count: Literal[11]
    restricted_residual_rank: Literal[198]
    marker_count: Literal[373279]
    maf: Literal[0.05]
    autosome_num: Literal[6]
    workers: Literal[2]
    threads_per_worker: Literal[8]
    maximum_concurrent_gcta_threads: Literal[16]
    smoke_map_count: Literal[8]
    full_map_count: Literal[8000]
    full_launch_requires_separate_explicit_mode: Literal[True]
    checkpoint_before_raw_map_removal: Literal[True]
    retain_all_phenotypes: Literal[True]
    retain_only_sentinel_maps: Literal[True]
    no_data_dependent_early_stop: Literal[True]


class ToolContract(_StrictModel):
    name: Literal["gcta64"]
    absolute_path: Literal["/opt/deepquery/bin/gcta64"]
    version: Literal["1.94.1_2022-11-15"]
    sha256: str

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("tool SHA-256 must be lowercase hexadecimal")
        return value


class StopGoContract(_StrictModel):
    go_requires: list[str]
    stop_on: list[str]
    full_launch_blocked_until_smoke_qualified: Literal[True]
    outcome_based_launch_decisions_permitted: Literal[False]


class ClaimsContract(_StrictModel):
    permitted: list[str]
    prohibited: list[str]


class Manifest(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    analysis_id: Literal["abamectin_ws283_restricted_residual_bootstrap_mvnpermute_pc10_loco_v1"]
    classification: Literal["preliminary_covariance_preserving_loco_calibration"]
    status: Literal["frozen_smoke_eligible_full_launch_not_yet_qualified"]
    validated: Literal[False]
    biological_claims_permitted: Literal[False]
    source_runs: list[SourceRun]
    assets: list[FrozenAsset]
    traits: list[TraitContract]
    bootstrap: BootstrapContract
    multiplicity: MultiplicityContract
    precision_power: PrecisionPower
    execution: ExecutionContract
    tool: ToolContract
    stop_go: StopGoContract
    claims: ClaimsContract

    @model_validator(mode="after")
    def exact_design(self) -> "Manifest":
        if [item.role for item in self.source_runs] != ["baseline", "calibration", "parametric_null"]:
            raise ValueError("source runs or order differ from the frozen contract")
        roles = [item.role for item in self.assets]
        if set(roles) != EXPECTED_ROLES or len(roles) != len(EXPECTED_ROLES):
            raise ValueError("asset roles are incomplete or duplicated")
        if [item.trait for item in self.traits] != list(TRAITS):
            raise ValueError("all four PC10 traits must appear in frozen order")
        if set(self.stop_go.go_requires) != {
            "all_source_hashes_and_parent_terminal_markers_verified",
            "four_transform_receipts_rank_198_and_reconstruction_qualified",
            "eight_smoke_maps_and_checkpoints_qualified",
            "smoke_walltime_projection_within_48_hours_for_8000_maps",
            "projected_peak_disk_below_25_gib_and_free_disk_above_100_gib",
            "no_competing_gcta_or_plink_process",
        }:
            raise ValueError("GO gate is incomplete")
        if not any("naive" in item.lower() for item in self.claims.prohibited):
            raise ValueError("claim boundary must explicitly prohibit naive permutation")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: Any) -> None:
    path = path.resolve()
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
    if not source.is_file():
        raise FileNotFoundError(f"restricted-residual bootstrap manifest is missing: {source}")
    return Manifest.model_validate_json(source.read_text(encoding="utf-8"))


def _asset(manifest: Manifest, role: str) -> FrozenAsset:
    matches = [item for item in manifest.assets if item.role == role]
    if len(matches) != 1:
        raise ValueError(f"manifest does not contain exactly one asset role {role}")
    return matches[0]


def _trait(manifest: Manifest, slug: str) -> TraitContract:
    matches = [item for item in manifest.traits if item.trait_slug == slug]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicated trait slug: {slug}")
    return matches[0]


def _resolve_asset(asset: FrozenAsset, roots: dict[str, Path]) -> Path:
    root = roots[asset.root_id]
    candidate = (root / asset.relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"asset escapes its root: {asset.role}") from exc
    return candidate


def _verify_asset(path: Path, asset: FrozenAsset) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen asset is missing: {asset.role}: {path}")
    observed_bytes = path.stat().st_size
    observed_hash = _sha256(path)
    if observed_bytes != asset.bytes or observed_hash != asset.sha256:
        raise ValueError(f"frozen asset identity mismatch: {asset.role}")
    return {
        "role": asset.role,
        "root_id": asset.root_id,
        "relative_path": asset.relative_path,
        "bytes": observed_bytes,
        "sha256": observed_hash,
    }


def verify_sources(
    manifest_path: str | Path,
    baseline_root: str | Path,
    calibration_root: str | Path,
    parametric_null_root: str | Path,
    repository_root: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Verify every frozen computational source before any phenotype is generated."""

    source = Path(manifest_path).resolve()
    manifest = load_manifest(source)
    roots = {
        "baseline": Path(baseline_root).resolve(),
        "calibration": Path(calibration_root).resolve(),
        "parametric_null": Path(parametric_null_root).resolve(),
        "repository": Path(repository_root).resolve(),
    }
    for role, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{role} root is not a directory: {root}")
    terminal = []
    for run in manifest.source_runs:
        root = roots[run.role]
        marker = root / run.terminal_marker
        if not marker.is_file() or (root / "FAILURE").exists():
            raise ValueError(f"source run is not cleanly terminal: {run.role}")
        terminal.append({"role": run.role, "run_id": run.run_id, "success": True})
    verified = [
        _verify_asset(_resolve_asset(asset, roots), asset) for asset in manifest.assets
    ]
    parametric_null_contract = parametric_null.load_manifest(_resolve_asset(_asset(manifest, "parametric_null_contract"), roots))
    if parametric_null_contract.analysis_id != "abamectin_qtl_ws283_parametric_polygenic_null_v1":
        raise ValueError("parametric-null calibration source contract has the wrong analysis identity")
    matched_relationship_model_result = _json_object(
        _resolve_asset(_asset(manifest, "matched_relationship_model_result"), roots), "matched relationship-model diagnostic result"
    )
    if (
        matched_relationship_model_result.get("interpretation", {}).get("next_lane")
        != "restricted_residual_or_covariance_preserving_bootstrap_for_the_intended_discovery_model_with_explicit_multiplicity"
    ):
        raise ValueError("matched relationship-model diagnostic result does not authorize the frozen restricted-residual bootstrap design question")
    receipt = {
        "schema_version": SOURCE_VERSION,
        "manifest_sha256": _sha256(source),
        "roots": {key: str(value) for key, value in roots.items()},
        "terminal_sources": terminal,
        "assets_verified": verified,
        "asset_count": len(verified),
        "matched_relationship_model_next_lane_verified": True,
        "qualified": True,
    }
    if output is not None:
        _write_json_atomic(Path(output), receipt)
    return receipt


def _stabilize_column_signs(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        vector = result[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            result[:, column] *= -1.0
    return result


def restricted_transform_arrays(
    y: np.ndarray,
    design: np.ndarray,
    covariance: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Construct Abney's null residual transform with deterministic QR signs."""

    y = np.asarray(y, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    n = y.shape[0]
    if (
        y.shape != (n,)
        or design.ndim != 2
        or design.shape[0] != n
        or covariance.shape != (n, n)
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(design))
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("restricted transform inputs are invalid")
    p = design.shape[1]
    if np.linalg.matrix_rank(design) != p or not 0 < p < n:
        raise ValueError("fixed-effect design is not full rank")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("null phenotype covariance is not positive definite")
    lower = np.linalg.cholesky(covariance)
    whitened_y = np.linalg.solve(lower, y)
    whitened_x = np.linalg.solve(lower, design)
    information = whitened_x.T @ whitened_x
    beta = np.linalg.solve(information, whitened_x.T @ whitened_y)
    whitened_residual = whitened_y - whitened_x @ beta
    q_full, _ = np.linalg.qr(whitened_x, mode="complete")
    residual_basis = _stabilize_column_signs(q_full[:, p:])
    xi = residual_basis.T @ whitened_residual
    fixed = design @ beta
    recolor = lower @ residual_basis
    reconstructed_residual = recolor @ xi
    reconstruction_error = float(np.max(np.abs((y - fixed) - reconstructed_residual)))
    orthonormal_error = float(
        np.max(np.abs(residual_basis.T @ residual_basis - np.eye(n - p)))
    )
    fixed_orthogonality_error = float(
        np.max(np.abs(residual_basis.T @ whitened_x))
    )
    if reconstruction_error > 1e-8 or orthonormal_error > 1e-10:
        raise ValueError("restricted residual basis failed numerical reconstruction")
    arrays = {
        "fixed": fixed,
        "recolor": recolor,
        "xi": xi,
        "beta": beta,
        "lower_cholesky": lower,
        "residual_basis": residual_basis,
    }
    diagnostics: dict[str, float | int] = {
        "sample_count": n,
        "fixed_effect_count": p,
        "restricted_residual_rank": n - p,
        "covariance_minimum_eigenvalue": float(eigenvalues[0]),
        "covariance_maximum_eigenvalue": float(eigenvalues[-1]),
        "covariance_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "information_condition_number": float(np.linalg.cond(information)),
        "reconstruction_max_abs_error": reconstruction_error,
        "basis_orthonormal_max_abs_error": orthonormal_error,
        "basis_fixed_orthogonality_max_abs_error": fixed_orthogonality_error,
        "xi_mean": float(np.mean(xi)),
        "xi_variance_ddof1": float(np.var(xi, ddof=1)),
        "xi_l2_norm": float(np.linalg.norm(xi)),
    }
    return arrays, diagnostics


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    stage = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with stage.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(stage, path)
    except BaseException:
        if stage.exists():
            stage.unlink()
        raise


def prepare_transform(
    manifest_path: str | Path,
    baseline_root: str | Path,
    calibration_root: str | Path,
    parametric_null_root: str | Path,
    trait_slug: str,
    transform_output: str | Path,
    receipt_output: str | Path,
) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = load_manifest(source)
    trait = _trait(manifest, trait_slug)
    roots = {
        "baseline": Path(baseline_root).resolve(),
        "calibration": Path(calibration_root).resolve(),
        "parametric_null": Path(parametric_null_root).resolve(),
        "repository": source.parent.parent.parent.resolve(),
    }
    roles = (
        "baseline_fam",
        "pc10_qcovar",
        "full_grm_bin",
        "full_grm_n_bin",
        "full_grm_id",
        "parametric_null_contract",
        trait.phenotype_role,
        trait.fit_role,
        trait.hsq_role,
    )
    paths = {}
    for role in roles:
        asset = _asset(manifest, role)
        path = _resolve_asset(asset, roots)
        _verify_asset(path, asset)
        paths[role] = path
    fam = paths["baseline_fam"]
    samples = parametric_null._read_fam(fam, manifest.execution.sample_count)
    design, terms, qcovar_hash = parametric_null._read_design_matrix(
        samples, 10, paths["pc10_qcovar"]
    )
    y = parametric_null._read_ordered_phenotype(paths[trait.phenotype_role], samples)
    grm_prefix = paths["full_grm_bin"].with_suffix("").with_suffix("")
    grm = parametric_null.read_gcta_grm(
        grm_prefix,
        fam,
        manifest.execution.marker_count,
        expected_samples=manifest.execution.sample_count,
        expected_hashes={
            "grm_bin": _asset(manifest, "full_grm_bin").sha256,
            "grm_n_bin": _asset(manifest, "full_grm_n_bin").sha256,
            "grm_id": _asset(manifest, "full_grm_id").sha256,
        },
    )
    fit_payload = _json_object(paths[trait.fit_role], "qualified parametric-null calibration fit")
    parametric_null_manifest_hash = _asset(manifest, "parametric_null_contract").sha256
    vg, ve, _ = parametric_null._validate_fit_receipt(
        fit_payload,
        parametric_null_manifest_hash,
        f"full_pc10_{trait_slug}",
        grm,
        terms,
    )
    covariance = vg * grm.matrix + ve * np.eye(len(samples))
    arrays, diagnostics = restricted_transform_arrays(y, design, covariance)
    if (
        diagnostics["sample_count"] != 209
        or diagnostics["fixed_effect_count"] != 11
        or diagnostics["restricted_residual_rank"] != 198
    ):
        raise ValueError("prepared transform does not have the frozen 209/11/198 dimensions")
    destination = Path(transform_output).resolve()
    _write_npz_atomic(destination, arrays)
    receipt = {
        "schema_version": TRANSFORM_VERSION,
        "manifest_sha256": _sha256(source),
        "trait": trait.trait,
        "trait_slug": trait_slug,
        "method": manifest.bootstrap.method,
        "formula": manifest.bootstrap.transform_formula,
        "inputs": {
            "fam_sha256": _sha256(fam),
            "phenotype_sha256": _sha256(paths[trait.phenotype_role]),
            "qcovar_sha256": qcovar_hash,
            "grm_bin_sha256": grm.receipt["grm_bin_sha256"],
            "grm_n_bin_sha256": grm.receipt["grm_n_bin_sha256"],
            "grm_id_sha256": grm.receipt["grm_id_sha256"],
            "fit_receipt_sha256": _sha256(paths[trait.fit_role]),
            "hsq_sha256": _sha256(paths[trait.hsq_role]),
            "Vg": vg,
            "Ve": ve,
        },
        "fixed_effect_terms": terms,
        "fixed_effects_gls": [float(value) for value in arrays["beta"]],
        "diagnostics": diagnostics,
        "transform": {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "numpy_version": np.__version__,
            "array_names": sorted(arrays),
        },
        "naive_phenotype_permutation_used": False,
        "naive_genotype_permutation_used": False,
        "qualified": True,
    }
    _write_json_atomic(Path(receipt_output), receipt)
    return receipt


def derive_permutation_seed(master_sha256: str, trait_slug: str, replicate: int) -> dict[str, Any]:
    if not _SHA256_RE.fullmatch(master_sha256) or trait_slug not in SLUG_TRAITS or replicate <= 0:
        raise ValueError("invalid permutation seed coordinates")
    parts = ("wormctx-restricted_residual_bootstrap-mvnpermute-v1", master_sha256, trait_slug, str(replicate))
    material = b"".join(len(item.encode("utf-8")).to_bytes(4, "big") + item.encode("utf-8") for item in parts)
    digest = hashlib.sha256(material).digest()
    return {
        "derivation": "SHA256_length_delimited_v1",
        "generator": "numpy.PCG64DXSM",
        "seed_sha256": digest.hex(),
        "seed_integer": int.from_bytes(digest, "big"),
    }


def _load_transform(path: Path, receipt: dict[str, Any]) -> dict[str, np.ndarray]:
    expected = receipt.get("transform", {})
    if _sha256(path) != expected.get("sha256") or path.stat().st_size != expected.get("bytes"):
        raise ValueError("transform NPZ differs from its qualification receipt")
    with np.load(path, allow_pickle=False) as bundle:
        names = set(bundle.files)
        if names != {"fixed", "recolor", "xi", "beta", "lower_cholesky", "residual_basis"}:
            raise ValueError("transform NPZ array inventory differs from contract")
        arrays = {name: np.asarray(bundle[name], dtype=np.float64) for name in bundle.files}
    rank = int(receipt["diagnostics"]["restricted_residual_rank"])
    sample_count = int(receipt["diagnostics"]["sample_count"])
    if arrays["fixed"].shape != (sample_count,) or arrays["recolor"].shape != (sample_count, rank) or arrays["xi"].shape != (rank,):
        raise ValueError("transform arrays have invalid dimensions")
    return arrays


def permuted_phenotype_values(
    fixed: np.ndarray,
    recolor: np.ndarray,
    xi: np.ndarray,
    seed: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed["seed_integer"]))
    permutation = rng.permutation(len(xi))
    values = fixed + recolor @ xi[permutation]
    if not np.all(np.isfinite(values)) or sorted(permutation.tolist()) != list(range(len(xi))):
        raise ValueError("permuted phenotype is invalid")
    return values, permutation


def _phenotype_bytes(ids: Sequence[tuple[str, str]], values: np.ndarray) -> bytes:
    if values.shape != (len(ids),):
        raise ValueError("phenotype values and IDs differ in length")
    return "".join(
        f"{fid}\t{iid}\t{format(float(value), '.17g')}\n"
        for (fid, iid), value in zip(ids, values, strict=True)
    ).encode("utf-8")


def generate_phenotype(
    manifest_path: str | Path,
    fam_path: str | Path,
    transform_path: str | Path,
    transform_receipt_path: str | Path,
    trait_slug: str,
    replicate: int,
    phenotype_output: str | Path,
    receipt_output: str | Path,
) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = load_manifest(source)
    trait = _trait(manifest, trait_slug)
    if not 1 <= replicate <= manifest.bootstrap.replicates_per_trait:
        raise ValueError("replicate is outside 1..2000")
    fam = Path(fam_path).resolve()
    fam_asset = _asset(manifest, "baseline_fam")
    _verify_asset(fam, fam_asset)
    ids = parametric_null._read_fam(fam, manifest.execution.sample_count)
    transform_source = Path(transform_path).resolve()
    transform_receipt_source = Path(transform_receipt_path).resolve()
    transform_receipt = _json_object(transform_receipt_source, "transform receipt")
    if (
        transform_receipt.get("schema_version") != TRANSFORM_VERSION
        or transform_receipt.get("manifest_sha256") != _sha256(source)
        or transform_receipt.get("trait_slug") != trait_slug
        or transform_receipt.get("qualified") is not True
    ):
        raise ValueError("transform receipt is not qualified for this trait and manifest")
    arrays = _load_transform(transform_source, transform_receipt)
    seed = derive_permutation_seed(manifest.bootstrap.master_seed_sha256, trait_slug, replicate)
    values, permutation = permuted_phenotype_values(
        arrays["fixed"], arrays["recolor"], arrays["xi"], seed
    )
    replay_values, replay_permutation = permuted_phenotype_values(
        arrays["fixed"], arrays["recolor"], arrays["xi"], seed
    )
    data = _phenotype_bytes(ids, values)
    replay = _phenotype_bytes(ids, replay_values)
    if not np.array_equal(values, replay_values) or not np.array_equal(permutation, replay_permutation) or data != replay:
        raise RuntimeError("deterministic restricted-residual bootstrap phenotype replay failed")
    phenotype_destination = Path(phenotype_output).resolve()
    receipt_destination = Path(receipt_output).resolve()
    if phenotype_destination.exists() or receipt_destination.exists():
        raise FileExistsError("phenotype outputs must not already exist")
    phenotype_destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_destination.parent.mkdir(parents=True, exist_ok=True)
    phenotype_hash = hashlib.sha256(data).hexdigest()
    receipt = {
        "schema_version": PHENOTYPE_VERSION,
        "manifest_sha256": _sha256(source),
        "trait": trait.trait,
        "trait_slug": trait_slug,
        "replicate": replicate,
        "method": manifest.bootstrap.method,
        "transform_receipt_sha256": _sha256(transform_receipt_source),
        "transform_npz_sha256": _sha256(transform_source),
        "permutation_seed": seed,
        "permutation_sha256": hashlib.sha256(
            permutation.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "permutation_is_exact_0_to_197": bool(
            np.array_equal(np.sort(permutation), np.arange(198))
        ),
        "restricted_coordinate_multiset_preserved": True,
        "naive_phenotype_permutation_used": False,
        "naive_genotype_permutation_used": False,
        "phenotype": {
            "path": str(phenotype_destination),
            "bytes": len(data),
            "sha256": phenotype_hash,
            "samples": len(ids),
            "sample_order": "exact_FAM_order",
            "float_format": manifest.bootstrap.phenotype_float_format,
        },
        "byte_replay_verified": True,
    }
    phenotype_stage = phenotype_destination.parent / f".{phenotype_destination.name}.tmp-{uuid.uuid4().hex}"
    receipt_stage = receipt_destination.parent / f".{receipt_destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        phenotype_stage.write_bytes(data)
        receipt_stage.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if _sha256(phenotype_stage) != phenotype_hash:
            raise RuntimeError("staged phenotype hash differs from replay")
        os.replace(phenotype_stage, phenotype_destination)
        os.replace(receipt_stage, receipt_destination)
    except BaseException:
        for stage in (phenotype_stage, receipt_stage):
            if stage.exists():
                stage.unlink()
        raise
    return receipt


def _qualify_gcta_log(
    log_path: Path,
    bfile_prefix: Path,
    phenotype: Path,
    qcovar: Path,
    output_prefix: Path,
    manifest: Manifest,
) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8")
    warnings = _qualify_log_text(text, log_path)
    options = _accepted_options(text, log_path)
    expected_names = {
        "--mlma-loco",
        "--bfile",
        "--pheno",
        "--qcovar",
        "--maf",
        "--autosome-num",
        "--thread-num",
        "--out",
    }
    if set(options) != expected_names or options["--mlma-loco"] is not None:
        raise ValueError("GCTA options differ from the exact restricted-residual bootstrap LOCO contract")
    expected_paths = {
        "--bfile": bfile_prefix,
        "--pheno": phenotype,
        "--qcovar": qcovar,
        "--out": output_prefix,
    }
    for option, expected in expected_paths.items():
        observed = options.get(option)
        if observed is None or Path(observed).resolve() != expected.resolve():
            raise ValueError(f"GCTA {option} differs from the exact restricted-residual bootstrap input")
    if (
        options["--maf"] != "0.05"
        or options["--autosome-num"] != "6"
        or options["--thread-num"] != str(manifest.execution.threads_per_worker)
    ):
        raise ValueError("GCTA numeric options differ from restricted-residual bootstrap")
    if "209 individuals are in common in these files." not in text:
        raise ValueError("GCTA log lacks exact 209-sample evidence")
    if "10 quantitative covariate(s) of 209 individuals are included" not in text:
        raise ValueError("GCTA log lacks exact PC10 evidence")
    convergence = [line.strip() for line in text.splitlines() if "converg" in line.lower()]
    if not convergence or any("not converg" in line.lower() for line in convergence):
        raise ValueError("GCTA LOCO log lacks positive convergence evidence")
    return {
        "path": str(log_path),
        "sha256": _sha256(log_path),
        "accepted_options": options,
        "convergence_evidence": convergence,
        "warnings": warnings,
    }


def qualify_map(
    manifest_path: str | Path,
    parametric_null_contract_path: str | Path,
    bim_path: str | Path,
    bfile_prefix: str | Path,
    qcovar_path: str | Path,
    map_path: str | Path,
    gcta_log_path: str | Path,
    gcta_output_prefix: str | Path,
    phenotype_path: str | Path,
    phenotype_receipt_path: str | Path,
    trait_slug: str,
    replicate: int,
    output: str | Path,
    retained_map: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = load_manifest(source)
    trait = _trait(manifest, trait_slug)
    if not 1 <= replicate <= manifest.bootstrap.replicates_per_trait:
        raise ValueError("map replicate is outside 1..2000")
    phenotype = Path(phenotype_path).resolve()
    phenotype_receipt_source = Path(phenotype_receipt_path).resolve()
    phenotype_receipt = _json_object(phenotype_receipt_source, "phenotype receipt")
    if (
        phenotype_receipt.get("schema_version") != PHENOTYPE_VERSION
        or phenotype_receipt.get("manifest_sha256") != _sha256(source)
        or phenotype_receipt.get("trait_slug") != trait_slug
        or phenotype_receipt.get("replicate") != replicate
        or phenotype_receipt.get("phenotype", {}).get("sha256") != _sha256(phenotype)
    ):
        raise ValueError("phenotype and receipt are not a qualified restricted-residual bootstrap tuple")
    parametric_null_contract_source = Path(parametric_null_contract_path).resolve()
    if _sha256(parametric_null_contract_source) != _asset(manifest, "parametric_null_contract").sha256:
        raise ValueError("parametric-null calibration diagnostic contract differs from frozen identity")
    parametric_null_manifest = parametric_null.load_manifest(parametric_null_contract_source)
    diagnostics = parametric_null.stream_map_diagnostics(
        Path(map_path).resolve(), Path(bim_path).resolve(), trait.trait, parametric_null_manifest
    )
    log = _qualify_gcta_log(
        Path(gcta_log_path).resolve(),
        Path(bfile_prefix).resolve(),
        phenotype,
        Path(qcovar_path).resolve(),
        Path(gcta_output_prefix).resolve(),
        manifest,
    )
    retained = None
    if retained_map is not None:
        retained_path = Path(retained_map).resolve()
        retained_path.parent.mkdir(parents=True, exist_ok=True)
        if retained_path.exists():
            raise FileExistsError(f"refusing to overwrite retained map: {retained_path}")
        shutil.copyfile(Path(map_path).resolve(), retained_path)
        if _sha256(retained_path) != diagnostics["map_sha256"]:
            raise RuntimeError("retained map differs from qualified source map")
        retained = {
            "path": str(retained_path),
            "bytes": retained_path.stat().st_size,
            "sha256": _sha256(retained_path),
        }
    receipt = {
        "schema_version": MAP_VERSION,
        "manifest_sha256": _sha256(source),
        "trait": trait.trait,
        "trait_slug": trait_slug,
        "replicate": replicate,
        "phenotype_sha256": _sha256(phenotype),
        "phenotype_receipt_sha256": _sha256(phenotype_receipt_source),
        "association_method": manifest.execution.association_method,
        "diagnostics": diagnostics,
        "gcta_log": log,
        "retained_map": retained,
        "qualified": True,
    }
    _write_json_atomic(Path(output), receipt)
    return receipt


def _load_observed_minimum(parametric_null_root: Path, slug: str, expected_contract_hash: str) -> dict[str, Any]:
    path = parametric_null_root / "observed" / f"full_pc10_{slug}.json"
    payload = _json_object(path, "parametric-null calibration observed-map receipt")
    if (
        payload.get("schema_version") != parametric_null.MAP_VERSION
        or payload.get("manifest_sha256") != expected_contract_hash
        or payload.get("cell", {}).get("id") != f"full_pc10_{slug}"
        or payload.get("replicate") != 0
    ):
        raise ValueError("parametric-null calibration observed-map receipt has the wrong identity")
    minimum = payload.get("diagnostics", {}).get("minimum_p")
    if not isinstance(minimum, dict) or not 0.0 < float(minimum.get("p", 0.0)) <= 1.0:
        raise ValueError("parametric-null calibration observed-map receipt lacks a valid minimum p-value")
    return {"path": str(path), "sha256": _sha256(path), "minimum_p": minimum}


def aggregate(
    manifest_path: str | Path,
    parametric_null_root: str | Path,
    checkpoint_root: str | Path,
    mode: Literal["smoke", "full"],
    output: str | Path,
) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = load_manifest(source)
    checkpoints = Path(checkpoint_root).resolve()
    parametric_null_source = Path(parametric_null_root).resolve()
    replicates = 2 if mode == "smoke" else manifest.bootstrap.replicates_per_trait
    cells = []
    for trait in manifest.traits:
        rows = []
        for replicate in range(1, replicates + 1):
            path = checkpoints / trait.trait_slug / f"rep-{replicate:04d}.json"
            payload = _json_object(path, "restricted-residual bootstrap map checkpoint")
            if (
                payload.get("schema_version") != MAP_VERSION
                or payload.get("manifest_sha256") != _sha256(source)
                or payload.get("trait_slug") != trait.trait_slug
                or payload.get("replicate") != replicate
                or payload.get("qualified") is not True
            ):
                raise ValueError(f"invalid checkpoint tuple: {trait.trait_slug}/{replicate}")
            rows.append(payload)
        min_p = [float(item["diagnostics"]["minimum_p"]["p"]) for item in rows]
        nominal_threshold = (
            manifest.multiplicity.overall_alpha / manifest.execution.marker_count
        )
        adjusted_threshold = (
            manifest.multiplicity.trait_alpha / manifest.execution.marker_count
        )
        nominal_events = sum(value <= nominal_threshold for value in min_p)
        adjusted_events = sum(value <= adjusted_threshold for value in min_p)
        cell: dict[str, Any] = {
            "trait": trait.trait,
            "trait_slug": trait.trait_slug,
            "replicates": replicates,
            "minimum_p_distribution": {
                "minimum": min(min_p),
                "median": statistics.median(min_p),
                "maximum": max(min_p),
            },
            "nominal_marker_bonferroni_alpha_005": {
                "threshold": nominal_threshold,
                "events": nominal_events,
                "rate": nominal_events / replicates,
                "binomial_interval": parametric_null._binomial_interval(nominal_events, replicates),
            },
            "four_trait_adjusted_marker_bonferroni_alpha_00125": {
                "threshold": adjusted_threshold,
                "events": adjusted_events,
                "rate": adjusted_events / replicates,
                "binomial_interval": parametric_null._binomial_interval(adjusted_events, replicates),
            },
        }
        if mode == "full":
            ordered = sorted(min_p)
            critical = ordered[manifest.multiplicity.critical_rank - 1]
            observed = _load_observed_minimum(
                parametric_null_source, trait.trait_slug, _asset(manifest, "parametric_null_contract").sha256
            )
            observed_p = float(observed["minimum_p"]["p"])
            count = sum(value <= observed_p for value in min_p)
            empirical_p = (1 + count) / (replicates + 1)
            cell["preliminary_traitwise_calibration"] = {
                "trait_alpha": manifest.multiplicity.trait_alpha,
                "critical_rank": manifest.multiplicity.critical_rank,
                "critical_min_p_strict_rule": critical,
                "observed": observed,
                "null_min_p_at_or_below_observed": count,
                "plus_one_empirical_p": empirical_p,
                "passes_four_trait_union_bound": empirical_p
                <= manifest.multiplicity.trait_alpha,
            }
            cell["nominal_fwer_inflation_diagnostic"] = {
                "critical_events": manifest.precision_power.diagnostic_critical_events,
                "events": nominal_events,
                "flags_rate_above_0_05": nominal_events
                >= manifest.precision_power.diagnostic_critical_events,
            }
        cells.append(cell)
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_VERSION,
        "manifest_sha256": _sha256(source),
        "mode": mode,
        "traits": cells,
        "completed_maps": replicates * len(cells),
        "expected_maps": manifest.execution.smoke_map_count
        if mode == "smoke"
        else manifest.execution.full_map_count,
        "multiplicity": manifest.multiplicity.model_dump(),
        "biological_claims_permitted": False,
        "production_threshold_claim_permitted": False,
        "qualified": True,
    }
    if mode == "full":
        trait_p = [
            item["preliminary_traitwise_calibration"]["plus_one_empirical_p"]
            for item in cells
        ]
        summary["four_trait_result"] = {
            "minimum_trait_empirical_p": min(trait_p),
            "bonferroni_adjusted_minimum_empirical_p": min(1.0, 4.0 * min(trait_p)),
            "any_trait_passes_union_bound": any(value <= 0.0125 for value in trait_p),
            "no_cross_trait_pooling": True,
        }
    _write_json_atomic(Path(output), summary)
    return summary


def contract_summary(manifest_path: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = load_manifest(source)
    payload = {
        "schema_version": "wormctx-abamectin-ws283-restricted_residual_bootstrap-local-qualification-1.0",
        "manifest_sha256": _sha256(source),
        "analysis_id": manifest.analysis_id,
        "traits": [item.trait for item in manifest.traits],
        "sample_fixed_residual_dimensions": [
            manifest.execution.sample_count,
            manifest.execution.fixed_effect_count,
            manifest.execution.restricted_residual_rank,
        ],
        "maps": manifest.execution.full_map_count,
        "trait_alpha": manifest.multiplicity.trait_alpha,
        "critical_rank": manifest.multiplicity.critical_rank,
        "p_value_resolution": manifest.precision_power.p_value_resolution,
        "full_launch_eligible": False,
        "full_launch_blockers": [
            "remote_transform_preflight_not_yet_qualified",
            "eight_map_remote_smoke_not_yet_qualified",
            "smoke_based_walltime_and_disk_projection_not_yet_qualified",
        ],
        "qualified_for_remote_smoke": True,
    }
    if output is not None:
        _write_json_atomic(Path(output), payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    contract = commands.add_parser("contract-summary")
    contract.add_argument("--manifest", type=Path, required=True)
    contract.add_argument("--output", type=Path)

    sources = commands.add_parser("verify-sources")
    sources.add_argument("--manifest", type=Path, required=True)
    sources.add_argument("--baseline-root", type=Path, required=True)
    sources.add_argument("--calibration-root", type=Path, required=True)
    sources.add_argument("--parametric_null-root", type=Path, required=True)
    sources.add_argument("--repository-root", type=Path, required=True)
    sources.add_argument("--output", type=Path, required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--baseline-root", type=Path, required=True)
    prepare.add_argument("--calibration-root", type=Path, required=True)
    prepare.add_argument("--parametric_null-root", type=Path, required=True)
    prepare.add_argument("--trait-slug", choices=tuple(SLUG_TRAITS), required=True)
    prepare.add_argument("--transform-output", type=Path, required=True)
    prepare.add_argument("--receipt-output", type=Path, required=True)

    generate = commands.add_parser("generate")
    generate.add_argument("--manifest", type=Path, required=True)
    generate.add_argument("--fam", type=Path, required=True)
    generate.add_argument("--transform", type=Path, required=True)
    generate.add_argument("--transform-receipt", type=Path, required=True)
    generate.add_argument("--trait-slug", choices=tuple(SLUG_TRAITS), required=True)
    generate.add_argument("--replicate", type=int, required=True)
    generate.add_argument("--phenotype-output", type=Path, required=True)
    generate.add_argument("--receipt-output", type=Path, required=True)

    qualify = commands.add_parser("qualify-map")
    qualify.add_argument("--manifest", type=Path, required=True)
    qualify.add_argument("--parametric_null-contract", type=Path, required=True)
    qualify.add_argument("--bim", type=Path, required=True)
    qualify.add_argument("--bfile-prefix", type=Path, required=True)
    qualify.add_argument("--qcovar", type=Path, required=True)
    qualify.add_argument("--map", type=Path, required=True)
    qualify.add_argument("--gcta-log", type=Path, required=True)
    qualify.add_argument("--gcta-output-prefix", type=Path, required=True)
    qualify.add_argument("--phenotype", type=Path, required=True)
    qualify.add_argument("--phenotype-receipt", type=Path, required=True)
    qualify.add_argument("--trait-slug", choices=tuple(SLUG_TRAITS), required=True)
    qualify.add_argument("--replicate", type=int, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    qualify.add_argument("--retained-map", type=Path)

    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--manifest", type=Path, required=True)
    aggregate_parser.add_argument("--parametric_null-root", type=Path, required=True)
    aggregate_parser.add_argument("--checkpoint-root", type=Path, required=True)
    aggregate_parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "contract-summary":
            result = contract_summary(args.manifest, args.output)
        elif args.command == "verify-sources":
            result = verify_sources(
                args.manifest,
                args.baseline_root,
                args.calibration_root,
                args.parametric_null_root,
                args.repository_root,
                args.output,
            )
        elif args.command == "prepare":
            result = prepare_transform(
                args.manifest,
                args.baseline_root,
                args.calibration_root,
                args.parametric_null_root,
                args.trait_slug,
                args.transform_output,
                args.receipt_output,
            )
        elif args.command == "generate":
            result = generate_phenotype(
                args.manifest,
                args.fam,
                args.transform,
                args.transform_receipt,
                args.trait_slug,
                args.replicate,
                args.phenotype_output,
                args.receipt_output,
            )
        elif args.command == "qualify-map":
            result = qualify_map(
                args.manifest,
                args.parametric_null_contract,
                args.bim,
                args.bfile_prefix,
                args.qcovar,
                args.map,
                args.gcta_log,
                args.gcta_output_prefix,
                args.phenotype,
                args.phenotype_receipt,
                args.trait_slug,
                args.replicate,
                args.output,
                args.retained_map,
            )
        else:
            result = aggregate(
                args.manifest,
                args.parametric_null_root,
                args.checkpoint_root,
                args.mode,
                args.output,
            )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
