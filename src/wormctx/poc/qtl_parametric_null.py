"""Frozen model-specific parametric polygenic-null calibration for WS283.

The calibration simulates phenotypes from fitted null mixed models and reruns
the predeclared association models.  It is deliberately model-specific: null
replicates calibrate the model that generated them and are never pooled across
relationship matrices, PC endpoints, or traits.
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
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .qtl import MLMA_HEADER, TRAITS, TRAIT_SLUGS
from .qtl_grm_sensitivity import (
    CHROMOSOMES,
    FrozenFile,
    ToolContract,
    _accepted_options,
    _path_below,
    _qualify_log_text,
    _read_fam,
    _read_float32,
    _sha256,
    _verify_frozen_files,
    _verify_internal_checksums,
)


SCHEMA_VERSION = "wormctx-abamectin-ws283-parametric-polygenic-null-1.0"
FIT_VERSION = "wormctx-abamectin-ws283-parametric-polygenic-null-fit-1.0"
PHENOTYPE_VERSION = "wormctx-abamectin-ws283-parametric-polygenic-null-phenotype-1.0"
MAP_VERSION = "wormctx-abamectin-ws283-parametric-polygenic-null-map-1.0"
SUMMARY_VERSION = "wormctx-abamectin-ws283-parametric-polygenic-null-summary-1.0"

RELATIONSHIP_KINDS = ("full", "ldpruned")
ENDPOINTS = ("pc0", "pc10")
REPLICATES_PER_CELL = 100
SENTINEL_REPLICATES = (1, 25, 50, 75, 100)
CELL_COUNT = len(RELATIONSHIP_KINDS) * len(ENDPOINTS) * len(TRAITS)
TOTAL_NULL_MAPS = CELL_COUNT * REPLICATES_PER_CELL
EXPECTED_BOUNDARY_CELLS = ("ldpruned_pc0_norm_n", "ldpruned_pc10_norm_n")
ASSOCIATION_MARKERS_BY_CHROMOSOME = {
    1: 31023,
    2: 76840,
    3: 55886,
    4: 53961,
    5: 115183,
    6: 40386,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
)
_CHI_SQUARE_1_MEDIAN = 0.4549364231195727


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ParentContract(_StrictModel):
    role: Literal["baseline", "calibration", "sensitivity"]
    run_id: str
    source_git_commit: str
    checksum_file_count: int = Field(gt=0)
    checksum_receipt_relative_path: Literal["receipts/SHA256SUMS.txt"]
    source_revision_relative_path: Literal["receipts/SOURCE_REVISION"]
    required_files: list[FrozenFile]

    @field_validator("source_git_commit")
    @classmethod
    def valid_commit(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("source_git_commit must be 40 lowercase hexadecimal characters")
        return value

    @model_validator(mode="after")
    def exact_parent_identity(self) -> "ParentContract":
        expected = {
            "baseline": (
                "abamectin-ws283-bd41637-20260717T200012Z",
                "bd41637f5ff2998333964cc954ee9f9f36a351d6",
                44,
            ),
            "calibration": (
                "abamectin-ws283-pc-calibration-20260720T215332Z-192322baefdf",
                "192322baefdfa246a8422de17e5a380fa7c5d93f",
                150,
            ),
            "sensitivity": (
                "abamectin-ws283-ldpruned-grm-sensitivity-20260720T231705Z-df9ae7333925",
                "df9ae7333925b481175fd93e171b5e53c4e8170b",
                315,
            ),
        }
        if (self.run_id, self.source_git_commit, self.checksum_file_count) != expected[self.role]:
            raise ValueError(f"{self.role} parent identity differs from the frozen contract")
        paths = [item.relative_path for item in self.required_files]
        if len(paths) != len(set(paths)):
            raise ValueError("parent required-file paths must be unique")
        mandatory = {
            "SUCCESS",
            self.checksum_receipt_relative_path,
            self.source_revision_relative_path,
        }
        if not mandatory.issubset(paths):
            raise ValueError(f"{self.role} parent omits a mandatory frozen file")
        return self


class RelationshipModelContract(_StrictModel):
    id: Literal["full", "ldpruned"]
    marker_count: Literal[373279, 1370]
    anchor_method: Literal["GCTA_make_grm_algorithm_0"]
    association_method: Literal["GCTA_MLMA_LOCO", "GCTA_MLMA_external_LOCO_GRM"]
    observed_parent: Literal["calibration", "sensitivity"]
    observed_model_prefix: Literal["pc", "ldgrm_pc"]

    @model_validator(mode="after")
    def exact_relationship_lane(self) -> "RelationshipModelContract":
        expected = {
            "full": (373279, "GCTA_MLMA_LOCO", "calibration", "pc"),
            "ldpruned": (
                1370,
                "GCTA_MLMA_external_LOCO_GRM",
                "sensitivity",
                "ldgrm_pc",
            ),
        }
        observed = (
            self.marker_count,
            self.association_method,
            self.observed_parent,
            self.observed_model_prefix,
        )
        if observed != expected[self.id]:
            raise ValueError(f"relationship lane {self.id} differs from the frozen contract")
        return self


class EndpointContract(_StrictModel):
    id: Literal["pc0", "pc10"]
    pc_count: Literal[0, 10]
    design_terms: list[str]
    qcovar_parent: Literal["calibration"] | None
    qcovar_relative_path: Literal["genotype/qcovars/pc10.qcovar"] | None

    @model_validator(mode="after")
    def exact_endpoint(self) -> "EndpointContract":
        expected_terms = ["intercept"] + (
            [f"PC{index}" for index in range(1, 11)] if self.id == "pc10" else []
        )
        if self.design_terms != expected_terms:
            raise ValueError(f"{self.id} design terms differ from the frozen contract")
        expected = (
            (0, None, None)
            if self.id == "pc0"
            else (10, "calibration", "genotype/qcovars/pc10.qcovar")
        )
        if (self.pc_count, self.qcovar_parent, self.qcovar_relative_path) != expected:
            raise ValueError(f"{self.id} covariate contract differs from the frozen contract")
        return self


class RemlContract(_StrictModel):
    method: Literal["GCTA_GREML"]
    constrained: Literal[True]
    estimate_fixed_effects: Literal[True]
    fixed_effect_reconstruction: Literal[
        "GCTA_hsq_reml_est_fix_verified_by_independent_GLS"
    ]
    fit_count: Literal[16]
    thread_count: Literal[16]
    phenotype_samples: Literal[209]
    variance_identity_absolute_tolerance: float = Field(gt=0.0, le=1e-6)
    fixed_effect_gls_absolute_tolerance: Literal[0.01]
    expected_boundary_cells: list[str]
    boundary_policy: Literal[
        "retain_expected_gcta_residual_floor_as_descriptive_stress_test"
    ]

    @model_validator(mode="after")
    def exact_boundary_preflight(self) -> "RemlContract":
        if self.expected_boundary_cells != list(EXPECTED_BOUNDARY_CELLS):
            raise ValueError("expected anchor boundary cells differ from the frozen preflight")
        return self


class SmokeGateContract(_StrictModel):
    replicates: list[int]
    required_cells: Literal[16]
    required_checkpoints: Literal[32]

    @model_validator(mode="after")
    def exact_smoke_gate(self) -> "SmokeGateContract":
        if self.replicates != [1, 2]:
            raise ValueError("smoke gate must run null replicates 1 and 2 in order")
        return self


class BootstrapContract(_StrictModel):
    replicates_per_cell: Literal[100]
    cell_count: Literal[16]
    total_null_maps: Literal[1600]
    random_generator: Literal["numpy.PCG64DXSM"]
    seed_derivation: Literal["SHA256_length_delimited_v1"]
    master_seed_sha256: str
    common_random_numbers: Literal[True]
    shared_stream_key_fields: list[str]
    stream_components: list[str]
    sentinel_replicates: list[int]
    phenotype_formula: Literal[
        "y=Xbeta+sqrt(Vg)*Khalf*zg+sqrt(Ve)*ze"
    ]
    phenotype_float_format: Literal[".17g"]
    byte_replay_required: Literal[True]
    empirical_p_correction: Literal["plus_one"]
    min_p_fwer_order_statistic: Literal[5]
    pooled_thresholds_permitted: Literal[False]
    smoke_gate: SmokeGateContract

    @field_validator("master_seed_sha256")
    @classmethod
    def valid_seed(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("master_seed_sha256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def exact_bootstrap(self) -> "BootstrapContract":
        if self.shared_stream_key_fields != ["trait_slug", "replicate", "component"]:
            raise ValueError("shared stream key must exclude relationship kind and PC endpoint")
        if self.stream_components != ["genetic", "residual"]:
            raise ValueError("bootstrap must have genetic and residual streams")
        if self.sentinel_replicates != list(SENTINEL_REPLICATES):
            raise ValueError("sentinel replicates differ from the frozen contract")
        if self.smoke_gate.required_checkpoints != (
            self.smoke_gate.required_cells * len(self.smoke_gate.replicates)
        ):
            raise ValueError("smoke gate checkpoint count is not cells times replicates")
        return self


class DiagnosticContract(_StrictModel):
    alpha: Literal[0.05]
    marker_count: Literal[373279]
    marker_bonferroni_p: float
    lambda_exclusions: list[str]
    qq_quantiles: list[float]
    clump_p2: Literal[0.05]
    clump_r2: Literal[0.2]
    clump_kb: Literal[1000]

    @model_validator(mode="after")
    def exact_diagnostics(self) -> "DiagnosticContract":
        if not math.isclose(
            self.marker_bonferroni_p, 0.05 / 373279, rel_tol=0.0, abs_tol=1e-20
        ):
            raise ValueError("marker Bonferroni threshold must equal 0.05/373279")
        if self.lambda_exclusions != [
            "none",
            "trait_published_intervals",
            "all_published_intervals",
        ]:
            raise ValueError("lambda exclusions differ from the frozen contract")
        if self.qq_quantiles != [0.5, 0.9, 0.95, 0.99, 0.999]:
            raise ValueError("QQ quantiles differ from the frozen contract")
        return self


class ClaimContract(_StrictModel):
    permitted: list[str]
    prohibited: list[str]


class ParametricNullManifest(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    analysis_id: Literal["abamectin_qtl_ws283_parametric_polygenic_null_v1"]
    classification: Literal["preliminary_model_specific_parametric_null_calibration"]
    status: Literal["predeclared_preliminary_calibration"]
    validated: Literal[False]
    biological_claims_permitted: Literal[False]
    parents: list[ParentContract]
    tools: list[ToolContract]
    relationship_models: list[RelationshipModelContract]
    endpoints: list[EndpointContract]
    traits: list[str]
    reml: RemlContract
    bootstrap: BootstrapContract
    diagnostics: DiagnosticContract
    published_intervals: list[dict[str, Any]]
    claims: ClaimContract

    @model_validator(mode="after")
    def exact_design(self) -> "ParametricNullManifest":
        if [item.role for item in self.parents] != ["baseline", "calibration", "sensitivity"]:
            raise ValueError("parents must be ordered baseline, calibration, sensitivity")
        if [item.name for item in self.tools] != ["plink2", "gcta64"]:
            raise ValueError("tools must be ordered plink2, gcta64")
        if [item.id for item in self.relationship_models] != list(RELATIONSHIP_KINDS):
            raise ValueError("relationship models must be ordered full, ldpruned")
        if [item.id for item in self.endpoints] != list(ENDPOINTS):
            raise ValueError("endpoints must be ordered pc0, pc10")
        if self.traits != list(TRAITS):
            raise ValueError("traits differ from the four frozen abamectin traits")
        prohibited = " ".join(self.claims.prohibited).lower()
        if "causal" not in prohibited or "model selection" not in prohibited:
            raise ValueError("claim boundary must prohibit causal claims and model selection")
        _validate_interval_payload(self.published_intervals)
        return self


def _validate_interval_payload(intervals: list[dict[str, Any]]) -> None:
    expected_keys = {"id", "trait", "chromosome", "start", "end"}
    seen: set[str] = set()
    for item in intervals:
        if set(item) != expected_keys:
            raise ValueError(
                "each published interval must contain id, trait, chromosome, start, end"
            )
        if (
            not isinstance(item["id"], str)
            or not _SAFE_TOKEN_RE.fullmatch(item["id"])
            or item["id"] in seen
            or item["trait"] not in TRAITS
            or item["chromosome"] not in CHROMOSOMES
            or not isinstance(item["start"], int)
            or not isinstance(item["end"], int)
            or item["start"] <= 0
            or item["end"] < item["start"]
        ):
            raise ValueError("published interval payload is invalid")
        seen.add(item["id"])
    if len(intervals) != 6:
        raise ValueError("the frozen design must contain the six File S3 intervals")


def load_manifest(path: str | Path) -> ParametricNullManifest:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"parametric-null manifest is not a file: {source}")
    return ParametricNullManifest.model_validate_json(source.read_text(encoding="utf-8"))


def canonical_kind(value: str) -> Literal["full", "ldpruned"]:
    aliases = {"dense": "full", "full": "full", "ld": "ldpruned", "ldpruned": "ldpruned"}
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(f"unknown relationship kind: {value}") from exc


def trait_from_slug(slug: str) -> str:
    matches = [trait for trait, candidate in TRAIT_SLUGS.items() if candidate == slug]
    if len(matches) != 1:
        raise ValueError(f"unknown trait slug: {slug}")
    return matches[0]


def cell_id(kind: str, endpoint: str, trait: str) -> str:
    normalized = canonical_kind(kind)
    if endpoint not in ENDPOINTS or trait not in TRAITS:
        raise ValueError("invalid bootstrap cell coordinates")
    return f"{normalized}_{endpoint}_{TRAIT_SLUGS[trait]}"


def iter_cells(manifest: ParametricNullManifest) -> list[dict[str, Any]]:
    cells = [
        {
            "id": cell_id(kind.id, endpoint.id, trait),
            "relationship_kind": kind.id,
            "endpoint": endpoint.id,
            "pc_count": endpoint.pc_count,
            "trait": trait,
            "trait_slug": TRAIT_SLUGS[trait],
        }
        for kind in manifest.relationship_models
        for endpoint in manifest.endpoints
        for trait in manifest.traits
    ]
    if len(cells) != CELL_COUNT or len({item["id"] for item in cells}) != CELL_COUNT:
        raise ValueError("manifest does not derive exactly 16 unique bootstrap cells")
    return cells


def _write_json_atomic(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"output must not already exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _emit_json(payload: Any, output: str | Path | None) -> None:
    if output is None:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    else:
        _write_json_atomic(Path(output).resolve(), payload)


def _parent_by_role(manifest: ParametricNullManifest, role: str) -> ParentContract:
    matches = [parent for parent in manifest.parents if parent.role == role]
    if len(matches) != 1:
        raise ValueError(f"manifest lacks exactly one {role} parent")
    return matches[0]


def verify_parents(
    manifest_path: str | Path,
    baseline_parent: str | Path,
    calibration_parent: str | Path,
    sensitivity_parent: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Verify all three immutable parent trees and their internal receipts."""

    manifest_source = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_source)
    roots = {
        "baseline": Path(baseline_parent).resolve(),
        "calibration": Path(calibration_parent).resolve(),
        "sensitivity": Path(sensitivity_parent).resolve(),
    }
    parent_receipts: list[dict[str, Any]] = []
    for role in ("baseline", "calibration", "sensitivity"):
        contract = _parent_by_role(manifest, role)
        root = roots[role]
        if not root.is_dir() or root.name != contract.run_id:
            raise FileNotFoundError(f"{role} parent root identity is wrong: {root}")
        verified = _verify_frozen_files(root, contract.required_files)
        checksum = _verify_internal_checksums(
            root,
            contract.checksum_receipt_relative_path,
            contract.checksum_file_count,
        )
        success = _path_below(root, "SUCCESS")
        if success.read_bytes() != b"SUCCESS\n":
            raise ValueError(f"{role} parent terminal marker is not exact SUCCESS")
        revision = _path_below(root, contract.source_revision_relative_path)
        if revision.read_text(encoding="utf-8").strip() != contract.source_git_commit:
            raise ValueError(f"{role} parent source revision differs from the contract")
        parent_receipts.append(
            {
                "role": role,
                "run_id": contract.run_id,
                "root": str(root),
                "source_git_commit": contract.source_git_commit,
                "required_files_verified": len(verified),
                "required_file_inventory_sha256": hashlib.sha256(
                    json.dumps(verified, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "internal_checksum_receipt": checksum,
                "success_sha256": _sha256(success),
                "source_revision_sha256": _sha256(revision),
            }
        )
    receipt = {
        "schema_version": "wormctx-abamectin-ws283-parametric-null-parent-verification-1.0",
        "manifest_sha256": _sha256(manifest_source),
        "parents": parent_receipts,
        "all_three_immutable_parents_verified": True,
    }
    if output is not None:
        _write_json_atomic(Path(output).resolve(), receipt)
    return receipt


@dataclass(frozen=True)
class GctaGrm:
    """A structurally verified dense GCTA GRM in exact FAM order."""

    prefix: Path
    ids: tuple[tuple[str, str], ...]
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    expected_markers: int
    receipt: dict[str, Any]

    def square_root_times(self, values: np.ndarray) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (len(self.ids),):
            raise ValueError("GRM square-root input has the wrong shape")
        clipped = np.maximum(self.eigenvalues, 0.0)
        return self.eigenvectors @ (np.sqrt(clipped) * vector)


def read_gcta_grm(
    prefix: str | Path,
    fam_path: str | Path,
    expected_markers: int,
    *,
    expected_samples: int | None = None,
    expected_hashes: dict[str, str] | None = None,
) -> GctaGrm:
    """Read and qualify GCTA lower-triangle float32 files without reordering IDs."""

    if expected_markers <= 0:
        raise ValueError("expected GRM marker count must be positive")
    fam = Path(fam_path).resolve()
    samples = _read_fam(fam, expected_samples or 209)
    grm_prefix = Path(prefix).resolve()
    paths = {
        "grm_bin": Path(f"{grm_prefix}.grm.bin"),
        "grm_n_bin": Path(f"{grm_prefix}.grm.N.bin"),
        "grm_id": Path(f"{grm_prefix}.grm.id"),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {role} file: {path}")
        if expected_hashes is not None:
            expected = expected_hashes.get(role)
            if expected is None or not _SHA256_RE.fullmatch(expected):
                raise ValueError(f"expected hash contract is missing {role}")
            if _sha256(path) != expected:
                raise ValueError(f"{role} identity differs from the frozen hash")
    ids: list[tuple[str, str]] = []
    for line_number, line in enumerate(paths["grm_id"].read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"GRM ID line {line_number} must contain FID and IID")
        ids.append((fields[0], fields[1]))
    if ids != samples:
        raise ValueError("GRM IDs/order differ from the exact FAM order")
    entries = len(samples) * (len(samples) + 1) // 2
    lower = _read_float32(paths["grm_bin"], entries)
    marker_counts = _read_float32(paths["grm_n_bin"], entries)
    if any(
        not math.isclose(value, expected_markers, rel_tol=0.0, abs_tol=1e-3)
        for value in marker_counts
    ):
        raise ValueError("GRM N binary does not bind every pair to the exact marker count")
    matrix = np.empty((len(samples), len(samples)), dtype=np.float64)
    cursor = 0
    for row in range(len(samples)):
        for column in range(row + 1):
            matrix[row, column] = lower[cursor]
            matrix[column, row] = lower[cursor]
            cursor += 1
    if not np.array_equal(matrix, matrix.T) or not np.all(np.isfinite(matrix)):
        raise ValueError("GRM is not finite and exactly symmetric")
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0.0):
        raise ValueError("GRM contains a non-positive diagonal")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = max(1e-5, spectral_scale * 1e-5)
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError(
            f"GRM is not PSD within float32 tolerance: {eigenvalues[0]} < {-tolerance}"
        )
    receipt = {
        "schema_version": "wormctx-gcta-grm-matrix-qualification-1.0",
        "prefix": str(grm_prefix),
        "fam_sha256": _sha256(fam),
        "samples": len(samples),
        "iid_order_sha256": hashlib.sha256(
            "".join(f"{fid}\t{iid}\n" for fid, iid in samples).encode("utf-8")
        ).hexdigest(),
        "expected_markers": expected_markers,
        "pairwise_marker_count_min": int(min(marker_counts)),
        "pairwise_marker_count_max": int(max(marker_counts)),
        "triangular_float32_entries": entries,
        "grm_bin_sha256": _sha256(paths["grm_bin"]),
        "grm_n_bin_sha256": _sha256(paths["grm_n_bin"]),
        "grm_id_sha256": _sha256(paths["grm_id"]),
        "diagonal_min": float(np.min(diagonal)),
        "diagonal_max": float(np.max(diagonal)),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "psd_tolerance": tolerance,
        "positive_semidefinite_within_float32_tolerance": True,
        "numpy_version": np.__version__,
        "k_half_eigen_factor_sha256": hashlib.sha256(
            eigenvectors.astype("<f8", copy=False).tobytes()
            + np.sqrt(np.maximum(eigenvalues, 0.0)).astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "k_half_eigen_factor_encoding": (
            "row-major little-endian float64 eigenvectors followed by clipped sqrt eigenvalues"
        ),
    }
    return GctaGrm(
        prefix=grm_prefix,
        ids=tuple(ids),
        matrix=matrix,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        expected_markers=expected_markers,
        receipt=receipt,
    )


def verify_grm(
    manifest_path: str | Path,
    fam_path: str | Path,
    grm_prefix: str | Path,
    kind: str,
    expected_markers: int,
    output: str | Path | None = None,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_source)
    normalized = canonical_kind(kind)
    contract = next(item for item in manifest.relationship_models if item.id == normalized)
    if expected_markers != contract.marker_count:
        raise ValueError("expected marker count differs from the relationship contract")
    grm = read_gcta_grm(
        grm_prefix,
        fam_path,
        expected_markers,
        expected_samples=manifest.reml.phenotype_samples,
    )
    log_path = Path(f"{Path(grm_prefix).resolve()}.log")
    if not log_path.is_file():
        raise FileNotFoundError(f"global GRM construction log is missing: {log_path}")
    log_text = log_path.read_text(encoding="utf-8")
    warnings = _qualify_log_text(log_text, log_path)
    options = _accepted_options(log_text, log_path)
    expected_options = {
        "--bfile",
        "--autosome-num",
        "--autosome",
        "--make-grm",
        "--make-grm-alg",
        "--thread-num",
        "--out",
    }
    if normalized == "ldpruned":
        expected_options.add("--extract")
    if set(options) != expected_options:
        raise ValueError("global GRM accepted options differ from the frozen contract")
    if options["--autosome"] is not None or options["--make-grm"] is not None:
        raise ValueError("global GRM flag-only options unexpectedly have values")
    if (
        options["--autosome-num"] != "6"
        or options["--make-grm-alg"] != "0"
        or options["--thread-num"] != "16"
        or options["--out"] is None
        or Path(options["--out"]).resolve() != Path(grm_prefix).resolve()
    ):
        raise ValueError("global GRM numeric/output options differ from the frozen contract")
    bfile = options["--bfile"]
    if bfile is None or not bfile.replace("\\", "/").endswith(
        "/genotype/abamectin_209_qc"
    ):
        raise ValueError("global GRM does not use the frozen baseline BFILE")
    if normalized == "ldpruned":
        extract = options["--extract"]
        if extract is None or not extract.replace("\\", "/").endswith(
            "/genotype/ws283_ld_pruned.prune.in"
        ):
            raise ValueError("LD-pruned global GRM does not use the frozen prune list")
    if f"{expected_markers} SNPs" not in log_text or "209 individuals" not in log_text:
        raise ValueError("global GRM log lacks exact marker/sample evidence")
    receipt = {
        **grm.receipt,
        "manifest_sha256": _sha256(manifest_source),
        "relationship_kind": normalized,
        "construction_log": {
            "path": str(log_path),
            "sha256": _sha256(log_path),
            "accepted_options": options,
            "warnings": warnings,
        },
    }
    if output is not None:
        _write_json_atomic(Path(output).resolve(), receipt)
    return receipt


def derive_stream_seed(
    master_seed_sha256: str,
    trait_slug: str,
    replicate: int,
    component: Literal["genetic", "residual"],
) -> dict[str, Any]:
    """Derive a PCG64DXSM seed; kind/endpoint are intentionally absent."""

    if not _SHA256_RE.fullmatch(master_seed_sha256):
        raise ValueError("master seed is not a lowercase SHA-256 digest")
    trait_from_slug(trait_slug)
    if not 1 <= replicate <= REPLICATES_PER_CELL:
        raise ValueError("replicate is outside 1..100")
    if component not in {"genetic", "residual"}:
        raise ValueError("unknown bootstrap stream component")
    parts = (
        b"wormctx-parametric-null-seed-v1",
        bytes.fromhex(master_seed_sha256),
        trait_slug.encode("utf-8"),
        str(replicate).encode("ascii"),
        component.encode("ascii"),
    )
    material = b"".join(len(part).to_bytes(8, "big") + part for part in parts)
    digest = hashlib.sha256(material).digest()
    return {
        "derivation": "SHA256_length_delimited_v1",
        "generator": "numpy.PCG64DXSM",
        "shared_stream_group": f"{trait_slug}/replicate-{replicate:03d}",
        "component": component,
        "seed_sha256": digest.hex(),
        "seed_integer": int.from_bytes(digest, "big"),
        "kind_and_endpoint_excluded": True,
    }


def _rng_from_seed(seed: dict[str, Any]) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(seed["seed_integer"]))


def _read_ordered_phenotype(
    path: Path, samples: Sequence[tuple[str, str]]
) -> np.ndarray:
    rows = path.read_text(encoding="utf-8-sig").splitlines()
    if len(rows) != len(samples):
        raise ValueError("phenotype row count differs from the exact FAM cohort")
    values = np.empty(len(samples), dtype=np.float64)
    for index, (line, expected_ids) in enumerate(zip(rows, samples, strict=True), 1):
        fields = line.split()
        if len(fields) != 3 or tuple(fields[:2]) != expected_ids:
            raise ValueError(f"phenotype IDs/order differ from FAM at row {index}")
        try:
            value = float(fields[2])
        except ValueError as exc:
            raise ValueError(f"phenotype row {index} is not numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"phenotype row {index} is not finite")
        values[index - 1] = value
    return values


def _read_design_matrix(
    samples: Sequence[tuple[str, str]],
    pc_count: int,
    qcovar_path: Path | None,
) -> tuple[np.ndarray, list[str], str | None]:
    if pc_count not in {0, 10}:
        raise ValueError("only the frozen PC0 and PC10 endpoints are permitted")
    if pc_count == 0:
        if qcovar_path is not None:
            raise ValueError("PC0 design must omit qcovar")
        return np.ones((len(samples), 1), dtype=np.float64), ["intercept"], None
    if qcovar_path is None or not qcovar_path.is_file():
        raise FileNotFoundError("PC10 design requires the frozen qcovar file")
    rows = qcovar_path.read_text(encoding="utf-8-sig").splitlines()
    if len(rows) != len(samples):
        raise ValueError("qcovar row count differs from the exact FAM cohort")
    matrix = np.ones((len(samples), pc_count + 1), dtype=np.float64)
    for index, (line, expected_ids) in enumerate(zip(rows, samples, strict=True), 1):
        fields = line.split()
        if len(fields) != pc_count + 2 or tuple(fields[:2]) != expected_ids:
            raise ValueError(f"qcovar IDs/order or column count differ at row {index}")
        try:
            values = [float(value) for value in fields[2:]]
        except ValueError as exc:
            raise ValueError(f"qcovar row {index} is not numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"qcovar row {index} is not finite")
        matrix[index - 1, 1:] = values
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError("fixed-effect design matrix is rank deficient")
    return matrix, ["intercept", *[f"PC{index}" for index in range(1, 11)]], _sha256(
        qcovar_path
    )


@dataclass(frozen=True)
class RemlComponents:
    variance_genetic: float
    variance_residual: float
    variance_phenotype: float
    heritability: float
    log_likelihood: float
    null_log_likelihood: float
    likelihood_ratio: float
    degrees_of_freedom: int
    p_value: float
    sample_count: int
    standard_errors: dict[str, float]
    fixed_effects: tuple[float, ...]
    fixed_effect_standard_errors: tuple[float, ...]


def parse_reml_hsq(
    path: str | Path,
    expected_samples: int,
    expected_fixed_effect_count: int | None = None,
) -> RemlComponents:
    """Parse GCTA 1.94.1 one-GRM estimates and ``--reml-est-fix`` tail."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"GCTA REML output is not a file: {source}")
    raw_lines = source.read_text(encoding="utf-8-sig").splitlines()
    if not raw_lines:
        raise ValueError("GCTA .hsq output is empty")
    blank_indexes = [index for index, line in enumerate(raw_lines) if not line.strip()]
    if blank_indexes != [11]:
        raise ValueError("GCTA .hsq must contain one blank before its fixed-effect tail")
    raw_rows = [line.split() for line in raw_lines if line.strip()]
    if not raw_rows or raw_rows[0] != ["Source", "Variance", "SE"]:
        raise ValueError("GCTA .hsq header must be Source, Variance, SE")
    expected_sources = (
        "V(G)",
        "V(e)",
        "Vp",
        "V(G)/Vp",
        "logL",
        "logL0",
        "LRT",
        "df",
        "Pval",
        "n",
    )
    source_rows = raw_rows[1:11]
    if [row[0] for row in source_rows] != list(expected_sources):
        raise ValueError("GCTA .hsq rows differ from the exact extended REML contract")
    if len(raw_rows) < 13 or raw_rows[11] != ["Fix_eff", "SE"]:
        raise ValueError("GCTA .hsq lacks the exact --reml-est-fix tail header")
    fixed_rows = raw_rows[12:]
    if expected_fixed_effect_count is not None and len(fixed_rows) != expected_fixed_effect_count:
        raise ValueError("GCTA .hsq fixed-effect count differs from the endpoint design")
    if len(fixed_rows) not in {1, 11}:
        raise ValueError("GCTA .hsq must contain one or eleven fixed-effect rows")
    values: dict[str, float] = {}
    standard_errors: dict[str, float] = {}
    for row in source_rows:
        if len(row) not in {2, 3} or not _NUMBER_RE.fullmatch(row[1]):
            raise ValueError(f"malformed GCTA .hsq row: {row}")
        value = float(row[1])
        if not math.isfinite(value):
            raise ValueError(f"non-finite GCTA .hsq estimate: {row[0]}")
        values[row[0]] = value
        if len(row) == 3 and _NUMBER_RE.fullmatch(row[2]):
            standard_error = float(row[2])
            if not math.isfinite(standard_error) or standard_error < 0.0:
                raise ValueError(f"invalid GCTA .hsq standard error: {row[0]}")
            standard_errors[row[0]] = standard_error
    if values["V(G)"] <= 0.0 or values["V(e)"] <= 0.0 or values["Vp"] <= 0.0:
        raise ValueError("GCTA REML variance components are outside the constrained null model")
    if not 0.0 < values["V(G)/Vp"] < 1.0:
        raise ValueError("GCTA REML heritability lies on or outside a constrained boundary")
    if not math.isclose(values["n"], round(values["n"]), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("GCTA REML n is not an integer")
    sample_count = int(round(values["n"]))
    if sample_count != expected_samples:
        raise ValueError("GCTA REML sample count differs from the frozen cohort")
    if not math.isclose(values["df"], round(values["df"]), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("GCTA REML degrees of freedom is not an integer")
    fixed_effects: list[float] = []
    fixed_effect_standard_errors: list[float] = []
    for row in fixed_rows:
        if len(row) != 2 or not all(_NUMBER_RE.fullmatch(field) for field in row):
            raise ValueError("GCTA .hsq fixed-effect row must contain estimate and SE")
        estimate, standard_error = map(float, row)
        if (
            not math.isfinite(estimate)
            or not math.isfinite(standard_error)
            or standard_error < 0.0
        ):
            raise ValueError("GCTA .hsq fixed-effect estimate or SE is invalid")
        fixed_effects.append(estimate)
        fixed_effect_standard_errors.append(standard_error)
    return RemlComponents(
        variance_genetic=values["V(G)"],
        variance_residual=values["V(e)"],
        variance_phenotype=values["Vp"],
        heritability=values["V(G)/Vp"],
        log_likelihood=values["logL"],
        null_log_likelihood=values["logL0"],
        likelihood_ratio=values["LRT"],
        degrees_of_freedom=int(round(values["df"])),
        p_value=values["Pval"],
        sample_count=sample_count,
        standard_errors=standard_errors,
        fixed_effects=tuple(fixed_effects),
        fixed_effect_standard_errors=tuple(fixed_effect_standard_errors),
    )


def gls_fixed_effects(
    y: np.ndarray,
    design: np.ndarray,
    grm: np.ndarray,
    variance_genetic: float,
    variance_residual: float,
) -> tuple[np.ndarray, float]:
    """Reconstruct REML fixed effects as GLS under the fitted covariance."""

    y_vector = np.asarray(y, dtype=np.float64)
    x_matrix = np.asarray(design, dtype=np.float64)
    k_matrix = np.asarray(grm, dtype=np.float64)
    sample_count = y_vector.shape[0]
    if (
        y_vector.shape != (sample_count,)
        or x_matrix.ndim != 2
        or x_matrix.shape[0] != sample_count
        or k_matrix.shape != (sample_count, sample_count)
        or variance_genetic < 0.0
        or variance_residual <= 0.0
        or not np.all(np.isfinite(y_vector))
        or not np.all(np.isfinite(x_matrix))
        or not np.all(np.isfinite(k_matrix))
    ):
        raise ValueError("invalid GLS inputs")
    covariance = variance_genetic * k_matrix + variance_residual * np.eye(sample_count)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("fitted phenotype covariance is not positive definite")
    solved_x = np.linalg.solve(covariance, x_matrix)
    solved_y = np.linalg.solve(covariance, y_vector)
    information = x_matrix.T @ solved_x
    if np.linalg.matrix_rank(information) != information.shape[0]:
        raise ValueError("GLS fixed-effect information matrix is rank deficient")
    beta = np.linalg.solve(information, x_matrix.T @ solved_y)
    condition = float(np.linalg.cond(information))
    if not np.all(np.isfinite(beta)) or not math.isfinite(condition):
        raise ValueError("GLS fixed-effect reconstruction is not finite")
    return beta, condition


def _qualify_reml_log(
    log_path: Path,
    grm_prefix: Path,
    phenotype_path: Path,
    qcovar_path: Path | None,
    output_prefix: Path,
    endpoint: EndpointContract,
    thread_count: int,
    expected_boundary: bool,
) -> dict[str, Any]:
    if not log_path.is_file():
        raise FileNotFoundError(f"GCTA REML log is not a file: {log_path}")
    text = log_path.read_text(encoding="utf-8")
    warnings = _qualify_log_text(text, log_path)
    options = _accepted_options(text, log_path)
    expected_names = {
        "--reml",
        "--grm",
        "--pheno",
        "--reml-est-fix",
        "--thread-num",
        "--out",
    }
    if endpoint.pc_count:
        expected_names.add("--qcovar")
    if set(options) != expected_names:
        raise ValueError("GCTA REML accepted options differ from the frozen fit contract")
    if options["--reml"] is not None or options["--reml-est-fix"] is not None:
        raise ValueError("GCTA REML flag-only options unexpectedly have values")
    expected_values = {
        "--grm": str(grm_prefix),
        "--pheno": str(phenotype_path),
        "--thread-num": str(thread_count),
        "--out": str(output_prefix),
    }
    if qcovar_path is not None:
        expected_values["--qcovar"] = str(qcovar_path)
    for option, expected in expected_values.items():
        observed = options[option]
        if observed is None or Path(observed).resolve() != Path(expected).resolve():
            raise ValueError(f"GCTA REML accepted {option} differs from the exact input")
    if "--reml-no-constrain" in text:
        raise ValueError("unconstrained REML is prohibited by the frozen null contract")
    boundary_indicators = (
        "boundary",
        "fixed at zero",
        "fixed to zero",
        "constrained at zero",
        "constrained to zero",
        "lower bound",
        "component(s) constrained",
    )
    boundary_lines = [
        line.strip()
        for line in text.splitlines()
        if any(indicator in line.lower() for indicator in boundary_indicators)
    ]
    exact_constrained_component_lines = [
        line for line in boundary_lines if "(1 component(s) constrained)" in line.lower()
    ]
    if expected_boundary and (
        not exact_constrained_component_lines
        or len(exact_constrained_component_lines) != len(boundary_lines)
    ):
        raise ValueError(
            "expected GCTA residual-floor cell lacks one-or-more pure single-component "
            "constrained lines: "
            f"{boundary_lines}"
        )
    if not expected_boundary and boundary_lines:
        raise ValueError(
            "unexpected GCTA REML boundary or constrained-floor evidence: "
            f"{boundary_lines}"
        )
    convergence_lines = [line.strip() for line in text.splitlines() if "converg" in line.lower()]
    if not convergence_lines or any("not converg" in line.lower() for line in convergence_lines):
        raise ValueError("GCTA REML log lacks positive convergence evidence")
    return {
        "log_sha256": _sha256(log_path),
        "accepted_options": options,
        "convergence_evidence": convergence_lines,
        "warnings": warnings,
        "anchor_boundary": expected_boundary,
        "boundary_evidence": boundary_lines,
        "constrained_component_evidence_count": len(exact_constrained_component_lines),
    }


def qualify_fit(
    manifest_path: str | Path,
    fam_path: str | Path,
    phenotype_path: str | Path,
    grm_prefix: str | Path,
    hsq_path: str | Path,
    log_path: str | Path,
    grm_kind: str,
    endpoint_id: str,
    trait_slug: str,
    output: str | Path,
    qcovar_path: str | Path | None = None,
) -> dict[str, Any]:
    """Qualify one global GREML fit and reconstruct its fixed effects by GLS."""

    manifest_source = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_source)
    kind = canonical_kind(grm_kind)
    trait = trait_from_slug(trait_slug)
    endpoints = [item for item in manifest.endpoints if item.id == endpoint_id]
    if len(endpoints) != 1:
        raise ValueError("fit endpoint is not pc0 or pc10")
    endpoint = endpoints[0]
    relationship = next(item for item in manifest.relationship_models if item.id == kind)
    expected_cell = cell_id(kind, endpoint.id, trait)
    expected_boundary = expected_cell in manifest.reml.expected_boundary_cells
    fam = Path(fam_path).resolve()
    phenotype = Path(phenotype_path).resolve()
    qcovar = Path(qcovar_path).resolve() if qcovar_path is not None else None
    samples = _read_fam(fam, manifest.reml.phenotype_samples)
    y = _read_ordered_phenotype(phenotype, samples)
    design, terms, qcovar_sha256 = _read_design_matrix(samples, endpoint.pc_count, qcovar)
    if terms != endpoint.design_terms:
        raise ValueError("computed design terms differ from the endpoint contract")
    grm = read_gcta_grm(
        grm_prefix,
        fam,
        relationship.marker_count,
        expected_samples=manifest.reml.phenotype_samples,
    )
    hsq = Path(hsq_path).resolve()
    components = parse_reml_hsq(
        hsq, manifest.reml.phenotype_samples, len(terms)
    )
    tolerance = manifest.reml.variance_identity_absolute_tolerance
    variance_identity_error = abs(
        components.variance_phenotype
        - components.variance_genetic
        - components.variance_residual
    )
    # GCTA prints six decimals.  The epsilon makes the configured inclusive
    # 1e-6 decimal tolerance robust to binary float representation only.
    if variance_identity_error > tolerance + 1e-12:
        raise ValueError("GCTA Vp does not equal V(G)+V(e) within the frozen tolerance")
    expected_h2 = components.variance_genetic / components.variance_phenotype
    if not math.isclose(components.heritability, expected_h2, rel_tol=1e-4, abs_tol=1e-5):
        raise ValueError("GCTA heritability is inconsistent with the variance components")
    gls_beta, condition = gls_fixed_effects(
        y,
        design,
        grm.matrix,
        components.variance_genetic,
        components.variance_residual,
    )
    beta = np.asarray(components.fixed_effects, dtype=np.float64)
    fixed_effect_gls_max_absolute_error = float(np.max(np.abs(beta - gls_beta)))
    if not np.allclose(
        beta,
        gls_beta,
        rtol=0.0,
        atol=manifest.reml.fixed_effect_gls_absolute_tolerance,
    ):
        raise ValueError(
            "GCTA .hsq fixed effects differ from independent GLS reconstruction"
        )
    output_prefix = hsq.with_suffix("")
    log_receipt = _qualify_reml_log(
        Path(log_path).resolve(),
        Path(grm_prefix).resolve(),
        phenotype,
        qcovar,
        output_prefix,
        endpoint,
        manifest.reml.thread_count,
        expected_boundary,
    )
    receipt = {
        "schema_version": FIT_VERSION,
        "manifest_sha256": _sha256(manifest_source),
        "cell": {
            "id": expected_cell,
            "relationship_kind": kind,
            "endpoint": endpoint.id,
            "trait": trait,
            "trait_slug": trait_slug,
        },
        "inputs": {
            "fam": {"path": str(fam), "sha256": _sha256(fam)},
            "phenotype": {"path": str(phenotype), "sha256": _sha256(phenotype)},
            "qcovar": None
            if qcovar is None
            else {"path": str(qcovar), "sha256": qcovar_sha256},
            "grm": grm.receipt,
            "hsq": {"path": str(hsq), "sha256": _sha256(hsq)},
            "log": {"path": str(Path(log_path).resolve()), **log_receipt},
        },
        "sample_count": components.sample_count,
        "variance_components": {
            "Vg": components.variance_genetic,
            "Ve": components.variance_residual,
            "Vp": components.variance_phenotype,
            "h2": components.heritability,
            "log_likelihood": components.log_likelihood,
            "null_log_likelihood": components.null_log_likelihood,
            "likelihood_ratio": components.likelihood_ratio,
            "degrees_of_freedom": components.degrees_of_freedom,
            "p_value": components.p_value,
            "standard_errors": components.standard_errors,
            "Vp_minus_Vg_minus_Ve_absolute_error": variance_identity_error,
            "constrained": True,
        },
        "fixed_effects": [
            {
                "term": term,
                "estimate": float(value),
                "standard_error": float(standard_error),
            }
            for term, value, standard_error in zip(
                terms,
                beta,
                components.fixed_effect_standard_errors,
                strict=True,
            )
        ],
        "fixed_effect_source": "GCTA_hsq_reml_est_fix_verified_by_independent_GLS",
        "independent_gls_fixed_effects": [float(value) for value in gls_beta],
        "fixed_effect_gls_max_absolute_error": fixed_effect_gls_max_absolute_error,
        "fixed_effect_gls_absolute_tolerance": (
            manifest.reml.fixed_effect_gls_absolute_tolerance
        ),
        "fixed_effect_gls_tolerance_basis": (
            "prebootstrap anchor preflight: six-decimal GCTA variance-component output "
            "under PC10 information-matrix conditioning"
        ),
        "gls_information_condition_number": condition,
        "gcta_log_fixed_effects_used": False,
        "gcta_log_fixed_effect_note": (
            "canonical estimates parsed from exact .hsq tail and independently GLS-verified"
        ),
        "anchor_boundary": expected_boundary,
        "threshold_eligible": not expected_boundary,
        "boundary_policy": manifest.reml.boundary_policy,
        "boundary_criterion": {
            "Vg_strictly_positive": True,
            "Ve_strictly_positive": True,
            "heritability_strictly_between_zero_and_one": True,
            "boundary_expectation_matched": True,
            "expected_boundary_evidence_present": expected_boundary,
            "no_unexpected_boundary_or_constrained_floor_log_indicator": True,
        },
        "qualified": True,
    }
    _write_json_atomic(Path(output).resolve(), receipt)
    return receipt


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _validate_fit_receipt(
    payload: dict[str, Any],
    manifest_sha256: str,
    expected_cell: str,
    expected_grm: GctaGrm,
    terms: Sequence[str],
) -> tuple[float, float, np.ndarray]:
    if payload.get("schema_version") != FIT_VERSION or payload.get("qualified") is not True:
        raise ValueError("fit receipt is not a qualified parametric-null fit")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("fit receipt is bound to a different manifest")
    if payload.get("cell", {}).get("id") != expected_cell:
        raise ValueError("fit receipt is bound to a different bootstrap cell")
    expected_boundary = expected_cell in EXPECTED_BOUNDARY_CELLS
    if (
        payload.get("anchor_boundary") is not expected_boundary
        or payload.get("threshold_eligible") is not (not expected_boundary)
    ):
        raise ValueError("fit receipt boundary eligibility differs from the frozen preflight")
    input_grm = payload.get("inputs", {}).get("grm", {})
    for field in ("grm_bin_sha256", "grm_n_bin_sha256", "grm_id_sha256"):
        if input_grm.get(field) != expected_grm.receipt[field]:
            raise ValueError("fit receipt is bound to a different anchor GRM")
    components = payload.get("variance_components", {})
    try:
        vg = float(components["Vg"])
        ve = float(components["Ve"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("fit receipt lacks numeric Vg and Ve") from exc
    if not math.isfinite(vg) or not math.isfinite(ve) or vg <= 0.0 or ve <= 0.0:
        raise ValueError("fit receipt contains invalid variance components")
    fixed_effects = payload.get("fixed_effects")
    if not isinstance(fixed_effects, list) or [item.get("term") for item in fixed_effects] != list(
        terms
    ):
        raise ValueError("fit receipt fixed-effect terms differ from the design")
    try:
        beta = np.asarray([float(item["estimate"]) for item in fixed_effects], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("fit receipt fixed effects are not numeric") from exc
    if beta.shape != (len(terms),) or not np.all(np.isfinite(beta)):
        raise ValueError("fit receipt fixed effects are not finite")
    return vg, ve, beta


def simulate_phenotype_values(
    design: np.ndarray,
    beta: np.ndarray,
    grm: GctaGrm,
    variance_genetic: float,
    variance_residual: float,
    master_seed_sha256: str,
    trait_slug: str,
    replicate: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Simulate ``X beta + sqrt(Vg) Khalf zg + sqrt(Ve) ze`` exactly."""

    x_matrix = np.asarray(design, dtype=np.float64)
    beta_vector = np.asarray(beta, dtype=np.float64)
    if x_matrix.shape != (len(grm.ids), len(beta_vector)):
        raise ValueError("simulation design and beta shapes do not match the GRM")
    if variance_genetic < 0.0 or variance_residual <= 0.0:
        raise ValueError("simulation variance components are outside the constrained model")
    genetic_seed = derive_stream_seed(master_seed_sha256, trait_slug, replicate, "genetic")
    residual_seed = derive_stream_seed(master_seed_sha256, trait_slug, replicate, "residual")
    genetic_standard_normal = _rng_from_seed(genetic_seed).standard_normal(len(grm.ids))
    residual_standard_normal = _rng_from_seed(residual_seed).standard_normal(len(grm.ids))
    fixed = x_matrix @ beta_vector
    genetic = math.sqrt(variance_genetic) * grm.square_root_times(genetic_standard_normal)
    residual = math.sqrt(variance_residual) * residual_standard_normal
    phenotype = fixed + genetic + residual
    if not np.all(np.isfinite(phenotype)):
        raise ValueError("simulated phenotype contains a non-finite value")
    components = {
        "fixed_sha256": hashlib.sha256(fixed.astype("<f8", copy=False).tobytes()).hexdigest(),
        "genetic_sha256": hashlib.sha256(
            genetic.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "residual_sha256": hashlib.sha256(
            residual.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "phenotype_float64_sha256": hashlib.sha256(
            phenotype.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "stream_seeds": {"genetic": genetic_seed, "residual": residual_seed},
    }
    return phenotype, components


def _phenotype_bytes(
    ids: Sequence[tuple[str, str]], values: np.ndarray
) -> bytes:
    if values.shape != (len(ids),):
        raise ValueError("phenotype byte encoder received a mismatched value vector")
    text = "".join(
        f"{fid}\t{iid}\t{format(float(value), '.17g')}\n"
        for (fid, iid), value in zip(ids, values, strict=True)
    )
    return text.encode("utf-8")


def simulate(
    manifest_path: str | Path,
    fam_path: str | Path,
    grm_prefix: str | Path,
    fit_path: str | Path,
    grm_kind: str,
    endpoint_id: str,
    trait_slug: str,
    replicate: int,
    phenotype_output: str | Path,
    receipt_output: str | Path,
    qcovar_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate one deterministic null phenotype and prove byte replay."""

    manifest_source = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_source)
    manifest_sha256 = _sha256(manifest_source)
    kind = canonical_kind(grm_kind)
    trait = trait_from_slug(trait_slug)
    endpoint_matches = [item for item in manifest.endpoints if item.id == endpoint_id]
    if len(endpoint_matches) != 1:
        raise ValueError("simulation endpoint is not pc0 or pc10")
    endpoint = endpoint_matches[0]
    relationship = next(item for item in manifest.relationship_models if item.id == kind)
    fam = Path(fam_path).resolve()
    samples = _read_fam(fam, manifest.reml.phenotype_samples)
    qcovar = Path(qcovar_path).resolve() if qcovar_path is not None else None
    design, terms, qcovar_sha256 = _read_design_matrix(samples, endpoint.pc_count, qcovar)
    grm = read_gcta_grm(
        grm_prefix,
        fam,
        relationship.marker_count,
        expected_samples=manifest.reml.phenotype_samples,
    )
    fit_source = Path(fit_path).resolve()
    fit = _load_json_object(fit_source, "fit receipt")
    expected_cell = cell_id(kind, endpoint.id, trait)
    vg, ve, beta = _validate_fit_receipt(
        fit, manifest_sha256, expected_cell, grm, terms
    )
    values, components = simulate_phenotype_values(
        design,
        beta,
        grm,
        vg,
        ve,
        manifest.bootstrap.master_seed_sha256,
        trait_slug,
        replicate,
    )
    replay_values, replay_components = simulate_phenotype_values(
        design,
        beta,
        grm,
        vg,
        ve,
        manifest.bootstrap.master_seed_sha256,
        trait_slug,
        replicate,
    )
    phenotype_bytes = _phenotype_bytes(grm.ids, values)
    replay_bytes = _phenotype_bytes(grm.ids, replay_values)
    if (
        not np.array_equal(values, replay_values)
        or components != replay_components
        or phenotype_bytes != replay_bytes
    ):
        raise RuntimeError("deterministic phenotype byte replay failed")
    phenotype_destination = Path(phenotype_output).resolve()
    receipt_destination = Path(receipt_output).resolve()
    if phenotype_destination.exists() or receipt_destination.exists():
        raise FileExistsError("phenotype and receipt outputs must not already exist")
    phenotype_destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_destination.parent.mkdir(parents=True, exist_ok=True)
    phenotype_sha256 = hashlib.sha256(phenotype_bytes).hexdigest()
    binding_payload = {
        "cell_id": expected_cell,
        "replicate": replicate,
        "fit_sha256": _sha256(fit_source),
        "phenotype_sha256": phenotype_sha256,
    }
    receipt = {
        "schema_version": PHENOTYPE_VERSION,
        "manifest_sha256": manifest_sha256,
        "cell": {
            "id": expected_cell,
            "relationship_kind": kind,
            "endpoint": endpoint.id,
            "trait": trait,
            "trait_slug": trait_slug,
        },
        "replicate": replicate,
        "shared_stream_group": components["stream_seeds"]["genetic"]["shared_stream_group"],
        "formula": manifest.bootstrap.phenotype_formula,
        "sample_order": "exact_FAM_order",
        "samples": len(samples),
        "inputs": {
            "fam_sha256": _sha256(fam),
            "qcovar_sha256": qcovar_sha256,
            "fit_receipt_sha256": _sha256(fit_source),
            "grm_bin_sha256": grm.receipt["grm_bin_sha256"],
            "grm_n_bin_sha256": grm.receipt["grm_n_bin_sha256"],
            "grm_id_sha256": grm.receipt["grm_id_sha256"],
            "numpy_version": np.__version__,
            "k_half_eigen_factor_sha256": grm.receipt["k_half_eigen_factor_sha256"],
            "Vg": vg,
            "Ve": ve,
            "fixed_effects": [
                {"term": term, "estimate": float(value)}
                for term, value in zip(terms, beta, strict=True)
            ],
        },
        "simulation_components": components,
        "phenotype": {
            "path": str(phenotype_destination),
            "bytes": len(phenotype_bytes),
            "sha256": phenotype_sha256,
            "float_format": ".17g",
        },
        "cell_replicate_binding_sha256": hashlib.sha256(
            json.dumps(binding_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "byte_replay_sha256": hashlib.sha256(replay_bytes).hexdigest(),
        "byte_replay_verified": True,
    }
    phenotype_stage = phenotype_destination.parent / (
        f".{phenotype_destination.name}.tmp-{uuid.uuid4().hex}"
    )
    receipt_stage = receipt_destination.parent / (
        f".{receipt_destination.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        phenotype_stage.write_bytes(phenotype_bytes)
        receipt_stage.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if hashlib.sha256(phenotype_stage.read_bytes()).hexdigest() != phenotype_sha256:
            raise RuntimeError("staged phenotype bytes do not match the replay receipt")
        os.replace(phenotype_stage, phenotype_destination)
        os.replace(receipt_stage, receipt_destination)
    except BaseException:
        for stage in (phenotype_stage, receipt_stage):
            if stage.exists():
                stage.unlink()
        raise
    return receipt


def _lambda_gc_from_p(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("lambda GC requires at least one p-value")
    p_values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(p_values)) or np.any(p_values <= 0.0) or np.any(p_values > 1.0):
        raise ValueError("lambda GC p-values must lie in (0,1]")
    count = len(p_values)
    if count % 2:
        middle_p = [float(np.partition(p_values, count // 2)[count // 2])]
    else:
        indexes = (count // 2 - 1, count // 2)
        partitioned = np.partition(p_values, indexes)
        middle_p = [float(partitioned[index]) for index in indexes]
    middle_chi_square = []
    for p_value in middle_p:
        z_score = NormalDist().inv_cdf(p_value / 2.0)
        middle_chi_square.append(z_score * z_score)
    result = statistics.fmean(middle_chi_square) / _CHI_SQUARE_1_MEDIAN
    if not math.isfinite(result):
        raise ValueError("lambda GC is not finite")
    return result


def _qq_from_p(values: Sequence[float], quantiles: Sequence[float]) -> list[dict[str, float]]:
    if not values:
        raise ValueError("QQ diagnostics require at least one p-value")
    observed_values = sorted(-math.log10(max(value, 5e-324)) for value in values)
    output = []
    for quantile in quantiles:
        position = quantile * (len(observed_values) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        observed = observed_values[lower]
        if upper != lower:
            observed += (observed_values[upper] - observed) * (position - lower)
        output.append(
            {
                "quantile": quantile,
                "expected_neg_log10_p": -math.log10(1.0 - quantile),
                "observed_neg_log10_p": observed,
            }
        )
    return output


def _inside_interval(
    chromosome: int, bp: int, intervals: Sequence[dict[str, Any]]
) -> bool:
    return any(
        chromosome == interval["chromosome"]
        and interval["start"] <= bp <= interval["end"]
        for interval in intervals
    )


def stream_map_diagnostics(
    map_path: str | Path,
    bim_path: str | Path,
    trait: str,
    manifest: ParametricNullManifest,
) -> dict[str, Any]:
    """Validate a map against BIM order while retaining only compact diagnostics."""

    source = Path(map_path).resolve()
    bim = Path(bim_path).resolve()
    if not source.is_file() or not bim.is_file():
        raise FileNotFoundError("association map or BIM is missing")
    if trait not in TRAITS:
        raise ValueError("map trait is not in the frozen design")
    all_p: list[float] = []
    outside_trait_p: list[float] = []
    outside_all_p: list[float] = []
    per_chromosome: dict[int, dict[str, Any]] = {}
    signature = hashlib.sha256()
    minimum: dict[str, Any] | None = None
    bonferroni_count = 0
    trait_intervals = [item for item in manifest.published_intervals if item["trait"] == trait]
    with source.open("r", encoding="utf-8-sig") as map_handle, bim.open(
        "r", encoding="utf-8-sig"
    ) as bim_handle:
        header = tuple(map_handle.readline().split())
        if header != MLMA_HEADER:
            raise ValueError(f"MLMA header mismatch: {header}")
        count = 0
        for count, bim_line in enumerate(bim_handle, 1):
            bim_fields = bim_line.split()
            if len(bim_fields) != 6:
                raise ValueError(f"BIM row {count} does not contain six fields")
            map_line = map_handle.readline()
            if not map_line:
                raise ValueError("association map ends before the frozen BIM")
            fields = map_line.split()
            if len(fields) != len(MLMA_HEADER):
                raise ValueError(f"MLMA row {count + 1} has the wrong field count")
            try:
                chromosome = int(fields[0])
                bp = int(fields[2])
                frequency, effect, standard_error, p_value = map(
                    float, (fields[5], fields[6], fields[7], fields[8])
                )
                bim_chromosome = int(bim_fields[0])
                bim_bp = int(bim_fields[3])
            except ValueError as exc:
                raise ValueError(f"MLMA or BIM row {count} contains a nonnumeric field") from exc
            snp = fields[1]
            if (chromosome, snp, bp) != (bim_chromosome, bim_fields[1], bim_bp):
                raise ValueError(f"MLMA marker identity/order differs from BIM at marker {count}")
            if chromosome not in CHROMOSOMES or bp <= 0:
                raise ValueError(f"MLMA row {count + 1} has an invalid coordinate")
            numeric = (frequency, effect, standard_error, p_value)
            if (
                not all(math.isfinite(value) for value in numeric)
                or not 0.0 <= frequency <= 1.0
                or standard_error <= 0.0
                or not 0.0 < p_value <= 1.0
                or not fields[3]
                or not fields[4]
            ):
                raise ValueError(f"MLMA row {count + 1} contains an invalid estimate")
            signature.update(
                (
                    f"{chromosome}\t{snp}\t{bp}\t{fields[3]}\t{fields[4]}\t"
                    f"{frequency:.17g}\n"
                ).encode("utf-8")
            )
            all_p.append(p_value)
            if not _inside_interval(chromosome, bp, trait_intervals):
                outside_trait_p.append(p_value)
            if not _inside_interval(chromosome, bp, manifest.published_intervals):
                outside_all_p.append(p_value)
            row_summary = {"chromosome": chromosome, "snp": snp, "bp": bp, "p": p_value}
            if minimum is None or (p_value, chromosome, bp, snp) < (
                minimum["p"],
                minimum["chromosome"],
                minimum["bp"],
                minimum["snp"],
            ):
                minimum = row_summary
            chromosome_minimum = per_chromosome.get(chromosome)
            if chromosome_minimum is None or (p_value, bp, snp) < (
                chromosome_minimum["p"],
                chromosome_minimum["bp"],
                chromosome_minimum["snp"],
            ):
                per_chromosome[chromosome] = row_summary
            if p_value <= manifest.diagnostics.marker_bonferroni_p:
                bonferroni_count += 1
        if map_handle.readline():
            raise ValueError("association map contains markers beyond the frozen BIM")
    if count != manifest.diagnostics.marker_count or len(all_p) != count:
        raise ValueError("association map marker count differs from 373279")
    if set(per_chromosome) != set(CHROMOSOMES) or minimum is None:
        raise ValueError("association map does not cover all six chromosomes")
    return {
        "markers": count,
        "map_bytes": source.stat().st_size,
        "map_sha256": _sha256(source),
        "marker_allele_frequency_signature_sha256": signature.hexdigest(),
        "minimum_p": minimum,
        "per_chromosome_minimum_p": [per_chromosome[item] for item in CHROMOSOMES],
        "bonferroni_marker_count": bonferroni_count,
        "lambda_gc": _lambda_gc_from_p(all_p),
        "lambda_gc_excluding_trait_intervals": _lambda_gc_from_p(outside_trait_p),
        "lambda_gc_excluding_all_intervals": _lambda_gc_from_p(outside_all_p),
        "markers_excluding_trait_intervals": len(outside_trait_p),
        "markers_excluding_all_intervals": len(outside_all_p),
        "qq_quantiles": _qq_from_p(all_p, manifest.diagnostics.qq_quantiles),
    }


def _resolve_clump_output(prefix_or_file: Path) -> Path | None:
    if prefix_or_file.is_file():
        return prefix_or_file
    candidates = [Path(f"{prefix_or_file}.clumps"), Path(f"{prefix_or_file}.clumped")]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError("clump prefix resolves to multiple result formats")
    return existing[0] if existing else None


def _clump_receipt(prefix_or_file: Path) -> dict[str, Any]:
    path = _resolve_clump_output(prefix_or_file)
    if path is None:
        return {
            "result_path": None,
            "result_sha256": None,
            "index_clumps": 0,
            "no_result_file_interpreted_as_zero": True,
        }
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        count = 0
    else:
        header = lines[0].split()
        if not any(value.upper().lstrip("#") in {"CHR", "CHROM"} for value in header):
            raise ValueError("PLINK clump result header is not recognized")
        count = len(lines) - 1
    return {
        "result_path": str(path.resolve()),
        "result_sha256": _sha256(path),
        "index_clumps": count,
        "no_result_file_interpreted_as_zero": False,
    }


def _qualify_association_logs(
    log_paths: Sequence[Path],
    kind: Literal["full", "ldpruned"],
    endpoint: EndpointContract,
    phenotype: Path,
    manifest: ParametricNullManifest,
) -> list[dict[str, Any]]:
    expected_count = 1 if kind == "full" else 6
    if len(log_paths) != expected_count or len(set(log_paths)) != expected_count:
        raise ValueError(f"{kind} map requires exactly {expected_count} unique GCTA log(s)")
    receipts = []
    seen_chromosomes: set[int] = set()
    for path in log_paths:
        if not path.is_file():
            raise FileNotFoundError(f"GCTA association log is missing: {path}")
        text = path.read_text(encoding="utf-8")
        warnings = _qualify_log_text(text, path)
        options = _accepted_options(text, path)
        expected_names = {
            "--mlma-loco" if kind == "full" else "--mlma",
            "--bfile",
            "--pheno",
            "--maf",
            "--autosome-num",
            "--thread-num",
            "--out",
        }
        if kind == "ldpruned":
            expected_names.update({"--extract", "--grm"})
        if endpoint.pc_count:
            expected_names.add("--qcovar")
        if set(options) != expected_names:
            raise ValueError(f"association accepted options differ from contract: {path}")
        flag = "--mlma-loco" if kind == "full" else "--mlma"
        if options[flag] is not None:
            raise ValueError(f"association flag unexpectedly has a value: {path}")
        if (
            options["--maf"] != "0.05"
            or options["--autosome-num"] != "6"
            or options["--thread-num"] != str(manifest.reml.thread_count)
        ):
            raise ValueError(f"association numeric options differ from contract: {path}")
        pheno_option = options["--pheno"]
        if pheno_option is None or Path(pheno_option).resolve() != phenotype:
            raise ValueError(f"association phenotype differs from the simulated input: {path}")
        if endpoint.pc_count:
            qcovar = options.get("--qcovar")
            if qcovar is None or not qcovar.replace("\\", "/").endswith(
                "/genotype/qcovars/pc10.qcovar"
            ):
                raise ValueError(f"PC10 association lacks the frozen qcovar: {path}")
            if "10 quantitative covariate(s) of 209 individuals are included" not in text:
                raise ValueError(f"PC10 association lacks covariate inclusion evidence: {path}")
        elif "--qcovar" in options or "quantitative covariate(s)" in text:
            raise ValueError(f"PC0 association unexpectedly includes qcovars: {path}")
        convergence = [line.strip() for line in text.splitlines() if "converg" in line.lower()]
        if not convergence or any("not converg" in line.lower() for line in convergence):
            raise ValueError(f"association log lacks positive convergence evidence: {path}")
        if "209 individuals are in common in these files." not in text:
            raise ValueError(f"association log lacks exact common-sample evidence: {path}")
        chromosome = None
        if kind == "ldpruned":
            extracts = re.findall(r"candidate_chr([1-6])\.snplist$", options["--extract"] or "")
            grms = re.findall(r"leave_chr([1-6])_out$", options["--grm"] or "")
            if len(extracts) != 1 or extracts != grms:
                raise ValueError(f"LD scan candidate/GRM chromosome mismatch: {path}")
            chromosome = int(extracts[0])
            if chromosome in seen_chromosomes:
                raise ValueError("duplicate chromosome among LD association logs")
            seen_chromosomes.add(chromosome)
            expected_markers = ASSOCIATION_MARKERS_BY_CHROMOSOME[chromosome]
            if f"Running association tests for {expected_markers} SNPs" not in text:
                raise ValueError(f"LD association log lacks expected marker evidence: {path}")
        receipts.append(
            {
                "chromosome": chromosome,
                "path": str(path),
                "sha256": _sha256(path),
                "accepted_options": options,
                "convergence_evidence": convergence,
                "warnings": warnings,
            }
        )
    if kind == "ldpruned" and seen_chromosomes != set(CHROMOSOMES):
        raise ValueError("LD association logs do not cover chromosomes 1-6 exactly")
    return sorted(receipts, key=lambda item: item["chromosome"] or 0)


def qualify_map(
    manifest_path: str | Path,
    bim_path: str | Path,
    map_path: str | Path,
    gcta_logs: Sequence[str | Path],
    grm_kind: str,
    endpoint_id: str,
    trait_slug: str,
    replicate: int,
    phenotype_path: str | Path,
    clump_prefix_or_file: str | Path,
    output: str | Path,
    retained_map: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically checkpoint one observed (0) or null (1..100) map."""

    if not 0 <= replicate <= REPLICATES_PER_CELL:
        raise ValueError("map replicate must be 0 (observed) or 1..100 (null)")
    manifest_source = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_source)
    kind = canonical_kind(grm_kind)
    trait = trait_from_slug(trait_slug)
    endpoint_matches = [item for item in manifest.endpoints if item.id == endpoint_id]
    if len(endpoint_matches) != 1:
        raise ValueError("map endpoint is not pc0 or pc10")
    endpoint = endpoint_matches[0]
    phenotype = Path(phenotype_path).resolve()
    if not phenotype.is_file():
        raise FileNotFoundError(f"map phenotype is missing: {phenotype}")
    map_source = Path(map_path).resolve()
    diagnostics = stream_map_diagnostics(map_source, bim_path, trait, manifest)
    logs = _qualify_association_logs(
        [Path(path).resolve() for path in gcta_logs], kind, endpoint, phenotype, manifest
    )
    clump = _clump_receipt(Path(clump_prefix_or_file).resolve())
    if clump["result_path"] is None and diagnostics["bonferroni_marker_count"] > 0:
        raise ValueError("clump output is absent despite Bonferroni-level association markers")
    if diagnostics["bonferroni_marker_count"] > 0 and clump["index_clumps"] <= 0:
        raise ValueError("Bonferroni-level markers did not produce a PLINK index clump")
    retained_receipt = None
    if retained_map is not None:
        retained = Path(retained_map).resolve()
        if retained.exists():
            raise FileExistsError(f"retained map must not already exist: {retained}")
        retained.parent.mkdir(parents=True, exist_ok=True)
        stage = retained.parent / f".{retained.name}.tmp-{uuid.uuid4().hex}"
        try:
            shutil.copyfile(map_source, stage)
            if _sha256(stage) != diagnostics["map_sha256"]:
                raise RuntimeError("retained map copy differs from its qualified source")
            os.replace(stage, retained)
        except BaseException:
            if stage.exists():
                stage.unlink()
            raise
        retained_receipt = {
            "path": str(retained),
            "bytes": retained.stat().st_size,
            "sha256": _sha256(retained),
        }
    elif replicate in SENTINEL_REPLICATES:
        raise ValueError("frozen sentinel replicates must retain their complete maps")
    receipt = {
        "schema_version": MAP_VERSION,
        "manifest_sha256": _sha256(manifest_source),
        "cell": {
            "id": cell_id(kind, endpoint.id, trait),
            "relationship_kind": kind,
            "endpoint": endpoint.id,
            "trait": trait,
            "trait_slug": trait_slug,
        },
        "replicate": replicate,
        "observed": replicate == 0,
        "phenotype": {
            "path": str(phenotype),
            "bytes": phenotype.stat().st_size,
            "sha256": _sha256(phenotype),
        },
        "association_logs": logs,
        "diagnostics": {**diagnostics, "ld_clump_count": clump["index_clumps"]},
        "clump": clump,
        "retained_map": retained_receipt,
        "raw_map_may_be_removed_after_checkpoint": retained_receipt is None,
        "qualified": True,
    }
    _write_json_atomic(Path(output).resolve(), receipt)
    return receipt


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile input is empty or probability is invalid")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    value = ordered[lower]
    if upper != lower:
        value += (ordered[upper] - value) * (position - lower)
    return value


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != REPLICATES_PER_CELL or not all(math.isfinite(value) for value in values):
        raise ValueError("each null distribution must contain 100 finite values")
    return {
        "replicates": len(values),
        "minimum": min(values),
        "q025": _linear_quantile(values, 0.025),
        "q05": _linear_quantile(values, 0.05),
        "median": _linear_quantile(values, 0.5),
        "q95": _linear_quantile(values, 0.95),
        "q975": _linear_quantile(values, 0.975),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def _binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> dict[str, Any]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial counts")
    alpha = 1.0 - confidence
    try:
        from scipy.stats import beta as beta_distribution  # type: ignore[import-not-found]

        lower = (
            0.0
            if successes == 0
            else float(beta_distribution.ppf(alpha / 2.0, successes, trials - successes + 1))
        )
        upper = (
            1.0
            if successes == trials
            else float(
                beta_distribution.ppf(
                    1.0 - alpha / 2.0, successes + 1, trials - successes
                )
            )
        )
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("SciPy returned a non-finite interval")
        method = "clopper_pearson_exact_scipy"
    except (ImportError, ValueError):
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        proportion = successes / trials
        denominator = 1.0 + z * z / trials
        center = (proportion + z * z / (2.0 * trials)) / denominator
        radius = (
            z
            * math.sqrt(
                proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
            )
            / denominator
        )
        lower = max(0.0, center - radius)
        upper = min(1.0, center + radius)
        method = "wilson_score_no_scipy"
    return {
        "confidence": confidence,
        "method": method,
        "successes": successes,
        "trials": trials,
        "lower": lower,
        "upper": upper,
    }


def plus_one_tail(
    null_values: Sequence[float], observed: float, direction: Literal["lower", "upper"]
) -> dict[str, Any]:
    """Model-specific Monte Carlo tail probability with the plus-one correction."""

    if len(null_values) != REPLICATES_PER_CELL:
        raise ValueError("plus-one calibration requires exactly 100 null replicates")
    if not math.isfinite(observed) or not all(math.isfinite(value) for value in null_values):
        raise ValueError("tail calibration values must be finite")
    if direction == "lower":
        events = sum(value <= observed for value in null_values)
    elif direction == "upper":
        events = sum(value >= observed for value in null_values)
    else:
        raise ValueError("tail direction must be lower or upper")
    return {
        "direction": direction,
        "observed": observed,
        "null_tail_events": events,
        "replicates": len(null_values),
        "plus_one_empirical_p": (events + 1) / (len(null_values) + 1),
        "raw_event_rate": events / len(null_values),
        "monte_carlo_interval_for_raw_event_rate": _binomial_interval(events, len(null_values)),
    }


def _discover_receipts(root: Path, schema_version: str) -> list[tuple[Path, dict[str, Any]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"receipt root is not a directory: {root}")
    output = []
    for path in sorted(root.rglob("*.json")):
        payload = _load_json_object(path, "calibration receipt")
        if payload.get("schema_version") == schema_version:
            output.append((path, payload))
    return output


def _verify_map_checkpoint(
    path: Path,
    payload: dict[str, Any],
    manifest_sha256: str,
    expected_cell: str,
    expected_replicate: int,
) -> None:
    if (
        payload.get("schema_version") != MAP_VERSION
        or payload.get("manifest_sha256") != manifest_sha256
        or payload.get("cell", {}).get("id") != expected_cell
        or payload.get("replicate") != expected_replicate
        or payload.get("observed") is not (expected_replicate == 0)
        or payload.get("qualified") is not True
    ):
        raise ValueError(f"map checkpoint identity/qualification failed: {path}")
    diagnostics = payload.get("diagnostics", {})
    if diagnostics.get("markers") != 373279:
        raise ValueError(f"map checkpoint marker count failed: {path}")
    required_metrics = (
        "lambda_gc",
        "lambda_gc_excluding_trait_intervals",
        "lambda_gc_excluding_all_intervals",
        "bonferroni_marker_count",
        "ld_clump_count",
    )
    for metric in required_metrics:
        value = diagnostics.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"map checkpoint metric {metric} is invalid: {path}")
    minimum = diagnostics.get("minimum_p", {})
    if not isinstance(minimum.get("p"), (int, float)) or not 0.0 <= minimum["p"] <= 1.0:
        raise ValueError(f"map checkpoint minimum p is invalid: {path}")
    phenotype = payload.get("phenotype", {})
    phenotype_path = Path(phenotype.get("path", ""))
    if (
        not phenotype_path.is_file()
        or phenotype_path.stat().st_size != phenotype.get("bytes")
        or _sha256(phenotype_path) != phenotype.get("sha256")
    ):
        raise ValueError(f"checkpoint phenotype identity failed: {path}")
    retained = payload.get("retained_map")
    if expected_replicate in SENTINEL_REPLICATES:
        if not isinstance(retained, dict):
            raise ValueError(f"sentinel checkpoint lacks its retained map: {path}")
        retained_path = Path(retained.get("path", ""))
        if (
            not retained_path.is_file()
            or retained_path.stat().st_size != retained.get("bytes")
            or _sha256(retained_path) != retained.get("sha256")
            or retained.get("sha256") != diagnostics.get("map_sha256")
        ):
            raise ValueError(f"sentinel retained-map identity failed: {path}")
    elif expected_replicate > 0 and retained is not None:
        raise ValueError(f"non-sentinel null checkpoint unexpectedly retained a map: {path}")
    clump = payload.get("clump", {})
    clump_path_value = clump.get("result_path")
    if clump_path_value is None:
        if (
            clump.get("result_sha256") is not None
            or clump.get("index_clumps") != 0
            or diagnostics.get("ld_clump_count") != 0
            or diagnostics.get("bonferroni_marker_count") != 0
            or clump.get("no_result_file_interpreted_as_zero") is not True
        ):
            raise ValueError(f"absent clump result is inconsistent with checkpoint: {path}")
    else:
        clump_path = Path(clump_path_value)
        if (
            not clump_path.is_file()
            or _sha256(clump_path) != clump.get("result_sha256")
            or clump.get("index_clumps") != diagnostics.get("ld_clump_count")
            or clump.get("no_result_file_interpreted_as_zero") is not False
        ):
            raise ValueError(f"checkpoint clump-result identity failed: {path}")


def _verify_simulation_receipt(
    path: Path,
    payload: dict[str, Any],
    checkpoint: dict[str, Any],
    manifest_sha256: str,
    expected_cell: str,
    expected_replicate: int,
    fit_sha256: str,
) -> None:
    phenotype = checkpoint["phenotype"]
    expected_binding = {
        "cell_id": expected_cell,
        "replicate": expected_replicate,
        "fit_sha256": fit_sha256,
        "phenotype_sha256": phenotype["sha256"],
    }
    binding_sha256 = hashlib.sha256(
        json.dumps(expected_binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        payload.get("schema_version") != PHENOTYPE_VERSION
        or payload.get("manifest_sha256") != manifest_sha256
        or payload.get("cell", {}).get("id") != expected_cell
        or payload.get("replicate") != expected_replicate
        or payload.get("byte_replay_verified") is not True
        or payload.get("byte_replay_sha256") != phenotype["sha256"]
        or payload.get("cell_replicate_binding_sha256") != binding_sha256
        or payload.get("inputs", {}).get("fit_receipt_sha256") != fit_sha256
        or payload.get("phenotype", {}).get("sha256") != phenotype["sha256"]
        or payload.get("phenotype", {}).get("bytes") != phenotype["bytes"]
        or Path(payload.get("phenotype", {}).get("path", ""))
        != Path(phenotype["path"])
    ):
        raise ValueError(f"simulation replay/binding receipt failed: {path}")
    expected_group = (
        f"{checkpoint['cell']['trait_slug']}/replicate-{expected_replicate:03d}"
    )
    if payload.get("shared_stream_group") != expected_group:
        raise ValueError(f"simulation shared-stream group differs from contract: {path}")
    inputs = payload.get("inputs", {})
    if inputs.get("numpy_version") != np.__version__ or not _SHA256_RE.fullmatch(
        str(inputs.get("k_half_eigen_factor_sha256", ""))
    ):
        raise ValueError(f"simulation transform provenance is incomplete: {path}")


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# WS283 model-specific parametric polygenic-null calibration",
        "",
        "This is a preliminary, model-specific calibration. Null distributions are not "
        "pooled, and the results do not select a preferred model or support causal claims.",
        "",
        "| Cell | Observed min P | 5th null min-P | Min-P empirical P | Observed lambda "
        "outside all intervals | Lambda empirical P | Observed clumps | Clump empirical P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        lambda_empirical_p = cell["empirical_tails"][
            "lambda_gc_excluding_all_intervals"
        ]["plus_one_empirical_p"]
        eligibility = "eligible" if cell["threshold_eligible"] else "descriptive boundary"
        lines.append(
            f"| {cell['id']} ({eligibility}) | {cell['observed']['minimum_p']:.6g} | "
            f"{cell['minimum_p_fwer']['conservative_fifth_order_statistic']:.6g} | "
            f"{cell['empirical_tails']['minimum_p']['plus_one_empirical_p']:.4f} | "
            f"{cell['observed']['lambda_gc_excluding_all_intervals']:.4f} | "
            f"{lambda_empirical_p:.4f} | "
            f"{cell['observed']['ld_clump_count']} | "
            f"{cell['empirical_tails']['ld_clump_count']['plus_one_empirical_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The per-cell Monte Carlo resolution is 1/101. For the 14 interior-anchor cells, "
            "the fifth sorted null minimum P is the predeclared conservative preliminary "
            "~5% genome-wide threshold. For the two expected residual-floor cells it is a "
            "descriptive order statistic only. No threshold is pooled across cells.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(
    manifest_path: str | Path,
    baseline_parent: str | Path,
    calibration_parent: str | Path,
    sensitivity_parent: str | Path,
    fit_root: str | Path,
    checkpoint_root: str | Path,
    observed_root: str | Path,
    output: str | Path,
    run_manifest: str | Path,
) -> dict[str, Any]:
    """Aggregate exactly 100 replicates in each of 16 cells, never pooling cells."""

    manifest_source = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_source)
    manifest_sha256 = _sha256(manifest_source)
    parents = verify_parents(
        manifest_source, baseline_parent, calibration_parent, sensitivity_parent
    )
    fits_root = Path(fit_root).resolve()
    checkpoints_root = Path(checkpoint_root).resolve()
    observed_receipt_root = Path(observed_root).resolve()
    simulation_root = checkpoints_root.parent / "simulations"
    fit_receipts = _discover_receipts(fits_root, FIT_VERSION)
    null_receipts = _discover_receipts(checkpoints_root, MAP_VERSION)
    observed_receipts = _discover_receipts(observed_receipt_root, MAP_VERSION)
    simulation_receipts = _discover_receipts(simulation_root, PHENOTYPE_VERSION)
    if len(fit_receipts) != CELL_COUNT:
        raise ValueError("fit root must contain exactly 16 qualified fit receipts")
    if len(null_receipts) != TOTAL_NULL_MAPS:
        raise ValueError("checkpoint root must contain exactly 1600 qualified null maps")
    if len(observed_receipts) != CELL_COUNT:
        raise ValueError("observed root must contain exactly 16 qualified observed maps")
    if len(simulation_receipts) != TOTAL_NULL_MAPS:
        raise ValueError("simulation root must contain exactly 1600 byte-replay receipts")
    fits_by_cell: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, payload in fit_receipts:
        identifier = payload.get("cell", {}).get("id")
        if not isinstance(identifier, str) or identifier in fits_by_cell:
            raise ValueError("fit receipts contain an invalid or duplicate cell")
        if (
            payload.get("manifest_sha256") != manifest_sha256
            or payload.get("qualified") is not True
        ):
            raise ValueError("fit receipt manifest binding or qualification failed")
        expected_boundary = identifier in EXPECTED_BOUNDARY_CELLS
        if (
            payload.get("anchor_boundary") is not expected_boundary
            or payload.get("threshold_eligible") is not (not expected_boundary)
        ):
            raise ValueError("fit receipt boundary eligibility differs from preflight")
        fits_by_cell[identifier] = (path, payload)
    null_by_key: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for path, payload in null_receipts:
        key = (payload.get("cell", {}).get("id"), payload.get("replicate"))
        if not isinstance(key[0], str) or not isinstance(key[1], int) or key in null_by_key:
            raise ValueError("null checkpoints contain an invalid or duplicate key")
        null_by_key[key] = (path, payload)
    observed_by_cell: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, payload in observed_receipts:
        identifier = payload.get("cell", {}).get("id")
        if not isinstance(identifier, str) or identifier in observed_by_cell:
            raise ValueError("observed checkpoints contain an invalid or duplicate cell")
        observed_by_cell[identifier] = (path, payload)
    simulations_by_key: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for path, payload in simulation_receipts:
        key = (payload.get("cell", {}).get("id"), payload.get("replicate"))
        if (
            not isinstance(key[0], str)
            or not isinstance(key[1], int)
            or key in simulations_by_key
        ):
            raise ValueError("simulation receipts contain an invalid or duplicate key")
        simulations_by_key[key] = (path, payload)

    cells = []
    all_signatures: set[str] = set()
    inventory: list[dict[str, Any]] = []
    expected_cells = iter_cells(manifest)
    for coordinates in expected_cells:
        identifier = coordinates["id"]
        if identifier not in fits_by_cell or identifier not in observed_by_cell:
            raise ValueError(f"cell is missing its fit or observed checkpoint: {identifier}")
        fit_path, fit = fits_by_cell[identifier]
        fit_sha256 = _sha256(fit_path)
        observed_path, observed = observed_by_cell[identifier]
        _verify_map_checkpoint(observed_path, observed, manifest_sha256, identifier, 0)
        null_payloads = []
        for replicate in range(1, REPLICATES_PER_CELL + 1):
            key = (identifier, replicate)
            if key not in null_by_key:
                raise ValueError(f"cell is missing null replicate {replicate}: {identifier}")
            checkpoint_path, checkpoint = null_by_key[key]
            _verify_map_checkpoint(
                checkpoint_path, checkpoint, manifest_sha256, identifier, replicate
            )
            if key not in simulations_by_key:
                raise ValueError(
                    f"cell is missing simulation replay receipt {replicate}: {identifier}"
                )
            simulation_path, simulation_receipt = simulations_by_key[key]
            _verify_simulation_receipt(
                simulation_path,
                simulation_receipt,
                checkpoint,
                manifest_sha256,
                identifier,
                replicate,
                fit_sha256,
            )
            null_payloads.append(checkpoint)
            inventory.append(
                {
                    "role": "null_checkpoint",
                    "cell": identifier,
                    "replicate": replicate,
                    "path": str(checkpoint_path),
                    "sha256": _sha256(checkpoint_path),
                }
            )
            inventory.append(
                {
                    "role": "simulation_receipt",
                    "cell": identifier,
                    "replicate": replicate,
                    "path": str(simulation_path),
                    "sha256": _sha256(simulation_path),
                }
            )
        inventory.extend(
            [
                {
                    "role": "fit",
                    "cell": identifier,
                    "path": str(fit_path),
                    "sha256": _sha256(fit_path),
                },
                {
                    "role": "observed_checkpoint",
                    "cell": identifier,
                    "path": str(observed_path),
                    "sha256": _sha256(observed_path),
                },
            ]
        )
        observed_diagnostics = observed["diagnostics"]
        null_diagnostics = [item["diagnostics"] for item in null_payloads]
        signatures = {
            item["marker_allele_frequency_signature_sha256"]
            for item in [observed_diagnostics, *null_diagnostics]
        }
        if len(signatures) != 1:
            raise ValueError(f"map allele/frequency signature changed within cell {identifier}")
        all_signatures.update(signatures)
        minimum_p = [float(item["minimum_p"]["p"]) for item in null_diagnostics]
        lambdas = [float(item["lambda_gc"]) for item in null_diagnostics]
        lambdas_trait = [
            float(item["lambda_gc_excluding_trait_intervals"]) for item in null_diagnostics
        ]
        lambdas_all = [
            float(item["lambda_gc_excluding_all_intervals"]) for item in null_diagnostics
        ]
        bonferroni = [float(item["bonferroni_marker_count"]) for item in null_diagnostics]
        clumps = [float(item["ld_clump_count"]) for item in null_diagnostics]
        observed_compact = {
            "minimum_p": float(observed_diagnostics["minimum_p"]["p"]),
            "minimum_p_marker": observed_diagnostics["minimum_p"],
            "lambda_gc": float(observed_diagnostics["lambda_gc"]),
            "lambda_gc_excluding_trait_intervals": float(
                observed_diagnostics["lambda_gc_excluding_trait_intervals"]
            ),
            "lambda_gc_excluding_all_intervals": float(
                observed_diagnostics["lambda_gc_excluding_all_intervals"]
            ),
            "bonferroni_marker_count": int(observed_diagnostics["bonferroni_marker_count"]),
            "ld_clump_count": int(observed_diagnostics["ld_clump_count"]),
        }
        fifth = sorted(minimum_p)[manifest.bootstrap.min_p_fwer_order_statistic - 1]
        nominal_bonferroni_events = sum(
            value <= manifest.diagnostics.marker_bonferroni_p for value in minimum_p
        )
        anchor_boundary = identifier in EXPECTED_BOUNDARY_CELLS
        threshold_eligible = not anchor_boundary
        cell_summary = {
            **coordinates,
            "anchor_boundary": anchor_boundary,
            "threshold_eligible": threshold_eligible,
            "boundary_policy": manifest.reml.boundary_policy,
            "fit": {
                "receipt_sha256": _sha256(fit_path),
                "variance_components": fit["variance_components"],
                "fixed_effects": fit["fixed_effects"],
            },
            "observed": observed_compact,
            "null_distributions": {
                "minimum_p": _distribution(minimum_p),
                "lambda_gc": _distribution(lambdas),
                "lambda_gc_excluding_trait_intervals": _distribution(lambdas_trait),
                "lambda_gc_excluding_all_intervals": _distribution(lambdas_all),
                "bonferroni_marker_count": _distribution(bonferroni),
                "ld_clump_count": _distribution(clumps),
            },
            "empirical_tails": {
                "minimum_p": plus_one_tail(minimum_p, observed_compact["minimum_p"], "lower"),
                "lambda_gc": plus_one_tail(lambdas, observed_compact["lambda_gc"], "upper"),
                "lambda_gc_excluding_trait_intervals": plus_one_tail(
                    lambdas_trait,
                    observed_compact["lambda_gc_excluding_trait_intervals"],
                    "upper",
                ),
                "lambda_gc_excluding_all_intervals": plus_one_tail(
                    lambdas_all,
                    observed_compact["lambda_gc_excluding_all_intervals"],
                    "upper",
                ),
                "bonferroni_marker_count": plus_one_tail(
                    bonferroni, float(observed_compact["bonferroni_marker_count"]), "upper"
                ),
                "ld_clump_count": plus_one_tail(
                    clumps, float(observed_compact["ld_clump_count"]), "upper"
                ),
            },
            "minimum_p_fwer": {
                "order_statistic": manifest.bootstrap.min_p_fwer_order_statistic,
                "replicates": REPLICATES_PER_CELL,
                "conservative_fifth_order_statistic": fifth,
                "threshold_eligible": threshold_eligible,
                "interpretation": (
                    "eligible_preliminary_model_specific_threshold"
                    if threshold_eligible
                    else "descriptive_order_statistic_expected_boundary_anchor"
                ),
                "observed_passes_preliminary_threshold": (
                    observed_compact["minimum_p"] <= fifth
                    if threshold_eligible
                    else None
                ),
                "pooled_across_cells": False,
            },
            "nominal_bonferroni_fwer": {
                "primary_null_metric": True,
                "threshold_eligible": threshold_eligible,
                "threshold": manifest.diagnostics.marker_bonferroni_p,
                "replicates_with_minimum_p_at_or_below_threshold": (
                    nominal_bonferroni_events
                ),
                "replicates": REPLICATES_PER_CELL,
                "rate": nominal_bonferroni_events / REPLICATES_PER_CELL,
                "binomial_interval": _binomial_interval(
                    nominal_bonferroni_events, REPLICATES_PER_CELL
                ),
            },
            "false_clump_burden": {
                "replicates_with_one_or_more": sum(value > 0 for value in clumps),
                "proportion_with_one_or_more": sum(value > 0 for value in clumps) / 100.0,
                "total_across_replicates": int(sum(clumps)),
            },
            "map_signature_sha256": next(iter(signatures)),
        }
        cells.append(cell_summary)
    expected_keys = {
        (item["id"], replicate)
        for item in expected_cells
        for replicate in range(1, REPLICATES_PER_CELL + 1)
    }
    if set(null_by_key) != expected_keys:
        raise ValueError("null checkpoint inventory contains coordinates outside the frozen design")
    if set(simulations_by_key) != expected_keys:
        raise ValueError(
            "simulation receipt inventory contains coordinates outside the frozen design"
        )
    if set(fits_by_cell) != {item["id"] for item in expected_cells} or set(
        observed_by_cell
    ) != {item["id"] for item in expected_cells}:
        raise ValueError("fit or observed inventory differs from the frozen 16 cells")
    if len(all_signatures) != 1:
        raise ValueError("map allele/frequency signature changed across bootstrap cells")
    summary = {
        "schema_version": SUMMARY_VERSION,
        "analysis_id": manifest.analysis_id,
        "classification": manifest.classification,
        "status": "preliminary_model_specific_null_calibration_complete",
        "validated": False,
        "biological_claims_permitted": False,
        "manifest_sha256": manifest_sha256,
        "design": {
            "relationship_models": list(RELATIONSHIP_KINDS),
            "endpoints": list(ENDPOINTS),
            "traits": list(TRAITS),
            "cells": CELL_COUNT,
            "replicates_per_cell": REPLICATES_PER_CELL,
            "total_null_maps": TOTAL_NULL_MAPS,
            "common_random_numbers": True,
            "plus_one_minimum_resolution": 1 / 101,
            "min_p_threshold_order_statistic": 5,
            "pooled_thresholds_permitted": False,
            "interior_anchor_cells": 14,
            "expected_boundary_anchor_cells": 2,
            "expected_boundary_cell_ids": list(EXPECTED_BOUNDARY_CELLS),
            "boundary_policy": manifest.reml.boundary_policy,
        },
        "map_signature_sha256": next(iter(all_signatures)),
        "cells": cells,
        "claim_boundary": {
            "model_specific_only": True,
            "preferred_model_selection_permitted": False,
            "causal_inference_permitted": False,
            "external_replication_established": False,
        },
    }
    destination = Path(output).resolve()
    run_manifest_destination = Path(run_manifest).resolve()
    aggregation_alias = destination.parent / "parametric_null_aggregation.json"
    if destination.exists() or run_manifest_destination.exists() or aggregation_alias.exists():
        raise FileExistsError(
            "aggregate output, aggregation alias, and run manifest must not already exist"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        summary_json = stage / "ws283_parametric_null_summary.json"
        summary_markdown = stage / "ws283_parametric_null_summary.md"
        summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        summary_markdown.write_text(_markdown_summary(summary), encoding="utf-8", newline="\n")
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    summary_json = destination / "ws283_parametric_null_summary.json"
    summary_markdown = destination / "ws283_parametric_null_summary.md"
    aggregation_payload = {
        "schema_version": "wormctx-abamectin-ws283-parametric-null-aggregation-1.0",
        "manifest_sha256": manifest_sha256,
        "rich_summary": {
            "path": str(summary_json),
            "sha256": _sha256(summary_json),
        },
        "calibration_cells": CELL_COUNT,
        "replicates_per_cell": REPLICATES_PER_CELL,
        "attempted_null_maps": TOTAL_NULL_MAPS,
        "completed_null_maps": TOTAL_NULL_MAPS,
        "cells": [
            {
                "cell_id": cell["id"],
                "replicates_attempted": REPLICATES_PER_CELL,
                "replicates_completed": REPLICATES_PER_CELL,
                "nominal_bonferroni_fwer": cell["nominal_bonferroni_fwer"],
                "anchor_boundary": cell["anchor_boundary"],
                "threshold_eligible": cell["threshold_eligible"],
            }
            for cell in cells
        ],
        "validated": False,
        "biological_claims_permitted": False,
    }
    _write_json_atomic(aggregation_alias, aggregation_payload)
    inventory_sorted = sorted(
        inventory,
        key=lambda item: (item["role"], item["cell"], item.get("replicate", 0)),
    )
    run_receipt = {
        "schema_version": "wormctx-abamectin-ws283-parametric-null-run-1.0",
        "manifest_sha256": manifest_sha256,
        "parent_verification": parents,
        "samples": 209,
        "calibration_cells": CELL_COUNT,
        "replicates_per_cell": REPLICATES_PER_CELL,
        "attempted_null_maps": TOTAL_NULL_MAPS,
        "completed_null_maps": TOTAL_NULL_MAPS,
        "full_grm_mlma_loco_calls": 800,
        "ld_grm_external_chromosome_scans": 4800,
        "anchor_reml_fits": CELL_COUNT,
        "interior_anchor_fits": 14,
        "expected_boundary_anchor_fits": 2,
        "expected_boundary_cells": list(EXPECTED_BOUNDARY_CELLS),
        "boundary_policy": manifest.reml.boundary_policy,
        "global_generating_grms": 2,
        "relationship_models": list(RELATIONSHIP_KINDS),
        "pc_endpoints": list(ENDPOINTS),
        "traits": list(TRAITS),
        "sentinel_raw_map_replicates": list(SENTINEL_REPLICATES),
        "validated": False,
        "biological_claims_permitted": False,
        "fit_receipts": CELL_COUNT,
        "observed_map_checkpoints": CELL_COUNT,
        "null_map_checkpoints": TOTAL_NULL_MAPS,
        "simulation_byte_replay_receipts": TOTAL_NULL_MAPS,
        "sentinel_maps_per_cell": len(SENTINEL_REPLICATES),
        "summary": {
            "json": {"path": str(summary_json), "sha256": _sha256(summary_json)},
            "markdown": {
                "path": str(summary_markdown),
                "sha256": _sha256(summary_markdown),
            },
            "aggregation_alias": {
                "path": str(aggregation_alias),
                "sha256": _sha256(aggregation_alias),
            },
        },
        "receipt_inventory_sha256": hashlib.sha256(
            json.dumps(inventory_sorted, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "receipt_inventory": inventory_sorted,
        "all_cells_complete": True,
        "pooled_thresholds_used": False,
    }
    _write_json_atomic(run_manifest_destination, run_receipt)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    parents = commands.add_parser("verify-parents")
    parents.add_argument("--manifest", type=Path, required=True)
    parents.add_argument("--baseline-parent", type=Path, required=True)
    parents.add_argument("--calibration-parent", type=Path, required=True)
    parents.add_argument("--sensitivity-parent", type=Path, required=True)
    parents.add_argument("--output", type=Path)

    grm = commands.add_parser("verify-grm")
    grm.add_argument("--manifest", type=Path, required=True)
    grm.add_argument("--fam", type=Path, required=True)
    grm.add_argument("--grm-prefix", type=Path, required=True)
    grm.add_argument("--grm-kind", choices=("full", "ldpruned", "dense", "ld"), required=True)
    grm.add_argument("--expected-markers", type=int, required=True)
    grm.add_argument("--output", type=Path)

    fit = commands.add_parser("qualify-fit")
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--fam", type=Path, required=True)
    fit.add_argument("--phenotype", type=Path, required=True)
    fit.add_argument("--qcovar", type=Path)
    fit.add_argument("--grm-prefix", type=Path, required=True)
    fit.add_argument("--hsq", type=Path, required=True)
    fit.add_argument("--log", type=Path, required=True)
    fit.add_argument("--grm-kind", choices=("full", "ldpruned", "dense", "ld"), required=True)
    fit.add_argument("--endpoint", choices=ENDPOINTS, required=True)
    fit.add_argument("--trait-slug", choices=tuple(TRAIT_SLUGS.values()), required=True)
    fit.add_argument("--output", type=Path, required=True)

    simulation = commands.add_parser("simulate")
    simulation.add_argument("--manifest", type=Path, required=True)
    simulation.add_argument("--fam", type=Path, required=True)
    simulation.add_argument("--qcovar", type=Path)
    simulation.add_argument("--grm-prefix", type=Path, required=True)
    simulation.add_argument("--fit", type=Path, required=True)
    simulation.add_argument(
        "--grm-kind", choices=("full", "ldpruned", "dense", "ld"), required=True
    )
    simulation.add_argument("--endpoint", choices=ENDPOINTS, required=True)
    simulation.add_argument("--trait-slug", choices=tuple(TRAIT_SLUGS.values()), required=True)
    simulation.add_argument("--replicate", type=int, required=True)
    simulation.add_argument("--phenotype-output", type=Path, required=True)
    simulation.add_argument("--receipt-output", type=Path, required=True)

    map_command = commands.add_parser("qualify-map")
    map_command.add_argument("--manifest", type=Path, required=True)
    map_command.add_argument("--bim", type=Path, required=True)
    map_command.add_argument("--map", type=Path, required=True)
    map_command.add_argument("--gcta-log", type=Path, action="append", required=True)
    map_command.add_argument(
        "--grm-kind", choices=("full", "ldpruned", "dense", "ld"), required=True
    )
    map_command.add_argument("--endpoint", choices=ENDPOINTS, required=True)
    map_command.add_argument("--trait-slug", choices=tuple(TRAIT_SLUGS.values()), required=True)
    map_command.add_argument("--replicate", type=int, required=True)
    map_command.add_argument("--phenotype", type=Path, required=True)
    clump_group = map_command.add_mutually_exclusive_group(required=True)
    clump_group.add_argument("--clump-prefix", type=Path)
    clump_group.add_argument("--clump-file", type=Path)
    map_command.add_argument("--output", type=Path, required=True)
    map_command.add_argument("--retained-map", type=Path)

    aggregate_command = commands.add_parser("aggregate")
    aggregate_command.add_argument("--manifest", type=Path, required=True)
    aggregate_command.add_argument("--baseline-parent", type=Path, required=True)
    aggregate_command.add_argument("--calibration-parent", type=Path, required=True)
    aggregate_command.add_argument("--sensitivity-parent", type=Path, required=True)
    aggregate_command.add_argument("--fit-root", type=Path, required=True)
    aggregate_command.add_argument("--checkpoint-root", type=Path, required=True)
    aggregate_command.add_argument("--observed-root", type=Path, required=True)
    aggregate_command.add_argument("--output", type=Path, required=True)
    aggregate_command.add_argument("--run-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify-parents":
            result = verify_parents(
                args.manifest,
                args.baseline_parent,
                args.calibration_parent,
                args.sensitivity_parent,
                args.output,
            )
        elif args.command == "verify-grm":
            result = verify_grm(
                args.manifest,
                args.fam,
                args.grm_prefix,
                args.grm_kind,
                args.expected_markers,
                args.output,
            )
        elif args.command == "qualify-fit":
            result = qualify_fit(
                args.manifest,
                args.fam,
                args.phenotype,
                args.grm_prefix,
                args.hsq,
                args.log,
                args.grm_kind,
                args.endpoint,
                args.trait_slug,
                args.output,
                args.qcovar,
            )
        elif args.command == "simulate":
            result = simulate(
                args.manifest,
                args.fam,
                args.grm_prefix,
                args.fit,
                args.grm_kind,
                args.endpoint,
                args.trait_slug,
                args.replicate,
                args.phenotype_output,
                args.receipt_output,
                args.qcovar,
            )
        elif args.command == "qualify-map":
            result = qualify_map(
                args.manifest,
                args.bim,
                args.map,
                args.gcta_log,
                args.grm_kind,
                args.endpoint,
                args.trait_slug,
                args.replicate,
                args.phenotype,
                args.clump_prefix if args.clump_prefix is not None else args.clump_file,
                args.output,
                args.retained_map,
            )
        elif args.command == "aggregate":
            result = aggregate(
                args.manifest,
                args.baseline_parent,
                args.calibration_parent,
                args.sensitivity_parent,
                args.fit_root,
                args.checkpoint_root,
                args.observed_root,
                args.output,
                args.run_manifest,
            )
        else:  # pragma: no cover - argparse enforces this
            raise ValueError(f"unsupported command: {args.command}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
