"""Gene-disjoint developmental baseline qualification and sealed evaluation.

The module operates in two phases. ``fit-select`` may deserialize only the
train and validation artifacts. ``evaluate-sealed`` atomically records the
single sealed-test opening before it reads the held-out artifact.
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import developmental_omix_stage_adapter as stage
from .developmental_omix import (
    SHA256_RE,
    _canonical_bytes,
    _sha256,
    _write_bytes_atomic,
    _write_json_atomic,
)


SCHEMA_VERSION = "wormctx-omix709-gene-disjoint-baseline-contract-1.0"
PREPARED_MANIFEST_VERSION = "wormctx-omix709-baseline-prepared-manifest-1.0"
PARTITION_VERSION = "wormctx-omix709-baseline-partition-1.0"
SELECTION_VERSION = "wormctx-omix709-baseline-selection-1.0"
SELECTION_VERIFY_VERSION = "wormctx-omix709-baseline-selection-verification-1.0"
SEALED_EVALUATION_VERSION = "wormctx-omix709-baseline-sealed-evaluation-1.0"
EVALUATION_VERIFY_VERSION = "wormctx-omix709-baseline-evaluation-verification-1.0"
SEALED_GATE_VERSION = "wormctx-omix709-sealed-test-opening-1.0"

FAMILY_ORDER = [
    "constant_control_mean",
    "compact_flat_ridge",
    "semantic_cell_aligned_linear",
    "nonlinear_edge_free",
]
PARTITION_ORDER = ["train", "validation", "sealed_test"]
SELECTION_FILES = {
    "model_states.json",
    "selection_manifest.json",
    "validation_metrics.json",
    "validation_predictions.json",
}
EVALUATION_FILES = {
    "evaluation_manifest.json",
    "test_bootstrap.json",
    "test_metrics.json",
    "test_predictions.json",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StageDependency(_StrictModel):
    contract_filename: str
    contract_sha256: str
    required_qualification_status: str
    required_model_launch_status: str

    @field_validator("contract_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("stage-adapter contract hash is malformed")
        return value


class SplitContract(_StrictModel):
    key: str
    whole_gene: bool
    train_genes: int = Field(gt=0)
    validation_genes: int = Field(gt=0)
    sealed_test_genes: int = Field(gt=0)
    exposed_partition: str
    exposed_genes_permitted_in_modeling: bool

    @model_validator(mode="after")
    def exact_split(self) -> "SplitContract":
        if self.key != "wormbase_gene_id" or not self.whole_gene:
            raise ValueError("the baseline split must be whole-gene by WormBase ID")
        if (self.train_genes, self.validation_genes, self.sealed_test_genes) != (
            410,
            137,
            137,
        ):
            raise ValueError("the baseline contract requires the frozen 410/137/137 split")
        if self.exposed_partition != "development_only_exposed_pilot":
            raise ValueError("the pilot-exposed partition identity differs")
        if self.exposed_genes_permitted_in_modeling:
            raise ValueError("pilot-exposed genes may not enter baseline modeling")
        return self


class RidgeContract(_StrictModel):
    alphas: list[float]
    penalize_intercept: bool

    @model_validator(mode="after")
    def valid_grid(self) -> "RidgeContract":
        if not self.alphas or any(value <= 0 or not math.isfinite(value) for value in self.alphas):
            raise ValueError("ridge alphas must be finite and positive")
        if self.alphas != sorted(set(self.alphas)):
            raise ValueError("ridge alphas must be unique and sorted")
        if self.penalize_intercept:
            raise ValueError("the ridge intercept must remain unpenalized")
        return self


class NonlinearContract(_StrictModel):
    architecture: str
    activation: str
    target_trainable_parameters: int = Field(gt=0)
    maximum_relative_parameter_delta: float = Field(ge=0.0, lt=1.0)
    learning_rates: list[float]
    l2_penalties: list[float]
    seeds: list[int]
    epochs: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_grid(self) -> "NonlinearContract":
        if self.architecture != "one_hidden_layer_edge_free_mlp" or self.activation != "tanh":
            raise ValueError("the nonlinear edge-free architecture differs")
        if any(value <= 0 or not math.isfinite(value) for value in self.learning_rates):
            raise ValueError("MLP learning rates must be finite and positive")
        if any(value < 0 or not math.isfinite(value) for value in self.l2_penalties):
            raise ValueError("MLP L2 penalties must be finite and nonnegative")
        if not self.learning_rates or not self.l2_penalties or not self.seeds:
            raise ValueError("MLP candidate grids must be nonempty")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("MLP seeds must be unique")
        return self


class ModelContract(_StrictModel):
    family_order: list[str]
    constant_value_source: str
    preprocessing: str
    include_missingness_indicators: bool
    zero_variance_policy: str
    flat_ridge: RidgeContract
    semantic_linear: RidgeContract
    nonlinear_edge_free: NonlinearContract
    primary_selection_metric: str
    selection_tie_breakers: list[str]
    refit_on_train_plus_validation_after_selection: bool

    @model_validator(mode="after")
    def exact_models(self) -> "ModelContract":
        if self.family_order != FAMILY_ORDER:
            raise ValueError("baseline family identities or order differ")
        if self.constant_value_source != "control_standardized_expected_mean_from_manifest":
            raise ValueError("constant baseline source differs")
        if self.preprocessing != "training_partition_mean_impute_and_zscore":
            raise ValueError("preprocessing policy differs")
        if not self.include_missingness_indicators:
            raise ValueError("explicit missingness indicators are required")
        if self.zero_variance_policy != "scale_one":
            raise ValueError("zero-variance features must use unit scale")
        if self.primary_selection_metric != "gene_macro_rmse":
            raise ValueError("primary selection metric differs")
        if self.selection_tie_breakers != ["gene_macro_mae", "candidate_id"]:
            raise ValueError("selection tie breakers differ")
        if not self.refit_on_train_plus_validation_after_selection:
            raise ValueError("selected models must refit on train plus validation")
        return self


class EvaluationContract(_StrictModel):
    metrics: list[str]
    rank_definition: str
    calibration_definition: str
    resampling_unit: str
    bootstrap_replicates: int = Field(gt=0)
    bootstrap_seed: int
    interval_quantiles: list[float]
    validation_permitted_for_selection: bool
    sealed_test_permitted_for_selection: bool
    sealed_test_openings: int
    gate_name_prefix: str

    @model_validator(mode="after")
    def exact_evaluation(self) -> "EvaluationContract":
        if self.metrics != [
            "gene_macro_rmse",
            "gene_macro_mae",
            "gene_mean_spearman",
            "calibration_intercept",
            "calibration_slope",
        ]:
            raise ValueError("evaluation metric set differs")
        if self.rank_definition != "spearman_of_gene_mean_observed_and_predicted":
            raise ValueError("rank definition differs")
        if self.calibration_definition != "ols_gene_mean_observed_on_predicted":
            raise ValueError("calibration definition differs")
        if self.resampling_unit != "whole_gene_cluster":
            raise ValueError("resampling must use whole-gene clusters")
        if self.interval_quantiles != [0.025, 0.975]:
            raise ValueError("bootstrap interval quantiles differ")
        if not self.validation_permitted_for_selection or self.sealed_test_permitted_for_selection:
            raise ValueError("selection partition policy differs")
        if self.sealed_test_openings != 1 or self.gate_name_prefix != "sealed-test-gate-":
            raise ValueError("sealed-test opening policy differs")
        return self


class BaselineContract(_StrictModel):
    schema_version: str
    analysis_id: str
    classification: str
    validated: bool
    biological_claims_permitted: bool
    stage_adapter: StageDependency
    split: SplitContract
    models: ModelContract
    evaluation: EvaluationContract

    @model_validator(mode="after")
    def exact_contract(self) -> "BaselineContract":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unknown developmental baseline contract")
        if self.validated or self.biological_claims_permitted:
            raise ValueError("baseline scaffold cannot assert validation or biology")
        if self.stage_adapter.required_qualification_status != "qualified_with_exclusions":
            raise ValueError("stage qualification requirement differs")
        if self.stage_adapter.required_model_launch_status != (
            "eligible_for_preregistered_gene_disjoint_baselines"
        ):
            raise ValueError("stage launch requirement differs")
        return self


class ArtifactContract(_StrictModel):
    partition: str
    filename: str
    bytes: int = Field(gt=0)
    sha256: str
    genes: int = Field(gt=0)
    embryos: int = Field(gt=0)

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("partition artifact hash is malformed")
        return value


class FeatureSchema(_StrictModel):
    flat_feature_names: list[str]
    semantic_feature_names: list[str]
    semantic_feature_cell_ids: list[str]
    outcome_id: str
    outcome_scale: str
    control_standardized_expected_mean: float

    @model_validator(mode="after")
    def valid_features(self) -> "FeatureSchema":
        if not self.flat_feature_names or len(self.flat_feature_names) != len(
            set(self.flat_feature_names)
        ):
            raise ValueError("flat feature names must be nonempty and unique")
        if not self.semantic_feature_names or len(self.semantic_feature_names) != len(
            set(self.semantic_feature_names)
        ):
            raise ValueError("semantic feature names must be nonempty and unique")
        if len(self.semantic_feature_names) != len(self.semantic_feature_cell_ids):
            raise ValueError("every semantic feature must bind one semantic cell")
        if not math.isfinite(self.control_standardized_expected_mean):
            raise ValueError("control-standardized expected mean must be finite")
        return self


class PreparedManifest(_StrictModel):
    schema_version: str
    analysis_id: str
    stage_adapter_qualification_sha256: str
    split_manifest_sha256: str
    source_stage_membership_sha256: str
    partition_artifacts_are_separate: bool
    sealed_test_artifact_not_read_during_selection: bool
    feature_schema: FeatureSchema
    artifacts: list[ArtifactContract]

    @field_validator(
        "stage_adapter_qualification_sha256",
        "split_manifest_sha256",
        "source_stage_membership_sha256",
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("prepared-manifest parent hash is malformed")
        return value

    @model_validator(mode="after")
    def exact_manifest(self) -> "PreparedManifest":
        if self.schema_version != PREPARED_MANIFEST_VERSION:
            raise ValueError("unknown prepared baseline manifest")
        if not self.partition_artifacts_are_separate:
            raise ValueError("partition artifacts must be physically separate")
        if not self.sealed_test_artifact_not_read_during_selection:
            raise ValueError("prepared manifest does not preserve the selection gate")
        if [item.partition for item in self.artifacts] != PARTITION_ORDER:
            raise ValueError("prepared partition identities or order differ")
        filenames = [item.filename for item in self.artifacts]
        if len(filenames) != len(set(filenames)):
            raise ValueError("prepared partition filenames must be distinct")
        return self


class Observation(_StrictModel):
    embryo_id: str
    wormbase_gene_id: str
    flat_features: list[float | None]
    semantic_features: list[float | None]
    outcome: float

    @field_validator("outcome")
    @classmethod
    def finite_outcome(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("outcome must be finite")
        return value


class PartitionPayload(_StrictModel):
    schema_version: str
    partition: str
    observations: list[Observation]

    @model_validator(mode="after")
    def exact_partition(self) -> "PartitionPayload":
        if self.schema_version != PARTITION_VERSION:
            raise ValueError("unknown prepared partition schema")
        if self.partition not in PARTITION_ORDER:
            raise ValueError("unknown prepared partition identity")
        if not self.observations:
            raise ValueError("prepared partition contains no observations")
        embryo_ids = [item.embryo_id for item in self.observations]
        if len(embryo_ids) != len(set(embryo_ids)):
            raise ValueError("prepared partition contains duplicate embryo IDs")
        return self


def load_contract(path: str | Path) -> BaselineContract:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"baseline contract is missing: {source}")
    return BaselineContract.model_validate_json(source.read_text(encoding="utf-8"))


def load_prepared_manifest(path: str | Path) -> PreparedManifest:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"prepared baseline manifest is missing: {source}")
    return PreparedManifest.model_validate_json(source.read_text(encoding="utf-8"))


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _contract_hash(contract_or_path: BaselineContract | str | Path) -> str:
    if isinstance(contract_or_path, (str, Path)):
        return _sha256(Path(contract_or_path).resolve())
    return _stable_hash(contract_or_path.model_dump(mode="json"))


def _verify_dependency(
    contract: BaselineContract,
    stage_root: Path,
    split_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage.verify_stage_adapter(stage_root)
    qualification = _read_json(stage_root / "qualification.json")
    analysis = _read_json(stage_root / "analysis_manifest.json")
    if (
        qualification.get("qualification_status")
        != contract.stage_adapter.required_qualification_status
        or qualification.get("model_launch_status")
        != contract.stage_adapter.required_model_launch_status
        or qualification.get("gates", {}).get("pilot_embryo_overlap_closed") is not True
        or qualification.get("gates", {}).get("sealed_test_opened") is not False
    ):
        raise ValueError("stage adapter is not eligible for gene-disjoint baselines")
    if analysis.get("contract_sha256") != contract.stage_adapter.contract_sha256:
        raise ValueError("stage-adapter contract identity differs")
    split_hash = _sha256(split_manifest_path)
    if analysis.get("parent_split_manifest_sha256") != split_hash:
        raise ValueError("split manifest is not the stage adapter's qualified parent")
    split = _read_json(split_manifest_path)
    entries = split.get("claim_bearing_genes")
    if not isinstance(entries, list):
        raise ValueError("split manifest has no claim-bearing gene entries")
    by_partition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_genes: set[str] = set()
    seen_embryos: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("split manifest gene entry is malformed")
        gene_id = item.get("gene_id")
        partition = item.get("partition")
        embryo_ids = item.get("embryo_ids")
        if (
            not isinstance(gene_id, str)
            or gene_id in seen_genes
            or partition not in PARTITION_ORDER
            or not isinstance(embryo_ids, list)
            or not embryo_ids
        ):
            raise ValueError("split manifest violates whole-gene partitioning")
        if any(not isinstance(value, str) or value in seen_embryos for value in embryo_ids):
            raise ValueError("split manifest embryo identities overlap")
        seen_genes.add(gene_id)
        seen_embryos.update(embryo_ids)
        by_partition[partition].append(item)
    expected = {
        "train": contract.split.train_genes,
        "validation": contract.split.validation_genes,
        "sealed_test": contract.split.sealed_test_genes,
    }
    observed = {name: len(by_partition[name]) for name in PARTITION_ORDER}
    if observed != expected:
        raise ValueError(
            f"whole-gene split counts differ: expected={expected}, observed={observed}"
        )
    exposed = split.get("development_only_exposed_pilot_genes", [])
    if not isinstance(exposed, list):
        raise ValueError("development-only partition is malformed")
    if seen_genes & {str(item.get("gene_id")) for item in exposed if isinstance(item, dict)}:
        raise ValueError("pilot-exposed genes overlap claim-bearing partitions")
    return split, analysis


def _artifact_by_partition(manifest: PreparedManifest) -> dict[str, ArtifactContract]:
    return {item.partition: item for item in manifest.artifacts}


def _load_partition(
    prepared_root: Path,
    artifact: ArtifactContract,
    feature_schema: FeatureSchema,
    expected_genes: Mapping[str, set[str]],
) -> list[Observation]:
    path = prepared_root / artifact.filename
    if not path.is_file() or path.stat().st_size != artifact.bytes:
        raise ValueError(f"{artifact.partition}: prepared artifact byte identity differs")
    if _sha256(path) != artifact.sha256:
        raise ValueError(f"{artifact.partition}: prepared artifact SHA-256 differs")
    payload = PartitionPayload.model_validate_json(path.read_text(encoding="utf-8"))
    if payload.partition != artifact.partition:
        raise ValueError("prepared artifact partition label differs")
    flat_width = len(feature_schema.flat_feature_names)
    semantic_width = len(feature_schema.semantic_feature_names)
    genes: set[str] = set()
    for row in payload.observations:
        if len(row.flat_features) != flat_width or len(row.semantic_features) != semantic_width:
            raise ValueError("prepared feature width differs from the frozen schema")
        for value in [*row.flat_features, *row.semantic_features]:
            if value is not None and not math.isfinite(value):
                raise ValueError("prepared features must be finite or null")
        genes.add(row.wormbase_gene_id)
    if genes != expected_genes[artifact.partition]:
        raise ValueError(f"{artifact.partition}: prepared genes differ from the frozen split")
    if len(genes) != artifact.genes or len(payload.observations) != artifact.embryos:
        raise ValueError(f"{artifact.partition}: prepared counts differ")
    return payload.observations


def _expected_gene_sets(split: Mapping[str, Any]) -> dict[str, set[str]]:
    result = {name: set() for name in PARTITION_ORDER}
    for item in split["claim_bearing_genes"]:
        result[str(item["partition"])].add(str(item["gene_id"]))
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = _rankdata(np.asarray(left, dtype=np.float64))
    y = _rankdata(np.asarray(right, dtype=np.float64))
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def gene_macro_metrics(
    observed: Sequence[float],
    predicted: Sequence[float],
    gene_ids: Sequence[str],
) -> dict[str, float | int | None]:
    if not observed or len(observed) != len(predicted) or len(observed) != len(gene_ids):
        raise ValueError("metric arrays must be nonempty and aligned")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, gene_id in enumerate(gene_ids):
        groups[gene_id].append(index)
    observed_array = np.asarray(observed, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    if not np.isfinite(observed_array).all() or not np.isfinite(predicted_array).all():
        raise ValueError("metric arrays must be finite")
    rmses: list[float] = []
    maes: list[float] = []
    observed_means: list[float] = []
    predicted_means: list[float] = []
    for gene_id in sorted(groups):
        indexes = groups[gene_id]
        residual = predicted_array[indexes] - observed_array[indexes]
        rmses.append(float(np.sqrt(np.mean(residual**2))))
        maes.append(float(np.mean(np.abs(residual))))
        observed_means.append(float(np.mean(observed_array[indexes])))
        predicted_means.append(float(np.mean(predicted_array[indexes])))
    prediction_variance = float(np.var(predicted_means))
    if prediction_variance == 0.0:
        slope = None
        intercept = float(np.mean(observed_means))
    else:
        slope = float(
            np.mean(
                (np.asarray(predicted_means) - np.mean(predicted_means))
                * (np.asarray(observed_means) - np.mean(observed_means))
            )
            / prediction_variance
        )
        intercept = float(np.mean(observed_means) - slope * np.mean(predicted_means))
    return {
        "genes": len(groups),
        "embryos": len(observed),
        "gene_macro_rmse": float(np.mean(rmses)),
        "gene_macro_mae": float(np.mean(maes)),
        "gene_mean_spearman": _spearman(observed_means, predicted_means),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def clustered_gene_bootstrap(
    observed: Sequence[float],
    predicted: Sequence[float],
    gene_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
    quantiles: Sequence[float] = (0.025, 0.975),
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, gene_id in enumerate(gene_ids):
        groups[gene_id].append(index)
    genes = sorted(groups)
    if len(genes) < 2:
        raise ValueError("clustered bootstrap requires at least two genes")
    observed_array = np.asarray(observed, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sampled = rng.choice(genes, size=len(genes), replace=True)
        sampled_observed: list[float] = []
        sampled_predicted: list[float] = []
        sampled_gene_ids: list[str] = []
        for draw, gene in enumerate(sampled):
            synthetic_gene = f"draw-{draw}:{gene}"
            for index in groups[str(gene)]:
                sampled_observed.append(float(observed_array[index]))
                sampled_predicted.append(float(predicted_array[index]))
                sampled_gene_ids.append(synthetic_gene)
        metrics = gene_macro_metrics(sampled_observed, sampled_predicted, sampled_gene_ids)
        for name in (
            "gene_macro_rmse",
            "gene_macro_mae",
            "gene_mean_spearman",
            "calibration_intercept",
            "calibration_slope",
        ):
            value = metrics[name]
            if value is not None:
                values[name].append(float(value))
    intervals: dict[str, Any] = {}
    for name in (
        "gene_macro_rmse",
        "gene_macro_mae",
        "gene_mean_spearman",
        "calibration_intercept",
        "calibration_slope",
    ):
        samples = values[name]
        intervals[name] = (
            None
            if not samples
            else {
                "lower": float(np.quantile(samples, quantiles[0])),
                "upper": float(np.quantile(samples, quantiles[1])),
                "finite_replicates": len(samples),
            }
        )
    return {
        "resampling_unit": "whole_gene_cluster",
        "replicates": replicates,
        "seed": seed,
        "quantiles": list(quantiles),
        "intervals": intervals,
    }


def _matrix(rows: Sequence[Observation], feature_kind: str) -> np.ndarray:
    values = [
        row.flat_features if feature_kind == "flat" else row.semantic_features
        for row in rows
    ]
    return np.asarray(
        [[np.nan if item is None else float(item) for item in row] for row in values],
        dtype=np.float64,
    )


def _fit_preprocessor(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("feature matrix must be nonempty and two-dimensional")
    valid = np.isfinite(matrix)
    support = valid.sum(axis=0)
    if np.any(support == 0):
        raise ValueError("a prepared feature is missing in every fitting observation")
    means = np.nansum(matrix, axis=0) / support
    imputed = np.where(valid, matrix, means)
    scales = np.std(imputed, axis=0)
    scales[scales == 0.0] = 1.0
    return {
        "means": means.tolist(),
        "scales": scales.tolist(),
        "input_width": int(matrix.shape[1]),
        "output_width": int(matrix.shape[1] * 2),
        "missingness_indicators_appended": True,
    }


def _apply_preprocessor(matrix: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
    means = np.asarray(state["means"], dtype=np.float64)
    scales = np.asarray(state["scales"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != means.size:
        raise ValueError("feature matrix width differs from fitted preprocessor")
    missing = ~np.isfinite(matrix)
    imputed = np.where(missing, means, matrix)
    standardized = (imputed - means) / scales
    return np.concatenate((standardized, missing.astype(np.float64)), axis=1)


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    design = np.concatenate((x, np.ones((x.shape[0], 1), dtype=np.float64)), axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    try:
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError as exc:
        raise ValueError("ridge fit is singular after regularization") from exc
    return {"weights": weights[:-1].tolist(), "intercept": float(weights[-1]), "alpha": alpha}


def _predict_ridge(state: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    return x @ np.asarray(state["weights"], dtype=np.float64) + float(state["intercept"])


def _mlp_width(input_width: int, contract: NonlinearContract) -> tuple[int, int, float]:
    width = max(1, round((contract.target_trainable_parameters - 1) / (input_width + 2)))
    actual = width * (input_width + 2) + 1
    delta = (
        abs(actual - contract.target_trainable_parameters)
        / contract.target_trainable_parameters
    )
    if delta > contract.maximum_relative_parameter_delta:
        raise ValueError("edge-free MLP cannot match the frozen parameter budget")
    return width, actual, delta


def _fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    nonlinear: NonlinearContract,
    *,
    learning_rate: float,
    l2: float,
    seed: int,
) -> dict[str, Any]:
    hidden, parameter_count, relative_delta = _mlp_width(x.shape[1], nonlinear)
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 1.0 / math.sqrt(x.shape[1]), size=(x.shape[1], hidden))
    b1 = np.zeros(hidden, dtype=np.float64)
    w2 = rng.normal(0.0, 1.0 / math.sqrt(hidden), size=hidden)
    b2 = 0.0
    y_mean = float(np.mean(y))
    y_scale = float(np.std(y)) or 1.0
    target = (y - y_mean) / y_scale
    parameters: list[np.ndarray] = [w1, b1, w2]
    moments = [np.zeros_like(value) for value in parameters]
    velocities = [np.zeros_like(value) for value in parameters]
    b2_m = 0.0
    b2_v = 0.0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for epoch in range(1, nonlinear.epochs + 1):
        hidden_values = np.tanh(x @ w1 + b1)
        prediction = hidden_values @ w2 + b2
        residual = prediction - target
        grad_prediction = 2.0 * residual / x.shape[0]
        grad_w2 = hidden_values.T @ grad_prediction + 2.0 * l2 * w2
        grad_b2 = float(np.sum(grad_prediction))
        grad_hidden = np.outer(grad_prediction, w2) * (1.0 - hidden_values**2)
        grad_w1 = x.T @ grad_hidden + 2.0 * l2 * w1
        grad_b1 = np.sum(grad_hidden, axis=0)
        gradients = [grad_w1, grad_b1, grad_w2]
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
            moments[index] = beta1 * moments[index] + (1.0 - beta1) * gradient
            velocities[index] = beta2 * velocities[index] + (1.0 - beta2) * gradient**2
            corrected_m = moments[index] / (1.0 - beta1**epoch)
            corrected_v = velocities[index] / (1.0 - beta2**epoch)
            parameter -= learning_rate * corrected_m / (np.sqrt(corrected_v) + epsilon)
        b2_m = beta1 * b2_m + (1.0 - beta1) * grad_b2
        b2_v = beta2 * b2_v + (1.0 - beta2) * grad_b2**2
        b2 -= learning_rate * (b2_m / (1.0 - beta1**epoch)) / (
            math.sqrt(b2_v / (1.0 - beta2**epoch)) + epsilon
        )
    return {
        "w1": w1.tolist(),
        "b1": b1.tolist(),
        "w2": w2.tolist(),
        "b2": b2,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "learning_rate": learning_rate,
        "l2": l2,
        "seed": seed,
        "epochs": nonlinear.epochs,
        "hidden_width": hidden,
        "parameter_count": parameter_count,
        "relative_parameter_delta": relative_delta,
        "edge_access": False,
    }


def _predict_mlp(state: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    w1 = np.asarray(state["w1"], dtype=np.float64)
    b1 = np.asarray(state["b1"], dtype=np.float64)
    w2 = np.asarray(state["w2"], dtype=np.float64)
    standardized = np.tanh(x @ w1 + b1) @ w2 + float(state["b2"])
    return standardized * float(state["y_scale"]) + float(state["y_mean"])


def _targets(rows: Sequence[Observation]) -> np.ndarray:
    return np.asarray([row.outcome for row in rows], dtype=np.float64)


def _gene_ids(rows: Sequence[Observation]) -> list[str]:
    return [row.wormbase_gene_id for row in rows]


def _embryo_ids(rows: Sequence[Observation]) -> list[str]:
    return [row.embryo_id for row in rows]


def _candidate_key(metrics: Mapping[str, Any], candidate_id: str) -> tuple[float, float, str]:
    return (
        float(metrics["gene_macro_rmse"]),
        float(metrics["gene_macro_mae"]),
        candidate_id,
    )


def _fit_family_candidates(
    contract: BaselineContract,
    family: str,
    train: Sequence[Observation],
    validation: Sequence[Observation],
    control_mean: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    y_train = _targets(train)
    y_validation = _targets(validation)
    validation_genes = _gene_ids(validation)
    candidates: list[tuple[str, dict[str, Any], np.ndarray]] = []
    if family == "constant_control_mean":
        candidate_id = "control_mean"
        state = {"family": family, "value": control_mean, "feature_kind": None}
        prediction = np.full(len(validation), control_mean, dtype=np.float64)
        candidates.append((candidate_id, state, prediction))
    else:
        feature_kind = "flat" if family == "compact_flat_ridge" else "semantic"
        train_raw = _matrix(train, feature_kind)
        validation_raw = _matrix(validation, feature_kind)
        preprocessing = _fit_preprocessor(train_raw)
        x_train = _apply_preprocessor(train_raw, preprocessing)
        x_validation = _apply_preprocessor(validation_raw, preprocessing)
        if family in {"compact_flat_ridge", "semantic_cell_aligned_linear"}:
            ridge = (
                contract.models.flat_ridge
                if family == "compact_flat_ridge"
                else contract.models.semantic_linear
            )
            for alpha in ridge.alphas:
                candidate_id = f"alpha={alpha:.12g}"
                model = _fit_ridge(x_train, y_train, alpha)
                state = {
                    "family": family,
                    "feature_kind": feature_kind,
                    "preprocessing": preprocessing,
                    "model": model,
                }
                candidates.append((candidate_id, state, _predict_ridge(model, x_validation)))
        elif family == "nonlinear_edge_free":
            nonlinear = contract.models.nonlinear_edge_free
            for learning_rate in nonlinear.learning_rates:
                for l2 in nonlinear.l2_penalties:
                    for seed in nonlinear.seeds:
                        candidate_id = f"lr={learning_rate:.12g}|l2={l2:.12g}|seed={seed}"
                        model = _fit_mlp(
                            x_train,
                            y_train,
                            nonlinear,
                            learning_rate=learning_rate,
                            l2=l2,
                            seed=seed,
                        )
                        state = {
                            "family": family,
                            "feature_kind": feature_kind,
                            "preprocessing": preprocessing,
                            "model": model,
                        }
                        candidates.append((candidate_id, state, _predict_mlp(model, x_validation)))
        else:  # pragma: no cover
            raise AssertionError(family)
    scored: list[dict[str, Any]] = []
    for candidate_id, state, prediction in candidates:
        metrics = gene_macro_metrics(y_validation.tolist(), prediction.tolist(), validation_genes)
        scored.append({"candidate_id": candidate_id, "metrics": metrics})
    best_index = min(
        range(len(candidates)),
        key=lambda index: _candidate_key(scored[index]["metrics"], candidates[index][0]),
    )
    candidate_id, state, prediction = candidates[best_index]
    selected = {
        "candidate_id": candidate_id,
        "validation_metrics": scored[best_index]["metrics"],
        "selection_partition": "validation",
        "sealed_test_used": False,
        "selection_state": state,
    }
    return selected, scored, prediction.tolist()


def _refit_selected(
    contract: BaselineContract,
    family: str,
    selected: Mapping[str, Any],
    rows: Sequence[Observation],
    control_mean: float,
) -> dict[str, Any]:
    if family == "constant_control_mean":
        return {"family": family, "value": control_mean, "feature_kind": None}
    selection_state = selected["selection_state"]
    feature_kind = str(selection_state["feature_kind"])
    raw = _matrix(rows, feature_kind)
    preprocessing = _fit_preprocessor(raw)
    x = _apply_preprocessor(raw, preprocessing)
    y = _targets(rows)
    if family in {"compact_flat_ridge", "semantic_cell_aligned_linear"}:
        alpha = float(selection_state["model"]["alpha"])
        model = _fit_ridge(x, y, alpha)
    else:
        previous = selection_state["model"]
        model = _fit_mlp(
            x,
            y,
            contract.models.nonlinear_edge_free,
            learning_rate=float(previous["learning_rate"]),
            l2=float(previous["l2"]),
            seed=int(previous["seed"]),
        )
    return {
        "family": family,
        "feature_kind": feature_kind,
        "preprocessing": preprocessing,
        "model": model,
        "fit_partitions": ["train", "validation"],
        "sealed_test_used": False,
    }


def _predict_state(state: Mapping[str, Any], rows: Sequence[Observation]) -> np.ndarray:
    family = str(state["family"])
    if family == "constant_control_mean":
        return np.full(len(rows), float(state["value"]), dtype=np.float64)
    raw = _matrix(rows, str(state["feature_kind"]))
    x = _apply_preprocessor(raw, state["preprocessing"])
    if family in {"compact_flat_ridge", "semantic_cell_aligned_linear"}:
        return _predict_ridge(state["model"], x)
    if family == "nonlinear_edge_free":
        return _predict_mlp(state["model"], x)
    raise ValueError(f"unknown fitted family: {family}")


def _write_bundle(root: Path, payloads: Mapping[str, Any], expected_files: set[str]) -> None:
    if set(payloads) != expected_files:
        raise ValueError("write-once bundle payload inventory differs")
    if root.exists():
        raise FileExistsError(f"output root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = root.parent / f".{root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for name in sorted(payloads):
            _write_json_atomic(temporary / name, payloads[name])
        lines = [f"{_sha256(temporary / name)}  {name}" for name in sorted(payloads)]
        _write_bytes_atomic(temporary / "SHA256SUMS.txt", ("\n".join(lines) + "\n").encode())
        _write_bytes_atomic(temporary / "SUCCESS", b"SUCCESS\n")
        os.replace(temporary, root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _verify_bundle(root: Path, expected_files: set[str]) -> None:
    expected = expected_files | {"SHA256SUMS.txt", "SUCCESS"}
    if not root.is_dir() or {item.name for item in root.iterdir()} != expected:
        raise ValueError("write-once bundle inventory differs")
    if (root / "SUCCESS").read_bytes() != b"SUCCESS\n":
        raise ValueError("write-once bundle SUCCESS marker differs")
    lines = (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    checked: set[str] = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None or match.group(2) not in expected_files or match.group(2) in checked:
            raise ValueError("write-once checksum manifest is malformed")
        if _sha256(root / match.group(2)) != match.group(1):
            raise ValueError(f"write-once checksum failed: {match.group(2)}")
        checked.add(match.group(2))
    if checked != expected_files:
        raise ValueError("write-once checksum coverage differs")


def _prepared_dependencies(
    manifest_path: Path,
    manifest: PreparedManifest,
    stage_root: Path,
    split_path: Path,
) -> None:
    if manifest.stage_adapter_qualification_sha256 != _sha256(
        stage_root / "qualification.json"
    ):
        raise ValueError("prepared data bind a different stage qualification")
    if manifest.source_stage_membership_sha256 != _sha256(stage_root / "stage_membership.json"):
        raise ValueError("prepared data bind different stage membership")
    if manifest.split_manifest_sha256 != _sha256(split_path):
        raise ValueError("prepared data bind a different gene split")
    if not manifest_path.is_file():  # pragma: no cover - defensive after load
        raise FileNotFoundError(manifest_path)


def fit_select(
    contract_or_path: BaselineContract | str | Path,
    stage_adapter_root: str | Path,
    split_manifest_path: str | Path,
    prepared_manifest_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    contract = (
        contract_or_path
        if isinstance(contract_or_path, BaselineContract)
        else load_contract(contract_or_path)
    )
    stage_root = Path(stage_adapter_root).resolve()
    split_path = Path(split_manifest_path).resolve()
    split, _stage_analysis = _verify_dependency(contract, stage_root, split_path)
    manifest_path = Path(prepared_manifest_path).resolve()
    manifest = load_prepared_manifest(manifest_path)
    _prepared_dependencies(manifest_path, manifest, stage_root, split_path)
    prepared_root = manifest_path.parent
    artifacts = _artifact_by_partition(manifest)
    expected_genes = _expected_gene_sets(split)
    # Intentional gate: the sealed-test artifact is not stat'ed, hashed, opened, or parsed here.
    train = _load_partition(
        prepared_root,
        artifacts["train"],
        manifest.feature_schema,
        expected_genes,
    )
    validation = _load_partition(
        prepared_root,
        artifacts["validation"],
        manifest.feature_schema,
        expected_genes,
    )
    selected: dict[str, Any] = {}
    candidate_metrics: dict[str, Any] = {}
    validation_predictions: list[dict[str, Any]] = []
    control_mean = manifest.feature_schema.control_standardized_expected_mean
    for family in FAMILY_ORDER:
        winner, scored, predictions = _fit_family_candidates(
            contract,
            family,
            train,
            validation,
            control_mean,
        )
        selected[family] = winner
        candidate_metrics[family] = scored
        for row, prediction in zip(validation, predictions, strict=True):
            validation_predictions.append(
                {
                    "family": family,
                    "embryo_id": row.embryo_id,
                    "wormbase_gene_id": row.wormbase_gene_id,
                    "observed": row.outcome,
                    "predicted": prediction,
                }
            )
    combined = [*train, *validation]
    final_states = {
        family: _refit_selected(contract, family, selected[family], combined, control_mean)
        for family in FAMILY_ORDER
    }
    manifest_hash = _sha256(manifest_path)
    selection_manifest = {
        "schema_version": SELECTION_VERSION,
        "analysis_id": contract.analysis_id,
        "contract_sha256": _contract_hash(contract_or_path),
        "prepared_manifest_sha256": manifest_hash,
        "stage_adapter_qualification_sha256": manifest.stage_adapter_qualification_sha256,
        "split_manifest_sha256": manifest.split_manifest_sha256,
        "fit_gene_counts": {
            "train": len(expected_genes["train"]),
            "validation": len(expected_genes["validation"]),
        },
        "family_order": FAMILY_ORDER,
        "selection_partition": "validation",
        "refit_partitions": ["train", "validation"],
        "sealed_test_artifact_declared_sha256": artifacts["sealed_test"].sha256,
        "sealed_test_artifact_bytes_read": 0,
        "sealed_test_deserialized": False,
        "sealed_test_openings": 0,
        "ready_for_single_sealed_evaluation": True,
    }
    payloads = {
        "model_states.json": {
            "schema_version": SELECTION_VERSION,
            "states": final_states,
            "sealed_test_used": False,
        },
        "selection_manifest.json": selection_manifest,
        "validation_metrics.json": {
            "schema_version": SELECTION_VERSION,
            "selected": {
                family: {
                    key: value
                    for key, value in selected[family].items()
                    if key != "selection_state"
                }
                for family in FAMILY_ORDER
            },
            "candidates": candidate_metrics,
        },
        "validation_predictions.json": {
            "schema_version": SELECTION_VERSION,
            "partition": "validation",
            "predictions": validation_predictions,
        },
    }
    output = Path(output_root).resolve()
    _write_bundle(output, payloads, SELECTION_FILES)
    return selection_manifest


def verify_selection(selection_root: str | Path) -> dict[str, Any]:
    root = Path(selection_root).resolve()
    _verify_bundle(root, SELECTION_FILES)
    manifest = _read_json(root / "selection_manifest.json")
    if (
        manifest.get("schema_version") != SELECTION_VERSION
        or manifest.get("sealed_test_deserialized") is not False
        or manifest.get("sealed_test_artifact_bytes_read") != 0
        or manifest.get("ready_for_single_sealed_evaluation") is not True
    ):
        raise ValueError("selection manifest violates the sealed-test gate")
    return {
        "schema_version": SELECTION_VERIFY_VERSION,
        "verified": True,
        "selection_manifest_sha256": _sha256(root / "selection_manifest.json"),
        "sealed_test_openings": 0,
    }


def sealed_gate_name(
    prepared_manifest_sha256: str,
    selection_manifest_sha256: str,
    sealed_artifact_sha256: str,
    prefix: str = "sealed-test-gate-",
) -> str:
    token = hashlib.sha256(
        f"{prepared_manifest_sha256}|{selection_manifest_sha256}|{sealed_artifact_sha256}".encode()
    ).hexdigest()
    return f"{prefix}{token[:16]}"


def _open_gate(
    gate_root: Path,
    *,
    expected_name: str,
    payload: Mapping[str, Any],
) -> None:
    if gate_root.name != expected_name:
        raise ValueError(f"sealed gate name differs; required={expected_name}")
    try:
        gate_root.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError("sealed test has already been opened for this selection") from exc
    _write_json_atomic(gate_root / "OPENED.json", payload)


def evaluate_sealed(
    contract_or_path: BaselineContract | str | Path,
    stage_adapter_root: str | Path,
    split_manifest_path: str | Path,
    prepared_manifest_path: str | Path,
    selection_root: str | Path,
    sealed_gate_root: str | Path,
    output_root: str | Path,
    *,
    open_sealed_test_once: bool,
) -> dict[str, Any]:
    if not open_sealed_test_once:
        raise PermissionError("explicit single sealed-test opening attestation is required")
    contract = (
        contract_or_path
        if isinstance(contract_or_path, BaselineContract)
        else load_contract(contract_or_path)
    )
    stage_root = Path(stage_adapter_root).resolve()
    split_path = Path(split_manifest_path).resolve()
    split, _stage_analysis = _verify_dependency(contract, stage_root, split_path)
    manifest_path = Path(prepared_manifest_path).resolve()
    manifest = load_prepared_manifest(manifest_path)
    _prepared_dependencies(manifest_path, manifest, stage_root, split_path)
    selection_path = Path(selection_root).resolve()
    selection_verification = verify_selection(selection_path)
    selection_manifest = _read_json(selection_path / "selection_manifest.json")
    prepared_hash = _sha256(manifest_path)
    if selection_manifest.get("prepared_manifest_sha256") != prepared_hash:
        raise ValueError("selection bundle binds a different prepared dataset")
    if selection_manifest.get("contract_sha256") != _contract_hash(contract_or_path):
        raise ValueError("selection bundle binds a different baseline contract")
    artifacts = _artifact_by_partition(manifest)
    sealed_artifact = artifacts["sealed_test"]
    evaluation_output = Path(output_root).resolve()
    if evaluation_output.exists():
        raise FileExistsError(f"sealed evaluation output root already exists: {evaluation_output}")
    required_gate_name = sealed_gate_name(
        prepared_hash,
        selection_verification["selection_manifest_sha256"],
        sealed_artifact.sha256,
        contract.evaluation.gate_name_prefix,
    )
    gate = Path(sealed_gate_root).resolve()
    _open_gate(
        gate,
        expected_name=required_gate_name,
        payload={
            "schema_version": SEALED_GATE_VERSION,
            "prepared_manifest_sha256": prepared_hash,
            "selection_manifest_sha256": selection_verification[
                "selection_manifest_sha256"
            ],
            "declared_sealed_artifact_sha256": sealed_artifact.sha256,
            "opening_ordinal": 1,
            "selection_complete_before_opening": True,
        },
    )
    try:
        expected_genes = _expected_gene_sets(split)
        test_rows = _load_partition(
            manifest_path.parent,
            sealed_artifact,
            manifest.feature_schema,
            expected_genes,
        )
        model_payload = _read_json(selection_path / "model_states.json")
        states = model_payload.get("states")
        if not isinstance(states, dict) or set(states) != set(FAMILY_ORDER):
            raise ValueError("selected model state inventory differs")
        y = _targets(test_rows)
        genes = _gene_ids(test_rows)
        predictions_payload: list[dict[str, Any]] = []
        metrics_payload: dict[str, Any] = {}
        bootstrap_payload: dict[str, Any] = {}
        for family in FAMILY_ORDER:
            prediction = _predict_state(states[family], test_rows)
            metrics_payload[family] = gene_macro_metrics(y.tolist(), prediction.tolist(), genes)
            bootstrap_payload[family] = clustered_gene_bootstrap(
                y.tolist(),
                prediction.tolist(),
                genes,
                replicates=contract.evaluation.bootstrap_replicates,
                seed=contract.evaluation.bootstrap_seed,
                quantiles=contract.evaluation.interval_quantiles,
            )
            for row, value in zip(test_rows, prediction, strict=True):
                predictions_payload.append(
                    {
                        "family": family,
                        "embryo_id": row.embryo_id,
                        "wormbase_gene_id": row.wormbase_gene_id,
                        "observed": row.outcome,
                        "predicted": float(value),
                    }
                )
        evaluation_manifest = {
            "schema_version": SEALED_EVALUATION_VERSION,
            "analysis_id": contract.analysis_id,
            "contract_sha256": _contract_hash(contract_or_path),
            "prepared_manifest_sha256": prepared_hash,
            "selection_manifest_sha256": selection_verification[
                "selection_manifest_sha256"
            ],
            "sealed_artifact_sha256": _sha256(manifest_path.parent / sealed_artifact.filename),
            "sealed_gate_opened_sha256": _sha256(gate / "OPENED.json"),
            "sealed_test_opening_ordinal": 1,
            "sealed_test_genes": len(expected_genes["sealed_test"]),
            "sealed_test_embryos": len(test_rows),
            "selection_reopened": False,
            "biological_claims_permitted": False,
        }
        payloads = {
            "evaluation_manifest.json": evaluation_manifest,
            "test_bootstrap.json": {
                "schema_version": SEALED_EVALUATION_VERSION,
                "families": bootstrap_payload,
            },
            "test_metrics.json": {
                "schema_version": SEALED_EVALUATION_VERSION,
                "families": metrics_payload,
            },
            "test_predictions.json": {
                "schema_version": SEALED_EVALUATION_VERSION,
                "partition": "sealed_test",
                "predictions": predictions_payload,
            },
        }
        _write_bundle(evaluation_output, payloads, EVALUATION_FILES)
        _write_bytes_atomic(gate / "SUCCESS", b"SUCCESS\n")
        return evaluation_manifest
    except BaseException:
        _write_bytes_atomic(gate / "FAILURE", b"FAILURE_AFTER_SEALED_OPEN\n")
        raise


def verify_evaluation(evaluation_root: str | Path) -> dict[str, Any]:
    root = Path(evaluation_root).resolve()
    _verify_bundle(root, EVALUATION_FILES)
    manifest = _read_json(root / "evaluation_manifest.json")
    if (
        manifest.get("schema_version") != SEALED_EVALUATION_VERSION
        or manifest.get("sealed_test_opening_ordinal") != 1
        or manifest.get("selection_reopened") is not False
        or manifest.get("biological_claims_permitted") is not False
    ):
        raise ValueError("sealed evaluation manifest differs")
    return {
        "schema_version": EVALUATION_VERIFY_VERSION,
        "verified": True,
        "evaluation_manifest_sha256": _sha256(root / "evaluation_manifest.json"),
        "sealed_test_opening_ordinal": 1,
    }


def _emit(payload: Any, output: str | Path | None = None) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(content)
    else:
        _write_bytes_atomic(Path(output).resolve(), content.encode("utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wormctx.poc.developmental_baselines",
        description="Fit gene-disjoint developmental baselines and open the sealed test once",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("fit-select")
    select.add_argument("--contract", required=True)
    select.add_argument("--stage-adapter-root", required=True)
    select.add_argument("--split-manifest", required=True)
    select.add_argument("--prepared-manifest", required=True)
    select.add_argument("--output-root", required=True)
    sealed = commands.add_parser("evaluate-sealed")
    sealed.add_argument("--contract", required=True)
    sealed.add_argument("--stage-adapter-root", required=True)
    sealed.add_argument("--split-manifest", required=True)
    sealed.add_argument("--prepared-manifest", required=True)
    sealed.add_argument("--selection-root", required=True)
    sealed.add_argument("--sealed-gate-root", required=True)
    sealed.add_argument("--output-root", required=True)
    sealed.add_argument("--open-sealed-test-once", action="store_true")
    verify_select = commands.add_parser("verify-selection")
    verify_select.add_argument("--selection-root", required=True)
    verify_select.add_argument("--output")
    verify_eval = commands.add_parser("verify-evaluation")
    verify_eval.add_argument("--evaluation-root", required=True)
    verify_eval.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fit-select":
        payload = fit_select(
            args.contract,
            args.stage_adapter_root,
            args.split_manifest,
            args.prepared_manifest,
            args.output_root,
        )
        _emit(payload)
    elif args.command == "evaluate-sealed":
        payload = evaluate_sealed(
            args.contract,
            args.stage_adapter_root,
            args.split_manifest,
            args.prepared_manifest,
            args.selection_root,
            args.sealed_gate_root,
            args.output_root,
            open_sealed_test_once=args.open_sealed_test_once,
        )
        _emit(payload)
    elif args.command == "verify-selection":
        _emit(verify_selection(args.selection_root), args.output)
    else:
        _emit(verify_evaluation(args.evaluation_root), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
