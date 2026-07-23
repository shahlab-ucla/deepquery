"""Fail-closed prospective transfer prospective RNAi and cross-background transfer closure.

No experiment or model is invoked here. The module validates and freezes a preregistration,
locks the complete blinded raw/QC universe before any partial release, and then closes staged
adaptation and interpretation bundles whose decision rules were fixed before results existed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Literal, Mapping, Sequence

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "wormctx-prospective_transfer-prospective-transfer-contract-1.0"
PREREG_VERSION = "wormctx-prospective_transfer-preregistration-1.0"
RESULT_INPUT_VERSION = "wormctx-prospective_transfer-result-input-1.0"
ADAPTATION_VERSION = "wormctx-prospective_transfer-adaptation-forecast-1.0"
BLINDED_RESULT_LOCK_VERSION = "wormctx-prospective_transfer-blinded-result-lock-1.0"
SUMMARY_VERSION = "wormctx-prospective_transfer-result-summary-1.0"
INTERPRETATION_VERSION = "wormctx-prospective_transfer-result-interpretation-1.0"
PREREG_BUNDLE_VERSION = "wormctx-prospective_transfer-preregistration-bundle-1.0"
RESULT_BUNDLE_VERSION = "wormctx-prospective_transfer-result-bundle-1.0"
ADAPTATION_BUNDLE_VERSION = "wormctx-prospective_transfer-adaptation-bundle-1.0"
BLINDED_RESULT_LOCK_BUNDLE_VERSION = "wormctx-prospective_transfer-blinded-result-lock-bundle-1.0"

PANEL_SIZE = 12
CANDIDATE_COUNT = 8
POSITIVE_CONTROL_COUNT = 2
NEGATIVE_CONTROL_COUNT = 2
BACKGROUND_COUNT = 3
REPLICATES = 3
SELECTION_ARMS = ("graph_eig", "random", "expert", "uncertainty_only")
ARCHITECTURES = ("graph", "edge_free")
TRANSFER_REGIMES = ("zero_shot", "low_shot", "retrained")
LINEAGE_LOCATIONS = (
    "AB",
    "MS",
    "E",
    "C",
    "D",
    "P4",
    "distributed",
    "none_detected",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _object_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PanelContract(_StrictModel):
    total_genes: Literal[12]
    candidates: Literal[8]
    positive_controls: Literal[2]
    negative_controls: Literal[2]
    selection_arms: list[str]
    candidates_per_selection_arm: Literal[2]
    candidate_sources_frozen_before_results: Literal[True]
    sealed_test_gene_selection_permitted: Literal[False]

    @model_validator(mode="after")
    def exact_panel(self) -> "PanelContract":
        if self.selection_arms != list(SELECTION_ARMS):
            raise ValueError("selection arms differ from the frozen contract")
        return self


class ForecastContract(_StrictModel):
    architectures: list[str]
    transfer_regimes: list[str]
    later_outcome: Literal["later_reporter_deviation_standardized"]
    lineage_vocabulary: list[str]
    uncertainty: Literal["prediction_interval_and_scalar_uncertainty"]
    nominal_interval_coverage: Literal[0.9]
    minimum_architecture_coverage: Literal[0.75]
    selective_risk_metric: Literal["rmse_among_answered_forecasts"]
    uncertainty_calibration_metric: Literal["mean_absolute_error_vs_scalar_uncertainty"]
    abstention_permitted: Literal[True]
    forced_prediction_permitted: Literal[False]
    forecasts_frozen_before_result_access: Literal[True]

    @model_validator(mode="after")
    def exact_forecasts(self) -> "ForecastContract":
        if self.architectures != list(ARCHITECTURES):
            raise ValueError("forecast architectures differ")
        if self.transfer_regimes != list(TRANSFER_REGIMES):
            raise ValueError("transfer regimes differ")
        if self.lineage_vocabulary != list(LINEAGE_LOCATIONS):
            raise ValueError("lineage vocabulary differs")
        return self


class AssayContract(_StrictModel):
    intervention: Literal["RNAi"]
    reference_background: Literal["N2"]
    natural_background_count: Literal[2]
    biological_replicates_per_gene_background: Literal[3]
    minimum_embryos_per_replicate: Literal[20]
    plate_unit: Literal["background_by_biological_replicate"]
    control_requirements_per_plate: Literal[
        "at_least_one_positive_and_one_negative_control"
    ]
    control_gate_level: Literal["strict_per_plate", "background_aggregate"]
    randomized_layout_required: Literal[True]
    operator_blinded_to_forecasts: Literal[True]
    scorer_blinded_to_gene_background_and_model: Literal[True]
    unblinding_after_result_lock_only: Literal[True]
    exclusion_codes: list[str]

    @model_validator(mode="after")
    def exact_exclusions(self) -> "AssayContract":
        if self.exclusion_codes != [
            "technical_failure",
            "insufficient_embryos",
            "contamination",
            "predeclared_image_qc_failure",
        ]:
            raise ValueError("assay exclusion codes differ")
        return self


class PowerContract(_StrictModel):
    method: Literal["clustered_simulation_frozen_before_results"]
    family_alpha: Literal[0.05]
    target_power: Literal[0.8]
    minimum_detectable_standardized_effect: Literal[0.5]
    assumed_intraclass_correlation: Literal[0.1]
    assumptions_receipt_required: Literal[True]
    underpowered_result_policy: Literal["retain_and_label_inconclusive"]


class TransferContract(_StrictModel):
    backgrounds: Literal[3]
    reference_plus_haplotype_selected_natural_isolates: Literal[True]
    independent_developmental_dataset_alternative_permitted: Literal[True]
    regimes: list[str]
    holdout_unit: Literal["strain_by_gene"]
    zero_shot_training: Literal["N2_only"]
    low_shot_adaptation: Literal["natural_background_controls_only"]
    retrained_test_rule: Literal[
        "one_outcome_blind_natural_background_holdout_per_candidate_gene"
    ]
    test_cells_may_enter_training_or_adaptation: Literal[False]

    @model_validator(mode="after")
    def exact_regimes(self) -> "TransferContract":
        if self.regimes != list(TRANSFER_REGIMES):
            raise ValueError("transfer regimes differ")
        return self


class InterpretationContract(_StrictModel):
    positive_control_minimum_absolute_effect: Literal[0.5]
    negative_control_maximum_absolute_effect: Literal[0.25]
    minimum_analyzable_replicates_per_cell: Literal[2]
    maximum_graph_rmse_for_predictive_signal: Literal[0.75]
    minimum_graph_rmse_advantage: Literal[0.05]
    minimum_graph_lineage_accuracy_advantage: Literal[0.1]
    minimum_transfer_improvement: Literal[0.05]
    branches: list[str]

    @model_validator(mode="after")
    def exact_branches(self) -> "InterpretationContract":
        if self.branches != [
            "control_failure_inconclusive",
            "insufficient_data_inconclusive",
            "insufficient_forecast_coverage_inconclusive",
            "graph_advantage_supported",
            "predictive_signal_without_graph_advantage",
            "no_predictive_signal",
        ]:
            raise ValueError("interpretation branches differ")
        return self


class ClosureContract(_StrictModel):
    preregistration_publication: Literal["atomic_write_once_directory"]
    blinded_result_publication: Literal[
        "complete_atomic_write_once_before_partial_release"
    ]
    result_publication: Literal["separate_atomic_write_once_directory"]
    checksum_algorithm: Literal["sha256"]
    result_must_bind_preregistration_sha256: Literal[True]
    adaptations_must_bind_blinded_result_lock_sha256: Literal[True]
    result_must_bind_blinded_result_lock_sha256: Literal[True]
    missing_or_extra_result_policy: Literal["refuse"]
    result_replacement_policy: Literal["refuse"]


class ClaimContract(_StrictModel):
    permitted: list[str]
    prohibited: list[str]


class ProspectiveTransferManifest(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    analysis_id: Literal["prospective_transfer_prospective_rnai_transfer_closure_v1"]
    classification: Literal["synthetic_only_preregistration_and_closure_scaffold"]
    status: Literal["blocked_on_models_experiment_and_transfer_backgrounds"]
    validated: Literal[False]
    biological_claims_permitted: Literal[False]
    panel: PanelContract
    forecasts: ForecastContract
    assay: AssayContract
    power: PowerContract
    transfer: TransferContract
    interpretation: InterpretationContract
    closure: ClosureContract
    claims: ClaimContract

    @model_validator(mode="after")
    def exact_claims(self) -> "ProspectiveTransferManifest":
        prohibited = " ".join(self.claims.prohibited).lower()
        for token in ("causal", "prospective validation", "external transfer"):
            if token not in prohibited:
                raise ValueError(f"claim boundary must prohibit {token}")
        return self


def load_manifest(path: str | Path) -> ProspectiveTransferManifest:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"prospective transfer manifest is not a file: {source}")
    return ProspectiveTransferManifest.model_validate_json(source.read_text(encoding="utf-8"))


class GenePanelRecord(_StrictModel):
    gene_id: str
    panel_role: Literal["candidate", "positive_control", "negative_control"]
    selection_arm: Literal[
        "graph_eig",
        "random",
        "expert",
        "uncertainty_only",
        "positive_control",
        "negative_control",
    ]
    candidate_source: str
    source_snapshot_sha256: str
    source_score: float | None
    expected_outcome_direction: Literal[
        "increase", "decrease", "near_zero", "unspecified"
    ]
    selected_before_result_access: Literal[True]
    sealed_test_metrics_used: Literal[False]

    @field_validator("gene_id")
    @classmethod
    def safe_gene(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("gene ID must be a safe token")
        return value

    @field_validator("source_snapshot_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("candidate source SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def role_matches_arm(self) -> "GenePanelRecord":
        expected = {
            "positive_control": "positive_control",
            "negative_control": "negative_control",
        }
        if self.panel_role in expected and self.selection_arm != expected[self.panel_role]:
            raise ValueError("control role and selection arm differ")
        if self.panel_role == "candidate" and self.selection_arm not in SELECTION_ARMS:
            raise ValueError("candidate must use a comparator selection arm")
        expected_direction = {
            "candidate": {"unspecified"},
            "positive_control": {"increase", "decrease"},
            "negative_control": {"near_zero"},
        }
        if self.expected_outcome_direction not in expected_direction[self.panel_role]:
            raise ValueError("control role and expected outcome direction differ")
        return self


class BackgroundRecord(_StrictModel):
    background_id: str
    role: Literal["reference", "natural_haplotype"]
    selection_source: str
    genotype_snapshot_sha256: str
    haplotype_panel_receipt_sha256: str | None
    selected_before_result_access: Literal[True]

    @field_validator("background_id")
    @classmethod
    def safe_background(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("background ID must be a safe token")
        return value

    @field_validator("genotype_snapshot_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("genotype snapshot SHA-256 must be lowercase hexadecimal")
        return value

    @field_validator("haplotype_panel_receipt_sha256")
    @classmethod
    def valid_optional_hash(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("haplotype receipt SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def exact_reference(self) -> "BackgroundRecord":
        if self.role == "reference" and (
            self.background_id != "N2" or self.haplotype_panel_receipt_sha256 is not None
        ):
            raise ValueError("reference background must be N2 without a haplotype receipt")
        if self.role == "natural_haplotype" and self.haplotype_panel_receipt_sha256 is None:
            raise ValueError("natural background requires a haplotype receipt")
        return self


class SelectionAssignmentReceipt(_StrictModel):
    random_seed_sha256: str
    random_candidate_universe_sha256: str
    expert_panel_receipt_sha256: str
    uncertainty_score_snapshot_sha256: str
    graph_eig_score_snapshot_sha256: str
    assignments_created_before_result_access: Literal[True]
    result_data_used_for_assignment: Literal[False]

    @field_validator(
        "random_seed_sha256",
        "random_candidate_universe_sha256",
        "expert_panel_receipt_sha256",
        "uncertainty_score_snapshot_sha256",
        "graph_eig_score_snapshot_sha256",
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("selection receipt SHA-256 must be lowercase hexadecimal")
        return value


class ForecastRecord(_StrictModel):
    gene_id: str
    background_id: str
    architecture: Literal["graph", "edge_free"]
    transfer_regime: Literal["zero_shot", "low_shot", "retrained"]
    model_artifact_sha256: str
    predicted_later_outcome: float | None
    interval_lower: float | None
    interval_upper: float | None
    scalar_uncertainty: float = Field(ge=0.0)
    lineage_location: str | None
    abstain: bool
    abstention_reason: str | None
    frozen_before_test_result_access: Literal[True]

    @field_validator("model_artifact_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("model artifact SHA-256 must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def forecast_semantics(self) -> "ForecastRecord":
        values = (self.predicted_later_outcome, self.interval_lower, self.interval_upper)
        if self.abstain:
            if any(value is not None for value in values) or self.lineage_location is not None:
                raise ValueError("abstention cannot carry a forced outcome or lineage call")
            if not self.abstention_reason:
                raise ValueError("abstention requires a reason")
        else:
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError("answered forecast requires finite outcome and interval")
            assert self.predicted_later_outcome is not None
            assert self.interval_lower is not None and self.interval_upper is not None
            if not self.interval_lower <= self.predicted_later_outcome <= self.interval_upper:
                raise ValueError("forecast interval does not contain its prediction")
            if self.lineage_location not in LINEAGE_LOCATIONS:
                raise ValueError("forecast uses an unknown lineage location")
            if self.abstention_reason is not None:
                raise ValueError("answered forecast cannot carry an abstention reason")
        return self


class PowerPlan(_StrictModel):
    assumptions_sha256: str
    method: Literal["clustered_simulation_frozen_before_results"]
    family_alpha: Literal[0.05]
    target_power: Literal[0.8]
    minimum_detectable_standardized_effect: Literal[0.5]
    intraclass_correlation: Literal[0.1]
    biological_replicates: Literal[3]
    embryos_per_replicate: int = Field(ge=20)

    @field_validator("assumptions_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("power assumptions SHA-256 must be lowercase hexadecimal")
        return value


class AssayPlan(_StrictModel):
    protocol_snapshot_sha256: str
    intervention: Literal["RNAi"]
    delivery_method: str
    dose: str
    exposure_window: str
    endpoint: Literal["later_reporter_deviation_standardized"]
    minimum_embryos_per_replicate: int = Field(ge=20)
    biological_replicates: Literal[3]
    randomization_seed_sha256: str
    operator_blinded_to_forecasts: Literal[True]
    scorer_blinded_to_gene_background_and_model: Literal[True]
    sample_mapping_access: Literal["unblinded_coordinator_only"]
    unblinding_after_result_lock: Literal[True]
    power: PowerPlan

    @field_validator("protocol_snapshot_sha256", "randomization_seed_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("assay SHA-256 must be lowercase hexadecimal")
        return value


class PlateAssignment(_StrictModel):
    blinded_sample_id: str
    gene_id: str
    background_id: str
    biological_replicate: int = Field(ge=1, le=3)
    plate_id: str
    well_id: str
    scorer_mapping_visible: Literal[False]


class TransferEvaluation(_StrictModel):
    regime: Literal["zero_shot", "low_shot", "retrained"]
    train_cells: list[str]
    adaptation_cells: list[str]
    test_cells: list[str]
    maximum_adaptation_replicates_per_cell: Literal[0, 1, 3]
    test_cells_used_for_training_or_adaptation: Literal[False]

    @model_validator(mode="after")
    def disjoint_cells(self) -> "TransferEvaluation":
        train = set(self.train_cells)
        adaptation = set(self.adaptation_cells)
        test = set(self.test_cells)
        if not test or train & adaptation or train & test or adaptation & test:
            raise ValueError("transfer train, adaptation, and test cells must be disjoint")
        expected_adaptation = {"zero_shot": 0, "low_shot": 1, "retrained": 3}
        if self.maximum_adaptation_replicates_per_cell != expected_adaptation[self.regime]:
            raise ValueError("transfer adaptation limit differs from its regime")
        if self.regime == "zero_shot" and adaptation:
            raise ValueError("zero-shot transfer cannot have adaptation cells")
        return self


class LockedAsset(_StrictModel):
    asset_id: str
    relative_path: str
    sha256: str
    bytes: int = Field(gt=0)

    @field_validator("asset_id")
    @classmethod
    def safe_asset_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("locked asset ID must be a safe token")
        return value

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError("locked asset path must be a safe POSIX-relative path")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("locked asset SHA-256 must be lowercase hexadecimal")
        return value


def _verify_locked_assets(
    assets: Sequence[LockedAsset], asset_root: str | Path
) -> dict[str, Any]:
    root = Path(asset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"locked asset root is not a directory: {root}")
    if (
        len({item.asset_id for item in assets}) != len(assets)
        or len({item.relative_path for item in assets}) != len(assets)
        or len({item.sha256 for item in assets}) != len(assets)
    ):
        raise ValueError("locked asset IDs, paths, and hashes must be unique")
    verified = []
    for asset in assets:
        path = (root / Path(*PurePosixPath(asset.relative_path).parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("locked asset escapes its declared root") from exc
        if not path.is_file():
            raise FileNotFoundError(f"declared locked asset is missing: {asset.relative_path}")
        if path.stat().st_size != asset.bytes or _sha256(path) != asset.sha256:
            raise ValueError(f"locked asset checksum or byte count differs: {asset.asset_id}")
        verified.append(asset.model_dump(mode="json"))
    return {"assets": verified, "verified": True}


class Preregistration(_StrictModel):
    schema_version: Literal[PREREG_VERSION]
    preregistration_id: str
    created_utc: AwareDatetime
    synthetic_fixture: Literal[True]
    external_model_calls_performed: Literal[False]
    result_data_accessed: Literal[False]
    referenced_assets: list[LockedAsset]
    panel: list[GenePanelRecord]
    selection_assignment_receipt: SelectionAssignmentReceipt
    backgrounds: list[BackgroundRecord]
    forecasts: list[ForecastRecord]
    assay: AssayPlan
    plate_assignments: list[PlateAssignment]
    transfer_evaluations: list[TransferEvaluation]
    independent_dataset_alternative: None
    result_interpretation_contract_sha256: str

    @field_validator("result_interpretation_contract_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("interpretation contract SHA-256 must be lowercase hexadecimal")
        return value


class AdaptationForecastReceipt(_StrictModel):
    schema_version: Literal[ADAPTATION_VERSION]
    adaptation_id: str
    created_utc: AwareDatetime
    stage: Literal["low_shot", "retrained"]
    status: Literal["frozen_after_permitted_adaptation_before_test_unblinding"]
    interpretation_scope: Literal[
        "post_control_adaptation_transfer_benchmark_nonprospective",
        "cross_fitted_post_outcome_transfer_benchmark_nonprospective",
    ]
    preregistration_bundle_sha256: str
    blinded_result_lock_bundle_sha256: str
    manifest_sha256: str
    synthetic_fixture: Literal[True]
    external_model_calls_performed: Literal[False]
    train_cells: list[str]
    adaptation_cells: list[str]
    test_cells: list[str]
    outcome_cells_accessed: list[str]
    outcome_replicates_accessed: dict[str, int]
    test_outcomes_accessed: Literal[False]
    training_data_lock_sha256: str
    forecasts: list[ForecastRecord]
    referenced_assets: list[LockedAsset]

    @field_validator(
        "preregistration_bundle_sha256",
        "blinded_result_lock_bundle_sha256",
        "manifest_sha256",
        "training_data_lock_sha256",
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("adaptation hash must be lowercase hexadecimal")
        return value


def _cell(gene_id: str, background_id: str) -> str:
    return f"{background_id}::{gene_id}"


def interpretation_contract_sha256(manifest: ProspectiveTransferManifest) -> str:
    """Hash the exact result-dependent interpretation contract."""
    return _object_sha256(manifest.interpretation.model_dump(mode="json"))


def manifest_sha256(manifest: ProspectiveTransferManifest) -> str:
    """Hash the exact prospective transfer manifest snapshot."""
    return _object_sha256(manifest.model_dump(mode="json"))


def _required_preregistration_asset_hashes(preregistration: Preregistration) -> set[str]:
    hashes = {item.source_snapshot_sha256 for item in preregistration.panel}
    receipt = preregistration.selection_assignment_receipt
    hashes.update(
        {
            receipt.random_seed_sha256,
            receipt.random_candidate_universe_sha256,
            receipt.expert_panel_receipt_sha256,
            receipt.uncertainty_score_snapshot_sha256,
            receipt.graph_eig_score_snapshot_sha256,
        }
    )
    for background in preregistration.backgrounds:
        hashes.add(background.genotype_snapshot_sha256)
        if background.haplotype_panel_receipt_sha256 is not None:
            hashes.add(background.haplotype_panel_receipt_sha256)
    hashes.update(item.model_artifact_sha256 for item in preregistration.forecasts)
    hashes.update(
        {
            preregistration.assay.protocol_snapshot_sha256,
            preregistration.assay.randomization_seed_sha256,
            preregistration.assay.power.assumptions_sha256,
        }
    )
    return hashes


def validate_preregistration(
    manifest: ProspectiveTransferManifest, preregistration: Preregistration
) -> dict[str, Any]:
    if preregistration.result_interpretation_contract_sha256 != (
        interpretation_contract_sha256(manifest)
    ):
        raise ValueError("preregistration is bound to a different interpretation contract")
    declared_asset_hashes = {item.sha256 for item in preregistration.referenced_assets}
    if declared_asset_hashes != _required_preregistration_asset_hashes(preregistration):
        raise ValueError("locked asset inventory does not exactly cover preregistration hashes")
    power = preregistration.assay.power
    if (
        preregistration.assay.minimum_embryos_per_replicate
        != manifest.assay.minimum_embryos_per_replicate
        or preregistration.assay.biological_replicates
        != manifest.assay.biological_replicates_per_gene_background
        or power.embryos_per_replicate < manifest.assay.minimum_embryos_per_replicate
        or power.method != manifest.power.method
        or power.family_alpha != manifest.power.family_alpha
        or power.target_power != manifest.power.target_power
        or power.minimum_detectable_standardized_effect
        != manifest.power.minimum_detectable_standardized_effect
        or power.intraclass_correlation
        != manifest.power.assumed_intraclass_correlation
    ):
        raise ValueError("assay and power plan differ from the frozen manifest")
    panel = preregistration.panel
    if len(panel) != PANEL_SIZE or len({item.gene_id for item in panel}) != PANEL_SIZE:
        raise ValueError("prospective transfer panel must contain 12 unique genes")
    by_role = defaultdict(list)
    by_arm = defaultdict(list)
    for item in panel:
        by_role[item.panel_role].append(item)
        by_arm[item.selection_arm].append(item)
    if (
        len(by_role["candidate"]) != CANDIDATE_COUNT
        or len(by_role["positive_control"]) != POSITIVE_CONTROL_COUNT
        or len(by_role["negative_control"]) != NEGATIVE_CONTROL_COUNT
    ):
        raise ValueError("candidate and control counts differ from the frozen panel")
    if any(len(by_arm[arm]) != 2 for arm in SELECTION_ARMS):
        raise ValueError("each comparator selection arm must contribute exactly two genes")

    backgrounds = preregistration.backgrounds
    if len(backgrounds) != BACKGROUND_COUNT or len(
        {item.background_id for item in backgrounds}
    ) != BACKGROUND_COUNT:
        raise ValueError("transfer arm must contain N2 and two unique natural backgrounds")
    if backgrounds[0].background_id != "N2" or backgrounds[0].role != "reference":
        raise ValueError("N2 must be the first reference background")
    if any(item.role != "natural_haplotype" for item in backgrounds[1:]):
        raise ValueError("the two transfer backgrounds must be haplotype-selected isolates")

    expected_forecasts = {
        (gene.gene_id, background.background_id, architecture, "zero_shot")
        for gene in panel
        for background in backgrounds
        for architecture in ARCHITECTURES
    }
    observed_forecasts = {
        (item.gene_id, item.background_id, item.architecture, item.transfer_regime)
        for item in preregistration.forecasts
    }
    if len(preregistration.forecasts) != len(expected_forecasts):
        raise ValueError("forecast records are duplicated or incomplete")
    if observed_forecasts != expected_forecasts:
        raise ValueError("forecast grid differs from panel by background by model contracts")

    expected_observations = {
        (gene.gene_id, background.background_id, replicate)
        for gene in panel
        for background in backgrounds
        for replicate in range(1, REPLICATES + 1)
    }
    observed_assignments = {
        (item.gene_id, item.background_id, item.biological_replicate)
        for item in preregistration.plate_assignments
    }
    blind_ids = [item.blinded_sample_id for item in preregistration.plate_assignments]
    plate_wells = [
        (item.plate_id, item.well_id) for item in preregistration.plate_assignments
    ]
    if (
        observed_assignments != expected_observations
        or len(preregistration.plate_assignments) != len(expected_observations)
        or len(set(blind_ids)) != len(blind_ids)
        or len(set(plate_wells)) != len(plate_wells)
    ):
        raise ValueError("blinded plate layout does not cover each expected observation once")
    gene_by_id = {item.gene_id: item for item in panel}
    by_plate = defaultdict(list)
    plate_by_background_replicate = defaultdict(set)
    background_replicate_by_plate = defaultdict(set)
    for item in preregistration.plate_assignments:
        by_plate[item.plate_id].append(gene_by_id[item.gene_id].panel_role)
        background_replicate = (item.background_id, item.biological_replicate)
        plate_by_background_replicate[background_replicate].add(item.plate_id)
        background_replicate_by_plate[item.plate_id].add(background_replicate)
    expected_plates = BACKGROUND_COUNT * REPLICATES
    if (
        len(by_plate) != expected_plates
        or any(len(plates) != 1 for plates in plate_by_background_replicate.values())
        or any(len(cells) != 1 for cells in background_replicate_by_plate.values())
    ):
        raise ValueError("each background-by-replicate must map to exactly one plate")
    if any(
        "positive_control" not in roles or "negative_control" not in roles
        for roles in by_plate.values()
    ):
        raise ValueError("every plate must contain positive and negative controls")

    if [item.regime for item in preregistration.transfer_evaluations] != list(
        TRANSFER_REGIMES
    ):
        raise ValueError("transfer evaluations differ from the frozen order")
    all_cells = {
        _cell(gene.gene_id, background.background_id)
        for gene in panel
        for background in backgrounds
    }
    candidate_ids = {item.gene_id for item in by_role["candidate"]}
    control_ids = {
        item.gene_id
        for role in ("positive_control", "negative_control")
        for item in by_role[role]
    }
    natural_ids = {item.background_id for item in backgrounds[1:]}
    for evaluation in preregistration.transfer_evaluations:
        supplied = set(evaluation.train_cells + evaluation.adaptation_cells + evaluation.test_cells)
        if not supplied <= all_cells:
            raise ValueError("transfer evaluation references an unknown strain-by-gene cell")
        if evaluation.regime == "zero_shot":
            expected_train = {_cell(gene.gene_id, "N2") for gene in panel}
            expected_test = {
                _cell(gene.gene_id, background)
                for gene in panel
                for background in natural_ids
            }
            if set(evaluation.train_cells) != expected_train or set(
                evaluation.test_cells
            ) != expected_test:
                raise ValueError("zero-shot transfer must train on N2 and test all natural cells")
        elif evaluation.regime == "low_shot":
            expected_train = {_cell(gene.gene_id, "N2") for gene in panel}
            expected_adaptation = {
                _cell(gene, background) for gene in control_ids for background in natural_ids
            }
            expected_test = {
                _cell(gene, background) for gene in candidate_ids for background in natural_ids
            }
            if (
                set(evaluation.train_cells) != expected_train
                or set(evaluation.adaptation_cells) != expected_adaptation
                or set(evaluation.test_cells) != expected_test
            ):
                raise ValueError("low-shot adaptation must use controls and test candidates")
        else:
            expected_train = {_cell(gene.gene_id, "N2") for gene in panel}
            test = set(evaluation.test_cells)
            if len(test) != CANDIDATE_COUNT:
                raise ValueError("retrained transfer needs one natural holdout per candidate")
            if {cell.split("::", 1)[1] for cell in test} != candidate_ids:
                raise ValueError("retrained holdouts must cover every candidate gene")
            if {cell.split("::", 1)[0] for cell in test} - natural_ids:
                raise ValueError("retrained holdouts must be natural-background cells")
            expected_adaptation = {
                _cell(gene.gene_id, background)
                for gene in panel
                for background in natural_ids
            } - test
            if (
                set(evaluation.train_cells) != expected_train
                or set(evaluation.adaptation_cells) != expected_adaptation
            ):
                raise ValueError(
                    "retrained transfer must train on N2 and adapt on non-holdout natural cells"
                )
    return {
        "schema_version": "wormctx-prospective_transfer-preregistration-qualification-1.0",
        "panel_genes": PANEL_SIZE,
        "backgrounds": BACKGROUND_COUNT,
        "forecasts": len(expected_forecasts),
        "expected_observations": len(expected_observations),
        "plates": len(by_plate),
        "transfer_regimes": list(TRANSFER_REGIMES),
        "synthetic_fixture": True,
        "biological_claims_permitted": False,
    }


def validate_adaptation_receipt(
    manifest: ProspectiveTransferManifest,
    preregistration: Preregistration,
    preregistration_bundle_sha256: str,
    blinded_result_lock_bundle_sha256: str,
    blinded_result_lock: BlindedResultLock,
    receipt: AdaptationForecastReceipt,
) -> dict[str, Any]:
    if receipt.preregistration_bundle_sha256 != preregistration_bundle_sha256:
        raise ValueError("adaptation receipt is bound to a different preregistration bundle")
    if receipt.created_utc <= preregistration.created_utc:
        raise ValueError("adaptation receipt timestamp must follow preregistration")
    if receipt.blinded_result_lock_bundle_sha256 != blinded_result_lock_bundle_sha256:
        raise ValueError("adaptation receipt is bound to a different blinded-result lock")
    if receipt.created_utc <= blinded_result_lock.locked_utc:
        raise ValueError("adaptation receipt must follow the blinded-result lock")
    if receipt.manifest_sha256 != manifest_sha256(manifest):
        raise ValueError("adaptation receipt is bound to a different manifest snapshot")
    evaluation = next(
        item
        for item in preregistration.transfer_evaluations
        if item.regime == receipt.stage
    )
    for field_name in ("train_cells", "adaptation_cells", "test_cells"):
        supplied = getattr(receipt, field_name)
        expected = getattr(evaluation, field_name)
        if len(supplied) != len(expected) or set(supplied) != set(expected):
            raise ValueError(f"adaptation {field_name} differ from the preregistered split")
    permitted_outcomes = set(evaluation.train_cells + evaluation.adaptation_cells)
    if (
        len(receipt.outcome_cells_accessed) != len(permitted_outcomes)
        or set(receipt.outcome_cells_accessed) != permitted_outcomes
        or permitted_outcomes & set(evaluation.test_cells)
    ):
        raise ValueError("adaptation outcome access is incomplete or includes held-out cells")
    expected_replicates = {
        **{cell: REPLICATES for cell in evaluation.train_cells},
        **{
            cell: evaluation.maximum_adaptation_replicates_per_cell
            for cell in evaluation.adaptation_cells
        },
    }
    if receipt.outcome_replicates_accessed != expected_replicates:
        raise ValueError("adaptation replicate access differs from the preregistered limit")
    expected_scope = {
        "low_shot": "post_control_adaptation_transfer_benchmark_nonprospective",
        "retrained": "cross_fitted_post_outcome_transfer_benchmark_nonprospective",
    }
    if receipt.interpretation_scope != expected_scope[receipt.stage]:
        raise ValueError("adaptation interpretation scope differs from its stage")
    test_pairs = {tuple(item.split("::", 1)[::-1]) for item in evaluation.test_cells}
    expected_forecasts = {
        (gene_id, background_id, architecture, receipt.stage)
        for gene_id, background_id in test_pairs
        for architecture in ARCHITECTURES
    }
    observed_forecasts = {
        (item.gene_id, item.background_id, item.architecture, item.transfer_regime)
        for item in receipt.forecasts
    }
    if (
        len(receipt.forecasts) != len(expected_forecasts)
        or observed_forecasts != expected_forecasts
    ):
        raise ValueError("adapted forecasts do not exactly cover held-out cells by architecture")
    required_hashes = {
        receipt.training_data_lock_sha256,
        *(item.model_artifact_sha256 for item in receipt.forecasts),
    }
    if {item.sha256 for item in receipt.referenced_assets} != required_hashes:
        raise ValueError("adaptation assets do not exactly cover training and model hashes")
    return {
        "schema_version": "wormctx-prospective_transfer-adaptation-qualification-1.0",
        "stage": receipt.stage,
        "train_cells": len(evaluation.train_cells),
        "adaptation_cells": len(evaluation.adaptation_cells),
        "heldout_test_cells": len(evaluation.test_cells),
        "forecasts": len(expected_forecasts),
        "interpretation_scope": receipt.interpretation_scope,
        "prospective_claim_permitted": False,
        "synthetic_fixture": True,
    }


class ResultObservation(_StrictModel):
    blinded_sample_id: str
    embryos_measured: int = Field(ge=0)
    observed_later_outcome: float | None
    observed_lineage_location: str | None
    qc_pass: bool
    exclusion_code: str | None

    @model_validator(mode="after")
    def result_semantics(self) -> "ResultObservation":
        if self.qc_pass:
            if (
                self.embryos_measured < 1
                or self.observed_later_outcome is None
                or not math.isfinite(self.observed_later_outcome)
                or self.observed_lineage_location not in LINEAGE_LOCATIONS
                or self.exclusion_code is not None
            ):
                raise ValueError("passing result requires finite outcome and lineage call")
        elif self.exclusion_code is None or any(
            value is not None
            for value in (self.observed_later_outcome, self.observed_lineage_location)
        ):
            raise ValueError("excluded result requires a code and no analyzed outcome")
        return self


class RawAsset(_StrictModel):
    asset_id: str
    relative_path: str
    sha256: str
    bytes: int = Field(gt=0)

    @field_validator("asset_id")
    @classmethod
    def safe_asset_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("raw asset ID must be a safe token")
        return value

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise ValueError("raw asset path must be a safe POSIX-relative path")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("raw asset SHA-256 must be lowercase hexadecimal")
        return value


class BlindedResultLock(_StrictModel):
    schema_version: Literal[BLINDED_RESULT_LOCK_VERSION]
    lock_id: str
    locked_utc: AwareDatetime
    preregistration_bundle_sha256: str
    manifest_sha256: str
    collected_before_any_partial_unblinding: Literal[True]
    scorer_blinded: Literal[True]
    sample_mapping_released: Literal[False]
    synthetic_fixture: Literal[True]
    external_experiment_performed: Literal[False]
    observations: list[ResultObservation]
    raw_assets: list[RawAsset]

    @field_validator("preregistration_bundle_sha256", "manifest_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("blinded-result lock hash must be lowercase hexadecimal")
        return value

    @model_validator(mode="after")
    def exact_assets(self) -> "BlindedResultLock":
        if not self.raw_assets:
            raise ValueError("blinded-result lock must declare at least one raw asset")
        if len({item.asset_id for item in self.raw_assets}) != len(self.raw_assets):
            raise ValueError("blinded-result raw asset IDs must be unique")
        if len({item.relative_path for item in self.raw_assets}) != len(self.raw_assets):
            raise ValueError("blinded-result raw asset paths must be unique")
        return self


class ResultInput(_StrictModel):
    schema_version: Literal[RESULT_INPUT_VERSION]
    preregistration_bundle_sha256: str
    blinded_result_lock_bundle_sha256: str
    adaptation_bundle_sha256: dict[str, str]
    result_locked_utc: AwareDatetime
    collected_before_unblinding: Literal[True]
    scorer_blinded: Literal[True]
    synthetic_fixture: Literal[True]
    external_experiment_performed: Literal[False]
    observations: list[ResultObservation]
    raw_assets: list[RawAsset]

    @field_validator("preregistration_bundle_sha256", "blinded_result_lock_bundle_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("preregistration bundle SHA-256 must be lowercase hexadecimal")
        return value

    @field_validator("adaptation_bundle_sha256")
    @classmethod
    def valid_adaptation_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != {"low_shot", "retrained"} or any(
            not _SHA256_RE.fullmatch(item) for item in value.values()
        ):
            raise ValueError("result input must bind both adaptation bundle hashes")
        return value

    @model_validator(mode="after")
    def exact_assets(self) -> "ResultInput":
        if not self.raw_assets:
            raise ValueError("result input must declare at least one raw asset")
        if len({item.asset_id for item in self.raw_assets}) != len(self.raw_assets):
            raise ValueError("raw asset IDs must be unique")
        if len({item.relative_path for item in self.raw_assets}) != len(self.raw_assets):
            raise ValueError("raw asset paths must be unique")
        return self


def _verify_raw_asset_inventory(
    assets: Sequence[RawAsset], raw_asset_root: str | Path
) -> dict[str, Any]:
    root = Path(raw_asset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw asset root is not a directory: {root}")
    verified = []
    for asset in assets:
        path = (root / Path(*PurePosixPath(asset.relative_path).parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("raw asset escapes its declared root") from exc
        if not path.is_file():
            raise FileNotFoundError(f"declared raw asset is missing: {asset.relative_path}")
        if path.stat().st_size != asset.bytes or _sha256(path) != asset.sha256:
            raise ValueError(f"raw asset checksum or byte count differs: {asset.asset_id}")
        verified.append(asset.model_dump(mode="json"))
    return {"assets": verified, "verified": True}


def _verify_raw_assets(results: ResultInput, raw_asset_root: str | Path) -> dict[str, Any]:
    return _verify_raw_asset_inventory(results.raw_assets, raw_asset_root)


def _write_new(path: Path, content: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"write-once file exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        stage.write_bytes(content)
        os.replace(stage, path)
    finally:
        if stage.exists():
            stage.unlink()


def _publish_bundle(
    output: str | Path, payloads: Mapping[str, bytes], terminal: str
) -> dict[str, Any]:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"write-once bundle exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        for name, content in payloads.items():
            _write_new(stage / name, content)
        checksum_lines = [f"{_sha256(stage / name)}  ./{name}" for name in payloads]
        _write_new(stage / "SHA256SUMS.txt", ("\n".join(checksum_lines) + "\n").encode())
        _write_new(stage / terminal, f"{terminal}\n".encode())
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {
        "output": str(destination),
        "bundle_sha256": _sha256(destination / "SHA256SUMS.txt"),
        "payload_count": len(payloads),
        "terminal": terminal,
    }


def _verify_bundle(
    root: str | Path, payload_names: Sequence[str], terminal: str
) -> dict[str, Any]:
    directory = Path(root).resolve()
    expected = sorted([*payload_names, "SHA256SUMS.txt", terminal])
    observed = sorted(item.name for item in directory.iterdir() if item.is_file())
    if observed != expected:
        raise ValueError("bundle inventory differs")
    if (directory / terminal).read_bytes() != f"{terminal}\n".encode():
        raise ValueError("bundle terminal marker differs")
    entries = []
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  \./([^/]+)", line)
        if match is None:
            raise ValueError("bundle checksum manifest is malformed")
        entries.append((match.group(2), match.group(1)))
    if [item[0] for item in entries] != list(payload_names):
        raise ValueError("bundle checksum coverage differs")
    for name, expected_hash in entries:
        if _sha256(directory / name) != expected_hash:
            raise ValueError(f"bundle checksum failed: {name}")
    return {
        "bundle_sha256": _sha256(directory / "SHA256SUMS.txt"),
        "payload_count": len(entries),
        "terminal": terminal,
    }


def freeze_preregistration(
    manifest: ProspectiveTransferManifest,
    preregistration: Preregistration,
    asset_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    qualification = validate_preregistration(manifest, preregistration)
    asset_verification = _verify_locked_assets(
        preregistration.referenced_assets, asset_root
    )
    qualification["manifest_sha256"] = manifest_sha256(manifest)
    payloads = {
        "manifest.json": _canonical_bytes(manifest.model_dump(mode="json")),
        "preregistration.json": _canonical_bytes(preregistration.model_dump(mode="json")),
        "qualification.json": _canonical_bytes(qualification),
        "asset_verification.json": _canonical_bytes(asset_verification),
        "expected_observations.jsonl": b"".join(
            _canonical_bytes(item.model_dump(mode="json"))
            for item in preregistration.plate_assignments
        ),
    }
    published = _publish_bundle(output, payloads, "FROZEN")
    return {"schema_version": PREREG_BUNDLE_VERSION, **published}


def verify_preregistration(
    root: str | Path, manifest: ProspectiveTransferManifest
) -> dict[str, Any]:
    verification = _verify_bundle(
        root,
        [
            "manifest.json",
            "preregistration.json",
            "qualification.json",
            "asset_verification.json",
            "expected_observations.jsonl",
        ],
        "FROZEN",
    )
    directory = Path(root).resolve()
    frozen_manifest = ProspectiveTransferManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest_sha256(frozen_manifest) != manifest_sha256(manifest):
        raise ValueError("current manifest differs from the frozen preregistration manifest")
    preregistration = Preregistration.model_validate_json(
        (directory / "preregistration.json").read_text(encoding="utf-8")
    )
    validate_preregistration(manifest, preregistration)
    return {**verification, "manifest_sha256": manifest_sha256(manifest)}


def validate_blinded_result_lock(
    manifest: ProspectiveTransferManifest,
    preregistration: Preregistration,
    preregistration_bundle_sha256: str,
    locked_results: BlindedResultLock,
) -> dict[str, Any]:
    if locked_results.preregistration_bundle_sha256 != preregistration_bundle_sha256:
        raise ValueError("blinded-result lock is bound to a different preregistration bundle")
    if locked_results.manifest_sha256 != manifest_sha256(manifest):
        raise ValueError("blinded-result lock is bound to a different manifest snapshot")
    if locked_results.locked_utc <= preregistration.created_utc:
        raise ValueError("blinded-result lock timestamp must follow preregistration")
    expected_blind_ids = {
        item.blinded_sample_id for item in preregistration.plate_assignments
    }
    observed_blind_ids = [item.blinded_sample_id for item in locked_results.observations]
    if (
        len(observed_blind_ids) != len(expected_blind_ids)
        or len(set(observed_blind_ids)) != len(observed_blind_ids)
        or set(observed_blind_ids) != expected_blind_ids
    ):
        raise ValueError("blinded-result lock must cover the exact preregistered blind-ID universe")
    return {
        "schema_version": "wormctx-prospective_transfer-blinded-result-lock-qualification-1.0",
        "observations": len(observed_blind_ids),
        "raw_assets": len(locked_results.raw_assets),
        "collected_before_any_partial_unblinding": True,
        "sample_mapping_released": False,
        "synthetic_fixture": True,
        "biological_claims_permitted": False,
    }


def freeze_blinded_result_lock(
    manifest: ProspectiveTransferManifest,
    preregistration_root: str | Path,
    locked_results: BlindedResultLock,
    raw_asset_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    prereg_verification = verify_preregistration(preregistration_root, manifest)
    preregistration = Preregistration.model_validate_json(
        (Path(preregistration_root).resolve() / "preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    qualification = validate_blinded_result_lock(
        manifest,
        preregistration,
        prereg_verification["bundle_sha256"],
        locked_results,
    )
    asset_verification = _verify_raw_asset_inventory(
        locked_results.raw_assets, raw_asset_root
    )
    payloads = {
        "blinded_result_lock.json": _canonical_bytes(
            locked_results.model_dump(mode="json")
        ),
        "qualification.json": _canonical_bytes(qualification),
        "raw_asset_verification.json": _canonical_bytes(asset_verification),
    }
    published = _publish_bundle(output, payloads, "BLINDED_RESULTS_LOCKED")
    return {
        "schema_version": BLINDED_RESULT_LOCK_BUNDLE_VERSION,
        **published,
    }


def verify_blinded_result_lock(
    root: str | Path,
    manifest: ProspectiveTransferManifest,
    preregistration_root: str | Path,
) -> dict[str, Any]:
    verification = _verify_bundle(
        root,
        [
            "blinded_result_lock.json",
            "qualification.json",
            "raw_asset_verification.json",
        ],
        "BLINDED_RESULTS_LOCKED",
    )
    prereg_verification = verify_preregistration(preregistration_root, manifest)
    preregistration = Preregistration.model_validate_json(
        (Path(preregistration_root).resolve() / "preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    locked_results = BlindedResultLock.model_validate_json(
        (Path(root).resolve() / "blinded_result_lock.json").read_text(encoding="utf-8")
    )
    validate_blinded_result_lock(
        manifest,
        preregistration,
        prereg_verification["bundle_sha256"],
        locked_results,
    )
    return {
        **verification,
        "locked_utc": locked_results.locked_utc.isoformat(),
    }


def freeze_adaptation_forecasts(
    manifest: ProspectiveTransferManifest,
    preregistration_root: str | Path,
    blinded_result_lock_root: str | Path,
    receipt: AdaptationForecastReceipt,
    asset_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    prereg_verification = verify_preregistration(preregistration_root, manifest)
    blinded_lock_verification = verify_blinded_result_lock(
        blinded_result_lock_root, manifest, preregistration_root
    )
    preregistration = Preregistration.model_validate_json(
        (Path(preregistration_root).resolve() / "preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    blinded_result_lock = BlindedResultLock.model_validate_json(
        (
            Path(blinded_result_lock_root).resolve() / "blinded_result_lock.json"
        ).read_text(encoding="utf-8")
    )
    qualification = validate_adaptation_receipt(
        manifest,
        preregistration,
        prereg_verification["bundle_sha256"],
        blinded_lock_verification["bundle_sha256"],
        blinded_result_lock,
        receipt,
    )
    asset_verification = _verify_locked_assets(receipt.referenced_assets, asset_root)
    payloads = {
        "adaptation_receipt.json": _canonical_bytes(receipt.model_dump(mode="json")),
        "qualification.json": _canonical_bytes(qualification),
        "asset_verification.json": _canonical_bytes(asset_verification),
    }
    published = _publish_bundle(output, payloads, "ADAPTATION_FROZEN")
    return {
        "schema_version": ADAPTATION_BUNDLE_VERSION,
        "stage": receipt.stage,
        **published,
    }


def verify_adaptation_bundle(
    root: str | Path,
    manifest: ProspectiveTransferManifest,
    preregistration_root: str | Path,
    blinded_result_lock_root: str | Path,
) -> dict[str, Any]:
    verification = _verify_bundle(
        root,
        ["adaptation_receipt.json", "qualification.json", "asset_verification.json"],
        "ADAPTATION_FROZEN",
    )
    prereg_verification = verify_preregistration(preregistration_root, manifest)
    blinded_lock_verification = verify_blinded_result_lock(
        blinded_result_lock_root, manifest, preregistration_root
    )
    preregistration = Preregistration.model_validate_json(
        (Path(preregistration_root).resolve() / "preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = AdaptationForecastReceipt.model_validate_json(
        (Path(root).resolve() / "adaptation_receipt.json").read_text(encoding="utf-8")
    )
    blinded_result_lock = BlindedResultLock.model_validate_json(
        (
            Path(blinded_result_lock_root).resolve() / "blinded_result_lock.json"
        ).read_text(encoding="utf-8")
    )
    validate_adaptation_receipt(
        manifest,
        preregistration,
        prereg_verification["bundle_sha256"],
        blinded_lock_verification["bundle_sha256"],
        blinded_result_lock,
        receipt,
    )
    return {**verification, "stage": receipt.stage}


def _aggregate_results(
    manifest: ProspectiveTransferManifest,
    preregistration: Preregistration,
    results: ResultInput,
) -> tuple[
    dict[tuple[str, str], tuple[float, str]],
    dict[tuple[str, str, int], float | None],
    dict[str, Any],
]:
    assignments = {item.blinded_sample_id: item for item in preregistration.plate_assignments}
    if len(results.observations) != len(assignments):
        raise ValueError("result input must contain exactly the preregistered observations")
    result_by_blind = {item.blinded_sample_id: item for item in results.observations}
    if len(result_by_blind) != len(results.observations) or set(result_by_blind) != set(
        assignments
    ):
        raise ValueError("result blind IDs are duplicated, missing, or extra")
    allowed_exclusions = set(manifest.assay.exclusion_codes)
    by_cell: dict[tuple[str, str], list[ResultObservation]] = defaultdict(list)
    by_replicate: dict[tuple[str, str, int], float | None] = {}
    exclusion_counts = defaultdict(int)
    for blind_id, result in result_by_blind.items():
        if result.qc_pass and (
            result.embryos_measured < manifest.assay.minimum_embryos_per_replicate
        ):
            raise ValueError("passing result is below the preregistered embryo minimum")
        if not result.qc_pass:
            if result.exclusion_code not in allowed_exclusions:
                raise ValueError("result uses an unregistered exclusion code")
            exclusion_counts[result.exclusion_code] += 1
        assignment = assignments[blind_id]
        by_cell[(assignment.gene_id, assignment.background_id)].append(result)
        by_replicate[
            (
                assignment.gene_id,
                assignment.background_id,
                assignment.biological_replicate,
            )
        ] = result.observed_later_outcome if result.qc_pass else None
    aggregated: dict[tuple[str, str], tuple[float, str]] = {}
    underpowered = []
    for cell, observations in by_cell.items():
        passing = [item for item in observations if item.qc_pass]
        if len(passing) < manifest.interpretation.minimum_analyzable_replicates_per_cell:
            underpowered.append(_cell(*cell))
            continue
        outcome = float(
            math.fsum(
                item.observed_later_outcome
                for item in passing
                if item.observed_later_outcome is not None
            )
            / len(passing)
        )
        locations = [item.observed_lineage_location for item in passing]
        location = max(sorted(set(locations)), key=lambda item: locations.count(item))
        assert location is not None
        aggregated[cell] = (outcome, location)
    return aggregated, by_replicate, {
        "expected_observations": len(assignments),
        "passing_observations": sum(item.qc_pass for item in results.observations),
        "exclusion_counts": dict(exclusion_counts),
        "analyzable_cells": len(aggregated),
        "underpowered_cells": sorted(underpowered),
    }


def _forecast_metrics(
    forecasts: Sequence[ForecastRecord],
    observed: Mapping[tuple[str, str], tuple[float, str]],
    target_cells: set[tuple[str, str]],
) -> dict[str, Any]:
    by_cell = {(item.gene_id, item.background_id): item for item in forecasts}
    if len(by_cell) != len(forecasts):
        raise ValueError("forecast scoring received duplicated strain-by-gene cells")
    available_outcomes = target_cells & set(observed)
    answered_cells = {
        cell
        for cell in available_outcomes
        if cell in by_cell and not by_cell[cell].abstain
    }
    target_count = len(target_cells)
    coverage = len(answered_cells) / target_count if target_count else 0.0
    base: dict[str, Any] = {
        "status": "ok" if answered_cells else "all_abstained_or_unavailable",
        "target_cells": target_count,
        "outcome_available_cells": len(available_outcomes),
        "answered": len(answered_cells),
        "abstained_or_missing_forecast": target_count - len(answered_cells),
        "coverage": coverage,
        "rmse": None,
        "mean_absolute_error": None,
        "lineage_accuracy": None,
        "interval_coverage": None,
        "mean_scalar_uncertainty": None,
        "uncertainty_calibration_mae": None,
    }
    if not answered_cells:
        return base
    errors = []
    absolute_errors = []
    lineage_correct = []
    interval_hits = []
    uncertainties = []
    for cell in sorted(answered_cells):
        forecast = by_cell[cell]
        prediction = forecast.predicted_later_outcome
        lower = forecast.interval_lower
        upper = forecast.interval_upper
        assert prediction is not None and lower is not None and upper is not None
        observed_outcome, observed_lineage = observed[cell]
        error = observed_outcome - prediction
        errors.append(error)
        absolute_errors.append(abs(error))
        lineage_correct.append(observed_lineage == forecast.lineage_location)
        interval_hits.append(lower <= observed_outcome <= upper)
        uncertainties.append(forecast.scalar_uncertainty)
    base.update(
        {
            "rmse": math.sqrt(math.fsum(item * item for item in errors) / len(errors)),
            "mean_absolute_error": math.fsum(absolute_errors) / len(absolute_errors),
            "lineage_accuracy": sum(lineage_correct) / len(lineage_correct),
            "interval_coverage": sum(interval_hits) / len(interval_hits),
            "mean_scalar_uncertainty": math.fsum(uncertainties) / len(uncertainties),
            "uncertainty_calibration_mae": (
                math.fsum(
                    abs(error - uncertainty)
                    for error, uncertainty in zip(absolute_errors, uncertainties, strict=True)
                )
                / len(absolute_errors)
            ),
        }
    )
    return base


def _common_architecture_metrics(
    graph_forecasts: Sequence[ForecastRecord],
    edge_forecasts: Sequence[ForecastRecord],
    observed: Mapping[tuple[str, str], tuple[float, str]],
    target_cells: set[tuple[str, str]],
) -> dict[str, Any]:
    graph_by_cell = {(item.gene_id, item.background_id): item for item in graph_forecasts}
    edge_by_cell = {(item.gene_id, item.background_id): item for item in edge_forecasts}
    common_cells = {
        cell
        for cell in target_cells & set(observed)
        if cell in graph_by_cell
        and cell in edge_by_cell
        and not graph_by_cell[cell].abstain
        and not edge_by_cell[cell].abstain
    }
    target_count = len(target_cells)
    coverage = len(common_cells) / target_count if target_count else 0.0
    if not common_cells:
        return {
            "status": "no_common_answered_forecasts",
            "target_cells": target_count,
            "common_answered": 0,
            "coverage": coverage,
            "graph_rmse": None,
            "edge_free_rmse": None,
            "rmse_advantage": None,
            "graph_lineage_accuracy": None,
            "edge_free_lineage_accuracy": None,
            "lineage_accuracy_advantage": None,
        }
    graph_metrics = _forecast_metrics(graph_forecasts, observed, common_cells)
    edge_metrics = _forecast_metrics(edge_forecasts, observed, common_cells)
    graph_rmse = graph_metrics["rmse"]
    edge_rmse = edge_metrics["rmse"]
    graph_lineage = graph_metrics["lineage_accuracy"]
    edge_lineage = edge_metrics["lineage_accuracy"]
    assert graph_rmse is not None and edge_rmse is not None
    assert graph_lineage is not None and edge_lineage is not None
    return {
        "status": "ok",
        "target_cells": target_count,
        "common_answered": len(common_cells),
        "coverage": coverage,
        "graph_rmse": graph_rmse,
        "edge_free_rmse": edge_rmse,
        "rmse_advantage": edge_rmse - graph_rmse,
        "graph_lineage_accuracy": graph_lineage,
        "edge_free_lineage_accuracy": edge_lineage,
        "lineage_accuracy_advantage": graph_lineage - edge_lineage,
    }


def _paired_regime_comparison(
    earlier_forecasts: Sequence[ForecastRecord],
    later_forecasts: Sequence[ForecastRecord],
    observed: Mapping[tuple[str, str], tuple[float, str]],
    target_cells: set[tuple[str, str]],
    minimum_coverage: float,
    minimum_improvement: float,
) -> dict[str, Any]:
    earlier_by_cell = {
        (item.gene_id, item.background_id): item for item in earlier_forecasts
    }
    later_by_cell = {
        (item.gene_id, item.background_id): item for item in later_forecasts
    }
    outcome_cells = target_cells & set(observed)
    earlier_answered = {
        cell
        for cell in outcome_cells
        if cell in earlier_by_cell and not earlier_by_cell[cell].abstain
    }
    later_answered = {
        cell
        for cell in outcome_cells
        if cell in later_by_cell and not later_by_cell[cell].abstain
    }
    common = earlier_answered & later_answered
    target_count = len(target_cells)
    common_coverage = len(common) / target_count if target_count else 0.0
    base: dict[str, Any] = {
        "target_cells": target_count,
        "earlier_answered": len(earlier_answered),
        "later_answered": len(later_answered),
        "common_answered": len(common),
        "common_coverage": common_coverage,
        "earlier_only_answered": len(earlier_answered - later_answered),
        "later_only_answered": len(later_answered - earlier_answered),
        "earlier_common_rmse": None,
        "later_common_rmse": None,
        "rmse_improvement": None,
        "supported": None,
    }
    if common_coverage < minimum_coverage:
        return {"status": "insufficient_common_answered_coverage", **base}
    earlier_common = _forecast_metrics(earlier_forecasts, observed, common)
    later_common = _forecast_metrics(later_forecasts, observed, common)
    earlier_rmse = earlier_common["rmse"]
    later_rmse = later_common["rmse"]
    if earlier_rmse is None or later_rmse is None:
        return {
            "status": "no_common_answered_outcomes",
            **base,
        }
    improvement = earlier_rmse - later_rmse
    return {
        "status": "ok",
        **base,
        "earlier_common_rmse": earlier_rmse,
        "later_common_rmse": later_rmse,
        "rmse_improvement": improvement,
        "supported": improvement >= minimum_improvement,
    }


def _result_interpretation(
    manifest: ProspectiveTransferManifest,
    preregistration: Preregistration,
    forecasts: Sequence[ForecastRecord],
    observed: Mapping[tuple[str, str], tuple[float, str]],
    replicate_observed: Mapping[tuple[str, str, int], float | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    panel_by_id = {item.gene_id: item for item in preregistration.panel}

    def signed_control_pass(gene: GenePanelRecord, outcome: float | None) -> bool:
        if outcome is None:
            return False
        if gene.expected_outcome_direction == "increase":
            return outcome >= manifest.interpretation.positive_control_minimum_absolute_effect
        if gene.expected_outcome_direction == "decrease":
            return outcome <= -manifest.interpretation.positive_control_minimum_absolute_effect
        return abs(outcome) <= manifest.interpretation.negative_control_maximum_absolute_effect

    background_control_summary = {}
    for background in (item.background_id for item in preregistration.backgrounds):
        records = []
        for gene in preregistration.panel:
            if gene.panel_role == "candidate":
                continue
            cell = (gene.gene_id, background)
            outcome = observed[cell][0] if cell in observed else None
            records.append(
                {
                    "gene_id": gene.gene_id,
                    "role": gene.panel_role,
                    "expected_direction": gene.expected_outcome_direction,
                    "observed_outcome": outcome,
                    "pass": signed_control_pass(gene, outcome),
                }
            )
        background_pass = len(records) == POSITIVE_CONTROL_COUNT + NEGATIVE_CONTROL_COUNT and all(
            item["pass"] for item in records
        )
        background_control_summary[background] = {
            "records": records,
            "pass": background_pass,
        }

    plate_control_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    plate_metadata: dict[str, dict[str, Any]] = {}
    for assignment in preregistration.plate_assignments:
        gene = panel_by_id[assignment.gene_id]
        if gene.panel_role == "candidate":
            continue
        outcome = replicate_observed.get(
            (
                assignment.gene_id,
                assignment.background_id,
                assignment.biological_replicate,
            )
        )
        plate_metadata[assignment.plate_id] = {
            "background_id": assignment.background_id,
            "biological_replicate": assignment.biological_replicate,
        }
        plate_control_records[assignment.plate_id].append(
            {
                "gene_id": gene.gene_id,
                "role": gene.panel_role,
                "expected_direction": gene.expected_outcome_direction,
                "observed_outcome": outcome,
                "pass": signed_control_pass(gene, outcome),
            }
        )
    plate_control_summary = {}
    for plate_id in sorted(plate_metadata):
        records = plate_control_records[plate_id]
        plate_control_summary[plate_id] = {
            **plate_metadata[plate_id],
            "records": records,
            "pass": (
                len(records) == POSITIVE_CONTROL_COUNT + NEGATIVE_CONTROL_COUNT
                and all(item["pass"] for item in records)
            ),
        }
    background_controls_pass = all(
        item["pass"] for item in background_control_summary.values()
    )
    plate_controls_pass = (
        len(plate_control_summary) == BACKGROUND_COUNT * REPLICATES
        and all(item["pass"] for item in plate_control_summary.values())
    )
    controls_pass = (
        plate_controls_pass
        if manifest.assay.control_gate_level == "strict_per_plate"
        else background_controls_pass
    )
    candidate_ids = {
        item.gene_id for item in preregistration.panel if item.panel_role == "candidate"
    }
    required_candidate_cells = {(gene_id, "N2") for gene_id in candidate_ids}
    for evaluation in preregistration.transfer_evaluations:
        required_candidate_cells.update(
            tuple(item.split("::", 1)[::-1])
            for item in evaluation.test_cells
            if item.split("::", 1)[1] in candidate_ids
        )
    missing_candidate_cells = sorted(
        _cell(gene_id, background_id)
        for gene_id, background_id in required_candidate_cells - set(observed)
    )
    primary_cells = {(gene_id, "N2") for gene_id in candidate_ids}
    primary_forecasts = [
        item
        for item in forecasts
        if item.transfer_regime == "zero_shot"
        and item.gene_id in candidate_ids
        and item.background_id == "N2"
    ]
    graph_forecasts = [
        item for item in primary_forecasts if item.architecture == "graph"
    ]
    edge_forecasts = [
        item for item in primary_forecasts if item.architecture == "edge_free"
    ]
    graph = _forecast_metrics(graph_forecasts, observed, primary_cells)
    edge = _forecast_metrics(edge_forecasts, observed, primary_cells)
    common = _common_architecture_metrics(
        graph_forecasts, edge_forecasts, observed, primary_cells
    )
    coverage_floor = manifest.forecasts.minimum_architecture_coverage
    coverage_pass = (
        graph["coverage"] >= coverage_floor
        and edge["coverage"] >= coverage_floor
        and common["coverage"] >= coverage_floor
    )
    predictive_signal = (
        graph["rmse"] is not None
        and graph["rmse"]
        <= manifest.interpretation.maximum_graph_rmse_for_predictive_signal
    )
    graph_advantage = (
        common["rmse_advantage"] is not None
        and common["lineage_accuracy_advantage"] is not None
        and common["rmse_advantage"]
        >= manifest.interpretation.minimum_graph_rmse_advantage
        and common["lineage_accuracy_advantage"]
        >= manifest.interpretation.minimum_graph_lineage_accuracy_advantage
    )
    if not controls_pass:
        branch = "control_failure_inconclusive"
    elif missing_candidate_cells:
        branch = "insufficient_data_inconclusive"
    elif not coverage_pass:
        branch = "insufficient_forecast_coverage_inconclusive"
    elif predictive_signal and graph_advantage:
        branch = "graph_advantage_supported"
    elif predictive_signal:
        branch = "predictive_signal_without_graph_advantage"
    else:
        branch = "no_predictive_signal"

    evaluation_by_regime = {
        item.regime: item for item in preregistration.transfer_evaluations
    }

    def regime_forecasts(
        regime: str, architecture: str, cells: set[tuple[str, str]]
    ) -> list[ForecastRecord]:
        return [
            item
            for item in forecasts
            if item.architecture == architecture
            and item.transfer_regime == regime
            and (item.gene_id, item.background_id) in cells
        ]

    transfer = {}
    for regime, evaluation in evaluation_by_regime.items():
        test_cells = {tuple(item.split("::", 1)[::-1]) for item in evaluation.test_cells}
        graph_stage = regime_forecasts(regime, "graph", test_cells)
        edge_stage = regime_forecasts(regime, "edge_free", test_cells)
        transfer[regime] = {
            "graph": _forecast_metrics(graph_stage, observed, test_cells),
            "edge_free": _forecast_metrics(edge_stage, observed, test_cells),
            "common_architecture": _common_architecture_metrics(
                graph_stage, edge_stage, observed, test_cells
            ),
            "interpretation_scope": (
                "prospective_zero_shot"
                if regime == "zero_shot"
                else "nonprospective_adaptation_benchmark"
            ),
        }

    low_cells = {
        tuple(item.split("::", 1)[::-1])
        for item in evaluation_by_regime["low_shot"].test_cells
    }
    retrained_cells = {
        tuple(item.split("::", 1)[::-1])
        for item in evaluation_by_regime["retrained"].test_cells
    }
    paired_transfer: dict[str, dict[str, Any]] = {
        "zero_to_low_shot": {
            "test_cells": len(low_cells),
            "zero_shot": _forecast_metrics(
                regime_forecasts("zero_shot", "graph", low_cells), observed, low_cells
            ),
            "low_shot": _forecast_metrics(
                regime_forecasts("low_shot", "graph", low_cells), observed, low_cells
            ),
        },
        "low_shot_to_retrained": {
            "test_cells": len(retrained_cells),
            "low_shot": _forecast_metrics(
                regime_forecasts("low_shot", "graph", retrained_cells),
                observed,
                retrained_cells,
            ),
            "retrained": _forecast_metrics(
                regime_forecasts("retrained", "graph", retrained_cells),
                observed,
                retrained_cells,
            ),
        },
    }
    low_comparison = _paired_regime_comparison(
        regime_forecasts("zero_shot", "graph", low_cells),
        regime_forecasts("low_shot", "graph", low_cells),
        observed,
        low_cells,
        coverage_floor,
        manifest.interpretation.minimum_transfer_improvement,
    )
    retrained_comparison = _paired_regime_comparison(
        regime_forecasts("low_shot", "graph", retrained_cells),
        regime_forecasts("retrained", "graph", retrained_cells),
        observed,
        retrained_cells,
        coverage_floor,
        manifest.interpretation.minimum_transfer_improvement,
    )
    paired_transfer["zero_to_low_shot"]["comparison"] = low_comparison
    paired_transfer["low_shot_to_retrained"]["comparison"] = retrained_comparison
    transfer_decision = {
        "zero_shot_prospective_evaluated": (
            transfer["zero_shot"]["graph"]["rmse"] is not None
            and transfer["zero_shot"]["graph"]["coverage"] >= coverage_floor
        ),
        "low_shot_nonprospective_improvement_supported": low_comparison["supported"],
        "retrained_cross_fitted_improvement_supported": retrained_comparison["supported"],
    }
    summary = {
        "schema_version": SUMMARY_VERSION,
        "controls": {
            "gate_level": manifest.assay.control_gate_level,
            "by_plate": plate_control_summary,
            "by_background": background_control_summary,
            "plate_gate_pass": plate_controls_pass,
            "background_aggregate_pass": background_controls_pass,
            "pass": controls_pass,
        },
        "missing_candidate_or_test_cells": missing_candidate_cells,
        "graph": graph,
        "edge_free": edge,
        "common_architecture_comparison": common,
        "coverage_floor": coverage_floor,
        "coverage_pass": coverage_pass,
        "transfer": transfer,
        "paired_transfer_comparisons": paired_transfer,
        "transfer_decision": transfer_decision,
    }
    interpretation = {
        "schema_version": INTERPRETATION_VERSION,
        "branch": branch,
        "controls_pass": controls_pass,
        "candidate_and_test_data_complete": not missing_candidate_cells,
        "coverage_pass": coverage_pass,
        "predictive_signal": predictive_signal,
        "graph_advantage": graph_advantage,
        "transfer_decision": transfer_decision,
        "causal_inference_permitted": False,
        "external_transfer_validated": False,
        "adapted_forecasts_support_prospective_claim": False,
    }
    return summary, interpretation


def ingest_results(
    manifest: ProspectiveTransferManifest,
    preregistration_root: str | Path,
    blinded_result_lock_root: str | Path,
    adaptation_roots: Mapping[str, str | Path],
    result_input_path: str | Path,
    raw_asset_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    prereg_verification = verify_preregistration(preregistration_root, manifest)
    blinded_lock_verification = verify_blinded_result_lock(
        blinded_result_lock_root, manifest, preregistration_root
    )
    blinded_result_lock = BlindedResultLock.model_validate_json(
        (
            Path(blinded_result_lock_root).resolve() / "blinded_result_lock.json"
        ).read_text(encoding="utf-8")
    )
    root = Path(preregistration_root).resolve()
    preregistration = Preregistration.model_validate_json(
        (root / "preregistration.json").read_text(encoding="utf-8")
    )
    results_path = Path(result_input_path).resolve()
    results = ResultInput.model_validate_json(results_path.read_text(encoding="utf-8"))
    if results.preregistration_bundle_sha256 != prereg_verification["bundle_sha256"]:
        raise ValueError("result input is bound to a different preregistration bundle")
    if (
        results.blinded_result_lock_bundle_sha256
        != blinded_lock_verification["bundle_sha256"]
    ):
        raise ValueError("result input is bound to a different blinded-result lock")
    locked_observations = {
        item.blinded_sample_id: item.model_dump(mode="json")
        for item in blinded_result_lock.observations
    }
    supplied_observations = {
        item.blinded_sample_id: item.model_dump(mode="json") for item in results.observations
    }
    locked_assets = sorted(
        (item.model_dump(mode="json") for item in blinded_result_lock.raw_assets),
        key=lambda item: item["asset_id"],
    )
    supplied_assets = sorted(
        (item.model_dump(mode="json") for item in results.raw_assets),
        key=lambda item: item["asset_id"],
    )
    if supplied_observations != locked_observations or supplied_assets != locked_assets:
        raise ValueError("final result differs from the complete blinded raw/QC lock")
    if set(adaptation_roots) != {"low_shot", "retrained"}:
        raise ValueError("result ingestion requires low-shot and retrained adaptation bundles")
    adaptation_verification = {}
    adaptation_receipts = {}
    adapted_forecasts = []
    for stage in ("low_shot", "retrained"):
        adaptation_root = Path(adaptation_roots[stage]).resolve()
        verification = verify_adaptation_bundle(
            adaptation_root,
            manifest,
            preregistration_root,
            blinded_result_lock_root,
        )
        if verification["stage"] != stage:
            raise ValueError("adaptation bundle stage and supplied role differ")
        receipt = AdaptationForecastReceipt.model_validate_json(
            (adaptation_root / "adaptation_receipt.json").read_text(encoding="utf-8")
        )
        adaptation_verification[stage] = verification
        adaptation_receipts[stage] = receipt
        adapted_forecasts.extend(receipt.forecasts)
    observed_adaptation_hashes = {
        stage: item["bundle_sha256"]
        for stage, item in adaptation_verification.items()
    }
    if results.adaptation_bundle_sha256 != observed_adaptation_hashes:
        raise ValueError("result input is bound to different adaptation bundles")
    if not (
        preregistration.created_utc
        < blinded_result_lock.locked_utc
        < adaptation_receipts["low_shot"].created_utc
        < adaptation_receipts["retrained"].created_utc
        < results.result_locked_utc
    ):
        raise ValueError(
            "preregistration, blinded result, adaptation, and final locks are out of order"
        )
    raw_asset_verification = _verify_raw_assets(results, raw_asset_root)
    observed, replicate_observed, ingestion = _aggregate_results(
        manifest, preregistration, results
    )
    summary, interpretation = _result_interpretation(
        manifest,
        preregistration,
        [*preregistration.forecasts, *adapted_forecasts],
        observed,
        replicate_observed,
    )
    summary["ingestion"] = ingestion
    payloads = {
        "result_input.json": results_path.read_bytes(),
        "summary.json": _canonical_bytes(summary),
        "interpretation.json": _canonical_bytes(interpretation),
        "provenance.json": _canonical_bytes(
            {
                "preregistration_bundle_sha256": prereg_verification["bundle_sha256"],
                "manifest_sha256": prereg_verification["manifest_sha256"],
                "blinded_result_lock_bundle_sha256": blinded_lock_verification[
                    "bundle_sha256"
                ],
                "adaptation_bundles": observed_adaptation_hashes,
                "result_input_sha256": _sha256(results_path),
                "raw_asset_verification": raw_asset_verification,
                "synthetic_fixture": True,
                "external_experiment_performed": False,
            }
        ),
    }
    published = _publish_bundle(output, payloads, "SUCCESS")
    return {
        "schema_version": RESULT_BUNDLE_VERSION,
        "interpretation_branch": interpretation["branch"],
        **published,
    }


def verify_result_bundle(root: str | Path) -> dict[str, Any]:
    return _verify_bundle(
        root,
        ["result_input.json", "summary.json", "interpretation.json", "provenance.json"],
        "SUCCESS",
    )
