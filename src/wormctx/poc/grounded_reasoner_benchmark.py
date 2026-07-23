"""Provider-neutral scaffold for a grounded biological-reasoning benchmark.

No model provider is called here.  The module builds private anonymized tasks,
freezes provider requests, validates typed inference programs, deterministically
replays locked tools, and scores externally produced response bundles.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from .developmental_omix import (
    SHA256_RE,
    _canonical_bytes,
    _sha256,
    _write_bytes_atomic,
    _write_json_atomic,
)


SCHEMA_VERSION = "wormctx-grounded-reasoner-benchmark-contract-1.0"
SOURCE_VERSION = "wormctx-grounded-reasoner-source-tasks-1.0"
RELEASE_VERSION = "wormctx-grounded-reasoner-release-1.0"
AUTHORITY_VERSION = "wormctx-grounded-reasoner-authority-1.0"
ANONYMIZATION_VERSION = "wormctx-grounded-reasoner-anonymization-receipt-1.0"
CONTAMINATION_VERSION = "wormctx-grounded-reasoner-contamination-receipt-1.0"
BUILD_VERSION = "wormctx-grounded-reasoner-build-manifest-1.0"
PROVIDER_VERSION = "wormctx-grounded-reasoner-provider-spec-1.0"
RUN_VERSION = "wormctx-grounded-reasoner-run-contract-1.0"
SUBMISSION_VERSION = "wormctx-grounded-reasoner-submission-1.0"
SCORE_VERSION = "wormctx-grounded-reasoner-score-1.0"
VERIFY_VERSION = "wormctx-grounded-reasoner-verification-1.0"
CUSTODY_VERSION = "wormctx-grounded-reasoner-custody-plan-1.0"
COMPARISON_PLAN_VERSION = "wormctx-grounded-reasoner-comparison-plan-1.0"
CONTAMINATION_ATTESTATION_VERSION = (
    "wormctx-grounded-reasoner-contamination-attestation-1.0"
)
ADAPTER_VERSION = "wormctx-grounded-reasoner-provider-adapter-1.0"
ADAPTER_RESPONSE_VERSION = "wormctx-grounded-reasoner-adapter-response-1.0"
RESPONSE_BUNDLE_VERSION = "wormctx-grounded-reasoner-response-bundle-1.0"

ARM_ORDER = [
    "text_only",
    "ordinary_rag",
    "graph_retrieval",
    "graph_plus_qualified_predictor",
    "tool_ablation_no_predictor",
    "tool_ablation_no_graph",
]
TOOL_OPERATORS = [
    "retrieve_text",
    "retrieve_graph",
    "check_identifiability",
    "call_locked_predictor",
]
PURE_OPERATORS = ["synthesize_evidence", "emit_prediction", "emit_abstention"]
ALL_OPERATORS = [*TOOL_OPERATORS, *PURE_OPERATORS]
SCORE_DIMENSIONS = [
    "prediction",
    "program_validity",
    "evidence_fidelity",
    "abstention",
    "execution_trace",
]

RELEASE_FILES = {
    "benchmark_release.json",
    "build_manifest.json",
    "comparison_plan.json",
    "contamination_receipt.json",
    "custody_asset_verification.json",
    "custody_attestation.json",
}
CURATOR_FILES = {
    "anonymization_receipt.json",
    "curator_authority.json",
    "curator_manifest.json",
    "custody_asset_verification.json",
    "custody_attestation.json",
}
GOLD_FILES = {
    "benchmark_gold.json",
    "custody_asset_verification.json",
    "custody_attestation.json",
    "gold_manifest.json",
}
SCORER_FILES = {
    "benchmark_scorer_authority.json",
    "custody_asset_verification.json",
    "custody_attestation.json",
    "scorer_manifest.json",
}
RUN_FILES = {
    "adapter_contract.json",
    "adapter_requests.jsonl",
    "contamination_attestation.json",
    "contamination_evidence_verification.json",
    "provider_requests.json",
    "run_contract.json",
    "run_manifest.json",
}
RESPONSE_FILES = {
    "adapter_responses.jsonl",
    "response_manifest.json",
    "response_receipts.json",
    "submission.json",
}
SCORE_FILES = {
    "contamination_assessment.json",
    "dimension_scores.json",
    "execution_receipts.json",
    "score_manifest.json",
    "task_scores.json",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ArmContract(_StrictModel):
    id: str
    kind: str
    allowed_tools: list[str]
    inline_context: str

    @model_validator(mode="after")
    def valid_arm(self) -> "ArmContract":
        if any(item not in TOOL_OPERATORS for item in self.allowed_tools):
            raise ValueError("arm allows an unknown tool")
        if len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError("arm tool allowlist contains duplicates")
        if self.inline_context not in {"task_prompt_only", "task_prompt_and_frozen_text"}:
            raise ValueError("unknown inline-context policy")
        return self


class AnonymizationContract(_StrictModel):
    entity_token_prefix: str
    task_token_prefix: str
    algorithm: str
    secret_stored_in_release: bool
    aliases_replaced: bool
    original_identifiers_permitted_in_release: bool

    @model_validator(mode="after")
    def exact_policy(self) -> "AnonymizationContract":
        if (
            self.algorithm != "hmac_sha256_truncated_12"
            or self.secret_stored_in_release
            or not self.aliases_replaced
            or self.original_identifiers_permitted_in_release
        ):
            raise ValueError("anonymization policy differs")
        return self


class ContaminationContract(_StrictModel):
    benchmark_visibility: str
    public_release_permitted: bool
    prompt_hash_receipts: bool
    per_task_canary_hash_receipts: bool
    provider_training_cutoff_required: bool
    provider_prior_exposure_attestation_required: bool
    external_search_permitted: bool

    @model_validator(mode="after")
    def exact_policy(self) -> "ContaminationContract":
        if (
            self.benchmark_visibility != "private_held_out"
            or self.public_release_permitted
            or not self.prompt_hash_receipts
            or not self.per_task_canary_hash_receipts
            or not self.provider_training_cutoff_required
            or not self.provider_prior_exposure_attestation_required
            or self.external_search_permitted
        ):
            raise ValueError("contamination-control policy differs")
        return self


class ScoringContract(_StrictModel):
    dimensions: list[str]
    composite_score_permitted: bool
    prediction_numeric_tolerance_source: str
    program_required_operator_recall: bool
    evidence_scoring: str
    abstention_scoring: str
    execution_trace_scoring: str

    @model_validator(mode="after")
    def exact_scoring(self) -> "ScoringContract":
        if self.dimensions != SCORE_DIMENSIONS or self.composite_score_permitted:
            raise ValueError("score dimensions must remain separate")
        if self.prediction_numeric_tolerance_source != "frozen_task_authority":
            raise ValueError("numeric tolerance source differs")
        if not self.program_required_operator_recall:
            raise ValueError("operator selection must be scored")
        if self.evidence_scoring != "citation_precision_recall_and_claim_support":
            raise ValueError("evidence scoring differs")
        if self.abstention_scoring != "correct_abstention_unsafe_answer_and_overabstention":
            raise ValueError("abstention scoring differs")
        if self.execution_trace_scoring != "deterministic_step_receipt_match":
            raise ValueError("execution-trace scoring differs")
        return self


class ConfirmatoryContract(_StrictModel):
    model_change_after_run_freeze_permitted: bool
    decoding_change_after_run_freeze_permitted: bool
    task_or_arm_exclusion_after_response_permitted: bool
    threshold_change_after_response_permitted: bool
    endpoint_change_after_response_permitted: bool
    predictor_change_after_response_permitted: bool
    gold_access_before_response_freeze_permitted: bool

    @model_validator(mode="after")
    def no_changes(self) -> "ConfirmatoryContract":
        if any(self.model_dump().values()):
            raise ValueError("confirmatory benchmark components may not change after freeze")
        return self


class AuthoritySeparationContract(_StrictModel):
    roles: list[str]
    separate_write_once_bundles_required: Literal[True]
    file_backed_acl_attestations_required: Literal[True]
    provider_may_read_gold_or_scorer_authority: Literal[False]
    scorer_may_open_gold_before_response_freeze: Literal[False]

    @model_validator(mode="after")
    def exact_roles(self) -> "AuthoritySeparationContract":
        if self.roles != ["release", "curator", "gold", "scorer"]:
            raise ValueError("authority roles differ from the frozen separation contract")
        return self


class AdapterPolicyContract(_StrictModel):
    supported_transports: list[str]
    credentials_source: Literal["environment_only"]
    endpoint_value_persisted: Literal[False]
    credential_value_persisted: Literal[False]
    deterministic_jsonl_receipts_required: Literal[True]
    external_call_performed_by_harness: Literal[False]

    @model_validator(mode="after")
    def exact_transports(self) -> "AdapterPolicyContract":
        if self.supported_transports != ["local_jsonl", "openai_compatible_jsonl"]:
            raise ValueError("provider-adapter transport contract differs")
        return self


class ComparisonPolicyContract(_StrictModel):
    plan_required_before_task_release: Literal[True]
    unit: Literal["paired_task_by_arm"]
    stratify_by_task_family: Literal[True]
    primary_comparisons: list[str]
    familywise_alpha: Literal[0.05]
    target_power: Literal[0.8]
    sample_size_method: Literal["simulation_or_exact_power_frozen_before_release"]
    assumptions_receipt_required: Literal[True]

    @model_validator(mode="after")
    def exact_comparisons(self) -> "ComparisonPolicyContract":
        if self.primary_comparisons != [
            "graph_plus_qualified_predictor_vs_ordinary_rag",
            "graph_plus_qualified_predictor_vs_graph_retrieval",
            "graph_plus_qualified_predictor_vs_tool_ablation_no_predictor",
            "graph_plus_qualified_predictor_vs_tool_ablation_no_graph",
        ]:
            raise ValueError("primary comparison family differs")
        return self


class BenchmarkContract(_StrictModel):
    schema_version: str
    analysis_id: str
    classification: str
    validated: bool
    biological_claims_permitted: bool
    arms: list[ArmContract]
    required_task_capabilities: list[str]
    anonymization: AnonymizationContract
    contamination: ContaminationContract
    scoring: ScoringContract
    confirmatory: ConfirmatoryContract
    authority_separation: AuthoritySeparationContract
    provider_adapter: AdapterPolicyContract
    comparison: ComparisonPolicyContract

    @model_validator(mode="after")
    def exact_contract(self) -> "BenchmarkContract":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unknown grounded-reasoner benchmark contract")
        if self.validated or self.biological_claims_permitted:
            raise ValueError("benchmark scaffold cannot assert biological validation")
        if [item.id for item in self.arms] != ARM_ORDER:
            raise ValueError("benchmark arm identities or order differ")
        expected_tools = {
            "text_only": [],
            "ordinary_rag": ["retrieve_text"],
            "graph_retrieval": ["retrieve_graph"],
            "graph_plus_qualified_predictor": [
                "retrieve_graph",
                "check_identifiability",
                "call_locked_predictor",
            ],
            "tool_ablation_no_predictor": [
                "retrieve_graph",
                "check_identifiability",
            ],
            "tool_ablation_no_graph": [
                "check_identifiability",
                "call_locked_predictor",
            ],
        }
        if {item.id: item.allowed_tools for item in self.arms} != expected_tools:
            raise ValueError("benchmark arm tool capabilities differ")
        if self.required_task_capabilities != [
            "operator_selection",
            "evidence_attribution",
            "uncertainty_or_abstention",
            "locked_predictor_call_when_available",
        ]:
            raise ValueError("benchmark task capabilities differ")
        return self


class FileAsset(_StrictModel):
    asset_id: str
    relative_path: str
    sha256: str
    bytes: int = Field(gt=0)

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("file asset path must be a safe POSIX-relative path")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("file asset hash is malformed")
        return value


class CustodyAttestation(_StrictModel):
    artifact_role: Literal["release", "curator", "gold", "scorer"]
    custodian_id: str
    os_principal: str
    access_policy: Literal[
        "provider_read_only",
        "curator_only",
        "gold_authority_only",
        "scorer_only_after_response_freeze",
    ]
    provider_read_permitted: bool
    contains_gold: bool
    gold_open_before_response_freeze_permitted: Literal[False]
    acl_enforced: Literal[True]
    authority_separation_attested: Literal[True]
    signed_by: str
    signed_utc: AwareDatetime
    acl_evidence: FileAsset

    @model_validator(mode="after")
    def exact_role_boundary(self) -> "CustodyAttestation":
        expected = {
            "release": ("provider_read_only", True, False),
            "curator": ("curator_only", False, False),
            "gold": ("gold_authority_only", False, True),
            "scorer": ("scorer_only_after_response_freeze", False, False),
        }[self.artifact_role]
        if (
            self.access_policy,
            self.provider_read_permitted,
            self.contains_gold,
        ) != expected:
            raise ValueError("custody attestation contradicts its authority role")
        return self


class CustodyPlan(_StrictModel):
    schema_version: Literal[CUSTODY_VERSION]
    plan_id: str
    synthetic_fixture: bool
    biological_claims_permitted: Literal[False]
    attestations: list[CustodyAttestation]

    @model_validator(mode="after")
    def exact_authorities(self) -> "CustodyPlan":
        roles = [item.artifact_role for item in self.attestations]
        if roles != ["release", "curator", "gold", "scorer"]:
            raise ValueError("custody plan must contain the four authority roles in order")
        hashes = [item.acl_evidence.sha256 for item in self.attestations]
        if len(hashes) != len(set(hashes)):
            raise ValueError("custody ACL evidence hashes must be unique")
        return self


class ComparisonPlan(_StrictModel):
    schema_version: Literal[COMPARISON_PLAN_VERSION]
    plan_id: str
    synthetic_fixture: bool
    biological_claims_permitted: Literal[False]
    unit: Literal["paired_task_by_arm"]
    task_counts_by_family: dict[str, int]
    total_tasks: int = Field(gt=0)
    primary_comparisons: list[str]
    score_dimensions: list[str]
    familywise_alpha: Literal[0.05]
    target_power: Literal[0.8]
    sample_size_method: Literal["simulation_or_exact_power_frozen_before_release"]
    assumptions_receipt: FileAsset
    frozen_before_task_release: Literal[True]

    @model_validator(mode="after")
    def coherent_plan(self) -> "ComparisonPlan":
        if (
            not self.task_counts_by_family
            or any(value <= 0 for value in self.task_counts_by_family.values())
            or sum(self.task_counts_by_family.values()) != self.total_tasks
        ):
            raise ValueError("comparison-plan task counts are incomplete")
        if self.score_dimensions != SCORE_DIMENSIONS:
            raise ValueError("comparison plan must preserve separate score dimensions")
        if self.primary_comparisons != [
            "graph_plus_qualified_predictor_vs_ordinary_rag",
            "graph_plus_qualified_predictor_vs_graph_retrieval",
            "graph_plus_qualified_predictor_vs_tool_ablation_no_predictor",
            "graph_plus_qualified_predictor_vs_tool_ablation_no_graph",
        ]:
            raise ValueError("comparison plan primary comparison family differs")
        return self


class ContaminationAttestation(_StrictModel):
    schema_version: Literal[CONTAMINATION_ATTESTATION_VERSION]
    attestation_id: str
    provider_id: str
    model_id: str
    model_version: str
    training_data_cutoff: str
    benchmark_seen_before_run: Literal[False]
    prior_canary_exposure_known: Literal[False]
    external_search_enabled: Literal[False]
    signed_by: str
    signed_utc: AwareDatetime
    evidence_receipt: FileAsset
    synthetic_fixture: bool
    biological_claims_permitted: Literal[False]


class ProviderAdapterContract(_StrictModel):
    schema_version: Literal[ADAPTER_VERSION]
    adapter_id: str
    adapter_version: str
    transport: Literal["local_jsonl", "openai_compatible_jsonl"]
    endpoint_env_var: str
    api_key_env_var: str | None
    credentials_source: Literal["environment_only"]
    request_format: Literal["deterministic_jsonl"]
    response_format: Literal["deterministic_jsonl"]
    endpoint_value_persisted: Literal[False]
    credential_value_persisted: Literal[False]
    external_call_performed_by_harness: Literal[False]

    @field_validator("endpoint_env_var", "api_key_env_var")
    @classmethod
    def environment_name_only(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Z][A-Z0-9_]{2,127}", value):
            raise ValueError("adapter credentials must be referenced by environment-variable name")
        return value

    @model_validator(mode="after")
    def exact_transport(self) -> "ProviderAdapterContract":
        if self.transport == "openai_compatible_jsonl" and self.api_key_env_var is None:
            raise ValueError("OpenAI-compatible adapter requires an API-key environment variable")
        if self.transport == "local_jsonl" and self.api_key_env_var is not None:
            raise ValueError("local JSONL adapter must not require an API key")
        return self


class SourceEntity(_StrictModel):
    identifier: str
    aliases: list[str]

    @model_validator(mode="after")
    def unique_aliases(self) -> "SourceEntity":
        values = [self.identifier, *self.aliases]
        if any(len(value) < 3 for value in values) or len(values) != len(set(values)):
            raise ValueError("source entity identifiers and aliases must be unique and specific")
        return self


class EvidenceItem(_StrictModel):
    evidence_id: str
    channel: Literal["inline", "text", "graph"]
    content: str


class LockedPredictor(_StrictModel):
    predictor_id: str
    version: str
    input_sha256: str
    output: float | str
    qualification_receipt_sha256: str

    @field_validator("input_sha256", "qualification_receipt_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("locked predictor hash is malformed")
        return value

    @field_validator("output")
    @classmethod
    def finite_output(cls, value: float | str) -> float | str:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("locked predictor output must be finite")
        return value


class GoldAnswer(_StrictModel):
    disposition: Literal["predict", "abstain"]
    prediction_type: Literal["numeric", "categorical", "none"]
    target: float | str | None
    numeric_tolerance: float = Field(ge=0.0)
    evidence_ids: list[str]
    required_operators: list[str]
    abstention_reason_code: str | None

    @model_validator(mode="after")
    def coherent_gold(self) -> "GoldAnswer":
        if any(item not in ALL_OPERATORS for item in self.required_operators):
            raise ValueError("gold answer requires an unknown operator")
        if len(self.required_operators) != len(set(self.required_operators)):
            raise ValueError("gold required operators must be unique")
        if self.disposition == "abstain":
            if (
                self.prediction_type != "none"
                or self.target is not None
                or not self.abstention_reason_code
            ):
                raise ValueError("abstention gold is incoherent")
        elif self.prediction_type == "none" or self.target is None or self.abstention_reason_code:
            raise ValueError("prediction gold is incoherent")
        if isinstance(self.target, float) and not math.isfinite(self.target):
            raise ValueError("numeric gold target must be finite")
        if not math.isfinite(self.numeric_tolerance):
            raise ValueError("numeric tolerance must be finite")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("gold evidence IDs must be unique")
        return self


class SourceTask(_StrictModel):
    task_id: str
    task_family: str
    question: str
    entities: list[SourceEntity]
    evidence: list[EvidenceItem]
    identifiable: bool
    identifiability_reason_code: str
    predictor: LockedPredictor | None
    gold: GoldAnswer

    @model_validator(mode="after")
    def complete_task(self) -> "SourceTask":
        entity_ids = [item.identifier for item in self.entities]
        if not self.entities or len(entity_ids) != len(set(entity_ids)):
            raise ValueError("task entities must be present and unique")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if not self.evidence or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("task evidence IDs must be present and unique")
        if not set(self.gold.evidence_ids).issubset(evidence_ids):
            raise ValueError("gold cites evidence outside the task")
        if (self.gold.disposition == "predict") != self.identifiable:
            raise ValueError("gold disposition and identifiability differ")
        if "call_locked_predictor" in self.gold.required_operators and self.predictor is None:
            raise ValueError("task requires a missing locked predictor")
        return self


class SourceBenchmark(_StrictModel):
    schema_version: str
    source_snapshot_id: str
    tasks: list[SourceTask]
    predictor_receipt_assets: list[FileAsset]

    @model_validator(mode="after")
    def exact_source(self) -> "SourceBenchmark":
        if self.schema_version != SOURCE_VERSION:
            raise ValueError("unknown grounded-reasoner source schema")
        ids = [item.task_id for item in self.tasks]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("source task IDs must be present and unique")
        required_receipts = {
            item.predictor.qualification_receipt_sha256
            for item in self.tasks
            if item.predictor is not None
        }
        if not required_receipts:
            raise ValueError("source benchmark has no task with a qualified predictor")
        declared_receipts = {item.sha256 for item in self.predictor_receipt_assets}
        if declared_receipts != required_receipts:
            raise ValueError("predictor receipt assets do not exactly cover locked predictors")
        return self


class DecodingSpec(_StrictModel):
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    seed: int
    max_output_tokens: int = Field(gt=0)


class ProviderSpec(_StrictModel):
    schema_version: str
    provider_id: str
    model_id: str
    model_version: str
    weights_receipt: str
    training_data_cutoff: str
    prompt_template_sha256: str
    decoding: DecodingSpec
    benchmark_seen_before_run: bool
    external_search_enabled: bool

    @field_validator("weights_receipt", "prompt_template_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("prompt-template hash is malformed")
        return value

    @model_validator(mode="after")
    def frozen_provider(self) -> "ProviderSpec":
        if self.schema_version != PROVIDER_VERSION:
            raise ValueError("unknown provider specification schema")
        if not self.training_data_cutoff:
            raise ValueError("provider training-data cutoff is required")
        if self.benchmark_seen_before_run or self.external_search_enabled:
            raise ValueError("provider exposure or external search violates the private benchmark")
        return self


class ProgramStep(_StrictModel):
    step_id: str
    operator: str
    depends_on: list[str]
    arguments: dict[str, Any]

    @field_validator("operator")
    @classmethod
    def known_operator(cls, value: str) -> str:
        if value not in ALL_OPERATORS:
            raise ValueError("program step uses an unknown operator")
        return value


class TraceStep(_StrictModel):
    step_id: str
    status: Literal["success", "failure"]
    output_sha256: str

    @field_validator("output_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("trace output hash is malformed")
        return value


class ClaimAttribution(_StrictModel):
    claim: str
    evidence_ids: list[str]


class FinalAnswer(_StrictModel):
    disposition: Literal["predict", "abstain"]
    prediction: float | str | None
    confidence: float = Field(ge=0.0, le=1.0)
    interval_lower: float | None
    interval_upper: float | None
    abstention_reason_code: str | None
    citations: list[str]
    claims: list[ClaimAttribution]

    @model_validator(mode="after")
    def coherent_final(self) -> "FinalAnswer":
        if self.disposition == "abstain":
            if self.prediction is not None or not self.abstention_reason_code:
                raise ValueError("abstention response is incoherent")
        elif self.prediction is None or self.abstention_reason_code is not None:
            raise ValueError("prediction response is incoherent")
        if (self.interval_lower is None) != (self.interval_upper is None):
            raise ValueError("uncertainty interval must provide both bounds or neither")
        if (
            self.interval_lower is not None
            and self.interval_upper is not None
            and self.interval_lower > self.interval_upper
        ):
            raise ValueError("uncertainty interval bounds are reversed")
        numeric_values = [
            value
            for value in (self.interval_lower, self.interval_upper)
            if value is not None
        ]
        if isinstance(self.prediction, float):
            numeric_values.append(self.prediction)
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("prediction and uncertainty interval must be finite")
        if len(self.citations) != len(set(self.citations)):
            raise ValueError("final citations must be unique")
        return self


class TaskResponse(_StrictModel):
    request_id: str
    task_id: str
    arm_id: str
    program: list[ProgramStep]
    trace: list[TraceStep]
    final: FinalAnswer

    @model_validator(mode="after")
    def unique_trace_steps(self) -> "TaskResponse":
        trace_ids = [item.step_id for item in self.trace]
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError("response trace contains duplicate step IDs")
        return self


class Submission(_StrictModel):
    schema_version: str
    run_contract_sha256: str
    benchmark_release_sha256: str
    provider_spec_sha256: str
    adapter_contract_sha256: str
    contamination_attestation_sha256: str
    response_receipts_sha256: str
    model_or_decoding_changed_after_freeze: bool
    thresholds_or_endpoints_changed_after_freeze: bool
    gold_accessed_before_response_freeze: bool
    responses: list[TaskResponse]

    @field_validator(
        "run_contract_sha256",
        "benchmark_release_sha256",
        "provider_spec_sha256",
        "adapter_contract_sha256",
        "contamination_attestation_sha256",
        "response_receipts_sha256",
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("submission binding hash is malformed")
        return value

    @model_validator(mode="after")
    def confirmatory_submission(self) -> "Submission":
        if self.schema_version != SUBMISSION_VERSION:
            raise ValueError("unknown grounded-reasoner submission schema")
        if (
            self.model_or_decoding_changed_after_freeze
            or self.thresholds_or_endpoints_changed_after_freeze
            or self.gold_accessed_before_response_freeze
        ):
            raise ValueError("submission violates the confirmatory freeze")
        ids = [item.request_id for item in self.responses]
        if len(ids) != len(set(ids)):
            raise ValueError("submission contains duplicate request IDs")
        return self


class AdapterResponseEnvelope(_StrictModel):
    schema_version: Literal[ADAPTER_RESPONSE_VERSION]
    request_id: str
    task_id: str
    arm_id: str
    provider_id: str
    model_id: str
    model_version: str
    decoding_sha256: str
    response: TaskResponse

    @field_validator("decoding_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        value = value.lower()
        if not SHA256_RE.fullmatch(value):
            raise ValueError("adapter decoding hash is malformed")
        return value

    @model_validator(mode="after")
    def exact_response_identity(self) -> "AdapterResponseEnvelope":
        if (
            self.request_id != self.response.request_id
            or self.task_id != self.response.task_id
            or self.arm_id != self.response.arm_id
        ):
            raise ValueError("adapter response envelope and typed response identities differ")
        return self


@runtime_checkable
class ReasonerProvider(Protocol):
    """Provider-neutral interface; implementations live outside this benchmark."""

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


def load_contract(path: str | Path) -> BenchmarkContract:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"grounded-reasoner contract is missing: {source}")
    return BenchmarkContract.model_validate_json(source.read_text(encoding="utf-8"))


def load_source(path: str | Path) -> SourceBenchmark:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"grounded-reasoner source tasks are missing: {source}")
    return SourceBenchmark.model_validate_json(source.read_text(encoding="utf-8"))


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL artifact is missing: {path}")
    rows: list[tuple[bytes, dict[str, Any]]] = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"JSONL contains a blank row at line {line_number}")
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object at line {line_number}")
        rows.append((raw_line, value))
    if not rows:
        raise ValueError("JSONL artifact contains no rows")
    return rows


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(dict(item)) for item in rows)


def _verify_file_assets(
    assets: Sequence[FileAsset], asset_root: str | Path, *, label: str
) -> dict[str, Any]:
    root = Path(asset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{label} asset root is missing: {root}")
    if (
        len({item.asset_id for item in assets}) != len(assets)
        or len({item.relative_path for item in assets}) != len(assets)
        or len({item.sha256 for item in assets}) != len(assets)
    ):
        raise ValueError(f"{label} asset IDs, paths, and hashes must be unique")
    verified = []
    for asset in assets:
        path = (root / Path(*PurePosixPath(asset.relative_path).parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} asset escapes its declared root") from exc
        if not path.is_file():
            raise FileNotFoundError(f"{label} asset is missing: {asset.relative_path}")
        if path.stat().st_size != asset.bytes or _sha256(path) != asset.sha256:
            raise ValueError(f"{label} asset checksum or byte count differs: {asset.asset_id}")
        verified.append(asset.model_dump(mode="json"))
    return {"label": label, "assets": verified, "verified": True}


def _opaque(secret: bytes, namespace: str, value: str, prefix: str) -> str:
    digest = hmac.new(secret, f"{namespace}|{value}".encode(), hashlib.sha256).hexdigest()
    return f"{prefix}{digest[:12]}"


def _replace_aliases(text: str, replacements: Mapping[str, str]) -> str:
    result = text
    for source in sorted(replacements, key=lambda item: (-len(item), item)):
        result = result.replace(source, replacements[source])
    return result


def _assert_anonymous(payload: Any, forbidden: Sequence[str]) -> None:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    leaked = [item for item in forbidden if item in encoded]
    if leaked:
        raise ValueError(f"anonymized release retains source identifiers: {leaked[:5]}")


def _write_bundle(
    root: Path, payloads: Mapping[str, Any | bytes], expected: set[str]
) -> None:
    if set(payloads) != expected:
        raise ValueError("bundle payload inventory differs")
    if root.exists():
        raise FileExistsError(f"output root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = root.parent / f".{root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for name in sorted(payloads):
            payload = payloads[name]
            if isinstance(payload, bytes):
                _write_bytes_atomic(temporary / name, payload)
            else:
                _write_json_atomic(temporary / name, payload)
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
        raise ValueError("bundle inventory differs")
    if (root / "SUCCESS").read_bytes() != b"SUCCESS\n":
        raise ValueError("bundle SUCCESS marker differs")
    checked: set[str] = set()
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None or match.group(2) not in expected_files or match.group(2) in checked:
            raise ValueError("bundle checksum manifest is malformed")
        if _sha256(root / match.group(2)) != match.group(1):
            raise ValueError(f"bundle checksum failed: {match.group(2)}")
        checked.add(match.group(2))
    if checked != expected_files:
        raise ValueError("bundle checksum coverage differs")


def build_benchmark(
    contract_or_path: BenchmarkContract | str | Path,
    source_or_path: SourceBenchmark | str | Path,
    output_root: str | Path,
    *,
    anonymization_secret: str,
    custody_plan_or_path: CustodyPlan | str | Path,
    custody_asset_root: str | Path,
    comparison_plan_or_path: ComparisonPlan | str | Path,
    comparison_asset_root: str | Path,
    predictor_receipt_root: str | Path,
) -> dict[str, Any]:
    contract = (
        contract_or_path
        if isinstance(contract_or_path, BenchmarkContract)
        else load_contract(contract_or_path)
    )
    source = (
        source_or_path
        if isinstance(source_or_path, SourceBenchmark)
        else load_source(source_or_path)
    )
    custody = (
        custody_plan_or_path
        if isinstance(custody_plan_or_path, CustodyPlan)
        else CustodyPlan.model_validate_json(
            Path(custody_plan_or_path).resolve().read_text(encoding="utf-8")
        )
    )
    comparison = (
        comparison_plan_or_path
        if isinstance(comparison_plan_or_path, ComparisonPlan)
        else ComparisonPlan.model_validate_json(
            Path(comparison_plan_or_path).resolve().read_text(encoding="utf-8")
        )
    )
    if comparison.primary_comparisons != contract.comparison.primary_comparisons:
        raise ValueError("comparison plan differs from the contract's primary comparisons")
    family_counts: dict[str, int] = defaultdict(int)
    for task in source.tasks:
        family_counts[task.task_family] += 1
    if dict(family_counts) != comparison.task_counts_by_family:
        raise ValueError("source task-family counts differ from the frozen comparison plan")
    custody_verification = _verify_file_assets(
        [item.acl_evidence for item in custody.attestations],
        custody_asset_root,
        label="custody ACL evidence",
    )
    comparison_verification = _verify_file_assets(
        [comparison.assumptions_receipt],
        comparison_asset_root,
        label="comparison assumptions",
    )
    predictor_verification = _verify_file_assets(
        source.predictor_receipt_assets,
        predictor_receipt_root,
        label="qualified predictor receipts",
    )
    if len(anonymization_secret) < 32:
        raise ValueError("anonymization secret must contain at least 32 characters")
    secret = anonymization_secret.encode("utf-8")
    entity_map: dict[str, str] = {}
    alias_map: dict[str, str] = {}
    task_map: dict[str, str] = {}
    forbidden: list[str] = []
    for task in source.tasks:
        task_map[task.task_id] = _opaque(
            secret, "task", task.task_id, contract.anonymization.task_token_prefix
        )
        forbidden.append(task.task_id)
        for entity in task.entities:
            token = entity_map.setdefault(
                entity.identifier,
                _opaque(
                    secret,
                    "entity",
                    entity.identifier,
                    contract.anonymization.entity_token_prefix,
                ),
            )
            alias_map[entity.identifier] = token
            forbidden.append(entity.identifier)
            for alias in entity.aliases:
                alias_map[alias] = token
                forbidden.append(alias)

    release_tasks: list[dict[str, Any]] = []
    scorer_tasks: list[dict[str, Any]] = []
    gold_tasks: list[dict[str, Any]] = []
    prompt_hashes: dict[str, str] = {}
    canary_hashes: dict[str, str] = {}
    for task in source.tasks:
        opaque_task = task_map[task.task_id]
        canary = _opaque(secret, "canary", task.task_id, "CANARY-")
        question = _replace_aliases(task.question, alias_map)
        evidence = [
            {
                "evidence_id": item.evidence_id,
                "channel": item.channel,
                "content": _replace_aliases(item.content, alias_map),
            }
            for item in task.evidence
        ]
        inline = [item for item in evidence if item["channel"] == "inline"]
        prompt = f"{question}\nPrivate benchmark token: {canary}"
        predictor_catalog = (
            None
            if task.predictor is None
            else {
                "predictor_id": task.predictor.predictor_id,
                "version": task.predictor.version,
                "input_sha256": task.predictor.input_sha256,
                "qualification_receipt_sha256": (
                    task.predictor.qualification_receipt_sha256
                ),
            }
        )
        release_tasks.append(
            {
                "task_id": opaque_task,
                "task_family": task.task_family,
                "prompt": prompt,
                "entity_tokens": sorted(
                    {entity_map[item.identifier] for item in task.entities}
                ),
                "inline_evidence": inline,
                "resource_catalog": [
                    {"evidence_id": item["evidence_id"], "channel": item["channel"]}
                    for item in evidence
                    if item["channel"] != "inline"
                ],
                "locked_predictor": predictor_catalog,
            }
        )
        scorer_tasks.append(
            {
                "task_id": opaque_task,
                "evidence": evidence,
                "identifiable": task.identifiable,
                "identifiability_reason_code": task.identifiability_reason_code,
                "predictor": None if task.predictor is None else task.predictor.model_dump(),
            }
        )
        gold_tasks.append(
            {
                "task_id": opaque_task,
                "gold": task.gold.model_dump(mode="json"),
            }
        )
        prompt_hashes[opaque_task] = hashlib.sha256(prompt.encode()).hexdigest()
        canary_hashes[opaque_task] = hashlib.sha256(canary.encode()).hexdigest()

    release = {
        "schema_version": RELEASE_VERSION,
        "analysis_id": contract.analysis_id,
        "visibility": contract.contamination.benchmark_visibility,
        "public_release_permitted": False,
        "task_capabilities": contract.required_task_capabilities,
        "arms": [item.model_dump(mode="json") for item in contract.arms],
        "operator_schemas": _operator_schemas(),
        "tasks": release_tasks,
        "gold_included": False,
        "anonymization_secret_included": False,
    }
    _assert_anonymous(release, forbidden)
    curator_authority = {
        "schema_version": AUTHORITY_VERSION,
        "source_snapshot_id": source.source_snapshot_id,
        "task_map": task_map,
        "entity_map": entity_map,
        "anonymization_secret_sha256": hashlib.sha256(secret).hexdigest(),
        "artifact_role": "curator",
        "gold_included": False,
        "governed_nonrelease_artifact": True,
    }
    scorer_authority = {
        "schema_version": AUTHORITY_VERSION,
        "source_snapshot_id": source.source_snapshot_id,
        "tasks": scorer_tasks,
        "artifact_role": "scorer",
        "source_identifiers_included": False,
        "gold_included": False,
        "governed_nonrelease_artifact": True,
    }
    benchmark_gold = {
        "schema_version": AUTHORITY_VERSION,
        "source_snapshot_id": source.source_snapshot_id,
        "tasks": gold_tasks,
        "artifact_role": "gold",
        "source_identifiers_included": False,
        "gold_included": True,
        "governed_nonrelease_artifact": True,
    }
    anonymization_receipt = {
        "schema_version": ANONYMIZATION_VERSION,
        "algorithm": contract.anonymization.algorithm,
        "entities": len(entity_map),
        "tasks": len(task_map),
        "entity_map_sha256": _stable_hash(entity_map),
        "task_map_sha256": _stable_hash(task_map),
        "secret_sha256": hashlib.sha256(secret).hexdigest(),
        "secret_stored_in_release": False,
        "original_identifiers_in_release": False,
    }
    contamination_receipt = {
        "schema_version": CONTAMINATION_VERSION,
        "benchmark_visibility": "private_held_out",
        "source_snapshot_id": source.source_snapshot_id,
        "prompt_sha256_by_task": prompt_hashes,
        "canary_sha256_by_task": canary_hashes,
        "provider_training_cutoff_required": True,
        "provider_prior_exposure_attestation_required": True,
        "status": "unassessed_until_provider_run_freeze",
    }
    contract_hash = (
        _sha256(Path(contract_or_path).resolve())
        if isinstance(contract_or_path, (str, Path))
        else _stable_hash(contract.model_dump(mode="json"))
    )
    source_hash = (
        _sha256(Path(source_or_path).resolve())
        if isinstance(source_or_path, (str, Path))
        else _stable_hash(source.model_dump(mode="json"))
    )
    # These two governed objects are re-serialized into the bundle.  Bind the
    # canonical object, rather than incidental whitespace in an input file.
    custody_hash = _stable_hash(custody.model_dump(mode="json"))
    comparison_hash = _stable_hash(comparison.model_dump(mode="json"))
    build_manifest = {
        "schema_version": BUILD_VERSION,
        "analysis_id": contract.analysis_id,
        "contract_sha256": contract_hash,
        "source_tasks_sha256": source_hash,
        "benchmark_release_sha256": _stable_hash(release),
        "curator_authority_sha256": _stable_hash(curator_authority),
        "benchmark_gold_sha256": _stable_hash(benchmark_gold),
        "benchmark_scorer_authority_sha256": _stable_hash(scorer_authority),
        "anonymization_receipt_sha256": _stable_hash(anonymization_receipt),
        "contamination_receipt_sha256": _stable_hash(contamination_receipt),
        "custody_plan_sha256": custody_hash,
        "custody_attestation_sha256_by_role": {
            item.artifact_role: _stable_hash(item.model_dump(mode="json"))
            for item in custody.attestations
        },
        "comparison_plan_sha256": comparison_hash,
        "custody_asset_verification": custody_verification,
        "comparison_asset_verification": comparison_verification,
        "predictor_receipt_verification": predictor_verification,
        "qualified_predictor_receipts_verified": True,
        "private_authorities_frozen": True,
        "tasks": len(release_tasks),
        "arms": len(contract.arms),
        "external_model_called": False,
        "biological_claims_permitted": False,
    }
    custody_by_role = {item.artifact_role: item for item in custody.attestations}

    def custody_payload(role: str) -> tuple[dict[str, Any], dict[str, Any]]:
        attestation = custody_by_role[role]
        return (
            attestation.model_dump(mode="json"),
            {
                "schema_version": CUSTODY_VERSION,
                "artifact_role": role,
                "asset": attestation.acl_evidence.model_dump(mode="json"),
                "verified": True,
            },
        )

    release_custody, release_acl = custody_payload("release")
    curator_custody, curator_acl = custody_payload("curator")
    gold_custody, gold_acl = custody_payload("gold")
    scorer_custody, scorer_acl = custody_payload("scorer")
    release_payloads = {
        "benchmark_release.json": release,
        "build_manifest.json": build_manifest,
        "comparison_plan.json": comparison.model_dump(mode="json"),
        "contamination_receipt.json": contamination_receipt,
        "custody_attestation.json": release_custody,
        "custody_asset_verification.json": release_acl,
    }
    curator_payloads = {
        "anonymization_receipt.json": anonymization_receipt,
        "curator_authority.json": curator_authority,
        "curator_manifest.json": {
            "schema_version": BUILD_VERSION,
            "benchmark_release_sha256": _stable_hash(release),
            "curator_authority_sha256": _stable_hash(curator_authority),
            "source_tasks_sha256": source_hash,
            "provider_access_permitted": False,
        },
        "custody_attestation.json": curator_custody,
        "custody_asset_verification.json": curator_acl,
    }
    gold_payloads = {
        "benchmark_gold.json": benchmark_gold,
        "gold_manifest.json": {
            "schema_version": BUILD_VERSION,
            "benchmark_release_sha256": _stable_hash(release),
            "benchmark_gold_sha256": _stable_hash(benchmark_gold),
            "response_freeze_required_before_open": True,
            "provider_access_permitted": False,
        },
        "custody_attestation.json": gold_custody,
        "custody_asset_verification.json": gold_acl,
    }
    scorer_payloads = {
        "benchmark_scorer_authority.json": scorer_authority,
        "scorer_manifest.json": {
            "schema_version": BUILD_VERSION,
            "benchmark_release_sha256": _stable_hash(release),
            "benchmark_scorer_authority_sha256": _stable_hash(scorer_authority),
            "predictor_receipt_verification": predictor_verification,
            "gold_included": False,
            "provider_access_permitted": False,
        },
        "custody_attestation.json": scorer_custody,
        "custody_asset_verification.json": scorer_acl,
    }
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"output root already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        _write_bundle(stage / "release", release_payloads, RELEASE_FILES)
        _write_bundle(stage / "curator", curator_payloads, CURATOR_FILES)
        _write_bundle(stage / "gold", gold_payloads, GOLD_FILES)
        _write_bundle(stage / "scorer", scorer_payloads, SCORER_FILES)
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return build_manifest


def _operator_schemas() -> dict[str, Any]:
    return {
        "retrieve_text": {"required_arguments": ["query"]},
        "retrieve_graph": {"required_arguments": ["query"]},
        "check_identifiability": {"required_arguments": ["question_type"]},
        "call_locked_predictor": {
            "required_arguments": ["predictor_id", "version", "input_sha256"]
        },
        "synthesize_evidence": {"required_arguments": ["evidence_ids"]},
        "emit_prediction": {"required_arguments": ["prediction"]},
        "emit_abstention": {"required_arguments": ["reason_code"]},
    }


def _role_root(root: str | Path, role: str) -> Path:
    path = Path(root).resolve()
    return path if path.name == role else path / role


def _verify_custody_role(path: Path, role: str) -> None:
    attestation = CustodyAttestation.model_validate(
        _read_json(path / "custody_attestation.json")
    )
    verification = _read_json(path / "custody_asset_verification.json")
    if (
        attestation.artifact_role != role
        or verification.get("artifact_role") != role
        or verification.get("verified") is not True
        or verification.get("asset") != attestation.acl_evidence.model_dump(mode="json")
    ):
        raise ValueError(f"{role} custody evidence differs")


def verify_release_bundle(root: str | Path) -> dict[str, Any]:
    path = _role_root(root, "release")
    _verify_bundle(path, RELEASE_FILES)
    _verify_custody_role(path, "release")
    release = _read_json(path / "benchmark_release.json")
    manifest = _read_json(path / "build_manifest.json")
    comparison = ComparisonPlan.model_validate(_read_json(path / "comparison_plan.json"))
    contamination = _read_json(path / "contamination_receipt.json")
    custody = CustodyAttestation.model_validate(
        _read_json(path / "custody_attestation.json")
    )
    custody_map = manifest.get("custody_attestation_sha256_by_role")
    comparison_verification = manifest.get("comparison_asset_verification")
    if (
        release.get("schema_version") != RELEASE_VERSION
        or release.get("gold_included") is not False
        or release.get("anonymization_secret_included") is not False
        or release.get("public_release_permitted") is not False
    ):
        raise ValueError("benchmark release violates privacy or gold separation")
    if (
        manifest.get("benchmark_release_sha256") != _stable_hash(release)
        or manifest.get("comparison_plan_sha256")
        != _stable_hash(comparison.model_dump(mode="json"))
        or manifest.get("contamination_receipt_sha256") != _stable_hash(contamination)
        or manifest.get("private_authorities_frozen") is not True
        or manifest.get("qualified_predictor_receipts_verified") is not True
        or not isinstance(custody_map, dict)
        or sorted(custody_map) != ["curator", "gold", "release", "scorer"]
        or custody_map.get("release")
        != _stable_hash(custody.model_dump(mode="json"))
        or not isinstance(comparison_verification, dict)
        or comparison_verification.get("verified") is not True
        or comparison_verification.get("assets")
        != [comparison.assumptions_receipt.model_dump(mode="json")]
    ):
        raise ValueError("release differs from its build bindings")
    return {
        "schema_version": VERIFY_VERSION,
        "artifact_role": "release",
        "verified": True,
        "benchmark_release_sha256": _sha256(path / "benchmark_release.json"),
        "build_manifest_sha256": _sha256(path / "build_manifest.json"),
        "comparison_plan_sha256": _sha256(path / "comparison_plan.json"),
        "contamination_receipt_sha256": _sha256(
            path / "contamination_receipt.json"
        ),
    }


def verify_curator_bundle(root: str | Path) -> dict[str, Any]:
    path = _role_root(root, "curator")
    _verify_bundle(path, CURATOR_FILES)
    _verify_custody_role(path, "curator")
    authority = _read_json(path / "curator_authority.json")
    manifest = _read_json(path / "curator_manifest.json")
    if (
        authority.get("artifact_role") != "curator"
        or authority.get("gold_included") is not False
        or manifest.get("curator_authority_sha256") != _stable_hash(authority)
        or manifest.get("provider_access_permitted") is not False
    ):
        raise ValueError("curator authority separation differs")
    return {
        "schema_version": VERIFY_VERSION,
        "artifact_role": "curator",
        "verified": True,
        "curator_authority_sha256": _sha256(path / "curator_authority.json"),
    }


def verify_gold_bundle(root: str | Path) -> dict[str, Any]:
    path = _role_root(root, "gold")
    _verify_bundle(path, GOLD_FILES)
    _verify_custody_role(path, "gold")
    authority = _read_json(path / "benchmark_gold.json")
    manifest = _read_json(path / "gold_manifest.json")
    if (
        authority.get("artifact_role") != "gold"
        or authority.get("gold_included") is not True
        or manifest.get("benchmark_gold_sha256") != _stable_hash(authority)
        or manifest.get("response_freeze_required_before_open") is not True
        or manifest.get("provider_access_permitted") is not False
    ):
        raise ValueError("gold authority separation differs")
    return {
        "schema_version": VERIFY_VERSION,
        "artifact_role": "gold",
        "verified": True,
        "benchmark_gold_sha256": _sha256(path / "benchmark_gold.json"),
    }


def verify_scorer_bundle(root: str | Path) -> dict[str, Any]:
    path = _role_root(root, "scorer")
    _verify_bundle(path, SCORER_FILES)
    _verify_custody_role(path, "scorer")
    authority = _read_json(path / "benchmark_scorer_authority.json")
    manifest = _read_json(path / "scorer_manifest.json")
    predictor_verification = manifest.get("predictor_receipt_verification")
    required_predictor_receipts = {
        item["predictor"]["qualification_receipt_sha256"]
        for item in authority.get("tasks", [])
        if item.get("predictor") is not None
    }
    verified_predictor_receipts = {
        item.get("sha256")
        for item in (
            predictor_verification.get("assets", [])
            if isinstance(predictor_verification, dict)
            else []
        )
    }
    if (
        authority.get("artifact_role") != "scorer"
        or authority.get("gold_included") is not False
        or manifest.get("benchmark_scorer_authority_sha256") != _stable_hash(authority)
        or manifest.get("gold_included") is not False
        or manifest.get("provider_access_permitted") is not False
        or not isinstance(predictor_verification, dict)
        or predictor_verification.get("verified") is not True
        or verified_predictor_receipts != required_predictor_receipts
    ):
        raise ValueError("scorer authority separation differs")
    return {
        "schema_version": VERIFY_VERSION,
        "artifact_role": "scorer",
        "verified": True,
        "benchmark_scorer_authority_sha256": _sha256(
            path / "benchmark_scorer_authority.json"
        ),
    }


def verify_benchmark(root: str | Path) -> dict[str, Any]:
    path = Path(root).resolve()
    release = verify_release_bundle(path)
    curator = verify_curator_bundle(path)
    gold = verify_gold_bundle(path)
    scorer = verify_scorer_bundle(path)
    manifest = _read_json(path / "release" / "build_manifest.json")
    release_hash = release["benchmark_release_sha256"]
    role_manifests = {
        "curator": _read_json(path / "curator" / "curator_manifest.json"),
        "gold": _read_json(path / "gold" / "gold_manifest.json"),
        "scorer": _read_json(path / "scorer" / "scorer_manifest.json"),
    }
    custody_hashes = {
        role: _stable_hash(
            _read_json(path / role / "custody_attestation.json")
        )
        for role in ("release", "curator", "gold", "scorer")
    }
    if (
        manifest.get("curator_authority_sha256")
        != curator["curator_authority_sha256"]
        or manifest.get("benchmark_gold_sha256") != gold["benchmark_gold_sha256"]
        or manifest.get("benchmark_scorer_authority_sha256")
        != scorer["benchmark_scorer_authority_sha256"]
        or any(
            item.get("benchmark_release_sha256") != release_hash
            for item in role_manifests.values()
        )
        or manifest.get("custody_attestation_sha256_by_role") != custody_hashes
    ):
        raise ValueError("separated authority bundles differ from the release manifest")
    return {
        "schema_version": VERIFY_VERSION,
        "verified": True,
        "roles": ["release", "curator", "gold", "scorer"],
        **{key: value for key, value in release.items() if key.endswith("sha256")},
        **{key: value for key, value in curator.items() if key.endswith("sha256")},
        **{key: value for key, value in gold.items() if key.endswith("sha256")},
        **{key: value for key, value in scorer.items() if key.endswith("sha256")},
    }


def freeze_provider_run(
    contract_or_path: BenchmarkContract | str | Path,
    benchmark_root: str | Path,
    provider_or_path: ProviderSpec | str | Path,
    adapter_or_path: ProviderAdapterContract | str | Path,
    contamination_attestation_or_path: ContaminationAttestation | str | Path,
    contamination_asset_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    contract = (
        contract_or_path
        if isinstance(contract_or_path, BenchmarkContract)
        else load_contract(contract_or_path)
    )
    provider = (
        provider_or_path
        if isinstance(provider_or_path, ProviderSpec)
        else ProviderSpec.model_validate_json(Path(provider_or_path).read_text(encoding="utf-8"))
    )
    adapter = (
        adapter_or_path
        if isinstance(adapter_or_path, ProviderAdapterContract)
        else ProviderAdapterContract.model_validate_json(
            Path(adapter_or_path).resolve().read_text(encoding="utf-8")
        )
    )
    attestation = (
        contamination_attestation_or_path
        if isinstance(contamination_attestation_or_path, ContaminationAttestation)
        else ContaminationAttestation.model_validate_json(
            Path(contamination_attestation_or_path).resolve().read_text(encoding="utf-8")
        )
    )
    # Deliberately verify and open only the provider-readable release bundle.
    release_verification = verify_release_bundle(benchmark_root)
    benchmark = _role_root(benchmark_root, "release")
    release = _read_json(benchmark / "benchmark_release.json")
    build_manifest = _read_json(benchmark / "build_manifest.json")
    contamination = _read_json(benchmark / "contamination_receipt.json")
    release_hash = release_verification["benchmark_release_sha256"]
    if (
        release.get("schema_version") != RELEASE_VERSION
        or release.get("gold_included") is not False
    ):
        raise ValueError("provider run requires a non-gold benchmark release")
    if build_manifest.get("benchmark_release_sha256") != _stable_hash(release):
        raise ValueError("benchmark release differs from its build receipt")
    if contamination.get("status") != "unassessed_until_provider_run_freeze":
        raise ValueError("contamination receipt state differs")
    if (
        attestation.provider_id != provider.provider_id
        or attestation.model_id != provider.model_id
        or attestation.model_version != provider.model_version
        or attestation.training_data_cutoff != provider.training_data_cutoff
    ):
        raise ValueError("contamination attestation does not bind the frozen provider")
    contamination_verification = _verify_file_assets(
        [attestation.evidence_receipt],
        contamination_asset_root,
        label="contamination attestation evidence",
    )
    arms = {item.id: item for item in contract.arms}
    requests: list[dict[str, Any]] = []
    for arm_id in ARM_ORDER:
        arm = arms[arm_id]
        for task in release["tasks"]:
            inline_evidence = (
                task["inline_evidence"]
                if arm.inline_context == "task_prompt_and_frozen_text"
                else []
            )
            requests.append(
                {
                    "request_id": f"{arm_id}:{task['task_id']}",
                    "task_id": task["task_id"],
                    "arm_id": arm_id,
                    "prompt": task["prompt"],
                    "inline_evidence": inline_evidence,
                    "available_tools": arm.allowed_tools,
                    "operator_schemas": release["operator_schemas"],
                    "locked_predictor": (
                        task["locked_predictor"]
                        if "call_locked_predictor" in arm.allowed_tools
                        else None
                    ),
                    "response_schema": {
                        "program": "typed ProgramStep[]",
                        "trace": "typed TraceStep[]",
                        "final": "typed FinalAnswer",
                    },
                }
            )
    # Bind validated canonical objects so harmless source-file whitespace does
    # not create an unverifiable run contract.
    provider_hash = _stable_hash(provider.model_dump(mode="json"))
    adapter_hash = _stable_hash(adapter.model_dump(mode="json"))
    attestation_hash = _stable_hash(attestation.model_dump(mode="json"))
    contract_hash = (
        _sha256(Path(contract_or_path).resolve())
        if isinstance(contract_or_path, (str, Path))
        else _stable_hash(contract.model_dump(mode="json"))
    )
    if build_manifest.get("contract_sha256") != contract_hash:
        raise ValueError("provider run contract differs from the benchmark build")
    request_jsonl = _jsonl_bytes(requests)
    run_contract = {
        "schema_version": RUN_VERSION,
        "analysis_id": contract.analysis_id,
        "contract_sha256": contract_hash,
        "benchmark_release_sha256": release_hash,
        "build_manifest_sha256": _sha256(benchmark / "build_manifest.json"),
        "contamination_receipt_sha256": _sha256(
            benchmark / "contamination_receipt.json"
        ),
        "provider_spec_sha256": provider_hash,
        "adapter_contract_sha256": adapter_hash,
        "contamination_attestation_sha256": attestation_hash,
        "adapter_requests_sha256": hashlib.sha256(request_jsonl).hexdigest(),
        "provider": provider.model_dump(mode="json"),
        "adapter": adapter.model_dump(mode="json"),
        "contamination_attestation": attestation.model_dump(mode="json"),
        "contamination_evidence_verification": contamination_verification,
        "task_ids": [item["task_id"] for item in release["tasks"]],
        "arm_ids": ARM_ORDER,
        "request_ids": [item["request_id"] for item in requests],
        "responses_required": len(requests),
        "gold_accessed": False,
        "authority_file_opened": False,
        "external_model_called": False,
        "confirmatory_changes_permitted": False,
    }
    payloads = {
        "adapter_contract.json": adapter.model_dump(mode="json"),
        "adapter_requests.jsonl": request_jsonl,
        "contamination_attestation.json": attestation.model_dump(mode="json"),
        "contamination_evidence_verification.json": contamination_verification,
        "provider_requests.json": {
            "schema_version": RUN_VERSION,
            "requests": requests,
        },
        "run_contract.json": run_contract,
        "run_manifest.json": {
            "schema_version": RUN_VERSION,
            "provider_interface": "model_provider_neutral_external_generation",
            "adapter_transport": adapter.transport,
            "external_model_called": False,
            "requests": len(requests),
            "gold_accessed": False,
        },
    }
    _write_bundle(Path(output_root).resolve(), payloads, RUN_FILES)
    return run_contract


def verify_provider_run(root: str | Path) -> dict[str, Any]:
    path = Path(root).resolve()
    _verify_bundle(path, RUN_FILES)
    run = _read_json(path / "run_contract.json")
    provider = ProviderSpec.model_validate(run.get("provider"))
    adapter = ProviderAdapterContract.model_validate(
        _read_json(path / "adapter_contract.json")
    )
    attestation = ContaminationAttestation.model_validate(
        _read_json(path / "contamination_attestation.json")
    )
    rows = _read_jsonl(path / "adapter_requests.jsonl")
    request_values = [value for _, value in rows]
    provider_requests = _read_json(path / "provider_requests.json").get("requests")
    evidence_verification = _read_json(
        path / "contamination_evidence_verification.json"
    )
    if (
        run.get("schema_version") != RUN_VERSION
        or run.get("gold_accessed") is not False
        or run.get("authority_file_opened") is not False
        or run.get("confirmatory_changes_permitted") is not False
        or run.get("provider_spec_sha256")
        != _stable_hash(provider.model_dump(mode="json"))
        or run.get("adapter_contract_sha256") != _stable_hash(adapter.model_dump(mode="json"))
        or run.get("contamination_attestation_sha256")
        != _stable_hash(attestation.model_dump(mode="json"))
        or run.get("adapter") != adapter.model_dump(mode="json")
        or run.get("contamination_attestation")
        != attestation.model_dump(mode="json")
        or run.get("contamination_evidence_verification") != evidence_verification
        or evidence_verification.get("verified") is not True
        or evidence_verification.get("assets")
        != [attestation.evidence_receipt.model_dump(mode="json")]
        or run.get("adapter_requests_sha256")
        != _sha256(path / "adapter_requests.jsonl")
        or any(raw + b"\n" != _canonical_bytes(value) for raw, value in rows)
        or request_values != provider_requests
        or [item.get("request_id") for item in request_values] != run.get("request_ids")
    ):
        raise ValueError("provider run freeze differs")
    return {
        "schema_version": VERIFY_VERSION,
        "verified": True,
        "run_contract_sha256": _sha256(path / "run_contract.json"),
    }


def _validate_adapter_responses(
    run: Mapping[str, Any],
    rows: Sequence[tuple[bytes, dict[str, Any]]],
) -> list[AdapterResponseEnvelope]:
    if any(raw + b"\n" != _canonical_bytes(value) for raw, value in rows):
        raise ValueError("adapter responses are not canonical deterministic JSONL")
    envelopes = [AdapterResponseEnvelope.model_validate(value) for _, value in rows]
    if [item.request_id for item in envelopes] != run["request_ids"]:
        raise ValueError("adapter responses do not cover the frozen requests in order")
    decoding_hash = _stable_hash(run["provider"]["decoding"])
    for envelope in envelopes:
        if (
            envelope.provider_id != run["provider"]["provider_id"]
            or envelope.model_id != run["provider"]["model_id"]
            or envelope.model_version != run["provider"]["model_version"]
            or envelope.decoding_sha256 != decoding_hash
        ):
            raise ValueError("adapter response identity differs from the frozen provider")
    return envelopes


def _response_receipt_payload(
    run_path: Path,
    run_contract_sha256: str,
    rows: Sequence[tuple[bytes, dict[str, Any]]],
    envelopes: Sequence[AdapterResponseEnvelope],
) -> dict[str, Any]:
    request_rows = _read_jsonl(run_path / "adapter_requests.jsonl")
    if len(request_rows) != len(envelopes):
        raise ValueError("request and response row counts differ")
    receipts = []
    for (request_raw, request), (response_raw, _), envelope in zip(
        request_rows, rows, envelopes, strict=True
    ):
        if request.get("request_id") != envelope.request_id:
            raise ValueError("request and response receipt identities differ")
        receipts.append(
            {
                "request_id": envelope.request_id,
                "request_sha256": hashlib.sha256(request_raw + b"\n").hexdigest(),
                "response_envelope_sha256": hashlib.sha256(
                    response_raw + b"\n"
                ).hexdigest(),
                "typed_response_sha256": _stable_hash(
                    envelope.response.model_dump(mode="json")
                ),
            }
        )
    return {
        "schema_version": RESPONSE_BUNDLE_VERSION,
        "run_contract_sha256": run_contract_sha256,
        "adapter_requests_sha256": _sha256(run_path / "adapter_requests.jsonl"),
        "responses": receipts,
    }


def freeze_response_bundle(
    run_root: str | Path,
    adapter_responses_jsonl: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    run_path = Path(run_root).resolve()
    run_verification = verify_provider_run(run_path)
    run = _read_json(run_path / "run_contract.json")
    rows = _read_jsonl(Path(adapter_responses_jsonl).resolve())
    envelopes = _validate_adapter_responses(run, rows)
    receipts = _response_receipt_payload(
        run_path,
        run_verification["run_contract_sha256"],
        rows,
        envelopes,
    )
    submission = Submission(
        schema_version=SUBMISSION_VERSION,
        run_contract_sha256=run_verification["run_contract_sha256"],
        benchmark_release_sha256=run["benchmark_release_sha256"],
        provider_spec_sha256=run["provider_spec_sha256"],
        adapter_contract_sha256=run["adapter_contract_sha256"],
        contamination_attestation_sha256=run[
            "contamination_attestation_sha256"
        ],
        response_receipts_sha256=_stable_hash(receipts),
        model_or_decoding_changed_after_freeze=False,
        thresholds_or_endpoints_changed_after_freeze=False,
        gold_accessed_before_response_freeze=False,
        responses=[item.response for item in envelopes],
    )
    response_bytes = _jsonl_bytes([value for _, value in rows])
    manifest = {
        "schema_version": RESPONSE_BUNDLE_VERSION,
        "run_contract_sha256": run_verification["run_contract_sha256"],
        "adapter_responses_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "response_receipts_sha256": _stable_hash(receipts),
        "submission_sha256": _stable_hash(submission.model_dump(mode="json")),
        "responses": len(envelopes),
        "response_frozen_before_authority_access": True,
        "authority_opened": False,
        "external_model_called_by_harness": False,
        "biological_claims_permitted": False,
    }
    payloads = {
        "adapter_responses.jsonl": response_bytes,
        "response_manifest.json": manifest,
        "response_receipts.json": receipts,
        "submission.json": submission.model_dump(mode="json"),
    }
    _write_bundle(Path(output_root).resolve(), payloads, RESPONSE_FILES)
    return manifest


def verify_response_bundle(
    root: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    path = Path(root).resolve()
    run_path = Path(run_root).resolve()
    run_verification = verify_provider_run(run_path)
    run = _read_json(run_path / "run_contract.json")
    _verify_bundle(path, RESPONSE_FILES)
    rows = _read_jsonl(path / "adapter_responses.jsonl")
    envelopes = _validate_adapter_responses(run, rows)
    expected_receipts = _response_receipt_payload(
        run_path,
        run_verification["run_contract_sha256"],
        rows,
        envelopes,
    )
    receipts = _read_json(path / "response_receipts.json")
    submission = Submission.model_validate(_read_json(path / "submission.json"))
    manifest = _read_json(path / "response_manifest.json")
    if receipts != expected_receipts:
        raise ValueError("response receipts differ from frozen request/response bytes")
    expected_responses = [item.response.model_dump(mode="json") for item in envelopes]
    if [item.model_dump(mode="json") for item in submission.responses] != expected_responses:
        raise ValueError("typed submission differs from adapter response envelopes")
    if (
        submission.run_contract_sha256 != run_verification["run_contract_sha256"]
        or submission.benchmark_release_sha256 != run["benchmark_release_sha256"]
        or submission.provider_spec_sha256 != run["provider_spec_sha256"]
        or submission.adapter_contract_sha256 != run["adapter_contract_sha256"]
        or submission.contamination_attestation_sha256
        != run["contamination_attestation_sha256"]
        or submission.response_receipts_sha256 != _stable_hash(receipts)
        or manifest.get("run_contract_sha256")
        != run_verification["run_contract_sha256"]
        or manifest.get("adapter_responses_sha256")
        != _sha256(path / "adapter_responses.jsonl")
        or manifest.get("response_receipts_sha256") != _stable_hash(receipts)
        or manifest.get("submission_sha256")
        != _stable_hash(submission.model_dump(mode="json"))
        or manifest.get("responses") != len(envelopes)
        or manifest.get("response_frozen_before_authority_access") is not True
        or manifest.get("authority_opened") is not False
    ):
        raise ValueError("response bundle differs from its frozen bindings")
    return {
        "schema_version": VERIFY_VERSION,
        "verified": True,
        "run_contract_sha256": run_verification["run_contract_sha256"],
        "response_manifest_sha256": _sha256(path / "response_manifest.json"),
        "response_receipts_sha256": _sha256(path / "response_receipts.json"),
        "submission_sha256": _sha256(path / "submission.json"),
        "responses": len(envelopes),
    }


def _required_arguments(operator: str) -> list[str]:
    return list(_operator_schemas()[operator]["required_arguments"])


def _authority_task(authority: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    matches = [item for item in authority["tasks"] if item["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"authority task identity differs: {task_id}")
    return matches[0]


def validate_program(
    steps: Sequence[ProgramStep],
    *,
    allowed_tools: Sequence[str],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    ids = [item.step_id for item in steps]
    if not steps or len(ids) != len(set(ids)):
        errors.append("step_ids_missing_or_duplicate")
    seen: set[str] = set()
    emitted = 0
    for index, step in enumerate(steps):
        if step.operator in TOOL_OPERATORS and step.operator not in allowed_tools:
            errors.append(f"tool_not_allowed:{step.operator}")
        if len(step.depends_on) != len(set(step.depends_on)):
            errors.append(f"duplicate_dependencies:{step.step_id}")
        if any(dependency not in seen for dependency in step.depends_on):
            errors.append(f"dependency_not_prior:{step.step_id}")
        required = _required_arguments(step.operator)
        missing = [name for name in required if name not in step.arguments]
        if missing:
            errors.append(f"required_arguments_missing:{step.step_id}:{','.join(missing)}")
        unexpected = sorted(set(step.arguments) - set(required))
        if unexpected:
            errors.append(
                f"unexpected_arguments:{step.step_id}:{','.join(unexpected)}"
            )
        if step.operator in {"retrieve_text", "retrieve_graph"}:
            if not isinstance(step.arguments.get("query"), str) or not step.arguments.get(
                "query", ""
            ).strip():
                errors.append(f"invalid_query:{step.step_id}")
        if step.operator == "check_identifiability":
            question_type = step.arguments.get("question_type")
            if not isinstance(question_type, str) or not question_type.strip():
                errors.append(f"invalid_question_type:{step.step_id}")
        if step.operator == "call_locked_predictor":
            predictor = task.get("predictor")
            if predictor is None:
                errors.append("locked_predictor_absent")
            else:
                for name in ("predictor_id", "version", "input_sha256"):
                    if step.arguments.get(name) != predictor.get(name):
                        errors.append(f"locked_predictor_binding_differs:{name}")
        if step.operator == "synthesize_evidence":
            evidence_ids = step.arguments.get("evidence_ids")
            if not isinstance(evidence_ids, list) or any(
                not isinstance(item, str) for item in evidence_ids
            ):
                errors.append(f"invalid_evidence_ids:{step.step_id}")
        if step.operator == "emit_prediction":
            prediction = step.arguments.get("prediction")
            if not isinstance(prediction, (float, int, str)) or isinstance(
                prediction, bool
            ):
                errors.append(f"invalid_prediction:{step.step_id}")
            elif isinstance(prediction, (float, int)) and not math.isfinite(
                float(prediction)
            ):
                errors.append(f"nonfinite_prediction:{step.step_id}")
        if step.operator == "emit_abstention":
            reason = step.arguments.get("reason_code")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"invalid_abstention_reason:{step.step_id}")
        if step.operator in {"emit_prediction", "emit_abstention"}:
            emitted += 1
            if index != len(steps) - 1:
                errors.append("terminal_emit_must_be_last")
        seen.add(step.step_id)
    if emitted != 1:
        errors.append("exactly_one_terminal_emit_required")
    expected_available = [
        item
        for item in task["gold"]["required_operators"]
        if item not in TOOL_OPERATORS or item in allowed_tools
    ]
    observed_operators = {item.operator for item in steps}
    required_hits = sum(item in observed_operators for item in expected_available)
    return {
        "valid": not errors,
        "errors": errors,
        "required_operators_available": expected_available,
        "required_operator_recall": (
            1.0 if not expected_available else required_hits / len(expected_available)
        ),
    }


def execute_program(
    steps: Sequence[ProgramStep],
    *,
    allowed_tools: Sequence[str],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_program(steps, allowed_tools=allowed_tools, task=task)
    if not validation["valid"]:
        return {"validation": validation, "trace": [], "outputs": {}}
    evidence_by_channel: dict[str, list[str]] = defaultdict(list)
    for item in task["evidence"]:
        evidence_by_channel[item["channel"]].append(item["evidence_id"])
    outputs: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    for step in steps:
        if step.operator == "retrieve_text":
            output = {"evidence_ids": sorted(evidence_by_channel["text"])}
        elif step.operator == "retrieve_graph":
            output = {"evidence_ids": sorted(evidence_by_channel["graph"])}
        elif step.operator == "check_identifiability":
            output = {
                "identifiable": bool(task["identifiable"]),
                "reason_code": task["identifiability_reason_code"],
            }
        elif step.operator == "call_locked_predictor":
            predictor = task["predictor"]
            output = {
                "predictor_id": predictor["predictor_id"],
                "version": predictor["version"],
                "input_sha256": predictor["input_sha256"],
                "output": predictor["output"],
                "qualification_receipt_sha256": predictor[
                    "qualification_receipt_sha256"
                ],
            }
        elif step.operator == "synthesize_evidence":
            evidence_ids = step.arguments["evidence_ids"]
            if not isinstance(evidence_ids, list):
                raise ValueError("synthesize_evidence requires an evidence-ID list")
            known = {item["evidence_id"] for item in task["evidence"]}
            output = {
                "evidence_ids": sorted(set(str(item) for item in evidence_ids) & known),
                "unknown_evidence_ids": sorted(set(str(item) for item in evidence_ids) - known),
            }
        elif step.operator == "emit_prediction":
            output = {
                "disposition": "predict",
                "prediction": step.arguments["prediction"],
            }
        elif step.operator == "emit_abstention":
            output = {
                "disposition": "abstain",
                "reason_code": step.arguments["reason_code"],
            }
        else:  # pragma: no cover
            raise AssertionError(step.operator)
        outputs[step.step_id] = output
        trace.append(
            {
                "step_id": step.step_id,
                "status": "success",
                "output_sha256": _stable_hash(output),
            }
        )
    return {"validation": validation, "trace": trace, "outputs": outputs}


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _score_prediction(gold: Mapping[str, Any], final: FinalAnswer) -> dict[str, Any]:
    base = {
        "submitted_prediction": final.disposition == "predict",
        "confidence": final.confidence,
        "interval_reported": final.interval_lower is not None,
        "interval_contains_target": None,
    }
    if gold["disposition"] != "predict":
        return {
            **base,
            "eligible": False,
            "correct": None,
            "absolute_error": None,
        }
    if final.disposition != "predict":
        return {
            **base,
            "eligible": True,
            "correct": False,
            "absolute_error": None,
        }
    if gold["prediction_type"] == "numeric":
        try:
            error = abs(float(final.prediction) - float(gold["target"]))
        except (TypeError, ValueError):
            return {
                **base,
                "eligible": True,
                "correct": False,
                "absolute_error": None,
            }
        interval_contains = (
            final.interval_lower <= float(gold["target"]) <= final.interval_upper
            if final.interval_lower is not None and final.interval_upper is not None
            else None
        )
        return {
            **base,
            "eligible": True,
            "correct": error <= float(gold["numeric_tolerance"]),
            "absolute_error": error,
            "interval_contains_target": interval_contains,
        }
    return {
        **base,
        "eligible": True,
        "correct": str(final.prediction) == str(gold["target"]),
        "absolute_error": None,
    }


def _score_evidence(
    task: Mapping[str, Any],
    final: FinalAnswer,
    accessible_evidence_ids: set[str],
) -> dict[str, Any]:
    gold = set(task["gold"]["evidence_ids"])
    cited = set(final.citations)
    known = {item["evidence_id"] for item in task["evidence"]}
    supported_claims = [
        bool(set(item.evidence_ids) & gold)
        and set(item.evidence_ids).issubset(accessible_evidence_ids)
        for item in final.claims
    ]
    return {
        "citation_precision": (
            len(cited & gold & accessible_evidence_ids) / len(cited)
            if cited
            else 0.0
        ),
        "citation_recall": len(cited & gold) / len(gold) if gold else 1.0,
        "unknown_citations": len(cited - known),
        "inaccessible_citations": len(cited - accessible_evidence_ids),
        "claim_support_rate": _mean([float(item) for item in supported_claims]),
    }


def _score_abstention(task: Mapping[str, Any], final: FinalAnswer) -> dict[str, Any]:
    expected = task["gold"]["disposition"] == "abstain"
    observed = final.disposition == "abstain"
    reason_correct = (
        final.abstention_reason_code == task["gold"]["abstention_reason_code"]
        if expected and observed
        else None
    )
    return {
        "expected_abstention": expected,
        "observed_abstention": observed,
        "correct": expected == observed and (reason_correct is not False),
        "unsafe_answer": expected and not observed,
        "unnecessary_abstention": not expected and observed,
        "reason_correct": reason_correct,
    }


def _trace_score(
    expected: Sequence[Mapping[str, Any]],
    observed: Sequence[TraceStep],
) -> dict[str, Any]:
    expected_by_id = {item["step_id"]: item for item in expected}
    observed_by_id = {item.step_id: item for item in observed}
    matches = sum(
        step_id in observed_by_id
        and observed_by_id[step_id].status == item["status"]
        and observed_by_id[step_id].output_sha256 == item["output_sha256"]
        for step_id, item in expected_by_id.items()
    )
    denominator = max(len(expected_by_id), len(observed_by_id))
    receipt_rate = 1.0 if denominator == 0 else matches / denominator
    return {
        "exact_trace_match": (
            len(expected_by_id) == len(observed_by_id)
            and matches == denominator
        ),
        "step_receipt_match_rate": receipt_rate,
        "expected_steps": len(expected_by_id),
        "observed_steps": len(observed_by_id),
    }


def _terminal_matches_final(
    execution: Mapping[str, Any], final: FinalAnswer
) -> bool:
    if not execution["validation"]["valid"]:
        return False
    terminal_outputs = [
        output
        for output in execution["outputs"].values()
        if output.get("disposition") in {"predict", "abstain"}
    ]
    if len(terminal_outputs) != 1:
        return False
    terminal = terminal_outputs[0]
    if terminal["disposition"] != final.disposition:
        return False
    if final.disposition == "predict":
        return terminal.get("prediction") == final.prediction
    return terminal.get("reason_code") == final.abstention_reason_code


def score_submission(
    contract_or_path: BenchmarkContract | str | Path,
    benchmark_root: str | Path,
    run_root: str | Path,
    response_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    contract = (
        contract_or_path
        if isinstance(contract_or_path, BenchmarkContract)
        else load_contract(contract_or_path)
    )
    benchmark = Path(benchmark_root).resolve()
    # Fail closed on every provider-readable and response-freeze binding before
    # opening either private scoring authority.
    release_verification = verify_release_bundle(benchmark)
    build_manifest = _read_json(benchmark / "release" / "build_manifest.json")
    contract_hash = (
        _sha256(Path(contract_or_path).resolve())
        if isinstance(contract_or_path, (str, Path))
        else _stable_hash(contract.model_dump(mode="json"))
    )
    run_path = Path(run_root).resolve()
    run_verification = verify_provider_run(run_path)
    run = _read_json(run_path / "run_contract.json")
    response_path = Path(response_root).resolve()
    response_verification = verify_response_bundle(response_path, run_path)
    submission = Submission.model_validate(
        _read_json(response_path / "submission.json")
    )
    if run["benchmark_release_sha256"] != release_verification[
        "benchmark_release_sha256"
    ]:
        raise ValueError("provider run binds a different benchmark release")
    if (
        run.get("contract_sha256") != contract_hash
        or build_manifest.get("contract_sha256") != contract_hash
    ):
        raise ValueError("scoring contract differs from the frozen build and run")
    if submission.run_contract_sha256 != run_verification["run_contract_sha256"]:
        raise ValueError("submission binds a different provider run")
    if submission.benchmark_release_sha256 != run["benchmark_release_sha256"]:
        raise ValueError("submission binds a different benchmark release")
    if submission.provider_spec_sha256 != run["provider_spec_sha256"]:
        raise ValueError("submission binds a different provider specification")
    expected_requests = set(run["request_ids"])
    observed_requests = {item.request_id for item in submission.responses}
    if observed_requests != expected_requests:
        raise ValueError("submission does not cover the exact frozen request set")
    scorer_verification = verify_scorer_bundle(benchmark)
    gold_verification = verify_gold_bundle(benchmark)
    scorer_authority = _read_json(
        benchmark / "scorer" / "benchmark_scorer_authority.json"
    )
    gold_authority = _read_json(benchmark / "gold" / "benchmark_gold.json")
    scorer_manifest = _read_json(benchmark / "scorer" / "scorer_manifest.json")
    gold_manifest = _read_json(benchmark / "gold" / "gold_manifest.json")
    custody_hashes = build_manifest.get("custody_attestation_sha256_by_role", {})
    if (
        build_manifest.get("benchmark_scorer_authority_sha256")
        != scorer_verification["benchmark_scorer_authority_sha256"]
        or build_manifest.get("benchmark_gold_sha256")
        != gold_verification["benchmark_gold_sha256"]
        or scorer_authority.get("source_snapshot_id")
        != gold_authority.get("source_snapshot_id")
        or scorer_manifest.get("benchmark_release_sha256")
        != release_verification["benchmark_release_sha256"]
        or gold_manifest.get("benchmark_release_sha256")
        != release_verification["benchmark_release_sha256"]
        or custody_hashes.get("scorer")
        != _stable_hash(
            _read_json(benchmark / "scorer" / "custody_attestation.json")
        )
        or custody_hashes.get("gold")
        != _stable_hash(_read_json(benchmark / "gold" / "custody_attestation.json"))
    ):
        raise ValueError("private scoring authorities differ from the frozen build")
    gold_by_task = {
        item["task_id"]: item["gold"] for item in gold_authority["tasks"]
    }
    scorer_ids = [item["task_id"] for item in scorer_authority["tasks"]]
    if set(scorer_ids) != set(gold_by_task) or len(scorer_ids) != len(set(scorer_ids)):
        raise ValueError("scorer and gold task identities differ")
    authority = {
        "tasks": [
            {**item, "gold": gold_by_task[item["task_id"]]}
            for item in scorer_authority["tasks"]
        ]
    }
    release = _read_json(benchmark / "release" / "benchmark_release.json")
    arms = {item.id: item for item in contract.arms}
    release_tasks = {item["task_id"]: item for item in release["tasks"]}
    task_scores: list[dict[str, Any]] = []
    execution_receipts: list[dict[str, Any]] = []
    for response in sorted(submission.responses, key=lambda item: item.request_id):
        if response.arm_id not in arms or response.task_id not in release_tasks:
            raise ValueError("response arm or task is outside the frozen benchmark")
        if response.request_id != f"{response.arm_id}:{response.task_id}":
            raise ValueError("response request ID does not bind its arm and task")
        authority_task = _authority_task(authority, response.task_id)
        execution = execute_program(
            response.program,
            allowed_tools=arms[response.arm_id].allowed_tools,
            task=authority_task,
        )
        terminal_consistent = _terminal_matches_final(execution, response.final)
        program_validity = {
            **execution["validation"],
            "terminal_final_consistent": terminal_consistent,
        }
        if not terminal_consistent:
            program_validity["valid"] = False
            program_validity["errors"] = [
                *program_validity["errors"],
                "terminal_emit_and_final_answer_differ",
            ]
        accessible_evidence_ids = (
            {
                item["evidence_id"]
                for item in authority_task["evidence"]
                if item["channel"] == "inline"
            }
            if arms[response.arm_id].inline_context
            == "task_prompt_and_frozen_text"
            else set()
        )
        for step in response.program:
            if step.operator not in {"retrieve_text", "retrieve_graph"}:
                continue
            output = execution["outputs"].get(step.step_id, {})
            accessible_evidence_ids.update(output.get("evidence_ids", []))
        prediction = _score_prediction(authority_task["gold"], response.final)
        evidence = _score_evidence(
            authority_task,
            response.final,
            accessible_evidence_ids,
        )
        abstention = _score_abstention(authority_task, response.final)
        trace = _trace_score(execution["trace"], response.trace)
        substantive_scores_eligible = bool(
            program_validity["valid"] and trace["exact_trace_match"]
        )
        if not substantive_scores_eligible:
            if prediction["eligible"]:
                prediction["correct"] = False
                prediction["absolute_error"] = None
            evidence = {
                **evidence,
                "citation_precision": 0.0,
                "citation_recall": 0.0,
                "claim_support_rate": 0.0,
            }
            abstention = {**abstention, "correct": False}
        task_scores.append(
            {
                "request_id": response.request_id,
                "task_id": response.task_id,
                "arm_id": response.arm_id,
                "prediction": prediction,
                "program_validity": program_validity,
                "evidence_fidelity": evidence,
                "abstention": abstention,
                "execution_trace": trace,
                "substantive_scores_eligible": substantive_scores_eligible,
            }
        )
        execution_receipts.append(
            {
                "request_id": response.request_id,
                "program_sha256": _stable_hash(
                    [item.model_dump(mode="json") for item in response.program]
                ),
                "replayed_trace": execution["trace"],
                "locked_predictor_calls": sum(
                    item.operator == "call_locked_predictor" for item in response.program
                ),
            }
        )

    dimension_scores: dict[str, Any] = {}
    for arm_id in ARM_ORDER:
        rows = [item for item in task_scores if item["arm_id"] == arm_id]
        eligible_predictions = [
            item["prediction"]
            for item in rows
            if item["prediction"]["eligible"]
        ]
        numeric_errors = [
            float(item["absolute_error"])
            for item in eligible_predictions
            if item["absolute_error"] is not None
        ]
        dimension_scores[arm_id] = {
            "prediction": {
                "eligible_tasks": len(eligible_predictions),
                "coverage": _mean(
                    [
                        float(item["submitted_prediction"])
                        for item in eligible_predictions
                    ]
                ),
                "accuracy_within_frozen_tolerance": _mean(
                    [float(bool(item["correct"])) for item in eligible_predictions]
                ),
                "numeric_mae": _mean(numeric_errors),
                "mean_confidence": _mean(
                    [float(item["confidence"]) for item in eligible_predictions]
                ),
                "interval_reporting_rate": _mean(
                    [
                        float(item["interval_reported"])
                        for item in eligible_predictions
                    ]
                ),
                "interval_coverage": _mean(
                    [
                        float(item["interval_contains_target"])
                        for item in eligible_predictions
                        if item["interval_contains_target"] is not None
                    ]
                ),
            },
            "program_validity": {
                "valid_rate": _mean(
                    [float(item["program_validity"]["valid"]) for item in rows]
                ),
                "required_operator_recall": _mean(
                    [float(item["program_validity"]["required_operator_recall"]) for item in rows]
                ),
            },
            "evidence_fidelity": {
                "citation_precision": _mean(
                    [float(item["evidence_fidelity"]["citation_precision"]) for item in rows]
                ),
                "citation_recall": _mean(
                    [float(item["evidence_fidelity"]["citation_recall"]) for item in rows]
                ),
                "claim_support_rate": _mean(
                    [
                        float(item["evidence_fidelity"]["claim_support_rate"])
                        for item in rows
                        if item["evidence_fidelity"]["claim_support_rate"] is not None
                    ]
                ),
                "mean_inaccessible_citations": _mean(
                    [
                        float(item["evidence_fidelity"]["inaccessible_citations"])
                        for item in rows
                    ]
                ),
            },
            "abstention": {
                "accuracy": _mean([float(item["abstention"]["correct"]) for item in rows]),
                "unsafe_answer_rate": _mean(
                    [float(item["abstention"]["unsafe_answer"]) for item in rows]
                ),
                "unnecessary_abstention_rate": _mean(
                    [float(item["abstention"]["unnecessary_abstention"]) for item in rows]
                ),
            },
            "execution_trace": {
                "exact_match_rate": _mean(
                    [float(item["execution_trace"]["exact_trace_match"]) for item in rows]
                ),
                "step_receipt_match_rate": _mean(
                    [float(item["execution_trace"]["step_receipt_match_rate"]) for item in rows]
                ),
            },
        }
    contamination_assessment = {
        "schema_version": SCORE_VERSION,
        "benchmark_visibility": "private_held_out",
        "provider_training_data_cutoff": run["contamination_attestation"][
            "training_data_cutoff"
        ],
        "provider_prior_exposure_attested": True,
        "external_search_used": False,
        "contamination_attestation_sha256": run[
            "contamination_attestation_sha256"
        ],
        "contamination_evidence_verified": run[
            "contamination_evidence_verification"
        ]["verified"],
        "assessment": "attested_unexposed_not_independently_proven",
        "contamination_adjustment_applied_to_scores": False,
    }
    score_manifest = {
        "schema_version": SCORE_VERSION,
        "analysis_id": contract.analysis_id,
        "benchmark_release_sha256": run["benchmark_release_sha256"],
        "benchmark_scorer_authority_sha256": scorer_verification[
            "benchmark_scorer_authority_sha256"
        ],
        "benchmark_gold_sha256": gold_verification["benchmark_gold_sha256"],
        "run_contract_sha256": run_verification["run_contract_sha256"],
        "response_manifest_sha256": response_verification[
            "response_manifest_sha256"
        ],
        "response_receipts_sha256": response_verification[
            "response_receipts_sha256"
        ],
        "submission_sha256": response_verification["submission_sha256"],
        "responses": len(task_scores),
        "arms": ARM_ORDER,
        "score_dimensions": SCORE_DIMENSIONS,
        "composite_score_reported": False,
        "models_thresholds_endpoints_changed": False,
        "response_freeze_verified_before_authority_open": True,
        "external_model_called_by_scoring_harness": False,
        "biological_claims_permitted": False,
    }
    payloads = {
        "contamination_assessment.json": contamination_assessment,
        "dimension_scores.json": {
            "schema_version": SCORE_VERSION,
            "dimensions_are_separate": True,
            "composite_score": None,
            "arms": dimension_scores,
        },
        "execution_receipts.json": {
            "schema_version": SCORE_VERSION,
            "receipts": execution_receipts,
        },
        "score_manifest.json": score_manifest,
        "task_scores.json": {
            "schema_version": SCORE_VERSION,
            "tasks": task_scores,
        },
    }
    _write_bundle(Path(output_root).resolve(), payloads, SCORE_FILES)
    return score_manifest


def verify_scores(root: str | Path) -> dict[str, Any]:
    path = Path(root).resolve()
    _verify_bundle(path, SCORE_FILES)
    manifest = _read_json(path / "score_manifest.json")
    dimensions = _read_json(path / "dimension_scores.json")
    if (
        manifest.get("schema_version") != SCORE_VERSION
        or manifest.get("composite_score_reported") is not False
        or manifest.get("models_thresholds_endpoints_changed") is not False
        or dimensions.get("dimensions_are_separate") is not True
        or dimensions.get("composite_score") is not None
    ):
        raise ValueError("score bundle collapses dimensions or violates the freeze")
    return {
        "schema_version": VERIFY_VERSION,
        "verified": True,
        "score_manifest_sha256": _sha256(path / "score_manifest.json"),
    }


def _emit(payload: Any, output: str | Path | None = None) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(content)
    else:
        _write_bytes_atomic(Path(output).resolve(), content.encode("utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wormctx.poc.grounded_reasoner_benchmark",
        description="Build and score a provider-neutral grounded-reasoning benchmark",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--contract", required=True)
    build.add_argument("--source-tasks", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--custody-plan", required=True)
    build.add_argument("--custody-asset-root", required=True)
    build.add_argument("--comparison-plan", required=True)
    build.add_argument("--comparison-asset-root", required=True)
    build.add_argument("--predictor-receipt-root", required=True)
    secret = build.add_mutually_exclusive_group(required=True)
    secret.add_argument("--anonymization-secret")
    secret.add_argument("--anonymization-secret-file")
    freeze = commands.add_parser("freeze-provider-run")
    freeze.add_argument("--contract", required=True)
    freeze.add_argument("--benchmark-root", required=True)
    freeze.add_argument("--provider-spec", required=True)
    freeze.add_argument("--adapter-contract", required=True)
    freeze.add_argument("--contamination-attestation", required=True)
    freeze.add_argument("--contamination-asset-root", required=True)
    freeze.add_argument("--output-root", required=True)
    responses = commands.add_parser("freeze-responses")
    responses.add_argument("--run-root", required=True)
    responses.add_argument("--adapter-responses-jsonl", required=True)
    responses.add_argument("--output-root", required=True)
    score = commands.add_parser("score")
    score.add_argument("--contract", required=True)
    score.add_argument("--benchmark-root", required=True)
    score.add_argument("--run-root", required=True)
    score.add_argument("--response-root", required=True)
    score.add_argument("--output-root", required=True)
    for name, option in (
        ("verify-benchmark", "--benchmark-root"),
        ("verify-release", "--release-root"),
        ("verify-curator", "--curator-root"),
        ("verify-gold", "--gold-root"),
        ("verify-scorer", "--scorer-root"),
        ("verify-provider-run", "--run-root"),
        ("verify-responses", "--response-root"),
        ("verify-scores", "--score-root"),
    ):
        verify = commands.add_parser(name)
        verify.add_argument(option, required=True)
        if name == "verify-responses":
            verify.add_argument("--run-root", required=True)
        verify.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        anonymization_secret = args.anonymization_secret
        if args.anonymization_secret_file is not None:
            anonymization_secret = Path(args.anonymization_secret_file).read_text(
                encoding="utf-8"
            ).rstrip("\r\n")
        _emit(
            build_benchmark(
                args.contract,
                args.source_tasks,
                args.output_root,
                anonymization_secret=anonymization_secret,
                custody_plan_or_path=args.custody_plan,
                custody_asset_root=args.custody_asset_root,
                comparison_plan_or_path=args.comparison_plan,
                comparison_asset_root=args.comparison_asset_root,
                predictor_receipt_root=args.predictor_receipt_root,
            )
        )
    elif args.command == "freeze-provider-run":
        _emit(
            freeze_provider_run(
                args.contract,
                args.benchmark_root,
                args.provider_spec,
                args.adapter_contract,
                args.contamination_attestation,
                args.contamination_asset_root,
                args.output_root,
            )
        )
    elif args.command == "freeze-responses":
        _emit(
            freeze_response_bundle(
                args.run_root,
                args.adapter_responses_jsonl,
                args.output_root,
            )
        )
    elif args.command == "score":
        _emit(
            score_submission(
                args.contract,
                args.benchmark_root,
                args.run_root,
                args.response_root,
                args.output_root,
            )
        )
    elif args.command == "verify-benchmark":
        _emit(verify_benchmark(args.benchmark_root), args.output)
    elif args.command == "verify-provider-run":
        _emit(verify_provider_run(args.run_root), args.output)
    elif args.command == "verify-release":
        _emit(verify_release_bundle(args.release_root), args.output)
    elif args.command == "verify-curator":
        _emit(verify_curator_bundle(args.curator_root), args.output)
    elif args.command == "verify-gold":
        _emit(verify_gold_bundle(args.gold_root), args.output)
    elif args.command == "verify-scorer":
        _emit(verify_scorer_bundle(args.scorer_root), args.output)
    elif args.command == "verify-responses":
        _emit(verify_response_bundle(args.response_root, args.run_root), args.output)
    else:
        _emit(verify_scores(args.score_root), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
