"""Composable build functions used by the CLI and tests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .coverage import coverage_cube, coverage_long
from .adapters.local_jsonl import LocalJsonlAdapter
from .adapters.wormbase_ws298 import WormBaseWs298Adapter
from .graph import (
    build_contextual_graph,
    build_kgx_context_sidecar,
    build_kgx_projection,
)
from .io import (
    load_observations,
    load_query,
    read_json,
    validate_observation_collection,
    write_json,
    write_jsonl,
    write_table,
)
from .manifests import load_snapshot_context_with_binding
from .models import SourceSnapshot
from .report import write_transport_html
from .transport import DIMENSIONS, score_observation


class _StrictReceiptModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NormalizationArtifactBinding(_StrictReceiptModel):
    artifact_id: str = Field(min_length=1)
    filename: str | None = None
    artifact_sha256: str
    expected_sha256: str
    artifact_spec_sha256: str
    byte_size: int = Field(ge=0)
    blob_uri: str
    artifact_receipt_sha256: str

    @field_validator(
        "artifact_sha256",
        "expected_sha256",
        "artifact_spec_sha256",
        "artifact_receipt_sha256",
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        return _require_sha256(value)

    @model_validator(mode="after")
    def consistent_content_identity(self) -> "NormalizationArtifactBinding":
        if self.artifact_sha256 != self.expected_sha256:
            raise ValueError("normalized artifact hash must equal its locked expected hash")
        if self.blob_uri != f"urn:sha256:{self.artifact_sha256}":
            raise ValueError("normalization artifact binding has an invalid blob URI")
        if self.filename is not None and (
            not self.filename or Path(self.filename).name != self.filename
        ):
            raise ValueError("normalization artifact filename must be a safe filename")
        return self


class NormalizationReceipt(_StrictReceiptModel):
    normalization_receipt_schema_version: Literal["1.1", "1.2"]
    created_at: datetime
    provenance_binding: Literal[
        "locked_manifest_and_release_receipts",
        "locked_manifest_all_artifact_receipts_and_primary_record_artifact",
    ]
    source_id: str = Field(min_length=1)
    source_release: str = Field(min_length=1)
    manifest_sha256: str
    release_receipt_sha256: str
    artifact_receipts: list[NormalizationArtifactBinding] = Field(min_length=1)
    adapter: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    schema_sha256: dict[str, str]
    mapping_sha256: dict[str, str]
    code_sha256: str
    output_filename: str
    output_sha256: str
    observation_count: int = Field(ge=0)
    observation_primary_artifact_counts: dict[str, int] | None = None

    @field_validator(
        "manifest_sha256",
        "release_receipt_sha256",
        "code_sha256",
        "output_sha256",
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        return _require_sha256(value)

    @field_validator("schema_sha256", "mapping_sha256")
    @classmethod
    def valid_hash_map(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("resource hash maps cannot be empty")
        if any(Path(name).name != name for name in value):
            raise ValueError("resource hash keys must be safe filenames")
        return {name: _require_sha256(digest) for name, digest in value.items()}

    @field_validator("output_filename")
    @classmethod
    def safe_output_filename(cls, value: str) -> str:
        if not value or Path(value).name != value:
            raise ValueError("output_filename must be a safe filename")
        return value

    @model_validator(mode="after")
    def unique_artifacts(self) -> "NormalizationReceipt":
        artifact_ids = [item.artifact_id for item in self.artifact_receipts]
        if artifact_ids != sorted(artifact_ids) or len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifact receipt bindings must be unique and sorted")
        if self.created_at.tzinfo is None:
            raise ValueError("normalization receipt timestamp must include a timezone")
        if self.normalization_receipt_schema_version == "1.1":
            if self.provenance_binding != "locked_manifest_and_release_receipts":
                raise ValueError("normalization receipt 1.1 has an invalid provenance binding")
            if self.observation_primary_artifact_counts is not None:
                raise ValueError(
                    "normalization receipt 1.1 cannot claim per-record artifact counts"
                )
            return self
        if self.provenance_binding != (
            "locked_manifest_all_artifact_receipts_and_primary_record_artifact"
        ):
            raise ValueError("normalization receipt 1.2 has an invalid provenance binding")
        if len(self.artifact_receipts) < 2:
            raise ValueError("normalization receipt 1.2 requires multiple artifacts")
        filenames = [item.filename for item in self.artifact_receipts]
        if any(filename is None for filename in filenames) or len(filenames) != len(
            set(filenames)
        ):
            raise ValueError(
                "normalization receipt 1.2 requires unique locked artifact filenames"
            )
        counts = self.observation_primary_artifact_counts
        if counts is None:
            raise ValueError(
                "normalization receipt 1.2 requires per-record primary artifact counts"
            )
        if list(counts) != sorted(counts) or set(counts) != set(artifact_ids):
            raise ValueError(
                "primary artifact count keys must be sorted and equal all artifact ids"
            )
        if any(isinstance(value, bool) or value < 0 for value in counts.values()):
            raise ValueError("primary artifact counts must be nonnegative integers")
        if sum(counts.values()) != self.observation_count:
            raise ValueError(
                "primary artifact counts must sum to the normalized observation count"
            )
        return self


class VerifiedNormalizationBuildBinding(_StrictReceiptModel):
    status: Literal["byte_bound_only", "full_provenance_verified"]
    verification_level: Literal["byte_bound_only", "full_provenance_verified"]
    normalization_receipt: str = Field(min_length=1)
    normalization_receipt_sha256: str
    normalization_receipt_schema_version: Literal["1.1", "1.2"]
    provenance_binding: Literal[
        "locked_manifest_and_release_receipts",
        "locked_manifest_all_artifact_receipts_and_primary_record_artifact",
    ]
    input_sha256: str
    observation_count: int = Field(ge=0)
    observation_primary_artifact_counts: dict[str, int] | None = None
    manifest_sha256: str
    release_receipt_sha256: str
    artifact_receipts: list[NormalizationArtifactBinding] = Field(min_length=1)
    schema_sha256: dict[str, str]
    mapping_sha256: dict[str, str]
    code_sha256: str
    adapter: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)

    @field_validator(
        "normalization_receipt_sha256",
        "input_sha256",
        "manifest_sha256",
        "release_receipt_sha256",
        "code_sha256",
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        return _require_sha256(value)

    @field_validator("schema_sha256", "mapping_sha256")
    @classmethod
    def valid_hash_map(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("normalization binding resource hashes cannot be empty")
        return {name: _require_sha256(digest) for name, digest in value.items()}

    @model_validator(mode="after")
    def status_matches_level(self) -> "VerifiedNormalizationBuildBinding":
        if self.status != self.verification_level:
            raise ValueError("normalization binding status and verification level differ")
        if Path(self.normalization_receipt).name != self.normalization_receipt:
            raise ValueError("normalization receipt binding must use a portable filename")
        artifact_ids = [item.artifact_id for item in self.artifact_receipts]
        if self.normalization_receipt_schema_version == "1.1":
            if self.provenance_binding != "locked_manifest_and_release_receipts":
                raise ValueError("normalization binding 1.1 has an invalid provenance mode")
            if self.observation_primary_artifact_counts is not None:
                raise ValueError(
                    "normalization binding 1.1 cannot claim per-record artifact counts"
                )
            return self
        if self.provenance_binding != (
            "locked_manifest_all_artifact_receipts_and_primary_record_artifact"
        ):
            raise ValueError("normalization binding 1.2 has an invalid provenance mode")
        counts = self.observation_primary_artifact_counts
        if (
            counts is None
            or list(counts) != sorted(counts)
            or set(counts) != set(artifact_ids)
            or any(isinstance(value, bool) or value < 0 for value in counts.values())
            or sum(counts.values()) != self.observation_count
        ):
            raise ValueError(
                "normalization binding 1.2 has invalid primary artifact counts"
            )
        return self


class UnboundBuildInputBinding(_StrictReceiptModel):
    status: Literal[
        "unbound_fixture_or_manual_input",
        "unbound_bundled_synthetic_fixture",
    ]
    warning: str = Field(min_length=1)


BuildNormalizationBinding = Annotated[
    VerifiedNormalizationBuildBinding | UnboundBuildInputBinding,
    Field(discriminator="status"),
]


class BuildFileReceipt(_StrictReceiptModel):
    path: str = Field(min_length=1)
    sha256: str
    byte_size: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        return _require_sha256(value)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        candidate = Path(value)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != value
            or value == "build_receipt.json"
        ):
            raise ValueError("build product path must be a safe relative POSIX path")
        return value


class BuildBindings(_StrictReceiptModel):
    normalization: list[BuildNormalizationBinding] = Field(min_length=1)
    inputs: dict[str, list[str]]
    schema_sha256: dict[str, str]
    mapping_sha256: dict[str, str]
    code_sha256: str
    code_file_sha256: dict[str, str]

    @field_validator("code_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        return _require_sha256(value)

    @field_validator("schema_sha256", "mapping_sha256", "code_file_sha256")
    @classmethod
    def valid_hash_map(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("build resource hash maps cannot be empty")
        return {name: _require_sha256(digest) for name, digest in value.items()}

    @field_validator("inputs")
    @classmethod
    def valid_inputs(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        allowed = {"observations_sha256", "query_sha256", "policy_sha256"}
        if not value or not set(value).issubset(allowed):
            raise ValueError("build input hash names are missing or unsupported")
        normalized: dict[str, list[str]] = {}
        for name, digests in value.items():
            if not digests or digests != sorted(set(digests)):
                raise ValueError("build input hash lists must be nonempty, sorted, and unique")
            normalized[name] = [_require_sha256(digest) for digest in digests]
        return normalized

    @model_validator(mode="after")
    def one_input_provenance_state(self) -> "BuildBindings":
        if len(self.normalization) != 1:
            raise ValueError("build receipt requires exactly one input provenance state")
        return self


class BuildReceipt(_StrictReceiptModel):
    build_schema_version: Literal["1.1"]
    created_at: datetime
    transformer: Literal["wormctx"]
    transformer_version: str = Field(min_length=1)
    bindings: BuildBindings
    summary: dict[str, Any]
    files: list[BuildFileReceipt] = Field(min_length=1)

    @model_validator(mode="after")
    def internally_consistent(self) -> "BuildReceipt":
        if self.created_at.tzinfo is None:
            raise ValueError("build receipt timestamp must include a timezone")
        paths = [item.path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("build receipt files must be sorted and unique")
        declared = _collect_produced_files(self.summary)
        if paths != declared:
            raise ValueError("build receipt files differ from summary produced_files")
        _require_authoritative_product_set(set(paths))
        if self.bindings.inputs != _collect_input_hashes(self.summary):
            raise ValueError("build receipt input hashes differ from summary bindings")
        claimed_normalization = [
            item.model_dump(mode="json") for item in self.bindings.normalization
        ]
        if claimed_normalization != _collect_values_for_key(
            self.summary, "input_binding"
        ):
            raise ValueError("build receipt normalization binding differs from summary")
        expected_input_names = {"observations_sha256"}
        if set(paths) != GRAPH_PRODUCT_FILES:
            expected_input_names.update({"query_sha256", "policy_sha256"})
        if set(self.bindings.inputs) != expected_input_names:
            raise ValueError("build input bindings do not match the authoritative product set")
        return self


GRAPH_PRODUCT_FILES = {
    "validated_observations.jsonl",
    "contextual_nodes.tsv",
    "contextual_edges.tsv",
    "kgx_nodes.tsv",
    "kgx_edges.tsv",
    "kgx_context_sidecar.jsonl",
}
TRANSPORT_PRODUCT_FILES = {
    "coverage_long.csv",
    "coverage_cube.csv",
    "transport_scores.csv",
    "transport_scores.json",
    "transport_map.html",
}


CONTEXTUAL_NODE_FIELDS = (
    "id",
    "name",
    "category",
    "source",
    "interpretation",
    "observation_status",
    "record_origin",
    "predicate",
    "proposition_negated",
    "assertion_method",
    "biolink_knowledge_level",
    "biolink_agent_type",
    "qualifiers",
    "context",
    "result",
    "canonical_observation",
    "canonical_sha256",
    "modality",
    "readout_resolution",
    "absolute_time",
    "context_gaps",
    "context_missing_reasons",
    "source_value",
    "mapping_record_id",
    "mapping_status",
    "reportedness",
    "intervention_type",
    "dose",
    "duration",
    "notes",
    "result_type",
    "summary",
    "group_role",
    "group_size",
    "group_members",
    "group_genetic_background",
    "measured_feature",
    "value",
    "statistic",
    "uncertainty",
    "uncertainty_type",
    "confidence_level",
    "p_value",
    "penetrance",
    "sample_size",
    "biological_replicates",
    "technical_replicates",
    "measurement_time",
    "comparison_group_ids",
    "asset_ids",
    "asset_uri",
    "media_type",
    "byte_size",
    "role",
    "evidence_type",
    "direction",
    "source_record_id",
    "publications",
    "figure_or_table",
    "source_url",
    "checksum_sha256",
    "source_release",
    "source_artifact",
    "retrieved_at",
    "license_uri",
    "transformation_activity",
    "transformer_version",
)
CONTEXTUAL_EDGE_FIELDS = (
    "id",
    "subject",
    "predicate",
    "object",
    "derived_from_observation",
    "observation_status",
    "negated",
    "interpretation",
)
KGX_NODE_FIELDS = ("id", "name", "category", "provided_by")
KGX_EDGE_FIELDS = (
    "id",
    "subject",
    "predicate",
    "object",
    "category",
    "primary_knowledge_source",
    "publications",
    "has_evidence",
    "knowledge_level",
    "agent_type",
    "negated",
)


def normalize_snapshot(
    manifest_path: str | Path,
    raw_root: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    snapshot, snapshot_binding = load_snapshot_context_with_binding(
        manifest_path, raw_root
    )
    adapters = {
        "local_jsonl": LocalJsonlAdapter,
        "wormbase_ws298_source_qualification": WormBaseWs298Adapter,
    }
    adapter_class = adapters.get(snapshot.manifest.adapter)
    if adapter_class is None:
        raise ValueError(
            f"adapter {snapshot.manifest.adapter!r} is not implemented; "
            "the source must remain disabled"
        )
    adapter = adapter_class()
    if adapter.adapter_version != snapshot.manifest.adapter_version:
        raise ValueError(
            f"adapter version mismatch: lock={snapshot.manifest.adapter_version}, "
            f"runtime={adapter.adapter_version}"
        )
    report = adapter.validate_raw(snapshot)
    report_payload = {
        "source_id": report.source_id,
        "snapshot_id": report.snapshot_id,
        "adapter_version": report.adapter_version,
        "schema_version": report.schema_version,
        "valid": report.valid,
        "checked_records": report.checked_records,
        "issues": [
            {**issue.__dict__, "severity": issue.severity.value}
            for issue in report.issues
        ],
    }
    if report_path:
        write_json(report_path, report_payload)
    if not report.valid:
        raise ValueError(f"snapshot validation failed with {len(report.issues)} issue(s)")

    observations = list(adapter.iter_observations(snapshot))
    receipts = tuple(sorted(snapshot.receipts, key=lambda item: item.artifact_id))
    receipt_aliases: dict[str, Any] = {}
    for receipt in receipts:
        for alias in (receipt.artifact_id, receipt.filename):
            existing = receipt_aliases.get(alias)
            if existing is not None and existing.artifact_id != receipt.artifact_id:
                raise ValueError(
                    f"artifact receipt alias {alias!r} is ambiguous across the snapshot"
                )
            receipt_aliases[alias] = receipt
    primary_artifact_counts = {
        receipt.artifact_id: 0 for receipt in receipts
    }
    for observation in observations:
        if len(receipts) == 1:
            receipt = receipts[0]
        else:
            emitted_artifact = observation.provenance.source_artifact
            receipt = receipt_aliases.get(emitted_artifact)
            if receipt is None:
                raise ValueError(
                    "multi-artifact adapter emitted an observation without an exact "
                    f"artifact-id or filename binding: {emitted_artifact!r}"
                )
            emitted_checksum = observation.provenance.checksum_sha256
            if emitted_checksum != receipt.sha256:
                raise ValueError(
                    "multi-artifact adapter emitted an observation whose checksum "
                    f"does not match {receipt.artifact_id!r}"
                )
        primary_artifact_counts[receipt.artifact_id] += 1
        source_url = receipt.final_url if receipt.final_url.startswith("https://") else None
        observation.provenance = SourceSnapshot(
            source_id=snapshot.manifest.source_id,
            source_release=snapshot.manifest.release,
            source_artifact=receipt.filename,
            source_url=source_url,
            retrieved_at=receipt.fetched_at,
            checksum_sha256=receipt.sha256,
            license_uri=receipt.license_uri,
            transformation_activity=(
                f"wormctx:normalize-{receipt.manifest_sha256[:16]}"
            ),
            transformer_version=adapter.adapter_version,
        )
    validate_observation_collection(observations)
    output = Path(output_path)
    write_jsonl(output, observations)
    is_multi_artifact = len(receipts) > 1
    normalization_receipt = NormalizationReceipt(
        normalization_receipt_schema_version=(
            "1.2" if is_multi_artifact else "1.1"
        ),
        created_at=datetime.now(timezone.utc),
        provenance_binding=(
            "locked_manifest_all_artifact_receipts_and_primary_record_artifact"
            if is_multi_artifact
            else "locked_manifest_and_release_receipts"
        ),
        source_id=snapshot_binding["source_id"],
        source_release=snapshot_binding["source_release"],
        manifest_sha256=snapshot_binding["manifest_sha256"],
        release_receipt_sha256=snapshot_binding["release_receipt_sha256"],
        artifact_receipts=snapshot_binding["artifact_receipts"],
        adapter=snapshot_binding["adapter"],
        adapter_version=snapshot_binding["adapter_version"],
        schema_sha256=_schema_hashes(),
        mapping_sha256=_mapping_hashes(),
        code_sha256=_code_hashes()["aggregate_sha256"],
        output_filename=output.name,
        output_sha256=_sha256(output),
        observation_count=len(observations),
        observation_primary_artifact_counts=(
            primary_artifact_counts if is_multi_artifact else None
        ),
    )
    receipt_path = output.with_suffix(output.suffix + ".normalization-receipt.json")
    write_json(
        receipt_path,
        normalization_receipt.model_dump(mode="json", exclude_none=True),
    )
    result = {
        "output": str(output.resolve()),
        "normalization_receipt": str(receipt_path.resolve()),
        "observation_count": len(observations),
        "manifest_sha256": snapshot_binding["manifest_sha256"],
        "artifact_sha256_by_id": {
            receipt.artifact_id: receipt.sha256 for receipt in receipts
        },
    }
    if len(receipts) == 1:
        result["artifact_sha256"] = receipts[0].sha256
    return result


def build_graph(input_path: str | Path, output: str | Path) -> dict[str, Any]:
    input_file = Path(input_path).resolve()
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observations = load_observations(input_file)
    produced_files = [
        "validated_observations.jsonl",
        "contextual_nodes.tsv",
        "contextual_edges.tsv",
        "kgx_nodes.tsv",
        "kgx_edges.tsv",
        "kgx_context_sidecar.jsonl",
    ]

    contextual = build_contextual_graph(observations)
    # Contextual graph construction must not fail because an exchange mapping is
    # incomplete. Standard KGX output skips those rows; the sidecar records why.
    kgx = build_kgx_projection(
        observations,
        require_reviewed_infores=True,
        on_unreviewed_infores="skip",
        on_unmapped_predicate="skip",
        require_edge_metadata=True,
        on_missing_edge_metadata="skip",
    )
    sidecar = build_kgx_context_sidecar(observations)
    with tempfile.TemporaryDirectory(
        prefix=".wormctx-graph-", dir=output_path.parent
    ) as directory:
        staging = Path(directory).resolve()
        write_jsonl(staging / "validated_observations.jsonl", observations)
        write_table(
            staging / "contextual_nodes.tsv",
            contextual.sorted_nodes(),
            fieldnames=CONTEXTUAL_NODE_FIELDS,
        )
        write_table(
            staging / "contextual_edges.tsv",
            contextual.sorted_edges(),
            fieldnames=CONTEXTUAL_EDGE_FIELDS,
        )
        write_table(
            staging / "kgx_nodes.tsv",
            kgx.sorted_nodes(),
            fieldnames=KGX_NODE_FIELDS,
        )
        write_table(
            staging / "kgx_edges.tsv",
            kgx.sorted_edges(),
            fieldnames=KGX_EDGE_FIELDS,
        )
        write_jsonl(staging / "kgx_context_sidecar.jsonl", sidecar)
        _publish_staged_files(staging, output_path, produced_files)
    return {
        "observation_count": len(observations),
        "contextual_node_count": len(contextual.nodes),
        "contextual_edge_count": len(contextual.edges),
        "kgx_node_count": len(kgx.nodes),
        "kgx_edge_count": len(kgx.edges),
        "kgx_sidecar_record_count": len(sidecar),
        "kgx_projection_policy": {
            "include_uncertain": False,
            "require_reviewed_infores": True,
            "on_unreviewed_infores": "skip",
            "on_unmapped_predicate": "skip",
            "require_edge_metadata": True,
            "on_missing_edge_metadata": "skip",
            "mode": "standard_strict_skip_blocked",
        },
        "input_hashes": {"observations_sha256": _sha256(input_file)},
        "produced_files": produced_files,
    }


def build_transport_map(
    input_path: str | Path,
    query_path: str | Path,
    policy_path: str | Path,
    output: str | Path,
    *,
    carry_forward_files: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    input_file = Path(input_path).resolve()
    query_file = Path(query_path).resolve()
    policy_file = Path(policy_path).resolve()
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observations = load_observations(input_file)
    query = load_query(query_file)
    policy = read_json(policy_file)
    scores = [score_observation(item, query.context, policy) for item in observations]

    long_rows = coverage_long(observations)
    cube_rows = coverage_cube(observations)
    score_rows = []
    for result in scores:
        row = {
            key: value
            for key, value in result.items()
            if key not in {"components", "hard_failures"}
        }
        row["hard_failures"] = result["hard_failures"]
        for component in result["components"]:
            prefix = component["dimension"]
            row[f"{prefix}_state"] = component["state"]
            row[f"{prefix}_score"] = component["score"]
            row[f"{prefix}_evidence"] = component["evidence_values"]
            row[f"{prefix}_query"] = component["query_values"]
        score_rows.append(row)
    score_fields = [
        "observation_id",
        "source_id",
        "score",
        "classification",
        "known_context_fraction",
        "hard_failures",
        "policy_id",
    ]
    for dimension in DIMENSIONS:
        score_fields.extend(
            (
                f"{dimension}_state",
                f"{dimension}_score",
                f"{dimension}_evidence",
                f"{dimension}_query",
            )
        )
    produced_files = [
        "coverage_long.csv",
        "coverage_cube.csv",
        "transport_scores.csv",
        "transport_scores.json",
        "transport_map.html",
    ]
    with tempfile.TemporaryDirectory(
        prefix=".wormctx-transport-", dir=output_path.parent
    ) as directory:
        staging = Path(directory).resolve()
        _seed_staging_with_declared_products(
            output_path,
            staging,
            list(carry_forward_files or []),
        )
        write_table(staging / "coverage_long.csv", long_rows, delimiter=",")
        write_table(staging / "coverage_cube.csv", cube_rows, delimiter=",")
        write_table(
            staging / "transport_scores.csv",
            score_rows,
            delimiter=",",
            fieldnames=score_fields,
        )
        write_json(staging / "transport_scores.json", scores)
        write_transport_html(staging / "transport_map.html", observations, query, scores)
        _publish_staged_files(staging, output_path, produced_files)
    counts = Counter(result["classification"] for result in scores)
    return {
        "coverage_row_count": len(long_rows),
        "coverage_cube_cell_count": len(cube_rows),
        "score_count": len(scores),
        "classification_counts": dict(sorted(counts.items())),
        "input_hashes": {
            "observations_sha256": _sha256(input_file),
            "query_sha256": _sha256(query_file),
            "policy_sha256": _sha256(policy_file),
        },
        "produced_files": produced_files,
    }


def write_build_receipt(output: str | Path, summary: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(output).resolve()
    produced_files = _collect_produced_files(summary)
    if not produced_files:
        raise ValueError(
            "build summary has no explicit produced_files; refusing to receipt a directory scan"
        )
    _require_authoritative_product_set(set(produced_files))
    actual_files = _authoritative_snapshot_files(output_path)
    actual_products = actual_files - {"build_receipt.json"}
    if actual_products != set(produced_files):
        raise ValueError(
            "output snapshot differs from declared authoritative products: "
            f"expected={sorted(produced_files)}, actual={sorted(actual_products)}"
        )
    files = []
    for relative_name in produced_files:
        path = (output_path / relative_name).resolve()
        if not path.is_relative_to(output_path) or not path.is_file():
            raise ValueError(f"declared build product is missing or unsafe: {relative_name}")
        files.append(
            {
                "path": relative_name,
                "sha256": _sha256(path),
                "byte_size": path.stat().st_size,
            }
        )
    code_binding = _code_hashes()
    receipt = BuildReceipt(
        build_schema_version="1.1",
        created_at=datetime.now(timezone.utc),
        transformer="wormctx",
        transformer_version="0.1.0",
        bindings={
            "normalization": _collect_values_for_key(summary, "input_binding"),
            "inputs": _collect_input_hashes(summary),
            "schema_sha256": _schema_hashes(),
            "mapping_sha256": _mapping_hashes(),
            "code_sha256": code_binding["aggregate_sha256"],
            "code_file_sha256": code_binding["file_sha256"],
        },
        summary=summary,
        files=files,
    )
    payload = receipt.model_dump(mode="json")
    write_json(output_path / "build_receipt.json", payload)
    return payload


def verify_build_receipt(
    output: str | Path,
    *,
    local_inputs: dict[str, list[str | Path]] | None = None,
    normalization_receipt_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    raw_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify an authoritative build snapshot and every locally supplied binding."""
    output_path = Path(output).resolve()
    receipt_path = output_path / "build_receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"build receipt is missing: {receipt_path}")
    receipt = BuildReceipt.model_validate(read_json(receipt_path))
    expected_products = {item.path for item in receipt.files}
    actual_files = _authoritative_snapshot_files(output_path)
    expected_files = expected_products | {"build_receipt.json"}
    if actual_files != expected_files:
        raise ValueError(
            "build snapshot has missing or extra authoritative files: "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    for item in receipt.files:
        path = (output_path / item.path).resolve()
        if path.stat().st_size != item.byte_size or _sha256(path) != item.sha256:
            raise ValueError(f"build product failed byte verification: {item.path}")

    code_binding = _code_hashes()
    if receipt.bindings.schema_sha256 != _schema_hashes():
        raise ValueError("build receipt schema hashes differ from this checkout")
    if receipt.bindings.mapping_sha256 != _mapping_hashes():
        raise ValueError("build receipt mapping hashes differ from this checkout")
    if receipt.bindings.code_sha256 != code_binding["aggregate_sha256"]:
        raise ValueError("build receipt code hash differs from this checkout")
    if receipt.bindings.code_file_sha256 != code_binding["file_sha256"]:
        raise ValueError("build receipt code-file hashes differ from this checkout")

    checked_inputs: dict[str, list[str]] = {}
    local_inputs = local_inputs or {}
    unsupported = set(local_inputs) - set(receipt.bindings.inputs)
    if unsupported:
        raise ValueError(
            f"local inputs were supplied for unbound names: {sorted(unsupported)}"
        )
    resolved_inputs: dict[str, list[Path]] = {}
    for name, paths in local_inputs.items():
        if not paths:
            raise ValueError(f"no local files supplied for input binding {name}")
        resolved = [Path(path).resolve() for path in paths]
        if any(not path.is_file() for path in resolved):
            raise ValueError(f"a local file is missing for input binding {name}")
        actual_hashes = sorted({_sha256(path) for path in resolved})
        if actual_hashes != receipt.bindings.inputs[name]:
            raise ValueError(f"local files do not match build input binding {name}")
        resolved_inputs[name] = resolved
        checked_inputs[name] = actual_hashes

    if (manifest_path is None) != (raw_root is None):
        raise ValueError("manifest_path and raw_root must be supplied together")
    if (manifest_path is not None or raw_root is not None) and (
        normalization_receipt_path is None
    ):
        raise ValueError(
            "full source verification requires a normalization receipt path"
        )

    claimed_binding = receipt.bindings.normalization[0]
    normalization_check: dict[str, Any]
    if isinstance(claimed_binding, UnboundBuildInputBinding):
        if normalization_receipt_path is not None:
            raise ValueError("build receipt explicitly declares an unbound input")
        normalization_check = {
            "status": claimed_binding.status,
            "warning": claimed_binding.warning,
        }
    elif normalization_receipt_path is None:
        normalization_check = {
            "status": "not_rechecked",
            "claimed_status": claimed_binding.status,
        }
    else:
        normalization_path = Path(normalization_receipt_path).resolve()
        if (
            not normalization_path.is_file()
            or _sha256(normalization_path)
            != claimed_binding.normalization_receipt_sha256
        ):
            raise ValueError("normalization receipt file does not match build binding")
        observation_inputs = resolved_inputs.get("observations_sha256", [])
        matching_observations = [
            path
            for path in observation_inputs
            if _sha256(path) == claimed_binding.input_sha256
        ]
        if len(matching_observations) != 1:
            raise ValueError(
                "normalization verification requires the uniquely bound observations input"
            )
        use_full_context = (
            claimed_binding.status == "full_provenance_verified"
            and manifest_path is not None
            and raw_root is not None
        )
        rechecked = verify_normalization_receipt(
            matching_observations[0],
            normalization_path,
            manifest_path=manifest_path if use_full_context else None,
            raw_root=raw_root if use_full_context else None,
        )
        claimed_payload = claimed_binding.model_dump(mode="json")
        if use_full_context or claimed_binding.status == "byte_bound_only":
            if rechecked != claimed_payload:
                raise ValueError(
                    "normalization receipt verification differs from build binding"
                )
        else:
            comparable_fields = set(claimed_payload) - {"status", "verification_level"}
            if any(
                claimed_payload[field] != rechecked[field]
                for field in comparable_fields
            ):
                raise ValueError(
                    "byte-level normalization verification differs from build binding"
                )
        normalization_check = {
            "status": rechecked["status"],
            "claimed_status": claimed_binding.status,
        }

    unchecked_inputs = sorted(set(receipt.bindings.inputs) - set(checked_inputs))
    return {
        "status": "authoritative_build_snapshot_verified",
        "build_receipt_sha256": _sha256(receipt_path),
        "product_count": len(receipt.files),
        "products": sorted(expected_products),
        "checked_inputs": checked_inputs,
        "unchecked_inputs": unchecked_inputs,
        "normalization": normalization_check,
        "checkout_bindings": "verified",
    }


def verify_normalization_receipt(
    input_path: str | Path,
    receipt_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    raw_root: str | Path | None = None,
) -> dict[str, Any]:
    input_file = Path(input_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    receipt = NormalizationReceipt.model_validate(read_json(receipt_file))
    actual_sha256 = _sha256(input_file)
    if receipt.output_filename != input_file.name:
        raise ValueError("normalization receipt output filename does not match the input")
    if receipt.output_sha256 != actual_sha256:
        raise ValueError("normalization receipt does not match the build input bytes")
    observations = load_observations(input_file)
    actual_observation_count = len(observations)
    if receipt.observation_count != actual_observation_count:
        raise ValueError("normalization receipt observation count does not match the input")
    actual_primary_artifact_counts = None
    if receipt.normalization_receipt_schema_version == "1.2":
        actual_primary_artifact_counts = _derive_primary_artifact_counts(
            observations,
            receipt,
        )
        if (
            actual_primary_artifact_counts
            != receipt.observation_primary_artifact_counts
        ):
            raise ValueError(
                "normalization receipt primary artifact counts do not match the input"
            )
    if (manifest_path is None) != (raw_root is None):
        raise ValueError("manifest_path and raw_root must be supplied together")

    verification_level = "byte_bound_only"
    if manifest_path is not None and raw_root is not None:
        _, snapshot_binding = load_snapshot_context_with_binding(manifest_path, raw_root)
        expected_artifact_receipts = snapshot_binding["artifact_receipts"]
        if all(item.filename is None for item in receipt.artifact_receipts):
            # Backward verification for 1.1 receipts created before filenames
            # became part of the portable artifact binding.
            expected_artifact_receipts = [
                {
                    key: value
                    for key, value in item.items()
                    if key != "filename"
                }
                for item in expected_artifact_receipts
            ]
        expected_snapshot = {
            "source_id": snapshot_binding["source_id"],
            "source_release": snapshot_binding["source_release"],
            "manifest_sha256": snapshot_binding["manifest_sha256"],
            "release_receipt_sha256": snapshot_binding["release_receipt_sha256"],
            "adapter": snapshot_binding["adapter"],
            "adapter_version": snapshot_binding["adapter_version"],
            "artifact_receipts": expected_artifact_receipts,
        }
        actual_snapshot = {
            "source_id": receipt.source_id,
            "source_release": receipt.source_release,
            "manifest_sha256": receipt.manifest_sha256,
            "release_receipt_sha256": receipt.release_receipt_sha256,
            "adapter": receipt.adapter,
            "adapter_version": receipt.adapter_version,
            "artifact_receipts": [
                item.model_dump(mode="json", exclude_none=True)
                for item in receipt.artifact_receipts
            ],
        }
        if actual_snapshot != expected_snapshot:
            raise ValueError(
                "normalization receipt does not match the fully verified source snapshot"
            )
        if receipt.schema_sha256 != _schema_hashes():
            raise ValueError("normalization receipt schema hashes differ from this checkout")
        if receipt.mapping_sha256 != _mapping_hashes():
            raise ValueError("normalization receipt mapping hashes differ from this checkout")
        if receipt.code_sha256 != _code_hashes()["aggregate_sha256"]:
            raise ValueError("normalization receipt code hash differs from this checkout")
        # The receipt is not a signature. Replaying the deterministic adapter is
        # what proves that these normalized bytes were derived from the verified
        # snapshot with the bound code, schema, and mappings.
        with tempfile.TemporaryDirectory(
            prefix=".wormctx-normalization-replay-"
        ) as directory:
            replay_path = Path(directory) / receipt.output_filename
            replay = normalize_snapshot(manifest_path, raw_root, replay_path)
            if replay["observation_count"] != receipt.observation_count:
                raise ValueError(
                    "normalization receipt does not match deterministic replay count"
                )
            if _sha256(replay_path) != actual_sha256:
                raise ValueError(
                    "normalization input bytes do not match deterministic replay"
                )
            replay_receipt = NormalizationReceipt.model_validate(
                read_json(replay["normalization_receipt"])
            )
            replay_closure = (
                replay_receipt.normalization_receipt_schema_version,
                replay_receipt.provenance_binding,
                replay_receipt.observation_primary_artifact_counts,
            )
            claimed_closure = (
                receipt.normalization_receipt_schema_version,
                receipt.provenance_binding,
                receipt.observation_primary_artifact_counts,
            )
            if replay_closure != claimed_closure:
                raise ValueError(
                    "normalization receipt artifact closure differs from replay"
                )
        verification_level = "full_provenance_verified"

    return {
        "status": verification_level,
        "verification_level": verification_level,
        "normalization_receipt": receipt_file.name,
        "normalization_receipt_sha256": _sha256(receipt_file),
        "normalization_receipt_schema_version": (
            receipt.normalization_receipt_schema_version
        ),
        "provenance_binding": receipt.provenance_binding,
        "input_sha256": actual_sha256,
        "observation_count": actual_observation_count,
        "observation_primary_artifact_counts": (
            actual_primary_artifact_counts
        ),
        "manifest_sha256": receipt.manifest_sha256,
        "release_receipt_sha256": receipt.release_receipt_sha256,
        "artifact_receipts": [
            item.model_dump(mode="json") for item in receipt.artifact_receipts
        ],
        "schema_sha256": receipt.schema_sha256,
        "mapping_sha256": receipt.mapping_sha256,
        "code_sha256": receipt.code_sha256,
        "adapter": receipt.adapter,
        "adapter_version": receipt.adapter_version,
    }


def _derive_primary_artifact_counts(
    observations: list[Any],
    receipt: NormalizationReceipt,
) -> dict[str, int]:
    """Recompute the 1.2 record-to-primary-artifact closure from normalized rows."""

    manifest_artifacts = {item.artifact_id: item for item in receipt.artifact_receipts}
    if len(manifest_artifacts) != len(receipt.artifact_receipts):
        raise ValueError("normalization receipt contains duplicate artifact ids")
    counts = {artifact_id: 0 for artifact_id in sorted(manifest_artifacts)}
    # Version 1.2 standardizes source_artifact to the locked artifact filename
    # and independently binds the checksum, preventing an adapter from charging
    # a record to an unused supporting artifact.
    by_filename: dict[str, NormalizationArtifactBinding] = {}
    for binding in receipt.artifact_receipts:
        if binding.filename is None or binding.filename in by_filename:
            raise ValueError(
                "normalization 1.2 artifact filenames are missing or ambiguous"
            )
        by_filename[binding.filename] = binding
    for observation in observations:
        binding = by_filename.get(observation.provenance.source_artifact)
        if (
            binding is None
            or observation.provenance.checksum_sha256 != binding.artifact_sha256
            or observation.provenance.source_id != receipt.source_id
            or observation.provenance.source_release != receipt.source_release
        ):
            raise ValueError(
                "normalization 1.2 row provenance does not match exactly one "
                "locked primary artifact"
            )
        counts[binding.artifact_id] += 1
    return counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("expected a lowercase 64-character SHA-256 digest")
    return value


def _schema_hashes() -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in sorted(_resource_directory("schemas").glob("*.yaml"))
    }


def _mapping_hashes() -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in sorted(_resource_directory("mappings").glob("*.sssom.tsv"))
    }


def _code_hashes() -> dict[str, Any]:
    source_root = Path(__file__).resolve().parent
    files = {
        path.relative_to(source_root).as_posix(): _sha256(path)
        for path in sorted(source_root.rglob("*.py"))
    }
    payload = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "aggregate_sha256": hashlib.sha256(payload).hexdigest(),
        "file_sha256": files,
    }


def _resource_directory(name: str) -> Path:
    """Resolve checked-out resources and hatch force-included wheel resources."""
    package_resource = Path(__file__).resolve().parent / "resources" / name
    if package_resource.is_dir():
        return package_resource
    checkout_resource = Path(__file__).resolve().parents[2] / name
    if checkout_resource.is_dir():
        return checkout_resource
    raise FileNotFoundError(
        f"wormctx resource directory {name!r} is missing from the package and checkout"
    )


def _collect_produced_files(summary: dict[str, Any]) -> list[str]:
    declared: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "produced_files":
                    if not isinstance(child, list) or not all(
                        isinstance(item, str) for item in child
                    ):
                        raise ValueError("produced_files must be a list of relative paths")
                    for item in child:
                        candidate = Path(item)
                        if (
                            not item
                            or candidate.is_absolute()
                            or ".." in candidate.parts
                            or item == "build_receipt.json"
                        ):
                            raise ValueError(f"unsafe declared build product: {item!r}")
                        declared.add(candidate.as_posix())
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(summary)
    return sorted(declared)


def _collect_values_for_key(value: Any, wanted_key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == wanted_key:
                values.append(child)
            else:
                values.extend(_collect_values_for_key(child, wanted_key))
    elif isinstance(value, list):
        for child in value:
            values.extend(_collect_values_for_key(child, wanted_key))
    return values


def _collect_input_hashes(summary: dict[str, Any]) -> dict[str, list[str]]:
    collected: dict[str, set[str]] = {}
    for value in _collect_values_for_key(summary, "input_hashes"):
        if not isinstance(value, dict):
            raise ValueError("input_hashes must be an object")
        for name, digest in value.items():
            if not isinstance(name, str) or not isinstance(digest, str):
                raise ValueError("input_hashes must map names to SHA-256 strings")
            collected.setdefault(name, set()).add(_require_sha256(digest))
    return {name: sorted(digests) for name, digests in sorted(collected.items())}


def _require_authoritative_product_set(products: set[str]) -> None:
    allowed_sets = (
        GRAPH_PRODUCT_FILES,
        TRANSPORT_PRODUCT_FILES,
        GRAPH_PRODUCT_FILES | TRANSPORT_PRODUCT_FILES,
    )
    if products not in allowed_sets:
        raise ValueError(
            "declared products are not a complete authoritative graph, transport, "
            "or composite snapshot"
        )


def _authoritative_snapshot_files(output: Path) -> set[str]:
    output = output.resolve()
    if not output.is_dir():
        raise ValueError(f"build output directory is missing: {output}")
    files: set[str] = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"build snapshot cannot contain symbolic links: {path}")
        relative = path.relative_to(output).as_posix()
        if path.is_dir():
            raise ValueError(f"build snapshot cannot contain subdirectories: {relative}")
        if not path.is_file():
            raise ValueError(f"build snapshot contains an unsupported entry: {relative}")
        files.add(relative)
    return files


def _publish_staged_files(
    staging: Path, output: Path, produced_files: list[str]
) -> None:
    """Publish a verified same-parent staging snapshot with rollback on failure."""
    staging = staging.resolve()
    output = output.resolve()
    if staging.parent != output.parent:
        raise ValueError("staging and output must share a parent for directory publication")
    for relative_name in produced_files:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe staged build product: {relative_name!r}")
        source = (staging / relative).resolve()
        if not source.is_relative_to(staging.resolve()) or not source.is_file():
            raise ValueError(f"staged build product is missing: {relative_name}")
    staged_products = _authoritative_snapshot_files(staging)
    _require_authoritative_product_set(staged_products)
    if not set(produced_files).issubset(staged_products):
        raise ValueError("staged snapshot omits newly produced authoritative files")
    if output.exists() and not output.is_dir():
        raise ValueError(f"build output exists but is not a directory: {output}")
    with tempfile.TemporaryDirectory(
        prefix=".wormctx-backup-", dir=output.parent
    ) as backup_directory:
        previous = Path(backup_directory) / "previous"
        had_previous = output.exists()
        if had_previous:
            os.replace(output, previous)
        try:
            os.replace(staging, output)
        except Exception:
            if had_previous and not output.exists() and previous.exists():
                os.replace(previous, output)
            raise


def _seed_staging_with_declared_products(
    output: Path, staging: Path, declared_products: list[str]
) -> None:
    """Carry forward only an explicitly declared complete prior product set."""
    output = output.resolve()
    staging = staging.resolve()
    if not declared_products:
        return
    declared = {Path(item).as_posix() for item in declared_products}
    _require_authoritative_product_set(declared)
    if not output.is_dir():
        raise ValueError("cannot carry products forward from a missing output snapshot")
    for relative_name in sorted(declared):
        source = (output / relative_name).resolve()
        if (
            not source.is_relative_to(output)
            or not source.is_file()
            or source.is_symlink()
        ):
            raise ValueError(f"declared prior build product is missing: {relative_name}")
        destination = (staging / relative_name).resolve()
        if not destination.is_relative_to(staging):
            raise ValueError(f"declared prior product is unsafe: {relative_name}")
        shutil.copy2(source, destination)
