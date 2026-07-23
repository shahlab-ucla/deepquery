"""Fail-closed OMIX709 source-curator and external-training handoff.

This module handles identity and data-use metadata only. It does not read OMIX
measurement values, prepare model tensors, fit models, or open a sealed test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .developmental_omix_stage_adapter import (
    OVERLAP_RECEIPT_VERSION,
    PilotOverlapReceipt,
)


PACKET_VERSION = "wormctx-omix709-source-curator-outreach-packet-1.0"
RESPONSE_VERSION = "wormctx-omix709-source-curator-rights-response-1.0"
HANDOFF_VERSION = "wormctx-omix709-source-curator-launch-handoff-1.0"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
UTC_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")

ASSERTION_KEYS = [
    "table_s3_uses_only_omix709_05_54_embryos",
    "omix709_05_54_and_05_55_are_biological_embryo_disjoint",
    "identifier_namespaces_are_complete_and_not_reused_across_archives",
    "any_reused_embryos_are_exhaustively_listed_in_overlap_pairs",
]
PERMISSION_KEYS = [
    "controlled_noncommercial_academic_analysis_permitted",
    "external_compute_source_transfer_permitted",
    "external_compute_processing_permitted",
    "model_training_parameter_updates_permitted",
    "trained_weights_and_derived_artifacts_retention_permitted",
    "public_release_of_source_data_or_trained_weights_permitted",
]
LAUNCH_REQUIRED_PERMISSION_KEYS = PERMISSION_KEYS[:5]
PAIR_KEYS = [
    "normalized_embryo_id",
    "scaleup_embryo_id",
    "omix709_05_55_source_member",
    "omix709_05_54_source_member",
]

EXPECTED_SOURCE_BINDINGS: dict[str, Any] = {
    "pilot_normalized_product_bytes": 62_315_263,
    "pilot_normalized_product_sha256": (
        "51fcda2051bb0d7cff376581e29e80be20da013211b4c40cb16a2186b7039659"
    ),
    "pilot_embryos": 251,
    "omix709_05_54_archive_bytes": 677_141_156,
    "omix709_05_54_archive_sha256": (
        "5e268e447c8f45fdd83823c029cfef6d67d389ad39013b66f9cb7191eb9a2030"
    ),
    "omix709_05_55_archive_bytes": 51_428_713,
    "omix709_05_55_archive_sha256": (
        "429f8888f3549738e66e28c7f2af440233982b96950350fd97b5d2b6d68d49c8"
    ),
    "scaleup_source_bundle_sha256": (
        "3b2bd2534beb81234492a79d947083ab523ac2afbe737e9454c1516dcb6c9aed"
    ),
    "split_manifest_sha256": (
        "60cdd6c3e4a236e86ed0a2cb95b4ea1b5ee1fbfc14004ffb54fa774c4287ba60"
    ),
    "attestation_request_sha256": (
        "c7cf62c22fbeef0781ec9ea55d7cec0f001eac749052673e66d1a48744b43eaf"
    ),
    "raw_identity_qualification_sha256": (
        "fb0d2cb4cd1735d7d82523939d4413b725f9ccb77b02727842c0c8df9e39bf8b"
    ),
    "main_member_inventory_sha256": (
        "bcbbb232d4c08a8c722902fff48af7044838bd4659d1d1c77a7c08338e83f3a3"
    ),
    "addendum_member_inventory_sha256": (
        "d25665463d2a4f702bf7454b57ba4123b83b7af7c6e21107a29edbde9383158b"
    ),
    "overlap_candidates_sha256": (
        "81d0e933f22294deab5d292be2fd1b7fa1ca081842ecbff4db3ee1ab6eeff423"
    ),
    "open_overlap_status_sha256": (
        "833b1cf87582b894da2650fa181efc70daf03d1c79e5a85e67ba71ab4f9cdcd2"
    ),
    "developmental_readiness_contract_sha256": (
        "16808c931c45918fd9fc5a78a6d4e226a2f765df1b03e8125555885e0a7b5310"
    ),
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SourceBindings(_StrictModel):
    pilot_normalized_product_bytes: int = Field(gt=0)
    pilot_normalized_product_sha256: str
    pilot_embryos: int = Field(gt=0)
    omix709_05_54_archive_bytes: int = Field(gt=0)
    omix709_05_54_archive_sha256: str
    omix709_05_55_archive_bytes: int = Field(gt=0)
    omix709_05_55_archive_sha256: str
    scaleup_source_bundle_sha256: str
    split_manifest_sha256: str
    attestation_request_sha256: str
    raw_identity_qualification_sha256: str
    main_member_inventory_sha256: str
    addendum_member_inventory_sha256: str
    overlap_candidates_sha256: str
    open_overlap_status_sha256: str
    developmental_readiness_contract_sha256: str

    @field_validator(
        "pilot_normalized_product_sha256",
        "omix709_05_54_archive_sha256",
        "omix709_05_55_archive_sha256",
        "scaleup_source_bundle_sha256",
        "split_manifest_sha256",
        "attestation_request_sha256",
        "raw_identity_qualification_sha256",
        "main_member_inventory_sha256",
        "addendum_member_inventory_sha256",
        "overlap_candidates_sha256",
        "open_overlap_status_sha256",
        "developmental_readiness_contract_sha256",
    )
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("source binding must be a lowercase SHA-256")
        return value


class CurrentStateFile(_StrictModel):
    relative_path: str = Field(min_length=1)
    bytes: int = Field(gt=0)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("current-state file hash must be a lowercase SHA-256")
        return value


class OutreachPacket(_StrictModel):
    schema_version: str
    packet_id: str
    status: Literal["awaiting_source_curator_and_rights_response"]
    classification: Literal["identity_and_rights_only_no_outcome_access"]
    biological_claims_permitted: Literal[False]
    outcome_values_permitted: Literal[False]
    outcome_prevalence_inspection_permitted: Literal[False]
    sealed_test_access_permitted: Literal[False]
    contact_executed: Literal[False]
    source_bindings: SourceBindings
    current_state_files: dict[str, CurrentStateFile]
    required_identity_assertions: list[str]
    required_overlap_pair_fields: list[str]
    required_permissions: list[str]
    response_schema_version: str
    durable_evidence_requirement: str
    authority_requirement: str
    external_compute_scope_requirement: str
    launch_rule: str

    @model_validator(mode="after")
    def frozen_packet(self) -> "OutreachPacket":
        if self.schema_version != PACKET_VERSION:
            raise ValueError("unknown source-curator outreach packet schema")
        if self.response_schema_version != RESPONSE_VERSION:
            raise ValueError("unknown source-curator response schema")
        if self.source_bindings.model_dump(mode="json") != EXPECTED_SOURCE_BINDINGS:
            raise ValueError("OMIX709 source bindings differ from the frozen gate")
        if self.required_identity_assertions != ASSERTION_KEYS:
            raise ValueError("identity assertion set or order differs")
        if self.required_overlap_pair_fields != PAIR_KEYS:
            raise ValueError("overlap pair field set or order differs")
        if self.required_permissions != PERMISSION_KEYS:
            raise ValueError("permission question set or order differs")
        required_files = {
            "attestation_request",
            "raw_identity_qualification",
            "open_overlap_status",
            "developmental_readiness_contract",
        }
        if set(self.current_state_files) != required_files:
            raise ValueError("current-state file set differs")
        return self


class AuthorityRecord(_StrictModel):
    name: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    role: str = Field(min_length=1)
    authority_basis: str = Field(min_length=1)
    attested_utc: str
    authority_explicitly_asserted: bool

    @field_validator("attested_utc")
    @classmethod
    def utc_timestamp(cls, value: str) -> str:
        if UTC_RE.fullmatch(value) is None:
            raise ValueError("authority attestation time must be UTC ISO-8601")
        return value


class DurableEvidence(_StrictModel):
    record_identifier: str = Field(min_length=1)
    bytes: int = Field(gt=0)
    sha256: str
    durable_record_locator: str = Field(min_length=1)
    signed_or_archived: bool

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("durable evidence must have a lowercase SHA-256")
        return value


class IdentityAssertions(_StrictModel):
    table_s3_uses_only_omix709_05_54_embryos: bool
    omix709_05_54_and_05_55_are_biological_embryo_disjoint: bool
    identifier_namespaces_are_complete_and_not_reused_across_archives: bool
    any_reused_embryos_are_exhaustively_listed_in_overlap_pairs: bool


class OverlapPair(_StrictModel):
    normalized_embryo_id: str = Field(min_length=1)
    scaleup_embryo_id: str = Field(min_length=1)
    omix709_05_55_source_member: str = Field(min_length=1)
    omix709_05_54_source_member: str = Field(min_length=1)


class RightsPermissions(_StrictModel):
    controlled_noncommercial_academic_analysis_permitted: bool
    external_compute_source_transfer_permitted: bool
    external_compute_processing_permitted: bool
    model_training_parameter_updates_permitted: bool
    trained_weights_and_derived_artifacts_retention_permitted: bool
    public_release_of_source_data_or_trained_weights_permitted: bool
    external_compute_scope: str = Field(min_length=1)
    terms_or_policy_identifier: str = Field(min_length=1)
    conditions: list[str]

    @field_validator("conditions")
    @classmethod
    def nonempty_conditions(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("permission conditions may not contain blank entries")
        return value


class CuratorRightsResponse(_StrictModel):
    schema_version: str
    outreach_packet_sha256: str
    source_bindings: SourceBindings
    identity_attestor: AuthorityRecord
    rights_attestor: AuthorityRecord
    durable_evidence: DurableEvidence
    assertions: IdentityAssertions
    mapping_basis: Literal["source_supported_exact_identity_receipt"]
    mapping_complete: bool
    overlap_pairs: list[OverlapPair]
    permissions: RightsPermissions
    outcome_values_used_for_identity_or_rights_decision: bool
    sealed_test_opened_for_identity_or_rights_decision: bool

    @field_validator("outreach_packet_sha256")
    @classmethod
    def valid_packet_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("outreach packet hash must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def complete_response(self) -> "CuratorRightsResponse":
        if self.schema_version != RESPONSE_VERSION:
            raise ValueError("unknown source-curator response schema")
        if not self.identity_attestor.authority_explicitly_asserted:
            raise ValueError("source identity authority was not explicitly asserted")
        if not self.rights_attestor.authority_explicitly_asserted:
            raise ValueError("data-use rights authority was not explicitly asserted")
        if not self.durable_evidence.signed_or_archived:
            raise ValueError("source response is not a signed or archived durable record")
        if not self.mapping_complete:
            raise ValueError("overlap mapping was not asserted complete")
        if self.outcome_values_used_for_identity_or_rights_decision:
            raise ValueError("identity/rights decision may not use outcome values")
        if self.sealed_test_opened_for_identity_or_rights_decision:
            raise ValueError("identity/rights decision may not open the sealed test")
        assertions = self.assertions
        if (
            not assertions.table_s3_uses_only_omix709_05_54_embryos
            or not assertions.identifier_namespaces_are_complete_and_not_reused_across_archives
            or not assertions.any_reused_embryos_are_exhaustively_listed_in_overlap_pairs
        ):
            raise ValueError("source authority did not establish complete overlap evidence")
        normalized = [item.normalized_embryo_id for item in self.overlap_pairs]
        scaleup = [item.scaleup_embryo_id for item in self.overlap_pairs]
        addendum = [item.omix709_05_55_source_member for item in self.overlap_pairs]
        main = [item.omix709_05_54_source_member for item in self.overlap_pairs]
        if any(len(values) != len(set(values)) for values in (normalized, scaleup, addendum, main)):
            raise ValueError("overlap pairs must be one-to-one in every identity namespace")
        disjoint = assertions.omix709_05_54_and_05_55_are_biological_embryo_disjoint
        if disjoint and self.overlap_pairs:
            raise ValueError("disjoint archives may not declare overlap pairs")
        if not disjoint and not self.overlap_pairs:
            raise ValueError("non-disjoint archives require exhaustive overlap pairs")
        return self


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {source}")
    return value


def load_packet(path: str | Path) -> OutreachPacket:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"outreach packet is missing: {source}")
    return OutreachPacket.model_validate_json(source.read_text(encoding="utf-8"))


def _verify_current_state_file(
    repository_root: Path,
    binding: CurrentStateFile,
) -> tuple[Path, dict[str, Any]]:
    path = (repository_root / binding.relative_path).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as error:
        raise ValueError("current-state path escapes the repository root") from error
    if not path.is_file():
        raise FileNotFoundError(f"current-state file is missing: {path}")
    if path.stat().st_size != binding.bytes:
        raise ValueError(f"current-state file byte identity differs: {binding.relative_path}")
    if _sha256(path) != binding.sha256:
        raise ValueError(f"current-state file SHA-256 differs: {binding.relative_path}")
    return path, _read_json(path)


def _safe_candidate_response_paths(repository_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    config_root = repository_root / "config" / "real"
    incoming_root = repository_root / "build" / "omix709-source-curator-incoming"
    if config_root.is_dir():
        candidates.update(config_root.glob("omix709_source_curator_response_*.json"))
        candidates.update(config_root.glob("omix709_pilot_overlap_receipt_*.json"))
    if incoming_root.is_dir():
        candidates.update(incoming_root.glob("*.json"))
    return sorted(
        path.resolve()
        for path in candidates
        if "_template_" not in path.name and not path.name.endswith("_template.json")
    )


def audit_current_state(
    packet_path: str | Path,
    *,
    observed_utc: str,
) -> dict[str, Any]:
    if UTC_RE.fullmatch(observed_utc) is None:
        raise ValueError("audit observation time must be UTC ISO-8601")
    packet_source = Path(packet_path).resolve()
    packet = load_packet(packet_source)
    repository_root = packet_source.parents[2]
    verified: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for role, binding in packet.current_state_files.items():
        path, payload = _verify_current_state_file(repository_root, binding)
        payloads[role] = payload
        verified[role] = {
            "relative_path": path.relative_to(repository_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    request = payloads["attestation_request"]
    if request.get("status") != "awaiting_source_curator_attestation":
        raise ValueError("attestation request is not in the frozen awaiting state")
    if set(request.get("requested_assertions", {})) != set(ASSERTION_KEYS):
        raise ValueError("attestation request assertion set differs")
    if any(value is not None for value in request["requested_assertions"].values()):
        raise ValueError("attestation request now contains an unreviewed assertion answer")
    required_response = request.get("required_response", {})
    if any(
        required_response.get(key) is not None
        for key in (
            "attestor_name",
            "attestor_role_and_source_authority",
            "attested_utc",
            "evidence_record_identifier",
            "evidence_record_sha256",
        )
    ) or required_response.get("overlap_pairs") != []:
        raise ValueError("attestation request now contains an unreviewed response")

    raw = payloads["raw_identity_qualification"]
    if (
        raw.get("current_overlap_contract_closed") is not False
        or raw.get("gates", {}).get("source_curator_attestation_received") is not False
        or raw.get("gates", {}).get("real_model_fit_permitted") is not False
        or raw.get("gates", {}).get("outcome_values_read") is not False
        or raw.get("gates", {}).get("sealed_test_opened") is not False
    ):
        raise ValueError("raw identity qualification no longer records the expected open gate")

    overlap = payloads["open_overlap_status"]
    if (
        overlap.get("receipt_supplied") is not False
        or overlap.get("embryo_level_disjointness_established") is not False
        or overlap.get("model_launch_gate")
        != "blocked_pending_pilot_embryo_overlap_receipt"
    ):
        raise ValueError("stage-adapter overlap status no longer records the expected open gate")

    readiness = payloads["developmental_readiness_contract"]
    if (
        readiness.get("validated") is not False
        or readiness.get("outcome_values_permitted") is not False
        or readiness.get("sealed_test_access_permitted") is not False
        or readiness.get("baseline", {}).get("real_fit_permitted_while_overlap_open") is not False
    ):
        raise ValueError("developmental readiness contract no longer records the expected block")

    candidates = _safe_candidate_response_paths(repository_root)
    if candidates:
        listed = ", ".join(path.relative_to(repository_root).as_posix() for path in candidates)
        raise ValueError(
            "candidate curator response exists and requires explicit evidence validation: "
            + listed
        )
    return {
        "schema_version": HANDOFF_VERSION,
        "observed_utc": observed_utc,
        "classification": "identity_and_rights_gate_audit_no_outcome_access",
        "outreach_packet_sha256": _sha256(packet_source),
        "candidate_response_paths": [
            path.relative_to(repository_root).as_posix() for path in candidates
        ],
        "valid_source_curator_response_found": False,
        "verified_open_state_files": verified,
        "identity_gate_status": "blocked_no_source_curator_response",
        "rights_gate_status": "blocked_no_explicit_external_compute_or_training_permission",
        "external_compute_training_launch_permitted": False,
        "model_launch_permitted": False,
        "partial_or_sealed_outcomes_accessed": False,
        "blockers": [
            "source_curator_response_missing",
            "source_identity_authority_missing",
            "data_use_rights_authority_missing",
            "durable_evidence_record_missing",
            "four_identity_assertions_unanswered",
            "overlap_pair_mapping_unresolved",
            "external_compute_permissions_unanswered",
            "model_training_and_weight_retention_permissions_unanswered",
        ],
    }


def _split_identity_sets(split_manifest: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    claim = {
        embryo_id
        for gene in split_manifest["claim_bearing_genes"]
        for embryo_id in gene["embryo_ids"]
    }
    exposed = {
        embryo_id
        for gene in split_manifest["development_only_exposed_pilot_genes"]
        for embryo_id in gene["embryo_ids"]
    }
    if claim & exposed:
        raise ValueError("split manifest has overlapping claim-bearing and exposed embryos")
    return claim, exposed


def _derived_overlap_receipt(response: CuratorRightsResponse) -> dict[str, Any]:
    bindings = response.source_bindings
    payload = {
        "schema_version": OVERLAP_RECEIPT_VERSION,
        "pilot_normalized_sha256": bindings.pilot_normalized_product_sha256,
        "pilot_normalized_bytes": bindings.pilot_normalized_product_bytes,
        "pilot_embryos": bindings.pilot_embryos,
        "scaleup_source_bundle_sha256": bindings.scaleup_source_bundle_sha256,
        "attestation_request_sha256": bindings.attestation_request_sha256,
        "source_evidence_record_identifier": response.durable_evidence.record_identifier,
        "source_evidence_record_bytes": response.durable_evidence.bytes,
        "source_evidence_record_sha256": response.durable_evidence.sha256,
        "attestor_name": response.identity_attestor.name,
        "attestor_role_and_source_authority": (
            f"{response.identity_attestor.role}; "
            f"{response.identity_attestor.organization}; "
            f"{response.identity_attestor.authority_basis}"
        ),
        "attested_utc": response.identity_attestor.attested_utc,
        "assertions": response.assertions.model_dump(mode="json"),
        "mapping_basis": response.mapping_basis,
        "mapping_complete": response.mapping_complete,
        "outcome_values_used_for_identity_decision": False,
        "pairs": [
            {
                "normalized_embryo_id": pair.normalized_embryo_id,
                "scaleup_embryo_id": pair.scaleup_embryo_id,
                "source_member": pair.omix709_05_55_source_member,
            }
            for pair in response.overlap_pairs
        ],
    }
    return PilotOverlapReceipt.model_validate(payload).model_dump(mode="json")


def validate_response(
    packet_path: str | Path,
    response_path: str | Path,
    evidence_path: str | Path,
    split_manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    packet_source = Path(packet_path).resolve()
    response_source = Path(response_path).resolve()
    evidence_source = Path(evidence_path).resolve()
    split_source = Path(split_manifest_path).resolve()
    packet = load_packet(packet_source)
    response = CuratorRightsResponse.model_validate_json(
        response_source.read_text(encoding="utf-8")
    )
    if response.outreach_packet_sha256 != _sha256(packet_source):
        raise ValueError("curator response is bound to a different outreach packet")
    if response.source_bindings != packet.source_bindings:
        raise ValueError("curator response is bound to different OMIX709 source identities")
    if not evidence_source.is_file():
        raise FileNotFoundError(f"durable evidence record is missing: {evidence_source}")
    if evidence_source.stat().st_size != response.durable_evidence.bytes:
        raise ValueError("durable evidence record byte identity differs")
    if _sha256(evidence_source) != response.durable_evidence.sha256:
        raise ValueError("durable evidence record SHA-256 differs")
    if _sha256(split_source) != packet.source_bindings.split_manifest_sha256:
        raise ValueError("split manifest SHA-256 differs from the frozen packet")
    split = _read_json(split_source)
    claim_ids, exposed_ids = _split_identity_sets(split)
    mapped = {pair.scaleup_embryo_id for pair in response.overlap_pairs}
    unknown = sorted(mapped - claim_ids - exposed_ids)
    if unknown:
        raise ValueError(f"overlap response contains unknown scale-up embryos: {unknown[:5]}")
    claim_overlap = sorted(mapped & claim_ids)
    if claim_overlap:
        raise ValueError(
            "pilot overlap reaches claim-bearing partitions: " + ", ".join(claim_overlap[:5])
        )

    permissions = response.permissions.model_dump(mode="json")
    denied = [
        key for key in LAUNCH_REQUIRED_PERMISSION_KEYS if permissions[key] is not True
    ]
    launch_permitted = not denied
    overlap_receipt = _derived_overlap_receipt(response)
    handoff = {
        "schema_version": HANDOFF_VERSION,
        "classification": "validated_identity_and_rights_handoff_no_outcome_access",
        "outreach_packet_sha256": _sha256(packet_source),
        "curator_response_sha256": _sha256(response_source),
        "durable_evidence_record_identifier": response.durable_evidence.record_identifier,
        "durable_evidence_record_bytes": response.durable_evidence.bytes,
        "durable_evidence_record_sha256": response.durable_evidence.sha256,
        "split_manifest_sha256": _sha256(split_source),
        "identity_attestor": response.identity_attestor.model_dump(mode="json"),
        "rights_attestor": response.rights_attestor.model_dump(mode="json"),
        "identity_gate_status": "closed_source_authorized_response_validated",
        "overlap_pairs": len(response.overlap_pairs),
        "claim_bearing_overlap_embryos": 0,
        "rights_gate_status": (
            "closed_explicit_permissions_granted"
            if launch_permitted
            else "blocked_explicit_permission_denial"
        ),
        "permissions": permissions,
        "launch_required_permissions": LAUNCH_REQUIRED_PERMISSION_KEYS,
        "denied_launch_required_permissions": denied,
        "external_compute_training_launch_permitted": launch_permitted,
        "model_launch_permitted": launch_permitted,
        "derived_overlap_receipt_sha256": (
            hashlib.sha256(_json_bytes(overlap_receipt)).hexdigest()
            if launch_permitted
            else None
        ),
        "partial_or_sealed_outcomes_accessed": False,
        "blockers": [f"permission_denied:{key}" for key in denied],
    }
    return handoff, overlap_receipt if launch_permitted else None


def _write_bundle(
    output_root: str | Path,
    handoff: Mapping[str, Any],
    overlap_receipt: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(output_root).resolve()
    if target.exists():
        raise FileExistsError(f"handoff output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        files: dict[str, bytes] = {
            "launch_handoff.json": _json_bytes(handoff),
        }
        if handoff["model_launch_permitted"] is True:
            if overlap_receipt is None:
                raise ValueError("eligible handoff requires a derived overlap receipt")
            files["pilot_overlap_receipt.json"] = _json_bytes(overlap_receipt)
            marker = "QUALIFIED"
        else:
            marker = "BLOCKED"
        files[marker] = b""
        for name, data in files.items():
            (temporary / name).write_bytes(data)
        checksum_lines = [
            f"{hashlib.sha256(data).hexdigest()}  {name}"
            for name, data in sorted(files.items())
        ]
        (temporary / "SHA256SUMS.txt").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n"
        )
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            for path in sorted(temporary.iterdir(), reverse=True):
                path.unlink()
            temporary.rmdir()
        raise
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit or validate the fail-closed OMIX709 curator/rights gate"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="freeze the current missing-response block")
    audit.add_argument("--packet", required=True)
    audit.add_argument("--observed-utc", required=True)
    audit.add_argument("--output-root", required=True)
    validate = commands.add_parser(
        "validate", help="validate a curator/rights response and write a launch handoff"
    )
    validate.add_argument("--packet", required=True)
    validate.add_argument("--response", required=True)
    validate.add_argument("--evidence-record", required=True)
    validate.add_argument("--split-manifest", required=True)
    validate.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "audit":
        handoff = audit_current_state(args.packet, observed_utc=args.observed_utc)
        output = _write_bundle(args.output_root, handoff)
    else:
        handoff, overlap = validate_response(
            args.packet,
            args.response,
            args.evidence_record,
            args.split_manifest,
        )
        output = _write_bundle(
            args.output_root,
            handoff,
            overlap if handoff["model_launch_permitted"] else None,
        )
    print(json.dumps({"output_root": str(output), **handoff}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
