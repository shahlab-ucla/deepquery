"""Small deterministic I/O helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from pydantic import BaseModel

from .models import (
    ContextualObservation,
    ExperimentalContext,
    NamedReference,
    TransportQuery,
)


class CollectionIntegrityError(ValueError):
    """A collection-level identifier or provenance invariant was violated."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_observation_collection(
    observations: Sequence[ContextualObservation],
) -> None:
    """Validate identities that cannot be checked one record at a time.

    Structural graph nodes are globally unique. Entity references may be reused
    only when their complete normalized identity is the same. Shared provenance
    and generated activity/snapshot nodes must also be semantically identical.
    """

    registry: dict[str, tuple[str, Any, bool]] = {}
    snapshot_by_key: dict[tuple[str, str, str], str] = {}

    def register(
        identifier: str,
        kind: str,
        payload: Any,
        *,
        reusable: bool = False,
        duplicate_code: str = "incompatible_identifier_collision",
    ) -> None:
        existing = registry.get(identifier)
        if existing is None:
            registry[identifier] = (kind, payload, reusable)
            return
        existing_kind, existing_payload, existing_reusable = existing
        if (
            reusable
            and existing_reusable
            and existing_kind == kind
            and _semantic_fingerprint(existing_payload) == _semantic_fingerprint(payload)
        ):
            return
        if existing_kind == kind and not reusable and not existing_reusable:
            raise CollectionIntegrityError(
                duplicate_code,
                f"duplicate {kind.replace('_', ' ')} id: {identifier}",
            )
        raise CollectionIntegrityError(
            "incompatible_identifier_collision",
            f"identifier {identifier!r} is reused incompatibly as "
            f"{existing_kind!r} and {kind!r}",
        )

    def register_reference(reference: NamedReference) -> None:
        identity = {
            "label": reference.label,
            "categories": sorted(set(reference.categories)),
        }
        existing = registry.get(reference.id)
        if existing is None:
            registry[reference.id] = ("entity_reference", identity, True)
            return
        existing_kind, existing_identity, existing_reusable = existing
        if existing_kind != "entity_reference" or not existing_reusable:
            raise CollectionIntegrityError(
                "incompatible_identifier_collision",
                f"identifier {reference.id!r} is reused incompatibly as "
                f"{existing_kind!r} and 'entity_reference'",
            )
        old_label = existing_identity.get("label")
        new_label = identity.get("label")
        if old_label and new_label and old_label != new_label:
            raise CollectionIntegrityError(
                "incompatible_entity_reuse",
                f"entity reference {reference.id!r} has conflicting labels "
                f"{old_label!r} and {new_label!r}",
            )
        old_categories = set(existing_identity.get("categories", []))
        new_categories = set(identity.get("categories", []))
        if old_categories and new_categories and old_categories != new_categories:
            raise CollectionIntegrityError(
                "incompatible_entity_reuse",
                f"entity reference {reference.id!r} has conflicting categories "
                f"{sorted(old_categories)!r} and {sorted(new_categories)!r}",
            )
        merged = {
            "label": old_label or new_label,
            "categories": sorted(old_categories | new_categories),
        }
        registry[reference.id] = ("entity_reference", merged, True)

    for observation in observations:
        register(
            observation.id,
            "observation",
            observation.model_dump(mode="json", exclude_none=True),
            duplicate_code="duplicate_observation_id",
        )

        for reference in _iter_named_references(observation):
            register_reference(reference)

        for index, intervention in enumerate(observation.context.interventions, start=1):
            register(
                f"{observation.id}/intervention/{index}",
                "generated_intervention",
                intervention.model_dump(mode="json", exclude_none=True),
            )

        for evidence in observation.evidence_lines:
            register(
                evidence.id,
                "evidence_line",
                evidence.model_dump(mode="json", exclude_none=True),
                duplicate_code="duplicate_evidence_id",
            )

        if observation.result:
            result = observation.result
            register(
                result.id,
                "observation_result",
                result.model_dump(mode="json", exclude_none=True),
                duplicate_code="duplicate_result_id",
            )
            for measurement in result.measurements:
                register(
                    measurement.id,
                    "measurement",
                    measurement.model_dump(mode="json", exclude_none=True),
                    duplicate_code="duplicate_measurement_id",
                )
            for group in result.comparison_groups:
                register(
                    group.id,
                    "comparison_group",
                    group.model_dump(mode="json", exclude_none=True),
                    duplicate_code="duplicate_comparison_group_id",
                )
            for asset in result.asset_references:
                register(
                    asset.id,
                    "result_asset",
                    asset.model_dump(mode="json", exclude_none=True),
                    duplicate_code="duplicate_result_asset_id",
                )

        provenance = observation.provenance.model_dump(mode="json", exclude_none=True)
        snapshot_key = (
            observation.provenance.source_id,
            observation.provenance.source_release,
            observation.provenance.source_artifact,
        )
        snapshot_fingerprint = _semantic_fingerprint(provenance)
        previous_snapshot = snapshot_by_key.get(snapshot_key)
        if previous_snapshot is not None and previous_snapshot != snapshot_fingerprint:
            raise CollectionIntegrityError(
                "inconsistent_source_snapshot",
                "observations sharing source_id/source_release/source_artifact must "
                "have identical snapshot provenance",
            )
        snapshot_by_key[snapshot_key] = snapshot_fingerprint
        register(
            _generated_snapshot_id(*snapshot_key),
            "source_snapshot",
            provenance,
            reusable=True,
        )

        if observation.provenance.transformation_activity:
            register(
                observation.provenance.transformation_activity,
                "transformation_activity",
                {"transformer_version": observation.provenance.transformer_version},
                reusable=True,
            )
        if observation.provenance.transformer_version:
            version = observation.provenance.transformer_version
            register(
                _generated_software_agent_id(version),
                "software_agent",
                {"transformer_version": version},
                reusable=True,
            )


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_text(target, payload)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object on {path}:{line_number}")
            yield value


def load_observations(path: str | Path) -> list[ContextualObservation]:
    observations = [ContextualObservation.model_validate(row) for row in iter_jsonl(path)]
    validate_observation_collection(observations)
    return observations


def load_query(path: str | Path) -> TransportQuery:
    data = read_json(path)
    if "query_id" in data:
        return TransportQuery.model_validate(data)
    return TransportQuery(
        query_id="wormctx:anonymous-query",
        context=ExperimentalContext.model_validate(data),
    )


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        if hasattr(row, "model_dump"):
            row = row.model_dump(mode="json", exclude_none=True)
        lines.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
    _atomic_text(target, "\n".join(lines) + ("\n" if lines else ""))


def write_table(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    delimiter: str = "\t",
    fieldnames: Sequence[str] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)
        fieldnames = ordered
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=target.parent
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            delimiter=delimiter,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(value) for key, value in row.items()})
        temporary = Path(handle.name)
    os.replace(temporary, target)


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict, set)):
        if isinstance(value, set):
            value = sorted(value)
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def _atomic_text(target: Path, payload: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=target.parent
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def _iter_named_references(value: Any) -> Iterator[NamedReference]:
    if isinstance(value, NamedReference):
        yield value
        return
    if isinstance(value, BaseModel):
        for field_name in value.__class__.model_fields:
            yield from _iter_named_references(getattr(value, field_name))
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_named_references(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_named_references(item)


def _semantic_fingerprint(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _generated_snapshot_id(source_id: str, release: str, artifact: str) -> str:
    raw = "\x1f".join((source_id, release, artifact))
    return f"wormctx:snapshot-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _generated_software_agent_id(version: str) -> str:
    digest = hashlib.sha256(version.encode("utf-8")).hexdigest()[:16]
    return f"wormctx:software-agent-{digest}"
