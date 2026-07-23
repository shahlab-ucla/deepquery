"""Offline reference adapter used for fixtures and hand-curated bridge bundles."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Iterator

from pydantic import ValidationError

from ..io import (
    CollectionIntegrityError,
    iter_jsonl,
    validate_observation_collection,
)
from ..models import ContextualObservation, DatasetProfile
from .base import Severity, SnapshotContext, ValidationIssue, ValidationReport


class LocalJsonlAdapter:
    source_id = "wormctx-local-jsonl"
    adapter_version = "0.1.0"

    def validate_raw(self, snapshot: Path | SnapshotContext) -> ValidationReport:
        path = self._path(snapshot)
        report = ValidationReport(
            source_id=self.source_id,
            snapshot_id=(snapshot.snapshot_id if isinstance(snapshot, SnapshotContext) else None),
            adapter_version=self.adapter_version,
        )
        observations: list[ContextualObservation] = []
        try:
            for line_number, row in enumerate(iter_jsonl(path), start=1):
                try:
                    observation = ContextualObservation.model_validate(row)
                    report.checked_records += 1
                    observations.append(observation)
                except ValidationError as exc:
                    report.add(
                        ValidationIssue(
                            severity=Severity.error,
                            code="schema_validation",
                            message=str(exc),
                            record_locator=f"line:{line_number}",
                        )
                    )
        except ValueError as exc:
            report.add(
                ValidationIssue(
                    severity=Severity.error,
                    code="invalid_jsonl",
                    message=str(exc),
                )
            )
        if observations and not any(
            issue.severity is Severity.error and issue.code == "schema_validation"
            for issue in report.issues
        ):
            try:
                validate_observation_collection(observations)
            except CollectionIntegrityError as exc:
                report.add(
                    ValidationIssue(
                        severity=Severity.error,
                        code=exc.code,
                        message=str(exc),
                    )
                )
        if report.checked_records == 0 and not report.issues:
            report.add(
                ValidationIssue(
                    severity=Severity.error,
                    code="empty_snapshot",
                    message="the JSONL snapshot contains no observation records",
                )
            )
        return report

    def iter_observations(
        self, snapshot: Path | SnapshotContext
    ) -> Iterator[ContextualObservation]:
        observations = [
            ContextualObservation.model_validate(row)
            for row in iter_jsonl(self._path(snapshot))
        ]
        validate_observation_collection(observations)
        yield from observations

    def dataset_profiles(
        self, snapshot: Path | SnapshotContext
    ) -> Iterator[DatasetProfile]:
        grouped: dict[tuple[str, str, str], list[ContextualObservation]] = defaultdict(list)
        for observation in self.iter_observations(snapshot):
            key = (
                observation.provenance.source_id,
                observation.provenance.source_release,
                observation.provenance.source_artifact,
            )
            grouped[key].append(observation)
        for (source_id, release, artifact), observations in sorted(grouped.items()):
            contexts = [item.context for item in observations]
            yield DatasetProfile(
                dataset_id=f"wormctx:dataset-{source_id}-{release}-{artifact}",
                source_id=source_id,
                source_release=release,
                source_artifact=artifact,
                taxon=contexts[0].organism_taxon.id,
                genetic_background=sorted(
                    {item.id for context in contexts for item in context.genetic_background}
                ),
                life_stage=sorted({item.id for context in contexts for item in context.life_stage}),
                intervention_type=sorted(
                    {
                        item.intervention_type
                        for context in contexts
                        for item in context.interventions
                    }
                ),
                perturbation_target=sorted(
                    {
                        item.target.id
                        for context in contexts
                        for item in context.interventions
                        if item.target
                    }
                ),
                compound_or_reagent=sorted(
                    {
                        item.reagent.id
                        for context in contexts
                        for item in context.interventions
                        if item.reagent
                    }
                ),
                assay=sorted({item.id for context in contexts for item in context.assays}),
                anatomy=sorted({item.id for context in contexts for item in context.anatomy}),
                modality=sorted({item for context in contexts for item in context.modality}),
                readout_resolution=sorted(
                    {item.value for context in contexts for item in context.readout_resolution}
                ),
                observation_count=len(observations),
                direct_or_derived=(
                    "derived"
                    if any(
                        item.record_origin.value in {"computed", "extracted"}
                        for item in observations
                    )
                    else "direct"
                ),
                missing_context={
                    gap.dimension: gap.reason.value
                    for context in contexts
                    for gap in context.context_gaps
                },
            )

    @staticmethod
    def _path(snapshot: Path | SnapshotContext) -> Path:
        if isinstance(snapshot, SnapshotContext):
            enabled = [item for item in snapshot.manifest.artifacts if item.enabled]
            if len(enabled) != 1:
                raise ValueError("LocalJsonlAdapter requires exactly one enabled artifact")
            return snapshot.artifact_path(enabled[0].artifact_id)
        return Path(snapshot)
