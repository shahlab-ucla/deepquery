"""Coverage products for the empirical data-transport map."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any, Iterable

from .io import validate_observation_collection
from .models import ContextualObservation
from .signatures import canonical_intervention_set_signature, context_values
from .transport import DIMENSIONS


UNSPECIFIED = "wormctx:unspecified"
CUBE_DIMENSIONS = (
    "organism_taxon",
    "genetic_background",
    "life_stage",
    "anatomy",
    "intervention_signature",
    "assay",
    "environment",
    "modality",
    "readout_resolution",
    "absolute_time",
)


def coverage_long(observations: Iterable[ContextualObservation]) -> list[dict[str, Any]]:
    observations = list(observations)
    validate_observation_collection(observations)
    rows: list[dict[str, Any]] = []
    for observation in observations:
        values = context_values(observation.context)
        gap_reasons = {
            gap.dimension: gap.reason.value for gap in observation.context.context_gaps
        }
        coarse_reasons = [
            reason.value for reason in observation.context.context_missing_reasons
        ]
        for dimension in DIMENSIONS:
            dimension_values = values[dimension]
            if dimension_values:
                for value in dimension_values:
                    rows.append(
                        {
                            "observation_id": observation.id,
                            "source_id": observation.provenance.source_id,
                            "source_release": observation.provenance.source_release,
                            "source_artifact": observation.provenance.source_artifact,
                            "source_checksum_sha256": observation.provenance.checksum_sha256,
                            "dimension": dimension,
                            "value_id": value,
                            "reportedness": "reported",
                            "missing_reason": "",
                        }
                    )
            else:
                reason = gap_reasons.get(
                    dimension, "|".join(coarse_reasons) or "not_reported"
                )
                reportedness = (
                    reason
                    if reason in {
                        "not_reported",
                        "not_applicable",
                        "ambiguous",
                        "explicitly_absent",
                    }
                    else "not_reported"
                )
                rows.append(
                    {
                        "observation_id": observation.id,
                        "source_id": observation.provenance.source_id,
                        "source_release": observation.provenance.source_release,
                        "source_artifact": observation.provenance.source_artifact,
                        "source_checksum_sha256": observation.provenance.checksum_sha256,
                        "dimension": dimension,
                        "value_id": UNSPECIFIED,
                        "reportedness": reportedness,
                        "missing_reason": reason,
                    }
                )
    return rows


def coverage_cube(observations: Iterable[ContextualObservation]) -> list[dict[str, Any]]:
    observations = list(observations)
    validate_observation_collection(observations)
    groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for observation in observations:
        values = context_values(observation.context)
        cube_values = {
            "organism_taxon": values["organism_taxon"],
            "genetic_background": values["genetic_background"],
            "life_stage": values["life_stage"],
            "anatomy": values["anatomy"],
            "intervention_signature": (_intervention_signature(observation),),
            "assay": values["assay"],
            "environment": values["environment"],
            "modality": values["modality"],
            "readout_resolution": values["readout_resolution"],
            "absolute_time": values["absolute_time"],
        }
        dimensions = [cube_values[name] or (UNSPECIFIED,) for name in CUBE_DIMENSIONS]
        for combination in product(*dimensions):
            key = (
                observation.provenance.source_id,
                observation.provenance.source_release,
                observation.provenance.source_artifact,
                observation.provenance.checksum_sha256 or "",
                *combination,
            )
            groups[key].add(observation.id)

    rows = []
    for key, observation_ids in sorted(groups.items()):
        source_id, source_release, source_artifact, source_checksum, *combination = key
        row: dict[str, Any] = {
            "source_id": source_id,
            "source_release": source_release,
            "source_artifact": source_artifact,
            "source_checksum_sha256": source_checksum or None,
        }
        row.update(dict(zip(CUBE_DIMENSIONS, combination, strict=True)))
        row["observation_count"] = len(observation_ids)
        row["observation_ids"] = sorted(observation_ids)
        rows.append(row)
    return rows


def _intervention_signature(observation: ContextualObservation) -> str:
    if not observation.context.interventions:
        return UNSPECIFIED
    return canonical_intervention_set_signature(observation.context.interventions)
