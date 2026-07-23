"""Canonical, shared signatures for contextual matching and coverage products."""

from __future__ import annotations

import json
from typing import Any, Iterable

from .models import ExperimentalContext, Intervention, QuantitativeValue


def canonical_quantity_signature(quantity: QuantitativeValue) -> str:
    """Return an unambiguous, deterministic identity for a quantitative value."""
    payload = quantity.model_dump(mode="json", exclude_none=True)
    unit = payload.pop("unit", None)
    if unit:
        payload["unit"] = unit["id"]
    return _canonical_json(payload)


def canonical_intervention_signature(intervention: Intervention) -> str:
    """Return the canonical identity of one intervention, including dose and duration."""
    payload: dict[str, Any] = {"type": intervention.intervention_type}
    for field in ("target", "reagent", "route"):
        value = getattr(intervention, field)
        if value is not None:
            payload[field] = value.id
    if intervention.dose is not None:
        payload["dose"] = json.loads(canonical_quantity_signature(intervention.dose))
    if intervention.duration is not None:
        payload["duration"] = json.loads(
            canonical_quantity_signature(intervention.duration)
        )
    return _canonical_json(payload)


def canonical_intervention_set_signature(
    interventions: Iterable[Intervention],
) -> str:
    """Return an order-independent canonical identity for an intervention set."""
    signatures = sorted(
        canonical_intervention_signature(item) for item in interventions
    )
    return f"[{','.join(signatures)}]"


def context_values(context: ExperimentalContext) -> dict[str, tuple[str, ...]]:
    """Project an experimental context into canonical comparison values."""
    return {
        "organism_taxon": (context.organism_taxon.id,),
        "genetic_background": tuple(
            sorted({item.id for item in context.genetic_background})
        ),
        "life_stage": tuple(sorted({item.id for item in context.life_stage})),
        "anatomy": tuple(sorted({item.id for item in context.anatomy})),
        "intervention": tuple(
            sorted(
                canonical_intervention_signature(item)
                for item in context.interventions
            )
        ),
        "assay": tuple(sorted({item.id for item in context.assays})),
        "environment": tuple(sorted({item.id for item in context.environments})),
        "modality": tuple(sorted(set(context.modality))),
        "readout_resolution": tuple(
            sorted({item.value for item in context.readout_resolution})
        ),
        "absolute_time": (
            (canonical_quantity_signature(context.absolute_time),)
            if context.absolute_time is not None
            else ()
        ),
    }


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
