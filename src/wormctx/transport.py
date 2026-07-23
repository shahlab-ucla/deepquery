"""Transparent context compatibility scoring for evidence transport."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Mapping

from .models import ContextualObservation, ExperimentalContext
from .signatures import (
    canonical_intervention_signature,
    canonical_quantity_signature,
    context_values,
)


DIMENSIONS = (
    "organism_taxon",
    "genetic_background",
    "life_stage",
    "anatomy",
    "intervention",
    "assay",
    "environment",
    "modality",
    "readout_resolution",
    "absolute_time",
)


@dataclass(frozen=True)
class DimensionScore:
    dimension: str
    state: str
    score: float | None
    weight: float
    evidence_values: tuple[str, ...]
    query_values: tuple[str, ...]
    matched_pair: tuple[str, str] | None = None
    matched_pairs: tuple[tuple[str, str], ...] = ()
    unmatched_query_values: tuple[str, ...] = ()
    hard_failure: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "state": self.state,
            "score": self.score,
            "weight": self.weight,
            "evidence_values": list(self.evidence_values),
            "query_values": list(self.query_values),
            "matched_pair": list(self.matched_pair) if self.matched_pair else None,
            "matched_pairs": [list(pair) for pair in self.matched_pairs],
            "unmatched_query_values": list(self.unmatched_query_values),
            "hard_failure": self.hard_failure,
        }


def score_observation(
    observation: ContextualObservation,
    query_context: ExperimentalContext,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    validate_policy(policy)
    evidence = context_values(observation.context)
    query = context_values(query_context)
    components: list[DimensionScore] = []
    weighted_total = 0.0
    total_weight = 0.0

    for dimension in DIMENSIONS:
        settings = policy["dimensions"][dimension]
        weight = float(settings["weight"])
        source_values = evidence[dimension]
        target_values = query[dimension]
        if not target_values:
            components.append(
                DimensionScore(
                    dimension=dimension,
                    state="not_requested",
                    score=None,
                    weight=weight,
                    evidence_values=source_values,
                    query_values=target_values,
                )
            )
            continue
        if not source_values:
            value = float(settings["unknown_score"])
            component = DimensionScore(
                dimension=dimension,
                state="unknown",
                score=value,
                weight=weight,
                evidence_values=source_values,
                query_values=target_values,
                unmatched_query_values=target_values,
                hard_failure=bool(settings.get("hard_gate", False)),
            )
        elif dimension == "intervention":
            component = _score_intervention_set(
                observation.context,
                query_context,
                weight,
                settings,
            )
        else:
            component = _best_pair(
                dimension,
                source_values,
                target_values,
                weight,
                settings,
                policy.get("compatibility", {}).get(dimension, {}),
            )
        components.append(component)
        if component.score is not None:
            weighted_total += component.score * weight
            total_weight += weight

    hard_failures = [item.dimension for item in components if item.hard_failure]
    score = weighted_total / total_weight if total_weight else 0.0
    thresholds = policy["thresholds"]
    if hard_failures:
        classification = "nontransportable"
    elif score >= float(thresholds["direct"]) and all(
        item.state in {"exact", "not_requested"} for item in components
    ):
        classification = "direct"
    elif score >= float(thresholds["near_context"]):
        classification = "near_context"
    elif score >= float(thresholds["transported"]):
        classification = "transported"
    else:
        classification = "low_context"

    known = sum(
        1
        for item in components
        if item.state not in {"unknown", "not_requested"}
    )
    requested = sum(1 for item in components if item.state != "not_requested")
    return {
        "observation_id": observation.id,
        "source_id": observation.provenance.source_id,
        "score": round(score, 6),
        "classification": classification,
        "known_context_fraction": round(known / requested, 6) if requested else 0.0,
        "hard_failures": hard_failures,
        "components": [item.as_dict() for item in components],
        "policy_id": policy.get("policy_id"),
    }


def validate_policy(policy: Mapping[str, Any]) -> None:
    required = {"policy_id", "thresholds", "dimensions"}
    missing = required.difference(policy)
    if missing:
        raise ValueError(f"transport policy is missing fields: {sorted(missing)}")
    dimensions = policy["dimensions"]
    if set(dimensions) != set(DIMENSIONS):
        missing_dimensions = sorted(set(DIMENSIONS).difference(dimensions))
        extra_dimensions = sorted(set(dimensions).difference(DIMENSIONS))
        raise ValueError(
            "transport policy dimension mismatch: "
            f"missing={missing_dimensions}, extra={extra_dimensions}"
        )
    thresholds = policy["thresholds"]
    try:
        transported_value = thresholds["transported"]
        near_value = thresholds["near_context"]
        direct_value = thresholds["direct"]
    except (KeyError, TypeError) as exc:
        raise ValueError("transport thresholds must contain numeric values") from exc
    transported = _finite_float(transported_value, "transported")
    near = _finite_float(near_value, "near_context")
    direct = _finite_float(direct_value, "direct")
    if not (0.0 <= transported <= near <= direct <= 1.0):
        raise ValueError("thresholds must satisfy 0 <= transported <= near <= direct <= 1")
    for name, settings in dimensions.items():
        try:
            weight_value = settings["weight"]
            unknown_value = settings["unknown_score"]
            mismatch_value = settings["mismatch_score"]
            hard_gate = settings["hard_gate"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid settings for transport dimension {name!r}") from exc
        weight = _finite_float(weight_value, f"{name}.weight")
        unknown = _finite_float(unknown_value, f"{name}.unknown_score")
        mismatch = _finite_float(mismatch_value, f"{name}.mismatch_score")
        if weight <= 0:
            raise ValueError(f"transport weight must be positive for {name!r}")
        if not (0.0 <= unknown <= 1.0 and 0.0 <= mismatch <= 1.0):
            raise ValueError(f"transport scores must be in [0, 1] for {name!r}")
        if not isinstance(hard_gate, bool):
            raise ValueError(f"hard_gate must be boolean for {name!r}")
    for dimension, pairs in policy.get("compatibility", {}).items():
        if dimension not in dimensions:
            raise ValueError(f"compatibility references unknown dimension {dimension!r}")
        for pair, value in pairs.items():
            if pair.count("|") != 1:
                raise ValueError(
                    f"compatibility key must be 'evidence|query', received {pair!r}"
                )
            score = _finite_float(value, f"compatibility[{pair}]")
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"compatibility score must be in [0, 1] for {pair!r}")


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite numeric value") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite numeric value")
    return number


def _best_pair(
    dimension: str,
    source_values: tuple[str, ...],
    target_values: tuple[str, ...],
    weight: float,
    settings: Mapping[str, Any],
    compatibility: Mapping[str, Any],
) -> DimensionScore:
    """Score one-to-one evidence coverage of every requested target value."""
    mismatch_score = float(settings["mismatch_score"])
    pair_results: list[list[tuple[float, str]]] = []
    for target in target_values:
        target_results: list[tuple[float, str]] = []
        for source in source_values:
            if source == target:
                target_results.append((1.0, "exact"))
                continue
            forward = f"{source}|{target}"
            if forward in compatibility:
                target_results.append((float(compatibility[forward]), "compatible"))
            else:
                target_results.append((mismatch_score, "mismatch"))
        pair_results.append(target_results)

    @lru_cache(maxsize=None)
    def assign(
        target_index: int, used_mask: int
    ) -> tuple[float, int, tuple[int, ...]]:
        if target_index == len(target_values):
            return 0.0, 0, ()
        tail_score, tail_quality, tail_assignment = assign(
            target_index + 1, used_mask
        )
        best_result = (
            mismatch_score + tail_score,
            tail_quality,
            (-1, *tail_assignment),
        )
        for source_index in range(len(source_values)):
            bit = 1 << source_index
            if used_mask & bit:
                continue
            pair_score, pair_state = pair_results[target_index][source_index]
            tail_score, tail_quality, tail_assignment = assign(
                target_index + 1, used_mask | bit
            )
            quality = {"mismatch": 0, "compatible": 1, "exact": 2}[pair_state]
            candidate = (
                pair_score + tail_score,
                quality + tail_quality,
                (source_index, *tail_assignment),
            )
            if candidate[:2] > best_result[:2]:
                best_result = candidate
        return best_result

    total, _, assignment = assign(0, 0)
    selected_states: list[str] = []
    matched_pairs: list[tuple[str, str]] = []
    unmatched: list[str] = []
    for target_index, source_index in enumerate(assignment):
        target = target_values[target_index]
        if source_index < 0:
            selected_states.append("mismatch")
            unmatched.append(target)
            continue
        source = source_values[source_index]
        selected_states.append(pair_results[target_index][source_index][1])
        matched_pairs.append((source, target))

    score = total / len(target_values)
    if all(item == "exact" for item in selected_states):
        state = "exact"
        score = 1.0
    elif all(item in {"exact", "compatible"} for item in selected_states):
        state = "compatible"
    else:
        state = "mismatch"
    hard_failure = bool(settings.get("hard_gate", False)) and state == "mismatch"
    return DimensionScore(
        dimension=dimension,
        state=state,
        score=score,
        weight=weight,
        evidence_values=source_values,
        query_values=target_values,
        matched_pair=matched_pairs[0] if len(matched_pairs) == 1 else None,
        matched_pairs=tuple(matched_pairs),
        unmatched_query_values=tuple(unmatched),
        hard_failure=hard_failure,
    )


def _score_intervention_set(
    evidence_context: ExperimentalContext,
    query_context: ExperimentalContext,
    weight: float,
    settings: Mapping[str, Any],
) -> DimensionScore:
    evidence_items = evidence_context.interventions
    query_items = query_context.interventions
    evidence_values = tuple(
        sorted(canonical_intervention_signature(item) for item in evidence_items)
    )
    query_values = tuple(
        sorted(canonical_intervention_signature(item) for item in query_items)
    )
    if not query_items:
        return DimensionScore(
            dimension="intervention",
            state="not_requested",
            score=None,
            weight=weight,
            evidence_values=evidence_values,
            query_values=query_values,
        )
    if not evidence_items:
        return DimensionScore(
            dimension="intervention",
            state="unknown",
            score=float(settings["unknown_score"]),
            weight=weight,
            evidence_values=evidence_values,
            query_values=query_values,
            unmatched_query_values=query_values,
            hard_failure=bool(settings.get("hard_gate", False)),
        )

    pair_results = [
        [
            _intervention_pair_result(evidence, query, settings)
            for evidence in evidence_items
        ]
        for query in query_items
    ]
    pair_scores = [
        [result[0] for result in query_results] for query_results in pair_results
    ]

    @lru_cache(maxsize=None)
    def best(query_index: int, used_mask: int) -> float:
        if query_index == len(query_items):
            return 0.0
        best_score = best(query_index + 1, used_mask)
        for evidence_index in range(len(evidence_items)):
            bit = 1 << evidence_index
            if used_mask & bit:
                continue
            candidate = pair_scores[query_index][evidence_index] + best(
                query_index + 1, used_mask | bit
            )
            best_score = max(best_score, candidate)
        return best_score

    score = best(0, 0) / len(query_items)
    extra_interventions = max(0, len(evidence_items) - len(query_items))
    score *= 0.85**extra_interventions
    if score >= 0.999999 and len(evidence_items) == len(query_items):
        state = "exact"
        score = 1.0
    elif score > float(settings["mismatch_score"]):
        state = "compatible"
    else:
        state = "mismatch"
        score = float(settings["mismatch_score"])
    hard_failure = bool(
        settings.get("hard_gate", False)
    ) and not _has_nonconflicting_assignment(pair_results)
    return DimensionScore(
        dimension="intervention",
        state=state,
        score=score,
        weight=weight,
        evidence_values=evidence_values,
        query_values=query_values,
        hard_failure=hard_failure,
    )


def _intervention_pair_result(
    evidence: Any, query: Any, settings: Mapping[str, Any]
) -> tuple[float, bool]:
    type_mismatch = evidence.intervention_type != query.intervention_type
    comparisons: list[float] = [
        float(settings["mismatch_score"]) if type_mismatch else 1.0
    ]
    explicit_mismatch = type_mismatch
    for field in ("target", "reagent", "route"):
        query_value = getattr(query, field)
        if query_value is None:
            continue
        evidence_value = getattr(evidence, field)
        if evidence_value is None:
            comparisons.append(float(settings["unknown_score"]))
        elif evidence_value.id == query_value.id:
            comparisons.append(1.0)
        else:
            comparisons.append(float(settings["mismatch_score"]))
            explicit_mismatch = True
    for field in ("dose", "duration"):
        query_value = getattr(query, field)
        if query_value is None:
            continue
        evidence_value = getattr(evidence, field)
        if evidence_value is None:
            comparisons.append(float(settings["unknown_score"]))
        elif canonical_quantity_signature(
            evidence_value
        ) == canonical_quantity_signature(query_value):
            comparisons.append(1.0)
        else:
            comparisons.append(float(settings["mismatch_score"]))
            explicit_mismatch = True
    return sum(comparisons) / len(comparisons), explicit_mismatch


def _has_nonconflicting_assignment(
    pair_results: list[list[tuple[float, bool]]],
) -> bool:
    """Check whether every query intervention has a distinct, non-conflicting match."""
    if not pair_results:
        return True
    evidence_count = len(pair_results[0])

    @lru_cache(maxsize=None)
    def assign(query_index: int, used_mask: int) -> bool:
        if query_index == len(pair_results):
            return True
        for evidence_index in range(evidence_count):
            bit = 1 << evidence_index
            if used_mask & bit or pair_results[query_index][evidence_index][1]:
                continue
            if assign(query_index + 1, used_mask | bit):
                return True
        return False

    return assign(0, 0)
