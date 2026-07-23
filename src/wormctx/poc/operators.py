"""Typed, deterministic operators for the integrated-inference proof of concept.

This module deliberately contains no learned-model or GPU dependency.  It is the
small trusted core that validates proposed designs and performs probability and
experiment-selection calculations.  Inputs are ordinary mappings (or objects
with matching attributes), so the functions can consume contract models without
depending on their implementation module.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Any


_PROBABILITY_TOLERANCE = 1e-9
_MISSING = object()


class OperatorName(str, Enum):
    """The fixed ten-operator vocabulary proposed for the first prototype."""

    DERIVE_TRAIT = "derive_trait"
    FIT_REACTION_NORM = "fit_reaction_norm"
    FIT_KINSHIP_AND_HAPLOTYPE_MODEL = "fit_kinship_and_haplotype_model"
    SCORE_PANGENOME_SEQUENCE_CHANGE = "score_pangenome_sequence_change"
    INFER_DEVELOPMENTAL_STATE = "infer_developmental_state"
    ESTIMATE_PERTURBATION_EFFECT = "estimate_perturbation_effect"
    ASSESS_EVIDENCE_DETECTABILITY = "assess_evidence_detectability"
    RETRIEVE_CONTEXTUAL_EVIDENCE = "retrieve_contextual_evidence"
    UPDATE_HYPOTHESIS_GRAPH = "update_hypothesis_graph"
    COMPUTE_EXPECTED_INFORMATION_GAIN = "compute_expected_information_gain"


@dataclass(frozen=True, slots=True)
class OperatorSignature:
    """A lightweight type signature for an inference operator."""

    name: OperatorName
    input_types: tuple[str, ...]
    output_type: str
    deterministic: bool


OPERATOR_VOCABULARY: Mapping[OperatorName, OperatorSignature] = MappingProxyType(
    {
        OperatorName.DERIVE_TRAIT: OperatorSignature(
            OperatorName.DERIVE_TRAIT,
            ("EpisodeGraph",),
            "TraitGraph",
            True,
        ),
        OperatorName.FIT_REACTION_NORM: OperatorSignature(
            OperatorName.FIT_REACTION_NORM,
            ("TraitGraph", "DesignGraph"),
            "ModelResult",
            True,
        ),
        OperatorName.FIT_KINSHIP_AND_HAPLOTYPE_MODEL: OperatorSignature(
            OperatorName.FIT_KINSHIP_AND_HAPLOTYPE_MODEL,
            ("TraitGraph", "GenotypeGraph", "DesignGraph"),
            "ModelResult",
            True,
        ),
        OperatorName.SCORE_PANGENOME_SEQUENCE_CHANGE: OperatorSignature(
            OperatorName.SCORE_PANGENOME_SEQUENCE_CHANGE,
            ("EpisodeGraph", "LocusGraph"),
            "EvidenceResult",
            True,
        ),
        OperatorName.INFER_DEVELOPMENTAL_STATE: OperatorSignature(
            OperatorName.INFER_DEVELOPMENTAL_STATE,
            ("EpisodeGraph",),
            "StateEstimate",
            False,
        ),
        OperatorName.ESTIMATE_PERTURBATION_EFFECT: OperatorSignature(
            OperatorName.ESTIMATE_PERTURBATION_EFFECT,
            ("EpisodeGraph", "DesignGraph"),
            "EffectEstimate",
            True,
        ),
        OperatorName.ASSESS_EVIDENCE_DETECTABILITY: OperatorSignature(
            OperatorName.ASSESS_EVIDENCE_DETECTABILITY,
            ("EvidenceSet", "AssayGraph"),
            "DetectabilityAssessment",
            True,
        ),
        OperatorName.RETRIEVE_CONTEXTUAL_EVIDENCE: OperatorSignature(
            OperatorName.RETRIEVE_CONTEXTUAL_EVIDENCE,
            ("EpisodeGraph", "QueryGraph"),
            "EvidenceSet",
            False,
        ),
        OperatorName.UPDATE_HYPOTHESIS_GRAPH: OperatorSignature(
            OperatorName.UPDATE_HYPOTHESIS_GRAPH,
            ("HypothesisGraph", "EvidenceSet"),
            "HypothesisPosterior",
            True,
        ),
        OperatorName.COMPUTE_EXPECTED_INFORMATION_GAIN: OperatorSignature(
            OperatorName.COMPUTE_EXPECTED_INFORMATION_GAIN,
            ("HypothesisPosterior", "ExperimentSet"),
            "ExperimentRanking",
            True,
        ),
    }
)


def operator_signature(name: OperatorName | str) -> OperatorSignature:
    """Return a signature, rejecting any operator outside the fixed vocabulary."""

    try:
        normalized = name if isinstance(name, OperatorName) else OperatorName(name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown typed operator: {name!r}") from exc
    return OPERATOR_VOCABULARY[normalized]


class InvalidDesignCode(str, Enum):
    """Stable machine-readable reasons that a proposed inference is invalid."""

    # Values intentionally match the POC contract labels without importing them.
    DOMINANCE_REQUIRES_HETEROZYGOTES = "dominance_from_homozygous_panel"
    STRUCTURAL_ABSENCE_IS_NOT_REFERENCE_SNP = "absent_locus_as_snp"
    DATABASE_ABSENCE_IS_NOT_POWERED_NEGATIVE = "database_absence_as_powered_negative"
    TRANSPORT_REQUIRES_IDENTIFIED_MODEL = "transport_without_identification"
    POST_TREATMENT_COVARIATE_REQUIRES_JUSTIFICATION = "post_treatment_covariate"


@dataclass(frozen=True, slots=True)
class DesignIssue:
    code: InvalidDesignCode
    message: str


class DesignValidationError(ValueError):
    """Raised when one or more deterministic scientific guardrails fail."""

    def __init__(self, issues: Sequence[DesignIssue]) -> None:
        self.issues = tuple(issues)
        detail = "; ".join(f"{issue.code.value}: {issue.message}" for issue in self.issues)
        super().__init__(f"invalid inference design: {detail}")


def _lookup(value: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _design_value(design: Any, *keys: str, default: Any = _MISSING) -> Any:
    """Read a field from a request or its conventional nested payloads."""

    containers = [design]
    for container_name in ("parameters", "design", "labels"):
        nested = _lookup(design, container_name)
        if nested is not _MISSING and nested is not None:
            containers.append(nested)
    for container in containers:
        for key in keys:
            result = _lookup(container, key)
            if result is not _MISSING:
                return result
    return default


def _normalized_token(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _contains_token(value: Any, expected: set[str]) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, str) or isinstance(value, Enum):
        return _normalized_token(value) in expected
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return any(_normalized_token(item) in expected for item in value)
    return False


def _declares_invalid_flag(design: Any, code: InvalidDesignCode) -> bool:
    flags = _design_value(design, "invalid_design_flags", default=())
    return _contains_token(flags, {code.value})


def _has_heterozygotes(design: Any) -> bool:
    explicit = _design_value(design, "has_heterozygotes")
    if explicit is not _MISSING:
        return explicit is True
    states = _design_value(design, "genotype_states", "zygosity_states")
    return _contains_token(states, {"heterozygote", "heterozygotes", "heterozygous", "het"})


def _dominance_requested(design: Any) -> bool:
    if _design_value(design, "estimate_dominance", default=False) is True:
        return True
    return _contains_token(
        _design_value(design, "estimand", "requested_estimands", "analysis_target"),
        {"dominance", "dominance_effect"},
    )


def _structural_absence_as_reference_snp(design: Any) -> bool:
    locus_state = _design_value(design, "locus_state", "assembly_status", "alignment_status")
    structurally_absent = _contains_token(
        locus_state,
        {
            "absent",
            "structurally_absent",
            "unassembled",
            "not_assembled",
            "unalignable",
            "unmapped",
        },
    )
    reference_snp = _design_value(design, "treat_as_reference_snp", default=False) is True
    reference_snp = reference_snp or _contains_token(
        _design_value(design, "variant_representation", "locus_representation"),
        {"reference_snp", "ordinary_reference_snp", "snp_reference"},
    )
    return structurally_absent and reference_snp


def _database_absence_as_powered_negative(design: Any) -> bool:
    basis = _design_value(design, "evidence_basis", "negative_evidence_basis", "evidence_source")
    database_absence = _contains_token(
        basis,
        {"database_absence", "absent_from_database", "not_found_in_database", "no_database_hit"},
    )
    powered_negative = _design_value(design, "powered_negative", default=False) is True
    powered_negative = powered_negative or _contains_token(
        _design_value(design, "evidence_interpretation", "claim", "conclusion"),
        {"powered_negative", "explicit_negative", "evidence_of_absence"},
    )
    return database_absence and powered_negative


def _unidentified_formal_transport(design: Any) -> bool:
    formal = _design_value(design, "formal_transport", default=False) is True
    formal = formal or _contains_token(
        _design_value(design, "transport_kind", "transport_mode", "inference_kind"),
        {"formal_transport", "formal_causal_transport", "causal_transportability"},
    )
    identified = _design_value(
        design,
        "source_target_model_identified",
        "transport_identified",
        default=False,
    )
    return formal and identified is not True


def _has_unjustified_post_treatment_covariate(design: Any) -> bool:
    global_justified = _design_value(
        design,
        "post_treatment_adjustment_justified",
        default=False,
    ) is True
    global_reason = _design_value(design, "post_treatment_justification", default=None)
    global_justified = global_justified or (
        isinstance(global_reason, str) and bool(global_reason.strip())
    )

    if _design_value(design, "condition_on_post_treatment", default=False) is True:
        return not global_justified

    explicit = _design_value(design, "post_treatment_covariates", default=_MISSING)
    if explicit is not _MISSING and explicit is not None:
        if isinstance(explicit, str):
            return bool(explicit.strip()) and not global_justified
        try:
            return bool(tuple(explicit)) and not global_justified
        except TypeError:
            return not global_justified

    covariates = _design_value(design, "covariates", default=())
    if isinstance(covariates, str) or not isinstance(covariates, Iterable):
        return False
    for covariate in covariates:
        if _lookup(covariate, "post_treatment", False) is not True:
            continue
        local_justified = _lookup(covariate, "adjustment_justified", False) is True
        local_reason = _lookup(covariate, "justification", None)
        local_justified = local_justified or (
            isinstance(local_reason, str) and bool(local_reason.strip())
        )
        if not (global_justified or local_justified):
            return True
    return False


def design_issues(design: Mapping[str, Any] | Any) -> tuple[DesignIssue, ...]:
    """Return all invalid-design findings in a stable, deterministic order.

    The accepted field names are intentionally ordinary mapping keys, making the
    validator usable with dictionaries, dataclasses, and Pydantic models alike.
    A missing prerequisite is never inferred as satisfied.
    """

    if design is None or isinstance(design, (str, bytes, bytearray)):
        raise TypeError("design must be a mapping or an object with named fields")

    issues: list[DesignIssue] = []
    if _declares_invalid_flag(
        design, InvalidDesignCode.DOMINANCE_REQUIRES_HETEROZYGOTES
    ) or (_dominance_requested(design) and not _has_heterozygotes(design)):
        issues.append(
            DesignIssue(
                InvalidDesignCode.DOMINANCE_REQUIRES_HETEROZYGOTES,
                "a dominance effect cannot be estimated without observed heterozygotes",
            )
        )
    if _declares_invalid_flag(
        design, InvalidDesignCode.STRUCTURAL_ABSENCE_IS_NOT_REFERENCE_SNP
    ) or _structural_absence_as_reference_snp(design):
        issues.append(
            DesignIssue(
                InvalidDesignCode.STRUCTURAL_ABSENCE_IS_NOT_REFERENCE_SNP,
                "an absent, unassembled, or unalignable locus needs a structural/pangenome state",
            )
        )
    if _declares_invalid_flag(
        design, InvalidDesignCode.DATABASE_ABSENCE_IS_NOT_POWERED_NEGATIVE
    ) or _database_absence_as_powered_negative(design):
        issues.append(
            DesignIssue(
                InvalidDesignCode.DATABASE_ABSENCE_IS_NOT_POWERED_NEGATIVE,
                "failure to find a database record does not establish a powered negative",
            )
        )
    if _declares_invalid_flag(
        design, InvalidDesignCode.TRANSPORT_REQUIRES_IDENTIFIED_MODEL
    ) or _unidentified_formal_transport(design):
        issues.append(
            DesignIssue(
                InvalidDesignCode.TRANSPORT_REQUIRES_IDENTIFIED_MODEL,
                "formal causal transport requires an identified source-target model",
            )
        )
    if _declares_invalid_flag(
        design, InvalidDesignCode.POST_TREATMENT_COVARIATE_REQUIRES_JUSTIFICATION
    ) or _has_unjustified_post_treatment_covariate(design):
        issues.append(
            DesignIssue(
                InvalidDesignCode.POST_TREATMENT_COVARIATE_REQUIRES_JUSTIFICATION,
                "conditioning on a post-treatment covariate requires explicit justification",
            )
        )
    return tuple(issues)


def validate_design(design: Mapping[str, Any] | Any) -> None:
    """Reject a scientifically invalid design; otherwise return ``None``."""

    issues = design_issues(design)
    if issues:
        raise DesignValidationError(issues)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _probability(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _distribution_values(
    probabilities: Mapping[Any, Any] | Iterable[Any], label: str
) -> list[float]:
    raw_values = probabilities.values() if isinstance(probabilities, Mapping) else probabilities
    if isinstance(raw_values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a non-empty probability distribution")
    try:
        values = [
            _probability(value, f"{label}[{index}]")
            for index, value in enumerate(raw_values)
        ]
    except TypeError as exc:
        raise ValueError(f"{label} must be a non-empty probability distribution") from exc
    if not values:
        raise ValueError(f"{label} must be a non-empty probability distribution")
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE):
        raise ValueError(f"{label} must sum to 1 (observed {total:.12g})")
    return values


def _mapping_distribution(
    probabilities: Mapping[Hashable, Any], label: str
) -> dict[Hashable, float]:
    if not isinstance(probabilities, Mapping):
        raise TypeError(f"{label} must map hypothesis identifiers to probabilities")
    values = _distribution_values(probabilities, label)
    return {key: value for key, value in zip(probabilities, values, strict=True)}


def _entropy_bits(probabilities: Iterable[float]) -> float:
    return -math.fsum(value * math.log2(value) for value in probabilities if value > 0.0)


def normalized_entropy(probabilities: Mapping[Any, Any] | Iterable[Any]) -> float:
    """Return Shannon entropy on [0, 1], normalized by ``log2(number of states)``."""

    values = _distribution_values(probabilities, "probabilities")
    if len(values) == 1:
        return 0.0
    return _entropy_bits(values) / math.log2(len(values))


def bayesian_outcome_update(
    prior: Mapping[Hashable, Any],
    outcome_likelihood: Mapping[Hashable, Any],
) -> dict[Hashable, float]:
    """Update ``P(hypothesis)`` using ``P(observed outcome | hypothesis)``."""

    prior_values = _mapping_distribution(prior, "prior")
    if not isinstance(outcome_likelihood, Mapping):
        raise TypeError("outcome_likelihood must map every hypothesis to a probability")
    if set(outcome_likelihood) != set(prior_values):
        raise ValueError("outcome_likelihood must contain exactly the prior hypothesis keys")
    likelihoods = {
        hypothesis: _probability(
            outcome_likelihood[hypothesis],
            f"outcome_likelihood[{hypothesis!r}]",
        )
        for hypothesis in prior_values
    }
    evidence_probability = math.fsum(
        prior_values[hypothesis] * likelihoods[hypothesis] for hypothesis in prior_values
    )
    if evidence_probability <= 0.0:
        raise ValueError("observed outcome has zero probability under the prior predictive model")
    return {
        hypothesis: prior_values[hypothesis] * likelihoods[hypothesis] / evidence_probability
        for hypothesis in prior_values
    }


def expected_information_gain(
    prior: Mapping[Hashable, Any],
    outcome_likelihoods: Mapping[Hashable, Mapping[Hashable, Any]],
    *,
    normalized: bool = True,
) -> float:
    """Return expected Shannon information gain for one experiment.

    ``outcome_likelihoods`` is outcome-major: each outcome maps every hypothesis
    to ``P(outcome | hypothesis)``.  For each hypothesis, its probabilities over
    all mutually exclusive outcomes must sum to one.  By default, entropy is
    divided by ``log2(number of hypotheses)`` so EIG is in [0, 1] and fits the
    POC probability-like contract.  Set ``normalized=False`` to return bits.
    """

    if not isinstance(normalized, bool):
        raise TypeError("normalized must be a boolean")
    prior_values = _mapping_distribution(prior, "prior")
    if not isinstance(outcome_likelihoods, Mapping) or not outcome_likelihoods:
        raise ValueError("outcome_likelihoods must define at least one outcome")

    validated: dict[Hashable, dict[Hashable, float]] = {}
    expected_keys = set(prior_values)
    for outcome, row in outcome_likelihoods.items():
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise ValueError(
                f"outcome {outcome!r} must contain exactly the prior hypothesis keys"
            )
        validated[outcome] = {
            hypothesis: _probability(row[hypothesis], f"P({outcome!r}|{hypothesis!r})")
            for hypothesis in prior_values
        }

    for hypothesis in prior_values:
        total = math.fsum(validated[outcome][hypothesis] for outcome in validated)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE):
            raise ValueError(
                "outcome probabilities must sum to 1 for every hypothesis "
                f"({hypothesis!r} sums to {total:.12g})"
            )

    expected_posterior_entropy = 0.0
    for row in validated.values():
        outcome_probability = math.fsum(
            prior_values[hypothesis] * row[hypothesis] for hypothesis in prior_values
        )
        if outcome_probability == 0.0:
            continue
        posterior = (
            prior_values[hypothesis] * row[hypothesis] / outcome_probability
            for hypothesis in prior_values
        )
        expected_posterior_entropy += outcome_probability * _entropy_bits(posterior)

    information_gain = _entropy_bits(prior_values.values()) - expected_posterior_entropy
    if information_gain < -_PROBABILITY_TOLERANCE:
        raise ArithmeticError("computed negative information gain from a valid probability model")
    information_gain = max(0.0, information_gain)
    if normalized and len(prior_values) > 1:
        information_gain /= math.log2(len(prior_values))
    return information_gain


def cost_risk_adjusted_utility(
    information_gain: Any,
    cost: Any,
    risk: Any,
    *,
    risk_weight: Any = 1.0,
) -> float:
    """Return ``information_gain / (cost + risk_weight * risk)``."""

    gain = _finite_number(information_gain, "information_gain")
    normalized_cost = _finite_number(cost, "cost")
    normalized_risk = _probability(risk, "risk")
    normalized_weight = _finite_number(risk_weight, "risk_weight")
    if gain < 0.0:
        raise ValueError("information_gain cannot be negative")
    if normalized_cost < 0.0:
        raise ValueError("cost cannot be negative")
    if normalized_weight < 0.0:
        raise ValueError("risk_weight cannot be negative")
    denominator = normalized_cost + normalized_weight * normalized_risk
    if denominator <= 0.0:
        raise ValueError("cost + risk_weight * risk must be positive")
    return gain / denominator


@dataclass(frozen=True, slots=True)
class RankedExperiment:
    rank: int
    experiment_id: str
    expected_information_gain: float
    utility: float
    cost: float
    risk: float

    @property
    def information_gain(self) -> float:
        """Backward-friendly short name for ``expected_information_gain``."""

        return self.expected_information_gain


def _required_experiment_value(experiment: Any, *keys: str) -> Any:
    for key in keys:
        value = _lookup(experiment, key)
        if value is not _MISSING:
            return value
    raise ValueError(f"experiment is missing required field {keys[0]!r}")


def _experiment_outcome_likelihoods(experiment: Any) -> Mapping[Hashable, Mapping[Hashable, Any]]:
    """Adapt either a direct likelihood mapping or contract-style outcome objects."""

    for key in ("outcome_likelihoods", "likelihoods_by_outcome"):
        direct = _lookup(experiment, key)
        if direct is not _MISSING:
            return direct

    outcomes = _lookup(experiment, "outcomes")
    if outcomes is _MISSING:
        raise ValueError("experiment is missing required field 'outcome_likelihoods'")
    result: dict[Hashable, Mapping[Hashable, Any]] = {}
    for index, outcome in enumerate(outcomes):
        outcome_id = _lookup(outcome, "id", index)
        likelihoods = _lookup(outcome, "likelihood_by_hypothesis")
        if likelihoods is _MISSING:
            raise ValueError(
                f"experiment outcome {outcome_id!r} is missing likelihood_by_hypothesis"
            )
        if outcome_id in result:
            raise ValueError(f"duplicate experiment outcome id: {outcome_id!r}")
        result[outcome_id] = likelihoods
    return result


def rank_experiments(
    prior: Mapping[Hashable, Any],
    experiments: Iterable[Mapping[str, Any] | Any],
    *,
    risk_weight: Any = 1.0,
) -> tuple[RankedExperiment, ...]:
    """Rank experiments deterministically by utility, EIG, cost, risk, then ID."""

    # Validate once even for an empty experiment list.
    _mapping_distribution(prior, "prior")
    normalized_weight = _finite_number(risk_weight, "risk_weight")
    if normalized_weight < 0.0:
        raise ValueError("risk_weight cannot be negative")

    scored: list[RankedExperiment] = []
    seen_ids: set[str] = set()
    for experiment in experiments:
        raw_id = _required_experiment_value(experiment, "experiment_id", "id")
        experiment_id = str(raw_id).strip()
        if not experiment_id:
            raise ValueError("experiment_id cannot be empty")
        if experiment_id in seen_ids:
            raise ValueError(f"duplicate experiment_id: {experiment_id!r}")
        seen_ids.add(experiment_id)

        likelihoods = _experiment_outcome_likelihoods(experiment)
        raw_cost = _required_experiment_value(experiment, "cost")
        raw_risk = _required_experiment_value(
            experiment,
            "risk",
            "technical_risk",
            "technical_failure_probability",
        )
        cost = _finite_number(raw_cost, f"cost for {experiment_id!r}")
        risk = _probability(raw_risk, f"risk for {experiment_id!r}")
        gain = expected_information_gain(prior, likelihoods)
        utility = cost_risk_adjusted_utility(
            gain,
            cost,
            risk,
            risk_weight=normalized_weight,
        )
        scored.append(
            RankedExperiment(
                rank=0,
                experiment_id=experiment_id,
                expected_information_gain=gain,
                utility=utility,
                cost=cost,
                risk=risk,
            )
        )

    scored.sort(
        key=lambda item: (
            -item.utility,
            -item.expected_information_gain,
            item.cost,
            item.risk,
            item.experiment_id,
        )
    )
    return tuple(
        RankedExperiment(
            rank=index,
            experiment_id=item.experiment_id,
            expected_information_gain=item.expected_information_gain,
            utility=item.utility,
            cost=item.cost,
            risk=item.risk,
        )
        for index, item in enumerate(scored, start=1)
    )


@dataclass(frozen=True, slots=True)
class EquivalenceClassConclusion:
    """A conclusion that preserves indistinguishable candidates as a set."""

    candidates: tuple[str, ...]
    reason: str

    @property
    def is_set_valued(self) -> bool:
        return len(self.candidates) > 1

    @property
    def resolved_candidate(self) -> str | None:
        return self.candidates[0] if len(self.candidates) == 1 else None


def equivalence_class_conclusion(
    candidate_scores: Mapping[str, Any],
    evidence_signatures: Mapping[str, Any] | None = None,
    *,
    tolerance: Any = 0.0,
    reason: str = "current design cannot distinguish candidates with equivalent evidence",
) -> EquivalenceClassConclusion:
    """Return the top candidate or its complete evidence-equivalence class.

    With no signatures, all candidates tied with the maximum score are returned.
    With signatures, any candidate whose signature equals a top candidate's
    signature is retained even if priors gave it a different score.  This avoids
    converting a design-specific resolution limit into an arbitrary single rank.
    """

    if not isinstance(candidate_scores, Mapping) or not candidate_scores:
        raise ValueError("candidate_scores must be a non-empty mapping")
    normalized_tolerance = _finite_number(tolerance, "tolerance")
    if normalized_tolerance < 0.0:
        raise ValueError("tolerance cannot be negative")
    scores: dict[str, float] = {}
    for raw_candidate, raw_score in candidate_scores.items():
        candidate = str(raw_candidate).strip()
        if not candidate:
            raise ValueError("candidate identifiers cannot be empty")
        if candidate in scores:
            raise ValueError(f"candidate identifiers collide after normalization: {candidate!r}")
        scores[candidate] = _finite_number(raw_score, f"score for {candidate!r}")

    maximum = max(scores.values())
    selected = {
        candidate
        for candidate, score in scores.items()
        if maximum - score <= normalized_tolerance
    }

    if evidence_signatures is not None:
        if not isinstance(evidence_signatures, Mapping):
            raise TypeError("evidence_signatures must map every candidate to a signature")
        normalized_signatures = {
            str(key).strip(): value for key, value in evidence_signatures.items()
        }
        if set(normalized_signatures) != set(scores):
            raise ValueError("evidence_signatures must contain exactly the candidate score keys")
        top_signatures = [normalized_signatures[candidate] for candidate in selected]
        selected.update(
            candidate
            for candidate, signature in normalized_signatures.items()
            if any(signature == top_signature for top_signature in top_signatures)
        )

    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("reason cannot be empty")
    return EquivalenceClassConclusion(tuple(sorted(selected)), normalized_reason)


def set_valued_conclusion(
    candidate_scores: Mapping[str, Any],
    evidence_signatures: Mapping[str, Any] | None = None,
    *,
    tolerance: Any = 0.0,
    reason: str = "current design cannot distinguish candidates with equivalent evidence",
) -> EquivalenceClassConclusion:
    """Alias with an inference-program-friendly name."""

    return equivalence_class_conclusion(
        candidate_scores,
        evidence_signatures,
        tolerance=tolerance,
        reason=reason,
    )
