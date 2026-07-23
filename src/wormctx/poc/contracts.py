"""Strict contracts for bounded proof-of-concept biological episodes.

The POC deliberately keeps its learning-facing representation separate from the
larger canonical evidence graph.  These contracts define the small, completely
materialized graph that can be batched by a future model without importing a
tensor framework.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MAX_NODES = 64
MAX_EDGES = 192
MAX_HYPOTHESES = 4
MAX_EXPERIMENTS = 8
MAX_FEATURES = 32
MAX_OPERATORS = 10

Identifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$",
    ),
]
ShortText = Annotated[str, Field(min_length=1, max_length=240)]
LongText = Annotated[str, Field(min_length=1, max_length=2_000)]
FiniteFloat = Annotated[float, Field(strict=True)]
Probability = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
PositiveCost = Annotated[float, Field(strict=True, gt=0.0, le=1_000_000.0)]
FeatureVector = Annotated[list[FiniteFloat], Field(max_length=MAX_FEATURES)]
LikelihoodRow = Annotated[
    list[Probability],
    Field(min_length=2, max_length=MAX_HYPOTHESES),
]


class StrictContract(BaseModel):
    """Base class that rejects coercion, unknown fields, and non-finite values."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        allow_inf_nan=False,
    )


class EpisodeFamily(str, Enum):
    developmental = "developmental"
    natural_variation = "natural_variation"
    cross_context = "cross_context"


class DatasetSplit(str, Enum):
    train = "train"
    val = "val"
    test = "test"


class NodeKind(str, Enum):
    pan_gene = "pan_gene"
    locus_instance = "locus_instance"
    transcript = "transcript"
    protein = "protein"
    allele = "allele"
    haplotype = "haplotype"
    strain = "strain"
    cell = "cell"
    lineage = "lineage"
    developmental_stage = "developmental_stage"
    compound = "compound"
    environment = "environment"
    phenotype = "phenotype"
    perturbation = "perturbation"
    assay = "assay"
    measurement = "measurement"
    covariate = "covariate"
    qtl_interval = "qtl_interval"
    evidence = "evidence"
    context = "context"
    hypothesis = "hypothesis"
    experiment = "experiment"


class EdgeKind(str, Enum):
    carried_by = "carried_by"
    lies_on_path = "lies_on_path"
    expressed_in = "expressed_in"
    descends_from = "descends_from"
    contacts = "contacts"
    intervenes_on = "intervenes_on"
    applied_during = "applied_during"
    transformed_into = "transformed_into"
    analyzed_by = "analyzed_by"
    supports = "supports"
    contradicts = "contradicts"
    has_detectability_for = "has_detectability_for"
    differs_from = "differs_from"
    distinguishes = "distinguishes"
    measured_in = "measured_in"
    has_covariate = "has_covariate"
    candidate_for = "candidate_for"


class Operator(str, Enum):
    """Closed POC inference-program vocabulary from the proposed architecture."""

    derive_trait = "derive_trait"
    fit_reaction_norm = "fit_reaction_norm"
    fit_kinship_and_haplotype_model = "fit_kinship_and_haplotype_model"
    score_pangenome_sequence_change = "score_pangenome_sequence_change"
    infer_developmental_state = "infer_developmental_state"
    estimate_perturbation_effect = "estimate_perturbation_effect"
    assess_evidence_detectability = "assess_evidence_detectability"
    retrieve_contextual_evidence = "retrieve_contextual_evidence"
    update_hypothesis_graph = "update_hypothesis_graph"
    compute_expected_information_gain = "compute_expected_information_gain"


class InvalidDesignFlag(str, Enum):
    dominance_from_homozygous_panel = "dominance_from_homozygous_panel"
    absent_locus_as_snp = "absent_locus_as_snp"
    database_absence_as_powered_negative = "database_absence_as_powered_negative"
    transport_without_identification = "transport_without_identification"
    post_treatment_covariate = "post_treatment_covariate"


class ResolutionCeiling(str, Enum):
    variant = "variant"
    gene = "gene"
    haplotype_block = "haplotype_block"
    regulatory_graph = "regulatory_graph"
    mechanistic_equivalence_class = "mechanistic_equivalence_class"
    contextual_applicability_only = "contextual_applicability_only"
    non_identifiable = "non_identifiable"


class GraphNode(StrictContract):
    id: Identifier
    kind: NodeKind
    label: ShortText
    # Feature semantics are fixed by the generator/model manifest.  Length is
    # intentionally variable so a trainer can pad or truncate at its boundary.
    numeric_features: FeatureVector = Field(default_factory=list)
    categorical_features: dict[Identifier, ShortText] = Field(
        default_factory=dict,
        max_length=MAX_FEATURES,
    )


class GraphEdge(StrictContract):
    id: Identifier
    source: Identifier
    kind: EdgeKind
    target: Identifier
    numeric_features: FeatureVector = Field(default_factory=list)


class EpisodeGraph(StrictContract):
    nodes: list[GraphNode] = Field(min_length=2, max_length=MAX_NODES)
    edges: list[GraphEdge] = Field(min_length=1, max_length=MAX_EDGES)

    @model_validator(mode="after")
    def validate_graph_integrity(self) -> "EpisodeGraph":
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph node ids must be unique")
        edge_ids = [edge.id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("graph edge ids must be unique")
        available_nodes = set(node_ids)
        for edge in self.edges:
            if edge.source not in available_nodes or edge.target not in available_nodes:
                raise ValueError(
                    f"edge {edge.id!r} must reference nodes present in this episode"
                )
            if edge.source == edge.target:
                raise ValueError(f"edge {edge.id!r} cannot be a self-loop")
        return self


class Hypothesis(StrictContract):
    id: Identifier
    label: ShortText
    statement: LongText
    equivalence_class: Identifier
    prior_probability: Probability
    numeric_features: FeatureVector = Field(default_factory=list)


class ExperimentOutcome(StrictContract):
    id: Identifier
    label: ShortText
    likelihood_by_hypothesis: dict[Identifier, Probability] = Field(
        min_length=2,
        max_length=MAX_HYPOTHESES,
    )


class CandidateExperiment(StrictContract):
    id: Identifier
    label: ShortText
    description: LongText
    target_hypothesis_ids: list[Identifier] = Field(
        min_length=2,
        max_length=MAX_HYPOTHESES,
    )
    outcomes: list[ExperimentOutcome] = Field(min_length=2, max_length=4)
    likelihood_hypothesis_order: list[Identifier] = Field(
        min_length=2,
        max_length=MAX_HYPOTHESES,
    )
    # Rows follow ``outcomes``; columns follow ``likelihood_hypothesis_order``.
    outcome_likelihood_matrix: list[LikelihoodRow] = Field(min_length=2, max_length=4)
    selection_prior_probability: Probability
    cost: PositiveCost
    risk: Probability
    duration_days: Annotated[int, Field(strict=True, ge=1, le=365)]
    expected_information_gain: Probability

    @model_validator(mode="after")
    def validate_experiment(self) -> "CandidateExperiment":
        if len(self.target_hypothesis_ids) != len(set(self.target_hypothesis_ids)):
            raise ValueError("target_hypothesis_ids must be unique")
        outcome_ids = [outcome.id for outcome in self.outcomes]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("experiment outcome ids must be unique")
        if len(self.likelihood_hypothesis_order) != len(
            set(self.likelihood_hypothesis_order)
        ):
            raise ValueError("likelihood_hypothesis_order must be unique")
        if len(self.outcome_likelihood_matrix) != len(self.outcomes):
            raise ValueError("outcome_likelihood_matrix requires one row per outcome")
        expected_width = len(self.likelihood_hypothesis_order)
        for row_index, row in enumerate(self.outcome_likelihood_matrix):
            if len(row) != expected_width:
                raise ValueError(
                    "each outcome_likelihood_matrix row requires one column per hypothesis"
                )
            expected_row = [
                self.outcomes[row_index].likelihood_by_hypothesis[hypothesis_id]
                for hypothesis_id in self.likelihood_hypothesis_order
            ]
            if any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(row, expected_row, strict=True)
            ):
                raise ValueError(
                    "outcome_likelihood_matrix must match the named outcome likelihoods"
                )
        return self


class EpisodeLabels(StrictContract):
    """Supervision targets for program induction, validity, and resolution."""

    operator_program: list[Operator] = Field(min_length=1, max_length=MAX_OPERATORS)
    operator_multilabel: list[Operator] = Field(min_length=1, max_length=MAX_OPERATORS)
    design_valid: bool
    invalid_design_flags: list[InvalidDesignFlag] = Field(
        default_factory=list,
        max_length=len(InvalidDesignFlag),
    )
    true_hypothesis_id: Identifier
    true_equivalence_class: Identifier
    resolution_ceiling: ResolutionCeiling
    resolution_note: LongText

    @model_validator(mode="after")
    def validate_labels(self) -> "EpisodeLabels":
        if len(self.operator_program) != len(set(self.operator_program)):
            raise ValueError("operator_program cannot repeat an operator in the POC")
        if len(self.operator_multilabel) != len(set(self.operator_multilabel)):
            raise ValueError("operator_multilabel must contain unique operators")
        if set(self.operator_multilabel) != set(self.operator_program):
            raise ValueError(
                "operator_multilabel must be the order-independent view of operator_program"
            )
        if len(self.invalid_design_flags) != len(set(self.invalid_design_flags)):
            raise ValueError("invalid_design_flags must be unique")
        if self.design_valid == bool(self.invalid_design_flags):
            raise ValueError(
                "design_valid must be false exactly when invalid_design_flags are present"
            )
        return self


class EpisodeProvenance(StrictContract):
    """An intentionally unmistakable marker that POC data are simulated."""

    origin: Literal["simulation"] = "simulation"
    synthetic: Literal[True] = True
    generator: Literal["wormctx.poc.simulation"] = "wormctx.poc.simulation"
    generator_version: Literal["1.0"] = "1.0"
    seed: Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
    scenario_id: Identifier


class BiologicalEpisode(StrictContract):
    id: Identifier
    family: EpisodeFamily
    question: LongText
    graph: EpisodeGraph
    hypotheses: list[Hypothesis] = Field(min_length=2, max_length=MAX_HYPOTHESES)
    candidate_experiments: list[CandidateExperiment] = Field(
        min_length=1,
        max_length=MAX_EXPERIMENTS,
    )
    labels: EpisodeLabels
    split_group: Identifier
    split: DatasetSplit
    provenance: EpisodeProvenance

    @model_validator(mode="after")
    def validate_episode_consistency(self) -> "BiologicalEpisode":
        hypothesis_ids = [hypothesis.id for hypothesis in self.hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("hypothesis ids must be unique")
        if not math.isclose(
            math.fsum(hypothesis.prior_probability for hypothesis in self.hypotheses),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("hypothesis prior probabilities must sum to one")

        available_hypotheses = set(hypothesis_ids)
        if self.labels.true_hypothesis_id not in available_hypotheses:
            raise ValueError("true_hypothesis_id must reference an episode hypothesis")
        true_hypothesis = next(
            item
            for item in self.hypotheses
            if item.id == self.labels.true_hypothesis_id
        )
        if true_hypothesis.equivalence_class != self.labels.true_equivalence_class:
            raise ValueError(
                "true_equivalence_class must match the labeled true hypothesis"
            )

        experiment_ids = [experiment.id for experiment in self.candidate_experiments]
        if len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("candidate experiment ids must be unique")
        if not math.isclose(
            math.fsum(
                experiment.selection_prior_probability
                for experiment in self.candidate_experiments
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("candidate experiment selection priors must sum to one")
        for experiment in self.candidate_experiments:
            target_ids = set(experiment.target_hypothesis_ids)
            if not target_ids <= available_hypotheses:
                raise ValueError(
                    f"experiment {experiment.id!r} targets an unknown hypothesis"
                )
            if experiment.likelihood_hypothesis_order != hypothesis_ids:
                raise ValueError(
                    "experiment likelihood_hypothesis_order must match episode hypothesis order"
                )
            for outcome in experiment.outcomes:
                if set(outcome.likelihood_by_hypothesis) != available_hypotheses:
                    raise ValueError(
                        f"outcome {outcome.id!r} must give a likelihood for every hypothesis"
                    )
            for hypothesis_id in hypothesis_ids:
                total = math.fsum(
                    outcome.likelihood_by_hypothesis[hypothesis_id]
                    for outcome in experiment.outcomes
                )
                if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
                    raise ValueError(
                        "outcome likelihoods for each experiment and hypothesis "
                        f"must sum to one; {experiment.id!r}/{hypothesis_id!r} sums to {total}"
                    )
        return self


# Stable string vocabularies for index construction in a dependency-free trainer.
# These are public rather than inferred ad hoc so checkpoints can record the
# exact categorical ordering that produced them.
FAMILY_ORDER = tuple(item.value for item in EpisodeFamily)
OPERATOR_ORDER = tuple(item.value for item in Operator)
INVALID_FLAG_ORDER = tuple(item.value for item in InvalidDesignFlag)
CONCLUSION_ORDER = tuple(item.value for item in ResolutionCeiling)
NODE_TYPE_ORDER = tuple(item.value for item in NodeKind)
EDGE_TYPE_ORDER = tuple(item.value for item in EdgeKind)


__all__ = [
    "BiologicalEpisode",
    "CandidateExperiment",
    "CONCLUSION_ORDER",
    "DatasetSplit",
    "EDGE_TYPE_ORDER",
    "EdgeKind",
    "EpisodeFamily",
    "EpisodeGraph",
    "EpisodeLabels",
    "EpisodeProvenance",
    "ExperimentOutcome",
    "GraphEdge",
    "GraphNode",
    "Hypothesis",
    "INVALID_FLAG_ORDER",
    "InvalidDesignFlag",
    "FAMILY_ORDER",
    "MAX_EDGES",
    "MAX_EXPERIMENTS",
    "MAX_HYPOTHESES",
    "MAX_NODES",
    "NodeKind",
    "NODE_TYPE_ORDER",
    "Operator",
    "OPERATOR_ORDER",
    "ResolutionCeiling",
]
