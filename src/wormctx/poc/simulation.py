"""Deterministic synthetic biological episodes for the local POC.

This module is intentionally limited to the standard library plus the Pydantic
contracts.  It provides data plumbing and supervision fixtures, not a claim of
biological realism.  Every generated episode carries an explicit simulation
provenance marker.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .contracts import (
    BiologicalEpisode,
    CandidateExperiment,
    DatasetSplit,
    EdgeKind,
    EpisodeFamily,
    EpisodeGraph,
    EpisodeLabels,
    EpisodeProvenance,
    ExperimentOutcome,
    GraphEdge,
    GraphNode,
    Hypothesis,
    InvalidDesignFlag,
    NodeKind,
    Operator,
    ResolutionCeiling,
)


DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)
RED_TEAM_FLAGS = tuple(InvalidDesignFlag)


def _checked_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be between zero and 2**63 - 1")
    return seed


def _checked_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least one")
    return value


def _family(value: EpisodeFamily | str) -> EpisodeFamily:
    if isinstance(value, EpisodeFamily):
        return value
    return EpisodeFamily(value)


def _invalid_flags(
    values: Iterable[InvalidDesignFlag | str],
) -> tuple[InvalidDesignFlag, ...]:
    flags = tuple(
        value if isinstance(value, InvalidDesignFlag) else InvalidDesignFlag(value)
        for value in values
    )
    if len(flags) != len(set(flags)):
        raise ValueError("invalid_flags must be unique")
    return flags


def _digest(seed: int, *parts: object) -> bytes:
    material = "\x1f".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(material.encode("utf-8")).digest()


def _rng(seed: int, *parts: object) -> random.Random:
    return random.Random(int.from_bytes(_digest(seed, *parts)[:8], "big"))


def _validate_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("split ratios must contain train, val, and test values")
    result = tuple(float(item) for item in ratios)
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        raise ValueError("split ratios must be finite and non-negative")
    if not math.isclose(math.fsum(result), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to one")
    if result[0] == 0.0:
        raise ValueError("the train split ratio must be positive")
    return result  # type: ignore[return-value]


def split_for_group(
    split_group: str,
    *,
    seed: int = 0,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> DatasetSplit:
    """Assign one group without relying on process-randomized Python hashes."""

    seed = _checked_seed(seed)
    train_ratio, val_ratio, _ = _validate_ratios(ratios)
    unit = int.from_bytes(_digest(seed, "split", split_group), "big") / 2**256
    if unit < train_ratio:
        return DatasetSplit.train
    if unit < train_ratio + val_ratio:
        return DatasetSplit.val
    return DatasetSplit.test


def assign_group_splits(
    split_groups: Iterable[str],
    *,
    seed: int = 0,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, DatasetSplit]:
    """Return deterministic, exactly group-disjoint train/val/test assignments.

    Ranking groups by a seeded SHA-256 digest makes the result independent of
    input order.  When at least three groups and all default ratios are nonzero,
    every split receives a group.
    """

    seed = _checked_seed(seed)
    split_ratios = _validate_ratios(ratios)
    groups = set(split_groups)
    if not groups:
        raise ValueError("at least one split group is required")
    if any(not group or any(character.isspace() for character in group) for group in groups):
        raise ValueError("split groups must be non-empty tokens without whitespace")

    ordered = sorted(groups, key=lambda group: (_digest(seed, "rank", group), group))
    count = len(ordered)
    raw_counts = [count * ratio for ratio in split_ratios]
    counts = [math.floor(value) for value in raw_counts]
    for index in sorted(
        range(3),
        key=lambda item: (-(raw_counts[item] - counts[item]), item),
    )[: count - sum(counts)]:
        counts[index] += 1

    nonzero_split_count = sum(ratio > 0.0 for ratio in split_ratios)
    if count >= nonzero_split_count:
        for empty_index, ratio in enumerate(split_ratios):
            if ratio == 0.0 or counts[empty_index] > 0:
                continue
            donor = max(
                (
                    index
                    for index, donor_count in enumerate(counts)
                    if donor_count > 1
                ),
                key=lambda index: (counts[index], split_ratios[index], -index),
            )
            counts[donor] -= 1
            counts[empty_index] += 1

    assignments: dict[str, DatasetSplit] = {}
    cursor = 0
    for split, split_count in zip(DatasetSplit, counts, strict=True):
        for group in ordered[cursor : cursor + split_count]:
            assignments[group] = split
        cursor += split_count
    return assignments


def _hypotheses(
    episode_id: str,
    family: EpisodeFamily,
    rng: random.Random,
) -> list[Hypothesis]:
    specifications = {
        EpisodeFamily.developmental: (
            (
                "direct regulator",
                "The perturbed factor directly controls the observed fate transition.",
                "direct_regulation",
            ),
            (
                "timing mediator",
                "The fate change is mediated by an earlier shift in division timing.",
                "timing_mediation",
            ),
            (
                "assay explanation",
                "The apparent fate change is a reporter or detectability effect.",
                "measurement_explanation",
            ),
        ),
        EpisodeFamily.natural_variation: (
            (
                "target modification",
                "A local allele changes the compound's molecular target.",
                "local_molecular_mechanism",
            ),
            (
                "uptake or metabolism",
                "The haplotype changes uptake, efflux, or compound metabolism.",
                "local_molecular_mechanism",
            ),
            (
                "baseline or multigene effect",
                "Baseline growth or a linked multigene haplotype drives the association.",
                "linked_haplotype",
            ),
        ),
        EpisodeFamily.cross_context: (
            (
                "shared mechanism",
                "The source-context mechanism remains operative in the target context.",
                "source_compatible_mechanism",
            ),
            (
                "allele-specific response",
                "The natural allele is not a scaled version of the source perturbation.",
                "source_compatible_mechanism",
            ),
            (
                "background interaction",
                "Genetic background or stage changes the direction of the effect.",
                "context_interaction",
            ),
        ),
    }[family]
    weights = [rng.uniform(0.7, 1.3) for _ in specifications]
    total = math.fsum(weights)
    return [
        Hypothesis(
            id=f"{episode_id}/hypothesis/{index}",
            label=label,
            statement=statement,
            equivalence_class=f"{episode_id}/equivalence/{equivalence_class}",
            prior_probability=float(weight / total),
            numeric_features=[float(weight / total), float(rng.uniform(0.0, 1.0))],
        )
        for index, ((label, statement, equivalence_class), weight) in enumerate(
            zip(specifications, weights, strict=True)
        )
    ]


def _outcomes(
    experiment_id: str,
    hypotheses: Sequence[Hypothesis],
    diagonal: float,
) -> tuple[list[ExperimentOutcome], list[list[float]]]:
    off_diagonal = (1.0 - diagonal) / (len(hypotheses) - 1)
    matrix = [
        [
            float(diagonal if outcome_index == hypothesis_index else off_diagonal)
            for hypothesis_index in range(len(hypotheses))
        ]
        for outcome_index in range(len(hypotheses))
    ]
    outcomes = [
        ExperimentOutcome(
            id=f"{experiment_id}/outcome/{outcome_index}",
            label=f"outcome favors {hypothesis.label}",
            likelihood_by_hypothesis={
                candidate.id: matrix[outcome_index][hypothesis_index]
                for hypothesis_index, candidate in enumerate(hypotheses)
            },
        )
        for outcome_index, hypothesis in enumerate(hypotheses)
    ]
    return outcomes, matrix


def _candidate_experiments(
    episode_id: str,
    family: EpisodeFamily,
    hypotheses: Sequence[Hypothesis],
    rng: random.Random,
) -> list[CandidateExperiment]:
    family_details = {
        EpisodeFamily.developmental: (
            "partial perturbation with time-resolved reporter",
            "Measure the earliest lineage-resolved reporter divergence after a "
            "partial perturbation.",
            900.0,
            0.12,
            8,
            "orthogonal fate and timing assay",
            "Measure fate and division timing with an orthogonal reporter in the same lineage.",
            1_350.0,
            0.16,
            12,
        ),
        EpisodeFamily.natural_variation: (
            "allele replacement across treatment doses",
            "Replace the candidate allele and measure a dose-by-time reaction norm.",
            2_500.0,
            0.20,
            24,
            "recombinant haplotype dissection",
            "Dissect the local haplotype with recombinants and matched untreated controls.",
            3_800.0,
            0.28,
            35,
        ),
        EpisodeFamily.cross_context: (
            "allele by background factorial replacement",
            "Test source and natural alleles in both source and target backgrounds.",
            3_200.0,
            0.24,
            30,
            "paired context perturbation assay",
            "Run a matched perturbation in both contexts with a common outcome definition.",
            2_100.0,
            0.18,
            18,
        ),
    }[family]
    experiment_specs = (
        (family_details[:5], 0.78, 0.65, 0.72),
        (family_details[5:], 0.60, 0.35, 0.48),
    )
    hypothesis_ids = [hypothesis.id for hypothesis in hypotheses]
    experiments: list[CandidateExperiment] = []
    for index, (details, diagonal, selection_prior, information_gain) in enumerate(
        experiment_specs
    ):
        label, description, cost, risk, duration = details
        experiment_id = f"{episode_id}/experiment/{index}"
        outcomes, matrix = _outcomes(experiment_id, hypotheses, diagonal)
        experiments.append(
            CandidateExperiment(
                id=experiment_id,
                label=label,
                description=description,
                target_hypothesis_ids=hypothesis_ids,
                outcomes=outcomes,
                likelihood_hypothesis_order=hypothesis_ids,
                outcome_likelihood_matrix=matrix,
                selection_prior_probability=float(selection_prior),
                cost=float(cost + rng.randint(0, 20)),
                risk=float(risk),
                duration_days=duration,
                expected_information_gain=float(information_gain),
            )
        )
    return experiments


def _node(
    episode_id: str,
    suffix: str,
    kind: NodeKind,
    label: str,
    numeric_features: Sequence[float],
    **categorical_features: str,
) -> GraphNode:
    return GraphNode(
        id=f"{episode_id}/node/{suffix}",
        kind=kind,
        label=label,
        numeric_features=[float(value) for value in numeric_features],
        categorical_features=categorical_features,
    )


def _edge(
    episode_id: str,
    index: int,
    source: GraphNode,
    kind: EdgeKind,
    target: GraphNode,
    *numeric_features: float,
) -> GraphEdge:
    return GraphEdge(
        id=f"{episode_id}/edge/{index}",
        source=source.id,
        kind=kind,
        target=target.id,
        numeric_features=[float(value) for value in numeric_features],
    )


def _developmental_graph(
    episode_id: str,
    hypotheses: Sequence[Hypothesis],
    experiments: Sequence[CandidateExperiment],
    flags: Sequence[InvalidDesignFlag],
    rng: random.Random,
) -> EpisodeGraph:
    gene = _node(episode_id, "gene", NodeKind.pan_gene, "candidate regulator", [0.82, 0.41])
    parent = _node(episode_id, "parent-cell", NodeKind.cell, "parent blastomere", [0.31, 0.54])
    cell = _node(episode_id, "cell", NodeKind.cell, "daughter cell at decision", [0.62, 0.57])
    stage = _node(episode_id, "stage", NodeKind.developmental_stage, "fate-decision stage", [350.0])
    perturbation = _node(
        episode_id,
        "perturbation",
        NodeKind.perturbation,
        "partial loss of function",
        [0.45],
        allele_type="partial_loss_of_function",
    )
    measurement = _node(
        episode_id,
        "measurement",
        NodeKind.measurement,
        "lineage reporter trajectory",
        [float(rng.uniform(-1.0, 1.0)), 0.12],
        modality="processed_trajectory",
    )
    evidence_features = {"observation_status": "positive", "detectability": "estimated"}
    if InvalidDesignFlag.database_absence_as_powered_negative in flags:
        evidence_features = {
            "observation_status": "not_recorded",
            "detectability": "unknown",
            "invalid_interpretation": "powered_negative",
        }
    evidence = _node(
        episode_id,
        "evidence",
        NodeKind.evidence,
        "contextual database evidence",
        [
            0.68,
            0.74,
            float(
                InvalidDesignFlag.database_absence_as_powered_negative in flags
            ),
        ],
        **evidence_features,
    )
    nodes = [gene, parent, cell, stage, perturbation, measurement, evidence]
    edges = [
        _edge(episode_id, 0, cell, EdgeKind.descends_from, parent, 1.0),
        _edge(episode_id, 1, gene, EdgeKind.expressed_in, cell, 0.82),
        _edge(episode_id, 2, perturbation, EdgeKind.intervenes_on, gene, 0.45),
        _edge(episode_id, 3, measurement, EdgeKind.measured_in, cell, 0.91),
        _edge(episode_id, 4, measurement, EdgeKind.applied_during, stage, 1.0),
        _edge(episode_id, 5, evidence, EdgeKind.has_detectability_for, measurement, 0.74),
    ]
    return _append_decision_nodes(episode_id, nodes, edges, hypotheses, experiments)


def _natural_variation_graph(
    episode_id: str,
    hypotheses: Sequence[Hypothesis],
    experiments: Sequence[CandidateExperiment],
    flags: Sequence[InvalidDesignFlag],
    rng: random.Random,
) -> EpisodeGraph:
    panel = _node(
        episode_id,
        "panel",
        NodeKind.strain,
        "natural isolate panel",
        [
            0.995,
            96.0,
            float(InvalidDesignFlag.dominance_from_homozygous_panel in flags),
        ],
        mating_design="homozygous_natural_isolates",
        requested_estimand=(
            "dominance"
            if InvalidDesignFlag.dominance_from_homozygous_panel in flags
            else "additive_haplotype_effect"
        ),
    )
    haplotype = _node(
        episode_id,
        "haplotype",
        NodeKind.haplotype,
        "local pangenome haplotype",
        [0.37, 0.22],
        path_state="alternate_path",
    )
    locus_features = {"presence_state": "present", "encoding": "pangenome_locus"}
    if InvalidDesignFlag.absent_locus_as_snp in flags:
        locus_features = {
            "presence_state": "absent_in_carriers",
            "invalid_encoding": "reference_coordinate_snp",
        }
    locus = _node(
        episode_id,
        "locus",
        NodeKind.locus_instance,
        "assembly-specific candidate locus",
        [
            0.64,
            0.19,
            float(InvalidDesignFlag.absent_locus_as_snp in flags),
        ],
        **locus_features,
    )
    trait = _node(
        episode_id,
        "trait",
        NodeKind.phenotype,
        "dose-by-time growth reaction norm",
        [float(rng.uniform(-0.8, 0.8)), 0.33],
        derivation="quality_control_then_reaction_norm",
    )
    compound = _node(episode_id, "compound", NodeKind.compound, "test compound", [0.71])
    assay = _node(
        episode_id,
        "assay",
        NodeKind.assay,
        "longitudinal growth assay",
        [0.86, 0.18],
        controls="treated_and_untreated",
    )
    covariate_features = {"measurement_time": "pre_treatment", "role": "baseline"}
    if InvalidDesignFlag.post_treatment_covariate in flags:
        covariate_features = {
            "measurement_time": "post_treatment",
            "invalid_role": "adjustment_covariate",
        }
    covariate = _node(
        episode_id,
        "covariate",
        NodeKind.covariate,
        "growth-rate covariate",
        [0.29, float(InvalidDesignFlag.post_treatment_covariate in flags)],
        **covariate_features,
    )
    nodes = [panel, haplotype, locus, trait, compound, assay, covariate]
    edges = [
        _edge(episode_id, 0, haplotype, EdgeKind.carried_by, panel, 0.37),
        _edge(episode_id, 1, locus, EdgeKind.lies_on_path, haplotype, 0.98),
        _edge(episode_id, 2, trait, EdgeKind.measured_in, assay, 0.86),
        _edge(episode_id, 3, compound, EdgeKind.applied_during, assay, 1.0),
        _edge(episode_id, 4, trait, EdgeKind.has_covariate, covariate, 0.29),
        _edge(episode_id, 5, locus, EdgeKind.candidate_for, trait, 0.64),
    ]
    return _append_decision_nodes(episode_id, nodes, edges, hypotheses, experiments)


def _cross_context_graph(
    episode_id: str,
    hypotheses: Sequence[Hypothesis],
    experiments: Sequence[CandidateExperiment],
    flags: Sequence[InvalidDesignFlag],
    rng: random.Random,
) -> EpisodeGraph:
    source = _node(
        episode_id,
        "source-context",
        NodeKind.context,
        "N2 null-perturbation source context",
        [0.92, 0.35],
        background="N2",
        allele="null",
        stage="embryo",
    )
    target_features = {
        "background": "natural_isolate",
        "allele": "regulatory",
        "stage": "adult",
        "transport_status": "not_claimed",
    }
    if InvalidDesignFlag.transport_without_identification in flags:
        target_features["transport_status"] = "claimed_without_identification"
    target = _node(
        episode_id,
        "target-context",
        NodeKind.context,
        "natural-allele target context",
        [
            0.51,
            0.77,
            float(InvalidDesignFlag.transport_without_identification in flags),
        ],
        **target_features,
    )
    allele = _node(
        episode_id,
        "allele",
        NodeKind.allele,
        "natural regulatory allele",
        [0.28, 0.62],
        allele_type="regulatory",
    )
    perturbation = _node(
        episode_id,
        "source-perturbation",
        NodeKind.perturbation,
        "source null perturbation",
        [1.0],
        allele_type="null",
    )
    phenotype = _node(
        episode_id,
        "phenotype",
        NodeKind.phenotype,
        "context-matched quantitative outcome",
        [float(rng.uniform(-0.5, 0.5)), 0.44],
    )
    evidence = _node(
        episode_id,
        "evidence",
        NodeKind.evidence,
        "paired contextual evidence",
        [0.55, 0.39],
        applicability="contextual_only",
    )
    nodes = [source, target, allele, perturbation, phenotype, evidence]
    edges = [
        _edge(episode_id, 0, target, EdgeKind.differs_from, source, 0.79),
        _edge(episode_id, 1, perturbation, EdgeKind.intervenes_on, allele, 0.24),
        _edge(episode_id, 2, evidence, EdgeKind.supports, phenotype, 0.55),
        _edge(episode_id, 3, phenotype, EdgeKind.measured_in, target, 0.44),
    ]
    return _append_decision_nodes(episode_id, nodes, edges, hypotheses, experiments)


def _append_decision_nodes(
    episode_id: str,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    hypotheses: Sequence[Hypothesis],
    experiments: Sequence[CandidateExperiment],
) -> EpisodeGraph:
    hypothesis_nodes = [
        _node(
            episode_id,
            f"hypothesis-{index}",
            NodeKind.hypothesis,
            hypothesis.label,
            [hypothesis.prior_probability, *hypothesis.numeric_features],
            hypothesis_id=hypothesis.id,
            equivalence_class=hypothesis.equivalence_class,
        )
        for index, hypothesis in enumerate(hypotheses)
    ]
    experiment_nodes = [
        _node(
            episode_id,
            f"experiment-{index}",
            NodeKind.experiment,
            experiment.label,
            [
                experiment.cost,
                experiment.risk,
                experiment.expected_information_gain,
            ],
            experiment_id=experiment.id,
        )
        for index, experiment in enumerate(experiments)
    ]
    edge_index = len(edges)
    for experiment_node in experiment_nodes:
        for hypothesis_node in hypothesis_nodes:
            edges.append(
                _edge(
                    episode_id,
                    edge_index,
                    experiment_node,
                    EdgeKind.distinguishes,
                    hypothesis_node,
                    1.0,
                )
            )
            edge_index += 1
    return EpisodeGraph(nodes=[*nodes, *hypothesis_nodes, *experiment_nodes], edges=edges)


def _operator_program(family: EpisodeFamily) -> list[Operator]:
    return {
        EpisodeFamily.developmental: [
            Operator.infer_developmental_state,
            Operator.estimate_perturbation_effect,
            Operator.assess_evidence_detectability,
            Operator.update_hypothesis_graph,
            Operator.compute_expected_information_gain,
        ],
        EpisodeFamily.natural_variation: [
            Operator.derive_trait,
            Operator.fit_reaction_norm,
            Operator.fit_kinship_and_haplotype_model,
            Operator.score_pangenome_sequence_change,
            Operator.assess_evidence_detectability,
            Operator.update_hypothesis_graph,
            Operator.compute_expected_information_gain,
        ],
        EpisodeFamily.cross_context: [
            Operator.retrieve_contextual_evidence,
            Operator.assess_evidence_detectability,
            Operator.estimate_perturbation_effect,
            Operator.update_hypothesis_graph,
            Operator.compute_expected_information_gain,
        ],
    }[family]


def _valid_resolution(family: EpisodeFamily) -> tuple[ResolutionCeiling, str]:
    return {
        EpisodeFamily.developmental: (
            ResolutionCeiling.regulatory_graph,
            "The simulated design resolves regulatory-graph alternatives but not "
            "molecular binding.",
        ),
        EpisodeFamily.natural_variation: (
            ResolutionCeiling.haplotype_block,
            "The simulated mapping design resolves a haplotype block, not necessarily one variant.",
        ),
        EpisodeFamily.cross_context: (
            ResolutionCeiling.contextual_applicability_only,
            "Without paired interventions, the evidence supports contextual applicability only.",
        ),
    }[family]


def generate_episode(
    family: EpisodeFamily | str,
    *,
    group_index: int,
    episode_index: int = 0,
    seed: int = 0,
    split: DatasetSplit | None = None,
    split_group: str | None = None,
    invalid_flags: Iterable[InvalidDesignFlag | str] = (),
) -> BiologicalEpisode:
    """Generate one deterministic, self-contained biological episode."""

    family = _family(family)
    seed = _checked_seed(seed)
    _checked_count("group_index + 1", group_index + 1)
    _checked_count("episode_index + 1", episode_index + 1)
    flags = _invalid_flags(invalid_flags)
    episode_id = (
        f"sim:{family.value}:g{group_index:04d}:e{episode_index:03d}:s{seed}"
    )
    split_group = split_group or f"sim:{family.value}:group-{group_index:04d}"
    split = split or split_for_group(split_group, seed=seed)
    rng = _rng(seed, family.value, group_index, episode_index, *(flag.value for flag in flags))
    hypotheses = _hypotheses(episode_id, family, rng)
    experiments = _candidate_experiments(episode_id, family, hypotheses, rng)
    graph_builder = {
        EpisodeFamily.developmental: _developmental_graph,
        EpisodeFamily.natural_variation: _natural_variation_graph,
        EpisodeFamily.cross_context: _cross_context_graph,
    }[family]
    graph = graph_builder(episode_id, hypotheses, experiments, flags, rng)
    true_hypothesis = hypotheses[(group_index + episode_index + seed) % len(hypotheses)]
    program = _operator_program(family)
    if flags:
        resolution = ResolutionCeiling.non_identifiable
        resolution_note = (
            "The requested conclusion is not identifiable because the simulated design "
            "contains: " + ", ".join(flag.value for flag in flags) + "."
        )
    else:
        resolution, resolution_note = _valid_resolution(family)
    question = {
        EpisodeFamily.developmental: (
            "Which regulatory explanation best accounts for the first lineage-resolved "
            "effect of this developmental perturbation?"
        ),
        EpisodeFamily.natural_variation: (
            "Which local pangenome mechanism best explains the strain-by-dose response?"
        ),
        EpisodeFamily.cross_context: (
            "How applicable is source-context perturbation evidence to this natural allele "
            "and target background?"
        ),
    }[family]
    return BiologicalEpisode(
        id=episode_id,
        family=family,
        question=question,
        graph=graph,
        hypotheses=hypotheses,
        candidate_experiments=experiments,
        labels=EpisodeLabels(
            operator_program=program,
            operator_multilabel=sorted(program, key=lambda item: item.value),
            design_valid=not flags,
            invalid_design_flags=list(flags),
            true_hypothesis_id=true_hypothesis.id,
            true_equivalence_class=true_hypothesis.equivalence_class,
            resolution_ceiling=resolution,
            resolution_note=resolution_note,
        ),
        split_group=split_group,
        split=split,
        provenance=EpisodeProvenance(seed=seed, scenario_id=episode_id),
    )


def _dataset_flags(
    family: EpisodeFamily,
    group_index: int,
    groups_per_family: int,
) -> tuple[InvalidDesignFlag, ...]:
    placements = {
        EpisodeFamily.developmental: (
            (InvalidDesignFlag.database_absence_as_powered_negative, 0),
        ),
        EpisodeFamily.natural_variation: (
            (InvalidDesignFlag.dominance_from_homozygous_panel, 0),
            (InvalidDesignFlag.absent_locus_as_snp, min(1, groups_per_family - 1)),
            (InvalidDesignFlag.post_treatment_covariate, min(2, groups_per_family - 1)),
        ),
        EpisodeFamily.cross_context: (
            (InvalidDesignFlag.transport_without_identification, 0),
        ),
    }[family]
    return tuple(flag for flag, target_group in placements if target_group == group_index)


def generate_synthetic_dataset(
    *,
    groups_per_family: int = 4,
    episodes_per_group: int = 1,
    seed: int = 0,
    include_red_team: bool = True,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> list[BiologicalEpisode]:
    """Generate all three episode families with group-disjoint dataset splits."""

    groups_per_family = _checked_count("groups_per_family", groups_per_family)
    episodes_per_group = _checked_count("episodes_per_group", episodes_per_group)
    seed = _checked_seed(seed)
    groups = [
        f"sim:{family.value}:group-{group_index:04d}"
        for family in EpisodeFamily
        for group_index in range(groups_per_family)
    ]
    assignments = assign_group_splits(groups, seed=seed, ratios=ratios)
    episodes: list[BiologicalEpisode] = []
    for family in EpisodeFamily:
        for group_index in range(groups_per_family):
            split_group = f"sim:{family.value}:group-{group_index:04d}"
            for episode_index in range(episodes_per_group):
                flags = (
                    _dataset_flags(family, group_index, groups_per_family)
                    if include_red_team and episode_index == 0
                    else ()
                )
                episodes.append(
                    generate_episode(
                        family,
                        group_index=group_index,
                        episode_index=episode_index,
                        seed=seed,
                        split=assignments[split_group],
                        split_group=split_group,
                        invalid_flags=flags,
                    )
                )
    return episodes


def generate_synthetic_episodes(**kwargs: object) -> list[BiologicalEpisode]:
    """Compatibility alias for :func:`generate_synthetic_dataset`."""

    return generate_synthetic_dataset(**kwargs)  # type: ignore[arg-type]


def generate_red_team_episodes(*, seed: int = 0) -> list[BiologicalEpisode]:
    """Generate one isolated episode for each required invalid-design trap."""

    seed = _checked_seed(seed)
    family_by_flag = {
        InvalidDesignFlag.dominance_from_homozygous_panel: EpisodeFamily.natural_variation,
        InvalidDesignFlag.absent_locus_as_snp: EpisodeFamily.natural_variation,
        InvalidDesignFlag.database_absence_as_powered_negative: EpisodeFamily.developmental,
        InvalidDesignFlag.transport_without_identification: EpisodeFamily.cross_context,
        InvalidDesignFlag.post_treatment_covariate: EpisodeFamily.natural_variation,
    }
    groups = [f"sim:redteam:{flag.value}" for flag in RED_TEAM_FLAGS]
    assignments = assign_group_splits(groups, seed=seed)
    episodes = []
    for index, flag in enumerate(RED_TEAM_FLAGS):
        split_group = groups[index]
        episodes.append(
            generate_episode(
                family_by_flag[flag],
                group_index=index,
                seed=seed,
                split=assignments[split_group],
                split_group=split_group,
                invalid_flags=(flag,),
            )
        )
    return episodes


@dataclass(frozen=True, slots=True)
class SyntheticEpisodeGenerator:
    """Small stateful facade convenient for command-line and trainer code."""

    seed: int = 0

    def __post_init__(self) -> None:
        _checked_seed(self.seed)

    def episode(
        self,
        family: EpisodeFamily | str,
        *,
        group_index: int,
        episode_index: int = 0,
        invalid_flags: Iterable[InvalidDesignFlag | str] = (),
    ) -> BiologicalEpisode:
        return generate_episode(
            family,
            group_index=group_index,
            episode_index=episode_index,
            seed=self.seed,
            invalid_flags=invalid_flags,
        )

    def dataset(
        self,
        *,
        groups_per_family: int = 4,
        episodes_per_group: int = 1,
        include_red_team: bool = True,
    ) -> list[BiologicalEpisode]:
        return generate_synthetic_dataset(
            groups_per_family=groups_per_family,
            episodes_per_group=episodes_per_group,
            seed=self.seed,
            include_red_team=include_red_team,
        )

    def red_team(self) -> list[BiologicalEpisode]:
        return generate_red_team_episodes(seed=self.seed)


__all__ = [
    "DEFAULT_SPLIT_RATIOS",
    "RED_TEAM_FLAGS",
    "SyntheticEpisodeGenerator",
    "assign_group_splits",
    "generate_episode",
    "generate_red_team_episodes",
    "generate_synthetic_dataset",
    "generate_synthetic_episodes",
    "split_for_group",
]
