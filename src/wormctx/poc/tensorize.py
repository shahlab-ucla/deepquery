"""Projection from validated episode contracts to bounded PyTorch tensors."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .contracts import (
    CONCLUSION_ORDER,
    EDGE_TYPE_ORDER,
    FAMILY_ORDER,
    INVALID_FLAG_ORDER,
    NODE_TYPE_ORDER,
    OPERATOR_ORDER,
    BiologicalEpisode,
)


VOCABULARY = {
    "families": FAMILY_ORDER,
    "operators": OPERATOR_ORDER,
    "invalid_flags": INVALID_FLAG_ORDER,
    "conclusions": CONCLUSION_ORDER,
    "node_types": NODE_TYPE_ORDER,
    "edge_types": EDGE_TYPE_ORDER,
}


def _index(values: Sequence[str]) -> dict[str, int]:
    return {value: position for position, value in enumerate(values)}


FAMILY_INDEX = _index(FAMILY_ORDER)
OPERATOR_INDEX = _index(OPERATOR_ORDER)
INVALID_INDEX = _index(INVALID_FLAG_ORDER)
CONCLUSION_INDEX = _index(CONCLUSION_ORDER)
NODE_INDEX = _index(NODE_TYPE_ORDER)
EDGE_INDEX = _index(EDGE_TYPE_ORDER)


def _padded_features(values: Sequence[float], size: int) -> list[float]:
    result = [float(value) for value in values[:size]]
    result.extend([0.0] * (size - len(result)))
    return result


def choose_experiment(episode: BiologicalEpisode, risk_weight: float = 1.0) -> int:
    """Return the utility-maximizing experiment label used by the simulation."""
    utilities = [
        item.expected_information_gain / (item.cost + risk_weight * item.risk)
        for item in episode.candidate_experiments
    ]
    return max(range(len(utilities)), key=lambda index: (utilities[index], -index))


def tensorize_episode(
    episode: BiologicalEpisode,
    *,
    max_nodes: int,
    max_edges: int,
    numeric_feature_dim: int,
    max_hypotheses: int,
    max_experiments: int,
) -> dict[str, Tensor | str]:
    if len(episode.graph.nodes) > max_nodes:
        raise ValueError(
            f"episode {episode.id!r} has {len(episode.graph.nodes)} nodes; max_nodes={max_nodes}"
        )
    if len(episode.graph.edges) > max_edges:
        raise ValueError(
            f"episode {episode.id!r} has {len(episode.graph.edges)} edges; max_edges={max_edges}"
        )
    if len(episode.hypotheses) > max_hypotheses:
        raise ValueError("max_hypotheses is smaller than a validated episode")
    if len(episode.candidate_experiments) > max_experiments:
        raise ValueError("max_experiments is smaller than a validated episode")

    node_by_id = {node.id: index for index, node in enumerate(episode.graph.nodes)}
    numeric_features = torch.zeros(max_nodes, numeric_feature_dim, dtype=torch.float32)
    node_type = torch.zeros(max_nodes, dtype=torch.long)
    node_mask = torch.zeros(max_nodes, dtype=torch.bool)
    for position, node in enumerate(episode.graph.nodes):
        numeric_features[position] = torch.tensor(
            _padded_features(node.numeric_features, numeric_feature_dim),
            dtype=torch.float32,
        )
        node_type[position] = NODE_INDEX[node.kind.value]
        node_mask[position] = True

    edge_source = torch.zeros(max_edges, dtype=torch.long)
    edge_target = torch.zeros(max_edges, dtype=torch.long)
    edge_type = torch.zeros(max_edges, dtype=torch.long)
    edge_mask = torch.zeros(max_edges, dtype=torch.bool)
    for position, edge in enumerate(episode.graph.edges):
        edge_source[position] = node_by_id[edge.source]
        edge_target[position] = node_by_id[edge.target]
        edge_type[position] = EDGE_INDEX[edge.kind.value]
        edge_mask[position] = True

    operator_targets = torch.zeros(len(OPERATOR_ORDER), dtype=torch.float32)
    for operator in episode.labels.operator_multilabel:
        operator_targets[OPERATOR_INDEX[operator.value]] = 1.0
    invalid_targets = torch.zeros(len(INVALID_FLAG_ORDER), dtype=torch.float32)
    for flag in episode.labels.invalid_design_flags:
        invalid_targets[INVALID_INDEX[flag.value]] = 1.0

    hypothesis_ids = [item.id for item in episode.hypotheses]
    hypothesis_mask = torch.zeros(max_hypotheses, dtype=torch.bool)
    hypothesis_mask[: len(hypothesis_ids)] = True
    experiment_mask = torch.zeros(max_experiments, dtype=torch.bool)
    experiment_mask[: len(episode.candidate_experiments)] = True

    return {
        "episode_id": episode.id,
        "split_group": episode.split_group,
        "numeric_features": numeric_features,
        "node_type": node_type,
        "node_mask": node_mask,
        "edge_source": edge_source,
        "edge_target": edge_target,
        "edge_type": edge_type,
        "edge_mask": edge_mask,
        "family_target": torch.tensor(FAMILY_INDEX[episode.family.value], dtype=torch.long),
        "operator_targets": operator_targets,
        "invalid_targets": invalid_targets,
        "conclusion_target": torch.tensor(
            CONCLUSION_INDEX[episode.labels.resolution_ceiling.value], dtype=torch.long
        ),
        "hypothesis_target": torch.tensor(
            hypothesis_ids.index(episode.labels.true_hypothesis_id), dtype=torch.long
        ),
        "hypothesis_mask": hypothesis_mask,
        "experiment_target": torch.tensor(choose_experiment(episode), dtype=torch.long),
        "experiment_mask": experiment_mask,
        "resolution_target": torch.tensor(
            resolution_score(episode.labels.resolution_ceiling.value), dtype=torch.float32
        ),
    }


def resolution_score(value: str) -> float:
    """An explicitly ordinal POC target; not a probability or confidence."""
    return {
        "variant": 1.0,
        "gene": 0.85,
        "haplotype_block": 0.65,
        "regulatory_graph": 0.70,
        "mechanistic_equivalence_class": 0.35,
        "contextual_applicability_only": 0.20,
        "non_identifiable": 0.0,
    }[value]


class EpisodeTensorDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        episodes: Sequence[BiologicalEpisode],
        *,
        max_nodes: int,
        numeric_feature_dim: int,
        max_hypotheses: int,
        max_experiments: int,
        max_edges: int | None = None,
    ) -> None:
        self.episodes = list(episodes)
        if not self.episodes:
            raise ValueError("an episode tensor dataset cannot be empty")
        self.max_edges = max_edges or max(len(item.graph.edges) for item in self.episodes)
        self.items = [
            tensorize_episode(
                episode,
                max_nodes=max_nodes,
                max_edges=self.max_edges,
                numeric_feature_dim=numeric_feature_dim,
                max_hypotheses=max_hypotheses,
                max_experiments=max_experiments,
            )
            for episode in self.episodes
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        return self.items[index]
