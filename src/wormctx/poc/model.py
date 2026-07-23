"""A small shared graph-conditioned reasoning core for the POC.

This is intentionally not a foundation model.  It is a trainable routing and
belief-state feasibility model over bounded episode graphs.  Authoritative
statistics remain outside the neural network in typed operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OutputDimensions:
    families: int
    operators: int
    invalid_flags: int
    conclusions: int
    hypotheses: int
    experiments: int


class RelationalMessageBlock(nn.Module):
    """Relation-aware message passing without PyG or scatter extensions."""

    def __init__(self, hidden_dim: int, relation_count: int, dropout: float) -> None:
        super().__init__()
        self.relation_embedding = nn.Embedding(relation_count, hidden_dim)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        states: Tensor,
        edge_source: Tensor,
        edge_target: Tensor,
        edge_type: Tensor,
        edge_mask: Tensor,
    ) -> Tensor:
        batch_size, node_count, hidden_dim = states.shape
        source_index = edge_source.clamp(min=0, max=node_count - 1)
        gather_index = source_index.unsqueeze(-1).expand(-1, -1, hidden_dim)
        source_states = torch.gather(states, dim=1, index=gather_index)
        relations = self.relation_embedding(edge_type)
        messages = self.message(source_states + relations)
        messages = messages * edge_mask.unsqueeze(-1)

        # One-hot destination aggregation is portable and avoids a dependency
        # on torch-scatter. Episode graphs are deliberately bounded, so this
        # dense operation is modest for the POC presets.
        destinations = F.one_hot(
            edge_target.clamp(min=0, max=node_count - 1), num_classes=node_count
        ).to(messages.dtype)
        destinations = destinations * edge_mask.unsqueeze(-1)
        aggregated = torch.einsum("ben,beh->bnh", destinations, messages)
        degree = destinations.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        aggregated = aggregated / degree
        delta = self.update(torch.cat((states, aggregated), dim=-1))
        return self.normalization(states + delta)


class SharedGraphReasoner(nn.Module):
    """One shared encoder with typed multi-task output heads."""

    def __init__(
        self,
        *,
        numeric_feature_dim: int,
        hidden_dim: int,
        node_type_count: int,
        edge_type_count: int,
        transformer_layers: int,
        message_passing_layers: int,
        attention_heads: int,
        dropout: float,
        outputs: OutputDimensions,
    ) -> None:
        super().__init__()
        self.outputs = outputs
        self.feature_projection = nn.Sequential(
            nn.Linear(numeric_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.node_type_embedding = nn.Embedding(node_type_count, hidden_dim)
        self.message_blocks = nn.ModuleList(
            RelationalMessageBlock(hidden_dim, edge_type_count, dropout)
            for _ in range(message_passing_layers)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.query_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.query_token, std=0.02)
        self.final_normalization = nn.LayerNorm(hidden_dim)

        def head(size: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, size),
            )

        self.family_head = head(outputs.families)
        self.operator_head = head(outputs.operators)
        self.invalid_head = head(outputs.invalid_flags)
        self.conclusion_head = head(outputs.conclusions)
        self.hypothesis_head = head(outputs.hypotheses)
        self.experiment_head = head(outputs.experiments)
        self.resolution_head = head(1)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        states = self.feature_projection(batch["numeric_features"])
        states = states + self.node_type_embedding(batch["node_type"])
        states = states * batch["node_mask"].unsqueeze(-1)
        for block in self.message_blocks:
            states = block(
                states,
                batch["edge_source"],
                batch["edge_target"],
                batch["edge_type"],
                batch["edge_mask"],
            )
            states = states * batch["node_mask"].unsqueeze(-1)

        query = self.query_token.expand(states.shape[0], -1, -1)
        sequence = torch.cat((query, states), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(
                    states.shape[0], 1, dtype=torch.bool, device=states.device
                ),
                ~batch["node_mask"].bool(),
            ),
            dim=1,
        )
        sequence = self.transformer(sequence, src_key_padding_mask=padding_mask)
        pooled = self.final_normalization(sequence[:, 0])
        return {
            "family_logits": self.family_head(pooled),
            "operator_logits": self.operator_head(pooled),
            "invalid_logits": self.invalid_head(pooled),
            "conclusion_logits": self.conclusion_head(pooled),
            "hypothesis_logits": self.hypothesis_head(pooled),
            "experiment_logits": self.experiment_head(pooled),
            "resolution": torch.sigmoid(self.resolution_head(pooled)).squeeze(-1),
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def masked_cross_entropy(logits: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    """Cross entropy that excludes padded candidate positions."""
    masked_logits = logits.masked_fill(~valid_mask.bool(), torch.finfo(logits.dtype).min)
    return F.cross_entropy(masked_logits, target)


def multitask_loss(
    predictions: dict[str, Tensor],
    batch: dict[str, Tensor],
    *,
    invalid_pos_weight: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    losses: dict[str, Tensor] = {
        "family": F.cross_entropy(predictions["family_logits"], batch["family_target"]),
        "operators": F.binary_cross_entropy_with_logits(
            predictions["operator_logits"], batch["operator_targets"]
        ),
        "invalid": F.binary_cross_entropy_with_logits(
            predictions["invalid_logits"],
            batch["invalid_targets"],
            pos_weight=invalid_pos_weight,
        ),
        "conclusion": F.cross_entropy(
            predictions["conclusion_logits"], batch["conclusion_target"]
        ),
        "hypothesis": masked_cross_entropy(
            predictions["hypothesis_logits"],
            batch["hypothesis_target"],
            batch["hypothesis_mask"],
        ),
        "experiment": masked_cross_entropy(
            predictions["experiment_logits"],
            batch["experiment_target"],
            batch["experiment_mask"],
        ),
        "resolution": F.smooth_l1_loss(
            predictions["resolution"], batch["resolution_target"]
        ),
    }
    weights = {
        "family": 0.25,
        "operators": 1.0,
        "invalid": 1.25,
        "conclusion": 0.75,
        "hypothesis": 0.75,
        "experiment": 0.75,
        "resolution": 0.5,
    }
    total = sum(losses[name] * weights[name] for name in losses)
    return total, losses


def model_metadata(model: SharedGraphReasoner) -> dict[str, Any]:
    return {
        "architecture": type(model).__name__,
        "parameter_count": model.parameter_count(),
        "output_dimensions": model.outputs.__dict__,
    }
