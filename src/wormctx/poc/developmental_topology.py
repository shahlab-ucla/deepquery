"""Governed topology benchmark primitives for developmental prediction.

The module is deliberately independent of the OMIX709 workbook reader.  A
forthcoming semantic-stage adapter may emit the tensor contract documented by
``validate_topology_batch`` without exposing outcomes to graph construction.
The benchmark models and null generators can therefore be exercised on
synthetic fixtures before any sealed developmental endpoint is opened.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TOPOLOGY_BENCHMARK_SCHEMA_VERSION = "wormctx-developmental-topology-benchmark-1.0"
PARENT_TO_CHILD = 0
CHILD_TO_PARENT = 1
SELF = 2
RELATION_COUNT = 3


class TopologyVariant(str, Enum):
    """Predeclared structural signals in the capacity-matched model ladder."""

    REAL_LINEAGE = "real_lineage"
    REWIRED_LINEAGE = "rewired_lineage"
    EDGE_FREE_SELF_LOOP = "edge_free_self_loop"
    SEMANTIC_IDENTITY_ONLY = "semantic_identity_only"
    DEPTH_STAGE_ONLY = "depth_stage_only"


@dataclass(frozen=True)
class TopologyModelConfig:
    """Shape-only model configuration; it contains no fitted quantities."""

    numeric_feature_dim: int
    semantic_vocab_size: int
    depth_vocab_size: int
    stage_vocab_size: int
    hidden_dim: int = 64
    message_passing_layers: int = 2
    outcome_dim: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        integer_fields = (
            "numeric_feature_dim",
            "semantic_vocab_size",
            "depth_vocab_size",
            "stage_vocab_size",
            "hidden_dim",
            "message_passing_layers",
            "outcome_dim",
        )
        for name in integer_fields:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class SelectionContract:
    """Fail-closed contract for validation-only model and seed selection."""

    split_unit: str
    tuning_partition: str
    sealed_partition: str
    primary_metric: str
    minimize: bool
    allowed_seeds: tuple[int, ...]
    sealed_test_openings: int = 1

    def __post_init__(self) -> None:
        if self.split_unit != "whole_gene":
            raise ValueError("topology selection must use whole_gene as the split unit")
        if self.tuning_partition != "validation":
            raise ValueError("hyperparameters and seeds may be selected only on validation")
        if self.sealed_partition != "sealed_test":
            raise ValueError("the final partition must be named sealed_test")
        if not self.primary_metric:
            raise ValueError("primary_metric cannot be empty")
        if not self.allowed_seeds or len(set(self.allowed_seeds)) != len(self.allowed_seeds):
            raise ValueError("allowed_seeds must be a nonempty unique sequence")
        if self.sealed_test_openings != 1:
            raise ValueError("the sealed test may be opened exactly once")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SelectionContract":
        return cls(
            split_unit=str(payload["split_unit"]),
            tuning_partition=str(payload["tuning_partition"]),
            sealed_partition=str(payload["sealed_partition"]),
            primary_metric=str(payload["primary_metric"]),
            minimize=bool(payload["minimize"]),
            allowed_seeds=tuple(int(item) for item in payload["allowed_seeds"]),
            sealed_test_openings=int(payload["sealed_test_openings"]),
        )


@dataclass(frozen=True)
class RewiredTreeNull:
    """One deterministic topology null and its auditable invariance receipt."""

    parent_index: Tensor
    receipt: dict[str, Any]


@dataclass(frozen=True)
class LearningCurvePoint:
    fraction: float
    seed: int
    gene_ids: tuple[str, ...]
    gene_set_sha256: str


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    """Load the predeclared scaffold config and validate its selection contract."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != TOPOLOGY_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unexpected topology benchmark schema version")
    SelectionContract.from_mapping(payload["selection"])
    declared = tuple(payload.get("model_ladder", ()))
    expected = tuple(item.value for item in TopologyVariant)
    if declared != expected:
        raise ValueError(f"model_ladder must be frozen in this order: {expected}")
    if payload.get("sealed_outcomes_inspected") is not False:
        raise ValueError("scaffold configuration must record sealed_outcomes_inspected=false")
    return payload


def topology_arrays_from_stage_adapter(
    lineage_manifest: Mapping[str, Any],
    stage_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert frozen outcome-blind adapter manifests into aligned graph arrays.

    This bridge deliberately consumes only ``lineage_manifest.json`` and
    ``stage_membership.json``. It cannot read an endpoint matrix. Numeric
    feature tensorization and modality-specific eligibility remain downstream.
    """

    if lineage_manifest.get("schema_version") != "wormctx-omix709-semantic-lineage-1.0":
        raise ValueError("unexpected semantic-lineage manifest schema")
    if stage_manifest.get("schema_version") != "wormctx-omix709-semantic-stage-membership-1.0":
        raise ValueError("unexpected semantic-stage manifest schema")
    if (
        stage_manifest.get("perturbation_measurement_values_read") is not False
        or stage_manifest.get("s4_outcome_values_read") is not False
    ):
        raise ValueError("topology construction requires an outcome-blind stage receipt")
    root = str(lineage_manifest["virtual_root"])
    cells = tuple(sorted(str(item) for item in lineage_manifest["cells_through_200"]))
    if not cells or len(set(cells)) != len(cells) or root not in cells:
        raise ValueError("cells_through_200 must uniquely contain the virtual root")
    position = {cell: index for index, cell in enumerate(cells)}
    parent_by_child: dict[str, str] = {}
    for edge in lineage_manifest["edges_through_200"]:
        parent = str(edge["parent"])
        child = str(edge["child"])
        if parent not in position or child not in position:
            raise ValueError("lineage edge references a cell outside cells_through_200")
        if child == root or child in parent_by_child:
            raise ValueError("each nonroot cell must have exactly one frozen parent")
        parent_by_child[child] = parent
    if set(parent_by_child) != set(cells) - {root}:
        raise ValueError("lineage edges must connect every nonroot cell")

    depth_cache = {root: 0}

    def depth(cell: str, trail: frozenset[str] = frozenset()) -> int:
        if cell in depth_cache:
            return depth_cache[cell]
        if cell in trail:
            raise ValueError("lineage manifest contains a cycle")
        parent = parent_by_child[cell]
        result = depth(parent, trail | {cell}) + 1
        depth_cache[cell] = result
        return result

    depths = tuple(depth(cell) for cell in cells)
    input_membership = stage_manifest.get("input_26", {})
    endpoint_membership = stage_manifest.get("endpoint_200", {})
    input_frontier = set(str(item) for item in input_membership.get("frontier", ()))
    endpoint_frontier = set(str(item) for item in endpoint_membership.get("frontier", ()))
    born_by_input = set(str(item) for item in input_membership.get("born_cells", ()))
    born_by_endpoint = set(str(item) for item in endpoint_membership.get("born_cells", ()))
    if len(input_frontier) != 26 or len(endpoint_frontier) != 200:
        raise ValueError("stage adapter must freeze exact 26- and 200-cell frontiers")
    if not input_frontier <= born_by_input or not endpoint_frontier <= born_by_endpoint:
        raise ValueError("each stage frontier must be a subset of cells born by that stage")
    if born_by_endpoint | {root} != set(cells) or not born_by_input <= born_by_endpoint:
        raise ValueError("stage memberships do not match cells_through_200")
    stage_ids = tuple(
        0 if cell == root else 1 if cell in born_by_input else 2 for cell in cells
    )
    parent_indices = tuple(
        -1 if cell == root else position[parent_by_child[cell]] for cell in cells
    )
    parent_tensor = torch.tensor(parent_indices, dtype=torch.long)
    depth_tensor = torch.tensor(depths, dtype=torch.long)
    validate_parent_forest(parent_tensor, depth_tensor)
    receipt_payload = {
        "cell_ids": cells,
        "parent_index": parent_indices,
        "depth_id": depths,
        "stage_id": stage_ids,
    }
    return {
        "cell_ids": cells,
        "semantic_id": torch.arange(len(cells), dtype=torch.long),
        "parent_index": parent_tensor,
        "depth_id": depth_tensor,
        "stage_id": torch.tensor(stage_ids, dtype=torch.long),
        "born_by_input_mask": torch.tensor(
            [cell in born_by_input for cell in cells], dtype=torch.bool
        ),
        "endpoint_frontier_mask": torch.tensor(
            [cell in endpoint_frontier for cell in cells], dtype=torch.bool
        ),
        "receipt": {
            "schema_version": TOPOLOGY_BENCHMARK_SCHEMA_VERSION,
            "source": "outcome_blind_semantic_stage_adapter",
            "cell_count_through_200": len(cells),
            "input_frontier_count": len(input_frontier),
            "endpoint_frontier_count": len(endpoint_frontier),
            "arrays_sha256": hashlib.sha256(
                json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "outcome_values_read": False,
        },
    }


def _hash_integer_sequence(values: Sequence[int]) -> str:
    encoded = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_parent_forest(
    parent_index: Tensor,
    depth_id: Tensor,
    node_mask: Tensor | None = None,
) -> None:
    """Validate a rooted forest whose every edge advances exactly one depth."""

    if parent_index.ndim != 1 or depth_id.ndim != 1 or parent_index.shape != depth_id.shape:
        raise ValueError("parent_index and depth_id must be same-length rank-one tensors")
    node_count = int(parent_index.numel())
    active = (
        torch.ones(node_count, dtype=torch.bool, device=parent_index.device)
        if node_mask is None
        else node_mask.bool()
    )
    if active.ndim != 1 or active.shape != parent_index.shape:
        raise ValueError("node_mask must match parent_index")
    if not bool(active.any()):
        raise ValueError("a topology must contain at least one active node")
    active_indices = torch.nonzero(active, as_tuple=False).flatten().tolist()
    roots = 0
    for child in active_indices:
        parent = int(parent_index[child].item())
        depth = int(depth_id[child].item())
        if depth < 0:
            raise ValueError("active node depths must be nonnegative")
        if parent < 0:
            roots += 1
            if depth != 0:
                raise ValueError("root nodes must have depth zero")
            continue
        if parent >= node_count or not bool(active[parent]):
            raise ValueError("every active nonroot parent must be an active node")
        if parent == child:
            raise ValueError("parent self-links are not permitted in the lineage forest")
        if int(depth_id[parent].item()) + 1 != depth:
            raise ValueError("each parent/child edge must advance exactly one depth")
    if roots < 1:
        raise ValueError("a topology must contain at least one root")


def _outdegrees(parent_index: Tensor) -> dict[int, int]:
    counts: dict[int, int] = {}
    for parent in parent_index.tolist():
        if int(parent) >= 0:
            counts[int(parent)] = counts.get(int(parent), 0) + 1
    return counts


def rewire_parent_index(
    parent_index: Tensor,
    depth_id: Tensor,
    *,
    seed: int,
) -> RewiredTreeNull:
    """Reassign children to same-depth parents while preserving parent outdegrees.

    Child identities and depths stay fixed.  At each depth, the original parent
    labels form a multiset of parent "slots"; seeded permutation of those slots
    preserves the exact outdegree of every parent.  Edges cannot create cycles
    because every new parent remains exactly one level above its child.
    """

    original = parent_index.detach().cpu().to(torch.long).clone()
    depths = depth_id.detach().cpu().to(torch.long).clone()
    validate_parent_forest(original, depths)
    rewired = original.clone()
    rng = random.Random(int(seed))
    for depth in sorted(set(int(item) for item in depths.tolist()) - {0}):
        children = [index for index, value in enumerate(depths.tolist()) if value == depth]
        slots = [int(original[child].item()) for child in children]
        candidate = list(slots)
        rng.shuffle(candidate)
        if candidate == slots and len(set(slots)) > 1:
            for shift in range(1, len(candidate)):
                rotated = candidate[shift:] + candidate[:shift]
                if rotated != slots:
                    candidate = rotated
                    break
        for child, parent in zip(children, candidate, strict=True):
            rewired[child] = parent
    validate_parent_forest(rewired, depths)
    if _outdegrees(rewired) != _outdegrees(original):  # pragma: no cover - invariant guard
        raise AssertionError("rewiring changed an outdegree")
    changed = int((rewired != original).sum().item())
    receipt = {
        "schema_version": TOPOLOGY_BENCHMARK_SCHEMA_VERSION,
        "null_type": "degree_depth_preserving_rewired_tree",
        "seed": int(seed),
        "node_count": int(original.numel()),
        "changed_parent_assignments": changed,
        "degree_preserved": True,
        "depth_preserved": True,
        "original_parent_sha256": _hash_integer_sequence(original.tolist()),
        "rewired_parent_sha256": _hash_integer_sequence(rewired.tolist()),
    }
    return RewiredTreeNull(parent_index=rewired, receipt=receipt)


def generate_rewired_tree_nulls(
    parent_index: Tensor,
    depth_id: Tensor,
    *,
    count: int,
    base_seed: int,
) -> tuple[RewiredTreeNull, ...]:
    """Generate a deterministic series of rewired trees with explicit seeds."""

    if count < 1:
        raise ValueError("count must be positive")
    return tuple(
        rewire_parent_index(parent_index, depth_id, seed=base_seed + 104729 * index)
        for index in range(count)
    )


def permute_group_labels(group_ids: Sequence[str], *, seed: int) -> tuple[str, ...]:
    """Bijectively relabel whole groups without splitting their sample membership."""

    unique = sorted(set(str(item) for item in group_ids))
    if len(unique) < 2:
        raise ValueError("at least two distinct group labels are required")
    shuffled = list(unique)
    random.Random(int(seed)).shuffle(shuffled)
    if shuffled == unique:
        shuffled = shuffled[1:] + shuffled[:1]
    mapping = dict(zip(unique, shuffled, strict=True))
    return tuple(mapping[str(item)] for item in group_ids)


def permute_sample_targets(target: Tensor, *, seed: int) -> tuple[Tensor, Tensor]:
    """Return a seeded sample-level target permutation and its explicit index."""

    if target.ndim < 1 or target.shape[0] < 2:
        raise ValueError("target must contain at least two samples")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(target.shape[0], generator=generator)
    return target.detach().cpu()[order], order


def validate_topology_batch(batch: Mapping[str, Tensor], config: TopologyModelConfig) -> None:
    """Validate the tensor boundary expected from the semantic-stage adapter.

    Required keys are ``numeric_features`` [B,N,F], ``node_mask`` [B,N],
    ``parent_index`` [B,N] (-1 for roots/padding), and the aligned integer
    ``semantic_id``, ``depth_id``, and ``stage_id`` arrays [B,N].
    """

    required = {
        "numeric_features",
        "node_mask",
        "parent_index",
        "semantic_id",
        "depth_id",
        "stage_id",
    }
    missing = required - set(batch)
    if missing:
        raise ValueError(f"topology batch is missing keys: {sorted(missing)}")
    features = batch["numeric_features"]
    if features.ndim != 3 or features.shape[-1] != config.numeric_feature_dim:
        raise ValueError("numeric_features has an incompatible shape")
    batch_size, node_count, _ = features.shape
    aligned_shape = (batch_size, node_count)
    for name in required - {"numeric_features"}:
        if tuple(batch[name].shape) != aligned_shape:
            raise ValueError(f"{name} must have shape {aligned_shape}")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("numeric_features must be finite after fold-local preprocessing")
    node_mask = batch["node_mask"].bool()
    if not bool(node_mask.any(dim=1).all()):
        raise ValueError("every sample must contain at least one active node")
    limits = {
        "semantic_id": config.semantic_vocab_size,
        "depth_id": config.depth_vocab_size,
        "stage_id": config.stage_vocab_size,
    }
    for name, limit in limits.items():
        values = batch[name][node_mask]
        if bool((values < 0).any()) or bool((values >= limit).any()):
            raise ValueError(f"active {name} values must be in [0, {limit})")
    for sample in range(batch_size):
        validate_parent_forest(
            batch["parent_index"][sample].detach().cpu(),
            batch["depth_id"][sample].detach().cpu(),
            node_mask[sample].detach().cpu(),
        )


def build_message_edges(
    parent_index: Tensor,
    node_mask: Tensor,
    *,
    use_lineage_edges: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build bidirectional parent/child plus self edges, or matched self controls."""

    if parent_index.ndim != 2 or node_mask.shape != parent_index.shape:
        raise ValueError("parent_index and node_mask must be aligned rank-two tensors")
    batch_size, node_count = parent_index.shape
    nodes = torch.arange(node_count, device=parent_index.device).expand(batch_size, -1)
    if use_lineage_edges:
        parent = parent_index.clamp(min=0, max=node_count - 1)
        child_valid = node_mask.bool() & parent_index.ge(0)
        source = torch.cat((parent, nodes, nodes), dim=1)
        target = torch.cat((nodes, parent, nodes), dim=1)
        edge_type = torch.cat(
            (
                torch.full_like(nodes, PARENT_TO_CHILD),
                torch.full_like(nodes, CHILD_TO_PARENT),
                torch.full_like(nodes, SELF),
            ),
            dim=1,
        )
        edge_mask = torch.cat((child_valid, child_valid, node_mask.bool()), dim=1)
    else:
        # Three typed self messages keep all relation-specific capacity active.
        source = torch.cat((nodes, nodes, nodes), dim=1)
        target = source.clone()
        edge_type = torch.cat(
            (
                torch.full_like(nodes, PARENT_TO_CHILD),
                torch.full_like(nodes, CHILD_TO_PARENT),
                torch.full_like(nodes, SELF),
            ),
            dim=1,
        )
        edge_mask = torch.cat((node_mask.bool(),) * RELATION_COUNT, dim=1)
    return source, target, edge_type, edge_mask


class DirectionalMessageBlock(nn.Module):
    """Portable direction-aware message passing without PyG dependencies."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.relation_embedding = nn.Embedding(RELATION_COUNT, hidden_dim)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
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
        _, node_count, hidden_dim = states.shape
        source_index = edge_source.unsqueeze(-1).expand(-1, -1, hidden_dim)
        source_state = torch.gather(states, 1, source_index)
        relation = self.relation_embedding(edge_type)
        messages = self.message(torch.cat((source_state, relation), dim=-1))
        messages = messages * edge_mask.unsqueeze(-1)
        destination = F.one_hot(edge_target, num_classes=node_count).to(messages.dtype)
        destination = destination * edge_mask.unsqueeze(-1)
        aggregate = torch.einsum("ben,beh->bnh", destination, messages)
        degree = destination.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        delta = self.update(torch.cat((states, aggregate / degree), dim=-1))
        return self.normalization(states + delta)


def _variant_signals(variant: TopologyVariant) -> tuple[bool, bool, bool]:
    """Return (lineage edges, semantic identity, depth/stage identity)."""

    if variant in {TopologyVariant.REAL_LINEAGE, TopologyVariant.REWIRED_LINEAGE}:
        return True, True, True
    if variant is TopologyVariant.EDGE_FREE_SELF_LOOP:
        return False, True, True
    if variant is TopologyVariant.SEMANTIC_IDENTITY_ONLY:
        return False, True, False
    if variant is TopologyVariant.DEPTH_STAGE_ONLY:
        return False, False, True
    raise AssertionError(variant)  # pragma: no cover


class DevelopmentalTopologyRegressor(nn.Module):
    """Capacity-matched predictor with an explicit structural ablation switch."""

    def __init__(self, config: TopologyModelConfig, variant: TopologyVariant) -> None:
        super().__init__()
        self.config = config
        self.variant = TopologyVariant(variant)
        hidden = config.hidden_dim
        self.numeric_projection = nn.Linear(config.numeric_feature_dim, hidden)
        self.semantic_embedding = nn.Embedding(config.semantic_vocab_size, hidden)
        self.depth_embedding = nn.Embedding(config.depth_vocab_size, hidden)
        self.stage_embedding = nn.Embedding(config.stage_vocab_size, hidden)
        self.input_normalization = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList(
            DirectionalMessageBlock(hidden, config.dropout)
            for _ in range(config.message_passing_layers)
        )
        self.pool_score = nn.Linear(hidden, 1)
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.outcome_dim * 2),
        )
        self.localization_head = nn.Linear(hidden, config.outcome_dim)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        validate_topology_batch(batch, self.config)
        use_edges, use_semantic, use_depth_stage = _variant_signals(self.variant)
        states = self.numeric_projection(batch["numeric_features"])
        if use_semantic:
            states = states + self.semantic_embedding(batch["semantic_id"])
        if use_depth_stage:
            states = states + self.depth_embedding(batch["depth_id"])
            states = states + self.stage_embedding(batch["stage_id"])
        states = F.gelu(self.input_normalization(states))
        node_mask = batch["node_mask"].bool()
        states = states * node_mask.unsqueeze(-1)
        edges = build_message_edges(
            batch["parent_index"], node_mask, use_lineage_edges=use_edges
        )
        for block in self.blocks:
            states = block(states, *edges)
            states = states * node_mask.unsqueeze(-1)
        attention_logits = self.pool_score(states).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~node_mask, torch.finfo(states.dtype).min)
        attention = torch.softmax(attention_logits, dim=1)
        pooled = torch.einsum("bn,bnh->bh", attention, states)
        outcome_parameters = self.outcome_head(pooled)
        mean, log_variance = outcome_parameters.chunk(2, dim=-1)
        localization = self.localization_head(states)
        localization = localization.masked_fill(~node_mask.unsqueeze(-1), -torch.inf)
        return {
            "mean": mean,
            "log_variance": log_variance.clamp(min=-8.0, max=6.0),
            "localization_logits": localization,
            "pooling_weights": attention,
            "node_states": states,
        }


def make_capacity_matched_models(
    config: TopologyModelConfig,
    variants: Iterable[TopologyVariant],
    *,
    initialization_seed: int,
) -> dict[TopologyVariant, DevelopmentalTopologyRegressor]:
    """Instantiate every variant from one identical parameter initialization."""

    requested = tuple(TopologyVariant(item) for item in variants)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("variants must be a nonempty unique sequence")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(initialization_seed))
        template = DevelopmentalTopologyRegressor(config, requested[0])
    state = template.state_dict()
    models: dict[TopologyVariant, DevelopmentalTopologyRegressor] = {}
    for variant in requested:
        model = DevelopmentalTopologyRegressor(config, variant)
        model.load_state_dict(state)
        models[variant] = model
    counts = {model.parameter_count() for model in models.values()}
    if len(counts) != 1:  # pragma: no cover - architecture invariant
        raise AssertionError("capacity-matched variants have unequal parameter counts")
    return models


def developmental_multitask_loss(
    predictions: Mapping[str, Tensor],
    target: Tensor,
    *,
    localization_target: Tensor | None = None,
    localization_weight: float = 0.25,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Gaussian outcome loss with an optional node-localization objective."""

    mean = predictions["mean"]
    log_variance = predictions["log_variance"]
    if target.shape != mean.shape:
        raise ValueError("target must match the predicted outcome shape")
    gaussian = 0.5 * (log_variance + (target - mean).square() * torch.exp(-log_variance))
    losses = {"outcome_gaussian_nll": gaussian.mean()}
    total = losses["outcome_gaussian_nll"]
    if localization_target is not None:
        logits = predictions["localization_logits"]
        if localization_target.shape != mean.shape:
            raise ValueError("localization_target must have shape [batch, outcome]")
        flat_logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[1])
        flat_target = localization_target.reshape(-1).long()
        valid = flat_target.ge(0)
        if bool(valid.any()):
            localization_loss = F.cross_entropy(flat_logits[valid], flat_target[valid])
            losses["localization_cross_entropy"] = localization_loss
            total = total + float(localization_weight) * localization_loss
    return total, losses


def _vector(values: Tensor, name: str) -> Tensor:
    result = values.detach().cpu().to(torch.float64).flatten()
    if result.numel() < 1 or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain finite values")
    return result


def _correlation(left: Tensor, right: Tensor) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.sqrt(left_centered.square().sum() * right_centered.square().sum())
    if float(denominator) == 0.0:
        return 0.0
    return float((left_centered * right_centered).sum() / denominator)


def prediction_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    gene_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Outcome metrics with optional whole-gene macro aggregation."""

    predicted = _vector(prediction, "prediction")
    observed = _vector(target, "target")
    if predicted.shape != observed.shape:
        raise ValueError("prediction and target must have equal lengths")
    error = predicted - observed
    result = {
        "rmse": float(torch.sqrt(error.square().mean())),
        "mae": float(error.abs().mean()),
        "pearson_r": _correlation(predicted, observed),
    }
    if gene_ids is not None:
        if len(gene_ids) != predicted.numel():
            raise ValueError("gene_ids must align with prediction rows")
        per_gene_rmse = []
        per_gene_mae = []
        for gene in sorted(set(str(item) for item in gene_ids)):
            indices = [index for index, item in enumerate(gene_ids) if str(item) == gene]
            group_error = error[indices]
            per_gene_rmse.append(float(torch.sqrt(group_error.square().mean())))
            per_gene_mae.append(float(group_error.abs().mean()))
        result["gene_macro_rmse"] = sum(per_gene_rmse) / len(per_gene_rmse)
        result["gene_macro_mae"] = sum(per_gene_mae) / len(per_gene_mae)
    return result


def regression_calibration_metrics(
    mean: Tensor,
    log_variance: Tensor,
    target: Tensor,
) -> dict[str, float]:
    """Gaussian NLL and empirical central-interval coverage diagnostics."""

    predicted = _vector(mean, "mean")
    log_var = _vector(log_variance, "log_variance")
    observed = _vector(target, "target")
    if predicted.shape != observed.shape or log_var.shape != observed.shape:
        raise ValueError("mean, log_variance, and target must align")
    variance = log_var.exp()
    standard_deviation = variance.sqrt()
    residual = observed - predicted
    result = {
        "gaussian_nll": float(
            (0.5 * (math.log(2.0 * math.pi) + log_var + residual.square() / variance)).mean()
        ),
        "standardized_residual_mean": float((residual / standard_deviation).mean()),
        "standardized_residual_rmse": float(
            torch.sqrt((residual / standard_deviation).square().mean())
        ),
    }
    for label, z_score in (("50", 0.67448975), ("80", 1.28155157), ("95", 1.95996398)):
        covered = residual.abs().le(z_score * standard_deviation)
        result[f"coverage_{label}"] = float(covered.to(torch.float64).mean())
    return result


def _tree_distance(parent: Sequence[int], source: int, target: int) -> int | None:
    ancestors: dict[int, int] = {}
    cursor = source
    distance = 0
    while cursor >= 0 and cursor not in ancestors:
        ancestors[cursor] = distance
        cursor = int(parent[cursor])
        distance += 1
    cursor = target
    distance = 0
    visited: set[int] = set()
    while cursor >= 0 and cursor not in visited:
        if cursor in ancestors:
            return ancestors[cursor] + distance
        visited.add(cursor)
        cursor = int(parent[cursor])
        distance += 1
    return None


def localization_metrics(
    logits: Tensor,
    target_index: Tensor,
    *,
    node_mask: Tensor | None = None,
    parent_index: Tensor | None = None,
) -> dict[str, float]:
    """Rank and lineage-distance metrics for earliest-defect localization."""

    if logits.ndim != 2 or target_index.ndim != 1 or logits.shape[0] != target_index.shape[0]:
        raise ValueError("logits must be [batch,nodes] and target_index [batch]")
    mask = torch.ones_like(logits, dtype=torch.bool) if node_mask is None else node_mask.bool()
    if mask.shape != logits.shape:
        raise ValueError("node_mask must match logits")
    masked = logits.detach().cpu().masked_fill(~mask.detach().cpu(), -torch.inf)
    targets = target_index.detach().cpu().long()
    if bool((targets < 0).any()) or bool((targets >= logits.shape[1]).any()):
        raise ValueError("target indices are outside the node range")
    if not all(bool(mask[index, target]) for index, target in enumerate(targets.tolist())):
        raise ValueError("every localization target must be an active node")
    order = masked.argsort(dim=1, descending=True)
    ranks = []
    for sample, target in enumerate(targets.tolist()):
        ranks.append(int(torch.nonzero(order[sample] == target, as_tuple=False)[0].item()) + 1)
    result = {
        "top1_accuracy": sum(rank == 1 for rank in ranks) / len(ranks),
        "top3_accuracy": sum(rank <= 3 for rank in ranks) / len(ranks),
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / len(ranks),
    }
    if parent_index is not None:
        parents = parent_index.detach().cpu()
        if parents.ndim == 1:
            parents = parents.unsqueeze(0).expand(logits.shape[0], -1)
        if tuple(parents.shape) != tuple(logits.shape):
            raise ValueError("parent_index must be [nodes] or [batch,nodes]")
        predicted = order[:, 0]
        distances = []
        for sample in range(logits.shape[0]):
            distance = _tree_distance(
                [int(item) for item in parents[sample].tolist()],
                int(predicted[sample].item()),
                int(targets[sample].item()),
            )
            if distance is not None:
                distances.append(distance)
        result["mean_tree_distance"] = (
            sum(distances) / len(distances) if distances else math.nan
        )
        result["tree_distance_estimable_fraction"] = len(distances) / logits.shape[0]
    return result


def build_learning_curve_plan(
    train_gene_ids: Sequence[str],
    *,
    fractions: Sequence[float],
    seeds: Sequence[int],
) -> tuple[LearningCurvePoint, ...]:
    """Create nested, outcome-blind whole-gene learning-curve subsets."""

    genes = sorted(set(str(item) for item in train_gene_ids))
    if len(genes) < 2:
        raise ValueError("at least two training genes are required")
    fraction_values = tuple(float(item) for item in fractions)
    if not fraction_values or tuple(sorted(set(fraction_values))) != fraction_values:
        raise ValueError("fractions must be unique and strictly increasing")
    if fraction_values[0] <= 0.0 or fraction_values[-1] > 1.0:
        raise ValueError("fractions must be in (0,1]")
    if not seeds or len(set(int(item) for item in seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty unique sequence")
    points = []
    for seed in seeds:
        ordered = list(genes)
        random.Random(int(seed)).shuffle(ordered)
        for fraction in fraction_values:
            count = max(1, math.ceil(fraction * len(ordered)))
            selected = tuple(sorted(ordered[:count]))
            digest = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
            points.append(
                LearningCurvePoint(
                    fraction=fraction,
                    seed=int(seed),
                    gene_ids=selected,
                    gene_set_sha256=digest,
                )
            )
    return tuple(points)


def select_validation_candidate(
    rows: Sequence[Mapping[str, Any]],
    contract: SelectionContract,
) -> dict[str, Any]:
    """Select a candidate using only its complete predeclared validation seeds."""

    if not rows:
        raise ValueError("candidate rows cannot be empty")
    grouped: dict[str, dict[int, float]] = {}
    for row in rows:
        if row.get("partition") != contract.tuning_partition:
            raise ValueError("candidate selection received a non-validation result")
        if row.get("metric") != contract.primary_metric:
            raise ValueError("candidate selection received an undeclared metric")
        seed = int(row["seed"])
        if seed not in contract.allowed_seeds:
            raise ValueError("candidate selection received an undeclared seed")
        candidate = str(row["candidate_id"])
        if seed in grouped.setdefault(candidate, {}):
            raise ValueError("candidate/seed validation result is duplicated")
        value = float(row["value"])
        if not math.isfinite(value):
            raise ValueError("candidate metric values must be finite")
        grouped[candidate][seed] = value
    required = set(contract.allowed_seeds)
    if any(set(values) != required for values in grouped.values()):
        raise ValueError("every candidate must have the complete predeclared seed set")
    summaries = [
        {
            "candidate_id": candidate,
            "validation_mean": sum(values.values()) / len(values),
            "seed_values": {str(seed): values[seed] for seed in sorted(values)},
        }
        for candidate, values in sorted(grouped.items())
    ]
    key = (
        (lambda item: (item["validation_mean"], item["candidate_id"]))
        if contract.minimize
        else (lambda item: (-item["validation_mean"], item["candidate_id"]))
    )
    selected = min(summaries, key=key)
    return {
        "selection_partition": contract.tuning_partition,
        "primary_metric": contract.primary_metric,
        "direction": "minimize" if contract.minimize else "maximize",
        "selected_candidate_id": selected["candidate_id"],
        "selected_validation_mean": selected["validation_mean"],
        "candidate_summaries": summaries,
        "sealed_test_used_for_selection": False,
    }
