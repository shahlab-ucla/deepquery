from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="developmental topology requires the poc extra")

from wormctx.poc import developmental_topology as topology  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/developmental_genetics/lineage_topology_ablation/config/topology_ablation.json"


def _model_config() -> topology.TopologyModelConfig:
    return topology.TopologyModelConfig(
        numeric_feature_dim=3,
        semantic_vocab_size=8,
        depth_vocab_size=4,
        stage_vocab_size=4,
        hidden_dim=16,
        message_passing_layers=2,
        outcome_dim=1,
        dropout=0.0,
    )


def _batch() -> dict[str, torch.Tensor]:
    parent = torch.tensor([-1, 0, 0, 1, 1, 2, 2], dtype=torch.long)
    depth = torch.tensor([0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    features = torch.tensor(
        [
            [
                [0.0, 0.2, 1.0],
                [0.1, 0.3, 1.0],
                [0.4, 0.2, 0.0],
                [0.7, 0.1, 1.0],
                [0.3, 0.8, 1.0],
                [0.2, 0.9, 0.0],
                [0.5, 0.5, 1.0],
            ],
            [
                [0.2, 0.1, 1.0],
                [0.6, 0.3, 0.0],
                [0.4, 0.4, 1.0],
                [0.1, 0.9, 1.0],
                [0.8, 0.1, 0.0],
                [0.3, 0.7, 1.0],
                [0.9, 0.2, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    return {
        "numeric_features": features,
        "node_mask": torch.ones(2, 7, dtype=torch.bool),
        "parent_index": parent.unsqueeze(0).expand(2, -1).clone(),
        "semantic_id": torch.arange(7).unsqueeze(0).expand(2, -1).clone(),
        "depth_id": depth.unsqueeze(0).expand(2, -1).clone(),
        "stage_id": depth.unsqueeze(0).expand(2, -1).clone(),
    }


def _adapter_manifests() -> tuple[dict, dict]:
    cells = ["P0"]
    edges = []
    frontier = ["P0"]
    input_frontier = []
    born_by_input = []
    next_cell = 1
    for event in range(199):
        parent = frontier.pop(0)
        daughters = [f"cell-{next_cell}", f"cell-{next_cell + 1}"]
        next_cell += 2
        cells.extend(daughters)
        edges.extend({"parent": parent, "child": child} for child in daughters)
        frontier.extend(daughters)
        if event == 24:
            input_frontier = list(frontier)
            born_by_input = list(cells)
    lineage = {
        "schema_version": "wormctx-omix709-semantic-lineage-1.0",
        "virtual_root": "P0",
        "cells_through_200": cells,
        "edges_through_200": edges,
    }
    stages = {
        "schema_version": "wormctx-omix709-semantic-stage-membership-1.0",
        "perturbation_measurement_values_read": False,
        "s4_outcome_values_read": False,
        "input_26": {"frontier": input_frontier, "born_cells": born_by_input},
        "endpoint_200": {"frontier": frontier, "born_cells": cells},
    }
    return lineage, stages


def test_predeclared_config_is_outcome_blind_and_validation_only() -> None:
    config = topology.load_benchmark_config(CONFIG)

    assert config["status"] == "coordinates_prepared_blocked_pending_overlap_and_outcome_tensor"
    assert config["real_preparation_binding"]["outcome_values_read"] is False
    assert config["real_preparation_binding"]["pilot_overlap_closed"] is False
    assert config["real_preparation_binding"]["real_training_permitted"] is False
    assert config["real_preparation_binding"]["lineage_cells"] == 399
    assert config["real_preparation_binding"]["lineage_edges"] == 398
    assert config["sealed_outcomes_inspected"] is False
    assert config["adapter_input_contract"]["graph_construction_may_read_outcomes"] is False
    assert config["selection"]["tuning_partition"] == "validation"
    assert config["selection"]["sealed_test_openings"] == 1
    assert config["model_ladder"] == [item.value for item in topology.TopologyVariant]


def test_stage_adapter_bridge_builds_aligned_outcome_blind_topology_arrays() -> None:
    lineage, stages = _adapter_manifests()
    arrays = topology.topology_arrays_from_stage_adapter(lineage, stages)

    assert len(arrays["cell_ids"]) == 399
    assert arrays["parent_index"].shape == (399,)
    assert arrays["depth_id"].shape == (399,)
    assert arrays["endpoint_frontier_mask"].sum().item() == 200
    assert arrays["born_by_input_mask"].sum().item() == 51
    assert arrays["receipt"]["outcome_values_read"] is False
    topology.validate_parent_forest(arrays["parent_index"], arrays["depth_id"])

    contaminated = dict(stages, s4_outcome_values_read=True)
    with pytest.raises(ValueError, match="outcome-blind"):
        topology.topology_arrays_from_stage_adapter(lineage, contaminated)


def test_parent_child_message_path_and_typed_self_control_are_explicit() -> None:
    batch = _batch()
    parent = batch["parent_index"][:1]
    mask = batch["node_mask"][:1]
    node_count = parent.shape[1]
    source, target, edge_type, edge_mask = topology.build_message_edges(
        parent, mask, use_lineage_edges=True
    )

    # Node 1 has parent 0: one parent->child and one child->parent message.
    assert (source[0, 1].item(), target[0, 1].item()) == (0, 1)
    assert edge_type[0, 1].item() == topology.PARENT_TO_CHILD
    assert (source[0, node_count + 1].item(), target[0, node_count + 1].item()) == (1, 0)
    assert edge_type[0, node_count + 1].item() == topology.CHILD_TO_PARENT
    assert edge_mask[0, 1] and edge_mask[0, node_count + 1]

    free_source, free_target, free_type, free_mask = topology.build_message_edges(
        parent, mask, use_lineage_edges=False
    )
    assert torch.equal(free_source, free_target)
    assert set(free_type.flatten().tolist()) == {
        topology.PARENT_TO_CHILD,
        topology.CHILD_TO_PARENT,
        topology.SELF,
    }
    assert free_mask.all()


def test_all_variants_are_capacity_matched_and_execute() -> None:
    config = _model_config()
    batch = _batch()
    models = topology.make_capacity_matched_models(
        config,
        list(topology.TopologyVariant),
        initialization_seed=1729,
    )

    counts = {model.parameter_count() for model in models.values()}
    assert len(counts) == 1
    outputs = {}
    for variant, model in models.items():
        model.eval()
        outputs[variant] = model(batch)
        assert outputs[variant]["mean"].shape == (2, 1)
        assert outputs[variant]["log_variance"].shape == (2, 1)
        assert outputs[variant]["localization_logits"].shape == (2, 7, 1)
        assert outputs[variant]["pooling_weights"].shape == (2, 7)
        assert torch.isfinite(outputs[variant]["mean"]).all()

    assert not torch.allclose(
        outputs[topology.TopologyVariant.REAL_LINEAGE]["mean"],
        outputs[topology.TopologyVariant.EDGE_FREE_SELF_LOOP]["mean"],
    )


def test_multitask_loss_backpropagates_through_outcome_and_localization() -> None:
    model = topology.DevelopmentalTopologyRegressor(
        _model_config(), topology.TopologyVariant.REAL_LINEAGE
    )
    predictions = model(_batch())
    loss, components = topology.developmental_multitask_loss(
        predictions,
        torch.tensor([[0.2], [-0.1]]),
        localization_target=torch.tensor([[3], [5]]),
    )

    assert set(components) == {"outcome_gaussian_nll", "localization_cross_entropy"}
    assert torch.isfinite(loss)
    loss.backward()
    assert model.numeric_projection.weight.grad is not None
    assert torch.isfinite(model.numeric_projection.weight.grad).all()


def test_rewired_tree_null_is_deterministic_and_preserves_degree_and_depth() -> None:
    parent = torch.tensor([-1, 0, 0, 1, 1, 2, 2], dtype=torch.long)
    depth = torch.tensor([0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    first = topology.rewire_parent_index(parent, depth, seed=41)
    second = topology.rewire_parent_index(parent, depth, seed=41)

    assert torch.equal(first.parent_index, second.parent_index)
    assert first.receipt == second.receipt
    assert first.receipt["degree_preserved"] is True
    assert first.receipt["depth_preserved"] is True
    assert first.receipt["changed_parent_assignments"] > 0
    assert not torch.equal(first.parent_index, parent)
    topology.validate_parent_forest(first.parent_index, depth)
    original_degree = torch.bincount(parent[parent >= 0], minlength=7)
    rewired_degree = torch.bincount(first.parent_index[first.parent_index >= 0], minlength=7)
    assert torch.equal(original_degree, rewired_degree)
    for child, new_parent in enumerate(first.parent_index.tolist()):
        if new_parent >= 0:
            assert depth[new_parent].item() + 1 == depth[child].item()

    series = topology.generate_rewired_tree_nulls(parent, depth, count=3, base_seed=100)
    assert [item.receipt["seed"] for item in series] == [100, 104829, 209558]


def test_label_permutation_hooks_are_seeded_and_group_safe() -> None:
    groups = ("g1", "g1", "g2", "g3", "g3")
    first = topology.permute_group_labels(groups, seed=7)
    second = topology.permute_group_labels(groups, seed=7)

    assert first == second
    assert first != groups
    assert first[0] == first[1]
    assert first[3] == first[4]
    assert set(first) == set(groups)

    target = torch.tensor([10.0, 20.0, 30.0, 40.0])
    permuted, order = topology.permute_sample_targets(target, seed=13)
    repeated, repeated_order = topology.permute_sample_targets(target, seed=13)
    assert torch.equal(order, repeated_order)
    assert torch.equal(permuted, repeated)
    assert sorted(permuted.tolist()) == sorted(target.tolist())


def test_validation_only_selection_rejects_sealed_or_incomplete_results() -> None:
    contract = topology.SelectionContract(
        split_unit="whole_gene",
        tuning_partition="validation",
        sealed_partition="sealed_test",
        primary_metric="gene_macro_rmse",
        minimize=True,
        allowed_seeds=(1, 2),
    )
    rows = [
        {
            "candidate_id": "model-a",
            "partition": "validation",
            "metric": "gene_macro_rmse",
            "seed": 1,
            "value": 1.1,
        },
        {
            "candidate_id": "model-a",
            "partition": "validation",
            "metric": "gene_macro_rmse",
            "seed": 2,
            "value": 1.3,
        },
        {
            "candidate_id": "model-b",
            "partition": "validation",
            "metric": "gene_macro_rmse",
            "seed": 1,
            "value": 0.9,
        },
        {
            "candidate_id": "model-b",
            "partition": "validation",
            "metric": "gene_macro_rmse",
            "seed": 2,
            "value": 1.1,
        },
    ]
    selected = topology.select_validation_candidate(rows, contract)
    assert selected["selected_candidate_id"] == "model-b"
    assert selected["sealed_test_used_for_selection"] is False

    sealed = [dict(rows[0], partition="sealed_test"), *rows[1:]]
    with pytest.raises(ValueError, match="non-validation"):
        topology.select_validation_candidate(sealed, contract)
    with pytest.raises(ValueError, match="complete"):
        topology.select_validation_candidate(rows[:-1], contract)


def test_prediction_calibration_localization_and_learning_curve_interfaces() -> None:
    prediction = torch.tensor([0.0, 2.0, 1.0, 3.0])
    target = torch.tensor([0.0, 1.0, 1.0, 2.0])
    metrics = topology.prediction_metrics(
        prediction, target, gene_ids=("a", "a", "b", "b")
    )
    assert metrics["rmse"] == pytest.approx(math.sqrt(0.5))
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["gene_macro_rmse"] == pytest.approx(math.sqrt(0.5))

    calibration = topology.regression_calibration_metrics(
        target, torch.zeros_like(target), target
    )
    assert calibration["gaussian_nll"] == pytest.approx(0.5 * math.log(2 * math.pi))
    assert calibration["coverage_50"] == 1.0
    assert calibration["coverage_95"] == 1.0

    logits = torch.tensor([[0.0, 3.0, 2.0], [3.0, 2.0, 1.0]])
    parents = torch.tensor([-1, 0, 0])
    localization = topology.localization_metrics(
        logits,
        torch.tensor([1, 1]),
        parent_index=parents,
    )
    assert localization["top1_accuracy"] == 0.5
    assert localization["top3_accuracy"] == 1.0
    assert localization["mean_reciprocal_rank"] == 0.75
    assert localization["mean_tree_distance"] == 0.5

    plan = topology.build_learning_curve_plan(
        [f"gene-{index}" for index in range(8)],
        fractions=(0.25, 0.5, 0.75, 1.0),
        seeds=(11, 12),
    )
    assert len(plan) == 8
    for seed in (11, 12):
        points = [item for item in plan if item.seed == seed]
        assert [len(item.gene_ids) for item in points] == [2, 4, 6, 8]
        for smaller, larger in zip(points, points[1:]):
            assert set(smaller.gene_ids) < set(larger.gene_ids)


def test_batch_contract_fails_closed_on_depth_inconsistent_parent() -> None:
    batch = _batch()
    batch["parent_index"][0, 3] = 0
    with pytest.raises(ValueError, match="advance exactly one depth"):
        topology.validate_topology_batch(batch, _model_config())
