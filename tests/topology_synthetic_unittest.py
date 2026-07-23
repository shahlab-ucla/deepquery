"""Standard-library runner for environments that have torch but not pytest.

The canonical pytest coverage remains in ``test_poc_developmental_topology``.
This focused runner exercises the same scientific invariants without changing
the immutable remote environment or installing packages into it.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

from wormctx.poc import developmental_topology as topology


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/developmental_genetics/lineage_topology_ablation/config/topology_ablation.json"


def _config() -> topology.TopologyModelConfig:
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
    generator = torch.Generator().manual_seed(1729)
    return {
        "numeric_features": torch.rand(2, 7, 3, generator=generator),
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


class TopologySyntheticTests(unittest.TestCase):
    def test_stage_adapter_bridge_is_outcome_blind_and_aligned(self) -> None:
        lineage, stages = _adapter_manifests()
        arrays = topology.topology_arrays_from_stage_adapter(lineage, stages)
        self.assertEqual(len(arrays["cell_ids"]), 399)
        self.assertEqual(arrays["endpoint_frontier_mask"].sum().item(), 200)
        self.assertEqual(arrays["born_by_input_mask"].sum().item(), 51)
        self.assertFalse(arrays["receipt"]["outcome_values_read"])
        topology.validate_parent_forest(arrays["parent_index"], arrays["depth_id"])
        contaminated = dict(stages, s4_outcome_values_read=True)
        with self.assertRaisesRegex(ValueError, "outcome-blind"):
            topology.topology_arrays_from_stage_adapter(lineage, contaminated)

    def test_config_and_validation_only_selection(self) -> None:
        payload = topology.load_benchmark_config(CONFIG)
        self.assertFalse(payload["sealed_outcomes_inspected"])
        self.assertEqual(payload["selection"]["tuning_partition"], "validation")
        self.assertEqual(payload["model_ladder"], [item.value for item in topology.TopologyVariant])

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
                "candidate_id": candidate,
                "partition": "validation",
                "metric": "gene_macro_rmse",
                "seed": seed,
                "value": value,
            }
            for candidate, seed, value in (
                ("a", 1, 1.1),
                ("a", 2, 1.3),
                ("b", 1, 0.9),
                ("b", 2, 1.1),
            )
        ]
        selected = topology.select_validation_candidate(rows, contract)
        self.assertEqual(selected["selected_candidate_id"], "b")
        self.assertFalse(selected["sealed_test_used_for_selection"])
        rows[0] = dict(rows[0], partition="sealed_test")
        with self.assertRaisesRegex(ValueError, "non-validation"):
            topology.select_validation_candidate(rows, contract)

    def test_real_edges_and_capacity_matched_controls_execute(self) -> None:
        batch = _batch()
        source, target, edge_type, edge_mask = topology.build_message_edges(
            batch["parent_index"][:1], batch["node_mask"][:1], use_lineage_edges=True
        )
        node_count = batch["parent_index"].shape[1]
        self.assertEqual((source[0, 1].item(), target[0, 1].item()), (0, 1))
        self.assertEqual(edge_type[0, 1].item(), topology.PARENT_TO_CHILD)
        self.assertEqual(
            (source[0, node_count + 1].item(), target[0, node_count + 1].item()),
            (1, 0),
        )
        self.assertTrue(edge_mask[0, 1])

        free_source, free_target, _, _ = topology.build_message_edges(
            batch["parent_index"][:1],
            batch["node_mask"][:1],
            use_lineage_edges=False,
        )
        self.assertTrue(torch.equal(free_source, free_target))

        models = topology.make_capacity_matched_models(
            _config(), list(topology.TopologyVariant), initialization_seed=1729
        )
        self.assertEqual(len({model.parameter_count() for model in models.values()}), 1)
        outputs = {}
        for variant, model in models.items():
            model.eval()
            outputs[variant] = model(batch)
            self.assertEqual(tuple(outputs[variant]["mean"].shape), (2, 1))
            self.assertEqual(tuple(outputs[variant]["localization_logits"].shape), (2, 7, 1))
            self.assertTrue(torch.isfinite(outputs[variant]["mean"]).all())
        self.assertFalse(
            torch.allclose(
                outputs[topology.TopologyVariant.REAL_LINEAGE]["mean"],
                outputs[topology.TopologyVariant.EDGE_FREE_SELF_LOOP]["mean"],
            )
        )

    def test_outcome_and_localization_loss_backpropagates(self) -> None:
        model = topology.DevelopmentalTopologyRegressor(
            _config(), topology.TopologyVariant.REAL_LINEAGE
        )
        loss, components = topology.developmental_multitask_loss(
            model(_batch()),
            torch.tensor([[0.2], [-0.1]]),
            localization_target=torch.tensor([[3], [5]]),
        )
        self.assertEqual(
            set(components), {"outcome_gaussian_nll", "localization_cross_entropy"}
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.numeric_projection.weight.grad)

    def test_rewiring_preserves_degree_and_depth(self) -> None:
        parent = torch.tensor([-1, 0, 0, 1, 1, 2, 2], dtype=torch.long)
        depth = torch.tensor([0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
        first = topology.rewire_parent_index(parent, depth, seed=41)
        second = topology.rewire_parent_index(parent, depth, seed=41)
        self.assertTrue(torch.equal(first.parent_index, second.parent_index))
        self.assertGreater(first.receipt["changed_parent_assignments"], 0)
        self.assertTrue(first.receipt["degree_preserved"])
        self.assertTrue(first.receipt["depth_preserved"])
        topology.validate_parent_forest(first.parent_index, depth)
        original_degree = torch.bincount(parent[parent >= 0], minlength=7)
        rewired_degree = torch.bincount(first.parent_index[first.parent_index >= 0], minlength=7)
        self.assertTrue(torch.equal(original_degree, rewired_degree))
        for child, new_parent in enumerate(first.parent_index.tolist()):
            if new_parent >= 0:
                self.assertEqual(depth[new_parent].item() + 1, depth[child].item())

    def test_permutation_hooks_preserve_units(self) -> None:
        groups = ("g1", "g1", "g2", "g3", "g3")
        permuted = topology.permute_group_labels(groups, seed=7)
        self.assertEqual(permuted, topology.permute_group_labels(groups, seed=7))
        self.assertNotEqual(permuted, groups)
        self.assertEqual(permuted[0], permuted[1])
        self.assertEqual(permuted[3], permuted[4])
        self.assertEqual(set(permuted), set(groups))

        target = torch.tensor([10.0, 20.0, 30.0, 40.0])
        shuffled, order = topology.permute_sample_targets(target, seed=13)
        repeated, repeated_order = topology.permute_sample_targets(target, seed=13)
        self.assertTrue(torch.equal(shuffled, repeated))
        self.assertTrue(torch.equal(order, repeated_order))

    def test_metric_and_learning_curve_interfaces(self) -> None:
        prediction = torch.tensor([0.0, 2.0, 1.0, 3.0])
        target = torch.tensor([0.0, 1.0, 1.0, 2.0])
        metrics = topology.prediction_metrics(
            prediction, target, gene_ids=("a", "a", "b", "b")
        )
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(0.5))
        self.assertAlmostEqual(metrics["gene_macro_rmse"], math.sqrt(0.5))
        calibration = topology.regression_calibration_metrics(
            target, torch.zeros_like(target), target
        )
        self.assertAlmostEqual(calibration["gaussian_nll"], 0.5 * math.log(2 * math.pi))
        self.assertEqual(calibration["coverage_95"], 1.0)

        localization = topology.localization_metrics(
            torch.tensor([[0.0, 3.0, 2.0], [3.0, 2.0, 1.0]]),
            torch.tensor([1, 1]),
            parent_index=torch.tensor([-1, 0, 0]),
        )
        self.assertEqual(localization["top1_accuracy"], 0.5)
        self.assertEqual(localization["mean_reciprocal_rank"], 0.75)
        self.assertEqual(localization["mean_tree_distance"], 0.5)

        plan = topology.build_learning_curve_plan(
            [f"gene-{index}" for index in range(8)],
            fractions=(0.25, 0.5, 0.75, 1.0),
            seeds=(11, 12),
        )
        self.assertEqual(len(plan), 8)
        for seed in (11, 12):
            points = [item for item in plan if item.seed == seed]
            self.assertEqual([len(item.gene_ids) for item in points], [2, 4, 6, 8])
            for smaller, larger in zip(points, points[1:]):
                self.assertLess(set(smaller.gene_ids), set(larger.gene_ids))

    def test_batch_validation_fails_closed(self) -> None:
        batch = _batch()
        batch["parent_index"][0, 3] = 0
        with self.assertRaisesRegex(ValueError, "advance exactly one depth"):
            topology.validate_topology_batch(batch, _config())


if __name__ == "__main__":
    unittest.main(verbosity=2)
