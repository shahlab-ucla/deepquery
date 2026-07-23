from __future__ import annotations

import unittest
import copy
import csv
import tempfile
from pathlib import Path

from wormctx.graph import (
    build_contextual_graph,
    build_kgx_context_sidecar,
    build_kgx_projection,
    GraphView,
    observations_from_contextual_graph,
)
from wormctx.io import iter_jsonl, load_observations, write_table
from wormctx.models import ContextualObservation
from wormctx.pipeline import CONTEXTUAL_NODE_FIELDS


ROOT = Path(__file__).resolve().parents[1]


class GraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observations = load_observations(ROOT / "data" / "examples" / "observations.jsonl")

    def test_contextual_graph_reifies_observations_and_provenance(self) -> None:
        graph = build_contextual_graph(self.observations)
        self.assertIn("wormctx:obs-caendr-001", graph.nodes)
        predicates = {edge["predicate"] for edge in graph.edges.values()}
        self.assertIn("wormctx:has_life_stage", predicates)
        self.assertIn("wormctx:has_intervention", predicates)
        self.assertIn("wormctx:has_evidence_line", predicates)
        self.assertIn("prov:wasDerivedFrom", predicates)
        self.assertIn("prov:wasGeneratedBy", predicates)
        self.assertIn("prov:used", predicates)

    def test_kgx_projection_is_loss_annotated(self) -> None:
        graph = build_kgx_projection(
            self.observations,
            require_reviewed_infores=False,
            require_edge_metadata=False,
        )
        self.assertEqual(len(graph.edges), len(self.observations) - 1)
        self.assertTrue(
            all(
                evidence.id in graph.nodes
                for item in self.observations
                if item.observation_status.value != "uncertain"
                for evidence in item.evidence_lines
            )
        )
        for edge in graph.edges.values():
            self.assertNotIn("derived_from_observation", edge)
            self.assertNotIn("context_json", edge)
            self.assertIn("has_evidence", edge)
        negative = next(
            edge
            for edge in graph.edges.values() if edge["subject"] == "wormctx:demo_gene_b"
        )
        self.assertTrue(negative["negated"])
        sidecar = build_kgx_context_sidecar(
            self.observations, require_reviewed_infores=False
        )
        negative_context = next(
            row for row in sidecar if row["observation_id"] == "wormctx:obs-negative-001"
        )
        self.assertEqual(negative_context["observation_status"], "explicit_negative")
        self.assertTrue(negative_context["proposition_negated"])
        uncertain = next(
            row for row in sidecar if row["observation_id"] == "wormctx:obs-literature-001"
        )
        self.assertEqual(uncertain["projection_status"], "skipped_uncertain")
        self.assertIsNone(uncertain["association_id"])

    def test_standard_kgx_requires_reviewed_infores(self) -> None:
        with self.assertRaisesRegex(ValueError, "reviewed Information Resource"):
            build_kgx_projection([self.observations[0]])
        skipped = build_kgx_projection(
            [self.observations[0]], on_unreviewed_infores="skip"
        )
        self.assertEqual(len(skipped.edges), 0)
        reviewed = self.observations[0].model_copy(deep=True)
        reviewed.provenance.source_id = "alliance"
        with self.assertRaisesRegex(ValueError, "knowledge_level.*agent_type"):
            build_kgx_projection([reviewed])
        metadata_sidecar = build_kgx_context_sidecar([reviewed])
        self.assertEqual(
            metadata_sidecar[0]["projection_status"],
            "blocked_missing_edge_metadata",
        )
        reviewed_payload = reviewed.model_dump(mode="json")
        reviewed_payload["biolink_knowledge_level"] = "observation"
        reviewed_payload["biolink_agent_type"] = "manual_agent"
        reviewed = ContextualObservation.model_validate(reviewed_payload)
        graph = build_kgx_projection([reviewed])
        edge = next(iter(graph.edges.values()))
        self.assertEqual(edge["primary_knowledge_source"], "infores:agrkb")

    def test_contextual_graph_round_trip_is_lossless(self) -> None:
        row = copy.deepcopy(next(iter_jsonl(ROOT / "data" / "examples" / "observations.jsonl")))
        row["assertion_method"] = {
            "id": "ECO:0000006",
            "label": "experimental evidence",
        }
        row["context"]["context_gaps"] = [
            {"dimension": "anatomy", "reason": "not_reported"}
        ]
        row["result"] = {
            "id": "wormctx:result-roundtrip-001",
            "result_type": "quantitative",
            "summary": "Synthetic round-trip result.",
            "comparison_groups": [
                {
                    "id": "wormctx:group-case-roundtrip",
                    "label": "case",
                    "role": "case",
                    "size": 10,
                },
                {
                    "id": "wormctx:group-control-roundtrip",
                    "label": "control",
                    "role": "control",
                    "size": 10,
                },
            ],
            "measurements": [
                {
                    "id": "wormctx:measurement-roundtrip-001",
                    "measured_feature": {"id": "wormctx:effect-size"},
                    "value": {"value": 0.5},
                    "uncertainty": {"lower_bound": 0.3, "upper_bound": 0.7},
                    "uncertainty_type": {"id": "STATO:0000196", "label": "confidence interval"},
                    "confidence_level": 0.95,
                    "p_value": 0.01,
                    "sample_size": 20,
                    "biological_replicates": 4,
                    "comparison_group_ids": [
                        "wormctx:group-case-roundtrip",
                        "wormctx:group-control-roundtrip",
                    ],
                    "asset_ids": ["wormctx:asset-roundtrip-001"],
                }
            ],
            "asset_references": [
                {
                    "id": "wormctx:asset-roundtrip-001",
                    "uri": "urn:sha256:" + "b" * 64,
                    "checksum_sha256": "b" * 64,
                    "media_type": "application/zarr",
                    "role": "trajectory array",
                }
            ],
        }
        enriched = ContextualObservation.model_validate(row)
        observations = [enriched, *self.observations[1:]]
        graph = build_contextual_graph(observations)
        rebuilt = observations_from_contextual_graph(graph)
        expected = sorted(observations, key=lambda item: item.id)
        self.assertEqual(
            [item.model_dump(mode="json") for item in rebuilt],
            [item.model_dump(mode="json") for item in expected],
        )
        self.assertIn("wormctx:result-roundtrip-001", graph.nodes)
        self.assertIn("wormctx:measurement-roundtrip-001", graph.nodes)
        self.assertIn("wormctx:asset-roundtrip-001", graph.nodes)
        self.assertEqual(
            graph.nodes["wormctx:asset-roundtrip-001"]["checksum_sha256"],
            "b" * 64,
        )
        # The pipeline's explicit column allowlist must retain the canonical
        # payload and checksum through its TSV boundary as well.
        with tempfile.TemporaryDirectory() as directory:
            node_path = Path(directory) / "contextual_nodes.tsv"
            write_table(
                node_path,
                graph.sorted_nodes(),
                fieldnames=CONTEXTUAL_NODE_FIELDS,
            )
            serialized = GraphView()
            with node_path.open("r", encoding="utf-8", newline="") as handle:
                for node in csv.DictReader(handle, delimiter="\t"):
                    serialized.nodes[node["id"]] = node
            rebuilt_from_tsv = observations_from_contextual_graph(serialized)
        self.assertEqual(
            [item.model_dump(mode="json") for item in rebuilt_from_tsv],
            [item.model_dump(mode="json") for item in expected],
        )
        graph.nodes[enriched.id]["canonical_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            observations_from_contextual_graph(graph)

    def test_not_provided_metadata_does_not_satisfy_the_completeness_gate(self) -> None:
        """`not_provided` is truthy but must be treated as missing edge metadata."""
        reviewed_payload = self.observations[0].model_dump(mode="json")
        reviewed_payload["provenance"]["source_id"] = "alliance"
        reviewed_payload["biolink_knowledge_level"] = "not_provided"
        reviewed_payload["biolink_agent_type"] = "not_provided"
        reviewed = ContextualObservation.model_validate(reviewed_payload)
        # Reviewed source + mapped predicate + certain, so only the metadata gate
        # can block it. Standard (error) mode must refuse it, not emit an edge.
        with self.assertRaisesRegex(ValueError, "knowledge_level.*agent_type"):
            build_kgx_projection([reviewed])
        skipped = build_kgx_projection([reviewed], on_missing_edge_metadata="skip")
        self.assertEqual(len(skipped.edges), 0)
        sidecar = build_kgx_context_sidecar([reviewed])
        self.assertEqual(
            sidecar[0]["projection_status"], "blocked_missing_edge_metadata"
        )
        self.assertIsNone(sidecar[0]["association_id"])

    def test_unmapped_predicate_is_not_silently_projected(self) -> None:
        observation = self.observations[0].model_copy(deep=True)
        observation.predicate = "RO:0000000"
        with self.assertRaisesRegex(ValueError, "no reviewed Biolink predicate mapping"):
            build_kgx_projection(
                [observation],
                require_reviewed_infores=False,
                require_edge_metadata=False,
            )

    def test_graph_view_rejects_incompatible_identifier_collisions(self) -> None:
        graph = GraphView()
        node = {"id": "wormctx:node-1", "name": "one", "category": ["biolink:Gene"]}
        graph.add_node(node)
        graph.add_node(dict(node))
        with self.assertRaisesRegex(ValueError, "node ID collision"):
            graph.add_node(
                {
                    "id": "wormctx:node-1",
                    "name": "two",
                    "category": ["biolink:PhenotypicFeature"],
                }
            )

        edge = {
            "id": "wormctx:edge-1",
            "subject": "wormctx:a",
            "predicate": "biolink:related_to",
            "object": "wormctx:b",
        }
        graph.add_edge(edge)
        graph.add_edge(dict(edge))
        conflicting = dict(edge)
        conflicting["object"] = "wormctx:c"
        with self.assertRaisesRegex(ValueError, "edge ID collision"):
            graph.add_edge(conflicting)


if __name__ == "__main__":
    unittest.main()
