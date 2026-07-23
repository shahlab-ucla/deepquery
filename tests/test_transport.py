from __future__ import annotations

import copy
import unittest
from pathlib import Path

from wormctx.coverage import UNSPECIFIED, coverage_cube, coverage_long
from wormctx.io import CollectionIntegrityError, load_observations, load_query, read_json
from wormctx.models import ContextGap, NamedReference, QuantitativeValue
from wormctx.signatures import (
    canonical_intervention_set_signature,
    context_values,
)
from wormctx.transport import score_observation, validate_policy


ROOT = Path(__file__).resolve().parents[1]


class TransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observations = load_observations(ROOT / "data" / "examples" / "observations.jsonl")
        self.query = load_query(ROOT / "data" / "examples" / "query_context.json")
        self.policy = read_json(ROOT / "config" / "transport_policy.json")

    def test_direct_evidence_outranks_transported_developmental_prior(self) -> None:
        scores = {
            item.id: score_observation(item, self.query.context, self.policy)
            for item in self.observations
        }
        direct = scores["wormctx:obs-caendr-001"]
        developmental = scores["wormctx:obs-development-001"]
        self.assertEqual(direct["classification"], "direct")
        self.assertEqual(direct["score"], 1.0)
        self.assertGreater(direct["score"], developmental["score"])
        self.assertNotEqual(developmental["classification"], "direct")

    def test_taxon_mismatch_is_hard_failure(self) -> None:
        context = self.query.context.model_copy(deep=True)
        context.organism_taxon = NamedReference(id="NCBITaxon:9606", label="Homo sapiens")
        result = score_observation(self.observations[0], context, self.policy)
        self.assertEqual(result["classification"], "nontransportable")
        self.assertIn("organism_taxon", result["hard_failures"])

    def test_compound_identity_changes_transport_score(self) -> None:
        observation = self.observations[0].model_copy(deep=True)
        matching = score_observation(observation, self.query.context, self.policy)
        observation.context.interventions[0].reagent = NamedReference(
            id="wormctx:demo_compound_b", label="demo compound B"
        )
        mismatching = score_observation(observation, self.query.context, self.policy)
        self.assertGreater(matching["score"], mismatching["score"])
        intervention = next(
            item
            for item in mismatching["components"]
            if item["dimension"] == "intervention"
        )
        self.assertEqual(intervention["state"], "compatible")

    def test_intervention_attributes_cannot_cross_match_between_records(self) -> None:
        observation = self.observations[1].model_copy(deep=True)
        query = self.query.context.model_copy(deep=True)
        query.interventions[0].target = NamedReference(
            id="wormctx:demo_gene_a", label="demo gene A"
        )
        result = score_observation(observation, query, self.policy)
        component = next(
            item for item in result["components"] if item["dimension"] == "intervention"
        )
        self.assertNotEqual(component["state"], "exact")
        self.assertNotEqual(result["classification"], "direct")

    def test_intervention_hard_gate_rejects_requested_attribute_conflict(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["dimensions"]["intervention"]["hard_gate"] = True
        observation = self.observations[0].model_copy(deep=True)
        observation.context.interventions[0].reagent = NamedReference(
            id="wormctx:demo_compound_b", label="demo compound B"
        )
        result = score_observation(observation, self.query.context, policy)
        self.assertEqual(result["classification"], "nontransportable")
        self.assertIn("intervention", result["hard_failures"])

    def test_absolute_time_is_a_transport_dimension(self) -> None:
        observation = self.observations[0].model_copy(deep=True)
        query = self.query.context.model_copy(deep=True)
        observation.context.absolute_time = QuantitativeValue(
            value=60.0, unit=NamedReference(id="UO:0000031", label="minute")
        )
        query.absolute_time = observation.context.absolute_time.model_copy(deep=True)
        matching = score_observation(observation, query, self.policy)
        query.absolute_time.value = 120.0
        mismatching = score_observation(observation, query, self.policy)
        self.assertGreater(matching["score"], mismatching["score"])
        component = next(
            item
            for item in mismatching["components"]
            if item["dimension"] == "absolute_time"
        )
        self.assertEqual(component["state"], "mismatch")
        coverage_row = next(
            row
            for row in coverage_long([observation])
            if row["dimension"] == "absolute_time"
        )
        self.assertEqual(coverage_row["reportedness"], "reported")
        self.assertEqual(
            coverage_row["value_id"],
            context_values(observation.context)["absolute_time"][0],
        )

    def test_dimension_specific_not_applicable_is_preserved(self) -> None:
        observation = self.observations[0].model_copy(deep=True)
        observation.context.context_gaps = [
            ContextGap(dimension="anatomy", reason="not_applicable")
        ]
        rows = coverage_long([observation])
        anatomy = next(row for row in rows if row["dimension"] == "anatomy")
        self.assertEqual(anatomy["reportedness"], "not_applicable")

    def test_coverage_preserves_unspecified_context(self) -> None:
        rows = coverage_long(self.observations)
        literature_rows = [
            row for row in rows if row["observation_id"] == "wormctx:obs-literature-001"
        ]
        self.assertTrue(any(row["value_id"] == UNSPECIFIED for row in literature_rows))
        self.assertTrue(any(row["reportedness"] == "not_reported" for row in literature_rows))
        cube = coverage_cube(self.observations)
        self.assertGreaterEqual(len(cube), len(self.observations))

    def test_coverage_uses_canonical_context_and_snapshot_identity(self) -> None:
        observation = self.observations[0].model_copy(deep=True)
        observation.context.interventions[0].duration = QuantitativeValue(
            value=2.0, unit=NamedReference(id="UO:0000031", label="minute")
        )
        intervention_signature = canonical_intervention_set_signature(
            observation.context.interventions
        )
        self.assertIn("dose", intervention_signature)
        self.assertIn("duration", intervention_signature)
        self.assertIn(
            context_values(observation.context)["intervention"][0],
            intervention_signature,
        )
        second_snapshot = observation.model_copy(deep=True)
        second_snapshot.id = "wormctx:obs-caendr-other-snapshot"
        second_snapshot.evidence_lines[0].id = (
            "wormctx:evidence-caendr-other-snapshot"
        )
        second_snapshot.provenance.source_release = "demo-2"
        second_snapshot.provenance.source_artifact = "synthetic-qtl-v2.tsv"
        cube = coverage_cube([observation, second_snapshot])
        self.assertEqual(len(cube), 2)
        cube_row = next(
            row
            for row in cube
            if row["source_release"] == observation.provenance.source_release
        )
        self.assertEqual(cube_row["intervention_signature"], intervention_signature)
        self.assertEqual(
            cube_row["source_release"], observation.provenance.source_release
        )
        self.assertEqual(
            cube_row["source_artifact"], observation.provenance.source_artifact
        )
        self.assertEqual(
            cube_row["source_checksum_sha256"],
            observation.provenance.checksum_sha256,
        )
        self.assertEqual(cube_row["absolute_time"], UNSPECIFIED)

        long_row = next(
            row
            for row in coverage_long([observation])
            if row["dimension"] == "organism_taxon"
        )
        self.assertEqual(
            long_row["source_release"], observation.provenance.source_release
        )
        self.assertEqual(
            long_row["source_artifact"], observation.provenance.source_artifact
        )
        self.assertEqual(
            long_row["source_checksum_sha256"],
            observation.provenance.checksum_sha256,
        )

    def test_coverage_rejects_inconsistent_snapshot_provenance(self) -> None:
        first = self.observations[0].model_copy(deep=True)
        second = first.model_copy(deep=True)
        second.id = "wormctx:obs-inconsistent-coverage-snapshot"
        second.evidence_lines[0].id = "wormctx:evidence-inconsistent-coverage-snapshot"
        second.provenance.checksum_sha256 = "d" * 64
        for builder in (coverage_long, coverage_cube):
            with self.subTest(builder=builder.__name__), self.assertRaisesRegex(
                CollectionIntegrityError, "identical snapshot provenance"
            ):
                builder([first, second])

    def test_invalid_policy_is_rejected(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["thresholds"]["direct"] = 0.2
        with self.assertRaisesRegex(ValueError, "thresholds"):
            validate_policy(policy)

    def test_all_requested_context_values_must_be_covered(self) -> None:
        query = self.query.context.model_copy(deep=True)
        query.life_stage.append(NamedReference(id="wormctx:embryo"))
        partial = score_observation(self.observations[0], query, self.policy)
        stage = next(
            item for item in partial["components"] if item["dimension"] == "life_stage"
        )
        self.assertNotEqual(partial["classification"], "direct")
        self.assertEqual(stage["state"], "mismatch")
        self.assertEqual(stage["unmatched_query_values"], ["wormctx:embryo"])

        policy = copy.deepcopy(self.policy)
        policy["dimensions"]["life_stage"]["hard_gate"] = True
        gated = score_observation(self.observations[0], query, policy)
        self.assertEqual(gated["classification"], "nontransportable")
        self.assertIn("life_stage", gated["hard_failures"])

        covering = self.observations[0].model_copy(deep=True)
        covering.context.life_stage.append(NamedReference(id="wormctx:embryo"))
        covered = score_observation(covering, query, self.policy)
        covered_stage = next(
            item for item in covered["components"] if item["dimension"] == "life_stage"
        )
        self.assertEqual(covered_stage["state"], "exact")
        self.assertEqual(covered["classification"], "direct")

    def test_nonfinite_policy_numbers_are_rejected(self) -> None:
        mutations = (
            ("threshold", lambda policy, value: policy["thresholds"].__setitem__("direct", value)),
            (
                "weight",
                lambda policy, value: policy["dimensions"]["life_stage"].__setitem__(
                    "weight", value
                ),
            ),
            (
                "unknown score",
                lambda policy, value: policy["dimensions"]["life_stage"].__setitem__(
                    "unknown_score", value
                ),
            ),
            (
                "mismatch score",
                lambda policy, value: policy["dimensions"]["life_stage"].__setitem__(
                    "mismatch_score", value
                ),
            ),
            (
                "compatibility",
                lambda policy, value: policy["compatibility"]["life_stage"].__setitem__(
                    "wormctx:adult|wormctx:young_adult", value
                ),
            ),
        )
        for label, mutate in mutations:
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(label=label, value=value):
                    policy = copy.deepcopy(self.policy)
                    mutate(policy, value)
                    with self.assertRaisesRegex(ValueError, "finite"):
                        validate_policy(policy)

    def test_canonical_signatures_refuse_nonfinite_constructed_values(self) -> None:
        unsafe = QuantitativeValue.model_construct(value=float("nan"))
        from wormctx.signatures import canonical_quantity_signature

        with self.assertRaises(ValueError):
            canonical_quantity_signature(unsafe)


if __name__ == "__main__":
    unittest.main()
