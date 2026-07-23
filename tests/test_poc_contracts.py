from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from wormctx.poc.contracts import (
    BiologicalEpisode,
    CONCLUSION_ORDER,
    EDGE_TYPE_ORDER,
    FAMILY_ORDER,
    INVALID_FLAG_ORDER,
    MAX_HYPOTHESES,
    NODE_TYPE_ORDER,
    OPERATOR_ORDER,
    EpisodeFamily,
    EpisodeGraph,
    GraphNode,
    InvalidDesignFlag,
    NodeKind,
)
from wormctx.poc.simulation import (
    assign_group_splits,
    generate_episode,
    generate_red_team_episodes,
    generate_synthetic_dataset,
)


class PocContractTests(unittest.TestCase):
    def test_each_red_team_request_is_visible_in_numeric_model_features(self) -> None:
        node_label_by_flag = {
            InvalidDesignFlag.dominance_from_homozygous_panel: "natural isolate panel",
            InvalidDesignFlag.absent_locus_as_snp: "assembly-specific candidate locus",
            InvalidDesignFlag.database_absence_as_powered_negative: (
                "contextual database evidence"
            ),
            InvalidDesignFlag.transport_without_identification: (
                "natural-allele target context"
            ),
            InvalidDesignFlag.post_treatment_covariate: "growth-rate covariate",
        }
        for episode in generate_red_team_episodes(seed=41):
            flag = episode.labels.invalid_design_flags[0]
            node = next(
                item
                for item in episode.graph.nodes
                if item.label == node_label_by_flag[flag]
            )
            self.assertEqual(
                node.numeric_features[-1],
                1.0,
                msg=f"{flag.value} must be visible at the tensor boundary",
            )

    def test_generator_is_deterministic_and_json_serializable(self) -> None:
        first = generate_synthetic_dataset(
            groups_per_family=3,
            episodes_per_group=2,
            seed=1729,
        )
        second = generate_synthetic_dataset(
            groups_per_family=3,
            episodes_per_group=2,
            seed=1729,
        )

        first_json = [episode.model_dump_json() for episode in first]
        self.assertEqual(
            first_json,
            [episode.model_dump_json() for episode in second],
        )
        self.assertNotEqual(
            first_json,
            [
                episode.model_dump_json()
                for episode in generate_synthetic_dataset(
                    groups_per_family=3,
                    episodes_per_group=2,
                    seed=1730,
                )
            ],
        )
        self.assertTrue(
            all(
                json.loads(payload)["provenance"]["origin"] == "simulation"
                for payload in first_json
            )
        )
        self.assertTrue(
            all(
                json.loads(payload)["provenance"]["synthetic"] is True
                for payload in first_json
            )
        )
        self.assertEqual(
            BiologicalEpisode.model_validate_json(first_json[0]),
            first[0],
        )

    def test_dataset_covers_families_and_has_group_disjoint_splits(self) -> None:
        episodes = generate_synthetic_dataset(
            groups_per_family=4,
            episodes_per_group=3,
            seed=11,
        )
        self.assertEqual({episode.family for episode in episodes}, set(EpisodeFamily))
        self.assertEqual(
            {episode.split.value for episode in episodes},
            {"train", "val", "test"},
        )

        groups_by_split = {
            split: {
                episode.split_group
                for episode in episodes
                if episode.split.value == split
            }
            for split in ("train", "val", "test")
        }
        self.assertTrue(groups_by_split["train"].isdisjoint(groups_by_split["val"]))
        self.assertTrue(groups_by_split["train"].isdisjoint(groups_by_split["test"]))
        self.assertTrue(groups_by_split["val"].isdisjoint(groups_by_split["test"]))
        self.assertTrue(
            all(
                len(
                    {
                        episode.split
                        for episode in episodes
                        if episode.split_group == group
                    }
                )
                == 1
                for group in {episode.split_group for episode in episodes}
            )
        )

        groups = [f"sim:test:group-{index}" for index in range(20)]
        forward = assign_group_splits(groups, seed=9)
        self.assertEqual(forward, assign_group_splits(reversed(groups), seed=9))

    def test_red_team_failure_modes_are_explicit_and_materialized(self) -> None:
        episodes = generate_red_team_episodes(seed=23)
        flags = {
            flag
            for episode in episodes
            for flag in episode.labels.invalid_design_flags
        }
        self.assertEqual(len(episodes), len(InvalidDesignFlag))
        self.assertEqual(flags, set(InvalidDesignFlag))
        self.assertTrue(all(not episode.labels.design_valid for episode in episodes))
        self.assertTrue(
            all(
                episode.labels.resolution_ceiling.value == "non_identifiable"
                for episode in episodes
            )
        )

        by_flag = {
            episode.labels.invalid_design_flags[0]: episode
            for episode in episodes
        }
        dominance_panel = next(
            node
            for node in by_flag[
                InvalidDesignFlag.dominance_from_homozygous_panel
            ].graph.nodes
            if node.label == "natural isolate panel"
        )
        self.assertEqual(
            dominance_panel.categorical_features["mating_design"],
            "homozygous_natural_isolates",
        )

        absent_locus = next(
            node
            for node in by_flag[InvalidDesignFlag.absent_locus_as_snp].graph.nodes
            if node.kind is NodeKind.locus_instance
        )
        self.assertEqual(
            absent_locus.categorical_features,
            {
                "presence_state": "absent_in_carriers",
                "invalid_encoding": "reference_coordinate_snp",
            },
        )

        database_evidence = next(
            node
            for node in by_flag[
                InvalidDesignFlag.database_absence_as_powered_negative
            ].graph.nodes
            if node.kind is NodeKind.evidence
        )
        self.assertEqual(
            database_evidence.categorical_features["observation_status"],
            "not_recorded",
        )
        self.assertEqual(
            database_evidence.categorical_features["invalid_interpretation"],
            "powered_negative",
        )

        target_context = next(
            node
            for node in by_flag[
                InvalidDesignFlag.transport_without_identification
            ].graph.nodes
            if node.label == "natural-allele target context"
        )
        self.assertEqual(
            target_context.categorical_features["transport_status"],
            "claimed_without_identification",
        )

        post_treatment = next(
            node
            for node in by_flag[
                InvalidDesignFlag.post_treatment_covariate
            ].graph.nodes
            if node.kind is NodeKind.covariate
        )
        self.assertEqual(
            post_treatment.categorical_features["measurement_time"],
            "post_treatment",
        )
        self.assertEqual(
            post_treatment.categorical_features["invalid_role"],
            "adjustment_covariate",
        )

    def test_vocabularies_are_fixed_and_complete(self) -> None:
        self.assertEqual(FAMILY_ORDER, tuple(item.value for item in EpisodeFamily))
        self.assertEqual(
            INVALID_FLAG_ORDER,
            tuple(item.value for item in InvalidDesignFlag),
        )
        self.assertEqual(len(OPERATOR_ORDER), 10)
        self.assertEqual(len(CONCLUSION_ORDER), len(set(CONCLUSION_ORDER)))
        self.assertEqual(len(NODE_TYPE_ORDER), len(set(NODE_TYPE_ORDER)))
        self.assertEqual(len(EDGE_TYPE_ORDER), len(set(EDGE_TYPE_ORDER)))

        for family in EpisodeFamily:
            with self.subTest(family=family):
                episode = generate_episode(family, group_index=7, seed=101)
                self.assertEqual(
                    set(episode.labels.operator_program),
                    set(episode.labels.operator_multilabel),
                )
                self.assertTrue(
                    all(
                        operator.value in OPERATOR_ORDER
                        for operator in episode.labels.operator_program
                    )
                )
                self.assertLessEqual(len(episode.hypotheses), MAX_HYPOTHESES)

    def test_experiment_priors_and_likelihood_matrices_are_normalized(self) -> None:
        episode = generate_episode(
            EpisodeFamily.natural_variation,
            group_index=2,
            seed=5,
        )
        self.assertAlmostEqual(
            sum(
                experiment.selection_prior_probability
                for experiment in episode.candidate_experiments
            ),
            1.0,
        )
        hypothesis_ids = [hypothesis.id for hypothesis in episode.hypotheses]
        equivalence_classes = [
            hypothesis.equivalence_class for hypothesis in episode.hypotheses
        ]
        self.assertLess(len(set(equivalence_classes)), len(equivalence_classes))
        for experiment in episode.candidate_experiments:
            self.assertEqual(experiment.likelihood_hypothesis_order, hypothesis_ids)
            self.assertEqual(
                len(experiment.outcome_likelihood_matrix),
                len(experiment.outcomes),
            )
            for outcome, row in zip(
                experiment.outcomes,
                experiment.outcome_likelihood_matrix,
                strict=True,
            ):
                self.assertEqual(
                    row,
                    [
                        outcome.likelihood_by_hypothesis[hypothesis_id]
                        for hypothesis_id in hypothesis_ids
                    ],
                )
            for column in range(len(hypothesis_ids)):
                self.assertAlmostEqual(
                    sum(
                        row[column]
                        for row in experiment.outcome_likelihood_matrix
                    ),
                    1.0,
                )

    def test_strict_contracts_reject_invalid_payloads(self) -> None:
        with self.assertRaises(ValidationError):
            GraphNode(
                id="node:one",
                kind=NodeKind.pan_gene,
                label="gene",
                numeric_features=["1.0"],
            )
        with self.assertRaises(ValidationError):
            GraphNode(
                id="node:one",
                kind=NodeKind.pan_gene,
                label="gene",
                numeric_features=[1.0],
                unexpected=True,
            )

        episode = generate_episode(
            EpisodeFamily.developmental,
            group_index=0,
            seed=3,
        )
        broken_graph = episode.graph.model_dump(mode="python")
        broken_graph["edges"][0]["source"] = "node:not-present"
        with self.assertRaises(ValidationError) as dangling:
            EpisodeGraph.model_validate(broken_graph)
        self.assertIn("must reference nodes", str(dangling.exception))

        too_many = episode.model_dump(mode="python")
        while len(too_many["hypotheses"]) <= MAX_HYPOTHESES:
            duplicate = dict(too_many["hypotheses"][-1])
            duplicate["id"] = (
                f"{episode.id}/hypothesis/extra-{len(too_many['hypotheses'])}"
            )
            too_many["hypotheses"].append(duplicate)
        with self.assertRaises(ValidationError):
            BiologicalEpisode.model_validate(too_many)

        bad_matrix = episode.model_dump(mode="python")
        bad_matrix["candidate_experiments"][0][
            "outcome_likelihood_matrix"
        ][0][0] = 0.01
        with self.assertRaises(ValidationError) as inconsistent:
            BiologicalEpisode.model_validate(bad_matrix)
        self.assertIn(
            "must match the named outcome likelihoods",
            str(inconsistent.exception),
        )


if __name__ == "__main__":
    unittest.main()
