from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from wormctx.poc.operators import (
    OPERATOR_VOCABULARY,
    DesignValidationError,
    InvalidDesignCode,
    OperatorName,
    bayesian_outcome_update,
    cost_risk_adjusted_utility,
    design_issues,
    equivalence_class_conclusion,
    expected_information_gain,
    normalized_entropy,
    operator_signature,
    rank_experiments,
    validate_design,
)


class TypedVocabularyTests(unittest.TestCase):
    def test_initial_vocabulary_is_fixed_and_typed(self) -> None:
        self.assertEqual(len(OPERATOR_VOCABULARY), 10)
        self.assertEqual(
            operator_signature("derive_trait").output_type,
            "TraitGraph",
        )
        self.assertIn(OperatorName.UPDATE_HYPOTHESIS_GRAPH, OPERATOR_VOCABULARY)
        with self.assertRaisesRegex(ValueError, "unknown typed operator"):
            operator_signature("invent_a_p_value")


class DesignGuardTests(unittest.TestCase):
    def assertRejected(self, design: dict[str, object], code: InvalidDesignCode) -> None:
        issues = design_issues(design)
        self.assertIn(code, {issue.code for issue in issues})
        with self.assertRaises(DesignValidationError) as raised:
            validate_design(design)
        self.assertIn(code, {issue.code for issue in raised.exception.issues})

    def test_rejects_dominance_without_heterozygotes(self) -> None:
        self.assertRejected(
            {"estimand": "dominance", "genotype_states": ["homozygous_reference"]},
            InvalidDesignCode.DOMINANCE_REQUIRES_HETEROZYGOTES,
        )
        validate_design({"estimand": "dominance", "genotype_states": ["heterozygous"]})

    def test_rejects_absent_or_unassembled_locus_as_reference_snp(self) -> None:
        for state in ("absent", "unassembled"):
            with self.subTest(state=state):
                self.assertRejected(
                    {
                        "locus_state": state,
                        "variant_representation": "reference_snp",
                    },
                    InvalidDesignCode.STRUCTURAL_ABSENCE_IS_NOT_REFERENCE_SNP,
                )

    def test_rejects_database_absence_as_powered_negative(self) -> None:
        self.assertRejected(
            {
                "evidence_basis": "database_absence",
                "evidence_interpretation": "powered_negative",
            },
            InvalidDesignCode.DATABASE_ABSENCE_IS_NOT_POWERED_NEGATIVE,
        )

    def test_rejects_formal_transport_without_identified_model(self) -> None:
        self.assertRejected(
            {
                "transport_kind": "formal_causal_transport",
                "source_target_model_identified": False,
            },
            InvalidDesignCode.TRANSPORT_REQUIRES_IDENTIFIED_MODEL,
        )
        validate_design(
            {
                "transport_kind": "formal_causal_transport",
                "source_target_model_identified": True,
            }
        )

    def test_rejects_unjustified_post_treatment_covariate(self) -> None:
        self.assertRejected(
            {"post_treatment_covariates": ["body_size_after_treatment"]},
            InvalidDesignCode.POST_TREATMENT_COVARIATE_REQUIRES_JUSTIFICATION,
        )
        validate_design(
            {
                "post_treatment_covariates": ["body_size_after_treatment"],
                "post_treatment_justification": "g-formula mediator analysis",
            }
        )

    def test_reports_multiple_issues_in_stable_rule_order(self) -> None:
        issues = design_issues(
            {
                "estimate_dominance": True,
                "has_heterozygotes": False,
                "locus_state": "unalignable",
                "treat_as_reference_snp": True,
                "evidence_basis": "database_absence",
                "powered_negative": True,
                "formal_transport": True,
                "source_target_model_identified": False,
                "condition_on_post_treatment": True,
            }
        )
        self.assertEqual([issue.code for issue in issues], list(InvalidDesignCode))

    def test_contract_style_invalid_flags_are_accepted_without_importing_contracts(self) -> None:
        episode = SimpleNamespace(
            labels=SimpleNamespace(
                invalid_design_flags=["transport_without_identification"]
            )
        )
        self.assertRejected(
            episode,
            InvalidDesignCode.TRANSPORT_REQUIRES_IDENTIFIED_MODEL,
        )


class ProbabilityOperatorTests(unittest.TestCase):
    def test_normalized_entropy_has_expected_extremes(self) -> None:
        self.assertEqual(normalized_entropy([1.0, 0.0]), 0.0)
        self.assertAlmostEqual(normalized_entropy([0.25] * 4), 1.0)
        self.assertAlmostEqual(normalized_entropy({"H1": 0.75, "H2": 0.25}), 0.8112781245)

    def test_bayesian_outcome_update(self) -> None:
        posterior = bayesian_outcome_update(
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.9, "H2": 0.1},
        )
        self.assertAlmostEqual(posterior["H1"], 0.9)
        self.assertAlmostEqual(sum(posterior.values()), 1.0)

    def test_expected_information_gain_sanity(self) -> None:
        prior = {"H1": 0.5, "H2": 0.5}
        perfect = {
            "positive": {"H1": 1.0, "H2": 0.0},
            "negative": {"H1": 0.0, "H2": 1.0},
        }
        uninformative = {
            "positive": {"H1": 0.5, "H2": 0.5},
            "negative": {"H1": 0.5, "H2": 0.5},
        }
        self.assertAlmostEqual(expected_information_gain(prior, perfect), 1.0)
        self.assertAlmostEqual(expected_information_gain(prior, uninformative), 0.0)

        four_way_prior = {f"H{index}": 0.25 for index in range(4)}
        four_way_perfect = {
            f"outcome-{outcome}": {
                f"H{hypothesis}": float(outcome == hypothesis)
                for hypothesis in range(4)
            }
            for outcome in range(4)
        }
        self.assertAlmostEqual(
            expected_information_gain(four_way_prior, four_way_perfect),
            1.0,
        )
        self.assertAlmostEqual(
            expected_information_gain(
                four_way_prior,
                four_way_perfect,
                normalized=False,
            ),
            2.0,
        )

    def test_probability_guards_are_explicit(self) -> None:
        invalid_distributions = (
            [],
            [0.2, 0.2],
            [-0.1, 1.1],
            [math.nan, math.nan],
            [math.inf, 0.0],
        )
        for probabilities in invalid_distributions:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    normalized_entropy(probabilities)

        with self.assertRaisesRegex(ValueError, "exactly the prior hypothesis keys"):
            bayesian_outcome_update({"H1": 1.0}, {"H2": 1.0})
        with self.assertRaisesRegex(ValueError, "zero probability"):
            bayesian_outcome_update({"H1": 1.0}, {"H1": 0.0})
        with self.assertRaisesRegex(ValueError, "sum to 1 for every hypothesis"):
            expected_information_gain(
                {"H1": 1.0},
                {"yes": {"H1": 0.7}, "no": {"H1": 0.7}},
            )


class ExperimentDecisionTests(unittest.TestCase):
    def test_cost_risk_adjusted_utility(self) -> None:
        self.assertAlmostEqual(
            cost_risk_adjusted_utility(0.9, cost=2.0, risk=0.5, risk_weight=2.0),
            0.3,
        )
        with self.assertRaises(ValueError):
            cost_risk_adjusted_utility(1.0, cost=0.0, risk=0.0)
        with self.assertRaisesRegex(ValueError, r"risk.*\[0, 1\]"):
            cost_risk_adjusted_utility(1.0, cost=1.0, risk=1.1)

    def test_experiment_ranking_uses_eig_cost_and_risk(self) -> None:
        prior = {"H1": 0.5, "H2": 0.5}
        perfect = {
            "yes": {"H1": 1.0, "H2": 0.0},
            "no": {"H1": 0.0, "H2": 1.0},
        }
        partial = {
            "yes": {"H1": 0.8, "H2": 0.2},
            "no": {"H1": 0.2, "H2": 0.8},
        }
        ranking = rank_experiments(
            prior,
            [
                {
                    "experiment_id": "perfect-but-expensive",
                    "outcome_likelihoods": perfect,
                    "cost": 20.0,
                    "risk": 0.1,
                },
                {
                    "experiment_id": "partial-and-cheap",
                    "outcome_likelihoods": partial,
                    "cost": 1.0,
                    "risk": 0.1,
                },
            ],
        )
        self.assertEqual([item.rank for item in ranking], [1, 2])
        self.assertEqual(ranking[0].experiment_id, "partial-and-cheap")
        self.assertGreater(ranking[0].utility, ranking[1].utility)

    def test_tied_ranking_is_deterministic_by_id(self) -> None:
        likelihoods = {
            "yes": {"H1": 0.5, "H2": 0.5},
            "no": {"H1": 0.5, "H2": 0.5},
        }
        ranking = rank_experiments(
            {"H1": 0.5, "H2": 0.5},
            [
                {"id": "b", "outcome_likelihoods": likelihoods, "cost": 1.0, "risk": 0.1},
                {"id": "a", "outcome_likelihoods": likelihoods, "cost": 1.0, "risk": 0.1},
            ],
        )
        self.assertEqual([item.experiment_id for item in ranking], ["a", "b"])

    def test_ranking_accepts_contract_style_outcome_objects(self) -> None:
        experiment = SimpleNamespace(
            id="reporter",
            cost=3.0,
            risk=0.2,
            outcomes=[
                SimpleNamespace(
                    id="yes",
                    likelihood_by_hypothesis={"H1": 0.9, "H2": 0.1},
                ),
                SimpleNamespace(
                    id="no",
                    likelihood_by_hypothesis={"H1": 0.1, "H2": 0.9},
                ),
            ],
        )
        ranking = rank_experiments({"H1": 0.5, "H2": 0.5}, [experiment])
        self.assertEqual(ranking[0].experiment_id, "reporter")
        self.assertGreater(ranking[0].expected_information_gain, 0.0)

    def test_equivalence_class_preserves_set_valued_answer(self) -> None:
        conclusion = equivalence_class_conclusion(
            {"gene-a": 0.55, "gene-b": 0.40, "gene-c": 0.05},
            {
                "gene-a": (1, 1, 0, 0),
                "gene-b": (1, 1, 0, 0),
                "gene-c": (0, 1, 0, 1),
            },
        )
        self.assertEqual(conclusion.candidates, ("gene-a", "gene-b"))
        self.assertTrue(conclusion.is_set_valued)
        self.assertIsNone(conclusion.resolved_candidate)


if __name__ == "__main__":
    unittest.main()
