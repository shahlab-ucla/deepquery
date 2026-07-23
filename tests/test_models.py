from __future__ import annotations

import copy
import unittest
from pathlib import Path

from pydantic import ValidationError

from wormctx.io import (
    CollectionIntegrityError,
    iter_jsonl,
    load_observations,
    validate_observation_collection,
    write_jsonl,
)
from wormctx.models import (
    ContextualObservation,
    ObservationStatus,
    QuantitativeValue,
    ReleaseManifest,
    ResultAssetReference,
)
from wormctx.adapters.local_jsonl import LocalJsonlAdapter


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "data" / "examples"


class ModelTests(unittest.TestCase):
    def test_example_observations_validate(self) -> None:
        observations = load_observations(EXAMPLES / "observations.jsonl")
        self.assertEqual(len(observations), 5)
        self.assertEqual(
            sum(item.observation_status is ObservationStatus.explicit_negative for item in observations),
            1,
        )

    def test_explicit_negative_requires_directional_evidence(self) -> None:
        row = copy.deepcopy(next(iter_jsonl(EXAMPLES / "observations.jsonl")))
        row["id"] = "wormctx:invalid-negative"
        row["observation_status"] = "explicit_negative"
        row["proposition_negated"] = True
        row["evidence_lines"][0]["direction"] = "neutral"
        with self.assertRaises(ValidationError):
            ContextualObservation.model_validate(row)

    def test_missing_is_not_materialized_as_negative(self) -> None:
        observations = load_observations(EXAMPLES / "observations.jsonl")
        literature = next(item for item in observations if item.id.endswith("literature-001"))
        self.assertEqual(literature.observation_status.value, "uncertain")
        self.assertEqual(
            [item.value for item in literature.context.context_missing_reasons],
            ["not_reported"],
        )

    def test_duplicate_observation_ids_are_rejected(self) -> None:
        source = EXAMPLES / "observations.jsonl"
        first = source.read_text(encoding="utf-8").splitlines()[0]
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.jsonl"
            duplicate.write_text(first + "\n" + first + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate observation id"):
                load_observations(duplicate)
            report = LocalJsonlAdapter().validate_raw(duplicate)
            self.assertFalse(report.valid)
            self.assertIn("duplicate_observation_id", {item.code for item in report.issues})

    def test_empty_jsonl_is_invalid(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            empty = Path(directory) / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            report = LocalJsonlAdapter().validate_raw(empty)
            self.assertFalse(report.valid)
            self.assertEqual(report.issues[0].code, "empty_snapshot")

    def test_negative_and_negation_state_are_bound(self) -> None:
        row = copy.deepcopy(next(iter_jsonl(EXAMPLES / "observations.jsonl")))
        row["id"] = "wormctx:bad-negation-state"
        row["observation_status"] = "explicit_negative"
        row["proposition_negated"] = False
        with self.assertRaises(ValidationError):
            ContextualObservation.model_validate(row)
        row["observation_status"] = "positive"
        row["proposition_negated"] = True
        with self.assertRaises(ValidationError):
            ContextualObservation.model_validate(row)

    def test_quantitative_result_preserves_comparison_and_asset_semantics(self) -> None:
        row = copy.deepcopy(next(iter_jsonl(EXAMPLES / "observations.jsonl")))
        digest = "a" * 64
        row["result"] = {
            "id": "wormctx:result-test-001",
            "result_type": "mixed",
            "summary": "Synthetic effect estimate with a trajectory payload.",
            "comparison_groups": [
                {
                    "id": "wormctx:group-exposed-001",
                    "label": "exposed",
                    "role": "exposure",
                    "size": 12,
                },
                {
                    "id": "wormctx:group-control-001",
                    "label": "vehicle control",
                    "role": "control",
                    "size": 12,
                },
            ],
            "measurements": [
                {
                    "id": "wormctx:measurement-effect-001",
                    "measured_feature": {
                        "id": "wormctx:normalized-effect",
                        "label": "normalized effect",
                    },
                    "value": {"value": 0.42},
                    "statistic": {"id": "STATO:0000303", "label": "effect estimate"},
                    "uncertainty": {"value": 0.08},
                    "uncertainty_type": {
                        "id": "STATO:0000264",
                        "label": "standard error",
                    },
                    "confidence_level": 0.95,
                    "p_value": 0.004,
                    "penetrance": {
                        "value": 0.75,
                        "unit": {"id": "UO:0000187", "label": "percent fraction"},
                    },
                    "sample_size": 24,
                    "biological_replicates": 3,
                    "technical_replicates": 2,
                    "comparison_group_ids": [
                        "wormctx:group-exposed-001",
                        "wormctx:group-control-001",
                    ],
                    "asset_ids": ["wormctx:asset-trajectory-001"],
                }
            ],
            "asset_references": [
                {
                    "id": "wormctx:asset-trajectory-001",
                    "uri": f"urn:sha256:{digest}",
                    "checksum_sha256": digest,
                    "media_type": "application/zarr",
                    "byte_size": 2048,
                    "role": "trajectory array",
                }
            ],
        }
        observation = ContextualObservation.model_validate(row)
        self.assertEqual(observation.result.measurements[0].sample_size, 24)
        self.assertEqual(
            observation.result.measurements[0].comparison_group_ids,
            ["wormctx:group-exposed-001", "wormctx:group-control-001"],
        )
        self.assertEqual(
            observation.result.asset_references[0].checksum_sha256, digest
        )

        row["result"]["asset_references"][0]["checksum_sha256"] = "not-a-digest"
        with self.assertRaises(ValidationError):
            ContextualObservation.model_validate(row)

    def test_linkml_required_fields_are_required_at_runtime(self) -> None:
        row = copy.deepcopy(next(iter_jsonl(EXAMPLES / "observations.jsonl")))
        del row["context"]["organism_taxon"]
        with self.assertRaises(ValidationError):
            ContextualObservation.model_validate(row)

        import json

        manifest = json.loads(
            (ROOT / "tests" / "fixtures" / "local_release.json").read_text(
                encoding="utf-8"
            )
        )
        no_method = copy.deepcopy(manifest)
        del no_method["artifacts"][0]["acquisition"]["method"]
        with self.assertRaises(ValidationError):
            ReleaseManifest.model_validate(no_method)
        no_acquisition = copy.deepcopy(manifest)
        del no_acquisition["artifacts"][0]["acquisition"]
        with self.assertRaises(ValidationError):
            ReleaseManifest.model_validate(no_acquisition)

    def test_context_gaps_are_unique_and_cannot_contradict_values(self) -> None:
        row = copy.deepcopy(next(iter_jsonl(EXAMPLES / "observations.jsonl")))
        row["context"]["context_gaps"] = [
            {"dimension": "anatomy", "reason": "not_reported"},
            {"dimension": "anatomy", "reason": "not_curated"},
        ]
        with self.assertRaisesRegex(ValidationError, "at most one gap"):
            ContextualObservation.model_validate(row)

        row = copy.deepcopy(next(iter_jsonl(EXAMPLES / "observations.jsonl")))
        row["context"]["context_gaps"] = [
            {"dimension": "life_stage", "reason": "not_reported"}
        ]
        with self.assertRaisesRegex(ValidationError, "cannot coexist"):
            ContextualObservation.model_validate(row)

    def test_quantitative_bounds_are_ordered(self) -> None:
        with self.assertRaisesRegex(ValidationError, "lower_bound"):
            QuantitativeValue(lower_bound=2.0, upper_bound=1.0)

    def test_collection_rejects_incompatible_entity_reuse_everywhere(self) -> None:
        first = copy.deepcopy(next(iter_jsonl(EXAMPLES / "observations.jsonl")))
        second = copy.deepcopy(first)
        second["id"] = "wormctx:obs-entity-conflict-002"
        second["evidence_lines"][0]["id"] = "wormctx:evidence-entity-conflict-002"
        second["subject"]["label"] = "conflicting gene identity"

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "entity-conflict.jsonl"
            write_jsonl(path, [first, second])
            with self.assertRaisesRegex(
                CollectionIntegrityError, "conflicting labels"
            ):
                load_observations(path)
            report = LocalJsonlAdapter().validate_raw(path)
            self.assertFalse(report.valid)
            self.assertIn(
                "incompatible_entity_reuse", {item.code for item in report.issues}
            )
            with self.assertRaisesRegex(
                CollectionIntegrityError, "conflicting labels"
            ):
                list(LocalJsonlAdapter().iter_observations(path))

    def test_collection_rejects_cross_kind_and_generated_id_collisions(self) -> None:
        row = copy.deepcopy(next(iter_jsonl(EXAMPLES / "observations.jsonl")))
        row["evidence_lines"][0]["id"] = row["subject"]["id"]
        observation = ContextualObservation.model_validate(row)
        with self.assertRaisesRegex(
            CollectionIntegrityError, "reused incompatibly"
        ):
            validate_observation_collection([observation])

        row = copy.deepcopy(next(iter_jsonl(EXAMPLES / "observations.jsonl")))
        row["qualifiers"] = [
            {
                "id": f"{row['id']}/intervention/1",
                "label": "collides with generated intervention node",
            }
        ]
        observation = ContextualObservation.model_validate(row)
        with self.assertRaisesRegex(
            CollectionIntegrityError, "generated_intervention"
        ):
            validate_observation_collection([observation])

    def test_collection_rejects_inconsistent_snapshot_provenance(self) -> None:
        first = copy.deepcopy(next(iter_jsonl(EXAMPLES / "observations.jsonl")))
        second = copy.deepcopy(first)
        second["id"] = "wormctx:obs-snapshot-conflict-002"
        second["evidence_lines"][0]["id"] = "wormctx:evidence-snapshot-conflict-002"
        second["provenance"]["checksum_sha256"] = "c" * 64
        observations = [
            ContextualObservation.model_validate(first),
            ContextualObservation.model_validate(second),
        ]
        with self.assertRaisesRegex(
            CollectionIntegrityError, "identical snapshot provenance"
        ):
            validate_observation_collection(observations)

    def test_nonfinite_quantitative_values_are_rejected(self) -> None:
        for field_name in ("value", "lower_bound", "upper_bound"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaises(ValidationError):
                        QuantitativeValue.model_validate({field_name: value})

    def test_sha256_asset_urn_must_match_checksum(self) -> None:
        digest = "a" * 64
        asset = ResultAssetReference(
            id="wormctx:asset-case-insensitive",
            uri=f"URN:SHA256:{digest.upper()}",
            checksum_sha256=digest,
            media_type="application/octet-stream",
        )
        self.assertEqual(asset.checksum_sha256, digest)
        with self.assertRaisesRegex(ValidationError, "must equal checksum"):
            ResultAssetReference(
                id="wormctx:asset-mismatch",
                uri=f"urn:sha256:{digest}",
                checksum_sha256="b" * 64,
                media_type="application/octet-stream",
            )

    def test_biolink_exchange_metadata_uses_pinned_enums(self) -> None:
        row = copy.deepcopy(next(iter_jsonl(EXAMPLES / "observations.jsonl")))
        row["biolink_knowledge_level"] = "statistical_association"
        row["biolink_agent_type"] = "data_analysis_pipeline"
        observation = ContextualObservation.model_validate(row)
        self.assertEqual(
            observation.biolink_knowledge_level.value, "statistical_association"
        )
        row["biolink_knowledge_level"] = "garbage"
        with self.assertRaises(ValidationError):
            ContextualObservation.model_validate(row)
        row["biolink_knowledge_level"] = "observation"
        row["biolink_agent_type"] = "garbage"
        with self.assertRaises(ValidationError):
            ContextualObservation.model_validate(row)


if __name__ == "__main__":
    unittest.main()
