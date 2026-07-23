from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from wormctx.poc.realconfig import (
    load_real_baseline_manifest,
    run_frozen_caendr_baseline,
)
from wormctx.poc.repro import sha256_file


class FrozenRealBaselineTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> dict[str, Path]:
        data = root / "cohort"
        data.mkdir()
        strains = [f"S{index:02d}" for index in range(12)]
        phenotype = data / "phenotypes.csv"
        with phenotype.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["strain", "condition", "trait", "phenotype"])
            for index, strain in enumerate(strains):
                writer.writerow([strain, "drug", "growth", 0.3 * index + index % 3])

        identifiers = data / "cohort.rel.id"
        identifiers.write_text(
            "".join(f"FAM {strain}\n" for strain in strains), encoding="utf-8"
        )
        latent = np.asarray(
            [[index // 4, index % 4, 1.0] for index in range(12)], dtype=np.float64
        )
        relationship = latent @ latent.T / 3.0 + np.eye(12)
        kinship = data / "cohort.rel"
        kinship.write_text(
            "".join(
                " ".join(f"{value:.12g}" for value in row) + "\n"
                for row in relationship
            ),
            encoding="utf-8",
        )
        eigenvectors = data / "cohort.eigenvec"
        eigenvectors.write_text(
            "#FID IID PC1 PC2 PC3\n"
            + "".join(
                f"FAM {strain} {index // 4} {index % 4} {index % 2}\n"
                for index, strain in enumerate(strains)
            ),
            encoding="utf-8",
        )
        return {
            "phenotypes": phenotype,
            "kinship": kinship,
            "kinship_ids": identifiers,
            "eigenvectors": eigenvectors,
        }

    def _write_manifest(
        self,
        root: Path,
        inputs: dict[str, Path],
        *,
        traits: list[str] | None = None,
    ) -> Path:
        payload = {
            "schema_version": "wormctx-real-baseline-1.0",
            "analysis_id": "fixture_drug_growth",
            "analysis_kind": "kinship_prediction_audit_not_qtl",
            "status": "exploratory_unvalidated",
            "validated": False,
            "rights_status": "review_required",
            "use_scope": "read_only_audit",
            **{
                role: {
                    "role": role,
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for role, path in inputs.items()
            },
            "phenotype_contract": {
                "condition": "drug",
                "traits": traits or ["growth"],
                "strain_column": "strain",
                "condition_column": "condition",
                "trait_column": "trait",
                "value_column": "phenotype",
                "replicate_policy": (
                    "finite_arithmetic_mean_within_condition_trait_strain"
                ),
            },
            "folds": {
                "method": (
                    "capacity_constrained_deterministic_kmeans_on_standardized_pcs"
                ),
                "seed": 41,
                "expected_group_count": 3,
                "maximum_group_size_difference": 1,
                "uses_phenotype_values": False,
                "min_samples": 9,
            },
            "known_limitations": [
                "fixture-only predictive audit; this is not a QTL analysis"
            ],
            "biological_claims_permitted": False,
        }
        manifest = root / "baseline.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return manifest

    def test_frozen_manifest_verifies_inputs_traits_and_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._write_inputs(root)
            manifest = self._write_manifest(root, inputs)
            result = run_frozen_caendr_baseline(manifest, root)

        self.assertTrue(result["frozen_analysis"]["inputs_verified_before_analysis"])
        self.assertFalse(result["frozen_analysis"]["biological_claims_permitted"])
        self.assertEqual(result["frozen_analysis"]["rights_status"], "review_required")
        self.assertEqual(
            result["frozen_analysis"]["analysis_kind"],
            "kinship_prediction_audit_not_qtl",
        )
        self.assertEqual(result["traits"][0]["analysis_id"], "drug::growth")
        self.assertEqual(result["traits"][0]["grouping"]["n_groups"], 3)

    def test_checksum_mismatch_stops_before_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._write_inputs(root)
            manifest = self._write_manifest(root, inputs)
            inputs["phenotypes"].write_text("changed\n", encoding="utf-8")
            with patch("wormctx.poc.realconfig.run_caendr_baseline") as runner:
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    run_frozen_caendr_baseline(manifest, root)
            runner.assert_not_called()

    def test_trait_contract_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._write_inputs(root)
            manifest = self._write_manifest(root, inputs, traits=["unseen_trait"])
            with self.assertRaisesRegex(ValueError, "condition/trait set differs"):
                run_frozen_caendr_baseline(manifest, root)

    def test_path_escape_and_role_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._write_inputs(root)
            manifest = self._write_manifest(root, inputs)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["phenotypes"]["relative_path"] = "../outside.csv"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_real_baseline_manifest(manifest)

            payload["phenotypes"]["relative_path"] = "cohort/phenotypes.csv"
            payload["phenotypes"]["role"] = "kinship"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_real_baseline_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
