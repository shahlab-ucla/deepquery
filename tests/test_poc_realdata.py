from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wormctx.poc.realdata import _deterministic_groups, run_caendr_baseline


class CaendrRealDataBaselineTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        shared = [f"S{index:02d}" for index in range(12)]
        kinship_ids = [*shared, "KINSHIP_ONLY"]
        pca_ids = [*reversed(shared), "PCA_ONLY"]

        phenotype_path = root / "phenotypes.csv"
        with phenotype_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["strain", "condition", "trait", "phenotype"])
            for index, strain in enumerate(shared):
                group = index // 4
                value = 1.5 * group + 0.2 * (index % 4) + (0.05 if index % 2 else -0.05)
                writer.writerow([strain, "control", "growth", f"{value:.8f}"])
            writer.writerow(["PHENOTYPE_ONLY", "control", "growth", "4.2"])

        kinship_ids_path = root / "cohort.rel.id"
        kinship_ids_path.write_text(
            "".join(f"FAM {identifier}\n" for identifier in kinship_ids), encoding="utf-8"
        )
        latent = np.asarray(
            [
                [index // 4 - 1.0, (index % 4 - 1.5) / 2.0, 1.0]
                for index in range(12)
            ]
            + [[0.25, -0.25, 1.0]],
            dtype=np.float64,
        )
        kinship = latent @ latent.T / latent.shape[1] + 0.35 * np.eye(len(latent))
        kinship_path = root / "cohort.rel"
        kinship_path.write_text(
            "".join(" ".join(f"{value:.12g}" for value in row) + "\n" for row in kinship),
            encoding="utf-8",
        )

        eigenvec_path = root / "cohort.eigenvec"
        lines = ["#FID IID PC1 PC2 PC3\n"]
        for identifier in pca_ids:
            if identifier == "PCA_ONLY":
                coordinates = (4.0, 4.0, 0.1)
            else:
                index = int(identifier[1:])
                coordinates = (
                    float(index // 4) * 5.0,
                    float(index % 4) * 0.2,
                    float((index % 2) * 2 - 1) * 0.1,
                )
            lines.append(
                f"FAM {identifier} {coordinates[0]} {coordinates[1]} {coordinates[2]}\n"
            )
        eigenvec_path.write_text("".join(lines), encoding="utf-8")
        return phenotype_path, kinship_path, kinship_ids_path, eigenvec_path

    def test_alignment_group_isolation_and_finite_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._write_fixture(Path(temporary))
            result = run_caendr_baseline(*paths, min_samples=9, seed=17)

        self.assertEqual(result["status"], "exploratory_unvalidated")
        self.assertFalse(result["validated"])
        self.assertTrue(result["read_only"])
        self.assertEqual(result["alignment"]["common_strain_count"], 12)
        self.assertEqual(
            result["alignment"]["common_strains_in_kinship_order"],
            [f"S{index:02d}" for index in range(12)],
        )
        self.assertEqual(
            result["alignment"]["dropped"]["phenotype_not_in_kinship"],
            ["PHENOTYPE_ONLY"],
        )
        self.assertEqual(
            result["alignment"]["dropped"]["kinship_not_in_phenotype"],
            ["KINSHIP_ONLY"],
        )
        self.assertEqual(
            result["alignment"]["dropped"]["pca_not_in_phenotype"],
            ["PCA_ONLY"],
        )

        trait = result["traits"][0]
        self.assertEqual(trait["status"], "analyzed_exploratory_unvalidated")
        grouping = trait["grouping"]
        self.assertEqual(
            grouping["method"],
            "capacity_constrained_deterministic_kmeans_on_standardized_pcs",
        )
        self.assertFalse(grouping["uses_phenotype_values"])
        self.assertTrue(grouping["balance_contract"]["satisfied"])
        group_sizes = list(grouping["balance_contract"]["group_sizes"].values())
        self.assertLessEqual(max(group_sizes) - min(group_sizes), 1)
        all_test_strains: set[str] = set()
        for fold in trait["cross_validation"]["folds"]:
            training = set(fold["train_strains"])
            testing = set(fold["test_strains"])
            self.assertTrue(training.isdisjoint(testing))
            self.assertTrue(testing.isdisjoint(all_test_strains))
            all_test_strains.update(testing)
            self.assertIn(fold["selected_alpha"], result["alpha_grid"])
        self.assertEqual(all_test_strains, set(trait["strains"]))
        for model_metrics in trait["metrics"].values():
            self.assertEqual(set(model_metrics), {"rmse", "pearson", "spearman"})
            self.assertTrue(all(math.isfinite(value) for value in model_metrics.values()))
        json.dumps(result, allow_nan=False)

    def test_outer_test_labels_do_not_select_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._write_fixture(root)
            first = run_caendr_baseline(*paths, min_samples=9, seed=23)
            first_trait = first["traits"][0]
            target_fold = first_trait["cross_validation"]["folds"][0]
            held_out = set(target_fold["test_strains"])

            phenotype_path = paths[0]
            with phenotype_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with phenotype_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["strain", "condition", "trait", "phenotype"]
                )
                writer.writeheader()
                for row in rows:
                    if row["strain"] in held_out:
                        row["phenotype"] = str(float(row["phenotype"]) + 10_000.0)
                    writer.writerow(row)

            second = run_caendr_baseline(*paths, min_samples=9, seed=23)
            self.assertEqual(
                second["traits"][0]["grouping"], first_trait["grouping"]
            )
            second_folds = second["traits"][0]["cross_validation"]["folds"]
            matching = next(
                fold
                for fold in second_folds
                if fold["held_out_group"] == target_fold["held_out_group"]
            )
            self.assertEqual(matching["test_strains"], target_fold["test_strains"])
            self.assertEqual(matching["selected_alpha"], target_fold["selected_alpha"])
            self.assertEqual(
                matching["inner_validation_rmse_by_alpha"],
                target_fold["inner_validation_rmse_by_alpha"],
            )

    def test_pathological_pca_outliers_obey_hard_balance_contract(self) -> None:
        strains = [f"DENSE_{index:02d}" for index in range(21)] + [
            "OUTLIER_NORTH",
            "OUTLIER_SOUTH",
            "OUTLIER_EAST",
            "OUTLIER_WEST",
        ]
        dense = np.asarray(
            [
                [index * 1e-4, ((index % 3) - 1) * 1e-4, ((index % 5) - 2) * 1e-5]
                for index in range(21)
            ],
            dtype=np.float64,
        )
        outliers = np.asarray(
            [
                [0.0, 100.0, 0.0],
                [0.0, -100.0, 0.0],
                [100.0, 0.0, 0.0],
                [-100.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        pcs = np.vstack([dense, outliers])

        first_labels, first_metadata, first_warnings = _deterministic_groups(
            pcs, strains, seed=31
        )
        second_labels, second_metadata, second_warnings = _deterministic_groups(
            pcs, strains, seed=31
        )

        sizes = np.bincount(first_labels, minlength=first_metadata["n_groups"])
        self.assertEqual(sizes.tolist(), [5, 5, 5, 5, 5])
        self.assertLessEqual(int(np.max(sizes) - np.min(sizes)), 1)
        self.assertTrue(first_metadata["balance_contract"]["satisfied"])
        self.assertFalse(first_metadata["uses_phenotype_values"])
        self.assertIn("PCA eigenvectors only", first_metadata["feature_source"])
        self.assertTrue(
            any("not inferred biological populations" in item for item in first_warnings)
        )
        np.testing.assert_array_equal(first_labels, second_labels)
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_warnings, second_warnings)


if __name__ == "__main__":
    unittest.main()
