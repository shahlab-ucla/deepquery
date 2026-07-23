from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from wormctx.poc import caendr_compendium_discovery_gate as gate


ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = ROOT / "experiments/natural_variation/phenotype_compendium_grouped_prediction/config/discovery_gate.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _spec(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "logical_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path], Path]:
    iids = [f"strain_{index:03d}" for index in range(203)]
    groups = [f"POP{index % 10 + 1:02d}" for index in range(203)]
    discovery = [f"discovery_trait_{index:04d}" for index in range(1184)]
    validation = [f"validation_trait_{index:04d}" for index in range(261)]
    paths = {
        "source_compendium": tmp_path / "source.csv",
        "stage1_qualification_receipt": tmp_path / "stage1.json",
        "stage1_discovery_trait_list": tmp_path / "discovery.txt",
        "stage1_validation_trait_list": tmp_path / "validation.txt",
        "stage2_kernel_qualification_receipt": tmp_path / "kernel_receipt.json",
        "stage2_population_groups": tmp_path / "groups.tsv",
        "stage2_float64_kernel": tmp_path / "kernel.npy",
        "stage2_pca10_scores": tmp_path / "pca.eigenvec",
    }
    paths["stage1_discovery_trait_list"].write_text(
        "\n".join(discovery) + "\n", encoding="utf-8"
    )
    paths["stage1_validation_trait_list"].write_text(
        "\n".join(validation) + "\n", encoding="utf-8"
    )
    paths["stage1_qualification_receipt"].write_text(
        json.dumps(
            {
                "models_executed": False,
                "compendium": {
                    "discovery_trait_sha256": _ordered_hash(discovery),
                    "validation_trait_sha256": _ordered_hash(validation),
                },
            }
        ),
        encoding="utf-8",
    )
    paths["stage2_population_groups"].write_text(
        "IID\tpopulation_group\n"
        + "".join(
            f"{iid}\t{group}\n" for iid, group in zip(iids, groups, strict=True)
        ),
        encoding="utf-8",
    )
    pca = np.column_stack(
        [
            np.sin((np.arange(203) + 1) * (dimension + 1) / 37.0)
            + np.arange(203) * (dimension + 1) / 10000.0
            for dimension in range(10)
        ]
    )
    paths["stage2_pca10_scores"].write_text(
        "#FID\tIID\t" + "\t".join(f"PC{index}" for index in range(1, 11)) + "\n"
        + "".join(
            f"0\t{iid}\t" + "\t".join(f"{value:.12g}" for value in row) + "\n"
            for iid, row in zip(iids, pca, strict=True)
        ),
        encoding="utf-8",
    )
    kernel = pca @ pca.T / pca.shape[1] + np.eye(203) * 0.25
    np.save(paths["stage2_float64_kernel"], kernel, allow_pickle=False)
    group_sizes = dict(sorted(gate.Counter(groups).items()))
    paths["stage2_kernel_qualification_receipt"].write_text(
        json.dumps(
            {
                "schema_version": (
                    "wormctx-caendr-compendium-203-kernel-qualification-1.0"
                ),
                "samples": 203,
                "ordered_iid_sha256": _ordered_hash(iids),
                "phenotype_values_accessed": False,
                "legacy_numeric_assets_accessed": False,
                "discovery_prediction_models_executed": False,
                "validation_outcomes_locked": True,
                "partition_identity": {
                    "discovery_trait_sha256": _ordered_hash(discovery),
                    "validation_trait_sha256": _ordered_hash(validation),
                },
                "population_groups": {
                    "selected_k": 10,
                    "group_sizes": group_sizes,
                },
            }
        ),
        encoding="utf-8",
    )
    with paths["source_compendium"].open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(gate.SOURCE_HEADER)
        for trait_index, trait in enumerate(discovery):
            for sample_index, iid in enumerate(iids[:150]):
                value = (
                    trait_index / 100.0
                    + math.sin(sample_index / 9.0)
                    + (sample_index % 13) / 10.0
                )
                writer.writerow(
                    ["synthetic", "c_elegans", trait, iid, f"{value:.12g}"]
                )
        for trait, iid in zip(validation, iids * 2, strict=False):
            writer.writerow(
                ["synthetic", "c_elegans", trait, iid, "DO_NOT_PARSE_LOCKED_VALUE"]
            )

    config = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    config["analysis_id"] = "synthetic_discovery_gate"
    config["inputs"] = [
        _spec(item["role"], paths[item["role"]]) for item in config["inputs"]
    ]
    config["python_tool"] = {
        "version": "synthetic",
        "sha256": _sha256(Path(sys.executable)),
        "maximum_threads": 1,
    }
    config["partition_contract"]["discovery_trait_sha256"] = _ordered_hash(discovery)
    config["partition_contract"]["validation_trait_sha256"] = _ordered_hash(validation)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    resource = tmp_path / "resource.txt"
    resource.write_text("synthetic single-thread coexistence audit\n", encoding="utf-8")
    return config_path, paths, resource


def _qualification(
    config: Path, paths: dict[str, Path], resource: Path, output: Path
) -> dict[str, object]:
    return gate.qualify_discovery_masks(
        config,
        paths["source_compendium"],
        paths["stage1_qualification_receipt"],
        paths["stage1_discovery_trait_list"],
        paths["stage1_validation_trait_list"],
        paths["stage2_kernel_qualification_receipt"],
        paths["stage2_population_groups"],
        paths["stage2_float64_kernel"],
        paths["stage2_pca10_scores"],
        sys.executable,
        resource,
        output,
    )


def test_real_contract_freezes_discovery_only_ladder_and_custody() -> None:
    config = gate.load_config(REAL_CONFIG)
    assert [item["id"] for item in config["model_ladder"]["models"]] == gate.MODEL_IDS
    assert config["smoke_contract"]["traits"] == 8
    assert config["smoke_contract"]["maximum_threads"] == 1
    assert config["validation_custody"]["locked"] is True
    assert config["validation_custody"]["runner_command_exists"] is False
    assert config["legacy_numeric_quarantine"]["accepted_as_inputs"] is False


def test_outcome_blind_qualification_and_discovery_smoke(tmp_path: Path) -> None:
    config, paths, resource = _fixture(tmp_path)
    qualification = tmp_path / "qualification"
    receipt = _qualification(config, paths, resource, qualification)

    assert receipt["qualified_discovery_traits"] == 1184
    assert len(receipt["smoke_traits"]) == 8
    assert receipt["discovery_value_magnitude_or_rank_evaluated"] is False
    assert receipt["models_executed"] is False
    custody = receipt["validation_custody"]
    assert custody["validation_outcome_value_field_evaluated"] is False
    assert custody["validation_trait_specific_masks_computed"] is False
    assert custody["validation_rows_routed_without_value_evaluation"] == 261
    assert "validation_trait_" not in (
        qualification / "trait_mask_manifest.tsv"
    ).read_text()

    smoke = tmp_path / "smoke"
    smoke_receipt = gate.run_discovery_smoke(
        config,
        paths["source_compendium"],
        paths["stage1_qualification_receipt"],
        paths["stage1_discovery_trait_list"],
        paths["stage1_validation_trait_list"],
        paths["stage2_kernel_qualification_receipt"],
        paths["stage2_population_groups"],
        paths["stage2_float64_kernel"],
        paths["stage2_pca10_scores"],
        sys.executable,
        qualification,
        resource,
        smoke,
    )
    assert smoke_receipt["technical_smoke_completed"] is True
    assert smoke_receipt["execution_thread_cap"] == 1
    assert smoke_receipt["null_model_fits"] == 8 * 10 * 3 * 4
    assert smoke_receipt["validation_custody"][
        "validation_outcome_value_field_evaluated"
    ] is False
    assert smoke_receipt["validation_custody"][
        "validation_predictions_generated"
    ] is False
    assert smoke_receipt["predictive_performance_claims_permitted"] is False
    prediction_lines = (smoke / "discovery_predictions.tsv").read_text().splitlines()
    assert len(prediction_lines) == 1 + 8 * 150 * 3
    assert "DO_NOT_PARSE_LOCKED_VALUE" not in "\n".join(prediction_lines)


def test_predictors_cannot_access_heldout_outcomes() -> None:
    rng = np.random.default_rng(90210)
    pca = rng.normal(size=(20, 10))
    raw = rng.normal(size=(20, 20))
    kernel = raw @ raw.T + np.eye(20)
    train = np.arange(15)
    test = np.arange(15, 20)
    y_train = rng.normal(size=15)

    for model in gate.MODEL_IDS:
        first = gate._predict_model(model, pca, kernel, train, test, y_train)
        second = gate._predict_model(model, pca, kernel, train, test, y_train.copy())
        np.testing.assert_array_equal(first, second)


def test_permutation_seed_is_deterministic_and_fold_specific() -> None:
    first = gate._permutation_seed("namespace:", "trait", "POP01", 1)
    assert first == gate._permutation_seed("namespace:", "trait", "POP01", 1)
    assert first != gate._permutation_seed("namespace:", "trait", "POP02", 1)
    assert first != gate._permutation_seed("namespace:", "trait", "POP01", 2)


@pytest.mark.skip(
    reason="the site-specific runner is intentionally absent from the public tree"
)
def test_runner_has_no_validation_mode_and_caps_every_math_runtime() -> None:
    runner = (
        ROOT / "scripts/run_caendr_compendium_discovery_gate.sh"
    ).read_text(encoding="utf-8")
    assert 'MODE" != "qualify" && "$MODE" != "smoke"' in runner
    assert "no validation command exists" in runner
    assert "OMP_NUM_THREADS=1" in runner
    assert "OPENBLAS_NUM_THREADS=1" in runner
    assert "MKL_NUM_THREADS=1" in runner
    assert "timeout 900" in runner
    assert "\nGROUPS=" not in runner
    assert "POPULATION_GROUPS=$8" in runner
