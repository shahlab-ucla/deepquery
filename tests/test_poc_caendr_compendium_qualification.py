from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from wormctx.poc.caendr_compendium_qualification import _load_config, qualify_compendium


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(path: Path) -> dict[str, object]:
    return {
        "logical_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _ordered_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, list[Path]]:
    iids = [f"strain_{index:03d}" for index in range(209)]
    fam = tmp_path / "panel.fam"
    fam.write_text(
        "".join(f"0\t{iid}\t0\t0\t0\t-9\n" for iid in iids), encoding="utf-8"
    )
    trait_names = [f"trait_{index:02d}" for index in range(5)]
    phenotypes = tmp_path / "phenotypes.csv"
    _write_csv(
        phenotypes,
        ["submitted_by", "species_name", "trait_name", "strain_name", "trait_value"],
        [
            ["submitter", "c_elegans", trait, iid, f"{trait_index + strain_index / 1000:.6f}"]
            for trait_index, trait in enumerate(trait_names)
            for strain_index, iid in enumerate(iids[:180])
        ],
    )
    h2_screen = tmp_path / "h2_screen.csv"
    _write_csv(
        h2_screen,
        ["trait", "n", "h2_HE", "is_expr", "h2_clip"],
        [[trait, 203, 0.5, "True", 0.5] for trait in trait_names],
    )
    h2_greml = tmp_path / "h2_greml.csv"
    _write_csv(
        h2_greml,
        ["trait", "n", "h2_greml"],
        [[trait, 203, 0.5] for trait in reversed(trait_names)],
    )
    top15 = tmp_path / "top15.csv"
    _write_csv(
        top15,
        ["trait", "n", "h2", "gblup_r", "gblup_rho"],
        [[trait, 203, 0.5, 0.2, 0.1] for trait in trait_names[:2]],
    )
    legacy = [h2_screen, h2_greml, top15]
    config = {
        "schema_version": "wormctx-caendr-compendium-qualification-config-1.1",
        "analysis_id": "synthetic_compendium_qualification",
        "classification": "retrospective_source_qualification_only",
        "biological_claims_permitted": False,
        "source": {
            "asset": _asset(phenotypes),
            "expected_header": [
                "submitted_by",
                "species_name",
                "trait_name",
                "strain_name",
                "trait_value",
            ],
            "expected_data_rows": 900,
            "expected_trait_count": 5,
        },
        "genotype_panel": {
            "asset": _asset(fam),
            "expected_samples": 209,
            "expected_fid": "0",
            "expected_ordered_iid_sha256": _ordered_hash(iids),
            "exact_order_required": True,
        },
        "legacy_quarantine": [
            {
                "id": "legacy_h2_screen_203",
                "asset": _asset(h2_screen),
                "expected_header": ["trait", "n", "h2_HE", "is_expr", "h2_clip"],
                "expected_rows": 5,
                "legacy_panel_sample_ceiling": 203,
                "expected_n_histogram": {"203": 5},
                "numerical_result_fields_used_for_design": False,
                "permitted_use": "fingerprint_and_structural_quarantine_only",
                "prohibited_uses": [
                    "trait_selection",
                    "hyperparameter_selection",
                    "model_comparison",
                    "preliminary_evidence",
                    "biological_claims",
                ],
            },
            {
                "id": "legacy_h2_greml_203",
                "asset": _asset(h2_greml),
                "expected_header": ["trait", "n", "h2_greml"],
                "expected_rows": 5,
                "legacy_panel_sample_ceiling": 203,
                "expected_n_histogram": {"203": 5},
                "numerical_result_fields_used_for_design": False,
                "permitted_use": "fingerprint_and_structural_quarantine_only",
                "prohibited_uses": [
                    "trait_selection",
                    "hyperparameter_selection",
                    "model_comparison",
                    "preliminary_evidence",
                    "biological_claims",
                ],
            },
            {
                "id": "legacy_gblup_top15_203",
                "asset": _asset(top15),
                "expected_header": ["trait", "n", "h2", "gblup_r", "gblup_rho"],
                "expected_rows": 2,
                "legacy_panel_sample_ceiling": 203,
                "expected_n_histogram": {"203": 2},
                "numerical_result_fields_used_for_design": False,
                "permitted_use": "fingerprint_and_structural_quarantine_only",
                "prohibited_uses": [
                    "trait_selection",
                    "hyperparameter_selection",
                    "model_comparison",
                    "preliminary_evidence",
                    "biological_claims",
                ],
            },
        ],
        "qualification_history": {
            "prior_model_free_run_id": "synthetic-v1",
            "prior_config_sha256": "0" * 64,
            "prior_receipt": {
                "logical_path": "prior/receipt.json",
                "bytes": 1,
                "sha256": "1" * 64,
            },
            "prior_minimum_panel_samples": 168,
            "prior_eligible_traits": 0,
            "revised_threshold_basis": (
                "source_panel_overlap_histogram_only_no_model_results"
            ),
            "phenotype_magnitude_or_rank_used_for_revision": False,
        },
        "eligibility_contract": {
            "selection_scope": (
                "all_traits_passing_outcome_independent_availability_and_schema_rules"
            ),
            "required_species": "c_elegans",
            "minimum_panel_fraction": 0.7,
            "minimum_panel_samples": 147,
            "duplicate_trait_strain_rows_permitted": False,
            "nonfinite_values_permitted": False,
            "outcome_magnitude_or_rank_used": False,
            "top_n_selection_permitted": False,
            "constant_outcome_policy": (
                "retain_in_manifest_and_mark_non_estimable_only_at_model_execution"
            ),
        },
        "partition_contract": {
            "algorithm": "sha256_namespace_plus_trait_name_modulo_5",
            "namespace": "synthetic-partition:",
            "modulus": 5,
            "validation_buckets": [0],
            "discovery_buckets": [1, 2, 3, 4],
            "legacy_outcomes_used_for_partition": False,
            "validation_locked_until_model_and_metric_contract_frozen": True,
            "operator_prior_outcome_blinding_asserted": False,
        },
        "future_benchmark_contract": {
            "models_executed_in_this_stage": False,
            "trait_selection": "all_mechanically_eligible_traits_no_top_n",
            "discovery_validation_order": (
                "freeze_model_metric_and_missingness_contract_on_discovery_before_validation"
            ),
            "legacy_203_outputs_accepted_as_baselines": False,
            "claims": "retrospective_technical_benchmark_only",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, [phenotypes, fam, *legacy]


def _run(config: Path, paths: list[Path], output: Path) -> dict[str, object]:
    return qualify_compendium(config, *paths, output)


def test_qualification_freezes_all_traits_and_quarantines_legacy(tmp_path: Path) -> None:
    config, paths = _fixture(tmp_path)
    output = tmp_path / "qualified"
    receipt = _run(config, paths, output)

    assert receipt["models_executed"] is False
    assert receipt["biological_claims_permitted"] is False
    assert receipt["compendium"]["traits"] == 5
    assert receipt["compendium"]["eligible_traits"] == 5
    assert (
        receipt["compendium"]["discovery_traits"]
        + receipt["compendium"]["validation_traits"]
        == 5
    )
    assert receipt["genotype_panel"]["samples"] == 209
    assert all(
        item["accepted_as_preliminary_evidence"] is False
        for item in receipt["legacy_quarantine"]
    )
    assert (
        receipt["legacy_quarantine_boundary"][
            "validation_partition_is_historically_outcome_naive"
        ]
        is False
    )
    assert (output / "SUCCESS").read_text(encoding="utf-8") == "qualified\n"
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "LEGACY_QUARANTINE.json",
        "eligible_discovery_traits.txt",
        "eligible_validation_traits.txt",
        "source_qualification_receipt.json",
        "trait_manifest.tsv",
    }
    trait_rows = (output / "trait_manifest.tsv").read_text(encoding="utf-8").splitlines()
    assert len(trait_rows) == 6


def test_config_refuses_outcome_rank_selection(tmp_path: Path) -> None:
    config, paths = _fixture(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["eligibility_contract"]["top_n_selection_permitted"] = True
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="eligibility contract differs"):
        _run(config, paths, tmp_path / "qualified")


def test_legacy_trait_roster_must_match_source_census(tmp_path: Path) -> None:
    config, paths = _fixture(tmp_path)
    h2_greml = paths[3]
    rows = list(csv.reader(h2_greml.read_text(encoding="utf-8").splitlines()))
    rows[1][0] = "not_in_source"
    _write_csv(h2_greml, rows[0], rows[1:])
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["legacy_quarantine"][1]["asset"] = _asset(h2_greml)
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="h2 trait rosters differ"):
        _run(config, paths, tmp_path / "qualified")


def test_public_source_contract_is_path_neutral_and_freezes_eligibility() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = (
        repository / "experiments/natural_variation/phenotype_compendium_grouped_prediction/config/source_qualification.json"
    )
    payload = _load_config(config)

    assert payload["eligibility_contract"]["minimum_panel_samples"] == 147
    assert payload["eligibility_contract"]["top_n_selection_permitted"] is False
    serialized = json.dumps(payload)
    assert "/home/" not in serialized
    assert "\\Users\\" not in serialized
