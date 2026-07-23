from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from wormctx.poc import qtl_vcf_haplotype_sensitivity as lane


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments/natural_variation/regional_genotype_profile_sensitivity/config/regional_genotype_profiles.json"
)


def test_frozen_contract_is_explicitly_not_pangenome_or_pav() -> None:
    manifest = lane.load_manifest(CONFIG)
    assert manifest.schema_version == lane.SCHEMA_VERSION
    assert manifest.cohort.samples == 209
    assert [item.id for item in manifest.regions] == ["chrv_left", "chrv_right"]
    assert [item.id for item in manifest.relationships] == list(lane.RELATIONSHIPS)
    assert manifest.analysis.permutations_per_cell == 2000
    assert manifest.execution.restricted_residual_bootstrap_terminal_success_required_before_heavy_launch is True
    assert manifest.biological_claims_permitted is False
    prohibited = " ".join(manifest.claims.prohibited).lower()
    assert "pangenome" in prohibited
    assert "presence-absence" in prohibited


def test_contract_rejects_relabeled_pangenome_release() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["derivation"]["input_kind"] = "pangenome_paths"
    with pytest.raises(ValidationError):
        lane.Manifest.model_validate(payload)


def test_regional_state_derivation_is_deterministic_and_carrier_gated() -> None:
    iids = [f"strain_{index:03d}" for index in range(60)]
    patterns = np.asarray(
        [
            [0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2],
            [0, 0, 2, 2, 0, 0, 2, 2, 0, 2, 0, 2],
            [2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0],
        ],
        dtype=np.float64,
    )
    dosage = np.vstack([patterns[index // 20] for index in range(60)])
    dosage[0, 0] = np.nan
    first = lane.derive_region_states(
        "chrv_left", iids, dosage, candidate_k=(2, 3, 4), minimum_carriers=5
    )
    second = lane.derive_region_states(
        "chrv_left", iids, dosage, candidate_k=(2, 3, 4), minimum_carriers=5
    )
    assert first.state_ids == second.state_ids
    assert first.receipt == second.receipt
    assert first.receipt["selected_k"] == 3
    assert sorted(first.receipt["state_carrier_counts"].values()) == [20, 20, 20]
    assert first.receipt["missing_dosages_mean_imputed"] == 1


def test_regional_derivation_rejects_all_rare_cluster_solutions() -> None:
    iids = [f"strain_{index:03d}" for index in range(10)]
    dosage = np.eye(10, dtype=np.float64)
    with pytest.raises(ValueError, match="carrier gate"):
        lane.derive_region_states(
            "chrv_right", iids, dosage, candidate_k=(8,), minimum_carriers=5
        )


def test_profile_reml_returns_positive_covariance() -> None:
    generator = np.random.default_rng(41)
    n = 70
    latent = generator.normal(size=(n, 5))
    kernel = latent @ latent.T / latent.shape[1]
    scale = np.sqrt(np.diag(kernel))
    kernel = kernel / scale[:, None] / scale[None, :]
    values, vectors = np.linalg.eigh(kernel)
    design = np.column_stack([np.ones(n), np.linspace(-1.0, 1.0, n)])
    covariance = 0.55 * kernel + 0.45 * np.eye(n)
    y = design @ np.asarray([0.5, -0.25]) + np.linalg.cholesky(covariance) @ generator.normal(size=n)
    fitted = lane.profile_reml(y, design, values, vectors)
    assert 0.0 <= fitted.h2 <= 0.99
    assert fitted.sigma2 > 0.0
    assert fitted.variance_residual >= 0.0
    assert np.linalg.eigvalsh(fitted.covariance)[0] > 0.0


def test_restricted_minp_detects_predeclared_strong_regional_signal() -> None:
    n = 60
    left = ["left:a" if index < 30 else "left:b" for index in range(n)]
    right = [f"right:{index % 3}" for index in range(n)]
    design = np.ones((n, 1), dtype=np.float64)
    generator = np.random.default_rng(7)
    y = np.asarray([0.0 if state == "left:a" else 2.5 for state in left])
    y += generator.normal(scale=0.35, size=n)
    first = lane.restricted_minp_test(
        y,
        design,
        np.eye(n),
        {"chrv_left": left, "chrv_right": right},
        permutations=499,
        seed=1234,
    )
    second = lane.restricted_minp_test(
        y,
        design,
        np.eye(n),
        {"chrv_left": left, "chrv_right": right},
        permutations=499,
        seed=1234,
    )
    assert first == second
    assert first["restricted_residual_rank"] == 59
    assert first["regions"][0]["marginal_empirical_p"] <= 0.01
    assert first["westfall_young_familywise_p"] <= 0.02
    assert first["regions"][1]["degrees_of_freedom"] == 2
