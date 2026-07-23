from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from wormctx.poc import qtl_restricted_residual_bootstrap as restricted_residual_bootstrap


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/natural_variation/restricted_residual_bootstrap_calibration/config/restricted_residual_bootstrap.json"


def test_manifest_freezes_four_pc10_traits_and_explicit_multiplicity() -> None:
    manifest = restricted_residual_bootstrap.load_manifest(CONFIG)
    assert [item.trait for item in manifest.traits] == list(restricted_residual_bootstrap.TRAITS)
    assert manifest.execution.sample_count == 209
    assert manifest.execution.fixed_effect_count == 11
    assert manifest.execution.restricted_residual_rank == 198
    assert manifest.bootstrap.replicates_per_trait == 2000
    assert manifest.execution.full_map_count == 8000
    assert manifest.multiplicity.trait_alpha == 0.0125
    assert manifest.multiplicity.critical_rank == 25
    assert manifest.multiplicity.pooled_trait_null_permitted is False
    assert manifest.multiplicity.cross_trait_exchangeability_assumed is False

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["traits"].pop()
    with pytest.raises(ValidationError, match="four PC10 traits"):
        restricted_residual_bootstrap.Manifest.model_validate(payload)


def test_manifest_makes_naive_permutations_unreachable() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["bootstrap"]["phenotype_label_permutation_permitted"] = True
    with pytest.raises(ValidationError):
        restricted_residual_bootstrap.Manifest.model_validate(payload)

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["bootstrap"]["genotype_label_permutation_permitted"] = True
    with pytest.raises(ValidationError):
        restricted_residual_bootstrap.Manifest.model_validate(payload)


def test_restricted_transform_reconstructs_and_has_n_minus_p_rank() -> None:
    rng = np.random.default_rng(8123)
    n = 15
    design = np.column_stack((np.ones(n), np.linspace(-1.0, 1.0, n)))
    raw = rng.normal(size=(n, n))
    covariance = raw @ raw.T + np.eye(n) * 0.75
    y = design @ np.array([2.0, -0.5]) + rng.normal(size=n)

    arrays, diagnostics = restricted_residual_bootstrap.restricted_transform_arrays(y, design, covariance)
    assert diagnostics["sample_count"] == n
    assert diagnostics["fixed_effect_count"] == 2
    assert diagnostics["restricted_residual_rank"] == n - 2
    assert diagnostics["reconstruction_max_abs_error"] < 1e-8
    assert diagnostics["basis_orthonormal_max_abs_error"] < 1e-10
    np.testing.assert_allclose(
        arrays["fixed"] + arrays["recolor"] @ arrays["xi"], y, rtol=0.0, atol=1e-8
    )


def test_permutation_is_deterministic_and_preserves_restricted_multiset() -> None:
    rng = np.random.default_rng(92)
    n = 14
    design = np.ones((n, 1))
    raw = rng.normal(size=(n, n))
    covariance = raw @ raw.T + np.eye(n)
    arrays, _ = restricted_residual_bootstrap.restricted_transform_arrays(rng.normal(size=n), design, covariance)
    seed = restricted_residual_bootstrap.derive_permutation_seed("a" * 64, "mean_EXT", 1)
    first, first_permutation = restricted_residual_bootstrap.permuted_phenotype_values(
        arrays["fixed"], arrays["recolor"], arrays["xi"], seed
    )
    second, second_permutation = restricted_residual_bootstrap.permuted_phenotype_values(
        arrays["fixed"], arrays["recolor"], arrays["xi"], seed
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_permutation, second_permutation)
    np.testing.assert_array_equal(np.sort(first_permutation), np.arange(n - 1))
    np.testing.assert_array_equal(
        np.sort(arrays["xi"][first_permutation]), np.sort(arrays["xi"])
    )
    assert restricted_residual_bootstrap.derive_permutation_seed("a" * 64, "mean_EXT", 2) != seed
    assert restricted_residual_bootstrap.derive_permutation_seed("a" * 64, "norm_n", 1) != seed


def test_gcta_gate_accepts_only_exact_pc10_mlma_loco(tmp_path: Path) -> None:
    manifest = restricted_residual_bootstrap.load_manifest(CONFIG)
    bfile = (tmp_path / "baseline/genotype/abamectin_209_qc").resolve()
    phenotype = (tmp_path / "rep-0001.phen").resolve()
    qcovar = (tmp_path / "calibration/genotype/qcovars/pc10.qcovar").resolve()
    output_prefix = (tmp_path / "scratch/map").resolve()
    log = tmp_path / "map.log"
    accepted = [
        "--mlma-loco",
        f"--bfile {bfile}",
        f"--pheno {phenotype}",
        f"--qcovar {qcovar}",
        "--maf 0.05",
        "--autosome-num 6",
        "--thread-num 8",
        f"--out {output_prefix}",
    ]
    log.write_text(
        "\n".join(
            [
                "Accepted options:",
                *accepted,
                "",
                "209 individuals are in common in these files.",
                "10 quantitative covariate(s) of 209 individuals are included.",
                "Log-likelihood ratio converged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    receipt = restricted_residual_bootstrap._qualify_gcta_log(
        log, bfile, phenotype, qcovar, output_prefix, manifest
    )
    assert receipt["accepted_options"]["--mlma-loco"] is None

    log.write_text(log.read_text(encoding="utf-8").replace("--mlma-loco", "--mlma"), encoding="utf-8")
    with pytest.raises(ValueError, match="options"):
        restricted_residual_bootstrap._qualify_gcta_log(log, bfile, phenotype, qcovar, output_prefix, manifest)


def test_contract_summary_keeps_full_launch_blocked_before_smoke() -> None:
    summary = restricted_residual_bootstrap.contract_summary(CONFIG)
    assert summary["qualified_for_remote_smoke"] is True
    assert summary["full_launch_eligible"] is False
    assert summary["maps"] == 8000
    assert len(summary["full_launch_blockers"]) == 3


def test_aggregate_keeps_nominal_and_four_trait_marker_thresholds_distinct() -> None:
    source = (ROOT / "src/wormctx/poc/qtl_restricted_residual_bootstrap.py").read_text(
        encoding="utf-8"
    )
    assert "nominal_marker_bonferroni_alpha_005" in source
    assert "four_trait_adjusted_marker_bonferroni_alpha_00125" in source
    assert "manifest.multiplicity.overall_alpha / manifest.execution.marker_count" in source
    assert "manifest.multiplicity.trait_alpha / manifest.execution.marker_count" in source


def test_assets_bind_matched_relationship_model_result_and_all_four_fit_receipts() -> None:
    manifest = restricted_residual_bootstrap.load_manifest(CONFIG)
    roles = {item.role for item in manifest.assets}
    assert roles == restricted_residual_bootstrap.EXPECTED_ROLES
    assert "matched_relationship_model_result" in roles
    for slug in restricted_residual_bootstrap.SLUG_TRAITS:
        assert f"fit_{slug}" in roles
        assert f"hsq_{slug}" in roles


@pytest.mark.skip(
    reason="legacy site-specific deployment assertions are superseded by the portable launcher"
)
def test_runner_and_deployer_freeze_smoke_before_full_and_checkpoint_cleanup() -> None:
    runner = (
        ROOT / "scripts/run_abamectin_ws283_restricted_residual_bootstrap_covariance_bootstrap.sh"
    ).read_text(encoding="utf-8")
    deployer = (
        ROOT / "scripts/deploy_abamectin_ws283_restricted_residual_bootstrap_covariance_bootstrap.ps1"
    ).read_text(encoding="utf-8")
    assert "--mlma-loco" in runner
    assert "--qcovar \"$QCOVAR\"" in runner
    assert "--thread-num \"$THREADS_PER_WORKER\"" in runner
    assert 'run_parallel_phase 1 2' in runner
    assert 'run_parallel_phase 3 2000' in runner
    assert 'if [[ "$MODE" == full ]]' in runner
    assert 'if [[ ! -s "$checkpoint" ]]' in runner
    assert 'remove_scratch "$scratch"' in runner
    assert "SMOKE_SUCCESS" in runner
    assert "PROJECTED_SECONDS" in runner and "172800" in runner
    assert '[ValidateSet("smoke", "full")]' in deployer
    assert 'if ($Mode -eq "full")' in deployer
    assert "FullQualificationReceipt" in deployer
    assert "full_launch_eligible" in deployer
