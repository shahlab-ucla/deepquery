from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "experiments"
    / "natural_variation"
    / "historical_marker_identity"
    / "config"
    / "observed_marker_sensitivity.json"
)
A1_RUNNER = ROOT / "scripts" / "run_abamectin_qtl_ws276_reconstruction.sh"
A2_RUNNER = ROOT / "scripts" / "run_abamectin_qtl_ws276_observed_sensitivity.sh"
A2_R = ROOT / "scripts" / "rrblup_ws276_observed_sensitivity.R"
A2_DEPLOY = ROOT / "scripts" / "deploy_ws276_observed_sensitivity.ps1"


@pytest.mark.skip(
    reason="the site-specific historical runner is intentionally absent"
)
def test_a1_historical_marker_gate_remains_fail_closed() -> None:
    script = A1_RUNNER.read_text(encoding="utf-8")

    assert 'PRUNED_MARKERS=$(wc -l < "$RUN_ROOT/genotype/markers.txt")' in script
    assert '"$COMPLETE_MARKERS" -ne 21342' in script
    assert "complete-case marker set is not exactly the historical 21,342 rows" in script
    assert "ALLOW_MARKER" not in script
    assert "75852" not in script


def test_a2_contract_predeclares_marker_universe_thresholds_and_claim_boundary() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    within = -math.log10(0.05 / 75_852)
    family = -math.log10(0.05 / (4 * 75_852))

    assert contract["classification"] == (
        "exploratory_current_ws276_isotype_marker_universe_sensitivity"
    )
    assert contract["strict_reproduction_claim_permitted"] is False
    assert contract["biological_claims_permitted"] is False
    assert contract["expected_current_markers"] == 75_852
    assert contract["expected_complete_markers"] == 75_831
    assert contract["expected_incomplete_markers"] == 21
    assert contract["analysis_marker_list_sha256"] == (
        "181d3dfe2db799b544dc1b17098095f591cbee3f77ef0c4a604211227a1cebe2"
    )
    assert contract["marker_list_sha256"] == (
        "1f0a0986ecd10f6f89dd1beb88fa240d506cfa870091c56436c95c0973a75675"
    )
    assert contract["expected_shared_coordinates"] == 12_601
    assert contract["expected_published_only"] == 8_345
    assert contract["expected_current_only"] == 63_251
    assert math.isclose(
        contract["association"]["within_trait_bonferroni_neg_log10_p"],
        within,
        abs_tol=1e-14,
    )
    assert math.isclose(
        contract["association"]["four_trait_familywise_neg_log10_p"],
        family,
        abs_tol=1e-14,
    )
    assert contract["association"]["p3d_true_can_originate_primary_hit"] is False
    assert "reproduction" in contract["claims"]["prohibited"]
    assert "independent validation" in contract["claims"]["prohibited"]


@pytest.mark.skip(
    reason="the site-specific sensitivity runner is intentionally absent"
)
def test_a2_runner_is_derived_from_exact_parent_and_never_retunes_markers() -> None:
    script = A2_RUNNER.read_text(encoding="utf-8")

    assert "EXPECTED_PARENT_ID=abamectin-ws276-reconstruction-20260718T133456Z" in script
    assert "grep -qx $'exit_code\\t70'" in script
    assert "46409f4e0c5025cc8b4181e4d1d7a55d588b80f1" in script
    assert "1f0a0986ecd10f6f89dd1beb88fa240d506cfa870091c56436c95c0973a75675" in script
    assert '"$(wc -l < "$PARENT_MARKERS")" -ne 75852' in script
    assert '"$COMPLETE_MARKERS" -ne 75831' in script
    assert '"$INCOMPLETE_MARKERS" -ne 21' in script
    assert "181d3dfe2db799b544dc1b17098095f591cbee3f77ef0c4a604211227a1cebe2" in script
    assert "6a006610325359ebeb2e7910eb98b58ebbb5dbe8650564bdc70785add8000cb1" in script
    assert 'cmp -s "$GENOTYPE_MATRIX" "$PARENT_GENOTYPE_MATRIX"' in script
    assert "--analysis-marker-list" in script
    assert "--indep-pairwise" not in script
    assert "--maf" not in script
    assert "prune.in" not in script
    assert "cp \"$PARENT_MARKERS\"" in script
    assert "cp \"$PARENT_VCF\"" in script
    assert script.index('"$RUN_ROOT/association/primary" FALSE') < script.index(
        '"$RUN_ROOT/association/p3d_true" TRUE'
    )
    assert "wormctx.poc.qtl_ws276_sensitivity" in script


def test_a2_r_runner_binds_kinship_to_matrix_samples_and_markers() -> None:
    script = A2_R.read_text(encoding="utf-8")

    assert "nrow(genotype) != 75852L" in script
    assert "nrow(analysis_genotype) != 75831L" in script
    assert "expected_matrix_sha256" in script
    assert "genotype_matrix_sha256" in script
    assert "wormctx-ws276-observed-sensitivity-kinship-1.0" in script
    assert "kinship_payload$sample_names" in script
    assert "kinship_payload$marker_ids" in script
    assert "existing kinship matrix was built from a different genotype matrix" in script
    assert 'P3D = p3d' in script
    assert '"biological_claims_permitted"' in script


@pytest.mark.skip(
    reason="the site-specific sensitivity deployer is intentionally absent"
)
def test_a2_deployer_is_immutable_hash_verified_and_non_destructive() -> None:
    script = A2_DEPLOY.read_text(encoding="utf-8")

    assert "status --porcelain=v1 --untracked-files=all" in script
    assert "archive --format=zip" in script
    assert "test ! -e '$Deployment'" in script
    assert "test ! -e '$RunRoot'" in script
    assert "sha256sum --check" in script
    assert "run_abamectin_qtl_ws276_observed_sensitivity.sh" in script
    assert "bash -n '$RemoteSourceRoot/scripts/run_abamectin" in script
    assert "ws276-sensitivity-results.tar.gz.sha256" in script
    assert "runner_receipt=`$(cat '$Deployment/runner-exit-code.txt')" in script
    assert "deployer-runner-exit-code.txt" in script
    assert "Retrieved result archive contains an unsafe or unexpected path" in script
    assert "Retrieved result archive contains a link or special entry" in script
    assert "FAILURE exit code conflicts with the deployer runner receipt" in script
    assert "Remove-Item" not in script
    assert "rm -" not in script
    assert "git reset" not in script
    assert "git checkout" not in script
