from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from wormctx.poc import caendr_compendium_full_discovery as full


ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = ROOT / "experiments/natural_variation/phenotype_compendium_grouped_prediction/config/full_grouped_prediction.json"


def test_real_contract_scales_only_the_frozen_discovery_ladder() -> None:
    config = full.load_config(REAL_CONFIG)
    execution = config["execution_contract"]
    assert execution["traits"] == 1181
    assert execution["models"] == [
        "training_mean",
        "pc10_ridge_fixed",
        "whole_genome_gblup_fixed",
    ]
    assert execution["folds_per_trait"] == 10
    assert execution["permutations_per_trait"] == 4
    assert execution["maximum_threads"] == 1
    assert execution["expected_prediction_rows"] == 717006
    assert execution["expected_null_model_fits"] == 141720
    assert config["projection"]["conservative_wall_seconds_upper_bound"] == 7200
    assert config["resource_coexistence"][
        "coexistence_with_vcf_haplotype_permitted"
    ] is True
    assert config["validation_custody"]["runner_command_exists"] is False
    assert config["claim_boundary"]["permutation_p_values_permitted"] is False


def _synthetic_terminal_contract(run_root: Path, deployment: Path) -> dict[str, object]:
    return {
        "run_root": str(run_root.resolve()),
        "deployment_root": str(deployment.resolve()),
        "runner_exit_relative_path": "runner-exit-code.txt",
        "runner_exit_bytes": 2,
        "runner_exit_sha256": (
            "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"
        ),
        "success_relative_path": "SUCCESS",
        "success_bytes": 8,
        "success_sha256": (
            "1f513d4ecec4e91ddd48da1a59b6d96f1b76c374dc1da641980782a34f43b102"
        ),
        "failure_relative_path": "FAILURE",
        "ended_utc_relative_path": "receipts/ended_utc.txt",
        "summary_relative_path": "summary/restricted_residual_bootstrap_summary.json",
        "summary_schema_version": "wormctx-abamectin-ws283-restricted_residual_bootstrap-summary-1.0",
        "summary_mode": "full",
        "summary_completed_maps": 8,
        "summary_expected_maps": 8,
        "checkpoint_root_relative_path": "receipts/checkpoints",
        "checkpoint_schema_version": "wormctx-abamectin-ws283-restricted_residual_bootstrap-map-1.0",
        "checkpoint_manifest_sha256": "a" * 64,
        "checkpoint_traits": ["a", "b", "c", "d"],
        "checkpoint_replicates_per_trait": 2,
        "checksum_relative_path": "receipts/SHA256SUMS.txt",
        "checksum_verification_log_relative_path": "logs/checksums.verify.log",
        "scratch_relative_path": "scratch",
        "active_process_pattern": "synthetic-restricted_residual_bootstrap",
        "must_be_terminal_and_process_clear_before_launch": True,
        "wait_poll_seconds": 60,
        "wait_maximum_polls": 1440,
    }


def _terminal_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    run_root = tmp_path / "restricted_residual_bootstrap"
    deployment = tmp_path / "deployment"
    for relative in (
        "receipts/checkpoints",
        "summary",
        "logs",
        "scratch",
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    deployment.mkdir()
    (run_root / "SUCCESS").write_bytes(b"SUCCESS\n")
    (deployment / "runner-exit-code.txt").write_bytes(b"0\n")
    (run_root / "receipts/ended_utc.txt").write_text(
        "2026-07-23T12:34:56Z\n", encoding="utf-8"
    )
    contract = _synthetic_terminal_contract(run_root, deployment)
    (run_root / "summary/restricted_residual_bootstrap_summary.json").write_text(
        json.dumps(
            {
                "schema_version": contract["summary_schema_version"],
                "manifest_sha256": contract["checkpoint_manifest_sha256"],
                "mode": "full",
                "completed_maps": 8,
                "expected_maps": 8,
                "qualified": True,
            }
        ),
        encoding="utf-8",
    )
    for trait in contract["checkpoint_traits"]:
        root = run_root / "receipts/checkpoints" / trait
        root.mkdir()
        for replicate in (1, 2):
            (root / f"rep-{replicate:04d}.json").write_text(
                json.dumps(
                    {
                        "schema_version": contract["checkpoint_schema_version"],
                        "manifest_sha256": contract[
                            "checkpoint_manifest_sha256"
                        ],
                        "trait_slug": trait,
                        "replicate": replicate,
                        "qualified": True,
                    }
                ),
                encoding="utf-8",
            )
    (run_root / "receipts/SHA256SUMS.txt").write_text(
        f"{'0' * 64}  placeholder\n", encoding="utf-8"
    )
    (run_root / "logs/checksums.verify.log").write_text(
        "synthetic checksum verification\n", encoding="utf-8"
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    return config_path, run_root, deployment, contract


def test_exhaustive_terminal_gate_requires_every_checkpoint_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, run_root, deployment, contract = _terminal_fixture(tmp_path)
    monkeypatch.setattr(
        full,
        "load_config",
        lambda _path: {"restricted_residual_bootstrap_terminal_prerequisite": contract},
    )

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "sha256sum":
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[0] == "ps":
            return subprocess.CompletedProcess(command, 0, "  1 init\n", "")
        raise AssertionError(command)

    monkeypatch.setattr(full.subprocess, "run", fake_run)
    output = tmp_path / "terminal-audit.json"
    receipt = full.verify_restricted_residual_bootstrap_terminal(
        config_path, run_root, deployment, output
    )
    assert receipt["qualified"] is True
    assert receipt["terminal_closed"] is True
    assert receipt["checkpoints"]["count"] == 8
    assert receipt["scratch_entries"] == 0
    assert output.is_file()

    missing = run_root / "receipts/checkpoints/a/rep-0002.json"
    missing.unlink()
    with pytest.raises(ValueError, match="checkpoint count differs"):
        full.verify_restricted_residual_bootstrap_terminal(
            config_path, run_root, deployment, tmp_path / "second-audit.json"
        )


@pytest.mark.skip(
    reason="the site-specific queue wrapper is intentionally absent from the public tree"
)
def test_runner_is_one_thread_fail_closed_and_has_no_validation_mode() -> None:
    runner = (
        ROOT / "scripts/queue_caendr_compendium_full_discovery.sh"
    ).read_text(encoding="utf-8")
    module = (
        ROOT / "src/wormctx/poc/caendr_compendium_full_discovery.py"
    ).read_text(encoding="utf-8")
    assert "OMP_NUM_THREADS=1" in runner
    assert "OPENBLAS_NUM_THREADS=1" in runner
    assert "MKL_NUM_THREADS=1" in runner
    assert "timeout 7200" in runner
    assert "verify-terminal" in runner
    assert '["sha256sum", "--check", "--quiet"' in module
    assert "matching_active_compute_processes" in module
    assert "validation_runner_command_exists" in module
    assert "add_parser(\"validation\")" not in module
    validation_branch = module.index("if trait in validation_set:")
    outcome_parse = module.index("value = float(fields[4])", validation_branch)
    assert validation_branch < outcome_parse


def test_smoke_continuation_rejects_any_numeric_drift(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    (smoke / "discovery_metrics.tsv").write_text(
        "trait\tmodel\tn\trmse\tmae\tgroup_macro_rmse\t"
        "rmse_over_outcome_sd\tpearson\tspearman\n"
        "t\ttraining_mean\t10\t1.0\t0.8\t1.1\t1.0\t\t\n",
        encoding="utf-8",
    )
    (smoke / "training_permutation_null_metrics.tsv").write_text(
        "trait\tmodel\treplicate\trmse\tgroup_macro_rmse\tpearson\tspearman\n"
        "t\ttraining_mean\t1\t1.2\t1.3\t\t\n",
        encoding="utf-8",
    )
    metrics = [["t", "training_mean", 10, 1.0, 0.8, 1.1, 1.0, "", ""]]
    null = [["t", "training_mean", 1, 1.2, 1.3, "", ""]]
    receipt = full._verify_smoke_continuation(smoke, {"t"}, metrics, null)
    assert receipt["exact_string_reproduction"] is True
    metrics[0][3] = 1.0001
    with pytest.raises(ValueError, match="does not exactly reproduce"):
        full._verify_smoke_continuation(smoke, {"t"}, metrics, null)


def test_smoke_continuation_is_order_independent_but_rejects_duplicates(
    tmp_path: Path,
) -> None:
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    header = (
        "trait\tmodel\tn\trmse\tmae\tgroup_macro_rmse\t"
        "rmse_over_outcome_sd\tpearson\tspearman\n"
    )
    (smoke / "discovery_metrics.tsv").write_text(
        header
        + "t2\ttraining_mean\t10\t2.0\t1.8\t2.1\t1.0\t\t\n"
        + "t1\ttraining_mean\t10\t1.0\t0.8\t1.1\t1.0\t\t\n",
        encoding="utf-8",
    )
    (smoke / "training_permutation_null_metrics.tsv").write_text(
        "trait\tmodel\treplicate\trmse\tgroup_macro_rmse\tpearson\tspearman\n"
        "t2\ttraining_mean\t1\t2.2\t2.3\t\t\n"
        "t1\ttraining_mean\t1\t1.2\t1.3\t\t\n",
        encoding="utf-8",
    )
    metrics = [
        ["t1", "training_mean", 10, 1.0, 0.8, 1.1, 1.0, "", ""],
        ["t2", "training_mean", 10, 2.0, 1.8, 2.1, 1.0, "", ""],
    ]
    null = [
        ["t1", "training_mean", 1, 1.2, 1.3, "", ""],
        ["t2", "training_mean", 1, 2.2, 2.3, "", ""],
    ]
    receipt = full._verify_smoke_continuation(
        smoke, {"t1", "t2"}, metrics, null
    )
    assert receipt["exact_string_reproduction"] is True

    with pytest.raises(ValueError, match="does not exactly reproduce"):
        full._verify_smoke_continuation(
            smoke, {"t1", "t2"}, [*metrics, metrics[0]], null
        )
