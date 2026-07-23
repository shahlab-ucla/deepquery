"""Run the frozen full CaeNDR discovery ladder after exhaustive restricted-residual bootstrap closure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import caendr_compendium_discovery_gate as gate
except ImportError:  # pragma: no cover - supports direct deployed-script execution
    import caendr_compendium_discovery_gate as gate


CONFIG_VERSION = "wormctx-caendr-compendium-full-discovery-config-1.0"
TERMINAL_AUDIT_VERSION = "wormctx-restricted_residual_bootstrap-exhaustive-terminal-audit-1.0"
RESULT_VERSION = "wormctx-caendr-compendium-full-discovery-result-1.0"
UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\n$")


def _require_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _asset_spec(payload: Mapping[str, Any], label: str) -> None:
    _require_keys(payload, {"role", "logical_path", "bytes", "sha256"}, label)
    if (
        not isinstance(payload["role"], str)
        or not payload["role"]
        or not isinstance(payload["logical_path"], str)
        or not payload["logical_path"]
        or not isinstance(payload["bytes"], int)
        or payload["bytes"] <= 0
        or not _valid_sha(payload["sha256"])
    ):
        raise ValueError(f"{label} is malformed")


def load_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require_keys(
        payload,
        {
            "schema_version",
            "analysis_id",
            "classification",
            "base_gate_config",
            "qualification_assets",
            "smoke_assets",
            "execution_contract",
            "projection",
            "restricted_residual_bootstrap_terminal_prerequisite",
            "resource_coexistence",
            "validation_custody",
            "claim_boundary",
        },
        "full-discovery config",
    )
    if (
        payload["schema_version"] != CONFIG_VERSION
        or payload["classification"]
        != "retrospective_discovery_only_full_technical_benchmark"
    ):
        raise ValueError("full-discovery config identity differs")
    _require_keys(
        payload["base_gate_config"],
        {"logical_path", "bytes", "sha256"},
        "base gate config",
    )
    if (
        payload["base_gate_config"]["bytes"] <= 0
        or not _valid_sha(payload["base_gate_config"]["sha256"])
    ):
        raise ValueError("base gate config asset is malformed")
    qualification_roles = [
        "discovery_qualification_receipt",
        "qualified_discovery_traits",
        "trait_mask_manifest",
        "frozen_model_contract",
        "qualification_success",
    ]
    smoke_roles = [
        "discovery_smoke_receipt",
        "discovery_smoke_metrics",
        "discovery_smoke_null_metrics",
        "smoke_success",
    ]
    if [item.get("role") for item in payload["qualification_assets"]] != qualification_roles:
        raise ValueError("qualification asset roles or order differ")
    if [item.get("role") for item in payload["smoke_assets"]] != smoke_roles:
        raise ValueError("smoke asset roles or order differ")
    for item in payload["qualification_assets"] + payload["smoke_assets"]:
        _asset_spec(item, f"asset {item.get('role')}")

    execution = payload["execution_contract"]
    _require_keys(
        execution,
        {
            "traits",
            "qualified_trait_sha256",
            "finite_trait_sample_pairs",
            "models",
            "fold_method",
            "folds_per_trait",
            "permutations_per_trait",
            "permutation_scope",
            "hyperparameter_tuning_permitted",
            "maximum_threads",
            "maximum_wall_seconds",
            "expected_prediction_rows",
            "expected_metric_rows",
            "expected_null_metric_rows",
            "expected_null_model_fits",
        },
        "execution contract",
    )
    if (
        execution["traits"] != 1181
        or execution["qualified_trait_sha256"]
        != "577895ff199c4716b1a4e0b451a617b2c2775be5edc6c4828657ccb98c004ce9"
        or execution["finite_trait_sample_pairs"] != 239002
        or execution["models"] != gate.MODEL_IDS
        or execution["fold_method"] != "leave_one_genotype_pc_group_out"
        or execution["folds_per_trait"] != 10
        or execution["permutations_per_trait"] != 4
        or execution["permutation_scope"] != "outer_training_outcomes_only"
        or execution["hyperparameter_tuning_permitted"] is not False
        or execution["maximum_threads"] != 1
        or execution["maximum_wall_seconds"] != 7200
        or execution["expected_prediction_rows"] != 717006
        or execution["expected_metric_rows"] != 3543
        or execution["expected_null_metric_rows"] != 14172
        or execution["expected_null_model_fits"] != 141720
    ):
        raise ValueError("full-discovery execution contract differs")

    projection = payload["projection"]
    if (
        projection.get("source_smoke_traits") != 8
        or projection.get("projection_supports_local_execution_within_few_days")
        is not True
        or projection.get("conservative_wall_seconds_upper_bound") != 7200
        or projection.get("conservative_output_bytes_upper_bound", 0) > 1073741824
        or projection.get("minimum_free_bytes_at_launch") != 10737418240
    ):
        raise ValueError("full-discovery projection differs or is unsafe")

    prerequisite = payload["restricted_residual_bootstrap_terminal_prerequisite"]
    _require_keys(
        prerequisite,
        {
            "run_root",
            "deployment_root",
            "runner_exit_relative_path",
            "runner_exit_bytes",
            "runner_exit_sha256",
            "success_relative_path",
            "success_bytes",
            "success_sha256",
            "failure_relative_path",
            "ended_utc_relative_path",
            "summary_relative_path",
            "summary_schema_version",
            "summary_mode",
            "summary_completed_maps",
            "summary_expected_maps",
            "checkpoint_root_relative_path",
            "checkpoint_schema_version",
            "checkpoint_manifest_sha256",
            "checkpoint_traits",
            "checkpoint_replicates_per_trait",
            "checksum_relative_path",
            "checksum_verification_log_relative_path",
            "scratch_relative_path",
            "active_process_pattern",
            "must_be_terminal_and_process_clear_before_launch",
            "wait_poll_seconds",
            "wait_maximum_polls",
        },
        "restricted-residual bootstrap terminal prerequisite",
    )
    if (
        prerequisite["runner_exit_bytes"] != 2
        or prerequisite["runner_exit_sha256"]
        != "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"
        or prerequisite["success_bytes"] != 8
        or prerequisite["success_sha256"]
        != "1f513d4ecec4e91ddd48da1a59b6d96f1b76c374dc1da641980782a34f43b102"
        or prerequisite["summary_schema_version"]
        != "wormctx-abamectin-ws283-restricted_residual_bootstrap-summary-1.0"
        or prerequisite["summary_mode"] != "full"
        or prerequisite["summary_completed_maps"] != 8000
        or prerequisite["summary_expected_maps"] != 8000
        or prerequisite["checkpoint_schema_version"]
        != "wormctx-abamectin-ws283-restricted_residual_bootstrap-map-1.0"
        or prerequisite["checkpoint_manifest_sha256"]
        != "7f3a8edf6a83a158887880fa6c8bf5d67fb2610db08c6a7df136cdc8267e7411"
        or prerequisite["checkpoint_traits"]
        != ["mean_EXT", "mean_TOF", "mean_norm_EXT", "norm_n"]
        or prerequisite["checkpoint_replicates_per_trait"] != 2000
        or prerequisite["must_be_terminal_and_process_clear_before_launch"] is not True
        or prerequisite["wait_poll_seconds"] != 60
        or prerequisite["wait_maximum_polls"] != 1440
    ):
        raise ValueError("restricted-residual bootstrap terminal prerequisite differs")

    coexistence = payload["resource_coexistence"]
    if (
        coexistence.get("full_discovery_maximum_threads") != 1
        or coexistence.get("vcf_haplotype_queue_maximum_threads") != 8
        or coexistence.get("host_online_cpus_observed") != 32
        or coexistence.get("minimum_available_memory_kib") != 33554432
        or coexistence.get("minimum_free_bytes") != 10737418240
        or coexistence.get("maximum_load_fraction") != 0.75
        or coexistence.get("coexistence_with_vcf_haplotype_permitted") is not True
        or coexistence.get("simultaneous_compendium_jobs_permitted") is not False
    ):
        raise ValueError("resource-coexistence contract differs")
    if payload["validation_custody"] != {
        "locked_traits": 261,
        "validation_trait_sha256": (
            "110d49079a54b2ca13cb413eed5d74508be9532282521a3a55ffdde532d75546"
        ),
        "outcome_value_field_evaluated": False,
        "predictions_generated": False,
        "metrics_generated": False,
        "runner_command_exists": False,
    }:
        raise ValueError("validation custody differs")
    if payload["claim_boundary"] != {
        "discovery_model_development_metrics_permitted": True,
        "confirmatory_performance_claims_permitted": False,
        "biological_claims_permitted": False,
        "independent_validation_established": False,
        "permutation_p_values_permitted": False,
        "legacy_numeric_results_accessed": False,
    }:
        raise ValueError("claim boundary differs")
    return payload


def _verify_asset(path: Path, spec: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is absent: {path}")
    observed = {"bytes": path.stat().st_size, "sha256": gate._sha256(path)}
    if observed["bytes"] != spec["bytes"] or observed["sha256"] != spec["sha256"]:
        raise ValueError(f"{label} byte count or SHA-256 differs")
    return observed


def _asset_by_role(config: Mapping[str, Any], collection: str, role: str) -> Mapping[str, Any]:
    return next(item for item in config[collection] if item["role"] == role)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(gate._canonical_json(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_utc_marker(path: Path) -> str:
    value = path.read_text(encoding="utf-8")
    if not UTC_PATTERN.fullmatch(value):
        raise ValueError(f"UTC marker is malformed: {path}")
    datetime.strptime(value.strip(), "%Y-%m-%dT%H:%M:%SZ")
    return value.strip()


def verify_restricted_residual_bootstrap_terminal(
    config_path: str | Path,
    restricted_residual_bootstrap_root: str | Path,
    restricted_residual_bootstrap_deployment: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Exhaustively verify restricted-residual bootstrap terminal state before permitting discovery compute."""

    config_source = Path(config_path).resolve()
    config = load_config(config_source)
    contract = config["restricted_residual_bootstrap_terminal_prerequisite"]
    run_root = Path(restricted_residual_bootstrap_root).resolve()
    deployment_root = Path(restricted_residual_bootstrap_deployment).resolve()
    if str(run_root) != contract["run_root"]:
        raise ValueError("restricted-residual bootstrap run root differs from the frozen prerequisite")
    if str(deployment_root) != contract["deployment_root"]:
        raise ValueError("restricted-residual bootstrap deployment root differs from the frozen prerequisite")

    success = run_root / contract["success_relative_path"]
    runner_exit = deployment_root / contract["runner_exit_relative_path"]
    failure = run_root / contract["failure_relative_path"]
    for path, bytes_expected, sha_expected, label in (
        (
            success,
            contract["success_bytes"],
            contract["success_sha256"],
            "restricted-residual bootstrap SUCCESS",
        ),
        (
            runner_exit,
            contract["runner_exit_bytes"],
            contract["runner_exit_sha256"],
            "restricted-residual bootstrap runner exit",
        ),
    ):
        if (
            not path.is_file()
            or path.stat().st_size != bytes_expected
            or gate._sha256(path) != sha_expected
        ):
            raise ValueError(f"{label} does not have the frozen terminal identity")
    if failure.exists():
        raise ValueError("restricted-residual bootstrap FAILURE marker exists")
    ended_utc = _parse_utc_marker(run_root / contract["ended_utc_relative_path"])

    summary_path = run_root / contract["summary_relative_path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema_version") != contract["summary_schema_version"]
        or summary.get("manifest_sha256")
        != contract["checkpoint_manifest_sha256"]
        or summary.get("mode") != contract["summary_mode"]
        or summary.get("completed_maps") != contract["summary_completed_maps"]
        or summary.get("expected_maps") != contract["summary_expected_maps"]
        or summary.get("qualified") is not True
    ):
        raise ValueError("restricted-residual bootstrap summary is not the frozen complete qualified result")

    checkpoint_root = run_root / contract["checkpoint_root_relative_path"]
    observed_paths = sorted(
        path.relative_to(checkpoint_root).as_posix()
        for path in checkpoint_root.rglob("*.json")
        if path.is_file()
    )
    expected_count = (
        len(contract["checkpoint_traits"])
        * contract["checkpoint_replicates_per_trait"]
    )
    if len(observed_paths) != expected_count:
        raise ValueError(
            f"restricted-residual bootstrap checkpoint count differs: {len(observed_paths)} != {expected_count}"
        )
    path_cursor = 0
    tuple_digest = hashlib.sha256()
    trait_counts: Counter[str] = Counter()
    for trait in contract["checkpoint_traits"]:
        for replicate in range(1, contract["checkpoint_replicates_per_trait"] + 1):
            relative = f"{trait}/rep-{replicate:04d}.json"
            if observed_paths[path_cursor] != relative:
                raise ValueError(f"restricted-residual bootstrap checkpoint path set differs at {relative}")
            path_cursor += 1
            checkpoint = checkpoint_root / relative
            item = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (
                item.get("schema_version") != contract["checkpoint_schema_version"]
                or item.get("manifest_sha256")
                != contract["checkpoint_manifest_sha256"]
                or item.get("trait_slug") != trait
                or item.get("replicate") != replicate
                or item.get("qualified") is not True
            ):
                raise ValueError(f"invalid restricted-residual bootstrap checkpoint tuple: {trait}/{replicate}")
            digest = gate._sha256(checkpoint)
            tuple_digest.update(f"{relative}\t{digest}\n".encode("utf-8"))
            trait_counts[trait] += 1

    scratch = run_root / contract["scratch_relative_path"]
    if not scratch.is_dir() or next(scratch.rglob("*"), None) is not None:
        raise ValueError("restricted-residual bootstrap scratch is absent or nonempty")
    checksum = run_root / contract["checksum_relative_path"]
    checksum_log = run_root / contract["checksum_verification_log_relative_path"]
    if not checksum.is_file() or not checksum_log.is_file() or checksum_log.stat().st_size == 0:
        raise ValueError("restricted-residual bootstrap checksum receipt or verification log is absent")
    verification = subprocess.run(
        ["sha256sum", "--check", "--quiet", str(checksum)],
        cwd=run_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if verification.returncode != 0:
        raise ValueError(f"restricted-residual bootstrap run checksum verification failed: {verification.stderr}")

    process_result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    blockers = []
    for line in process_result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if pid_text == str(os.getpid()):
            continue
        if str(run_root) in command and any(
            token in command
            for token in (
                "gcta-1.94.1",
                "plink2",
                "qtl_restricted_residual_bootstrap",
                "run_abamectin_ws283_restricted_residual_bootstrap_covariance_bootstrap",
            )
        ):
            blockers.append(stripped)
    if blockers:
        raise ValueError(f"restricted-residual bootstrap compute process remains active: {blockers}")

    receipt = {
        "schema_version": TERMINAL_AUDIT_VERSION,
        "qualified": True,
        "terminal_closed": True,
        "config": {
            "bytes": config_source.stat().st_size,
            "sha256": gate._sha256(config_source),
        },
        "run_root": str(run_root),
        "deployment_root": str(deployment_root),
        "runner_exit": {
            "bytes": runner_exit.stat().st_size,
            "sha256": gate._sha256(runner_exit),
            "exit_code": 0,
        },
        "success": {
            "bytes": success.stat().st_size,
            "sha256": gate._sha256(success),
        },
        "failure_marker_absent": True,
        "ended_utc": ended_utc,
        "summary": {
            "bytes": summary_path.stat().st_size,
            "sha256": gate._sha256(summary_path),
            "completed_maps": summary["completed_maps"],
            "expected_maps": summary["expected_maps"],
            "qualified": True,
        },
        "checkpoints": {
            "count": expected_count,
            "per_trait": dict(trait_counts),
            "ordered_path_and_sha256_digest": tuple_digest.hexdigest(),
            "all_schema_manifest_trait_replicate_and_qualification_tuples_verified": True,
        },
        "checksum_manifest": {
            "bytes": checksum.stat().st_size,
            "sha256": gate._sha256(checksum),
            "verification_exit_code": 0,
        },
        "checksum_verification_log": {
            "bytes": checksum_log.stat().st_size,
            "sha256": gate._sha256(checksum_log),
        },
        "scratch_entries": 0,
        "matching_active_compute_processes": 0,
    }
    _atomic_json(Path(output_path).resolve(), receipt)
    return receipt


def _verify_frozen_stage(
    config: Mapping[str, Any],
    qualification: Path,
    smoke: Path,
) -> dict[str, dict[str, Any]]:
    paths = {
        "discovery_qualification_receipt": (
            qualification / "discovery_qualification_receipt.json"
        ),
        "qualified_discovery_traits": (
            qualification / "qualified_discovery_traits.txt"
        ),
        "trait_mask_manifest": qualification / "trait_mask_manifest.tsv",
        "frozen_model_contract": qualification / "frozen_model_contract.json",
        "qualification_success": qualification / "SUCCESS",
        "discovery_smoke_receipt": smoke / "discovery_smoke_receipt.json",
        "discovery_smoke_metrics": smoke / "discovery_metrics.tsv",
        "discovery_smoke_null_metrics": (
            smoke / "training_permutation_null_metrics.tsv"
        ),
        "smoke_success": smoke / "SUCCESS",
    }
    verified = {}
    for role, path in paths.items():
        collection = (
            "qualification_assets"
            if role
            in {
                "discovery_qualification_receipt",
                "qualified_discovery_traits",
                "trait_mask_manifest",
                "frozen_model_contract",
                "qualification_success",
            }
            else "smoke_assets"
        )
        verified[role] = _verify_asset(
            path, _asset_by_role(config, collection, role), role
        )
    return verified


def _read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    if not rows:
        raise ValueError(f"TSV is empty: {path}")
    return rows[0], rows[1:]


def _verify_smoke_continuation(
    smoke: Path,
    smoke_traits: set[str],
    metric_rows: Sequence[Sequence[Any]],
    null_rows: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    metric_header, expected_metrics = _read_tsv(smoke / "discovery_metrics.tsv")
    null_header, expected_null = _read_tsv(
        smoke / "training_permutation_null_metrics.tsv"
    )
    if metric_header != [
        "trait",
        "model",
        "n",
        "rmse",
        "mae",
        "group_macro_rmse",
        "rmse_over_outcome_sd",
        "pearson",
        "spearman",
    ]:
        raise ValueError("smoke metric header differs")
    if null_header != [
        "trait",
        "model",
        "replicate",
        "rmse",
        "group_macro_rmse",
        "pearson",
        "spearman",
    ]:
        raise ValueError("smoke null metric header differs")
    observed_metrics = [
        [str(value) for value in row] for row in metric_rows if row[0] in smoke_traits
    ]
    observed_null = [
        [str(value) for value in row] for row in null_rows if row[0] in smoke_traits
    ]
    expected_metric_map = {
        (row[0], row[1]): row for row in expected_metrics
    }
    observed_metric_map = {
        (row[0], row[1]): row for row in observed_metrics
    }
    expected_null_map = {
        (row[0], row[1], row[2]): row for row in expected_null
    }
    observed_null_map = {
        (row[0], row[1], row[2]): row for row in observed_null
    }
    if (
        len(expected_metric_map) != len(expected_metrics)
        or len(observed_metric_map) != len(observed_metrics)
        or len(expected_null_map) != len(expected_null)
        or len(observed_null_map) != len(observed_null)
        or observed_metric_map != expected_metric_map
        or observed_null_map != expected_null_map
    ):
        raise ValueError("full run does not exactly reproduce the frozen smoke subset")
    return {
        "traits": len(smoke_traits),
        "metric_rows": len(expected_metrics),
        "null_metric_rows": len(expected_null),
        "exact_string_reproduction": True,
    }


def run_full_discovery(
    full_config_path: str | Path,
    gate_config_path: str | Path,
    source_path: str | Path,
    stage1_receipt_path: str | Path,
    discovery_path: str | Path,
    validation_path: str | Path,
    kernel_receipt_path: str | Path,
    groups_path: str | Path,
    kernel_path: str | Path,
    pca_path: str | Path,
    python_path: str | Path,
    qualification_dir: str | Path,
    smoke_dir: str | Path,
    terminal_audit_path: str | Path,
    resource_audit_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run all qualified discovery traits under the already-frozen smoke ladder."""

    full_config_source = Path(full_config_path).resolve()
    full_config = load_config(full_config_source)
    gate_config_source = Path(gate_config_path).resolve()
    base_spec = full_config["base_gate_config"]
    if (
        gate_config_source.stat().st_size != base_spec["bytes"]
        or gate._sha256(gate_config_source) != base_spec["sha256"]
    ):
        raise ValueError("base discovery-gate config differs")
    gate_config = gate.load_config(gate_config_source)
    paths = {
        "source_compendium": Path(source_path).resolve(),
        "stage1_qualification_receipt": Path(stage1_receipt_path).resolve(),
        "stage1_discovery_trait_list": Path(discovery_path).resolve(),
        "stage1_validation_trait_list": Path(validation_path).resolve(),
        "stage2_kernel_qualification_receipt": Path(kernel_receipt_path).resolve(),
        "stage2_population_groups": Path(groups_path).resolve(),
        "stage2_float64_kernel": Path(kernel_path).resolve(),
        "stage2_pca10_scores": Path(pca_path).resolve(),
    }
    (
        verified_inputs,
        discovery,
        validation,
        iids,
        groups,
        pca,
        kernel,
    ) = gate._verify_upstreams(gate_config, paths, Path(python_path).resolve())
    qualification = Path(qualification_dir).resolve()
    smoke = Path(smoke_dir).resolve()
    verified_stage = _verify_frozen_stage(full_config, qualification, smoke)

    terminal_audit_source = Path(terminal_audit_path).resolve()
    terminal_audit = json.loads(terminal_audit_source.read_text(encoding="utf-8"))
    if (
        terminal_audit.get("schema_version") != TERMINAL_AUDIT_VERSION
        or terminal_audit.get("qualified") is not True
        or terminal_audit.get("terminal_closed") is not True
        or terminal_audit.get("run_root")
        != full_config["restricted_residual_bootstrap_terminal_prerequisite"]["run_root"]
        or terminal_audit.get("checkpoints", {}).get("count") != 8000
        or terminal_audit.get("checksum_manifest", {}).get(
            "verification_exit_code"
        )
        != 0
        or terminal_audit.get("matching_active_compute_processes") != 0
    ):
        raise ValueError("restricted-residual bootstrap terminal audit is absent, incomplete, or differs")

    qualification_receipt = json.loads(
        (qualification / "discovery_qualification_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    selected = gate._read_lines(qualification / "qualified_discovery_traits.txt")
    execution = full_config["execution_contract"]
    if (
        qualification_receipt.get("schema_version") != gate.QUALIFICATION_VERSION
        or qualification_receipt.get("models_executed") is not False
        or qualification_receipt.get("validation_outcomes_locked") is not True
        or qualification_receipt.get("validation_custody", {}).get(
            "validation_outcome_value_field_evaluated"
        )
        is not False
        or selected
        != gate._read_lines(qualification / "qualified_discovery_traits.txt")
        or len(selected) != execution["traits"]
        or gate._ordered_hash(selected) != execution["qualified_trait_sha256"]
        or gate._ordered_hash(selected)
        != qualification_receipt.get("qualified_discovery_trait_sha256")
        or any(trait not in set(discovery) for trait in selected)
    ):
        raise ValueError("qualified discovery trait set or custody contract differs")
    smoke_traits = gate._read_lines(qualification / "smoke_traits.txt")
    smoke_receipt = json.loads(
        (smoke / "discovery_smoke_receipt.json").read_text(encoding="utf-8")
    )
    if (
        smoke_receipt.get("schema_version") != gate.SMOKE_VERSION
        or smoke_receipt.get("technical_smoke_completed") is not True
        or smoke_receipt.get("execution_thread_cap") != 1
        or smoke_receipt.get("traits") != smoke_traits
        or smoke_receipt.get("validation_custody", {}).get(
            "validation_outcome_value_field_evaluated"
        )
        is not False
    ):
        raise ValueError("bounded smoke is not a custody-safe terminal precursor")

    group_order = sorted(set(groups))
    mask_contracts = gate._read_mask_manifest(
        qualification / "trait_mask_manifest.tsv", set(selected), group_order
    )
    validation_set = set(validation)
    selected_set = set(selected)
    roster_set = set(iids)
    values: dict[str, dict[str, float]] = {trait: {} for trait in selected}
    validation_rows_routed = 0
    nonselected_rows_routed = 0
    with paths["source_compendium"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != gate.SOURCE_HEADER:
            raise ValueError("source compendium header differs")
        for line_number, fields in enumerate(reader, 2):
            if len(fields) != len(gate.SOURCE_HEADER):
                raise ValueError(f"source row {line_number} field count differs")
            trait = fields[2]
            if trait in validation_set:
                validation_rows_routed += 1
                continue
            if trait not in selected_set or fields[3] not in roster_set:
                nonselected_rows_routed += 1
                continue
            strain = fields[3]
            try:
                value = float(fields[4])
            except ValueError as error:
                raise ValueError(
                    f"qualified discovery outcome is nonnumeric at row {line_number}"
                ) from error
            if not math.isfinite(value) or strain in values[trait]:
                raise ValueError(
                    "qualified discovery outcome is nonfinite or duplicated "
                    f"at row {line_number}"
                )
            values[trait][strain] = value

    iid_index = {iid: index for index, iid in enumerate(iids)}
    group_by_iid = dict(zip(iids, groups, strict=True))
    prediction_rows: list[list[Any]] = []
    metric_rows: list[list[Any]] = []
    null_rows: list[list[Any]] = []
    null_fit_count = 0
    permutations = execution["permutations_per_trait"]
    permutation_namespace = gate_config["smoke_contract"][
        "permutation_seed_namespace"
    ]
    for trait in selected:
        ordered_finite = [iid for iid in iids if iid in values[trait]]
        contract = mask_contracts[trait]
        if (
            len(ordered_finite) != contract["finite_samples"]
            or gate._ordered_hash(ordered_finite)
            != contract["ordered_finite_iid_sha256"]
        ):
            raise ValueError(f"qualified discovery mask drifted: {trait}")
        finite_indices = np.asarray([iid_index[iid] for iid in ordered_finite])
        y = np.asarray([values[trait][iid] for iid in ordered_finite], dtype=np.float64)
        finite_groups = [group_by_iid[iid] for iid in ordered_finite]
        local_group = np.asarray(finite_groups)
        predictions = {
            model_id: np.empty(len(ordered_finite), dtype=np.float64)
            for model_id in gate.MODEL_IDS
        }
        null_predictions = {
            (model_id, replicate): np.empty(len(ordered_finite), dtype=np.float64)
            for model_id in gate.MODEL_IDS
            for replicate in range(1, permutations + 1)
        }
        tested = np.zeros(len(ordered_finite), dtype=np.int64)
        for heldout in group_order:
            test_local = np.flatnonzero(local_group == heldout)
            train_local = np.flatnonzero(local_group != heldout)
            train_global = finite_indices[train_local]
            test_global = finite_indices[test_local]
            y_train = y[train_local]
            tested[test_local] += 1
            for model_id in gate.MODEL_IDS:
                predictions[model_id][test_local] = gate._predict_model(
                    model_id,
                    pca,
                    kernel,
                    train_global,
                    test_global,
                    y_train,
                )
                for replicate in range(1, permutations + 1):
                    seed = gate._permutation_seed(
                        permutation_namespace, trait, heldout, replicate
                    )
                    permuted = y_train[
                        np.random.default_rng(seed).permutation(len(y_train))
                    ]
                    null_predictions[(model_id, replicate)][
                        test_local
                    ] = gate._predict_model(
                        model_id,
                        pca,
                        kernel,
                        train_global,
                        test_global,
                        permuted,
                    )
                    null_fit_count += 1
        if not np.array_equal(tested, np.ones(len(tested), dtype=np.int64)):
            raise ValueError(f"finite samples are not tested exactly once: {trait}")
        for model_id in gate.MODEL_IDS:
            if not np.all(np.isfinite(predictions[model_id])):
                raise ValueError(f"nonfinite full prediction: {trait}/{model_id}")
            metrics = gate._metrics(y, predictions[model_id], finite_groups)
            metric_rows.append(
                [
                    trait,
                    model_id,
                    *[
                        "" if metrics[key] is None else metrics[key]
                        for key in (
                            "n",
                            "rmse",
                            "mae",
                            "group_macro_rmse",
                            "rmse_over_outcome_sd",
                            "pearson",
                            "spearman",
                        )
                    ],
                ]
            )
            for local_index, iid in enumerate(ordered_finite):
                prediction_rows.append(
                    [
                        trait,
                        model_id,
                        iid,
                        finite_groups[local_index],
                        y[local_index],
                        predictions[model_id][local_index],
                    ]
                )
            for replicate in range(1, permutations + 1):
                null_metric = gate._metrics(
                    y, null_predictions[(model_id, replicate)], finite_groups
                )
                null_rows.append(
                    [
                        trait,
                        model_id,
                        replicate,
                        *[
                            "" if null_metric[key] is None else null_metric[key]
                            for key in (
                                "rmse",
                                "group_macro_rmse",
                                "pearson",
                                "spearman",
                            )
                        ],
                    ]
                )

    if (
        len(prediction_rows) != execution["expected_prediction_rows"]
        or len(metric_rows) != execution["expected_metric_rows"]
        or len(null_rows) != execution["expected_null_metric_rows"]
        or null_fit_count != execution["expected_null_model_fits"]
    ):
        raise ValueError("full-discovery row or fit count differs from projection")
    smoke_reproduction = _verify_smoke_continuation(
        smoke, set(smoke_traits), metric_rows, null_rows
    )

    output = Path(output_dir).resolve()
    temporary, publish = gate._atomic_output(output)
    try:
        resource_copy = temporary / "resource_preflight.txt"
        terminal_copy = temporary / "restricted_residual_bootstrap_terminal_audit.json"
        shutil.copyfile(Path(resource_audit_path).resolve(), resource_copy)
        shutil.copyfile(terminal_audit_source, terminal_copy)
        predictions_path = temporary / "discovery_predictions.tsv"
        gate._write_tsv(
            predictions_path,
            ["trait", "model", "IID", "heldout_group", "observed", "predicted"],
            prediction_rows,
        )
        metrics_path = temporary / "discovery_metrics.tsv"
        gate._write_tsv(
            metrics_path,
            [
                "trait",
                "model",
                "n",
                "rmse",
                "mae",
                "group_macro_rmse",
                "rmse_over_outcome_sd",
                "pearson",
                "spearman",
            ],
            metric_rows,
        )
        null_path = temporary / "training_permutation_null_metrics.tsv"
        gate._write_tsv(
            null_path,
            [
                "trait",
                "model",
                "replicate",
                "rmse",
                "group_macro_rmse",
                "pearson",
                "spearman",
            ],
            null_rows,
        )
        receipt = {
            "schema_version": RESULT_VERSION,
            "analysis_id": full_config["analysis_id"],
            "classification": full_config["classification"],
            "config": {
                "bytes": full_config_source.stat().st_size,
                "sha256": gate._sha256(full_config_source),
            },
            "base_gate_config": {
                "bytes": gate_config_source.stat().st_size,
                "sha256": gate._sha256(gate_config_source),
            },
            "verified_inputs": verified_inputs,
            "verified_qualification_and_smoke_assets": verified_stage,
            "restricted_residual_bootstrap_terminal_audit": gate._resource_asset(terminal_copy),
            "resource_coexistence_audit": gate._resource_asset(resource_copy),
            "execution_thread_cap": 1,
            "traits": len(selected),
            "trait_sha256": gate._ordered_hash(selected),
            "finite_trait_sample_pairs": sum(len(item) for item in values.values()),
            "models": gate.MODEL_IDS,
            "fold_method": execution["fold_method"],
            "folds_per_trait": execution["folds_per_trait"],
            "permutations_per_trait": permutations,
            "prediction_rows": len(prediction_rows),
            "metric_rows": len(metric_rows),
            "null_metric_rows": len(null_rows),
            "null_model_fits": null_fit_count,
            "smoke_continuation": smoke_reproduction,
            "leakage_controls": {
                "trait_set_frozen_by_finite_mask_and_group_balance_only": True,
                "folds_used_outcome_magnitude": False,
                "hyperparameter_tuning_permitted": False,
                "pc_centering_and_scaling": "outer_training_samples_only",
                "outcome_centering": "outer_training_samples_only",
                "outer_test_outcomes_used_for_fit_or_selection": False,
                "permutations_used_outer_training_outcomes_only": True,
            },
            "validation_custody": {
                "locked_traits": len(validation),
                "validation_trait_sha256": gate._ordered_hash(validation),
                "rows_routed_without_value_evaluation": validation_rows_routed,
                "validation_outcome_value_field_evaluated": False,
                "validation_predictions_generated": False,
                "validation_metrics_generated": False,
                "validation_runner_command_exists": False,
            },
            "unqualified_or_nonpanel_rows_routed_without_value_evaluation": (
                nonselected_rows_routed
            ),
            "legacy_numeric_assets_accessed": False,
            "permutation_p_values_reported": False,
            "confirmatory_performance_claims_permitted": False,
            "biological_claims_permitted": False,
            "full_discovery_completed": True,
        }
        receipt_path = temporary / "full_discovery_receipt.json"
        receipt_path.write_bytes(gate._canonical_json(receipt))
        assets = {
            path.name: {"bytes": path.stat().st_size, "sha256": gate._sha256(path)}
            for path in (
                receipt_path,
                predictions_path,
                metrics_path,
                null_path,
                resource_copy,
                terminal_copy,
            )
        }
        (temporary / "MANIFEST.json").write_bytes(gate._canonical_json(assets))
        (temporary / "SUCCESS").write_text(
            "full_discovery_completed\n", encoding="utf-8"
        )
        publish()
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    terminal = commands.add_parser("verify-terminal")
    terminal.add_argument("--config", required=True)
    terminal.add_argument("--restricted_residual_bootstrap-root", required=True)
    terminal.add_argument("--restricted_residual_bootstrap-deployment", required=True)
    terminal.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--gate-config", required=True)
    run.add_argument("--source", required=True)
    run.add_argument("--stage1-receipt", required=True)
    run.add_argument("--discovery-list", required=True)
    run.add_argument("--validation-list", required=True)
    run.add_argument("--kernel-receipt", required=True)
    run.add_argument("--groups", required=True)
    run.add_argument("--kernel", required=True)
    run.add_argument("--pca", required=True)
    run.add_argument("--python", required=True)
    run.add_argument("--qualification-dir", required=True)
    run.add_argument("--smoke-dir", required=True)
    run.add_argument("--terminal-audit", required=True)
    run.add_argument("--resource-audit", required=True)
    run.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-config":
        config = load_config(args.config)
        receipt = {
            "schema_version": config["schema_version"],
            "analysis_id": config["analysis_id"],
            "traits": config["execution_contract"]["traits"],
            "maximum_threads": config["execution_contract"]["maximum_threads"],
            "maximum_wall_seconds": config["execution_contract"][
                "maximum_wall_seconds"
            ],
            "validation_runner_command_exists": config["validation_custody"][
                "runner_command_exists"
            ],
            "qualified": True,
        }
    elif args.command == "verify-terminal":
        receipt = verify_restricted_residual_bootstrap_terminal(
            args.config, args.restricted_residual_bootstrap_root, args.restricted_residual_bootstrap_deployment, args.output
        )
    else:
        receipt = run_full_discovery(
            args.config,
            args.gate_config,
            args.source,
            args.stage1_receipt,
            args.discovery_list,
            args.validation_list,
            args.kernel_receipt,
            args.groups,
            args.kernel,
            args.pca,
            args.python,
            args.qualification_dir,
            args.smoke_dir,
            args.terminal_audit,
            args.resource_audit,
            args.output_dir,
        )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
