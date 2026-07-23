"""Qualify discovery masks and run a bounded CaeNDR compendium smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CONFIG_VERSION = "wormctx-caendr-compendium-discovery-gate-config-1.0"
QUALIFICATION_VERSION = "wormctx-caendr-compendium-discovery-qualification-1.0"
SMOKE_VERSION = "wormctx-caendr-compendium-discovery-smoke-1.0"
EXPECTED_INPUT_ROLES = [
    "source_compendium",
    "stage1_qualification_receipt",
    "stage1_discovery_trait_list",
    "stage1_validation_trait_list",
    "stage2_kernel_qualification_receipt",
    "stage2_population_groups",
    "stage2_float64_kernel",
    "stage2_pca10_scores",
]
SOURCE_HEADER = [
    "submitted_by",
    "species_name",
    "trait_name",
    "strain_name",
    "trait_value",
]
MODEL_IDS = ["training_mean", "pc10_ridge_fixed", "whole_genome_gblup_fixed"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _asset_spec(payload: Mapping[str, Any], label: str) -> None:
    _require_keys(payload, {"role", "logical_path", "bytes", "sha256"}, label)
    if (
        not payload["role"]
        or not payload["logical_path"]
        or not isinstance(payload["bytes"], int)
        or payload["bytes"] <= 0
        or not _valid_sha(payload["sha256"])
    ):
        raise ValueError(f"{label} is invalid")


def load_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require_keys(
        payload,
        {
            "schema_version",
            "analysis_id",
            "classification",
            "inputs",
            "python_tool",
            "partition_contract",
            "mask_contract",
            "fold_contract",
            "model_ladder",
            "smoke_contract",
            "validation_custody",
            "claim_boundary",
            "legacy_numeric_quarantine",
        },
        "discovery gate config",
    )
    if (
        payload["schema_version"] != CONFIG_VERSION
        or payload["classification"]
        != "retrospective_discovery_only_technical_benchmark"
    ):
        raise ValueError("discovery gate config identity differs")
    if [item.get("role") for item in payload["inputs"]] != EXPECTED_INPUT_ROLES:
        raise ValueError("discovery gate input roles or order differ")
    for item in payload["inputs"]:
        _asset_spec(item, f"input {item.get('role')}")

    tool = payload["python_tool"]
    _require_keys(tool, {"version", "sha256", "maximum_threads"}, "Python tool")
    if (
        not tool["version"]
        or not _valid_sha(tool["sha256"])
        or tool["maximum_threads"] != 1
    ):
        raise ValueError("Python tool contract differs")

    partition = payload["partition_contract"]
    if partition != {
        "discovery_traits": 1184,
        "validation_traits": 261,
        "discovery_trait_sha256": partition.get("discovery_trait_sha256"),
        "validation_trait_sha256": partition.get("validation_trait_sha256"),
        "bucket_identity_must_not_change": True,
    } or not all(
        _valid_sha(partition[key])
        for key in ("discovery_trait_sha256", "validation_trait_sha256")
    ):
        raise ValueError("frozen partition contract differs")

    mask = payload["mask_contract"]
    if mask != {
        "panel_samples": 203,
        "minimum_finite_samples": 143,
        "derivation": "ceil_0.70_times_203_and_stage1_147_minus_four_absent",
        "value_finiteness_evaluated": True,
        "value_magnitude_or_rank_evaluated": False,
        "trait_variance_used_for_qualification": False,
        "minimum_qualified_discovery_traits": 800,
    }:
        raise ValueError("discovery mask contract differs")

    folds = payload["fold_contract"]
    if folds != {
        "method": "leave_one_genotype_pc_group_out",
        "expected_groups": 10,
        "minimum_heldout_samples": 3,
        "minimum_training_samples": 120,
        "minimum_samples_per_nonheldout_group": 3,
        "every_finite_sample_tested_exactly_once": True,
        "phenotype_magnitude_used_for_folds": False,
    }:
        raise ValueError("discovery fold contract differs")

    ladder = payload["model_ladder"]
    if ladder != {
        "models": [
            {"id": "training_mean", "hyperparameters": {}},
            {"id": "pc10_ridge_fixed", "hyperparameters": {"ridge": 1.0}},
            {
                "id": "whole_genome_gblup_fixed",
                "hyperparameters": {"ridge": 1.0},
            },
        ],
        "hyperparameter_tuning_permitted": False,
        "outer_test_used_for_selection": False,
        "pc_centering_and_scaling": "outer_training_samples_only",
        "outcome_centering": "outer_training_samples_only",
        "missing_outcome_imputation_permitted": False,
        "constant_traits_retained_as_nonestimable": True,
    }:
        raise ValueError("discovery model ladder differs")

    smoke = payload["smoke_contract"]
    if smoke != {
        "traits": 8,
        "selection": "sha256_namespace_plus_trait_name_ascending",
        "selection_namespace": smoke.get("selection_namespace"),
        "permutations_per_trait": 4,
        "permutation_scope": "outer_training_outcomes_only",
        "permutation_seed_namespace": smoke.get("permutation_seed_namespace"),
        "maximum_threads": 1,
        "maximum_wall_seconds": 900,
        "performance_claims_permitted": False,
    } or not smoke["selection_namespace"] or not smoke["permutation_seed_namespace"]:
        raise ValueError("discovery smoke contract differs")

    custody = payload["validation_custody"]
    if custody != {
        "locked": True,
        "runner_command_exists": False,
        "outcome_value_field_evaluated": False,
        "trait_specific_masks_computed": False,
        "per_trait_availability_reported": False,
        "list_used_only_for_exclusion_and_identity": True,
    }:
        raise ValueError("validation custody contract differs")
    if payload["claim_boundary"] != {
        "biological_claims_permitted": False,
        "predictive_performance_claims_permitted": False,
        "independent_validation_established": False,
        "technical_smoke_only": True,
    }:
        raise ValueError("discovery claim boundary differs")
    if payload["legacy_numeric_quarantine"] != {
        "accepted_as_inputs": False,
        "paths_accepted_by_runner": False,
        "values_accessed": False,
    }:
        raise ValueError("legacy numeric quarantine differs")
    return payload


def _by_role(config: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    return next(item for item in config["inputs"] if item["role"] == role)


def _verify(path: Path, specification: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    size = path.stat().st_size
    digest = _sha256(path)
    if size != specification["bytes"] or digest != specification["sha256"]:
        raise ValueError(f"{label} byte count or SHA-256 differs")
    return {"bytes": size, "sha256": digest, "qualified": True}


def _read_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line or "\t" in line or "\r" in line for line in lines):
        raise ValueError(f"line list is empty or malformed: {path}")
    if len(lines) != len(set(lines)):
        raise ValueError(f"line list has duplicates: {path}")
    return lines


def _verify_partition(
    config: Mapping[str, Any], discovery_path: Path, validation_path: Path
) -> tuple[list[str], list[str]]:
    discovery = _read_lines(discovery_path)
    validation = _read_lines(validation_path)
    contract = config["partition_contract"]
    if (
        len(discovery) != contract["discovery_traits"]
        or len(validation) != contract["validation_traits"]
        or _ordered_hash(discovery) != contract["discovery_trait_sha256"]
        or _ordered_hash(validation) != contract["validation_trait_sha256"]
        or set(discovery) & set(validation)
    ):
        raise ValueError("frozen discovery/validation partition differs")
    return discovery, validation


def _read_groups(
    path: Path, kernel_receipt: Mapping[str, Any]
) -> tuple[list[str], list[str], dict[str, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "IID\tpopulation_group":
        raise ValueError("population-group header differs")
    rows = [line.split("\t") for line in lines[1:]]
    if any(len(row) != 2 or not row[0] or not row[1] for row in rows):
        raise ValueError("population-group row differs")
    iids = [row[0] for row in rows]
    groups = [row[1] for row in rows]
    sizes = dict(sorted(Counter(groups).items()))
    expected = kernel_receipt["population_groups"]
    if (
        len(iids) != 203
        or len(iids) != len(set(iids))
        or _ordered_hash(iids) != kernel_receipt["ordered_iid_sha256"]
        or expected["selected_k"] != 10
        or sizes != expected["group_sizes"]
    ):
        raise ValueError("population groups differ from the stage-2 receipt")
    return iids, groups, sizes


def _read_pca(path: Path, expected_iids: Sequence[str]) -> np.ndarray:
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    expected_header = ["#FID", "IID", *[f"PC{index}" for index in range(1, 11)]]
    if not rows or rows[0] != expected_header:
        raise ValueError("PCA header differs")
    if [row[1] for row in rows[1:]] != list(expected_iids):
        raise ValueError("PCA order differs from the stage-2 roster")
    try:
        values = np.asarray(
            [[float(value) for value in row[2:]] for row in rows[1:]],
            dtype=np.float64,
        )
    except (IndexError, ValueError) as error:
        raise ValueError("PCA score table is malformed") from error
    if values.shape != (203, 10) or not np.all(np.isfinite(values)):
        raise ValueError("PCA score matrix differs")
    return values


def _read_kernel(path: Path) -> np.ndarray:
    try:
        matrix = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError("stage-2 kernel is not a safe NPY array") from error
    matrix = np.asarray(matrix, dtype=np.float64)
    if (
        matrix.shape != (203, 203)
        or not np.all(np.isfinite(matrix))
        or not np.array_equal(matrix, matrix.T)
    ):
        raise ValueError("stage-2 kernel shape, finiteness, or symmetry differs")
    if float(np.linalg.eigvalsh(matrix)[0]) < -1e-5:
        raise ValueError("stage-2 kernel is not PSD")
    return matrix


def _verify_upstreams(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    python_path: Path,
) -> tuple[
    dict[str, Any],
    list[str],
    list[str],
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
]:
    verified = {
        role: _verify(paths[role], _by_role(config, role), role)
        for role in EXPECTED_INPUT_ROLES
    }
    if not python_path.is_file() or _sha256(python_path) != config["python_tool"]["sha256"]:
        raise ValueError("Python executable differs")
    discovery, validation = _verify_partition(
        config,
        paths["stage1_discovery_trait_list"],
        paths["stage1_validation_trait_list"],
    )
    stage1 = json.loads(
        paths["stage1_qualification_receipt"].read_text(encoding="utf-8")
    )
    partition = config["partition_contract"]
    if (
        stage1.get("models_executed") is not False
        or stage1.get("compendium", {}).get("discovery_trait_sha256")
        != partition["discovery_trait_sha256"]
        or stage1.get("compendium", {}).get("validation_trait_sha256")
        != partition["validation_trait_sha256"]
    ):
        raise ValueError("stage-1 receipt does not bind the frozen partition")
    kernel_receipt = json.loads(
        paths["stage2_kernel_qualification_receipt"].read_text(encoding="utf-8")
    )
    if (
        kernel_receipt.get("schema_version")
        != "wormctx-caendr-compendium-203-kernel-qualification-1.0"
        or kernel_receipt.get("samples") != 203
        or kernel_receipt.get("phenotype_values_accessed") is not False
        or kernel_receipt.get("legacy_numeric_assets_accessed") is not False
        or kernel_receipt.get("discovery_prediction_models_executed") is not False
        or kernel_receipt.get("partition_identity", {}).get("discovery_trait_sha256")
        != partition["discovery_trait_sha256"]
        or kernel_receipt.get("partition_identity", {}).get("validation_trait_sha256")
        != partition["validation_trait_sha256"]
        or kernel_receipt.get("validation_outcomes_locked") is not True
    ):
        raise ValueError("stage-2 kernel receipt semantics differ")
    iids, groups, _sizes = _read_groups(
        paths["stage2_population_groups"], kernel_receipt
    )
    pca = _read_pca(paths["stage2_pca10_scores"], iids)
    kernel = _read_kernel(paths["stage2_float64_kernel"])
    return verified, discovery, validation, iids, groups, pca, kernel


def _resource_asset(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("resource coexistence audit is missing")
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _mask_failure_reasons(
    counts: Mapping[str, int], config: Mapping[str, Any]
) -> list[str]:
    mask = config["mask_contract"]
    folds = config["fold_contract"]
    total = sum(counts.values())
    reasons: list[str] = []
    if total < mask["minimum_finite_samples"]:
        reasons.append("finite_samples_below_minimum")
    groups = sorted(counts)
    for heldout in groups:
        if counts[heldout] < folds["minimum_heldout_samples"]:
            reasons.append(f"{heldout}_heldout_below_minimum")
        if total - counts[heldout] < folds["minimum_training_samples"]:
            reasons.append(f"{heldout}_training_below_minimum")
        for training_group in groups:
            if (
                training_group != heldout
                and counts[training_group]
                < folds["minimum_samples_per_nonheldout_group"]
            ):
                reasons.append(
                    f"{heldout}_training_{training_group}_below_minimum"
                )
    return sorted(set(reasons))


def _smoke_selection(
    qualified: Sequence[str], contract: Mapping[str, Any]
) -> list[str]:
    namespace = contract["selection_namespace"]
    ordered = sorted(
        qualified,
        key=lambda trait: (
            hashlib.sha256((namespace + trait).encode("utf-8")).hexdigest(),
            trait,
        ),
    )
    return ordered[: contract["traits"]]


def _atomic_output(output: Path) -> tuple[Path, Any]:
    if output.exists():
        raise FileExistsError(f"write-once output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()

    def publish() -> None:
        os.replace(temporary, output)

    return temporary, publish


def qualify_discovery_masks(
    config_path: str | Path,
    source_path: str | Path,
    stage1_receipt_path: str | Path,
    discovery_path: str | Path,
    validation_path: str | Path,
    kernel_receipt_path: str | Path,
    groups_path: str | Path,
    kernel_path: str | Path,
    pca_path: str | Path,
    python_path: str | Path,
    resource_audit_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build discovery masks without using outcome magnitude or validation values."""

    config_source = Path(config_path).resolve()
    config = load_config(config_source)
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
    verified, discovery, validation, iids, groups, _pca, _kernel = _verify_upstreams(
        config, paths, Path(python_path).resolve()
    )
    discovery_set = set(discovery)
    validation_set = set(validation)
    roster_set = set(iids)
    finite_by_trait = {trait: set() for trait in discovery}
    discovery_rows_evaluated = 0
    validation_rows_routed = 0
    validation_traits_seen: set[str] = set()
    other_rows_routed = 0

    with paths["source_compendium"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("source compendium is empty") from error
        if header != SOURCE_HEADER:
            raise ValueError("source compendium header differs")
        for line_number, fields in enumerate(reader, 2):
            if len(fields) != len(SOURCE_HEADER):
                raise ValueError(f"source row {line_number} field count differs")
            if fields[1] != "c_elegans":
                raise ValueError(f"source row {line_number} species differs")
            trait = fields[2]
            strain = fields[3]
            if trait in validation_set:
                validation_rows_routed += 1
                validation_traits_seen.add(trait)
                continue
            if trait not in discovery_set:
                other_rows_routed += 1
                continue
            if strain not in roster_set:
                continue
            try:
                value = float(fields[4])
            except ValueError as error:
                raise ValueError(
                    f"discovery value is nonnumeric at source row {line_number}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"discovery value is nonfinite at source row {line_number}"
                )
            if strain in finite_by_trait[trait]:
                raise ValueError(f"duplicate discovery trait/strain at row {line_number}")
            finite_by_trait[trait].add(strain)
            discovery_rows_evaluated += 1
    if validation_traits_seen != validation_set:
        raise ValueError("not every locked validation trait was routed by identity")

    group_order = sorted(set(groups))
    iid_group = dict(zip(iids, groups, strict=True))
    mask_rows: list[list[Any]] = []
    fold_records: list[dict[str, Any]] = []
    qualified: list[str] = []
    count_histogram: Counter[int] = Counter()
    minimum_group_histogram: Counter[int] = Counter()
    failure_histogram: Counter[str] = Counter()
    for trait in discovery:
        finite = finite_by_trait[trait]
        ordered_finite = [iid for iid in iids if iid in finite]
        counts = {
            group: sum(iid_group[iid] == group for iid in ordered_finite)
            for group in group_order
        }
        reasons = _mask_failure_reasons(counts, config)
        is_qualified = not reasons
        if is_qualified:
            qualified.append(trait)
        for reason in reasons:
            failure_histogram[reason] += 1
        count_histogram[len(ordered_finite)] += 1
        minimum_group_histogram[min(counts.values())] += 1
        mask_rows.append(
            [
                trait,
                "discovery",
                len(ordered_finite),
                _ordered_hash(ordered_finite),
                "true" if is_qualified else "false",
                ";".join(reasons),
                *[counts[group] for group in group_order],
            ]
        )
        for heldout in group_order:
            test_iids = [iid for iid in ordered_finite if iid_group[iid] == heldout]
            train_iids = [iid for iid in ordered_finite if iid_group[iid] != heldout]
            fold_records.append(
                {
                    "trait": trait,
                    "qualified": is_qualified,
                    "heldout_group": heldout,
                    "train_samples": len(train_iids),
                    "test_samples": len(test_iids),
                    "ordered_train_iid_sha256": _ordered_hash(train_iids),
                    "ordered_test_iid_sha256": _ordered_hash(test_iids),
                    "phenotype_magnitude_used": False,
                }
            )
    if len(qualified) < config["mask_contract"]["minimum_qualified_discovery_traits"]:
        raise ValueError("too few discovery traits pass the frozen mask/fold gate")
    selected = _smoke_selection(qualified, config["smoke_contract"])
    if len(selected) != config["smoke_contract"]["traits"]:
        raise ValueError("too few qualified traits for the bounded smoke")

    output = Path(output_dir).resolve()
    temporary, publish = _atomic_output(output)
    try:
        resource_copy = temporary / "resource_preflight.txt"
        shutil.copyfile(Path(resource_audit_path).resolve(), resource_copy)
        mask_path = temporary / "trait_mask_manifest.tsv"
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "trait",
                "bucket",
                "finite_samples",
                "ordered_finite_iid_sha256",
                "qualified",
                "failure_reasons",
                *group_order,
            ]
        )
        writer.writerows(mask_rows)
        mask_path.write_text(stream.getvalue(), encoding="utf-8")
        qualified_path = temporary / "qualified_discovery_traits.txt"
        qualified_path.write_text("\n".join(qualified) + "\n", encoding="utf-8")
        smoke_path = temporary / "smoke_traits.txt"
        smoke_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
        folds_path = temporary / "fold_manifest.jsonl"
        folds_path.write_bytes(b"".join(_canonical_json(item) for item in fold_records))
        custody = {
            "locked_validation_traits": len(validation),
            "validation_trait_sha256": _ordered_hash(validation),
            "validation_traits_seen_by_identity": len(validation_traits_seen),
            "validation_rows_routed_without_value_evaluation": validation_rows_routed,
            "validation_outcome_value_field_evaluated": False,
            "validation_trait_specific_masks_computed": False,
            "validation_per_trait_availability_reported": False,
            "validation_model_command_exists": False,
        }
        custody_path = temporary / "validation_custody.json"
        custody_path.write_bytes(_canonical_json(custody))
        model_contract = {
            "model_ladder": config["model_ladder"],
            "fold_contract": config["fold_contract"],
            "smoke_contract": config["smoke_contract"],
            "selected_traits": selected,
            "selected_trait_sha256": _ordered_hash(selected),
            "selection_used_outcome_magnitude_or_rank": False,
            "validation_custody": config["validation_custody"],
        }
        model_path = temporary / "frozen_model_contract.json"
        model_path.write_bytes(_canonical_json(model_contract))
        receipt = {
            "schema_version": QUALIFICATION_VERSION,
            "analysis_id": config["analysis_id"],
            "classification": config["classification"],
            "config": {
                "bytes": config_source.stat().st_size,
                "sha256": _sha256(config_source),
            },
            "verified_inputs": verified,
            "python_tool": config["python_tool"],
            "resource_coexistence_audit": _resource_asset(resource_copy),
            "samples": len(iids),
            "ordered_iid_sha256": _ordered_hash(iids),
            "population_group_sizes": dict(sorted(Counter(groups).items())),
            "discovery_traits_frozen": len(discovery),
            "discovery_trait_sha256": _ordered_hash(discovery),
            "qualified_discovery_traits": len(qualified),
            "qualified_discovery_trait_sha256": _ordered_hash(qualified),
            "smoke_traits": selected,
            "smoke_trait_sha256": _ordered_hash(selected),
            "finite_sample_histogram": dict(sorted(count_histogram.items())),
            "minimum_group_sample_histogram": dict(
                sorted(minimum_group_histogram.items())
            ),
            "failure_reason_histogram": dict(sorted(failure_histogram.items())),
            "discovery_panel_values_evaluated_for_finiteness": (
                discovery_rows_evaluated
            ),
            "discovery_value_magnitude_or_rank_evaluated": False,
            "discovery_trait_variance_evaluated": False,
            "validation_custody": custody,
            "other_source_rows_routed_without_value_evaluation": other_rows_routed,
            "model_contract": model_contract,
            "legacy_numeric_assets_accessed": False,
            "models_executed": False,
            "validation_outcomes_locked": True,
            "biological_claims_permitted": False,
        }
        receipt_path = temporary / "discovery_qualification_receipt.json"
        receipt_path.write_bytes(_canonical_json(receipt))
        assets = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (
                receipt_path,
                mask_path,
                qualified_path,
                smoke_path,
                folds_path,
                custody_path,
                model_path,
                resource_copy,
            )
        }
        (temporary / "MANIFEST.json").write_bytes(_canonical_json(assets))
        (temporary / "SUCCESS").write_text("qualified\n", encoding="utf-8")
        publish()
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def _read_mask_manifest(
    path: Path, selected: set[str], group_order: Sequence[str]
) -> dict[str, dict[str, Any]]:
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    expected = [
        "trait",
        "bucket",
        "finite_samples",
        "ordered_finite_iid_sha256",
        "qualified",
        "failure_reasons",
        *group_order,
    ]
    if not rows or rows[0] != expected:
        raise ValueError("trait-mask manifest header differs")
    output: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        if len(row) != len(expected) or row[0] not in selected:
            continue
        output[row[0]] = {
            "finite_samples": int(row[2]),
            "ordered_finite_iid_sha256": row[3],
            "qualified": row[4] == "true",
            "failure_reasons": row[5],
            "group_counts": {
                group: int(value)
                for group, value in zip(group_order, row[6:], strict=True)
            },
        }
    if set(output) != selected or any(
        not item["qualified"] or item["failure_reasons"] for item in output.values()
    ):
        raise ValueError("selected smoke traits are not mask/fold qualified")
    return output


def _predict_mean(y_train: np.ndarray, test_count: int) -> np.ndarray:
    return np.full(test_count, float(np.mean(y_train)), dtype=np.float64)


def _predict_pc(
    pca: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    y_train: np.ndarray,
    ridge: float,
) -> np.ndarray:
    mean = pca[train].mean(axis=0)
    scale = pca[train].std(axis=0, ddof=0)
    if np.any(scale <= np.finfo(np.float64).eps) or not np.all(np.isfinite(scale)):
        raise ValueError("training-only PC standardization is singular")
    x_train = (pca[train] - mean) / scale
    x_test = (pca[test] - mean) / scale
    outcome_mean = float(np.mean(y_train))
    system = x_train.T @ x_train + ridge * np.eye(x_train.shape[1])
    target = x_train.T @ (y_train - outcome_mean)
    try:
        coefficients = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, target, rcond=None)[0]
    return outcome_mean + x_test @ coefficients


def _predict_kernel(
    kernel: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    y_train: np.ndarray,
    ridge: float,
) -> np.ndarray:
    outcome_mean = float(np.mean(y_train))
    system = kernel[np.ix_(train, train)] + ridge * np.eye(len(train))
    try:
        coefficients = np.linalg.solve(system, y_train - outcome_mean)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(
            system, y_train - outcome_mean, rcond=None
        )[0]
    return outcome_mean + kernel[np.ix_(test, train)] @ coefficients


def _predict_model(
    model_id: str,
    pca: np.ndarray,
    kernel: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    if model_id == "training_mean":
        return _predict_mean(y_train, len(test))
    if model_id == "pc10_ridge_fixed":
        return _predict_pc(pca, train, test, y_train, 1.0)
    if model_id == "whole_genome_gblup_fixed":
        return _predict_kernel(kernel, train, test, y_train, 1.0)
    raise ValueError(f"unknown model ID: {model_id}")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    left = first - np.mean(first)
    right = second - np.mean(second)
    denominator = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.clip(np.sum(left * right) / denominator, -1.0, 1.0))


def _metrics(
    observed: np.ndarray, predicted: np.ndarray, groups: Sequence[str]
) -> dict[str, Any]:
    residual = observed - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    outcome_sd = float(np.std(observed, ddof=0))
    group_rmse = []
    group_array = np.asarray(groups)
    for group in sorted(set(groups)):
        selected = group_array == group
        group_rmse.append(float(np.sqrt(np.mean(residual[selected] ** 2))))
    return {
        "n": len(observed),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(residual))),
        "group_macro_rmse": float(math.fsum(group_rmse) / len(group_rmse)),
        "rmse_over_outcome_sd": None if outcome_sd == 0.0 else rmse / outcome_sd,
        "pearson": _correlation(observed, predicted),
        "spearman": _correlation(
            _average_ranks(observed), _average_ranks(predicted)
        ),
    }


def _permutation_seed(
    namespace: str, trait: str, heldout_group: str, replicate: int
) -> int:
    payload = f"{namespace}{trait}\0{heldout_group}\0{replicate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _write_tsv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def run_discovery_smoke(
    config_path: str | Path,
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
    resource_audit_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the frozen one-thread smoke on hash-selected discovery traits only."""

    config_source = Path(config_path).resolve()
    config = load_config(config_source)
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
    verified, discovery, validation, iids, groups, pca, kernel = _verify_upstreams(
        config, paths, Path(python_path).resolve()
    )
    qualification = Path(qualification_dir).resolve()
    receipt_path = qualification / "discovery_qualification_receipt.json"
    qualification_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        qualification_receipt.get("schema_version") != QUALIFICATION_VERSION
        or qualification_receipt.get("models_executed") is not False
        or qualification_receipt.get("validation_outcomes_locked") is not True
        or qualification_receipt.get("validation_custody", {}).get(
            "validation_outcome_value_field_evaluated"
        )
        is not False
        or (qualification / "SUCCESS").read_text(encoding="utf-8") != "qualified\n"
    ):
        raise ValueError("discovery qualification is not a terminal custody-safe gate")
    selected = _read_lines(qualification / "smoke_traits.txt")
    if (
        selected != qualification_receipt["smoke_traits"]
        or _ordered_hash(selected) != qualification_receipt["smoke_trait_sha256"]
        or len(selected) != config["smoke_contract"]["traits"]
        or any(trait not in set(discovery) for trait in selected)
    ):
        raise ValueError("smoke trait selection differs from the frozen qualification")
    group_order = sorted(set(groups))
    mask_contracts = _read_mask_manifest(
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
        if next(reader) != SOURCE_HEADER:
            raise ValueError("source compendium header differs")
        for line_number, fields in enumerate(reader, 2):
            if len(fields) != len(SOURCE_HEADER):
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
                    f"selected discovery outcome is nonnumeric at row {line_number}"
                ) from error
            if not math.isfinite(value) or strain in values[trait]:
                raise ValueError(
                    f"selected discovery outcome is nonfinite or duplicated at row {line_number}"
                )
            values[trait][strain] = value

    iid_index = {iid: index for index, iid in enumerate(iids)}
    group_by_iid = dict(zip(iids, groups, strict=True))
    prediction_rows: list[list[Any]] = []
    metric_rows: list[list[Any]] = []
    null_rows: list[list[Any]] = []
    summary: dict[str, Any] = {}
    null_fit_count = 0
    for trait in selected:
        ordered_finite = [iid for iid in iids if iid in values[trait]]
        contract = mask_contracts[trait]
        if (
            len(ordered_finite) != contract["finite_samples"]
            or _ordered_hash(ordered_finite)
            != contract["ordered_finite_iid_sha256"]
        ):
            raise ValueError(f"selected discovery mask drifted: {trait}")
        finite_indices = np.asarray([iid_index[iid] for iid in ordered_finite])
        y = np.asarray([values[trait][iid] for iid in ordered_finite], dtype=np.float64)
        finite_groups = [group_by_iid[iid] for iid in ordered_finite]
        local_group = np.asarray(finite_groups)
        predictions = {
            model_id: np.empty(len(ordered_finite), dtype=np.float64)
            for model_id in MODEL_IDS
        }
        null_predictions = {
            (model_id, replicate): np.empty(len(ordered_finite), dtype=np.float64)
            for model_id in MODEL_IDS
            for replicate in range(1, config["smoke_contract"]["permutations_per_trait"] + 1)
        }
        tested = np.zeros(len(ordered_finite), dtype=np.int64)
        for heldout in group_order:
            test_local = np.flatnonzero(local_group == heldout)
            train_local = np.flatnonzero(local_group != heldout)
            train_global = finite_indices[train_local]
            test_global = finite_indices[test_local]
            y_train = y[train_local]
            tested[test_local] += 1
            for model_id in MODEL_IDS:
                predictions[model_id][test_local] = _predict_model(
                    model_id,
                    pca,
                    kernel,
                    train_global,
                    test_global,
                    y_train,
                )
                for replicate in range(
                    1, config["smoke_contract"]["permutations_per_trait"] + 1
                ):
                    seed = _permutation_seed(
                        config["smoke_contract"]["permutation_seed_namespace"],
                        trait,
                        heldout,
                        replicate,
                    )
                    permutation = np.random.default_rng(seed).permutation(len(y_train))
                    permuted = y_train[permutation]
                    null_predictions[(model_id, replicate)][
                        test_local
                    ] = _predict_model(
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
        summary[trait] = {}
        for model_id in MODEL_IDS:
            if not np.all(np.isfinite(predictions[model_id])):
                raise ValueError(f"nonfinite smoke prediction: {trait}/{model_id}")
            metrics = _metrics(y, predictions[model_id], finite_groups)
            summary[trait][model_id] = metrics
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
            for replicate in range(
                1, config["smoke_contract"]["permutations_per_trait"] + 1
            ):
                null_metric = _metrics(
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

    output = Path(output_dir).resolve()
    temporary, publish = _atomic_output(output)
    try:
        resource_copy = temporary / "resource_preflight.txt"
        shutil.copyfile(Path(resource_audit_path).resolve(), resource_copy)
        predictions_path = temporary / "discovery_predictions.tsv"
        _write_tsv(
            predictions_path,
            ["trait", "model", "IID", "heldout_group", "observed", "predicted"],
            prediction_rows,
        )
        metrics_path = temporary / "discovery_metrics.tsv"
        _write_tsv(
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
        _write_tsv(
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
            "schema_version": SMOKE_VERSION,
            "analysis_id": config["analysis_id"],
            "classification": "bounded_discovery_only_technical_smoke",
            "config": {
                "bytes": config_source.stat().st_size,
                "sha256": _sha256(config_source),
            },
            "qualification_receipt": {
                "bytes": receipt_path.stat().st_size,
                "sha256": _sha256(receipt_path),
            },
            "verified_inputs": verified,
            "resource_coexistence_audit": _resource_asset(resource_copy),
            "execution_thread_cap": 1,
            "traits": selected,
            "trait_sha256": _ordered_hash(selected),
            "models": MODEL_IDS,
            "fold_method": config["fold_contract"]["method"],
            "permutations_per_trait": config["smoke_contract"][
                "permutations_per_trait"
            ],
            "null_model_fits": null_fit_count,
            "metrics": summary,
            "leakage_controls": {
                "trait_selection_used_outcome_magnitude_or_rank": False,
                "folds_used_outcome_magnitude": False,
                "hyperparameter_tuning_permitted": False,
                "pc_centering_and_scaling": "outer_training_samples_only",
                "outcome_centering": "outer_training_samples_only",
                "outer_test_outcomes_used_for_fit_or_selection": False,
                "permutations_used_outer_training_outcomes_only": True,
            },
            "validation_custody": {
                "locked_traits": len(validation),
                "validation_trait_sha256": _ordered_hash(validation),
                "rows_routed_without_value_evaluation": validation_rows_routed,
                "validation_outcome_value_field_evaluated": False,
                "validation_predictions_generated": False,
                "validation_metrics_generated": False,
            },
            "nonselected_rows_routed_without_value_evaluation": nonselected_rows_routed,
            "legacy_numeric_assets_accessed": False,
            "biological_claims_permitted": False,
            "predictive_performance_claims_permitted": False,
            "technical_smoke_completed": True,
        }
        receipt_path_out = temporary / "discovery_smoke_receipt.json"
        receipt_path_out.write_bytes(_canonical_json(receipt))
        assets = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (
                receipt_path_out,
                predictions_path,
                metrics_path,
                null_path,
                resource_copy,
            )
        }
        (temporary / "MANIFEST.json").write_bytes(_canonical_json(assets))
        (temporary / "SUCCESS").write_text("technical_smoke_completed\n", encoding="utf-8")
        publish()
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--stage1-receipt", required=True)
    parser.add_argument("--discovery-list", required=True)
    parser.add_argument("--validation-list", required=True)
    parser.add_argument("--kernel-receipt", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--resource-audit", required=True)
    parser.add_argument("--output-dir", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    qualify = subparsers.add_parser("qualify")
    _common_arguments(qualify)
    smoke = subparsers.add_parser("smoke")
    _common_arguments(smoke)
    smoke.add_argument("--qualification-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = [
        args.config,
        args.source,
        args.stage1_receipt,
        args.discovery_list,
        args.validation_list,
        args.kernel_receipt,
        args.groups,
        args.kernel,
        args.pca,
        args.python,
    ]
    if args.command == "qualify":
        receipt = qualify_discovery_masks(
            *common, args.resource_audit, args.output_dir
        )
    else:
        receipt = run_discovery_smoke(
            *common,
            args.qualification_dir,
            args.resource_audit,
            args.output_dir,
        )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
