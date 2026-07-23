"""Qualify the retrospective CaeNDR phenotype compendium without fitting a model.

This lane deliberately separates source/schema qualification from downstream phenotype
modeling.  It inventories every trait, binds the exact 209-strain genotype order, freezes a
mechanical trait partition, and records legacy 203-sample results as prohibited inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


CONFIG_VERSION = "wormctx-caendr-compendium-qualification-config-1.1"
RECEIPT_VERSION = "wormctx-caendr-compendium-source-qualification-1.1"
PARTITION_ALGORITHM = "sha256_namespace_plus_trait_name_modulo_5"
EXPECTED_SOURCE_HEADER = [
    "submitted_by",
    "species_name",
    "trait_name",
    "strain_name",
    "trait_value",
]
EXPECTED_LEGACY_HEADERS = {
    "legacy_h2_screen_203": ["trait", "n", "h2_HE", "is_expr", "h2_clip"],
    "legacy_h2_greml_203": ["trait", "n", "h2_greml"],
    "legacy_gblup_top15_203": ["trait", "n", "h2", "gblup_r", "gblup_rho"],
}


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


def _ordered_text_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _require_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _validate_asset_spec(payload: dict[str, Any], label: str) -> None:
    _require_keys(payload, {"logical_path", "bytes", "sha256"}, label)
    if not isinstance(payload["logical_path"], str) or not payload["logical_path"]:
        raise ValueError(f"{label} logical path is empty")
    if not isinstance(payload["bytes"], int) or payload["bytes"] <= 0:
        raise ValueError(f"{label} byte count is invalid")
    sha = payload["sha256"]
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(character not in "0123456789abcdef" for character in sha)
    ):
        raise ValueError(f"{label} SHA-256 is invalid")


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(
        payload,
        {
            "schema_version",
            "analysis_id",
            "classification",
            "biological_claims_permitted",
            "source",
            "genotype_panel",
            "legacy_quarantine",
            "qualification_history",
            "eligibility_contract",
            "partition_contract",
            "future_benchmark_contract",
        },
        "config",
    )
    if payload["schema_version"] != CONFIG_VERSION:
        raise ValueError("qualification config schema differs")
    if payload["classification"] != "retrospective_source_qualification_only":
        raise ValueError("classification must remain retrospective source qualification")
    if payload["biological_claims_permitted"] is not False:
        raise ValueError("source qualification cannot permit biological claims")

    source = payload["source"]
    _require_keys(
        source,
        {"asset", "expected_header", "expected_data_rows", "expected_trait_count"},
        "source",
    )
    _validate_asset_spec(source["asset"], "source asset")
    if source["expected_header"] != EXPECTED_SOURCE_HEADER:
        raise ValueError("source CSV header contract differs")
    if source["expected_data_rows"] <= 0 or source["expected_trait_count"] <= 0:
        raise ValueError("source expected counts must be positive")

    panel = payload["genotype_panel"]
    _require_keys(
        panel,
        {
            "asset",
            "expected_samples",
            "expected_fid",
            "expected_ordered_iid_sha256",
            "exact_order_required",
        },
        "genotype panel",
    )
    _validate_asset_spec(panel["asset"], "genotype panel asset")
    if (
        panel["expected_samples"] != 209
        or panel["expected_fid"] != "0"
        or panel["exact_order_required"] is not True
    ):
        raise ValueError("genotype panel must remain the exact 209-strain FAM order")

    quarantine = payload["legacy_quarantine"]
    if [item.get("id") for item in quarantine] != list(EXPECTED_LEGACY_HEADERS):
        raise ValueError("legacy quarantine inventory or order differs")
    for item in quarantine:
        _require_keys(
            item,
            {
                "id",
                "asset",
                "expected_header",
                "expected_rows",
                "legacy_panel_sample_ceiling",
                "expected_n_histogram",
                "numerical_result_fields_used_for_design",
                "permitted_use",
                "prohibited_uses",
            },
            f"legacy quarantine {item.get('id')}",
        )
        _validate_asset_spec(item["asset"], f"legacy asset {item['id']}")
        if item["expected_header"] != EXPECTED_LEGACY_HEADERS[item["id"]]:
            raise ValueError(f"legacy header contract differs: {item['id']}")
        if item["expected_rows"] <= 0 or item["legacy_panel_sample_ceiling"] != 203:
            raise ValueError("legacy outputs must remain explicitly bound to a 203-sample panel")
        n_histogram = item["expected_n_histogram"]
        if (
            not isinstance(n_histogram, dict)
            or not n_histogram
            or any(not str(key).isdigit() for key in n_histogram)
            or any(not isinstance(value, int) or value <= 0 for value in n_histogram.values())
            or sum(n_histogram.values()) != item["expected_rows"]
            or max(int(key) for key in n_histogram) != item["legacy_panel_sample_ceiling"]
        ):
            raise ValueError(f"legacy n histogram is invalid: {item['id']}")
        if item["numerical_result_fields_used_for_design"] is not False:
            raise ValueError("legacy numerical results cannot inform the new design")
        if item["permitted_use"] != "fingerprint_and_structural_quarantine_only":
            raise ValueError("legacy output permitted use differs")
        required_prohibitions = {
            "trait_selection",
            "hyperparameter_selection",
            "model_comparison",
            "preliminary_evidence",
            "biological_claims",
        }
        if set(item["prohibited_uses"]) != required_prohibitions:
            raise ValueError(f"legacy quarantine prohibitions differ: {item['id']}")

    history = payload["qualification_history"]
    _require_keys(
        history,
        {
            "prior_model_free_run_id",
            "prior_config_sha256",
            "prior_receipt",
            "prior_minimum_panel_samples",
            "prior_eligible_traits",
            "revised_threshold_basis",
            "phenotype_magnitude_or_rank_used_for_revision",
        },
        "qualification history",
    )
    _validate_asset_spec(history["prior_receipt"], "prior qualification receipt")
    if (
        not history["prior_model_free_run_id"]
        or len(history["prior_config_sha256"]) != 64
        or history["prior_minimum_panel_samples"] != 168
        or history["prior_eligible_traits"] != 0
        or history["revised_threshold_basis"]
        != "source_panel_overlap_histogram_only_no_model_results"
        or history["phenotype_magnitude_or_rank_used_for_revision"] is not False
    ):
        raise ValueError("qualification history does not bind the model-free v1 audit")

    eligibility = payload["eligibility_contract"]
    _require_keys(
        eligibility,
        {
            "selection_scope",
            "required_species",
            "minimum_panel_fraction",
            "minimum_panel_samples",
            "duplicate_trait_strain_rows_permitted",
            "nonfinite_values_permitted",
            "outcome_magnitude_or_rank_used",
            "top_n_selection_permitted",
            "constant_outcome_policy",
        },
        "eligibility contract",
    )
    if eligibility != {
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
    }:
        raise ValueError("mechanical trait-eligibility contract differs")
    if eligibility["minimum_panel_samples"] != math.ceil(
        panel["expected_samples"] * eligibility["minimum_panel_fraction"]
    ):
        raise ValueError("minimum panel count does not equal the frozen fraction rule")

    partition = payload["partition_contract"]
    _require_keys(
        partition,
        {
            "algorithm",
            "namespace",
            "modulus",
            "validation_buckets",
            "discovery_buckets",
            "legacy_outcomes_used_for_partition",
            "validation_locked_until_model_and_metric_contract_frozen",
            "operator_prior_outcome_blinding_asserted",
        },
        "partition contract",
    )
    if (
        partition["algorithm"] != PARTITION_ALGORITHM
        or not isinstance(partition["namespace"], str)
        or not partition["namespace"]
        or partition["modulus"] != 5
        or partition["validation_buckets"] != [0]
        or partition["discovery_buckets"] != [1, 2, 3, 4]
        or partition["legacy_outcomes_used_for_partition"] is not False
        or partition["validation_locked_until_model_and_metric_contract_frozen"] is not True
        or partition["operator_prior_outcome_blinding_asserted"] is not False
    ):
        raise ValueError("trait partition contract differs")

    future = payload["future_benchmark_contract"]
    required_future = {
        "models_executed_in_this_stage": False,
        "trait_selection": "all_mechanically_eligible_traits_no_top_n",
        "discovery_validation_order": (
            "freeze_model_metric_and_missingness_contract_on_discovery_before_validation"
        ),
        "legacy_203_outputs_accepted_as_baselines": False,
        "claims": "retrospective_technical_benchmark_only",
    }
    if future != required_future:
        raise ValueError("future benchmark boundary differs")
    return payload


def _verify_asset(path: Path, specification: dict[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != specification["bytes"]:
        raise ValueError(f"{label} byte count differs from the frozen asset")
    observed_sha = _sha256(path)
    if observed_sha != specification["sha256"]:
        raise ValueError(f"{label} SHA-256 differs from the frozen asset")
    return {
        "logical_path": specification["logical_path"],
        "bytes": observed_bytes,
        "sha256": observed_sha,
        "qualified": True,
    }


def _read_fam(path: Path, config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    iids: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"FAM line {line_number} does not contain six fields")
        if fields[0] != config["expected_fid"]:
            raise ValueError(f"FAM line {line_number} has an unexpected FID")
        if not fields[1] or any(character in fields[1] for character in "\t\r\n"):
            raise ValueError(f"FAM line {line_number} has an unsafe IID")
        iids.append(fields[1])
    if len(iids) != config["expected_samples"] or len(iids) != len(set(iids)):
        raise ValueError("FAM does not contain exactly 209 unique IIDs")
    ordered_sha = _ordered_text_sha256(iids)
    if ordered_sha != config["expected_ordered_iid_sha256"]:
        raise ValueError("FAM ordered IID hash differs from the frozen panel")
    return iids, {
        "samples": len(iids),
        "ordered_iid_sha256": ordered_sha,
        "first_iid": iids[0],
        "last_iid": iids[-1],
    }


@dataclass
class TraitStats:
    rows: int = 0
    finite_rows: int = 0
    nonfinite_rows: int = 0
    invalid_required_field_rows: int = 0
    duplicate_trait_strain_rows: int = 0
    strains: set[str] = field(default_factory=set)
    finite_panel_strains: set[str] = field(default_factory=set)
    species: set[str] = field(default_factory=set)


def _read_compendium(
    path: Path,
    panel_iids: set[str],
    source_config: dict[str, Any],
    eligibility: dict[str, Any],
    partition: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[str]]:
    traits: dict[str, TraitStats] = {}
    submitters: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()
    all_strains: set[str] = set()
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("phenotype compendium is empty") from error
        if header != EXPECTED_SOURCE_HEADER:
            raise ValueError("phenotype compendium header differs")
        for row_count, row in enumerate(reader, 1):
            if len(row) != len(EXPECTED_SOURCE_HEADER):
                raise ValueError(f"phenotype row {row_count + 1} has the wrong field count")
            submitted_by, species, trait_name, strain_name, raw_value = row
            if not trait_name or any(character in trait_name for character in "\t\r\n"):
                raise ValueError(f"phenotype row {row_count + 1} has an unsafe trait name")
            if any(character in strain_name for character in "\t\r\n"):
                raise ValueError(f"phenotype row {row_count + 1} has an unsafe strain name")
            stats = traits.setdefault(trait_name, TraitStats())
            stats.rows += 1
            stats.species.add(species)
            submitters[submitted_by] += 1
            species_counts[species] += 1
            all_strains.add(strain_name)
            required_fields_valid = all(row)
            if not required_fields_valid:
                stats.invalid_required_field_rows += 1
            if strain_name in stats.strains:
                stats.duplicate_trait_strain_rows += 1
            stats.strains.add(strain_name)
            try:
                value_is_finite = math.isfinite(float(raw_value))
            except ValueError:
                value_is_finite = False
            if value_is_finite:
                stats.finite_rows += 1
                if strain_name in panel_iids:
                    stats.finite_panel_strains.add(strain_name)
            else:
                stats.nonfinite_rows += 1

    if row_count != source_config["expected_data_rows"]:
        raise ValueError(
            f"phenotype row count differs: expected {source_config['expected_data_rows']}, "
            f"got {row_count}"
        )
    if len(traits) != source_config["expected_trait_count"]:
        raise ValueError("phenotype trait count differs from the frozen source")

    trait_manifest: list[dict[str, Any]] = []
    discovery: list[str] = []
    validation: list[str] = []
    for trait_name in sorted(traits):
        stats = traits[trait_name]
        reasons: list[str] = []
        if stats.species != {eligibility["required_species"]}:
            reasons.append("required_species_mismatch")
        if stats.invalid_required_field_rows:
            reasons.append("empty_required_field")
        if stats.nonfinite_rows:
            reasons.append("nonfinite_trait_value")
        if stats.duplicate_trait_strain_rows:
            reasons.append("duplicate_trait_strain_row")
        if len(stats.finite_panel_strains) < eligibility["minimum_panel_samples"]:
            reasons.append("panel_overlap_below_minimum")
        eligible = not reasons
        digest = hashlib.sha256(
            (partition["namespace"] + trait_name).encode("utf-8")
        ).hexdigest()
        bucket = int(digest, 16) % partition["modulus"]
        split = "validation" if bucket in partition["validation_buckets"] else "discovery"
        if eligible:
            (validation if split == "validation" else discovery).append(trait_name)
        trait_manifest.append(
            {
                "trait_name": trait_name,
                "trait_name_sha256": hashlib.sha256(trait_name.encode("utf-8")).hexdigest(),
                "partition_sha256": digest,
                "partition_bucket": bucket,
                "split": split,
                "rows": stats.rows,
                "unique_strains": len(stats.strains),
                "finite_rows": stats.finite_rows,
                "nonfinite_rows": stats.nonfinite_rows,
                "duplicate_trait_strain_rows": stats.duplicate_trait_strain_rows,
                "finite_exact_panel_strains": len(stats.finite_panel_strains),
                "eligible": eligible,
                "eligibility_reasons": ";".join(reasons) if reasons else "eligible",
            }
        )

    census_traits = [item["trait_name"] for item in trait_manifest]
    eligible_traits = discovery + validation
    summary = {
        "header": EXPECTED_SOURCE_HEADER,
        "data_rows": row_count,
        "traits": len(traits),
        "trait_census_sha256": _ordered_text_sha256(census_traits),
        "unique_strains": len(all_strains),
        "strain_census_sha256": _ordered_text_sha256(sorted(all_strains)),
        "submitted_by_counts": dict(sorted(submitters.items())),
        "species_counts": dict(sorted(species_counts.items())),
        "trait_row_count_histogram": {
            str(key): value
            for key, value in sorted(Counter(item["rows"] for item in trait_manifest).items())
        },
        "trait_panel_overlap_histogram": {
            str(key): value
            for key, value in sorted(
                Counter(item["finite_exact_panel_strains"] for item in trait_manifest).items()
            )
        },
        "eligible_traits": len(eligible_traits),
        "eligible_trait_sha256": _ordered_text_sha256(sorted(eligible_traits)),
        "discovery_traits": len(discovery),
        "discovery_trait_sha256": _ordered_text_sha256(discovery),
        "validation_traits": len(validation),
        "validation_trait_sha256": _ordered_text_sha256(validation),
        "traits_ineligible_by_reason": dict(
            sorted(
                Counter(
                    reason
                    for item in trait_manifest
                    for reason in (
                        []
                        if item["eligibility_reasons"] == "eligible"
                        else item["eligibility_reasons"].split(";")
                    )
                ).items()
            )
        ),
    }
    return summary, trait_manifest, discovery, validation


def _audit_legacy(
    path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    trait_ids: list[str] = []
    observed_n: Counter[int] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"legacy quarantine file is empty: {config['id']}") from error
        if header != config["expected_header"]:
            raise ValueError(f"legacy quarantine header differs: {config['id']}")
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(
                    f"legacy quarantine row has wrong field count: {config['id']}:{line_number}"
                )
            if not row[0] or row[0] in trait_ids:
                raise ValueError(
                    f"legacy quarantine trait IDs are empty or repeated: {config['id']}"
                )
            trait_ids.append(row[0])
            try:
                observed_n[int(row[1])] += 1
            except ValueError as error:
                raise ValueError(f"legacy quarantine n is not integral: {config['id']}") from error
    if len(trait_ids) != config["expected_rows"]:
        raise ValueError(f"legacy quarantine row count differs: {config['id']}")
    observed_histogram = {str(key): value for key, value in sorted(observed_n.items())}
    if observed_histogram != config["expected_n_histogram"]:
        raise ValueError(f"legacy quarantine n histogram differs: {config['id']}")
    if max(observed_n) != config["legacy_panel_sample_ceiling"]:
        raise ValueError(f"legacy quarantine panel ceiling differs: {config['id']}")
    return {
        "id": config["id"],
        "rows": len(trait_ids),
        "legacy_panel_sample_ceiling": config["legacy_panel_sample_ceiling"],
        "n_histogram": observed_histogram,
        "ordered_trait_sha256": _ordered_text_sha256(trait_ids),
        "trait_set_sha256": _ordered_text_sha256(sorted(trait_ids)),
        "numerical_result_fields_used_for_design": False,
        "permitted_use": "fingerprint_and_structural_quarantine_only",
        "accepted_as_benchmark_input": False,
        "accepted_as_preliminary_evidence": False,
    }, trait_ids


def _trait_manifest_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    columns = [
        "trait_name",
        "trait_name_sha256",
        "partition_sha256",
        "partition_bucket",
        "split",
        "rows",
        "unique_strains",
        "finite_rows",
        "nonfinite_rows",
        "duplicate_trait_strain_rows",
        "finite_exact_panel_strains",
        "eligible",
        "eligibility_reasons",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        fields = [
            str(row[column]).lower()
            if isinstance(row[column], bool)
            else str(row[column])
            for column in columns
        ]
        lines.append("\t".join(fields))
    return ("\n".join(lines) + "\n").encode("utf-8")


def qualify_compendium(
    config_path: str | Path,
    phenotype_path: str | Path,
    fam_path: str | Path,
    legacy_h2_screen_path: str | Path,
    legacy_h2_greml_path: str | Path,
    legacy_gblup_top15_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify sources and atomically emit a model-free qualification receipt."""

    config_source = Path(config_path).resolve()
    if not config_source.is_file():
        raise FileNotFoundError(f"qualification config is not a file: {config_source}")
    config = _load_config(config_source)
    phenotype = Path(phenotype_path).resolve()
    fam = Path(fam_path).resolve()
    legacy_paths = [
        Path(legacy_h2_screen_path).resolve(),
        Path(legacy_h2_greml_path).resolve(),
        Path(legacy_gblup_top15_path).resolve(),
    ]
    source_asset = _verify_asset(phenotype, config["source"]["asset"], "phenotype compendium")
    panel_asset = _verify_asset(fam, config["genotype_panel"]["asset"], "209-strain FAM")
    legacy_assets = [
        _verify_asset(path, item["asset"], item["id"])
        for path, item in zip(legacy_paths, config["legacy_quarantine"], strict=True)
    ]
    panel_iids, panel_summary = _read_fam(fam, config["genotype_panel"])
    compendium, trait_manifest, discovery, validation = _read_compendium(
        phenotype,
        set(panel_iids),
        config["source"],
        config["eligibility_contract"],
        config["partition_contract"],
    )
    legacy_audits: list[dict[str, Any]] = []
    legacy_traits: list[list[str]] = []
    for path, item, asset in zip(
        legacy_paths, config["legacy_quarantine"], legacy_assets, strict=True
    ):
        audit, traits = _audit_legacy(path, item)
        audit["asset"] = asset
        legacy_audits.append(audit)
        legacy_traits.append(traits)
    source_trait_set = {item["trait_name"] for item in trait_manifest}
    h2_screen_set = set(legacy_traits[0])
    h2_greml_set = set(legacy_traits[1])
    if h2_screen_set != h2_greml_set:
        raise ValueError("legacy 203-sample h2 trait rosters differ from each other")
    if not h2_screen_set.issubset(source_trait_set):
        raise ValueError("legacy 203-sample h2 roster is not a subset of the source census")
    if not set(legacy_traits[2]).issubset(h2_screen_set):
        raise ValueError("legacy top-15 roster is not a subset of the source trait census")
    validation_set = set(validation)
    top15_set = set(legacy_traits[2])
    source_only_traits = sorted(source_trait_set - h2_screen_set)

    receipt = {
        "schema_version": RECEIPT_VERSION,
        "analysis_id": config["analysis_id"],
        "classification": "retrospective_source_qualification_only",
        "biological_claims_permitted": False,
        "models_executed": False,
        "config": {
            "bytes": config_source.stat().st_size,
            "sha256": _sha256(config_source),
        },
        "source_asset": source_asset,
        "genotype_panel_asset": panel_asset,
        "genotype_panel": panel_summary,
        "compendium": compendium,
        "eligibility_contract": config["eligibility_contract"],
        "qualification_history": config["qualification_history"],
        "partition_contract": config["partition_contract"],
        "future_benchmark_contract": config["future_benchmark_contract"],
        "legacy_quarantine": legacy_audits,
        "legacy_quarantine_boundary": {
            "legacy_result_values_used_for_eligibility": False,
            "legacy_result_values_used_for_partition": False,
            "legacy_outputs_accepted_as_preliminary_evidence": False,
            "legacy_outputs_accepted_as_benchmark_baselines": False,
            "legacy_h2_trait_roster_count": len(h2_screen_set),
            "legacy_h2_trait_roster_is_source_subset": True,
            "source_traits_absent_from_legacy_h2_count": len(source_only_traits),
            "source_traits_absent_from_legacy_h2": source_only_traits,
            "source_traits_absent_from_legacy_h2_sha256": _ordered_text_sha256(
                source_only_traits
            ),
            "top15_validation_partition_overlap_count": len(top15_set & validation_set),
            "top15_discovery_partition_overlap_count": len(top15_set - validation_set),
            "validation_partition_is_historically_outcome_naive": False,
            "reason_not_outcome_naive": (
                "legacy h2 results exist for 1,458 of the 1,461 source traits"
            ),
        },
        "claim_boundary": {
            "permitted": [
                "source and schema qualification",
                "mechanical availability-based trait census",
                "prospective-from-freeze discovery-validation partition identity",
            ],
            "prohibited": [
                "heritability inference",
                "predictive performance inference",
                "trait prioritization from legacy top-N outputs",
                "biological discovery or validation",
            ],
        },
    }

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"qualification output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        files = {
            "source_qualification_receipt.json": _canonical_json(receipt),
            "trait_manifest.tsv": _trait_manifest_bytes(trait_manifest),
            "eligible_discovery_traits.txt": (
                "\n".join(discovery) + "\n"
            ).encode("utf-8"),
            "eligible_validation_traits.txt": (
                "\n".join(validation) + "\n"
            ).encode("utf-8"),
            "LEGACY_QUARANTINE.json": _canonical_json(
                {
                    "schema_version": RECEIPT_VERSION,
                    "legacy_outputs": legacy_audits,
                    "permitted_use": "fingerprint_and_structural_quarantine_only",
                    "prohibited_as_model_or_evidence_input": True,
                }
            ),
        }
        for name, content in files.items():
            (temporary / name).write_bytes(content)
        manifest = {
            name: {"bytes": (temporary / name).stat().st_size, "sha256": _sha256(temporary / name)}
            for name in sorted(files)
        }
        (temporary / "MANIFEST.json").write_bytes(_canonical_json(manifest))
        (temporary / "SUCCESS").write_text("qualified\n", encoding="utf-8")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phenotypes", required=True)
    parser.add_argument("--fam", required=True)
    parser.add_argument("--legacy-h2-screen", required=True)
    parser.add_argument("--legacy-h2-greml", required=True)
    parser.add_argument("--legacy-gblup-top15", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    receipt = qualify_compendium(
        args.config,
        args.phenotypes,
        args.fam,
        args.legacy_h2_screen,
        args.legacy_h2_greml,
        args.legacy_gblup_top15,
        args.output_dir,
    )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
