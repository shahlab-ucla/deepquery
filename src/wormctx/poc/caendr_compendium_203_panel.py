"""Qualify a fresh phenotype-free 203-strain CaeNDR genotype panel and kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CONFIG_VERSION = "wormctx-caendr-compendium-203-kernel-config-1.0"
PREFLIGHT_VERSION = "wormctx-caendr-compendium-203-kernel-preflight-1.0"
RECEIPT_VERSION = "wormctx-caendr-compendium-203-kernel-qualification-1.0"
EXPECTED_ROLES = [
    "source_vcf",
    "source_vcf_index",
    "identity_only_legacy_roster",
    "stage1_qualification_receipt",
    "stage1_discovery_trait_list",
    "stage1_validation_trait_list",
]
EXPECTED_TOOLS = ["bcftools", "plink2", "gcta64", "python"]


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


def _require_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _asset_spec(payload: dict[str, Any], label: str) -> None:
    _require_keys(payload, {"role", "logical_path", "bytes", "sha256"}, label)
    if not payload["role"] or not payload["logical_path"]:
        raise ValueError(f"{label} identifiers are empty")
    if not isinstance(payload["bytes"], int) or payload["bytes"] <= 0:
        raise ValueError(f"{label} byte count is invalid")
    if not _valid_sha(payload["sha256"]):
        raise ValueError(f"{label} SHA-256 is invalid")


def load_kernel_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require_keys(
        payload,
        {
            "schema_version",
            "analysis_id",
            "classification",
            "biological_claims_permitted",
            "phenotype_paths_accepted_by_kernel_runner",
            "inputs",
            "source_vcf_contract",
            "roster_contract",
            "partition_identity",
            "qc_contract",
            "kernel_contract",
            "group_contract",
            "tools",
            "legacy_numeric_quarantine",
        },
        "kernel config",
    )
    if payload["schema_version"] != CONFIG_VERSION:
        raise ValueError("203-kernel config schema differs")
    if payload["classification"] != "phenotype_free_genotype_kernel_qualification":
        raise ValueError("203-kernel classification differs")
    if payload["biological_claims_permitted"] is not False:
        raise ValueError("kernel qualification cannot permit biological claims")
    if payload["phenotype_paths_accepted_by_kernel_runner"] is not False:
        raise ValueError("kernel runner cannot accept phenotype paths")

    inputs = payload["inputs"]
    if [item.get("role") for item in inputs] != EXPECTED_ROLES:
        raise ValueError("203-kernel input roles or order differ")
    for item in inputs:
        _asset_spec(item, f"input {item.get('role')}")

    source_contract = payload["source_vcf_contract"]
    if source_contract != {
        "release": "CaeNDR_20250625",
        "assembly": "PRJNA13758.WS283_inferred_not_declared_in_vcf_header",
        "expected_samples": 684,
        "ordered_sample_sha256": source_contract.get("ordered_sample_sha256"),
        "input_chromosomes": ["I", "II", "III", "IV", "V", "X", "MtDNA"],
        "analysis_chromosome_map": {
            "I": 1,
            "II": 2,
            "III": 3,
            "IV": 4,
            "V": 5,
            "X": 6,
        },
        "mtdna_excluded": True,
    } or not _valid_sha(source_contract["ordered_sample_sha256"]):
        raise ValueError("source VCF contract differs")

    roster = payload["roster_contract"]
    _require_keys(
        roster,
        {
            "expected_samples",
            "expected_header",
            "expected_fid",
            "ordered_iid_sha256",
            "set_must_equal_vcf_compendium_intersection",
            "legacy_numeric_fields_accepted",
            "identity_roster_only",
        },
        "roster contract",
    )
    if (
        roster["expected_samples"] != 203
        or roster["expected_header"] != ["#FID", "IID"]
        or roster["expected_fid"] != "0"
        or not _valid_sha(roster["ordered_iid_sha256"])
        or roster["set_must_equal_vcf_compendium_intersection"] is not True
        or roster["legacy_numeric_fields_accepted"] is not False
        or roster["identity_roster_only"] is not True
    ):
        raise ValueError("203-strain roster contract differs")

    partition = payload["partition_identity"]
    if partition != {
        "stage1_schema_version": "wormctx-caendr-compendium-source-qualification-1.1",
        "eligible_traits": 1445,
        "discovery_traits": 1184,
        "validation_traits": 261,
        "discovery_trait_sha256": partition.get("discovery_trait_sha256"),
        "validation_trait_sha256": partition.get("validation_trait_sha256"),
        "trait_bucket_identity_must_not_change": True,
        "validation_outcomes_locked": True,
    } or not all(
        _valid_sha(partition[key])
        for key in ("discovery_trait_sha256", "validation_trait_sha256")
    ):
        raise ValueError("stage-1 partition identity differs")

    qc = payload["qc_contract"]
    expected_qc = {
        "chromosomes": [1, 2, 3, 4, 5, 6],
        "snps_only_just_acgt": True,
        "biallelic_max_alleles": 2,
        "duplicate_marker_policy": "exclude_all",
        "variant_id_template": "@:#",
        "mind": 0.05,
        "geno": 0.05,
        "maf": 0.05,
        "hardy_weinberg_filter": False,
        "phenotype_directed_filtering": False,
        "minimum_qualified_markers": qc.get("minimum_qualified_markers"),
    }
    if (
        qc != expected_qc
        or not isinstance(qc["minimum_qualified_markers"], int)
        or qc["minimum_qualified_markers"] <= 10
    ):
        raise ValueError("203-panel marker-QC contract differs")

    kernel = payload["kernel_contract"]
    if kernel != {
        "method": "gcta_make_grm_alg0",
        "autosome_num": 6,
        "samples": 203,
        "pairwise_marker_counts_must_equal_marker_count": True,
        "emit_float64_npy": True,
        "psd_absolute_tolerance": 1e-5,
    }:
        raise ValueError("203-panel kernel contract differs")

    group = payload["group_contract"]
    if group != {
        "pca_ld_pruning": {
            "window": "500kb",
            "step_variants": 1,
            "r2": 0.2,
            "indep_order": 2,
        },
        "pc_count": 10,
        "algorithm": (
            "standardized_pc10_deterministic_farthest_first_kmeans_silhouette_v1"
        ),
        "candidate_k": list(range(5, 13)),
        "minimum_group_size": 5,
        "phenotype_values_accessed": False,
        "labels_are_external_ancestry_assignments": False,
    }:
        raise ValueError("203-panel population-group contract differs")

    tools = payload["tools"]
    if [item.get("id") for item in tools] != EXPECTED_TOOLS:
        raise ValueError("203-kernel tool inventory differs")
    for item in tools:
        _require_keys(item, {"id", "version", "sha256"}, f"tool {item.get('id')}")
        if not item["version"] or not _valid_sha(item["sha256"]):
            raise ValueError(f"tool contract is invalid: {item['id']}")

    quarantine = payload["legacy_numeric_quarantine"]
    required_ids = {
        "legacy_relationship_matrix",
        "legacy_pca_eigenvectors",
        "legacy_pca_eigenvalues",
        "legacy_h2_screen",
        "legacy_h2_greml",
        "legacy_gblup_top15",
    }
    if {item.get("id") for item in quarantine} != required_ids:
        raise ValueError("legacy numeric quarantine inventory differs")
    for item in quarantine:
        _require_keys(
            item,
            {"id", "bytes", "sha256", "accepted_as_input", "permitted_use"},
            f"quarantine {item.get('id')}",
        )
        if (
            item["bytes"] <= 0
            or not _valid_sha(item["sha256"])
            or item["accepted_as_input"] is not False
            or item["permitted_use"] != "identity_and_exclusion_receipt_only"
        ):
            raise ValueError(f"legacy numeric quarantine differs: {item['id']}")
    return payload


def _by_role(config: dict[str, Any], role: str) -> dict[str, Any]:
    return next(item for item in config["inputs"] if item["role"] == role)


def _verify(path: Path, specification: dict[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    if path.stat().st_size != specification["bytes"]:
        raise ValueError(f"{label} byte count differs")
    digest = _sha256(path)
    if digest != specification["sha256"]:
        raise ValueError(f"{label} SHA-256 differs")
    return {"bytes": path.stat().st_size, "sha256": digest, "qualified": True}


def _read_nonempty_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line or "\t" in line or "\r" in line for line in lines):
        raise ValueError(f"line-list asset is empty or malformed: {path}")
    if len(lines) != len(set(lines)):
        raise ValueError(f"line-list asset contains duplicates: {path}")
    return lines


def _read_roster(path: Path, config: dict[str, Any]) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].split() != config["expected_header"]:
        raise ValueError("identity-only roster header differs")
    iids: list[str] = []
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split()
        if len(fields) != 2 or fields[0] != config["expected_fid"] or not fields[1]:
            raise ValueError(f"identity-only roster line {line_number} is invalid")
        iids.append(fields[1])
    if len(iids) != config["expected_samples"] or len(iids) != len(set(iids)):
        raise ValueError("identity-only roster does not contain 203 unique IIDs")
    if _ordered_hash(iids) != config["ordered_iid_sha256"]:
        raise ValueError("identity-only roster order differs")
    return iids


def prepare_kernel_inputs(
    config_path: str | Path,
    source_vcf_path: str | Path,
    source_vcf_index_path: str | Path,
    roster_path: str | Path,
    stage1_receipt_path: str | Path,
    discovery_list_path: str | Path,
    validation_list_path: str | Path,
    vcf_sample_list_path: str | Path,
    tool_paths: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify all phenotype-free inputs and emit normalized roster/build controls."""

    config_source = Path(config_path).resolve()
    config = load_kernel_config(config_source)
    paths = {
        "source_vcf": Path(source_vcf_path).resolve(),
        "source_vcf_index": Path(source_vcf_index_path).resolve(),
        "identity_only_legacy_roster": Path(roster_path).resolve(),
        "stage1_qualification_receipt": Path(stage1_receipt_path).resolve(),
        "stage1_discovery_trait_list": Path(discovery_list_path).resolve(),
        "stage1_validation_trait_list": Path(validation_list_path).resolve(),
    }
    verified_inputs = {
        role: _verify(path, _by_role(config, role), role)
        for role, path in paths.items()
    }
    if len(tool_paths) != len(EXPECTED_TOOLS):
        raise ValueError("tool path count differs")
    verified_tools: dict[str, Any] = {}
    for specification, raw_path in zip(config["tools"], tool_paths, strict=True):
        path = Path(raw_path).resolve()
        if not path.is_file() or _sha256(path) != specification["sha256"]:
            raise ValueError(f"tool binary differs: {specification['id']}")
        verified_tools[specification["id"]] = {
            "sha256": specification["sha256"],
            "version": specification["version"],
        }

    roster = _read_roster(
        paths["identity_only_legacy_roster"], config["roster_contract"]
    )
    vcf_samples = _read_nonempty_lines(Path(vcf_sample_list_path).resolve())
    source_contract = config["source_vcf_contract"]
    if (
        len(vcf_samples) != source_contract["expected_samples"]
        or _ordered_hash(vcf_samples) != source_contract["ordered_sample_sha256"]
    ):
        raise ValueError("VCF sample header differs from the frozen 684-isotype order")
    if not set(roster).issubset(vcf_samples):
        raise ValueError("203 identity roster is not a subset of the VCF samples")
    vcf_ordered_roster = [iid for iid in vcf_samples if iid in set(roster)]
    if vcf_ordered_roster != roster:
        raise ValueError("identity roster does not equal the VCF-ordered 203 subset")

    discovery = _read_nonempty_lines(paths["stage1_discovery_trait_list"])
    validation = _read_nonempty_lines(paths["stage1_validation_trait_list"])
    partition = config["partition_identity"]
    if (
        len(discovery) != partition["discovery_traits"]
        or len(validation) != partition["validation_traits"]
        or _ordered_hash(discovery) != partition["discovery_trait_sha256"]
        or _ordered_hash(validation) != partition["validation_trait_sha256"]
        or set(discovery) & set(validation)
    ):
        raise ValueError("stage-1 trait partition lists differ")
    stage1 = json.loads(
        paths["stage1_qualification_receipt"].read_text(encoding="utf-8")
    )
    if (
        stage1.get("schema_version") != partition["stage1_schema_version"]
        or stage1.get("compendium", {}).get("discovery_traits") != len(discovery)
        or stage1.get("compendium", {}).get("validation_traits") != len(validation)
        or stage1.get("compendium", {}).get("discovery_trait_sha256")
        != _ordered_hash(discovery)
        or stage1.get("compendium", {}).get("validation_trait_sha256")
        != _ordered_hash(validation)
        or stage1.get("models_executed") is not False
    ):
        raise ValueError(
            "stage-1 qualification receipt does not bind the trait partition"
        )

    receipt = {
        "schema_version": PREFLIGHT_VERSION,
        "analysis_id": config["analysis_id"],
        "phenotype_paths_accepted": False,
        "phenotype_values_accessed": False,
        "legacy_numeric_assets_accessed": False,
        "inputs": verified_inputs,
        "tools": verified_tools,
        "source_vcf_samples": len(vcf_samples),
        "source_vcf_ordered_sample_sha256": _ordered_hash(vcf_samples),
        "roster_samples": len(roster),
        "roster_ordered_iid_sha256": _ordered_hash(roster),
        "partition_identity": partition,
    }
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"preflight output already exists: {output}")
    output.mkdir(parents=True)
    (output / "roster.samples.txt").write_text(
        "\n".join(roster) + "\n", encoding="utf-8"
    )
    (output / "roster.keep.tsv").write_text(
        "".join(f"0\t{iid}\n" for iid in roster), encoding="utf-8"
    )
    chromosome_map = config["source_vcf_contract"]["analysis_chromosome_map"]
    (output / "rename_chromosomes.tsv").write_text(
        "".join(f"{key}\t{value}\n" for key, value in chromosome_map.items()),
        encoding="utf-8",
    )
    (output / "preflight_receipt.json").write_bytes(_canonical_json(receipt))
    return receipt


def _read_fam(path: Path, expected_iids: Sequence[str]) -> list[str]:
    iids: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split()
        if len(fields) != 6 or fields[0] != "0":
            raise ValueError(f"fresh FAM line {line_number} differs")
        iids.append(fields[1])
    if iids != list(expected_iids):
        raise ValueError("fresh FAM does not exactly preserve the 203 roster order")
    return iids


def _read_bim(path: Path, minimum_markers: int) -> tuple[list[str], dict[str, int]]:
    marker_ids: list[str] = []
    chromosome_counts: Counter[str] = Counter()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"fresh BIM line {line_number} differs")
        chromosome, marker, _cm, position, allele1, allele2 = fields
        if chromosome not in {"1", "2", "3", "4", "5", "6"}:
            raise ValueError("fresh BIM contains a non-analysis chromosome")
        if not position.isdigit() or int(position) <= 0:
            raise ValueError("fresh BIM position is invalid")
        if (
            allele1 not in {"A", "C", "G", "T"}
            or allele2 not in {"A", "C", "G", "T"}
            or allele1 == allele2
        ):
            raise ValueError("fresh BIM contains a non-biallelic ACGT marker")
        marker_ids.append(marker)
        chromosome_counts[chromosome] += 1
    if len(marker_ids) < minimum_markers or len(marker_ids) != len(set(marker_ids)):
        raise ValueError("fresh panel has too few markers or duplicate marker IDs")
    if set(chromosome_counts) != {"1", "2", "3", "4", "5", "6"}:
        raise ValueError("fresh panel does not contain exactly chromosomes 1-6")
    return marker_ids, dict(
        sorted(chromosome_counts.items(), key=lambda item: int(item[0]))
    )


def _read_plink_table(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"PLINK table is empty: {path}")
    return lines[0].split(), [line.split() for line in lines[1:]]


def _verify_qc_tables(
    marker_ids: Sequence[str],
    iids: Sequence[str],
    afreq_path: Path,
    vmiss_path: Path,
    smiss_path: Path,
    maf: float,
    geno: float,
    mind: float,
) -> dict[str, Any]:
    afreq_header, afreq_rows = _read_plink_table(afreq_path)
    base_afreq_header = ["#CHROM", "ID", "REF", "ALT", "ALT_FREQS", "OBS_CT"]
    provisional_afreq_header = [
        "#CHROM",
        "ID",
        "REF",
        "ALT",
        "PROVISIONAL_REF?",
        "ALT_FREQS",
        "OBS_CT",
    ]
    if afreq_header == base_afreq_header:
        frequency_index = 4
    elif afreq_header == provisional_afreq_header:
        frequency_index = 5
    else:
        raise ValueError("PLINK allele-frequency header differs")
    if any(len(row) != len(afreq_header) for row in afreq_rows):
        raise ValueError("allele-frequency row differs")
    if [row[1] for row in afreq_rows] != list(marker_ids):
        raise ValueError("allele-frequency marker order differs from the BIM")
    minor_frequencies: list[float] = []
    for row in afreq_rows:
        if frequency_index == 5 and row[4] not in {"Y", "N"}:
            raise ValueError("PLINK provisional-reference indicator differs")
        frequency = float(row[frequency_index])
        if not math.isfinite(frequency):
            raise ValueError("allele-frequency value is nonfinite")
        minor_frequencies.append(min(frequency, 1.0 - frequency))
    if min(minor_frequencies) + 1e-12 < maf:
        raise ValueError("fresh panel violates the frozen MAF threshold")

    vmiss_header, vmiss_rows = _read_plink_table(vmiss_path)
    if vmiss_header != ["#CHROM", "ID", "MISSING_CT", "OBS_CT", "F_MISS"]:
        raise ValueError("PLINK variant-missingness header differs")
    if any(len(row) != len(vmiss_header) for row in vmiss_rows):
        raise ValueError("variant-missingness row differs")
    if [row[1] for row in vmiss_rows] != list(marker_ids):
        raise ValueError("variant-missingness marker order differs from the BIM")
    variant_missingness = [float(row[4]) for row in vmiss_rows]
    if any(
        not math.isfinite(value) or value > geno + 1e-12
        for value in variant_missingness
    ):
        raise ValueError("fresh panel violates the frozen variant-missingness threshold")

    smiss_header, smiss_rows = _read_plink_table(smiss_path)
    if smiss_header != ["#FID", "IID", "MISSING_CT", "OBS_CT", "F_MISS"]:
        raise ValueError("PLINK sample-missingness header differs")
    if any(len(row) != len(smiss_header) for row in smiss_rows):
        raise ValueError("sample-missingness row differs")
    if [row[1] for row in smiss_rows] != list(iids):
        raise ValueError("sample-missingness order differs from the FAM")
    sample_missingness = [float(row[4]) for row in smiss_rows]
    if any(
        not math.isfinite(value) or value > mind + 1e-12
        for value in sample_missingness
    ):
        raise ValueError("fresh panel violates the frozen sample-missingness threshold")
    return {
        "allele_frequency_header": afreq_header,
        "provisional_reference_column_present": frequency_index == 5,
        "minimum_minor_allele_frequency": min(minor_frequencies),
        "maximum_variant_missingness": max(variant_missingness),
        "maximum_sample_missingness": max(sample_missingness),
    }


def _load_pca(path: Path, expected_iids: Sequence[str]) -> np.ndarray:
    header, rows = _read_plink_table(path)
    if header != ["#FID", "IID", *[f"PC{index}" for index in range(1, 11)]]:
        raise ValueError("PCA score header differs")
    if [row[1] for row in rows] != list(expected_iids):
        raise ValueError("PCA score order differs from the 203 roster")
    values = np.asarray(
        [[float(value) for value in row[2:]] for row in rows], dtype=np.float64
    )
    if values.shape != (len(expected_iids), 10) or not np.all(np.isfinite(values)):
        raise ValueError("PCA score matrix differs")
    return values


def _load_eigenvalues(path: Path) -> np.ndarray:
    values = np.asarray(
        [float(line) for line in path.read_text(encoding="utf-8").splitlines()],
        dtype=np.float64,
    )
    if (
        values.shape != (10,)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
        or np.any(values[1:] > values[:-1])
    ):
        raise ValueError("PCA eigenvalues differ")
    return values


def _canonicalize_labels(assignments: np.ndarray, ids: Sequence[str]) -> np.ndarray:
    clusters = sorted(
        np.unique(assignments).tolist(),
        key=lambda group: min(
            ids[index] for index in np.where(assignments == group)[0]
        ),
    )
    mapping = {old: new for new, old in enumerate(clusters)}
    return np.asarray([mapping[int(item)] for item in assignments], dtype=np.int64)


def _kmeans(values: np.ndarray, ids: Sequence[str], k: int) -> np.ndarray:
    norms = np.einsum("ij,ij->i", values, values)
    first = min(range(len(ids)), key=lambda index: (float(norms[index]), ids[index]))
    centers = [values[first].copy()]
    chosen = {first}
    while len(centers) < k:
        distances = np.stack(
            [
                np.einsum("ij,ij->i", values - center, values - center)
                for center in centers
            ],
            axis=1,
        )
        nearest = np.min(distances, axis=1)
        remaining = [index for index in range(len(ids)) if index not in chosen]
        selected = min(
            remaining, key=lambda index: (-float(nearest[index]), ids[index])
        )
        chosen.add(selected)
        centers.append(values[selected].copy())
    center_matrix = np.stack(centers)
    previous: np.ndarray | None = None
    for _ in range(300):
        distances = np.sum(
            (values[:, None, :] - center_matrix[None, :, :]) ** 2, axis=2
        )
        assignments = np.argmin(distances, axis=1)
        if previous is not None and np.array_equal(assignments, previous):
            break
        if len(np.unique(assignments)) != k:
            raise ValueError(
                f"deterministic k-means produced an empty cluster for k={k}"
            )
        previous = assignments.copy()
        center_matrix = np.stack(
            [values[assignments == group].mean(axis=0) for group in range(k)]
        )
    else:
        raise ValueError(f"deterministic k-means failed to converge for k={k}")
    return _canonicalize_labels(assignments, ids)


def _silhouette(values: np.ndarray, assignments: np.ndarray) -> float:
    distances = np.sqrt(np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2))
    scores: list[float] = []
    for index, group in enumerate(assignments):
        same = np.where(assignments == group)[0]
        same = same[same != index]
        if len(same) == 0:
            raise ValueError("silhouette is undefined for singleton groups")
        within = float(distances[index, same].mean())
        between = min(
            float(distances[index, assignments == other].mean())
            for other in np.unique(assignments)
            if other != group
        )
        denominator = max(within, between)
        scores.append(0.0 if denominator == 0.0 else (between - within) / denominator)
    return float(np.mean(scores))


def _derive_groups(
    ids: Sequence[str],
    values: np.ndarray,
    candidate_k: Sequence[int],
    minimum_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    if np.any(~np.isfinite(scales)) or np.any(scales <= np.finfo(np.float64).eps):
        raise ValueError(
            "each PC must have a nonzero finite population standard deviation"
        )
    standardized = (values - means) / scales
    candidates: list[tuple[float, int, np.ndarray]] = []
    evaluations: list[dict[str, Any]] = []
    for k in candidate_k:
        assignments = _kmeans(standardized, ids, k)
        counts = np.bincount(assignments, minlength=k)
        eligible = bool(np.min(counts) >= minimum_size)
        score = _silhouette(standardized, assignments) if eligible else None
        evaluations.append(
            {
                "k": int(k),
                "eligible": eligible,
                "minimum_group_size": int(np.min(counts)),
                "maximum_group_size": int(np.max(counts)),
                "silhouette": score,
            }
        )
        if score is not None:
            candidates.append((score, int(k), assignments))
    if not candidates:
        raise ValueError("no genotype-PC grouping satisfies the frozen minimum size")
    best = min(candidates, key=lambda item: (-round(item[0], 12), item[1]))
    return best[2], evaluations


def _load_gcta_kernel(
    grm_path: Path,
    n_path: Path,
    id_path: Path,
    expected_iids: Sequence[str],
    marker_count: int,
    require_constant_n: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    ids: list[str] = []
    for line in id_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[0] != "0":
            raise ValueError("GCTA GRM ID file differs")
        ids.append(fields[1])
    if ids != list(expected_iids):
        raise ValueError("GCTA GRM IDs differ from the 203 roster")
    expected_entries = len(ids) * (len(ids) + 1) // 2
    triangular = np.fromfile(grm_path, dtype="<f4")
    marker_counts = np.fromfile(n_path, dtype="<f4")
    if triangular.shape != (expected_entries,) or marker_counts.shape != (
        expected_entries,
    ):
        raise ValueError("GCTA triangular GRM payload shape differs")
    if not np.all(np.isfinite(triangular)) or not np.all(np.isfinite(marker_counts)):
        raise ValueError("GCTA GRM payload contains nonfinite values")
    if require_constant_n and not np.all(marker_counts == float(marker_count)):
        raise ValueError(
            "GCTA pairwise marker counts do not equal the qualified marker count"
        )
    matrix = np.empty((len(ids), len(ids)), dtype=np.float64)
    cursor = 0
    for row in range(len(ids)):
        count = row + 1
        matrix[row, :count] = triangular[cursor : cursor + count]
        matrix[:count, row] = triangular[cursor : cursor + count]
        cursor += count
    eigenvalues = np.linalg.eigvalsh(matrix)
    return matrix, {
        "triangular_entries": expected_entries,
        "pairwise_marker_count_min": int(np.min(marker_counts)),
        "pairwise_marker_count_max": int(np.max(marker_counts)),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "diagonal_min": float(np.min(np.diag(matrix))),
        "diagonal_max": float(np.max(np.diag(matrix))),
    }


def qualify_kernel_outputs(
    config_path: str | Path,
    preflight_dir: str | Path,
    bed_path: str | Path,
    bim_path: str | Path,
    fam_path: str | Path,
    afreq_path: str | Path,
    vmiss_path: str | Path,
    smiss_path: str | Path,
    prune_in_path: str | Path,
    pca_path: str | Path,
    eigenval_path: str | Path,
    grm_path: str | Path,
    grm_n_path: str | Path,
    grm_id_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify fresh panel/QC/PCA/GRM outputs and emit compact downstream assets."""

    config_source = Path(config_path).resolve()
    config = load_kernel_config(config_source)
    preflight = Path(preflight_dir).resolve()
    preflight_receipt_path = preflight / "preflight_receipt.json"
    preflight_receipt = json.loads(preflight_receipt_path.read_text(encoding="utf-8"))
    if (
        preflight_receipt.get("schema_version") != PREFLIGHT_VERSION
        or preflight_receipt.get("phenotype_values_accessed") is not False
        or preflight_receipt.get("legacy_numeric_assets_accessed") is not False
    ):
        raise ValueError("kernel preflight receipt differs")
    expected_iids = _read_nonempty_lines(preflight / "roster.samples.txt")
    if _ordered_hash(expected_iids) != config["roster_contract"]["ordered_iid_sha256"]:
        raise ValueError("normalized roster order differs")

    generated = {
        "bed": Path(bed_path).resolve(),
        "bim": Path(bim_path).resolve(),
        "fam": Path(fam_path).resolve(),
        "afreq": Path(afreq_path).resolve(),
        "vmiss": Path(vmiss_path).resolve(),
        "smiss": Path(smiss_path).resolve(),
        "prune_in": Path(prune_in_path).resolve(),
        "pca": Path(pca_path).resolve(),
        "eigenval": Path(eigenval_path).resolve(),
        "grm_bin": Path(grm_path).resolve(),
        "grm_n_bin": Path(grm_n_path).resolve(),
        "grm_id": Path(grm_id_path).resolve(),
    }
    for role, path in generated.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"generated kernel asset is missing: {role}: {path}"
            )

    iids = _read_fam(generated["fam"], expected_iids)
    qc = config["qc_contract"]
    marker_ids, chromosome_counts = _read_bim(
        generated["bim"], qc["minimum_qualified_markers"]
    )
    qc_stats = _verify_qc_tables(
        marker_ids,
        iids,
        generated["afreq"],
        generated["vmiss"],
        generated["smiss"],
        qc["maf"],
        qc["geno"],
        qc["mind"],
    )
    pruned_markers = _read_nonempty_lines(generated["prune_in"])
    if len(pruned_markers) <= 10 or not set(pruned_markers).issubset(marker_ids):
        raise ValueError("LD-pruned PCA marker list differs from the qualified panel")
    pca = _load_pca(generated["pca"], iids)
    eigenvalues = _load_eigenvalues(generated["eigenval"])
    group = config["group_contract"]
    assignments, evaluations = _derive_groups(
        iids, pca, group["candidate_k"], group["minimum_group_size"]
    )
    selected_k = len(np.unique(assignments))
    group_labels = [f"POP{int(value) + 1:02d}" for value in assignments]
    group_sizes = dict(sorted(Counter(group_labels).items()))

    kernel_contract = config["kernel_contract"]
    matrix, kernel_stats = _load_gcta_kernel(
        generated["grm_bin"],
        generated["grm_n_bin"],
        generated["grm_id"],
        iids,
        len(marker_ids),
        kernel_contract["pairwise_marker_counts_must_equal_marker_count"],
    )
    if kernel_stats["minimum_eigenvalue"] < -kernel_contract["psd_absolute_tolerance"]:
        raise ValueError("fresh 203-strain kernel is not PSD within the frozen tolerance")

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"kernel qualification output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        kernel_npy = temporary / "whole_genome_kernel.float64.npy"
        np.save(kernel_npy, matrix, allow_pickle=False)
        group_path = temporary / "population_groups.tsv"
        group_path.write_text(
            "IID\tpopulation_group\n"
            + "".join(
                f"{iid}\t{label}\n"
                for iid, label in zip(iids, group_labels, strict=True)
            ),
            encoding="utf-8",
        )
        receipt = {
            "schema_version": RECEIPT_VERSION,
            "analysis_id": config["analysis_id"],
            "classification": config["classification"],
            "biological_claims_permitted": False,
            "phenotype_paths_accepted_by_kernel_runner": False,
            "phenotype_values_accessed": False,
            "legacy_numeric_assets_accessed": False,
            "legacy_numeric_assets_accepted_as_inputs": False,
            "config": {
                "bytes": config_source.stat().st_size,
                "sha256": _sha256(config_source),
            },
            "preflight_receipt": {
                "bytes": preflight_receipt_path.stat().st_size,
                "sha256": _sha256(preflight_receipt_path),
            },
            "samples": len(iids),
            "ordered_iid_sha256": _ordered_hash(iids),
            "markers": len(marker_ids),
            "ordered_marker_sha256": _ordered_hash(marker_ids),
            "chromosome_marker_counts": chromosome_counts,
            "qc_statistics": qc_stats,
            "pca": {
                "markers": len(pruned_markers),
                "ordered_marker_sha256": _ordered_hash(pruned_markers),
                "pc_count": 10,
                "eigenvalues": [float(value) for value in eigenvalues],
                "scores_sha256": _sha256(generated["pca"]),
            },
            "population_groups": {
                "selected_k": selected_k,
                "group_sizes": group_sizes,
                "candidate_evaluations": evaluations,
                "labels_are_external_ancestry_assignments": False,
                "phenotype_values_accessed": False,
            },
            "kernel": {
                "method": kernel_contract["method"],
                "shape": [len(iids), len(iids)],
                "dtype": "float64",
                **kernel_stats,
                "psd_absolute_tolerance": kernel_contract["psd_absolute_tolerance"],
            },
            "generated_assets": {
                role: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                for role, path in generated.items()
            },
            "partition_identity": config["partition_identity"],
            "validation_outcomes_locked": True,
            "discovery_prediction_models_executed": False,
        }
        receipt_path = temporary / "kernel_qualification_receipt.json"
        receipt_path.write_bytes(_canonical_json(receipt))
        manifest = {
            name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for name, path in {
                "kernel_qualification_receipt.json": receipt_path,
                "population_groups.tsv": group_path,
                "whole_genome_kernel.float64.npy": kernel_npy,
            }.items()
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
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--source-vcf", required=True)
    preflight.add_argument("--source-vcf-index", required=True)
    preflight.add_argument("--roster", required=True)
    preflight.add_argument("--stage1-receipt", required=True)
    preflight.add_argument("--discovery-list", required=True)
    preflight.add_argument("--validation-list", required=True)
    preflight.add_argument("--vcf-samples", required=True)
    preflight.add_argument("--bcftools", required=True)
    preflight.add_argument("--plink2", required=True)
    preflight.add_argument("--gcta", required=True)
    preflight.add_argument("--python", required=True)
    preflight.add_argument("--output-dir", required=True)

    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--config", required=True)
    qualify.add_argument("--preflight-dir", required=True)
    for argument in (
        "bed",
        "bim",
        "fam",
        "afreq",
        "vmiss",
        "smiss",
        "prune-in",
        "pca",
        "eigenval",
        "grm",
        "grm-n",
        "grm-id",
    ):
        qualify.add_argument(f"--{argument}", required=True)
    qualify.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "preflight":
        receipt = prepare_kernel_inputs(
            args.config,
            args.source_vcf,
            args.source_vcf_index,
            args.roster,
            args.stage1_receipt,
            args.discovery_list,
            args.validation_list,
            args.vcf_samples,
            [args.bcftools, args.plink2, args.gcta, args.python],
            args.output_dir,
        )
    else:
        receipt = qualify_kernel_outputs(
            args.config,
            args.preflight_dir,
            args.bed,
            args.bim,
            args.fam,
            args.afreq,
            args.vmiss,
            args.smiss,
            args.prune_in,
            args.pca,
            args.eigenval,
            args.grm,
            args.grm_n,
            args.grm_id,
            args.output_dir,
        )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
