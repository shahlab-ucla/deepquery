"""Predeclared WS283 population-structure calibration for the abamectin POC.

This module verifies the immutable technical-dry-run parent, converts a
PLINK2 PCA report into fixed GCTA quantitative-covariate files, and summarizes
the frozen PC0/PC3/PC5/PC10 MLMA-LOCO sensitivity.  It deliberately reports
calibration and rank stability only; it cannot upgrade the parent to a
biological result or select a preferred PC count after seeing known loci.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .qtl import TRAITS, TRAIT_SLUGS, _lambda_gc, _parse_mlma, load_qtl_manifest


SCHEMA_VERSION = "wormctx-abamectin-ws283-pc-calibration-1.0"
SUMMARY_VERSION = "wormctx-abamectin-ws283-pc-calibration-summary-1.0"
MODEL_IDS = ("pc0", "pc3", "pc5", "pc10")
_SHA256 = frozenset("0123456789abcdef")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FrozenParentFile(_StrictModel):
    relative_path: str
    bytes: int = Field(ge=0)
    sha256: str

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("relative_path must use POSIX separators")
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("relative_path must remain below parent_run")
        return path.as_posix()

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in _SHA256 for character in value):
            raise ValueError("sha256 must be lowercase hexadecimal")
        return value


class ParentContract(_StrictModel):
    run_id: Literal["abamectin-ws283-bd41637-20260717T200012Z"]
    source_git_commit: Literal["bd41637f5ff2998333964cc954ee9f9f36a351d6"]
    post_qc_samples: Literal[209]
    post_qc_markers: Literal[373279]
    required_files: list[FrozenParentFile]

    @model_validator(mode="after")
    def required_roles_are_unique(self) -> "ParentContract":
        paths = [item.relative_path for item in self.required_files]
        if len(paths) != len(set(paths)):
            raise ValueError("parent required file paths must be unique")
        expected = {
            "SUCCESS",
            "receipts/SOURCE_REVISION",
            "receipts/frozen_analysis_manifest.json",
            "receipts/post_qc_counts.tsv",
            "genotype/abamectin_209_qc.bed",
            "genotype/abamectin_209_qc.bim",
            "genotype/abamectin_209_qc.fam",
            *(f"prepared/cohort/{TRAIT_SLUGS[trait]}.phen" for trait in TRAITS),
            *(f"association/{TRAIT_SLUGS[trait]}.loco.mlma" for trait in TRAITS),
        }
        if set(paths) != expected:
            raise ValueError("parent required file set differs from the frozen contract")
        return self


class ToolContract(_StrictModel):
    name: Literal["plink2", "gcta64"]
    version: str
    absolute_path: str
    sha256: str

    @field_validator("absolute_path")
    @classmethod
    def normalized_absolute_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("tool path must be normalized and absolute")
        return path.as_posix()

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in _SHA256 for character in value):
            raise ValueError("sha256 must be lowercase hexadecimal")
        return value


class LdPruningContract(_StrictModel):
    method: Literal["plink2_indep_pairwise_unphased_hardcall_r2"]
    window_kb: Literal[500]
    step_variants: Literal[1]
    r2: Literal[0.2]
    indep_order: Literal[2]


class PcaContract(_StrictModel):
    source: Literal["parent_post_qc_bfile"]
    ld_pruning: LdPruningContract
    computed_pc_count: Literal[10]
    association_pc_counts: list[int]
    approximate_pca: Literal[False]

    @model_validator(mode="after")
    def fixed_counts(self) -> "PcaContract":
        if self.association_pc_counts != [0, 3, 5, 10]:
            raise ValueError("association_pc_counts must be [0, 3, 5, 10]")
        return self


class AssociationModel(_StrictModel):
    id: Literal["pc0", "pc3", "pc5", "pc10"]
    pc_count: Literal[0, 3, 5, 10]
    source: Literal["immutable_parent_result", "new_qcovar_sensitivity"]
    primary: bool


class AssociationContract(_StrictModel):
    method: Literal["GCTA_MLMA_LOCO"]
    gcta_autosome_num: Literal[6]
    thread_count: Literal[16]
    maf: Literal[0.05]
    models: list[AssociationModel]
    covariate_fitting: Literal["gcta_default_preadjustment"]
    model_selection_from_results_permitted: Literal[False]

    @model_validator(mode="after")
    def exact_models(self) -> "AssociationContract":
        if [item.id for item in self.models] != list(MODEL_IDS):
            raise ValueError(f"models must appear in this order: {list(MODEL_IDS)}")
        if [item.pc_count for item in self.models] != [0, 3, 5, 10]:
            raise ValueError("model PC counts differ from the predeclared contract")
        if self.models[0].source != "immutable_parent_result" or not self.models[0].primary:
            raise ValueError("pc0 must remain the inherited primary model")
        if any(item.primary or item.source != "new_qcovar_sensitivity" for item in self.models[1:]):
            raise ValueError("PC-adjusted models must remain non-primary sensitivities")
        return self


class ClumpContract(_StrictModel):
    method: Literal["plink2_unphased"]
    p1: float
    p2: Literal[0.05]
    r2: Literal[0.2]
    kb: Literal[1000]


class DiagnosticContract(_StrictModel):
    alpha: Literal[0.05]
    marker_bonferroni_p: float
    historical_effective_test_p_reference_only: float
    qq_quantiles: list[float]
    lambda_exclusions: list[str]
    ld_clump: ClumpContract
    top_set_sizes: list[int]

    @model_validator(mode="after")
    def frozen_diagnostics(self) -> "DiagnosticContract":
        expected = 0.05 / 373279
        if not math.isclose(self.marker_bonferroni_p, expected, rel_tol=0.0, abs_tol=1e-20):
            raise ValueError("marker Bonferroni threshold differs from 0.05 / 373279")
        if not math.isclose(self.ld_clump.p1, expected, rel_tol=0.0, abs_tol=1e-20):
            raise ValueError("clump p1 must equal the marker Bonferroni threshold")
        if self.qq_quantiles != [0.5, 0.9, 0.95, 0.99, 0.999]:
            raise ValueError("QQ quantiles differ from the predeclared contract")
        if self.lambda_exclusions != [
            "none",
            "trait_published_intervals",
            "all_published_intervals",
        ]:
            raise ValueError("lambda exclusions differ from the predeclared contract")
        if self.top_set_sizes != [100, 1000]:
            raise ValueError("top-set sizes differ from the predeclared contract")
        return self


class ClaimContract(_StrictModel):
    permitted: list[str]
    prohibited: list[str]


class CalibrationManifest(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    analysis_id: Literal["abamectin_qtl_ws283_caendr20250625_pc_calibration_v1"]
    classification: Literal["exploratory_ws283_population_structure_calibration"]
    status: Literal["predeclared_preliminary_calibration_sensitivity"]
    validated: Literal[False]
    biological_claims_permitted: Literal[False]
    parent: ParentContract
    tools: list[ToolContract]
    pca: PcaContract
    association: AssociationContract
    diagnostics: DiagnosticContract
    claims: ClaimContract

    @model_validator(mode="after")
    def exact_tools_and_claim_boundary(self) -> "CalibrationManifest":
        if [tool.name for tool in self.tools] != ["plink2", "gcta64"]:
            raise ValueError("tools must be frozen in plink2, gcta64 order")
        prohibited = " ".join(self.claims.prohibited).lower()
        if "causal" not in prohibited or "biological validation" not in prohibited:
            raise ValueError("claim boundary must prohibit causal and validation claims")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_calibration_manifest(path: str | Path) -> CalibrationManifest:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"calibration manifest is not a file: {source}")
    return CalibrationManifest.model_validate_json(source.read_text(encoding="utf-8"))


def verify_parent(manifest_path: str | Path, parent_run: str | Path) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_calibration_manifest(manifest_source)
    root = Path(parent_run).resolve()
    if not root.is_dir() or root.name != manifest.parent.run_id:
        raise ValueError("parent run directory differs from the frozen contract")
    verified: list[dict[str, Any]] = []
    for item in manifest.parent.required_files:
        path = root.joinpath(*PurePosixPath(item.relative_path).parts)
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen parent file: {item.relative_path}")
        observed_bytes = path.stat().st_size
        observed_hash = _sha256(path)
        if observed_bytes != item.bytes or observed_hash != item.sha256:
            raise ValueError(f"parent file identity mismatch: {item.relative_path}")
        verified.append(
            {
                "relative_path": item.relative_path,
                "bytes": observed_bytes,
                "sha256": observed_hash,
            }
        )
    if (root / "SUCCESS").read_text(encoding="utf-8").strip() != "SUCCESS":
        raise ValueError("parent SUCCESS marker is malformed")
    revision = (root / "receipts/SOURCE_REVISION").read_text(encoding="utf-8").strip()
    if revision != manifest.parent.source_git_commit:
        raise ValueError("parent source revision differs from the frozen contract")
    counts = dict(
        line.split("\t", 1)
        for line in (root / "receipts/post_qc_counts.tsv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if counts != {
        "post_qc_samples": "209",
        "post_qc_markers": "373279",
        "chromosomes": "1,2,3,4,5,6",
    }:
        raise ValueError("parent post-QC counts differ from the frozen contract")
    return {
        "verified": True,
        "parent_run": str(root),
        "parent_run_id": root.name,
        "parent_source_git_commit": revision,
        "manifest_sha256": _sha256(manifest_source),
        "verified_files": verified,
    }


def _read_fam(path: Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"FAM line {line_number} must contain six fields")
        samples.append((fields[0], fields[1]))
    if len(samples) != 209 or len(samples) != len(set(samples)):
        raise ValueError("FAM must contain exactly 209 unique FID/IID pairs")
    return samples


def build_qcovars(
    manifest_path: str | Path,
    eigenvec_path: str | Path,
    fam_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    manifest = load_calibration_manifest(manifest_path)
    eigenvec = Path(eigenvec_path).resolve()
    fam = Path(fam_path).resolve()
    if not eigenvec.is_file() or not fam.is_file():
        raise FileNotFoundError("PCA eigenvector or FAM input is missing")
    samples = _read_fam(fam)
    lines = eigenvec.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError("PCA eigenvector file is empty")
    header = lines[0].split()
    normalized = [field.lstrip("#") for field in header]
    if "IID" not in normalized:
        raise ValueError("PCA eigenvector header has no IID column")
    iid_index = normalized.index("IID")
    fid_index = normalized.index("FID") if "FID" in normalized else None
    pc_indices = []
    for index in range(1, manifest.pca.computed_pc_count + 1):
        label = f"PC{index}"
        if label not in normalized:
            raise ValueError(f"PCA eigenvector header has no {label} column")
        pc_indices.append(normalized.index(label))
    by_sample: dict[tuple[str, str], list[float]] = {}
    fam_by_iid = {iid: fid for fid, iid in samples}
    if len(fam_by_iid) != len(samples):
        raise ValueError("FAM IIDs must be unique when PCA omits FID")
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split()
        if len(fields) != len(header):
            raise ValueError(f"PCA eigenvector line {line_number} has the wrong field count")
        iid = fields[iid_index]
        fid = fields[fid_index] if fid_index is not None else fam_by_iid.get(iid)
        if fid is None:
            raise ValueError(f"PCA IID is absent from FAM: {iid}")
        key = (fid, iid)
        if key in by_sample:
            raise ValueError(f"duplicate PCA sample: {key}")
        try:
            values = [float(fields[index]) for index in pc_indices]
        except ValueError as exc:
            raise ValueError(f"non-numeric PCA score on line {line_number}") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite PCA score on line {line_number}")
        by_sample[key] = values
    if set(by_sample) != set(samples):
        missing = sorted(set(samples) - set(by_sample))[:5]
        extra = sorted(set(by_sample) - set(samples))[:5]
        raise ValueError(f"PCA/FAM sample mismatch: missing={missing}, extra={extra}")

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"qcovar output must not already exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    outputs: dict[str, dict[str, Any]] = {}
    try:
        stage.mkdir()
        for count in manifest.pca.association_pc_counts[1:]:
            path = stage / f"pc{count}.qcovar"
            with path.open("w", encoding="utf-8", newline="") as handle:
                for fid, iid in samples:
                    scores = "\t".join(format(value, ".17g") for value in by_sample[(fid, iid)][:count])
                    handle.write(f"{fid}\t{iid}\t{scores}\n")
            outputs[path.name] = {
                "pc_count": count,
                "samples": len(samples),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        metadata = {
            "schema_version": "wormctx-abamectin-ws283-qcovars-1.0",
            "source_eigenvec": str(eigenvec),
            "source_eigenvec_sha256": _sha256(eigenvec),
            "source_fam": str(fam),
            "source_fam_sha256": _sha256(fam),
            "sample_order": "parent_fam_order",
            "outputs": outputs,
        }
        _write_json(stage / "qcovar_manifest.json", metadata)
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return metadata


def _in_intervals(row: Any, intervals: list[Any]) -> bool:
    return any(
        row.chromosome == interval.chromosome and interval.start <= row.bp <= interval.end
        for interval in intervals
    )


def _qq_quantiles(rows: list[Any], quantiles: list[float]) -> list[dict[str, float]]:
    values = sorted(-math.log10(max(row.p, 5e-324)) for row in rows)
    result = []
    for quantile in quantiles:
        position = quantile * (len(values) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        observed = values[lower]
        if upper != lower:
            observed += (values[upper] - values[lower]) * (position - lower)
        result.append(
            {
                "quantile": quantile,
                "expected_neg_log10_p": -math.log10(1.0 - quantile),
                "observed_neg_log10_p": observed,
            }
        )
    return result


def _ranked(rows: list[Any]) -> list[Any]:
    return sorted(rows, key=lambda row: (row.p, row.chromosome, row.bp, row.snp))


def _deterministic_spearman(reference: list[Any], comparison: list[Any]) -> float:
    if len(reference) != len(comparison):
        raise ValueError("rank comparison marker counts differ")
    reference_order = {row.snp: index for index, row in enumerate(_ranked(reference))}
    comparison_order = {row.snp: index for index, row in enumerate(_ranked(comparison))}
    if set(reference_order) != set(comparison_order):
        raise ValueError("rank comparison marker identities differ")
    count = len(reference_order)
    if count < 2:
        raise ValueError("rank comparison requires at least two markers")
    squared_difference = sum(
        (reference_order[snp] - comparison_order[snp]) ** 2 for snp in reference_order
    )
    return 1.0 - (6.0 * squared_difference) / (count * (count * count - 1))


def _top_overlap(reference: list[Any], comparison: list[Any], count: int) -> dict[str, Any]:
    reference_set = {row.snp for row in _ranked(reference)[:count]}
    comparison_set = {row.snp for row in _ranked(comparison)[:count]}
    intersection = len(reference_set & comparison_set)
    union = len(reference_set | comparison_set)
    return {
        "set_size": count,
        "intersection_count": intersection,
        "union_count": union,
        "jaccard": intersection / union if union else 1.0,
    }


def _read_clump_counts(path: Path) -> dict[tuple[str, str], int]:
    if not path.is_file():
        raise FileNotFoundError(f"clump count table is not a file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["model", "trait", "index_clumps"]:
            raise ValueError("clump count header mismatch")
        counts: dict[tuple[str, str], int] = {}
        for row in reader:
            key = (row["model"], row["trait"])
            if key in counts or key[0] not in MODEL_IDS or key[1] not in TRAITS:
                raise ValueError(f"invalid or duplicate clump row: {key}")
            value = int(row["index_clumps"])
            if value < 0:
                raise ValueError("clump counts cannot be negative")
            counts[key] = value
    expected = {(model, trait) for model in MODEL_IDS for trait in TRAITS}
    if set(counts) != expected:
        raise ValueError("clump table does not cover every model/trait combination")
    return counts


def _interval_summary(rows: list[Any], intervals: list[Any]) -> list[dict[str, Any]]:
    ordered = _ranked(rows)
    global_rank = {row.snp: index + 1 for index, row in enumerate(ordered)}
    output = []
    for interval in intervals:
        candidates = [row for row in rows if _in_intervals(row, [interval])]
        lead = min(candidates, key=lambda row: (row.p, row.bp, row.snp)) if candidates else None
        output.append(
            {
                "id": interval.id,
                "chromosome": interval.chromosome,
                "start": interval.start,
                "end": interval.end,
                "tested_markers": len(candidates),
                "lead_marker": None
                if lead is None
                else {
                    "snp": lead.snp,
                    "bp": lead.bp,
                    "p": lead.p,
                    "global_rank": global_rank[lead.snp],
                },
            }
        )
    return output


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Abamectin WS283 population-structure calibration",
        "",
        "This is a predeclared exploratory calibration sensitivity. Biological claims are prohibited.",
        "",
        "| Model | Trait | Lambda | Lambda outside trait intervals | Lambda outside all intervals | Bonferroni markers | LD clumps |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in summary["models"]:
        for trait in model["traits"]:
            lines.append(
                f"| {model['id']} | {trait['trait']} | {trait['lambda_gc']:.4f} | "
                f"{trait['lambda_gc_excluding_trait_intervals']:.4f} | "
                f"{trait['lambda_gc_excluding_all_intervals']:.4f} | "
                f"{trait['bonferroni_marker_count']} | {trait['ld_clump_count']} |"
            )
    lines.extend(["", "## Published-interval rank stability", ""])
    lines.append("| Model | Trait | Interval | Lead marker | Lead p | Global rank |")
    lines.append("|---|---|---|---|---:|---:|")
    for model in summary["models"]:
        for trait in model["traits"]:
            for interval in trait["published_intervals"]:
                lead = interval["lead_marker"]
                lines.append(
                    f"| {model['id']} | {trait['trait']} | {interval['id']} | "
                    f"{lead['snp'] if lead else 'none'} | "
                    f"{format(lead['p'], '.6g') if lead else 'n/a'} | "
                    f"{lead['global_rank'] if lead else 'n/a'} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "PC counts were fixed before these results. No model may be selected from known-locus recovery. "
            "Inflation, interval rank, and clump counts are diagnostics, not causal evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_calibration(
    manifest_path: str | Path,
    parent_manifest_path: str | Path,
    results_root: str | Path,
    clump_counts_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_calibration_manifest(manifest_source)
    parent_manifest_source = Path(parent_manifest_path).resolve()
    parent_manifest = load_qtl_manifest(parent_manifest_source)
    results = Path(results_root).resolve()
    if not results.is_dir():
        raise FileNotFoundError(f"calibration results root is not a directory: {results}")
    clump_counts_source = Path(clump_counts_path).resolve()
    clump_counts = _read_clump_counts(clump_counts_source)
    all_intervals = list(parent_manifest.published_intervals)
    parsed: dict[str, dict[str, list[Any]]] = {}
    marker_identity: list[tuple[int, str, int]] | None = None
    for model in MODEL_IDS:
        parsed[model] = {}
        for trait in TRAITS:
            path = results / model / f"{TRAIT_SLUGS[trait]}.loco.mlma"
            rows = _parse_mlma(path, set(parent_manifest.marker_qc.chromosomes))
            identity = [(row.chromosome, row.snp, row.bp) for row in rows]
            if marker_identity is None:
                marker_identity = identity
            elif identity != marker_identity:
                raise ValueError(f"marker identity/order differs for {model}/{trait}")
            if len(rows) != manifest.parent.post_qc_markers:
                raise ValueError(f"marker count differs for {model}/{trait}")
            parsed[model][trait] = rows

    models = []
    baseline = parsed["pc0"]
    for model_contract in manifest.association.models:
        traits = []
        for trait in TRAITS:
            rows = parsed[model_contract.id][trait]
            trait_intervals = [item for item in all_intervals if item.trait == trait]
            outside_trait = [row for row in rows if not _in_intervals(row, trait_intervals)]
            outside_all = [row for row in rows if not _in_intervals(row, all_intervals)]
            if not outside_trait or not outside_all:
                raise ValueError("published-interval exclusion removed every marker")
            trait_summary: dict[str, Any] = {
                "trait": trait,
                "result_sha256": _sha256(
                    results / model_contract.id / f"{TRAIT_SLUGS[trait]}.loco.mlma"
                ),
                "marker_count": len(rows),
                "lambda_gc": _lambda_gc(rows),
                "lambda_gc_excluding_trait_intervals": _lambda_gc(outside_trait),
                "lambda_gc_excluding_all_intervals": _lambda_gc(outside_all),
                "bonferroni_marker_count": sum(
                    row.p <= manifest.diagnostics.marker_bonferroni_p for row in rows
                ),
                "historical_reference_marker_count": sum(
                    row.p <= manifest.diagnostics.historical_effective_test_p_reference_only
                    for row in rows
                ),
                "ld_clump_count": clump_counts[(model_contract.id, trait)],
                "qq_quantiles": _qq_quantiles(rows, manifest.diagnostics.qq_quantiles),
                "published_intervals": _interval_summary(rows, trait_intervals),
            }
            if model_contract.id != "pc0":
                trait_summary["rank_stability_to_pc0"] = {
                    "deterministic_spearman_p_rank": _deterministic_spearman(
                        baseline[trait], rows
                    ),
                    "top_sets": [
                        _top_overlap(baseline[trait], rows, count)
                        for count in manifest.diagnostics.top_set_sizes
                    ],
                }
            traits.append(trait_summary)
        models.append(
            {
                "id": model_contract.id,
                "pc_count": model_contract.pc_count,
                "source": model_contract.source,
                "primary": model_contract.primary,
                "traits": traits,
            }
        )

    summary: dict[str, Any] = {
        "schema_version": SUMMARY_VERSION,
        "analysis_id": manifest.analysis_id,
        "classification": manifest.classification,
        "status": manifest.status,
        "validated": False,
        "biological_claims_permitted": False,
        "manifest_sha256": _sha256(manifest_source),
        "parent_manifest_sha256": _sha256(parent_manifest_source),
        "clump_counts_sha256": _sha256(clump_counts_source),
        "marker_count": manifest.parent.post_qc_markers,
        "sample_count": manifest.parent.post_qc_samples,
        "models": models,
        "claims": manifest.claims.model_dump(mode="json"),
        "limitations": {
            "pc_counts_predeclared_not_selected": True,
            "gcta_default_covariate_preadjustment": True,
            "alternative_grm_performed": False,
            "haplotype_analysis_performed": False,
            "causal_or_mechanistic_interpretation_permitted": False,
        },
    }
    json.dumps(summary, allow_nan=False)
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"calibration summary output must not exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        _write_json(stage / "ws283_pc_calibration_summary.json", summary)
        (stage / "ws283_pc_calibration_summary.md").write_text(
            _markdown(summary), encoding="utf-8", newline="\n"
        )
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return summary


def _parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="command", required=True)
    verify = subcommands.add_parser("verify-parent")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--parent-run", type=Path, required=True)
    qcovars = subcommands.add_parser("build-qcovars")
    qcovars.add_argument("--manifest", type=Path, required=True)
    qcovars.add_argument("--eigenvec", type=Path, required=True)
    qcovars.add_argument("--fam", type=Path, required=True)
    qcovars.add_argument("--output", type=Path, required=True)
    summarize = subcommands.add_parser("summarize")
    summarize.add_argument("--manifest", type=Path, required=True)
    summarize.add_argument("--parent-manifest", type=Path, required=True)
    summarize.add_argument("--results-root", type=Path, required=True)
    summarize.add_argument("--clump-counts", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    return command


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify-parent":
            result = verify_parent(args.manifest, args.parent_run)
        elif args.command == "build-qcovars":
            result = build_qcovars(args.manifest, args.eigenvec, args.fam, args.output)
        elif args.command == "summarize":
            result = summarize_calibration(
                args.manifest,
                args.parent_manifest,
                args.results_root,
                args.clump_counts,
                args.output,
            )
        else:  # pragma: no cover - argparse enforces this
            raise ValueError(f"unsupported command: {args.command}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
