"""Frozen preparation and reporting for the abamectin WS283 SNP-LMM POC.

This lane is deliberately narrow.  It prepares one predeclared four-trait
cohort from the Evans et al. supporting-data table and summarizes already-run
GCTA MLMA-LOCO files.  It does not infer causal genes, mechanisms, or
haplotypes, and every emitted receipt keeps biological claims disabled.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import NormalDist
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


QTL_SCHEMA_VERSION = "wormctx-abamectin-qtl-1.0"
QTL_PREPARATION_VERSION = "wormctx-abamectin-qtl-preparation-1.0"
QTL_SUMMARY_VERSION = "wormctx-abamectin-qtl-summary-1.0"
TRAITS = ("mean.EXT", "mean.TOF", "mean.norm.EXT", "norm.n")
TRAIT_SLUGS = {
    "mean.EXT": "mean_EXT",
    "mean.TOF": "mean_TOF",
    "mean.norm.EXT": "mean_norm_EXT",
    "norm.n": "norm_n",
}
MLMA_HEADER = ("Chr", "SNP", "bp", "A1", "A2", "Freq", "b", "se", "p")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MISSING = {"", ".", "na", "n/a", "nan", "null", "none"}
_CHI_SQUARE_1_MEDIAN = 0.4549364231195727


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FrozenFile(_StrictModel):
    role: Literal[
        "phenotype",
        "genotype_bed",
        "genotype_bim",
        "genotype_fam",
        "source_vcf",
        "source_vcf_index",
    ]
    relative_path: str
    sha256: str
    bytes: int = Field(gt=0)
    use: Literal["analysis_input", "lineage_only"]

    @field_validator("relative_path")
    @classmethod
    def portable_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("relative_path must use POSIX separators")
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("relative_path must remain beneath data_root")
        if any(part in {"", "."} for part in path.parts):
            raise ValueError("relative_path contains an empty or current-directory segment")
        return path.as_posix()

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("sha256 must be lowercase hexadecimal")
        return value


class ToolContract(_StrictModel):
    name: Literal["plink2", "gcta64"]
    version: str
    absolute_path: str
    sha256: str

    @field_validator("absolute_path")
    @classmethod
    def absolute_posix_path(cls, value: str) -> str:
        if not value.startswith("/") or ".." in PurePosixPath(value).parts:
            raise ValueError("tool path must be an absolute normalized POSIX path")
        return PurePosixPath(value).as_posix()

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("sha256 must be lowercase hexadecimal")
        return value


class PhenotypeContract(_StrictModel):
    source: Literal["Evans_et_al_2021_PLOS_Pathogens_File_S2"]
    condition: Literal["abamectin"]
    columns: list[str]
    traits: list[str]
    input_rows: Literal[948]
    input_strains: Literal[237]
    finite_strains_per_trait: Literal[210]
    common_missing_strains: Literal[27]
    expected_common_genotyped_strains: Literal[209]
    expected_finite_absent_from_genotype: list[str]
    one_row_per_strain_trait: Literal[True]
    identical_finite_cohort_all_traits: Literal[True]

    @model_validator(mode="after")
    def exact_contract(self) -> "PhenotypeContract":
        if self.columns != ["", "strain", "condition", "trait", "phenotype"]:
            raise ValueError("phenotype columns differ from the frozen File S2 derivative")
        if self.traits != list(TRAITS):
            raise ValueError(f"traits must be predeclared in this order: {list(TRAITS)}")
        if self.expected_finite_absent_from_genotype != ["JU1580"]:
            raise ValueError("the frozen finite non-genotyped audit must contain only JU1580")
        return self


class CohortContract(_StrictModel):
    match_key: Literal["fam_iid_equals_phenotype_strain"]
    order: Literal["source_fam_order"]
    fam_fid: Literal["0"]
    all_traits_required: Literal[True]
    expected_samples: Literal[209]


class MarkerQcContract(_StrictModel):
    chromosomes: list[int]
    chromosome_presentation: dict[str, str]
    snps_only: Literal[True]
    biallelic_only: Literal[True]
    maf: Literal[0.05]
    geno: Literal[0.05]
    mind: Literal[0.05]
    hardy_weinberg_filter: Literal[False]
    phenotype_directed_marker_filtering: Literal[False]

    @model_validator(mode="after")
    def exact_chromosomes(self) -> "MarkerQcContract":
        if self.chromosomes != [1, 2, 3, 4, 5, 6]:
            raise ValueError("chromosomes must be numeric I-V,X in order")
        expected = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "X"}
        if self.chromosome_presentation != expected:
            raise ValueError("chromosome presentation map must be I-V,X")
        return self


class AssociationContract(_StrictModel):
    method: Literal["GCTA_MLMA_LOCO"]
    gcta_autosome_num: Literal[6]
    fixed_pc_covariates: Literal[0]
    thread_count: int = Field(ge=1)
    alpha: Literal[0.05]
    published_effective_tests: Literal[963]
    top_hits_per_trait: int = Field(ge=1, le=100)
    same_common_cohort_all_traits: Literal[True]


class PublishedInterval(_StrictModel):
    id: str
    trait: Literal["mean.EXT", "mean.TOF", "norm.n"]
    chromosome: int = Field(ge=1, le=6)
    start: int = Field(gt=0)
    end: int = Field(gt=0)
    published_peak: int = Field(gt=0)
    source_table: Literal["Evans_et_al_2021_File_S3"]

    @model_validator(mode="after")
    def valid_interval(self) -> "PublishedInterval":
        if not _SAFE_TOKEN.fullmatch(self.id):
            raise ValueError("interval id must be path-safe")
        if not self.start <= self.published_peak <= self.end:
            raise ValueError("published peak must fall inside its interval")
        if "VC" in self.id.upper():
            raise ValueError("VC is linkage/NIL evidence and must not enter the wild-isolate scan")
        return self


class ProvenanceContract(_StrictModel):
    genotype_release: Literal["CaeNDR_20250625"]
    coordinate_system: Literal["WS283_inferred_not_declared_in_vcf_header"]
    phenotype_publication_doi: Literal["10.1371/journal.ppat.1009297"]
    phenotype_repo_commit: Literal["c197efe22cd5175aeb66aab05d56765cf9702e7f"]
    retained_plink_conversion_status: Literal[
        "sample_order_verified_but_generating_script_conflicts_with_retained_log"
    ]
    modernized_poc_limitation: Literal[True]
    numeric_chromosome_6_modeling_note: Literal[
        "biological_X_treated_as_sixth_diploid_inbred_chromosome_for_GCTA_LOCO_POC"
    ]


class QtlManifest(_StrictModel):
    schema_version: Literal["wormctx-abamectin-qtl-1.0"]
    analysis_id: str
    status: Literal["technical_dry_run_non_publishable"]
    validated: Literal[False]
    biological_claims_permitted: Literal[False]
    phenotype_rights_status: Literal["cc_by_publication_source_transform_receipt_missing"]
    inputs: list[FrozenFile]
    tools: list[ToolContract]
    phenotype: PhenotypeContract
    cohort: CohortContract
    marker_qc: MarkerQcContract
    association: AssociationContract
    published_intervals: list[PublishedInterval]
    traits_without_expected_published_interval: list[str]
    provenance: ProvenanceContract

    @model_validator(mode="after")
    def exact_frozen_analysis(self) -> "QtlManifest":
        if not _SAFE_TOKEN.fullmatch(self.analysis_id):
            raise ValueError("analysis_id must be a path-safe token")
        expected_roles = {
            "phenotype",
            "genotype_bed",
            "genotype_bim",
            "genotype_fam",
            "source_vcf",
            "source_vcf_index",
        }
        roles = [item.role for item in self.inputs]
        if len(roles) != len(set(roles)) or set(roles) != expected_roles:
            raise ValueError(
                f"inputs must contain exactly these unique roles: {sorted(expected_roles)}"
            )
        tool_names = [item.name for item in self.tools]
        if tool_names != ["plink2", "gcta64"]:
            raise ValueError("tools must contain plink2 then gcta64")
        if self.traits_without_expected_published_interval != ["mean.norm.EXT"]:
            raise ValueError("mean.norm.EXT must be the sole null published-interval trait")
        observed = [
            (item.id, item.trait, item.chromosome, item.start, item.end, item.published_peak)
            for item in self.published_intervals
        ]
        if observed != _expected_interval_tuples():
            raise ValueError("published intervals differ from the six frozen File S3 intervals")
        return self


def _expected_interval_tuples() -> list[tuple[str, str, int, int, int, int]]:
    return [
        ("mean_EXT_VL", "mean.EXT", 5, 1_747_612, 4_333_001, 2_693_128),
        ("mean_TOF_VL", "mean.TOF", 5, 1_757_246, 4_333_001, 2_693_128),
        ("mean_TOF_VR_glc-1", "mean.TOF", 5, 15_983_112, 16_599_066, 16_276_775),
        ("norm_n_II", "norm.n", 2, 13_756_151, 14_937_792, 14_121_786),
        ("norm_n_III", "norm.n", 3, 3_061_633, 4_632_949, 3_526_374),
        ("norm_n_V", "norm.n", 5, 13_606_517, 16_754_986, 15_965_095),
    ]


def load_qtl_manifest(path: str | Path) -> QtlManifest:
    return QtlManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _input_by_role(manifest: QtlManifest, role: str) -> FrozenFile:
    return next(item for item in manifest.inputs if item.role == role)


def _resolve_and_verify(data_root: Path, item: FrozenFile) -> Path:
    target = (data_root / PurePosixPath(item.relative_path)).resolve()
    if not target.is_relative_to(data_root):
        raise ValueError(f"input escapes data_root: {item.relative_path}")
    if not target.is_file():
        raise FileNotFoundError(f"frozen {item.role} input is not a file: {target}")
    observed_bytes = target.stat().st_size
    if observed_bytes != item.bytes:
        raise ValueError(
            f"frozen input size mismatch for {item.role}: "
            f"expected {item.bytes}, observed {observed_bytes}"
        )
    observed_hash = _sha256_file(target)
    if observed_hash != item.sha256:
        raise ValueError(
            f"frozen input checksum mismatch for {item.role}: "
            f"expected {item.sha256}, observed {observed_hash}"
        )
    return target


@dataclass(frozen=True)
class _FamSample:
    fid: str
    iid: str


def _read_fam(path: Path, expected_fid: str) -> list[_FamSample]:
    samples: list[_FamSample] = []
    seen_iids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"FAM line {line_number} must contain exactly six fields")
        fid, iid = fields[:2]
        if fid != expected_fid:
            raise ValueError(
                f"FAM line {line_number} FID differs from frozen value {expected_fid!r}"
            )
        if not iid or iid in seen_iids:
            raise ValueError(f"FAM contains blank or duplicate IID at line {line_number}: {iid!r}")
        seen_iids.add(iid)
        samples.append(_FamSample(fid=fid, iid=iid))
    if not samples:
        raise ValueError("FAM contains no samples")
    return samples


def _parse_phenotypes(
    path: Path, contract: PhenotypeContract
) -> tuple[dict[str, dict[str, float]], set[str], set[str], int]:
    by_trait: dict[str, dict[str, float]] = {trait: {} for trait in TRAITS}
    all_strains: set[str] = set()
    observed_cells: set[tuple[str, str]] = set()
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != contract.columns:
            raise ValueError(
                "phenotype header mismatch: "
                f"expected={contract.columns}, observed={reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, 2):
            row_count += 1
            strain = (row["strain"] or "").strip()
            condition = (row["condition"] or "").strip()
            trait = (row["trait"] or "").strip()
            raw_value = (row["phenotype"] or "").strip()
            if not strain:
                raise ValueError(f"phenotype row {row_number} has blank strain")
            if condition != contract.condition:
                raise ValueError(
                    f"phenotype row {row_number} has unexpected condition {condition!r}"
                )
            if trait not in by_trait:
                raise ValueError(f"phenotype row {row_number} has unexpected trait {trait!r}")
            cell = (strain, trait)
            if cell in observed_cells:
                raise ValueError(
                    f"duplicate phenotype strain/trait cell at row {row_number}: {cell}"
                )
            observed_cells.add(cell)
            all_strains.add(strain)
            if raw_value.casefold() in _MISSING:
                continue
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"phenotype row {row_number} is non-numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"phenotype row {row_number} has a non-finite non-missing value")
            by_trait[trait][strain] = value

    if row_count != contract.input_rows:
        raise ValueError(f"expected {contract.input_rows} phenotype rows, observed {row_count}")
    if len(all_strains) != contract.input_strains:
        raise ValueError(
            f"expected {contract.input_strains} phenotype strains, observed {len(all_strains)}"
        )
    expected_cells = {(strain, trait) for strain in all_strains for trait in TRAITS}
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)[:5]
        raise ValueError(
            "phenotype table is not a complete one-row-per-strain/trait grid: "
            f"{missing}"
        )
    finite_sets = [set(by_trait[trait]) for trait in TRAITS]
    counts = {trait: len(by_trait[trait]) for trait in TRAITS}
    if any(count != contract.finite_strains_per_trait for count in counts.values()):
        raise ValueError(f"unexpected finite phenotype counts: {counts}")
    if any(values != finite_sets[0] for values in finite_sets[1:]):
        raise ValueError("finite phenotype cohorts differ across the four predeclared traits")
    missing = all_strains - finite_sets[0]
    if len(missing) != contract.common_missing_strains:
        raise ValueError(
            f"expected {contract.common_missing_strains} common missing strains, "
            f"observed {len(missing)}"
        )
    return by_trait, all_strains, missing, row_count


def prepare_abamectin_qtl(
    manifest_path: str | Path, data_root: str | Path, output: str | Path
) -> dict[str, Any]:
    """Verify frozen inputs and atomically prepare the 209-sample phenotype cohort."""

    manifest_source = Path(manifest_path).resolve()
    manifest = load_qtl_manifest(manifest_source)
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data_root is not a directory: {root}")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"QTL preparation output must not already exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    phenotype_input = _resolve_and_verify(root, _input_by_role(manifest, "phenotype"))
    fam_input = _resolve_and_verify(root, _input_by_role(manifest, "genotype_fam"))
    by_trait, all_strains, missing_strains, row_count = _parse_phenotypes(
        phenotype_input, manifest.phenotype
    )
    fam = _read_fam(fam_input, manifest.cohort.fam_fid)
    genotype_iids = {sample.iid for sample in fam}
    common_finite = set(by_trait[TRAITS[0]])
    absent_finite = sorted(common_finite - genotype_iids)
    if absent_finite != manifest.phenotype.expected_finite_absent_from_genotype:
        raise ValueError(
            "finite phenotype strains absent from genotype differ from contract: "
            f"expected={manifest.phenotype.expected_finite_absent_from_genotype}, "
            f"observed={absent_finite}"
        )
    selected = [sample for sample in fam if sample.iid in common_finite]
    if len(selected) != manifest.cohort.expected_samples:
        raise ValueError(
            f"expected {manifest.cohort.expected_samples} common genotyped strains, "
            f"observed {len(selected)}"
        )
    selected_ids = {sample.iid for sample in selected}
    if any(set(by_trait[trait]) & genotype_iids != selected_ids for trait in TRAITS):
        raise ValueError("selected cohort is not identical for every predeclared trait")

    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        keep_path = stage / "cohort.keep"
        keep_path.write_text(
            "".join(f"{sample.fid}\t{sample.iid}\n" for sample in selected),
            encoding="utf-8",
            newline="\n",
        )
        cohort_path = stage / "cohort.tsv"
        cohort_path.write_text(
            "FID\tIID\tstrain\n"
            + "".join(f"{sample.fid}\t{sample.iid}\t{sample.iid}\n" for sample in selected),
            encoding="utf-8",
            newline="\n",
        )
        output_files: dict[str, dict[str, Any]] = {}
        for path, rows in ((keep_path, len(selected)), (cohort_path, len(selected))):
            output_files[path.name] = {
                "bytes": path.stat().st_size,
                "rows": rows,
                "sha256": _sha256_file(path),
            }
        for trait in TRAITS:
            path = stage / f"{TRAIT_SLUGS[trait]}.phen"
            path.write_text(
                "".join(
                    f"{sample.fid}\t{sample.iid}\t{format(by_trait[trait][sample.iid], '.17g')}\n"
                    for sample in selected
                ),
                encoding="utf-8",
                newline="\n",
            )
            output_files[path.name] = {
                "bytes": path.stat().st_size,
                "rows": len(selected),
                "sha256": _sha256_file(path),
                "trait": trait,
            }

        receipt: dict[str, Any] = {
            "schema_version": QTL_PREPARATION_VERSION,
            "analysis_id": manifest.analysis_id,
            "status": manifest.status,
            "validated": False,
            "biological_claims_permitted": False,
            "source_manifest": {
                "path": str(manifest_source),
                "sha256": _sha256_file(manifest_source),
            },
            "verified_inputs": {
                "phenotype": {
                    "relative_path": _input_by_role(manifest, "phenotype").relative_path,
                    "sha256": _sha256_file(phenotype_input),
                    "bytes": phenotype_input.stat().st_size,
                },
                "genotype_fam": {
                    "relative_path": _input_by_role(manifest, "genotype_fam").relative_path,
                    "sha256": _sha256_file(fam_input),
                    "bytes": fam_input.stat().st_size,
                },
            },
            "phenotype_audit": {
                "rows": row_count,
                "strains": len(all_strains),
                "traits": list(TRAITS),
                "finite_strains_per_trait": {
                    trait: len(by_trait[trait]) for trait in TRAITS
                },
                "common_missing_strains": sorted(missing_strains),
                "phenotype_roster_absent_from_genotype": sorted(all_strains - genotype_iids),
                "finite_absent_from_genotype": absent_finite,
            },
            "cohort": {
                "samples": len(selected),
                "order": manifest.cohort.order,
                "match_key": manifest.cohort.match_key,
                "all_traits_required": True,
                "iid_order_sha256": hashlib.sha256(
                    "".join(f"{sample.iid}\n" for sample in selected).encode("utf-8")
                ).hexdigest(),
            },
            "outputs": output_files,
            "limitations": {
                "phenotype_transform_receipt_missing": True,
                "retained_plink_conversion_provenance_conflict": True,
                "haplotype_analysis_performed": False,
            },
        }
        _write_json(stage / "phenotype_manifest.json", receipt)
        os.replace(stage, destination)
        return receipt
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


@dataclass(frozen=True)
class _MlmaRow:
    chromosome: int
    snp: str
    bp: int
    a1: str
    a2: str
    frequency: float
    effect: float
    standard_error: float
    p: float


def _parse_mlma(path: Path, chromosomes: set[int]) -> list[_MlmaRow]:
    if not path.is_file():
        raise FileNotFoundError(f"GCTA MLMA-LOCO result is not a file: {path}")
    rows: list[_MlmaRow] = []
    seen_snps: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError(f"MLMA file is empty: {path}")
        header = tuple(header_line.split())
        if header != MLMA_HEADER:
            raise ValueError(f"MLMA header mismatch in {path}: {header}")
        for line_number, line in enumerate(handle, 2):
            fields = line.split()
            if len(fields) != len(MLMA_HEADER):
                raise ValueError(f"MLMA row {line_number} in {path} has {len(fields)} fields")
            try:
                chromosome = int(fields[0])
                bp = int(fields[2])
                frequency, effect, standard_error, p_value = map(
                    float, (fields[5], fields[6], fields[7], fields[8])
                )
            except ValueError as exc:
                raise ValueError(f"MLMA row {line_number} in {path} is not numeric") from exc
            snp, a1, a2 = fields[1], fields[3], fields[4]
            numeric = (frequency, effect, standard_error, p_value)
            if chromosome not in chromosomes or bp <= 0:
                raise ValueError(f"MLMA row {line_number} has invalid chromosome/position")
            if not snp or snp in seen_snps:
                raise ValueError(f"MLMA row {line_number} has blank or duplicate SNP {snp!r}")
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(f"MLMA row {line_number} contains non-finite values")
            if not 0.0 <= frequency <= 1.0:
                raise ValueError(f"MLMA row {line_number} has frequency outside [0,1]")
            if standard_error <= 0.0 or not 0.0 <= p_value <= 1.0:
                raise ValueError(f"MLMA row {line_number} has invalid se or p")
            seen_snps.add(snp)
            rows.append(
                _MlmaRow(
                    chromosome, snp, bp, a1, a2, frequency, effect, standard_error, p_value
                )
            )
    if not rows:
        raise ValueError(f"MLMA file contains no marker rows: {path}")
    return rows


def _lambda_gc(rows: list[_MlmaRow]) -> float:
    chi_square = []
    for row in rows:
        lower_tail = max(row.p / 2.0, 5e-324)
        z_score = NormalDist().inv_cdf(lower_tail)
        chi_square.append(z_score * z_score)
    return statistics.median(chi_square) / _CHI_SQUARE_1_MEDIAN


def _row_json(row: _MlmaRow, intervals: list[PublishedInterval]) -> dict[str, Any]:
    return {
        "chromosome": row.chromosome,
        "snp": row.snp,
        "bp": row.bp,
        "a1": row.a1,
        "a2": row.a2,
        "frequency": row.frequency,
        "effect": row.effect,
        "standard_error": row.standard_error,
        "p": row.p,
        "published_interval_matches": [
            item.id
            for item in intervals
            if item.chromosome == row.chromosome and item.start <= row.bp <= item.end
        ],
    }


def _trait_summary(
    trait: str, rows: list[_MlmaRow], manifest: QtlManifest, source: Path
) -> dict[str, Any]:
    alpha = manifest.association.alpha
    effective_threshold = alpha / manifest.association.published_effective_tests
    bonferroni_threshold = alpha / len(rows)
    ordered = sorted(rows, key=lambda row: (row.p, row.chromosome, row.bp, row.snp))
    intervals = [item for item in manifest.published_intervals if item.trait == trait]
    recoveries: list[dict[str, Any]] = []
    for interval in intervals:
        candidates = [
            row
            for row in rows
            if row.chromosome == interval.chromosome and interval.start <= row.bp <= interval.end
        ]
        lead = min(candidates, key=lambda row: (row.p, row.bp, row.snp)) if candidates else None
        recoveries.append(
            {
                "id": interval.id,
                "chromosome": interval.chromosome,
                "start": interval.start,
                "end": interval.end,
                "published_peak": interval.published_peak,
                "tested_markers": len(candidates),
                "lead_marker": _row_json(lead, [interval]) if lead else None,
                "recovered_at_published_effective_threshold": bool(
                    lead is not None and lead.p <= effective_threshold
                ),
                "recovered_at_naive_bonferroni_threshold": bool(
                    lead is not None and lead.p <= bonferroni_threshold
                ),
            }
        )
    return {
        "trait": trait,
        "result_file": str(source),
        "result_sha256": _sha256_file(source),
        "marker_count": len(rows),
        "genomic_inflation_lambda_gc": _lambda_gc(rows),
        "thresholds": {
            "published_effective_test": effective_threshold,
            "published_effective_tests": manifest.association.published_effective_tests,
            "naive_bonferroni": bonferroni_threshold,
            "alpha": alpha,
        },
        "significant_marker_counts": {
            "published_effective_test": sum(row.p <= effective_threshold for row in rows),
            "naive_bonferroni": sum(row.p <= bonferroni_threshold for row in rows),
        },
        "top_hits": [
            _row_json(row, intervals)
            for row in ordered[: manifest.association.top_hits_per_trait]
        ],
        "published_interval_recovery": recoveries,
        "expected_published_interval_count": len(intervals),
        "no_expected_published_interval": trait
        in manifest.traits_without_expected_published_interval,
    }


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Abamectin WS283 MLMA-LOCO technical summary",
        "",
        (
            "This is an exploratory, non-publishable technical dry run. "
            "Biological claims are not permitted."
        ),
        "",
        (
            "| Trait | Markers | Lambda GC | Effective-test hits | "
            "Bonferroni hits | Expected intervals |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["traits"]:
        counts = item["significant_marker_counts"]
        lines.append(
            f"| {item['trait']} | {item['marker_count']} | "
            f"{item['genomic_inflation_lambda_gc']:.6g} | "
            f"{counts['published_effective_test']} | {counts['naive_bonferroni']} | "
            f"{item['expected_published_interval_count']} |"
        )
    lines.extend(["", "## Published File S3 interval recovery", ""])
    for item in summary["traits"]:
        lines.append(f"### {item['trait']}")
        lines.append("")
        recoveries = item["published_interval_recovery"]
        if not recoveries:
            lines.append("No wild-isolate association interval was reported for this trait.")
            lines.append("")
            continue
        lines.append("| Interval | Chr | Span | Lead marker | Lead p | Effective | Bonferroni |")
        lines.append("|---|---:|---:|---|---:|---|---|")
        for recovery in recoveries:
            lead = recovery["lead_marker"]
            lines.append(
                f"| {recovery['id']} | {recovery['chromosome']} | "
                f"{recovery['start']}-{recovery['end']} | "
                f"{lead['snp'] if lead else 'none'} | "
                f"{format(lead['p'], '.6g') if lead else 'n/a'} | "
                f"{str(recovery['recovered_at_published_effective_threshold']).lower()} | "
                f"{str(recovery['recovered_at_naive_bonferroni_threshold']).lower()} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            (
                "These are SNP mixed-model diagnostics, not haplotype, causal-gene, "
                "or mechanism results. The VC linkage/NIL interval is intentionally "
                "excluded from this wild-isolate benchmark."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def summarize_abamectin_qtl(
    manifest_path: str | Path, results_root: str | Path, output: str | Path
) -> dict[str, Any]:
    """Validate and atomically summarize four exact ``*.loco.mlma`` outputs."""

    manifest_source = Path(manifest_path).resolve()
    manifest = load_qtl_manifest(manifest_source)
    results = Path(results_root).resolve()
    if not results.is_dir():
        raise FileNotFoundError(f"results_root is not a directory: {results}")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"QTL summary output must not already exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    trait_results: list[dict[str, Any]] = []
    chromosomes = set(manifest.marker_qc.chromosomes)
    for trait in TRAITS:
        source = results / f"{TRAIT_SLUGS[trait]}.loco.mlma"
        rows = _parse_mlma(source, chromosomes)
        trait_results.append(_trait_summary(trait, rows, manifest, source))
    marker_counts = {item["marker_count"] for item in trait_results}
    if len(marker_counts) != 1:
        raise ValueError(f"four-trait MLMA marker counts differ: {sorted(marker_counts)}")
    summary: dict[str, Any] = {
        "schema_version": QTL_SUMMARY_VERSION,
        "analysis_id": manifest.analysis_id,
        "status": manifest.status,
        "validated": False,
        "biological_claims_permitted": False,
        "manifest_sha256": _sha256_file(manifest_source),
        "method": manifest.association.method,
        "gcta_autosome_num": manifest.association.gcta_autosome_num,
        "traits": trait_results,
        "published_interval_contract": {
            "source": "Evans_et_al_2021_File_S3",
            "expected_intervals": 6,
            "observed_intervals": sum(
                item["expected_published_interval_count"] for item in trait_results
            ),
            "null_trait": "mean.norm.EXT",
            "vc_interval_included": False,
        },
        "limitations": {
            "technical_dry_run_non_publishable": True,
            "haplotype_analysis_performed": False,
            "causal_or_mechanistic_interpretation_permitted": False,
        },
    }
    json.dumps(summary, allow_nan=False)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        stage.mkdir()
        _write_json(stage / "qtl_summary.json", summary)
        (stage / "qtl_summary.md").write_text(
            _markdown_summary(summary), encoding="utf-8", newline="\n"
        )
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return summary
