"""Receipted isotype-sensitivity reconstruction of the Evans et al. GWAS.

This module preserves the earlier rehosted isotype-imputed lane.  The archived
all-strain hard-filter release payload and original CSI were recovered later,
but all tested release representations failed the historical marker-identity
gate.  This module must not relabel its legacy isotype results as a strict
reproduction, an original-source recovery, or a biological claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
import uuid
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "wormctx-abamectin-ws276-reconstruction-1.0"
TRAITS = ("mean.EXT", "mean.TOF", "mean.norm.EXT", "norm.n")
TRAIT_SLUGS = {
    "mean.EXT": "mean_EXT",
    "mean.TOF": "mean_TOF",
    "mean.norm.EXT": "mean_norm_EXT",
    "norm.n": "norm_n",
}
CHROMOSOME_ORDER = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "X": 6, "MtDNA": 7}
_MISSING = {"", ".", "na", "n/a", "nan", "null", "none"}
_CHI_SQUARE_1_MEDIAN = 0.4549364231195727


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FrozenAsset(_StrictModel):
    role: Literal["phenotype", "source_vcf", "source_vcf_index", "published_s3"]
    filename: str
    bytes: int = Field(gt=0)
    sha256: str
    source: str

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value


class CohortContract(_StrictModel):
    raw_rows: Literal[948]
    raw_strains: Literal[237]
    finite_per_trait: Literal[210]
    missing_per_trait: Literal[27]
    mapping_source_commit: Literal["2abedc5d51289f612e9b0931f1237eb039852c0d"]
    mapping_source_git_blob: Literal["4d6658d1d2f7babff3b43433780b656e6a5d864f"]
    aliases: dict[str, str]
    collision_policy: Literal["isotype_reference_phenotype_wins"]
    expected_collision: dict[str, list[str]]
    expected_analysis_samples: Literal[209]

    @model_validator(mode="after")
    def frozen_mapping(self) -> "CohortContract":
        expected_aliases = {
            "ECA248": "CB4855",
            "ECA250": "CB4857",
            "ECA251": "CB4858",
            "JU1580": "JU1793",
        }
        if self.aliases != expected_aliases:
            raise ValueError("cohort aliases differ from the commit-pinned lookup reconstruction")
        if self.expected_collision != {"JU1793": ["JU1580", "JU1793"]}:
            raise ValueError("JU1793 must be the sole frozen isotype collision")
        return self


class PreprocessingContract(_StrictModel):
    bcftools_version: Literal["1.9"]
    plink_version: Literal["1.90b6.12"]
    require_no_missing_genotypes: Literal[True]
    snps_only: Literal[True]
    biallelic_only: Literal[True]
    maf: Literal[0.05]
    plink_geno_default: Literal[0.1]
    ld_window: Literal[50]
    ld_step: Literal[10]
    ld_r2: Literal[0.8]
    expected_complete_markers: Literal[21342]


class AssociationContract(_StrictModel):
    method: Literal["rrBLUP_GWAS_EMMA"]
    rrblup_version: Literal["4.6"]
    r_version: Literal["3.6.0"]
    p3d_primary: Literal[False]
    p3d_sensitivity: Literal[True]
    min_maf: Literal[0.05]
    effective_tests_deposited: float
    threshold_log10p: float
    ci_markers_each_side: Literal[150]
    qtl_grouping_marker_gap: Literal[1000]
    expected_published_nonzero_markers: Literal[20946]

    @model_validator(mode="after")
    def deposited_threshold(self) -> "AssociationContract":
        if not math.isclose(self.effective_tests_deposited, 962.616, abs_tol=1e-12):
            raise ValueError("effective-test count must equal the value deposited in File S3")
        if not math.isclose(self.threshold_log10p, 4.28448307163607, abs_tol=1e-14):
            raise ValueError("threshold must equal the value deposited in File S3")
        if not math.isclose(
            -math.log10(0.05 / self.effective_tests_deposited),
            self.threshold_log10p,
            abs_tol=1e-12,
        ):
            raise ValueError("effective-test count and deposited threshold disagree")
        return self


class PublishedInterval(_StrictModel):
    trait: Literal["mean.EXT", "mean.TOF", "norm.n"]
    chromosome: Literal["II", "III", "V"]
    start: int = Field(gt=0)
    end: int = Field(gt=0)
    peak: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_geometry(self) -> "PublishedInterval":
        if not self.start <= self.peak <= self.end:
            raise ValueError("published interval must satisfy start <= peak <= end")
        return self


class Ws276Manifest(_StrictModel):
    schema_version: Literal["wormctx-abamectin-ws276-reconstruction-1.0"]
    analysis_id: str
    status: Literal["historical_isotype_reconstruction_non_publishable"]
    strict_reproduction_claim_permitted: Literal[False]
    biological_claims_permitted: Literal[False]
    assets: list[FrozenAsset]
    traits: list[str]
    cohort: CohortContract
    preprocessing: PreprocessingContract
    association: AssociationContract
    expected_significant_markers: dict[str, int]
    published_intervals: list[PublishedInterval]
    provenance_gaps: list[str]

    @model_validator(mode="after")
    def exact_contract(self) -> "Ws276Manifest":
        roles = [asset.role for asset in self.assets]
        if roles != ["phenotype", "source_vcf", "source_vcf_index", "published_s3"]:
            raise ValueError("assets must be phenotype, VCF, index, and File S3 in order")
        if self.traits != list(TRAITS):
            raise ValueError(f"traits must be {list(TRAITS)}")
        if self.expected_significant_markers != {
            "mean.EXT": 2,
            "mean.TOF": 2,
            "mean.norm.EXT": 0,
            "norm.n": 101,
        }:
            raise ValueError("published significant-marker contract changed")
        if len(self.published_intervals) != 6:
            raise ValueError("the published wild-isolate contract has six intervals")
        expected_intervals = [
            {
                "trait": "mean.EXT",
                "chromosome": "V",
                "start": 1_747_612,
                "end": 4_333_001,
                "peak": 2_693_128,
            },
            {
                "trait": "mean.TOF",
                "chromosome": "V",
                "start": 1_757_246,
                "end": 4_333_001,
                "peak": 2_693_128,
            },
            {
                "trait": "mean.TOF",
                "chromosome": "V",
                "start": 15_983_112,
                "end": 16_599_066,
                "peak": 16_276_775,
            },
            {
                "trait": "norm.n",
                "chromosome": "II",
                "start": 13_756_151,
                "end": 14_937_792,
                "peak": 14_121_786,
            },
            {
                "trait": "norm.n",
                "chromosome": "III",
                "start": 3_061_633,
                "end": 4_632_949,
                "peak": 3_526_374,
            },
            {
                "trait": "norm.n",
                "chromosome": "V",
                "start": 13_606_517,
                "end": 16_754_986,
                "peak": 15_965_095,
            },
        ]
        observed_intervals = [item.model_dump() for item in self.published_intervals]
        if observed_intervals != expected_intervals:
            raise ValueError("published interval tuples differ from the frozen File S3 contract")
        if not self.provenance_gaps:
            raise ValueError("a reconstruction must retain its provenance gaps")
        return self


def load_ws276_manifest(path: str | Path) -> Ws276Manifest:
    return Ws276Manifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _asset(manifest: Ws276Manifest, role: str) -> FrozenAsset:
    return next(item for item in manifest.assets if item.role == role)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify_asset(path: Path, asset: FrozenAsset) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {asset.role}: {path}")
    if path.stat().st_size != asset.bytes:
        raise ValueError(
            f"{asset.role} size mismatch: expected {asset.bytes}, observed {path.stat().st_size}"
        )
    observed = _sha256(path)
    if observed != asset.sha256:
        raise ValueError(
            f"{asset.role} checksum mismatch: expected {asset.sha256}, observed {observed}"
        )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _parse_phenotype(
    path: Path, manifest: Ws276Manifest
) -> tuple[dict[str, dict[str, float]], set[str], int]:
    by_trait: dict[str, dict[str, float]] = {trait: {} for trait in TRAITS}
    all_strains: set[str] = set()
    seen: set[tuple[str, str]] = set()
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_header = ["", "strain", "condition", "trait", "phenotype"]
        if reader.fieldnames != expected_header:
            raise ValueError(
                "phenotype header mismatch: "
                f"expected={expected_header}, observed={reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, 2):
            rows += 1
            strain = (row["strain"] or "").strip()
            trait = (row["trait"] or "").strip()
            condition = (row["condition"] or "").strip()
            if not strain or trait not in by_trait or condition != "abamectin":
                raise ValueError(f"invalid phenotype identity fields at row {line_number}")
            cell = (strain, trait)
            if cell in seen:
                raise ValueError(f"duplicate phenotype cell at row {line_number}: {cell}")
            seen.add(cell)
            all_strains.add(strain)
            raw = (row["phenotype"] or "").strip()
            if raw.casefold() in _MISSING:
                continue
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"non-numeric phenotype at row {line_number}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite phenotype at row {line_number}")
            by_trait[trait][strain] = value

    cohort = manifest.cohort
    if rows != cohort.raw_rows or len(all_strains) != cohort.raw_strains:
        raise ValueError(
            f"phenotype dimensions changed: rows={rows}, strains={len(all_strains)}"
        )
    expected_cells = {(strain, trait) for strain in all_strains for trait in TRAITS}
    if seen != expected_cells:
        raise ValueError("phenotype input is not a complete strain-by-trait grid")
    finite_sets = [set(by_trait[trait]) for trait in TRAITS]
    if any(len(values) != cohort.finite_per_trait for values in finite_sets):
        raise ValueError("finite phenotype counts differ from the frozen contract")
    if any(values != finite_sets[0] for values in finite_sets[1:]):
        raise ValueError("finite phenotype cohorts differ by trait")
    if len(all_strains - finite_sets[0]) != cohort.missing_per_trait:
        raise ValueError("common missing phenotype count differs from the frozen contract")
    return by_trait, all_strains, rows


def prepare_ws276_reconstruction(
    manifest_path: str | Path,
    phenotype_path: str | Path,
    vcf_samples_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Create the mapped 209-isotype cohort and four rrBLUP phenotype files."""

    manifest = load_ws276_manifest(manifest_path)
    phenotype = Path(phenotype_path).resolve()
    _verify_asset(phenotype, _asset(manifest, "phenotype"))
    samples_source = Path(vcf_samples_path).resolve()
    if not samples_source.is_file():
        raise FileNotFoundError(f"VCF sample inventory is missing: {samples_source}")
    vcf_samples = [line.strip() for line in samples_source.read_text(encoding="utf-8").splitlines()]
    if (
        not vcf_samples
        or len(vcf_samples) != len(set(vcf_samples))
        or any(not item for item in vcf_samples)
    ):
        raise ValueError("VCF sample inventory is empty or contains blank/duplicate IDs")

    by_trait, all_strains, row_count = _parse_phenotype(phenotype, manifest)
    raw_finite = set(by_trait[TRAITS[0]])
    groups: dict[str, list[str]] = {}
    for strain in sorted(raw_finite):
        isotype = manifest.cohort.aliases.get(strain, strain)
        groups.setdefault(isotype, []).append(strain)
    collisions = {key: values for key, values in groups.items() if len(values) > 1}
    if collisions != manifest.cohort.expected_collision:
        raise ValueError(
            f"isotype collision set changed: expected={manifest.cohort.expected_collision}, "
            f"observed={collisions}"
        )

    representatives: dict[str, str] = {}
    for isotype, sources in groups.items():
        if len(sources) == 1:
            representatives[isotype] = sources[0]
        elif isotype in sources:
            representatives[isotype] = isotype
        else:
            raise ValueError(f"collision lacks an isotype-reference phenotype: {isotype}={sources}")
    if len(representatives) != manifest.cohort.expected_analysis_samples:
        raise ValueError(f"expected 209 reconstructed isotypes, observed {len(representatives)}")
    absent = sorted(set(representatives) - set(vcf_samples))
    if absent:
        raise ValueError(f"reconstructed phenotype isotypes absent from VCF: {absent}")

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"reconstruction output must not already exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    sample_order = sorted(representatives)
    try:
        stage.mkdir()
        (stage / "sample_order.txt").write_text(
            "".join(f"{sample}\n" for sample in sample_order),
            encoding="utf-8",
            newline="\n",
        )
        for trait in TRAITS:
            with (stage / f"{TRAIT_SLUGS[trait]}.tsv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["strain", trait])
                for isotype in sample_order:
                    value = by_trait[trait][representatives[isotype]]
                    writer.writerow([isotype, format(value, ".17g")])
        with (stage / "source_to_analysis.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["source_strain", "analysis_isotype", "selected", "reason"])
            for isotype, sources in sorted(groups.items()):
                selected = representatives[isotype]
                for source in sources:
                    reason = "reference_wins_collision" if len(sources) > 1 else (
                        "commit_pinned_alias" if source != isotype else "identity"
                    )
                    writer.writerow([source, isotype, str(source == selected).lower(), reason])
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "analysis_id": manifest.analysis_id,
            "status": manifest.status,
            "strict_reproduction_claim_permitted": False,
            "biological_claims_permitted": False,
            "phenotype_rows": row_count,
            "phenotype_strains": len(all_strains),
            "finite_source_strains": len(raw_finite),
            "analysis_isotypes": len(sample_order),
            "vcf_inventory_samples": len(vcf_samples),
            "mapping_source_commit": manifest.cohort.mapping_source_commit,
            "mapping_source_git_blob": manifest.cohort.mapping_source_git_blob,
            "aliases_applied": manifest.cohort.aliases,
            "collisions": collisions,
            "selected_collision_representative": {"JU1793": "JU1793"},
            "dropped_source_phenotypes": ["JU1580"],
            "provenance_gaps": manifest.provenance_gaps,
        }
        _write_json(stage / "cohort_receipt.json", receipt)
        stage.replace(destination)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return receipt


def _read_mapping(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing rrBLUP mapping: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["marker", "CHROM", "POS", "log10p"]:
            raise ValueError(f"rrBLUP mapping header mismatch in {path}: {reader.fieldnames}")
        for line_number, row in enumerate(reader, 2):
            marker = (row["marker"] or "").strip()
            chromosome = (row["CHROM"] or "").strip()
            if marker in seen or chromosome not in CHROMOSOME_ORDER:
                raise ValueError(f"duplicate marker or invalid chromosome at {path}:{line_number}")
            seen.add(marker)
            try:
                position = int(row["POS"])
                log10p = float(row["log10p"])
            except ValueError as exc:
                raise ValueError(f"invalid rrBLUP value at {path}:{line_number}") from exc
            if position <= 0 or not math.isfinite(log10p) or log10p < 0:
                raise ValueError(f"out-of-range rrBLUP value at {path}:{line_number}")
            rows.append(
                {"marker": marker, "CHROM": chromosome, "POS": position, "log10p": log10p}
            )
    if not rows:
        raise ValueError(f"empty rrBLUP mapping: {path}")
    return rows


def _read_run_metadata(
    path: Path,
    manifest: Ws276Manifest,
    *,
    expected_p3d: bool,
    expected_kinship_action: str,
) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing rrBLUP run metadata: {path}")
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["key", "value"]:
            raise ValueError(f"rrBLUP metadata header mismatch in {path}: {reader.fieldnames}")
        for line_number, row in enumerate(reader, 2):
            key = (row["key"] or "").strip()
            value = (row["value"] or "").strip()
            if not key or not value or key in values:
                raise ValueError(f"blank or duplicate rrBLUP metadata at {path}:{line_number}")
            values[key] = value

    exact = {
        "classification": "historical_isotype_imputed_reconstruction",
        "strict_reproduction": "false",
        "P3D": str(expected_p3d).upper(),
        "rrBLUP_version": manifest.association.rrblup_version,
        "input_samples": str(manifest.cohort.expected_analysis_samples),
        "MtDNA_preserved": "true",
        "kinship_action": expected_kinship_action,
    }
    mismatches = {
        key: {"expected": expected, "observed": values.get(key)}
        for key, expected in exact.items()
        if values.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"rrBLUP mode metadata contract failed in {path}: {mismatches}")
    try:
        cores = int(values["cores"])
        input_markers = int(values["input_markers"])
        complete_markers = int(values["complete_markers"])
        tested_markers = {
            trait: int(values[f"{TRAIT_SLUGS[trait]}_tested_markers"])
            for trait in TRAITS
        }
    except (KeyError, ValueError) as exc:
        raise ValueError(f"rrBLUP numeric metadata is missing or invalid in {path}") from exc
    if (
        cores <= 0
        or not 0 < complete_markers <= input_markers
    ):
        raise ValueError(f"rrBLUP numeric metadata is out of range in {path}")
    if any(count <= 0 for count in tested_markers.values()):
        raise ValueError(f"rrBLUP tested-marker counts are out of range in {path}")
    return values


def _read_published_s3(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {trait: {} for trait in TRAITS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"marker", "CHROM", "POS", "log10p", "trait"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("published File S3 header lacks mapping columns")
        for line_number, row in enumerate(reader, 2):
            raw_trait = (row["trait"] or "").strip()
            trait = raw_trait.removeprefix("abamectin_")
            if trait not in result:
                raise ValueError(f"unexpected File S3 trait at row {line_number}: {raw_trait}")
            marker = (row["marker"] or "").strip()
            value = {
                "marker": marker,
                "CHROM": (row["CHROM"] or "").strip(),
                "POS": int(row["POS"]),
                "log10p": float(row["log10p"]),
            }
            previous = result[trait].get(marker)
            if previous is not None and previous != value:
                raise ValueError(f"inconsistent duplicated File S3 marker: {trait}/{marker}")
            result[trait][marker] = value
    return result


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2
        for ordered_index in order[index:end]:
            ranks[ordered_index] = rank
        index = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def _lambda_gc(rows: list[dict[str, Any]]) -> float:
    chi_square: list[float] = []
    normal = NormalDist()
    for row in rows:
        p_value = 10.0 ** (-row["log10p"])
        if p_value >= 1.0:
            chi_square.append(0.0)
        elif p_value > 0.0:
            # NormalDist cannot represent a CDF closer to one than machine
            # precision. The cap prevents extreme hits from breaking a median-
            # based diagnostic and has no practical effect on lambda GC.
            upper_cdf = 1.0 - max(p_value / 2.0, 1e-16)
            chi_square.append(normal.inv_cdf(upper_cdf) ** 2)
    return statistics.median(chi_square) / _CHI_SQUARE_1_MEDIAN


def _intervals(
    rows: list[dict[str, Any]], threshold: float, flank: int, grouping_gap: int
) -> list[dict[str, Any]]:
    retained = [row for row in rows if row["log10p"] != 0]
    retained.sort(key=lambda row: (CHROMOSOME_ORDER[row["CHROM"]], row["POS"], row["marker"]))
    by_chromosome: dict[str, list[dict[str, Any]]] = {}
    for row in retained:
        by_chromosome.setdefault(row["CHROM"], []).append(row)
    intervals: list[dict[str, Any]] = []
    for chromosome, chromosome_rows in by_chromosome.items():
        significant_indices = [
            index for index, row in enumerate(chromosome_rows) if row["log10p"] >= threshold
        ]
        groups: list[list[int]] = []
        for index in significant_indices:
            if groups and index - groups[-1][-1] < grouping_gap:
                groups[-1].append(index)
            else:
                groups.append([index])
        for group in groups:
            start_index = max(0, group[0] - flank)
            end_index = min(len(chromosome_rows) - 1, group[-1] + flank)
            peak = max((chromosome_rows[index] for index in group), key=lambda row: row["log10p"])
            intervals.append(
                {
                    "chromosome": chromosome,
                    "start": chromosome_rows[start_index]["POS"],
                    "end": chromosome_rows[end_index]["POS"],
                    "peak": peak["POS"],
                    "peak_marker": peak["marker"],
                    "peak_log10p": peak["log10p"],
                    "significant_markers": len(group),
                }
            )
    return intervals


def summarize_ws276_reconstruction(
    manifest_path: str | Path,
    results_root: str | Path,
    published_s3_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Compare primary and P3D sensitivity scans with the deposited File S3 map."""

    manifest = load_ws276_manifest(manifest_path)
    published_path = Path(published_s3_path).resolve()
    _verify_asset(published_path, _asset(manifest, "published_s3"))
    reference = _read_published_s3(published_path)
    root = Path(results_root).resolve()
    threshold = manifest.association.threshold_log10p
    modes = {
        "primary_emma_p3d_false": ("primary", False, "created"),
        "sensitivity_emmax_p3d_true": ("p3d_true", True, "reused"),
    }
    summary_modes: dict[str, Any] = {}
    marker_universes: dict[str, dict[str, dict[str, tuple[str, int]]]] = {}
    for mode, (directory, expected_p3d, expected_kinship_action) in modes.items():
        metadata = _read_run_metadata(
            root / directory / "run_metadata.tsv",
            manifest,
            expected_p3d=expected_p3d,
            expected_kinship_action=expected_kinship_action,
        )
        trait_summaries: list[dict[str, Any]] = []
        mode_marker_universes: dict[str, dict[str, tuple[str, int]]] = {}
        for trait in TRAITS:
            rows = _read_mapping(root / directory / f"{TRAIT_SLUGS[trait]}_raw_mapping.tsv")
            metadata_marker_count = int(metadata[f"{TRAIT_SLUGS[trait]}_tested_markers"])
            if metadata_marker_count != len(rows):
                raise ValueError(
                    f"rrBLUP metadata/result marker count differs for {mode}/{trait}: "
                    f"metadata={metadata_marker_count}, result={len(rows)}"
                )
            mode_marker_universes[trait] = {
                row["marker"]: (row["CHROM"], row["POS"]) for row in rows
            }
            nonzero = {row["marker"]: row for row in rows if row["log10p"] != 0}
            published = reference[trait]
            shared = sorted(set(nonzero) & set(published))
            observed_values = [nonzero[marker]["log10p"] for marker in shared]
            expected_values = [published[marker]["log10p"] for marker in shared]
            significant = [row for row in rows if row["log10p"] >= threshold]
            observed_intervals = _intervals(
                rows,
                threshold,
                manifest.association.ci_markers_each_side,
                manifest.association.qtl_grouping_marker_gap,
            )
            expected_intervals = [
                item.model_dump()
                for item in manifest.published_intervals
                if item.trait == trait
            ]
            observed_interval_keys = [
                (item["chromosome"], item["start"], item["end"], item["peak"])
                for item in observed_intervals
            ]
            expected_interval_keys = [
                (item["chromosome"], item["start"], item["end"], item["peak"])
                for item in expected_intervals
            ]
            top = sorted(rows, key=lambda row: (-row["log10p"], row["marker"]))[:10]
            trait_summaries.append(
                {
                    "trait": trait,
                    "raw_marker_count": len(rows),
                    "nonzero_marker_count": len(nonzero),
                    "published_marker_count": len(published),
                    "shared_marker_count": len(shared),
                    "marker_set_exact": set(nonzero) == set(published),
                    "coordinate_set_exact": all(
                        nonzero[marker]["CHROM"] == published[marker]["CHROM"]
                        and nonzero[marker]["POS"] == published[marker]["POS"]
                        for marker in shared
                    ),
                    "max_abs_log10p_difference": max(
                        (abs(x - y) for x, y in zip(observed_values, expected_values)),
                        default=None,
                    ),
                    "spearman_log10p": _correlation(
                        _average_ranks(observed_values), _average_ranks(expected_values)
                    ),
                    "lambda_gc": _lambda_gc(rows),
                    "threshold_log10p": threshold,
                    "significant_marker_count": len(significant),
                    "expected_significant_marker_count": (
                        manifest.expected_significant_markers[trait]
                    ),
                    "significant_count_exact": len(significant)
                    == manifest.expected_significant_markers[trait],
                    "intervals": observed_intervals,
                    "expected_intervals": expected_intervals,
                    "intervals_exact": observed_interval_keys == expected_interval_keys,
                    "top_markers": top,
                }
            )
        summary_modes[mode] = {"run_metadata": metadata, "traits": trait_summaries}
        marker_universes[mode] = mode_marker_universes

    primary_traits = summary_modes["primary_emma_p3d_false"]["traits"]
    all_mode_traits = [
        item
        for mode_summary in summary_modes.values()
        for item in mode_summary["traits"]
    ]
    gates = {
        "raw_marker_count_21342_all_traits": all(
            item["raw_marker_count"] == manifest.preprocessing.expected_complete_markers
            for item in primary_traits
        ),
        "complete_marker_count_21342_primary": int(
            summary_modes["primary_emma_p3d_false"]["run_metadata"]["complete_markers"]
        )
        == manifest.preprocessing.expected_complete_markers,
        "complete_marker_count_21342_both_modes": all(
            int(mode_summary["run_metadata"]["complete_markers"])
            == manifest.preprocessing.expected_complete_markers
            for mode_summary in summary_modes.values()
        ),
        "raw_marker_count_21342_both_modes": all(
            item["raw_marker_count"] == manifest.preprocessing.expected_complete_markers
            for item in all_mode_traits
        ),
        "marker_coordinates_identical_between_modes_all_traits": all(
            marker_universes["primary_emma_p3d_false"][trait]
            == marker_universes["sensitivity_emmax_p3d_true"][trait]
            for trait in TRAITS
        ),
        "published_nonzero_count_20946_all_traits": all(
            item["published_marker_count"]
            == manifest.association.expected_published_nonzero_markers
            for item in primary_traits
        ),
        "published_marker_set_exact_all_traits": all(
            item["marker_set_exact"] and item["coordinate_set_exact"] for item in primary_traits
        ),
        "numerical_tolerance_1e_6_all_traits": all(
            item["max_abs_log10p_difference"] is not None
            and item["max_abs_log10p_difference"] <= 1e-6
            for item in primary_traits
        ),
        "significant_counts_exact_all_traits": all(
            item["significant_count_exact"] for item in primary_traits
        ),
        "intervals_exact_all_traits": all(item["intervals_exact"] for item in primary_traits),
    }
    statistical_match = all(gates.values())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": manifest.analysis_id,
        "status": manifest.status,
        "strict_reproduction_claim_permitted": False,
        "biological_claims_permitted": False,
        "classification": (
            "functional_statistical_match_with_unresolved_source_provenance"
            if statistical_match
            else "partial_historical_isotype_reconstruction"
        ),
        "deposited_effective_tests": manifest.association.effective_tests_deposited,
        "deposited_threshold_log10p": threshold,
        "effective_tests_recomputed": False,
        "functional_match_gates": gates,
        "modes": summary_modes,
        "provenance_gaps": manifest.provenance_gaps,
    }

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"summary output must not already exist: {destination}")
    destination.mkdir(parents=True)
    _write_json(destination / "ws276_reconstruction_summary.json", summary)
    lines = [
        "# WS276 abamectin historical isotype reconstruction",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "Strict-reproduction and biological claims remain disabled for this legacy "
        "isotype lane; the recovered all-strain release payload failed the separate "
        "marker-identity gate, and the original execution receipt remains missing.",
        "",
        "| Mode | Trait | Raw markers | Shared S3 markers | Max delta log10p | "
        "Lambda GC | Significant | Intervals exact |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for mode, mode_summary in summary_modes.items():
        for item in mode_summary["traits"]:
            delta = item["max_abs_log10p_difference"]
            lines.append(
                f"| {mode} | {item['trait']} | {item['raw_marker_count']} | "
                f"{item['shared_marker_count']} | {delta if delta is not None else 'NA'} | "
                f"{item['lambda_gc']:.4f} | {item['significant_marker_count']} | "
                f"{item['intervals_exact']} |"
            )
    (destination / "ws276_reconstruction_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return summary
