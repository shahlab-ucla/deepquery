"""Summarize the frozen WS276 current-marker sensitivity lane.

This module is deliberately separate from :mod:`qtl_ws276`.  The historical
reconstruction keeps its 21,342-marker acceptance gate; this lane describes
association results from the 75,852-marker universe emitted by the same
preprocessing command on the currently available isotype VCF.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
import uuid
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic import model_validator


SCHEMA_VERSION = "wormctx-abamectin-ws276-observed-sensitivity-1.0"
ANALYSIS_ID = "abamectin_qtl_ws276_current_isotype_marker_universe_sensitivity_v1"
CLASSIFICATION = "exploratory_current_ws276_isotype_marker_universe_sensitivity"
STATUS = "predeclared_preliminary_same_phenotype_source_representation_sensitivity"
PARENT_RUN_ID = "abamectin-ws276-reconstruction-20260718T133456Z-46409f4e0c50"
PARENT_SOURCE_GIT_COMMIT = "46409f4e0c5025cc8b4181e4d1d7a55d588b80f1"
PARENT_FAILURE_REASON = (
    "complete-case marker set is not exactly the historical 21,342 rows"
)
TRAITS = ("mean.EXT", "mean.TOF", "mean.norm.EXT", "norm.n")
TRAIT_SLUGS = {
    "mean.EXT": "mean_EXT",
    "mean.TOF": "mean_TOF",
    "mean.norm.EXT": "mean_norm_EXT",
    "norm.n": "norm_n",
}
CHROMOSOME_ORDER = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "X": 6, "MtDNA": 7}
EXPECTED_CURRENT_MARKERS = 75_852
EXPECTED_COMPLETE_MARKERS = 75_831
EXPECTED_INCOMPLETE_MARKERS = 21
EXPECTED_PUBLISHED_MARKERS = 20_946
EXPECTED_SHARED_COORDINATES = 12_601
EXPECTED_PUBLISHED_ONLY = 8_345
EXPECTED_CURRENT_ONLY = 63_251
EXPECTED_MARKER_LIST_SHA256 = (
    "1f0a0986ecd10f6f89dd1beb88fa240d506cfa870091c56436c95c0973a75675"
)
EXPECTED_ANALYSIS_MARKER_LIST_SHA256 = (
    "181d3dfe2db799b544dc1b17098095f591cbee3f77ef0c4a604211227a1cebe2"
)
EXPECTED_ANALYSIS_SHARED_COORDINATES = 12_601
EXPECTED_ANALYSIS_PUBLISHED_ONLY = 8_345
EXPECTED_ANALYSIS_CURRENT_ONLY = 63_230
EXPECTED_INCOMPLETE_PUBLISHED_OVERLAP = 0
PUBLISHED_S3_SHA256 = "7f846221728b568e0eb2967dead051f6e21f0b316a9677bccb6580d640e692b4"
ALPHA = 0.05
WITHIN_TRAIT_THRESHOLD = 6.18099703203937
FOUR_TRAIT_FAMILYWISE_THRESHOLD = 6.78305702336733
DEPOSITED_THRESHOLD_REFERENCE_ONLY = 4.28448307163607
TOP_SHARED_COUNT = 100
TOP_SHARED_FRACTION = 0.01
_CHI_SQUARE_1_MEDIAN = 0.4549364231195727

Coordinate = tuple[str, int]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _validate_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
    return value


class PublishedInterval(_StrictModel):
    trait: Literal["mean.EXT", "mean.TOF", "norm.n"]
    chromosome: Literal["II", "III", "V"]
    start: int = Field(gt=0)
    peak: int = Field(gt=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_geometry(self) -> "PublishedInterval":
        if not self.start <= self.peak <= self.end:
            raise ValueError("published interval must satisfy start <= peak <= end")
        return self


class AssociationContract(_StrictModel):
    method: Literal["rrBLUP_GWAS_EMMA"]
    rrblup_version: Literal["4.6"]
    r_version: Literal["3.6.0"]
    min_maf: Literal[0.05]
    primary_p3d: Literal[False]
    calibration_p3d: Literal[True]
    within_trait_bonferroni_neg_log10_p: float
    four_trait_familywise_neg_log10_p: float
    deposited_threshold_reference_only: float
    p3d_true_can_originate_primary_hit: Literal[False]

    @model_validator(mode="after")
    def frozen_thresholds(self) -> "AssociationContract":
        expected_within = -math.log10(ALPHA / EXPECTED_CURRENT_MARKERS)
        expected_familywise = -math.log10(
            ALPHA / (len(TRAITS) * EXPECTED_CURRENT_MARKERS)
        )
        checks = (
            (
                self.within_trait_bonferroni_neg_log10_p,
                expected_within,
                "within-trait threshold",
            ),
            (
                self.four_trait_familywise_neg_log10_p,
                expected_familywise,
                "four-trait familywise threshold",
            ),
            (
                self.deposited_threshold_reference_only,
                DEPOSITED_THRESHOLD_REFERENCE_ONLY,
                "deposited reference threshold",
            ),
        )
        for observed, expected, label in checks:
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-13):
                raise ValueError(f"{label} differs from the frozen sensitivity contract")
        return self


class MarkerUniverseContract(_StrictModel):
    selection: Literal[
        "all markers emitted by the frozen public-candidate preprocessing command used by "
        "the A1 diagnostic on the current isotype VCF"
    ]
    historical_command_identity_proven: Literal[False]
    phenotype_guided_selection: Literal[False]
    ld_parameter_retuning: Literal[False]
    downsample_to_historical_count: Literal[False]
    restrict_scan_to_published_overlap: Literal[False]
    chromosome_counts: dict[str, int]

    @model_validator(mode="after")
    def frozen_chromosome_counts(self) -> "MarkerUniverseContract":
        expected = {
            "I": 5_661,
            "II": 20_201,
            "III": 12_811,
            "IV": 11_282,
            "V": 18_633,
            "X": 7_225,
            "MtDNA": 39,
        }
        if self.chromosome_counts != expected:
            raise ValueError("chromosome counts differ from the frozen 75,852-marker list")
        if sum(self.chromosome_counts.values()) != EXPECTED_CURRENT_MARKERS:
            raise ValueError("chromosome counts do not sum to 75,852")
        return self


class ArtifactReceipt(_StrictModel):
    bytes: int = Field(gt=0)
    sha256: str

    _valid_sha256 = field_validator("sha256")(_validate_sha256)


class ParentArtifacts(_StrictModel):
    filtered_vcf: ArtifactReceipt
    filtered_vcf_index: ArtifactReceipt
    sample_order: ArtifactReceipt
    cohort_receipt: ArtifactReceipt
    source_to_analysis: ArtifactReceipt
    genotype_matrix: ArtifactReceipt
    analysis_marker_list: ArtifactReceipt
    incomplete_marker_list: ArtifactReceipt
    preprocessing_counts: ArtifactReceipt
    incomplete_by_chromosome: ArtifactReceipt
    failure_marker: ArtifactReceipt
    deployer_runner_exit: ArtifactReceipt
    phenotypes: dict[str, str]

    @field_validator("phenotypes")
    @classmethod
    def valid_phenotypes(cls, value: dict[str, str]) -> dict[str, str]:
        expected_names = {f"{TRAIT_SLUGS[trait]}.tsv" for trait in TRAITS}
        if set(value) != expected_names:
            raise ValueError("parent phenotype receipts must cover exactly the four traits")
        return {name: _validate_sha256(digest) for name, digest in value.items()}

    @model_validator(mode="after")
    def frozen_artifacts(self) -> "ParentArtifacts":
        expected = {
            "filtered_vcf": {
                "bytes": 46_032_067,
                "sha256": "15df06950eb3e2d8649a9c1a21081b844ca6af620566311e31279b3edc95ecee",
            },
            "filtered_vcf_index": {
                "bytes": 63_798,
                "sha256": "2ab386a58e57dd9201e34628445f8954d349e5d5409f23865282bf1392f8577b",
            },
            "sample_order": {
                "bytes": 1_425,
                "sha256": "4c063af69f395a0cc465a8da64815beccec52e9e3dff6036d0040d7b2e7b51c3",
            },
            "cohort_receipt": {
                "bytes": 1_565,
                "sha256": "afc5b886a1a807273612088686e2da1f84dda0db9892aba85bedbf7eeefe0b54",
            },
            "source_to_analysis": {
                "bytes": 5_917,
                "sha256": "64ee0f6196b14feeaeec43595d6eb92b779c4f831be284120faf89709672e4ab",
            },
            "genotype_matrix": {
                "bytes": 44_961_783,
                "sha256": "6a006610325359ebeb2e7910eb98b58ebbb5dbe8650564bdc70785add8000cb1",
            },
            "analysis_marker_list": {
                "bytes": 843_976,
                "sha256": "181d3dfe2db799b544dc1b17098095f591cbee3f77ef0c4a604211227a1cebe2",
            },
            "incomplete_marker_list": {
                "bytes": 237,
                "sha256": "ce59fde61770d90c9d9a1eaecc740912ce0175262500015f149a264a00950a4f",
            },
            "preprocessing_counts": {
                "bytes": 170,
                "sha256": "77fd545deaaf784611b22540ccb873aaafadd5ecd6ab8dd9a56f4fac5435e7ae",
            },
            "incomplete_by_chromosome": {
                "bytes": 67,
                "sha256": "df6574aea35dbdb09fd776c32d89383fe08ea33315e19e7c801087fd72a0db6d",
            },
            "failure_marker": {
                "bytes": 51,
                "sha256": "c2e9a30a6a8702b331bab66daf512d2f955a86633e9d6dc2a0261427c7f66cdf",
            },
            "deployer_runner_exit": {
                "bytes": 3,
                "sha256": "6442bc26a7c562f5afe6467dab36365c709909f6a81afcecfc0c25cff0f1bab0",
            },
            "phenotypes": {
                "mean_EXT.tsv": (
                    "471b2b3b5e6d65f8e1ac12f973a0ec929967aa6d4ebef49ed05c28b928bc9832"
                ),
                "mean_TOF.tsv": (
                    "a3335cbf4768f9e9573134daecfed65560ee2cf7e961c32c39eaaccbf4660b87"
                ),
                "mean_norm_EXT.tsv": (
                    "d9f738abd9d6bb8ea019443dea11590419c31af2bc069ded34fc062cfa49a1e0"
                ),
                "norm_n.tsv": (
                    "c7c4c54c1c30fdc0dd5ade0d4e8ea0634417ba05a6c9a2502231082c4d57ac4a"
                ),
            },
        }
        if self.model_dump() != expected:
            raise ValueError("parent artifact receipts differ from the frozen A1 evidence")
        return self


class ClaimsContract(_StrictModel):
    permitted: list[str]
    prohibited: list[str]

    @model_validator(mode="after")
    def frozen_claims(self) -> "ClaimsContract":
        if self.permitted != [
            "preliminary same-phenotype source-representation sensitivity",
            "descriptive coordinate concordance",
            "descriptive rank robustness",
        ]:
            raise ValueError("permitted claims differ from the frozen exploratory boundary")
        if self.prohibited != [
            "reproduction",
            "replication",
            "independent validation",
            "recovery of the historical marker universe",
            "causal locus, gene, or allele",
            "biological negative conclusion",
        ]:
            raise ValueError("prohibited claims differ from the frozen exploratory boundary")
        return self


class Ws276SensitivityContract(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    analysis_id: Literal[ANALYSIS_ID]
    classification: Literal[CLASSIFICATION]
    status: Literal[STATUS]
    strict_reproduction_claim_permitted: Literal[False]
    biological_claims_permitted: Literal[False]
    parent_run_id: Literal[PARENT_RUN_ID]
    parent_source_git_commit: Literal[PARENT_SOURCE_GIT_COMMIT]
    parent_failure_exit_code: Literal[70]
    parent_failure_reason: Literal[PARENT_FAILURE_REASON]
    marker_list_sha256: Literal[EXPECTED_MARKER_LIST_SHA256]
    expected_current_markers: Literal[EXPECTED_CURRENT_MARKERS]
    expected_complete_markers: Literal[EXPECTED_COMPLETE_MARKERS]
    expected_incomplete_markers: Literal[EXPECTED_INCOMPLETE_MARKERS]
    analysis_marker_list_sha256: Literal[EXPECTED_ANALYSIS_MARKER_LIST_SHA256]
    expected_published_markers: Literal[EXPECTED_PUBLISHED_MARKERS]
    expected_shared_coordinates: Literal[EXPECTED_SHARED_COORDINATES]
    expected_published_only: Literal[EXPECTED_PUBLISHED_ONLY]
    expected_current_only: Literal[EXPECTED_CURRENT_ONLY]
    expected_analysis_shared_coordinates: Literal[EXPECTED_ANALYSIS_SHARED_COORDINATES]
    expected_analysis_published_only: Literal[EXPECTED_ANALYSIS_PUBLISHED_ONLY]
    expected_analysis_current_only: Literal[EXPECTED_ANALYSIS_CURRENT_ONLY]
    expected_incomplete_published_overlap: Literal[EXPECTED_INCOMPLETE_PUBLISHED_OVERLAP]
    coordinate_identity_caveat: Literal[
        "File S3 lacks REF and ALT, so shared CHROM:POS coordinates do not prove allele identity."
    ]
    traits: list[str]
    alpha: Literal[0.05]
    association: AssociationContract
    marker_universe: MarkerUniverseContract
    parent_artifacts: ParentArtifacts
    published_intervals: list[PublishedInterval]
    claims: ClaimsContract

    @model_validator(mode="after")
    def frozen_contract(self) -> "Ws276SensitivityContract":
        if self.traits != list(TRAITS):
            raise ValueError(f"traits must be {list(TRAITS)}")
        if self.expected_shared_coordinates + self.expected_current_only != (
            self.expected_current_markers
        ):
            raise ValueError("shared plus current-only coordinates must equal current markers")
        if self.expected_complete_markers + self.expected_incomplete_markers != (
            self.expected_current_markers
        ):
            raise ValueError("complete plus incomplete markers must equal current markers")
        if self.expected_shared_coordinates + self.expected_published_only != (
            self.expected_published_markers
        ):
            raise ValueError("shared plus published-only coordinates must equal published markers")
        if (
            self.expected_analysis_shared_coordinates
            + self.expected_analysis_current_only
            != self.expected_complete_markers
        ):
            raise ValueError(
                "analysis shared plus analysis current-only must equal complete markers"
            )
        if (
            self.expected_analysis_shared_coordinates
            + self.expected_analysis_published_only
            != self.expected_published_markers
        ):
            raise ValueError(
                "analysis shared plus analysis published-only must equal published markers"
            )
        if (
            self.expected_analysis_shared_coordinates
            + self.expected_incomplete_published_overlap
            != self.expected_shared_coordinates
        ):
            raise ValueError(
                "analysis shared plus excluded published overlap must equal full shared"
            )
        expected_intervals = [
            ("mean.EXT", "V", 1_747_612, 2_693_128, 4_333_001),
            ("mean.TOF", "V", 1_757_246, 2_693_128, 4_333_001),
            ("mean.TOF", "V", 15_983_112, 16_276_775, 16_599_066),
            ("norm.n", "II", 13_756_151, 14_121_786, 14_937_792),
            ("norm.n", "III", 3_061_633, 3_526_374, 4_632_949),
            ("norm.n", "V", 13_606_517, 15_965_095, 16_754_986),
        ]
        observed_intervals = [
            (item.trait, item.chromosome, item.start, item.peak, item.end)
            for item in self.published_intervals
        ]
        if observed_intervals != expected_intervals:
            raise ValueError("published intervals differ from the frozen File S3 contract")
        return self


def load_ws276_sensitivity_contract(path: str | Path) -> Ws276SensitivityContract:
    """Load the fail-closed A2 analysis contract."""

    return Ws276SensitivityContract.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
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


def _coordinate_sort_key(coordinate: Coordinate) -> tuple[int, int]:
    return CHROMOSOME_ORDER[coordinate[0]], coordinate[1]


def _read_marker_list(
    path: Path, contract: Ws276SensitivityContract
) -> set[Coordinate]:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen current-marker list: {path}")
    observed_sha256 = _sha256(path)
    if observed_sha256 != contract.marker_list_sha256:
        raise ValueError(
            "current-marker list checksum mismatch: "
            f"expected {contract.marker_list_sha256}, observed {observed_sha256}"
        )
    coordinates: set[Coordinate] = set()
    chromosome_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2 or fields[0] not in CHROMOSOME_ORDER:
                raise ValueError(f"invalid marker-list row at {path}:{line_number}")
            try:
                position = int(fields[1])
            except ValueError as exc:
                raise ValueError(f"invalid marker position at {path}:{line_number}") from exc
            coordinate = (fields[0], position)
            if position <= 0 or coordinate in coordinates:
                raise ValueError(f"duplicate or invalid marker at {path}:{line_number}")
            coordinates.add(coordinate)
            chromosome_counts[fields[0]] += 1
    if len(coordinates) != contract.expected_current_markers:
        raise ValueError(
            "current-marker count mismatch: "
            f"expected {contract.expected_current_markers}, observed {len(coordinates)}"
        )
    if dict(chromosome_counts) != contract.marker_universe.chromosome_counts:
        raise ValueError(
            "current-marker chromosome counts differ from the predeclared contract: "
            f"{dict(chromosome_counts)}"
        )
    return coordinates


def _read_analysis_marker_list(
    path: Path,
    current_coordinates: set[Coordinate],
    contract: Ws276SensitivityContract,
) -> tuple[set[Coordinate], set[Coordinate]]:
    """Read the post-``na.omit`` universe and prove its source-only derivation."""

    if not path.is_file():
        raise FileNotFoundError(f"missing complete analysis-marker list: {path}")
    observed_sha256 = _sha256(path)
    if observed_sha256 != contract.analysis_marker_list_sha256:
        raise ValueError(
            "analysis-marker list checksum mismatch: "
            f"expected {contract.analysis_marker_list_sha256}, observed {observed_sha256}"
        )
    coordinates: set[Coordinate] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2 or fields[0] not in CHROMOSOME_ORDER:
                raise ValueError(f"invalid analysis-marker row at {path}:{line_number}")
            try:
                position = int(fields[1])
            except ValueError as exc:
                raise ValueError(
                    f"invalid analysis-marker position at {path}:{line_number}"
                ) from exc
            coordinate = (fields[0], position)
            if position <= 0 or coordinate in coordinates:
                raise ValueError(
                    f"duplicate or invalid analysis marker at {path}:{line_number}"
                )
            coordinates.add(coordinate)
    if len(coordinates) != contract.expected_complete_markers:
        raise ValueError(
            "complete analysis-marker count mismatch: "
            f"expected {contract.expected_complete_markers}, observed {len(coordinates)}"
        )
    extra = coordinates - current_coordinates
    if extra:
        raise ValueError(
            "analysis-marker list is not a subset of the frozen current-marker list: "
            f"first_extra={sorted(extra, key=_coordinate_sort_key)[:3]}"
        )
    excluded = current_coordinates - coordinates
    if len(excluded) != contract.expected_incomplete_markers:
        raise ValueError(
            "incomplete-marker difference mismatch: "
            f"expected {contract.expected_incomplete_markers}, observed {len(excluded)}"
        )
    excluded_chromosomes = Counter(coordinate[0] for coordinate in excluded)
    if excluded_chromosomes != {"MtDNA": contract.expected_incomplete_markers}:
        raise ValueError(
            "the 21 complete-case exclusions must all be MtDNA markers: "
            f"observed={dict(excluded_chromosomes)}"
        )
    return coordinates, excluded


def _read_mapping(path: Path, expected_coordinates: set[Coordinate]) -> dict[Coordinate, float]:
    if not path.is_file():
        raise FileNotFoundError(f"missing rrBLUP sensitivity mapping: {path}")
    rows: dict[Coordinate, float] = {}
    markers: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["marker", "CHROM", "POS", "log10p"]:
            raise ValueError(f"rrBLUP mapping header mismatch in {path}: {reader.fieldnames}")
        for line_number, row in enumerate(reader, 2):
            marker = (row["marker"] or "").strip()
            chromosome = (row["CHROM"] or "").strip()
            try:
                position = int(row["POS"])
                score = float(row["log10p"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid rrBLUP value at {path}:{line_number}") from exc
            coordinate = (chromosome, position)
            if (
                chromosome not in CHROMOSOME_ORDER
                or position <= 0
                or not marker
                or marker != f"{chromosome}_{position}"
                or marker in markers
                or coordinate in rows
                or not math.isfinite(score)
                or score < 0
            ):
                raise ValueError(f"invalid or duplicate rrBLUP marker at {path}:{line_number}")
            markers.add(marker)
            rows[coordinate] = score
    observed_coordinates = set(rows)
    if observed_coordinates != expected_coordinates:
        missing = sorted(expected_coordinates - observed_coordinates, key=_coordinate_sort_key)
        extra = sorted(observed_coordinates - expected_coordinates, key=_coordinate_sort_key)
        raise ValueError(
            f"rrBLUP coordinate universe differs in {path}: "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"first_missing={missing[:3]}, first_extra={extra[:3]}"
        )
    return rows


def _read_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing rrBLUP sensitivity metadata: {path}")
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
    return values


def _validate_metadata(
    values: dict[str, str],
    contract: Ws276SensitivityContract,
    *,
    expected_p3d: bool,
    expected_kinship_action: str,
    path: Path,
) -> None:
    expected = {
        "classification": CLASSIFICATION,
        "strict_reproduction": "false",
        "biological_claims_permitted": "false",
        "P3D": str(expected_p3d).upper(),
        "rrBLUP_version": contract.association.rrblup_version,
        "input_samples": "209",
        "input_markers": str(contract.expected_current_markers),
        "complete_markers": str(contract.expected_complete_markers),
        "MtDNA_preserved": "true",
        "kinship_action": expected_kinship_action,
    }
    expected.update(
        {
            f"{TRAIT_SLUGS[trait]}_tested_markers": str(
                contract.expected_complete_markers
            )
            for trait in TRAITS
        }
    )
    mismatches = {
        key: {"expected": expected_value, "observed": values.get(key)}
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"rrBLUP sensitivity metadata contract failed in {path}: {mismatches}")
    try:
        cores = int(values["cores"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"rrBLUP cores metadata is missing or invalid in {path}") from exc
    if cores <= 0 or not values.get("kinship_path"):
        raise ValueError(f"rrBLUP cores or kinship path is invalid in {path}")
    try:
        observed_genotype_sha256 = _validate_sha256(values["genotype_matrix_sha256"])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"rrBLUP genotype-matrix SHA-256 is missing or invalid in {path}"
        ) from exc
    if observed_genotype_sha256 != contract.parent_artifacts.genotype_matrix.sha256:
        raise ValueError(
            f"rrBLUP genotype-matrix SHA-256 differs from the frozen A1 parent in {path}"
        )


def _validate_cross_mode_metadata(
    primary: dict[str, str], calibration: dict[str, str]
) -> None:
    for key, label in (
        ("kinship_path", "kinship path"),
        ("genotype_matrix_sha256", "genotype-matrix SHA-256"),
    ):
        if primary[key] != calibration[key]:
            raise ValueError(f"P3D modes did not reuse the exact same {label}")


def _read_published_s3(path: Path) -> dict[str, dict[Coordinate, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen published File S3: {path}")
    observed_sha256 = _sha256(path)
    if observed_sha256 != PUBLISHED_S3_SHA256:
        raise ValueError(
            "published File S3 checksum mismatch: "
            f"expected {PUBLISHED_S3_SHA256}, observed {observed_sha256}"
        )
    result: dict[str, dict[Coordinate, float]] = {trait: {} for trait in TRAITS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"marker", "CHROM", "POS", "log10p", "trait"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("published File S3 header lacks mapping columns")
        for line_number, row in enumerate(reader, 2):
            raw_trait = (row["trait"] or "").strip()
            trait = raw_trait.removeprefix("abamectin_")
            chromosome = (row["CHROM"] or "").strip()
            try:
                position = int(row["POS"])
                score = float(row["log10p"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid File S3 value at row {line_number}") from exc
            coordinate = (chromosome, position)
            if (
                trait not in result
                or chromosome not in CHROMOSOME_ORDER
                or position <= 0
                or not math.isfinite(score)
                or score < 0
            ):
                raise ValueError(f"out-of-range File S3 value at row {line_number}")
            previous = result[trait].get(coordinate)
            if previous is not None and previous != score:
                raise ValueError(
                    f"inconsistent duplicated File S3 coordinate: {trait}/{coordinate}"
                )
            result[trait][coordinate] = score
    return result


def _validate_coordinate_contract(
    current: set[Coordinate],
    published: dict[str, dict[Coordinate, float]],
    contract: Ws276SensitivityContract,
) -> tuple[set[Coordinate], set[Coordinate]]:
    published_sets = [set(published[trait]) for trait in TRAITS]
    if any(values != published_sets[0] for values in published_sets[1:]):
        raise ValueError("published File S3 coordinate universe differs between traits")
    published_coordinates = published_sets[0]
    shared = current & published_coordinates
    published_only = published_coordinates - current
    current_only = current - published_coordinates
    observed = {
        "current": len(current),
        "published": len(published_coordinates),
        "shared": len(shared),
        "published_only": len(published_only),
        "current_only": len(current_only),
    }
    expected = {
        "current": contract.expected_current_markers,
        "published": contract.expected_published_markers,
        "shared": contract.expected_shared_coordinates,
        "published_only": contract.expected_published_only,
        "current_only": contract.expected_current_only,
    }
    if observed != expected:
        raise ValueError(
            f"current-versus-published coordinate contract failed: "
            f"expected={expected}, observed={observed}"
        )
    return published_coordinates, shared


def _validate_analysis_coordinate_contract(
    analysis: set[Coordinate],
    incomplete: set[Coordinate],
    published: set[Coordinate],
    contract: Ws276SensitivityContract,
) -> set[Coordinate]:
    shared = analysis & published
    observed = {
        "analysis_shared": len(shared),
        "analysis_published_only": len(published - analysis),
        "analysis_current_only": len(analysis - published),
        "incomplete_published_overlap": len(incomplete & published),
    }
    expected = {
        "analysis_shared": contract.expected_analysis_shared_coordinates,
        "analysis_published_only": contract.expected_analysis_published_only,
        "analysis_current_only": contract.expected_analysis_current_only,
        "incomplete_published_overlap": contract.expected_incomplete_published_overlap,
    }
    if observed != expected:
        raise ValueError(
            "analysis-versus-published coordinate contract failed: "
            f"expected={expected}, observed={observed}"
        )
    return shared


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


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def _spearman(
    left: dict[Coordinate, float],
    right: dict[Coordinate, float],
    coordinates: set[Coordinate],
) -> float | None:
    ordered = sorted(coordinates, key=_coordinate_sort_key)
    return _correlation(
        _average_ranks([left[coordinate] for coordinate in ordered]),
        _average_ranks([right[coordinate] for coordinate in ordered]),
    )


def _top_coordinates(
    scores: dict[Coordinate, float], coordinates: set[Coordinate], count: int
) -> set[Coordinate]:
    if count <= 0 or count > len(coordinates):
        raise ValueError("top-coordinate count is outside the available coordinate universe")
    ordered = sorted(
        coordinates,
        key=lambda coordinate: (-scores[coordinate], *_coordinate_sort_key(coordinate)),
    )
    return set(ordered[:count])


def _jaccard(left: set[Coordinate], right: set[Coordinate]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _top_concordance(
    left: dict[Coordinate, float],
    right: dict[Coordinate, float],
    coordinates: set[Coordinate],
    count: int,
) -> dict[str, Any]:
    left_top = _top_coordinates(left, coordinates, count)
    right_top = _top_coordinates(right, coordinates, count)
    return {
        "set_size": count,
        "intersection_count": len(left_top & right_top),
        "union_count": len(left_top | right_top),
        "jaccard": _jaccard(left_top, right_top),
    }


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _lambda_gc(scores: dict[Coordinate, float]) -> float:
    chi_square: list[float] = []
    normal = NormalDist()
    for score in scores.values():
        p_value = 10.0 ** (-score)
        if p_value >= 1.0:
            chi_square.append(0.0)
        elif p_value > 0.0:
            upper_cdf = 1.0 - max(p_value / 2.0, 1e-16)
            chi_square.append(normal.inv_cdf(upper_cdf) ** 2)
    return statistics.median(chi_square) / _CHI_SQUARE_1_MEDIAN


def _coordinate_record(coordinate: Coordinate, score: float) -> dict[str, Any]:
    return {
        "marker": f"{coordinate[0]}_{coordinate[1]}",
        "CHROM": coordinate[0],
        "POS": coordinate[1],
        "log10p": score,
    }


def _interval_metrics(
    trait: str,
    scores: dict[Coordinate, float],
    published: dict[Coordinate, float],
    intervals: list[PublishedInterval],
    *,
    primary: bool,
) -> list[dict[str, Any]]:
    chromosome_coordinates: dict[str, list[Coordinate]] = {}
    for coordinate in scores:
        chromosome_coordinates.setdefault(coordinate[0], []).append(coordinate)
    result: list[dict[str, Any]] = []
    for interval in intervals:
        if interval.trait != trait:
            continue
        in_interval = [
            coordinate
            for coordinate in chromosome_coordinates.get(interval.chromosome, [])
            if interval.start <= coordinate[1] <= interval.end
        ]
        current_maximum = (
            min(
                in_interval,
                key=lambda coordinate: (-scores[coordinate], coordinate[1]),
            )
            if in_interval
            else None
        )
        chromosome_markers = chromosome_coordinates.get(interval.chromosome, [])
        nearest = (
            min(
                chromosome_markers,
                key=lambda coordinate: (abs(coordinate[1] - interval.peak), coordinate[1]),
            )
            if chromosome_markers
            else None
        )
        exact_peak = (interval.chromosome, interval.peak)
        maximum_score = scores[current_maximum] if current_maximum is not None else None
        nearest_score = scores[nearest] if nearest is not None else None
        exact_score = scores.get(exact_peak)
        result.append(
            {
                "trait": trait,
                "chromosome": interval.chromosome,
                "published_start": interval.start,
                "published_peak": interval.peak,
                "published_end": interval.end,
                "published_peak_log10p": published.get(exact_peak),
                "current_marker_count_in_interval": len(in_interval),
                "current_interval_maximum": (
                    _coordinate_record(current_maximum, maximum_score)
                    if current_maximum is not None and maximum_score is not None
                    else None
                ),
                "current_interval_maximum_distance_from_published_peak": (
                    abs(current_maximum[1] - interval.peak)
                    if current_maximum is not None
                    else None
                ),
                "current_interval_maximum_crosses_within_trait_threshold": (
                    maximum_score is not None and maximum_score >= WITHIN_TRAIT_THRESHOLD
                ),
                "current_interval_maximum_crosses_familywise_threshold": (
                    maximum_score is not None
                    and maximum_score >= FOUR_TRAIT_FAMILYWISE_THRESHOLD
                ),
                "current_interval_familywise_screen_hit": (
                    primary
                    and maximum_score is not None
                    and maximum_score >= FOUR_TRAIT_FAMILYWISE_THRESHOLD
                ),
                "published_peak_exact_coordinate_present_current": exact_score is not None,
                "published_peak_exact_coordinate_current_log10p": exact_score,
                "nearest_current_marker_to_published_peak": (
                    _coordinate_record(nearest, nearest_score)
                    if nearest is not None and nearest_score is not None
                    else None
                ),
                "nearest_current_marker_distance": (
                    abs(nearest[1] - interval.peak) if nearest is not None else None
                ),
            }
        )
    return result


def _write_top20(
    path: Path,
    trait: str,
    mode: str,
    scores: dict[Coordinate, float],
    published_coordinates: set[Coordinate],
    intervals: list[PublishedInterval],
    *,
    primary: bool,
) -> None:
    ordered = sorted(
        scores,
        key=lambda coordinate: (-scores[coordinate], *_coordinate_sort_key(coordinate)),
    )[:20]
    trait_intervals = [interval for interval in intervals if interval.trait == trait]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "rank",
                "mode",
                "trait",
                "marker",
                "CHROM",
                "POS",
                "log10p",
                "crosses_within_trait_threshold",
                "crosses_four_trait_familywise_threshold",
                "familywise_screen_hit",
                "published_coordinate_overlap",
                "inside_published_interval",
            ]
        )
        for rank, coordinate in enumerate(ordered, 1):
            score = scores[coordinate]
            inside_interval = any(
                interval.chromosome == coordinate[0]
                and interval.start <= coordinate[1] <= interval.end
                for interval in trait_intervals
            )
            writer.writerow(
                [
                    rank,
                    mode,
                    trait,
                    f"{coordinate[0]}_{coordinate[1]}",
                    coordinate[0],
                    coordinate[1],
                    format(score, ".17g"),
                    str(score >= WITHIN_TRAIT_THRESHOLD).lower(),
                    str(score >= FOUR_TRAIT_FAMILYWISE_THRESHOLD).lower(),
                    str(primary and score >= FOUR_TRAIT_FAMILYWISE_THRESHOLD).lower(),
                    str(coordinate in published_coordinates).lower(),
                    str(inside_interval).lower(),
                ]
            )


def _mode_trait_summary(
    trait: str,
    mode: str,
    scores: dict[Coordinate, float],
    published: dict[Coordinate, float],
    shared: set[Coordinate],
    contract: Ws276SensitivityContract,
    top20_relative_path: str,
    *,
    primary: bool,
) -> dict[str, Any]:
    ordered_shared = sorted(shared, key=_coordinate_sort_key)
    absolute_differences = [
        abs(scores[coordinate] - published[coordinate]) for coordinate in ordered_shared
    ]
    top_one_percent_count = math.ceil(TOP_SHARED_FRACTION * len(shared))
    within_crossings = sum(score >= WITHIN_TRAIT_THRESHOLD for score in scores.values())
    familywise_crossings = sum(
        score >= FOUR_TRAIT_FAMILYWISE_THRESHOLD for score in scores.values()
    )
    return {
        "trait": trait,
        "mode": mode,
        "P3D": not primary,
        "raw_marker_count": len(scores),
        "nonzero_marker_count": sum(score != 0 for score in scores.values()),
        "lambda_gc": _lambda_gc(scores),
        "within_trait_threshold_crossing_count": within_crossings,
        "four_trait_familywise_threshold_crossing_count": familywise_crossings,
        "familywise_screen_hit_count": familywise_crossings if primary else 0,
        "p3d_true_can_originate_primary_hit": False,
        "top20_table": top20_relative_path,
        "concordance_to_published_same_phenotype": {
            "coordinate_identity_only": True,
            "coordinate_identity_caveat": contract.coordinate_identity_caveat,
            "shared_coordinate_count": len(shared),
            "spearman_log10p": _spearman(scores, published, shared),
            "median_absolute_log10p_difference": _quantile(absolute_differences, 0.5),
            "p95_absolute_log10p_difference": _quantile(absolute_differences, 0.95),
            "top_100": _top_concordance(
                scores, published, shared, min(TOP_SHARED_COUNT, len(shared))
            ),
            "top_1_percent": _top_concordance(
                scores, published, shared, top_one_percent_count
            ),
        },
        "published_interval_and_peak_metrics": _interval_metrics(
            trait,
            scores,
            published,
            contract.published_intervals,
            primary=primary,
        ),
    }


def summarize_ws276_current_marker_sensitivity(
    contract_path: str | Path,
    marker_list_path: str | Path,
    analysis_marker_list_path: str | Path,
    results_root: str | Path,
    published_s3_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Validate and summarize A2 without upgrading its exploratory claim boundary."""

    contract = load_ws276_sensitivity_contract(contract_path)
    marker_path = Path(marker_list_path).resolve()
    current_coordinates = _read_marker_list(marker_path, contract)
    analysis_marker_path = Path(analysis_marker_list_path).resolve()
    analysis_coordinates, incomplete_coordinates = _read_analysis_marker_list(
        analysis_marker_path, current_coordinates, contract
    )
    published_path = Path(published_s3_path).resolve()
    published = _read_published_s3(published_path)
    published_coordinates, full_shared = _validate_coordinate_contract(
        current_coordinates, published, contract
    )
    analysis_shared = _validate_analysis_coordinate_contract(
        analysis_coordinates,
        incomplete_coordinates,
        published_coordinates,
        contract,
    )

    root = Path(results_root).resolve()
    modes = {
        "primary_emma_p3d_false": ("primary", False, "created"),
        "calibration_emmax_p3d_true": ("p3d_true", True, "reused"),
    }
    mode_maps: dict[str, dict[str, dict[Coordinate, float]]] = {}
    mode_metadata: dict[str, dict[str, str]] = {}
    for mode, (directory, p3d, kinship_action) in modes.items():
        mode_dir = root / directory
        metadata_path = mode_dir / "run_metadata.tsv"
        metadata = _read_metadata(metadata_path)
        _validate_metadata(
            metadata,
            contract,
            expected_p3d=p3d,
            expected_kinship_action=kinship_action,
            path=metadata_path,
        )
        mode_metadata[mode] = metadata
        mode_maps[mode] = {
            trait: _read_mapping(
                mode_dir / f"{TRAIT_SLUGS[trait]}_raw_mapping.tsv",
                analysis_coordinates,
            )
            for trait in TRAITS
        }
    _validate_cross_mode_metadata(
        mode_metadata["primary_emma_p3d_false"],
        mode_metadata["calibration_emmax_p3d_true"],
    )

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"sensitivity summary output must not already exist: {destination}")
    # Keep the staging name short enough for Windows' legacy path limit while
    # retaining a collision-resistant, same-filesystem atomic replacement.
    stage = destination.with_name(f".a2-{uuid.uuid4().hex[:12]}")
    stage.mkdir(parents=True)
    try:
        top20_dir = stage / "top20"
        top20_dir.mkdir()
        summary_modes: dict[str, Any] = {}
        for mode, (_, p3d, _) in modes.items():
            primary = not p3d
            trait_summaries: list[dict[str, Any]] = []
            for trait in TRAITS:
                mode_slug = "primary" if primary else "p3d_true"
                filename = f"{mode_slug}.{TRAIT_SLUGS[trait]}.top20.tsv"
                _write_top20(
                    top20_dir / filename,
                    trait,
                    mode,
                    mode_maps[mode][trait],
                    published_coordinates,
                    contract.published_intervals,
                    primary=primary,
                )
                trait_summaries.append(
                    _mode_trait_summary(
                        trait,
                        mode,
                        mode_maps[mode][trait],
                        published[trait],
                        analysis_shared,
                        contract,
                        f"top20/{filename}",
                        primary=primary,
                    )
                )
            summary_modes[mode] = {
                "P3D": p3d,
                "role": "primary" if primary else "calibration_sensitivity",
                "can_originate_familywise_screen_hit": primary,
                "run_metadata": mode_metadata[mode],
                "traits": trait_summaries,
            }

        cross_mode: dict[str, Any] = {}
        cross_mode_top_count = math.ceil(
            TOP_SHARED_FRACTION * contract.expected_complete_markers
        )
        for trait in TRAITS:
            primary_scores = mode_maps["primary_emma_p3d_false"][trait]
            calibration_scores = mode_maps["calibration_emmax_p3d_true"][trait]
            primary_hits = {
                coordinate
                for coordinate, score in primary_scores.items()
                if score >= FOUR_TRAIT_FAMILYWISE_THRESHOLD
            }
            calibration_crossings = {
                coordinate
                for coordinate, score in calibration_scores.items()
                if score >= FOUR_TRAIT_FAMILYWISE_THRESHOLD
            }
            cross_mode[trait] = {
                "coordinate_count": len(analysis_coordinates),
                "spearman_log10p": _spearman(
                    primary_scores, calibration_scores, analysis_coordinates
                ),
                "top_100": _top_concordance(
                    primary_scores,
                    calibration_scores,
                    analysis_coordinates,
                    min(TOP_SHARED_COUNT, len(analysis_coordinates)),
                ),
                "top_1_percent": _top_concordance(
                    primary_scores,
                    calibration_scores,
                    analysis_coordinates,
                    cross_mode_top_count,
                ),
                "primary_familywise_screen_hit_count": len(primary_hits),
                "calibration_numeric_familywise_crossing_count": len(
                    calibration_crossings
                ),
                "familywise_crossing_overlap_count": len(
                    primary_hits & calibration_crossings
                ),
                "calibration_only_can_originate_primary_hit": False,
            }

        summary = {
            "schema_version": SCHEMA_VERSION,
            "analysis_id": contract.analysis_id,
            "classification": CLASSIFICATION,
            "status": STATUS,
            "claims": {
                "strict_reproduction_claim_permitted": False,
                "biological_claims_permitted": False,
                "independent_replication_claim_permitted": False,
                "permitted": contract.claims.permitted,
                "prohibited": contract.claims.prohibited,
                "same_phenotype_concordance_is_not_independent_replication": True,
            },
            "parent_failed_reconstruction": {
                "run_id": contract.parent_run_id,
                "source_git_commit": contract.parent_source_git_commit,
                "exit_code": contract.parent_failure_exit_code,
                "reason": contract.parent_failure_reason,
                "failure_remains_unmodified": True,
            },
            "inputs": {
                "contract": str(Path(contract_path).resolve()),
                "contract_sha256": _sha256(Path(contract_path).resolve()),
                "marker_list": str(marker_path),
                "marker_list_sha256": _sha256(marker_path),
                "analysis_marker_list": str(analysis_marker_path),
                "analysis_marker_list_sha256": _sha256(analysis_marker_path),
                "published_s3": str(published_path),
                "published_s3_sha256": _sha256(published_path),
                "results_root": str(root),
            },
            "marker_universe": {
                "current_coordinates": len(current_coordinates),
                "complete_analysis_coordinates": len(analysis_coordinates),
                "incomplete_coordinates": len(incomplete_coordinates),
                "published_coordinates": len(published_coordinates),
                "full_current_shared_coordinates": len(full_shared),
                "analysis_shared_coordinates": len(analysis_shared),
                "incomplete_coordinates_shared_with_published": len(
                    incomplete_coordinates & published_coordinates
                ),
                "published_only_coordinates": len(published_coordinates - current_coordinates),
                "current_only_coordinates": len(current_coordinates - published_coordinates),
                "analysis_published_only_coordinates": len(
                    published_coordinates - analysis_coordinates
                ),
                "analysis_current_only_coordinates": len(
                    analysis_coordinates - published_coordinates
                ),
                "full_published_coordinate_recovery_fraction": len(full_shared)
                / len(published_coordinates),
                "analysis_published_coordinate_recovery_fraction": len(analysis_shared)
                / len(published_coordinates),
                "scan_restricted_to_shared_coordinates": False,
                "coordinate_identity_caveat": contract.coordinate_identity_caveat,
            },
            "multiplicity": {
                "alpha": contract.alpha,
                "marker_denominator": contract.expected_current_markers,
                "trait_family_size": len(TRAITS),
                "within_trait_bonferroni_neg_log10_p": WITHIN_TRAIT_THRESHOLD,
                "four_trait_familywise_neg_log10_p": FOUR_TRAIT_FAMILYWISE_THRESHOLD,
                "deposited_threshold_reference_only": DEPOSITED_THRESHOLD_REFERENCE_ONLY,
                "deposited_threshold_used_for_calls": False,
                "P3D_true_in_discovery_family": False,
                "P3D_true_can_originate_primary_hit": False,
            },
            "modes": summary_modes,
            "p3d_cross_mode_concordance": cross_mode,
        }
        _write_json(stage / "ws276_current_marker_sensitivity_summary.json", summary)
        stage.replace(destination)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return summary


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--contract", type=Path, required=True)
    command.add_argument("--marker-list", type=Path, required=True)
    command.add_argument("--analysis-marker-list", type=Path, required=True)
    command.add_argument("--results-root", type=Path, required=True)
    command.add_argument("--published-s3", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = summarize_ws276_current_marker_sensitivity(
            args.contract,
            args.marker_list,
            args.analysis_marker_list,
            args.results_root,
            args.published_s3,
            args.output,
        )
    except (OSError, RuntimeError, ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
