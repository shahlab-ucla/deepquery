from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wormctx.poc import qtl_ws276_sensitivity as sensitivity
from wormctx.poc.qtl_ws276_sensitivity import (
    ANALYSIS_ID,
    CLASSIFICATION,
    FOUR_TRAIT_FAMILYWISE_THRESHOLD,
    PARENT_FAILURE_REASON,
    PARENT_RUN_ID,
    PARENT_SOURCE_GIT_COMMIT,
    TRAITS,
    TRAIT_SLUGS,
    WITHIN_TRAIT_THRESHOLD,
    _read_mapping,
    _read_analysis_marker_list,
    _read_marker_list,
    _validate_cross_mode_metadata,
    _validate_metadata,
    load_ws276_sensitivity_contract,
    summarize_ws276_current_marker_sensitivity,
)


BASE_CONFIG = (
    Path(__file__).parents[1]
    / "experiments"
    / "natural_variation"
    / "historical_marker_identity"
    / "config"
    / "observed_marker_sensitivity.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_sensitivity_contract_and_thresholds_load() -> None:
    contract = load_ws276_sensitivity_contract(BASE_CONFIG)

    assert contract.classification == CLASSIFICATION
    assert contract.analysis_id == ANALYSIS_ID
    assert contract.parent_run_id == PARENT_RUN_ID
    assert contract.parent_source_git_commit == PARENT_SOURCE_GIT_COMMIT
    assert contract.parent_failure_reason == PARENT_FAILURE_REASON
    assert contract.expected_current_markers == 75_852
    assert contract.expected_complete_markers == 75_831
    assert contract.expected_incomplete_markers == 21
    assert contract.analysis_marker_list_sha256 == (
        "181d3dfe2db799b544dc1b17098095f591cbee3f77ef0c4a604211227a1cebe2"
    )
    assert contract.expected_analysis_shared_coordinates == 12_601
    assert contract.expected_analysis_current_only == 63_230
    assert contract.expected_incomplete_published_overlap == 0
    assert contract.expected_shared_coordinates == 12_601
    assert contract.strict_reproduction_claim_permitted is False
    assert contract.biological_claims_permitted is False
    assert contract.association.primary_p3d is False
    assert contract.association.calibration_p3d is True
    assert contract.association.p3d_true_can_originate_primary_hit is False
    assert contract.association.within_trait_bonferroni_neg_log10_p == pytest.approx(
        WITHIN_TRAIT_THRESHOLD, abs=1e-14
    )
    assert contract.association.four_trait_familywise_neg_log10_p == pytest.approx(
        FOUR_TRAIT_FAMILYWISE_THRESHOLD, abs=1e-14
    )


def test_contract_cannot_enable_reproduction_or_change_threshold(tmp_path: Path) -> None:
    value = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    value["strict_reproduction_claim_permitted"] = True
    value["association"]["four_trait_familywise_neg_log10_p"] -= 1
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_ws276_sensitivity_contract(path)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("analysis_id",), "renamed_analysis"),
        (("parent_run_id",), "wrong-parent-run"),
        (("parent_source_git_commit",), "0" * 40),
        (("parent_failure_reason",), "wrong failure"),
        (
            ("parent_artifacts", "filtered_vcf", "sha256"),
            "0" * 64,
        ),
        (("parent_artifacts", "source_to_analysis", "bytes"), 5_918),
        (("parent_artifacts", "genotype_matrix", "sha256"), "e" * 64),
        (("parent_artifacts", "failure_marker", "bytes"), 52),
        (
            ("parent_artifacts", "phenotypes", "norm_n.tsv"),
            "f" * 64,
        ),
    ],
)
def test_contract_freezes_analysis_and_parent_artifact_identity(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    document = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    invalid = tmp_path / "invalid-identity.json"
    invalid.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_ws276_sensitivity_contract(invalid)


def _small_contract(tmp_path: Path):
    contract = load_ws276_sensitivity_contract(BASE_CONFIG)
    marker_list = tmp_path / "markers.txt"
    marker_list.write_text(
        "MtDNA\t100\nV\t2693128\nV\t3000000\nII\t14121786\n",
        encoding="utf-8",
        newline="\n",
    )
    analysis_marker_list = tmp_path / "analysis_markers.txt"
    analysis_marker_list.write_text(
        "V\t2693128\nV\t3000000\nII\t14121786\n",
        encoding="utf-8",
        newline="\n",
    )
    contract.marker_list_sha256 = _sha256(marker_list)
    contract.analysis_marker_list_sha256 = _sha256(analysis_marker_list)
    contract.expected_current_markers = 4
    contract.expected_complete_markers = 3
    contract.expected_incomplete_markers = 1
    contract.expected_published_markers = 4
    contract.expected_shared_coordinates = 3
    contract.expected_published_only = 1
    contract.expected_current_only = 1
    contract.expected_analysis_shared_coordinates = 2
    contract.expected_analysis_published_only = 2
    contract.expected_analysis_current_only = 1
    contract.expected_incomplete_published_overlap = 1
    contract.parent_artifacts.genotype_matrix.sha256 = "a" * 64
    contract.marker_universe.chromosome_counts = {"MtDNA": 1, "V": 2, "II": 1}
    return contract, marker_list, analysis_marker_list


def _write_published_s3(path: Path) -> None:
    rows = [
        ("MtDNA", 100, 1.0),
        ("V", 2_693_128, 6.0),
        ("II", 14_121_786, 5.0),
        ("III", 3_526_374, 4.0),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["marker", "CHROM", "POS", "log10p", "trait"])
        for trait in TRAITS:
            for chromosome, position, score in rows:
                writer.writerow(
                    [
                        f"{chromosome}_{position}",
                        chromosome,
                        position,
                        score,
                        f"abamectin_{trait}",
                    ]
                )


def _write_results(root: Path) -> None:
    rows = [
        ("V", 2_693_128),
        ("V", 3_000_000),
        ("II", 14_121_786),
    ]
    for mode in ("primary", "p3d_true"):
        directory = root / mode
        directory.mkdir(parents=True)
        for trait_index, trait in enumerate(TRAITS):
            with (directory / f"{TRAIT_SLUGS[trait]}_raw_mapping.tsv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["marker", "CHROM", "POS", "log10p"])
                for marker_index, (chromosome, position) in enumerate(rows):
                    if mode == "primary":
                        score = [7.0, 2.0, 4.0][marker_index] + trait_index * 0.01
                    else:
                        # This sensitivity-only crossing must never become a primary hit.
                        score = [6.5, 8.0, 4.5][marker_index] + trait_index * 0.01
                    writer.writerow(
                        [f"{chromosome}_{position}", chromosome, position, score]
                    )
        metadata = {
            "classification": CLASSIFICATION,
            "strict_reproduction": "false",
            "biological_claims_permitted": "false",
            "P3D": "FALSE" if mode == "primary" else "TRUE",
            "cores": "2",
            "rrBLUP_version": "4.6",
            "input_samples": "209",
            "input_markers": "4",
            "complete_markers": "3",
            "MtDNA_preserved": "true",
            "kinship_action": "created" if mode == "primary" else "reused",
            "kinship_path": "/immutable/current_ws276_kinship.rds",
            "genotype_matrix_sha256": "a" * 64,
        }
        metadata.update({f"{TRAIT_SLUGS[trait]}_tested_markers": "3" for trait in TRAITS})
        with (directory / "run_metadata.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["key", "value"])
            writer.writerows(metadata.items())


def test_summarizer_emits_exploratory_concordance_without_claim_upgrade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract, marker_list, analysis_marker_list = _small_contract(tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    published = tmp_path / "FileS3.csv"
    _write_published_s3(published)
    results = tmp_path / "association"
    _write_results(results)
    output = tmp_path / "summary"

    monkeypatch.setattr(
        sensitivity, "load_ws276_sensitivity_contract", lambda _path: contract
    )
    monkeypatch.setattr(sensitivity, "PUBLISHED_S3_SHA256", _sha256(published))
    monkeypatch.setattr(sensitivity, "TOP_SHARED_COUNT", 2)

    summary = summarize_ws276_current_marker_sensitivity(
        contract_path,
        marker_list,
        analysis_marker_list,
        results,
        published,
        output,
    )

    assert summary["classification"] == CLASSIFICATION
    assert summary["claims"]["strict_reproduction_claim_permitted"] is False
    assert summary["claims"]["biological_claims_permitted"] is False
    assert summary["claims"]["independent_replication_claim_permitted"] is False
    assert summary["claims"][
        "same_phenotype_concordance_is_not_independent_replication"
    ]
    assert summary["marker_universe"] == {
        "current_coordinates": 4,
        "complete_analysis_coordinates": 3,
        "incomplete_coordinates": 1,
        "published_coordinates": 4,
        "full_current_shared_coordinates": 3,
        "analysis_shared_coordinates": 2,
        "incomplete_coordinates_shared_with_published": 1,
        "published_only_coordinates": 1,
        "current_only_coordinates": 1,
        "analysis_published_only_coordinates": 2,
        "analysis_current_only_coordinates": 1,
        "full_published_coordinate_recovery_fraction": 0.75,
        "analysis_published_coordinate_recovery_fraction": 0.5,
        "scan_restricted_to_shared_coordinates": False,
        "coordinate_identity_caveat": contract.coordinate_identity_caveat,
    }
    assert summary["multiplicity"]["deposited_threshold_used_for_calls"] is False
    assert summary["multiplicity"]["P3D_true_can_originate_primary_hit"] is False

    primary = summary["modes"]["primary_emma_p3d_false"]
    calibration = summary["modes"]["calibration_emmax_p3d_true"]
    assert primary["can_originate_familywise_screen_hit"] is True
    assert calibration["can_originate_familywise_screen_hit"] is False
    assert primary["traits"][0]["familywise_screen_hit_count"] == 1
    assert calibration["traits"][0]["four_trait_familywise_threshold_crossing_count"] == 1
    assert calibration["traits"][0]["familywise_screen_hit_count"] == 0

    concordance = primary["traits"][0]["concordance_to_published_same_phenotype"]
    assert concordance["shared_coordinate_count"] == 2
    assert concordance["top_100"]["set_size"] == 2
    assert concordance["top_1_percent"]["set_size"] == 1
    interval = primary["traits"][0]["published_interval_and_peak_metrics"][0]
    assert interval["published_peak_exact_coordinate_present_current"] is True
    assert interval["published_peak_exact_coordinate_current_log10p"] == 7.0
    assert interval["current_interval_familywise_screen_hit"] is True

    cross_mode = summary["p3d_cross_mode_concordance"]["mean.EXT"]
    assert cross_mode["calibration_numeric_familywise_crossing_count"] == 1
    assert cross_mode["calibration_only_can_originate_primary_hit"] is False
    assert (output / "ws276_current_marker_sensitivity_summary.json").is_file()
    top20 = output / primary["traits"][0]["top20_table"]
    assert top20.is_file()
    assert "familywise_screen_hit" in top20.read_text(encoding="utf-8").splitlines()[0]


def test_marker_and_mapping_contracts_fail_closed(tmp_path: Path) -> None:
    contract, marker_list, analysis_marker_list = _small_contract(tmp_path)
    expected = _read_marker_list(marker_list, contract)
    analysis, excluded = _read_analysis_marker_list(
        analysis_marker_list, expected, contract
    )
    assert len(analysis) == 3
    assert excluded == {("MtDNA", 100)}
    analysis_marker_list.write_text(
        analysis_marker_list.read_text(encoding="utf-8").replace("3000000", "3000001"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="analysis-marker list checksum mismatch"):
        _read_analysis_marker_list(analysis_marker_list, expected, contract)
    marker_list.write_text(marker_list.read_text(encoding="utf-8") + "X\t500\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _read_marker_list(marker_list, contract)

    mapping = tmp_path / "mapping.tsv"
    with mapping.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["marker", "CHROM", "POS", "log10p"])
        for chromosome, position in sorted(expected):
            writer.writerow([f"{chromosome}_{position}", chromosome, position, 1.0])
    mapping.write_text(
        mapping.read_text(encoding="utf-8").replace(
            "MtDNA_100\tMtDNA\t100", "MtDNA_101\tMtDNA\t101"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coordinate universe differs"):
        _read_mapping(mapping, expected)


def test_metadata_rejects_swapped_p3d_role(tmp_path: Path) -> None:
    contract, _, _ = _small_contract(tmp_path)
    metadata = {
        "classification": CLASSIFICATION,
        "strict_reproduction": "false",
        "biological_claims_permitted": "false",
        "P3D": "TRUE",
        "cores": "2",
        "rrBLUP_version": "4.6",
        "input_samples": "209",
        "input_markers": "4",
        "complete_markers": "3",
        "MtDNA_preserved": "true",
        "kinship_action": "created",
        "kinship_path": "/immutable/K.rds",
        "genotype_matrix_sha256": "a" * 64,
    }
    metadata.update({f"{TRAIT_SLUGS[trait]}_tested_markers": "3" for trait in TRAITS})

    with pytest.raises(ValueError, match="metadata contract failed"):
        _validate_metadata(
            metadata,
            contract,
            expected_p3d=False,
            expected_kinship_action="created",
            path=tmp_path / "run_metadata.tsv",
        )


def test_metadata_requires_claim_boundary_and_valid_genotype_hash(tmp_path: Path) -> None:
    contract, _, _ = _small_contract(tmp_path)
    metadata = {
        "classification": CLASSIFICATION,
        "strict_reproduction": "false",
        "biological_claims_permitted": "false",
        "P3D": "FALSE",
        "cores": "2",
        "rrBLUP_version": "4.6",
        "input_samples": "209",
        "input_markers": "4",
        "complete_markers": "3",
        "MtDNA_preserved": "true",
        "kinship_action": "created",
        "kinship_path": "/immutable/K.rds",
        "genotype_matrix_sha256": "a" * 64,
    }
    metadata.update({f"{TRAIT_SLUGS[trait]}_tested_markers": "3" for trait in TRAITS})

    missing_claim_boundary = dict(metadata)
    del missing_claim_boundary["biological_claims_permitted"]
    with pytest.raises(ValueError, match="metadata contract failed"):
        _validate_metadata(
            missing_claim_boundary,
            contract,
            expected_p3d=False,
            expected_kinship_action="created",
            path=tmp_path / "missing-claim.tsv",
        )

    invalid_hash = dict(metadata)
    invalid_hash["genotype_matrix_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="SHA-256 is missing or invalid"):
        _validate_metadata(
            invalid_hash,
            contract,
            expected_p3d=False,
            expected_kinship_action="created",
            path=tmp_path / "invalid-hash.tsv",
        )


def test_p3d_modes_must_share_genotype_hash_and_kinship_path() -> None:
    primary = {
        "kinship_path": "/immutable/K.rds",
        "genotype_matrix_sha256": "a" * 64,
    }
    calibration = dict(primary)
    _validate_cross_mode_metadata(primary, calibration)

    calibration["genotype_matrix_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="same genotype-matrix SHA-256"):
        _validate_cross_mode_metadata(primary, calibration)
