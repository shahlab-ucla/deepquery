from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wormctx.poc import qtl_ws276 as qtl_ws276_module
from wormctx.poc.cli import main
from wormctx.poc.qtl_ws276 import (
    TRAITS,
    TRAIT_SLUGS,
    load_ws276_manifest,
    prepare_ws276_reconstruction,
    summarize_ws276_reconstruction,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "experiments"
    / "natural_variation"
    / "historical_marker_identity"
    / "config"
    / "historical_isotype_reconstruction.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_phenotypes(path: Path) -> tuple[list[str], list[str]]:
    identities = [f"S{index:03d}" for index in range(205)]
    special = ["ECA248", "ECA250", "ECA251", "JU1580", "JU1793"]
    finite = identities + special
    missing = [f"M{index:03d}" for index in range(27)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["", "strain", "condition", "trait", "phenotype"])
        row_id = 0
        for strain_index, strain in enumerate(finite + missing):
            for trait_index, trait in enumerate(TRAITS):
                value = "NA" if strain in missing else str(strain_index + trait_index / 10)
                writer.writerow([row_id, strain, "abamectin", trait, value])
                row_id += 1
    return finite, missing


def _write_s3(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["", "marker", "CHROM", "POS", "log10p", "trait"])
        row_id = 0
        for trait in TRAITS:
            for marker, position, value in [("I_100", 100, 1.0), ("V_200", 200, 5.0)]:
                row_id += 1
                writer.writerow(
                    [row_id, marker, marker.split("_")[0], position, value, f"abamectin_{trait}"]
                )


def _synthetic_contract(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    phenotype = tmp_path / "FileS2_wipheno.csv"
    _write_phenotypes(phenotype)
    s3 = tmp_path / "FileS3_wimap.csv"
    _write_s3(s3)
    samples = tmp_path / "vcf_samples.txt"
    sample_ids = [f"S{index:03d}" for index in range(205)]
    sample_ids += ["CB4855", "CB4857", "CB4858", "JU1793"]
    samples.write_text("\n".join(reversed(sample_ids)) + "\n", encoding="utf-8")

    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    assets = {item["role"]: item for item in config["assets"]}
    assets["phenotype"].update(
        {"bytes": phenotype.stat().st_size, "sha256": _sha256(phenotype)}
    )
    assets["published_s3"].update({"bytes": s3.stat().st_size, "sha256": _sha256(s3)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(config), encoding="utf-8")
    return manifest, phenotype, samples, s3


def test_real_reconstruction_manifest_is_locked_nonpublishable() -> None:
    manifest = load_ws276_manifest(BASE_CONFIG)
    assert manifest.strict_reproduction_claim_permitted is False
    assert manifest.biological_claims_permitted is False
    assert manifest.preprocessing.expected_complete_markers == 21_342
    assert manifest.association.threshold_log10p == pytest.approx(4.28448307163607)
    assert manifest.association.expected_published_nonzero_markers == 20_946
    assert manifest.cohort.aliases["JU1580"] == "JU1793"
    assert [item.model_dump() for item in manifest.published_intervals] == [
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
    assert len(manifest.provenance_gaps) >= 4


def test_prepare_maps_aliases_and_keeps_reference_collision_phenotype(tmp_path: Path) -> None:
    manifest, phenotype, samples, _ = _synthetic_contract(tmp_path)
    output = tmp_path / "prepared"

    receipt = prepare_ws276_reconstruction(manifest, phenotype, samples, output)

    assert receipt["finite_source_strains"] == 210
    assert receipt["analysis_isotypes"] == 209
    assert receipt["dropped_source_phenotypes"] == ["JU1580"]
    order = (output / "sample_order.txt").read_text(encoding="utf-8").splitlines()
    assert order == sorted(order)
    assert "CB4855" in order and "ECA248" not in order
    assert "JU1793" in order and "JU1580" not in order
    with (output / "mean_EXT.tsv").open(encoding="utf-8", newline="") as handle:
        values = {
            row["strain"]: float(row["mean.EXT"])
            for row in csv.DictReader(handle, delimiter="\t")
        }
    # The synthetic finite order puts JU1580 at 208 and JU1793 at 209.
    assert values["JU1793"] == 209.0
    mapping = (output / "source_to_analysis.tsv").read_text(encoding="utf-8")
    assert "JU1580\tJU1793\tfalse\treference_wins_collision" in mapping
    assert "JU1793\tJU1793\ttrue\treference_wins_collision" in mapping
    with pytest.raises(FileExistsError, match="must not already exist"):
        prepare_ws276_reconstruction(manifest, phenotype, samples, output)


def test_prepare_rejects_missing_reconstructed_isotype(tmp_path: Path) -> None:
    manifest, phenotype, samples, _ = _synthetic_contract(tmp_path)
    lines = samples.read_text(encoding="utf-8").splitlines()
    samples.write_text("\n".join(line for line in lines if line != "CB4855") + "\n")
    with pytest.raises(ValueError, match="absent from VCF:.*CB4855"):
        prepare_ws276_reconstruction(manifest, phenotype, samples, tmp_path / "prepared")


def test_manifest_cannot_enable_strict_reproduction_claim(tmp_path: Path) -> None:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["strict_reproduction_claim_permitted"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_ws276_manifest(path)


def test_manifest_cannot_change_a_published_interval(tmp_path: Path) -> None:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["published_intervals"][0]["start"] += 1
    path = tmp_path / "invalid-interval.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValidationError, match="interval tuples differ"):
        load_ws276_manifest(path)


def _write_results(root: Path) -> None:
    for mode in ("primary", "p3d_true"):
        directory = root / mode
        directory.mkdir(parents=True)
        for trait in TRAITS:
            with (directory / f"{TRAIT_SLUGS[trait]}_raw_mapping.tsv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(["marker", "CHROM", "POS", "log10p"])
                writer.writerow(["I_100", "I", 100, 1.0])
                writer.writerow(["V_200", "V", 200, 5.0 if mode == "primary" else 4.5])
        metadata = {
            "classification": "historical_isotype_imputed_reconstruction",
            "strict_reproduction": "false",
            "P3D": "FALSE" if mode == "primary" else "TRUE",
            "cores": "2",
            "rrBLUP_version": "4.6",
            "input_samples": "209",
            "input_markers": "21342",
            "complete_markers": "2",
            "MtDNA_preserved": "true",
            "kinship_action": "created" if mode == "primary" else "reused",
            "kinship_path": "/immutable/ws276_kinship.rds",
        }
        metadata.update({f"{TRAIT_SLUGS[trait]}_tested_markers": "2" for trait in TRAITS})
        with (directory / "run_metadata.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["key", "value"])
            writer.writerows(metadata.items())


def test_summarize_compares_primary_and_sensitivity_without_upgrading_claim(
    tmp_path: Path,
) -> None:
    manifest, _, _, s3 = _synthetic_contract(tmp_path)
    results = tmp_path / "association"
    _write_results(results)
    output = tmp_path / "summary"

    summary = summarize_ws276_reconstruction(manifest, results, s3, output)

    assert summary["strict_reproduction_claim_permitted"] is False
    assert summary["effective_tests_recomputed"] is False
    assert summary["classification"] == "partial_historical_isotype_reconstruction"
    assert not all(summary["functional_match_gates"].values())
    assert summary["functional_match_gates"][
        "marker_coordinates_identical_between_modes_all_traits"
    ]
    primary = summary["modes"]["primary_emma_p3d_false"]["traits"]
    sensitivity = summary["modes"]["sensitivity_emmax_p3d_true"]["traits"]
    assert all(item["marker_set_exact"] for item in primary)
    assert all(item["max_abs_log10p_difference"] == 0 for item in primary)
    assert all(item["max_abs_log10p_difference"] == 0.5 for item in sensitivity)
    assert (output / "ws276_reconstruction_summary.json").is_file()
    assert "Strict-reproduction and biological claims remain disabled" in (
        output / "ws276_reconstruction_summary.md"
    ).read_text(encoding="utf-8")


def test_summarize_rejects_swapped_p3d_metadata(tmp_path: Path) -> None:
    manifest, _, _, s3 = _synthetic_contract(tmp_path)
    results = tmp_path / "association"
    _write_results(results)
    primary_metadata = results / "primary" / "run_metadata.tsv"
    primary_metadata.write_text(
        primary_metadata.read_text(encoding="utf-8").replace("P3D\tFALSE", "P3D\tTRUE"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mode metadata contract failed"):
        summarize_ws276_reconstruction(manifest, results, s3, tmp_path / "summary")


def test_summarize_fails_gate_for_sensitivity_marker_drift(tmp_path: Path) -> None:
    manifest, _, _, s3 = _synthetic_contract(tmp_path)
    results = tmp_path / "association"
    _write_results(results)
    sensitivity = results / "p3d_true" / "mean_EXT_raw_mapping.tsv"
    sensitivity.write_text(
        sensitivity.read_text(encoding="utf-8").replace("I_100\tI\t100", "I_101\tI\t101"),
        encoding="utf-8",
    )

    summary = summarize_ws276_reconstruction(manifest, results, s3, tmp_path / "summary")

    assert not summary["functional_match_gates"][
        "marker_coordinates_identical_between_modes_all_traits"
    ]


@pytest.mark.parametrize(
    ("gates", "expected_exit"),
    [({"marker_gate": True, "interval_gate": True}, 0), ({"marker_gate": False}, 3)],
)
def test_ws276_summary_cli_exit_tracks_acceptance_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gates: dict[str, bool],
    expected_exit: int,
) -> None:
    monkeypatch.setattr(
        qtl_ws276_module,
        "summarize_ws276_reconstruction",
        lambda *_args, **_kwargs: {"functional_match_gates": gates},
    )
    exit_code = main(
        [
            "qtl-ws276-summarize",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--results-root",
            str(tmp_path / "association"),
            "--published-s3",
            str(tmp_path / "s3.csv"),
            "--output",
            str(tmp_path / "summary"),
        ]
    )
    assert exit_code == expected_exit
