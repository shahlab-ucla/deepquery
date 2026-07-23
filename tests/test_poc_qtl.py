from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wormctx.poc.qtl import (
    TRAITS,
    TRAIT_SLUGS,
    load_qtl_manifest,
    prepare_abamectin_qtl,
    summarize_abamectin_qtl,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "experiments"
    / "natural_variation"
    / "modern_snp_association_reanalysis"
    / "config"
    / "association_contract.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_phenotypes(path: Path, *, unexpected_trait: bool = False) -> list[str]:
    finite = [f"S{index:03d}" for index in range(209)] + ["JU1580"]
    missing = ["ECA252"] + [f"M{index:03d}" for index in range(26)]
    strains = finite + missing
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["", "strain", "condition", "trait", "phenotype"])
        row_index = 0
        for strain_index, strain in enumerate(strains):
            for trait_index, trait in enumerate(TRAITS):
                if unexpected_trait and row_index == 0:
                    trait = "unplanned.trait"
                value = "NA" if strain in missing else f"{strain_index + trait_index / 10:.1f}"
                writer.writerow([row_index, strain, "abamectin", trait, value])
                row_index += 1
    return finite


def _write_fam(path: Path, finite: list[str]) -> None:
    # Reverse order proves that preparation follows genotype order, not CSV order.
    path.write_text(
        "".join(f"0 {strain} 0 0 0 -9\n" for strain in reversed(finite[:-1])),
        encoding="utf-8",
    )


def _synthetic_manifest(
    tmp_path: Path, *, unexpected_trait: bool = False, bad_phenotype_hash: bool = False
) -> tuple[Path, Path]:
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    phenotype = data_root / "phenotypes.csv"
    fam = data_root / "source.fam"
    finite = _write_phenotypes(phenotype, unexpected_trait=unexpected_trait)
    _write_fam(fam, finite)
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    by_role = {item["role"]: item for item in config["inputs"]}
    by_role["phenotype"].update(
        {
            "relative_path": phenotype.name,
            "bytes": phenotype.stat().st_size,
            "sha256": "0" * 64 if bad_phenotype_hash else _sha256(phenotype),
        }
    )
    by_role["genotype_fam"].update(
        {
            "relative_path": fam.name,
            "bytes": fam.stat().st_size,
            "sha256": _sha256(fam),
        }
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(config), encoding="utf-8")
    return manifest, data_root


def test_prepare_frozen_common_cohort_and_receipts(tmp_path: Path) -> None:
    manifest, data_root = _synthetic_manifest(tmp_path)
    output = tmp_path / "prepared"

    result = prepare_abamectin_qtl(manifest, data_root, output)

    assert result["biological_claims_permitted"] is False
    assert result["cohort"]["samples"] == 209
    assert result["phenotype_audit"]["finite_absent_from_genotype"] == ["JU1580"]
    assert len(result["phenotype_audit"]["common_missing_strains"]) == 27
    keep = (output / "cohort.keep").read_text(encoding="utf-8").splitlines()
    assert len(keep) == 209
    assert keep[0] == "0\tS208"
    assert keep[-1] == "0\tS000"
    for trait in TRAITS:
        path = output / f"{TRAIT_SLUGS[trait]}.phen"
        assert len(path.read_text(encoding="utf-8").splitlines()) == 209
        assert result["outputs"][path.name]["sha256"] == _sha256(path)
    with pytest.raises(FileExistsError, match="must not already exist"):
        prepare_abamectin_qtl(manifest, data_root, output)


def test_prepare_stops_on_checksum_mismatch_before_output(tmp_path: Path) -> None:
    manifest, data_root = _synthetic_manifest(tmp_path, bad_phenotype_hash=True)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="checksum mismatch for phenotype"):
        prepare_abamectin_qtl(manifest, data_root, output)
    assert not output.exists()


def test_prepare_rejects_unexpected_trait(tmp_path: Path) -> None:
    manifest, data_root = _synthetic_manifest(tmp_path, unexpected_trait=True)
    with pytest.raises(ValueError, match="unexpected trait"):
        prepare_abamectin_qtl(manifest, data_root, tmp_path / "prepared")


def test_prepare_rejects_changed_finite_cohort(tmp_path: Path) -> None:
    manifest, data_root = _synthetic_manifest(tmp_path)
    fam = data_root / "source.fam"
    lines = fam.read_text(encoding="utf-8").splitlines()
    fam.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    config = json.loads(manifest.read_text(encoding="utf-8"))
    item = next(value for value in config["inputs"] if value["role"] == "genotype_fam")
    item["bytes"] = fam.stat().st_size
    item["sha256"] = _sha256(fam)
    manifest.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="finite phenotype strains absent from genotype"):
        prepare_abamectin_qtl(manifest, data_root, tmp_path / "prepared")


def _write_mlma(path: Path, trait: str, *, malformed: bool = False) -> None:
    header = ["Chr", "SNP", "bp", "A1", "A2", "Freq", "b", "se", "p"]
    if malformed:
        header[-1] = "P_value"
    rows = [
        [5, "mVL", 2693128, "A", "G", 0.2, 0.5, 0.1, 1e-8 if trait != "norm.n" else 0.2],
        [2, "mNII", 14121786, "C", "T", 0.3, -0.2, 0.1, 1e-8 if trait == "norm.n" else 0.2],
        [1, "mNull", 100, "A", "C", 0.4, 0.1, 0.2, 1e-8 if trait == "mean.norm.EXT" else 0.9],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_four_results(root: Path, *, malformed_trait: str | None = None) -> None:
    root.mkdir()
    for trait in TRAITS:
        _write_mlma(
            root / f"{TRAIT_SLUGS[trait]}.loco.mlma",
            trait,
            malformed=trait == malformed_trait,
        )


def test_summarize_thresholds_interval_recovery_and_null_trait(tmp_path: Path) -> None:
    results = tmp_path / "association"
    _write_four_results(results)
    output = tmp_path / "summary"

    summary = summarize_abamectin_qtl(BASE_CONFIG, results, output)

    by_trait = {item["trait"]: item for item in summary["traits"]}
    assert all(item["marker_count"] == 3 for item in by_trait.values())
    assert by_trait["mean.EXT"]["published_interval_recovery"][0][
        "recovered_at_published_effective_threshold"
    ] is True
    assert by_trait["mean.norm.EXT"]["no_expected_published_interval"] is True
    assert by_trait["mean.norm.EXT"]["published_interval_recovery"] == []
    assert summary["published_interval_contract"] == {
        "source": "Evans_et_al_2021_File_S3",
        "expected_intervals": 6,
        "observed_intervals": 6,
        "null_trait": "mean.norm.EXT",
        "vc_interval_included": False,
    }
    assert (output / "qtl_summary.json").is_file()
    assert "VC linkage/NIL interval is intentionally excluded" in (
        output / "qtl_summary.md"
    ).read_text(encoding="utf-8")


def test_summarize_rejects_malformed_mlma_without_output(tmp_path: Path) -> None:
    results = tmp_path / "association"
    _write_four_results(results, malformed_trait="mean.TOF")
    output = tmp_path / "summary"
    with pytest.raises(ValueError, match="MLMA header mismatch"):
        summarize_abamectin_qtl(BASE_CONFIG, results, output)
    assert not output.exists()


def test_summarize_accepts_whitespace_delimited_gcta_output(tmp_path: Path) -> None:
    results = tmp_path / "association"
    _write_four_results(results)
    for path in results.glob("*.loco.mlma"):
        path.write_text(
            path.read_text(encoding="utf-8").replace("\t", "   "),
            encoding="utf-8",
        )
    summary = summarize_abamectin_qtl(BASE_CONFIG, results, tmp_path / "summary")
    assert [item["marker_count"] for item in summary["traits"]] == [3, 3, 3, 3]


def test_manifest_rejects_vc_interval(tmp_path: Path) -> None:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["published_intervals"][0]["id"] = "VC_linkage"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValidationError, match="VC is linkage/NIL evidence"):
        load_qtl_manifest(path)


@pytest.mark.skip(
    reason="the site-specific runner is intentionally absent from the public tree"
)
def test_remote_runner_enforces_immutable_receipted_contract() -> None:
    script = (ROOT / "scripts" / "run_abamectin_qtl_remote.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ "$RUN_ROOT" != /* ]]' in script
    assert 'if ! mkdir -- "$RUN_ROOT"' in script
    assert 'mkdir -p "$RUN_ROOT"' not in script
    assert 'cp "$REPO_ROOT/SOURCE_REVISION"' in script
    assert 'git -C "$REPO_ROOT" status --porcelain=v1' in script
    assert '--autosome-num 6' in script
    assert 'MLMA_ROWS' in script
    assert '"$MLMA_ROWS" -ne "$POST_QC_MARKERS"' in script
    assert "printf 'SUCCESS\\n'" in script
