from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wormctx.poc import qtl_matched_relationship_model_diagnostic as matched_relationship_model
from wormctx.poc import qtl_parametric_null as parametric_null


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/natural_variation/matched_relationship_model_diagnostic/config/matched_relationship_model.json"
RESULT = (
    ROOT
    / "experiments"
    / "natural_variation"
    / "matched_relationship_model_diagnostic"
    / "results"
    / "summary.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_freezes_four_cells_400_maps_and_byte_reuse() -> None:
    manifest = matched_relationship_model.load_manifest(CONFIG)
    assert tuple(cell.id for cell in manifest.cells) == matched_relationship_model.EXPECTED_CELL_IDS
    assert manifest.execution.new_map_count == 400
    assert manifest.execution.phenotypes_are_reused_byte_for_byte is True
    assert manifest.execution.phenotype_regeneration_permitted is False
    assert manifest.execution.pooled_cells_permitted is False
    assert manifest.classification.endswith("not_threshold_calibration")

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["cells"].reverse()
    with pytest.raises(ValidationError, match="cells or their order"):
        matched_relationship_model.Manifest.model_validate(payload)

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["execution"]["phenotype_regeneration_permitted"] = True
    with pytest.raises(ValidationError):
        matched_relationship_model.Manifest.model_validate(payload)


def test_checksum_parser_rejects_unsafe_duplicate_and_malformed_paths(tmp_path: Path) -> None:
    receipt = tmp_path / "SHA256SUMS.txt"
    digest = "a" * 64
    receipt.write_text(
        f"{digest}  ./safe/a.txt\n{digest}  ./safe/b.txt\n",
        encoding="utf-8",
    )
    assert matched_relationship_model.parse_checksum_closure(receipt) == {
        "safe/a.txt": digest,
        "safe/b.txt": digest,
    }

    receipt.write_text(f"{digest}  ./safe/a.txt\n{digest}  ./safe/a.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        matched_relationship_model.parse_checksum_closure(receipt)
    receipt.write_text(f"{digest}  ./../escape.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        matched_relationship_model.parse_checksum_closure(receipt)


def _diagnostics() -> dict[str, object]:
    return {
        "minimum_p": {"p": 0.01, "chromosome": 1, "bp": 1, "snp": "m1"},
        "per_chromosome_minimum_p": [
            {"p": 0.01, "chromosome": chromosome, "bp": chromosome, "snp": f"m{chromosome}"}
            for chromosome in range(1, 7)
        ],
        "bonferroni_marker_count": 0,
        "lambda_gc": 1.0,
        "lambda_gc_excluding_trait_intervals": 1.0,
        "lambda_gc_excluding_all_intervals": 1.0,
        "qq_quantiles": [
            {"quantile": q, "expected_neg_log10_p": 0.0, "observed_neg_log10_p": 0.0}
            for q in (0.5, 0.9, 0.95, 0.99, 0.999)
        ],
    }


def test_selected_tuple_requires_same_phenotype_bytes_in_simulation_and_loco(
    tmp_path: Path,
) -> None:
    manifest = matched_relationship_model.load_manifest(CONFIG)
    cell = manifest.cells[0]
    phenotype_rel, simulation_rel, checkpoint_rel = matched_relationship_model._selected_paths(cell, 1)
    for relative in (phenotype_rel, simulation_rel, checkpoint_rel):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
    phenotype = tmp_path / phenotype_rel
    phenotype.write_text("0\tS1\t1.25\n", encoding="utf-8", newline="\n")
    expected_cell = {
        "id": cell.id,
        "relationship_kind": cell.relationship_kind,
        "endpoint": cell.endpoint,
        "trait": cell.trait,
        "trait_slug": cell.trait_slug,
    }
    simulation = {
        "schema_version": parametric_null.PHENOTYPE_VERSION,
        "cell": expected_cell,
        "replicate": 1,
        "byte_replay_verified": True,
        "byte_replay_sha256": _sha(phenotype),
        "phenotype": {"bytes": phenotype.stat().st_size, "sha256": _sha(phenotype)},
    }
    checkpoint = {
        "schema_version": parametric_null.MAP_VERSION,
        "cell": expected_cell,
        "replicate": 1,
        "qualified": True,
        "observed": False,
        "phenotype": {"sha256": _sha(phenotype)},
        "diagnostics": _diagnostics(),
    }
    (tmp_path / simulation_rel).write_text(json.dumps(simulation), encoding="utf-8")
    (tmp_path / checkpoint_rel).write_text(json.dumps(checkpoint), encoding="utf-8")
    closure = {
        phenotype_rel: _sha(phenotype),
        simulation_rel: _sha(tmp_path / simulation_rel),
        checkpoint_rel: _sha(tmp_path / checkpoint_rel),
    }
    selected = matched_relationship_model._verify_selected_tuple(manifest, tmp_path, closure, cell, 1)
    assert selected["phenotype_sha256"] == _sha(phenotype)

    checkpoint["phenotype"]["sha256"] = "0" * 64
    (tmp_path / checkpoint_rel).write_text(json.dumps(checkpoint), encoding="utf-8")
    closure[checkpoint_rel] = _sha(tmp_path / checkpoint_rel)
    with pytest.raises(ValueError, match="same phenotype"):
        matched_relationship_model._verify_selected_tuple(manifest, tmp_path, closure, cell, 1)


def test_log_gate_accepts_only_single_whole_grm_mlma(tmp_path: Path) -> None:
    manifest = matched_relationship_model.load_manifest(CONFIG)
    cell = manifest.cells[0]
    source = tmp_path / "parametric_null"
    bfile = tmp_path / "parent/genotype/abamectin_209_qc"
    phenotype = source / "null/phenotypes/full_pc0_mean_EXT/rep-001.phen"
    log = tmp_path / "map.log"
    options = [
        ("--mlma", None),
        ("--bfile", str(bfile)),
        ("--grm", str(source / "grm/global/full")),
        ("--pheno", str(phenotype)),
        ("--maf", "0.05"),
        ("--autosome-num", "6"),
        ("--thread-num", "16"),
        ("--out", str(tmp_path / "map")),
    ]
    accepted = [name if value is None else f"{name} {value}" for name, value in options]
    log.write_text(
        "\n".join(
            [
                "Accepted options:",
                *accepted,
                "",
                "209 individuals are in common in these files.",
                "Log-likelihood ratio converged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    receipt = matched_relationship_model._qualify_mlma_log(
        log, manifest, source, cell, phenotype, bfile, None
    )
    assert receipt["accepted_options"]["--mlma"] is None

    log.write_text(log.read_text(encoding="utf-8").replace("--mlma\n", "--mlma-loco\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="options|mlma-loco"):
        matched_relationship_model._qualify_mlma_log(log, manifest, source, cell, phenotype, bfile, None)


@pytest.mark.skip(
    reason="legacy site-specific deployment assertions are superseded by the portable launcher"
)
def test_runner_and_deployer_are_fail_closed_and_do_not_regenerate_phenotypes() -> None:
    runner = (ROOT / "scripts/run_abamectin_ws283_matched_relationship_model_matched_model.sh").read_text(
        encoding="utf-8"
    )
    deployer = (
        ROOT / "scripts/deploy_abamectin_ws283_matched_relationship_model_matched_model.ps1"
    ).read_text(encoding="utf-8")
    assert runner.count("--mlma \\") == 1
    assert "--mlma-loco" not in runner
    assert "${MODULE[@]}\" verify-source" in runner
    assert '--output "$SOURCE_VERIFY_TMP"' not in runner
    assert '> "$SOURCE_VERIFY_TMP"' in runner
    assert "null/phenotypes/$cell_id/$tag.phen" in runner
    assert "simulate" not in runner
    assert "test ! -e '$Deployment'" in deployer
    assert "test ! -e '$RunRoot'" in deployer
    assert "comm=" in deployer
    assert "gcta-1[.]94[.]1|plink2" in deployer
    assert '"src/wormctx/poc/qtl_calibration.py"' in deployer


def test_paired_distribution_retains_direction_and_cell_specific_n() -> None:
    comparison = matched_relationship_model._metric_comparison(
        [{"x": 0.5}, {"x": 1.0}, {"x": 2.0}],
        [{"x": 1.0}, {"x": 1.0}, {"x": 1.0}],
        "x",
    )
    assert comparison["matched_lower_than_loco"] == 1
    assert comparison["ties"] == 1
    assert comparison["matched_higher_than_loco"] == 1
    assert comparison["paired_difference_matched_minus_loco"]["replicates"] == 3


def test_executed_result_freezes_terminal_closure_and_conservative_outcome() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["terminal_closure"]["completed_maps"] == 400
    assert result["terminal_closure"]["source_phenotypes_reused"] == 400
    assert result["terminal_closure"]["phenotypes_regenerated"] == 0
    assert [row["cell"] for row in result["nominal_bonferroni_fwer"]] == list(
        matched_relationship_model.EXPECTED_CELL_IDS
    )
    assert all(
        row["matched_whole_relationship_events"] == 0
        for row in result["nominal_bonferroni_fwer"]
    )
    assert result["interpretation"]["generator_fitter_mismatch_is_material"] is True
    assert (
        result["interpretation"]["matched_whole_relationship_is_nominally_calibrated"]
        is False
    )
    assert result["interpretation"]["whole_relationship_selected_as_discovery_model"] is False
