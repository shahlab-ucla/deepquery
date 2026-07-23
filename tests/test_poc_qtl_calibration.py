from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wormctx.poc import qtl_calibration as calibration
from wormctx.poc.qtl import TRAITS, TRAIT_SLUGS, _MlmaRow


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/natural_variation/population_structure_sensitivity/config/principal_component_sensitivity.json"
PARENT_CONFIG = ROOT / "experiments/natural_variation/modern_snp_association_reanalysis/config/association_contract.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibration_manifest_freezes_models_and_claim_boundary(tmp_path: Path) -> None:
    manifest = calibration.load_calibration_manifest(CONFIG)
    assert [item.id for item in manifest.association.models] == ["pc0", "pc3", "pc5", "pc10"]
    assert manifest.association.model_selection_from_results_permitted is False
    assert manifest.biological_claims_permitted is False

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["pca"]["association_pc_counts"] = [0, 2, 5, 10]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="association_pc_counts"):
        calibration.load_calibration_manifest(changed)


def _write_fam(path: Path) -> None:
    path.write_text(
        "".join(f"0 S{index:03d} 0 0 0 -9\n" for index in range(209)),
        encoding="utf-8",
    )


def _write_eigenvec(path: Path) -> None:
    header = ["#FID", "IID", *(f"PC{index}" for index in range(1, 11))]
    lines = ["\t".join(header)]
    for sample in range(209):
        scores = [f"{(sample + 1) * pc / 100000:.8f}" for pc in range(1, 11)]
        lines.append("\t".join(["0", f"S{sample:03d}", *scores]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_qcovars_preserves_fam_order_and_exact_pc_counts(tmp_path: Path) -> None:
    fam = tmp_path / "source.fam"
    eigenvec = tmp_path / "source.eigenvec"
    _write_fam(fam)
    _write_eigenvec(eigenvec)

    result = calibration.build_qcovars(CONFIG, eigenvec, fam, tmp_path / "qcovars")

    assert set(result["outputs"]) == {"pc3.qcovar", "pc5.qcovar", "pc10.qcovar"}
    for count in (3, 5, 10):
        path = tmp_path / "qcovars" / f"pc{count}.qcovar"
        rows = path.read_text(encoding="utf-8").splitlines()
        assert len(rows) == 209
        assert len(rows[0].split("\t")) == count + 2
        assert rows[0].split("\t")[:2] == ["0", "S000"]
        assert rows[-1].split("\t")[:2] == ["0", "S208"]
        assert result["outputs"][path.name]["sha256"] == _sha256(path)
    with pytest.raises(FileExistsError, match="must not already exist"):
        calibration.build_qcovars(CONFIG, eigenvec, fam, tmp_path / "qcovars")


def _synthetic_parent(tmp_path: Path) -> tuple[Path, Path]:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = tmp_path / payload["parent"]["run_id"]
    parent.mkdir()
    special = {
        "SUCCESS": "SUCCESS\n",
        "receipts/SOURCE_REVISION": payload["parent"]["source_git_commit"] + "\n",
        "receipts/post_qc_counts.tsv": (
            "post_qc_samples\t209\npost_qc_markers\t373279\nchromosomes\t1,2,3,4,5,6\n"
        ),
    }
    for item in payload["parent"]["required_files"]:
        path = parent.joinpath(*item["relative_path"].split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(special.get(item["relative_path"], "fixture\n"), encoding="utf-8")
        item["bytes"] = path.stat().st_size
        item["sha256"] = _sha256(path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, parent


def test_verify_parent_is_content_addressed_and_fail_closed(tmp_path: Path) -> None:
    manifest, parent = _synthetic_parent(tmp_path)
    result = calibration.verify_parent(manifest, parent)
    assert result["verified"] is True
    assert len(result["verified_files"]) == 15

    target = parent / "genotype/abamectin_209_qc.fam"
    target.write_text(target.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        calibration.verify_parent(manifest, parent)


def _synthetic_rows(model: str) -> list[_MlmaRow]:
    multiplier = {"pc0": 1.0, "pc3": 2.0, "pc5": 3.0, "pc10": 4.0}[model]
    values = [
        (5, "m_vl", 2693128, 1e-9 * multiplier),
        (5, "m_vr", 16100000, 1e-8 * multiplier),
        (2, "m_ii", 14121786, 1e-5 * multiplier),
        (1, "m_null1", 1000, 0.25 / multiplier),
        (1, "m_null2", 2000, 0.75 / multiplier),
    ]
    return [
        _MlmaRow(chromosome, snp, bp, "A", "G", 0.2, 0.5, 0.1, p)
        for chromosome, snp, bp, p in values
    ]


def test_summarize_calibration_reports_excluded_lambda_and_rank_stability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = calibration.load_calibration_manifest(CONFIG)
    manifest.parent.post_qc_markers = 5
    monkeypatch.setattr(calibration, "load_calibration_manifest", lambda _: manifest)

    results = tmp_path / "association"
    for model in calibration.MODEL_IDS:
        model_root = results / model
        model_root.mkdir(parents=True)
        for trait in TRAITS:
            (model_root / f"{TRAIT_SLUGS[trait]}.loco.mlma").write_text(
                "synthetic\n", encoding="utf-8"
            )

    def parse(path: Path, chromosomes: set[int]) -> list[_MlmaRow]:
        del chromosomes
        return _synthetic_rows(path.parent.name)

    monkeypatch.setattr(calibration, "_parse_mlma", parse)
    clumps = tmp_path / "clumps.tsv"
    with clumps.open("w", encoding="utf-8", newline="") as handle:
        handle.write("model\ttrait\tindex_clumps\n")
        for model in calibration.MODEL_IDS:
            for trait in TRAITS:
                handle.write(f"{model}\t{trait}\t1\n")

    summary = calibration.summarize_calibration(
        CONFIG,
        PARENT_CONFIG,
        results,
        clumps,
        tmp_path / "summary",
    )

    assert [item["id"] for item in summary["models"]] == list(calibration.MODEL_IDS)
    assert summary["models"][0]["primary"] is True
    assert summary["models"][1]["traits"][0]["rank_stability_to_pc0"][
        "deterministic_spearman_p_rank"
    ] == pytest.approx(1.0)
    assert summary["models"][0]["traits"][0]["lambda_gc_excluding_all_intervals"] > 0
    assert summary["models"][0]["traits"][0]["published_intervals"][0]["lead_marker"][
        "snp"
    ] == "m_vl"
    assert (tmp_path / "summary/ws283_pc_calibration_summary.json").is_file()
    assert "model may be selected" in (
        tmp_path / "summary/ws283_pc_calibration_summary.md"
    ).read_text(encoding="utf-8")


@pytest.mark.skip(
    reason="site-specific launchers are replaced by the portable execution layer"
)
def test_remote_runner_and_deployer_preserve_frozen_calibration_contract() -> None:
    runner = (ROOT / "scripts/run_abamectin_qtl_ws283_pc_calibration.sh").read_text(
        encoding="utf-8"
    )
    assert "verify-parent" in runner
    assert "--indep-pairwise 500kb 1 0.2" in runner
    assert "--indep-order 2" in runner
    assert "--pca 10" in runner
    assert "for model in pc3 pc5 pc10" in runner
    assert "--qcovar" in runner
    assert "--mlma-loco" in runner
    assert "--clump-p1 1.3394806565598387e-7" in runner
    assert "--clump-unphased" in runner
    assert "alternative_grm_performed" in runner
    assert "printf 'SUCCESS\\n'" in runner

    deployer = (ROOT / "scripts/deploy_ws283_pc_calibration.ps1").read_text(
        encoding="utf-8"
    )
    assert "status --porcelain=v1 --untracked-files=all" in deployer
    assert "git @GitOptions archive" in deployer
    assert "StrictHostKeyChecking=yes" in deployer
    assert "test ! -e '$RunRoot'" in deployer
    assert "bash -n '$RemoteSourceRoot/scripts/run_abamectin_qtl_ws283_pc_calibration.sh'" in deployer
    assert "Result archive contains a link or special entry" in deployer
    assert "Result bundle must contain exactly one terminal marker" in deployer
