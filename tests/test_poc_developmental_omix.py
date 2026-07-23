from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wormctx.poc import developmental_omix as omix


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/developmental_genetics/pilot_early_to_late_prediction/config/pilot_prediction.json"
RAW_HEADER = (
    "\ttime\tcell_name\tX\tY\tZ\tsize\traw-expression\tblot-correction\n"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_embryo(path: Path, offset: float = 0.0, *, duplicate: bool = False) -> None:
    rows = [
        (0, 1, "ABa", 10.0, 20.0, 4.0, 28.0, 100.0, 101.0),
        (1, 1, "ABp", 20.0, 20.0, 4.0, 29.0, 105.0, 106.0),
        (2, 2, "ABal", 8.0, 22.0, 4.5, 25.0, 110.0, 111.0),
        (3, 2, "ABar", 12.0, 22.0, 4.5, 25.5, 112.0, 113.0),
    ]
    if duplicate:
        rows.append((4, 1, "ABa", 11.0, 21.0, 4.0, 28.0, 100.0, 101.0))
    lines = [RAW_HEADER]
    for index, time, cell, x, y, z, size, raw, corrected in rows:
        values = (
            index,
            time,
            cell,
            x + offset,
            y + offset,
            z,
            size,
            raw + offset,
            corrected + offset,
        )
        lines.append("\t".join(str(value) for value in values) + "\n")
    path.write_text("".join(lines), encoding="utf-8", newline="\n")


def _fixture_roots(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
    control = tmp_path / "Raw_data_Control"
    addendum = tmp_path / "Raw_data_RNAi_add"
    control.mkdir()
    addendum.mkdir()

    for embryo in range(1, 5):
        _write_embryo(control / f"ctr_emb{embryo}_raw_data.txt", embryo / 100)
    for embryo in range(1, 3):
        _write_embryo(addendum / f"CTR_add_emb{embryo}_raw_data.txt", embryo / 50)
        _write_embryo(
            addendum / f"C30H7.2_add_emb{embryo}_raw_data.txt",
            1.0 + embryo / 10,
        )
        _write_embryo(
            addendum / f"VPR-1_add_emb{embryo}_raw_data.txt",
            2.0 + embryo / 10,
        )

    archives = tmp_path / "archives"
    archives.mkdir()
    control_archive = archives / "OMIX709-05-53.rar"
    addendum_archive = archives / "OMIX709-05-55.rar"
    control_archive.write_bytes(b"tiny deterministic control archive fixture\n")
    addendum_archive.write_bytes(b"tiny deterministic addendum archive fixture\n")
    receipts = {
        "reference_control": omix.verify_source_archive(
            control_archive,
            control_archive.stat().st_size,
            _sha256(control_archive),
        ),
        "addendum": omix.verify_source_archive(
            addendum_archive,
            addendum_archive.stat().st_size,
            _sha256(addendum_archive),
        ),
    }
    return control, addendum, receipts


def _normalize_fixture(tmp_path: Path, name: str = "normalized.json") -> tuple[Path, dict]:
    control, addendum, receipts = _fixture_roots(tmp_path)
    output = tmp_path / name
    receipt = omix.normalize_dataset(control, addendum, output, receipts)
    assert output.is_file()
    assert receipt["sha256"] == _sha256(output)
    return output, receipt


def _payload_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    records = payload["embryos"]
    assert isinstance(records, list)
    return records


def test_manifest_and_cli_freeze_the_real_developmental_contract() -> None:
    manifest = omix.load_manifest(CONFIG)
    encoded = json.dumps(manifest.model_dump(mode="json"), sort_keys=True)

    for frozen_value in (
        "cf85a12ec6b6eabf900e303a630e366f61e2a7aeff0b050feb38bb26a06c07a4",
        "429f8888f3549738e66e28c7f2af440233982b96950350fd97b5d2b6d68d49c8",
        "leave_one_perturbation_target_out",
        "reference_controls_only",
        "semantic_cell_id",
        "supportive",
        "not_supportive",
        "not_estimable",
    ):
        assert frozen_value in encoded
    assert "gene_identity_features_permitted" in encoded
    assert '"gene_identity_features_permitted": false' in encoded

    help_text = omix._parser().format_help()
    for command in ("normalize", "qualify", "run", "verify-run"):
        assert command in help_text


def test_source_archive_identity_is_exact_and_fail_closed(tmp_path: Path) -> None:
    archive = tmp_path / "OMIX709-05-53.rar"
    archive.write_bytes(b"content addressed fixture\n")
    expected_hash = _sha256(archive)
    receipt = omix.verify_source_archive(archive, archive.stat().st_size, expected_hash)

    assert receipt["verified"] is True
    assert receipt["bytes"] == archive.stat().st_size
    assert receipt["sha256"] == expected_hash

    with pytest.raises(ValueError, match="(bytes|size|identity)"):
        omix.verify_source_archive(archive, archive.stat().st_size + 1, expected_hash)
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="(SHA-256|sha256|hash|identity)"):
        omix.verify_source_archive(archive, archive.stat().st_size, expected_hash)


def test_normalization_is_deterministic_and_retains_semantic_cells(
    tmp_path: Path,
) -> None:
    control, addendum, receipts = _fixture_roots(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_receipt = omix.normalize_dataset(control, addendum, first, receipts)
    second_receipt = omix.normalize_dataset(control, addendum, second, receipts)

    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["sha256"] == second_receipt["sha256"] == _sha256(first)
    records = _payload_records(first)
    assert len(records) == 10
    assert {record["role"] for record in records} == {
        "reference_control",
        "external_control",
        "perturbation",
    }
    for record in records:
        assert record["embryo_id"]
        assert len(record["observations"]) == 4
        assert {item["cell_id"] for item in record["observations"]} == {
            "ABa",
            "ABp",
            "ABal",
            "ABar",
        }
        assert all(isinstance(item["cell_id"], str) for item in record["observations"])

    qualified = omix.qualify_dataset(
        first,
        expected_counts={
            "total_embryos": 10,
            "reference_control_embryos": 4,
            "external_control_embryos": 2,
            "perturbation_embryos": 4,
            "perturbation_conditions": 2,
        },
        expected_sha256=first_receipt["sha256"],
    )
    assert qualified["qualified"] is True
    with pytest.raises(FileExistsError, match="exist"):
        omix.normalize_dataset(control, addendum, first, receipts)


def test_normalization_rejects_duplicate_semantic_cell_at_one_time(tmp_path: Path) -> None:
    control, addendum, receipts = _fixture_roots(tmp_path)
    _write_embryo(control / "ctr_emb1_raw_data.txt", duplicate=True)

    with pytest.raises(ValueError, match="(duplicate|cell.*time|time.*cell)"):
        omix.normalize_dataset(
            control,
            addendum,
            tmp_path / "invalid.json",
            receipts,
        )


def test_leave_one_condition_out_folds_hold_out_whole_embryos(
    tmp_path: Path,
) -> None:
    normalized, _ = _normalize_fixture(tmp_path)
    records = _payload_records(normalized)
    folds = omix.build_leave_one_condition_out_folds(records)

    perturbations = [record for record in records if record["role"] == "perturbation"]
    perturbation_ids = {record["embryo_id"] for record in perturbations}
    condition_by_id = {
        record["embryo_id"]: record["condition"] for record in perturbations
    }
    assert [fold["held_out_condition"] for fold in folds] == ["C30H7.2", "VPR-1"]

    seen_test: set[str] = set()
    for fold in folds:
        train = set(fold["train_embryo_ids"])
        test = set(fold["test_embryo_ids"])
        assert train.isdisjoint(test)
        assert test
        assert {condition_by_id[item] for item in test} == {
            fold["held_out_condition"]
        }
        assert all(
            condition_by_id[item] != fold["held_out_condition"]
            for item in train
            if item in condition_by_id
        )
        assert seen_test.isdisjoint(test)
        seen_test.update(test)
    assert seen_test == perturbation_ids
    assert folds == omix.build_leave_one_condition_out_folds(list(reversed(records)))


def test_primary_features_mask_gene_identity_and_ignore_row_order(tmp_path: Path) -> None:
    normalized, _ = _normalize_fixture(tmp_path)
    records = _payload_records(normalized)
    perturbations = [record for record in records if record["role"] == "perturbation"]

    first = omix.build_primary_features(perturbations)
    reordered = copy.deepcopy(list(reversed(perturbations)))
    for record in reordered:
        record["observations"].reverse()
    second = omix.build_primary_features(reordered)

    assert first["embryo_ids"] == second["embryo_ids"]
    assert first["feature_names"] == second["feature_names"]
    np.testing.assert_allclose(first["matrix"], second["matrix"], rtol=0, atol=0)
    assert {"ABa", "ABp", "ABal", "ABar"}.issubset(first["semantic_cell_ids"])

    relabeled = copy.deepcopy(perturbations)
    for index, record in enumerate(relabeled):
        record["condition"] = f"secret-target-{index}"
    third = omix.build_primary_features(relabeled)
    assert first["feature_names"] == third["feature_names"]
    np.testing.assert_allclose(first["matrix"], third["matrix"], rtol=0, atol=0)
    forbidden = ("condition", "gene", "target", "filename", "embryo_serial")
    assert not any(
        token in feature.lower()
        for feature in first["feature_names"]
        for token in forbidden
    )


def test_control_threshold_is_deterministic_and_records_control_only_source() -> None:
    scores = {f"reference-{index:02d}": float(index) for index in range(1, 21)}
    forward = omix.calibrate_control_threshold(scores, alpha=0.05)
    reverse = omix.calibrate_control_threshold(dict(reversed(list(scores.items()))), alpha=0.05)

    assert forward == reverse
    assert forward["calibration_source"] == "reference_controls_only"
    assert forward["n_controls"] == 20
    assert forward["alpha"] == 0.05
    assert forward["threshold"] == 20.0
    assert forward["quantile_method"]
    with pytest.raises(ValueError, match="(finite|score)"):
        omix.calibrate_control_threshold({"bad-control": float("nan")})


def test_run_separates_technical_success_from_scientific_status_and_replays(
    tmp_path: Path,
) -> None:
    normalized, _ = _normalize_fixture(tmp_path)
    first_root = tmp_path / "run-first"
    second_root = tmp_path / "run-second"
    first = omix.run_analysis(normalized, first_root, seed=1729)
    second = omix.run_analysis(normalized, second_root, seed=1729)

    assert first == second
    assert first["technical_status"] == "success"
    assert first["scientific_status"] in {"not_supportive", "not_estimable"}
    assert first["technical_status"] != first["scientific_status"]
    assert first["scientific_status"] != "supportive"

    first_summary = first_root / "omix709_developmental_summary.json"
    second_summary = second_root / "omix709_developmental_summary.json"
    assert first_summary.read_bytes() == second_summary.read_bytes()
    verification = omix.verify_run(first_root)
    assert verification["verified"] is True

    first_summary.write_bytes(first_summary.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="(checksum|hash|identity|manifest)"):
        omix.verify_run(first_root)


@pytest.mark.skip(
    reason="the governed-data runner is intentionally absent from the public tree"
)
def test_remote_runner_freezes_resource_rights_and_scientific_parent_boundaries() -> None:
    runner = (ROOT / "scripts/run_omix709_developmental_robustness.sh").read_text(
        encoding="utf-8"
    )

    for required in (
        "ACADEMIC_USE_ATTESTED",
        "controlled noncommercial academic-use attestation is required",
        "33554432",
        "maximum_load1_exclusive",
        "THREADS must be 1 or 2",
        "concurrency_only_null_run_path.txt",
        '"scientific_parent_runs": []',
        "verify-run",
        "SHA256SUMS.txt",
    ):
        assert required in runner
    assert "import numpy, pydantic" in runner
    assert "sklearn" not in runner
    assert "! -path './SUCCESS' ! -path './FAILURE'" in runner
    assert "! -name SUCCESS" not in runner


@pytest.mark.skip(
    reason="the governed-data deployer is intentionally absent from the public tree"
)
def test_deployer_ships_only_the_frozen_normalized_product() -> None:
    deployer = (ROOT / "scripts/deploy_omix709_developmental_robustness.ps1").read_text(
        encoding="utf-8"
    )

    for required in (
        "AcademicUseAttested",
        "62315263",
        "51fcda2051bb0d7cff376581e29e80be20da013211b4c40cb16a2186b7039659",
        "OMIX709-05-53.rar",
        "OMIX709-05-55.rar",
        "omix709_normalized.json",
        "normalized-input.tar.gz",
        "END {exit !(NR == 4)}",
        "grep -Fqx -- 'deployment-manifest.json'",
        'ValidateSet("Launch", "StartPrepared", "Status", "Retrieve")',
    ):
        assert required in deployer
    assert "OMIX709-05-53.rar' '$Remote" not in deployer
    assert "OMIX709-05-55.rar' '$Remote" not in deployer
    assert "input_members=" not in deployer
    assert "PermittedLegacyInnerMarker" in deployer
    assert "Windows SCP path limit" in deployer


def test_cross_platform_summary_comparison_tolerates_only_last_bit_floats(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(
        json.dumps(
            {
                "technical_status": "success",
                "scientific_status": "not_supportive",
                "counts": {"embryos": 251},
                "metric": 0.02877154986201814,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            {
                "technical_status": "success",
                "scientific_status": "not_supportive",
                "counts": {"embryos": 251},
                "metric": 0.028771549862018138,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    receipt = omix.compare_summaries(reference, candidate, absolute_tolerance=1e-12)
    assert receipt["equivalent_within_tolerance"] is True
    assert receipt["byte_identical"] is False
    assert receipt["maximum_absolute_difference"] < 1e-12

    changed = json.loads(candidate.read_text(encoding="utf-8"))
    changed["scientific_status"] = "supportive"
    candidate.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    rejected = omix.compare_summaries(reference, candidate, absolute_tolerance=1e-12)
    assert rejected["equivalent_within_tolerance"] is False
    assert rejected["mismatches"][0]["path"] == "summary.scientific_status"
