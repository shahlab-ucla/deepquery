from __future__ import annotations

import copy
from pathlib import Path

import pytest

from wormctx.poc import developmental_raw_archive_identity as raw_identity
from wormctx.poc.developmental_omix import _sha256


ROOT = Path(__file__).resolve().parents[1]
READINESS_CONFIG = ROOT / "experiments/developmental_genetics/semantic_stage_and_lineage_preparation/config/readiness_gate.json"


def _files(root: Path, payloads: dict[str, bytes]) -> None:
    root.mkdir()
    for name, content in payloads.items():
        (root / name).write_bytes(content)


def test_opaque_member_inventory_and_exact_candidate_scan(tmp_path: Path) -> None:
    main = tmp_path / "Raw_data_RNAi"
    add = tmp_path / "Raw_data_RNAi_add"
    _files(main, {"geneA_emb1.txt": b"\xff\x00A", "geneB_emb1.txt": b"B"})
    _files(add, {"geneA_emb1.txt": b"different", "renamed.txt": b"B"})

    main_inventory = raw_identity._inventory_directory(
        main,
        expected_name="Raw_data_RNAi",
        expected_files=2,
        source_archive_sha256="a" * 64,
    )
    add_inventory = raw_identity._inventory_directory(
        add,
        expected_name="Raw_data_RNAi_add",
        expected_files=2,
        source_archive_sha256="b" * 64,
    )
    candidates = raw_identity._compare_inventories(main_inventory, add_inventory)

    assert main_inventory["access"]["member_bytes_hashed_opaquely"] is True
    assert main_inventory["access"]["member_values_decoded"] is False
    assert candidates["counts"] == {
        "exact_relative_path_matches": 1,
        "exact_basename_matches": 1,
        "exact_content_sha256_matches": 1,
    }
    assert candidates["candidate_resolution_required"] is True
    assert candidates["biological_embryo_disjointness_established"] is False
    assert candidates["current_overlap_contract_closed"] is False


def test_no_exact_match_is_still_not_contract_closure(tmp_path: Path) -> None:
    main = tmp_path / "Raw_data_RNAi"
    add = tmp_path / "Raw_data_RNAi_add"
    _files(main, {"one.txt": b"one"})
    _files(add, {"two.txt": b"two"})
    first = raw_identity._inventory_directory(
        main,
        expected_name="Raw_data_RNAi",
        expected_files=1,
        source_archive_sha256="a" * 64,
    )
    second = raw_identity._inventory_directory(
        add,
        expected_name="Raw_data_RNAi_add",
        expected_files=1,
        source_archive_sha256="b" * 64,
    )
    result = raw_identity._compare_inventories(first, second)

    assert result["candidate_resolution_required"] is False
    assert result["interpretation"].startswith("no_exact_match_supports")
    assert result["current_overlap_contract_closed"] is False


def test_full_identity_bundle_remains_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    main_archive_dir = tmp_path / "main-archive"
    add_archive_dir = tmp_path / "add-archive"
    main_archive_dir.mkdir()
    add_archive_dir.mkdir()
    main_archive = main_archive_dir / "OMIX709-05-54.rar"
    add_archive = add_archive_dir / "OMIX709-05-55.rar"
    main_archive.write_bytes(b"synthetic-main")
    add_archive.write_bytes(b"synthetic-add")
    main = tmp_path / "Raw_data_RNAi"
    add = tmp_path / "Raw_data_RNAi_add"
    _files(main, {"main-1.txt": b"m1", "main-2.txt": b"m2"})
    _files(add, {"add-1.txt": b"a1"})

    contract = raw_identity._load_readiness_contract(READINESS_CONFIG)
    contract = copy.deepcopy(contract)
    policy = contract["overlap"]["raw_archive_supporting_evidence"]
    policy["expected_bytes"] = main_archive.stat().st_size
    policy["expected_sha256"] = _sha256(main_archive)
    policy["expected_embryos"] = 2
    policy["comparison_expected_bytes"] = add_archive.stat().st_size
    policy["comparison_expected_embryos"] = 1
    policy["comparison_expected_sha256"] = _sha256(add_archive)
    monkeypatch.setattr(raw_identity, "_load_readiness_contract", lambda _path: contract)

    output = tmp_path / "qualification"
    result = raw_identity.qualify_raw_archives(
        READINESS_CONFIG,
        main_archive,
        add_archive,
        main,
        add,
        output,
    )

    assert result["evidence_status"] == "supporting_identity_evidence_complete"
    assert result["model_launch_status"] == "blocked_pending_pilot_embryo_overlap_receipt"
    assert result["current_overlap_contract_closed"] is False
    assert result["gates"]["real_model_fit_permitted"] is False
    assert raw_identity.verify_raw_identity(output)["verified"] is True


def test_inventory_rejects_unexpected_member_count(tmp_path: Path) -> None:
    root = tmp_path / "Raw_data_RNAi"
    _files(root, {"only.txt": b"opaque"})

    with pytest.raises(ValueError, match="member count differs"):
        raw_identity._inventory_directory(
            root,
            expected_name="Raw_data_RNAi",
            expected_files=2,
            source_archive_sha256="a" * 64,
        )
