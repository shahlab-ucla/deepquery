"""Identity-only OMIX709 raw-archive inventory and overlap-candidate scan.

The scanner hashes archive members as opaque bytes.  It never parses a row,
decodes a measurement column, or opens a prepared endpoint.  Absence of exact
member-name/content matches is supporting evidence only and cannot close the
frozen biological-embryo overlap gate by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

READINESS_SCHEMA_VERSION = "wormctx-omix709-developmental-readiness-contract-1.0"
ARCHIVE_RECEIPT_VERSION = "wormctx-omix709-raw-archive-identity-receipt-1.0"
MEMBER_INVENTORY_VERSION = "wormctx-omix709-raw-member-inventory-1.0"
OVERLAP_CANDIDATE_VERSION = "wormctx-omix709-raw-overlap-candidates-1.0"
QUALIFICATION_VERSION = "wormctx-omix709-raw-identity-qualification-1.0"
VERIFY_VERSION = "wormctx-omix709-raw-identity-verification-1.0"
OUTPUT_FILES = {
    "archive_receipt.json",
    "main_member_inventory.json",
    "addendum_member_inventory.json",
    "overlap_candidates.json",
    "qualification.json",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"output must not already exist: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    content = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    _write_bytes_atomic(path, content)


def _load_readiness_contract(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != READINESS_SCHEMA_VERSION:
        raise ValueError("unknown developmental readiness contract")
    if any(
        payload.get(name) is not False
        for name in (
            "outcome_values_permitted",
            "outcome_prevalence_inspection_permitted",
            "sealed_test_access_permitted",
        )
    ):
        raise ValueError("raw identity qualification requires an outcome-blind contract")
    policy = payload.get("overlap", {}).get("raw_archive_supporting_evidence", {})
    if (
        policy.get("omix_file_id") != "OMIX709-05-54"
        or policy.get("filename") != "OMIX709-05-54.rar"
        or policy.get("expected_bytes") != 677141156
        or policy.get("expected_sha256")
        != "5e268e447c8f45fdd83823c029cfef6d67d389ad39013b66f9cb7191eb9a2030"
        or policy.get("expected_embryos") != 2075
        or policy.get("expected_extracted_root") != "Raw_data_RNAi"
        or policy.get("comparison_omix_file_id") != "OMIX709-05-55"
        or policy.get("comparison_filename") != "OMIX709-05-55.rar"
        or policy.get("comparison_expected_bytes") != 51428713
        or policy.get("comparison_expected_sha256")
        != "429f8888f3549738e66e28c7f2af440233982b96950350fd97b5d2b6d68d49c8"
        or policy.get("comparison_expected_embryos") != 146
        or policy.get("comparison_expected_extracted_root") != "Raw_data_RNAi_add"
        or policy.get("permitted_access")
        != "archive_identity_member_names_member_sizes_and_member_sha256_only"
        or policy.get("outcome_columns_permitted") is not False
        or policy.get("no_exact_member_or_hash_match_interpretation")
        != "supporting_evidence_only_not_contract_closure"
    ):
        raise ValueError("raw archive supporting-evidence contract differs")
    return payload


def _inventory_directory(
    root: str | Path,
    *,
    expected_name: str,
    expected_files: int,
    source_archive_sha256: str,
) -> dict[str, Any]:
    directory = Path(root).resolve()
    if not directory.is_dir() or directory.name != expected_name:
        raise ValueError(f"extracted root must be named {expected_name}")
    members: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("raw archive inventory may not follow symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("raw archive member is not a regular file")
        relative = path.relative_to(directory).as_posix()
        members.append(
            {
                "relative_path": relative,
                "basename": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if len(members) != expected_files:
        raise ValueError(
            f"{expected_name} member count differs: expected {expected_files}, "
            f"observed {len(members)}"
        )
    if len({item["relative_path"] for item in members}) != len(members):
        raise ValueError("raw archive inventory contains duplicate relative paths")
    return {
        "schema_version": MEMBER_INVENTORY_VERSION,
        "extracted_root": expected_name,
        "source_archive_sha256": source_archive_sha256,
        "file_count": len(members),
        "total_bytes": sum(int(item["bytes"]) for item in members),
        "members": members,
        "access": {
            "member_names_read": True,
            "member_sizes_read": True,
            "member_bytes_hashed_opaquely": True,
            "member_values_decoded": False,
            "outcome_columns_decoded": False,
        },
    }


def _compare_inventories(
    main: Mapping[str, Any], addendum: Mapping[str, Any]
) -> dict[str, Any]:
    main_by_path = {str(item["relative_path"]): item for item in main["members"]}
    add_by_path = {str(item["relative_path"]): item for item in addendum["members"]}
    exact_paths = sorted(set(main_by_path) & set(add_by_path))

    main_by_basename: dict[str, list[str]] = defaultdict(list)
    add_by_basename: dict[str, list[str]] = defaultdict(list)
    main_by_hash: dict[str, list[str]] = defaultdict(list)
    add_by_hash: dict[str, list[str]] = defaultdict(list)
    for item in main["members"]:
        main_by_basename[str(item["basename"])].append(str(item["relative_path"]))
        main_by_hash[str(item["sha256"])].append(str(item["relative_path"]))
    for item in addendum["members"]:
        add_by_basename[str(item["basename"])].append(str(item["relative_path"]))
        add_by_hash[str(item["sha256"])].append(str(item["relative_path"]))

    basename_matches = [
        {
            "basename": name,
            "main_members": sorted(main_by_basename[name]),
            "addendum_members": sorted(add_by_basename[name]),
        }
        for name in sorted(set(main_by_basename) & set(add_by_basename))
    ]
    hash_matches = [
        {
            "sha256": digest,
            "main_members": sorted(main_by_hash[digest]),
            "addendum_members": sorted(add_by_hash[digest]),
        }
        for digest in sorted(set(main_by_hash) & set(add_by_hash))
    ]
    any_candidate = bool(exact_paths or basename_matches or hash_matches)
    return {
        "schema_version": OVERLAP_CANDIDATE_VERSION,
        "exact_relative_path_matches": exact_paths,
        "exact_basename_matches": basename_matches,
        "exact_content_sha256_matches": hash_matches,
        "counts": {
            "exact_relative_path_matches": len(exact_paths),
            "exact_basename_matches": len(basename_matches),
            "exact_content_sha256_matches": len(hash_matches),
        },
        "candidate_resolution_required": any_candidate,
        "interpretation": (
            "fatal_overlap_candidate_requires_source_curator_resolution"
            if any_candidate
            else "no_exact_match_supports_separate_source_objects_but_does_not_close_contract"
        ),
        "biological_embryo_disjointness_established": False,
        "current_overlap_contract_closed": False,
        "outcome_values_read": False,
    }


def _write_bundle(output_root: Path, payloads: Mapping[str, Any]) -> None:
    if output_root.exists():
        raise FileExistsError(f"raw identity output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".ri-{uuid.uuid4().hex[:8]}"
    temporary.mkdir()
    try:
        for name in sorted(payloads):
            _write_json_atomic(temporary / name, payloads[name])
        lines = [f"{_sha256(temporary / name)}  {name}" for name in sorted(payloads)]
        _write_bytes_atomic(
            temporary / "SHA256SUMS.txt", ("\n".join(lines) + "\n").encode("utf-8")
        )
        _write_bytes_atomic(temporary / "SUCCESS", b"SUCCESS\n")
        os.replace(temporary, output_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def qualify_raw_archives(
    readiness_contract_path: str | Path,
    main_archive_path: str | Path,
    addendum_archive_path: str | Path,
    main_extracted_root: str | Path,
    addendum_extracted_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    contract_path = Path(readiness_contract_path).resolve()
    contract = _load_readiness_contract(contract_path)
    policy = contract["overlap"]["raw_archive_supporting_evidence"]
    main_archive = Path(main_archive_path).resolve()
    addendum_archive = Path(addendum_archive_path).resolve()
    if (
        not main_archive.is_file()
        or main_archive.name != policy["filename"]
        or main_archive.stat().st_size != policy["expected_bytes"]
    ):
        raise ValueError("OMIX709-05-54 archive identity or completeness differs")
    if (
        not addendum_archive.is_file()
        or addendum_archive.name != policy["comparison_filename"]
        or addendum_archive.stat().st_size != policy["comparison_expected_bytes"]
    ):
        raise ValueError("OMIX709-05-55 archive identity or completeness differs")
    main_sha256 = _sha256(main_archive)
    addendum_sha256 = _sha256(addendum_archive)
    if main_sha256 != policy["expected_sha256"]:
        raise ValueError("OMIX709-05-54 archive SHA-256 differs from the frozen source")
    if addendum_sha256 != policy["comparison_expected_sha256"]:
        raise ValueError("OMIX709-05-55 archive SHA-256 differs from the frozen pilot source")

    main_inventory = _inventory_directory(
        main_extracted_root,
        expected_name=policy["expected_extracted_root"],
        expected_files=policy["expected_embryos"],
        source_archive_sha256=main_sha256,
    )
    addendum_inventory = _inventory_directory(
        addendum_extracted_root,
        expected_name=policy["comparison_expected_extracted_root"],
        expected_files=policy["comparison_expected_embryos"],
        source_archive_sha256=addendum_sha256,
    )
    candidates = _compare_inventories(main_inventory, addendum_inventory)
    archive_receipt = {
        "schema_version": ARCHIVE_RECEIPT_VERSION,
        "readiness_contract_sha256": _sha256(contract_path),
        "archives": [
            {
                "omix_file_id": policy["omix_file_id"],
                "filename": main_archive.name,
                "bytes": main_archive.stat().st_size,
                "sha256": main_sha256,
                "expected_embryos": policy["expected_embryos"],
                "extracted_root": policy["expected_extracted_root"],
            },
            {
                "omix_file_id": policy["comparison_omix_file_id"],
                "filename": addendum_archive.name,
                "bytes": addendum_archive.stat().st_size,
                "sha256": addendum_sha256,
                "expected_embryos": policy["comparison_expected_embryos"],
                "extracted_root": policy["comparison_expected_extracted_root"],
            },
        ],
        "access": {
            "archive_and_member_identity_only": True,
            "member_values_decoded": False,
            "outcome_columns_decoded": False,
            "outcome_prevalence_inspected": False,
            "sealed_test_opened": False,
        },
    }
    qualification = {
        "schema_version": QUALIFICATION_VERSION,
        "technical_status": "success",
        "evidence_status": (
            "overlap_candidate_requires_curator_resolution"
            if candidates["candidate_resolution_required"]
            else "supporting_identity_evidence_complete"
        ),
        "model_launch_status": "blocked_pending_pilot_embryo_overlap_receipt",
        "current_overlap_contract_closed": False,
        "counts": {
            "main_members": main_inventory["file_count"],
            "addendum_members": addendum_inventory["file_count"],
            **candidates["counts"],
        },
        "gates": {
            "both_archive_identities_frozen": True,
            "complete_member_inventories_frozen": True,
            "opaque_member_hashes_computed": True,
            "source_curator_attestation_received": False,
            "outcome_values_read": False,
            "outcome_prevalence_inspected": False,
            "sealed_test_opened": False,
            "real_model_fit_permitted": False,
        },
        "limitation": (
            "Archive/member identity comparison cannot exclude a renamed or reprocessed "
            "biological duplicate and therefore cannot close the frozen gate alone."
        ),
        "biological_claims_permitted": False,
    }
    _write_bundle(
        Path(output_root).resolve(),
        {
            "archive_receipt.json": archive_receipt,
            "main_member_inventory.json": main_inventory,
            "addendum_member_inventory.json": addendum_inventory,
            "overlap_candidates.json": candidates,
            "qualification.json": qualification,
        },
    )
    return qualification


def verify_raw_identity(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw identity output is missing: {root}")
    if {item.name for item in root.iterdir()} != OUTPUT_FILES | {"SHA256SUMS.txt", "SUCCESS"}:
        raise ValueError("raw identity output inventory differs")
    if (root / "SUCCESS").read_bytes() != b"SUCCESS\n":
        raise ValueError("raw identity SUCCESS marker differs")
    checked: set[str] = set()
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None or match.group(2) not in OUTPUT_FILES or match.group(2) in checked:
            raise ValueError("raw identity checksum manifest is malformed")
        if _sha256(root / match.group(2)) != match.group(1):
            raise ValueError(f"raw identity checksum failed: {match.group(2)}")
        checked.add(match.group(2))
    if checked != OUTPUT_FILES:
        raise ValueError("raw identity checksum coverage differs")
    qualification = json.loads((root / "qualification.json").read_text(encoding="utf-8"))
    candidates = json.loads((root / "overlap_candidates.json").read_text(encoding="utf-8"))
    if (
        qualification.get("schema_version") != QUALIFICATION_VERSION
        or qualification.get("technical_status") != "success"
        or qualification.get("current_overlap_contract_closed") is not False
        or qualification.get("gates", {}).get("outcome_values_read") is not False
        or qualification.get("gates", {}).get("sealed_test_opened") is not False
        or candidates.get("biological_embryo_disjointness_established") is not False
        or candidates.get("current_overlap_contract_closed") is not False
    ):
        raise ValueError("raw identity qualification overstates evidence or access")
    return {
        "schema_version": VERIFY_VERSION,
        "verified": True,
        "checked_files": len(checked),
        "evidence_status": qualification["evidence_status"],
        "current_overlap_contract_closed": False,
        "outcome_values_read": False,
        "sealed_test_opened": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wormctx.poc.developmental_raw_archive_identity",
        description="Freeze identity-only OMIX709 raw-archive overlap evidence",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    qualify = commands.add_parser("qualify")
    qualify.add_argument("--readiness-contract", required=True)
    qualify.add_argument("--main-archive", required=True)
    qualify.add_argument("--addendum-archive", required=True)
    qualify.add_argument("--main-extracted-root", required=True)
    qualify.add_argument("--addendum-extracted-root", required=True)
    qualify.add_argument("--output-root", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "qualify":
        payload = qualify_raw_archives(
            args.readiness_contract,
            args.main_archive,
            args.addendum_archive,
            args.main_extracted_root,
            args.addendum_extracted_root,
            args.output_root,
        )
    else:
        payload = verify_raw_identity(args.output_root)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
