"""Qualify WS276/WS283 coordinate identity without phenotype or association inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


CONFIG_VERSION = "wormctx-ws276-ws283-coordinate-qualification-config-1.0"
RECEIPT_VERSION = "wormctx-ws276-ws283-coordinate-identity-1.0"
WS276_ASSEMBLY = "PRJNA13758.WS276"
WS283_ASSEMBLY = "PRJNA13758.WS283_inferred_not_declared_in_vcf_header"
EXPECTED_REGIONS = {
    "chrv_left": (1_747_612, 4_333_001),
    "chrv_right": (13_606_517, 16_754_986),
}
_ALLOWED_BASES = frozenset(b"ACGTNRYKMSWBDHVacgtnrykmswbdhv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class FastaRecord:
    name: str
    length: int
    offset: int
    line_bases: int
    line_width: int


@dataclass(frozen=True)
class FastaScan:
    records: tuple[FastaRecord, ...]
    chromosome_v: bytes

    @property
    def fai_bytes(self) -> bytes:
        return "".join(
            f"{item.name}\t{item.length}\t{item.offset}\t"
            f"{item.line_bases}\t{item.line_width}\n"
            for item in self.records
        ).encode("ascii")


def scan_fasta(path: str | Path) -> FastaScan:
    """Scan a regular FASTA, emit its FAI representation, and retain chromosome V."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"FASTA is not a file: {source}")
    records: list[FastaRecord] = []
    chromosome_v = bytearray()
    name: str | None = None
    length = 0
    offset = 0
    line_bases = 0
    line_width = 0
    short_line_seen = False

    def finish_record() -> None:
        nonlocal name, length, offset, line_bases, line_width, short_line_seen
        if name is None:
            return
        if length <= 0 or line_bases <= 0 or line_width < line_bases:
            raise ValueError(f"FASTA record {name!r} is empty or malformed")
        records.append(FastaRecord(name, length, offset, line_bases, line_width))
        name = None
        length = 0
        offset = 0
        line_bases = 0
        line_width = 0
        short_line_seen = False

    with source.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                break
            if raw.startswith(b">"):
                finish_record()
                try:
                    header = raw[1:].strip().decode("ascii")
                except UnicodeDecodeError as error:
                    raise ValueError("FASTA header must be ASCII") from error
                name = header.split()[0] if header else ""
                if not name or any(item.name == name for item in records):
                    raise ValueError("FASTA record names must be nonempty and unique")
                offset = handle.tell()
                continue
            if name is None:
                raise ValueError("FASTA sequence occurs before the first header")
            sequence = raw.rstrip(b"\r\n")
            if not sequence or any(base not in _ALLOWED_BASES for base in sequence):
                raise ValueError(f"FASTA record {name!r} contains an invalid sequence line")
            if line_bases == 0:
                line_bases = len(sequence)
                line_width = len(raw)
            elif short_line_seen:
                raise ValueError(f"FASTA record {name!r} has sequence after a short line")
            elif len(sequence) != line_bases or len(raw) != line_width:
                if len(sequence) > line_bases:
                    raise ValueError(f"FASTA record {name!r} has inconsistent line width")
                short_line_seen = True
            length += len(sequence)
            if name == "V":
                chromosome_v.extend(sequence.upper())
    finish_record()
    if not records or not chromosome_v:
        raise ValueError("FASTA must contain a nonempty chromosome named V")
    return FastaScan(tuple(records), bytes(chromosome_v))


def _verify_asset(path: Path, specification: dict[str, Any], label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    if path.stat().st_size != specification["bytes"]:
        raise ValueError(f"{label} byte count differs from the frozen input")
    digest = _sha256(path)
    if digest != specification["sha256"]:
        raise ValueError(f"{label} SHA-256 differs from the frozen input")
    return digest


def qualify_coordinate_identity(
    config_path: str | Path,
    source_fasta_path: str | Path,
    target_fasta_path: str | Path,
    target_fai_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify both references and atomically publish a coordinate-identity receipt."""

    config_source = Path(config_path).resolve()
    config = json.loads(config_source.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_VERSION:
        raise ValueError("coordinate qualification config schema differs")
    if config.get("source_assembly") != WS276_ASSEMBLY:
        raise ValueError("coordinate qualification source assembly differs")
    if config.get("target_assembly") != WS283_ASSEMBLY:
        raise ValueError("coordinate qualification target assembly differs")
    expected_flags = {
        "generated_before_state_ingestion": True,
        "phenotype_paths_accepted": False,
        "association_result_paths_accepted": False,
        "operator_outcome_blinding_asserted": False,
        "outcome_access_claim_scope": "qualification_execution_path_only",
        "numerical_coordinate_equality_required": True,
        "full_chromosome_v_sequence_identity_required": True,
    }
    for key, expected in expected_flags.items():
        if config.get(key) != expected:
            raise ValueError(f"coordinate qualification flag differs: {key}")
    if config.get("regions") != [
        {"region_id": key, "start": value[0], "end": value[1]}
        for key, value in EXPECTED_REGIONS.items()
    ]:
        raise ValueError("coordinate qualification regions differ from the frozen design")

    source_fasta = Path(source_fasta_path).resolve()
    target_fasta = Path(target_fasta_path).resolve()
    target_fai = Path(target_fai_path).resolve()
    source_sha = _verify_asset(source_fasta, config["source_fasta"], "source FASTA")
    target_sha = _verify_asset(target_fasta, config["target_fasta"], "target FASTA")
    target_fai_sha = _verify_asset(target_fai, config["target_fai"], "target FAI")
    source_scan = scan_fasta(source_fasta)
    target_scan = scan_fasta(target_fasta)
    if target_scan.fai_bytes != target_fai.read_bytes():
        raise ValueError("target FAI differs from a fresh scan of the frozen FASTA")
    source_v = source_scan.chromosome_v
    target_v = target_scan.chromosome_v
    if len(source_v) != len(target_v) or source_v != target_v:
        raise ValueError("WS276 and WS283 chromosome V sequences are not identical")
    source_v_sha = hashlib.sha256(source_v).hexdigest()
    target_v_sha = hashlib.sha256(target_v).hexdigest()
    regions: list[dict[str, Any]] = []
    for region_id, (start, end) in EXPECTED_REGIONS.items():
        source_interval = source_v[start - 1 : end]
        target_interval = target_v[start - 1 : end]
        if len(source_interval) != end - start + 1 or source_interval != target_interval:
            raise ValueError(f"coordinate identity failed for {region_id}")
        digest = hashlib.sha256(source_interval).hexdigest()
        regions.append(
            {
                "mapping_mode": "sequence_identical_coordinate_identity",
                "region_id": region_id,
                "source_assembly": WS276_ASSEMBLY,
                "source_chromosome": 5,
                "source_end": end,
                "source_interval_sha256": digest,
                "source_start": start,
                "target_assembly": WS283_ASSEMBLY,
                "target_chromosome": 5,
                "target_end": end,
                "target_interval_sha256": digest,
                "target_start": start,
            }
        )

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"coordinate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        source_fai = temporary / "ws276_reference.fa.fai"
        source_fai.write_bytes(source_scan.fai_bytes)
        receipt = {
            "association_results_accessed": False,
            "full_reference_byte_identical": source_sha == target_sha,
            "generated_before_state_ingestion": True,
            "operator_outcome_blinding_asserted": False,
            "outcome_access_claim_scope": "qualification_execution_path_only",
            "phenotype_accessed": False,
            "regions": regions,
            "schema_version": RECEIPT_VERSION,
            "source_assembly": WS276_ASSEMBLY,
            "source_chromosome_v_length": len(source_v),
            "source_chromosome_v_sha256": source_v_sha,
            "source_reference_fai_sha256": _sha256(source_fai),
            "source_reference_fasta_sha256": source_sha,
            "target_assembly": WS283_ASSEMBLY,
            "target_chromosome_v_length": len(target_v),
            "target_chromosome_v_sha256": target_v_sha,
            "target_reference_fai_sha256": target_fai_sha,
            "target_reference_fasta_sha256": target_sha,
        }
        receipt_path = temporary / "coordinate_identity_receipt.json"
        receipt_path.write_bytes(_canonical_json(receipt))
        inventory = {
            "coordinate_identity_receipt.json": _sha256(receipt_path),
            "ws276_reference.fa.fai": _sha256(source_fai),
        }
        (temporary / "MANIFEST.json").write_bytes(_canonical_json(inventory))
        (temporary / "SUCCESS").write_text("qualified\n", encoding="utf-8")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-fasta", required=True)
    parser.add_argument("--target-fasta", required=True)
    parser.add_argument("--target-fai", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    receipt = qualify_coordinate_identity(
        args.config,
        args.source_fasta,
        args.target_fasta,
        args.target_fai,
        args.output_dir,
    )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
