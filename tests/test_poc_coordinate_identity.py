from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from wormctx.poc import coordinate_identity as identity


def _write_fasta(path: Path, chromosome_v: bytes) -> None:
    path.write_bytes(
        b">I synthetic\nACGTACGT\n>V synthetic\n"
        + chromosome_v[:8]
        + b"\n"
        + chromosome_v[8:]
        + b"\n"
    )


def test_fasta_scan_emits_fai_and_canonical_chromosome_v(tmp_path: Path) -> None:
    fasta = tmp_path / "reference.fa"
    _write_fasta(fasta, b"acgtnrykmswbdhv")

    scan = identity.scan_fasta(fasta)

    assert scan.chromosome_v == b"ACGTNRYKMSWBDHV"
    assert [item.name for item in scan.records] == ["I", "V"]
    assert scan.records[1].length == 15
    assert scan.fai_bytes.startswith(b"I\t8\t13\t8\t9\nV\t15\t")


def test_fasta_scan_rejects_sequence_after_short_line(tmp_path: Path) -> None:
    fasta = tmp_path / "invalid.fa"
    fasta.write_bytes(b">V\nACGT\nAC\nACGT\n")

    with pytest.raises(ValueError, match="after a short line"):
        identity.scan_fasta(fasta)


def test_fasta_scan_rejects_invalid_bases(tmp_path: Path) -> None:
    fasta = tmp_path / "invalid.fa"
    fasta.write_bytes(b">V\nACGTZ\n")

    with pytest.raises(ValueError, match="invalid sequence line"):
        identity.scan_fasta(fasta)


def test_interval_digest_uses_one_based_closed_coordinates() -> None:
    sequence = b"AACCGGTT"
    observed = hashlib.sha256(sequence[1:6]).hexdigest()
    expected = hashlib.sha256(b"ACCGG").hexdigest()

    assert observed == expected
