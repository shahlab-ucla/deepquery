"""Qualify the bounded WormBase WS298 source qualification source profile.

This is a source/schema/integrity gate, not a biological analysis.  It reads
only the seven artifacts named by an immutable contract and publishes a small
receipt.  It never ingests the ACeDB archives or the full 17.9 GB mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wormctx.adapters.wormbase_ws298 import (
    EMPTY_GAF_QUALIFIER_RECEIPT_TOKEN,
    profile_table,
    read_gaf,
    read_gene_ids,
    read_obo,
    sha256_file,
)


CONTRACT_SCHEMA_VERSION = "wormctx-wormbase-ws298-source_qualification-source-contract-1.0"
RECEIPT_SCHEMA_VERSION = "wormctx-wormbase-ws298-source_qualification-source-qualification-1.0"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FrozenAsset(_StrictModel):
    artifact_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    role: Literal[
        "release_checksum_ledger",
        "release_letter",
        "gene_identifiers",
        "phenotype_associations",
        "development_associations",
        "phenotype_ontology",
        "development_ontology",
    ]
    relative_path: str
    source_url: str
    bytes: int = Field(gt=0)
    sha256: str
    md5_in_release_ledger: str | None = None
    compression: Literal["none", "gzip"]
    format: Literal["md5sum", "plain_text", "csv6", "gaf2.0", "obo1.2"]
    rights_status: Literal["cleared", "review_required"]
    license_uri: str | None = None
    external_transfer_permitted: bool

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != value:
            raise ValueError("artifact relative_path must be safe canonical POSIX")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")
        return value

    @field_validator("md5_in_release_ledger")
    @classmethod
    def valid_md5(cls, value: str | None) -> str | None:
        if value is not None and not MD5_PATTERN.fullmatch(value):
            raise ValueError("md5 must contain 32 lowercase hexadecimal characters")
        return value

    @model_validator(mode="after")
    def rights_contract(self) -> "FrozenAsset":
        if self.rights_status == "cleared" and self.license_uri is None:
            raise ValueError("rights-cleared artifacts require an explicit license")
        if self.rights_status == "review_required" and self.external_transfer_permitted:
            raise ValueError("review-required artifacts cannot permit external transfer")
        return self


class ExpectedProfile(_StrictModel):
    gene_rows: int = Field(gt=0)
    phenotype_association_rows: int = Field(gt=0)
    phenotype_unique_genes: int = Field(gt=0)
    phenotype_unique_terms: int = Field(gt=0)
    phenotype_negated_rows: int = Field(ge=0)
    phenotype_non_not_qualifier_rows: int = Field(ge=0)
    development_association_rows: int = Field(gt=0)
    development_unique_genes: int = Field(gt=0)
    development_unique_terms: int = Field(gt=0)
    development_negated_rows: int = Field(ge=0)
    development_anatomy_term_qualified_rows: int = Field(ge=0)
    phenotype_ontology_terms: int = Field(gt=0)
    phenotype_ontology_obsolete_terms: int = Field(ge=0)
    phenotype_ontology_alternate_ids: int = Field(ge=0)
    phenotype_ontology_ambiguous_ids: int = Field(ge=0)
    phenotype_ontology_nameless_obsolete_terms: int = Field(ge=0)
    development_ontology_terms: int = Field(gt=0)
    development_ontology_obsolete_terms: int = Field(ge=0)
    development_ontology_alternate_ids: int = Field(ge=0)
    development_ontology_ambiguous_ids: int = Field(ge=0)
    development_ontology_nameless_obsolete_terms: int = Field(ge=0)
    unresolved_gene_references: int = Field(ge=0)
    unresolved_gene_rows: int = Field(ge=0)
    unresolved_phenotype_references: Literal[0]
    unresolved_development_references: Literal[0]
    ambiguous_referenced_phenotype_ids: Literal[0]
    ambiguous_referenced_development_ids: Literal[0]


class ClaimBoundary(_StrictModel):
    biological_claims_permitted: Literal[False]
    permitted: list[str] = Field(min_length=1)
    prohibited: list[str] = Field(min_length=1)


class QualificationContract(_StrictModel):
    schema_version: Literal[CONTRACT_SCHEMA_VERSION]
    analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    classification: Literal["retrospective_source_qualification_only"]
    source_root_id: Literal["wormbase_ws298_mirror"]
    provider: Literal["WormBase"]
    release: Literal["WS298"]
    release_is_immutable: Literal[True]
    taxon: Literal["NCBITaxon:6239"]
    bioproject: Literal["NCBI:PRJNA13758"]
    genome_assembly: Literal["WBcel235/GCA_000002985.3/ce11"]
    adapter: Literal["wormbase_ws298_source_qualification"]
    adapter_version: Literal["0.1.0"]
    scope: Literal["gene_development_stage_phenotype"]
    total_expected_bytes: int = Field(gt=0)
    artifacts: list[FrozenAsset] = Field(min_length=7, max_length=7)
    expected_profile: ExpectedProfile
    required_gaf_date_generated: Literal["2025-11-05"]
    empty_gaf_qualifier_receipt_token: Literal["UNQUALIFIED"]
    required_ontology_data_versions: dict[str, str]
    release_letter_expected_gene_total: Literal[49164]
    rights_policy: dict[str, bool]
    claim_boundary: ClaimBoundary

    @model_validator(mode="after")
    def closed_scope(self) -> "QualificationContract":
        expected_roles = {
            "release_checksum_ledger",
            "release_letter",
            "gene_identifiers",
            "phenotype_associations",
            "development_associations",
            "phenotype_ontology",
            "development_ontology",
        }
        roles = [asset.role for asset in self.artifacts]
        identifiers = [asset.artifact_id for asset in self.artifacts]
        paths = [asset.relative_path for asset in self.artifacts]
        if set(roles) != expected_roles or len(roles) != len(set(roles)):
            raise ValueError("contract must contain each bounded source qualification role exactly once")
        if len(identifiers) != len(set(identifiers)) or len(paths) != len(set(paths)):
            raise ValueError("artifact identifiers and paths must be unique")
        if sum(asset.bytes for asset in self.artifacts) != self.total_expected_bytes:
            raise ValueError("total_expected_bytes differs from artifact byte sizes")
        if set(self.required_ontology_data_versions) != {
            "phenotype_ontology",
            "development_ontology",
        }:
            raise ValueError("ontology data-version contract is incomplete")
        if self.rights_policy != {
            "controlled_noncommercial_academic_local_use_attested": True,
            "external_compute_transfer_requires_all_artifacts_cleared": True,
            "redistribution_requires_all_artifacts_cleared": True,
        }:
            raise ValueError("rights policy must remain fail closed")
        return self


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration root must be an object")
    return value


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_ledger(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{32})  (\S+)", line)
        if match is None:
            raise ValueError(f"CHECKSUMS line {line_number} is malformed")
        digest, relative_path = match.groups()
        if relative_path in entries:
            raise ValueError(f"CHECKSUMS repeats {relative_path!r}")
        entries[relative_path] = digest
    if not entries:
        raise ValueError("CHECKSUMS has no entries")
    return entries


def _role_paths(
    contract: QualificationContract, source_root: Path
) -> dict[str, Path]:
    resolved_root = source_root.resolve()
    result: dict[str, Path] = {}
    for asset in contract.artifacts:
        path = (resolved_root / Path(*PurePosixPath(asset.relative_path).parts)).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError("artifact path escapes the source root")
        result[asset.role] = path
    return result


def qualify_source(
    config_path: str | Path,
    source_root: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    config_file = Path(config_path).resolve()
    root = Path(source_root).resolve()
    output = Path(output_root).resolve()
    contract = QualificationContract.model_validate(_read_json(config_file))
    if output.exists():
        raise FileExistsError(f"qualification output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = _role_paths(contract, root)

    asset_receipts: list[dict[str, object]] = []
    for asset in contract.artifacts:
        path = paths[asset.role]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != asset.bytes or actual_sha256 != asset.sha256:
            raise ValueError(f"frozen bytes differ for {asset.artifact_id}")
        asset_receipts.append(
            {
                "artifact_id": asset.artifact_id,
                "role": asset.role,
                "relative_path": asset.relative_path,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "compression": asset.compression,
                "format": asset.format,
                "rights_status": asset.rights_status,
                "license_uri": asset.license_uri,
                "external_transfer_permitted": asset.external_transfer_permitted,
            }
        )

    ledger = _release_ledger(paths["release_checksum_ledger"])
    assets_by_role = {asset.role: asset for asset in contract.artifacts}
    ledger_verified = 0
    for asset in contract.artifacts:
        if asset.md5_in_release_ledger is None:
            continue
        ledger_digest = ledger.get(asset.relative_path)
        actual_md5 = _md5_file(paths[asset.role])
        if ledger_digest != asset.md5_in_release_ledger or actual_md5 != ledger_digest:
            raise ValueError(f"release MD5 binding failed for {asset.artifact_id}")
        ledger_verified += 1

    letter = paths["release_letter"].read_text(encoding="utf-8")
    if "New release of WormBase WS298" not in letter:
        raise ValueError("release letter does not identify WS298")
    if f"C. elegans gene data ({contract.release_letter_expected_gene_total} genes in total)" not in letter:
        raise ValueError("release letter gene census differs from contract")

    genes = read_gene_ids(paths["gene_identifiers"])
    phenotype = read_gaf(
        paths["phenotype_associations"],
        expected_aspect="P",
        expected_object_prefix="WBPhenotype:",
    )
    development = read_gaf(
        paths["development_associations"],
        expected_aspect="L",
        expected_object_prefix="WBls:",
        allowed_qualifiers=frozenset({"", "Anatomy_term"}),
    )
    phenotype_obo = read_obo(
        paths["phenotype_ontology"],
        expected_ontology="wbphenotype",
        expected_prefix="WBPhenotype:",
    )
    development_obo = read_obo(
        paths["development_ontology"],
        expected_ontology="wbls/wbls-simple",
        expected_prefix="WBls:",
    )

    for name, table in (("phenotype", phenotype), ("development", development)):
        required = f"!date-generated: {contract.required_gaf_date_generated}"
        if required not in table.headers:
            raise ValueError(f"{name} GAF date differs from contract")
    ontology_versions = {
        "phenotype_ontology": phenotype_obo.headers.get("data-version", (None,))[0],
        "development_ontology": development_obo.headers.get("data-version", (None,))[0],
    }
    if ontology_versions != contract.required_ontology_data_versions:
        raise ValueError("ontology data versions differ from contract")

    gene_ids = {record.gene_id for record in genes}
    missing_genes = sorted(
        {
            record.gene_id
            for table in (phenotype, development)
            for record in table.records
            if record.gene_id not in gene_ids
        }
    )
    missing_gene_set = set(missing_genes)
    missing_gene_rows = sum(
        record.gene_id in missing_gene_set
        for table in (phenotype, development)
        for record in table.records
    )
    missing_phenotypes = sorted(
        {
            record.object_id
            for record in phenotype.records
            if record.object_id not in phenotype_obo.accepted_ids
        }
    )
    missing_development = sorted(
        {
            record.object_id
            for record in development.records
            if record.object_id not in development_obo.accepted_ids
        }
    )
    ambiguous_phenotypes = sorted(
        {
            record.object_id
            for record in phenotype.records
            if record.object_id in phenotype_obo.ambiguous_ids
        }
    )
    ambiguous_development = sorted(
        {
            record.object_id
            for record in development.records
            if record.object_id in development_obo.ambiguous_ids
        }
    )
    phenotype_profile = profile_table(phenotype)
    development_profile = profile_table(development)
    actual_profile = {
        "gene_rows": len(genes),
        "phenotype_association_rows": phenotype_profile["data_rows"],
        "phenotype_unique_genes": phenotype_profile["unique_genes"],
        "phenotype_unique_terms": phenotype_profile["unique_objects"],
        "phenotype_negated_rows": phenotype_profile["negated_rows"],
        "phenotype_non_not_qualifier_rows": sum(
            count
            for qualifier, count in phenotype_profile["qualifier_counts"].items()
            if qualifier not in {EMPTY_GAF_QUALIFIER_RECEIPT_TOKEN, "NOT"}
        ),
        "development_association_rows": development_profile["data_rows"],
        "development_unique_genes": development_profile["unique_genes"],
        "development_unique_terms": development_profile["unique_objects"],
        "development_negated_rows": development_profile["negated_rows"],
        "development_anatomy_term_qualified_rows": development_profile[
            "qualifier_counts"
        ].get("Anatomy_term", 0),
        "phenotype_ontology_terms": len(phenotype_obo.terms),
        "phenotype_ontology_obsolete_terms": sum(
            term.obsolete for term in phenotype_obo.terms.values()
        ),
        "phenotype_ontology_alternate_ids": sum(
            len(term.alt_ids) for term in phenotype_obo.terms.values()
        ),
        "phenotype_ontology_ambiguous_ids": len(phenotype_obo.ambiguous_ids),
        "phenotype_ontology_nameless_obsolete_terms": sum(
            term.name is None for term in phenotype_obo.terms.values()
        ),
        "development_ontology_terms": len(development_obo.terms),
        "development_ontology_obsolete_terms": sum(
            term.obsolete for term in development_obo.terms.values()
        ),
        "development_ontology_alternate_ids": sum(
            len(term.alt_ids) for term in development_obo.terms.values()
        ),
        "development_ontology_ambiguous_ids": len(development_obo.ambiguous_ids),
        "development_ontology_nameless_obsolete_terms": sum(
            term.name is None for term in development_obo.terms.values()
        ),
        "unresolved_gene_references": len(missing_genes),
        "unresolved_gene_rows": missing_gene_rows,
        "unresolved_phenotype_references": len(missing_phenotypes),
        "unresolved_development_references": len(missing_development),
        "ambiguous_referenced_phenotype_ids": len(ambiguous_phenotypes),
        "ambiguous_referenced_development_ids": len(ambiguous_development),
    }
    if actual_profile != contract.expected_profile.model_dump(mode="json"):
        raise ValueError(
            "source profile differs from frozen expectation: "
            f"actual={actual_profile!r}"
        )

    review_required = sorted(
        asset.artifact_id
        for asset in contract.artifacts
        if asset.rights_status == "review_required"
    )
    config_bytes = config_file.read_bytes()
    receipt: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "analysis_id": contract.analysis_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": contract.classification,
        "qualification_status": "qualified_source_inventory_with_adapter_blockers",
        "provider": contract.provider,
        "release": contract.release,
        "taxon": contract.taxon,
        "scope": contract.scope,
        "source_root_id": contract.source_root_id,
        "config": {
            "filename": config_file.name,
            "bytes": len(config_bytes),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
        "assets": sorted(asset_receipts, key=lambda item: str(item["artifact_id"])),
        "release_ledger": {
            "entries": len(ledger),
            "selected_assets_verified": ledger_verified,
            "checksum_algorithm": "MD5_upstream_plus_SHA256_local",
        },
        "schemas": {
            "gene_identifiers": "headerless_CSV_exactly_6_columns",
            "phenotype_associations": "GAF_2.0_exactly_17_columns_aspect_P",
            "empty_gaf_qualifier_receipt_token": (
                contract.empty_gaf_qualifier_receipt_token
            ),
            "development_associations": "GAF_2.0_exactly_17_columns_aspect_L",
            "ontologies": (
                "OBO_1.2_primary_ids_unique_alias_ambiguity_receipted_"
                "referenced_ambiguity_rejected"
            ),
        },
        "profile": actual_profile,
        "gene_profile": {
            "status_counts": dict(
                sorted(Counter(record.status for record in genes).items())
            ),
            "gene_type_counts": dict(
                sorted(Counter(record.gene_type for record in genes).items())
            ),
        },
        "phenotype_gaf_profile": phenotype_profile,
        "development_gaf_profile": development_profile,
        "ontology_profile": {
            "phenotype": {
                "data_version": ontology_versions["phenotype_ontology"],
                "terms": len(phenotype_obo.terms),
                "alternate_ids": sum(
                    len(term.alt_ids) for term in phenotype_obo.terms.values()
                ),
                "obsolete_terms": actual_profile[
                    "phenotype_ontology_obsolete_terms"
                ],
                "nameless_obsolete_terms": sum(
                    term.name is None for term in phenotype_obo.terms.values()
                ),
                "ambiguous_primary_or_alternate_ids": len(
                    phenotype_obo.ambiguous_ids
                ),
                "embedded_license": assets_by_role[
                    "phenotype_ontology"
                ].license_uri,
            },
            "development": {
                "data_version": ontology_versions["development_ontology"],
                "terms": len(development_obo.terms),
                "alternate_ids": sum(
                    len(term.alt_ids) for term in development_obo.terms.values()
                ),
                "obsolete_terms": actual_profile[
                    "development_ontology_obsolete_terms"
                ],
                "nameless_obsolete_terms": sum(
                    term.name is None for term in development_obo.terms.values()
                ),
                "ambiguous_primary_or_alternate_ids": len(
                    development_obo.ambiguous_ids
                ),
                "embedded_license": assets_by_role[
                    "development_ontology"
                ].license_uri,
            },
        },
        "referential_integrity": {
            "unresolved_gene_ids": missing_genes,
            "unresolved_phenotype_ids": missing_phenotypes,
            "unresolved_development_ids": missing_development,
            "ambiguous_referenced_phenotype_ids": ambiguous_phenotypes,
            "ambiguous_referenced_development_ids": ambiguous_development,
            "normalization_permitted": not (
                missing_genes
                or missing_phenotypes
                or missing_development
                or ambiguous_phenotypes
                or ambiguous_development
            ),
        },
        "rights_gate": {
            "controlled_noncommercial_academic_local_use_attested": True,
            "review_required_artifacts": review_required,
            "external_compute_transfer_permitted": not review_required,
            "redistribution_permitted": not review_required,
            "status": (
                "blocked_pending_artifact_level_rights_review"
                if review_required
                else "cleared"
            ),
        },
        "adapter_boundary": {
            "offline_adapter_implemented": True,
            "rights_safe_synthetic_fixture_required_before_real_build": True,
            "generic_multi_artifact_normalization_binding_implemented": False,
            "reviewed_WormBase_infores_mapping_implemented": False,
            "reviewed_GAF_evidence_to_ECO_mapping_implemented": False,
            "real_normalization_status": (
                "blocked_on_unresolved_subject_identity"
                if missing_genes
                else "source_identity_gate_passed"
            ),
        },
        "claim_boundary": contract.claim_boundary.model_dump(mode="json"),
        "biological_claims_permitted": False,
        "models_executed": False,
    }

    staging = Path(tempfile.mkdtemp(prefix=".ws298-source_qualification-", dir=output.parent))
    try:
        receipt_path = staging / "source_qualification_receipt.json"
        receipt_path.write_bytes(_canonical_bytes(receipt))
        receipt_sha256 = sha256_file(receipt_path)
        (staging / "SHA256SUMS.txt").write_text(
            f"{receipt_sha256}  source_qualification_receipt.json\n",
            encoding="utf-8",
            newline="\n",
        )
        (staging / "SUCCESS").write_text(
            "qualified\n", encoding="utf-8", newline="\n"
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = qualify_source(args.config, args.source_root, args.output_root)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
