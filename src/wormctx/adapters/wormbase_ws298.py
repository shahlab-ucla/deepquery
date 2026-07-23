"""Bounded offline adapter for the WormBase WS298 source qualification source profile.

The adapter intentionally accepts only five small, release-pinned biological
artifacts: gene identifiers, phenotype and developmental-stage GAF files, and
their two OBO ontologies.  Acquisition and rights decisions remain outside the
adapter.  The qualification lane in :mod:`wormctx.poc.wormbase_ws298_source_qualification`
verifies those boundaries before any real normalization is attempted.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Iterator, Mapping, TextIO

from ..models import (
    BiolinkAgentType,
    BiolinkKnowledgeLevel,
    ContextGap,
    ContextualObservation,
    DatasetProfile,
    EvidenceDirection,
    EvidenceLine,
    ExperimentalContext,
    Interpretation,
    MappingStatus,
    MissingContextReason,
    NamedReference,
    ObservationStatus,
    RecordOrigin,
    SourceSnapshot,
)
from .base import Severity, SnapshotContext, ValidationIssue, ValidationReport


ADAPTER_VERSION = "0.1.0"
EMPTY_GAF_QUALIFIER_RECEIPT_TOKEN = "UNQUALIFIED"
ARTIFACT_IDS = {
    "gene_ids": "gene-ids",
    "phenotype_gaf": "phenotype-associations",
    "development_gaf": "development-associations",
    "phenotype_obo": "phenotype-ontology",
    "development_obo": "development-ontology",
}
@dataclass(frozen=True)
class GeneRecord:
    taxon_id: str
    gene_id: str
    public_name: str
    sequence_name: str
    status: str
    gene_type: str


@dataclass(frozen=True)
class GafRecord:
    line_number: int
    columns: tuple[str, ...]

    @property
    def gene_id(self) -> str:
        return self.columns[1]

    @property
    def gene_label(self) -> str:
        return self.columns[2]

    @property
    def qualifier(self) -> str:
        return self.columns[3]

    @property
    def object_id(self) -> str:
        return self.columns[4]

    @property
    def references(self) -> tuple[str, ...]:
        return tuple(item for item in self.columns[5].split("|") if item)

    @property
    def evidence_code(self) -> str:
        return self.columns[6]

    @property
    def aspect(self) -> str:
        return self.columns[8]

    @property
    def taxon(self) -> str:
        return self.columns[12]


@dataclass(frozen=True)
class GafTable:
    headers: tuple[str, ...]
    records: tuple[GafRecord, ...]


@dataclass(frozen=True)
class OboTerm:
    identifier: str
    name: str | None
    alt_ids: tuple[str, ...]
    obsolete: bool


@dataclass(frozen=True)
class OboOntology:
    headers: Mapping[str, tuple[str, ...]]
    terms: Mapping[str, OboTerm]

    @cached_property
    def accepted_ids(self) -> frozenset[str]:
        return frozenset(set(self.terms) | {
            alt_id for term in self.terms.values() for alt_id in term.alt_ids
        })

    @cached_property
    def alt_id_owners(self) -> dict[str, tuple[str, ...]]:
        owners: dict[str, list[str]] = {}
        for term in self.terms.values():
            for alt_id in term.alt_ids:
                owners.setdefault(alt_id, []).append(term.identifier)
        return {
            alt_id: tuple(sorted(set(term_ids)))
            for alt_id, term_ids in owners.items()
        }

    @cached_property
    def ambiguous_ids(self) -> frozenset[str]:
        return frozenset({
            alt_id
            for alt_id, owners in self.alt_id_owners.items()
            if len(owners) != 1 or alt_id in self.terms
        })

    def resolve_term(self, identifier: str) -> OboTerm:
        if identifier in self.ambiguous_ids:
            raise ValueError(f"ontology identifier is ambiguous: {identifier}")
        if identifier in self.terms:
            return self.terms[identifier]
        owners = self.alt_id_owners.get(identifier, ())
        if len(owners) != 1:
            raise KeyError(identifier)
        return self.terms[owners[0]]


@dataclass(frozen=True)
class WormBaseWs298Inputs:
    gene_ids: Path
    phenotype_gaf: Path
    development_gaf: Path
    phenotype_obo: Path
    development_obo: Path
    phenotype_artifact_name: str | None = None
    development_artifact_name: str | None = None
    source_id: str = "wormbase-ws298-source_qualification"
    source_release: str = "WS298"
    source_url: str = (
        "https://ftp.ebi.ac.uk/pub/databases/wormbase/releases/WS298/"
    )
    license_uri: str | None = None
    excluded_subjects: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )


def _open_text(path: Path) -> TextIO:
    # Snapshot blobs are content-addressed and therefore have no source suffix.
    # Detect gzip bytes as well as accepting ordinary named fixture files.
    with path.open("rb") as probe:
        is_gzip = probe.read(2) == b"\x1f\x8b"
    if path.suffix == ".gz" or is_gzip:
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="r", encoding="utf-8", newline="")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gene_ids(path: Path) -> tuple[GeneRecord, ...]:
    records: list[GeneRecord] = []
    with _open_text(path) as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 6:
                raise ValueError(
                    f"geneIDs row {line_number} has {len(row)} columns; expected 6"
                )
            record = GeneRecord(*row)
            if record.taxon_id != "6239":
                raise ValueError(f"geneIDs row {line_number} has unexpected taxon")
            if not (
                record.gene_id.startswith("WBGene")
                and len(record.gene_id) == 14
                and record.gene_id[6:].isdigit()
            ):
                raise ValueError(f"geneIDs row {line_number} has invalid gene identifier")
            records.append(record)
    if not records:
        raise ValueError("geneIDs file has no records")
    identifiers = [record.gene_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("geneIDs file contains duplicate WormBase gene identifiers")
    return tuple(records)


def read_gaf(
    path: Path,
    *,
    expected_aspect: str,
    expected_object_prefix: str,
    allowed_qualifiers: frozenset[str] = frozenset({"", "NOT"}),
) -> GafTable:
    headers: list[str] = []
    records: list[GafRecord] = []
    with _open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("!"):
                headers.append(line)
                continue
            columns = tuple(line.split("\t"))
            if len(columns) != 17:
                raise ValueError(
                    f"GAF row {line_number} has {len(columns)} columns; expected 17"
                )
            record = GafRecord(line_number=line_number, columns=columns)
            if columns[0] != "WB" or record.taxon != "taxon:6239":
                raise ValueError(f"GAF row {line_number} has unexpected source or taxon")
            if not record.gene_id.startswith("WBGene"):
                raise ValueError(f"GAF row {line_number} has invalid gene identifier")
            if record.aspect != expected_aspect:
                raise ValueError(f"GAF row {line_number} has unexpected aspect")
            if not record.object_id.startswith(expected_object_prefix):
                raise ValueError(f"GAF row {line_number} has unexpected object prefix")
            if record.qualifier not in allowed_qualifiers:
                raise ValueError(
                    f"GAF row {line_number} has an unsupported qualifier: "
                    f"{record.qualifier!r}"
                )
            records.append(record)
    required_headers = {
        "!gaf-version: 2.0",
        "!generated-by: WormBase",
        "!project-release: WS298",
    }
    missing = required_headers - set(headers)
    if missing:
        raise ValueError(f"GAF headers are missing: {sorted(missing)}")
    if not records:
        raise ValueError("GAF file has no records")
    return GafTable(headers=tuple(headers), records=tuple(records))


def read_obo(path: Path, *, expected_ontology: str, expected_prefix: str) -> OboOntology:
    headers: dict[str, list[str]] = {}
    term_rows: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] | None = None
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line == "[Term]":
                current = {}
                term_rows.append(current)
                continue
            if line.startswith("["):
                current = None
                continue
            if not line or line.startswith("!") or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            target = headers if current is None else current
            target.setdefault(key, []).append(value)
    if expected_ontology not in headers.get("ontology", []):
        raise ValueError(f"OBO ontology header does not contain {expected_ontology!r}")
    terms: dict[str, OboTerm] = {}
    for index, row in enumerate(term_rows, start=1):
        identifiers = row.get("id", [])
        names = row.get("name", [])
        obsolete = row.get("is_obsolete", []) == ["true"]
        if len(identifiers) != 1 or len(names) > 1 or (not names and not obsolete):
            raise ValueError(
                f"OBO term stanza {index} must have one id and, unless obsolete, one name"
            )
        identifier = identifiers[0]
        if not identifier.startswith(expected_prefix):
            raise ValueError(f"OBO term stanza {index} has unexpected identifier prefix")
        if identifier in terms:
            raise ValueError(f"OBO contains duplicate term identifier {identifier}")
        terms[identifier] = OboTerm(
            identifier=identifier,
            name=names[0] if names else None,
            alt_ids=tuple(dict.fromkeys(row.get("alt_id", []))),
            obsolete=obsolete,
        )
    if not terms:
        raise ValueError("OBO file has no term stanzas")
    return OboOntology(
        headers={key: tuple(values) for key, values in headers.items()},
        terms=terms,
    )


class WormBaseWs298Adapter:
    """Normalize the bounded GAF profile without fetching or widening scope."""

    source_id = "wormbase-ws298-source_qualification"
    adapter_version = ADAPTER_VERSION

    def validate_raw(
        self, snapshot: WormBaseWs298Inputs | SnapshotContext
    ) -> ValidationReport:
        inputs = self._inputs(snapshot)
        report = ValidationReport(
            source_id=inputs.source_id,
            snapshot_id=(
                snapshot.snapshot_id if isinstance(snapshot, SnapshotContext) else None
            ),
            adapter_version=self.adapter_version,
            schema_version="wormctx-wormbase-ws298-source_qualification-validation-1.0",
        )
        try:
            genes = read_gene_ids(inputs.gene_ids)
            phenotype = read_gaf(
                inputs.phenotype_gaf,
                expected_aspect="P",
                expected_object_prefix="WBPhenotype:",
            )
            development = read_gaf(
                inputs.development_gaf,
                expected_aspect="L",
                expected_object_prefix="WBls:",
                allowed_qualifiers=frozenset({"", "Anatomy_term"}),
            )
            phenotype_obo = read_obo(
                inputs.phenotype_obo,
                expected_ontology="wbphenotype",
                expected_prefix="WBPhenotype:",
            )
            development_obo = read_obo(
                inputs.development_obo,
                expected_ontology="wbls/wbls-simple",
                expected_prefix="WBls:",
            )
            gene_ids = {record.gene_id for record in genes}
            all_records = phenotype.records + development.records
            for subject_id, disposition in inputs.excluded_subjects.items():
                matching = [
                    record for record in all_records if record.gene_id == subject_id
                ]
                if any(
                    record.gene_label != disposition["expected_label"]
                    for record in matching
                ):
                    raise ValueError(
                        f"frozen exclusion identity changed for {subject_id}"
                    )
                if inputs.source_id == self.source_id:
                    phenotype_matching = [
                        record
                        for record in phenotype.records
                        if record.gene_id == subject_id
                    ]
                    development_matching = [
                        record
                        for record in development.records
                        if record.gene_id == subject_id
                    ]
                    if (
                        len(phenotype_matching)
                        != disposition["expected_phenotype_rows"]
                        or development_matching
                    ):
                        raise ValueError(
                            f"frozen exclusion row census changed for {subject_id}"
                        )
            missing_genes = {
                record.gene_id
                for table in (phenotype, development)
                for record in table.records
                if record.gene_id not in gene_ids
                and record.gene_id not in inputs.excluded_subjects
            }
            missing_phenotypes = {
                record.object_id
                for record in phenotype.records
                if record.object_id not in phenotype_obo.accepted_ids
            }
            missing_stages = {
                record.object_id
                for record in development.records
                if record.object_id not in development_obo.accepted_ids
            }
            ambiguous_phenotypes = {
                record.object_id
                for record in phenotype.records
                if record.object_id in phenotype_obo.ambiguous_ids
            }
            ambiguous_stages = {
                record.object_id
                for record in development.records
                if record.object_id in development_obo.ambiguous_ids
            }
            if (
                missing_genes
                or missing_phenotypes
                or missing_stages
                or ambiguous_phenotypes
                or ambiguous_stages
            ):
                raise ValueError(
                    "cross-file references are unresolved: "
                    f"genes={len(missing_genes)}, phenotypes={len(missing_phenotypes)}, "
                    f"stages={len(missing_stages)}, "
                    f"ambiguous_phenotypes={len(ambiguous_phenotypes)}, "
                    f"ambiguous_stages={len(ambiguous_stages)}"
                )
            report.checked_records = len(genes) + len(phenotype.records) + len(
                development.records
            )
        except (OSError, UnicodeError, ValueError, csv.Error) as exc:
            report.add(
                ValidationIssue(
                    severity=Severity.error,
                    code="ws298_source_qualification_source_validation",
                    message=str(exc),
                )
            )
        return report

    def iter_observations(
        self, snapshot: WormBaseWs298Inputs | SnapshotContext
    ) -> Iterator[ContextualObservation]:
        inputs = self._inputs(snapshot)
        report = self.validate_raw(inputs)
        if not report.valid:
            raise ValueError(report.issues[0].message)
        genes = {record.gene_id: record for record in read_gene_ids(inputs.gene_ids)}
        phenotype_obo = read_obo(
            inputs.phenotype_obo,
            expected_ontology="wbphenotype",
            expected_prefix="WBPhenotype:",
        )
        development_obo = read_obo(
            inputs.development_obo,
            expected_ontology="wbls/wbls-simple",
            expected_prefix="WBls:",
        )
        tables = (
            (
                "phenotype",
                read_gaf(
                    inputs.phenotype_gaf,
                    expected_aspect="P",
                    expected_object_prefix="WBPhenotype:",
                ),
                phenotype_obo,
                inputs.phenotype_gaf,
                inputs.phenotype_artifact_name or inputs.phenotype_gaf.name,
            ),
            (
                "development",
                read_gaf(
                    inputs.development_gaf,
                    expected_aspect="L",
                    expected_object_prefix="WBls:",
                    allowed_qualifiers=frozenset({"", "Anatomy_term"}),
                ),
                development_obo,
                inputs.development_gaf,
                inputs.development_artifact_name or inputs.development_gaf.name,
            ),
        )
        for kind, table, ontology, artifact_path, artifact_name in tables:
            artifact_sha256 = sha256_file(artifact_path)
            for record in table.records:
                if record.gene_id in inputs.excluded_subjects:
                    continue
                yield self._observation(
                    inputs,
                    kind,
                    record,
                    genes[record.gene_id],
                    ontology,
                    artifact_path,
                    artifact_name,
                    artifact_sha256,
                )

    def dataset_profiles(
        self, snapshot: WormBaseWs298Inputs | SnapshotContext
    ) -> Iterator[DatasetProfile]:
        inputs = self._inputs(snapshot)
        phenotype = read_gaf(
            inputs.phenotype_gaf,
            expected_aspect="P",
            expected_object_prefix="WBPhenotype:",
        )
        development = read_gaf(
            inputs.development_gaf,
            expected_aspect="L",
            expected_object_prefix="WBls:",
            allowed_qualifiers=frozenset({"", "Anatomy_term"}),
        )
        yield DatasetProfile(
            dataset_id="wormctx:dataset-wormbase-ws298-source_qualification-phenotype",
            source_id=inputs.source_id,
            source_release=inputs.source_release,
            source_artifact=(
                inputs.phenotype_artifact_name or inputs.phenotype_gaf.name
            ),
            taxon="NCBITaxon:6239",
            modality=["curated_phenotype_association"],
            observation_count=sum(
                record.gene_id not in inputs.excluded_subjects
                for record in phenotype.records
            ),
            direct_or_derived="direct",
            missing_context={"life_stage": "not_reported"},
        )
        yield DatasetProfile(
            dataset_id="wormctx:dataset-wormbase-ws298-source_qualification-development",
            source_id=inputs.source_id,
            source_release=inputs.source_release,
            source_artifact=(
                inputs.development_artifact_name or inputs.development_gaf.name
            ),
            taxon="NCBITaxon:6239",
            life_stage=sorted({record.object_id for record in development.records}),
            modality=["curated_developmental_stage_association"],
            observation_count=sum(
                record.gene_id not in inputs.excluded_subjects
                for record in development.records
            ),
            direct_or_derived="direct",
            missing_context={"assay": "not_reported"},
        )

    @staticmethod
    def _inputs(
        snapshot: WormBaseWs298Inputs | SnapshotContext,
    ) -> WormBaseWs298Inputs:
        if isinstance(snapshot, WormBaseWs298Inputs):
            return snapshot
        by_key = {
            key: snapshot.artifact_path(artifact_id)
            for key, artifact_id in ARTIFACT_IDS.items()
        }
        receipts = {item.artifact_id: item for item in snapshot.receipts}
        return WormBaseWs298Inputs(
            **by_key,
            phenotype_artifact_name=receipts[
                ARTIFACT_IDS["phenotype_gaf"]
            ].filename,
            development_artifact_name=receipts[
                ARTIFACT_IDS["development_gaf"]
            ].filename,
            source_id=snapshot.manifest.source_id,
            source_release=snapshot.manifest.release,
            source_url=str(snapshot.manifest.landing_page or "https://wormbase.org/"),
            license_uri=(
                str(snapshot.manifest.license_uri)
                if snapshot.manifest.license_uri is not None
                else None
            ),
        )

    @staticmethod
    def _observation(
        inputs: WormBaseWs298Inputs,
        kind: str,
        record: GafRecord,
        gene: GeneRecord,
        ontology: OboOntology,
        artifact_path: Path,
        artifact_name: str,
        artifact_sha256: str,
    ) -> ContextualObservation:
        row_key = "\t".join(record.columns)
        identity = hashlib.sha256(
            f"{kind}\n{record.line_number}\n{row_key}".encode("utf-8")
        ).hexdigest()[:24]
        object_term = ontology.resolve_term(record.object_id)
        negated = record.qualifier == "NOT"
        if kind == "phenotype":
            predicate = "biolink:has_phenotype"
            object_categories = ["biolink:PhenotypicFeature"]
            modality = ["curated_phenotype_association"]
            life_stage: list[NamedReference] = []
            gaps = WormBaseWs298Adapter._context_gaps(life_stage_missing=True)
        else:
            predicate = "wormctx:observed_during_life_stage"
            object_categories = ["biolink:NamedThing"]
            modality = ["curated_developmental_stage_association"]
            life_stage = [
                NamedReference(id=record.object_id, label=object_term.name)
            ]
            gaps = WormBaseWs298Adapter._context_gaps(life_stage_missing=False)
        publications = [
            NamedReference(id=reference)
            for reference in record.references
            if reference.startswith(("WB_REF:", "PMID:"))
        ]
        return ContextualObservation(
            id=f"wormctx:ws298-source_qualification-{kind}-{identity}",
            subject=NamedReference(
                id=f"WB:{record.gene_id}",
                label=gene.public_name or gene.sequence_name,
                categories=["biolink:Gene"],
                source_value=record.gene_id,
            ),
            predicate=predicate,
            object=NamedReference(
                id=record.object_id,
                label=object_term.name,
                categories=object_categories,
            ),
            interpretation=Interpretation.observation,
            observation_status=(
                ObservationStatus.explicit_negative
                if negated
                else ObservationStatus.positive
            ),
            proposition_negated=negated,
            context=ExperimentalContext(
                organism_taxon=NamedReference(
                    id="NCBITaxon:6239", label="Caenorhabditis elegans"
                ),
                life_stage=life_stage,
                modality=modality,
                context_gaps=gaps,
            ),
            evidence_lines=[
                EvidenceLine(
                    id=f"wormctx:evidence-ws298-source_qualification-{kind}-{identity}",
                    evidence_type=NamedReference(
                        id="ECO:0000006",
                        label="experimental evidence",
                        source_value=record.evidence_code,
                        mapping_record_id="wormctx:ws298-source_qualification-broad-evidence-map",
                        mapping_status=MappingStatus.mapped_broad,
                    ),
                    direction=EvidenceDirection.supports,
                    source_record_id=(
                        f"{artifact_name}:line:{record.line_number}"
                    ),
                    publications=publications,
                    notes=(
                        "Source GAF evidence code is preserved in source_value; "
                        "ECO:0000006 is a deliberately broad provisional mapping."
                    ),
                )
            ],
            provenance=SourceSnapshot(
                source_id=inputs.source_id,
                source_release=inputs.source_release,
                source_artifact=artifact_name,
                source_url=inputs.source_url,
                checksum_sha256=artifact_sha256,
                license_uri=inputs.license_uri,
                transformation_activity="wormctx:normalize-ws298-source_qualification",
                transformer_version=ADAPTER_VERSION,
            ),
            record_origin=RecordOrigin.curated,
            biolink_knowledge_level=BiolinkKnowledgeLevel.knowledge_assertion,
            biolink_agent_type=BiolinkAgentType.manual_agent,
            qualifiers=(
                [
                    NamedReference(
                        id=f"wormctx:gaf-qualifier-{record.qualifier}",
                        label=f"source GAF qualifier {record.qualifier}",
                        source_value=record.qualifier,
                    )
                ]
                if record.qualifier not in {"", "NOT"}
                else []
            ),
        )

    @staticmethod
    def _context_gaps(*, life_stage_missing: bool) -> list[ContextGap]:
        dimensions = [
            "genetic_background",
            "anatomy",
            "intervention",
            "assay",
            "environment",
            "readout_resolution",
            "absolute_time",
        ]
        if life_stage_missing:
            dimensions.append("life_stage")
        return [
            ContextGap(
                dimension=dimension,
                reason=MissingContextReason.not_reported,
                notes="Not represented in the bounded WS298 GAF source row.",
            )
            for dimension in dimensions
        ]


def profile_table(table: GafTable) -> dict[str, object]:
    """Return deterministic source-profile metrics used by source qualification qualification."""

    return {
        "data_rows": len(table.records),
        "unique_genes": len({record.gene_id for record in table.records}),
        "unique_objects": len({record.object_id for record in table.records}),
        "negated_rows": sum(record.qualifier == "NOT" for record in table.records),
        "qualifier_counts": dict(
            sorted(
                Counter(
                    record.qualifier or EMPTY_GAF_QUALIFIER_RECEIPT_TOKEN
                    for record in table.records
                ).items()
            )
        ),
        "evidence_code_counts": dict(
            sorted(Counter(record.evidence_code for record in table.records).items())
        ),
        "date_counts": dict(
            sorted(Counter(record.columns[13] for record in table.records).items())
        ),
    }
