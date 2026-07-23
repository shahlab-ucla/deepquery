"""Validated runtime models for the LinkML-aligned canonical records."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


CURIE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*:[^\s]+$")
SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
SAFE_PATH_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class MappingStatus(str, Enum):
    asserted = "asserted"
    mapped_exact = "mapped_exact"
    mapped_broad = "mapped_broad"
    mapped_narrow = "mapped_narrow"
    mapped_related = "mapped_related"
    unmapped = "unmapped"


class Reportedness(str, Enum):
    reported = "reported"
    not_reported = "not_reported"
    not_applicable = "not_applicable"
    ambiguous = "ambiguous"
    explicitly_absent = "explicitly_absent"


class NamedReference(StrictModel):
    id: str
    label: str | None = None
    categories: list[str] = Field(default_factory=list)
    source_value: str | None = None
    mapping_record_id: str | None = None
    mapping_status: MappingStatus = MappingStatus.asserted
    reportedness: Reportedness = Reportedness.reported

    @model_validator(mode="after")
    def validate_identifier(self) -> "NamedReference":
        if not CURIE_PATTERN.match(self.id):
            raise ValueError(f"identifier must be a CURIE or compact URI: {self.id!r}")
        for category in self.categories:
            if not CURIE_PATTERN.match(category):
                raise ValueError(f"category must be a CURIE: {category!r}")
        if self.mapping_record_id and not CURIE_PATTERN.match(self.mapping_record_id):
            raise ValueError("mapping_record_id must be a CURIE")
        return self


class QuantitativeValue(StrictModel):
    value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    comparator: str | None = None
    unit: NamedReference | None = None
    text: str | None = None

    @model_validator(mode="after")
    def require_content(self) -> "QuantitativeValue":
        if all(
            part is None
            for part in (self.value, self.lower_bound, self.upper_bound, self.text)
        ):
            raise ValueError("quantitative value requires a number, bound, or text")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("quantitative lower_bound cannot exceed upper_bound")
        return self


class ResultType(str, Enum):
    qualitative = "qualitative"
    quantitative = "quantitative"
    categorical = "categorical"
    trajectory = "trajectory"
    image_derived = "image_derived"
    textual_claim = "textual_claim"
    mixed = "mixed"


class ResultAssetReference(StrictModel):
    """Immutable reference to a non-inline result payload.

    The digest is mandatory: an observation may point at a table, array, image,
    or trajectory without silently depending on mutable bytes at ``uri``.
    """

    id: str
    uri: str
    checksum_sha256: str
    media_type: str
    byte_size: int | None = None
    role: str | None = None

    @model_validator(mode="after")
    def validate_asset(self) -> "ResultAssetReference":
        if not CURIE_PATTERN.match(self.id):
            raise ValueError("result asset id must be a CURIE")
        if not self.uri or ":" not in self.uri or any(char.isspace() for char in self.uri):
            raise ValueError("result asset uri must be an absolute URI")
        if not SHA256_PATTERN.match(self.checksum_sha256):
            raise ValueError("checksum_sha256 must contain exactly 64 hexadecimal characters")
        sha256_urn_prefix = "urn:sha256:"
        if self.uri.lower().startswith(sha256_urn_prefix):
            uri_digest = self.uri[len(sha256_urn_prefix) :]
            if (
                not SHA256_PATTERN.fullmatch(uri_digest)
                or uri_digest.lower() != self.checksum_sha256.lower()
            ):
                raise ValueError(
                    "a sha256 URN digest must equal checksum_sha256"
                )
        if self.byte_size is not None and self.byte_size < 0:
            raise ValueError("result asset byte_size cannot be negative")
        return self


class ComparisonGroupRole(str, Enum):
    case = "case"
    control = "control"
    reference = "reference"
    baseline = "baseline"
    exposure = "exposure"
    other = "other"


class ComparisonGroup(StrictModel):
    """A named arm or cohort used to interpret a measurement or effect."""

    id: str
    label: str
    role: ComparisonGroupRole
    size: int | None = None
    members: list[NamedReference] = Field(default_factory=list)
    genetic_background: list[NamedReference] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_group(self) -> "ComparisonGroup":
        if not CURIE_PATTERN.match(self.id):
            raise ValueError("comparison group id must be a CURIE")
        if self.size is not None and self.size < 1:
            raise ValueError("comparison group size must be at least one when reported")
        return self


class Measurement(StrictModel):
    """A reported scalar, interval, or textual measurement of one feature."""

    id: str
    measured_feature: NamedReference
    value: QuantitativeValue
    statistic: NamedReference | None = None
    uncertainty: QuantitativeValue | None = None
    uncertainty_type: NamedReference | None = None
    confidence_level: float | None = None
    p_value: float | None = None
    penetrance: QuantitativeValue | None = None
    sample_size: int | None = None
    biological_replicates: int | None = None
    technical_replicates: int | None = None
    measurement_time: QuantitativeValue | None = None
    comparison_group_ids: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_measurement(self) -> "Measurement":
        if not CURIE_PATTERN.match(self.id):
            raise ValueError("measurement id must be a CURIE")
        for field_name in ("sample_size", "biological_replicates", "technical_replicates"):
            count = getattr(self, field_name)
            if count is not None and count < 1:
                raise ValueError(f"{field_name} must be at least one when reported")
        if len(self.asset_ids) != len(set(self.asset_ids)):
            raise ValueError("measurement asset_ids must be unique")
        if any(not CURIE_PATTERN.match(asset_id) for asset_id in self.asset_ids):
            raise ValueError("measurement asset_ids must be CURIEs")
        if len(self.comparison_group_ids) != len(set(self.comparison_group_ids)):
            raise ValueError("measurement comparison_group_ids must be unique")
        if any(
            not CURIE_PATTERN.match(group_id)
            for group_id in self.comparison_group_ids
        ):
            raise ValueError("measurement comparison_group_ids must be CURIEs")
        for field_name in ("confidence_level", "p_value"):
            value = getattr(self, field_name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between zero and one")
        if self.uncertainty is not None and self.uncertainty_type is None:
            raise ValueError("uncertainty requires an explicit uncertainty_type")
        return self


class ObservationResult(StrictModel):
    """The reported result, distinct from the semantic outcome in ``object``."""

    id: str
    result_type: ResultType
    summary: str | None = None
    comparison_groups: list[ComparisonGroup] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    asset_references: list[ResultAssetReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result(self) -> "ObservationResult":
        if not CURIE_PATTERN.match(self.id):
            raise ValueError("observation result id must be a CURIE")
        if not self.summary and not self.measurements and not self.asset_references:
            raise ValueError("observation result requires a summary, measurement, or asset")
        if self.result_type is ResultType.quantitative and not self.measurements:
            raise ValueError("a quantitative result requires at least one measurement")
        if self.result_type in {ResultType.trajectory, ResultType.image_derived} and not self.asset_references:
            raise ValueError(
                "trajectory and image-derived results require a checksummed result asset"
            )
        measurement_ids = [item.id for item in self.measurements]
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError("measurement ids must be unique within a result")
        asset_ids = [item.id for item in self.asset_references]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("result asset ids must be unique within a result")
        available_assets = set(asset_ids)
        group_ids = [item.id for item in self.comparison_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("comparison group ids must be unique within a result")
        available_groups = set(group_ids)
        for measurement in self.measurements:
            unknown = set(measurement.asset_ids) - available_assets
            if unknown:
                raise ValueError(
                    f"measurement {measurement.id!r} references unknown result assets: "
                    f"{sorted(unknown)!r}"
                )
            unknown_groups = set(measurement.comparison_group_ids) - available_groups
            if unknown_groups:
                raise ValueError(
                    f"measurement {measurement.id!r} references unknown comparison "
                    f"groups: {sorted(unknown_groups)!r}"
                )
        return self


class Intervention(StrictModel):
    intervention_type: str
    target: NamedReference | None = None
    reagent: NamedReference | None = None
    route: NamedReference | None = None
    dose: QuantitativeValue | None = None
    duration: QuantitativeValue | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_type(self) -> "Intervention":
        if not CURIE_PATTERN.match(self.intervention_type):
            raise ValueError("intervention_type must be a CURIE")
        return self


class ReadoutResolution(str, Enum):
    organism = "organism"
    tissue = "tissue"
    cell = "cell"
    subcellular = "subcellular"
    spatial = "spatial"
    temporal = "temporal"
    trajectory = "trajectory"


class MissingContextReason(str, Enum):
    not_reported = "not_reported"
    not_applicable = "not_applicable"
    ambiguous = "ambiguous"
    explicitly_absent = "explicitly_absent"
    not_available = "not_available"
    not_curated = "not_curated"
    not_yet_normalized = "not_yet_normalized"


KNOWN_CONTEXT_DIMENSIONS = {
    "organism_taxon",
    "genetic_background",
    "life_stage",
    "anatomy",
    "intervention",
    "assay",
    "environment",
    "modality",
    "readout_resolution",
    "absolute_time",
}


class ContextGap(StrictModel):
    dimension: str
    reason: MissingContextReason
    source_value: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_dimension(self) -> "ContextGap":
        if self.dimension not in KNOWN_CONTEXT_DIMENSIONS:
            raise ValueError(f"unknown context dimension: {self.dimension!r}")
        return self


class ExperimentalContext(StrictModel):
    organism_taxon: NamedReference
    genetic_background: list[NamedReference] = Field(default_factory=list)
    life_stage: list[NamedReference] = Field(default_factory=list)
    anatomy: list[NamedReference] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    assays: list[NamedReference] = Field(default_factory=list)
    environments: list[NamedReference] = Field(default_factory=list)
    modality: list[str] = Field(default_factory=list)
    readout_resolution: list[ReadoutResolution] = Field(default_factory=list)
    absolute_time: QuantitativeValue | None = None
    context_gaps: list[ContextGap] = Field(default_factory=list)
    # Deprecated coarse fallback retained for migration of early fixture bundles.
    context_missing_reasons: list[MissingContextReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_context_gaps(self) -> "ExperimentalContext":
        dimensions = [gap.dimension for gap in self.context_gaps]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("context_gaps must contain at most one gap per dimension")
        present = {
            "organism_taxon": True,
            "genetic_background": bool(self.genetic_background),
            "life_stage": bool(self.life_stage),
            "anatomy": bool(self.anatomy),
            "intervention": bool(self.interventions),
            "assay": bool(self.assays),
            "environment": bool(self.environments),
            "modality": bool(self.modality),
            "readout_resolution": bool(self.readout_resolution),
            "absolute_time": self.absolute_time is not None,
        }
        contradictory = sorted(
            gap.dimension for gap in self.context_gaps if present[gap.dimension]
        )
        if contradictory:
            raise ValueError(
                "context gaps cannot coexist with reported values for: "
                + ", ".join(contradictory)
            )
        return self


class EvidenceDirection(str, Enum):
    supports = "supports"
    disputes = "disputes"
    neutral = "neutral"


class EvidenceLine(StrictModel):
    id: str
    evidence_type: NamedReference
    direction: EvidenceDirection
    source_record_id: str
    publications: list[NamedReference] = Field(default_factory=list)
    figure_or_table: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_id(self) -> "EvidenceLine":
        if not CURIE_PATTERN.match(self.id):
            raise ValueError("evidence line id must be a CURIE")
        if not self.evidence_type.id.startswith("ECO:"):
            raise ValueError("evidence_type must use the pinned ECO vocabulary")
        return self


class SourceSnapshot(StrictModel):
    source_id: str
    source_release: str
    source_artifact: str
    source_url: HttpUrl | None = None
    retrieved_at: datetime | None = None
    checksum_sha256: str | None = None
    license_uri: HttpUrl | None = None
    transformation_activity: str | None = None
    transformer_version: str | None = None

    @model_validator(mode="after")
    def validate_provenance(self) -> "SourceSnapshot":
        if self.checksum_sha256 and not SHA256_PATTERN.match(self.checksum_sha256):
            raise ValueError("checksum_sha256 must contain exactly 64 hexadecimal characters")
        if self.transformation_activity and not CURIE_PATTERN.match(
            self.transformation_activity
        ):
            raise ValueError("transformation_activity must be a CURIE")
        return self


class Interpretation(str, Enum):
    observation = "observation"
    association = "association"
    causal_claim = "causal_claim"


class ObservationStatus(str, Enum):
    positive = "positive"
    explicit_negative = "explicit_negative"
    uncertain = "uncertain"


class RecordOrigin(str, Enum):
    curated = "curated"
    author_reported = "author_reported"
    extracted = "extracted"
    computed = "computed"


class BiolinkKnowledgeLevel(str, Enum):
    """Biolink 4.4.3 KnowledgeLevelEnum permissible values.

    Text co-occurrence is expressed through ``agent_type = text_mining_agent``,
    not as a knowledge level; ``text_co_occurrence`` is not a member of the
    Biolink KnowledgeLevelEnum and is deliberately absent here.
    """

    knowledge_assertion = "knowledge_assertion"
    logical_entailment = "logical_entailment"
    prediction = "prediction"
    statistical_association = "statistical_association"
    observation = "observation"
    not_provided = "not_provided"


class BiolinkAgentType(str, Enum):
    """Biolink 4.4.3 AgentTypeEnum permissible values."""

    manual_agent = "manual_agent"
    automated_agent = "automated_agent"
    data_analysis_pipeline = "data_analysis_pipeline"
    computational_model = "computational_model"
    text_mining_agent = "text_mining_agent"
    image_processing_agent = "image_processing_agent"
    manual_validation_of_automated_agent = "manual_validation_of_automated_agent"
    not_provided = "not_provided"


class ContextualObservation(StrictModel):
    id: str
    subject: NamedReference
    predicate: str
    object: NamedReference
    interpretation: Interpretation
    observation_status: ObservationStatus
    proposition_negated: bool = False
    context: ExperimentalContext
    # Many curated claims do not expose a quantitative payload. When a result is
    # available it is modeled explicitly rather than being flattened into object.
    result: ObservationResult | None = None
    evidence_lines: list[EvidenceLine] = Field(min_length=1)
    provenance: SourceSnapshot
    record_origin: RecordOrigin
    assertion_method: NamedReference | None = None
    biolink_knowledge_level: BiolinkKnowledgeLevel | None = None
    biolink_agent_type: BiolinkAgentType | None = None
    qualifiers: list[NamedReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_observation(self) -> "ContextualObservation":
        if not CURIE_PATTERN.match(self.id):
            raise ValueError("observation id must be a CURIE")
        if not CURIE_PATTERN.match(self.predicate):
            raise ValueError("predicate must be a CURIE")
        if self.observation_status is ObservationStatus.explicit_negative and not any(
            line.direction is EvidenceDirection.supports
            for line in self.evidence_lines
        ):
            raise ValueError("an explicit negative requires supporting evidence")
        if self.observation_status is ObservationStatus.explicit_negative and not self.proposition_negated:
            raise ValueError("an explicit negative must explicitly negate its proposition")
        if self.observation_status is ObservationStatus.positive and self.proposition_negated:
            raise ValueError("a positive observation cannot negate its proposition")
        if self.assertion_method and not self.assertion_method.id.startswith("ECO:"):
            raise ValueError("assertion_method must use the pinned ECO vocabulary")
        return self


class ObservationCollection(StrictModel):
    schema_version: str
    observations: list[ContextualObservation]


class TransportQuery(StrictModel):
    query_id: str
    description: str | None = None
    context: ExperimentalContext


class RightsStatus(str, Enum):
    cleared = "cleared"
    review_required = "review_required"
    metadata_only = "metadata_only"
    restricted = "restricted"


class ImplementationStatus(str, Enum):
    planned = "planned"
    scaffolded = "scaffolded"
    ready = "ready"
    blocked = "blocked"


class AcquisitionRequest(StrictModel):
    method: str
    parameters: dict[str, str | list[str]] = Field(default_factory=dict)
    body_sha256: str | None = None
    accept: str | None = None
    non_secret_headers: dict[str, str] = Field(default_factory=dict)
    pagination_state: str | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "AcquisitionRequest":
        self.method = self.method.upper()
        if self.method not in {"GET", "POST"}:
            raise ValueError("acquisition method must be GET or POST")
        if self.body_sha256 and not SHA256_PATTERN.match(self.body_sha256):
            raise ValueError("body_sha256 must contain exactly 64 hexadecimal characters")
        forbidden = {"authorization", "cookie", "proxy-authorization"}
        if forbidden.intersection(key.lower() for key in self.non_secret_headers):
            raise ValueError("secrets and authentication headers cannot be stored in a lock")
        return self


class ArtifactSpec(StrictModel):
    artifact_id: str
    role: str
    url: str
    filename: str
    media_type: str | None = None
    compression: str | None = None
    expected_sha256: str
    expected_size: int | None = None
    rights_status: RightsStatus
    license_uri: HttpUrl | None = None
    acquisition: AcquisitionRequest
    enabled: bool

    @model_validator(mode="after")
    def validate_hash(self) -> "ArtifactSpec":
        if not SAFE_PATH_TOKEN.match(self.artifact_id):
            raise ValueError("artifact_id must be a path-safe token")
        if not SHA256_PATTERN.match(self.expected_sha256):
            raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
        if PathLikeName.is_unsafe(self.filename):
            raise ValueError("filename must be a single safe relative filename")
        if self.expected_size is not None and self.expected_size < 0:
            raise ValueError("expected_size cannot be negative")
        if self.rights_status is RightsStatus.cleared and self.license_uri is None:
            raise ValueError("a rights-cleared artifact requires an explicit license_uri")
        return self


class PathLikeName:
    """Namespace for path-token checks without importing pathlib into model logic."""

    @staticmethod
    def is_unsafe(value: str) -> bool:
        return (
            not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or "\x00" in value
        )


class ReleaseManifest(StrictModel):
    manifest_schema_version: str
    source_id: str
    provider: str
    release: str
    release_is_mutable: bool
    adapter: str
    adapter_version: str
    landing_page: HttpUrl | None = None
    taxon: str | None = None
    sequence_assembly: str | None = None
    annotation_release: str | None = None
    bioproject: str | None = None
    coordinate_convention: str | None = None
    rights_status: RightsStatus
    license_uri: HttpUrl | None = None
    allowed_hosts: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactSpec]

    @model_validator(mode="after")
    def immutable_build_contract(self) -> "ReleaseManifest":
        if not SAFE_PATH_TOKEN.match(self.source_id):
            raise ValueError("source_id must be a path-safe token")
        if not SAFE_PATH_TOKEN.match(self.release):
            raise ValueError("release must be a path-safe token")
        if self.release.lower() in {"latest", "current", "unresolved"}:
            raise ValueError("release aliases cannot appear in an immutable release manifest")
        artifact_ids = [item.artifact_id for item in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifact_id values must be unique within a release")
        return self


class ArtifactReceipt(StrictModel):
    # Versionless receipts are the legacy 1.1 form.  Version 2.0 removes the
    # host-specific absolute path and binds the blob to a raw-root-relative,
    # content-addressed locator instead.
    artifact_receipt_schema_version: Literal["2.0"] | None = None
    source_id: str
    release: str
    artifact_id: str
    manifest_sha256: str
    artifact_spec_sha256: str
    adapter: str
    adapter_version: str
    role: str
    filename: str
    requested_url: str
    final_url: str
    redirect_chain: list[str]
    fetched_at: datetime
    sha256: str
    expected_sha256: str
    byte_size: int
    blob_uri: str
    blob_locator: str | None = None
    blob_path: str | None = None
    rights_status: RightsStatus
    license_uri: HttpUrl
    response_status: int | None = None
    response_content_type: str | None = None
    response_etag: str | None = None
    response_last_modified: str | None = None

    @model_validator(mode="after")
    def receipt_location_contract(self) -> "ArtifactReceipt":
        if self.artifact_receipt_schema_version == "2.0":
            if self.blob_path is not None:
                raise ValueError("portable artifact receipts cannot contain blob_path")
            if not self.blob_locator:
                raise ValueError("portable artifact receipts require blob_locator")
        else:
            if not self.blob_path:
                raise ValueError("legacy artifact receipts require blob_path")
            if self.blob_locator is not None:
                raise ValueError("legacy artifact receipts cannot contain blob_locator")
        return self


class CatalogArtifactIntent(StrictModel):
    artifact_id: str
    role: str
    url: HttpUrl
    filename: str
    media_type: str | None = None
    enabled: bool


class SourceReference(StrictModel):
    taxon: str | None = None
    sequence_assembly: str | None = None
    annotation_release: str | None = None
    bioproject: str | None = None
    coordinate_convention: str | None = None


class SourceCatalogEntry(StrictModel):
    source_id: str
    title: str
    provider: str
    release: str
    release_is_mutable: bool
    landing_page: HttpUrl
    download_root: HttpUrl | None = None
    adapter: str
    priority: int
    rights_status: RightsStatus
    enabled: bool
    inventory_complete: bool
    implementation_status: ImplementationStatus
    reference: SourceReference | None = None
    notes: str
    artifacts: list[CatalogArtifactIntent]

    @model_validator(mode="after")
    def validate_source_id(self) -> "SourceCatalogEntry":
        if not SAFE_PATH_TOKEN.match(self.source_id):
            raise ValueError("source_id must be a path-safe token")
        return self


class SourceCatalog(StrictModel):
    catalog_schema_version: str
    sources: list[SourceCatalogEntry]

    @model_validator(mode="after")
    def validate_catalog(self) -> "SourceCatalog":
        if self.catalog_schema_version != "1.0":
            raise ValueError("unsupported source catalog schema version")
        source_ids = [item.source_id for item in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique within a catalog")
        return self


class DatasetProfile(StrictModel):
    dataset_id: str
    source_id: str
    source_release: str
    source_artifact: str
    taxon: str
    genetic_background: list[str] = Field(default_factory=list)
    life_stage: list[str] = Field(default_factory=list)
    intervention_type: list[str] = Field(default_factory=list)
    perturbation_target: list[str] = Field(default_factory=list)
    compound_or_reagent: list[str] = Field(default_factory=list)
    assay: list[str] = Field(default_factory=list)
    anatomy: list[str] = Field(default_factory=list)
    modality: list[str] = Field(default_factory=list)
    readout_resolution: list[str] = Field(default_factory=list)
    observation_count: int
    direct_or_derived: str
    rights_status: RightsStatus | None = None
    license_uri: HttpUrl | None = None
    missing_context: dict[str, str] = Field(default_factory=dict)


def jsonable(value: BaseModel | Enum | Any) -> Any:
    """Convert models and enums to ordinary JSON-compatible values."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Enum):
        return value.value
    return value
