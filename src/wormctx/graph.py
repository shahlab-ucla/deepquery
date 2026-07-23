"""Faithful reified graph and loss-aware Biolink/KGX projection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from .io import validate_observation_collection
from .models import (
    BiolinkAgentType,
    BiolinkKnowledgeLevel,
    ContextualObservation,
    NamedReference,
    ObservationStatus,
)


BIOLINK_PREDICATE_MAP = {
    "biolink:affects": "biolink:affects",
    "biolink:associated_with": "biolink:associated_with",
    "biolink:related_to": "biolink:related_to",
    "biolink:has_phenotype": "biolink:has_phenotype",
}
# Concrete Biolink categories only. biolink:GenomicEntity is a mixin, not an
# instantiable category, so it is intentionally excluded (KGX/Biolink validation
# rejects a mixin used as a node category).
BIOLINK_CATEGORIES = {
    "biolink:BiologicalProcess",
    "biolink:ChemicalEntity",
    "biolink:Gene",
    "biolink:InformationContentEntity",
    "biolink:NamedThing",
    "biolink:OrganismTaxon",
    "biolink:PhenotypicFeature",
    "biolink:Procedure",
    "biolink:Publication",
}

# Exact, reviewed mappings only. Substring matching would incorrectly bless local
# mirrors, transformed derivatives, and similarly named sources as AGRKB itself.
REVIEWED_INFORES_MAP = {
    "alliance": "infores:agrkb",
    "alliance-genome-resources": "infores:agrkb",
}


def _edge_metadata_is_missing(value: Any) -> bool:
    """A strict KGX association needs a concrete knowledge level and agent type.

    ``not_provided`` is a valid Biolink sentinel, but it is truthy (a non-empty
    ``str`` enum member), so a plain ``not value`` test would wrongly accept it as
    "metadata complete" and emit an association whose provenance is explicitly
    unknown. Treat ``None`` and either enum's ``not_provided`` as missing.
    """
    return value is None or value in {
        BiolinkKnowledgeLevel.not_provided,
        BiolinkAgentType.not_provided,
    }


@dataclass
class GraphView:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_node(self, node: dict[str, Any]) -> None:
        identifier = node["id"]
        if identifier in self.nodes:
            existing = self.nodes[identifier]
            if existing == node:
                return
            if (
                existing.get("_identity_kind") == "entity_reference"
                and node.get("_identity_kind") == "entity_reference"
            ):
                old_name = existing.get("name")
                new_name = node.get("name")
                if old_name and new_name and old_name != new_name:
                    raise ValueError(
                        f"incompatible graph node ID collision for {identifier!r}: "
                        "conflicting entity labels"
                    )
                old_categories = set(existing.get("category", [])) - {
                    "biolink:NamedThing"
                }
                new_categories = set(node.get("category", [])) - {
                    "biolink:NamedThing"
                }
                if (
                    old_categories
                    and new_categories
                    and old_categories != new_categories
                ):
                    raise ValueError(
                        f"incompatible graph node ID collision for {identifier!r}: "
                        "conflicting entity categories"
                    )
                existing["name"] = old_name or new_name
                categories = old_categories | new_categories
                existing["category"] = sorted(categories) or ["biolink:NamedThing"]
                return
            raise ValueError(
                f"incompatible graph node ID collision for {identifier!r}"
            )
        self.nodes[identifier] = node

    def add_edge(self, edge: dict[str, Any]) -> None:
        identifier = edge["id"]
        if identifier in self.edges:
            if self.edges[identifier] == edge:
                return
            raise ValueError(
                f"incompatible graph edge ID collision for {identifier!r}"
            )
        self.edges[identifier] = edge

    def sorted_nodes(self) -> list[dict[str, Any]]:
        return [self.nodes[key] for key in sorted(self.nodes)]

    def sorted_edges(self) -> list[dict[str, Any]]:
        return [self.edges[key] for key in sorted(self.edges)]


def build_contextual_graph(observations: Iterable[ContextualObservation]) -> GraphView:
    observations = list(observations)
    validate_observation_collection(observations)
    graph = GraphView()
    for observation in observations:
        _reference_node(graph, observation.subject, observation.provenance.source_id)
        _reference_node(graph, observation.object, observation.provenance.source_id)
        canonical = observation.model_dump(mode="json", exclude_none=True)
        canonical_sha256 = _canonical_sha256(canonical)
        graph.add_node(
            {
                "id": observation.id,
                "name": f"Contextual observation {observation.id}",
                "category": [
                    "biolink:InformationContentEntity",
                    "wormctx:ContextualObservation",
                ],
                "source": observation.provenance.source_id,
                "interpretation": observation.interpretation.value,
                "observation_status": observation.observation_status.value,
                "record_origin": observation.record_origin.value,
                "predicate": observation.predicate,
                "proposition_negated": observation.proposition_negated,
                "assertion_method": (
                    observation.assertion_method.model_dump(mode="json", exclude_none=True)
                    if observation.assertion_method
                    else None
                ),
                "biolink_knowledge_level": (
                    observation.biolink_knowledge_level.value
                    if observation.biolink_knowledge_level
                    else None
                ),
                "biolink_agent_type": (
                    observation.biolink_agent_type.value
                    if observation.biolink_agent_type
                    else None
                ),
                "qualifiers": [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in observation.qualifiers
                ],
                "context": observation.context.model_dump(mode="json", exclude_none=True),
                "result": (
                    observation.result.model_dump(mode="json", exclude_none=True)
                    if observation.result
                    else None
                ),
                # This canonical payload is the lossless graph contract. The
                # additional reified nodes/edges are query indexes over it.
                "canonical_observation": canonical,
                "canonical_sha256": canonical_sha256,
            }
        )
        _edge(graph, observation.id, "wormctx:has_subject", observation.subject.id)
        _edge(graph, observation.id, "wormctx:has_object", observation.object.id)
        _edge(
            graph,
            observation.subject.id,
            observation.predicate,
            observation.object.id,
            derived_from_observation=observation.id,
            observation_status=observation.observation_status.value,
            negated=observation.proposition_negated,
            interpretation=observation.interpretation.value,
        )

        if observation.assertion_method:
            _reference_node(
                graph, observation.assertion_method, observation.provenance.source_id
            )
            _edge(
                graph,
                observation.id,
                "wormctx:has_assertion_method",
                observation.assertion_method.id,
            )
        for qualifier in observation.qualifiers:
            _reference_node(graph, qualifier, observation.provenance.source_id)
            _edge(graph, observation.id, "wormctx:has_qualifier", qualifier.id)

        context_refs: list[tuple[str, NamedReference]] = [
            ("wormctx:has_organism_taxon", observation.context.organism_taxon)
        ]
        context_refs.extend(
            ("wormctx:has_genetic_background", value)
            for value in observation.context.genetic_background
        )
        context_refs.extend(
            ("wormctx:has_life_stage", value) for value in observation.context.life_stage
        )
        context_refs.extend(
            ("wormctx:has_anatomical_context", value) for value in observation.context.anatomy
        )
        context_refs.extend(
            ("wormctx:used_assay", value) for value in observation.context.assays
        )
        context_refs.extend(
            ("wormctx:has_environment", value) for value in observation.context.environments
        )
        for predicate, reference in context_refs:
            _reference_node(graph, reference, observation.provenance.source_id)
            _edge(graph, observation.id, predicate, reference.id)

        # Preserve scalar and missingness semantics on the observation node as
        # well as inside the canonical payload.
        observation_node = graph.nodes[observation.id]
        observation_node["modality"] = list(observation.context.modality)
        observation_node["readout_resolution"] = [
            item.value for item in observation.context.readout_resolution
        ]
        observation_node["absolute_time"] = (
            observation.context.absolute_time.model_dump(mode="json", exclude_none=True)
            if observation.context.absolute_time
            else None
        )
        observation_node["context_gaps"] = [
            item.model_dump(mode="json", exclude_none=True)
            for item in observation.context.context_gaps
        ]
        observation_node["context_missing_reasons"] = [
            item.value for item in observation.context.context_missing_reasons
        ]

        for index, intervention in enumerate(observation.context.interventions):
            intervention_id = f"{observation.id}/intervention/{index + 1}"
            graph.add_node(
                {
                    "id": intervention_id,
                    "name": intervention.notes or intervention.intervention_type,
                    "category": ["biolink:Procedure", "wormctx:Intervention"],
                    "source": observation.provenance.source_id,
                    "intervention_type": intervention.intervention_type,
                    "dose": (
                        intervention.dose.model_dump(mode="json", exclude_none=True)
                        if intervention.dose
                        else None
                    ),
                    "duration": (
                        intervention.duration.model_dump(mode="json", exclude_none=True)
                        if intervention.duration
                        else None
                    ),
                    "notes": intervention.notes,
                }
            )
            _edge(graph, observation.id, "wormctx:has_intervention", intervention_id)
            if intervention.target:
                _reference_node(graph, intervention.target, observation.provenance.source_id)
                _edge(graph, intervention_id, "wormctx:has_target", intervention.target.id)
            if intervention.reagent:
                _reference_node(graph, intervention.reagent, observation.provenance.source_id)
                _edge(graph, intervention_id, "wormctx:uses_reagent", intervention.reagent.id)
            if intervention.route:
                _reference_node(graph, intervention.route, observation.provenance.source_id)
                _edge(graph, intervention_id, "wormctx:has_route", intervention.route.id)

        if observation.result:
            result = observation.result
            graph.add_node(
                {
                    "id": result.id,
                    "name": result.summary or f"Result for {observation.id}",
                    "category": ["wormctx:ObservationResult"],
                    "source": observation.provenance.source_id,
                    "result_type": result.result_type.value,
                    "summary": result.summary,
                }
            )
            _edge(graph, observation.id, "wormctx:has_result", result.id)
            asset_ids = {item.id for item in result.asset_references}
            for group in result.comparison_groups:
                graph.add_node(
                    {
                        "id": group.id,
                        "name": group.label,
                        "category": ["wormctx:ComparisonGroup"],
                        "source": observation.provenance.source_id,
                        "group_role": group.role.value,
                        "group_size": group.size,
                        "group_members": [item.id for item in group.members],
                        "group_genetic_background": [
                            item.id for item in group.genetic_background
                        ],
                        "notes": group.notes,
                    }
                )
                _edge(graph, result.id, "wormctx:has_comparison_group", group.id)
                for member in group.members:
                    _reference_node(graph, member, observation.provenance.source_id)
                    _edge(graph, group.id, "wormctx:has_member", member.id)
                for background in group.genetic_background:
                    _reference_node(graph, background, observation.provenance.source_id)
                    _edge(
                        graph,
                        group.id,
                        "wormctx:has_genetic_background",
                        background.id,
                    )
            for measurement in result.measurements:
                graph.add_node(
                    {
                        "id": measurement.id,
                        "name": measurement.notes or measurement.measured_feature.label,
                        "category": ["wormctx:Measurement"],
                        "source": observation.provenance.source_id,
                        "measured_feature": measurement.measured_feature.model_dump(
                            mode="json", exclude_none=True
                        ),
                        "value": measurement.value.model_dump(
                            mode="json", exclude_none=True
                        ),
                        "statistic": (
                            measurement.statistic.model_dump(
                                mode="json", exclude_none=True
                            )
                            if measurement.statistic
                            else None
                        ),
                        "uncertainty": (
                            measurement.uncertainty.model_dump(
                                mode="json", exclude_none=True
                            )
                            if measurement.uncertainty
                            else None
                        ),
                        "uncertainty_type": (
                            measurement.uncertainty_type.model_dump(
                                mode="json", exclude_none=True
                            )
                            if measurement.uncertainty_type
                            else None
                        ),
                        "confidence_level": measurement.confidence_level,
                        "p_value": measurement.p_value,
                        "penetrance": (
                            measurement.penetrance.model_dump(
                                mode="json", exclude_none=True
                            )
                            if measurement.penetrance
                            else None
                        ),
                        "sample_size": measurement.sample_size,
                        "biological_replicates": measurement.biological_replicates,
                        "technical_replicates": measurement.technical_replicates,
                        "measurement_time": (
                            measurement.measurement_time.model_dump(
                                mode="json", exclude_none=True
                            )
                            if measurement.measurement_time
                            else None
                        ),
                        "comparison_group_ids": list(
                            measurement.comparison_group_ids
                        ),
                        "asset_ids": list(measurement.asset_ids),
                        "notes": measurement.notes,
                    }
                )
                _edge(graph, result.id, "wormctx:has_measurement", measurement.id)
                _reference_node(
                    graph,
                    measurement.measured_feature,
                    observation.provenance.source_id,
                )
                _edge(
                    graph,
                    measurement.id,
                    "wormctx:measures_feature",
                    measurement.measured_feature.id,
                )
                if measurement.statistic:
                    _reference_node(
                        graph,
                        measurement.statistic,
                        observation.provenance.source_id,
                    )
                    _edge(
                        graph,
                        measurement.id,
                        "wormctx:has_statistic",
                        measurement.statistic.id,
                    )
                if measurement.uncertainty_type:
                    _reference_node(
                        graph,
                        measurement.uncertainty_type,
                        observation.provenance.source_id,
                    )
                    _edge(
                        graph,
                        measurement.id,
                        "wormctx:has_uncertainty_type",
                        measurement.uncertainty_type.id,
                    )
                for group_id in measurement.comparison_group_ids:
                    _edge(
                        graph,
                        measurement.id,
                        "wormctx:compares_group",
                        group_id,
                    )
                for asset_id in measurement.asset_ids:
                    # The Pydantic result validator has already established this.
                    if asset_id in asset_ids:
                        _edge(graph, measurement.id, "wormctx:derived_from_asset", asset_id)
            for asset in result.asset_references:
                graph.add_node(
                    {
                        "id": asset.id,
                        "name": asset.role or asset.id,
                        "category": ["prov:Entity", "wormctx:ResultAsset"],
                        "source": observation.provenance.source_id,
                        "asset_uri": asset.uri,
                        "checksum_sha256": asset.checksum_sha256,
                        "media_type": asset.media_type,
                        "byte_size": asset.byte_size,
                        "role": asset.role,
                    }
                )
                _edge(graph, result.id, "wormctx:has_result_asset", asset.id)

        for evidence in observation.evidence_lines:
            graph.add_node(
                {
                    "id": evidence.id,
                    "name": evidence.notes or evidence.source_record_id,
                    "category": ["biolink:InformationContentEntity", "wormctx:EvidenceLine"],
                    "source": observation.provenance.source_id,
                    "evidence_type": evidence.evidence_type.id,
                    "direction": evidence.direction.value,
                    "source_record_id": evidence.source_record_id,
                    "publications": [item.id for item in evidence.publications],
                    "figure_or_table": evidence.figure_or_table,
                    "notes": evidence.notes,
                }
            )
            _edge(graph, observation.id, "wormctx:has_evidence_line", evidence.id)
            _reference_node(
                graph, evidence.evidence_type, observation.provenance.source_id
            )
            _edge(
                graph,
                evidence.id,
                "wormctx:has_evidence_type",
                evidence.evidence_type.id,
            )
            for publication in evidence.publications:
                _reference_node(graph, publication, observation.provenance.source_id)
                _edge(graph, evidence.id, "wormctx:reported_in", publication.id)

        snapshot_id = _snapshot_id(observation)
        graph.add_node(
            {
                "id": snapshot_id,
                "name": (
                    f"{observation.provenance.source_id} "
                    f"{observation.provenance.source_release} "
                    f"{observation.provenance.source_artifact}"
                ),
                "category": ["biolink:Dataset", "prov:Entity"],
                "source": observation.provenance.source_id,
                "source_url": (
                    str(observation.provenance.source_url)
                    if observation.provenance.source_url
                    else None
                ),
                "checksum_sha256": observation.provenance.checksum_sha256,
                "source_release": observation.provenance.source_release,
                "source_artifact": observation.provenance.source_artifact,
                "retrieved_at": (
                    observation.provenance.retrieved_at.isoformat()
                    if observation.provenance.retrieved_at
                    else None
                ),
                "license_uri": (
                    str(observation.provenance.license_uri)
                    if observation.provenance.license_uri
                    else None
                ),
                "transformation_activity": observation.provenance.transformation_activity,
                "transformer_version": observation.provenance.transformer_version,
            }
        )
        _edge(graph, observation.id, "prov:wasDerivedFrom", snapshot_id)
        if observation.provenance.transformation_activity:
            activity_id = observation.provenance.transformation_activity
            graph.add_node(
                {
                    "id": activity_id,
                    "name": f"Transformation {activity_id}",
                    "category": ["prov:Activity"],
                }
            )
            _edge(graph, observation.id, "prov:wasGeneratedBy", activity_id)
            _edge(graph, activity_id, "prov:used", snapshot_id)
            if observation.provenance.transformer_version:
                software_id = _software_agent_id(
                    observation.provenance.transformer_version
                )
                graph.add_node(
                    {
                        "id": software_id,
                        "name": f"wormctx {observation.provenance.transformer_version}",
                        "category": ["prov:SoftwareAgent"],
                    }
                )
                _edge(graph, activity_id, "prov:wasAssociatedWith", software_id)
    return graph


def observations_from_contextual_graph(graph: GraphView) -> list[ContextualObservation]:
    """Reconstruct and integrity-check canonical observations from a graph view."""
    observations: list[ContextualObservation] = []
    for node in graph.nodes.values():
        categories = node.get("category", [])
        if isinstance(categories, str):
            categories = json.loads(categories) if categories else []
        if "wormctx:ContextualObservation" not in categories:
            continue
        payload = node.get("canonical_observation")
        if payload is None:
            raise ValueError(f"contextual observation node {node['id']!r} lacks canonical payload")
        if isinstance(payload, str):
            payload = json.loads(payload)
        expected = node.get("canonical_sha256")
        actual = _canonical_sha256(payload)
        if expected != actual:
            raise ValueError(f"canonical payload checksum mismatch for {node['id']!r}")
        observation = ContextualObservation.model_validate(payload)
        if observation.id != node["id"]:
            raise ValueError(f"canonical payload identifier mismatch for {node['id']!r}")
        observations.append(observation)
    observations = sorted(observations, key=lambda item: item.id)
    validate_observation_collection(observations)
    return observations


def build_kgx_projection(
    observations: Iterable[ContextualObservation],
    *,
    include_uncertain: bool = False,
    require_reviewed_infores: bool = True,
    on_unreviewed_infores: str = "error",
    on_unmapped_predicate: str = "error",
    require_edge_metadata: bool = True,
    on_missing_edge_metadata: str = "error",
) -> GraphView:
    """Build the pairwise exchange projection.

    Standard export is conservative: uncertain observations are omitted and a
    reviewed Information Resource Registry mapping is mandatory. Callers making
    a synthetic/demo projection may explicitly relax the latter; the sidecar
    records that choice.
    """
    if on_unreviewed_infores not in {"error", "skip"}:
        raise ValueError("on_unreviewed_infores must be 'error' or 'skip'")
    if on_unmapped_predicate not in {"error", "skip"}:
        raise ValueError("on_unmapped_predicate must be 'error' or 'skip'")
    if on_missing_edge_metadata not in {"error", "skip"}:
        raise ValueError("on_missing_edge_metadata must be 'error' or 'skip'")
    observations = list(observations)
    validate_observation_collection(observations)
    graph = GraphView()
    for observation in observations:
        if (
            observation.observation_status is ObservationStatus.uncertain
            and not include_uncertain
        ):
            continue
        predicate = BIOLINK_PREDICATE_MAP.get(observation.predicate)
        if predicate is None:
            if on_unmapped_predicate == "skip":
                continue
            raise ValueError(
                f"no reviewed Biolink predicate mapping for {observation.predicate!r}; "
                "retain it only in the contextual graph until mapped"
            )
        infores = _infores(observation.provenance.source_id)
        if require_reviewed_infores and infores is None:
            if on_unreviewed_infores == "skip":
                continue
            raise ValueError(
                "standard KGX export requires a reviewed Information Resource "
                f"Registry mapping for source {observation.provenance.source_id!r}"
            )
        missing_edge_metadata = [
            name
            for name, value in (
                ("knowledge_level", observation.biolink_knowledge_level),
                ("agent_type", observation.biolink_agent_type),
            )
            if _edge_metadata_is_missing(value)
        ]
        if require_edge_metadata and missing_edge_metadata:
            if on_missing_edge_metadata == "skip":
                continue
            raise ValueError(
                "standard KGX export requires explicit "
                + " and ".join(missing_edge_metadata)
                + f" for observation {observation.id!r}"
            )
        _kgx_reference_node(graph, observation.subject, infores)
        _kgx_reference_node(graph, observation.object, infores)
        publications = sorted(
            {
                publication.id
                for evidence in observation.evidence_lines
                for publication in evidence.publications
            }
        )
        for evidence in observation.evidence_lines:
            graph.add_node(
                {
                    "id": evidence.id,
                    "name": evidence.notes or evidence.source_record_id,
                    "category": ["biolink:InformationContentEntity"],
                    "provided_by": [infores] if infores else [],
                }
            )
        edge = {
            "id": _projection_edge_id(observation, predicate),
            "subject": observation.subject.id,
            "predicate": predicate,
            "object": observation.object.id,
            "category": ["biolink:Association"],
            "publications": publications,
            "has_evidence": [item.id for item in observation.evidence_lines],
            "knowledge_level": (
                observation.biolink_knowledge_level.value
                if observation.biolink_knowledge_level
                else None
            ),
            "agent_type": (
                observation.biolink_agent_type.value
                if observation.biolink_agent_type
                else None
            ),
            "negated": observation.proposition_negated,
        }
        if infores:
            edge["primary_knowledge_source"] = infores
        graph.add_edge(edge)
    return graph


def build_kgx_context_sidecar(
    observations: Iterable[ContextualObservation],
    *,
    include_uncertain: bool = False,
    require_reviewed_infores: bool = True,
    require_edge_metadata: bool = True,
) -> list[dict[str, Any]]:
    """Preserve canonical context that cannot safely live on an ordinary KGX edge."""
    observations = list(observations)
    validate_observation_collection(observations)
    rows = []
    for observation in observations:
        predicate = BIOLINK_PREDICATE_MAP.get(observation.predicate)
        infores = _infores(observation.provenance.source_id)
        warnings = ["experimental context is carried in this sidecar, not the KGX edge"]
        projection_status = "projected"
        # Precedence mirrors build_kgx_projection: uncertain is checked before the
        # predicate mapping, so an uncertain+unmapped record is labelled the same
        # way here as it is skipped there.
        if (
            observation.observation_status is ObservationStatus.uncertain
            and not include_uncertain
        ):
            projection_status = "skipped_uncertain"
            warnings.append("uncertain observation omitted from the KGX projection")
        elif predicate is None:
            projection_status = "blocked_unmapped_predicate"
            warnings.append("predicate has no reviewed Biolink mapping")
        if infores is None:
            warnings.append(
                "primary source has no reviewed Information Resource Registry mapping"
            )
            if require_reviewed_infores and projection_status == "projected":
                projection_status = "blocked_unreviewed_infores"
        missing_edge_metadata = [
            name
            for name, value in (
                ("knowledge_level", observation.biolink_knowledge_level),
                ("agent_type", observation.biolink_agent_type),
            )
            if _edge_metadata_is_missing(value)
        ]
        if missing_edge_metadata:
            warnings.append(
                "missing required KGX association metadata: "
                + ", ".join(missing_edge_metadata)
            )
            if require_edge_metadata and projection_status == "projected":
                projection_status = "blocked_missing_edge_metadata"
        projected = projection_status == "projected"
        rows.append(
            {
                "association_id": (
                    _projection_edge_id(observation, predicate)
                    if projected and predicate
                    else None
                ),
                "observation_id": observation.id,
                "projection_status": projection_status,
                "observation_status": observation.observation_status.value,
                "proposition_negated": observation.proposition_negated,
                "interpretation": observation.interpretation.value,
                "record_origin": observation.record_origin.value,
                "original_subject_categories": observation.subject.categories,
                "original_predicate": observation.predicate,
                "original_object_categories": observation.object.categories,
                "context": observation.context.model_dump(mode="json", exclude_none=True),
                "result": (
                    observation.result.model_dump(mode="json", exclude_none=True)
                    if observation.result
                    else None
                ),
                "evidence_lines": [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in observation.evidence_lines
                ],
                "assertion_method": (
                    observation.assertion_method.model_dump(mode="json", exclude_none=True)
                    if observation.assertion_method
                    else None
                ),
                "qualifiers": [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in observation.qualifiers
                ],
                "source_snapshot": observation.provenance.model_dump(
                    mode="json", exclude_none=True
                ),
                "projection_warnings": warnings,
            }
        )
    return rows


def _reference_node(graph: GraphView, reference: NamedReference, source: str) -> None:
    graph.add_node(
        {
            "id": reference.id,
            "name": reference.label,
            "category": reference.categories or ["biolink:NamedThing"],
            "_identity_kind": "entity_reference",
        }
    )


def _kgx_reference_node(
    graph: GraphView, reference: NamedReference, infores: str | None
) -> None:
    categories = [item for item in reference.categories if item in BIOLINK_CATEGORIES]
    graph.add_node(
        {
            "id": reference.id,
            "name": reference.label,
            "category": categories or ["biolink:NamedThing"],
            "_identity_kind": "entity_reference",
        }
    )


def _edge(
    graph: GraphView,
    subject: str,
    predicate: str,
    object_: str,
    **properties: Any,
) -> None:
    identifier = _edge_id(subject, predicate, object_, properties.get("derived_from_observation"))
    graph.add_edge(
        {
            "id": identifier,
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            **properties,
        }
    )


def _edge_id(subject: str, predicate: str, object_: str, salt: str | None = None) -> str:
    value = "\x1f".join((subject, predicate, object_, salt or ""))
    return f"wormctx:edge-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def _snapshot_id(observation: ContextualObservation) -> str:
    raw = "\x1f".join(
        (
            observation.provenance.source_id,
            observation.provenance.source_release,
            observation.provenance.source_artifact,
        )
    )
    return f"wormctx:snapshot-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _projection_edge_id(observation: ContextualObservation, predicate: str) -> str:
    return _edge_id(observation.subject.id, predicate, observation.object.id, observation.id)


def _software_agent_id(version: str) -> str:
    digest = hashlib.sha256(version.encode("utf-8")).hexdigest()[:16]
    return f"wormctx:software-agent-{digest}"


def _infores(source_id: str) -> str | None:
    return REVIEWED_INFORES_MAP.get(source_id.lower())


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
