from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from wormctx.adapters.wormbase_ws298 import (
    EMPTY_GAF_QUALIFIER_RECEIPT_TOKEN,
    WormBaseWs298Adapter,
    WormBaseWs298Inputs,
    profile_table,
    read_gaf,
    read_gene_ids,
    read_obo,
)
from wormctx.io import read_json, write_jsonl
from wormctx.manifests import fetch_release
from wormctx.pipeline import (
    build_graph,
    normalize_snapshot,
    verify_build_receipt,
    verify_normalization_receipt,
    write_build_receipt,
)
from wormctx.poc.wormbase_ws298_source_qualification import (
    QualificationContract,
    qualify_source,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
REAL_CONFIG = (
    ROOT
    / "experiments"
    / "context_graph"
    / "wormbase_provenance_ingestion"
    / "config"
    / "source_qualification.json"
)


def _fixture_inputs() -> WormBaseWs298Inputs:
    return WormBaseWs298Inputs(
        gene_ids=FIXTURES / "wormbase_ws298_source_qualification_gene_ids.csv",
        phenotype_gaf=FIXTURES / "wormbase_ws298_source_qualification_phenotype.gaf",
        development_gaf=FIXTURES / "wormbase_ws298_source_qualification_development.gaf",
        phenotype_obo=FIXTURES / "wormbase_ws298_source_qualification_phenotype.obo",
        development_obo=FIXTURES / "wormbase_ws298_source_qualification_development.obo",
        source_id="wormbase-ws298-source_qualification-fixture",
        source_release="WS298",
        source_url="https://example.org/wormbase-ws298-source_qualification-fixture/",
        license_uri="https://creativecommons.org/publicdomain/zero/1.0/",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _write_normalization_fixture(root: Path) -> tuple[Path, Path]:
    """Create a six-artifact, rights-cleared manifest from author-created fixtures."""

    sources = {
        "gene-ids": FIXTURES / "wormbase_ws298_source_qualification_gene_ids.csv",
        "phenotype-associations": FIXTURES / "wormbase_ws298_source_qualification_phenotype.gaf",
        "development-associations": FIXTURES
        / "wormbase_ws298_source_qualification_development.gaf",
        "phenotype-ontology": FIXTURES / "wormbase_ws298_source_qualification_phenotype.obo",
        "development-ontology": FIXTURES / "wormbase_ws298_source_qualification_development.obo",
    }
    filenames = {
        "gene-ids": "gene_ids.csv",
        "phenotype-associations": "phenotype.gaf",
        "development-associations": "development.gaf",
        "phenotype-ontology": "phenotype.obo",
        "development-ontology": "development.obo",
    }
    artifacts = []
    for artifact_id, fixture_path in sources.items():
        destination = root / filenames[artifact_id]
        destination.write_bytes(fixture_path.read_bytes())
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "role": artifact_id,
                "url": destination.name,
                "filename": destination.name,
                "media_type": "text/plain",
                "expected_sha256": _sha(destination),
                "expected_size": destination.stat().st_size,
                "rights_status": "cleared",
                "license_uri": "https://creativecommons.org/publicdomain/zero/1.0/",
                "acquisition": {"method": "GET"},
                "enabled": True,
            }
        )
    support = root / "release_support.txt"
    support.write_text(
        "Author-created support artifact with no normalized rows.\n",
        encoding="utf-8",
        newline="\n",
    )
    artifacts.append(
        {
            "artifact_id": "release-support",
            "role": "release_support",
            "url": support.name,
            "filename": support.name,
            "media_type": "text/plain",
            "expected_sha256": _sha(support),
            "expected_size": support.stat().st_size,
            "rights_status": "cleared",
            "license_uri": "https://creativecommons.org/publicdomain/zero/1.0/",
            "acquisition": {"method": "GET"},
            "enabled": True,
        }
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_schema_version": "1.0",
                # Keep receipt paths below the legacy Windows MAX_PATH limit;
                # scientific identity here comes from fixture bytes, not this token.
                "source_id": "wbf",
                "provider": "wormctx tests",
                "release": "r1",
                "release_is_mutable": False,
                "adapter": "wormbase_ws298_source_qualification",
                "adapter_version": "0.1.0",
                "taxon": "NCBITaxon:6239",
                "rights_status": "cleared",
                "license_uri": (
                    "https://creativecommons.org/publicdomain/zero/1.0/"
                ),
                "allowed_hosts": [],
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    raw = root / "raw"
    fetch_release(manifest, raw)
    return manifest, raw


def test_real_contract_freezes_only_the_seven_bounded_files() -> None:
    contract = QualificationContract.model_validate(
        json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    )
    assert len(contract.artifacts) == 7
    assert contract.total_expected_bytes == 5_382_876
    assert {artifact.role for artifact in contract.artifacts} == {
        "release_checksum_ledger",
        "release_letter",
        "gene_identifiers",
        "phenotype_associations",
        "development_associations",
        "phenotype_ontology",
        "development_ontology",
    }
    assert {
        artifact.artifact_id
        for artifact in contract.artifacts
        if artifact.rights_status == "review_required"
    } == {
        "ws298-checksums",
        "ws298-release-letter",
        "ws298-gene-ids",
        "ws298-phenotype-associations",
        "ws298-development-associations",
    }
    payload = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    payload["artifacts"][0]["relative_path"] = "../outside"
    with pytest.raises(ValueError, match="safe canonical POSIX"):
        QualificationContract.model_validate(payload)


def test_rights_review_artifact_cannot_enable_external_transfer() -> None:
    payload = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    payload["artifacts"][0]["external_transfer_permitted"] = True
    with pytest.raises(ValueError, match="cannot permit external transfer"):
        QualificationContract.model_validate(payload)


def test_fixture_parsers_and_adapter_preserve_negation_and_stage_context() -> None:
    inputs = _fixture_inputs()
    assert len(read_gene_ids(inputs.gene_ids)) == 3
    assert len(
        read_gaf(
            inputs.phenotype_gaf,
            expected_aspect="P",
            expected_object_prefix="WBPhenotype:",
        ).records
    ) == 2
    assert len(
        read_obo(
            inputs.development_obo,
            expected_ontology="wbls/wbls-simple",
            expected_prefix="WBls:",
        ).terms
    ) == 2

    adapter = WormBaseWs298Adapter()
    report = adapter.validate_raw(inputs)
    assert report.valid
    assert report.checked_records == 7
    observations = list(adapter.iter_observations(inputs))
    assert len(observations) == 4
    negative = next(item for item in observations if item.proposition_negated)
    assert negative.observation_status.value == "explicit_negative"
    development = [
        item
        for item in observations
        if item.predicate == "wormctx:observed_during_life_stage"
    ]
    assert len(development) == 2
    assert development[0].context.life_stage[0].id.startswith("WBls:")
    assert all(
        item.evidence_lines[0].evidence_type.mapping_status.value == "mapped_broad"
        for item in observations
    )
    fixture_profile = profile_table(
        read_gaf(
            inputs.phenotype_gaf,
            expected_aspect="P",
            expected_object_prefix="WBPhenotype:",
        )
    )
    assert fixture_profile["qualifier_counts"] == {
        "NOT": 1,
        EMPTY_GAF_QUALIFIER_RECEIPT_TOKEN: 1,
    }
    assert "" not in fixture_profile["qualifier_counts"]


def test_obo_allows_nameless_terms_only_when_source_marks_them_obsolete(
    tmp_path: Path,
) -> None:
    obo = tmp_path / "nameless-obsolete.obo"
    header = "format-version: 1.2\nontology: wbphenotype\n\n[Term]\n"
    obo.write_text(
        header + "id: WBPhenotype:9000003\nis_obsolete: true\n",
        encoding="utf-8",
    )
    ontology = read_obo(
        obo,
        expected_ontology="wbphenotype",
        expected_prefix="WBPhenotype:",
    )
    assert ontology.terms["WBPhenotype:9000003"].name is None
    obo.write_text(header + "id: WBPhenotype:9000003\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unless obsolete"):
        read_obo(
            obo,
            expected_ontology="wbphenotype",
            expected_prefix="WBPhenotype:",
        )


def test_fixture_reuses_context_graph_and_receipt_boundaries(tmp_path: Path) -> None:
    observations = list(WormBaseWs298Adapter().iter_observations(_fixture_inputs()))
    source = tmp_path / "wormbase_ws298_source_qualification_fixture_observations.jsonl"
    output = tmp_path / "wormbase_ws298_source_qualification_fixture_graph"
    write_jsonl(source, observations)
    graph = build_graph(source, output)
    receipt = write_build_receipt(
        output,
        {
            "graph": graph,
            "input_binding": {
                "status": "unbound_bundled_synthetic_fixture",
                "warning": (
                    "Author-created CC0 source qualification fixture; not a real WS298 normalization receipt."
                ),
            },
        },
    )
    assert graph["observation_count"] == 4
    assert graph["contextual_node_count"] > 4
    assert graph["kgx_edge_count"] == 0
    assert receipt["bindings"]["normalization"][0]["status"] == (
        "unbound_bundled_synthetic_fixture"
    )
    verified = verify_build_receipt(
        output, local_inputs={"observations_sha256": [source]}
    )
    assert verified["status"] == "authoritative_build_snapshot_verified"


def test_multi_artifact_normalization_binds_exact_filename_sha_and_zero_use(
    tmp_path: Path,
) -> None:
    manifest, raw = _write_normalization_fixture(tmp_path)
    normalized = tmp_path / "normalized" / "observations.jsonl"
    result = normalize_snapshot(manifest, raw, normalized)
    receipt = read_json(result["normalization_receipt"])

    assert receipt["normalization_receipt_schema_version"] == "1.2"
    assert receipt["provenance_binding"] == (
        "locked_manifest_all_artifact_receipts_and_primary_record_artifact"
    )
    assert receipt["observation_primary_artifact_counts"] == {
        "development-associations": 2,
        "development-ontology": 0,
        "gene-ids": 0,
        "phenotype-associations": 2,
        "phenotype-ontology": 0,
        "release-support": 0,
    }
    expected_files = {
        "development-associations": "development.gaf",
        "development-ontology": "development.obo",
        "gene-ids": "gene_ids.csv",
        "phenotype-associations": "phenotype.gaf",
        "phenotype-ontology": "phenotype.obo",
        "release-support": "release_support.txt",
    }
    assert {
        binding["artifact_id"]: binding["filename"]
        for binding in receipt["artifact_receipts"]
    } == expected_files
    assert {
        binding["artifact_id"]: binding["artifact_sha256"]
        for binding in receipt["artifact_receipts"]
    } == {
        artifact_id: _sha(tmp_path / filename)
        for artifact_id, filename in expected_files.items()
    }

    full_binding = verify_normalization_receipt(
        normalized,
        result["normalization_receipt"],
        manifest_path=manifest,
        raw_root=raw,
    )
    assert full_binding["status"] == "full_provenance_verified"
    assert full_binding["observation_primary_artifact_counts"] == (
        receipt["observation_primary_artifact_counts"]
    )

    build_root = tmp_path / "graph"
    graph = build_graph(normalized, build_root)
    graph["input_binding"] = full_binding
    build_receipt = write_build_receipt(build_root, graph)
    claimed = build_receipt["bindings"]["normalization"][0]
    assert claimed["normalization_receipt_schema_version"] == "1.2"
    assert claimed["observation_primary_artifact_counts"]["release-support"] == 0
    verified = verify_build_receipt(
        build_root,
        local_inputs={"observations_sha256": [normalized]},
        normalization_receipt_path=result["normalization_receipt"],
        manifest_path=manifest,
        raw_root=raw,
    )
    assert verified["normalization"]["status"] == "full_provenance_verified"


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("source_artifact", "not-a-locked-filename.gaf"),
        ("checksum_sha256", "0" * 64),
    ],
)
def test_multi_artifact_verifier_rejects_wrong_row_binding(
    tmp_path: Path,
    field: str,
    wrong_value: str,
) -> None:
    manifest, raw = _write_normalization_fixture(tmp_path)
    normalized = tmp_path / "normalized" / "observations.jsonl"
    result = normalize_snapshot(manifest, raw, normalized)
    rows = [
        json.loads(line)
        for line in normalized.read_text(encoding="utf-8").splitlines()
    ]
    target_filename = rows[0]["provenance"]["source_artifact"]
    for row in rows:
        if row["provenance"]["source_artifact"] == target_filename:
            row["provenance"][field] = wrong_value
    normalized.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    forged_receipt = read_json(result["normalization_receipt"])
    forged_receipt["output_sha256"] = _sha(normalized)
    forged_receipt_path = tmp_path / f"forged-{field}.json"
    forged_receipt_path.write_text(json.dumps(forged_receipt), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="row provenance does not match exactly one locked primary artifact",
    ):
        verify_normalization_receipt(normalized, forged_receipt_path)


def test_multi_artifact_normalizer_rejects_missing_emitted_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, raw = _write_normalization_fixture(tmp_path)
    original = WormBaseWs298Adapter.iter_observations

    def missing_binding(
        self: WormBaseWs298Adapter,
        snapshot: object,
    ):
        for index, observation in enumerate(original(self, snapshot)):
            if index == 0:
                observation.provenance.source_artifact = "missing.gaf"
            yield observation

    monkeypatch.setattr(
        WormBaseWs298Adapter,
        "iter_observations",
        missing_binding,
    )
    with pytest.raises(ValueError, match="without an exact artifact-id or filename"):
        normalize_snapshot(
            manifest,
            raw,
            tmp_path / "missing-binding" / "observations.jsonl",
        )


@pytest.mark.parametrize("emitted_sha", ["0" * 64, None])
def test_multi_artifact_normalizer_rejects_wrong_or_missing_emitted_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    emitted_sha: str | None,
) -> None:
    manifest, raw = _write_normalization_fixture(tmp_path)
    original = WormBaseWs298Adapter.iter_observations

    def wrong_sha(
        self: WormBaseWs298Adapter,
        snapshot: object,
    ):
        for index, observation in enumerate(original(self, snapshot)):
            if index == 0:
                observation.provenance.checksum_sha256 = emitted_sha
            yield observation

    monkeypatch.setattr(WormBaseWs298Adapter, "iter_observations", wrong_sha)
    with pytest.raises(ValueError, match="checksum does not match"):
        normalize_snapshot(
            manifest,
            raw,
            tmp_path / "wrong-sha" / "observations.jsonl",
        )


def test_multi_artifact_normalizer_rejects_ambiguous_filename_binding(
    tmp_path: Path,
) -> None:
    manifest, raw = _write_normalization_fixture(tmp_path)
    payload = read_json(manifest)
    gene_spec = next(
        item for item in payload["artifacts"] if item["artifact_id"] == "gene-ids"
    )
    support_spec = next(
        item
        for item in payload["artifacts"]
        if item["artifact_id"] == "release-support"
    )
    support_spec.update(
        {
            "url": gene_spec["url"],
            "filename": gene_spec["filename"],
            "expected_sha256": gene_spec["expected_sha256"],
            "expected_size": gene_spec["expected_size"],
        }
    )
    ambiguous_manifest = tmp_path / "ambiguous-manifest.json"
    ambiguous_manifest.write_text(json.dumps(payload), encoding="utf-8")
    ambiguous_raw = tmp_path / "ambiguous-raw"
    fetch_release(ambiguous_manifest, ambiguous_raw)
    with pytest.raises(ValueError, match="ambiguous across the snapshot"):
        normalize_snapshot(
            ambiguous_manifest,
            ambiguous_raw,
            tmp_path / "ambiguous" / "observations.jsonl",
        )


def _write_qualification_fixture(root: Path) -> tuple[Path, dict[str, object]]:
    source = root / "source"
    source.mkdir()
    role_sources = {
        "gene_identifiers": FIXTURES / "wormbase_ws298_source_qualification_gene_ids.csv",
        "phenotype_associations": FIXTURES / "wormbase_ws298_source_qualification_phenotype.gaf",
        "development_associations": FIXTURES / "wormbase_ws298_source_qualification_development.gaf",
        "phenotype_ontology": FIXTURES / "wormbase_ws298_source_qualification_phenotype.obo",
        "development_ontology": FIXTURES / "wormbase_ws298_source_qualification_development.obo",
    }
    role_names = {
        "gene_identifiers": "gene_ids.csv",
        "phenotype_associations": "phenotype.gaf",
        "development_associations": "development.gaf",
        "phenotype_ontology": "phenotype.obo",
        "development_ontology": "development.obo",
    }
    for role, original in role_sources.items():
        (source / role_names[role]).write_bytes(original.read_bytes())
    letter = source / "letter.WS298"
    letter.write_text(
        "New release of WormBase WS298\n"
        "C. elegans gene data (49164 genes in total)\n",
        encoding="utf-8",
        newline="\n",
    )
    ledger_paths = [letter] + [source / name for name in role_names.values()]
    checksums = source / "CHECKSUMS"
    checksums.write_text(
        "".join(f"{_md5(path)}  {path.name}\n" for path in ledger_paths),
        encoding="utf-8",
        newline="\n",
    )

    formats = {
        "release_checksum_ledger": "md5sum",
        "release_letter": "plain_text",
        "gene_identifiers": "csv6",
        "phenotype_associations": "gaf2.0",
        "development_associations": "gaf2.0",
        "phenotype_ontology": "obo1.2",
        "development_ontology": "obo1.2",
    }
    role_to_path = {
        "release_checksum_ledger": checksums,
        "release_letter": letter,
        **{role: source / name for role, name in role_names.items()},
    }
    cleared = {"phenotype_ontology", "development_ontology"}
    artifacts = []
    for role, path in role_to_path.items():
        artifacts.append(
            {
                "artifact_id": f"fixture-{role.replace('_', '-')}",
                "role": role,
                "relative_path": path.name,
                "source_url": f"https://example.org/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
                "md5_in_release_ledger": (
                    None if role == "release_checksum_ledger" else _md5(path)
                ),
                "compression": "none",
                "format": formats[role],
                "rights_status": "cleared" if role in cleared else "review_required",
                "license_uri": (
                    "https://creativecommons.org/publicdomain/zero/1.0/"
                    if role in cleared
                    else None
                ),
                "external_transfer_permitted": role in cleared,
            }
        )
    config: dict[str, object] = {
        "schema_version": "wormctx-wormbase-ws298-source_qualification-source-contract-1.0",
        "analysis_id": "wormbase_ws298_source_qualification_fixture_qualification",
        "classification": "retrospective_source_qualification_only",
        "source_root_id": "wormbase_ws298_mirror",
        "provider": "WormBase",
        "release": "WS298",
        "release_is_immutable": True,
        "taxon": "NCBITaxon:6239",
        "bioproject": "NCBI:PRJNA13758",
        "genome_assembly": "WBcel235/GCA_000002985.3/ce11",
        "adapter": "wormbase_ws298_source_qualification",
        "adapter_version": "0.1.0",
        "scope": "gene_development_stage_phenotype",
        "total_expected_bytes": sum(path.stat().st_size for path in role_to_path.values()),
        "artifacts": artifacts,
        "expected_profile": {
            "gene_rows": 3,
            "phenotype_association_rows": 2,
            "phenotype_unique_genes": 2,
            "phenotype_unique_terms": 2,
            "phenotype_negated_rows": 1,
            "phenotype_non_not_qualifier_rows": 0,
            "development_association_rows": 2,
            "development_unique_genes": 2,
            "development_unique_terms": 2,
            "development_negated_rows": 0,
            "development_anatomy_term_qualified_rows": 0,
            "phenotype_ontology_terms": 2,
            "phenotype_ontology_obsolete_terms": 0,
            "phenotype_ontology_alternate_ids": 0,
            "phenotype_ontology_ambiguous_ids": 0,
            "phenotype_ontology_nameless_obsolete_terms": 0,
            "development_ontology_terms": 2,
            "development_ontology_obsolete_terms": 0,
            "development_ontology_alternate_ids": 0,
            "development_ontology_ambiguous_ids": 0,
            "development_ontology_nameless_obsolete_terms": 0,
            "unresolved_gene_references": 0,
            "unresolved_gene_rows": 0,
            "unresolved_phenotype_references": 0,
            "unresolved_development_references": 0,
            "ambiguous_referenced_phenotype_ids": 0,
            "ambiguous_referenced_development_ids": 0,
        },
        "required_gaf_date_generated": "2025-11-05",
        "empty_gaf_qualifier_receipt_token": "UNQUALIFIED",
        "required_ontology_data_versions": {
            "phenotype_ontology": "fixture/2026-07-22",
            "development_ontology": "fixture/2026-07-22",
        },
        "release_letter_expected_gene_total": 49164,
        "rights_policy": {
            "controlled_noncommercial_academic_local_use_attested": True,
            "external_compute_transfer_requires_all_artifacts_cleared": True,
            "redistribution_requires_all_artifacts_cleared": True,
        },
        "claim_boundary": {
            "biological_claims_permitted": False,
            "permitted": ["fixture source qualification"],
            "prohibited": ["biological inference"],
        },
    }
    return source, config


def test_qualification_writes_atomic_compact_receipt_and_detects_tampering(
    tmp_path: Path,
) -> None:
    source, config = _write_qualification_fixture(tmp_path)
    config_path = tmp_path / "wormbase_ws298_source_qualification_fixture_contract.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "qualified"
    receipt = qualify_source(config_path, source, output)
    assert receipt["profile"]["phenotype_association_rows"] == 2
    assert receipt["rights_gate"]["external_compute_transfer_permitted"] is False
    assert receipt["models_executed"] is False
    assert {path.name for path in output.iterdir()} == {
        "source_qualification_receipt.json",
        "SHA256SUMS.txt",
        "SUCCESS",
    }
    with pytest.raises(FileExistsError):
        qualify_source(config_path, source, output)

    changed = copy.deepcopy(config)
    changed["artifacts"][2]["sha256"] = "0" * 64
    changed_path = tmp_path / "wormbase_ws298_source_qualification_tampered_contract.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen bytes differ"):
        qualify_source(changed_path, source, tmp_path / "tampered-output")
    assert not (tmp_path / "tampered-output").exists()
