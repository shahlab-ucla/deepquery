from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from wormctx.poc import developmental_readiness as readiness
from wormctx.poc.developmental_omix import _write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/developmental_genetics/semantic_stage_and_lineage_preparation/config/readiness_gate.json"
STAGE_CONTRACT = ROOT / "experiments/developmental_genetics/semantic_stage_and_lineage_preparation/config/semantic_stage_adapter.json"
BASELINE_CONTRACT = ROOT / "experiments/developmental_genetics/gene_disjoint_outcome_prediction/config/gene_disjoint_baselines.json"
OVERLAP_REQUEST = (
    ROOT / "experiments/developmental_genetics/semantic_stage_and_lineage_preparation/config/overlap_resolution_template.json"
)
STAGE_ROOT = ROOT / "build/developmental_source-open-v11"
SPLIT = ROOT / "build/omix709-s1s4-qualification-20260721/split_manifest.json"
REAL_INPUTS = (STAGE_ROOT, SPLIT)


def test_readiness_contract_preserves_open_overlap_and_outcome_boundary() -> None:
    contract = readiness.load_contract(CONFIG)

    assert contract["stage_adapter"]["contract_sha256"] == (
        "d89471fa5c12856eb83353a93bc6c3ae8dc082dbb6ace97b1c8b2dd857a1e120"
    )
    assert contract["baseline"]["required_stage_contract_sha256"] == (
        contract["stage_adapter"]["contract_sha256"]
    )
    assert contract["overlap"]["public_records_establish_biological_embryo_disjointness"] is False
    assert contract["overlap"]["gene_quarantine_alone_closes_current_contract"] is False
    assert contract["preparation"]["topology_coordinates_may_read_outcomes"] is False
    assert contract["sealed_test_access_permitted"] is False


def test_contract_rejects_silent_gene_quarantine_amendment(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["overlap"]["gene_quarantine_alone_closes_current_contract"] = True
    changed = tmp_path / "changed.json"
    _write_json_atomic(changed, payload)

    with pytest.raises(ValueError, match="overlap policy differs"):
        readiness.load_contract(changed)


def test_real_topology_coordinates_are_outcome_blind_and_hash_frozen() -> None:
    if not all(path.exists() for path in REAL_INPUTS):
        pytest.skip("local governed developmental source qualification artifacts are not installed")
    lineage = json.loads((STAGE_ROOT / "lineage_manifest.json").read_text(encoding="utf-8"))
    stages = json.loads((STAGE_ROOT / "stage_membership.json").read_text(encoding="utf-8"))
    coordinates = readiness.topology_coordinates_from_stage(lineage, stages)

    assert coordinates["receipt"] == {
        "cell_count_through_200": 399,
        "edge_count_through_200": 398,
        "input_frontier_count": 26,
        "input_born_count": 50,
        "endpoint_frontier_count": 200,
        "maximum_depth": 9,
        "arrays_sha256": "1829d7258cdb3140160d65c3529ff3f1185d028c48b39190b2726cb8aa256ded",
        "outcome_values_read": False,
    }
    assert sum(coordinates["born_by_input_mask"]) == 50
    assert sum(coordinates["endpoint_frontier_mask"]) == 200

    contaminated = dict(stages, perturbation_measurement_values_read=True)
    with pytest.raises(ValueError, match="outcome-blind"):
        readiness.topology_coordinates_from_stage(lineage, contaminated)


def test_real_split_is_exact_whole_gene_and_pilot_quarantined() -> None:
    if not SPLIT.exists():
        pytest.skip("local governed split manifest is not installed")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    contract = readiness.load_contract(CONFIG)
    result = readiness._validate_split(split, contract["split"])

    assert result["partition_counts"] == {
        "train": {"genes": 410, "embryos": 1229},
        "validation": {"genes": 137, "embryos": 407},
        "sealed_test": {"genes": 137, "embryos": 398},
        "development_only_exposed_pilot": {"genes": 14, "embryos": 41},
    }
    assert result["total_genes"] == 698
    assert result["total_embryos"] == 2075
    assert result["pilot_exposed_genes_permitted_in_modeling"] is False

    changed = json.loads(json.dumps(split))
    changed["development_only_exposed_pilot_genes"][0]["embryo_ids"][0] = (
        changed["claim_bearing_genes"][0]["embryo_ids"][0]
    )
    with pytest.raises(ValueError, match="overlap split partitions"):
        readiness._validate_split(changed, contract["split"])


def test_real_readiness_bundle_is_blocked_but_prepared(tmp_path: Path) -> None:
    if not all(path.exists() for path in REAL_INPUTS):
        pytest.skip("local governed developmental source qualification artifacts are not installed")
    output = tmp_path / "ready"
    result = readiness.prepare_readiness(
        CONFIG,
        STAGE_CONTRACT,
        STAGE_ROOT,
        SPLIT,
        BASELINE_CONTRACT,
        OVERLAP_REQUEST,
        output,
    )

    assert result["safe_preparation_complete"] is True
    assert result["execution_status"] == "blocked_pending_pilot_embryo_overlap_receipt"
    assert result["next_modeling_step_permitted"] is False
    assert result["gates"]["topology_coordinates_materialized"] is True
    assert result["gates"]["outcome_values_read"] is False
    assert result["gates"]["sealed_test_opened"] is False
    assert readiness.verify_readiness(output, CONFIG)["verified"] is True


def test_readiness_verifier_rejects_posthoc_status_change(tmp_path: Path) -> None:
    source = ROOT / "build/omix709-developmental-readiness-20260722-v3"
    if not source.exists():
        pytest.skip("local readiness bundle is not installed")
    copied = tmp_path / "copied"
    shutil.copytree(source, copied)
    payload = json.loads((copied / "readiness.json").read_text(encoding="utf-8"))
    payload["next_modeling_step_permitted"] = True
    (copied / "readiness.json").unlink()
    _write_json_atomic(copied / "readiness.json", payload)

    with pytest.raises(ValueError, match="checksum failed"):
        readiness.verify_readiness(copied, CONFIG)
