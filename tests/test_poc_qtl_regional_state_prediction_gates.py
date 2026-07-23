from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from wormctx.poc import qtl_regional_state_prediction_gates as gates
from wormctx.poc import qtl_heldout_prediction as heldout


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments"
    / "natural_variation"
    / "pangenome_state_qualification"
    / "config"
    / "state_input_handoff.json"
)


def test_real_handoff_manifest_freezes_partial_gate_state() -> None:
    manifest = gates.load_handoff_manifest(CONFIG)

    assert gates.gate_summary(manifest) == {
        "coordinate_identity": "qualified",
        "state_panel": "missing",
        "ldpruned_chr5_excluded_grm": "qualified",
        "full_kernels": "qualified",
        "population_groups": "qualified",
        "phenotype_provenance": "blocked_on_raw_measurement_provenance",
    }
    assert gates.coordinate_identity_ready(manifest.coordinate_identity)
    assert manifest.coordinate_identity.qualification_receipt is not None
    assert manifest.coordinate_identity.qualification_receipt.sha256 == (
        "16de24d5cd2721bc1a9cdec6868e212e62bd2d1e9b4279b5b6e5163f83329a2c"
    )
    assert manifest.classification == (
        "outcome_independent_execution_path_partial_real_input_qualification"
    )
    assert manifest.biological_claims_permitted is False
    assert manifest.null_resource_guard.heavyweight_derivations_released is True
    assert manifest.null_resource_guard.audit_observation == "terminal_success_checksum_qualified"


def test_manifest_does_not_relabel_small_variants_or_divergence_as_pav() -> None:
    manifest = gates.load_handoff_manifest(CONFIG)
    state = manifest.state_panel

    assert state.observed_small_variant_vcf_accepted_as_v1_haplotype_or_pav is False
    assert state.divergent_region_bed_accepted_as_presence_absence_calls is False
    assert state.qualified_panel_assets == []


def test_qualified_state_panel_cannot_use_an_arbitrary_nonempty_asset_list() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))["state_panel"]
    payload["status"] = "qualified"
    payload["qualified_panel_assets"] = [payload["observed_small_variant_vcf"]]
    payload["qualified_panel_evidence"] = {
        "schema_version": "wormctx-abamectin-haplotype-pav-panel-qualification-1.0",
        "samples": 209,
        "ordered_iid_sha256": "0" * 64,
        "state_count": 1,
        "complete_binary_or_categorical_calls": True,
        "outcome_blind": True,
    }

    with pytest.raises(ValidationError, match="exact frozen asset set"):
        gates.StatePanelGate.model_validate(payload)


def test_full_kernel_status_cannot_be_qualified_from_metadata_only() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))["full_kernels"]
    payload["status"] = "qualified"
    payload["qualified_output_assets"] = []
    payload["numerical_qualification_receipt"] = None

    with pytest.raises(ValidationError, match="exact frozen output asset set"):
        gates.FullKernelDerivationGate.model_validate(payload)


def test_qualified_full_kernel_manifest_freezes_real_outputs_and_receipt() -> None:
    gate = gates.load_handoff_manifest(CONFIG).full_kernels

    assert gate.status == "qualified"
    assert [asset.role for asset in gate.qualified_output_assets] == [
        "whole_genome_grm_bin",
        "whole_genome_grm_n_bin",
        "whole_genome_grm_id",
        "whole_genome_float64_npy",
        "genome_excluding_chrv_grm_bin",
        "genome_excluding_chrv_grm_n_bin",
        "genome_excluding_chrv_grm_id",
        "genome_excluding_chrv_float64_npy",
        "full_kernel_marker_manifest",
        "full_kernel_run_manifest",
        "full_kernel_checksum_receipt",
    ]
    assert gate.numerical_qualification_receipt is not None
    assert gate.numerical_qualification_receipt.sha256 == (
        "1841b916a43c04003da97d40f411175242d38420a6173b2f398510c2f42c983c"
    )


def test_coordinate_gate_refuses_interval_hash_mismatch() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))["coordinate_identity"]
    payload["source"] = json.loads(json.dumps(payload["target"]))
    payload["source"]["assembly"] = gates.WS276_ASSEMBLY
    payload["source"]["fasta"]["role"] = "ws276_reference_fasta"
    payload["source"]["fai"]["role"] = "ws276_reference_fai"
    payload["source"]["intervals"][0]["canonical_sequence_sha256"] = "0" * 64
    payload["status"] = "qualified"

    with pytest.raises(ValidationError, match="coordinate identity"):
        gates.CoordinateIdentityGate.model_validate(payload)


def test_coordinate_gate_accepts_only_complete_matching_interval_evidence() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))["coordinate_identity"]
    payload["source"] = json.loads(json.dumps(payload["target"]))
    payload["source"]["assembly"] = gates.WS276_ASSEMBLY
    payload["source"]["fasta"]["role"] = "ws276_reference_fasta"
    payload["source"]["fai"]["role"] = "ws276_reference_fai"
    payload["status"] = "qualified"

    gate = gates.CoordinateIdentityGate.model_validate(payload)
    assert gates.coordinate_identity_ready(gate)


@pytest.mark.skip(
    reason="the real coordinate receipt is intentionally excluded from the public tree"
)
def test_coordinate_receipt_is_frozen_and_verifiable_from_repository() -> None:
    manifest = gates.load_handoff_manifest(CONFIG)
    verified = gates.verify_frozen_assets(
        manifest,
        {"repository": ROOT},
        roles=["ws276_ws283_coordinate_identity_receipt"],
    )

    assert len(verified) == 1
    assert verified[0].bytes == 2117
    assert verified[0].sha256 == (
        "16de24d5cd2721bc1a9cdec6868e212e62bd2d1e9b4279b5b6e5163f83329a2c"
    )


def test_frozen_asset_verifier_checks_bytes_and_sha256(tmp_path: Path) -> None:
    payload = b"governed input\n"
    source = tmp_path / "asset.bin"
    source.write_bytes(payload)
    asset = gates.FrozenAsset(
        role="test_asset",
        root_id="test_root",
        relative_path="asset.bin",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    receipt = gates.verify_frozen_asset(asset, {"test_root": tmp_path})
    assert receipt.qualified

    source.write_bytes(payload + b"changed")
    with pytest.raises(ValueError, match="byte count mismatch"):
        gates.verify_frozen_asset(asset, {"test_root": tmp_path})


def test_frozen_asset_rejects_paths_outside_governed_root() -> None:
    with pytest.raises(ValidationError, match="governed root-relative"):
        gates.FrozenAsset(
            role="escape",
            root_id="root",
            relative_path="../escape",
            bytes=1,
            sha256="0" * 64,
        )


def test_group_derivation_is_deterministic_and_phenotype_free() -> None:
    rng = np.random.default_rng(1729)
    ids = [f"strain_{index:03d}" for index in range(40)]
    # The frozen method standardizes every PC before clustering.  Give every
    # synthetic PC genuine between-group structure so that standardization does
    # not turn eight noise-only columns into equal-weight clustering features.
    centers = np.asarray(
        [
            [np.sin((group + 1) * (column + 1)) for column in range(10)]
            for group in range(5)
        ],
        dtype=np.float64,
    ) * 12.0
    values = np.vstack(
        [centers[group] + rng.normal(0.0, 0.05, size=(8, 10)) for group in range(5)]
    )

    first, evaluations = gates.derive_groups_from_matrix(
        ids, values, candidate_k=(5, 6), minimum_group_size=5
    )
    second, repeated = gates.derive_groups_from_matrix(
        ids, values, candidate_k=(5, 6), minimum_group_size=5
    )

    np.testing.assert_array_equal(first, second)
    assert evaluations == repeated
    assert len(np.unique(first)) == 5
    assert min(np.bincount(first)) >= 5


def test_group_derivation_fails_when_no_candidate_meets_minimum_size() -> None:
    rng = np.random.default_rng(13)
    ids = [f"strain_{index:03d}" for index in range(20)]
    values = rng.normal(size=(20, 10))

    with pytest.raises(ValueError, match="no candidate grouping"):
        gates.derive_groups_from_matrix(ids, values, candidate_k=(5,), minimum_group_size=10)


def test_ldpruned_grm_is_bound_to_the_same_fam_and_order_as_grouping() -> None:
    manifest = gates.load_handoff_manifest(CONFIG)
    fam = manifest.full_kernels.input_files[2]
    verification = manifest.ldpruned_chr5_excluded_grm.verification

    assert verification.fam_sha256 == fam.sha256
    assert verification.ordered_iid_sha256 == manifest.population_groups.source_ordered_iid_sha256
    assert verification.minimum_eigenvalue >= -verification.psd_tolerance


@pytest.mark.skip(
    reason="the real population-group bundle is intentionally excluded from the public tree"
)
def test_real_population_group_bundle_is_frozen_and_outcome_blind() -> None:
    manifest = gates.load_handoff_manifest(CONFIG)
    assignments = ROOT / "mappings" / "abamectin_ws283_pc10_population_groups_20260722.tsv"
    receipt = (
        ROOT
        / "mappings"
        / "abamectin_ws283_pc10_population_groups_20260722.receipt.json"
    )

    payload = gates.verify_population_group_bundle(manifest, assignments, receipt)

    assert payload["selected_k"] == 5
    assert payload["group_sizes"] == {
        "POP01": 163,
        "POP02": 8,
        "POP03": 13,
        "POP04": 15,
        "POP05": 10,
    }
    assert payload["phenotype_values_accessed"] is False
    assert payload["schema_version"] == gates.GROUP_RECEIPT_VERSION
    assert payload["eigenvalue_count"] == 10
    assert payload["eigenvalues_used_for_clustering"] is False
    assert payload["axis_weighting"] == "equal_weight_after_per_pc_standardization"
    assert payload["outcome_access_claim_scope"] == "derivation_execution_path_only"
    assert payload["operator_outcome_blinding_asserted"] is False
    assert payload["labels_are_external_ancestry_assignments"] is False
    verified = gates.verify_frozen_assets(
        manifest,
        {"repository": ROOT},
        roles=[
            "phenotype_blind_population_groups",
            "phenotype_blind_population_group_receipt",
        ],
    )
    assert [item.sha256 for item in verified] == [
        "60fee103992f92e2bd2ee62eeba2426a882f36e250334d4814fc09084dfa87c3",
        "fc6c4ad412155c8c4a12f08b74a7b7881cd58334341bb0b8513ffdbf9e84f515",
    ]


def test_pca_eigenvalue_loader_rejects_corrupt_source(tmp_path: Path) -> None:
    valid = tmp_path / "valid.eigenval"
    valid.write_text(
        "\n".join(str(value) for value in range(10, 0, -1)) + "\n",
        encoding="utf-8",
    )
    np.testing.assert_array_equal(gates._load_pca_eigenvalues(valid), np.arange(10, 0, -1))

    invalid = tmp_path / "invalid.eigenval"
    invalid.write_text("10\n9\n8\n7\n6\n5\n4\n3\n2\n11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nonincreasing"):
        gates._load_pca_eigenvalues(invalid)


def _write_gcta_kernel(
    root: Path,
    name: str,
    fam_rows: list[tuple[str, str]],
    matrix: np.ndarray,
    markers: int,
) -> None:
    prefix = root / name
    lower = matrix[np.tril_indices(len(fam_rows))].astype("<f4")
    lower.tofile(Path(f"{prefix}.grm.bin"))
    np.full(lower.shape, markers, dtype="<f4").tofile(Path(f"{prefix}.grm.N.bin"))
    Path(f"{prefix}.grm.id").write_text(
        "".join(f"{fid}\t{iid}\n" for fid, iid in fam_rows), encoding="utf-8"
    )
    Path(f"{prefix}.log").write_text("synthetic GCTA log\n", encoding="utf-8")


def test_gcta_kernel_qualification_preserves_raw_matrix_and_heldout_prediction_accepts_it(
    tmp_path: Path,
) -> None:
    fam_rows = [(f"F{index:03d}", f"I{index:03d}") for index in range(209)]
    rng = np.random.default_rng(709)
    features = rng.normal(size=(209, 12))
    matrix = features @ features.T / features.shape[1] + np.eye(209) * 0.25
    _write_gcta_kernel(tmp_path, "whole_genome", fam_rows, matrix, 373279)

    observed, receipt = gates._qualify_gcta_kernel(
        tmp_path / "whole_genome", fam_rows, 373279
    )

    np.testing.assert_array_equal(observed, observed.T)
    assert receipt["samples"] == 209
    assert receipt["pairwise_marker_count_min"] == 373279
    assert receipt["pairwise_marker_count_max"] == 373279
    assert receipt["minimum_eigenvalue"] >= -receipt["psd_tolerance"]
    np.testing.assert_array_equal(
        heldout._validate_kernel("whole", observed, 209), observed
    )


def test_gcta_kernel_qualification_rejects_pairwise_count_drift(tmp_path: Path) -> None:
    fam_rows = [(f"F{index:03d}", f"I{index:03d}") for index in range(209)]
    _write_gcta_kernel(tmp_path, "genome_excluding_chrv", fam_rows, np.eye(209), 258096)
    counts_path = tmp_path / "genome_excluding_chrv.grm.N.bin"
    counts = np.fromfile(counts_path, dtype="<f4")
    counts[0] -= 1
    counts.tofile(counts_path)

    with pytest.raises(ValueError, match="pairwise marker counts"):
        gates._qualify_gcta_kernel(
            tmp_path / "genome_excluding_chrv", fam_rows, 258096
        )


def test_full_kernel_bundle_is_write_once_float64_and_heldout_prediction_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fam = tmp_path / "cohort.fam"
    bim = tmp_path / "cohort.bim"
    fam.write_text("synthetic\n", encoding="utf-8")
    bim.write_text("synthetic\n", encoding="utf-8")
    marker_root = tmp_path / "markers"
    marker_root.mkdir()
    (marker_root / "marker_manifest.json").write_text("{}\n", encoding="utf-8")
    (marker_root / "SUCCESS").write_text("qualified\n", encoding="utf-8")
    fam_rows = [(f"F{index:03d}", f"I{index:03d}") for index in range(209)]
    marker_receipt = {"ordered_fid_iid_sha256": "0" * 64}
    monkeypatch.setattr(
        gates,
        "_load_marker_manifest",
        lambda *_args: (marker_receipt, fam_rows),
    )
    grm_root = tmp_path / "grms"
    grm_root.mkdir()
    rng = np.random.default_rng(283)
    first_features = rng.normal(size=(209, 18))
    second_features = rng.normal(size=(209, 14))
    matrices = {
        "whole_genome": first_features @ first_features.T / 18 + np.eye(209),
        "genome_excluding_chrv": second_features @ second_features.T / 14 + np.eye(209),
    }
    _write_gcta_kernel(grm_root, "whole_genome", fam_rows, matrices["whole_genome"], 373279)
    _write_gcta_kernel(
        grm_root,
        "genome_excluding_chrv",
        fam_rows,
        matrices["genome_excluding_chrv"],
        258096,
    )
    output = tmp_path / "qualified"

    receipt = gates.qualify_full_kernels(
        CONFIG, fam, bim, marker_root, grm_root, output
    )

    assert receipt["schema_version"] == gates.KERNEL_QUALIFICATION_VERSION
    assert (output / "SUCCESS").read_text(encoding="utf-8") == "qualified\n"
    for name in ("whole_genome", "genome_excluding_chrv"):
        matrix = np.load(output / f"{name}.npy", allow_pickle=False)
        assert matrix.dtype == np.float64
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(heldout._validate_kernel(name, matrix, 209), matrix)
    with pytest.raises(FileExistsError, match="already exists"):
        gates.qualify_full_kernels(CONFIG, fam, bim, marker_root, grm_root, output)
