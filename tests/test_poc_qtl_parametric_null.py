from __future__ import annotations

import array
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from wormctx.poc import qtl_parametric_null as null


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/natural_variation/parametric_polygenic_null_calibration/config/parametric_null.json"


def test_manifest_freezes_three_parents_sixteen_cells_and_1600_maps() -> None:
    manifest = null.load_manifest(CONFIG)

    assert [item.role for item in manifest.parents] == [
        "baseline",
        "calibration",
        "sensitivity",
    ]
    assert [item.id for item in manifest.relationship_models] == ["full", "ldpruned"]
    assert [item.id for item in manifest.endpoints] == ["pc0", "pc10"]
    assert len(null.iter_cells(manifest)) == 16
    assert manifest.bootstrap.replicates_per_cell == 100
    assert manifest.bootstrap.total_null_maps == 1600
    assert manifest.bootstrap.sentinel_replicates == [1, 25, 50, 75, 100]
    assert manifest.bootstrap.smoke_gate.replicates == [1, 2]
    assert manifest.bootstrap.smoke_gate.required_checkpoints == 32
    assert manifest.reml.expected_boundary_cells == [
        "ldpruned_pc0_norm_n",
        "ldpruned_pc10_norm_n",
    ]
    assert manifest.bootstrap.pooled_thresholds_permitted is False

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        null.ParametricNullManifest.model_validate(payload)

    payload.pop("unexpected")
    payload["bootstrap"]["shared_stream_key_fields"].insert(0, "relationship_kind")
    with pytest.raises(ValidationError, match="exclude relationship kind"):
        null.ParametricNullManifest.model_validate(payload)


def _independent_seed_digest(master: str, trait_slug: str, replicate: int, component: str) -> str:
    parts = (
        b"wormctx-parametric-null-seed-v1",
        bytes.fromhex(master),
        trait_slug.encode(),
        str(replicate).encode("ascii"),
        component.encode("ascii"),
    )
    material = b"".join(len(part).to_bytes(8, "big") + part for part in parts)
    return hashlib.sha256(material).hexdigest()


def test_sha256_pcg64dxsm_seed_is_order_independent_and_common_across_cells() -> None:
    master = null.load_manifest(CONFIG).bootstrap.master_seed_sha256
    first = null.derive_stream_seed(master, "mean_EXT", 25, "genetic")
    replay = null.derive_stream_seed(master, "mean_EXT", 25, "genetic")
    residual = null.derive_stream_seed(master, "mean_EXT", 25, "residual")

    assert first == replay
    assert first["seed_sha256"] == _independent_seed_digest(
        master, "mean_EXT", 25, "genetic"
    )
    assert first["shared_stream_group"] == "mean_EXT/replicate-025"
    assert first["kind_and_endpoint_excluded"] is True
    assert first["seed_sha256"] != residual["seed_sha256"]


def _write_float32(path: Path, values: list[float]) -> None:
    with path.open("wb") as handle:
        array.array("f", values).tofile(handle)


def _small_grm(tmp_path: Path) -> tuple[Path, Path]:
    fam = tmp_path / "cohort.fam"
    fam.write_text(
        "0 S0 0 0 0 -9\n0 S1 0 0 0 -9\n0 S2 0 0 0 -9\n",
        encoding="utf-8",
        newline="\n",
    )
    prefix = tmp_path / "anchor"
    Path(f"{prefix}.grm.id").write_text(
        "0\tS0\n0\tS1\n0\tS2\n", encoding="utf-8", newline="\n"
    )
    _write_float32(Path(f"{prefix}.grm.bin"), [1.0, 0.2, 1.0, 0.1, 0.3, 1.0])
    _write_float32(Path(f"{prefix}.grm.N.bin"), [5.0] * 6)
    return fam, prefix


def test_read_gcta_grm_binds_float32_matrix_and_exact_fam_order(tmp_path: Path) -> None:
    fam, prefix = _small_grm(tmp_path)
    grm = null.read_gcta_grm(prefix, fam, 5, expected_samples=3)

    np.testing.assert_allclose(
        grm.matrix,
        np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]]),
        atol=1e-7,
    )
    assert grm.ids == (("0", "S0"), ("0", "S1"), ("0", "S2"))
    assert grm.receipt["positive_semidefinite_within_float32_tolerance"] is True
    assert grm.receipt["numpy_version"] == np.__version__
    assert len(grm.receipt["k_half_eigen_factor_sha256"]) == 64

    Path(f"{prefix}.grm.id").write_text(
        "0\tS1\n0\tS0\n0\tS2\n", encoding="utf-8", newline="\n"
    )
    with pytest.raises(ValueError, match="IDs/order"):
        null.read_gcta_grm(prefix, fam, 5, expected_samples=3)


def test_parse_reml_and_gls_require_interior_constrained_fit(tmp_path: Path) -> None:
    hsq = tmp_path / "fit.hsq"
    hsq.write_text(
        "Source\tVariance\tSE\n"
        "V(G)\t0.4\t0.1\n"
        "V(e)\t0.6\t0.1\n"
        "Vp\t1.0\t0.1\n"
        "V(G)/Vp\t0.4\t0.1\n"
        "logL\t-10.5\n"
        "logL0\t-12.0\n"
        "LRT\t3.0\n"
        "df\t1\n"
        "Pval\t0.04\n"
        "n\t3\n"
        "\n"
        "Fix_eff\tSE\n"
        "3.0\t0.2\n",
        encoding="utf-8",
        newline="\n",
    )
    fit = null.parse_reml_hsq(hsq, 3)
    assert fit.variance_genetic == 0.4
    assert fit.variance_residual == 0.6
    assert fit.fixed_effects == (3.0,)
    y = np.array([2.0, 3.0, 4.0])
    x = np.column_stack([np.ones(3), np.array([-1.0, 0.0, 1.0])])
    beta, condition = null.gls_fixed_effects(y, x, np.eye(3), 0.4, 0.6)
    np.testing.assert_allclose(beta, [3.0, 1.0], atol=1e-12)
    assert condition >= 1.0

    hsq.write_text(
        hsq.read_text(encoding="utf-8").replace("V(G)\t0.4", "V(G)\t0.0", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="constrained null model"):
        null.parse_reml_hsq(hsq, 3)


def test_reml_log_requires_exactly_the_two_preflight_boundary_cells(tmp_path: Path) -> None:
    manifest = null.load_manifest(CONFIG)
    prefix = tmp_path / "reml"
    phenotype = tmp_path / "trait.phen"
    grm = tmp_path / "grm"
    log = tmp_path / "reml.log"
    log.write_text(
        "Accepted options:\n"
        "--reml\n"
        f"--grm {grm}\n"
        f"--pheno {phenotype}\n"
        "--reml-est-fix\n"
        "--thread-num 16\n"
        f"--out {prefix}\n"
        "\n"
        "Log-likelihood converged.\n"
        "(1 component(s) constrained)\n",
        encoding="utf-8",
    )
    endpoint = manifest.endpoints[0]
    receipt = null._qualify_reml_log(
        log, grm, phenotype, None, prefix, endpoint, 16, True
    )
    assert receipt["anchor_boundary"] is True
    assert receipt["boundary_evidence"] == ["(1 component(s) constrained)"]

    with pytest.raises(ValueError, match="unexpected"):
        null._qualify_reml_log(log, grm, phenotype, None, prefix, endpoint, 16, False)


def _manual_identity_grm() -> null.GctaGrm:
    matrix = np.eye(3)
    return null.GctaGrm(
        prefix=Path("identity"),
        ids=(("0", "S0"), ("0", "S1"), ("0", "S2")),
        matrix=matrix,
        eigenvalues=np.ones(3),
        eigenvectors=np.eye(3),
        expected_markers=5,
        receipt={"k_half_eigen_factor_sha256": "a" * 64},
    )


def test_explicit_polygenic_null_formula_replays_exact_float64_bytes() -> None:
    manifest = null.load_manifest(CONFIG)
    design = np.ones((3, 1))
    beta = np.array([2.5])
    grm = _manual_identity_grm()
    first, first_receipt = null.simulate_phenotype_values(
        design,
        beta,
        grm,
        0.4,
        0.6,
        manifest.bootstrap.master_seed_sha256,
        "norm_n",
        1,
    )
    replay, replay_receipt = null.simulate_phenotype_values(
        design,
        beta,
        grm,
        0.4,
        0.6,
        manifest.bootstrap.master_seed_sha256,
        "norm_n",
        1,
    )

    assert np.array_equal(first, replay)
    assert first_receipt == replay_receipt
    assert null._phenotype_bytes(grm.ids, first) == null._phenotype_bytes(grm.ids, replay)
    assert first_receipt["stream_seeds"]["genetic"]["kind_and_endpoint_excluded"] is True


def _small_stream_manifest() -> null.ParametricNullManifest:
    manifest = null.load_manifest(CONFIG).model_copy(deep=True)
    manifest.diagnostics.marker_count = 6
    return manifest


def _write_small_map_and_bim(tmp_path: Path, p_values: list[float]) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bim = tmp_path / "markers.bim"
    result = tmp_path / "map.mlma"
    bim.write_text(
        "".join(
            f"{chromosome}\tm{chromosome}\t0\t{chromosome * 100}\tA\tG\n"
            for chromosome in range(1, 7)
        ),
        encoding="utf-8",
        newline="\n",
    )
    rows = ["\t".join(null.MLMA_HEADER)]
    rows.extend(
        f"{chromosome}\tm{chromosome}\t{chromosome * 100}\tA\tG\t0.2\t0.1\t0.2\t"
        f"{p_values[chromosome - 1]}"
        for chromosome in range(1, 7)
    )
    result.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    return result, bim


def test_streaming_map_diagnostics_validate_order_positive_p_and_exact_lambda(
    tmp_path: Path,
) -> None:
    p_values = [0.01, 0.05, 0.2, 0.4, 0.8, 1.0]
    result, bim = _write_small_map_and_bim(tmp_path, p_values)
    diagnostics = null.stream_map_diagnostics(
        result, bim, "mean.EXT", _small_stream_manifest()
    )
    brute = []
    for p_value in p_values:
        z_score = null.NormalDist().inv_cdf(p_value / 2.0)
        brute.append(z_score * z_score)
    assert diagnostics["markers"] == 6
    assert diagnostics["minimum_p"]["snp"] == "m1"
    assert diagnostics["lambda_gc"] == pytest.approx(
        float(np.median(brute)) / null._CHI_SQUARE_1_MEDIAN
    )
    assert len(diagnostics["marker_allele_frequency_signature_sha256"]) == 64

    invalid, invalid_bim = _write_small_map_and_bim(tmp_path / "invalid", p_values)
    invalid.write_text(
        invalid.read_text(encoding="utf-8").replace("\t0.01\n", "\t0\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid estimate"):
        null.stream_map_diagnostics(
            invalid, invalid_bim, "mean.EXT", _small_stream_manifest()
        )


def test_plus_one_tail_and_nominal_fwer_primitives_are_cell_specific() -> None:
    null_values = [index / 1000.0 for index in range(1, 101)]
    tail = null.plus_one_tail(null_values, 0.005, "lower")

    assert tail["null_tail_events"] == 5
    assert tail["plus_one_empirical_p"] == pytest.approx(6 / 101)
    assert tail["replicates"] == 100
    assert tail["monte_carlo_interval_for_raw_event_rate"]["trials"] == 100
    assert sorted(null_values)[4] == 0.005


def test_cli_exposes_frozen_execution_subcommands() -> None:
    parser = null._parser()
    help_text = parser.format_help()
    for command in (
        "verify-parents",
        "verify-grm",
        "qualify-fit",
        "simulate",
        "qualify-map",
        "aggregate",
    ):
        assert command in help_text


@pytest.mark.skip(
    reason="the site-specific runner is intentionally absent from the public tree"
)
def test_runner_removes_only_empty_scratch_directories_before_terminal_gate() -> None:
    script = (
        ROOT / "scripts" / "run_abamectin_qtl_ws283_parametric_null.sh"
    ).read_text(encoding="utf-8")
    cleanup = (
        'find "$RUN_ROOT/scratch" -mindepth 1 -depth -type d -empty -delete'
    )
    fail_closed_gate = (
        'if find "$RUN_ROOT/scratch" -mindepth 1 -print -quit | grep -q .; then'
    )

    assert script.count(cleanup) == 1
    assert script.index(cleanup) < script.index(fail_closed_gate)
    assert 'rm -rf -- "$RUN_ROOT/scratch"' not in script
    assert "any file,\n# symlink, or non-empty directory remains visible" in script
