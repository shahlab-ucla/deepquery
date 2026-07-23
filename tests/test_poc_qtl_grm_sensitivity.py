from __future__ import annotations

import array
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wormctx.poc import qtl_grm_sensitivity as grm
from wormctx.poc.qtl import MLMA_HEADER, TRAITS, TRAIT_SLUGS, _MlmaRow


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments"
    / "natural_variation"
    / "chromosome_excluded_relationship_sensitivity"
    / "config"
    / "relationship_sensitivity.json"
)
RUNNER = ROOT / "scripts" / "run_abamectin_qtl_ws283_ldpruned_grm_sensitivity.sh"
DEPLOYER = ROOT / "scripts" / "deploy_ws283_ldpruned_grm_sensitivity.ps1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_manifest_freezes_dual_parents_six_grms_eight_maps_and_claim_boundary(
    tmp_path: Path,
) -> None:
    manifest = grm.load_grm_manifest(CONFIG)

    assert manifest.baseline_parent.post_qc_samples == 209
    assert manifest.baseline_parent.post_qc_markers == 373_279
    assert manifest.baseline_parent.checksum_file_count == 44
    assert manifest.calibration_parent.ld_pruned_markers == 1_370
    assert manifest.calibration_parent.checksum_file_count == 150
    baseline_files = {item.relative_path for item in manifest.baseline_parent.required_files}
    calibration_files = {
        item.relative_path for item in manifest.calibration_parent.required_files
    }
    assert "receipts/SHA256SUMS.txt" in baseline_files
    assert "receipts/deployer-runner-exit-code.txt" in calibration_files
    assert "logs/checksums.verify.log" in calibration_files
    assert [item.chromosome for item in manifest.relationship_matrix.chromosomes] == list(
        grm.CHROMOSOMES
    )
    assert [item.relationship_markers for item in manifest.relationship_matrix.chromosomes] == [
        1_224,
        970,
        1_128,
        1_221,
        1_080,
        1_227,
    ]
    assert all(
        len(item.association_list_sha256) == 64
        and len(item.relationship_list_sha256) == 64
        for item in manifest.relationship_matrix.chromosomes
    )
    assert manifest.association.chromosome_scan_count == 48
    assert manifest.association.assembled_map_count == 8
    assert [item.id for item in manifest.association.models] == [
        "ldgrm_pc0",
        "ldgrm_pc10",
    ]
    assert manifest.association.models[0].qcovar_relative_path is None
    assert manifest.association.models[1].qcovar_relative_path.endswith("pc10.qcovar")
    assert manifest.association.thread_count == 16
    assert manifest.association.model_selection_from_results_permitted is False
    assert manifest.biological_claims_permitted is False

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["association"]["thread_count"] = 8
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        grm.load_grm_manifest(changed)


def test_manifest_rejects_marker_hash_tamper_and_extra_fields(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["relationship_matrix"]["chromosomes"][0][
        "association_list_sha256"
    ] = "A" * 64
    changed = tmp_path / "uppercase-hash.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="sha256"):
        grm.load_grm_manifest(changed)

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["association"]["models"][0]["unfrozen_choice"] = True
    changed = tmp_path / "extra-field.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="extra"):
        grm.load_grm_manifest(changed)


def test_parent_gate_rejects_success_with_coexisting_failure(tmp_path: Path) -> None:
    manifest = grm.load_grm_manifest(CONFIG)
    baseline = tmp_path / manifest.baseline_parent.run_id
    calibration = tmp_path / manifest.calibration_parent.run_id
    baseline.mkdir()
    calibration.mkdir()
    for parent in (baseline, calibration):
        (parent / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
    (calibration / "FAILURE").write_text("exit_code\t1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="FAILURE"):
        grm.verify_parents(CONFIG, baseline, calibration)


def test_internal_checksum_verification_is_exact_and_fail_closed(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "nested/second.txt"
    second.parent.mkdir()
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    receipt = tmp_path / "SHA256SUMS.txt"
    receipt.write_text(
        f"{_sha256(first)}  ./first.txt\n{_sha256(second)}  ./nested/second.txt\n",
        encoding="utf-8",
    )

    result = grm._verify_internal_checksums(tmp_path, "SHA256SUMS.txt", 2)
    assert result["verified_file_count"] == 2

    second.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        grm._verify_internal_checksums(tmp_path, "SHA256SUMS.txt", 2)

    receipt.write_text(f"{_sha256(first)}  ./../first.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="wrong entry count|unsafe checksum path"):
        grm._verify_internal_checksums(tmp_path, "SHA256SUMS.txt", 1)


def _write_small_bim_and_prune(tmp_path: Path) -> tuple[Path, Path]:
    bim = tmp_path / "fixture.bim"
    bim_lines: list[str] = []
    for chromosome in grm.CHROMOSOMES:
        bim_lines.extend(
            [
                f"{chromosome}\tm{chromosome}a\t0\t{chromosome * 100 + 1}\tA\tG",
                f"{chromosome}\tm{chromosome}b\t0\t{chromosome * 100 + 2}\tC\tT",
            ]
        )
    bim.write_text("\n".join(bim_lines) + "\n", encoding="utf-8", newline="\n")

    # Deliberately reverse this list: every derived list must use baseline BIM order.
    prune = tmp_path / "fixture.prune.in"
    prune.write_text(
        "\n".join(f"m{chromosome}a" for chromosome in reversed(grm.CHROMOSOMES))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return bim, prune


def _small_manifest(bim: Path, prune: Path) -> grm.GrmSensitivityManifest:
    manifest = grm.load_grm_manifest(CONFIG)
    manifest.baseline_parent.post_qc_samples = 3
    manifest.baseline_parent.post_qc_markers = 12
    manifest.calibration_parent.post_qc_samples = 3
    manifest.calibration_parent.post_qc_markers = 12
    manifest.calibration_parent.ld_pruned_markers = 6
    manifest.relationship_matrix.total_ld_pruned_markers = 6

    for item in manifest.baseline_parent.required_files:
        if item.relative_path == "genotype/abamectin_209_qc.bim":
            item.bytes = bim.stat().st_size
            item.sha256 = _sha256(bim)
    for item in manifest.calibration_parent.required_files:
        if item.relative_path == "genotype/ws283_ld_pruned.prune.in":
            item.bytes = prune.stat().st_size
            item.sha256 = _sha256(prune)

    for contract in manifest.relationship_matrix.chromosomes:
        chromosome = contract.chromosome
        contract.association_markers = 2
        contract.pruned_on_chromosome = 1
        contract.relationship_markers = 5
        candidates = f"m{chromosome}a\nm{chromosome}b\n"
        relationship = "".join(
            f"m{other}a\n" for other in grm.CHROMOSOMES if other != chromosome
        )
        contract.association_list_sha256 = _sha256_text(candidates)
        contract.relationship_list_sha256 = _sha256_text(relationship)
    return manifest


def _prepare_small_marker_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[grm.GrmSensitivityManifest, Path, Path, Path]:
    bim, prune = _write_small_bim_and_prune(tmp_path)
    manifest = _small_manifest(bim, prune)
    monkeypatch.setattr(grm, "load_grm_manifest", lambda _: manifest)
    output = tmp_path / "marker-lists"
    grm.prepare_marker_lists(CONFIG, bim, prune, output)
    return manifest, bim, prune, output


def test_prepare_marker_lists_builds_six_exact_disjoint_complements_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, output = _prepare_small_marker_lists(tmp_path, monkeypatch)
    receipt = json.loads((output / "marker_manifest.json").read_text(encoding="utf-8"))

    assert receipt["association_markers"] == 12
    assert receipt["ld_pruned_markers"] == 6
    assert len(receipt["chromosomes"]) == 6
    assert (output / "association/candidate_chr1.snplist").read_text(
        encoding="utf-8"
    ) == "m1a\nm1b\n"
    assert (output / "relationship/grm_exclude_chr1.snplist").read_text(
        encoding="utf-8"
    ) == "m2a\nm3a\nm4a\nm5a\nm6a\n"
    for chromosome_receipt in receipt["chromosomes"]:
        chromosome = chromosome_receipt["chromosome"]
        candidates = set(
            (output / f"association/candidate_chr{chromosome}.snplist")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        relationship = set(
            (output / f"relationship/grm_exclude_chr{chromosome}.snplist")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(candidates) == 2
        assert len(relationship) == 5
        assert candidates.isdisjoint(relationship)

    with pytest.raises(FileExistsError, match="must not exist"):
        grm.prepare_marker_lists(
            CONFIG,
            tmp_path / "fixture.bim",
            tmp_path / "fixture.prune.in",
            output,
        )


def test_prepare_marker_lists_rejects_unknown_pruned_marker_without_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bim, prune = _write_small_bim_and_prune(tmp_path)
    prune.write_text("m1a\nm2a\nm3a\nm4a\nm5a\nunknown\n", encoding="utf-8")
    manifest = _small_manifest(bim, prune)
    monkeypatch.setattr(grm, "load_grm_manifest", lambda _: manifest)
    output = tmp_path / "marker-lists"

    with pytest.raises(ValueError, match="absent from BIM"):
        grm.prepare_marker_lists(CONFIG, bim, prune, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".marker-lists.tmp-*"))


def _write_fam(path: Path) -> None:
    path.write_text(
        "".join(f"0 S{sample} 0 0 0 -9\n" for sample in range(3)),
        encoding="utf-8",
        newline="\n",
    )


def _write_float32(path: Path, values: list[float]) -> None:
    with path.open("wb") as handle:
        array.array("f", values).tofile(handle)


def _write_small_grms(root: Path) -> None:
    root.mkdir()
    ids = "0\tS0\n0\tS1\n0\tS2\n"
    identity_lower_triangle = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    for chromosome in grm.CHROMOSOMES:
        prefix = root / f"leave_chr{chromosome}_out"
        Path(f"{prefix}.grm.id").write_text(ids, encoding="utf-8", newline="\n")
        _write_float32(Path(f"{prefix}.grm.bin"), identity_lower_triangle)
        _write_float32(Path(f"{prefix}.grm.N.bin"), [5.0] * 6)


def test_verify_grms_binds_six_binary_matrices_to_fam_and_marker_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _, _, marker_root = _prepare_small_marker_lists(tmp_path, monkeypatch)
    fam = tmp_path / "fixture.fam"
    _write_fam(fam)
    for item in manifest.baseline_parent.required_files:
        if item.relative_path == "genotype/abamectin_209_qc.fam":
            item.bytes = fam.stat().st_size
            item.sha256 = _sha256(fam)
    grm_root = tmp_path / "grms"
    _write_small_grms(grm_root)

    receipt = grm.verify_grms(
        CONFIG,
        fam,
        marker_root / "marker_manifest.json",
        grm_root,
        tmp_path / "grm-verification.json",
    )

    assert receipt["samples"] == 3
    assert len(receipt["grms"]) == 6
    assert [item["excluded_chromosome"] for item in receipt["grms"]] == list(
        grm.CHROMOSOMES
    )
    assert all(item["relationship_markers"] == 5 for item in receipt["grms"])
    assert all(item["triangular_float32_entries"] == 6 for item in receipt["grms"])
    assert all(
        item["positive_semidefinite_within_float32_tolerance"] is True
        for item in receipt["grms"]
    )


def test_verify_grms_rejects_altered_marker_manifest_and_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, marker_root = _prepare_small_marker_lists(tmp_path, monkeypatch)
    fam = tmp_path / "fixture.fam"
    _write_fam(fam)
    grm_root = tmp_path / "grms"
    _write_small_grms(grm_root)
    marker_manifest = marker_root / "marker_manifest.json"
    payload = json.loads(marker_manifest.read_text(encoding="utf-8"))
    payload["chromosomes"][0]["relationship_markers"] = 4
    marker_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen contract"):
        grm.verify_grms(
            CONFIG,
            fam,
            marker_manifest,
            grm_root,
            tmp_path / "altered-manifest.json",
        )

    # Restore the content-addressed receipt, then alter one of the files it binds.
    payload["chromosomes"][0]["relationship_markers"] = 5
    marker_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    relationship = marker_root / "relationship/grm_exclude_chr1.snplist"
    relationship.write_text("m2a\nm3a\nm4a\nm5a\ntampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="altered"):
        grm.verify_grms(
            CONFIG,
            fam,
            marker_manifest,
            grm_root,
            tmp_path / "altered-list.json",
        )


def test_verify_grms_rejects_id_order_and_non_psd_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, marker_root = _prepare_small_marker_lists(tmp_path, monkeypatch)
    fam = tmp_path / "fixture.fam"
    _write_fam(fam)
    grm_root = tmp_path / "grms"
    _write_small_grms(grm_root)
    id_path = grm_root / "leave_chr2_out.grm.id"
    id_path.write_text("0\tS1\n0\tS0\n0\tS2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="IDs/order"):
        grm.verify_grms(
            CONFIG,
            fam,
            marker_root / "marker_manifest.json",
            grm_root,
            tmp_path / "bad-id.json",
        )

    id_path.write_text("0\tS0\n0\tS1\n0\tS2\n", encoding="utf-8")
    _write_float32(
        grm_root / "leave_chr2_out.grm.bin",
        [1.0, 2.0, 1.0, 0.0, 0.0, 1.0],
    )
    with pytest.raises(ValueError, match="semidefinite|semi-definite|PSD|eigen"):
        grm.verify_grms(
            CONFIG,
            fam,
            marker_root / "marker_manifest.json",
            grm_root,
            tmp_path / "bad-psd.json",
        )


def _gcta_log(
    options: list[tuple[str, str | None]],
    evidence: list[str],
    heading: str = "Accepted options:",
) -> str:
    accepted = [
        name if value is None else f"{name} {value}" for name, value in options
    ]
    return "\n".join([heading, *accepted, "", *evidence, ""])


def _write_runtime_logs(
    manifest: grm.GrmSensitivityManifest,
    marker_root: Path,
    grm_root: Path,
    chunks: Path,
) -> None:
    grm_root.mkdir()
    marker_rows = {
        item["chromosome"]: item
        for item in json.loads(
            (marker_root / "marker_manifest.json").read_text(encoding="utf-8")
        )["chromosomes"]
    }
    baseline_prefix = (
        f"/parents/{manifest.baseline_parent.run_id}/genotype/abamectin_209_qc"
    )
    for chromosome in grm.CHROMOSOMES:
        complement = marker_root / marker_rows[chromosome]["relationship_list"]
        prefix = grm_root / f"leave_chr{chromosome}_out"
        options = [
            ("--bfile", baseline_prefix),
            ("--autosome-num", "6"),
            ("--autosome", None),
            ("--extract", str(complement)),
            ("--make-grm", None),
            ("--make-grm-alg", "0"),
            ("--thread-num", "16"),
            ("--out", str(prefix)),
        ]
        Path(f"{prefix}.log").write_text(
            _gcta_log(
                options,
                ["209 individuals", "5 SNPs", "Computing GRM"],
                "Options:\n",
            ),
            encoding="utf-8",
        )

    for model_contract in manifest.association.models:
        for trait in TRAITS:
            trait_slug = TRAIT_SLUGS[trait]
            trait_root = chunks / model_contract.id / trait_slug
            trait_root.mkdir(parents=True)
            for chromosome in grm.CHROMOSOMES:
                candidate = marker_root / marker_rows[chromosome]["association_list"]
                grm_prefix = grm_root / f"leave_chr{chromosome}_out"
                prefix = trait_root / f"chr{chromosome}"
                options: list[tuple[str, str | None]] = [
                    ("--mlma", None),
                    ("--bfile", baseline_prefix),
                    ("--extract", str(candidate)),
                    ("--grm", str(grm_prefix)),
                    (
                        "--pheno",
                        f"/parents/{manifest.baseline_parent.run_id}/prepared/cohort/"
                        f"{trait_slug}.phen",
                    ),
                    ("--maf", "0.05"),
                    ("--autosome-num", "6"),
                    ("--thread-num", "16"),
                    ("--out", str(prefix)),
                ]
                evidence = [
                    "209 individuals are in common in these files.",
                    "Log-likelihood ratio converged.",
                    "Running association tests for 2 SNPs",
                ]
                if model_contract.pc_count == 10:
                    options.append(
                        (
                            "--qcovar",
                            f"/parents/{manifest.calibration_parent.run_id}/genotype/"
                            "qcovars/pc10.qcovar",
                        )
                    )
                    evidence.append(
                        "10 quantitative covariate(s) of 209 individuals are included"
                    )
                Path(f"{prefix}.log").write_text(
                    _gcta_log(options, evidence), encoding="utf-8"
                )


def test_runtime_log_qualification_proves_exact_commands_counts_and_convergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _, _, marker_root = _prepare_small_marker_lists(tmp_path, monkeypatch)
    grm_root = tmp_path / "grms"
    chunks = tmp_path / "chunks"
    _write_runtime_logs(manifest, marker_root, grm_root, chunks)

    receipt = grm.verify_runtime_logs(
        CONFIG,
        marker_root / "marker_manifest.json",
        grm_root,
        chunks,
        tmp_path / "runtime.json",
    )

    assert receipt["grm_log_count"] == 6
    assert receipt["association_log_count"] == 48
    assert receipt["all_reml_fits_converged"] is True
    pc0 = [item for item in receipt["association_scans"] if item["model"] == "ldgrm_pc0"]
    pc10 = [item for item in receipt["association_scans"] if item["model"] == "ldgrm_pc10"]
    assert len(pc0) == len(pc10) == 24
    assert all("--qcovar" not in item["accepted_options"] for item in pc0)
    assert all(item["accepted_options"]["--qcovar"].endswith("pc10.qcovar") for item in pc10)


def test_runtime_log_qualification_rejects_pc0_qcovar_and_nonconvergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _, _, marker_root = _prepare_small_marker_lists(tmp_path, monkeypatch)
    grm_root = tmp_path / "grms"
    chunks = tmp_path / "chunks"
    _write_runtime_logs(manifest, marker_root, grm_root, chunks)
    pc0_log = chunks / "ldgrm_pc0/mean_EXT/chr1.log"
    pc0_log.write_text(
        pc0_log.read_text(encoding="utf-8") + "--qcovar unexpected\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="PC0.*qcovar"):
        grm.verify_runtime_logs(
            CONFIG,
            marker_root / "marker_manifest.json",
            grm_root,
            chunks,
            tmp_path / "pc0-qcovar.json",
        )

    restored_pc0_log = pc0_log.read_text(encoding="utf-8").replace(
        "--qcovar unexpected\n", ""
    )
    pc0_log.write_text(restored_pc0_log, encoding="utf-8")
    pc10_log = chunks / "ldgrm_pc10/mean_EXT/chr1.log"
    pc10_log.write_text(
        pc10_log.read_text(encoding="utf-8") + "failed to converge\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="failure indicator"):
        grm.verify_runtime_logs(
            CONFIG,
            marker_root / "marker_manifest.json",
            grm_root,
            chunks,
            tmp_path / "nonconvergence.json",
        )


def _write_chunks(root: Path) -> None:
    for model in grm.MODEL_IDS:
        for trait in TRAITS:
            trait_root = root / model / TRAIT_SLUGS[trait]
            trait_root.mkdir(parents=True)
            for chromosome in grm.CHROMOSOMES:
                path = trait_root / f"chr{chromosome}.mlma"
                rows = ["\t".join(MLMA_HEADER)]
                rows.extend(
                    [
                        (
                            f"{chromosome}\tm{chromosome}a\t{chromosome * 100 + 1}"
                            "\tA\tG\t0.2\t0.5\t0.1\t0.01"
                        ),
                        (
                            f"{chromosome}\tm{chromosome}b\t{chromosome * 100 + 2}"
                            "\tC\tT\t0.3\t-0.2\t0.1\t0.20"
                        ),
                    ]
                )
                path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")


def test_assemble_maps_requires_48_exact_chunks_and_emits_eight_full_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bim, prune = _write_small_bim_and_prune(tmp_path)
    manifest = _small_manifest(bim, prune)
    monkeypatch.setattr(grm, "load_grm_manifest", lambda _: manifest)
    chunks = tmp_path / "chunks"
    _write_chunks(chunks)

    receipt = grm.assemble_maps(CONFIG, bim, chunks, tmp_path / "assembled")

    assert receipt["map_count"] == 8
    assert receipt["chunk_count"] == 48
    assert receipt["marker_count_per_map"] == 12
    for item in receipt["maps"]:
        assembled = tmp_path / "assembled" / item["relative_path"]
        assert assembled.is_file()
        assert len(assembled.read_text(encoding="utf-8").splitlines()) == 13
        assert item["markers"] == 12


def test_assemble_maps_rejects_chunk_marker_tamper_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bim, prune = _write_small_bim_and_prune(tmp_path)
    manifest = _small_manifest(bim, prune)
    monkeypatch.setattr(grm, "load_grm_manifest", lambda _: manifest)
    chunks = tmp_path / "chunks"
    _write_chunks(chunks)
    tampered = chunks / "ldgrm_pc10/mean_EXT/chr4.mlma"
    tampered.write_text(
        tampered.read_text(encoding="utf-8").replace("m4b", "wrong-marker"),
        encoding="utf-8",
    )
    output = tmp_path / "assembled"

    with pytest.raises(ValueError, match="identity/order"):
        grm.assemble_maps(CONFIG, bim, chunks, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".assembled.tmp-*"))


def test_matched_reference_gate_includes_alleles_and_frequency() -> None:
    reference = [_MlmaRow(1, "m1", 101, "A", "G", 0.2, 0.5, 0.1, 0.01)]
    matching = [_MlmaRow(1, "m1", 101, "A", "G", 0.2, -0.1, 0.2, 0.5)]
    grm._assert_matched_map_signature(reference, matching, "fixture")

    wrong_allele = [_MlmaRow(1, "m1", 101, "G", "A", 0.2, 0.5, 0.1, 0.01)]
    with pytest.raises(ValueError, match="allele/frequency"):
        grm._assert_matched_map_signature(reference, wrong_allele, "fixture")

    wrong_frequency = [_MlmaRow(1, "m1", 101, "A", "G", 0.21, 0.5, 0.1, 0.01)]
    with pytest.raises(ValueError, match="allele/frequency"):
        grm._assert_matched_map_signature(reference, wrong_frequency, "fixture")


@pytest.mark.skip(
    reason="the site-specific runner is intentionally absent from the public tree"
)
def test_runner_executes_six_external_grms_48_scans_and_eight_assemblies() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "BASELINE_PARENT CALIBRATION_PARENT NEW_RUN_ROOT" in script
    assert "THREADS" in script and "16" in script
    assert "verify-parents" in script
    assert "prepare-marker-lists" in script
    assert "for chromosome in 1 2 3 4 5 6" in script
    assert "--make-grm" in script
    assert "--make-grm-alg 0" in script
    assert "--make-grm-inbred" not in script
    assert "--autosome-num 6" in script
    assert "grm_exclude_chr${chromosome}.snplist" in script
    assert "candidate_chr${chromosome}.snplist" in script
    assert "--mlma" in script
    assert "--mlma-loco" not in script
    assert "--grm" in script
    assert "--qcovar" in script
    assert "--maf 0.05" in script
    assert "verify-grms" in script
    assert "verify-runtime-logs" in script
    assert "assemble-maps" in script
    assert "--clump-p1 1.3394806565598387e-7" in script
    assert "--clump-p2 0.05" in script
    assert "--clump-r2 0.2" in script
    assert "--clump-kb 1000" in script
    assert "printf 'SUCCESS\\n'" in script
    assert script.index("SHA256SUMS.txt") < script.index("printf 'SUCCESS\\n'")
    assert "deployer-runner-exit-code.txt" not in script


@pytest.mark.skip(
    reason="the site-specific deployer is intentionally absent from the public tree"
)
def test_deployer_is_immutable_dual_parent_non_destructive_and_audits_bundle() -> None:
    script = DEPLOYER.read_text(encoding="utf-8")

    assert "status --porcelain=v1 --untracked-files=all" in script
    assert "archive --format=zip" in script
    assert "abamectin-ws283-bd41637-20260717T200012Z" in script
    assert "abamectin-ws283-pc-calibration-20260720T215332Z-192322baefdf" in script
    assert "StrictHostKeyChecking=yes" in script
    assert "IdentitiesOnly=yes" in script
    assert "test ! -e '$Deployment'" in script
    assert "test ! -e '$RunRoot'" in script
    assert "run_abamectin_qtl_ws283_ldpruned_grm_sensitivity.sh" in script
    assert "deployer-runner-exit-code.txt" in script
    assert "SHA256SUMS.txt" in script
    assert "Result archive contains a link or special entry" in script
    assert "Result bundle must contain exactly one terminal marker" in script
    assert "FAILURE exit code conflicts with the deployer runner receipt" in script
    assert "[System.IO.Path]::GetRelativePath" not in script
    assert ".Substring($LocalPrefix.Length)" in script
    assert "Remove-Item" not in script
    assert "rm -" not in script
    assert "git checkout" not in script
    assert "git reset" not in script
