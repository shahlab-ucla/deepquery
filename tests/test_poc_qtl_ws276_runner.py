from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHELL_RUNNER = ROOT / "scripts" / "run_abamectin_qtl_ws276_reconstruction.sh"
R_RUNNER = ROOT / "scripts" / "rrblup_ws276.R"
ENVIRONMENT = ROOT / "environment" / "historical-marker-reconstruction.yml"


@pytest.mark.skip(
    reason="the site-specific historical runner is intentionally absent"
)
def test_ws276_shell_runner_is_immutable_and_honestly_classified() -> None:
    script = SHELL_RUNNER.read_text(encoding="utf-8")

    assert "PHENOTYPE VCF PUBLISHED_S3 NEW_RUN_ROOT ENV_PREFIX [PYTHON]" in script
    assert '[[ "$RUN_ROOT_ARG" != /* ]]' in script
    assert 'if ! mkdir -- "$RUN_ROOT"' in script
    assert 'mkdir -p "$RUN_ROOT"' not in script
    assert "historical_isotype_imputed_reconstruction" in script
    assert "strict_reproduction\tfalse" in script
    assert "exact_all_strain_imputed_vcf_and_original_run_manifest_unavailable" in script
    assert "historical_isotype_reconstruction_non_publishable" in script
    assert "historical_marker_identity/config/historical_isotype_reconstruction.json" in script
    assert 'git -C "$REPO_ROOT" status --porcelain=v1' in script
    assert "SOURCE_ARCHIVE_SHA256" in script
    assert "verified_git_archive_from_clean_worktree" in script
    assert "SHA256SUMS.txt" in script
    assert "sha256sum --check" in script
    assert "printf 'SUCCESS\\n'" in script
    assert "FAILED_ACCEPTANCE" in script
    assert 'if [[ "$SUMMARY_STATUS" -ne 0 ]]' in script
    assert script.index("FAILED_ACCEPTANCE") < script.index("printf 'SUCCESS\\n'")


@pytest.mark.skip(
    reason="the site-specific historical runner is intentionally absent"
)
def test_ws276_shell_runner_hard_verifies_all_frozen_assets() -> None:
    script = SHELL_RUNNER.read_text(encoding="utf-8")

    expected_contracts = (
        ("51760", "c158ff3c976ea25a5bcf2545fe459d6c0216fa015f312bed4e6d62694b7dfcae"),
        ("74544444", "fd22d4722716fc3e88875817b96631953f2797ee028e963f639ac71c6a0c94e8"),
        ("66829", "6109ab85594e379345cbb9aebac67dad34b00267f2430291863a86c79bf48303"),
        ("13387059", "7f846221728b568e0eb2967dead051f6e21f0b316a9677bccb6580d640e692b4"),
    )
    assert "require_asset()" in script
    assert script.index("require_asset phenotype") < script.index('if ! mkdir -- "$RUN_ROOT"')
    for expected_bytes, expected_hash in expected_contracts:
        assert expected_bytes in script
        assert expected_hash in script


@pytest.mark.skip(
    reason="the site-specific historical runner is intentionally absent"
)
def test_ws276_shell_runner_hard_gates_historical_tool_versions() -> None:
    script = SHELL_RUNNER.read_text(encoding="utf-8")

    assert "bcftools must be exactly version 1.9" in script
    assert "tabix/htslib must be exactly version 1.9" in script
    assert "PLINK must be exactly version 1.90b6.12" in script
    assert "Rscript must be exactly version 3.6.0" in script
    assert "rrBLUP must be exactly version 4.6" in script
    assert '[[ "$RRBLUP_VERSION_OUTPUT" != "4.6" ]]' in script
    assert 'ENV_MANAGER=${WS276_ENV_MANAGER:-' in script
    assert '"$ENV_MANAGER" list --explicit -p "$ENV_PREFIX"' in script
    assert 'cp "$ENV_HISTORY"' in script

    environment = ENVIRONMENT.read_text(encoding="utf-8")
    assert "r-base==3.6.0" in environment
    assert "r-rrblup==4.6=r36h6115d3f_1" in environment
    assert "r-rrblup=4.6\n" not in environment


@pytest.mark.skip(
    reason="the site-specific historical runner is intentionally absent"
)
def test_ws276_shell_runner_uses_recovered_historical_preprocessing() -> None:
    script = SHELL_RUNNER.read_text(encoding="utf-8")

    assert "bcftools" in script.lower()
    assert "filter -i 'N_MISSING=0'" in script
    for flag in (
        "--snps-only",
        "--biallelic-only",
        "--maf 0.05",
        "--set-missing-var-ids @:#",
        "--indep-pairwise 50 10 0.8",
        "--geno",
        "--allow-extra-chr",
    ):
        assert flag in script
    assert "sort -k1,1d -k2,2n" in script
    assert "sorted_samples.txt" in script
    assert 'value = -1' in script
    assert 'value = 1' in script
    assert 'value = "NA"' in script
    assert "21,342" in script
    assert 'PRUNED_MARKERS=$(wc -l < "$RUN_ROOT/genotype/markers.txt")' in script
    assert '"$GENOTYPE_ROWS" -ne "$PRUNED_MARKERS"' in script
    assert '"$COMPLETE_MARKERS" -ne 21342' in script
    assert "complete-case marker set is not exactly the historical 21,342 rows" in script
    assert script.index('COMPLETE_MARKERS=$(wc -l < "$ANALYSIS_MARKERS")') < script.index(
        '"$COMPLETE_MARKERS" -ne 21342'
    )
    assert '"$(wc -l < "$SAMPLE_ORDER")" -ne 209' in script
    assert 'NF != 213' in script
    assert '$1 == "MtDNA"' in script
    assert "--not-chr" not in script


@pytest.mark.skip(
    reason="the site-specific historical runner is intentionally absent"
)
def test_ws276_shell_runner_calls_both_models_and_python_auditors() -> None:
    script = SHELL_RUNNER.read_text(encoding="utf-8")

    assert "qtl-ws276-prepare" in script
    assert '--vcf-samples "$RUN_ROOT/prepared/vcf_samples.txt"' in script
    assert "qtl-ws276-summarize" in script
    assert '--published-s3 "$PUBLISHED_S3"' in script
    assert '"$RUN_ROOT/association/primary" FALSE' in script
    assert '"$RUN_ROOT/association/p3d_true" TRUE' in script
    assert "4.28448307163607" in script
    assert "deposited_effective_tests\t962.616" in script
    assert "paper_rounded_effective_tests\t963" in script
    assert "ws276_reconstruction_summary.json" in script


def test_rrblup_runner_reuses_kinship_and_emits_raw_four_trait_results() -> None:
    script = R_RUNNER.read_text(encoding="utf-8")

    assert 'requireNamespace("rrBLUP"' in script
    assert "rrBLUP::A.mat(t(analysis_values), n.core = cores)" in script
    assert "rrBLUP::GWAS(" in script
    assert "min.MAF = 0.05" in script
    assert "P3D = p3d" in script
    assert "saveRDS(kinship_payload" in script
    assert "readRDS(kinship_path)" in script
    assert 'p3d_text %in% c("TRUE", "FALSE")' in script
    assert 'mean_EXT = "mean.EXT"' in script
    assert 'mean_TOF = "mean.TOF"' in script
    assert 'mean_norm_EXT = "mean.norm.EXT"' in script
    assert 'norm_n = "norm.n"' in script
    assert 'paste0(slug, "_raw_mapping.tsv")' in script


def test_rrblup_runner_gates_matrix_shape_encoding_and_mtdna() -> None:
    script = R_RUNNER.read_text(encoding="utf-8")

    assert "nrow(analysis_genotype) != 21342L" in script
    assert "complete-case genotype matrix has %d rows" in script
    assert "length(sample_names) != 209L" in script
    assert "genotype_values == -1 | genotype_values == 1" in script
    assert '"MtDNA" %in% as.character(genotype$CHROM)' in script
    assert "complete.cases(genotype_values)" in script
    assert "historical NA removal" in script
    assert '"strict_reproduction", "P3D"' in script
    assert '"historical_isotype_imputed_reconstruction", "false"' in script
