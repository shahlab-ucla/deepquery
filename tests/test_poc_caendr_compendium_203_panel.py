from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wormctx.poc.caendr_compendium_203_panel import (
    load_kernel_config,
    prepare_kernel_inputs,
    qualify_kernel_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = (
    ROOT / "experiments/natural_variation/phenotype_compendium_grouped_prediction/config/genotype_kernel_qualification.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordered_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _spec(role: str, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "logical_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _preflight_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path], list[Path]]:
    roster = [f"strain_{index:03d}" for index in range(203)]
    vcf_samples = [*roster, *[f"other_{index:03d}" for index in range(481)]]
    discovery = [f"discovery_trait_{index:04d}" for index in range(1184)]
    validation = [f"validation_trait_{index:04d}" for index in range(261)]

    paths = {
        "source_vcf": tmp_path / "source.vcf.gz",
        "source_vcf_index": tmp_path / "source.vcf.gz.tbi",
        "identity_only_legacy_roster": tmp_path / "roster.rel.id",
        "stage1_qualification_receipt": tmp_path / "stage1.json",
        "stage1_discovery_trait_list": tmp_path / "discovery.txt",
        "stage1_validation_trait_list": tmp_path / "validation.txt",
        "vcf_samples": tmp_path / "vcf.samples.txt",
    }
    paths["source_vcf"].write_bytes(b"synthetic-vcf")
    paths["source_vcf_index"].write_bytes(b"synthetic-index")
    paths["identity_only_legacy_roster"].write_text(
        "#FID\tIID\n" + "".join(f"0\t{iid}\n" for iid in roster),
        encoding="utf-8",
    )
    paths["stage1_discovery_trait_list"].write_text(
        "\n".join(discovery) + "\n", encoding="utf-8"
    )
    paths["stage1_validation_trait_list"].write_text(
        "\n".join(validation) + "\n", encoding="utf-8"
    )
    paths["vcf_samples"].write_text(
        "\n".join(vcf_samples) + "\n", encoding="utf-8"
    )
    paths["stage1_qualification_receipt"].write_text(
        json.dumps(
            {
                "schema_version": "wormctx-caendr-compendium-source-qualification-1.1",
                "compendium": {
                    "discovery_traits": len(discovery),
                    "validation_traits": len(validation),
                    "discovery_trait_sha256": _ordered_hash(discovery),
                    "validation_trait_sha256": _ordered_hash(validation),
                },
                "models_executed": False,
            }
        ),
        encoding="utf-8",
    )

    tools: list[Path] = []
    for identifier in ("bcftools", "plink2", "gcta64", "python"):
        tool = tmp_path / identifier
        tool.write_bytes(f"synthetic-{identifier}".encode())
        tools.append(tool)

    config = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    config["analysis_id"] = "synthetic_203_panel"
    config["inputs"] = [
        _spec(item["role"], paths[item["role"]]) for item in config["inputs"]
    ]
    config["source_vcf_contract"]["ordered_sample_sha256"] = _ordered_hash(
        vcf_samples
    )
    config["roster_contract"]["ordered_iid_sha256"] = _ordered_hash(roster)
    config["partition_identity"]["discovery_trait_sha256"] = _ordered_hash(
        discovery
    )
    config["partition_identity"]["validation_trait_sha256"] = _ordered_hash(
        validation
    )
    config["qc_contract"]["minimum_qualified_markers"] = 12
    for item, tool in zip(config["tools"], tools, strict=True):
        item["version"] = "synthetic"
        item["sha256"] = _sha256(tool)
    config_path = tmp_path / "kernel_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, paths, tools


def _run_preflight(
    config: Path, paths: dict[str, Path], tools: list[Path], output: Path
) -> dict[str, object]:
    return prepare_kernel_inputs(
        config,
        paths["source_vcf"],
        paths["source_vcf_index"],
        paths["identity_only_legacy_roster"],
        paths["stage1_qualification_receipt"],
        paths["stage1_discovery_trait_list"],
        paths["stage1_validation_trait_list"],
        paths["vcf_samples"],
        tools,
        output,
    )


def _write_qualification_fixture(tmp_path: Path, roster: list[str]) -> dict[str, Path]:
    generated = {
        name: tmp_path / name
        for name in (
            "panel.bed",
            "panel.bim",
            "panel.fam",
            "panel.afreq",
            "panel.vmiss",
            "panel.smiss",
            "panel.prune.in",
            "panel.eigenvec",
            "panel.eigenval",
            "panel.grm.bin",
            "panel.grm.N.bin",
            "panel.grm.id",
        )
    }
    markers = [
        (str(chromosome), f"{chromosome}:{100 + offset}", 100 + offset)
        for chromosome in range(1, 7)
        for offset in range(3)
    ]
    generated["panel.bed"].write_bytes(b"synthetic-bed")
    generated["panel.bim"].write_text(
        "".join(
            f"{chromosome}\t{marker}\t0\t{position}\tA\tG\n"
            for chromosome, marker, position in markers
        ),
        encoding="utf-8",
    )
    generated["panel.fam"].write_text(
        "".join(f"0\t{iid}\t0\t0\t0\t-9\n" for iid in roster),
        encoding="utf-8",
    )
    generated["panel.afreq"].write_text(
        "#CHROM\tID\tREF\tALT\tPROVISIONAL_REF?\tALT_FREQS\tOBS_CT\n"
        + "".join(
            f"{chromosome}\t{marker}\tG\tA\tY\t0.2\t406\n"
            for chromosome, marker, _position in markers
        ),
        encoding="utf-8",
    )
    generated["panel.vmiss"].write_text(
        "#CHROM\tID\tMISSING_CT\tOBS_CT\tF_MISS\n"
        + "".join(
            f"{chromosome}\t{marker}\t0\t406\t0\n"
            for chromosome, marker, _position in markers
        ),
        encoding="utf-8",
    )
    generated["panel.smiss"].write_text(
        "#FID\tIID\tMISSING_CT\tOBS_CT\tF_MISS\n"
        + "".join(f"0\t{iid}\t0\t36\t0\n" for iid in roster),
        encoding="utf-8",
    )
    generated["panel.prune.in"].write_text(
        "\n".join(marker for _chromosome, marker, _position in markers) + "\n",
        encoding="utf-8",
    )

    angles = np.arange(len(roster), dtype=np.float64) * (2.0 * np.pi / 12.0)
    cluster = np.arange(len(roster)) % 12
    scores = np.column_stack(
        [
            np.cos(angles + dimension / 7.0)
            + cluster * (dimension + 1) / 20.0
            + np.arange(len(roster)) / (10000.0 + dimension)
            for dimension in range(10)
        ]
    )
    generated["panel.eigenvec"].write_text(
        "#FID\tIID\t" + "\t".join(f"PC{index}" for index in range(1, 11)) + "\n"
        + "".join(
            f"0\t{iid}\t" + "\t".join(f"{value:.12g}" for value in row) + "\n"
            for iid, row in zip(roster, scores, strict=True)
        ),
        encoding="utf-8",
    )
    generated["panel.eigenval"].write_text(
        "\n".join(str(value) for value in range(10, 0, -1)) + "\n",
        encoding="utf-8",
    )

    kernel = np.eye(len(roster), dtype=np.float32)
    triangular = np.concatenate([kernel[row, : row + 1] for row in range(len(roster))])
    triangular.astype("<f4").tofile(generated["panel.grm.bin"])
    np.full(triangular.shape, len(markers), dtype="<f4").tofile(
        generated["panel.grm.N.bin"]
    )
    generated["panel.grm.id"].write_text(
        "".join(f"0\t{iid}\n" for iid in roster), encoding="utf-8"
    )
    return generated


def test_real_contract_is_phenotype_free_and_preserves_frozen_buckets() -> None:
    config = load_kernel_config(REAL_CONFIG)
    assert config["phenotype_paths_accepted_by_kernel_runner"] is False
    assert config["partition_identity"]["discovery_traits"] == 1184
    assert config["partition_identity"]["validation_traits"] == 261
    assert all(
        item["accepted_as_input"] is False
        for item in config["legacy_numeric_quarantine"]
    )


def test_preflight_binds_exact_roster_tools_and_trait_buckets(tmp_path: Path) -> None:
    config, paths, tools = _preflight_fixture(tmp_path)
    output = tmp_path / "preflight"
    receipt = _run_preflight(config, paths, tools, output)

    assert receipt["source_vcf_samples"] == 684
    assert receipt["roster_samples"] == 203
    assert receipt["phenotype_values_accessed"] is False
    assert receipt["legacy_numeric_assets_accessed"] is False
    assert receipt["partition_identity"]["discovery_traits"] == 1184
    assert receipt["partition_identity"]["validation_traits"] == 261
    assert len((output / "roster.samples.txt").read_text().splitlines()) == 203


def test_preflight_rejects_any_trait_partition_drift(tmp_path: Path) -> None:
    config, paths, tools = _preflight_fixture(tmp_path)
    paths["stage1_validation_trait_list"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte count differs"):
        _run_preflight(config, paths, tools, tmp_path / "preflight")


def test_qualification_emits_exact_float64_kernel_and_pc_groups(tmp_path: Path) -> None:
    config, paths, tools = _preflight_fixture(tmp_path)
    preflight = tmp_path / "preflight"
    _run_preflight(config, paths, tools, preflight)
    roster = (preflight / "roster.samples.txt").read_text().splitlines()
    generated = _write_qualification_fixture(tmp_path, roster)
    output = tmp_path / "qualified"
    receipt = qualify_kernel_outputs(
        config,
        preflight,
        generated["panel.bed"],
        generated["panel.bim"],
        generated["panel.fam"],
        generated["panel.afreq"],
        generated["panel.vmiss"],
        generated["panel.smiss"],
        generated["panel.prune.in"],
        generated["panel.eigenvec"],
        generated["panel.eigenval"],
        generated["panel.grm.bin"],
        generated["panel.grm.N.bin"],
        generated["panel.grm.id"],
        output,
    )

    kernel = np.load(output / "whole_genome_kernel.float64.npy", allow_pickle=False)
    assert kernel.shape == (203, 203)
    assert kernel.dtype == np.float64
    np.testing.assert_array_equal(kernel, np.eye(203))
    assert receipt["samples"] == 203
    assert receipt["markers"] == 18
    assert receipt["qc_statistics"]["provisional_reference_column_present"] is True
    assert receipt["kernel"]["pairwise_marker_count_min"] == 18
    assert receipt["population_groups"]["selected_k"] in range(5, 13)
    assert min(receipt["population_groups"]["group_sizes"].values()) >= 5
    assert receipt["validation_outcomes_locked"] is True
    assert receipt["discovery_prediction_models_executed"] is False
    assert (output / "SUCCESS").read_text(encoding="utf-8") == "qualified\n"


def test_multibase_alleles_are_rejected(tmp_path: Path) -> None:
    config, paths, tools = _preflight_fixture(tmp_path)
    preflight = tmp_path / "preflight"
    _run_preflight(config, paths, tools, preflight)
    roster = (preflight / "roster.samples.txt").read_text().splitlines()
    generated = _write_qualification_fixture(tmp_path, roster)
    bim = generated["panel.bim"]
    bim.write_text(bim.read_text().replace("\tA\tG\n", "\tAC\tG\n", 1))

    with pytest.raises(ValueError, match="non-biallelic ACGT"):
        qualify_kernel_outputs(
            config,
            preflight,
            generated["panel.bed"],
            generated["panel.bim"],
            generated["panel.fam"],
            generated["panel.afreq"],
            generated["panel.vmiss"],
            generated["panel.smiss"],
            generated["panel.prune.in"],
            generated["panel.eigenvec"],
            generated["panel.eigenval"],
            generated["panel.grm.bin"],
            generated["panel.grm.N.bin"],
            generated["panel.grm.id"],
            tmp_path / "qualified",
        )


@pytest.mark.skip(
    reason="the site-specific runner is intentionally absent from the public tree"
)
def test_runner_has_no_phenotype_or_legacy_numeric_argument() -> None:
    runner = (ROOT / "scripts/run_caendr_compendium_203_kernel.sh").read_text()
    assert "PHENOTYPE" not in runner
    assert "pheno_only.rel " not in runner
    assert "h2_screen" not in runner
    assert "--make-grm-alg 0" in runner
    assert "--indep-pairwise 500kb 1 0.2" in runner
