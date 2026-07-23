"""Command-line interface for the bounded inference POC."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from wormctx.cli import repository_root

from .config import load_config
from .contracts import BiologicalEpisode
from .generation import dataset_summary, generate_configured_episodes, split_episodes
from .operators import design_issues
from .repro import (
    runtime_manifest,
    sha256_file,
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
)


def parser() -> argparse.ArgumentParser:
    root = repository_root()
    command = argparse.ArgumentParser(prog="wormpoc", description=__doc__)
    subcommands = command.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="inspect POC runtime and contracts")
    doctor.add_argument("--config", type=Path, default=root / "config" / "poc_smoke.json")

    generate = subcommands.add_parser(
        "generate", help="materialize deterministic simulation episodes without torch"
    )
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)

    validate = subcommands.add_parser("validate", help="validate an episode JSONL file")
    validate.add_argument("--input", type=Path, required=True)

    run = subcommands.add_parser("run", help="train, evaluate, and receipt the POC")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)

    verify = subcommands.add_parser("verify-run", help="verify a completed run checksum set")
    verify.add_argument("--run", type=Path, required=True)

    baseline = subcommands.add_parser(
        "real-baseline",
        help="run an explicitly exploratory grouped CaeNDR kinship baseline",
    )
    baseline.add_argument("--phenotypes", type=Path, required=True)
    baseline.add_argument("--kinship", type=Path, required=True)
    baseline.add_argument("--kinship-ids", type=Path, required=True)
    baseline.add_argument("--eigenvec", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--min-samples", type=int, default=40)
    baseline.add_argument("--seed", type=int, default=20260717)

    frozen_baseline = subcommands.add_parser(
        "real-baseline-frozen",
        help="verify a frozen input/trait/fold contract before a CaeNDR audit",
    )
    frozen_baseline.add_argument("--manifest", type=Path, required=True)
    frozen_baseline.add_argument("--data-root", type=Path, required=True)
    frozen_baseline.add_argument("--output", type=Path, required=True)

    qtl_prepare = subcommands.add_parser(
        "qtl-prepare",
        help="verify and prepare the frozen 209-isotype abamectin QTL cohort",
    )
    qtl_prepare.add_argument("--manifest", type=Path, required=True)
    qtl_prepare.add_argument("--data-root", type=Path, required=True)
    qtl_prepare.add_argument("--output", type=Path, required=True)

    qtl_summarize = subcommands.add_parser(
        "qtl-summarize",
        help="validate and summarize four predeclared GCTA MLMA-LOCO outputs",
    )
    qtl_summarize.add_argument("--manifest", type=Path, required=True)
    qtl_summarize.add_argument("--results-root", type=Path, required=True)
    qtl_summarize.add_argument("--output", type=Path, required=True)

    ws276_prepare = subcommands.add_parser(
        "qtl-ws276-prepare",
        help="prepare the explicit WS276 historical isotype reconstruction cohort",
    )
    ws276_prepare.add_argument("--manifest", type=Path, required=True)
    ws276_prepare.add_argument("--phenotype", type=Path, required=True)
    ws276_prepare.add_argument("--vcf-samples", type=Path, required=True)
    ws276_prepare.add_argument("--output", type=Path, required=True)

    ws276_summarize = subcommands.add_parser(
        "qtl-ws276-summarize",
        help="compare WS276 rrBLUP reconstruction scans with deposited File S3",
    )
    ws276_summarize.add_argument("--manifest", type=Path, required=True)
    ws276_summarize.add_argument("--results-root", type=Path, required=True)
    ws276_summarize.add_argument("--published-s3", type=Path, required=True)
    ws276_summarize.add_argument("--output", type=Path, required=True)
    return command


def _doctor(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = repository_root()
    torch_available = importlib.util.find_spec("torch") is not None
    safetensors_available = importlib.util.find_spec("safetensors") is not None
    runtime = runtime_manifest()
    return {
        "config_valid": True,
        "config": str(config_path.resolve()),
        "mode": config.mode,
        "repository_root": str(root),
        "runtime": runtime,
        "torch_available": torch_available,
        "safetensors_available": safetensors_available,
        "cuda_required": config.acceptance.require_cuda,
        "ready_for_requested_run": (
            torch_available
            and safetensors_available
            and (
                not config.acceptance.require_cuda
                or bool(runtime.get("cuda_available", False))
            )
        ),
        "claims_scope": "simulation_framework_feasibility_only",
    }


def _generate(config_path: Path, output: Path) -> dict[str, Any]:
    config = load_config(config_path)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"generation output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    episodes = generate_configured_episodes(config)
    splits = split_episodes(episodes)
    hashes: dict[str, str] = {}
    for split, values in splits.items():
        path = output / f"{split}.jsonl"
        write_jsonl(
            path,
            (item.model_dump(mode="json", exclude_none=True) for item in values),
        )
        hashes[split] = sha256_file(path)
    manifest = {
        "scope": "scientifically_structured_simulation_only",
        "biological_claims_permitted": False,
        "config_sha256": sha256_file(config_path),
        "split_sha256": hashes,
        "summary": dataset_summary(episodes),
    }
    write_json(output / "manifest.json", manifest)
    write_checksums(output)
    return manifest


def _validate(input_path: Path) -> dict[str, Any]:
    episodes = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            episode = BiologicalEpisode.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"invalid episode at line {line_number}: {exc}") from exc
        expected = {flag.value for flag in episode.labels.invalid_design_flags}
        observed = {issue.code.value for issue in design_issues(episode)}
        if expected != observed:
            raise ValueError(
                f"operator guard mismatch for {episode.id}: labels={sorted(expected)}, "
                f"guards={sorted(observed)}"
            )
        episodes.append(episode)
    if not episodes:
        raise ValueError("episode JSONL is empty")
    return {
        "valid": True,
        "input": str(input_path.resolve()),
        "sha256": sha256_file(input_path),
        "summary": dataset_summary(episodes),
    }


def _verify_run(run: Path) -> dict[str, Any]:
    run = run.resolve()
    failures = verify_checksums(run)
    manifest_path = run / "run_manifest.json"
    if not manifest_path.is_file():
        failures.append("missing:run_manifest.json")
    success = (run / "SUCCESS").is_file()
    return {
        "run": str(run),
        "checksums_valid": not failures,
        "failures": failures,
        "success_marker": success,
        "verified": not failures and success,
    }


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "doctor":
        return _doctor(args.config)
    if args.command == "generate":
        return _generate(args.config, args.output)
    if args.command == "validate":
        return _validate(args.input)
    if args.command == "verify-run":
        return _verify_run(args.run)
    if args.command == "real-baseline":
        from .realdata import run_caendr_baseline

        result = run_caendr_baseline(
            args.phenotypes,
            args.kinship,
            args.kinship_ids,
            args.eigenvec,
            min_samples=args.min_samples,
            seed=args.seed,
        )
        write_json(args.output, result)
        return result
    if args.command == "real-baseline-frozen":
        from .realconfig import run_frozen_caendr_baseline

        if args.output.exists():
            raise ValueError(f"frozen baseline output already exists: {args.output.resolve()}")
        result = run_frozen_caendr_baseline(args.manifest, args.data_root)
        write_json(args.output, result)
        return result
    if args.command == "qtl-prepare":
        from .qtl import prepare_abamectin_qtl

        return prepare_abamectin_qtl(args.manifest, args.data_root, args.output)
    if args.command == "qtl-summarize":
        from .qtl import summarize_abamectin_qtl

        return summarize_abamectin_qtl(args.manifest, args.results_root, args.output)
    if args.command == "qtl-ws276-prepare":
        from .qtl_ws276 import prepare_ws276_reconstruction

        return prepare_ws276_reconstruction(
            args.manifest,
            args.phenotype,
            args.vcf_samples,
            args.output,
        )
    if args.command == "qtl-ws276-summarize":
        from .qtl_ws276 import summarize_ws276_reconstruction

        return summarize_ws276_reconstruction(
            args.manifest,
            args.results_root,
            args.published_s3,
            args.output,
        )
    if args.command == "run":
        from .training import run_training

        return run_training(
            load_config(args.config),
            output_root=args.output_root,
            repository_root=repository_root(),
        )
    raise ValueError(f"unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (ImportError, OSError, RuntimeError, ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.command == "run" and not result.get("passed", False):
        return 3
    if args.command == "verify-run" and not result.get("verified", False):
        return 3
    if args.command == "qtl-ws276-summarize":
        gates = result.get("functional_match_gates")
        if not gates or not all(gates.values()):
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
