"""Command-line entry point for the offline-first starter pipeline."""

from __future__ import annotations

import argparse
import importlib.util
from importlib import resources
import json
import platform
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import __version__
from .adapters.local_jsonl import LocalJsonlAdapter
from .io import write_json
from .manifests import (
    ManifestError,
    catalog_plan,
    fetch_release,
    list_catalog,
    migrate_snapshot_receipts,
)
from .pipeline import (
    build_graph,
    build_transport_map,
    normalize_snapshot,
    verify_build_receipt,
    verify_normalization_receipt,
    write_build_receipt,
)


def repository_root() -> Path:
    """Return checkout resources or their installed-wheel equivalent.

    Hatch places the committed contracts below ``wormctx/resources`` in a
    wheel.  Keeping this lookup here lets CLI defaults behave identically from
    a checkout and from an installed distribution.
    """

    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "config" / "source_catalog.json").is_file():
        return checkout
    packaged = resources.files("wormctx").joinpath("resources")
    return Path(str(packaged))


def parser() -> argparse.ArgumentParser:
    root = repository_root()
    command = argparse.ArgumentParser(prog="wormctx", description=__doc__)
    command.add_argument("--version", action="version", version=__version__)
    subcommands = command.add_subparsers(dest="command", required=True)

    subcommands.add_parser("doctor", help="check local runtime and repository contracts")

    sources = subcommands.add_parser("sources", help="inspect source catalogs")
    source_commands = sources.add_subparsers(dest="sources_command", required=True)
    for name in ("list", "plan"):
        child = source_commands.add_parser(name)
        child.add_argument(
            "--catalog",
            type=Path,
            default=root / "config" / "source_catalog.json",
        )

    fetch = subcommands.add_parser("fetch", help="fetch an immutable, rights-cleared manifest")
    fetch.add_argument("--manifest", type=Path, required=True)
    fetch.add_argument("--raw-root", type=Path, required=True)
    fetch.add_argument("--max-bytes", type=int, default=5 * 1024**3)

    migrate = subcommands.add_parser(
        "migrate-receipts",
        help="verify a snapshot and copy it to a fresh root with portable receipts",
    )
    migrate.add_argument("--manifest", type=Path, required=True)
    migrate.add_argument("--source-raw-root", type=Path, required=True)
    migrate.add_argument("--destination-raw-root", type=Path, required=True)

    validate = subcommands.add_parser("validate", help="validate canonical observation JSONL")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--report", type=Path)

    normalize = subcommands.add_parser(
        "normalize", help="verify a fetched snapshot, run its adapter, and bind provenance"
    )
    normalize.add_argument("--manifest", type=Path, required=True)
    normalize.add_argument("--raw-root", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--report", type=Path)

    build = subcommands.add_parser("build", help="build contextual and KGX graph views")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--normalization-receipt", type=Path)
    build.add_argument(
        "--manifest",
        type=Path,
        help="locked source manifest used for full normalization provenance verification",
    )
    build.add_argument(
        "--raw-root",
        type=Path,
        help="verified blob/receipt root used with --manifest",
    )

    transport = subcommands.add_parser(
        "transport-map", help="build coverage and query-specific compatibility products"
    )
    transport.add_argument("--input", type=Path, required=True)
    transport.add_argument("--query", type=Path, required=True)
    transport.add_argument(
        "--policy", type=Path, default=root / "config" / "transport_policy.json"
    )
    transport.add_argument("--output", type=Path, required=True)

    verify_build = subcommands.add_parser(
        "verify-build", help="verify a build receipt and authoritative product snapshot"
    )
    verify_build.add_argument("--output", type=Path, required=True)
    verify_build.add_argument("--observations", type=Path, action="append")
    verify_build.add_argument("--query", type=Path, action="append")
    verify_build.add_argument("--policy", type=Path, action="append")
    verify_build.add_argument("--normalization-receipt", type=Path)
    verify_build.add_argument("--manifest", type=Path)
    verify_build.add_argument("--raw-root", type=Path)

    demo = subcommands.add_parser("demo", help="run the complete offline fixture pipeline")
    # Build products are emitted output, not packaged resources: default to a
    # caller-relative path so a wheel install never writes into site-packages.
    demo.add_argument("--output", type=Path, default=Path("build") / "demo")
    return command


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (ManifestError, ValidationError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


def dispatch(args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]] | None:
    root = repository_root()
    if args.command == "doctor":
        required = [
            root / "schemas" / "contextual_observation.yaml",
            root / "schemas" / "source_manifest.yaml",
            root / "config" / "transport_policy.json",
            root / "config" / "source_catalog.json",
            root / "config" / "standards_lock.json",
            root / "sql" / "canonical.sql",
        ]
        return {
            "wormctx_version": __version__,
            "python": platform.python_version(),
            "repository_root": str(root),
            "contracts": {str(path.relative_to(root)): path.is_file() for path in required},
            "core_ready": all(path.is_file() for path in required),
            "semantic_tooling": {
                name: importlib.util.find_spec(module) is not None
                for name, module in {
                    "linkml": "linkml",
                    "kgx": "kgx",
                    "oaklib": "oaklib",
                    "sssom": "sssom",
                }.items()
            },
            "production_ready": False,
            "production_blockers": [
                "resolve and rights-review immutable production release locks",
                "install and run pinned LinkML/KGX/OAK validation",
                "implement and validate source-specific adapters",
            ],
        }
    if args.command == "sources":
        if args.sources_command == "list":
            return list_catalog(args.catalog)
        return catalog_plan(args.catalog)
    if args.command == "fetch":
        receipts = fetch_release(
            args.manifest, args.raw_root, max_bytes=args.max_bytes
        )
        return [item.model_dump(mode="json", exclude_none=True) for item in receipts]
    if args.command == "migrate-receipts":
        return migrate_snapshot_receipts(
            args.manifest,
            args.source_raw_root,
            args.destination_raw_root,
        )
    if args.command == "validate":
        adapter = LocalJsonlAdapter()
        report = adapter.validate_raw(args.input)
        result = {
            "source_id": report.source_id,
            "valid": report.valid,
            "checked_records": report.checked_records,
            "issues": [
                {**issue.__dict__, "severity": issue.severity.value}
                for issue in report.issues
            ],
        }
        if args.report:
            write_json(args.report, result)
        if not report.valid:
            raise ValueError(f"validation failed with {len(report.issues)} issue(s)")
        return result
    if args.command == "normalize":
        return normalize_snapshot(
            args.manifest,
            args.raw_root,
            args.output,
            report_path=args.report,
        )
    if args.command == "build":
        summary = build_graph(args.input, args.output)
        summary["input_binding"] = (
            verify_normalization_receipt(
                args.input,
                args.normalization_receipt,
                manifest_path=args.manifest,
                raw_root=args.raw_root,
            )
            if args.normalization_receipt
            else {
                "status": "unbound_fixture_or_manual_input",
                "warning": "production builds must provide --normalization-receipt",
            }
        )
        write_build_receipt(args.output, summary)
        return summary
    if args.command == "transport-map":
        summary = build_transport_map(args.input, args.query, args.policy, args.output)
        summary["input_binding"] = {
            "status": "unbound_fixture_or_manual_input",
            "warning": "transport input was not accompanied by a normalization receipt",
        }
        write_build_receipt(args.output, summary)
        return summary
    if args.command == "verify-build":
        local_inputs = {
            name: paths
            for name, paths in {
                "observations_sha256": args.observations,
                "query_sha256": args.query,
                "policy_sha256": args.policy,
            }.items()
            if paths
        }
        return verify_build_receipt(
            args.output,
            local_inputs=local_inputs,
            normalization_receipt_path=args.normalization_receipt,
            manifest_path=args.manifest,
            raw_root=args.raw_root,
        )
    if args.command == "demo":
        input_path = root / "data" / "examples" / "observations.jsonl"
        query_path = root / "data" / "examples" / "query_context.json"
        policy_path = root / "config" / "transport_policy.json"
        graph_summary = build_graph(input_path, args.output)
        transport_summary = build_transport_map(
            input_path,
            query_path,
            policy_path,
            args.output,
            carry_forward_files=graph_summary["produced_files"],
        )
        summary = {
            "graph": graph_summary,
            "transport": transport_summary,
            "input_binding": {
                "status": "unbound_bundled_synthetic_fixture",
                "warning": "demo inputs are committed fixtures, not a fetched source snapshot",
            },
        }
        write_build_receipt(args.output, summary)
        return summary
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
