from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import wormctx.execution as execution_module
from wormctx.execution import (
    ExecutionFailed,
    ExecutionManifest,
    ExecutionRoots,
    FileBinding,
    OutputBinding,
    PreflightError,
    PythonModuleEntrypoint,
    ReceiptExistsError,
    ResourceAvailability,
    ResourceRequest,
    execute_manifest,
    preflight_execution,
    verify_execution_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_EXPERIMENT = (
    ROOT
    / "experiments"
    / "graph_conditioned_inference"
    / "synthetic_routing_and_design_selection"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def execution_case(tmp_path: Path) -> tuple[ExecutionManifest, ExecutionRoots, dict[str, str]]:
    config_root = tmp_path / "config"
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    config_root.mkdir()
    input_root.mkdir()
    output_root.mkdir()
    (config_root / "settings.json").write_text('{"indent": 2}\n', encoding="utf-8")
    (input_root / "payload.json").write_text('{"answer": 42}\n', encoding="utf-8")

    manifest = ExecutionManifest(
        experiment_id="portable-json-validation",
        entrypoint=PythonModuleEntrypoint(
            module="json.tool",
            argv=("{input:payload}", "{output:pretty-json}"),
        ),
        configs=(
            FileBinding(
                name="settings",
                path="settings.json",
                sha256=_digest(config_root / "settings.json"),
            ),
        ),
        inputs=(
            FileBinding(
                name="payload",
                path="payload.json",
                sha256=_digest(input_root / "payload.json"),
            ),
        ),
        outputs=(OutputBinding(name="pretty-json", path="results/pretty.json"),),
        resources=ResourceRequest(cpu_cores=1, ram_gib=0.1),
        required_environment=("DEEPQUERY_TEST_KEY",),
        timeout_seconds=30,
    )
    roots = ExecutionRoots(
        config_root=config_root.resolve(),
        input_root=input_root.resolve(),
        output_root=output_root.resolve(),
    )
    environment = {"DEEPQUERY_TEST_KEY": "never-serialize-this-secret"}
    return manifest, roots, environment


def _available() -> ResourceAvailability:
    return ResourceAvailability(cpu_cores=8, ram_gib=32, gpu_vram_gib=(24,))


def test_linux_ram_discovery_prefers_memavailable() -> None:
    meminfo_path = MagicMock(spec=Path)
    meminfo_path.read_text.return_value = (
        "MemTotal:       131900000 kB\n"
        "MemFree:          7000000 kB\n"
        "MemAvailable:    114294784 kB\n"
    )
    with (
        patch.object(execution_module.sys, "platform", "linux"),
        patch.object(execution_module.os, "sysconf", create=True) as sysconf,
    ):
        assert execution_module._discover_posix_ram_gib(meminfo_path) == 109.0
    meminfo_path.read_text.assert_called_once_with(encoding="ascii")
    sysconf.assert_not_called()


@pytest.mark.parametrize(
    "meminfo_result",
    [
        "MemAvailable: not-a-number kB\n",
        FileNotFoundError("/proc/meminfo"),
    ],
)
def test_linux_ram_discovery_falls_back_to_sysconf(
    meminfo_result: str | OSError,
) -> None:
    meminfo_path = MagicMock(spec=Path)
    if isinstance(meminfo_result, OSError):
        meminfo_path.read_text.side_effect = meminfo_result
    else:
        meminfo_path.read_text.return_value = meminfo_result
    with (
        patch.object(execution_module.sys, "platform", "linux"),
        patch.object(
            execution_module.os,
            "sysconf",
            side_effect=[4096, 2_097_152],
            create=True,
        ) as sysconf,
    ):
        assert execution_module._discover_posix_ram_gib(meminfo_path) == 8.0
    assert sysconf.call_args_list == [
        (("SC_PAGE_SIZE",),),
        (("SC_AVPHYS_PAGES",),),
    ]


@pytest.mark.parametrize(
    "manifest_name",
    ["portable_smoke.json", "rtx3090_gpu.json"],
)
def test_public_execution_manifests_preflight_against_bound_configs(
    manifest_name: str,
    tmp_path: Path,
) -> None:
    manifest = ExecutionManifest.model_validate_json(
        (SYNTHETIC_EXPERIMENT / "execution" / manifest_name).read_bytes()
    )
    output = tmp_path / "output"
    output.mkdir()
    prepared = preflight_execution(
        manifest,
        ExecutionRoots(
            config_root=(SYNTHETIC_EXPERIMENT / "config").resolve(),
            input_root=ROOT.resolve(),
            output_root=output.resolve(),
        ),
        trusted_modules={"wormctx.poc"},
        environment={},
        availability=_available(),
        dry_run=True,
    )
    assert prepared.manifest_sha256 == manifest.canonical_sha256()
    assert prepared.command[1:3] == ("-m", "wormctx.poc")


def test_context_graph_smoke_manifest_preflights_against_rights_safe_inputs(
    tmp_path: Path,
) -> None:
    source = (
        ROOT
        / "experiments"
        / "context_graph"
        / "context_graph_transport_smoke"
        / "execution"
        / "portable_smoke.json"
    )
    manifest = ExecutionManifest.model_validate_json(source.read_bytes())
    output = tmp_path / "output"
    output.mkdir()
    prepared = preflight_execution(
        manifest,
        ExecutionRoots(
            config_root=ROOT.resolve(),
            input_root=ROOT.resolve(),
            output_root=output.resolve(),
        ),
        trusted_modules={"wormctx.__main__"},
        environment={},
        availability=_available(),
        dry_run=True,
    )
    assert prepared.verified_inputs[0].name == "transport-policy"
    assert {item.name for item in prepared.verified_inputs[1:]} == {
        "observations",
        "query-context",
    }
    assert prepared.command[1:3] == ("-m", "wormctx.__main__")


def test_manifest_hash_is_canonical_across_unordered_declarations(tmp_path: Path) -> None:
    first_file = tmp_path / "first"
    second_file = tmp_path / "second"
    first_file.write_bytes(b"first")
    second_file.write_bytes(b"second")
    common = {
        "experiment_id": "canonical-manifest-check",
        "entrypoint": {"module": "json.tool", "argv": ["--help"]},
        "outputs": [
            {"name": "second-output", "path": "second.json"},
            {"name": "first-output", "path": "first.json"},
        ],
        "resources": {"cpu_cores": 1, "ram_gib": 1},
    }
    left = ExecutionManifest.model_validate(
        {
            **common,
            "inputs": [
                {"name": "second", "path": "second", "sha256": _digest(second_file)},
                {"name": "first", "path": "first", "sha256": _digest(first_file)},
            ],
            "required_environment": ["SECOND_VARIABLE", "FIRST_VARIABLE"],
        }
    )
    right = ExecutionManifest.model_validate(
        {
            **common,
            "outputs": list(reversed(common["outputs"])),
            "inputs": list(reversed(left.model_dump(mode="json")["inputs"])),
            "required_environment": ["FIRST_VARIABLE", "SECOND_VARIABLE"],
        }
    )
    assert left.canonical_sha256() == right.canonical_sha256()
    assert left.canonical_bytes() == right.canonical_bytes()


@pytest.mark.parametrize("experiment_id", ["es-2", "s1", "opaque", "../escape", "Bad-Name"])
def test_experiment_id_must_be_descriptive(experiment_id: str) -> None:
    with pytest.raises(ValidationError):
        ExecutionManifest(
            experiment_id=experiment_id,
            entrypoint={"module": "json.tool"},
            outputs=[{"name": "result", "path": "result.json"}],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inputs", [{"name": "payload", "path": "../payload", "sha256": "0" * 64}]),
        ("inputs", [{"name": "payload", "path": "/etc/passwd", "sha256": "0" * 64}]),
        ("inputs", [{"name": "payload", "path": r"C:\private", "sha256": "0" * 64}]),
        ("outputs", [{"name": "result", "path": "../result.json"}]),
    ],
)
def test_declared_paths_reject_absolute_and_traversal(field: str, value: list[dict]) -> None:
    payload = {
        "experiment_id": "path-safety-check",
        "entrypoint": {"module": "json.tool"},
        "outputs": [{"name": "result", "path": "result.json"}],
        field: value,
    }
    with pytest.raises(ValidationError):
        ExecutionManifest.model_validate(payload)


def test_manifest_cannot_smuggle_a_shell_command() -> None:
    with pytest.raises(ValidationError):
        ExecutionManifest.model_validate(
            {
                "experiment_id": "shell-command-rejection",
                "entrypoint": {"module": "json.tool"},
                "outputs": [{"name": "result", "path": "result.json"}],
                "command": "python -m json.tool; whoami",
            }
        )
    with pytest.raises(ValidationError):
        PythonModuleEntrypoint(module="json.tool;whoami")
    with pytest.raises(ValidationError):
        PythonModuleEntrypoint(module="json.tool", argv=("../private",))
    with pytest.raises(ValidationError):
        PythonModuleEntrypoint(module="json.tool", argv=("{unknown:value}",))


def test_argv_placeholder_must_name_a_declared_binding() -> None:
    with pytest.raises(ValidationError, match="undeclared input"):
        ExecutionManifest(
            experiment_id="placeholder-binding-check",
            entrypoint={"module": "json.tool", "argv": ["{input:missing}"]},
            outputs=[{"name": "result", "path": "result.json"}],
        )


def test_preflight_resolves_placeholders_beneath_roots(execution_case) -> None:
    manifest, roots, environment = execution_case
    prepared = preflight_execution(
        manifest,
        roots,
        trusted_modules={"json.tool"},
        environment=environment,
        availability=_available(),
    )
    assert prepared.command[:3] == (str(Path(sys.executable).resolve()), "-m", "json.tool")
    assert Path(prepared.command[3]).is_relative_to(roots.input_root)
    assert Path(prepared.command[4]).is_relative_to(roots.output_root)
    assert prepared.verified_inputs[0].kind == "config"
    assert prepared.verified_inputs[1].kind == "input"


def test_preflight_rejects_tampered_input(execution_case) -> None:
    manifest, roots, environment = execution_case
    (roots.input_root / "payload.json").write_text('{"answer": 0}\n', encoding="utf-8")
    with pytest.raises(PreflightError, match="sha256"):
        preflight_execution(
            manifest,
            roots,
            trusted_modules={"json.tool"},
            environment=environment,
            availability=_available(),
        )


def test_preflight_fails_closed_for_module_environment_and_resources(execution_case) -> None:
    manifest, roots, environment = execution_case
    with pytest.raises(PreflightError, match="trust set"):
        preflight_execution(
            manifest,
            roots,
            trusted_modules={"some.other.module"},
            environment=environment,
            availability=_available(),
        )
    with pytest.raises(PreflightError, match="environment"):
        preflight_execution(
            manifest,
            roots,
            trusted_modules={"json.tool"},
            environment={},
            availability=_available(),
        )
    gpu_manifest = manifest.model_copy(
        update={"resources": ResourceRequest(cpu_cores=1, ram_gib=1, gpus=1, min_vram_gib=24)}
    )
    with pytest.raises(PreflightError, match="GPU"):
        preflight_execution(
            gpu_manifest,
            roots,
            trusted_modules={"json.tool"},
            environment=environment,
            availability=ResourceAvailability(
                cpu_cores=8,
                ram_gib=32,
                gpu_vram_gib=(12,),
            ),
        )


def test_dry_run_writes_sanitized_write_once_receipt(execution_case) -> None:
    manifest, roots, environment = execution_case
    with patch("wormctx.execution.subprocess.run") as run:
        outcome = execute_manifest(
            manifest,
            roots,
            trusted_modules={"json.tool"},
            environment=environment,
            availability=_available(),
            dry_run=True,
        )
    run.assert_not_called()
    assert outcome.status == "dry_run"
    raw_receipt = outcome.receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(raw_receipt)
    assert receipt["status"] == "dry_run"
    assert receipt["required_environment"] == ["DEEPQUERY_TEST_KEY"]
    assert environment["DEEPQUERY_TEST_KEY"] not in raw_receipt
    assert str(roots.config_root) not in raw_receipt
    assert str(roots.input_root) not in raw_receipt
    assert str(roots.output_root) not in raw_receipt
    assert socket.gethostname() not in raw_receipt
    assert "127.0.0.1" not in raw_receipt

    with pytest.raises(ReceiptExistsError):
        execute_manifest(
            manifest,
            roots,
            trusted_modules={"json.tool"},
            environment=environment,
            availability=_available(),
            dry_run=True,
        )


def test_execute_uses_shell_false_and_binds_outputs(execution_case) -> None:
    manifest, roots, environment = execution_case
    outcome = execute_manifest(
        manifest,
        roots,
        trusted_modules={"json.tool"},
        environment=environment,
        availability=_available(),
    )
    assert outcome.status == "succeeded"
    result_path = roots.output_root / "results" / "pretty.json"
    assert json.loads(result_path.read_text(encoding="utf-8")) == {"answer": 42}
    raw_receipt = outcome.receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(raw_receipt)
    assert receipt["produced_outputs"] == [
        {
            "byte_size": result_path.stat().st_size,
            "file_count": 1,
            "kind": "file",
            "name": "pretty-json",
            "path": "results/pretty.json",
            "sha256": _digest(result_path),
        }
    ]
    assert environment["DEEPQUERY_TEST_KEY"] not in raw_receipt
    assert manifest.entrypoint.argv[0] not in raw_receipt


def test_subprocess_invocation_is_an_argv_vector_with_shell_disabled(execution_case) -> None:
    manifest, roots, environment = execution_case

    def create_output(command, **kwargs):
        assert isinstance(command, list)
        Path(command[-1]).write_text('{"answer": 42}\n', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    with patch("wormctx.execution.subprocess.run", side_effect=create_output) as run:
        execute_manifest(
            manifest,
            roots,
            trusted_modules={"json.tool"},
            environment=environment,
            availability=_available(),
        )
    assert run.call_args.kwargs["shell"] is False
    assert run.call_args.args[0][1:3] == ["-m", "json.tool"]


def test_zero_exit_without_declared_output_is_a_failure_with_sanitized_receipt(
    execution_case,
) -> None:
    manifest, roots, environment = execution_case
    with patch(
        "wormctx.execution.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0),
    ):
        with pytest.raises(ExecutionFailed) as failure:
            execute_manifest(
                manifest,
                roots,
                trusted_modules={"json.tool"},
                environment=environment,
                availability=_available(),
            )
    outcome = failure.value.outcome
    assert outcome is not None
    receipt = json.loads(outcome.receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == 0
    assert receipt["produced_outputs"] == []
    assert environment["DEEPQUERY_TEST_KEY"] not in outcome.receipt_path.read_text(
        encoding="utf-8"
    )


def test_existing_output_is_never_overwritten(execution_case) -> None:
    manifest, roots, environment = execution_case
    output = roots.output_root / "results" / "pretty.json"
    output.parent.mkdir()
    output.write_text("keep me", encoding="utf-8")
    with pytest.raises(PreflightError, match="already exists"):
        execute_manifest(
            manifest,
            roots,
            trusted_modules={"json.tool"},
            environment=environment,
            availability=_available(),
        )
    assert output.read_text(encoding="utf-8") == "keep me"


def test_directory_output_receives_recursive_checksum_closure(execution_case) -> None:
    manifest, roots, environment = execution_case
    directory_manifest = manifest.model_copy(
        update={
            "outputs": (
                OutputBinding(
                    name="run-directory",
                    path="runs/portable-run",
                    kind="directory",
                ),
            ),
            "entrypoint": PythonModuleEntrypoint(
                module="json.tool",
                argv=(),
            ),
        }
    )

    def create_directory(_command, **_kwargs):
        destination = roots.output_root / "runs" / "portable-run"
        destination.mkdir(parents=True)
        (destination / "metrics.json").write_text('{"loss": 0.5}\n', encoding="utf-8")
        (destination / "SUCCESS").write_text("qualified\n", encoding="utf-8")
        return subprocess.CompletedProcess([], 0)

    with patch("wormctx.execution.subprocess.run", side_effect=create_directory):
        outcome = execute_manifest(
            directory_manifest,
            roots,
            trusted_modules={"json.tool"},
            environment=environment,
            availability=_available(),
        )

    receipt = json.loads(outcome.receipt_path.read_text(encoding="utf-8"))
    produced = receipt["produced_outputs"][0]
    assert produced["kind"] == "directory"
    assert produced["file_count"] == 2
    assert produced["byte_size"] > 0
    assert len(produced["sha256"]) == 64


def test_receipts_namespace_cannot_be_declared_as_output() -> None:
    with pytest.raises(ValidationError, match="reserved"):
        ExecutionManifest(
            experiment_id="receipt-namespace-rejection",
            entrypoint={"module": "json.tool"},
            outputs=[
                {
                    "name": "authority-collision",
                    "path": "receipts/result.json",
                }
            ],
        )


def test_overlapping_output_declarations_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must not overlap"):
        ExecutionManifest(
            experiment_id="overlapping-output-rejection",
            entrypoint={"module": "json.tool"},
            outputs=[
                {
                    "name": "run-directory",
                    "path": "run",
                    "kind": "directory",
                },
                {
                    "name": "nested-result",
                    "path": "run/result.json",
                    "kind": "file",
                },
            ],
        )


def test_successful_receipt_replays_and_detects_output_tampering(execution_case) -> None:
    manifest, roots, environment = execution_case
    outcome = execute_manifest(
        manifest,
        roots,
        trusted_modules={"json.tool"},
        environment=environment,
        availability=_available(),
    )
    verification = verify_execution_outputs(outcome.receipt_path, roots.output_root)
    assert verification["status"] == "execution_outputs_verified"
    assert verification["output_count"] == 1
    assert verification["total_output_bytes"] > 0

    (roots.output_root / "results" / "pretty.json").write_text(
        '{"answer": 0}\n',
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="differs from receipt"):
        verify_execution_outputs(outcome.receipt_path, roots.output_root)


def test_symlink_escape_is_rejected_when_supported(execution_case, tmp_path: Path) -> None:
    manifest, roots, environment = execution_case
    outside = tmp_path / "outside"
    outside.mkdir()
    link = roots.output_root / "results"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not available to this user")
    with pytest.raises(PreflightError, match="symlink"):
        preflight_execution(
            manifest,
            roots,
            trusted_modules={"json.tool"},
            environment=environment,
            availability=_available(),
        )
