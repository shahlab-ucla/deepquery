from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from wormctx.poc import grounded_reasoner_benchmark as reasoner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/biological_reasoning/graph_grounded_reasoner_evaluation/config/reasoner_evaluation.json"
SECRET = "synthetic-private-anonymization-secret-20260721"
SIGNED_UTC = "2026-07-21T12:00:00Z"


def _asset(root: Path, name: str, content: bytes) -> reasoner.FileAsset:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_bytes(content)
    return reasoner.FileAsset(
        asset_id=name.removesuffix(".txt"),
        relative_path=name,
        sha256=hashlib.sha256(content).hexdigest(),
        bytes=len(content),
    )


def _source(predictor_receipt: reasoner.FileAsset) -> reasoner.SourceBenchmark:
    return reasoner.SourceBenchmark.model_validate(
        {
            "schema_version": reasoner.SOURCE_VERSION,
            "source_snapshot_id": "synthetic-private-snapshot",
            "predictor_receipt_assets": [predictor_receipt.model_dump(mode="json")],
            "tasks": [
                {
                    "task_id": "developmental_target_alpha",
                    "task_family": "developmental_genetics",
                    "question": (
                        "Predict the qualified endpoint for WBGene00000001 "
                        "(alpha_gene), or abstain if it is not identifiable."
                    ),
                    "entities": [
                        {"identifier": "WBGene00000001", "aliases": ["alpha_gene"]}
                    ],
                    "evidence": [
                        {
                            "evidence_id": "E-IN-01",
                            "channel": "inline",
                            "content": "alpha_gene has a frozen assay definition.",
                        },
                        {
                            "evidence_id": "E-TX-01",
                            "channel": "text",
                            "content": "WBGene00000001 has developmental context.",
                        },
                        {
                            "evidence_id": "E-GR-01",
                            "channel": "graph",
                            "content": "alpha_gene connects to the endpoint node.",
                        },
                    ],
                    "identifiable": True,
                    "identifiability_reason_code": "qualified_predictor_available",
                    "predictor": {
                        "predictor_id": "locked-development-v1",
                        "version": "1.0",
                        "input_sha256": "a" * 64,
                        "output": 1.25,
                        "qualification_receipt_sha256": predictor_receipt.sha256,
                    },
                    "gold": {
                        "disposition": "predict",
                        "prediction_type": "numeric",
                        "target": 1.25,
                        "numeric_tolerance": 0.01,
                        "evidence_ids": ["E-IN-01", "E-TX-01", "E-GR-01"],
                        "required_operators": [
                            "retrieve_text",
                            "retrieve_graph",
                            "check_identifiability",
                            "call_locked_predictor",
                            "synthesize_evidence",
                            "emit_prediction",
                        ],
                        "abstention_reason_code": None,
                    },
                },
                {
                    "task_id": "natural_variation_dominance_beta",
                    "task_family": "natural_variation",
                    "question": (
                        "Estimate dominance for WBGene00000002 (beta_gene), "
                        "or abstain if the contrast is not identifiable."
                    ),
                    "entities": [
                        {"identifier": "WBGene00000002", "aliases": ["beta_gene"]}
                    ],
                    "evidence": [
                        {
                            "evidence_id": "E-IN-02",
                            "channel": "inline",
                            "content": "beta_gene has a frozen contrast definition.",
                        },
                        {
                            "evidence_id": "E-TX-02",
                            "channel": "text",
                            "content": "WBGene00000002 has no heterozygote record.",
                        },
                        {
                            "evidence_id": "E-GR-02",
                            "channel": "graph",
                            "content": "beta_gene lacks an identifying graph path.",
                        },
                    ],
                    "identifiable": False,
                    "identifiability_reason_code": "no_heterozygotes",
                    "predictor": None,
                    "gold": {
                        "disposition": "abstain",
                        "prediction_type": "none",
                        "target": None,
                        "numeric_tolerance": 0.0,
                        "evidence_ids": ["E-IN-02", "E-TX-02", "E-GR-02"],
                        "required_operators": [
                            "retrieve_text",
                            "retrieve_graph",
                            "check_identifiability",
                            "synthesize_evidence",
                            "emit_abstention",
                        ],
                        "abstention_reason_code": "no_heterozygotes",
                    },
                },
            ],
        }
    )


def _provider() -> reasoner.ProviderSpec:
    return reasoner.ProviderSpec.model_validate(
        {
            "schema_version": reasoner.PROVIDER_VERSION,
            "provider_id": "synthetic-provider",
            "model_id": "synthetic-reasoner",
            "model_version": "frozen-test-version",
            "weights_receipt": "b" * 64,
            "training_data_cutoff": "2025-01-01",
            "prompt_template_sha256": "c" * 64,
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 19,
                "max_output_tokens": 2048,
            },
            "benchmark_seen_before_run": False,
            "external_search_enabled": False,
        }
    )


def _adapter() -> reasoner.ProviderAdapterContract:
    return reasoner.ProviderAdapterContract.model_validate(
        {
            "schema_version": reasoner.ADAPTER_VERSION,
            "adapter_id": "synthetic-jsonl-adapter",
            "adapter_version": "1.0",
            "transport": "local_jsonl",
            "endpoint_env_var": "WORMCTX_LOCAL_ENDPOINT",
            "api_key_env_var": None,
            "credentials_source": "environment_only",
            "request_format": "deterministic_jsonl",
            "response_format": "deterministic_jsonl",
            "endpoint_value_persisted": False,
            "credential_value_persisted": False,
            "external_call_performed_by_harness": False,
        }
    )


def _plans(asset_root: Path) -> tuple[
    reasoner.SourceBenchmark,
    reasoner.CustodyPlan,
    reasoner.ComparisonPlan,
    reasoner.ContaminationAttestation,
]:
    predictor = _asset(asset_root, "predictor-receipt.txt", b"qualified predictor\n")
    source = _source(predictor)
    attestations = []
    policies = {
        "release": ("provider_read_only", True, False),
        "curator": ("curator_only", False, False),
        "gold": ("gold_authority_only", False, True),
        "scorer": ("scorer_only_after_response_freeze", False, False),
    }
    for role, (policy, provider_read, contains_gold) in policies.items():
        acl = _asset(asset_root, f"acl-{role}.txt", f"{role} ACL\n".encode())
        attestations.append(
            {
                "artifact_role": role,
                "custodian_id": f"synthetic-{role}-custodian",
                "os_principal": f"wormctx-{role}",
                "access_policy": policy,
                "provider_read_permitted": provider_read,
                "contains_gold": contains_gold,
                "gold_open_before_response_freeze_permitted": False,
                "acl_enforced": True,
                "authority_separation_attested": True,
                "signed_by": "synthetic-fixture",
                "signed_utc": SIGNED_UTC,
                "acl_evidence": acl.model_dump(mode="json"),
            }
        )
    custody = reasoner.CustodyPlan.model_validate(
        {
            "schema_version": reasoner.CUSTODY_VERSION,
            "plan_id": "synthetic-four-role-custody",
            "synthetic_fixture": True,
            "biological_claims_permitted": False,
            "attestations": attestations,
        }
    )
    assumptions = _asset(
        asset_root,
        "comparison-assumptions.txt",
        b"synthetic paired-arm power assumptions; no biological claim\n",
    )
    comparison = reasoner.ComparisonPlan.model_validate(
        {
            "schema_version": reasoner.COMPARISON_PLAN_VERSION,
            "plan_id": "synthetic-frozen-comparison-plan",
            "synthetic_fixture": True,
            "biological_claims_permitted": False,
            "unit": "paired_task_by_arm",
            "task_counts_by_family": {
                "developmental_genetics": 1,
                "natural_variation": 1,
            },
            "total_tasks": 2,
            "primary_comparisons": [
                "graph_plus_qualified_predictor_vs_ordinary_rag",
                "graph_plus_qualified_predictor_vs_graph_retrieval",
                "graph_plus_qualified_predictor_vs_tool_ablation_no_predictor",
                "graph_plus_qualified_predictor_vs_tool_ablation_no_graph",
            ],
            "score_dimensions": reasoner.SCORE_DIMENSIONS,
            "familywise_alpha": 0.05,
            "target_power": 0.8,
            "sample_size_method": "simulation_or_exact_power_frozen_before_release",
            "assumptions_receipt": assumptions.model_dump(mode="json"),
            "frozen_before_task_release": True,
        }
    )
    exposure = _asset(
        asset_root,
        "contamination-evidence.txt",
        b"synthetic private-benchmark non-exposure evidence\n",
    )
    contamination = reasoner.ContaminationAttestation.model_validate(
        {
            "schema_version": reasoner.CONTAMINATION_ATTESTATION_VERSION,
            "attestation_id": "synthetic-contamination-attestation",
            "provider_id": _provider().provider_id,
            "model_id": _provider().model_id,
            "model_version": _provider().model_version,
            "training_data_cutoff": _provider().training_data_cutoff,
            "benchmark_seen_before_run": False,
            "prior_canary_exposure_known": False,
            "external_search_enabled": False,
            "signed_by": "synthetic-fixture",
            "signed_utc": SIGNED_UTC,
            "evidence_receipt": exposure.model_dump(mode="json"),
            "synthetic_fixture": True,
            "biological_claims_permitted": False,
        }
    )
    return source, custody, comparison, contamination


def _built(
    tmp_path: Path,
) -> tuple[
    reasoner.BenchmarkContract,
    Path,
    Path,
    reasoner.ContaminationAttestation,
]:
    contract = reasoner.load_contract(CONFIG)
    asset_root = tmp_path / "assets"
    source, custody, comparison, contamination = _plans(asset_root)
    benchmark = tmp_path / "benchmark"
    reasoner.build_benchmark(
        contract,
        source,
        benchmark,
        anonymization_secret=SECRET,
        custody_plan_or_path=custody,
        custody_asset_root=asset_root,
        comparison_plan_or_path=comparison,
        comparison_asset_root=asset_root,
        predictor_receipt_root=asset_root,
    )
    return contract, benchmark, asset_root, contamination


def _run(
    contract: reasoner.BenchmarkContract,
    benchmark: Path,
    asset_root: Path,
    contamination: reasoner.ContaminationAttestation,
    run_root: Path,
) -> dict[str, Any]:
    return reasoner.freeze_provider_run(
        contract,
        benchmark,
        _provider(),
        _adapter(),
        contamination,
        asset_root,
        run_root,
    )


def _program(
    task: dict[str, Any], available_tools: list[str]
) -> tuple[list[reasoner.ProgramStep], reasoner.FinalAnswer]:
    steps: list[reasoner.ProgramStep] = []
    dependencies: list[str] = []
    for operator, step_id in (
        ("retrieve_text", "text"),
        ("retrieve_graph", "graph"),
        ("check_identifiability", "identify"),
    ):
        if operator not in available_tools:
            continue
        arguments = (
            {"question_type": task["gold"]["disposition"]}
            if operator == "check_identifiability"
            else {"query": "frozen biological task"}
        )
        steps.append(
            reasoner.ProgramStep(
                step_id=step_id,
                operator=operator,
                depends_on=list(dependencies),
                arguments=arguments,
            )
        )
        dependencies.append(step_id)
    if "call_locked_predictor" in available_tools and task["predictor"] is not None:
        predictor = task["predictor"]
        steps.append(
            reasoner.ProgramStep(
                step_id="predictor",
                operator="call_locked_predictor",
                depends_on=list(dependencies),
                arguments={
                    "predictor_id": predictor["predictor_id"],
                    "version": predictor["version"],
                    "input_sha256": predictor["input_sha256"],
                },
            )
        )
        dependencies.append("predictor")
    channels = {"inline"}
    if "retrieve_text" in available_tools:
        channels.add("text")
    if "retrieve_graph" in available_tools:
        channels.add("graph")
    accessible = [
        item["evidence_id"]
        for item in task["evidence"]
        if item["channel"] in channels and item["evidence_id"] in task["gold"]["evidence_ids"]
    ]
    steps.append(
        reasoner.ProgramStep(
            step_id="synthesis",
            operator="synthesize_evidence",
            depends_on=list(dependencies),
            arguments={"evidence_ids": accessible},
        )
    )
    dependencies.append("synthesis")
    gold = task["gold"]
    if gold["disposition"] == "predict":
        final = reasoner.FinalAnswer(
            disposition="predict",
            prediction=gold["target"],
            confidence=0.8,
            interval_lower=float(gold["target"]) - 0.1,
            interval_upper=float(gold["target"]) + 0.1,
            abstention_reason_code=None,
            citations=accessible,
            claims=[
                reasoner.ClaimAttribution(
                    claim="Synthetic fixture prediction.",
                    evidence_ids=accessible,
                )
            ],
        )
        terminal = reasoner.ProgramStep(
            step_id="terminal",
            operator="emit_prediction",
            depends_on=dependencies,
            arguments={"prediction": gold["target"]},
        )
    else:
        final = reasoner.FinalAnswer(
            disposition="abstain",
            prediction=None,
            confidence=0.9,
            interval_lower=None,
            interval_upper=None,
            abstention_reason_code=gold["abstention_reason_code"],
            citations=accessible,
            claims=[
                reasoner.ClaimAttribution(
                    claim="Synthetic fixture is not identifiable.",
                    evidence_ids=accessible,
                )
            ],
        )
        terminal = reasoner.ProgramStep(
            step_id="terminal",
            operator="emit_abstention",
            depends_on=dependencies,
            arguments={"reason_code": gold["abstention_reason_code"]},
        )
    steps.append(terminal)
    return steps, final


def _combined_tasks(benchmark: Path) -> dict[str, dict[str, Any]]:
    scorer = json.loads(
        (benchmark / "scorer/benchmark_scorer_authority.json").read_text(encoding="utf-8")
    )
    gold = json.loads(
        (benchmark / "gold/benchmark_gold.json").read_text(encoding="utf-8")
    )
    gold_by_id = {item["task_id"]: item["gold"] for item in gold["tasks"]}
    return {
        item["task_id"]: {**item, "gold": gold_by_id[item["task_id"]]}
        for item in scorer["tasks"]
    }


def _adapter_response_file(
    contract: reasoner.BenchmarkContract,
    benchmark: Path,
    run_root: Path,
    path: Path,
    *,
    corrupt_trace: bool = False,
    omit_last: bool = False,
) -> Path:
    run = json.loads((run_root / "run_contract.json").read_text(encoding="utf-8"))
    tasks = _combined_tasks(benchmark)
    arms = {item.id: item for item in contract.arms}
    rows = []
    for request_id in run["request_ids"]:
        arm_id, task_id = request_id.split(":", 1)
        task = tasks[task_id]
        program, final = _program(task, arms[arm_id].allowed_tools)
        execution = reasoner.execute_program(
            program, allowed_tools=arms[arm_id].allowed_tools, task=task
        )
        response = reasoner.TaskResponse(
            request_id=request_id,
            task_id=task_id,
            arm_id=arm_id,
            program=program,
            trace=[reasoner.TraceStep.model_validate(item) for item in execution["trace"]],
            final=final,
        )
        rows.append(
            reasoner.AdapterResponseEnvelope(
                schema_version=reasoner.ADAPTER_RESPONSE_VERSION,
                request_id=request_id,
                task_id=task_id,
                arm_id=arm_id,
                provider_id=run["provider"]["provider_id"],
                model_id=run["provider"]["model_id"],
                model_version=run["provider"]["model_version"],
                decoding_sha256=reasoner._stable_hash(run["provider"]["decoding"]),
                response=response,
            ).model_dump(mode="json")
        )
    if corrupt_trace:
        rows[0]["response"]["trace"][0]["output_sha256"] = "f" * 64
    if omit_last:
        rows.pop()
    path.write_bytes(reasoner._jsonl_bytes(rows))
    return path


def _frozen_responses(
    contract: reasoner.BenchmarkContract,
    benchmark: Path,
    run_root: Path,
    tmp_path: Path,
    *,
    corrupt_trace: bool = False,
) -> Path:
    raw = _adapter_response_file(
        contract,
        benchmark,
        run_root,
        tmp_path / "adapter-responses.jsonl",
        corrupt_trace=corrupt_trace,
    )
    response_root = tmp_path / "responses"
    reasoner.freeze_response_bundle(run_root, raw, response_root)
    return response_root


def test_contract_freezes_authorities_adapter_comparison_and_scores() -> None:
    contract = reasoner.load_contract(CONFIG)
    assert [item.id for item in contract.arms] == reasoner.ARM_ORDER
    assert contract.authority_separation.roles == ["release", "curator", "gold", "scorer"]
    assert contract.provider_adapter.credentials_source == "environment_only"
    assert contract.comparison.unit == "paired_task_by_arm"
    assert contract.scoring.dimensions == reasoner.SCORE_DIMENSIONS
    payload = contract.model_dump(mode="json")
    payload["confirmatory"]["threshold_change_after_response_permitted"] = True
    with pytest.raises(ValueError, match="may not change"):
        reasoner.BenchmarkContract.model_validate(payload)


def test_build_creates_four_verified_write_once_bundles_without_release_leakage(
    tmp_path: Path,
) -> None:
    _, benchmark, _, _ = _built(tmp_path)
    verification = reasoner.verify_benchmark(benchmark)
    assert verification["roles"] == ["release", "curator", "gold", "scorer"]
    for role in verification["roles"]:
        assert (benchmark / role / "SUCCESS").read_bytes() == b"SUCCESS\n"
    release_text = (benchmark / "release/benchmark_release.json").read_text(encoding="utf-8")
    for forbidden in (
        "developmental_target_alpha",
        "natural_variation_dominance_beta",
        "WBGene00000001",
        "WBGene00000002",
        "alpha_gene",
        "beta_gene",
    ):
        assert forbidden not in release_text
    assert "gold" not in json.loads(release_text)["tasks"][0]
    with pytest.raises(FileExistsError, match="already exists"):
        _built(tmp_path)


def test_build_rejects_tampered_predictor_receipt(tmp_path: Path) -> None:
    contract = reasoner.load_contract(CONFIG)
    asset_root = tmp_path / "assets"
    source, custody, comparison, _ = _plans(asset_root)
    (asset_root / "predictor-receipt.txt").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="qualified predictor receipts"):
        reasoner.build_benchmark(
            contract,
            source,
            tmp_path / "benchmark",
            anonymization_secret=SECRET,
            custody_plan_or_path=custody,
            custody_asset_root=asset_root,
            comparison_plan_or_path=comparison,
            comparison_asset_root=asset_root,
            predictor_receipt_root=asset_root,
        )


def test_provider_freeze_opens_only_release_and_persists_env_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, benchmark, assets, contamination = _built(tmp_path)
    real_read = reasoner._read_json

    def guarded_read(path: Path) -> dict[str, Any]:
        if path.name in {
            "curator_authority.json",
            "benchmark_gold.json",
            "benchmark_scorer_authority.json",
        }:
            raise AssertionError("provider freeze opened a private authority")
        return real_read(path)

    monkeypatch.setattr(reasoner, "_read_json", guarded_read)
    run_root = tmp_path / "run"
    run = _run(contract, benchmark, assets, contamination, run_root)
    assert run["responses_required"] == 12
    assert run["authority_file_opened"] is False
    assert reasoner.verify_provider_run(run_root)["verified"] is True
    adapter_text = (run_root / "adapter_contract.json").read_text(encoding="utf-8")
    assert "WORMCTX_LOCAL_ENDPOINT" in adapter_text
    assert "http://" not in adapter_text and "https://" not in adapter_text


def test_provider_freeze_rejects_mismatched_contamination_attestation(tmp_path: Path) -> None:
    contract, benchmark, assets, contamination = _built(tmp_path)
    bad = contamination.model_copy(update={"model_version": "different-model"})
    with pytest.raises(ValueError, match="does not bind"):
        _run(contract, benchmark, assets, bad, tmp_path / "run")


def test_adapter_contract_rejects_endpoint_values_and_local_api_keys() -> None:
    payload = _adapter().model_dump(mode="json")
    payload["endpoint_env_var"] = "https://provider.invalid/v1"
    with pytest.raises(ValueError, match="environment-variable name"):
        reasoner.ProviderAdapterContract.model_validate(payload)
    payload = _adapter().model_dump(mode="json")
    payload["api_key_env_var"] = "SECRET_API_KEY"
    with pytest.raises(ValueError, match="must not require"):
        reasoner.ProviderAdapterContract.model_validate(payload)


def test_typed_program_enforces_ablation_and_predictor_receipt_binding(tmp_path: Path) -> None:
    source, _, _, _ = _plans(tmp_path / "assets")
    task = source.tasks[0].model_dump(mode="json")
    predictor = task["predictor"]
    call = reasoner.ProgramStep(
        step_id="predictor",
        operator="call_locked_predictor",
        depends_on=[],
        arguments={
            "predictor_id": predictor["predictor_id"],
            "version": predictor["version"],
            "input_sha256": predictor["input_sha256"],
        },
    )
    terminal = reasoner.ProgramStep(
        step_id="terminal",
        operator="emit_prediction",
        depends_on=["predictor"],
        arguments={"prediction": predictor["output"]},
    )
    ablated = reasoner.validate_program([call, terminal], allowed_tools=[], task=task)
    assert "tool_not_allowed:call_locked_predictor" in ablated["errors"]
    mismatched = call.model_copy(deep=True)
    mismatched.arguments["input_sha256"] = "d" * 64
    invalid = reasoner.validate_program(
        [mismatched, terminal], allowed_tools=["call_locked_predictor"], task=task
    )
    assert "locked_predictor_binding_differs:input_sha256" in invalid["errors"]


def test_response_freeze_rejects_incomplete_response_set(tmp_path: Path) -> None:
    contract, benchmark, assets, contamination = _built(tmp_path)
    run_root = tmp_path / "run"
    _run(contract, benchmark, assets, contamination, run_root)
    raw = _adapter_response_file(
        contract, benchmark, run_root, tmp_path / "incomplete.jsonl", omit_last=True
    )
    with pytest.raises(ValueError, match="cover the frozen requests in order"):
        reasoner.freeze_response_bundle(run_root, raw, tmp_path / "responses")


def test_scoring_verifies_response_freeze_before_opening_private_authorities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, benchmark, assets, contamination = _built(tmp_path)
    run_root = tmp_path / "run"
    _run(contract, benchmark, assets, contamination, run_root)
    responses = _frozen_responses(contract, benchmark, run_root, tmp_path)
    (responses / "submission.json").write_text("{}\n", encoding="utf-8")
    real_read = reasoner._read_json

    def guarded_read(path: Path) -> dict[str, Any]:
        if path.name in {"benchmark_gold.json", "benchmark_scorer_authority.json"}:
            raise AssertionError("private authority opened before response verification")
        return real_read(path)

    monkeypatch.setattr(reasoner, "_read_json", guarded_read)
    with pytest.raises(ValueError, match="checksum failed"):
        reasoner.score_submission(
            contract, benchmark, run_root, responses, tmp_path / "scores"
        )


def test_synthetic_scoring_uses_frozen_responses_and_separate_dimensions(tmp_path: Path) -> None:
    contract, benchmark, assets, contamination = _built(tmp_path)
    run_root = tmp_path / "run"
    _run(contract, benchmark, assets, contamination, run_root)
    responses = _frozen_responses(contract, benchmark, run_root, tmp_path)
    assert reasoner.verify_response_bundle(responses, run_root)["responses"] == 12
    score_root = tmp_path / "scores"
    manifest = reasoner.score_submission(
        contract, benchmark, run_root, responses, score_root
    )
    assert manifest["responses"] == 12
    assert manifest["response_freeze_verified_before_authority_open"] is True
    assert manifest["composite_score_reported"] is False
    assert reasoner.verify_scores(score_root)["verified"] is True
    scores = json.loads((score_root / "dimension_scores.json").read_text(encoding="utf-8"))
    assert scores["composite_score"] is None
    for arm in scores["arms"].values():
        assert arm["prediction"]["accuracy_within_frozen_tolerance"] == 1.0
        assert arm["program_validity"]["valid_rate"] == 1.0
        assert arm["abstention"]["accuracy"] == 1.0
        assert arm["execution_trace"]["exact_match_rate"] == 1.0
    for arm_id, expected_recall in (
        ("text_only", 1 / 3),
        ("ordinary_rag", 2 / 3),
        ("graph_retrieval", 2 / 3),
    ):
        assert scores["arms"][arm_id]["evidence_fidelity"][
            "citation_recall"
        ] == pytest.approx(expected_recall)


def test_trace_mismatch_is_frozen_but_substantive_scoring_fails_closed(
    tmp_path: Path,
) -> None:
    contract, benchmark, assets, contamination = _built(tmp_path)
    run_root = tmp_path / "run"
    _run(contract, benchmark, assets, contamination, run_root)
    responses = _frozen_responses(
        contract, benchmark, run_root, tmp_path, corrupt_trace=True
    )
    scores_root = tmp_path / "scores"
    reasoner.score_submission(contract, benchmark, run_root, responses, scores_root)
    tasks = json.loads((scores_root / "task_scores.json").read_text(encoding="utf-8"))["tasks"]
    failed = [item for item in tasks if not item["execution_trace"]["exact_trace_match"]]
    assert len(failed) == 1
    assert failed[0]["substantive_scores_eligible"] is False
    assert failed[0]["prediction"]["correct"] is False


def test_submission_rejects_post_freeze_changes() -> None:
    with pytest.raises(ValueError, match="violates the confirmatory freeze"):
        reasoner.Submission.model_validate(
            {
                "schema_version": reasoner.SUBMISSION_VERSION,
                "run_contract_sha256": "a" * 64,
                "benchmark_release_sha256": "b" * 64,
                "provider_spec_sha256": "c" * 64,
                "adapter_contract_sha256": "d" * 64,
                "contamination_attestation_sha256": "e" * 64,
                "response_receipts_sha256": "f" * 64,
                "model_or_decoding_changed_after_freeze": True,
                "thresholds_or_endpoints_changed_after_freeze": False,
                "gold_accessed_before_response_freeze": False,
                "responses": [],
            }
        )
