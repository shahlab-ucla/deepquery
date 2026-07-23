"""End-to-end, receipt-producing training path for the bounded POC."""

from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from wormctx.pipeline import (
    build_graph,
    build_transport_map,
    verify_build_receipt,
    write_build_receipt,
)

from .config import PocConfig
from .contracts import (
    EDGE_TYPE_ORDER,
    FAMILY_ORDER,
    INVALID_FLAG_ORDER,
    NODE_TYPE_ORDER,
    OPERATOR_ORDER,
)
from .generation import dataset_summary, generate_configured_episodes, split_episodes
from .metrics import (
    evaluate_constant_baseline,
    evaluate_model,
    fit_invalid_thresholds,
    fit_constant_baseline,
)
from .model import (
    OutputDimensions,
    SharedGraphReasoner,
    model_metadata,
    multitask_loss,
)
from .repro import (
    canonical_json_bytes,
    git_manifest,
    runtime_manifest,
    sha256_bytes,
    sha256_file,
    utc_now,
    verify_checksums,
    write_checksums,
    write_json,
    write_jsonl,
    write_text,
)
from .tensorize import (
    VOCABULARY,
    EpisodeTensorDataset,
)


def _seed_everything(seed: int, deterministic: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            # The fused/memory-efficient attention kernels are fast but may use
            # nondeterministic backward algorithms on Ampere. The math kernel is
            # slower and reproducible, which is the correct tradeoff for this POC.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)


def _device(config: PocConfig) -> torch.device:
    requested = config.training.device
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requires CUDA but torch.cuda.is_available() is false")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" or torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_context_graph_preflight(repository_root: Path, output: Path) -> dict[str, Any]:
    """Exercise the inherited evidence/transport path unchanged and verify its receipt."""
    observations = repository_root / "data" / "examples" / "observations.jsonl"
    query = repository_root / "data" / "examples" / "query_context.json"
    policy = repository_root / "config" / "transport_policy.json"
    graph_summary = build_graph(observations, output)
    transport_summary = build_transport_map(
        observations,
        query,
        policy,
        output,
        carry_forward_files=graph_summary["produced_files"],
    )
    summary = {
        "graph": graph_summary,
        "transport": transport_summary,
        "input_binding": {
            "status": "unbound_bundled_synthetic_fixture",
            "warning": "context preflight uses committed synthetic fixtures, not production data",
        },
    }
    write_build_receipt(output, summary)
    verification = verify_build_receipt(
        output,
        local_inputs={
            "observations_sha256": [observations],
            "query_sha256": [query],
            "policy_sha256": [policy],
        },
    )
    return {"summary": summary, "verification": verification}


def _loader(
    dataset: EpisodeTensorDataset,
    config: PocConfig,
    *,
    shuffle: bool,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(config.training.seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def _tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }


def _train_epoch(
    model: SharedGraphReasoner,
    loader: DataLoader[Any],
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: PocConfig,
    invalid_pos_weight: Tensor,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    examples = 0
    use_amp = config.training.amp and device.type == "cuda"
    for raw_batch in loader:
        batch = _tensor_batch(raw_batch, device)
        batch_size = int(batch["family_target"].shape[0])
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            predictions = model(batch)
            loss, components = multitask_loss(
                predictions,
                batch,
                invalid_pos_weight=invalid_pos_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        totals["total"] = totals.get("total", 0.0) + float(loss.detach()) * batch_size
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        examples += batch_size
    return {name: value / examples for name, value in totals.items()}


def _save_training_state(
    path: Path,
    *,
    epoch: int,
    model: SharedGraphReasoner,
    optimizer: AdamW,
    best_score: float,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_validation_score": best_score,
        },
        temporary,
    )
    os.replace(temporary, path)


def _validation_score(metrics: dict[str, float]) -> float:
    terms = (
        metrics["operator_micro_f1"],
        metrics["invalid_flag_micro_recall"],
        metrics["invalid_flag_micro_f1"],
        metrics["invalid_design_rejection_accuracy"],
        metrics["family_accuracy"],
        metrics["conclusion_accuracy"],
    )
    return sum(terms) / len(terms)


def _run_id(config: PocConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = sha256_bytes(canonical_json_bytes(config.model_dump(mode="json")))[:10]
    return f"{config.run_name}-{stamp}-{digest}"


def _markdown_report(
    *,
    run_id: str,
    passed: bool,
    model_info: dict[str, Any],
    data_info: dict[str, Any],
    test_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    gates: dict[str, dict[str, Any]],
    peak_vram_gib: float,
    elapsed_seconds: float,
) -> str:
    metric_names = (
        "family_accuracy",
        "operator_micro_f1",
        "operator_exact_match",
        "invalid_flag_micro_recall",
        "invalid_design_rejection_accuracy",
        "conclusion_accuracy",
        "hypothesis_accuracy",
        "experiment_accuracy",
        "resolution_mae",
    )
    metric_rows = "\n".join(
        f"| {name} | {test_metrics[name]:.4f} | {baseline_metrics[name]:.4f} |"
        for name in metric_names
    )
    gate_rows = "\n".join(
        f"| {name} | {'PASS' if item['passed'] else 'FAIL'} | "
        f"{item['observed']} | {item['required']} |"
        for name, item in gates.items()
    )
    return f"""# C. elegans graph-conditioned inference POC

Run `{run_id}` {'passed' if passed else 'did not pass'} its predeclared engineering gates.

## Scope

This is an **exploratory framework-feasibility run over provenance-labeled simulations**.
It is not a biological discovery, a calibrated mechanism posterior, or evidence that an
experiment will work in vivo. The inherited `wormctx` contextual graph was also rebuilt
and receipt-verified from its committed synthetic fixture.

## Runtime

- Trainable parameters: {model_info['parameter_count']:,}
- Episodes: {data_info['episode_count']:,}
- Peak allocated GPU memory: {peak_vram_gib:.3f} GiB
- Elapsed wall time: {elapsed_seconds:.1f} seconds

## Locked test results

| Metric | graph model | constant baseline |
|---|---:|---:|
{metric_rows}

Hypothesis accuracy is included as a diagnostic, not an acceptance gate: the simulator's
true hypothesis index is intentionally weakly supervised and is not a real causal label.

## Acceptance gates

| Gate | Status | Observed | Required |
|---|---|---:|---:|
{gate_rows}

## Next scientific gate

Select one rights-reviewed CaeNDR trait with raw/control measurements and a published
benchmark locus. Reproduce its conventional phenotype, kinship, and haplotype baseline
before using any neural score in a biological ranking. Keep processed developmental data
as a secondary routing demonstration until its observation and split contracts are frozen.
"""


def run_training(
    config: PocConfig,
    *,
    output_root: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    _seed_everything(config.training.seed, config.training.deterministic)
    device = _device(config)
    output_root = Path(output_root).resolve()
    repository_root = Path(repository_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(config)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    (run_dir / "data").mkdir()
    (run_dir / "model").mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "reports").mkdir()

    write_json(run_dir / "config.resolved.json", config.model_dump(mode="json"))
    write_json(run_dir / "vocabulary.json", VOCABULARY)
    context_preflight = run_context_graph_preflight(
        repository_root, run_dir / "context_graph_preflight"
    )

    episodes = generate_configured_episodes(config)
    splits = split_episodes(episodes)
    data_info = dataset_summary(episodes)
    split_hashes: dict[str, str] = {}
    for split, values in splits.items():
        path = run_dir / "data" / f"{split}.jsonl"
        write_jsonl(
            path,
            (item.model_dump(mode="json", exclude_none=True) for item in values),
        )
        split_hashes[split] = sha256_file(path)
    data_manifest = {
        "scope": "scientifically_structured_simulation_only",
        "biological_claims_permitted": False,
        "summary": data_info,
        "split_sha256": split_hashes,
        "group_disjoint": True,
        "generator": "wormctx.poc.simulation:1.0",
    }
    write_json(run_dir / "data" / "manifest.json", data_manifest)

    max_edges = max(len(item.graph.edges) for item in episodes)
    dataset_args = {
        "max_nodes": config.model.max_nodes,
        "numeric_feature_dim": config.model.numeric_feature_dim,
        "max_hypotheses": config.model.max_hypotheses,
        "max_experiments": config.model.max_experiments,
        "max_edges": max_edges,
    }
    datasets = {
        split: EpisodeTensorDataset(values, **dataset_args)
        for split, values in splits.items()
    }
    loaders = {
        split: _loader(dataset, config, shuffle=split == "train")
        for split, dataset in datasets.items()
    }
    # Evaluation loaders never shuffle; the training loader above is used only for updates.
    train_eval_loader = _loader(datasets["train"], config, shuffle=False)
    baseline = fit_constant_baseline(train_eval_loader)
    baseline_metrics = evaluate_constant_baseline(baseline, loaders["test"])
    invalid_targets = torch.stack(
        [item["invalid_targets"] for item in datasets["train"].items]
    )
    positive = invalid_targets.sum(dim=0)
    negative = invalid_targets.shape[0] - positive
    invalid_pos_weight = torch.sqrt(negative / positive.clamp_min(1.0)).to(device)

    output_dimensions = OutputDimensions(
        families=len(FAMILY_ORDER),
        operators=len(OPERATOR_ORDER),
        invalid_flags=len(INVALID_FLAG_ORDER),
        conclusions=len(VOCABULARY["conclusions"]),
        hypotheses=config.model.max_hypotheses,
        experiments=config.model.max_experiments,
    )
    model = SharedGraphReasoner(
        numeric_feature_dim=config.model.numeric_feature_dim,
        hidden_dim=config.model.hidden_dim,
        node_type_count=len(NODE_TYPE_ORDER),
        edge_type_count=len(EDGE_TYPE_ORDER),
        transformer_layers=config.model.transformer_layers,
        message_passing_layers=config.model.message_passing_layers,
        attention_heads=config.model.attention_heads,
        dropout=config.model.dropout,
        outputs=output_dimensions,
    ).to(device)
    model_info = model_metadata(model)
    model_info["invalid_positive_class_weights"] = [
        float(value) for value in invalid_pos_weight.detach().cpu()
    ]
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=config.training.amp and device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_state: dict[str, Tensor] | None = None
    best_invalid_thresholds: Tensor | None = None
    best_epoch = 0
    for epoch in range(1, config.training.epochs + 1):
        epoch_started = time.perf_counter()
        train_losses = _train_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            config,
            invalid_pos_weight,
        )
        validation_thresholds = fit_invalid_thresholds(model, loaders["val"], device)
        validation_metrics, _ = evaluate_model(
            model,
            loaders["val"],
            device,
            invalid_thresholds=validation_thresholds,
        )
        score = _validation_score(validation_metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_invalid_thresholds = validation_thresholds.clone()
        _save_training_state(
            run_dir / "model" / "training_state.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_score=best_score,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_losses,
                "validation_metrics": validation_metrics,
                "validation_composite": score,
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
        )
        write_jsonl(run_dir / "logs" / "epochs.jsonl", history)

    if best_state is None or best_invalid_thresholds is None:
        raise RuntimeError("training produced no checkpoint")
    final_path = run_dir / "model" / "best.safetensors"
    save_safetensors(
        {name: value.contiguous() for name, value in best_state.items()},
        str(final_path),
        metadata={
            "schema_version": config.schema_version,
            "run_id": run_id,
            "best_epoch": str(best_epoch),
            "claims_scope": "simulation_framework_feasibility_only",
        },
    )
    # Reload from the portable, non-pickle checkpoint before locked test evaluation.
    model.load_state_dict(load_safetensors(str(final_path), device=str(device)))
    checkpoint_reloaded = True
    write_json(
        run_dir / "model" / "decision_thresholds.json",
        {
            "fit_split": "validation",
            "invalid_flag_thresholds": {
                name: float(best_invalid_thresholds[index])
                for index, name in enumerate(INVALID_FLAG_ORDER)
            },
            "test_data_used_for_threshold_selection": False,
        },
    )
    test_metrics, prediction_rows = evaluate_model(
        model,
        loaders["test"],
        device,
        include_predictions=True,
        invalid_thresholds=best_invalid_thresholds,
    )
    write_jsonl(run_dir / "predictions.test.jsonl", prediction_rows)
    write_json(
        run_dir / "metrics.json",
        {
            "best_epoch": best_epoch,
            "best_validation_composite": best_score,
            "test": test_metrics,
            "constant_baseline_test": baseline_metrics,
        },
    )
    peak_vram_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    gates = {
        "context_graph_receipt": {
            "observed": context_preflight["verification"]["status"],
            "required": "authoritative_build_snapshot_verified",
            "passed": context_preflight["verification"]["status"]
            == "authoritative_build_snapshot_verified",
        },
        "cuda": {
            "observed": device.type,
            "required": "cuda" if config.acceptance.require_cuda else "cpu_or_cuda",
            "passed": device.type == "cuda" or not config.acceptance.require_cuda,
        },
        "operator_micro_f1": {
            "observed": test_metrics["operator_micro_f1"],
            "required": config.acceptance.minimum_operator_micro_f1,
            "passed": test_metrics["operator_micro_f1"]
            >= config.acceptance.minimum_operator_micro_f1,
        },
        "invalid_flag_recall": {
            "observed": test_metrics["invalid_flag_micro_recall"],
            "required": config.acceptance.minimum_invalid_flag_recall,
            "passed": test_metrics["invalid_flag_micro_recall"]
            >= config.acceptance.minimum_invalid_flag_recall,
        },
        "invalid_flag_f1": {
            "observed": test_metrics["invalid_flag_micro_f1"],
            "required": config.acceptance.minimum_invalid_flag_f1,
            "passed": test_metrics["invalid_flag_micro_f1"]
            >= config.acceptance.minimum_invalid_flag_f1,
        },
        "invalid_design_accuracy": {
            "observed": test_metrics["invalid_design_rejection_accuracy"],
            "required": config.acceptance.minimum_invalid_design_accuracy,
            "passed": test_metrics["invalid_design_rejection_accuracy"]
            >= config.acceptance.minimum_invalid_design_accuracy,
        },
        "peak_vram_gib": {
            "observed": peak_vram_gib,
            "required": config.acceptance.maximum_peak_vram_gib,
            "passed": peak_vram_gib <= config.acceptance.maximum_peak_vram_gib,
        },
        "checkpoint_reload": {
            "observed": checkpoint_reloaded,
            "required": True,
            "passed": checkpoint_reloaded,
        },
    }
    passed = all(item["passed"] for item in gates.values())
    elapsed_seconds = time.perf_counter() - started
    run_manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "mode": config.mode,
        "claims_scope": "simulation_framework_feasibility_only",
        "warning": "neural scores are not calibrated biological posteriors",
        "config_sha256": sha256_file(run_dir / "config.resolved.json"),
        "data": data_manifest,
        "model": model_info,
        "runtime": runtime_manifest(),
        "git": git_manifest(repository_root),
        "device": str(device),
        "peak_vram_gib": peak_vram_gib,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_reloaded": checkpoint_reloaded,
        "context_graph_preflight": context_preflight["verification"],
        "acceptance_gates": gates,
        "passed": passed,
    }
    write_json(run_dir / "run_manifest.json", run_manifest)
    write_text(
        run_dir / "reports" / "summary.md",
        _markdown_report(
            run_id=run_id,
            passed=passed,
            model_info=model_info,
            data_info=data_info,
            test_metrics=test_metrics,
            baseline_metrics=baseline_metrics,
            gates=gates,
            peak_vram_gib=peak_vram_gib,
            elapsed_seconds=elapsed_seconds,
        ),
    )
    write_checksums(run_dir)
    checksum_failures = verify_checksums(run_dir)
    if checksum_failures:
        raise RuntimeError(f"run checksum verification failed: {checksum_failures}")
    if passed:
        write_text(run_dir / "SUCCESS", f"{run_id}\n")
    else:
        write_text(run_dir / "FAILED_ACCEPTANCE", f"{run_id}\n")
    return {
        "run_id": run_id,
        "run_directory": str(run_dir),
        "passed": passed,
        "test_metrics": test_metrics,
        "constant_baseline_test": baseline_metrics,
        "peak_vram_gib": peak_vram_gib,
        "elapsed_seconds": elapsed_seconds,
        "parameter_count": model_info["parameter_count"],
    }
