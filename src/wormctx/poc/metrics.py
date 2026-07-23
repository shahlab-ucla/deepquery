"""Evaluation and cheap constant baselines for the POC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .contracts import (
    CONCLUSION_ORDER,
    FAMILY_ORDER,
    INVALID_FLAG_ORDER,
    OPERATOR_ORDER,
)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(predicted: Tensor, target: Tensor, prefix: str) -> dict[str, float]:
    predicted = predicted.bool()
    target = target.bool()
    true_positive = float((predicted & target).sum().item())
    false_positive = float((predicted & ~target).sum().item())
    false_negative = float((~predicted & target).sum().item())
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    exact = float((predicted == target).all(dim=1).float().mean().item())
    return {
        f"{prefix}_micro_precision": precision,
        f"{prefix}_micro_recall": recall,
        f"{prefix}_micro_f1": f1,
        f"{prefix}_exact_match": exact,
    }


@dataclass(frozen=True)
class ConstantBaseline:
    family: int
    operator_prediction: Tensor
    invalid_prediction: Tensor
    conclusion: int
    hypothesis: int
    experiment: int
    resolution: float


def fit_constant_baseline(loader: Any) -> ConstantBaseline:
    family: list[Tensor] = []
    operators: list[Tensor] = []
    invalid: list[Tensor] = []
    conclusion: list[Tensor] = []
    hypothesis: list[Tensor] = []
    experiment: list[Tensor] = []
    resolution: list[Tensor] = []
    for batch in loader:
        family.append(batch["family_target"])
        operators.append(batch["operator_targets"])
        invalid.append(batch["invalid_targets"])
        conclusion.append(batch["conclusion_target"])
        hypothesis.append(batch["hypothesis_target"])
        experiment.append(batch["experiment_target"])
        resolution.append(batch["resolution_target"])
    family_values = torch.cat(family)
    conclusion_values = torch.cat(conclusion)
    hypothesis_values = torch.cat(hypothesis)
    experiment_values = torch.cat(experiment)
    return ConstantBaseline(
        family=int(torch.bincount(family_values).argmax().item()),
        operator_prediction=torch.cat(operators).mean(dim=0).ge(0.5),
        invalid_prediction=torch.cat(invalid).mean(dim=0).ge(0.5),
        conclusion=int(torch.bincount(conclusion_values).argmax().item()),
        hypothesis=int(torch.bincount(hypothesis_values).argmax().item()),
        experiment=int(torch.bincount(experiment_values).argmax().item()),
        resolution=float(torch.cat(resolution).mean().item()),
    )


def evaluate_constant_baseline(baseline: ConstantBaseline, loader: Any) -> dict[str, float]:
    targets = _collect_targets(loader)
    count = targets["family"].shape[0]
    family_prediction = torch.full((count,), baseline.family, dtype=torch.long)
    conclusion_prediction = torch.full((count,), baseline.conclusion, dtype=torch.long)
    hypothesis_prediction = torch.full((count,), baseline.hypothesis, dtype=torch.long)
    experiment_prediction = torch.full((count,), baseline.experiment, dtype=torch.long)
    operator_prediction = baseline.operator_prediction.unsqueeze(0).expand(count, -1)
    invalid_prediction = baseline.invalid_prediction.unsqueeze(0).expand(count, -1)
    resolution_prediction = torch.full((count,), baseline.resolution)
    return _aggregate_metrics(
        family_prediction,
        operator_prediction,
        invalid_prediction,
        conclusion_prediction,
        hypothesis_prediction,
        experiment_prediction,
        resolution_prediction,
        targets,
    )


def _collect_targets(loader: Any) -> dict[str, Tensor]:
    collected: dict[str, list[Tensor]] = {
        "family": [],
        "operators": [],
        "invalid": [],
        "conclusion": [],
        "hypothesis": [],
        "experiment": [],
        "resolution": [],
    }
    keys = {
        "family": "family_target",
        "operators": "operator_targets",
        "invalid": "invalid_targets",
        "conclusion": "conclusion_target",
        "hypothesis": "hypothesis_target",
        "experiment": "experiment_target",
        "resolution": "resolution_target",
    }
    for batch in loader:
        for target_name, batch_name in keys.items():
            collected[target_name].append(batch[batch_name].detach().cpu())
    return {name: torch.cat(parts) for name, parts in collected.items()}


def _aggregate_metrics(
    family_prediction: Tensor,
    operator_prediction: Tensor,
    invalid_prediction: Tensor,
    conclusion_prediction: Tensor,
    hypothesis_prediction: Tensor,
    experiment_prediction: Tensor,
    resolution_prediction: Tensor,
    targets: dict[str, Tensor],
) -> dict[str, float]:
    metrics = {
        "family_accuracy": float(
            (family_prediction == targets["family"]).float().mean().item()
        ),
        "conclusion_accuracy": float(
            (conclusion_prediction == targets["conclusion"]).float().mean().item()
        ),
        "hypothesis_accuracy": float(
            (hypothesis_prediction == targets["hypothesis"]).float().mean().item()
        ),
        "experiment_accuracy": float(
            (experiment_prediction == targets["experiment"]).float().mean().item()
        ),
        "resolution_mae": float(
            (resolution_prediction - targets["resolution"]).abs().mean().item()
        ),
        "invalid_design_rejection_accuracy": float(
            (
                invalid_prediction.any(dim=1)
                == targets["invalid"].bool().any(dim=1)
            )
            .float()
            .mean()
            .item()
        ),
    }
    metrics.update(binary_metrics(operator_prediction, targets["operators"], "operator"))
    metrics.update(binary_metrics(invalid_prediction, targets["invalid"], "invalid_flag"))
    return metrics


def _move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def fit_invalid_thresholds(
    model: nn.Module,
    loader: Any,
    device: torch.device,
) -> Tensor:
    """Choose per-flag thresholds on validation data only, maximizing F1 then accuracy."""
    model.eval()
    probability_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_tensor_batch(raw_batch, device)
            probability_parts.append(model(batch)["invalid_logits"].sigmoid().cpu())
            target_parts.append(batch["invalid_targets"].bool().cpu())
    probabilities = torch.cat(probability_parts)
    targets = torch.cat(target_parts)
    candidate_thresholds = torch.linspace(0.02, 0.98, 97)
    chosen: list[float] = []
    for flag_index in range(probabilities.shape[1]):
        flag_probability = probabilities[:, flag_index]
        flag_target = targets[:, flag_index]
        best_key: tuple[float, float, float] | None = None
        best_threshold = 0.5
        for threshold in candidate_thresholds:
            predicted = flag_probability.ge(threshold)
            true_positive = float((predicted & flag_target).sum())
            false_positive = float((predicted & ~flag_target).sum())
            false_negative = float((~predicted & flag_target).sum())
            precision = _safe_divide(true_positive, true_positive + false_positive)
            recall = _safe_divide(true_positive, true_positive + false_negative)
            f1 = _safe_divide(2.0 * precision * recall, precision + recall)
            accuracy = float((predicted == flag_target).float().mean())
            key = (f1, accuracy, -abs(float(threshold) - 0.5))
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(threshold)
        chosen.append(best_threshold)
    return torch.tensor(chosen, dtype=torch.float32)


def evaluate_model(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    *,
    include_predictions: bool = False,
    invalid_thresholds: Tensor | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    target_parts: dict[str, list[Tensor]] = {
        name: []
        for name in (
            "family",
            "operators",
            "invalid",
            "conclusion",
            "hypothesis",
            "experiment",
            "resolution",
        )
    }
    prediction_parts: dict[str, list[Tensor]] = {
        name: [] for name in target_parts
    }
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_tensor_batch(raw_batch, device)
            output = model(batch)
            family_probability = output["family_logits"].softmax(dim=-1)
            operator_probability = output["operator_logits"].sigmoid()
            invalid_probability = output["invalid_logits"].sigmoid()
            conclusion_probability = output["conclusion_logits"].softmax(dim=-1)
            hypothesis_logits = output["hypothesis_logits"].masked_fill(
                ~batch["hypothesis_mask"], torch.finfo(output["hypothesis_logits"].dtype).min
            )
            experiment_logits = output["experiment_logits"].masked_fill(
                ~batch["experiment_mask"], torch.finfo(output["experiment_logits"].dtype).min
            )
            hypothesis_probability = hypothesis_logits.softmax(dim=-1)
            experiment_probability = experiment_logits.softmax(dim=-1)

            thresholds = (
                invalid_thresholds.to(device)
                if invalid_thresholds is not None
                else torch.full(
                    (invalid_probability.shape[1],),
                    0.5,
                    dtype=invalid_probability.dtype,
                    device=device,
                )
            )
            current_predictions = {
                "family": family_probability.argmax(dim=-1),
                "operators": operator_probability.ge(0.5),
                "invalid": invalid_probability.ge(thresholds.unsqueeze(0)),
                "conclusion": conclusion_probability.argmax(dim=-1),
                "hypothesis": hypothesis_probability.argmax(dim=-1),
                "experiment": experiment_probability.argmax(dim=-1),
                "resolution": output["resolution"],
            }
            current_targets = {
                "family": batch["family_target"],
                "operators": batch["operator_targets"],
                "invalid": batch["invalid_targets"],
                "conclusion": batch["conclusion_target"],
                "hypothesis": batch["hypothesis_target"],
                "experiment": batch["experiment_target"],
                "resolution": batch["resolution_target"],
            }
            for name in target_parts:
                target_parts[name].append(current_targets[name].detach().cpu())
                prediction_parts[name].append(current_predictions[name].detach().cpu())

            if include_predictions:
                for index, episode_id in enumerate(raw_batch["episode_id"]):
                    predicted_operators = [
                        OPERATOR_ORDER[position]
                        for position in range(len(OPERATOR_ORDER))
                        if bool(current_predictions["operators"][index, position])
                    ]
                    predicted_flags = [
                        INVALID_FLAG_ORDER[position]
                        for position in range(len(INVALID_FLAG_ORDER))
                        if bool(current_predictions["invalid"][index, position])
                    ]
                    rows.append(
                        {
                            "episode_id": episode_id,
                            "split_group": raw_batch["split_group"][index],
                            "predicted_family": FAMILY_ORDER[
                                int(current_predictions["family"][index])
                            ],
                            "target_family": FAMILY_ORDER[int(batch["family_target"][index])],
                            "predicted_operators": predicted_operators,
                            "target_operators": [
                                OPERATOR_ORDER[position]
                                for position in range(len(OPERATOR_ORDER))
                                if bool(batch["operator_targets"][index, position])
                            ],
                            "predicted_invalid_flags": predicted_flags,
                            "target_invalid_flags": [
                                INVALID_FLAG_ORDER[position]
                                for position in range(len(INVALID_FLAG_ORDER))
                                if bool(batch["invalid_targets"][index, position])
                            ],
                            "predicted_resolution_ceiling": CONCLUSION_ORDER[
                                int(current_predictions["conclusion"][index])
                            ],
                            "target_resolution_ceiling": CONCLUSION_ORDER[
                                int(batch["conclusion_target"][index])
                            ],
                            "predicted_hypothesis_index": int(
                                current_predictions["hypothesis"][index]
                            ),
                            "target_hypothesis_index": int(batch["hypothesis_target"][index]),
                            "predicted_experiment_index": int(
                                current_predictions["experiment"][index]
                            ),
                            "target_experiment_index": int(batch["experiment_target"][index]),
                            "predicted_resolution_score": float(output["resolution"][index]),
                            "target_resolution_score": float(batch["resolution_target"][index]),
                        }
                    )
    targets = {name: torch.cat(parts) for name, parts in target_parts.items()}
    predictions = {name: torch.cat(parts) for name, parts in prediction_parts.items()}
    metrics = _aggregate_metrics(
        predictions["family"],
        predictions["operators"],
        predictions["invalid"],
        predictions["conclusion"],
        predictions["hypothesis"],
        predictions["experiment"],
        predictions["resolution"],
        targets,
    )
    return metrics, rows
