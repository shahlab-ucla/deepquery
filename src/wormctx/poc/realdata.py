"""Read-only CaeNDR phenotype-prediction baseline for exploratory data audits.

This module intentionally implements a conventional baseline rather than a
biological discovery workflow.  It aligns long-form phenotypes, a PLINK square
relationship matrix, and PLINK PCA eigenvectors, then evaluates a training-mean
predictor against kinship kernel ridge regression (the usual GBLUP prediction
form) with nested, population-group-held-out cross-validation.

The returned evidence is exploratory and unvalidated.  It cannot identify QTLs,
causal variants, biological mechanisms, or calibrated biological effects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_ALPHA_GRID = tuple(float(value) for value in np.logspace(-4, 4, 9))
_MISSING_PHENOTYPES = {"", ".", "na", "n/a", "nan", "null", "none"}
_MAX_KMEANS_ITERATIONS = 200
_MAX_CLUSTERING_PCS = 10


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _resolve_input(path: str | Path, label: str) -> Path:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"{label} input is not a file: {target}")
    return target


def _load_phenotypes(
    path: Path,
) -> tuple[dict[tuple[str, str], dict[str, float]], set[str], list[str]]:
    observations: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_strains: set[str] = set()
    warnings: list[str] = []
    missing_value_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("phenotype CSV has no header")
        normalized: dict[str, str] = {}
        for raw_name in reader.fieldnames:
            name = raw_name.strip().lower()
            if name in normalized:
                raise ValueError(f"phenotype CSV has duplicate column {name!r}")
            normalized[name] = raw_name
        required = {"strain", "condition", "trait", "phenotype"}
        missing_columns = sorted(required - normalized.keys())
        if missing_columns:
            raise ValueError(
                "phenotype CSV is missing required columns: " + ", ".join(missing_columns)
            )

        for row_number, row in enumerate(reader, start=2):
            strain = (row.get(normalized["strain"]) or "").strip()
            condition = (row.get(normalized["condition"]) or "").strip()
            trait = (row.get(normalized["trait"]) or "").strip()
            raw_value = (row.get(normalized["phenotype"]) or "").strip()
            if not strain or not condition or not trait:
                raise ValueError(
                    f"phenotype CSV row {row_number} has a blank strain, condition, or trait"
                )
            all_strains.add(strain)
            if raw_value.casefold() in _MISSING_PHENOTYPES:
                missing_value_rows += 1
                continue
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"phenotype CSV row {row_number} has a non-numeric phenotype"
                ) from exc
            if not math.isfinite(value):
                missing_value_rows += 1
                continue
            observations[(condition, trait)][strain].append(value)

    if not observations:
        raise ValueError("phenotype CSV contains no finite phenotype observations")
    if missing_value_rows:
        warnings.append(
            f"ignored {missing_value_rows} row(s) with missing or non-finite phenotypes"
        )

    aggregated: dict[tuple[str, str], dict[str, float]] = {}
    replicated_cells = 0
    extra_replicates = 0
    for analysis_key, by_strain in observations.items():
        aggregated[analysis_key] = {}
        for strain, values in by_strain.items():
            if len(values) > 1:
                replicated_cells += 1
                extra_replicates += len(values) - 1
            aggregated[analysis_key][strain] = float(math.fsum(values) / len(values))
    if replicated_cells:
        warnings.append(
            "averaged "
            f"{extra_replicates} extra replicate row(s) across {replicated_cells} "
            "condition/trait/strain cell(s)"
        )
    return aggregated, all_strains, warnings


def _load_kinship_ids(path: Path) -> list[str]:
    identifiers: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if not identifiers and len(fields) >= 2:
            if fields[0].lstrip("#").casefold() == "fid" and fields[1].casefold() == "iid":
                continue
        identifier = fields[1] if len(fields) >= 2 else fields[0]
        identifier = identifier.strip()
        if not identifier:
            raise ValueError(f"kinship ID file line {line_number} has a blank identifier")
        identifiers.append(identifier)
    if not identifiers:
        raise ValueError("kinship ID file contains no identifiers")
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"kinship ID file contains duplicate IID values: {preview}")
    return identifiers


def _load_kinship(matrix_path: Path, ids_path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    identifiers = _load_kinship_ids(ids_path)
    rows: list[list[float]] = []
    for line_number, line in enumerate(
        matrix_path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        fields = line.replace(",", " ").split()
        if not fields:
            continue
        try:
            row = [float(value) for value in fields]
        except ValueError as exc:
            raise ValueError(
                f"kinship matrix line {line_number} contains a non-numeric value"
            ) from exc
        rows.append(row)
    expected = len(identifiers)
    if len(rows) != expected or any(len(row) != expected for row in rows):
        dimensions = f"{len(rows)}x{len(rows[0]) if rows else 0}"
        raise ValueError(
            f"kinship matrix must be square and match {expected} IDs; observed {dimensions}"
        )
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("kinship matrix contains non-finite values")

    warnings: list[str] = []
    scale = max(1.0, float(np.max(np.abs(matrix))))
    maximum_asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if maximum_asymmetry > 1e-5 * scale:
        raise ValueError(
            "kinship matrix is not symmetric "
            f"(maximum absolute asymmetry {maximum_asymmetry:.6g})"
        )
    if maximum_asymmetry > 0.0:
        matrix = (matrix + matrix.T) / 2.0
        warnings.append(
            "symmetrized the kinship matrix within numeric tolerance "
            f"(maximum absolute asymmetry {maximum_asymmetry:.6g})"
        )

    eigenvalues = np.linalg.eigvalsh(matrix)
    eigen_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    minimum_eigenvalue = float(eigenvalues[0])
    if minimum_eigenvalue < -1e-5 * eigen_scale:
        raise ValueError(
            "kinship matrix is not positive semidefinite within tolerance "
            f"(minimum eigenvalue {minimum_eigenvalue:.6g})"
        )
    if minimum_eigenvalue < 0.0:
        warnings.append(
            "kinship matrix has a small negative eigenvalue within numeric tolerance "
            f"({minimum_eigenvalue:.6g}); ridge regularization is retained"
        )
    return identifiers, matrix, warnings


def _load_eigenvectors(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    lines = [
        line.split()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.split()
    ]
    if not lines:
        raise ValueError("PCA eigenvector file is empty")

    header_tokens = [token.lstrip("#").casefold() for token in lines[0]]
    has_header = "iid" in header_tokens and any(token.startswith("pc") for token in header_tokens)
    if has_header:
        iid_index = header_tokens.index("iid")
        pc_indices = [
            index for index, token in enumerate(header_tokens) if token.startswith("pc")
        ]
        data_lines = lines[1:]
    else:
        if len(lines[0]) < 3:
            raise ValueError(
                "headerless PCA eigenvector rows must contain FID, IID, and at least one PC"
            )
        iid_index = 1
        pc_indices = list(range(2, len(lines[0])))
        data_lines = lines
    if not pc_indices:
        raise ValueError("PCA eigenvector file contains no PC columns")

    identifiers: list[str] = []
    vectors: list[list[float]] = []
    required_columns = max([iid_index, *pc_indices]) + 1
    for line_number, fields in enumerate(data_lines, start=2 if has_header else 1):
        if len(fields) < required_columns:
            raise ValueError(f"PCA eigenvector line {line_number} has too few columns")
        identifier = fields[iid_index].strip()
        if not identifier:
            raise ValueError(f"PCA eigenvector line {line_number} has a blank IID")
        try:
            vector = [float(fields[index]) for index in pc_indices]
        except ValueError as exc:
            raise ValueError(
                f"PCA eigenvector line {line_number} contains a non-numeric PC"
            ) from exc
        identifiers.append(identifier)
        vectors.append(vector)

    if not identifiers:
        raise ValueError("PCA eigenvector file contains no sample rows")
    if len(set(identifiers)) != len(identifiers):
        duplicates = sorted(
            {item for item in identifiers if identifiers.count(item) > 1}
        )
        raise ValueError(
            "PCA eigenvector file contains duplicate IID values: " + ", ".join(duplicates[:5])
        )
    matrix = np.asarray(vectors, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("PCA eigenvector file contains non-finite PC values")
    warnings: list[str] = []
    if matrix.shape[1] > _MAX_CLUSTERING_PCS:
        warnings.append(
            f"used the first {_MAX_CLUSTERING_PCS} of {matrix.shape[1]} PCs for grouping"
        )
    return identifiers, matrix, warnings


def _stable_order_key(seed: int, identifier: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{identifier}".encode("utf-8")).digest()


def _group_count(sample_count: int) -> int:
    return min(sample_count, 5, max(3, int(round(math.sqrt(sample_count)))))


def _minimum_cost_capacity_assignment(
    distances: np.ndarray,
    capacities: np.ndarray,
) -> np.ndarray:
    """Assign rows to centers exactly while respecting every center capacity.

    Center columns are expanded into identical capacity slots and the square
    linear-assignment problem is solved with the deterministic Hungarian
    algorithm.  This is practical for the hundreds of CaeNDR strains targeted
    by the POC and avoids an optional SciPy dependency.
    """

    sample_count, group_count = distances.shape
    if len(capacities) != group_count or int(np.sum(capacities)) != sample_count:
        raise ValueError("balanced-group capacities do not match the distance matrix")
    slot_groups = np.repeat(np.arange(group_count, dtype=np.int64), capacities)
    costs = distances[:, slot_groups]

    # Hungarian shortest-augmenting-path algorithm, using one-based work arrays.
    row_potential = np.zeros(sample_count + 1, dtype=np.float64)
    column_potential = np.zeros(sample_count + 1, dtype=np.float64)
    matched_row = np.zeros(sample_count + 1, dtype=np.int64)
    predecessor = np.zeros(sample_count + 1, dtype=np.int64)
    for row in range(1, sample_count + 1):
        matched_row[0] = row
        current_column = 0
        minimum_reduced_cost = np.full(sample_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(sample_count + 1, dtype=bool)
        while True:
            used[current_column] = True
            current_row = int(matched_row[current_column])
            delta = math.inf
            next_column = 0
            for column in range(1, sample_count + 1):
                if used[column]:
                    continue
                reduced_cost = (
                    costs[current_row - 1, column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced_cost < minimum_reduced_cost[column]:
                    minimum_reduced_cost[column] = reduced_cost
                    predecessor[column] = current_column
                candidate = minimum_reduced_cost[column]
                if candidate < delta or (candidate == delta and column < next_column):
                    delta = float(candidate)
                    next_column = column
            if not math.isfinite(delta):
                raise RuntimeError("balanced linear assignment could not find an augmenting path")
            for column in range(sample_count + 1):
                if used[column]:
                    row_potential[matched_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum_reduced_cost[column] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while True:
            previous_column = int(predecessor[current_column])
            matched_row[current_column] = matched_row[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    assigned_slot = np.empty(sample_count, dtype=np.int64)
    for column in range(1, sample_count + 1):
        assigned_slot[int(matched_row[column]) - 1] = column - 1
    return slot_groups[assigned_slot]


def _deterministic_groups(
    pc_matrix: np.ndarray,
    strains: Sequence[str],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    sample_count = len(strains)
    group_count = _group_count(sample_count)
    base_capacity, extra_capacity_count = divmod(sample_count, group_count)
    capacities = np.asarray(
        [
            base_capacity + (1 if group < extra_capacity_count else 0)
            for group in range(group_count)
        ],
        dtype=np.int64,
    )
    used_dimension_count = min(pc_matrix.shape[1], _MAX_CLUSTERING_PCS, sample_count - 1)
    raw = np.asarray(pc_matrix[:, :used_dimension_count], dtype=np.float64)
    standard_deviations = np.std(raw, axis=0)
    varying = standard_deviations > np.finfo(np.float64).eps * 100.0
    warnings: list[str] = []

    if not np.any(varying):
        ordered = sorted(range(sample_count), key=lambda index: (strains[index], index))
        labels = np.empty(sample_count, dtype=np.int64)
        for rank, index in enumerate(ordered):
            labels[index] = rank % group_count
        warnings.append(
            "PCA coordinates were constant for this trait; used a deterministic balanced "
            "strain-ID grouping fallback"
        )
        method = "deterministic_balanced_strain_id_fallback"
        dimension_count = 0
        iterations = 0
        unconstrained_sizes: list[int] | None = None
    else:
        values = raw[:, varying]
        values = (values - np.mean(values, axis=0)) / np.std(values, axis=0)
        stable_indices = sorted(
            range(sample_count),
            key=lambda index: (
                _stable_order_key(seed, strains[index]),
                strains[index],
            ),
        )
        stable_rank = {index: rank for rank, index in enumerate(stable_indices)}
        chosen = [stable_indices[0]]
        while len(chosen) < group_count:
            squared_distances = np.min(
                np.stack(
                    [np.sum((values - values[index]) ** 2, axis=1) for index in chosen],
                    axis=1,
                ),
                axis=1,
            )
            candidates = [index for index in stable_indices if index not in chosen]
            next_index = max(
                candidates,
                key=lambda index: (float(squared_distances[index]), -stable_rank[index]),
            )
            chosen.append(next_index)
        centers = values[np.asarray(chosen)].copy()
        labels = np.zeros(sample_count, dtype=np.int64)
        previous_labels: np.ndarray | None = None
        best_labels: np.ndarray | None = None
        best_objective = math.inf
        seen_assignments: set[tuple[int, ...]] = set()
        iterations = 0

        for iteration in range(1, _MAX_KMEANS_ITERATIONS + 1):
            iterations = iteration
            distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = _minimum_cost_capacity_assignment(distances, capacities)
            new_centers = np.stack(
                [np.mean(values[labels == group], axis=0) for group in range(group_count)]
            )
            objective = float(np.sum((values - new_centers[labels]) ** 2))
            assignment_key = tuple(int(label) for label in labels)
            if objective < best_objective:
                best_objective = objective
                best_labels = labels.copy()
            if previous_labels is not None and np.array_equal(labels, previous_labels):
                centers = new_centers
                break
            if assignment_key in seen_assignments:
                break
            seen_assignments.add(assignment_key)
            previous_labels = labels.copy()
            centers = new_centers
        if best_labels is None:
            raise RuntimeError("balanced PCA grouping produced no assignment")
        labels = best_labels
        final_centers = np.stack(
            [np.mean(values[labels == group], axis=0) for group in range(group_count)]
        )
        unconstrained = np.argmin(
            np.sum((values[:, None, :] - final_centers[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        unconstrained_sizes = [
            int(value)
            for value in np.bincount(unconstrained, minlength=group_count).tolist()
        ]
        method = "capacity_constrained_deterministic_kmeans_on_standardized_pcs"
        dimension_count = int(np.count_nonzero(varying))

    canonical_groups = sorted(
        range(group_count),
        key=lambda group: min(
            strains[index] for index in range(sample_count) if int(labels[index]) == group
        ),
    )
    remapping = {old: new for new, old in enumerate(canonical_groups)}
    labels = np.asarray([remapping[int(label)] for label in labels], dtype=np.int64)
    group_sizes = [
        int(value) for value in np.bincount(labels, minlength=group_count).tolist()
    ]
    balance_satisfied = max(group_sizes) - min(group_sizes) <= 1
    if not balance_satisfied:
        raise RuntimeError("balanced PCA grouping violated its hard size contract")
    warnings.append(
        "capacity-constrained PCA groups are balanced ancestry-informed evaluation "
        "partitions, not inferred biological populations"
    )
    if unconstrained_sizes is not None and max(unconstrained_sizes) - min(
        unconstrained_sizes
    ) > 1:
        warnings.append(
            "the hard balance contract replaced unconstrained nearest-centroid group "
            f"sizes {unconstrained_sizes} with balanced sizes {sorted(group_sizes)}"
        )
    metadata: dict[str, Any] = {
        "method": method,
        "seed": int(seed),
        "feature_source": "PCA eigenvectors only; phenotype values are not used",
        "uses_phenotype_values": False,
        "assignment_solver": "exact_minimum_cost_capacity_assignment",
        "iterations": int(iterations),
        "n_groups": int(group_count),
        "available_pc_dimensions": int(pc_matrix.shape[1]),
        "used_varying_pc_dimensions": dimension_count,
        "balance_contract": {
            "rule": "maximum_group_size_minus_minimum_group_size_lte_1",
            "satisfied": balance_satisfied,
            "minimum_group_size": min(group_sizes),
            "maximum_group_size": max(group_sizes),
            "group_sizes": {
                str(group): group_sizes[group] for group in range(group_count)
            },
        },
        "assignments": [
            {"strain": strain, "group": int(labels[index])}
            for index, strain in enumerate(strains)
        ],
        "groups": {
            str(group): [
                strain
                for index, strain in enumerate(strains)
                if int(labels[index]) == group
            ]
            for group in range(group_count)
        },
    }
    return labels, metadata, warnings


def _kernel_predict(
    kernel: np.ndarray,
    phenotype: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    alpha: float,
) -> np.ndarray:
    training_mean = float(np.mean(phenotype[train_indices]))
    centered = phenotype[train_indices] - training_mean
    training_kernel = kernel[np.ix_(train_indices, train_indices)]
    system = training_kernel + float(alpha) * np.eye(len(train_indices), dtype=np.float64)
    try:
        coefficients = np.linalg.solve(system, centered)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, centered, rcond=None)[0]
    predictions = training_mean + kernel[np.ix_(test_indices, train_indices)] @ coefficients
    return np.asarray(predictions, dtype=np.float64)


def _choose_alpha(
    kernel: np.ndarray,
    phenotype: np.ndarray,
    labels: np.ndarray,
    outer_test_group: int,
) -> tuple[float, dict[str, float]]:
    outer_training_groups = [
        int(group) for group in sorted(set(labels.tolist())) if group != outer_test_group
    ]
    squared_errors: dict[float, list[float]] = {alpha: [] for alpha in _ALPHA_GRID}
    for inner_test_group in outer_training_groups:
        inner_test = np.flatnonzero(labels == inner_test_group)
        inner_train = np.flatnonzero(
            (labels != outer_test_group) & (labels != inner_test_group)
        )
        if not len(inner_train) or not len(inner_test):
            raise ValueError("nested group cross-validation produced an empty split")
        for alpha in _ALPHA_GRID:
            prediction = _kernel_predict(
                kernel, phenotype, inner_train, inner_test, alpha
            )
            errors = (phenotype[inner_test] - prediction) ** 2
            squared_errors[alpha].extend(float(error) for error in errors)

    scores = {
        alpha: float(math.sqrt(math.fsum(errors) / len(errors)))
        for alpha, errors in squared_errors.items()
    }
    selected = min(_ALPHA_GRID, key=lambda alpha: (scores[alpha], -alpha))
    serialized_scores = {f"{alpha:.12g}": scores[alpha] for alpha in _ALPHA_GRID}
    return float(selected), serialized_scores


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _correlation(
    observed: np.ndarray,
    predicted: np.ndarray,
    metric_name: str,
) -> tuple[float, str | None]:
    observed_centered = observed - np.mean(observed)
    predicted_centered = predicted - np.mean(predicted)
    denominator = float(
        np.sqrt(np.sum(observed_centered**2) * np.sum(predicted_centered**2))
    )
    if denominator <= np.finfo(np.float64).eps:
        return 0.0, f"reported {metric_name}=0 because one input was constant"
    value = float(np.sum(observed_centered * predicted_centered) / denominator)
    return max(-1.0, min(1.0, value)), None


def _metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
) -> tuple[dict[str, float], list[str]]:
    rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
    pearson, pearson_warning = _correlation(observed, predicted, "Pearson correlation")
    spearman, spearman_warning = _correlation(
        _average_ranks(observed), _average_ranks(predicted), "Spearman correlation"
    )
    warnings = [
        f"{model_name}: {warning}"
        for warning in (pearson_warning, spearman_warning)
        if warning is not None
    ]
    return {"rmse": rmse, "pearson": pearson, "spearman": spearman}, warnings


def _evaluate_trait(
    condition: str,
    trait: str,
    phenotype_by_strain: dict[str, float],
    aligned_order: Sequence[str],
    kinship: np.ndarray,
    kinship_index: dict[str, int],
    pca_by_strain: dict[str, np.ndarray],
    min_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[str]]:
    analysis_id = f"{condition}::{trait}"
    strains = [strain for strain in aligned_order if strain in phenotype_by_strain]
    result: dict[str, Any] = {
        "analysis_id": analysis_id,
        "condition": condition,
        "trait": trait,
        "n_samples": len(strains),
        "status": "skipped",
    }
    if len(strains) < min_samples:
        reason = (
            f"only {len(strains)} aligned finite samples; min_samples={min_samples}"
        )
        result["reason"] = reason
        return result, [f"{analysis_id}: skipped ({reason})"]

    sample_indices = np.asarray([kinship_index[strain] for strain in strains], dtype=np.int64)
    trait_kernel = kinship[np.ix_(sample_indices, sample_indices)]
    phenotype = np.asarray(
        [phenotype_by_strain[strain] for strain in strains], dtype=np.float64
    )
    pc_matrix = np.stack([pca_by_strain[strain] for strain in strains])
    labels, grouping, warnings = _deterministic_groups(pc_matrix, strains, seed)

    baseline_predictions = np.empty(len(strains), dtype=np.float64)
    kinship_predictions = np.empty(len(strains), dtype=np.float64)
    folds: list[dict[str, Any]] = []
    for held_out_group in sorted(set(labels.tolist())):
        test_indices = np.flatnonzero(labels == held_out_group)
        train_indices = np.flatnonzero(labels != held_out_group)
        selected_alpha, inner_scores = _choose_alpha(
            trait_kernel, phenotype, labels, int(held_out_group)
        )
        training_mean = float(np.mean(phenotype[train_indices]))
        baseline_predictions[test_indices] = training_mean
        fold_kinship = _kernel_predict(
            trait_kernel, phenotype, train_indices, test_indices, selected_alpha
        )
        kinship_predictions[test_indices] = fold_kinship
        fold_baseline_rmse = float(
            np.sqrt(np.mean((phenotype[test_indices] - training_mean) ** 2))
        )
        fold_kinship_rmse = float(
            np.sqrt(np.mean((phenotype[test_indices] - fold_kinship) ** 2))
        )
        folds.append(
            {
                "held_out_group": int(held_out_group),
                "train_strains": [strains[index] for index in train_indices],
                "test_strains": [strains[index] for index in test_indices],
                "selected_alpha": selected_alpha,
                "inner_validation_rmse_by_alpha": inner_scores,
                "fold_rmse": {
                    "training_mean": fold_baseline_rmse,
                    "kinship_kernel_ridge_gblup": fold_kinship_rmse,
                },
            }
        )

    baseline_metrics, baseline_warnings = _metrics(
        phenotype, baseline_predictions, "training_mean"
    )
    kinship_metrics, kinship_warnings = _metrics(
        phenotype, kinship_predictions, "kinship_kernel_ridge_gblup"
    )
    warnings.extend(baseline_warnings)
    warnings.extend(kinship_warnings)
    predictions = [
        {
            "strain": strain,
            "group": int(labels[index]),
            "observed": float(phenotype[index]),
            "training_mean": float(baseline_predictions[index]),
            "kinship_kernel_ridge_gblup": float(kinship_predictions[index]),
        }
        for index, strain in enumerate(strains)
    ]
    result.update(
        {
            "status": "analyzed_exploratory_unvalidated",
            "strains": list(strains),
            "grouping": grouping,
            "cross_validation": {
                "protocol": (
                    "outer leave-one-PCA-group-out; alpha selected only by nested "
                    "leave-one-training-group-out RMSE"
                ),
                "folds": folds,
            },
            "metrics": {
                "training_mean": baseline_metrics,
                "kinship_kernel_ridge_gblup": kinship_metrics,
            },
            "predictions": predictions,
            "warnings": list(warnings),
        }
    )
    return result, [f"{analysis_id}: {warning}" for warning in warnings]


def run_caendr_baseline(
    phenotype_path: str | Path,
    kinship_path: str | Path,
    kinship_ids_path: str | Path,
    eigenvec_path: str | Path,
    *,
    min_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Run a read-only, exploratory CaeNDR phenotype-prediction audit.

    Hyperparameters are selected in nested group folds contained wholly within
    each outer training split.  Consequently, no outer test phenotype is used
    to select ``alpha``.  All returned values are JSON serializable with
    ``allow_nan=False``.
    """

    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 3:
        raise ValueError("min_samples must be an integer of at least 3")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    phenotype_file = _resolve_input(phenotype_path, "phenotype")
    kinship_file = _resolve_input(kinship_path, "kinship")
    kinship_ids_file = _resolve_input(kinship_ids_path, "kinship IDs")
    eigenvec_file = _resolve_input(eigenvec_path, "PCA eigenvector")

    phenotypes, phenotype_strains, warnings = _load_phenotypes(phenotype_file)
    kinship_ids, kinship, kinship_warnings = _load_kinship(
        kinship_file, kinship_ids_file
    )
    pca_ids, pca_matrix, pca_warnings = _load_eigenvectors(eigenvec_file)
    warnings.extend(kinship_warnings)
    warnings.extend(pca_warnings)

    kinship_set = set(kinship_ids)
    pca_set = set(pca_ids)
    aligned_set = phenotype_strains & kinship_set & pca_set
    aligned_order = [identifier for identifier in kinship_ids if identifier in aligned_set]
    pca_by_strain = {
        identifier: pca_matrix[index] for index, identifier in enumerate(pca_ids)
    }
    kinship_index = {identifier: index for index, identifier in enumerate(kinship_ids)}

    if not aligned_order:
        warnings.append("no strain IID is shared by phenotypes, kinship, and PCA inputs")
    dropped = {
        "phenotype_not_in_kinship": sorted(phenotype_strains - kinship_set),
        "phenotype_not_in_pca": sorted(phenotype_strains - pca_set),
        "kinship_not_in_phenotype": sorted(kinship_set - phenotype_strains),
        "pca_not_in_phenotype": sorted(pca_set - phenotype_strains),
        "kinship_not_in_pca": sorted(kinship_set - pca_set),
        "pca_not_in_kinship": sorted(pca_set - kinship_set),
    }
    for category, identifiers in dropped.items():
        if identifiers:
            warnings.append(f"alignment {category}: {len(identifiers)} strain(s)")

    trait_results: list[dict[str, Any]] = []
    for condition, trait in sorted(phenotypes):
        trait_result, trait_warnings = _evaluate_trait(
            condition,
            trait,
            phenotypes[(condition, trait)],
            aligned_order,
            kinship,
            kinship_index,
            pca_by_strain,
            min_samples,
            seed,
        )
        trait_results.append(trait_result)
        warnings.extend(trait_warnings)

    output: dict[str, Any] = {
        "analysis_type": "caendr_conventional_prediction_baseline",
        "status": "exploratory_unvalidated",
        "validated": False,
        "read_only": True,
        "scope": (
            "Exploratory, unvalidated phenotype-prediction audit only. This output does "
            "not identify QTLs, causal variants, biological mechanisms, or calibrated "
            "biological effects."
        ),
        "seed": int(seed),
        "min_samples": int(min_samples),
        "alpha_grid": list(_ALPHA_GRID),
        "inputs": {
            "phenotype": {
                "path": str(phenotype_file),
                "sha256": _sha256_file(phenotype_file),
            },
            "kinship": {
                "path": str(kinship_file),
                "sha256": _sha256_file(kinship_file),
            },
            "kinship_ids": {
                "path": str(kinship_ids_file),
                "sha256": _sha256_file(kinship_ids_file),
            },
            "eigenvectors": {
                "path": str(eigenvec_file),
                "sha256": _sha256_file(eigenvec_file),
            },
        },
        "alignment": {
            "phenotype_strain_count": len(phenotype_strains),
            "kinship_strain_count": len(kinship_ids),
            "pca_strain_count": len(pca_ids),
            "common_strain_count": len(aligned_order),
            "common_strains_in_kinship_order": list(aligned_order),
            "dropped": dropped,
        },
        "traits": trait_results,
        "warnings": warnings,
    }
    # Fail here rather than return a payload containing an accidental NaN/Infinity.
    json.dumps(output, allow_nan=False)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Print the baseline audit as JSON for ``python -m wormctx.poc.realdata``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phenotype_path")
    parser.add_argument("kinship_path")
    parser.add_argument("kinship_ids_path")
    parser.add_argument("eigenvec_path")
    parser.add_argument("--min-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = run_caendr_baseline(
        args.phenotype_path,
        args.kinship_path,
        args.kinship_ids_path,
        args.eigenvec_path,
        min_samples=args.min_samples,
        seed=args.seed,
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
