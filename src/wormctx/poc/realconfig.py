"""Frozen-input contract for exploratory real-data baseline audits."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .realdata import run_caendr_baseline
from .repro import sha256_file


REAL_BASELINE_SCHEMA_VERSION = "wormctx-real-baseline-1.0"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FrozenInput(_StrictModel):
    role: Literal["phenotypes", "kinship", "kinship_ids", "eigenvectors"]
    relative_path: str
    sha256: str

    @field_validator("relative_path")
    @classmethod
    def portable_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("relative_path must use portable POSIX separators")
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("relative_path must stay beneath the supplied data root")
        if any(part in {"", "."} for part in path.parts):
            raise ValueError("relative_path contains an empty or current-directory segment")
        return path.as_posix()

    @field_validator("sha256")
    @classmethod
    def lowercase_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value


class PhenotypeContract(_StrictModel):
    condition: str
    traits: list[str] = Field(min_length=1)
    strain_column: Literal["strain"]
    condition_column: Literal["condition"]
    trait_column: Literal["trait"]
    value_column: Literal["phenotype"]
    replicate_policy: Literal["finite_arithmetic_mean_within_condition_trait_strain"]

    @model_validator(mode="after")
    def unique_nonempty_traits(self) -> "PhenotypeContract":
        if not self.condition:
            raise ValueError("condition must not be blank")
        if any(not trait for trait in self.traits):
            raise ValueError("trait names must not be blank")
        if len(self.traits) != len(set(self.traits)):
            raise ValueError("trait names must be unique")
        return self


class FoldContract(_StrictModel):
    method: Literal["capacity_constrained_deterministic_kmeans_on_standardized_pcs"]
    seed: int
    expected_group_count: int = Field(ge=3, le=20)
    maximum_group_size_difference: Literal[1]
    uses_phenotype_values: Literal[False]
    min_samples: int = Field(ge=3)


class RealBaselineManifest(_StrictModel):
    schema_version: Literal["wormctx-real-baseline-1.0"]
    analysis_id: str
    analysis_kind: Literal["kinship_prediction_audit_not_qtl"]
    status: Literal["exploratory_unvalidated"]
    validated: Literal[False]
    rights_status: Literal["review_required", "cleared"]
    use_scope: Literal["read_only_audit"]
    phenotypes: FrozenInput
    kinship: FrozenInput
    kinship_ids: FrozenInput
    eigenvectors: FrozenInput
    phenotype_contract: PhenotypeContract
    folds: FoldContract
    known_limitations: list[str] = Field(min_length=1)
    biological_claims_permitted: Literal[False]

    @model_validator(mode="after")
    def identity_and_roles(self) -> "RealBaselineManifest":
        if not _SAFE_NAME.fullmatch(self.analysis_id):
            raise ValueError("analysis_id must be a path-safe token")
        expected_roles = {
            "phenotypes": self.phenotypes.role,
            "kinship": self.kinship.role,
            "kinship_ids": self.kinship_ids.role,
            "eigenvectors": self.eigenvectors.role,
        }
        mismatches = [name for name, role in expected_roles.items() if name != role]
        if mismatches:
            raise ValueError("frozen input roles do not match their manifest fields")
        if any(not item for item in self.known_limitations):
            raise ValueError("known limitations must not be blank")
        if len(self.known_limitations) != len(set(self.known_limitations)):
            raise ValueError("known limitations must be unique")
        return self


def load_real_baseline_manifest(path: str | Path) -> RealBaselineManifest:
    source = Path(path)
    return RealBaselineManifest.model_validate_json(source.read_text(encoding="utf-8"))


def _resolve_frozen_input(data_root: Path, item: FrozenInput) -> Path:
    candidate = (data_root / PurePosixPath(item.relative_path)).resolve()
    if not candidate.is_relative_to(data_root):
        raise ValueError(f"frozen input escapes the data root: {item.relative_path}")
    if not candidate.is_file():
        raise FileNotFoundError(f"frozen input is not a file: {candidate}")
    observed = sha256_file(candidate)
    if observed != item.sha256:
        raise ValueError(
            f"frozen input checksum mismatch for {item.role}: "
            f"expected {item.sha256}, observed {observed}"
        )
    return candidate


def run_frozen_caendr_baseline(
    manifest_path: str | Path,
    data_root: str | Path,
) -> dict:
    """Verify every predeclared input before running the exploratory baseline."""

    source = Path(manifest_path).resolve()
    manifest = load_real_baseline_manifest(source)
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root is not a directory: {root}")
    inputs = {
        name: _resolve_frozen_input(root, item)
        for name, item in {
            "phenotypes": manifest.phenotypes,
            "kinship": manifest.kinship,
            "kinship_ids": manifest.kinship_ids,
            "eigenvectors": manifest.eigenvectors,
        }.items()
    }
    result = run_caendr_baseline(
        inputs["phenotypes"],
        inputs["kinship"],
        inputs["kinship_ids"],
        inputs["eigenvectors"],
        min_samples=manifest.folds.min_samples,
        seed=manifest.folds.seed,
    )
    observed_traits = {
        (item["condition"], item["trait"])
        for item in result["traits"]
        if item["status"] == "analyzed_exploratory_unvalidated"
    }
    expected_traits = {
        (manifest.phenotype_contract.condition, trait)
        for trait in manifest.phenotype_contract.traits
    }
    if observed_traits != expected_traits:
        raise ValueError(
            "analyzed condition/trait set differs from the frozen phenotype contract: "
            f"expected={sorted(expected_traits)}, observed={sorted(observed_traits)}"
        )
    for item in result["traits"]:
        grouping = item.get("grouping")
        if not grouping:
            continue
        balance = grouping["balance_contract"]
        if grouping["method"] != manifest.folds.method:
            raise ValueError("observed grouping method differs from the frozen fold contract")
        if grouping["n_groups"] != manifest.folds.expected_group_count:
            raise ValueError("observed group count differs from the frozen fold contract")
        if grouping["uses_phenotype_values"] is not False:
            raise ValueError("grouping unexpectedly used phenotype values")
        if (
            balance["maximum_group_size"] - balance["minimum_group_size"]
            > manifest.folds.maximum_group_size_difference
        ):
            raise ValueError("observed groups violate the frozen balance contract")

    result["frozen_analysis"] = {
        "schema_version": REAL_BASELINE_SCHEMA_VERSION,
        "analysis_id": manifest.analysis_id,
        "analysis_kind": manifest.analysis_kind,
        "manifest_path": str(source),
        "manifest_sha256": sha256_file(source),
        "rights_status": manifest.rights_status,
        "use_scope": manifest.use_scope,
        "biological_claims_permitted": False,
        "inputs_verified_before_analysis": True,
        "phenotype_contract": manifest.phenotype_contract.model_dump(mode="json"),
        "fold_contract": manifest.folds.model_dump(mode="json"),
        "known_limitations": manifest.known_limitations,
    }
    # Assert serializability and prohibit accidental NaN/Infinity before returning.
    json.dumps(result, allow_nan=False)
    return result
