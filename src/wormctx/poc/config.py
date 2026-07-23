"""Validated, JSON-serializable POC configuration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import POC_SCHEMA_VERSION


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SimulationConfig(StrictConfig):
    seed: int
    train_groups_per_family: int = Field(ge=2)
    validation_groups_per_family: int = Field(ge=1)
    test_groups_per_family: int = Field(ge=1)
    episodes_per_group: int = Field(ge=1, le=64)


class ModelConfig(StrictConfig):
    hidden_dim: int = Field(ge=32, le=1024)
    transformer_layers: int = Field(ge=1, le=12)
    message_passing_layers: int = Field(ge=1, le=8)
    attention_heads: int = Field(ge=1, le=16)
    dropout: float = Field(ge=0.0, lt=0.8)
    max_nodes: int = Field(ge=8, le=4096)
    numeric_feature_dim: int = Field(ge=4, le=256)
    max_hypotheses: int = Field(ge=2, le=16)
    max_experiments: int = Field(ge=2, le=16)

    @model_validator(mode="after")
    def attention_divides_hidden_dimension(self) -> "ModelConfig":
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        return self


class TrainingConfig(StrictConfig):
    seed: int
    epochs: int = Field(ge=1, le=10000)
    batch_size: int = Field(ge=1, le=4096)
    learning_rate: float = Field(gt=0.0, le=1.0)
    weight_decay: float = Field(ge=0.0, le=1.0)
    gradient_clip: float = Field(gt=0.0, le=1000.0)
    num_workers: int = Field(ge=0, le=64)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    amp: bool = True
    deterministic: bool = True


class AcceptanceConfig(StrictConfig):
    require_cuda: bool = False
    minimum_invalid_flag_recall: float = Field(ge=0.0, le=1.0)
    minimum_invalid_flag_f1: float = Field(ge=0.0, le=1.0)
    minimum_invalid_design_accuracy: float = Field(ge=0.0, le=1.0)
    minimum_operator_micro_f1: float = Field(ge=0.0, le=1.0)
    maximum_peak_vram_gib: float = Field(gt=0.0, le=256.0)


class PocConfig(StrictConfig):
    schema_version: str
    run_name: str
    mode: Literal["confirmatory", "exploratory"]
    simulation: SimulationConfig
    model: ModelConfig
    training: TrainingConfig
    acceptance: AcceptanceConfig

    @model_validator(mode="after")
    def validate_identity(self) -> "PocConfig":
        if self.schema_version != POC_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {POC_SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        if not SAFE_NAME.fullmatch(self.run_name):
            raise ValueError("run_name must be a path-safe token")
        return self


def load_config(path: str | Path) -> PocConfig:
    """Load a strict JSON config without resolving mutable defaults."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return PocConfig.model_validate(payload)
