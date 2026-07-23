# DeepQuery

DeepQuery is a research codebase for representing biological evidence with explicit
experimental context and evaluating controlled inference procedures over that evidence.
It combines a provenance-bound context graph, deterministic scientific operators,
conventional statistical baselines, graph-conditioned prediction models, and guarded
evaluation harnesses.

The repository is organized by scientific experiment. Descriptive names are used throughout
the public tree; historical shorthand identifiers are not used as experiment names.

DeepQuery is not yet a trained general biological-reasoning model. The implemented neural
component is a bounded graph-conditioned routing and belief-state model. Authoritative
statistics, validity checks, and claim boundaries remain in deterministic typed code.

## Repository layout

- `src/wormctx/`: reusable graph, provenance, inference, and evaluation implementation.
- `experiments/`: portable contracts, public-safe summaries, and per-experiment records.
- `docs/algorithms/`: mathematical and procedural descriptions of implemented methods.
- `docs/architecture/`: component boundaries and data flow.
- `docs/implementation/`: local, remote, and container execution plans.
- `docs/reproducibility/`: data governance, receipts, and result-release rules.
- `schemas/`, `sql/`, `mappings/`: canonical data and interoperability contracts.
- `tests/`: synthetic and rights-safe verification.

The full experiment catalog is in [experiments/README.md](experiments/README.md).

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -q
```

Run the rights-safe graph demonstration:

```bash
deepquery demo --output build/demo
deepquery verify-build \
  --output build/demo \
  --observations data/examples/observations.jsonl \
  --query data/examples/query_context.json \
  --policy config/transport_policy.json
```

Validate the synthetic graph-conditioned inference contract:

```bash
deepquery-poc doctor \
  --config experiments/graph_conditioned_inference/synthetic_routing_and_design_selection/config/smoke.json
```

GPU execution requires the optional inference dependencies:

```bash
python -m pip install -e ".[poc]"
deepquery-poc run \
  --config experiments/graph_conditioned_inference/synthetic_routing_and_design_selection/config/full_gpu.json \
  --output-root runs/synthetic-gpu
```

## What has been demonstrated

- Context-bearing observations can be validated, normalized, reified as a graph, projected
  to exchange formats with explicit loss accounting, and bound to content-addressed receipts.
- Typed operators can reject invalid biological designs before a learned model is allowed to
  route or summarize them.
- Natural-variation baselines can be separated from calibration diagnostics, model
  sensitivity, state qualification, and held-out prediction.
- Developmental experiments can enforce gene-disjoint splits, sealed outcomes, semantic-stage
  alignment, and authentic-versus-rewired topology controls.
- A provider-neutral reasoning benchmark can freeze tasks, tools, authority boundaries,
  response bundles, and dimension-specific scores without giving a model access to gold data.

Several real-data lanes remain deliberately blocked when source identity, rights, upstream
receipts, or scientific calibration are incomplete. A blocked result is part of the evidence:
the software records why a claim cannot yet be made.

## Claim boundary

The code and public summaries support engineering feasibility, reproducibility analysis,
calibration diagnostics, and experiment qualification. They do not by themselves establish
causal mechanisms, validated biomarkers, prospective utility, or general biological
reasoning. Each experiment directory states its own narrower boundary.

## Public data boundary

This repository contains code, contracts, public source identifiers, aggregate summaries,
and synthetic or author-created fixtures. It intentionally excludes raw governed data,
private correspondence, machine-specific run receipts, credentials, host information,
queued-process state, private benchmark tasks and gold answers, and review-gated source rows.
See [docs/reproducibility/data-and-result-boundaries.md](docs/reproducibility/data-and-result-boundaries.md).

## Documentation

- [System architecture](docs/architecture/system.md)
- [Context graph and transport algorithms](docs/algorithms/context-graph-and-transport.md)
- [Graph-conditioned inference](docs/algorithms/graph-conditioned-inference.md)
- [Natural-variation calibration](docs/algorithms/natural-variation-calibration.md)
- [Developmental prediction and topology controls](docs/algorithms/developmental-prediction-and-topology.md)
- [Phenotype-compendium prediction](docs/algorithms/phenotype-compendium-prediction.md)
- [Graph-grounded reasoning evaluation](docs/algorithms/graph-grounded-reasoning-evaluation.md)
- [Implementation roadmap](docs/implementation/roadmap.md)
- [Portable execution](docs/implementation/portable-execution.md)

## License

Code is released under the [MIT License](LICENSE). Fixture and mapping artifacts retain their
own embedded licensing statements where applicable.

