# Synthetic routing and design selection

Question: can one relation-aware graph encoder learn a shared routing contract across bounded
developmental, natural-variation, and cross-context episodes?

The benchmark generates deterministic group-disjoint episodes, applies five invalid-design
guards, and supervises task family, operator set, validity flags, conclusion, hypothesis,
next experiment, and resolution. The smoke and full GPU configurations are under `config/`.
Provider-neutral execution manifests under `execution/` bind those configs by SHA-256,
request either CPU smoke resources or one 24-GB-class accelerator, and checksum the complete
run directory.

Validate the CPU-capable manifest without starting the experiment:

```bash
mkdir -p build/portable-smoke
deepquery execute \
  --manifest experiments/graph_conditioned_inference/synthetic_routing_and_design_selection/execution/portable_smoke.json \
  --config-root experiments/graph_conditioned_inference/synthetic_routing_and_design_selection/config \
  --input-root . \
  --output-root build/portable-smoke \
  --dry-run
```

Status: executable synthetic feasibility benchmark. It measures contract learning and
engineering behavior, not biological validity.

One full checked GPU run has completed and passed every predeclared engineering gate. Its
path-free evidence record and interpretation are under [`evidence/`](evidence/README.md).
The next model-level controls are a second seed and deterministic checkpoint replay.

Implementation: `wormctx.poc.{generation,operators,tensorize,model,training,metrics}`.
