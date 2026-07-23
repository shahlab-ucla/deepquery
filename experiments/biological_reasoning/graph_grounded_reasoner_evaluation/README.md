# Graph-grounded reasoner evaluation

Question: does structured graph and tool grounding improve valid, evidence-faithful reasoning
over text-only and ordinary retrieval baselines?

The harness freezes six paired arms, anonymized private tasks, authority-separated gold and
scoring bundles, typed programs, deterministic tool replay, contamination controls, and
dimension-specific scores.

Status: provider-neutral build, replay, and score contracts are tested. No language-model
result is included, and private tasks and gold answers are never committed.

Implementation: `wormctx.poc.grounded_reasoner_benchmark`.

