# Synthetic routing and design selection

Question: can one relation-aware graph encoder learn a shared routing contract across bounded
developmental, natural-variation, and cross-context episodes?

The benchmark generates deterministic group-disjoint episodes, applies five invalid-design
guards, and supervises task family, operator set, validity flags, conclusion, hypothesis,
next experiment, and resolution. The smoke and full GPU configurations are under `config/`.

Status: executable synthetic feasibility benchmark. It measures contract learning and
engineering behavior, not biological validity.

Implementation: `wormctx.poc.{generation,operators,tensorize,model,training,metrics}`.

