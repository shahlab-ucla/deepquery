# Developmental prediction and topology controls

## Scientific unit and leakage boundary

The indivisible unit is an embryo nested within a perturbation target. Rows from one embryo
cannot cross splits. The primary generalization question is gene-disjoint: all embryos from a
perturbation gene belong to train, validation, or sealed test, never more than one.

Split assignment is frozen before outcome values are opened. Stage, lineage, and feature
eligibility are derived from outcome-blind metadata.

## Semantic stage preparation

Source tables are scanned structurally to establish sheets, headers, dimensions, embryo
identifiers, and permitted columns. Semantic cell names are mapped to a lineage manifest.
The adapter freezes:

- cells born by the input stage;
- the exact 26-cell input frontier;
- cells born by the endpoint;
- the exact 200-cell endpoint frontier;
- a rooted lineage through the endpoint;
- hashes of aligned stage and lineage arrays.

Secondary outcome-call sheets remain excluded when their semantics or embryo headers are not
qualified. The continuous endpoint and missingness rules are separate frozen decisions.

## Baseline ladder

Models are evaluated in increasing structural complexity:

1. training-set constant predictor;
2. target-blind flat ridge regression;
3. semantic-cell ridge regression;
4. topology-aware model.

Each model receives the same allowed early prefix and split. Hyperparameters are selected on
validation genes only. Metrics are computed per embryo and summarized per held-out gene so
large perturbation groups do not dominate.

For ridge regression,

\[
\hat B_\lambda=(X^TX+\lambda I)^{-1}X^TY
\]

or its dual form is used according to matrix shape. Feature standardization parameters are
fit on the training partition and replayed unchanged.

## Topology-aware representation

The lineage is represented by a parent index, depth, semantic identity, and stage identity.
Validation requires one root, exactly one parent for every nonroot active node, acyclicity,
and depth increasing by one along each parent edge.

The topology model receives early measurements and lineage coordinates. Its effect is assessed
against controlled alternatives:

- authentic lineage edges;
- edge-free model with comparable parameter capacity;
- depth/stage-matched model without parent identity;
- deterministic degree/depth-preserving rewired trees;
- shuffled semantic identity where applicable.

Rewiring is generated without outcome access and accompanied by an invariance receipt. A
topology benefit requires improvement over capacity-matched and rewired controls, not merely
over a constant baseline.

## Learning curves and seeds

Training-gene fractions and gene sets are frozen for each learning-curve point. Multiple seeds
separate optimization variation from split variation. Results report per-seed predictions,
aggregate uncertainty, and failures; unsuccessful runs are not silently dropped.

## Sealed evaluation

The sealed test is mounted only after the model family, topology controls, endpoint,
missingness rule, and thresholds are frozen. A counter records test openings. Repeated opening
or post-test model selection invalidates confirmatory interpretation.

## Interpretation boundary

Prediction across unseen perturbation genes is stronger than embryo-random prediction but
still does not establish developmental mechanism. Authentic-topology improvement would show
that lineage structure adds predictive information under the tested representation; it would
not prove that the learned messages correspond to causal signaling.

