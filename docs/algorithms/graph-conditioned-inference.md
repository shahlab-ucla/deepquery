# Graph-conditioned inference

## Role of the learned model

The graph-conditioned model is a bounded controller over structured episodes. It proposes an
inference family, operator set, invalid-design flags, a conclusion class, a hypothesis,
and a next experiment. Deterministic operators still calculate statistics and enforce
scientific validity.

## Episode tensorization

Each episode is converted to:

- numeric node features \(X\in\mathbb{R}^{N\times F}\);
- node-type indices;
- directed source and destination edge indices;
- edge-relation indices;
- masks for nodes, edges, hypotheses, and experiment candidates;
- supervised targets for seven output tasks.

Episode sizes are bounded and padded within a batch. Candidate masks ensure padding can never
receive probability mass in hypothesis or experiment selection.

## Relation-aware message passing

Initial node state is

\[
h_i^{(0)}=\operatorname{GELU}(\operatorname{LN}(W_x x_i))+e_{\tau(i)},
\]

where \(e_{\tau(i)}\) is the node-type embedding.

For edge \(i\rightarrow j\) of relation \(r\),

\[
m_{ij}^{(\ell)}=f_m(h_i^{(\ell)}+e_r).
\]

Incoming messages are averaged at each destination and combined with the prior state:

\[
\bar m_j^{(\ell)} =
\frac{1}{\max(1,|\mathcal N^-_j|)}
\sum_{i\in\mathcal N^-_j}m_{ij}^{(\ell)},
\]

\[
h_j^{(\ell+1)}=
\operatorname{LN}\left(h_j^{(\ell)}
f_u([h_j^{(\ell)},\bar m_j^{(\ell)}])\right).
\]

The implementation uses one-hot destination aggregation rather than external scatter
extensions. This is computationally acceptable because episode graphs are deliberately small.

## Transformer pooling

A learned query token is prepended to the message-passed node sequence. A Transformer encoder
uses node padding masks, and the final query-token state is the episode representation.
Independent two-layer heads produce:

- family logits;
- multi-label operator logits;
- multi-label invalid-design logits;
- conclusion logits;
- masked hypothesis logits;
- masked experiment logits;
- a sigmoid resolution estimate.

## Objective

The training objective is a weighted sum:

\[
\mathcal L =
0.25L_{\text{family}}
+1.0L_{\text{operators}}
+1.25L_{\text{invalid}}
+0.75L_{\text{conclusion}}
+0.75L_{\text{hypothesis}}
+0.75L_{\text{experiment}}
+0.5L_{\text{resolution}}.
\]

Categorical targets use cross-entropy, operator and invalid flags use binary cross-entropy,
and resolution uses smooth L1 loss. The invalid-design term is weighted most heavily because
an unsafe route is more consequential than an imperfect family label.

## Split and evaluation rules

Generation is deterministic from a frozen seed. Splits are group-disjoint where an episode
family defines a biological group. Evaluation reports each task separately, including invalid
flag recall, operator selection, conclusion accuracy, ranking accuracy, and resolution error.
A single composite score is not sufficient to claim reasoning quality.

## Failure interpretation

Good routing performance shows that a shared graph encoder can learn the synthetic contract.
It does not show biological validity. Progression to real tasks requires qualified graphs,
locked deterministic tools, explicit task authority, contamination controls, and sealed
evaluation.

