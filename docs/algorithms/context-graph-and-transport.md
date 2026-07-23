# Context graph and transport algorithms

## Canonical observation

The canonical unit is a contextual observation

\[
o=(s,p,t,c,e,m,\pi),
\]

where \(s\), \(p\), and \(t\) are the subject, predicate, and target; \(c\) is
experimental context; \(e\) is the evidence description; \(m\) is the measurement or
result structure; and \(\pi\) is provenance.

Context is modeled rather than embedded in free text. It can include organism, genetic
background, stage, anatomy, cell, environment, intervention, assay, protocol, and comparison
group. Missing fields require an explicit reason such as not reported, not applicable, or not
yet mapped.

## Manifest-bound acquisition

For each enabled artifact:

1. validate an immutable release URL and allowed host;
2. reject unresolved rights or mutable aliases;
3. stream bytes into a content-addressed blob;
4. verify expected byte count and SHA-256;
5. write an artifact receipt and release-level closure;
6. expose the snapshot to adapters only through verified paths.

Legacy absolute blob locations are not silently rewritten. Receipt migration verifies the
source snapshot and creates new portable locators in a fresh destination.

## Adapter normalization

An adapter performs source-specific parsing, but returns source-independent observations.
Normalization is two-phase:

1. `validate_raw` checks syntax, identifiers, ontology prefixes, referential integrity,
   qualifiers, expected row classes, and rights-dependent gates.
2. `normalize` yields observations only after validation succeeds.

Multi-artifact receipts bind each emitted observation to the exact source filename and digest.
Support artifacts that emit zero rows remain recorded so absence of use is auditable.

## Reified graph construction

A contextual observation becomes a small graph centered on an observation node. Subject and
target entities connect to that node; context, evidence, measurement, comparison, and source
entities attach as typed relations. Reification prevents two studies with the same
gene–phenotype pair but different conditions from collapsing into one indistinguishable edge.

Node identifiers are deterministic. Duplicate normalized observations are rejected or merged
only under an explicit identity rule. Graph construction validates the whole observation
collection before emitting files.

## Loss-aware exchange projection

The internal graph is richer than a pairwise KGX association. Projection therefore applies
an allowlist:

1. verify that subject and target categories are concrete Biolink categories;
2. map only reviewed predicates;
3. require exchange-level knowledge and agent metadata;
4. emit the pairwise association when the contract is satisfied;
5. write all nonprojected context and the reason for omission to a sidecar.

The sidecar is part of the output contract. A small KGX edge count is not treated as lost
evidence if the faithful reified graph and projection decisions remain available.

## Context coverage and transport score

For a query context \(q\), each observation receives per-dimension agreement values. Exact
agreement scores highest; compatible broader context receives partial credit; conflicts,
unknown values, and structurally missing fields are distinguished.

A policy-weighted score has the form

\[
T(o,q)=\frac{\sum_d w_d a_d(o,q)}{\sum_d w_d},
\]

where \(a_d\) is the agreement for dimension \(d\) and \(w_d\) is its declared importance.
The score is descriptive relevance, not an identified causal transport estimand. Formal
transport claims remain blocked unless a separate source/target causal model is supplied.

## Verification

Build verification recomputes input, query, policy, and output digests; checks expected file
sets and counts; and replays semantic invariants. A success marker without a matching receipt
closure is insufficient.

