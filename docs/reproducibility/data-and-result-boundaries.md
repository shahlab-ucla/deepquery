# Data and result boundaries

## Committed

The public repository may contain:

- source and schema code;
- experiment contracts with portable or container-standard paths;
- public release identifiers, URLs, counts, and expected digests;
- synthetic or author-created fixtures;
- aggregate metrics that have passed release review;
- explicit blocked status and claim boundaries.

## Never committed

- credentials, private keys, tokens, or authentication configuration;
- hostnames, user accounts, personal paths, or machine inventory;
- queue identifiers, process identifiers, live command lines, or free-resource snapshots;
- raw governed data or derivatives that preserve restricted rows;
- large genotype, imaging, embedding, database, or checkpoint products;
- source correspondence and individual attestations;
- private reasoning tasks, original aliases, gold answers, scorer secrets, or HMAC keys;
- review-gated row-level source dispositions.

Ignored paths include `artifacts/`, `runs/`, content-addressed data stores, archive formats,
genotype formats, logs, environment files, and common credential file extensions.

## Public summaries

A public summary is recreated from a terminal private receipt. It contains only:

- schema version and descriptive experiment identity;
- scientific design and sample/trait/map counts;
- aggregate metrics and uncertainty;
- terminal, qualified, active, or blocked status;
- explicit interpretation and prohibited claims.

It excludes filesystem paths, command lines, process state, private hashes whose only purpose
is locating local artifacts, and row-level governed content.

## Rights-aware behavior

Code can describe how to validate a source without redistributing the source. A manifest with
`review_required` rights cannot enable external transfer or normalization by changing another
field. Rights decisions are authority inputs, not model predictions.

If real input is absent, tests use synthetic fixtures. Tests that require private receipts are
marked as such and skipped in the public checkout; the deterministic interface around the gate
remains tested.

## Secret and strategy scans

Before each release, scan tracked files and staged history for:

- private-key headers and common token formats;
- email addresses other than explicit placeholders;
- personal or remote absolute paths;
- host addresses and SSH identity names;
- private strategic-planning and internal execution-board language;
- forbidden data extensions and files larger than the repository limit.

The scan complements, but does not replace, review of staged diffs and Git history.
