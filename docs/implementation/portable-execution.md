# Portable local-to-remote execution

## Objective

The same committed source and experiment contract should execute locally or on a remote Linux
workstation without embedding a hostname, account name, identity-file location, or data path
in Git.

## Execution inputs

The public source bundle contains:

- one clean Git revision;
- experiment contract;
- source code and schemas;
- dependency lock or container recipe;
- rights-safe test fixtures.

The operator supplies outside Git:

- remote destination and authentication;
- governed or large source data;
- upstream private receipts;
- licensed external executables;
- output and scratch roots.

Credentials are never copied into the source archive or written to a run manifest.

## Preflight

Before transfer:

1. require a clean worktree;
2. record the commit and contract SHA-256;
3. run compilation and the rights-safe tests;
4. build an archive with `git archive`, not a filesystem copy;
5. hash the archive;
6. confirm that ignored data and artifact directories are absent.

On the execution host:

1. verify the archive digest before extraction;
2. extract into a new deployment directory;
3. confirm the embedded source revision;
4. validate the contract without opening outcomes;
5. inspect CPU, memory, disk, accelerator, driver, and runtime availability;
6. verify source and upstream-artifact digests;
7. create a new run root that did not previously exist.

## Direct remote backend

A direct backend may use SSH and SCP, but connection values must be parameters or environment
variables supplied by the operator. Strict host-key checking and noninteractive batch mode are
required. The private identity file remains outside the repository.

The remote command should receive only quoted positional arguments or a generated run manifest.
Avoid interpolating unvalidated paths into shell fragments. Absolute remote paths must pass an
allowlist and must not be the filesystem root or a home-directory wildcard.

## Scheduler backend

A scheduler submission should request one experiment task per immutable run root. The job
script receives:

- container or environment identifier;
- contract path;
- artifact roots;
- output root;
- resource limits;
- optional checkpoint to resume.

The scheduler job identifier is operational metadata and stays in the private run receipt, not
the public experiment contract.

## GPU execution

The GPU preflight records, without credentials:

- accelerator model and count;
- driver and CUDA runtime;
- framework and CUDA versions;
- free and total device memory;
- deterministic-algorithm settings;
- visible-device mask.

The process must fail before training if the requested device is absent. Automatic fallback to
CPU is permitted only for configurations that explicitly allow it.

## Output contract

A run root is append-only except for designated scratch and atomic staging files. Required
outputs are:

```text
resolved_config.json
run_manifest.json
environment.json
logs/
checkpoints/
predictions-or-statistics/
metrics/
SHA256SUMS.txt
SUCCESS or FAILURE
```

`SUCCESS` is written last. A result bundle is retrieved only after its checksum file verifies
on the execution host. Retrieval verifies the archive hash before extraction and rejects
absolute paths, parent traversal, links, and special filesystem entries.

## Restart behavior

Restart is explicit:

- immutable inputs and completed checkpoints are verified;
- the new run records the parent run and checkpoint digest;
- incomplete scratch may be removed only after proving it is under the designated run root;
- completed outputs are never overwritten;
- a restarted run receives a new run identifier and terminal receipt.

## Equivalence criterion

Local and remote runs are considered semantically equivalent when resolved contracts, input
digests, split identities, deterministic seeds, model/statistical settings, and aggregate
outputs agree within the prespecified numeric tolerance. Byte equality is required for
deterministic artifacts; floating outputs use an explicit tolerance.

