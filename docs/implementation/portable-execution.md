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
4. run `deepquery source-bundle create --output-directory <bundle-directory>`;
5. hash the archive;
6. confirm that ignored data and artifact directories are absent.

The bundle command resolves a committed revision, refuses a dirty worktree, delegates file
selection to `git archive`, checks every archive member, verifies the substituted
`SOURCE_REVISION`, and writes a path-neutral digest receipt. Before extraction, verify both
objects:

```bash
deepquery source-bundle verify \
  --archive <bundle-directory>/deepquery-<revision>.tar.gz \
  --receipt <bundle-directory>/deepquery-<revision>.tar.gz.receipt.json
```

On the execution host:

1. verify the archive digest before extraction;
2. extract into a new deployment directory;
3. confirm the embedded source revision;
4. validate the contract without opening outcomes;
5. inspect CPU, memory, disk, accelerator, driver, and runtime availability;
6. verify source and upstream-artifact digests;
7. create a new run root that did not previously exist.

Accelerator qualification is independent of experiment execution:

```bash
deepquery hardware \
  --minimum-vram-gib 23 \
  --benchmark \
  --output <run-root>/accelerator-qualification.json
```

The hardware receipt includes only portable runtime and accelerator fields. It never reads or
records a hostname, account name, network address, environment value, or working directory.

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

The portable GPU preflight records, without credentials:

- accelerator model and count;
- framework and framework CUDA-build versions;
- compute capability and total device memory;
- the resource threshold and qualification decision;
- bounded benchmark timing and peak framework allocation, when requested.

The process must fail before training if the requested device is absent. Automatic fallback to
CPU is permitted only for configurations that explicitly allow it.

The checked-in 24-GB-class execution contract can be dry-run or executed with:

```bash
deepquery execute \
  --manifest experiments/graph_conditioned_inference/synthetic_routing_and_design_selection/execution/rtx3090_gpu.json \
  --config-root experiments/graph_conditioned_inference/synthetic_routing_and_design_selection/config \
  --input-root . \
  --output-root <fresh-run-root> \
  --dry-run
```

Remove `--dry-run` only after the source bundle, accelerator receipt, mounted inputs, and
fresh output root have qualified. The execution layer invokes an explicitly trusted Python
module with an argument vector and `shell=False`; it does not accept a command string.
Install DeepQuery into the selected interpreter before execution. Relative `PYTHONPATH`
entries are not an environment contract because the subprocess intentionally changes into
the isolated output root.

The first full reference execution completed this sequence on a 24-GB-class Ampere device.
The source archive, hardware qualification, graph fixture, 24-epoch model run, embedded
checksums, and 223-MB recursive output closure all verified. The machine-neutral metrics and
receipt digests are recorded in
`experiments/graph_conditioned_inference/synthetic_routing_and_design_selection/evidence/`.
An exact replay reproduced every checkpoint tensor, and a second seed committed before
execution also passed all gates. This establishes software and execution feasibility only; it
does not validate a biological prediction.

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

The generic execution receipt additionally hashes every declared file output. Directory
outputs receive a recursive, path-sorted closure over each regular member's relative path,
byte count, and SHA-256. Symlinks, special entries, empty directories, undeclared paths, and
pre-existing destinations fail closed.

After retrieval, replay that closure before interpreting results:

```bash
deepquery verify-execution \
  --receipt <retrieved-run>/receipts/<experiment>/<manifest>.execution.json \
  --output-root <retrieved-run>
```

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

## Container boundary

The OCI and Apptainer definitions in `containers/` contain only committed source, contracts,
and rights-safe fixtures. Data and results are external mounts. The CPU base validates
packaging; an accelerator base is supplied as a digest-pinned build argument compatible with
the execution system's driver. The portable constraints deliberately do not claim that one
Windows, CPU, or CUDA package resolution is universal.
