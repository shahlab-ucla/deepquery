# Portable execution contract

DeepQuery treats experiment execution as a sequence of verifiable transformations rather
than as an informal command copied between machines. The current implementation has three
parts:

1. a committed source revision is exported as a self-identifying archive;
2. a data-only execution manifest is validated and resolved against explicit filesystem and
   resource roots; and
3. a trusted Python module is run without a shell, after which declared outputs are closed by
   checksums and summarized in a write-once receipt.

A separate hardware-qualification procedure records a privacy-preserving runtime profile and
can run a bounded numerical workload. These mechanisms establish source identity, input
identity, invocation structure, resource admissibility, and output identity. They do not, by
themselves, establish the scientific validity of an experiment.

The implementation is in
[`src/wormctx/bundles.py`](../../src/wormctx/bundles.py),
[`src/wormctx/execution.py`](../../src/wormctx/execution.py), and
[`src/wormctx/hardware.py`](../../src/wormctx/hardware.py).

## 1. End-to-end protocol

The intended execution sequence is:

```text
committed revision
    -> source archive + archive receipt
    -> archive verification before extraction
    -> typed ExecutionManifest
    -> preflight against explicit roots and measured resources
    -> python -m <trusted module> <ordered argv>, with no shell
    -> declared file and directory verification
    -> exclusive, durable execution receipt
    -> post-transfer receipt replay against retrieved outputs
```

The source archive, execution manifest, inputs, and outputs have distinct hashes. This is
deliberate. A source digest answers “which code bytes were transported?”; a manifest digest
answers “which abstract invocation was authorized?”; input digests answer “which data bytes
were read?”; and output digests answer “which result bytes were produced?”

## 2. Source-bundle algorithm

`create_source_bundle` accepts a Git worktree, an output directory, and a revision. The
default revision is `HEAD`, and the worktree must be clean unless the caller explicitly
disables that guard.

The algorithm is:

1. Confirm that the source directory is a Git worktree.
2. Resolve the requested revision to a full, lowercase, 40-character commit digest. Revision
   strings that are empty, begin with an option prefix, or contain a null byte are rejected.
   Resolution uses Git's explicit end-of-options boundary.
3. Inspect tracked, modified, and untracked state. With the default policy, any dirty state
   stops bundle creation.
4. Refuse to reuse an archive or receipt name already present in the destination.
5. Run `git archive` on the resolved commit, not on the mutable working directory. Every
   member is placed below a revision-derived top-level prefix.
6. Inspect the completed archive before it is accepted.
7. Read `SOURCE_REVISION` from inside the archive and require it to equal the full resolved
   commit digest.
8. Compute the archive SHA-256 digest and byte count, then write a strict receipt containing
   the revision, filename, digest, byte count, archive prefix, and member count.

The embedded revision is normally produced by a tracked `SOURCE_REVISION` template together
with Git's `export-subst` attribute. This makes the archived file identify the commit even
though the template in the worktree remains stable.

### Archive safety checks

`inspect_source_archive` examines every tar member before extraction. It rejects:

- an empty archive;
- absolute paths or parent traversal;
- invalid or prefix-free member names;
- members outside the declared top-level prefix;
- symbolic links and hard links; and
- device or other special filesystem entries recognized by the tar reader.

Verification repeats these structural checks and also requires exact agreement with the
receipt's filename, byte count, archive digest, member count, prefix, and embedded revision.
Consequently, extraction is authorized only after both byte identity and path structure have
been checked.

The bundle receipt itself is canonical JSON: keys are sorted, separators are compact, text is
ASCII-encoded, and one terminal newline is added. This representation makes incidental JSON
formatting differences less likely, although the archive SHA-256—not a hash of the
receipt—is the authority for archive bytes.

## 3. Execution manifest

`ExecutionManifest` is a strict, frozen Pydantic model. Unknown fields are forbidden. It
contains:

- a descriptive, multi-token experiment identifier;
- a dotted Python module name and ordered argument vector;
- named configuration and input files, each with a relative path and SHA-256 digest;
- one or more named file or directory outputs;
- CPU, RAM, GPU-count, and per-GPU-memory requirements;
- names of required environment variables, never their values; and
- a bounded subprocess timeout.

The manifest cannot contain a shell command. Its entrypoint is restricted to a syntactically
valid dotted module, and execution always has the form:

```text
<resolved Python executable> -m <trusted module> <expanded argument vector>
```

Arguments may refer to declared bindings through exact placeholders such as
`{input:phenotypes}` or `{output:result-directory}`. Braces that do not form a known
placeholder are rejected. Literal arguments cannot contain absolute paths, parent traversal,
null bytes, carriage returns, or newlines, and the vector is capped at 512 items.

All declared paths are normalized relative POSIX paths. Absolute POSIX paths, absolute
Windows paths, backslashes, drive or scheme separators, empty path components, `.` and `..`
are invalid. Binding names must be descriptive lowercase identifiers. Names and paths must be
unique within each binding class, output paths cannot overlap as parent and child, and
experiment outputs cannot occupy the reserved `receipts/` namespace.

Resource bounds are validated when the manifest is constructed. A GPU request must state a
positive per-device memory requirement; a CPU-only request must state zero accelerator
memory. The schema also imposes finite operational upper bounds on cores, memory, device
count, and timeout.

## 4. Canonical manifest identity

The manifest digest is designed to be independent of declaration order where order has no
semantic meaning. Before serialization:

- configurations, inputs, and outputs are sorted by `(name, path)`; and
- required environment-variable names are sorted.

The argument vector is not sorted because argument order is semantic. The normalized model is
serialized as strict, compact JSON with sorted object keys, UTF-8 encoding, and non-finite
numbers forbidden. If this byte sequence is \(C(M)\), the manifest identity is:

\[
H_M = \operatorname{SHA256}(C(M)).
\]

The resulting digest determines the receipt filename. Reordering equivalent binding
declarations therefore produces the same manifest identity, while changing an argument,
input digest, output declaration, resource request, or timeout produces a different one.

## 5. Preflight algorithm

`preflight_execution` converts a portable manifest into an in-memory
`PreparedExecution`. Absolute filesystem paths appear only in this transient object; they do
not enter the public manifest or persisted receipt.

Preflight proceeds in fail-closed order:

1. Require the entrypoint module to appear in a caller-supplied, nonempty trust set.
2. Require every named environment variable to exist and contain a nonempty string. Only its
   name is retained for the receipt.
3. Resolve the configuration, input, and output roots. Each must be an existing directory
   given as an explicit absolute path.
4. Compare measured or caller-supplied resource availability with the manifest request.
5. Resolve the Python executable and require it to be an existing regular file.
6. Resolve every configuration and input path beneath its corresponding root. Every path
   component is checked for symbolic links, the fully resolved path must remain beneath the
   root, and the target must be a regular file.
7. Stream each configuration and input through SHA-256 and require exact agreement with its
   binding.
8. Resolve every declared output beneath the output root and require it not to exist.
9. Derive the receipt path from the experiment identifier, canonical manifest digest, and
   dry-run or execution status. A pre-existing receipt is an error.
10. Expand declared placeholders to the verified absolute paths and construct the ordered
    `python -m` argument vector.

Automatic resource discovery uses logical CPU count, available physical memory, and GPU
memory reported by the locally available accelerator-management utility. Callers may instead
supply `ResourceAvailability`, which is the intended integration point for a scheduler or
another independently verified resource inventory.

No experiment code is imported or executed during preflight.

## 6. Subprocess and receipt protocol

`execute_manifest` always calls preflight first. A dry run writes a `dry_run` receipt without
starting a subprocess. A real run creates only the parent directories required for declared
outputs and then invokes the prepared argument vector with:

- `shell=False`;
- the resolved output root as the working directory;
- the supplied environment mapping; and
- the manifest timeout.

Avoiding a shell removes shell parsing, interpolation, pipelines, and redirection from the
execution authority. The explicit module trust set is equally important: syntactic validation
of a module name does not make arbitrary Python code trustworthy.

After the subprocess exits, exit code zero is necessary but not sufficient for success. Every
declared output must exist with the declared kind and pass checksum closure. Only then is a
`succeeded` receipt constructed.

Receipts are opened with operating-system exclusive creation. If the path already exists, the
write fails rather than replacing prior evidence. After writing, the stream is flushed and
`fsync` is called. The receipt records:

- canonical manifest digest and experiment identifier;
- trusted entrypoint module;
- status, timezone-aware start and finish times, monotonic elapsed time, and exit code;
- requested resources;
- required environment-variable names;
- verified relative input paths and digests;
- declared outputs; and
- checksum-closed produced outputs for successful runs.

It intentionally omits absolute roots, host identity, argument values, environment values,
and subprocess output. A caller may therefore publish the receipt without publishing
machine-specific paths or sensitive configuration values, provided the relative binding
names and paths are themselves suitable for release.

`verify_execution_outputs` provides the corresponding retrieval-side check. It accepts a
successful receipt and an explicit output root, requires the produced closure to equal the
declared closure, recomputes every file or directory digest, and returns a path-neutral
verification summary that includes the receipt digest.

## 7. File and directory checksum closure

A file output is represented by:

\[
(p,\ H_f,\ n,\ 1),
\qquad
H_f = \operatorname{SHA256}(\text{file bytes}),
\]

where \(p\) is its declared relative path and \(n\) is its byte count.

A directory output is closed recursively. Regular files are sorted by path. For each file
\(i\), the implementation records:

```json
{"path":"relative/posix/path","sha256":"...","byte_size":123}
```

These records are encoded as compact ASCII JSON with sorted keys. The directory digest is the
SHA-256 of that encoded list:

\[
H_D =
\operatorname{SHA256}\left(
  \operatorname{CanonicalJSON}
  \left[
    (p_i, H_i, n_i)
  \right]_{i=1}^{k}
\right).
\]

The receipt also stores \(\sum_i n_i\) and \(k\). Symbolic links, special entries, and
directories containing no regular files are rejected. The digest therefore commits to every
regular file's relative name, content, and size. It intentionally does not commit to
timestamps, ownership, permissions, directory metadata, or empty directories.

## 8. Hardware qualification

The hardware module provides a privacy-preserving qualification layer separate from basic
execution preflight.

### Sanitized detection

`detect_hardware` uses PyTorch as its accelerator probe and reports only:

- Python version, implementation, and normalized operating-system family;
- framework availability, version, CUDA build version, and a stable status code; and
- accelerator backend, device index and count, model label, compute capability, and total
  memory.

It does not query machine names, user identities, network addresses, environment variables,
or filesystem paths. Control characters are removed from labels and lengths are bounded.
Framework-import failures are represented by stable codes rather than raw exception text,
which could contain local library paths.

### Admission decision

`evaluate_accelerator` converts a minimum memory requirement from GiB to an integer byte
threshold using a ceiling operation. An available CUDA device qualifies only when reported
total memory is at least this threshold. CPU fallback is never implicit: the caller must
enable it explicitly. The result distinguishes “accelerator requirement met,” “CPU fallback
permitted,” and “execution rejected.”

### Bounded numerical benchmark

`run_bounded_matmul_benchmark` creates two seeded float32 square matrices and repeatedly
multiplies them. Matrix dimension, warmup iterations, and timed iterations have hard upper
bounds that are checked before allocation. For CUDA execution, the procedure:

1. seeds both the framework and accelerator random generators;
2. performs the requested warmup;
3. synchronizes and resets peak allocation statistics;
4. synchronizes immediately before and after the timed region; and
5. records elapsed time, time per iteration, peak allocated memory, and one finite result
   sample.

The benchmark is observational: the seed fixes the generated workload, but elapsed time is
not treated as deterministic. `create_accelerator_receipt` combines detection, admission, and
the optional benchmark into strict JSON. If no backend is permitted, it records that the
benchmark was not run.

## 9. Core invariants

The implementation maintains the following invariants:

| Boundary | Invariant |
| --- | --- |
| Source | A default bundle contains only a committed revision from a clean worktree. |
| Archive | Every member is relative, prefix-contained, and free of links and special entries. |
| Revision | The archive's embedded full revision equals the receipt's resolved revision. |
| Manifest | Unknown fields, shell commands, opaque identifiers, unsafe paths, and inconsistent resource requests are rejected. |
| Inputs | Every configuration and input is a regular file beneath an explicit root and matches its declared SHA-256. |
| Authority | Only an explicitly trusted Python module can be invoked. |
| Outputs | Declared outputs must be absent before execution and present with the correct kind after a zero exit. |
| Receipt | A receipt is exclusive-created, never overwritten, and contains no absolute roots or environment values. |
| Success | Exit code zero without complete output closure is recorded as failure. |
| Accelerator | CPU fallback occurs only under explicit permission. |

## 10. Threat model

The contract is designed to reduce risks from accidental drift and partially untrusted
execution descriptions. It detects modified archives, tampered input files, path traversal,
symbolic-link escapes visible during validation, undeclared or untrusted entrypoints,
insufficient measured resources, missing outputs, and attempts to overwrite a prior receipt.
It also binds the exact output content observed at receipt time. Receipt minimization reduces
disclosure of machine-specific execution details.

The trusted Python module is outside this security boundary. Once authorized, it executes
with the invoking process's filesystem and environment privileges and can perform operations
not represented in the manifest. The contract is therefore an invocation and provenance
control, not a sandbox.

SHA-256 provides integrity only when the expected receipt or digest is obtained through an
independently trusted channel. The current receipts are not digitally signed. An attacker who
can replace both an artifact and its expected receipt is not detected by hashing alone.

The implementation also assumes that the operating system, Python interpreter, Git
executable, PyTorch runtime, and filesystem semantics are sufficiently trustworthy for the
measurement being made. It does not defend against a privileged process changing files
during hashing or after receipt creation.

## 11. Failure semantics

Failures are explicit and do not promote partial work:

- schema and path errors fail at model construction;
- trust, environment, resource, executable, path, digest, or output-existence errors raise
  `PreflightError` before experiment execution;
- an existing receipt raises `ReceiptExistsError`;
- inability to start the subprocess or a timeout produces a failed receipt with no exit code;
- a nonzero exit produces a failed receipt with that exit code;
- a zero exit followed by missing, mistyped, empty, linked, special, or otherwise unverifiable
  output produces a failed receipt with exit code zero and no produced-output claims; and
- only complete output closure produces a successful receipt.

`ExecutionFailed` carries a minimal `ExecutionOutcome` pointing to the authoritative failed
receipt. Dry-run and real-execution receipts have distinct filenames, so a completed dry run
does not masquerade as an executed experiment.

## 12. Current limitations

The present implementation has intentionally narrow boundaries:

- Source-bundle receipts and execution receipts are integrity records, not signatures or
  attestations.
- Bundle destination checks occur before archive creation, but archive and bundle-receipt
  publication are not yet a single atomic transaction.
- Preflight path checks and later subprocess access are separated in time. A hostile actor
  with concurrent write access could exploit this time-of-check/time-of-use interval.
- A trusted module may create undeclared files under the output root. The receipt binds only
  declared outputs and does not currently reject unrelated additions.
- A process interruption before final receipt creation can leave partial outputs without a
  terminal receipt. Because outputs are write-once, recovery requires an explicit,
  independently checked cleanup or a fresh output root.
- Directory closure does not encode permissions, ownership, timestamps, empty directories,
  or directory entries themselves.
- Files are hashed without snapshotting or locking. Concurrent mutation can invalidate the
  meaning of a digest even though ordinary post-run workflows avoid such mutation.
- Retrieval-side verification replays output closure but does not re-run the experiment or
  re-verify that its original inputs remain available.
- The subprocess inherits the supplied environment and standard streams. Sensitive values
  are omitted from the receipt but remain visible to the trusted module.
- Basic automatic GPU-resource discovery and the PyTorch hardware profile are separate
  mechanisms. Their results are not yet joined into one signed admission record.
- Automatic resource discovery measures available RAM at one instant and total accelerator
  memory, not guaranteed future capacity. It does not reserve resources.
- The bounded matrix benchmark is not a model-training benchmark. It does not measure data
  loading, mixed precision, sustained utilization, thermal behavior, storage performance, or
  end-to-end experiment throughput.
- A fixed random seed does not guarantee bitwise-identical floating-point results across
  framework versions, accelerator architectures, or mathematical-library implementations.

These limitations are recorded so that a successful receipt is interpreted correctly: it
proves that one declared invocation passed the implemented checks and produced a specific
set of bytes, not that all environmental or scientific sources of variation have been
eliminated.
