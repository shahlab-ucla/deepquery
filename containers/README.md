# Portable container execution

These definitions package the DeepQuery Python runtime, public experiment contracts, and
rights-safe tests. They do not embed raw input data, completed run directories, credentials,
machine identities, or scheduler state. Inputs are mounted read-only at `/data/input`; outputs
are mounted separately at `/data/output`.

## Base-image contract

The default base is a general Python 3.11 Linux image suitable for CPU validation. The
`BASE_IMAGE`/`base_image` build argument may instead name a locally approved accelerator
image. A replacement must provide:

- Linux user-management tools for `Containerfile` builds;
- Python 3.11 or newer and `pip`;
- a PyTorch version accepted by `pyproject.toml`; and
- for accelerated runs, a user-space CUDA runtime compatible with the host driver.

The repository does not pin a CUDA release. CUDA libraries belong to the selected base image,
while the host driver remains outside the container. Record the chosen base-image digest,
driver, framework version, and `pip freeze` output in each private run receipt.

For an authoritative build, create and verify a source bundle first, then build with the
extracted bundle directory as the context. This expands `SOURCE_REVISION` to the exact
archived commit instead of relying on mutable worktree state.

## Docker-compatible build and run

Build a CPU validation image from the repository root:

```bash
docker build --file containers/Containerfile --tag deepquery:local .
```

For an accelerator-capable image, first choose a compatible base and pass it without changing
the tracked definition:

```bash
export DEEPQUERY_GPU_BASE_IMAGE="your-approved-python-and-cuda-base@sha256:..."
docker build \
  --file containers/Containerfile \
  --build-arg BASE_IMAGE="${DEEPQUERY_GPU_BASE_IMAGE}" \
  --tag deepquery:gpu \
  .
```

Inspect the packaged runtime:

```bash
docker run --rm deepquery:local
```

Run with an accelerator and external data roots:

```bash
mkdir -p inputs runs
docker run --rm --gpus all \
  --mount type=bind,src="$(pwd)/inputs",dst=/data/input,readonly \
  --mount type=bind,src="$(pwd)/runs",dst=/data/output \
  deepquery:gpu \
  execute \
  --manifest /opt/deepquery/experiments/graph_conditioned_inference/synthetic_routing_and_design_selection/execution/rtx3090_gpu.json \
  --config-root /opt/deepquery/experiments/graph_conditioned_inference/synthetic_routing_and_design_selection/config \
  --input-root /data/input \
  --output-root /data/output
```

The image runs as an unprivileged `deepquery` user by default. Override `APP_UID` and
`APP_GID` at build time when a shared filesystem requires a particular numeric identity.

## Apptainer build and run

Build from the repository root. The default produces a CPU validation image:

```bash
apptainer build deepquery.sif containers/deepquery.def
```

Select an approved accelerator base with a build argument:

```bash
export DEEPQUERY_GPU_BASE_IMAGE="your-approved-python-and-cuda-base@sha256:..."
apptainer build \
  --build-arg base_image="${DEEPQUERY_GPU_BASE_IMAGE}" \
  deepquery-gpu.sif \
  containers/deepquery.def
```

Run the packaged doctor:

```bash
apptainer run deepquery.sif \
  doctor
```

Run with accelerator passthrough and explicit bind mounts:

```bash
mkdir -p inputs runs
apptainer run --nv --containall \
  --bind "$(pwd)/inputs:/data/input:ro" \
  --bind "$(pwd)/runs:/data/output:rw" \
  deepquery-gpu.sif \
  execute \
  --manifest /opt/deepquery/experiments/graph_conditioned_inference/synthetic_routing_and_design_selection/execution/rtx3090_gpu.json \
  --config-root /opt/deepquery/experiments/graph_conditioned_inference/synthetic_routing_and_design_selection/config \
  --input-root /data/input \
  --output-root /data/output
```

Apptainer normally executes with the invoking user's identity, so the definition does not
create a privileged runtime account. Site policy controls whether image builds require a
privileged builder, fakeroot, or a separately operated build service.

## Validation and result handling

Before an experiment run:

1. record the source revision and container digest;
2. run `doctor` and the rights-safe test suite;
3. verify mounted inputs against the experiment contract;
4. create a fresh output root; and
5. preserve the resolved config, dependency inventory, checksums, and terminal status with
   the result.

The image definition proves packaging mechanics only. Accelerator availability, driver
compatibility, numerical reproducibility across framework builds, and access to experiment
inputs must be verified on the execution system.
