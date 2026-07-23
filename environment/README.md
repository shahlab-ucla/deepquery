# Environment strategy

DeepQuery separates portable dependency intent from execution-specific locking:

- `pyproject.toml` is the authoritative package and optional-feature contract.
- `requirements-container.txt` selects the inference and qualification dependency groups.
- `constraints-container.txt` repeats supported cross-platform version bounds without
  selecting a CUDA build.
- each executed run records the resolved package inventory, base-image digest, Python,
  framework, accelerator runtime, and driver versions in its private receipt.

A single committed `pip freeze` is intentionally not treated as portable across CPU, CUDA,
operating-system, and architecture combinations. To reproduce an executed environment,
rebuild from the recorded image digest and verify its recorded package inventory. When exact
wheel retention is required, store a hashed wheelhouse or platform-specific lock as a
versioned release artifact after license and redistribution review; do not replace these
cross-platform bounds with a lock generated on an unrelated runtime.

The historical reconstruction environment in this directory is independent of the current
container strategy and remains unchanged.
