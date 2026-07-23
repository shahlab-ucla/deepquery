# RTX-class synthetic feasibility evidence

The full checked configuration completed on one 24-GB-class Ampere accelerator and passed
every predeclared engineering gate. The result is evidence that the current graph encoder,
multi-head reasoning contract, checkpoint path, and portable executor can run together. It is
not evidence of biological validity.

Key locked-test results:

| Metric | graph model | constant baseline |
|---|---:|---:|
| family accuracy | 1.0000 | 0.3333 |
| operator micro-F1 | 1.0000 | 0.7586 |
| invalid-design rejection accuracy | 0.9167 | 0.5833 |
| invalid-flag micro-F1 | 0.9091 | 0.0000 |
| conclusion accuracy | 0.9167 | 0.4167 |
| experiment accuracy | 1.0000 | 0.6667 |
| resolution MAE | 0.1045 | 0.2722 |
| hypothesis accuracy | 0.3333 | 0.3333 |

The unchanged hypothesis accuracy is an informative negative control. The simulator weakly
supervises the true hypothesis index, so the run does not support a claim that causal
hypothesis selection has been learned.

An exact replay in a fresh output root produced byte-identical thresholds, predictions,
metrics, split files, data manifest, resolved configuration, and vocabulary. All 117 named
checkpoint tensors were bitwise identical. The safetensors container hash changed because its
metadata includes the timestamped run identifier; tensor values did not change.

The machine-neutral evidence record is
[`rtx3090_synthetic_feasibility.json`](rtx3090_synthetic_feasibility.json). It binds the source
revision, source archive, execution manifest, output-closure receipt, workload, metrics, and
acceptance decisions. Hostnames, account names, network addresses, authentication material,
absolute paths, and interpreter locations are excluded. The 223-MB model run directory is not
committed; its recursive closure digest is recorded so a retained or repeated run can be
checked against the same contract.

The second simulation/training seed is frozen in
[`full_gpu_second_seed.json`](../config/full_gpu_second_seed.json) and bound by
[`rtx3090_gpu_second_seed.json`](../execution/rtx3090_gpu_second_seed.json). After its paired
run, conventional real-data baselines remain required before biological interpretation.
