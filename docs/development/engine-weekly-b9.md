# B9 retained-model prediction check

B9 loads the approved P2 commerce/25B model. It predicts the complete frozen
population with `num_workers=1`, then `num_workers=2`. Both calls use the public
`Trainer.predict` API. B9 adds no fit.

The two visible GPUs must be distinct devices of the same model. Each must meet
the P2 GPU class and memory requirement in the approved input manifest. The
model's training wheel can differ from the installed test wheel. The runner
checks the installed wheel bytes and records its version and dependencies.

Each scorer receives public Lightning callbacks. They reject training, require
finite CUDA score tensors, and count completed prediction batches. Before the
first batch, they compare the loaded model state with the retained checkpoint.
Python's spawn method imports the runner before a scorer starts. That import
installs the same callbacks in each child process. The runner does not replace
private `kumo_core` methods or disable product memory recovery.

A pass needs one complete GPU receipt for the first call and two receipts for
the second call. The second call must use distinct processes and GPU identifiers.
Both calls must retain the full prediction population. Each call must match the
stored P2 predictions. The two calls must also match each other. Entity IDs and
ranks must match, and scores must stay within P2's approved `atol` and `rtol` values. A
missing receipt, CPU fallback, changed checkpoint, partial population, or excess
prediction difference fails the run. Time remains a report.

Invoke `weekly_b9.py` directly with the selected wheel's isolated Python
interpreter. Supply `--input-root`, `--manifest-sha256`, `--reference-root`,
`--reference-sha256`, `--output`, `--scratch`, `--host-lock`, `--wheel`,
`--wheel-sha256`, and `--version`. Keep scratch outside the evidence, input, and reference
trees. Scratch must use the native graph's filesystem. Expose exactly two matching GPUs.

B9 uses a data-only `manifest.json` with format `kumo-engine-weekly-p2-input-v1`.
It contains only `P2-fixed`. Its case fields and file checksums follow
[the weekly input schema](engine-weekly-ci.md#inputs-and-reference-approval).
It does not require the other ten cases on the B9 host. The main weekly runner
still requires all eleven cases and rejects this P2-only format.

Import the retained model with `weekly_reference.py --retained-p2` and the approved
legacy bundle checksum. Supply its usual input and output arguments. The importer
audits the complete original eleven-case evidence, then copies only P2's model,
prediction coordinates, and source-case record. It preserves the checkpoint and
original fit ID. It imports data without loading a model or fitting one.
The output `reference.json` uses format `kumo-engine-weekly-p2-reference-v1`.
The main weekly runner rejects this narrower reference. Review the imported
reference and prediction tolerance before approving its checksum for B9.

The shared host lock covers input hashing, runtime setup, both calls, and the final input audit. The runner
keeps original models, raw outputs, canonical predictions, device receipts, and
failures. It removes only its fresh scratch directory.

The weekly workflow adds a separate `b9` job for manual candidate runs. It waits
for P1 to finish before reserving the shared host. It runs after a P1 failure
when the CPU checks, configuration, and wheel build succeeded. This preserves
independent evidence for each check. It downloads the same candidate wheel,
makes sure that its build receipt matches, and installs it in a fresh environment.
The job exposes exactly two GPUs and has a 120-minute timeout. Reference-preparation
mode skips it. There is no cron trigger.

Add `b9` to `KUMO_ENGINE_WEEKLY_CONFIG` after reference review. It requires
`runner` with the existing `group` and `labels`, absolute `input_root`,
`reference_root`, `host_lock`, and `work_root`, plus `manifest_sha256` and
`reference_sha256`. Use the same host lock as other workloads on that host.
The job does not create a VM or approve a reference. Missing bindings stop
configuration before prediction starts.

The job uploads results, device receipts, raw Parquet, and canonical predictions
as a separate `weekly-b9` artifact. It then removes its fresh environment and
work directory. It preserves immutable inputs and the original reference models.
A failed B9 job fails the workflow. P2 is a fixed performance workload without
reserved quality answers. B9 therefore compares predictions rather than a quality
score. Elapsed time remains a report until a separate timing policy is approved.

The focused CPU guard runs with:

```bash
python packages/kumo-engine/test/test_weekly_b9.py
```

The Engine test suite also runs `test_weekly_b9_observer.py`. It exercises the
spawn import with real Lightning classes on CPU. It makes sure that the observer
rejects training and CPU prediction. It does not qualify prediction on two GPUs.
