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
each top-K set must match exactly. Scores are compared by class ID within P2's
approved `atol` and `rtol` values. Ranking order remains exact by default. An
explicit positive `rank_atol` permits a reversed pair only when its score gap
stays within that budget in both results. A missing receipt, CPU fallback,
changed checkpoint, partial population, or excess
prediction difference fails the run. Time remains a report.

Before applying the gate, the runner records common-class score differences,
changed rankings, added or removed recommendations, and up to five examples.
These measurements remain in the result when a comparison fails. They do not
approve a numerical difference or choose the rank budget automatically.

The [calibration record](engine-weekly-b9-calibration.json) contains a direct
trial with the published 1.0.0 wheel and the retained P2 model. Both CUDA calls
completed with no fit. The one-GPU call took 175.66 seconds; the two-GPU call took
112.43 seconds. Those times include load, prediction, and canonicalization.
The complete command took 44 minutes 14 seconds, including input admission and
graph staging. Every comparison preserved the top-ten membership for all 200000
entities. Some near-equal scores changed order. The original exact-rank,
zero-tolerance trial remains `FAIL`.

The proposed policy is `atol=0.002`, `rtol=0`, and `rank_atol=0.0004`. It adds
25 percent operational headroom to the largest measured score difference and
reversed-pair gap, then rounds each upward to one significant digit. Applying
this policy to all retained outputs passes all three comparisons. Smaller score
and rank budgets, and a deliberately changed member ID, fail. This separate
comparison does not change the original native result. A complete checksum audit
after the trial confirms unchanged inputs and reference payload.

The policy still needs reference review. One pilot does not establish a universal
tolerance. It does not qualify a different candidate wheel or the full GitHub
workflow. These values are calibration proposals, not runner defaults. B9 keeps
exact top-K membership and does not whitelist observed entities.

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
legacy bundle checksum. Supply `--legacy-root`, `--legacy-sha256`, `--input-root`,
`--manifest-sha256`, and a fresh `--output` directory. The importer
audits the complete original eleven-case evidence, then copies only P2's model,
prediction coordinates, and source-case record. It preserves the checkpoint and
original fit ID. It imports data without loading a model or fitting one.
The output `reference.json` uses format `kumo-engine-weekly-p2-reference-v1`.
The main weekly runner rejects this narrower reference. Review the imported
reference and prediction tolerance before approving its checksum for B9.

The shared host lock covers input hashing, runtime setup, both calls, and the final input audit.
Before it imports the model runtime, the runner rejects active GPU work anywhere
on that host. The runner
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
