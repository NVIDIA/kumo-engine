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
Both calls must retain the full prediction population. Entity IDs and ranks must
match, and scores must stay within P2's approved `atol` and `rtol` values. A
missing receipt, CPU fallback, changed checkpoint, partial population, or excess
prediction difference fails the run. Time remains a report.

Invoke `weekly_b9.py` directly with the selected wheel's isolated Python
interpreter. Supply `--input-root`, `--manifest-sha256`, `--reference-root`,
`--reference-sha256`, `--output`, `--scratch`, `--host-lock`, `--wheel`,
`--wheel-sha256`, and `--version`. Keep scratch outside the evidence, input, and reference
trees. Scratch must use the native graph's filesystem. Expose exactly two matching GPUs. The reference is the reviewed eleven-case
bundle described in [the weekly guide](engine-weekly-ci.md).

The shared host lock covers input hashing, runtime setup, both calls, and the final input audit. The runner
keeps original models, raw outputs, canonical predictions, device receipts, and
failures. It removes only its fresh scratch directory. This separate check has
no scheduled job. Two-device capacity and the spawn callback path need a real
GPU run before the change is ready for use.

The focused CPU guard runs with:

```bash
python packages/kumo-engine/test/test_weekly_b9.py
```

The Engine test suite also runs `test_weekly_b9_observer.py`. It exercises the
spawn import with real Lightning classes on CPU. It makes sure that the observer
rejects training and CPU prediction. It does not qualify prediction on two GPUs.
