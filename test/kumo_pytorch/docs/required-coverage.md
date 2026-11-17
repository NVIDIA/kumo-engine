# Required AML coverage

CI uses the reviewed node IDs in `../required-tests.json`: 339 CPU cases and
five CUDA reference cases. The CPU job installs Polars and exercises both the
generic adapter case and the AML-specific Polars cases. CUDA cases are explicitly
delegated to the dedicated GPU job, not accepted as skips on CPU.

`--required-coverage=cpu` or `--required-coverage=gpu` requires the corresponding
inventory. Missing/deleted cases, unexpected additions, collection-time skips,
missing imports, deselection of required cases, runtime skips and xfails fail
execution. Required test bodies must execute; setup failures remain failures.
Ordinary test assertions are unchanged. Local focused runs may omit the option.

The expanded training-acceptance module contributes 22 required CPU cases:
two fixture paths for all five combinations, plus harness and model-regression
checks. All 22 pass after removing unused final non-root branches from the
standalone GraphSAGE test model; see [training acceptance](training-acceptance.md).

When intentionally adding or renaming cases, collect from the repository root:

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini \
  --confcutdir=test/kumo_pytorch test/kumo_pytorch/model_checks --collect-only -q
```

Review the collected IDs against `required-tests.json` and update the inventory
in the same change as the tests. Keep IDs starting with `model_checks/`, sorted
within each owner. Never remove a required case merely to pass enforcement.
Collection alone does not establish execution acceptance.

## DiskGraph-only scope

Per the user's 2026-11-17 instruction, 87 InMemory variants were removed from
the standalone inventory. The cutoff-alignment and fresh-process benchmark-state
checks now use DiskGraph instead of InMemory, retaining their assertions.
Polars/GPU timing-reference cases are unchanged. Pre-existing Engine tests are
not removed; new package sweeps and golden boundary additions use DiskGraph.

Local execution before expanding training acceptance from two combinations to
five (326 CPU cases): **325 passed, one failed, zero
skipped**, with five GPU cases explicitly delegated. The failure remains
`test_raw_aml_multisource_history[filters-and-windows]`: `DG-ROW-FILTER` rejects
the required operation. It remains a normal failure. The converted checks passed.
Separately, all **105** focused package checks passed, including the 29 new
DiskGraph golden boundary checks. These runs use the existing compatible local
native extension; branch-native CI validation remains separate.

## Native AnchorTable migration

Benchmarking, AML training/evaluation, history checks, inference, finetuning and
checkpoint continuation now materialize Parquet AnchorTables and iterate the
owned population without an explicit request argument. Bounded correctness
probes retain expected records only for independent assertions and table
projection. The benchmark reuses the streaming population projection directly.
Three added required cases check native-path rejection, cache-reset rejection,
and complete per-anchor length reuse across a partial final batch.

This migration exposes a Fusion fresh-process checkpoint failure: aggregate
materialization changes the internal anchor fingerprint on reload, so
`TaskBatches.load_state_dict()` rejects the same original population/index as a
different task. The two Fusion checkpoint cases and the combined raw-AML case
remain ordinary failures. `restore-fingerprints.json` records both identities;
the package's check is not bypassed. The pre-existing `DG-ROW-FILTER` failure
also remains. These are local fixture findings, not full AML/GPU CI results.

## GitHub merge enforcement

Inspection on 2026-11-17 found that main's active repository ruleset requires
`kumo-core-gpu-ci`, but no PyTorch check. Existing review/code-owner requirements
must remain intact. The current account has maintain access, not admin access.
No repository protection settings have been changed.

PR #284 owns the proposed always-triggered `kumo-pytorch-gate`. It remains a
separate PR and must also account for this standalone suite: changes under
`test/kumo_pytorch/**` need to trigger the relevant checks and the final gate must
require the CPU AML coverage job. Do not require a path-filtered workflow by
itself: unrelated PRs may never produce its check. Once the always-triggered gate
exists and is verified on affected and unrelated PRs, an administrator must add
its exact context and GitHub Actions integration to the existing required checks,
preserving `kumo-core-gpu-ci` and all other protections. Until then, merge
enforcement is **deferred at the user's request**, not implemented by a
successful local test run.

Verification of actual scheduled execution is **deferred until after merge**,
per the user's instruction. Scheduled real-data/GPU execution is distinct from
PR fixture coverage and must not be claimed from local CPU tests.

Historical validation before the DiskGraph-only scope change: all 413 CPU
cases executed; 412 passed, one required multi-source filter case failed with
`DG-ROW-FILTER`, and none skipped. All six Polars cases passed. The five GPU
cases were explicitly delegated, not locally validated. Seven subprocess
fault-injection checks passed. These results use the existing compatible native
extension; CI must rebuild and validate the final native revision.
