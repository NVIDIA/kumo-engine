# Requirements and coverage owners

This is a source-to-test map, not a claim that the latest full run passes.
[The roadmap](../PLAN.md#remaining-acceptance-work) tracks remaining acceptance;
[required coverage](required-coverage.md) distinguishes execution, accepted
refusals and failures. Historical execution counts remain in Git history.

## PDF reconciliation

The original **kumo-pytorch priorities and testing plan.pdf** was reconciled
against all three pages on 2026-11-16 (SHA-256
`e6ee1a780028594b1ff1bbfa7f382a84b844df18cd65775c5bc26370a2953655`).
The later 1.2 criteria are mapped below. User decisions supersede document
cadence and reference-engine suggestions: PR/nightly coverage is identical,
Polars/GPU comparisons are removed, and T01 real-data execution is on demand.
Synthetic and MBD bindings are now included; the earlier AML-only deferral ended.

| Original PDF section | Execution owner | Acceptance boundary |
| --- | --- | --- |
| 1. Independent harness and datasets | [Inventory](../datasets/inputs.json), [downloader](../data.py), [binding catalog](../workloads/bindings.json) | Verify pinned sources/anchors; no Arena runtime dependency or packaged test assets |
| 2. Starter workloads | [Task/model guide](aml-prototype.md), [expanded binding features](binding-coverage.md) | Independently check each declared input, target, mask and loss; a column's presence is not task coverage |
| 3. Coverage sweeps | [Sweep map](aml-prototype.md#history-coverage-sweeps), existing package tests and conformance | Preserve independent expected answers, boundary traps and size axes; tiny fixtures do not prove out-of-core operation |
| 4. Benchmark stages and metrics | [Complete checklist](../PLAN.md#complete-metric-checklist), [benchmark runner](benchmarking.md), [fitting memory](fitting-memory.md) | All applicable metrics and experiment conditions required; incomplete measurements fail |
| 5. Correctness | Package tests, source oracles, [difficult behavior](difficult-behavior.md), lifecycle and model fixtures | Cutoff/no leakage, eligible membership, values, masks, fitted state and objectives; ties may permute within their allowed semantics |
| 6. Automation | [Full-suite plan](../README.md#organization-and-adding-bindings), [inventory/gate](required-coverage.md) | Same required PR/nightly plan, no silent skips; preserve failed/partial reports |

## 1.2 criteria

| Criterion | Implemented assertion/measurement owner | Remaining acceptance |
| --- | --- | --- |
| 1. Keyless tables | [Package keyless tests](../../../packages/experimental/kumo-pytorch/tests/test_keyless_tables.py) and difficult-behavior occurrence checks | Final-revision execution; do not confuse duplicate values with duplicate occurrences |
| 2. Subsecond clocks | [Clock precision](../../../packages/experimental/kumo-pytorch/tests/test_clock_precision.py), [time boundaries](../../../packages/experimental/kumo-pytorch/tests/test_time_boundaries.py) | Final-revision execution |
| 3. Bounded fitting | [Chunk boundaries](../../../packages/experimental/kumo-pytorch/tests/test_fitting_chunks.py), [25k/400k experiment](fitting-memory.md) | Real-data memory evidence; retain all repetitions and report growth without a threshold |
| 4. Multi-source histories | [Package histories](../../../packages/experimental/kumo-pytorch/tests/test_multisource_history.py), [source probes](../history_checks.py) | Verify supported behavior or approved preflight refusal; late errors remain failures |
| 5. Special-token IDs | [Special IDs](../../../packages/experimental/kumo-pytorch/tests/test_special_token_ids.py), [fresh-process lifecycle](../../../packages/experimental/kumo-pytorch/tests/test_inference_lifecycle.py) | Final-revision execution, including persistence and explicit IDs |
| 6. Real-data workloads | [Population runner](population.md), [binding feature map](binding-coverage.md) | Complete declared populations on the final revision; no cross-engine batch-parity requirement |
| 7. Lifecycle | [Reload/inference](../lifecycle.py), [finetuning](../training_lifecycle.py), [iterator/model/optimizer/RNG resume](../checkpoint_resume.py) | Passing real-data child-process evidence; no refitting; preserve downstream pretraining-to-involvement checks |
| 8. Training | [Training acceptance](training-acceptance.md), retained [controlled learning](../model_checks/test_learning.py) | Two deterministic 100-update CPU runs per active binding; finite loss/gradients, every trainable parameter updated, loss decreases |
| 9. Speed and memory | [Benchmarks](benchmarking.md), [fitting-memory checks](fitting-memory.md) | Complete required stage/count/rate/memory/cache evidence, including native length planning and reuse where claimed |
| 10. Automation | [Required coverage and activation](required-coverage.md), full-suite gate | Final-run acceptance; runner access, required-check activation and post-merge automatic schedule verification remain separate |

## Review rules

- Existing package tests own generic semantics; add binding/source checks only
  for a distinct boundary. The [golden audit](arena-golden-audit.md) records that mapping.
- Histories check cutoff, chronology, eligible membership, values, multiplicity
  and limits. Permit equal-time permutations and eligible clipping alternatives.
- Graph checks remap by identity and verify features/connectivity; allowable
  tied-neighbor choices still need semantic validation.
- Floating statistics use justified numerical tolerances. Counts, masks,
  structure and native identity/order guarantees remain exact.
- Independently verify source labels before comparing projected float32 targets.
- A clean refusal proves rejection only. Failed execution is not converted into
  an accepted refusal or a claim of feature/performance coverage.
