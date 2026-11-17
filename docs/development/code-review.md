# Reviewing behavior and system design

Use this guide for human, agent and CodeRabbit reviews. It describes review
reasoning, not new contribution requirements or merge gates. Apply only the
lenses relevant to the changed behavior; a small fix does not automatically
need a design document, benchmark or additional abstraction.

Read the PR's problem, description, existing change map and linked design when
available, then trace the implementation and affected consumers. If context is
unavailable, state the limit and use the PR and repository evidence you can
inspect. Missing context alone is not a proven defect. The existing
[review workflow](review.md) owns PR metadata and change-map conventions. Use
[ML guidance](ml.md), [ML review examples](ml-review-patterns.md) and the
[UI guide](ui.md) as domain-specific companions; historical examples are not
universal requirements.

## Establish the problem and the right fix

Identify the failing scenario or requested behavior and inspect the current
path before evaluating the patch. Does the proposed cause explain the observed
behavior? Does the fix remove that cause, or mask a symptom, add a second policy
or impose a restriction on valid inputs? Look for an existing owner, helper or
contract before proposing another one.

Consider the smallest sufficient fix and its tradeoffs: simpler arithmetic
instead of a derived artifact, one shared validation contract instead of a
special bypass, or retaining existing behavior when the alleged defect is not
real. Do not generalize for hypothetical callers. No change needed is a valid
conclusion. Review substantive disagreements with the design, not just whether
its prose accurately describes the code.

## Follow the affected contracts

| Lens | Questions to resolve from source and evidence |
| --- | --- |
| Correctness | Which IDs, units, timestamps, shapes, ordering and numerical tolerances cross this boundary? What happens for empty, null, missing, skewed and boundary inputs? Are historical artifacts and unaffected paths still interpreted correctly? |
| Ownership and lifecycle | Who creates, owns, closes or invalidates each buffer, cache, file, worker and artifact? Does state remain valid across repeated calls, epochs, save/reload, failure and cancellation? Can a failed operation publish partial state or alter its source? |
| Scalability | What is allocated or scanned per row, batch, shard, thread, process or machine? Derive cost from the actual representation. Include private versus shared memory, peak memory, I/O, copies and skew; a small output batch does not necessarily bound preparation memory. |
| Performance | Does the changed branch actually run? Is dispatch overhead worthwhile on small inputs? Are timing, completed device work and cache state comparable? Separate kernel, loader and end-to-end effects, and throughput from latency or memory tradeoffs. |
| Usability | Can a user discover and run the intended installed API? Are defaults, optional dependencies, supported cases and errors coherent? Does the complete load/prepare/predict/save/reload/publish path preserve identity, configuration and results? |
| Code quality | Is policy owned once, with clear contracts and bounded scope? Does an abstraction remove actual duplication? Are new dependencies and configuration justified? Flag consequential complexity, unsafe ownership or inconsistent policies with concrete effects; leave formatting to automated checks. |

Trace across native/Python boundaries, SDK/service adapters, artifact producers
and consumers, and configuration where the change crosses them. Do not demand
changes to every layer merely because those layers exist.

## Assess the evidence, not just the test count

Prefer an independent oracle or invariant over a test that repeats the new
implementation. Check that the regression fails before the correction and that
tests force or observe the optimized/fallback path being claimed. Exercise
relevant serial/parallel and CPU/GPU paths, worker partitioning, restored state
and subsequent calls when those behaviors change. Missing unavailable hardware
results should be described as unverified, not silently treated as passing.

For performance claims, inspect the baseline/candidate revisions, workload and
scale, machine/resources, process/thread/batch settings, warm-up/cache state,
repetitions and reported variation. Compare equal work and semantic outputs;
report material regressions and latency/memory tradeoffs as well as wins. A
smoke run establishes execution, not production-scale throughput. A faster
kernel does not by itself establish faster training. Request a focused
measurement only when it can resolve a material claim or plausible regression;
do not require every PR to run a full benchmark matrix.

## Lessons from public reviews

- [Engine #93](https://github.com/NVIDIA/kumo-engine/pull/93) compares computed
  offset positions with the old decoder, including truncation and boundary
  cases. It explains why the simpler representation avoids a new artifact and
  separates process memory, cold reads, warm throughput and latency.
- [Engine #102](https://github.com/NVIDIA/kumo-engine/pull/102) distinguishes a
  measured parallel crossover from a load-balancing heuristic. The
  [coverage review](https://github.com/NVIDIA/kumo-engine/pull/102#discussion_r4394109371)
  resulted in forcing the parallel branch rather than relying on random input
  to cross its dispatch threshold. Its measurements acknowledge noise and
  warm-up effects.
- [Engine #327](https://github.com/NVIDIA/kumo-engine/pull/327) checks exact
  inputs, targets and loss masks across CPU/CUDA and reports loader speedups
  separately from training performance.
- [Engine #339](https://github.com/NVIDIA/kumo-engine/pull/339#discussion_r4570287474)
  checks actual tensors and index maps before changing a budget formula: the
  suspected dense allocation did not exist. [Engine #341](https://github.com/NVIDIA/kumo-engine/pull/341)
  fixes a regression through the shared empty-projection contract rather than
  adding another length-only admission rule.

## Make findings useful

Report a concrete trigger, affected path, consequence, supporting evidence and
suggested correction. Separate proven defects from questions and optional
improvements. Do not turn a plausible concern into a defect without checking
reachable callers, actual layouts and established contracts. Avoid pre-existing
unrelated issues, duplicate findings, praise and style noise.

[CodeRabbit configuration](../../.coderabbit.yaml) applies this guide through
its code-guideline context, general and subsystem path instructions, and the
existing advisory change-map design check. The design check assesses the whole
proposal; inline findings identify concrete code locations. Automated review
can miss defects or raise incorrect concerns; a passing check is not proof of
correctness, scalability or merge readiness. Draft auto-review and existing
check modes are unchanged.
