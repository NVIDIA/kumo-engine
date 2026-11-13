# Local evidence-wheel acceptance

This procedure builds an experimental CPU candidate from committed `supply-chain-xai` source.
It preserves the prototype wheel, environment, active run, models and retained evidence.
It does not qualify GPU, concurrent scorers, multi-seed batches or production statistical coverage.

## Build and install

From the Engine checkout, after reviewing and committing the change:

```bash
git branch --show-current
git status --short
git rev-parse HEAD
bash scripts/xai/build-test-wheel.sh
```

The script refuses a different branch, tracked changes or uncommitted package sources.
It creates a new `/tmp/engine-xai-wheel.*` directory, clones the committed source without hardlinks, and invokes the repository-supported builder:

```bash
export KUMO_REPO="$RUN/source"
unset BUNDLE_VERSION OUTPUT_DIR
bash "$RUN/source/packages/kumo-engine/scripts/build_wheel.sh" --output-dir "$RUN/wheels"
```

The enclosing script pins build dependencies from the existing prototype's read-only `build-requirements.txt`, installs cargo-zigbuild 0.24.1, and records the Python, Rust, Cargo, OS and glibc versions.
Override `XAI_PYTHON`, `XAI_RUST_BIN` or `XAI_PROTOTYPE` only deliberately; the resulting versions and copied inputs are retained.
Build and test caches, virtual environments and native outputs belong to the new run directory.
No system package installation or global configuration change is required.

Installation uses a fresh environment, the prototype's frozen dependency export excluding `kumo-engine`, CPU Torch, and the newly built wheel:

```bash
"$RUN/test-venv/bin/python" -m pip --isolated install --index-url https://download.pytorch.org/whl/cpu 'torch==2.13.0+cpu'
"$RUN/test-venv/bin/python" -m pip --isolated install --index-url https://pypi.org/simple -r "$RUN/runtime-requirements.txt" "$WHEEL"
"$RUN/test-venv/bin/python" -I -m pip --isolated check
```

The script records `source-revision.txt`, `wheel.sha256`, dependency inventories, input checksums, logs, resource measurements and acceptance reports.
Builds are source-identifiable, not claimed byte-for-byte reproducible.
Network access to package registries and local multiprocessing sockets are needed.

## Acceptance gate

Tests execute outside both checkouts with source overlays removed and Python isolated mode enabled.
The native harness checks installed module locations and module bytes against the wheel, plus wheel metadata and the committed revision.
Source-overlay test runs are development evidence only.

The gate runs these stages sequentially:

1. Unit and adversarial tests cover stable row identity, repeated entities/cutoffs, exact relation endpoints, missing/zero inputs, computed zero gradients, malformed bundles, interrupted publication, capability rejection, calibration and controlled removal.
2. The incremental collector writes 10,000 rows and checks that peak-memory growth from 1,000 to 10,000 stays below 32 MiB.
3. Small static and temporal native models are trained, saved and reloaded.
4. Original capture is compared with ordinary prediction, with one forward and one differentiation per explained row, signed finite differences in transformed-input coordinates, graph-only mode, separate replay, sink failure recovery and unsupported-head reporting.
5. Retained bundles are copied and reopened after scratch cleanup; controlled record removal reconstructs and checks the baseline before evaluating a permitted change.
6. Synthetic interval calibration and independent evaluation record empirical coverage and widths.
7. A four-row supply-chain smoke scores the copied retained model and historical snapshot at 6 July 2023, then exercises the existing prototype's artifact consumer.

The supply-chain smoke does not run the reserved evaluation or fit a calibrator to that population.
Statistical deployment acceptance still requires a new, declared calibration population and a separate forward evaluation population.
Chronological separation does not establish exchangeability or guarantee coverage under drift.

Engine's native temporal sampler is inclusive at its cutoff.
The adversarial fixture includes an equal-cutoff record and verifies that capture retains the actual graph while reporting a strict-before eligibility violation.
Changing to strict sampling is a future scoring-policy change, not an after-the-fact graph edit.

Allow approximately two hours for a cold native build and up to one hour for the bounded acceptance stages; actual time depends on download and compilation caches.
The script limits Cargo to two jobs and numerical libraries to two threads, uses CPU only, and records `/usr/bin/time -v` measurements.
Reserve at least 30 GiB scratch space; the actual peak is measured, not guaranteed by that estimate.
Build, unit, native and supply stages have explicit timeouts.

## Review ownership and extension gates

The implementing agent orchestrates integration and owns the public prediction API and final commit.
At most three pod agents work concurrently, with one native/build workload at a time.
Builders do not approve their own implementation.

| Pod | Builder ownership | Independent validation | Method work |
| --- | --- | --- | --- |
| Retention | Engine evidence package and bundle tests | Orchestrator plus scoring/interval pod adversarial review | Versioning, integrity, retention semantics |
| Original pass | Core prediction, collector, streaming hooks and wheel harness | Retention and interval pod reviews | Coordinate semantics and temporal audit |
| Methods | Calibration, controlled-removal adapters and tests | Retention pod plus orchestrator | Exchangeability, perturbation validity, empirical evaluation |

Ideators may discuss across pods; file changes stay with the assigned builder until handoff.
Straightforward integration covers durable retention, original graph capture, exact relation matching and explicit availability.
Controlled removals and interval calibration remain experimental methods with narrower assumptions and separate empirical acceptance.
Raw-source derivatives, additive attribution, GPU capture, multiple scorers/seeds, arbitrary graph edits and causal explanations are not implemented.

## Rollback

The candidate is installed only in its new test environment.
Rollback consists of ceasing use of that environment and keeping the retained prototype environment selected.
Do not replace the prototype wheel or delete its artifacts as part of this procedure.
Keep failed candidate logs and partial directories until reviewed; remove only the explicitly identified candidate run directory when retention is no longer needed.
For later application migration, first run the existing consumer against `export_legacy()` output in a separate directory and explicitly version any application-schema changes.

See [bundle API and semantics](evidence-bundles.md) for supported cases and unavailable explanations.
