# Native I/O evidence and reproduction

This package separates six formal five-pair groups (90 trials) from seven n=1 calibration/diagnostic groups (21 trials). The report series is dated 2026-11-09; formal cells finished on 2026-11-10 UTC. No graph files, model weights, Parquet, tensor dumps or native wheel binaries are included.

## Replay saved results

From the repository root, with the repository Python dependencies installed as described in the [setup guide](../../../../../docs/development/setup.md):

```bash
python packages/kumo-engine/benchmark/native_io_results.py --replay packages/kumo-engine/benchmark/results/native-io-2026-11-09/formal
python packages/kumo-engine/benchmark/native_io_results.py --replay packages/kumo-engine/benchmark/results/native-io-2026-11-09/calibration
```

Replay never trains or reads the original graph. It uses the committed canonical matrix validator, checks its four file hashes against the package, checks raw/controls hashes, reconstructs controls and executed argument vectors, recomputes duration/pressure eligibility, and validates predictions/work across regimes/cache variants per task. It compares regenerated summaries with the recorded summaries. It requires the same validator file contents, not checkout HEAD 75ca: a final documentation/helper commit can descend from the measured source.

`raw-trials.jsonl.gz` contains original result text and SHA256 per record; full predictions, phase/native counters, cgroup/process telemetry and original specs are preserved once. `controls.json` deduplicates source/build/data/model/runtime controls. `manifest.json` contains compact summaries, exact child argument vectors and metadata. Non-diagnostic calibration replay is strict, but each pilot remains `EXPLORATORY` with null intervals. Formal five-pair intervals remain exploratory inference. Historical proof files do not enter either timing estimate.

The [root archive audit](evidence/dg424-root-archive-audit.json) independently checked all 111 original texts/source hashes and unique paths. The [formal audit](evidence/dg424-independent-formal-audit.json) independently checked all 90 trials, actual backends/cache/rings/work/pressure, cross-regime predictions and medians/paired ratios without importing the canonical validator; it did not recompute bootstrap intervals. The [completion proof](evidence/dg424-phase7-formal-completion-proof.json) records clean measured source, terminal processes and cleanup. The [acceptance check](evidence/package-verification.json) records actual replay plus five rejected corrupt/ineligible cases.

## Reproduce the acceptance checks

The original acceptance source bytes are archived with their hashes in [the index](evidence-index.json). Extract into a fresh scratch location, then run from the repository root:

```bash
gzip -dc packages/kumo-engine/benchmark/results/native-io-2026-11-09/evidence/dg424-final-result-verification.py.gz > /tmp/native-io-result-verification.py
python /tmp/native-io-result-verification.py --package packages/kumo-engine/benchmark/results/native-io-2026-11-09/formal --scratch /tmp/native-io-negative-fresh --output /tmp/native-io-negative-results.json
```

The scratch directory must not exist. The five cases corrupt the archive, alter control content while updating its file hash, alter recorded argv, shorten a formal training interval below 30 s, and remove measured constrained-training pressure. They use the real archived trial data and validator; no synthetic canonical statistics implementation is substituted. Recorded original workspace argv is in [verification-command.json](reproduce/verification-command.json).

## Regenerate workloads and timing cells

The measured source is 75ca0e2fd6797827feda61d637b59b75e2d6f678 (upstream main 329e2ef5 plus the port/benchmark phases). Use a separate clean checkout of that source for new timing cells: the driver refuses a dirty source tree. The [release record](evidence/dg424-benchmark-release-75ca0e2f-provenance.json) pins the native/wheel/ingest hashes, optimized profile, Python/Torch versions and exact source files. Recorded toolchain override was installed Rust 1.96.1 rather than repository-pinned 1.99.1, with `RUSTFLAGS=-C target-feature=+aes,+neon`. A different build/platform is a new experiment and needs new provenance and native smoke checks.

The [native build command](reproduce/native-build-command.json) is reconstructed from the executed package-local backend command and release provenance, with the [captured optimized build log](evidence/dg424-benchmark-release-75ca0e2f-build-release.log.gz). It is explicitly not an original argv capture. The backend builds/stages optimized ingest and the release PyO3 wheel. Verify the selected toolchain actually resolves to the recorded 1.96.1; a later `stable` can resolve differently.

Install/build the matching native candidate using the repository development setup, with the matching Engine/Core/API/PQL source roots. The measured host was ARM CPU/local NVMe/ext4 with io_uring/O_DIRECT capability and user systemd memory/CPU scopes. The recorded environment has no usable CUDA. Set the source roots and single-threaded numerical libraries explicitly:

```bash
export PYTHONPATH="$PWD:$PWD/packages/kumo-engine:$PWD/packages/kumo-core/src:$PWD/packages/kumo-api:$PWD/packages/kumo-pql"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Preparation runs outside all measured cells. Each task uses scale 768, feature width 256, seed 0 and at most 2,048 held-out prediction entities. Generate into fresh paths:

```bash
for task in binary regression temporal_link; do
  python packages/kumo-engine/benchmark/native_io_tasks.py --prepare --task "$task" --data-root /local/native-io-source --prepared-dir "/local/native-io-prepared/$task" --scale 768 --feature-width 256 --seed 0 --prediction-entities 2048
done
```

Prepared manifests under [provenance](provenance/) retain the original generator config, source-file hashes, full pristine-file oracle, actual index bytes and preparation timings. Runtime-generated namespace/path metadata can vary when regenerating a graph; do not silently claim a new index is the original pinned snapshot. Pin the new prepared manifests, preserve identical inputs across the three backends, and treat it as a new variant when provenance differs.

The [six-group schedule](reproduce/dg424-phase7-formal-selected.tsv), [original controller command](reproduce/dg424-phase7-formal-selected-75ca-v1-command.sh.gz), and [original sequential controller source](reproduce/dg424-phase6-sequential-plan.sh.gz) preserve exactly executed settings and clean-source checks. Their original local paths are historical; adapt paths/build provenance and CPU affinity to the new host and keep that new invocation. For example, a new binary constrained group uses:

```bash
python packages/kumo-engine/benchmark/native_io_matrix.py run --prepared-root /local/native-io-prepared --output-dir /local/native-io-new/binary-constrained --runner "$PWD/packages/kumo-engine/benchmark/native_io_tasks.py" --python "$(command -v python)" --build-provenance /local/native-release-proof.json --expected-native-sha256 REBUILT_NATIVE_SHA256 --tasks binary --regimes constrained --repetitions 5 --steps 1280 --batch-size 256 --workers 0 --loader-mode standalone --threads 4 --torch-threads 1 --cpus 5-9,15-19 --resident-memory-gib 64 --constrained-memory-gib 4 --direct-cache-gib 2 --io-rings 8 --io-qd 256 --io-batch-mult 1 --model-seed 0 --order-seed 20261109 --min-training-seconds 30 --timeout 3600
```

Use the schedule's cache/step settings for the other groups: resident 64 GiB/cache 16 GiB; constrained 4 GiB/cache 2 GiB for scalars and cache 1 GiB for LP; 1,280 batches for scalars and 256 for LP. The matrix creates full private-inode copies, evicts owned files outside the scope, warms resident files inside the scope, runs fresh processes, and cleans successful trial/model copies after saving results. Resident labels kernel page-cache residency; it is not a same-reader hot native cache. The native cache starts empty in each fresh process.

## Package a new experiment

Package only complete homogeneous groups, with a unique variant name per source/control group; keep five-pair groups separate from n=1 calibration. For example:

```bash
python packages/kumo-engine/benchmark/native_io_results.py --variant binary-constrained=/local/native-io-new/binary-constrained --output-dir /local/native-io-new/package --repetitions 5 --order-seed 20261109
```

Repeat `--variant NAME=DIRECTORY` for the six final groups. The helper separately checks cross-regime/cache correctness without pooling timing groups. Different work schedules or legacy instrumentation cannot be mixed into a formal package. `--diagnostic` may archive incompatible historical results, marking invalid groups explicitly; it does not relax the validator or infer missing fields. [packaging-commands.json](packaging-commands.json) preserves the actual package invocations used here. Each group is measured with total scope memory including native/model/OS-cache allocations, workers 0, standalone loading, four Rayon threads and one Torch thread; this is not production automatic-worker qualification.

## Evidence integrity and historical scripts

[evidence-index.json](evidence-index.json) maps claim roles to durable files and both original/stored SHA256 values. Small records are plain JSON/log/patch; larger records and original scripts are deterministic gzip so their exact original source bytes are preserved. The archived historical scripts may contain original local paths; they are provenance/reproduction recipes, not executable imports required by replay. All substantive report claims have durable proof or raw trial data here. No temporary path is required to replay saved results.

The seed regression patch is archived as `evidence/dg424-production-seed-repro.patch.gz`. Its original `.patch.sha256` record applies to the decompressed bytes (SHA256 `f3b41b476f634001bf88c7c4972a23096d1796ae9b5766bb46c73e3af5ffdd5f`), not the gzip container. Extract to `/tmp/dg424-production-seed-repro.patch` to use the recorded checksum path. Source logs are likewise preserved byte-for-byte in gzip; padding and blank lines were not stripped.
