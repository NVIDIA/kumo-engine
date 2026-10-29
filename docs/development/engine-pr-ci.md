# Engine PR check

The `Kumo Engine PR` workflow builds and tests the candidate at the exact PR
head. It supplements the existing source, package, native, connector, and GPU
workflows. It does not change release jobs or enforce learning-quality limits.

NVIDIA GPU jobs use the existing trusted `copy-pr-bot` route. A push to
`pull-request/<number>` starts the workflow. The selection step checks that
the copied commit is still the PR head. A manual run also requires the PR
number and matching revision. Repository owners can require the stable
`kumo-engine-pr-ci` check after validating runner and source access.

## Selected work

Every code PR selects all eight learning types. Each task trains one normal
model with seed zero on small synthetic PR 172 data. Scores report only.
The check requires completed CUDA training, finite and complete predictions,
hidden-answer and chronological checks, and exact saved and restored model
state. Repeated, partial, and reloaded prediction differences remain diagnostic.
There is no quality reference or scrambled-answer fit in this PR check.

Changed paths also select installation, behavior, and source checks. Shared
runtime, dependency, sampling, and unknown changes select the broad behavior
roster. Packaging changes select installation. Test files select source checks.
Only documentation changes skip learning. A changed Engine quickstart or
example still runs the documented example. Renames inspect both paths.
Missing inventories, stale heads, failed selection, and skipped required jobs
fail the final check.

The behavior checks reuse small package fixtures. They cover recommendation
counts and unique IDs, time filters, invalid options, exports and custom tables,
continued training, and native sampling behavior. One retained model also
predicts in a fresh process after reconstructing its graph from copied tables.
The shared test also runs on the existing CPU owner suite. The PR worker
requires CUDA and verifies the child's requested device and completed work.
Public recommendation exclusions remain unsupported. The check does not claim
coverage of that unsupported option.

Installation and behavior can add training runs. The report records each
physical Core fit attempt, completed optimizer updates, CUDA batches, failures,
and completed fit counts. Eight describes the primary task roster only.
The source job also reads the existing source workflow at the same PR revision
and rejects failed jobs even when that workflow allows their failure.

## Artifact and input identity

The CPU job requires a clean checkout, builds the full bundled wheel, and
records the full source revision, tree, version, and archive SHA-256. The GPU
job installs that archive in a fresh Python 3.12 environment. It verifies every
installed runtime member against the archive and runs `pip check`. Dependency
versions and the actual GPU model are retained with the results.

The small inputs contain 556,127 adtech rows, 1,442,968 delivery rows, and 74,416
commerce rows. They were generated once from the recorded PR172 revision.
The committed archive is 13,834,517 bytes. It preserves the exact approved
table bytes, manifests, and validation stamps. CI checks archive and file
hashes, inventory, and actual Parquet row counts before training. It uses
local queries and graph definitions. It reads no private repository and
requires no fixture-access token. No private generator code is included.

The graph adapter uses public Engine APIs and fixed table, column, type, key,
time, delimiter, and edge metadata. Its full graph configurations and all
eight resolved query types match the retained generator's definitions.

The GPU runner defaults to the repository's existing
`linux-amd64-gpu-t4-latest-1` label. `KUMO_ENGINE_PR_GPU_RUNNER` can select an
admitted compatible runner. The workflow provisions no hosts and does not use
the retained benchmark VMs. The existing RTX measurements do not prove T4
runtime. Each run records its actual device and rejects CPU fallback.

## Timing and evidence

The target is 20 minutes through the last required result. It includes build,
queues, installation, generation, selected checks, reporting, and uploads.
Per-job timeouts are execution bounds. They do not prove the complete target.
The recorded manual trial and worker timings do not replace a GitHub run.

The Rust cache includes workspace crates and keys them to the native sources
and bundle builder. A miss still builds from source. Every run retains the
exact-source and glibc checks. Cache improvements need runtime measurement.

The [recorded attempt at `4a508a93`](https://github.com/NVIDIA/kumo-engine/actions/runs/40924279550)
took 25 minutes 56 seconds and failed before any primary fit. Its exact wheel
build passed in 13 minutes 15 seconds. The GPU job lacked the generator token
required by that earlier implementation. Committed fixtures remove that
dependency. The owner source suite also failed its credential selection and
a shared test's unconditional CUDA check. Both paths have focused fixes.
This is a failed measurement, not a passing runtime estimate. The current
control jobs use Ubuntu 22. The existing owner workflow retains its own runners
and waiting time.

After a run finishes, download its run metadata and the complete job inventory,
including all pages for its recorded attempt. Then audit that run:

```sh
python3 .github/scripts/engine-pr-timing.py \
  --run run.json --jobs jobs.json --revision FULL_COMMIT_SHA \
  --output timing.json
```

The audit includes queue time and all jobs through their final completion.
It rejects incomplete attempts, stale commits, and failed jobs hidden by a
green workflow. Audit the other required workflows too before claiming the
complete PR target. The existing source workflow previously took 1,302 seconds.
It remains active and can exceed the target independently of the new worker.

The workflow keeps selection, build identity, installation, task results,
physical-fit receipts, test results, generator logs, and failed-run evidence.
No nightly, weekly, release, or reference-calibration job is added here.
