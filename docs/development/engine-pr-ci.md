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

## Agent-local feedback

Vedaant Jain requested a local feedback command on October 29, 2026. Its
selected tests must finish in strictly less than five minutes. This requirement
is separate from the complete GPU CI timeline.

Use Python 3.12 on macOS or Linux, with Git and Bash available. Prepare a local
environment once from the repository root. This setup is outside the timed run:

```sh
python3.12 -m venv .venv-agent-local
.venv-agent-local/bin/python -m pip install -r test/scripts/agent-local-requirements.txt
```

Run the marked tests with this one command:

```sh
.venv-agent-local/bin/python -I scripts/agent_local_tests.py
```

The entry point uses the existing pytest tests marked `agent_local`. It supplies
an isolated configuration, disables plugin autoload and inherited pytest
options, stops parent conftest discovery at `test/scripts`, and names the test
files explicitly. Running a root-level `pytest -m agent_local` is insufficient:
root collection can import service fixtures before marker selection.

The selection covers changed-file and required-job decisions, complete source
and timing evidence, bundled fixture integrity, package identity, finite and
complete outputs, and score calculations. Separate supervisor regressions in full CI exercise failure, skip, incomplete
selection, reporting time and collection-time timeout handling. Keeping those
subprocess regressions outside the marked subset avoids nested supervisors.
Remote API calls
in owner and workflow unit tests use controlled responses. They do not establish
network access. The Torch checkpoint test is not selected locally. Actual GPU
training and checkpoint checks remain in full CI.

A parent process times the complete pytest command, including Python startup,
collection, fixture setup, execution and pytest reporting. It stops the suite
at its deadline. A timeout, failure, missing report, empty selection, unmarked
selected test, any skip, or elapsed time of 300 seconds or more cannot pass.
The result retains exact selected node IDs, source hashes, environment and
package versions, elapsed time, pytest logs and JUnit output. The printed path
identifies this local evidence. Dependency installation is never hidden inside
the timed command.

The final complete local command took 5.04 seconds on macOS ARM64 with
Python 3.12.13. All 56 selected tests completed, including 100 subtest cases,
with no skips or failures. This is a prepared-environment observation, not a
guarantee for every machine. The retained test list and environment identify
that measurement.

Normal CI runs these same tests along with its existing larger selection. The
local marker does not remove any CI coverage or any of the eight primary GPU
learning tasks. Nightly, weekly and release conditions remain unchanged.

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

Vedaant Jain accepts about 22 minutes through the last required result and
asks that CI stay below 25 minutes. Twenty minutes remains preferred. The
complete timeline includes build,
queues, installation, fixture validation, selected checks, reporting, and uploads.
Per-job timeouts are execution bounds. They do not prove the complete target.
The source wrapper waits up to 30 minutes for the existing source workflow.
Its job limit leaves time for setup and result upload. A source workflow that
exceeds the accepted target still reports that miss. The longer observation
window allows the wrapper to retain the completed test result.
The recorded manual trial and worker timings do not replace a GitHub run.

The Rust cache includes workspace crates and keys them to the native sources
and bundle builder. A miss still builds from source. Every run retains the
exact-source and glibc checks. Cache improvements need runtime measurement.

The [first run with bundled fixtures at `ca562fb`](https://github.com/NVIDIA/kumo-engine/actions/runs/40947297560)
completed its GPU checks successfully. It verified eight normal seed-zero
primary fits and 39 additional installation and behavior fits on a Tesla T4.
The selected worker took 462.61 seconds. The GPU job, including setup,
installation, validation and upload, took 10 minutes 45 seconds.

That cold build took 14 minutes 33 seconds. Native compilation occupied
766.69 seconds. The job saved the workspace-crate cache. The complete Engine workflow took 28 minutes
27 seconds and failed because the previous 15-minute source observer stopped
before the existing source workflow completed. The source workflow later
passed in 24 minutes 55 seconds, with 49 successful jobs and one conditional
skip. The longer observer preserves that completed result. These historical measurements
remain failed observations under their original 20-minute policy.

The [measured run at `eb1bd5f`](https://github.com/NVIDIA/kumo-engine/actions/runs/40952212441)
passed all five Engine jobs in 21 minutes 17 seconds. The build used the exact
native cache and took 9 minutes 24 seconds. The GPU job took 10 minutes
40 seconds. Its selected worker completed eight primary fits and 39 additional
fits in 473.89 seconds. The existing source workflow passed in 19 minutes
17 seconds. These are measured observations, not a guarantee for every queue
or runner. The original record remains a 77-second miss against 20 minutes.
Under Vedaant's accepted 22-minute policy, the same observation meets the
accepted target and stays below the 25-minute guardrail.

After a run finishes, download its run metadata and the complete job inventory,
including all pages for its recorded attempt. Then audit that run:

```sh
python3 .github/scripts/engine-pr-timing.py \
  --run run.json --jobs jobs.json --revision FULL_COMMIT_SHA \
  --output timing.json
```

The audit includes queue time and all jobs through their final completion.
It reports the preferred 1,200-second metric separately. A verified successful
run can pass the accepted 1,320-second target, including equality. A run above
that target remains a target miss. Runs at 1,500 seconds or more also breach
the strict guardrail. Historical report files keep their original policy.
This full-CI policy does not relax the strictly under-300-second local limit.
It rejects incomplete attempts, stale commits, and failed jobs hidden by a
green workflow. Audit the other required workflows too before claiming the
complete PR target. The earlier source workflow took 1,495 seconds; the later
run took 1,157 seconds. Keep both observations and their runner variation visible.

The workflow keeps selection, build identity, installation, task results,
physical-fit receipts, test results, fixture integrity, and failed-run evidence.
No nightly, weekly, release, or reference-calibration job is added here.
