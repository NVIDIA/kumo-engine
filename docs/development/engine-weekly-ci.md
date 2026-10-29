# Engine weekly check

The first weekly block contains eleven primary model fits. Ten fits cover seed-0 normal models and real scrambled-answer controls for Q13 through Q17. P2 adds one fixed training and prediction workload on the full commerce/25B graph. The first reference block uses `kumo-engine==1.0.0`. A later candidate block trains eleven fresh models and compares them with those saved references. Every candidate fit ID must differ from every reference fit ID.

The runner keeps execution and input checks strict. It reports quality, time, memory, and prediction differences without new numerical limits. The separate performance reference is `1.0.0+group5.20261023.1`. The approved 1.0.0 P2 observation does not replace that reference. P2 has no frozen quality answers. Its scores remain null with an explicit reason. A missing or incompatible performance reference remains visible in the report.

| Case | Dataset | Query | Metric | Primary fits |
| --- | --- | --- | --- | --- |
| Q13 | streaming/100M | next_titles | Recall@50 | Normal and scrambled-answer control |
| Q14 | community/100M | messages_30d | Mean absolute error | Normal and scrambled-answer control |
| Q15 | payments/100M core | flagged | AUROC | Normal and scrambled-answer control |
| Q16 | fraud/100M | account_flagged | AUROC | Normal and scrambled-answer control |
| Q17 | commerce/1B | next_merchants | Recall@10 | Normal and scrambled-answer control |
| P2 | commerce/25B | next_merchants | Complete fixed prediction workload | One fixed workload |

Q15 requires the payments core variant. The full payments variant does not satisfy it. Q15 and Q16 retain their six-hop model plans. P2 loads the full graph and uses frozen example populations. It does not mean one epoch over all twenty-five billion graph rows. Do not reduce data, batches, or work to reach a time target.

## Freeze inputs and admit capacity

A manifest is a file that lists the cases and their exact inputs. Before a fit, freeze its source, wheel, data, query, splits, model plan, control, dependencies, and device profile. Reuse existing validated data and prepared graphs when their identities match. Keep reserved answers out of plan selection and control construction. Preserve chronological populations and query eligibility filters.

Use schema 1 with `reference`, `performance_reference`, and eleven `cases`. The reference contains version `1.0.0` and the approved wheel SHA-256. Each case contains `id`, `workload`, `dataset`, `query`, `seed`, `condition`, `metric`, `hardware_profile`, `identity`, `inputs`, `capacity`, and `worker`. The first five workloads have conditions `normal` and `scrambled_answer_control`. P2 uses `fixed_scale_workload`. All cases use seed 0. Immutable manifest revisions can fill pending inputs for other cases. Each selected case must be complete before its fit. Collection accepts later revisions only when that complete case and the global roster, references, and policies match exactly. Each attempt retains its original manifest bytes and SHA-256.

The `identity` object contains eight SHA-256 values: `source_sha256`, `data_sha256`, `query_sha256`, `split_sha256`, `model_sha256`, `control_sha256`, `dependency_sha256`, and `device_profile_sha256`. The dependency value hashes the actual dependency-profile file. Store any canonical package-map hash separately. The device profile describes allowed hardware. Each result records the physical GPU UUID separately. The host capacity receipt binds this initial route to its actual instance and GPU.

The `inputs` object maps frozen file paths to their hashes. Include the worker configuration, frozen plan, descriptors, source manifests, and imported worker dependencies. The `worker` object contains its `path` and `sha256`. Extra fields can identify its configuration and prepared inputs. The adapter exports `run_case(spec, output)` and returns the measured result. Version that adapter with the frozen source or include it in the reviewed change.

The `capacity` object contains a receipt `path` and `sha256`. Its receipt must have `status: ADMITTED`, the matching `case_id` and `hardware_profile`, `native_capacity_checked: true`, and the actual `gpu_name`. Its `observed_hardware` object contains `instance_id`, `gpu_uuid`, `driver`, `vcpus`, `memory_total_bytes`, and `gpu_memory_bytes`. The runner compares those values with the executing host before fitting. Retain the native memory, storage, input, owner, process, and lock checks behind that receipt. Software installation or a CUDA probe alone does not admit model capacity.

The dedicated RTX host has about 124.77 GiB of operating-system RAM and 1.9 TB of local scratch. Earlier graph preparation used 127.68 GiB for commerce/1B and 497.76 GiB for commerce/25B. The 25B attempt added 2233.52 GiB of scratch. Those measurements do not establish capacity on the weekly RTX host. Q17 retains the approved RTX route. P2 can use the separately admitted Scale GPU profile when its owner, processes, and locks permit use. This workflow provisions no hosts and changes no shared services.

## Create or reuse reference evidence

Use the approved installed wheel environment and frozen manifest. Run one ready case through `weekly_check.py run --mode reference`. The command acquires the existing host lock and refuses a changed input, incomplete capacity receipt, or different wheel. The shared fit observer records every physical fit and completed CUDA batches. The worker must perform exactly one fit. A failed physical attempt remains separate from the eleven logical cases.

```bash
python -I packages/kumo-engine/scripts/weekly_check.py run \
  --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA256" \
  --case Q13-normal --mode reference --version 1.0.0 \
  --wheel "$REFERENCE_WHEEL" --wheel-sha256 "$REFERENCE_WHEEL_SHA256" \
  --expected-prefix "$REFERENCE_VENV" --dependency-profile "$DEPENDENCY_PROFILE" \
  --host-lock "$HOST_LOCK" --output "$NEW_CASE_DIRECTORY"
```

The output directory must be new. Reusing a directory never starts another fit. The runner writes `fits.json` before model work and preserves a failed case report when an operation fails. A complete case includes finite predictions, actual training work, CUDA receipts, actual plans, split fingerprints, resource samples, and saved/loaded state. It also includes full, repeated, partial, and reloaded prediction exports. Keep the saved model and complete raw evidence on durable storage.

The adapter returns the existing scale-worker result fields, plus the exact case `identity`. Training work includes completed batches, examples, and optimizer updates. Reload evidence must identify the selected checkpoint and restored tensors. Standard artifact names include `requested-plan.json`, `selected-model-plan.json`, `generation-plan.json`, `resolved_model_plan.json`, `trainer_config.json`, `runtime_manifest.json`, `feature_schema.json`, `device.json`, `measurements.json`, and `memory.json`. Resource samples must contain `overall.samples`. A saved checkpoint path ends in `model/checkpoint`.

The runner independently recomputes MAE, AUROC, and Recall@K from saved result coordinates. Native boolean binary targets remain valid. If the frozen native population makes a quality metric undefined, keep that population. Report the undefined score as `null` with `quality_missing_reason`. Do not silently drop empty link answers or replace them with a new score convention. Predictions still must be complete and finite.

After all eleven cases complete, collect their evidence into a new bundle. The collector refuses missing or repeated cases, mixed wheels, changed files, missing models, and reused fit identities. Verify local copies before reclaiming scratch. Reuse an old fit only after its whole identity and complete evidence match the selected case. A repair that only scores or collects a retained fit does not need another training attempt.

```bash
python -I packages/kumo-engine/scripts/weekly_check.py collect \
  --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA256" \
  --mode reference --case-reports "$CASE_ROOT"/*/weekly-case.json \
  --output "$NEW_REFERENCE_BUNDLE"
```

Record the SHA-256 of the bundle's `manifest.json` outside the bundle. The bundle includes complete models and evidence. Its manifest pins every file and all eleven case identities. The collector copies complete case directories, so admit space for that copy before collection. Do not place comparison output inside an immutable bundle.

## Bind the scheduled workflow

The workflow uses existing registered runners and existing durable storage. Configure the repository variable `KUMO_ENGINE_WEEKLY_CONFIG` before a run. It contains `manifest_sha256`, `reference_sha256`, and `profiles.rtx` plus `profiles.scale`. Each profile supplies `runner` as a list of existing runner labels and absolute paths for `root`, `manifest`, `dependency_profile`, `reference_wheel`, `reference_root`, and `host_lock`. Reference creation can omit `reference_sha256` because that block does not exist yet.

```json
{
  "manifest_sha256": "<frozen manifest SHA-256>",
  "reference_sha256": "<complete reference manifest SHA-256>",
  "profiles": {
    "rtx": {
      "runner": ["self-hosted", "linux", "x64", "<existing RTX label>"],
      "root": "/absolute/weekly-runs",
      "manifest": "/absolute/frozen-manifest.json",
      "dependency_profile": "/absolute/dependency-profile.json",
      "reference_wheel": "/absolute/kumo_engine-1.0.0.whl",
      "reference_root": "/absolute/approved-reference-bundle",
      "host_lock": "/absolute/gpu-study.lock"
    },
    "scale": {
      "runner": ["self-hosted", "linux", "x64", "<existing Scale label>"],
      "root": "/absolute/weekly-runs",
      "manifest": "/absolute/frozen-manifest.json",
      "dependency_profile": "/absolute/dependency-profile.json",
      "reference_wheel": "/absolute/kumo_engine-1.0.0.whl",
      "reference_root": "/absolute/approved-reference-bundle",
      "host_lock": "/absolute/gpu-study.lock"
    }
  }
}
```

Use the same pinned manifest and frozen inputs on both profiles. The selected paths must exist on their respective hosts. The GPU jobs require Python 3.12, enough admitted storage, and access to the declared public dependency packages. They create fresh isolated environments. They do not use the historical candidate readiness environment.

Manual `reference` mode creates only the eleven approved reference fits. Scheduled runs use `candidate` mode. A manual candidate run can select a revision. The build receipt records its exact commit, tree, wheel, and wheel hash. The candidate mode checks a complete matching reference before installation and training. Each case retains the build receipt for the exact source commit, tree, and wheel. It runs ten cases on the RTX profile and one P2 case on Scale. Jobs preserve failures and upload raw evidence. The final job collects the complete block and compares matching cases.

The scheduled time is Sunday at 00:30 UTC. Change that schedule only as part of a reviewed configuration change. Runs do not cancel an earlier weekly run. Host locks still control actual GPU use. Missing runner bindings, storage, capacity, source, or reference evidence must fail explicitly. This change does not register runners or create another service.

## Evidence and remaining validation

This implementation depends on the existing shared PR workers and committed `nightly_check.py` foundation at `7f28d954d6e19f8ea63d85461b17ed8fb66f9f2b`. It leaves the PR and nightly workflows unchanged. This initial receipt format pins host paths and native admission to one instance. Moving a case to another physical host needs a separately reviewed input and reference route. The model-free tests exercise manifest admission, complete evidence collection, score recomputation, failures, and reference matching. They do not establish GPU model results or a finished GitHub run.

Keep the PR draft until the concrete frozen adapters and access bindings are reviewed. The first reference execution and a later candidate comparison are separate runs. Do not claim that the candidate route passed end to end until it actually runs. Do not launch an extra candidate block merely to validate the first reference task.

B6 reuses each fitted model for repeated, partial, and reloaded predictions. P1 adds five CPU sampling operations and no model fits. B9 needs fresh prediction from retained models on two matching GPUs. The one-GPU weekly host alone cannot prove B9. This primary runner does not implement those separate P1 or two-GPU B9 operations. Record their actual status separately before claiming the complete first-weekly behavior coverage.

Training variation, four additional P2 repetitions, deliberate faults, new numerical gates, and broader release qualification remain deferred. The training-stage target is under one hour. The comparison reports that target as unproved when exact training-stage timestamps are absent. When all timestamps exist, it reports the span from the first actual fit start to the last actual fit end. That conservative span keeps any gaps visible. Measure the training critical path separately from data preparation, builds, installation, transport, queues, scoring, and reporting. Keep the complete workflow elapsed time visible too.
