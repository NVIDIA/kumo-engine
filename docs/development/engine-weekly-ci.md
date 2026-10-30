# Weekly learning and model compatibility checks

The weekly runner trains eleven seed-zero models and tests saved models across
wheel versions. The `rtx` profile runs the ten Q13-Q17 cases. The `scale` profile
runs P2 on its separately admitted capacity. A wheel is the installable Python package. Candidate work uses
one new fit per case. A fit is one model training operation.

Q13 predicts the next titles on streaming/100M. Q14 predicts message counts on
community/100M. Q15 predicts flagged payments on payments/100M core. Q16 predicts
flagged accounts on fraud/100M. Q17 predicts the next merchants on commerce/1B.
Each task has a normal model and a model trained with scrambled answers.
P2 adds one fixed commerce/25B workload. P2 has no approved quality answers.

For Q13, Q14, Q16, and Q17, the normal model must retain at least half the
reference's separation from its scrambled-answer control. Lower MAE means better
quality for Q14. Higher scores mean better quality for the other tasks. The
reference must first demonstrate a positive separation for each gated task.
The half-gap rule is a proposed correctness margin, not a calibrated performance
limit. Reference approval includes review of this margin.

Q15 scores remain reports because the recorded normal and control scores do not
establish useful learning. Its predictions and scores must still be complete
and finite. Native graph counts, temporal sampling, source time columns, and
feature names must match the frozen inputs. Reserved answers cannot enter the
feature schema. Time and memory differences remain reports for every task.

Before each candidate fit, the installed candidate loads that case's approved
reference model. It predicts the full stored population and compares the output
with the stored reference predictions. This operation adds no fit. Entity IDs,
labels, and ranks must match. Scores must stay within the manifest's approved
absolute and relative tolerances. A public Lightning callback also compares the
restored weights with the saved checkpoint. Checkpoint-format failures, missing
rows, changed weights, and excess score drift fail the case. A candidate also reloads its own saved model.

P1 means five CPU sampling operations on commerce/100M. B9 means prediction with
the retained P2 model on one GPU and then two matching GPUs. They are separate
changes and do not run in this workflow.

## Inputs and reference approval

`weekly_check.py` accepts a data-only `manifest.json` with format
`kumo-engine-weekly-input-v2`. Each of the eleven case IDs is fixed:
`Q13-normal`, `Q13-control`, through `Q17-normal`, `Q17-control`, and `P2-fixed`.
The reference version and wheel checksum are manifest fields.

The top-level `files` map binds relative data paths to SHA-256 checksums. Each
case's `files` list selects its complete payload. Relative paths cannot escape
the input root or use symbolic links. Workers always come from this repository.
Input artifacts cannot choose a Python module or shell script.

Each case declares `dataset`, `metric`, `arm`, `seed`, `query`, `gpu_class`,
`capacity`, and `tolerance`. `gpu_class` contains `compute_capability` and
`minimum_memory_bytes`. `capacity` contains `memory_bytes` and `scratch_bytes`.
`tolerance` contains finite nonnegative `atol` and `rtol` values. GPU identifiers
and driver versions are observations. They do not restrict host replacement.

`query` binds the training query. If prediction uses a different frozen query,
declare `prediction_query` as well. P2 uses a bounded training query and the
registered prediction query. Both retain the stored prediction population.

Each case also binds these relative data paths:

| Field | Contents |
| --- | --- |
| `snapshot` | JSON with `id`, `index_path`, and `graph_config` for a prepared `GraphSnapshot` |
| `dataset_manifest` | The published source manifest with full dataset counts and per-table row counts |
| `training_table` | JSON with `staging_dir`, `train_table_path`, `manifest_path`, `required_stages`, and `generation_plan` |
| `prediction_table` | JSON with `staging_dir`, `table_path`, `row_count`, `tt_id`, `timestamps`, and `required_stages` |
| `model_plan` | The frozen public `ModelPlan` JSON, with one experiment, seed zero, and no refits |
| `population` | Parquet with unique `ENTITY1` values and the frozen `TIMESTAMP` where required |
| `answers` | Reserved Parquet with `ENTITY1` and `TARGET1`, or null for P2 |
| `catalog` | For ranking, Parquet with the complete integer `CLASS` catalog |

Ranking cases also declare `raw_limit`. The reader sorts raw Parquet by entity,
descending score, and class. It uses one thread and a 2 GB memory limit, with
spill outside the evidence tree. It rejects incomplete populations, invalid
catalog keys, and duplicate coordinates. It keeps the approved public top-K
behavior when the returned boundary has ties. Reference replay still requires
the same stored item order and score tolerance.

All immutable files under the prepared snapshot and staging directories must
appear in the checksum map and the case's file list. Exclude the native index's
`train_tables` cache. Each worker creates a private graph directory in scratch,
with hard links to the frozen graph files. Native compilation writes there.
Scratch must use the same filesystem as the graph. Graph bytes remain pinned
and are checked again after work completes. Preparation must retain the full
named datasets, queries, splits, scrambled answers, model recipes, and prediction
populations. The old host-script manifest is not accepted as this format.
Converting and auditing the retained input payload remains necessary before GPU use.
A profile validates only its own data files, so the other host's graph does not
have to be copied onto it.

To prepare a new reference, select `mode=reference` with the approved wheel version
and checksum in the input manifest. Review all eleven fits, the learning gaps,
full predictions, saved models, runtime, and resource measurements. Approve the
result only after this review. For a new two-profile reference, download both
profile artifacts and place their `cases` directories beside the collected
`reference.json`. The collector does not upload a duplicate copy of these models.
Bind the reviewed `reference.json` checksum in the
repository configuration. A successful preparation run does not approve itself.
Keep model bundles trusted because loading a checkpoint deserializes model data.

Existing approved models do not need new fits for this format. Run
`weekly_reference.py` against the original extracted reference bundle and its
approved `manifest.json` checksum. Supply `--legacy-root`, `--legacy-sha256`,
`--input-root`, `--manifest-sha256`, and a fresh `--output` directory. If the
reference arrives as an archive, first match its approved archive checksum and
extract it with Python's data extraction filter.

Each input case must include `legacy_identity` from the original frozen case.
The importer makes sure that the dataset, query, condition, wheel, fit, saved
checkpoint, and completed CUDA evidence match. It copies the original models
and stored predictions without importing historical Python workers or calling
an engine API. It preserves the original bundle and records zero new fits.
Review the converted `reference.json` and its source checksum before approval.

Candidate dependencies do not have to equal reference dependencies. Each worker
runs `pip check` and records its actual packages. Installed package bytes must
match the selected wheel. Review the dependency record with the source and
wheel receipts.

## Manual execution and evidence

The GPU workflow is dispatch-only. Pull requests run CPU guards only. Keep the
schedule disabled until one complete candidate run passes on the admitted host.
The managed runner grant alone does not establish commerce/25B capacity.

Set `KUMO_ENGINE_WEEKLY_CONFIG` to JSON with `manifest_sha256`,
`reference_sha256`, and `profiles`. Both `rtx` and `scale` profiles declare
absolute `input_root`, `reference_root`, `work_root`, and `host_lock` paths.
Each also declares `runner` with the existing runner `group` and `labels`.
`work_root` must use the native graph's filesystem. Use the same host lock as
other work on each host. Both profiles must admit their frozen data, references,
and capacity before either profile starts a fit. Admission also checks the actual
GPU class through `nvidia-smi`. A missing configuration fails a manual dispatch.

`revision` must resolve to a commit in the repository's default branch history.
An ephemeral build job checks this before producing a wheel. Candidate source
never runs from a fork checkout on the persistent GPU host.

The runner reserves each host before hashing inputs and holds its lock across
that profile's full block. A failed case leaves its error and available fit ledger, then the remaining cases run. Engine scratch
stays outside the evidence tree and is removed after each worker. Models,
canonical predictions, raw predictions, logs, and failures stay in one artifact
per profile. The final collection checks all eleven fits, the candidate's eleven
reference replays, and the learning gates. It uploads only a compact summary, so
large model evidence is not uploaded twice. The workflow removes each temporary
environment after upload. Download and review a reference artifact before the artifact's seven-day retention period expires.

Local CPU guards run without CUDA:

```bash
python packages/kumo-engine/test/test_weekly_check.py
python packages/kumo-engine/test/test_weekly_reference.py
python packages/kumo-engine/test/test_weekly_workflow.py
```

The Engine package test suite also runs `test_weekly_model_replay.py`. It uses
a small real CPU model to exercise model loading and prediction against staged
native inputs. It does not qualify the full weekly datasets or a published wheel.

These guards do not establish a passing GPU run or cross-version compatibility
of a particular published reference. The rebuild stays in draft until PR137
settles, the retained inputs pass the new manifest audit, and an actual candidate
completes the eleven cases and eleven reference replays.
