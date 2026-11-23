# Changelog

## Unreleased — 2.0.0a2

A rewrite of the v1 notebook example into a reference-architecture kit. Nothing
from `v1.0.0` (tag, and branch `v1`) is compatible: no checkpoint, config, or
metric carries over. The exact v1 replay and its errata are on branch
`archive/v1-replay`. This entry summarizes the kit against `v1.0.0`; the
commit history has the steps in between. Items marked **Breaking since
2.0.0a1** changed after the GTC Berlin lab's snapshot (below), and the
[2.0.0a1 to 2.0.0a2 migration guide](docs/migration/2.0.0a1-to-2.0.0a2.md)
maps each of them; `tfm migrate-config` rewrites a 2.0.0a1 config.

### Since 2.0.0a1

- New: `tfm migrate-config` rewrites a 2.0.0a1 config (such as a checkpoint's
  `config.yaml`) into this release's keys and checks that it loads; it
  refuses a setting this release cannot run. The GTC Berlin lab's Blueprint v2
  checkpoint config migrates to `benchmarks/ablations/blueprint_v2_final_d.yaml`.
- `recipes/blueprint_v1/config.cached.yaml` is `config.yaml` with
  `data.token_cache` (2.0.0a1's `tabformer_hf_decoder/config.gb10_cached.yaml`),
  and `config.yaml` trains with `training.backend: ddp`, as 2.0.0a1's
  `config.gb10.yaml` did.
- Flat configs whose pretraining windows are wider than their row, or whose
  rows are wider than the model's context, are refused at load.
- `notebooks/pragmatiq`: a new tour (00) and notebooks 01, 02, 03b, and 04
  adapted from the GTC Berlin lab's Blueprint v2 track. The series runs on the
  final-sweep checkpoint: `tfm.runtime.lab.LAB_CONFIG` is
  `benchmarks/ablations/blueprint_v2_final_d.yaml`, the checkpoint folder is
  `blueprint_v2_lab_tuned`, and `fetch_lab_assets` migrates a 2.0.0a1
  checkpoint's `config.yaml`. Notebook 05 tells the final sweep.
- `notebooks/blueprint_v1/04` measures the embedding space it plots
  (silhouette and distance ratios by customer, in the full space and in 3-D).

### Recipes and entry points

- Three reference architectures, each run with
  `tfm <stage> --recipe <slug> [--config <variant|path>]`:
  - `blueprint_v1`: NVIDIA's TFM Blueprint on TabFormer, with v1's label
    shift fixed and each transaction scored with the user's recent history
    (ADR 0017).
  - `pragma_like`: a PRAGMA-style profile, event, and history encoder stack
    (ADR 0007).
  - `blueprint_v2`: PRAGMA's encoders under a causal history decoder (ADR 0012).
- `pragma_like/config.yaml` is matched to `blueprint_v2/config.yaml` on
  everything but the architecture, and a unit test keeps it that way.
- `train`, `infer`, `evaluate`, and `report` each run the stages before
  them. `smoke` runs `config.smoke.yaml` on CPU, and `--reuse-checkpoint`
  re-evaluates a trained run.
- Slot-by-slot ablations live in `benchmarks/ablations/`. Each recipe has one
  notebook series per dataset: `notebooks/blueprint_v1` and
  `notebooks/pragmatiq`.
- **Breaking since 2.0.0a1:** recipes `tabformer_hf_decoder`,
  `profile_history_kvt`, `blueprint_v2_hybrid`, `decoder_fraud_baseline`, and
  `building_block_comparison`, and their per-recipe scripts, are replaced as
  the migration guide lists.

### Composition and contracts

- `TrxFM(...)` with the core exports `Flat`, `Decoder`, `NoPE`,
  `EagerAttention`, `LastTokenPool`, `NextEvent`, and `Binary`. The
  hierarchical stacks are in `tfm.experimental`.
- Five building-block slots (ADR 0014). Validation refuses the pairings that
  would leak or cannot learn, before any data is prepared.
- Event, entity, and sequence labels are distinct types. Profiles and labels
  are cut at each row's cutoff, and the fraud probe is entity-disjoint, with
  per-side label windows (ADR 0013).
- New `tfm.api.register_backbone` adds a `model.backbone` name for the flat
  stack. Built in: `eager`, `huggingface` (Llama, which requires
  `position: rope`), and `gru`.
- New experimental `flat_encoder` granularity (ADR 0019, `tfm.experimental`):
  `FlatEncoder`, a bidirectional encoder on `transaction_field_tokenizer`
  rows, trained by the new `masked_fields` objective (`MaskedFields`) and read
  out by `last_event_mean` (`LastEventMeanPool`). The field tokenizer reports
  its field id ranges as a `FieldVocabulary`.
- `pragma_like` and `blueprint_v2` share their event and profile encoders
  (`tfm.layers.kvt_encoders`). Older `pragma_like` checkpoints still load.
- `TrxFM.to_config` reports the declared position, the feed-forward width, and
  the head objectives.
- **Breaking since 2.0.0a1:** `tfm.data.collate.sequence_fraud_label` is
  now `sequence_label(..., label="fraud")`. `collate_batch` takes `label` and
  `amount_key`.
- Surprise scoring (`tfm.evaluation.surprise`) accepts any model that
  implements `NextEventScorer`, not just one class.

### Data and tokens

- Adapters for generated data, TabFormer (`tfm prepare-tabformer`), and
  pragmatiq books (ADR 0015). Each dataset has one role, and results from
  different datasets are never combined.
- Spliced identity-theft fraud on pragmatiq (`data.fraud_labels:
  spliced_identity_theft`), because a single event identifies the book's own
  fraud labels.
- Tokenizers:
  - `transaction_field_tokenizer`, with `features.fields`;
  - the `key_value_time` tokenizer, with `categorical_keys` and `hashed_keys`;
  - the `flat_key_value` and `event_fields` schemes;
  - `rapids_field_tokenizer`, an optional cuDF extra.
- `tfm cache-tokens` writes a pragmatiq book's tokens to disk once (ADR 0016).
- `features.exclude_keys` (`key_value_time` only) drops identity keys, such as
  TabFormer's `user`, from fitting and encoding. Tokenizer states and token
  caches without it load unchanged.

### Training

- Training backends `eager`, `ddp`, and `fsdp2` behind one contract suite
  (ADR 0011). Under `torchrun`, `training.batch_size` is per rank, and only
  rank 0 evaluates and writes artifacts.
- `training.precision: bf16` and sequence packing are available, along with
  these optimizer options: Muon, cosine schedules, and gradient clipping.
- LoRA and full fine-tuning, with a linear or joint-fusion head
  (`fraud_task.finetune`).
- **Breaking since 2.0.0a1:** `training.backend: nemo` is removed; use
  `fsdp2`.

### Evaluation

- One headline per task (ADR 0018): XGBoost on the raw row with and without
  the frozen embedding, with hand-built `tabular_xgboost` as the reference.
  The model card opens with this table. One constructor,
  `xgboost_classifier`, builds every XGBoost model, on CPU or with
  `device: cuda`.
- The logistic `raw` / `embedding` / `combined` stack runs only when the
  XGBoost arms are off, which is the CPU smoke path. `single_event_raw` is
  dropped, because it repeated the history row's raw arm.
- History-window fraud probe, with a single-event control. Next-event
  surprise is an extra readout (`evaluation.surprise`).
- `evaluation.history_window.max_rows: null` scores every evaluation-side
  row. `blueprint_v1/config.yaml` uses it and one fixed XGBoost setting, as
  `blueprint_v2` and `pragma_like` already did.
- **Breaking since 2.0.0a1:**
  `tfm.evaluation.fraud.baselines.evaluate_fraud_baselines` and its helpers
  are removed. Use `evaluate_binary_probes` or the runtime evaluator.

### Configuration

- Every key changes behavior or is rejected. A removed key fails with an error
  that names its replacement.
- **Breaking since 2.0.0a1:** optional knobs are grouped into sub-sections:
  `data.pretrain_windows`, `training.optimizer.muon`,
  `evaluation.history_window`, and `fraud_task.supervised_head`. Also, these
  keys are removed: `data.pretrain_window_stride`,
  `training.packing_seq_length`, and joint fusion's `cross_layers` and
  `plr_frequencies`.

### Records and documentation

- Records live under `benchmarks/results/`, with an index that maps the
  recipe names in older records to the current ones; superseded records are
  in `benchmarks/results/archive/`. The roadmap is a checklist per milestone.
- Three-seed records for both milestones: `blueprint_v1_seeds` on TabFormer,
  and `blueprint_v2_seeds` for `blueprint_v2` beside `pragma_like` on
  pragmatiq.
- `pragma_like_tabformer_seed42` scores PRAGMA-like's encoder beside v1's
  decoder on the same TabFormer rows
  (`benchmarks/ablations/pragma_like_tabformer.yaml`).
  `flat_encoder_tabformer_seed42` scores ADR 0019's flat BERT encoder on v1's
  own tokens and the same rows
  (`benchmarks/ablations/flat_encoder_tabformer.yaml`).
- `blueprint_v2_final_sweep` tries `blueprint_v2` at 1,500 or 3,000
  pretraining steps, by LoRA rank 8 or 16, at three seeds, against a selection
  rule fixed in advance. The recipe config is unchanged; the winner, 3,000
  steps with rank 16, is `benchmarks/ablations/blueprint_v2_final_d.yaml`,
  whose seed 42 is the GTC Berlin lab's checkpoint.
  `benchmarks/blueprint_v2_final_sweep.sh` reruns the sweep and
  `benchmarks/blueprint_v2_final_sweep.py` scores it.
- A documentation site (`make docs`) renders the concepts, the ADRs,
  the notebooks, the results, and the API reference.

## 2.0.0a1 — 2026-10-15

The SDK the GTC Berlin 2026 training lab (`x-fx-141-v1`, `main` at `944882d`)
vendors and installs at `/opt/tfm`: `v2` at `9ed45db`, the TabFormer token
cache, and the lab's edits below. Its full entry, against `v1.0.0`, is this
file at tag `v2.0.0a1`. Recipe names and config keys in this section are
2.0.0a1's.

### Changes from the GTC Berlin lab

- New: `training.log_every` (default `None`, silent) prints step, epoch, loss,
  learning rate, gradient norm, token throughput, and time left every that many
  steps, from the first rank only. Every run now writes its per-step curve to
  `train_log.json`, whatever `log_every` is. The eager, DDP, and FSDP2 trainers
  feed the same `tfm.training.progress.TrainingProgress`; FSDP2 does not clip,
  so it reports no gradient norm. The TabFormer GB10 configs set
  `log_every: 50`.
- New: recipe jobs print each stage (setup, train, infer, evaluate) with the
  time since the job started, from the first rank only.
- New: `tfm.evaluation.representation.project(vectors, dims=2|3)` projects
  embeddings to two or three dimensions; `project_2d` is its `dims=2` case.
- New: `TrxFM.to_config()` reports an optional backbone's own `describe()`
  (the HuggingFace Llama backbone reports its depth, heads, KV heads, and
  sizes) instead of the unbuilt `Decoder` spec, or `None` without one.
  `cost_from_composition` refuses a composition that does not run the eager
  `DecoderModule`.
- `embeddings_manifest.json` names the arrays of `embeddings.npz`.
- New, internal and untested beyond the cost guard: `tfm.api.resolve` (slot
  names to components) and `tfm.diagnose.cost` (parameter and FLOP estimates
  from `TrxFM.to_config()`).

### TabFormer token cache and data-parallel training

- New: `tfm cache-tokens` also builds a flat-field token cache of TabFormer's
  train split (`transaction_field_tokenizer`, precomputed splits,
  `data.pretrain_window_events`). With `data.token_cache` set, pretraining
  windows are read from disk, and `train.py` builds no event objects on any
  rank. On all of TabFormer the cache builds once in about 75 s (8 workers,
  7.5 GB peak) against about 9 min and 48 GB per run, per rank, for the object
  path. `evaluate.py` and `report.py` still load events for their probes. The
  cached windows equal the object path's tensor for tensor (ADR 0016).
- New: `setup_run(..., pretraining_only=True)` and
  `tfm.runtime.prepare_pretraining`; the `train` command uses them.
- Fix: `training.backend: ddp` failed on its second step with two or more
  ranks whenever the model carried a task head that pretraining does not train
  (such as `heads: fraud`). DDP now sees only the loss, so untrained heads are
  skipped.
- `tabformer_hf_decoder/config.gb10.yaml` trains with `training.backend: ddp`:
  unchanged on one GPU, data-parallel under `torchrun --nproc-per-node N`.
