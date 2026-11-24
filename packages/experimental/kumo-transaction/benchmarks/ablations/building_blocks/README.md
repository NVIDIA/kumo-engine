# Building-block ablation (experimental)

**Support tier:** `experimental` (CPU synthetic smoke)

## Goal

Answer "what does this slot buy?" by changing one building block at a time
and scoring every variant the same way. The harness is the method; the fixture
is too small for its numbers to rank anything.

## Architecture

The base is the Blueprint v2 hybrid at CPU size
([`config.yaml`](config.yaml)). Each row of [`variants.yaml`](variants.yaml)
applies a few dotted-key overrides to it:

| Variant | Slot | Change |
|---|---|---|
| `v2_hybrid` | baseline | none |
| `no_time_positions` | event representation | `NoPE` instead of TimeRoPE in the history decoder |
| `fixed_event_fields` | event representation | the 12 flat Blueprint fields as key-value tokens |
| `flat_decoder` | sequence architecture | one flat stream through a causal decoder (token scheme and objective change with it) |
| `next_fields_only` | objective | drop the within-event MLM auxiliary |
| `plus_contrastive` | objective | add the CoLES-style contrastive auxiliary |
| `mean_readout` | readout | mean over event states instead of the last one |

Add a row to compare something else. `tfm.runtime.compare.apply_overrides`
validates every variant as a full `RecipeConfig`, so a pairing the kit refuses
fails before anything trains.

## Task

Self-supervised pretraining per variant, then the fraud probe of
[ADR 0013](../../../docs/architecture/0013-fraud-probe-split-integrity.md) with
its `raw`, `embedding`, and `combined` arms.

## Data assumptions

Every variant shares the base `data` section and seed, so the book, cutoffs,
label windows, and probe user split are identical. The smoke test asserts
that. The one exception is inside the book: `fixed_event_fields` drops app
events, which have none of the 12 fields, so its `raw` aggregates (event
count) differ from the other rows.

With no `TFM_PRAGMATIQ_ROOT`, a tiny synthetic pragmatiq-shaped fixture is
generated. Point `data.path` or `TFM_PRAGMATIQ_ROOT` at a real book and raise
the sizes in a copy of `config.yaml` for a comparison worth reading.

## Commands

```bash
uv run python benchmarks/ablations/building_blocks/smoke_test.py
uv run python benchmarks/ablations/building_blocks/compare.py
uv run python benchmarks/ablations/building_blocks/compare.py --seeds 42 43 44
uv run python benchmarks/ablations/building_blocks/compare.py --only v2_hybrid mean_readout
```

`compare.py` is the train, evaluate, and report entry point for every variant
in one command; there is no separate inference step because the probes embed
their own rows. Hardware: CPU, about a minute for all seven variants.

## Expected outputs

Under `outputs/ablation_building_blocks/<run-id>/` (or `--run-dir`):

- `comparison.md`: one row per variant, ROC-AUC per arm as median [min, max]
  over seeds, with the evaluation positives that support it;
- `comparison.json`: every variant's overrides and full metrics per seed;
- `<variant>/seed_<n>/`: the usual run artifacts, including
  `resolved_config.yaml` and `probe_manifest.json`.

## Limitations

The default fixture has 48 users and four evaluation positives, so the table
shows that the comparison runs, not which slot is better. One seed is not a
result: pass `--seeds`. Losses are not compared because the objectives differ.
Variants train for the same number of steps, not the same compute. No
production, fraud, or paper-fidelity claims.
