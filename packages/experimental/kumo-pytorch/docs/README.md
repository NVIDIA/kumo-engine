# kumo-pytorch guide

This guide explains kumo-pytorch one idea at a time. If you haven't yet, start
with the [quick start](../README.md#quick-start) in the main README, then read
the pages below in order. Each page builds on the one before it, and its code
runs if you paste the blocks in sequence.

1. [Concepts](01-concepts.md) introduces the three declarations, what a
   prediction time is, and which parts of the work stay yours.
2. [Contracts, step by step](02-contracts.md) walks through a full contract:
   the schema, what each example looks at (values, aggregates, event histories,
   graph neighborhoods, candidates), and what the model receives.
3. [Encodings and IDs](03-encodings-and-ids.md) shows how values become numbers:
   choosing encoders, sharing them between inputs, and the ID spaces a model
   embeds.
4. [Engines and data](04-engines-and-data.md) explains what an engine does,
   how to bind your Parquet files, the DiskGraph engine you use for real data,
   and the small in-memory engine the guide uses for its examples.
5. [Training batches](05-training-batches.md) adds labels, fits encoders on the
   training split, and covers batch layouts, deterministic order and splitting
   work across GPUs.
6. [Inference and serving](06-inference-and-serving.md) saves what you trained
   with, loads it in a fresh process, and uses it for batch scoring and for
   serving single requests.
7. [Model input shapes](07-model-input-shapes.md) builds the inputs for four
   published sequence models (PRAGMA, nuFormer, PANTHER and TREASURE) from the
   same tables.
8. [Design principles](08-design-principles.md) explains why the contract looks
   the way it does.

When you need the details on something, these pages are the place to look:

- [Installation](installation.md), if you want CUDA wheels, pip instead of uv,
  or the DiskGraph engine.
- [Supported features and limitations](limitations.md), to check whether
  the DiskGraph engine supports something you need.
- The [API reference](api/README.md), for the exact parameters and errors of
  every public class and function.
- The [behavior guarantees](../conformance/spec/guarantees/), for the precise
  rules the package follows, each with small tested examples.

If you'd rather learn from a complete project, the
[examples](../examples/README.md) solve three modeling problems end to end.

The package's own tests live in `tests/`, `conformance/` (the checks behind the
behavior guarantees) and `e2e/` (end-to-end training and serving runs on
generated data). You don't need any of them to use kumo-pytorch;
[installation](installation.md#run-the-tests-and-notebooks-from-a-checkout)
shows how to run them.
