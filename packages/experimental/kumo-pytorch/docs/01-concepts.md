# 1. Concepts

A relational dataset is tables connected by keys: a user row connects to many
purchase rows, a purchase to one item. Turning that into training data takes
several separate decisions. kumo-pytorch gives each decision its own
declaration, so changing one never silently changes another.

## Three declarations, one contract

| Question | Declaration |
| --- | --- |
| What is stored, how are rows related, and which column is each table's clock? | `DataSchema` |
| What may one example read at its prediction time? | `SampleSpec` |
| How are those values encoded and arranged for the model? | `BatchSpec` |

`DataLoaderSpec(schema=..., sample=..., batch=...)` is the whole contract. It
can't be changed once built, you can save it as JSON, and its fingerprint lets
you check later exactly what a model was trained on.

The contract doesn't say where your data lives or what reads it. A `Dataset`
says where each table's data is (Parquet files or in-memory rows), and an
engine reads it and builds the batches.

The package ships two engines, both implementations of the `Engine` interface.
You'll use the DiskGraph engine (`DiskGraphEngine`) for real data; it's built on
kumo-engine's DiskGraph, which the
[kumo-engine README](https://github.com/NVIDIA/kumo-engine#readme) describes.
The in-memory engine (`InMemoryEngine`) runs the contract in plain Python over a
few rows, for trying things out on toy data. The guide's small examples use it so
they can print their results. You always choose the engine yourself when you
call `prepare`, and [engines and data](04-engines-and-data.md) covers both.

## What is stored

`DataSchema` names tables, column dtypes, primary keys, relations and clocks. A
dtype describes a stored value, such as an `int64` or a UTC timestamp. It does
not decide whether the model receives a normalized number or a category ID;
that is an encoding choice made later.

Every table declares its clock, or says it has none (`NON_TEMPORAL`). The
clock is what point-in-time filtering compares against.

## What one example may know

An **anchor** is one example to build: an entity (a row of the entity table,
by primary key) and a **prediction time**. In code, the prediction time is the
`as_of_time` field, and some pages call it the cutoff. You supply anchors, as
`SampleRequest`s or an `AnchorTable`. You also decide how to split them and
which candidates to score; kumo-pytorch doesn't generate any of these.

`SampleSpec` selects what that example may read at that time: the entity's own
values, past events, aggregates over related rows, graph neighborhoods,
caller-supplied candidates, and typed calculations such as calendar parts and
event ages. Historical windows are half-open, `(start, stop]` in seconds from
the prediction time, so an event exactly at the prediction time is history and
anything later is not. Labels look strictly forward from the same time.

**No input ever reads a row from after its example's prediction time.**

## What reaches the model

`BatchSpec` holds named inputs and targets. Each input says which sampled
values it uses, how each one is encoded (with `Raw`, `ZScore`, `Buckets`,
`Vocabulary` or `Text`), and how the result is laid out as tensors. Pages 2 and
3 go through the options. Targets are encoded the same way and never appear in
inputs.

Nothing is shared between inputs unless you ask for it. [Encodings and
IDs](03-encodings-and-ids.md) shows how to share an encoder or an ID numbering
between inputs.

## The lifecycle

```text
prepare(contract, dataset, engine) ──► Plan     validate, resolve types, bind data
plan.materialize(anchors, fit_split) ──► Task  bind anchors, fit encoders on the fit split
task.batches(split=...) ──► Batch              inputs, targets, metadata
task.to_artifact().save(path)                  contract + fitted state, for inference
```

Validation happens before data is read wherever the contract alone decides it.
If the engine can't run part of a contract, `prepare` raises an error that names
the declaration and suggests what to try instead.

## What stays yours

You write the model, its loss, the optimizer, the training loop and
checkpointing, and you supply the anchors and labels. A batch is plain PyTorch
tensors in nested dictionaries, so you can adapt it to any model. Use the
vocabularies and statistics the task fitted rather than fitting your own on the
side, so training and inference encode values the same way.

Next: [contracts, step by step](02-contracts.md).
