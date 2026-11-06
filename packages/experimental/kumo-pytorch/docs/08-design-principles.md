# 8. Design principles

This page explains a few of the choices behind the contract's design, for when
you're wondering why something works the way it does.

## Declare each decision once

Stored column types belong to the schema; calculated result types belong to
calculations; encoders read both. We don't infer a separate "kind" for each
column, and no encoder is chosen automatically from a dtype. You pick the
encoder, so a float can be standardized, bucketed, returned raw or treated as a
category.

## Separate selection from presentation

`SampleSpec` owns which rows and values an example may read, and every fixed
calculation over them: arithmetic, calendar parts, grouping, anchor-relative
ages. `BatchSpec` owns how the selected values are encoded and arranged.
Adding an output never gives access to different rows, and the same sample can
feed very different models ([page 7](07-model-input-shapes.md) shows four)
without changing what data they read.

## Keep one encoding path

Columns, token rows, composite rows, graph attributes and independently encoded
targets all use the same encoder implementations and fitted state. Model code
reads the task's fitted metadata (vocabulary sizes, special IDs) instead of
learning its own mappings, so training, batch scoring and serving cannot drift
apart.

## Make sharing and numbering explicit

A named encoder shares fitted state, a named ID space shares output numbering,
and a field-name dictionary identifies fields. You state each of these where you
use it, as [encodings and IDs](03-encodings-and-ids.md) describes.

## Order only what is meaningful

Named declarations are maps, so the order you write them in doesn't affect
behavior or fingerprints. Where order changes the result, as with output field
sequences, operands, composite components, event order and marker placement,
you give it explicitly.

## Make alignment visible

An input that adds values to another input's events (`EventsOf`) or token
positions (`OccurrencesOf`) names that input, so it's clear which rows line up.
Calendar parts and event ages are ordinary sampled values that you ask for
explicitly, not something hidden in layout metadata.

## Keep missing, unknown and padding apart

A NULL source value is not an observed zero, an unseen category or a padded
position. Validity masks and reserved special IDs preserve those differences
all the way to the model.

## Exact results

Raw typed values aren't converted into strings or floats. Calculations have
checked result types, and integer sums are exact and fail on overflow instead
of wrapping. If the engine can't compute something exactly, `prepare` says so
and names the declaration and a next step.

## Unsupported features

Features we haven't built yet, such as array columns, string-encoded lists and
artifact migrations, raise an error instead of guessing at a behavior. The
[limitations](limitations.md) page lists them, and the
[behavior guarantees](../conformance/spec/guarantees/) give the exact rules.
