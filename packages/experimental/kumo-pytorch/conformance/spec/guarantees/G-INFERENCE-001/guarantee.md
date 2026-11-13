# G-INFERENCE-001: inference batches carry the model's inputs and no targets

## Rule

`plan.materialize(anchors, fitted=state, inference=True)` returns a task whose
every batch is an inference batch. Inference is a mode of the same contract,
not a second batch spec:

- Supervised labels (`LabelTarget`, `StructuredTarget`, `CandidateTarget`) are
  not computed and not emitted. The samples are built without the contract's
  row-computed targets, so no row is read for a target, including every row
  after a cutoff that only a target would read. Caller-supplied labels
  (`ExternalTarget`) read no rows; their declarations stay, and they are not
  emitted.
- Input-changing self-supervised targets (`MaskedValuesTarget`) are off: the
  inputs are the unmasked values.
- Targets that leave the inputs as they are (`MatchingGroupTarget`,
  `AlignedEventTarget`) are not emitted; fields an aligned target withholds stay
  withheld. A `NextTokenTarget` is not emitted either, and its stream is the
  whole history up to the cutoff, with no final token held back: the model
  sees every token and predicts the one after the last, as causal inference
  does. Training holds the final token back only because it is the last
  prediction target.
- `Batch.targets` is empty. The metadata still keys every output:
  `sample_ids`, `entity_ids`, `as_of_time`, `splits`, `is_padding` and the
  contract and artifact fingerprints.

Inference never fits. Without `fitted` it is refused, and `fit_split` is
refused with it. `fitted` is the complete state or its input half
(`fitted.inputs`); inference reads only the input half, so both give the same
inputs. The input half must belong to the contract's inputs (its input
fingerprint equals the contract's); a complete state must also belong to the
whole contract (its contract fingerprint equals the contract's). A task given
only the input half keeps only the input half as its fitted state, and writes
no artifact: an artifact holds complete state.

Every anchor is scored: unless the caller passes `drop_last`, the last partial
batch is kept, whatever `Batching.drop_last` the contract trains with.

## Boundaries

An input that does not depend on a target is identical in inference and
training batches for the same anchors and batch arguments. A row that only a
target would read, such as one after the cutoff inside a label's window,
changes no inference batch. Inputs keep their own windows: an input declared
over the current event reads what it reads in training. Labels a request or anchor table
supplies are ignored, not emitted. The task's output schema lists no targets.
Given complete state, its fitted state and manifest are the training state's,
and so is the batches' artifact fingerprint. Given the input half, its manifest
and the batches' artifact fingerprint describe the input half alone: equal to
the training state's when the contract fits no target state (for example, only
`MaskedValuesTarget`), different otherwise.

On the DiskGraph engine, which tied neighbors a `GraphSample(strategy="recent")`
keeps at its fanout limit depends on the index layout, which is the known gap
`GAP-GRAPH-TIES-DG` against G-GRAPH-001 rule 2 in
[registry.py](../../../registry.py). Until that gap is fixed, samples with a tie
at a fanout limit are excluded from the future-row rule on the DiskGraph engine.
The `graph_inference_exact` property checks the rule on graphs for every other
sample, and for tied samples wherever the index is built deterministically (one
shard or one thread).

## Scope

The golden case below fixes the raw input values an inference batch carries,
checked by the conformance harness on the DiskGraph engine. Inference mode
itself, the absent targets, unmasked inputs, future rows and the kept last batch
need encoded batches, so they are covered by tests in
[test_inference_lifecycle.py](../../../../tests/test_inference_lifecycle.py):
the golden case run in inference mode with a label added that counts the rows
after each cutoff; no targets with unchanged inputs and metadata; masking off;
transactions appended after the cutoff change no inference batch while they
change training labels; no sample spec handed to the engine carries a
row-computed target; the DiskGraph engine's model output format equals its
inspectable one; the input half gives the same inputs as the complete state,
reads no target row, and is kept as the task's state; and the refusals above.

`check(..., inference=True)` and `prepare(..., inference=True)` check what
inference runs. The contract must be well formed, targets included, but only its
inputs must be ones the engine runs: a contract whose inputs the engine runs
and whose targets it does not build is refused for training and accepted for
inference. The plan keeps the whole contract and its fingerprints. Its tasks are
inference tasks: `materialize` defaults to inference, refuses `inference=False`,
and checks an anchor table the same way. A plan prepared without
`inference=True` checks the targets too, and also materializes inference tasks.
Tests in the same file cover a label with a row filter, a label through a timed
intermediate table and a label with a decimal operand, none of which the
DiskGraph engine builds: it refuses each contract for training and scores it
for inference in both output formats, with the same batches as batch inference
over those rows read in process.

## Golden examples

- [C-INFERENCE-001](golden/C-INFERENCE-001.json): user 1 has events at 3, 8, 10
  and 15, so its count is 3 at 10 (the event at 10 is included, the one at 15
  is not) and 4 at 16; user 2's events at 12 and 30 give 0 at 10. A label
  counting the rows after each cutoff changes none of these inputs.
