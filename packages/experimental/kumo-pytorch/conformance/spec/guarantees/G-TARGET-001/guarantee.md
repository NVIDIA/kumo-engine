# G-TARGET-001: Explicit labels and input-derived answers

## Rule

Separate targets declare their type and encoding. Input-derived targets reuse
the selected input's final IDs and alignment; they do not fit another mapping.
Masked-value answers are recorded before corruption, then only the named input
is changed using its resolved reserved role IDs. Next-token answers shift the
selected sample sequence without crossing sample boundaries.

## Boundaries

Padding is not supervised. Structural marker supervision follows the objective's
explicit sequence rule, not tensor contents alone. A missing external label is
invalid for inference; a provided label must match its declared type. Input
encoder fitting excludes label observations even when targets reference that
encoder. Aligned event labels follow event identity through filtering and
clipping, not accidental array position. Masking needs a compatible discrete
input and declared mask role. Obvious exposure of the same sampled field in
another input emits a named warning without silently masking that input.
An input may have at most one masked-value or next-token target: duplicate
corruption/shift targets, including mixing both kinds on one input, are rejected
at declaration rather than depending on target map order.

Text-valued aligned-event and candidate targets retain flat piece values and
piece_offsets for each real target row. Their valid mask describes those rows,
and field offsets map samples to their real rows even with a padded parent.
Candidate targets additionally retain the parent's candidate_valid layout.
Empty text, NULL text and padded candidate slots are not interchangeable.

## Scope

This guarantee does not infer hidden labels, prove absence of indirect leakage,
or promise masking for every source/layout. Supported target/input combinations
are validated explicitly. Group equality uses raw typed values, not coincident
unknown vocabulary IDs. Fitted thresholds use the declared fitting split and
remain target state, not learned SampleSpec calculations.

## Golden examples

- [C-TARGET-001](golden/C-TARGET-001.json): probability-one masking returns
  original final dictionary IDs as answers and the mask role as input.
