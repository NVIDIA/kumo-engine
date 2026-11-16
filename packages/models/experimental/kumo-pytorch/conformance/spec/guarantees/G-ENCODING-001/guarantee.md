# G-ENCODING-001: Explicit scalar encodings

## Rule

Each selected field has one explicit Raw, ZScore, Buckets, Vocabulary, or Text
encoding. The same fitted encoding gives the same logical answer through row
and column execution. Vocabulary keys retain typed identity, not printed
spellings. Vocabulary defaults to missing ID 0 and unknown ID 1; with those
defaults ordinary learned IDs begin at 2. Explicit missing/unknown IDs must be
distinct nonnegative int64 values; learned entries use the lowest available
non-reserved IDs. Supplied IDs are explicit and remain exact.

## Boundaries

Learned vocabularies order categories by descending frequency then canonical
typed value for ties; size limits count ordinary entries only. Positive and
negative floating zero identify one category. Numerical encoding rejects
nonfinite observations rather than converting them to strings or NULL.
ZScore uses population mean/scale, ignores NULL while fitting, defaults to
mean 0/scale 1 for an empty population, and uses scale 1 for a constant one.
Buckets equality enters the higher range, with missing ID 0 and ordinary range
IDs starting at 1; explicit cuts are finite and strictly increasing. Learned
quantile cuts use linear interpolation and collapse duplicate cuts. Text emits
piece IDs in order; its declared empty policy distinguishes empty text from
NULL. Text supplied state completely defines its tokenizer.

## Scope

These are local encoder IDs. G-IDSPACE-001 defines any final dictionary
translation. Approximate float64 numerical fitting does not permit approximate
integer IDs, typed category equality, or Raw values. Numerical golden answers
use exact representable examples; any nonzero tolerance must be stated on the
particular array assertion, never inferred from an engine run.

ZScore and Buckets operate on finite binary64 approximations of their numerical
inputs. Wide integers and decimals may round in that explicitly approximate
domain; row and column paths apply the same conversion. This does not change
the exact stored values or Vocabulary keys. Supplied cuts must remain distinct
and increasing after conversion, rather than silently merging indistinguishable
boundaries. Values outside the finite binary64 domain fail with context.

## Golden examples

- [C-ENCODING-001](golden/C-ENCODING-001.json): float categories 1.0/1.5 retain
  distinct supplied IDs; explicit cuts place equality in the higher range.
- [C-ENCODING-002](golden/C-ENCODING-002.json): fitting 1 and 3 gives mean 2,
  scale 1, normalized outputs -1 and 1; NULL is not a fitting observation.
- [C-ENCODING-003](golden/C-ENCODING-003.json): an explicitly supplied BPE
  vocabulary and merge list turn "red bike" into IDs 8 and 11; empty and NULL
  retain distinct presence and piece boundaries.
