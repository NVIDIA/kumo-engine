# G-CALC-001: Named typed calculations

## Rule

Calculations refer to named sampled values. Their result DType is inferred
from the operation and resolved operand types unless explicitly supplied.
Constants, external labels, and MapValues require explicit types. Explicit
result declarations are checked for compatibility with the operation.
Dependencies are evaluated before their consumers regardless of declaration
map order. Operand sequences remain ordered: subtract(a,b) differs from
subtract(b,a). Event calculations use that event's fields; EventTimeDelta uses
the sample anchor minus the event timestamp in its declared unit.

## Boundaries

CalendarPart extracts explicitly selected UTC calendar components. MapValues
has explicit nonoverlapping groups and an unknown-value policy. These are fixed
operations, not user-provided functions or fitted transformations. Unknown
dependencies, cycles, incompatible operands/results, and repeated-valued operands for
scalar-only calculations fail validation. Result declarations do not permit
integer wraparound, truncation of fractional means, or string parsing.
Arithmetic uses Arrow's numeric promotion, including mixed integer/float
operands. Integer sums are exact and checked against their result type; they
never wrap or round. Floating-point rounding and nonfinite overflow results
are preserved;
an output declaration does not recover lost precision. MapValues
normalizes members to their resolved source type and rejects groups that then
overlap. Decimal arithmetic preserves the declared exact decimal domain rather
than accepting an intermediate rounded or widened Arrow result as its answer.
Raw floating distinct-count and mode follow stored bit identity, including
distinct positive/negative zero; this differs from Vocabulary's category-zero
equivalence. Reduction mode retains deterministic type/printed-value ordering
for equal-frequency ties.

## Scope

Sampling computes typed values independently of the batch encoder. Dependency
resolution does not reorder ordered field output. Count/count-distinct return
int64. Signed integer sums return int64 and unsigned integer sums return
uint64; 2^53 plus 1 remains 9007199254740993. Float sums, mean, division,
and population/sample moment calculations return float64.
Existing row/cutoff rules remain in G-ROWS-001. This guarantee does not
introduce learned SampleSpec calculations or cross-event arithmetic.

DiskGraph uses integer accumulation for integer sums and float64 intermediates
for floating sums and moments. Its current aggregation query refuses decimal
operands, although direct typed decimal field reads are supported. Integer
min/max/mode preserve native scalar values; count and count-distinct have
checked int64 results. Expressly approximate float64 means and moments remain
supported.

## Golden examples

- [C-CALC-001](golden/C-CALC-001.json): reverse-declared dependencies and
  reversed subtraction operands produce distinct, hand-calculated results;
  mixed integer/float addition produces a float64 value.
- [C-CALC-002](golden/C-CALC-002.json): integer sums retain values beyond
  float64's exact range; grouped counts sum exactly; floating sums retain
  rounding and overflow behavior.
- [C-CALC-003](golden/C-CALC-003.json): positive and negative floating zero
  count as two distinct raw stored values, independently of category encoding.
