# Typed Timestamps for DiskGraph

## Problem

Timestamps in DiskGraph are raw `i64`. The compiler cannot distinguish an
epoch-seconds timestamp from an epoch-nanoseconds timestamp from a window
offset from a row count. This has caused bugs at two levels.

### Boundary bugs: wrong unit entering the system

The kumo backend generates observation timestamps as pandas nanoseconds
(e.g. `604800000000000` for 1970-01-08). The diskgraph compiler needs
these in seconds. The caller knows they're nanoseconds, but this
information is not passed — instead, a heuristic guesses the unit from
the value's magnitude:

```python
if max_ts > 10**16: return "nanoseconds"   # 17+ digits
if max_ts > 10**13: return "microseconds"  # 14-16 digits
```

`604800000000000` is 15 digits — classified as microseconds. Divided by
10^6 instead of 10^9. The observation timestamp becomes `604800000`
(year 1989) instead of `604800` (1970-01-08). The training table is
silently wrong.

Three independent magnitude-guessing heuristics exist in the codebase.
All are ambiguous for 15-digit values. Each was written independently
because the unit was already lost by the time the code needed it.

A separate boundary bug (`dc3c328`): split time range boundaries were
converted to nanoseconds, then compared against training table timestamps
in seconds. Every row was assigned to the wrong split stage.

### Interior bugs: wrong arithmetic on correctly-converted values

In `ttgen_stats.rs`, split boundary computation:

```rust
boundaries.push(max_ts + *off * 86_400);
```

The `off` values are split offsets. The code assumed they were day-counts
and multiplied by 86400 to get seconds. But if the offsets were already
in seconds (having been converted upstream), the multiplication produces
`604800 * 86400` — nonsense.

Both the timestamp and the offset are `i64`. The code has no way to tell
whether the offset is in days or seconds. This bug is entirely inside
Rust, after the boundary — the values entered the system correctly.

## How timestamps flow through DiskGraph

Three subsystems handle timestamps:

**Ingest** — source data arrives in arbitrary units. Ingest normalizes
to epoch seconds and stores in rowpack. `DatasetMetadata.timestamp_unit`
is configurable (defaults to Seconds, never overridden). Sort key
encoding uses `(value - anchor) / step` in the same unit.

**TT generation** — Python passes observation timestamps to the Rust
executor. The executor reads entity timestamps from rowpack, compares
against observations, computes window bounds, accumulates stats, and
writes output. All arithmetic assumes one unit.

**Sampling** — the SDK sends seed timestamps with a declared
`seed_timestamp_unit`. The engine converts to the index unit at entry.

## Fix 1: Explicit source unit at the boundary

The compiler has a `timestamp_unit` parameter for the **target** unit
(always Seconds). What's missing is the **source** unit — what the
caller's timestamps are actually in. Add `input_timestamp_unit`:

```python
compiler.compile(
    timestamps=nanos_values,
    input_timestamp_unit='nanoseconds',
)
```

The caller declares the source unit. Rust converts once. The heuristic
is eliminated.

This fixes the boundary bugs **when the caller remembers to pass the
parameter**. But `input_timestamp_unit` is optional with a default
(seconds). If a new caller — or an AI coding agent generating SDK
calls — omits it, the raw values pass through as-is, assumed to be
seconds. If they're nanoseconds, the bug reappears silently. The
parameter approach relies on every caller getting it right.

This also does not fix the interior bugs — those involve values that
were already correctly converted.

## Fix 2: Typed wrappers in Rust

The TT-gen pipeline works with two kinds of time values:

- **Absolute timestamps** — observation anchors, entity creation times,
  split range boundaries.
- **Durations** — window offsets, lag offsets, split date offsets.

Today both are `i64`. The executor does things like:

```rust
let window_start = observation_ts + offset_lo;  // i64 + i64
let span = max_ts - min_ts;                     // i64 - i64
```

Both of these make sense. But `i64` also allows:

```rust
let wrong = offset * 86_400;  // already in seconds — was this needed?
let nonsense = observation_ts + other_ts;  // adding two dates?
```

The `* 86_400` bug was the first case. With separate types for
timestamps and durations:

- `Timestamp + Duration = Timestamp` — compiles, correct
- `Timestamp + Timestamp` — does not compile
- `Duration * scalar = Duration` — compiles, but the programmer must
  decide whether it's needed (the constructor `from_days(7)` already
  produces the right seconds value — no manual multiplication)

Construction goes through named methods that declare the source unit:

```rust
impl EpochSeconds {
    pub fn from_secs(secs: i64) -> Self { Self(secs) }
    pub fn from_nanos(ns: i64) -> Self { Self(ns / 1_000_000_000) }
    pub fn from_millis(ms: i64) -> Self { Self(ms / 1_000) }
    pub fn from_days(days: i32) -> Self { Self(days as i64 * 86_400) }
    pub fn raw(self) -> i64 { self.0 }  // only at I/O boundaries
}

impl DurationSeconds {
    pub fn from_secs(secs: i64) -> Self { Self(secs) }
    pub fn from_days(days: i64) -> Self { Self(days * 86_400) }
    pub fn raw(self) -> i64 { self.0 }
}
```

The inner field is private — there is no `EpochSeconds(value)` that
accepts raw `i64`. The only way to construct a typed timestamp is through
a method that names the source unit. The only way to extract the raw
value is `.raw()`, which should only appear at I/O boundaries (Arrow
output, PyO3 serialization, rowpack reads).

This is `#[repr(transparent)]` over `i64` — zero runtime cost, same
memory layout, same machine code after compilation.

### How the bugs are prevented

**Bug 1 (boundary):** The Rust boundary calls
`EpochSeconds::from_nanos(value)`. The constructor name declares the
source unit. No heuristic. If the caller's values are seconds, they call
`from_secs` instead.

**Bug 2 (interior `* 86_400`):** Split offsets are `DurationSeconds`.
`DurationSeconds::from_days(7)` produces 604800 (seconds). The split
boundary computation is `max_ts + offset` where
`max_ts: EpochSeconds` and `offset: DurationSeconds`. No manual
multiplication. The `* 86_400` disappears.

**Bug 3 (boundary — seconds vs nanos):** Split boundaries and training
table timestamps are both `EpochSeconds`. Same type, comparison works.
A nanosecond value cannot be compared against `EpochSeconds` without
going through `from_nanos()` first.

## Concrete vs generic types

The typed wrapper approach requires two types (timestamp + duration).
The question is whether they should be concrete or generic.

### Option A: Concrete newtypes

```rust
pub struct EpochSeconds(i64);
pub struct DurationSeconds(i64);
```

Simple. One unit, no generics. If a second unit is ever needed, either
duplicate the types (`EpochMillis`, `DurationMillis`) or refactor.

### Option B: Generic phantom types

```rust
pub trait UnitSystem: Copy + Send + Sync {
    fn abs_from_secs(secs: i64) -> Abs<Self>;
    fn abs_from_nanos(ns: i64) -> Abs<Self>;
    fn abs_raw(ts: Abs<Self>) -> i64;
    fn dur_from_secs(secs: i64) -> Dur<Self>;
    fn dur_raw(dur: Dur<Self>) -> i64;
}

pub struct Abs<U: UnitSystem>(i64, PhantomData<U>);
pub struct Dur<U: UnitSystem>(i64, PhantomData<U>);

pub struct Seconds;
impl UnitSystem for Seconds { ... }
```

Monomorphization produces identical code to Option A. Adding a new unit =
one marker struct + one impl block + one match arm at the dispatch site.
`DatasetMetadata.timestamp_unit` is already configurable.

### Implementation cost

Options A and B require the same work:
- Type definitions (~30 lines)
- Arithmetic impls (~40 lines, identical structure)
- Serde as transparent i64 (~20 lines)
- Updating plan types, resolved types, executor, stats

Option B adds a trait definition (~15 lines) and uses `<U>` type
parameters instead of concrete names. Call sites are identical:
`Abs::from_secs(v)` vs `EpochSeconds::from_secs(v)`.

## What's practical today

The system uses seconds exclusively. The three subsystems can be
addressed incrementally.

**TT generation:** Explicit `input_timestamp_unit` at the boundary +
typed wrappers in plan types and executor interior. This is the main
change.

**Ingest:** No changes needed. Rowpack normalizes to seconds. The
executor wraps rowpack values with `from_secs()` — identity for
seconds, documents the contract. If a non-seconds unit is needed later,
this call site switches to `from_raw()`.

**Sampling:** No changes needed. The engine already has runtime unit
conversion at entry. Making it type-safe would cascade across 4 crates.
Not justified when the conversion is one contained function.

## What would it take to add milliseconds?

Regardless of which option:

1. Ingest: store millis in rowpack (or convert at read)
2. Sort key metadata: anchor and step in millis
3. Top-level dispatch: one match arm

With Option B (generics), the TT-gen pipeline needs zero changes. With
Option A, the TT-gen pipeline needs new types and a manual audit.

## Open questions

1. Concrete newtypes (Option A) or generics (Option B)?

2. Should rowpack return values in the configured index unit (requires
   ingest changes), or always return seconds with `from_secs()` at the
   call site?

## Related issues

- kumo-ai/kumo#28174 — Custom TT ingestion drops `timestamp_format`
  from feats. `nanosAsLong=True` Spark config makes int64 nanos look
  like micros to the digit-counting heuristic.
