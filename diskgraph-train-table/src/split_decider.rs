//! Split decider trait + impls.
//!
//! A `SplitDecider<U>` is a per-split-family object that observes rows during
//! pass A (`accumulate`), derives cutoffs at end-of-pass-A (`derive_cutoffs`),
//! and assigns one byte per row during pass B (`decide_for_view`).
//!
//! Tier-1 stateless impls: `TimeRangeDecider`, `BacktestingDecider`,
//! `ManualDecider`. Tier-2 deciders: `RandomDecider`, `RandomLinkDecider`,
//! `FixedDecider`, `TemporalDecider`, `DateOffsetDecider`,
//! `ApproxDateOffsetDecider`, `ColumnDecider`.
//!
//! The pass-B per-row API uses a per-batch view: `batch_view(batch,
//! row_offset) -> BatchView` followed by `decide_for_view(view, idx)`.
//! Downcasts and column lookups happen once per batch, not once per row.
//! `required_columns` returns `Vec<String>` so deciders that need
//! user-configured column names (`ColumnDecider` in particular) can return
//! them. Each decider's BatchView resolves its time column index from the
//! batch schema at view-construction time; the downcast cache is reused
//! for every row in the batch.
//!
//! ## Performance contract
//!
//! `accumulate` and `decide_for_view` run on the per-row hot path at 50B-row
//! scale. The trait stays generic-dispatched: `RowProcessor<U, D>` is
//! monomorphized over `D: SplitDecider<U>`, so calls inline. There is no
//! `&dyn`, no `Box<dyn>`, no vtable.
//!
//! ## Send + Sync
//!
//! `SplitDecider<U>: Send + Sync`. Pass A clones one decider per rayon
//! worker and folds them at the end (`Send`). Pass B shares a single
//! `&decider` across rayon workers (one per row group) so the decider must
//! also be `Sync`.
//!
//! ## Streaming vs legacy permutation semantics
//!
//! The legacy pybind one-shot functions in
//! `diskgraph-pybind/src/split_assignments.rs` use a permutation-shuffle
//! model (`StdRng::seed_from_u64(seed).shuffle(perm)` then assign by index).
//! The streaming TT-NG model produces the SAME aggregate ratios but is
//! NOT byte-for-byte equivalent for `RandomDecider` / `RandomLinkDecider`
//! (it uses content-hash bucketing) and is APPROXIMATE for `TemporalDecider`
//! (t-digest quantiles). `FixedDecider`, `DateOffsetDecider`, and
//! `ApproxDateOffsetDecider` are exact within numerical tolerance of the
//! legacy logic. Tests document these divergences.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::ttgen_stats::TTRowRef;
use arrow_array::{Array, Int64Array, RecordBatch, TimestampNanosecondArray};
use diskgraph_core::accumulators::TDigestEstimator;
use diskgraph_core::time::{Abs, Dur, UnitSystem, add_calendar_months};
use diskgraph_core::tt_builder::Value;

/// Sentinel byte for "row not assigned to any stage" — consumed by `pass_b`
/// when writing `split.bin`. Must match the design doc's split.bin alphabet
/// (`0=train`, `1=val`, `2=test`, `255=unused`).
pub(crate) const UNUSED_STAGE: u8 = 255;

/// Per-split-family trait.
///
/// `<U>` is on the trait (not the methods) because deciders store cutoffs in
/// the same `U` they decide rows in.
///
/// `Send + Sync`: pass A clones one decider per rayon worker and folds them
/// at the end (`Send`). Pass B shares a single `&decider` across rayon
/// workers (one per row group) over `Arc<File>`, so the decider must also
/// be `Sync`.
///
/// ## Per-batch view (`BatchView`)
///
/// `BatchView<'a>` is a per-batch downcast cache. Pass B calls
/// `batch_view(&batch, row_offset)` ONCE per `RecordBatch`, then
/// `decide_for_view(&view, idx)` for each row. This hoists the column
/// schema lookup and `as_any().downcast_ref()` out of the per-row loop —
/// at 50B rows that's a measurable difference. The view borrows from
/// `&batch`, so its lifetime is tied to the batch (GAT).
///
/// `row_offset` on `batch_view` is the global row index of the first row
/// in this batch (`= cumulative row count of preceding row groups + rows
/// already consumed in earlier batches of this group`). `FixedDecider`
/// uses it; tier-1 / hash-based / timestamp-based deciders ignore it.
pub(crate) trait SplitDecider<U: UnitSystem>: Send + Sync {
    /// Per-batch downcast cache. Constructed once per `RecordBatch` in
    /// pass B; consumed by `decide_for_view`. The lifetime parameter
    /// borrows from `&'a RecordBatch`.
    type BatchView<'a>: Send
    where
        Self: 'a;

    /// Build the per-batch view. Looks up needed columns by name in the
    /// batch schema and caches typed references. `row_offset` is the
    /// global row index of `batch[0]` (used by `FixedDecider`).
    fn batch_view<'a>(
        &self,
        batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>>;

    /// Decide one row, given the view and row index within the batch.
    /// Hot-path: must be a tight loop body. `#[inline]` is recommended.
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8;

    /// Observe one row during pass A. Most stateless families (TimeRange,
    /// Manual, Backtesting, Random, RandomLink) make this a no-op; stateful
    /// families (Temporal, DateOffset, ApproxDateOffset, Fixed) update
    /// internal accumulators here.
    fn accumulate(&mut self, row: &TTRowRef<'_, U>);

    /// Compute final cutoffs from accumulated state. Called once per job
    /// after all per-worker deciders have been merged. No-op for
    /// stateless families.
    fn derive_cutoffs(&mut self);

    /// Validate cutoffs derived from the accumulated state before pass B
    /// consumes them. Most deciders have no additional post-derivation
    /// invariant; temporal deciders can use this hook to turn an invalid
    /// split into a user-facing TT-gen error instead of silently routing rows
    /// to `UNUSED_STAGE`.
    fn validate_derived_cutoffs(&self) -> crate::errors::TTGenResult<()> {
        Ok(())
    }

    /// Assign one row to a stage byte (`0`, `1`, `2`, or `UNUSED_STAGE`).
    /// In-memory `TTRowRef` form; used by tests and any in-pass-A
    /// computation. `FixedDecider` / `RandomDecider` may have to return
    /// `UNUSED_STAGE` here (they need batch context that `TTRowRef`
    /// doesn't carry).
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8;

    /// Fold another decider's per-worker state into self.
    /// Stateless families: no-op. Stateful: combine accumulators.
    fn merge(&mut self, other: &Self)
    where
        Self: Sized;

    /// Column names this decider needs from `tt.parquet` for pass B.
    /// `Vec<String>` (not `&'static [&'static str]`) because deciders
    /// like `ColumnDecider` carry user-configured column names that
    /// can't be `'static`. `NoSplitDecider` returns `vec![]`.
    fn required_columns(&self) -> Vec<String>;
}

// ─── BatchView helpers ────────────────────────────────────────────────────

/// A BatchView that resolves the `TIMESTAMP` column once and caches the
/// typed reference. Reused by every tier-1 / Temporal / DateOffset /
/// Approx / Random decider that needs a timestamp.
///
/// Stores a clone of the `ArrayRef` (Arrow arrays are `Arc`-backed; cloning
/// is a refcount bump, no data copy). Then re-downcasts on construction
/// and stores both the keep-alive Arc plus the typed pointer-equivalent.
/// We avoid self-referential structs by re-running the downcast inside
/// `ts_secs_at` — but that's a single `as_any()` + ptr-cast per call,
/// not a schema-name lookup, and inlines.
pub(crate) struct TsBatchView<'a> {
    /// Reference to the timestamp column (TimestampNanosecondArray or Int64Array).
    /// We cache it once and re-downcast on access; downcast is a cheap
    /// `Any::is` + ptr cast and the compiler typically hoists it out of
    /// the hot loop.
    ts_col: &'a dyn Array,
    /// Global row index of `batch[0]`. Used by FixedDecider (its view
    /// alias adds this directly; tier-1 / temporal deciders ignore it).
    pub(crate) start_offset: u64,
}

impl<'a> TsBatchView<'a> {
    pub(crate) fn new(batch: &'a RecordBatch, start_offset: u64) -> crate::errors::TTGenResult<Self> {
        let (idx, _) = batch
            .schema_ref()
            .column_with_name("TIMESTAMP")
            .ok_or_else(|| {
                crate::errors::TTGenError::SystemError(
                    "split decider: TIMESTAMP column not found in batch".to_string(),
                )
            })?;
        let ts_col = batch.column(idx).as_ref();
        Ok(Self {
            ts_col,
            start_offset,
        })
    }

    /// Read the timestamp at `row_idx` as epoch seconds.
    #[inline]
    pub(crate) fn ts_secs_at(&self, row_idx: usize) -> i64 {
        if let Some(arr) = self.ts_col.as_any().downcast_ref::<TimestampNanosecondArray>() {
            arr.value(row_idx) / 1_000_000_000
        } else if let Some(arr) = self.ts_col.as_any().downcast_ref::<Int64Array>() {
            arr.value(row_idx)
        } else {
            panic!(
                "TsBatchView: TIMESTAMP column has unexpected dtype {:?}",
                self.ts_col.data_type()
            );
        }
    }

    /// Read the timestamp at `row_idx` as a typed `Abs<U>`.
    #[inline]
    pub(crate) fn ts_at<U: UnitSystem>(&self, row_idx: usize) -> Abs<U> {
        Abs::<U>::from_secs(self.ts_secs_at(row_idx))
    }
}

/// BatchView that doesn't read any column — used by FixedDecider /
/// NoSplitDecider. Holds only the batch's start offset.
pub(crate) struct OffsetOnlyView {
    pub(crate) start_offset: u64,
}

// ─── NoSplitDecider ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NoSplitDecider;

impl<U: UnitSystem> SplitDecider<U> for NoSplitDecider {
    type BatchView<'a> = OffsetOnlyView;

    #[inline]
    fn batch_view<'a>(
        &self,
        _batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        Ok(OffsetOnlyView { start_offset: row_offset })
    }

    #[inline(always)]
    fn decide_for_view(&self, _view: &Self::BatchView<'_>, _row_idx: usize) -> u8 {
        UNUSED_STAGE
    }

    #[inline(always)]
    fn accumulate(&mut self, _row: &TTRowRef<'_, U>) {}

    #[inline(always)]
    fn derive_cutoffs(&mut self) {}

    #[inline(always)]
    fn decide(&self, _row: &TTRowRef<'_, U>) -> u8 {
        UNUSED_STAGE
    }

    #[inline(always)]
    fn merge(&mut self, _other: &Self) {}

    #[inline(always)]
    fn required_columns(&self) -> Vec<String> {
        Vec::new()
    }
}

// ─── TimeRangeDecider ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct TimeRangeDecider<U: UnitSystem> {
    ranges: Vec<(Abs<U>, Abs<U>)>,
    /// `None` is the legacy static-query predicate `(start, end]`. Temporal
    /// queries carry the target window plus anchor offset and use
    /// `[start, end - shift]` instead.
    end_shift: Option<OffsetUnit<U>>,
}

impl<U: UnitSystem> TimeRangeDecider<U> {
    pub(crate) fn new(ranges: Vec<(Abs<U>, Abs<U>)>) -> Self {
        Self {
            ranges,
            end_shift: None,
        }
    }

    pub(crate) fn with_end_shift(mut self, end_shift: Option<OffsetUnit<U>>) -> Self {
        self.end_shift = end_shift;
        self
    }

    #[inline]
    pub(crate) fn decide_ts(&self, ts: Abs<U>) -> u8 {
        for (idx, (start, end)) in self.ranges.iter().enumerate() {
            let in_range = match self.end_shift {
                Some(shift) => ts >= *start && apply_offset(ts, &shift) <= *end,
                None => ts > *start && ts <= *end,
            };
            if in_range {
                return idx as u8;
            }
        }
        UNUSED_STAGE
    }
}

impl<U: UnitSystem> SplitDecider<U> for TimeRangeDecider<U> {
    type BatchView<'a> = TsBatchView<'a>;

    #[inline]
    fn batch_view<'a>(
        &self,
        batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        TsBatchView::new(batch, row_offset)
    }

    #[inline]
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8 {
        self.decide_ts(view.ts_at::<U>(row_idx))
    }

    #[inline(always)]
    fn accumulate(&mut self, _row: &TTRowRef<'_, U>) {}

    #[inline(always)]
    fn derive_cutoffs(&mut self) {}

    #[inline]
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8 {
        self.decide_ts(row.ts)
    }

    #[inline(always)]
    fn merge(&mut self, _other: &Self) {}

    #[inline]
    fn required_columns(&self) -> Vec<String> {
        vec!["TIMESTAMP".to_string()]
    }
}

// ─── BacktestingDecider ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub(crate) struct BacktestingDecider<U: UnitSystem> {
    test_ts: Abs<U>,
}

impl<U: UnitSystem> BacktestingDecider<U> {
    pub(crate) fn new(test_ts: Abs<U>) -> Self {
        Self { test_ts }
    }

    #[inline]
    fn decide_ts(&self, ts: Abs<U>) -> u8 {
        if ts == self.test_ts {
            2
        } else {
            0
        }
    }
}

impl<U: UnitSystem> SplitDecider<U> for BacktestingDecider<U> {
    type BatchView<'a> = TsBatchView<'a>;

    #[inline]
    fn batch_view<'a>(
        &self,
        batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        TsBatchView::new(batch, row_offset)
    }

    #[inline]
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8 {
        self.decide_ts(view.ts_at::<U>(row_idx))
    }

    #[inline(always)]
    fn accumulate(&mut self, _row: &TTRowRef<'_, U>) {}

    #[inline(always)]
    fn derive_cutoffs(&mut self) {}

    #[inline]
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8 {
        self.decide_ts(row.ts)
    }

    #[inline(always)]
    fn merge(&mut self, _other: &Self) {}

    #[inline]
    fn required_columns(&self) -> Vec<String> {
        vec!["TIMESTAMP".to_string()]
    }
}

// ─── ManualDecider ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct ManualDecider<U: UnitSystem> {
    train_values: HashSet<Abs<U>>,
    val_values: HashSet<Abs<U>>,
    test_values: HashSet<Abs<U>>,
}

impl<U: UnitSystem> ManualDecider<U> {
    pub(crate) fn new(
        train_values: HashSet<Abs<U>>,
        val_values: HashSet<Abs<U>>,
        test_values: HashSet<Abs<U>>,
    ) -> Self {
        Self {
            train_values,
            val_values,
            test_values,
        }
    }

    #[inline]
    fn decide_ts(&self, ts: Abs<U>) -> u8 {
        if self.train_values.contains(&ts) {
            0
        } else if self.val_values.contains(&ts) {
            1
        } else if self.test_values.contains(&ts) {
            2
        } else {
            UNUSED_STAGE
        }
    }
}

impl<U: UnitSystem> SplitDecider<U> for ManualDecider<U> {
    type BatchView<'a> = TsBatchView<'a>;

    #[inline]
    fn batch_view<'a>(
        &self,
        batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        TsBatchView::new(batch, row_offset)
    }

    #[inline]
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8 {
        self.decide_ts(view.ts_at::<U>(row_idx))
    }

    #[inline(always)]
    fn accumulate(&mut self, _row: &TTRowRef<'_, U>) {}

    #[inline(always)]
    fn derive_cutoffs(&mut self) {}

    #[inline]
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8 {
        self.decide_ts(row.ts)
    }

    #[inline(always)]
    fn merge(&mut self, _other: &Self) {}

    #[inline]
    fn required_columns(&self) -> Vec<String> {
        vec!["TIMESTAMP".to_string()]
    }
}

// ─── Tier-2 deciders ───────────────────────────────────────────────────────

/// Convert ratio Vec<f64> (len 2 or 3) to cumulative-ratio cutoffs in [0,1).
/// E.g. [0.7, 0.2, 0.1] → cutoffs [0.7, 0.9]; [0.8, 0.2] → cutoffs [0.8].
/// For len-2 ratios there's only one cutoff (train | val); len-3 has two
/// cutoffs (train | val | test).
fn ratio_to_cumulative_cutoffs(ratio: &[f64]) -> Vec<f64> {
    let total: f64 = ratio.iter().sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let mut acc = 0.0;
    let mut out = Vec::with_capacity(ratio.len().saturating_sub(1));
    // All but the last bucket produce a cutoff; the last bucket is "everything
    // above the last cutoff", so we don't emit a cutoff for it.
    for r in ratio.iter().take(ratio.len().saturating_sub(1)) {
        acc += (r / total).max(0.0);
        out.push(acc);
    }
    out
}

/// Bucket index from a fractional value `v ∈ [0,1)` and cumulative cutoffs.
/// E.g. cutoffs=[0.7, 0.9]; v=0.5 → 0, v=0.8 → 1, v=0.95 → 2.
#[inline]
fn bucket_by_cutoffs(v: f64, cutoffs: &[f64]) -> u8 {
    for (i, &c) in cutoffs.iter().enumerate() {
        if v < c {
            return i as u8;
        }
    }
    cutoffs.len() as u8
}

/// Splitmix64 — fast deterministic 64-bit hash mixer. Used by RandomDecider
/// for content-hashing. Not cryptographic; we only need uniformity and
/// determinism.
#[inline]
pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Hash a `Value` into the running 64-bit hash state. Includes a variant
/// discriminator byte so cross-variant collisions don't occur (e.g.
/// `Int(0x6e6f)` and `Str("\x6e\x6f")` would otherwise collide).
///
/// NaN / -0 canonicalization for floats so equivalent values hash equal.
#[inline]
pub(crate) fn hash_value(v: &Value, h: u64) -> u64 {
    match v {
        Value::Int(i) => splitmix64(h ^ (*i as u64) ^ 0x01),
        Value::Float(f) => {
            let bits = if f.is_nan() {
                f64::NAN.to_bits()
            } else if *f == 0.0 {
                0u64
            } else {
                f.to_bits()
            };
            splitmix64(h ^ bits ^ 0x02)
        }
        Value::Str(s) => {
            let mut h = splitmix64(h ^ 0x03);
            for chunk in s.as_bytes().chunks(8) {
                let mut buf = [0u8; 8];
                buf[..chunk.len()].copy_from_slice(chunk);
                h = splitmix64(h ^ u64::from_le_bytes(buf));
            }
            h
        }
        Value::Bool(b) => splitmix64(h ^ (*b as u64) ^ 0x04),
        Value::List(items) => {
            let mut h = splitmix64(h ^ 0x05);
            for item in items {
                h = hash_value(item, h);
            }
            h
        }
    }
}

/// Hash `(entity_pk, ts_raw, seed)` to a uniform double in `[0, 1)`.
///
/// This is the per-row content-hash for `RandomDecider`. Mixing the entity
/// in (not just the timestamp) is required for correctness under timestamp
/// clustering — without it, all rows sharing a timestamp collapse into the
/// same stage, breaking both the per-row randomness and the aggregate
/// ratio contract. See the design notes on `RandomDecider` for details.
#[inline]
pub(crate) fn hash_entity_ts_to_unit_interval(entity: &Value, ts_raw: i64, seed: u64) -> f64 {
    let h0 = splitmix64(seed);
    let h1 = hash_value(entity, h0);
    let h = splitmix64(h1 ^ (ts_raw as u64));
    // Use the top 53 bits to get a uniform double in [0,1).
    (h >> 11) as f64 / (1u64 << 53) as f64
}

// ─── EntityTsBatchView ─────────────────────────────────────────────────────

/// Per-batch view for deciders that hash `(entity_pk, ts)` together.
/// Caches downcasts of both `ENTITY1` and `TIMESTAMP` once per batch; the
/// hot loop only does a typed array lookup per row.
///
/// Reading the entity column constructs a `Value` per row by branching on
/// the underlying Arrow dtype (Int64 → `Value::Int`, Utf8 → `Value::Str`,
/// Float64 → `Value::Float`, Boolean → `Value::Bool`). Customers' primary
/// key columns can in principle be any of these; we cover the common
/// scalar dtypes seen in tt.parquet.
pub(crate) struct EntityTsBatchView<'a> {
    entity_col: &'a dyn Array,
    ts_col: &'a dyn Array,
    pub(crate) start_offset: u64,
}

impl<'a> EntityTsBatchView<'a> {
    pub(crate) fn new(
        batch: &'a RecordBatch,
        start_offset: u64,
    ) -> crate::errors::TTGenResult<Self> {
        let (ts_idx, _) = batch
            .schema_ref()
            .column_with_name("TIMESTAMP")
            .ok_or_else(|| {
                crate::errors::TTGenError::SystemError(
                    "split decider: TIMESTAMP column not found in batch".to_string(),
                )
            })?;
        let (entity_idx, _) =
            batch.schema_ref().column_with_name("ENTITY1").ok_or_else(|| {
                crate::errors::TTGenError::SystemError(
                    "split decider: ENTITY1 column not found in batch".to_string(),
                )
            })?;
        Ok(Self {
            entity_col: batch.column(entity_idx).as_ref(),
            ts_col: batch.column(ts_idx).as_ref(),
            start_offset,
        })
    }

    /// Read the timestamp at `row_idx` as epoch seconds.
    #[inline]
    pub(crate) fn ts_secs_at(&self, row_idx: usize) -> i64 {
        if let Some(arr) = self.ts_col.as_any().downcast_ref::<TimestampNanosecondArray>() {
            arr.value(row_idx) / 1_000_000_000
        } else if let Some(arr) = self.ts_col.as_any().downcast_ref::<Int64Array>() {
            arr.value(row_idx)
        } else {
            panic!(
                "EntityTsBatchView: TIMESTAMP column has unexpected dtype {:?}",
                self.ts_col.data_type()
            );
        }
    }

    /// Read the entity primary key at `row_idx` as a `Value`. Branches by
    /// dtype; supports Int64 / Utf8 / Float64 / Boolean — the common PK
    /// dtypes a customer table uses.
    #[inline]
    pub(crate) fn entity_pk_at(&self, row_idx: usize) -> Value {
        use arrow_array::{BooleanArray, Float64Array, StringArray};
        if let Some(arr) = self.entity_col.as_any().downcast_ref::<Int64Array>() {
            Value::Int(arr.value(row_idx))
        } else if let Some(arr) = self.entity_col.as_any().downcast_ref::<StringArray>() {
            Value::Str(arr.value(row_idx).to_string())
        } else if let Some(arr) = self.entity_col.as_any().downcast_ref::<Float64Array>() {
            Value::Float(arr.value(row_idx))
        } else if let Some(arr) = self.entity_col.as_any().downcast_ref::<BooleanArray>() {
            Value::Bool(arr.value(row_idx))
        } else {
            panic!(
                "EntityTsBatchView: ENTITY1 column has unsupported dtype {:?}",
                self.entity_col.data_type()
            );
        }
    }
}

// ─── RandomDecider ─────────────────────────────────────────────────────────

/// Stateless, content-hash-based random split.
///
/// **Divergence from legacy**: the pybind one-shot `RandomSplit` permutes
/// row indices via `StdRng::seed_from_u64(seed).shuffle(perm)` then assigns
/// by index. That is NOT streaming-friendly. The streaming `RandomDecider`
/// instead hashes content per row to a uniform [0,1) value and buckets by
/// the cumulative ratio cutoffs. It produces the SAME aggregate ratios in
/// expectation but DIFFERENT per-row stage assignments. Users who need
/// byte-for-byte parity with legacy `RandomSplit` should not migrate to
/// the streaming TT-NG path.
///
/// Hash input is `(entity_pk, ts_raw, seed)`. Streaming-friendly content-hash
/// bucketing produces uniform per-row stage assignments even under timestamp
/// clustering — fixing a correctness bug in the ts-only formulation where
/// same-timestamp rows would all land in the same stage.
#[derive(Debug, Clone)]
pub(crate) struct RandomDecider<U: UnitSystem> {
    cutoffs: Vec<f64>,
    seed: u64,
    _marker: std::marker::PhantomData<U>,
}

impl<U: UnitSystem> RandomDecider<U> {
    pub(crate) fn new(ratio: Vec<f64>, seed: u64) -> Self {
        Self {
            cutoffs: ratio_to_cumulative_cutoffs(&ratio),
            seed,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline]
    fn decide_entity_ts(&self, entity: &Value, ts_raw: i64) -> u8 {
        if self.cutoffs.is_empty() {
            return UNUSED_STAGE;
        }
        bucket_by_cutoffs(
            hash_entity_ts_to_unit_interval(entity, ts_raw, self.seed),
            &self.cutoffs,
        )
    }
}

impl<U: UnitSystem> SplitDecider<U> for RandomDecider<U> {
    type BatchView<'a> = EntityTsBatchView<'a>;

    #[inline]
    fn batch_view<'a>(
        &self,
        batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        EntityTsBatchView::new(batch, row_offset)
    }

    #[inline]
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8 {
        let entity = view.entity_pk_at(row_idx);
        let ts_raw = view.ts_secs_at(row_idx);
        self.decide_entity_ts(&entity, ts_raw)
    }

    #[inline(always)]
    fn accumulate(&mut self, _row: &TTRowRef<'_, U>) {}

    #[inline(always)]
    fn derive_cutoffs(&mut self) {}

    #[inline]
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8 {
        self.decide_entity_ts(row.entity_pk, row.ts.raw())
    }

    #[inline(always)]
    fn merge(&mut self, _other: &Self) {}

    #[inline]
    fn required_columns(&self) -> Vec<String> {
        vec!["TIMESTAMP".to_string(), "ENTITY1".to_string()]
    }
}

// ─── RandomLinkDecider ─────────────────────────────────────────────────────

/// Non-array variant of `RandomLinkSplit`. Same semantics as `RandomDecider`
/// for the per-row case; the array case (where the link target is a list
/// and each element gets an independent stage) is handled by the
/// regroup pass and falls back to NoSplitDecider with a warning at
/// the executor's dispatch site.
#[derive(Debug, Clone)]
pub(crate) struct RandomLinkDecider<U: UnitSystem> {
    inner: RandomDecider<U>,
}

impl<U: UnitSystem> RandomLinkDecider<U> {
    pub(crate) fn new(ratio: Vec<f64>, seed: u64) -> Self {
        Self {
            inner: RandomDecider::new(ratio, seed),
        }
    }
}

impl<U: UnitSystem> SplitDecider<U> for RandomLinkDecider<U> {
    type BatchView<'a> = EntityTsBatchView<'a>;

    #[inline]
    fn batch_view<'a>(
        &self,
        batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        <RandomDecider<U> as SplitDecider<U>>::batch_view(&self.inner, batch, row_offset)
    }

    #[inline]
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8 {
        <RandomDecider<U> as SplitDecider<U>>::decide_for_view(&self.inner, view, row_idx)
    }

    #[inline(always)]
    fn accumulate(&mut self, _row: &TTRowRef<'_, U>) {}

    #[inline(always)]
    fn derive_cutoffs(&mut self) {}

    #[inline]
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8 {
        <RandomDecider<U> as SplitDecider<U>>::decide(&self.inner, row)
    }

    #[inline(always)]
    fn merge(&mut self, _other: &Self) {}

    #[inline]
    fn required_columns(&self) -> Vec<String> {
        <RandomDecider<U> as SplitDecider<U>>::required_columns(&self.inner)
    }
}

// ─── FixedDecider ──────────────────────────────────────────────────────────

/// Position-based split: rows are assigned to stages by their global row
/// index. Train fills first, then val, then test. Per-worker accumulator
/// is a row counter; `derive_cutoffs` converts (total_rows, ratio) into
/// `[train_end, val_end]` global row indices.
///
/// **Exact match** with legacy `FixedSplit` (the pybind permutation is
/// `(0..N)`, the identity, so per-index assignment is also positional).
/// Apportionment uses largest-remainder-method to match the pybind logic.
#[derive(Debug, Clone)]
pub(crate) struct FixedDecider<U: UnitSystem> {
    /// Per-worker row counter. Folded via `merge`; finalized by
    /// `derive_cutoffs` into `cutoffs_global`.
    row_count: u64,
    ratio: Vec<f64>,
    /// Set by `derive_cutoffs`. Length 1 (two-way split) or 2 (three-way).
    cutoffs_global: Vec<u64>,
    _marker: std::marker::PhantomData<U>,
}

impl<U: UnitSystem> FixedDecider<U> {
    pub(crate) fn new(ratio: Vec<f64>) -> Self {
        Self {
            row_count: 0,
            ratio,
            cutoffs_global: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Largest-remainder-method apportionment, matching pybind's logic
    /// in `stage_assignments_by_ratio_on_perm`. Produces stage sizes
    /// summing exactly to `total_rows`.
    fn apportion(total_rows: u64, ratio: &[f64]) -> Vec<u64> {
        let mut ratio_vals: Vec<f64> = ratio.iter().take(3).map(|x| x.max(0.0)).collect();
        if ratio_vals.is_empty() {
            return vec![total_rows];
        }
        let total: f64 = ratio_vals.iter().sum();
        if total <= 0.0 {
            return vec![total_rows];
        }
        for x in &mut ratio_vals {
            *x /= total;
        }
        let ideal: Vec<f64> = ratio_vals.iter().map(|x| x * total_rows as f64).collect();
        let mut sizes: Vec<u64> = ideal.iter().map(|v| v.floor() as u64).collect();
        let assigned_floor: u64 = sizes.iter().sum();
        let mut remaining = total_rows.saturating_sub(assigned_floor);

        let mut order: Vec<usize> = (0..sizes.len()).collect();
        order.sort_by(|&a, &b| {
            let ra = ideal[a] - (sizes[a] as f64);
            let rb = ideal[b] - (sizes[b] as f64);
            rb.partial_cmp(&ra)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });

        let mut k = 0usize;
        while remaining > 0 && k < order.len() {
            sizes[order[k]] += 1;
            remaining -= 1;
            k += 1;
            if k == order.len() {
                k = 0;
            }
        }
        sizes
    }

    #[inline]
    fn decide_global(&self, global_idx: u64) -> u8 {
        for (i, &cutoff) in self.cutoffs_global.iter().enumerate() {
            if global_idx < cutoff {
                return i as u8;
            }
        }
        self.cutoffs_global.len() as u8
    }
}

impl<U: UnitSystem> SplitDecider<U> for FixedDecider<U> {
    type BatchView<'a> = OffsetOnlyView;

    #[inline]
    fn batch_view<'a>(
        &self,
        _batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        Ok(OffsetOnlyView { start_offset: row_offset })
    }

    #[inline]
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8 {
        self.decide_global(view.start_offset + row_idx as u64)
    }

    #[inline]
    fn accumulate(&mut self, _row: &TTRowRef<'_, U>) {
        self.row_count += 1;
    }

    fn derive_cutoffs(&mut self) {
        let sizes = Self::apportion(self.row_count, &self.ratio);
        // cutoffs are cumulative sums excluding the final stage.
        let mut acc = 0u64;
        let mut out = Vec::with_capacity(sizes.len().saturating_sub(1));
        for s in sizes.iter().take(sizes.len().saturating_sub(1)) {
            acc += s;
            out.push(acc);
        }
        self.cutoffs_global = out;
    }

    #[inline]
    fn decide(&self, _row: &TTRowRef<'_, U>) -> u8 {
        // FixedDecider needs the global row index, not row content.
        // Pass A's per-row path doesn't compute global indices, so this
        // is unused on the pass-A side. Pass B uses `decide_for_view`.
        UNUSED_STAGE
    }

    fn merge(&mut self, other: &Self) {
        self.row_count += other.row_count;
    }

    #[inline]
    fn required_columns(&self) -> Vec<String> {
        // Position-based — no column needed. Pass B passes
        // `OffsetOnlyView`, not a real column projection.
        Vec::new()
    }
}

// ─── TemporalDecider ───────────────────────────────────────────────────────

/// Quantile-based temporal split. Per-worker accumulates a t-digest of
/// timestamp raw values; `derive_cutoffs` queries quantiles at the
/// cumulative ratio thresholds.
///
/// **Approximate vs legacy**: the pybind one-shot `TemporalSplit` sorts
/// all timestamps and assigns by index. The streaming TT-NG version
/// uses t-digest quantile estimates (200 centroids by default), which
/// have well below 1% absolute error at the median when queried directly
/// via `TDigestEstimator::quantile_at`. For small fixtures the
/// stage counts may differ by a few rows from the legacy expectation.
#[derive(Debug, Clone)]
pub(crate) struct TemporalDecider<U: UnitSystem> {
    ratio: Vec<f64>,
    digest: TDigestEstimator,
    /// Set by `derive_cutoffs`. Length 1 or 2.
    cutoffs: Vec<Abs<U>>,
}

impl<U: UnitSystem> TemporalDecider<U> {
    pub(crate) fn new(ratio: Vec<f64>) -> Self {
        Self {
            ratio,
            digest: TDigestEstimator::new(200),
            cutoffs: Vec::new(),
        }
    }

    #[inline]
    fn decide_ts(&self, ts: Abs<U>) -> u8 {
        for (i, c) in self.cutoffs.iter().enumerate() {
            if ts < *c {
                return i as u8;
            }
        }
        self.cutoffs.len() as u8
    }
}

impl<U: UnitSystem> SplitDecider<U> for TemporalDecider<U> {
    type BatchView<'a> = TsBatchView<'a>;

    #[inline]
    fn batch_view<'a>(
        &self,
        batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        TsBatchView::new(batch, row_offset)
    }

    #[inline]
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8 {
        self.decide_ts(view.ts_at::<U>(row_idx))
    }

    #[inline]
    fn accumulate(&mut self, row: &TTRowRef<'_, U>) {
        self.digest.observe(row.ts.raw() as f64);
    }

    fn derive_cutoffs(&mut self) {
        let cum = ratio_to_cumulative_cutoffs(&self.ratio);
        if cum.is_empty() {
            return;
        }
        let mut out = Vec::with_capacity(cum.len());
        for &q in &cum {
            // Direct `quantile_at` query. Replaces the legacy
            // `quantile_via_public_api` workaround that interpolated between
            // q25/q50/q75 (~1-2% absolute error). `quantile_at` flushes the
            // internal buffer and queries at exactly `q`.
            let q_val = self.digest.quantile_at(q);
            out.push(Abs::<U>::from_raw(q_val.round() as i64));
        }
        self.cutoffs = out;
    }

    #[inline]
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8 {
        self.decide_ts(row.ts)
    }

    fn merge(&mut self, other: &Self) {
        // TDigestEstimator::merge takes `&mut other`; we take `&Self`.
        // Clone the other's digest to satisfy the API (cheap — centroids
        // are bounded by `delta`).
        let mut other_clone = other.digest.clone();
        self.digest.merge(&mut other_clone);
    }

    #[inline]
    fn required_columns(&self) -> Vec<String> {
        vec!["TIMESTAMP".to_string()]
    }
}

// ─── DateOffsetDecider ─────────────────────────────────────────────────────

/// One offset entry in a `DateOffsetDecider`. Variants:
/// - `Fixed`: a typed [`Dur<U>`] (seconds/days/weeks/...). Pure additive
///   arithmetic; matches tier-2 semantics.
/// - `Months`: a calendar-month offset (signed). Applied via
///   [`add_calendar_months`] from `diskgraph-core`. Honors end-of-month
///   clamping (e.g. Jan 31 + 1 month = Feb 28/29).
/// - `Years`: a calendar-year offset (signed). Equivalent to
///   `Months(years * 12)` — kept distinct for plan readability.
///
/// Calendar variants exist because months/years are not fixed-duration
/// quantities; they require a date anchor. Mixing variants in one
/// `Vec<OffsetUnit<U>>` is permitted (each is applied to `max_ts`
/// independently).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OffsetUnit<U: UnitSystem> {
    Fixed(Dur<U>),
    Months(i32),
    Years(i32),
}

impl<U: UnitSystem> OffsetUnit<U> {
    /// Returns true if this is a fixed-duration offset (no calendar arithmetic).
    pub(crate) fn is_fixed(&self) -> bool {
        matches!(self, OffsetUnit::Fixed(_))
    }

    /// If both offsets are `Fixed`, return their typed `Dur<U>` difference.
    /// Otherwise compute the difference by anchoring at `max_ts`:
    ///   `(max_ts + self) - (max_ts + other)`
    /// which yields a fixed `Dur<U>` valid for the specific anchor.
    fn difference_at(&self, other: &Self, anchor: Abs<U>) -> Dur<U> {
        match (self, other) {
            (OffsetUnit::Fixed(a), OffsetUnit::Fixed(b)) => *a - *b,
            _ => apply_offset(anchor, self) - apply_offset(anchor, other),
        }
    }
}

/// Apply an `OffsetUnit` to an anchor timestamp.
///
/// - For `Fixed(d)` returns `anchor + d`.
/// - For `Months(m)` / `Years(y)` calls [`add_calendar_months`].
///
/// **End-of-month convention**: months arithmetic clamps the day-of-month
/// to the last valid day of the target month. So `2024-03-31 + 1 month`
/// = `2024-04-30` (not `2024-05-01`), and `2024-01-31 - 1 month` (i.e.
/// `+ Months(-1)`) = `2023-12-31` (no clamp needed). This matches
/// `pd.DateOffset(months=N)` and the existing `add_calendar_months`
/// semantics already shared with the temporal-aggregation window code.
pub(crate) fn apply_offset<U: UnitSystem>(anchor: Abs<U>, offset: &OffsetUnit<U>) -> Abs<U> {
    match offset {
        OffsetUnit::Fixed(d) => anchor + *d,
        OffsetUnit::Months(m) => add_calendar_months(anchor, *m),
        OffsetUnit::Years(y) => add_calendar_months(anchor, y.saturating_mul(12)),
    }
}

/// Subtract a calendar-aware offset from an anchor.
///
/// `subtract_calendar_offset(anchor, OffsetUnit::Months(m))` =
/// `add_calendar_months(anchor, -m)`. For `Fixed(d)` this is `anchor - d`.
pub(crate) fn subtract_calendar_offset<U: UnitSystem>(
    anchor: Abs<U>,
    offset: OffsetUnit<U>,
) -> Abs<U> {
    match offset {
        OffsetUnit::Fixed(d) => anchor - d,
        OffsetUnit::Months(m) => add_calendar_months(anchor, -m),
        OffsetUnit::Years(y) => add_calendar_months(anchor, -y.saturating_mul(12)),
    }
}

/// Port of kumo `BaseTemporalSplit.adjust_boundaries` (split.py:298-365) —
/// the `flexible_boundaries=True` (production-default) boundary snapping.
///
/// `boundaries` is the flat point list `[min, max+off0, .., max+offk,
/// max+tf+anchor]`; consecutive points are the (train, val, test) windows (each
/// window's finish IS the next's start). kumo snaps each window start onto the
/// grid `(max+tf+anchor) - tf - k*step` (newest-first), then restores `min`.
/// In flat form only the interior cuts `boundaries[1..N-1]` move: point 0 stays
/// `min`, the last is fixed, and kumo's `cum_offset` finish-clamp is a no-op
/// (a window's finish is its successor's already-snapped start). The grid cursor
/// `step_offset` is monotonic and shared across windows, matching kumo exactly.
pub(crate) fn flex_adjust_boundaries<U: UnitSystem>(
    boundaries: &mut [Abs<U>],
    tf: Dur<U>,
    anchor: Dur<U>,
    timeframe_step: Option<Dur<U>>,
) {
    let n = boundaries.len();
    if n < 2 {
        return;
    }
    let step = timeframe_step.unwrap_or(tf);
    let mut step_offset = boundaries[n - 1] - tf;
    for i in (1..n - 1).rev() {
        let mut start = boundaries[i];
        let finish = boundaries[i + 1];
        if start + anchor + tf > finish {
            start = finish - tf - anchor;
        }
        while step_offset - anchor > start {
            step_offset = step_offset - step;
        }
        boundaries[i] = step_offset - anchor;
    }
}

/// Validate the Kumo `flexible_boundaries` postcondition for DateOffsetSplit.
///
/// Boundary point zero is the observed minimum timestamp and point one is the
/// snapped end of the training window. A train seed at `ts` is valid only when
/// `ts + timeframe + anchor <= train_end`; therefore the first window must
/// itself hold at least that shift. Legacy Kumo rejects this configuration
/// after snapping rather than publishing an empty training split.
fn validate_flexible_train_window<U: UnitSystem>(
    boundaries: &[Abs<U>],
    shift: Dur<U>,
) -> crate::errors::TTGenResult<()> {
    if boundaries.len() >= 2 && boundaries[0] + shift > boundaries[1] {
        return Err(crate::errors::TTGenError::InvalidPlan(
            "Not enough data for the split - the training set will end up empty. \
             Provide a longer history of data or reduce the size of the window."
                .to_string(),
        ));
    }
    Ok(())
}

/// Date-offset split: cutoffs are computed from `max_ts` minus a sequence
/// of offsets.
///
/// **Exact** within typed `Abs<U>`/`Dur<U>` arithmetic for `OffsetUnit::Fixed`
/// entries, and exact-up-to-end-of-month-clamp for `Months`/`Years` entries.
/// Per-worker accumulates `min_ts`/`max_ts`; `derive_cutoffs` produces
/// `boundaries` = `[min_ts, max_ts + offsets[0], ..., max_ts + offsets[k],
/// max_ts + timeframe + anchor]`. Pass-A/Pass-B per-row decision matches the
/// legacy `assign_date_offset_split`: a row goes to stage `i` if
/// `ts >= start_i` and `ts + timeframe + anchor <= end_i`.
///
/// Calendar-month and calendar-year offsets are supported via
/// [`OffsetUnit::Months`] / [`OffsetUnit::Years`]. The `timeframe` and
/// `anchor_start_offset` remain typed `Dur<U>` (callers must convert if they
/// need a calendar tf — typically tf=0 / anchor=0 for calendar-month splits).
/// When `timeframe` is `None` the default is derived from the offsets via
/// [`OffsetUnit::difference_at`] anchored at `max_ts`.
#[derive(Debug, Clone)]
pub(crate) struct DateOffsetDecider<U: UnitSystem> {
    /// Construction-time offsets, len 2 or 3.
    offsets: Vec<OffsetUnit<U>>,
    anchor_start_offset: Dur<U>,
    /// `timeframe` parameter; if `None`, defaults are derived from offsets.
    timeframe: Option<Dur<U>>,
    /// Per-worker min/max accumulator.
    min_ts: Option<Abs<U>>,
    max_ts: Option<Abs<U>>,
    /// Set by `derive_cutoffs`. `boundaries.len() == offsets.len() + 2`.
    boundaries: Vec<Abs<U>>,
    /// Cached `timeframe + anchor_start_offset` shift for per-row decide.
    shift: Dur<U>,
    /// kumo `flexible_boundaries` (prod default `true`). When `true`,
    /// `derive_cutoffs` snaps split starts to the sampling grid so each split
    /// holds ≥1 full timeframe; when `false`, raw offset boundaries are used.
    flexible_boundaries: bool,
    /// kumo `timeframe_step`; sampling-grid spacing for flexible snapping.
    /// `None` ⇒ grid spacing == `timeframe`.
    timeframe_step: Option<Dur<U>>,
}

impl<U: UnitSystem> DateOffsetDecider<U> {
    /// Construct from typed `Dur<U>` offsets (legacy fixed-duration path).
    /// Maps each `Dur<U>` to `OffsetUnit::Fixed(d)`. Behavior unchanged from
    /// legacy (pre-calendar-month-offset).
    pub(crate) fn new(
        offsets: Vec<Dur<U>>,
        anchor_start_offset: Dur<U>,
        timeframe: Option<Dur<U>>,
    ) -> Self {
        Self::new_with_units(
            offsets.into_iter().map(OffsetUnit::Fixed).collect(),
            anchor_start_offset,
            timeframe,
        )
    }

    /// Construct from `OffsetUnit<U>` offsets, supporting calendar units.
    pub(crate) fn new_with_units(
        offsets: Vec<OffsetUnit<U>>,
        anchor_start_offset: Dur<U>,
        timeframe: Option<Dur<U>>,
    ) -> Self {
        Self {
            offsets,
            anchor_start_offset,
            timeframe,
            min_ts: None,
            max_ts: None,
            boundaries: Vec::new(),
            shift: Dur::<U>::zero(),
            flexible_boundaries: false,
            timeframe_step: None,
        }
    }

    /// Enable kumo `flexible_boundaries` snapping with an optional
    /// `timeframe_step` grid spacing. Builder-style so the existing
    /// constructors keep their rigid (`flexible=false`) default.
    pub(crate) fn with_flexible_boundaries(
        mut self,
        flexible: bool,
        timeframe_step: Option<Dur<U>>,
    ) -> Self {
        self.flexible_boundaries = flexible;
        self.timeframe_step = timeframe_step;
        self
    }

    #[inline]
    fn decide_ts(&self, ts: Abs<U>) -> u8 {
        if self.boundaries.len() < 2 {
            return UNUSED_STAGE;
        }
        let shifted = ts + self.shift;
        let mut out = UNUSED_STAGE;
        // Match pybind: scan all (start, end) pairs; LAST match wins.
        for (split_idx, w) in self.boundaries.windows(2).enumerate() {
            let start = w[0];
            let end = w[1];
            let stage_id = split_idx.min(2) as u8;
            if ts >= start && shifted <= end {
                out = stage_id;
            }
        }
        out
    }
}

impl<U: UnitSystem> SplitDecider<U> for DateOffsetDecider<U> {
    type BatchView<'a> = TsBatchView<'a>;

    #[inline]
    fn batch_view<'a>(
        &self,
        batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        TsBatchView::new(batch, row_offset)
    }

    #[inline]
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8 {
        self.decide_ts(view.ts_at::<U>(row_idx))
    }

    #[inline]
    fn accumulate(&mut self, row: &TTRowRef<'_, U>) {
        self.min_ts = Some(match self.min_ts {
            Some(cur) => cur.min(row.ts),
            None => row.ts,
        });
        self.max_ts = Some(match self.max_ts {
            Some(cur) => cur.max(row.ts),
            None => row.ts,
        });
    }

    fn derive_cutoffs(&mut self) {
        let (min_ts, max_ts) = match (self.min_ts, self.max_ts) {
            (Some(a), Some(b)) => (a, b),
            _ => return,
        };
        // Compute timeframe with default-derivation matching pybind. For
        // calendar offsets the difference is anchored at `max_ts` (yields a
        // fixed Dur<U> appropriate for the per-row shift used by decide_ts).
        let tf: Dur<U> = match self.timeframe {
            Some(t) => t,
            None => {
                if self.offsets.len() == 2 {
                    self.offsets[1].difference_at(&self.offsets[0], max_ts)
                } else if !self.offsets.is_empty() {
                    Dur::<U>::zero() - self.offsets[0]
                        .difference_at(&OffsetUnit::Fixed(Dur::<U>::zero()), max_ts)
                } else {
                    Dur::<U>::zero()
                }
            }
        };
        self.shift = tf + self.anchor_start_offset;

        let mut boundaries: Vec<Abs<U>> = Vec::with_capacity(self.offsets.len() + 2);
        boundaries.push(min_ts);
        for off in &self.offsets {
            boundaries.push(apply_offset(max_ts, off));
        }
        boundaries.push(max_ts + tf + self.anchor_start_offset);
        if self.flexible_boundaries {
            flex_adjust_boundaries(&mut boundaries, tf, self.anchor_start_offset, self.timeframe_step);
        }
        self.boundaries = boundaries;
    }

    fn validate_derived_cutoffs(&self) -> crate::errors::TTGenResult<()> {
        // Match Kumo exactly: the final empty-train guard is part of
        // `flexible_boundaries`; rigid DateOffsetSplit preserves its legacy
        // raw-boundary behavior.
        if self.flexible_boundaries {
            validate_flexible_train_window(&self.boundaries, self.shift)?;
        }
        Ok(())
    }

    #[inline]
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8 {
        self.decide_ts(row.ts)
    }

    fn merge(&mut self, other: &Self) {
        self.min_ts = match (self.min_ts, other.min_ts) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        self.max_ts = match (self.max_ts, other.max_ts) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
    }

    #[inline]
    fn required_columns(&self) -> Vec<String> {
        vec!["TIMESTAMP".to_string()]
    }
}

// ─── ApproxDateOffsetDecider ───────────────────────────────────────────────

/// Approximate-date-offset split: groups rows by unique timestamp, sorts
/// timestamps, and finds cutoff timestamps where the cumulative row count
/// crosses each ratio threshold. Matches pybind `assign_approx_date_offset_split`
/// (lines 231-308). Per-worker accumulates a `HashMap<Abs<U>, u64>` count;
/// `derive_cutoffs` finalizes a `HashMap<Abs<U>, u8>` stage map.
///
/// **Exact** match with legacy when run on the same row set. The streaming
/// model rebuilds the same per-timestamp counts at merge time.
#[derive(Debug, Clone)]
pub(crate) struct ApproxDateOffsetDecider<U: UnitSystem> {
    ratio: Vec<f64>,
    /// Per-worker per-timestamp count.
    counts: HashMap<Abs<U>, u64>,
    /// Set by `derive_cutoffs`. Maps each unique timestamp to its stage.
    /// Per-row decide is a HashMap lookup (O(1) avg).
    stage_by_ts: HashMap<Abs<U>, u8>,
}

impl<U: UnitSystem> ApproxDateOffsetDecider<U> {
    pub(crate) fn new(ratio: Vec<f64>) -> Self {
        Self {
            ratio,
            counts: HashMap::new(),
            stage_by_ts: HashMap::new(),
        }
    }

    #[inline]
    fn decide_ts(&self, ts: Abs<U>) -> u8 {
        *self.stage_by_ts.get(&ts).unwrap_or(&UNUSED_STAGE)
    }
}

impl<U: UnitSystem> SplitDecider<U> for ApproxDateOffsetDecider<U> {
    type BatchView<'a> = TsBatchView<'a>;

    #[inline]
    fn batch_view<'a>(
        &self,
        batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        TsBatchView::new(batch, row_offset)
    }

    #[inline]
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8 {
        self.decide_ts(view.ts_at::<U>(row_idx))
    }

    #[inline]
    fn accumulate(&mut self, row: &TTRowRef<'_, U>) {
        *self.counts.entry(row.ts).or_insert(0) += 1;
    }

    fn derive_cutoffs(&mut self) {
        let num_split = self.ratio.len();
        if !(2..=3).contains(&num_split) {
            return;
        }
        if self.counts.is_empty() {
            return;
        }
        // Sort unique timestamps ascending.
        let mut groups: Vec<(Abs<U>, u64)> =
            self.counts.iter().map(|(k, v)| (*k, *v)).collect();
        groups.sort_by_key(|x| x.0);

        let total_rows: u64 = groups.iter().map(|(_, c)| *c).sum();
        if total_rows == 0 {
            return;
        }
        // Cumulative ratio per group.
        let mut cum: Vec<f64> = Vec::with_capacity(groups.len());
        let mut run = 0u64;
        for (_, c) in &groups {
            run += *c;
            cum.push(run as f64 / total_rows as f64);
        }

        let n = groups.len();
        let mut train_val_idx = cum
            .iter()
            .position(|v| *v >= self.ratio[0])
            .unwrap_or(n.saturating_sub(1));
        train_val_idx = train_val_idx.min(n.saturating_sub(num_split));

        let mut val_test_idx: Option<usize> = None;
        if num_split == 3 {
            let mut idx = cum
                .iter()
                .position(|v| *v >= (self.ratio[0] + self.ratio[1]))
                .unwrap_or(n.saturating_sub(1));
            idx = idx.min(n.saturating_sub(2));
            idx = idx.max(train_val_idx.saturating_add(1));
            val_test_idx = Some(idx);
        }

        let mut stage_by_ts: HashMap<Abs<U>, u8> = HashMap::with_capacity(n);
        for (i, (ts, _)) in groups.iter().enumerate() {
            let stage = if i <= train_val_idx {
                0u8
            } else if num_split == 2 {
                1u8
            } else {
                let vt = val_test_idx.unwrap();
                if i <= vt {
                    1u8
                } else {
                    2u8
                }
            };
            stage_by_ts.insert(*ts, stage);
        }
        self.stage_by_ts = stage_by_ts;
    }

    #[inline]
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8 {
        self.decide_ts(row.ts)
    }

    fn merge(&mut self, other: &Self) {
        for (k, v) in &other.counts {
            *self.counts.entry(*k).or_insert(0) += *v;
        }
    }

    #[inline]
    fn required_columns(&self) -> Vec<String> {
        vec!["TIMESTAMP".to_string()]
    }
}

// ─── ColumnDecider ─────────────────────────────────────────────────────────
//
// Three sub-cases of "ColumnSplit", each backed by a concrete decider type
// (no enum dispatch — keeps the per-row hot path identical to the other
// tier deciders, since `RowProcessor<U, D>` is monomorphized over D):
//
//   (α) `ColumnDeciderManual`         — explicit train/val/test value sets
//       on a string-valued split column. Stateless. Per-row decide is a
//       single HashSet<String> lookup. Cutoffs come from the plan
//       (manual_train_values / manual_val_values / manual_test_values).
//
//   (β) `ColumnDeciderNumericExplicit`  — explicit numeric cutoffs on a
//       numeric (Int64 / Float64) split column. Stateless. Per-row decide
//       buckets `value` against the configured cutoffs. Construction is
//       exposed for unit tests; the SplitSpec input schema does not yet
//       carry a "cutoffs" field, so this variant is not currently
//       reachable through `execute_tt_pass`'s dispatch (documented).
//
//   (γ) `ColumnDeciderRatioNumeric`     — ratio-derived cutoffs on a
//       numeric split column. Pass A accumulates a t-digest of the
//       split column values; `derive_cutoffs` picks quantiles at the
//       cumulative ratio thresholds. Per-row decide buckets `value`
//       against those derived cutoffs. Per-worker merge folds two
//       t-digests via `TDigestEstimator::merge`.
//
// (δ) Reservoir-sampling-based string-ratio is **scoped out**.
//     Reservoir sampling is more invasive than t-digest merge (rayon
//     workers each carry an independent reservoir; merging two reservoirs
//     to maintain a uniform sample requires a Reservoir merge primitive
//     that doesn't exist in `diskgraph_core::accumulators`). See the
//     `Err(InvalidPlan)` branch in `execute_tt_pass`'s ColumnSplit arm
//     for the explicit deferral message.
//
// Pass B BatchView for (α) and (γ): a typed downcast of the configured
// column from the `RecordBatch` schema. The view caches the column index
// + downcast pointer once per batch, identical pattern to `TsBatchView`.

/// BatchView for `ColumnDeciderManual` — caches a downcast `StringArray`
/// reference for the configured column. Constructed once per `RecordBatch`
/// in pass B; consumed by `decide_for_view`.
pub(crate) struct StringColumnBatchView<'a> {
    col: &'a arrow_array::StringArray,
    pub(crate) start_offset: u64,
}

impl<'a> StringColumnBatchView<'a> {
    fn new(
        batch: &'a RecordBatch,
        column_name: &str,
        start_offset: u64,
    ) -> crate::errors::TTGenResult<Self> {
        let (idx, _) = batch
            .schema_ref()
            .column_with_name(column_name)
            .ok_or_else(|| {
                crate::errors::TTGenError::SystemError(format!(
                    "ColumnDecider: column '{column_name}' not found in batch schema"
                ))
            })?;
        let col = batch.column(idx).as_any().downcast_ref::<arrow_array::StringArray>().ok_or_else(|| {
            crate::errors::TTGenError::SystemError(format!(
                "ColumnDecider: column '{column_name}' is not a Utf8 StringArray (dtype={:?})",
                batch.column(idx).data_type()
            ))
        })?;
        Ok(Self { col, start_offset })
    }

    #[inline]
    fn value_at(&self, row_idx: usize) -> Option<&str> {
        if self.col.is_null(row_idx) {
            None
        } else {
            Some(self.col.value(row_idx))
        }
    }
}

/// BatchView for numeric-column deciders. Caches an Int64 / Float64 /
/// Utf8 downcast and yields f64 per row. Utf8 support exists because the
/// executor's `split_projection` path writes the projected split column
/// to `tt.parquet` as Utf8 today (see `resolve_split_projection`); pass B
/// reads it back and we parse the string to f64 per row. Non-parseable
/// strings yield `None` (treated as UNUSED at decide time).
pub(crate) struct NumericColumnBatchView<'a> {
    col: &'a dyn Array,
    pub(crate) start_offset: u64,
}

impl<'a> NumericColumnBatchView<'a> {
    fn new(
        batch: &'a RecordBatch,
        column_name: &str,
        start_offset: u64,
    ) -> crate::errors::TTGenResult<Self> {
        let (idx, _) = batch
            .schema_ref()
            .column_with_name(column_name)
            .ok_or_else(|| {
                crate::errors::TTGenError::SystemError(format!(
                    "ColumnDecider: column '{column_name}' not found in batch schema"
                ))
            })?;
        let col = batch.column(idx).as_ref();
        // Validate dtype here so we don't surprise per-row callers.
        let any = col.as_any();
        if any.downcast_ref::<Int64Array>().is_none()
            && any.downcast_ref::<arrow_array::Float64Array>().is_none()
            && any.downcast_ref::<arrow_array::StringArray>().is_none()
        {
            return Err(crate::errors::TTGenError::SystemError(format!(
                "ColumnDecider numeric: column '{column_name}' is not Int64/Float64/Utf8 (dtype={:?})",
                col.data_type()
            )));
        }
        Ok(Self { col, start_offset })
    }

    #[inline]
    fn value_at(&self, row_idx: usize) -> Option<f64> {
        let any = self.col.as_any();
        if let Some(arr) = any.downcast_ref::<Int64Array>() {
            if arr.is_null(row_idx) {
                None
            } else {
                Some(arr.value(row_idx) as f64)
            }
        } else if let Some(arr) = any.downcast_ref::<arrow_array::Float64Array>() {
            if arr.is_null(row_idx) {
                None
            } else {
                Some(arr.value(row_idx))
            }
        } else if let Some(arr) = any.downcast_ref::<arrow_array::StringArray>() {
            // Utf8 path: tt.parquet's split projection column is written
            // as Utf8 today. Parse the string to f64; non-parseable
            // (e.g. "None" sentinel for null projections) → None.
            if arr.is_null(row_idx) {
                None
            } else {
                arr.value(row_idx).parse::<f64>().ok()
            }
        } else {
            // Validated at construction time; this is unreachable.
            None
        }
    }
}

/// Coerce a `Value` (rowpack-decoded scalar) to a string for set-membership
/// comparison. Used by `ColumnDeciderManual::decide` (the in-pass-A path)
/// and by tests. Behavior:
///   - `Value::Str(s)` → `s.clone()`
///   - `Value::Int(i)` → `i.to_string()`
///   - `Value::Float(f)` → `f.to_string()` (matches Rust's `Display`)
///   - `Value::Bool(b)` → `"true"` / `"false"` (matches Rust's `Display`)
///   - `Value::List(_)` → empty string (lists aren't valid split-column
///     values; treated as no-match)
fn value_to_string_key(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::List(_) => String::new(),
    }
}

/// Coerce a `Value` to f64 for numeric ColumnDecider variants. `Some` for
/// Int / Float; `None` for everything else (treated as a no-op /
/// UNUSED row).
#[inline]
fn value_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Int(i) => Some(*i as f64),
        Value::Float(f) => Some(*f),
        _ => None,
    }
}

// ─── ColumnDeciderManual (α) ─────────────────────────────────────────────

/// Explicit train/val/test value sets on a (typically string-valued) split
/// column. Stateless: pass A `accumulate` is a no-op; `decide` and
/// `decide_for_view` perform a single HashSet lookup against the
/// configured sets.
///
/// Cutoffs come from the plan: `manual_train_values`, `manual_val_values`,
/// `manual_test_values` are passed in as `Vec<String>` from the plan and
/// converted to `HashSet<String>` here.
///
/// Mirrors today's `ManualDecider` (which operates on `Abs<U>` timestamps)
/// but on an arbitrary user-configured column. `required_columns()` returns
/// `vec![column_name]`.
#[derive(Debug, Clone)]
pub(crate) struct ColumnDeciderManual {
    column_name: String,
    train_set: HashSet<String>,
    val_set: HashSet<String>,
    test_set: HashSet<String>,
}

impl ColumnDeciderManual {
    pub(crate) fn new(
        column_name: String,
        train_values: Vec<String>,
        val_values: Vec<String>,
        test_values: Vec<String>,
    ) -> Self {
        Self {
            column_name,
            train_set: train_values.into_iter().collect(),
            val_set: val_values.into_iter().collect(),
            test_set: test_values.into_iter().collect(),
        }
    }

    #[inline]
    fn decide_str(&self, s: &str) -> u8 {
        if self.train_set.contains(s) {
            0
        } else if self.val_set.contains(s) {
            1
        } else if self.test_set.contains(s) {
            2
        } else {
            UNUSED_STAGE
        }
    }
}

impl<U: UnitSystem> SplitDecider<U> for ColumnDeciderManual {
    type BatchView<'a> = StringColumnBatchView<'a>;

    #[inline]
    fn batch_view<'a>(
        &self,
        batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        StringColumnBatchView::new(batch, &self.column_name, row_offset)
    }

    #[inline]
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8 {
        match view.value_at(row_idx) {
            Some(s) => self.decide_str(s),
            None => UNUSED_STAGE,
        }
    }

    #[inline(always)]
    fn accumulate(&mut self, _row: &TTRowRef<'_, U>) {}

    #[inline(always)]
    fn derive_cutoffs(&mut self) {}

    #[inline]
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8 {
        let Some(v) = row.split_column_value else {
            return UNUSED_STAGE;
        };
        self.decide_str(&value_to_string_key(v))
    }

    #[inline(always)]
    fn merge(&mut self, _other: &Self) {}

    #[inline]
    fn required_columns(&self) -> Vec<String> {
        vec![self.column_name.clone()]
    }
}

// ─── ColumnDeciderNumericExplicit (β) ──────────────────────────────────

/// Explicit numeric cutoffs on a numeric (Int64 / Float64) split column.
/// Stateless: pass A `accumulate` is a no-op; per-row decide buckets the
/// column value against `cutoffs` (sorted ascending).
///
/// `cutoffs.len() == 1` produces a binary split (stage 0 / stage 1);
/// `cutoffs.len() == 2` produces a three-way split.
///
/// **Note**: the SplitSpec input schema doesn't currently carry a
/// "cutoffs" field. This variant is exposed for unit tests + future
/// SplitSpec extensions. The dispatch in `execute_tt_pass` does NOT
/// route to this variant today — it falls into (α) when `manual_*_values`
/// are set, into (γ) when `ratio` is set, and `Err(InvalidPlan)`
/// otherwise.
#[derive(Debug, Clone)]
pub(crate) struct ColumnDeciderNumericExplicit {
    column_name: String,
    cutoffs: Vec<f64>,
}

impl ColumnDeciderNumericExplicit {
    pub(crate) fn new(column_name: String, mut cutoffs: Vec<f64>) -> Self {
        cutoffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Self { column_name, cutoffs }
    }

    #[inline]
    fn decide_f64(&self, v: f64) -> u8 {
        for (i, &c) in self.cutoffs.iter().enumerate() {
            if v < c {
                return i as u8;
            }
        }
        self.cutoffs.len() as u8
    }
}

impl<U: UnitSystem> SplitDecider<U> for ColumnDeciderNumericExplicit {
    type BatchView<'a> = NumericColumnBatchView<'a>;

    #[inline]
    fn batch_view<'a>(
        &self,
        batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        NumericColumnBatchView::new(batch, &self.column_name, row_offset)
    }

    #[inline]
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8 {
        match view.value_at(row_idx) {
            Some(v) => self.decide_f64(v),
            None => UNUSED_STAGE,
        }
    }

    #[inline(always)]
    fn accumulate(&mut self, _row: &TTRowRef<'_, U>) {}

    #[inline(always)]
    fn derive_cutoffs(&mut self) {}

    #[inline]
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8 {
        let Some(v) = row.split_column_value.and_then(value_to_f64) else {
            return UNUSED_STAGE;
        };
        self.decide_f64(v)
    }

    #[inline(always)]
    fn merge(&mut self, _other: &Self) {}

    #[inline]
    fn required_columns(&self) -> Vec<String> {
        vec![self.column_name.clone()]
    }
}

// ─── ColumnDeciderRatioNumeric (γ) ─────────────────────────────────────

/// Ratio-derived split on a numeric column. Pass A accumulates a t-digest
/// over per-row split-column values (read from `row.split_column_value`);
/// `derive_cutoffs` quantizes the digest at the cumulative ratio thresholds
/// and stores those as the per-row decision boundaries.
///
/// Mirrors `TemporalDecider` but on an arbitrary numeric column.
/// Quantiles are queried via `TDigestEstimator::quantile_at`
/// directly — no quartile-anchored interpolation workaround.
#[derive(Debug, Clone)]
pub(crate) struct ColumnDeciderRatioNumeric {
    column_name: String,
    ratio: Vec<f64>,
    digest: TDigestEstimator,
    cutoffs: Vec<f64>,
}

impl ColumnDeciderRatioNumeric {
    pub(crate) fn new(column_name: String, ratio: Vec<f64>) -> Self {
        Self {
            column_name,
            ratio,
            digest: TDigestEstimator::new(200),
            cutoffs: Vec::new(),
        }
    }

    #[inline]
    fn decide_f64(&self, v: f64) -> u8 {
        if self.cutoffs.is_empty() {
            return UNUSED_STAGE;
        }
        for (i, &c) in self.cutoffs.iter().enumerate() {
            if v < c {
                return i as u8;
            }
        }
        self.cutoffs.len() as u8
    }
}

impl<U: UnitSystem> SplitDecider<U> for ColumnDeciderRatioNumeric {
    type BatchView<'a> = NumericColumnBatchView<'a>;

    #[inline]
    fn batch_view<'a>(
        &self,
        batch: &'a RecordBatch,
        row_offset: u64,
    ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
        NumericColumnBatchView::new(batch, &self.column_name, row_offset)
    }

    #[inline]
    fn decide_for_view(&self, view: &Self::BatchView<'_>, row_idx: usize) -> u8 {
        match view.value_at(row_idx) {
            Some(v) => self.decide_f64(v),
            None => UNUSED_STAGE,
        }
    }

    #[inline]
    fn accumulate(&mut self, row: &TTRowRef<'_, U>) {
        if let Some(v) = row.split_column_value.and_then(value_to_f64) {
            self.digest.observe(v);
        }
    }

    fn derive_cutoffs(&mut self) {
        let cum = ratio_to_cumulative_cutoffs(&self.ratio);
        if cum.is_empty() {
            return;
        }
        let mut out = Vec::with_capacity(cum.len());
        for &q in &cum {
            // Direct `quantile_at` query (see TemporalDecider).
            out.push(self.digest.quantile_at(q));
        }
        self.cutoffs = out;
    }

    #[inline]
    fn decide(&self, row: &TTRowRef<'_, U>) -> u8 {
        let Some(v) = row.split_column_value.and_then(value_to_f64) else {
            return UNUSED_STAGE;
        };
        self.decide_f64(v)
    }

    fn merge(&mut self, other: &Self) {
        // Mirror TemporalDecider::merge — TDigestEstimator::merge takes
        // `&mut other`, so we clone (cheap: bounded by `delta` centroids).
        let mut other_clone = other.digest.clone();
        self.digest.merge(&mut other_clone);
    }

    #[inline]
    fn required_columns(&self) -> Vec<String> {
        vec![self.column_name.clone()]
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::TTGenError;
    use crate::ttgen_stats::TTRowRef;
    use diskgraph_core::time::{Abs, Seconds};
    use diskgraph_core::tt_builder::Value;

    fn sample_row<'a>(
        entity_pk: &'a Value,
        ts: Abs<Seconds>,
        target: &'a Value,
    ) -> TTRowRef<'a, Seconds> {
        TTRowRef {
            entity_pk,
            ts,
            target,
            frequency: None,
            target_is_numerical: true,
            target_is_categorical: false,
            target_stype: "numerical",
            split_column_value: None,
        }
    }

    fn sample_row_with_split<'a>(
        entity_pk: &'a Value,
        ts: Abs<Seconds>,
        target: &'a Value,
        split_value: &'a Value,
    ) -> TTRowRef<'a, Seconds> {
        TTRowRef {
            entity_pk,
            ts,
            target,
            frequency: None,
            target_is_numerical: true,
            target_is_categorical: false,
            target_stype: "numerical",
            split_column_value: Some(split_value),
        }
    }

    /// Build a tiny `RecordBatch` with an `ENTITY` column (Int64) at idx 0
    /// and a `TIMESTAMP` column (TimestampNanosecondArray, ns-scaled) at
    /// idx 1. Mirrors the schema produced by `ChannelBatchWriter`.
    fn batch_with_ts_secs(ts_secs: &[i64]) -> RecordBatch {
        use arrow_schema::{DataType, Field, Schema, TimeUnit};
        use std::sync::Arc;
        let entities: Vec<i64> = (0..ts_secs.len() as i64).collect();
        let entity_arr = Int64Array::from(entities);
        let ts_nanos: Vec<i64> = ts_secs.iter().map(|s| s * 1_000_000_000).collect();
        let ts_arr = TimestampNanosecondArray::from(ts_nanos);
        let schema = Arc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(entity_arr), Arc::new(ts_arr)]).unwrap()
    }

    // ─── NoSplitDecider ────────────────────────────────────────────────────

    #[test]
    fn test_no_split_decider_decide_returns_unused() {
        let decider = NoSplitDecider;
        let entity = Value::Int(1);
        let target = Value::Int(42);
        let row = sample_row(&entity, Abs::from_raw(1000), &target);
        assert_eq!(<NoSplitDecider as SplitDecider<Seconds>>::decide(&decider, &row), UNUSED_STAGE);
        assert_eq!(UNUSED_STAGE, 255);
    }

    #[test]
    fn test_no_split_decider_required_columns_empty() {
        let d = NoSplitDecider;
        let cols = <NoSplitDecider as SplitDecider<Seconds>>::required_columns(&d);
        assert!(cols.is_empty());
    }

    #[test]
    fn test_no_split_decider_decide_for_view_returns_unused() {
        let d = NoSplitDecider;
        let batch = batch_with_ts_secs(&[10, 20, 30]);
        let view = <NoSplitDecider as SplitDecider<Seconds>>::batch_view(&d, &batch, 0).unwrap();
        for i in 0..3 {
            assert_eq!(
                <NoSplitDecider as SplitDecider<Seconds>>::decide_for_view(&d, &view, i),
                UNUSED_STAGE
            );
        }
    }

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoSplitDecider>();
        assert_send_sync::<TimeRangeDecider<Seconds>>();
        assert_send_sync::<BacktestingDecider<Seconds>>();
        assert_send_sync::<ManualDecider<Seconds>>();
        assert_send_sync::<RandomDecider<Seconds>>();
        assert_send_sync::<RandomLinkDecider<Seconds>>();
        assert_send_sync::<FixedDecider<Seconds>>();
        assert_send_sync::<TemporalDecider<Seconds>>();
        assert_send_sync::<DateOffsetDecider<Seconds>>();
        assert_send_sync::<ApproxDateOffsetDecider<Seconds>>();
    }

    // ─── TimeRangeDecider ──────────────────────────────────────────────────

    fn time_range_three_stage() -> TimeRangeDecider<Seconds> {
        TimeRangeDecider::new(vec![
            (Abs::from_raw(0), Abs::from_raw(100)),
            (Abs::from_raw(101), Abs::from_raw(200)),
            (Abs::from_raw(201), Abs::from_raw(300)),
        ])
    }

    #[test]
    fn test_time_range_decider_in_range_returns_index() {
        let d = time_range_three_stage();
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for (ts, expect) in [(50, 0), (150, 1), (250, 2), (350, UNUSED_STAGE)] {
            let row = sample_row(&entity, Abs::from_raw(ts), &target);
            assert_eq!(d.decide(&row), expect, "ts={ts}");
        }
    }

    #[test]
    fn test_time_range_decider_decide_for_view_matches_decide() {
        let d = time_range_three_stage();
        let ts_secs = vec![50_i64, 150, 250, 350, -10];
        let batch = batch_with_ts_secs(&ts_secs);
        let view = d.batch_view(&batch, 0).unwrap();
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for (i, &ts) in ts_secs.iter().enumerate() {
            let by_view = d.decide_for_view(&view, i);
            let by_decide =
                d.decide(&sample_row(&entity, Abs::<Seconds>::from_secs(ts), &target));
            assert_eq!(by_view, by_decide, "ts={ts}");
        }
    }

    #[test]
    fn test_time_range_decider_required_columns() {
        let d = time_range_three_stage();
        assert_eq!(d.required_columns(), vec!["TIMESTAMP".to_string()]);
    }

    #[test]
    fn test_time_range_decider_temporal_adjacent_boundaries() {
        let d = TimeRangeDecider::<Seconds>::new(vec![
            (Abs::from_secs(0), Abs::from_secs(20)),
            (Abs::from_secs(20), Abs::from_secs(30)),
            (Abs::from_secs(30), Abs::from_secs(40)),
        ])
        .with_end_shift(Some(OffsetUnit::Fixed(Dur::from_secs(10))));

        assert_eq!(d.decide_ts(Abs::from_secs(10)), 0);
        assert_eq!(d.decide_ts(Abs::from_secs(20)), 1);
        assert_eq!(d.decide_ts(Abs::from_secs(30)), 2);
        assert_eq!(d.decide_ts(Abs::from_secs(40)), UNUSED_STAGE);
    }

    #[test]
    fn test_time_range_decider_static_shared_boundary_belongs_to_earlier_stage() {
        let d = TimeRangeDecider::<Seconds>::new(vec![
            (Abs::from_secs(0), Abs::from_secs(20)),
            (Abs::from_secs(20), Abs::from_secs(30)),
        ]);

        assert_eq!(d.decide_ts(Abs::from_secs(20)), 0);
        assert_eq!(d.decide_ts(Abs::from_secs(30)), 1);
        assert_eq!(d.decide_ts(Abs::from_secs(0)), UNUSED_STAGE);
    }

    /// A decider that takes a column-name string at construction and
    /// returns it from `required_columns()` integrates correctly. This
    /// exercises the SplitDecider trait's `Vec<String>` return shape (a
    /// non-static String works just like the `vec!["TIMESTAMP".to_string()]`
    /// returned by tier-1 deciders today). No production decider needs this
    /// shape yet; the test ensures the API supports a future ColumnDecider.
    #[test]
    fn test_decider_required_columns_from_constructor_string() {
        // Build a "fake" tier-1 decider whose required column is configured
        // at construction. We use a wrapper around NoSplitDecider since we
        // only need to assert the required_columns shape, not actually
        // exercise pass B.
        struct ColumnNameDecider {
            column_name: String,
        }
        impl<U: UnitSystem> SplitDecider<U> for ColumnNameDecider {
            type BatchView<'a> = OffsetOnlyView;
            fn batch_view<'a>(
                &self,
                _batch: &'a RecordBatch,
                row_offset: u64,
            ) -> crate::errors::TTGenResult<Self::BatchView<'a>> {
                Ok(OffsetOnlyView { start_offset: row_offset })
            }
            fn decide_for_view(&self, _view: &Self::BatchView<'_>, _row_idx: usize) -> u8 {
                UNUSED_STAGE
            }
            fn accumulate(&mut self, _row: &TTRowRef<'_, U>) {}
            fn derive_cutoffs(&mut self) {}
            fn decide(&self, _row: &TTRowRef<'_, U>) -> u8 {
                UNUSED_STAGE
            }
            fn merge(&mut self, _other: &Self) {}
            fn required_columns(&self) -> Vec<String> {
                vec![self.column_name.clone()]
            }
        }

        let d = ColumnNameDecider {
            column_name: "TARGET2".to_string(),
        };
        let cols = <ColumnNameDecider as SplitDecider<Seconds>>::required_columns(&d);
        assert_eq!(cols, vec!["TARGET2".to_string()]);

        // Also verify it still works when the column name varies.
        let d2 = ColumnNameDecider {
            column_name: "ENTITY1".to_string(),
        };
        assert_eq!(
            <ColumnNameDecider as SplitDecider<Seconds>>::required_columns(&d2),
            vec!["ENTITY1".to_string()]
        );
    }

    // ─── BacktestingDecider ────────────────────────────────────────────────

    #[test]
    fn test_backtesting_decider_test_ts_returns_test() {
        let d = BacktestingDecider::<Seconds>::new(Abs::from_raw(1000));
        let entity = Value::Int(0);
        let target = Value::Int(0);
        assert_eq!(d.decide(&sample_row(&entity, Abs::from_raw(1000), &target)), 2);
        assert_eq!(d.decide(&sample_row(&entity, Abs::from_raw(999), &target)), 0);
    }

    #[test]
    fn test_backtesting_decider_decide_for_view_matches() {
        let d = BacktestingDecider::<Seconds>::new(Abs::from_secs(1000));
        let batch = batch_with_ts_secs(&[999, 1000, 1001]);
        let view = d.batch_view(&batch, 0).unwrap();
        assert_eq!(d.decide_for_view(&view, 0), 0);
        assert_eq!(d.decide_for_view(&view, 1), 2);
        assert_eq!(d.decide_for_view(&view, 2), 0);
    }

    // ─── ManualDecider ─────────────────────────────────────────────────────

    fn manual_decider_basic() -> ManualDecider<Seconds> {
        let train: HashSet<Abs<Seconds>> = [10, 20, 30].into_iter().map(Abs::from_raw).collect();
        let val: HashSet<Abs<Seconds>> = [40, 50].into_iter().map(Abs::from_raw).collect();
        let test: HashSet<Abs<Seconds>> = [60, 70].into_iter().map(Abs::from_raw).collect();
        ManualDecider::new(train, val, test)
    }

    #[test]
    fn test_manual_decider_in_set_returns_stage() {
        let d = manual_decider_basic();
        let entity = Value::Int(0);
        let target = Value::Int(0);
        assert_eq!(d.decide(&sample_row(&entity, Abs::from_raw(20), &target)), 0);
        assert_eq!(d.decide(&sample_row(&entity, Abs::from_raw(40), &target)), 1);
        assert_eq!(d.decide(&sample_row(&entity, Abs::from_raw(70), &target)), 2);
        assert_eq!(d.decide(&sample_row(&entity, Abs::from_raw(35), &target)), UNUSED_STAGE);
    }

    #[test]
    fn test_manual_decider_decide_for_view_matches() {
        let d = manual_decider_basic();
        let batch = batch_with_ts_secs(&[10, 20, 40, 60, 99]);
        let view = d.batch_view(&batch, 0).unwrap();
        let expected = [0, 0, 1, 2, UNUSED_STAGE];
        for (i, e) in expected.iter().enumerate() {
            assert_eq!(d.decide_for_view(&view, i), *e, "row {i}");
        }
    }

    // ─── RandomDecider ─────────────────────────────────────────────────────

    #[test]
    fn test_random_decider_deterministic() {
        let d: RandomDecider<Seconds> = RandomDecider::new(vec![0.7, 0.2, 0.1], 42);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        // Same input → same output every time.
        let r1 = d.decide(&sample_row(&entity, Abs::from_raw(123), &target));
        let r2 = d.decide(&sample_row(&entity, Abs::from_raw(123), &target));
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_random_decider_aggregate_ratio_approximate() {
        // Hash-based bucketing should land roughly at the configured
        // ratio in expectation. With 1000 rows and ratio 0.7/0.2/0.1,
        // each stage count should be within 10% of its expected size.
        let d: RandomDecider<Seconds> = RandomDecider::new(vec![0.7, 0.2, 0.1], 42);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        let mut counts = [0u64; 4];
        for ts in 0..1000_i64 {
            let row = sample_row(&entity, Abs::from_raw(ts), &target);
            let stage = d.decide(&row);
            counts[stage as usize] += 1;
        }
        // Generous tolerance: hash uniformity is approximate.
        assert!((counts[0] as i64 - 700).abs() < 80, "train count: {}", counts[0]);
        assert!((counts[1] as i64 - 200).abs() < 60, "val count: {}", counts[1]);
        assert!((counts[2] as i64 - 100).abs() < 50, "test count: {}", counts[2]);
        assert_eq!(counts[3], 0); // no UNUSED in three-way split
    }

    #[test]
    fn test_random_decider_seed_changes_assignment() {
        // Different seeds → different per-row assignments (with very high
        // probability). Aggregate counts are still ~ratio.
        let d1: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], 1);
        let d2: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], 2);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        let mut diffs = 0;
        for ts in 0..100_i64 {
            let row = sample_row(&entity, Abs::from_raw(ts), &target);
            if d1.decide(&row) != d2.decide(&row) {
                diffs += 1;
            }
        }
        assert!(diffs > 30, "seeds should produce different assignments; diffs={diffs}");
    }

    #[test]
    fn test_random_decider_decide_for_view_matches_decide() {
        let d: RandomDecider<Seconds> = RandomDecider::new(vec![0.6, 0.4], 7);
        let ts_secs: Vec<i64> = (0..20).map(|i| i * 13 + 1).collect();
        let batch = batch_with_ts_secs(&ts_secs);
        let view = d.batch_view(&batch, 0).unwrap();
        let target = Value::Int(0);
        // `batch_with_ts_secs` populates ENTITY1 with `0..ts_secs.len()`.
        // Mirror that here so the per-row `decide` agrees with
        // `decide_for_view` on the entity, not just the timestamp.
        for (i, &ts) in ts_secs.iter().enumerate() {
            let by_view = d.decide_for_view(&view, i);
            let entity = Value::Int(i as i64);
            let by_decide = d.decide(&sample_row(&entity, Abs::from_secs(ts), &target));
            assert_eq!(by_view, by_decide);
        }
    }

    #[test]
    fn test_random_decider_merge_is_noop() {
        let mut a: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], 1);
        let b: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], 1);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        let row = sample_row(&entity, Abs::from_raw(42), &target);
        let before = a.decide(&row);
        a.merge(&b);
        a.derive_cutoffs();
        assert_eq!(a.decide(&row), before);
    }

    // ─── RandomLinkDecider ─────────────────────────────────────────────────

    #[test]
    fn test_random_link_decider_matches_random() {
        let d_link: RandomLinkDecider<Seconds> = RandomLinkDecider::new(vec![0.7, 0.2, 0.1], 42);
        let d_rnd: RandomDecider<Seconds> = RandomDecider::new(vec![0.7, 0.2, 0.1], 42);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for ts in 0..50_i64 {
            let row = sample_row(&entity, Abs::from_raw(ts), &target);
            assert_eq!(d_link.decide(&row), d_rnd.decide(&row), "ts={ts}");
        }
    }

    // ─── RandomDecider entity-aware tests (correctness fix) ────────────────
    //
    // These tests verify that `RandomDecider` hashes `(entity_pk, ts, seed)`,
    // not just `(ts, seed)`. The ts-only hash silently broke production
    // datasets with timestamp clustering: same-ts rows all landed in one
    // stage, the aggregate ratio skewed, and fresh-ts customer data could
    // entirely miss the train stage.

    /// Build a tiny `RecordBatch` with `ENTITY1` (Int64) + `TIMESTAMP`
    /// (TimestampNanosecondArray) where the entities and timestamps are
    /// supplied directly. Used by the entity-aware view tests.
    fn batch_with_int_entities_and_ts(entities: &[i64], ts_secs: &[i64]) -> RecordBatch {
        use arrow_schema::{DataType, Field, Schema, TimeUnit};
        use std::sync::Arc;
        assert_eq!(entities.len(), ts_secs.len());
        let entity_arr = Int64Array::from(entities.to_vec());
        let ts_nanos: Vec<i64> = ts_secs.iter().map(|s| s * 1_000_000_000).collect();
        let ts_arr = TimestampNanosecondArray::from(ts_nanos);
        let schema = Arc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Int64, false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(entity_arr), Arc::new(ts_arr)]).unwrap()
    }

    fn batch_with_str_entities_and_ts(entities: &[&str], ts_secs: &[i64]) -> RecordBatch {
        use arrow_array::StringArray;
        use arrow_schema::{DataType, Field, Schema, TimeUnit};
        use std::sync::Arc;
        assert_eq!(entities.len(), ts_secs.len());
        let entity_arr = StringArray::from(entities.to_vec());
        let ts_nanos: Vec<i64> = ts_secs.iter().map(|s| s * 1_000_000_000).collect();
        let ts_arr = TimestampNanosecondArray::from(ts_nanos);
        let schema = Arc::new(Schema::new(vec![
            Field::new("ENTITY1", DataType::Utf8, false),
            Field::new("TIMESTAMP", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(entity_arr), Arc::new(ts_arr)]).unwrap()
    }

    #[test]
    fn test_random_decider_distinguishes_entities_at_same_ts() {
        // 1000 distinct entities ALL at the same timestamp must split into
        // multiple stages (this is the core correctness bug: ts-only hashing
        // crammed every same-ts row into a single stage).
        let d: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.25, 0.25], 42);
        let target = Value::Int(0);
        let mut counts = [0u64; 3];
        let entities: Vec<Value> = (0..1000).map(Value::Int).collect();
        for entity in &entities {
            let row = sample_row(entity, Abs::from_raw(123_456_789), &target);
            let stage = d.decide(&row);
            assert!(stage < 3, "stage byte {stage} out of alphabet");
            counts[stage as usize] += 1;
        }
        // The buggy ts-only impl crams ALL 1000 rows into one stage; assert
        // that at least 2 stages are non-empty (proves entity is in the hash).
        let nonempty = counts.iter().filter(|&&c| c > 0).count();
        assert!(
            nonempty >= 2,
            "expected entity-distinguished split; got {counts:?}"
        );
        // Aggregate within ±5% of (500, 250, 250).
        assert!(
            (counts[0] as i64 - 500).abs() <= 50,
            "train within 5%: counts={counts:?}"
        );
        assert!(
            (counts[1] as i64 - 250).abs() <= 50,
            "val within 5%: counts={counts:?}"
        );
        assert!(
            (counts[2] as i64 - 250).abs() <= 50,
            "test within 5%: counts={counts:?}"
        );
    }

    #[test]
    fn test_random_decider_aggregate_ratios_under_ts_clustering() {
        // 10000 rows clustered onto 3 timestamps. The ts-only hash sends each
        // ts-cluster 100% to one stage, so aggregate skews wildly. With the
        // entity-aware hash, the aggregate must stay close to 80/10/10.
        let d: RandomDecider<Seconds> = RandomDecider::new(vec![0.8, 0.1, 0.1], 42);
        let target = Value::Int(0);
        let mut counts = [0u64; 3];
        let mut entity_idx: i64 = 0;
        let cluster_sizes = [(0_i64, 5000_usize), (1_i64, 3000), (2_i64, 2000)];
        for (ts, n) in &cluster_sizes {
            for _ in 0..*n {
                let entity = Value::Int(entity_idx);
                let row = sample_row(&entity, Abs::from_raw(*ts), &target);
                let stage = d.decide(&row);
                counts[stage as usize] += 1;
                entity_idx += 1;
            }
        }
        // ±2% tolerance over 10000 rows.
        assert!(
            (counts[0] as i64 - 8000).abs() <= 200,
            "train within 2% of 8000: counts={counts:?}"
        );
        assert!(
            (counts[1] as i64 - 1000).abs() <= 200,
            "val within 2% of 1000: counts={counts:?}"
        );
        assert!(
            (counts[2] as i64 - 1000).abs() <= 200,
            "test within 2% of 1000: counts={counts:?}"
        );
    }

    #[test]
    fn test_random_decider_deterministic_entity_aware() {
        // Same (entity, ts, seed) on two distinct decider instances → same
        // stage. (The pre-existing `test_random_decider_deterministic` only
        // checks repeatability of one instance; this is the entity-aware
        // version.)
        let d1: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], 7);
        let d2: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], 7);
        let target = Value::Int(0);
        for i in 0..100i64 {
            let entity = Value::Int(i);
            let r1 = d1.decide(&sample_row(&entity, Abs::from_raw(42), &target));
            let r2 = d2.decide(&sample_row(&entity, Abs::from_raw(42), &target));
            assert_eq!(r1, r2, "entity={i}");
        }
    }

    #[test]
    fn test_random_decider_different_seeds_diverge_per_entity() {
        // Same (entity, ts), two seeds → at least 100 of 1000 entities differ.
        let d1: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], 1);
        let d2: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], 2);
        let target = Value::Int(0);
        let mut diffs = 0;
        for i in 0..1000i64 {
            let entity = Value::Int(i);
            let row = sample_row(&entity, Abs::from_raw(0), &target);
            if d1.decide(&row) != d2.decide(&row) {
                diffs += 1;
            }
        }
        assert!(diffs > 100, "expected >100 diffs across seeds; got {diffs}");
    }

    #[test]
    fn test_random_decider_value_variants() {
        // All Value variants work as entity_pk and produce uniform splits.
        let d: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], 42);
        let target = Value::Int(0);

        // Int variant uniformity
        let mut int_counts = [0u64; 2];
        for i in 0..1000i64 {
            let entity = Value::Int(i);
            int_counts[d.decide(&sample_row(&entity, Abs::from_raw(0), &target)) as usize] += 1;
        }
        assert!(
            (int_counts[0] as i64 - 500).abs() <= 80,
            "Int uniform: {int_counts:?}"
        );

        // Float variant uniformity
        let mut float_counts = [0u64; 2];
        for i in 0..1000i64 {
            let entity = Value::Float(i as f64 + 0.5);
            float_counts[d.decide(&sample_row(&entity, Abs::from_raw(0), &target)) as usize] += 1;
        }
        assert!(
            (float_counts[0] as i64 - 500).abs() <= 80,
            "Float uniform: {float_counts:?}"
        );

        // Str variant uniformity
        let mut str_counts = [0u64; 2];
        for i in 0..1000i64 {
            let entity = Value::Str(format!("e_{i}"));
            str_counts[d.decide(&sample_row(&entity, Abs::from_raw(0), &target)) as usize] += 1;
        }
        assert!(
            (str_counts[0] as i64 - 500).abs() <= 80,
            "Str uniform: {str_counts:?}"
        );

        // Bool variant — only two distinct values, just verify it doesn't
        // panic and stages are valid bytes.
        let bool_t = Value::Bool(true);
        let bool_f = Value::Bool(false);
        let s_t = d.decide(&sample_row(&bool_t, Abs::from_raw(0), &target));
        let s_f = d.decide(&sample_row(&bool_f, Abs::from_raw(0), &target));
        assert!(s_t < 2 && s_f < 2);

        // Cross-variant collision check: Int(0x6e) and Str("\x6e") would
        // collide under naive byte-only encoding. The variant discriminator
        // byte must keep them apart with very high probability.
        let int_val = Value::Int(0x6e);
        let str_val = Value::Str("\x6e".to_string());
        // Repeat across many seeds — at least one seed must produce
        // different stage assignments. (Probability of collision across
        // 50 seeds is < 2^-50 if discriminator is in place.)
        let mut diffs = 0;
        for seed in 0..50u64 {
            let dd: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], seed);
            let s_int = dd.decide(&sample_row(&int_val, Abs::from_raw(0), &target));
            let s_str = dd.decide(&sample_row(&str_val, Abs::from_raw(0), &target));
            if s_int != s_str {
                diffs += 1;
            }
        }
        assert!(
            diffs > 0,
            "Int(0x6e) and Str(\"\\x6e\") collided across all 50 seeds — variant tag missing"
        );
    }

    #[test]
    fn test_random_link_decider_per_row_inherits_fix() {
        // RandomLinkDecider wraps RandomDecider; the same-ts entity-distinct
        // splitting must hold for it too.
        let d: RandomLinkDecider<Seconds> = RandomLinkDecider::new(vec![0.5, 0.25, 0.25], 42);
        let target = Value::Int(0);
        let mut counts = [0u64; 3];
        for i in 0..1000i64 {
            let entity = Value::Int(i);
            let row = sample_row(&entity, Abs::from_raw(123_456_789), &target);
            let stage = d.decide(&row);
            counts[stage as usize] += 1;
        }
        let nonempty = counts.iter().filter(|&&c| c > 0).count();
        assert!(
            nonempty >= 2,
            "RandomLinkDecider: entity-distinguished split required; counts={counts:?}"
        );
    }

    #[test]
    fn test_random_decider_decide_for_view_int_entities() {
        // The pass-B view path (decide_for_view) must agree with the pass-A
        // path (decide) for Int-typed entity columns. 50 rows at the same ts
        // must split across stages.
        let d: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], 42);
        let entities: Vec<i64> = (0..50).collect();
        let ts_secs = vec![777i64; 50];
        let batch = batch_with_int_entities_and_ts(&entities, &ts_secs);
        let view = d.batch_view(&batch, 0).unwrap();
        let target = Value::Int(0);
        let mut counts = [0u64; 2];
        for (i, &eid) in entities.iter().enumerate() {
            let by_view = d.decide_for_view(&view, i);
            let entity = Value::Int(eid);
            let by_decide =
                d.decide(&sample_row(&entity, Abs::from_secs(ts_secs[i]), &target));
            assert_eq!(by_view, by_decide, "entity={eid}");
            counts[by_view as usize] += 1;
        }
        let nonempty = counts.iter().filter(|&&c| c > 0).count();
        assert!(
            nonempty >= 2,
            "view path: entity-distinguished split required; counts={counts:?}"
        );
    }

    #[test]
    fn test_random_decider_decide_for_view_str_entities() {
        // Same as Int test, for Utf8 entity column.
        let d: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], 42);
        let entity_strs: Vec<String> = (0..50).map(|i| format!("e_{i}")).collect();
        let entity_refs: Vec<&str> = entity_strs.iter().map(|s| s.as_str()).collect();
        let ts_secs = vec![777i64; 50];
        let batch = batch_with_str_entities_and_ts(&entity_refs, &ts_secs);
        let view = d.batch_view(&batch, 0).unwrap();
        let target = Value::Int(0);
        let mut counts = [0u64; 2];
        for (i, name) in entity_strs.iter().enumerate() {
            let by_view = d.decide_for_view(&view, i);
            let entity = Value::Str(name.clone());
            let by_decide =
                d.decide(&sample_row(&entity, Abs::from_secs(ts_secs[i]), &target));
            assert_eq!(by_view, by_decide, "entity={name}");
            counts[by_view as usize] += 1;
        }
        let nonempty = counts.iter().filter(|&&c| c > 0).count();
        assert!(
            nonempty >= 2,
            "view path (str): entity-distinguished split; counts={counts:?}"
        );
    }

    #[test]
    fn test_random_decider_required_columns_includes_entity1() {
        // Pass-B projection must include ENTITY1 so the view can read it.
        let d: RandomDecider<Seconds> = RandomDecider::new(vec![0.5, 0.5], 42);
        let cols =
            <RandomDecider<Seconds> as SplitDecider<Seconds>>::required_columns(&d);
        assert!(
            cols.iter().any(|c| c == "TIMESTAMP"),
            "TIMESTAMP missing: {cols:?}"
        );
        assert!(
            cols.iter().any(|c| c == "ENTITY1"),
            "ENTITY1 missing: {cols:?}"
        );
    }

    // ─── FixedDecider ──────────────────────────────────────────────────────

    #[test]
    fn test_fixed_decider_apportion_three_way() {
        // 100 rows, 0.7/0.2/0.1 → [70, 20, 10]
        let sizes = FixedDecider::<Seconds>::apportion(100, &[0.7, 0.2, 0.1]);
        assert_eq!(sizes, vec![70, 20, 10]);
    }

    #[test]
    fn test_fixed_decider_apportion_largest_remainder() {
        // 10 rows, 0.7/0.2/0.1 → ideal=7,2,1 → exact (no remainder).
        let sizes = FixedDecider::<Seconds>::apportion(10, &[0.7, 0.2, 0.1]);
        assert_eq!(sizes.iter().sum::<u64>(), 10);
    }

    #[test]
    fn test_fixed_decider_apportion_unbalanced() {
        // 7 rows, 0.7/0.2/0.1 → ideal=4.9, 1.4, 0.7; floors=4,1,0; rem=2.
        // Remainders 0.9, 0.4, 0.7 → order [0, 2, 1] → +1 to [0,2] →
        // sizes=[5,1,1].
        let sizes = FixedDecider::<Seconds>::apportion(7, &[0.7, 0.2, 0.1]);
        assert_eq!(sizes.iter().sum::<u64>(), 7);
        assert_eq!(sizes[0], 5);
    }

    #[test]
    fn test_fixed_decider_accumulate_and_derive_cutoffs() {
        let mut d: FixedDecider<Seconds> = FixedDecider::new(vec![0.7, 0.2, 0.1]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for ts in 0..100_i64 {
            d.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        d.derive_cutoffs();
        assert_eq!(d.cutoffs_global, vec![70, 90]);
    }

    #[test]
    fn test_fixed_decider_decide_for_view_uses_global_offset() {
        let mut d: FixedDecider<Seconds> = FixedDecider::new(vec![0.7, 0.2, 0.1]);
        d.row_count = 100;
        d.derive_cutoffs(); // cutoffs at 70, 90
        // Build a batch of 10 rows starting at global offset 65.
        let ts_secs: Vec<i64> = (0..10).collect();
        let batch = batch_with_ts_secs(&ts_secs);
        let view = d.batch_view(&batch, 65).unwrap();
        // global=65..75: 65→0, 66→0, ..., 69→0, 70→1, 71→1, ..., 74→1
        for i in 0..10 {
            let global = 65 + i as u64;
            let expected = if global < 70 { 0 } else if global < 90 { 1 } else { 2 };
            assert_eq!(d.decide_for_view(&view, i), expected, "global={global}");
        }
    }

    #[test]
    fn test_fixed_decider_merge_sums_counts() {
        let mut a: FixedDecider<Seconds> = FixedDecider::new(vec![0.5, 0.5]);
        let mut b: FixedDecider<Seconds> = FixedDecider::new(vec![0.5, 0.5]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for _ in 0..30 {
            a.accumulate(&sample_row(&entity, Abs::from_raw(0), &target));
        }
        for _ in 0..40 {
            b.accumulate(&sample_row(&entity, Abs::from_raw(0), &target));
        }
        a.merge(&b);
        a.derive_cutoffs();
        // 70 total, 50/50 → [35, 35]; cutoff = 35.
        assert_eq!(a.row_count, 70);
        assert_eq!(a.cutoffs_global, vec![35]);
    }

    #[test]
    fn test_fixed_decider_required_columns_empty() {
        let d: FixedDecider<Seconds> = FixedDecider::new(vec![0.5, 0.5]);
        let cols = <FixedDecider<Seconds> as SplitDecider<Seconds>>::required_columns(&d);
        assert!(cols.is_empty());
    }

    // ─── TemporalDecider ───────────────────────────────────────────────────

    #[test]
    fn test_temporal_decider_accumulate_and_derive_cutoffs() {
        let mut d: TemporalDecider<Seconds> = TemporalDecider::new(vec![0.5, 0.5]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for ts in 0..1000_i64 {
            d.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        d.derive_cutoffs();
        assert_eq!(d.cutoffs.len(), 1);
        // q50 of [0, 999] is around 500; t-digest has a few-percent tolerance.
        let cutoff_raw = d.cutoffs[0].raw();
        assert!(
            (cutoff_raw - 500).abs() < 50,
            "expected ~500, got {cutoff_raw}"
        );
    }

    #[test]
    fn test_temporal_decider_three_way_aggregate_ratio() {
        let mut d: TemporalDecider<Seconds> = TemporalDecider::new(vec![0.6, 0.2, 0.2]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for ts in 0..1000_i64 {
            d.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        d.derive_cutoffs();
        assert_eq!(d.cutoffs.len(), 2);
        let mut counts = [0u64; 3];
        for ts in 0..1000_i64 {
            let stage = d.decide(&sample_row(&entity, Abs::from_raw(ts), &target));
            counts[stage as usize] += 1;
        }
        // Loose bound: t-digest interpolation is approximate, especially
        // off-quartile ratios. Within 200 rows of expected.
        assert!((counts[0] as i64 - 600).abs() < 200, "train: {}", counts[0]);
    }

    #[test]
    fn test_temporal_decider_merge() {
        let mut a: TemporalDecider<Seconds> = TemporalDecider::new(vec![0.5, 0.5]);
        let mut b: TemporalDecider<Seconds> = TemporalDecider::new(vec![0.5, 0.5]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for ts in 0..500_i64 {
            a.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        for ts in 500..1000_i64 {
            b.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        a.merge(&b);
        a.derive_cutoffs();
        // After merge, q50 should be ~500 over [0, 999].
        let cutoff_raw = a.cutoffs[0].raw();
        assert!(
            (cutoff_raw - 500).abs() < 100,
            "expected ~500 after merge, got {cutoff_raw}"
        );
    }

    // ─── DateOffsetDecider ─────────────────────────────────────────────────

    #[test]
    fn test_date_offset_decider_two_way_basic() {
        // Two offsets, no anchor. Mirrors pybind logic: boundaries =
        // [min, max+off0, max+off1+timeframe (default = off1-off0)]
        // = [0, 90, 100+(100-90)+0=110]. The `timeframe` default for
        // 2-offset case is offsets[1]-offsets[0]=10. shift = tf+anchor = 10.
        let offsets = vec![Dur::<Seconds>::from_raw(-10), Dur::<Seconds>::from_raw(0)];
        let mut d: DateOffsetDecider<Seconds> =
            DateOffsetDecider::new(offsets, Dur::<Seconds>::zero(), None);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for ts in [0_i64, 30, 60, 100] {
            d.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        d.derive_cutoffs();
        // boundaries=[0, 90, 100, 110]. Stages:
        //   row in [0, 90] AND row+10 in [0, 90]   → 0
        //   row in [90, 100] AND row+10 in [0, 100] → 1
        // ts=0: shifted=10. 0>=0 && 10<=90 → 0; 0>=90? no; 0>=100? no → 0.
        // ts=30: shifted=40. 30>=0 && 40<=90 → 0; 30>=90? no → 0.
        // ts=60: shifted=70. 60>=0 && 70<=90 → 0 → 0.
        // ts=100: shifted=110. 100>=0 && 110<=90? no; 100>=90 && 110<=100? no;
        //         100>=100 && 110<=110 → 2.
        let s0 = d.decide(&sample_row(&entity, Abs::from_raw(0), &target));
        let s30 = d.decide(&sample_row(&entity, Abs::from_raw(30), &target));
        let s60 = d.decide(&sample_row(&entity, Abs::from_raw(60), &target));
        let s100 = d.decide(&sample_row(&entity, Abs::from_raw(100), &target));
        assert_eq!(s0, 0, "ts=0");
        assert_eq!(s30, 0, "ts=30");
        assert_eq!(s60, 0, "ts=60");
        // s100 may be 2 or UNUSED depending on inclusive bounds — match
        // pybind's `>=`/`<=` semantics: shifted=110, end[2]=110, so 2.
        assert_eq!(s100, 2, "ts=100");
    }

    #[test]
    fn test_date_offset_repro_540_300_tf60() {
        // Reproduce the e2e exactsplit scenario in isolation:
        // offsets [-540d, -300d], timeframe=60d, flexible=False (raw cutoffs).
        // Python reference on the real TT: train=728, val=257, test=556.
        let day = 86400_i64;
        let offsets = vec![
            Dur::<Seconds>::from_raw(-540 * day),
            Dur::<Seconds>::from_raw(-300 * day),
        ];
        let mut d: DateOffsetDecider<Seconds> = DateOffsetDecider::new(
            offsets,
            Dur::<Seconds>::zero(),
            Some(Dur::<Seconds>::from_raw(60 * day)),
        );
        let entity = Value::Int(0);
        let target = Value::Int(0);
        let min_ts = 959_817_600_i64; // 2000-06-01
        let max_ts = 1_400_889_600_i64; // 2014-05-24
        d.accumulate(&sample_row(&entity, Abs::from_raw(min_ts), &target));
        d.accumulate(&sample_row(&entity, Abs::from_raw(max_ts), &target));
        d.derive_cutoffs();
        let train_ts = 1_338_508_800_i64; // 2012-06-01 -> expect 0 (train)
        let val_ts = 1_362_096_000_i64; // 2013-03-01 -> expect 1 (val)
        let test_ts = 1_393_632_000_i64; // 2014-03-01 -> expect 2 (test)
        let st = d.decide(&sample_row(&entity, Abs::from_raw(train_ts), &target));
        let sv = d.decide(&sample_row(&entity, Abs::from_raw(val_ts), &target));
        let se = d.decide(&sample_row(&entity, Abs::from_raw(test_ts), &target));
        println!(
            "REPRO boundaries={:?} shift={:?} train_stage={} val_stage={} test_stage={}",
            d.boundaries, d.shift, st, sv, se
        );
        assert_eq!(st, 0, "2012-06-01 should be TRAIN");
        assert_eq!(sv, 1, "2013-03-01 should be VAL (this is the e2e val=0 bug)");
        assert_eq!(se, 2, "2014-03-01 should be TEST");
    }

    #[test]
    fn test_date_offset_decider_merge_min_max() {
        let mut a: DateOffsetDecider<Seconds> = DateOffsetDecider::new(
            vec![Dur::<Seconds>::from_raw(-10), Dur::<Seconds>::from_raw(0)],
            Dur::<Seconds>::zero(),
            None,
        );
        let mut b: DateOffsetDecider<Seconds> = DateOffsetDecider::new(
            vec![Dur::<Seconds>::from_raw(-10), Dur::<Seconds>::from_raw(0)],
            Dur::<Seconds>::zero(),
            None,
        );
        let entity = Value::Int(0);
        let target = Value::Int(0);
        a.accumulate(&sample_row(&entity, Abs::from_raw(50), &target));
        a.accumulate(&sample_row(&entity, Abs::from_raw(70), &target));
        b.accumulate(&sample_row(&entity, Abs::from_raw(10), &target));
        b.accumulate(&sample_row(&entity, Abs::from_raw(120), &target));
        a.merge(&b);
        assert_eq!(a.min_ts.unwrap().raw(), 10);
        assert_eq!(a.max_ts.unwrap().raw(), 120);
    }

    #[test]
    fn test_date_offset_decider_decide_for_view_matches() {
        let offsets = vec![Dur::<Seconds>::from_raw(-10), Dur::<Seconds>::from_raw(0)];
        let mut d: DateOffsetDecider<Seconds> =
            DateOffsetDecider::new(offsets, Dur::<Seconds>::zero(), None);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for ts in [0_i64, 30, 60, 100] {
            d.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        d.derive_cutoffs();
        let ts_secs = vec![0_i64, 30, 60, 100];
        let batch = batch_with_ts_secs(&ts_secs);
        let view = d.batch_view(&batch, 0).unwrap();
        for (i, &ts) in ts_secs.iter().enumerate() {
            let by_view = d.decide_for_view(&view, i);
            let by_decide = d.decide(&sample_row(&entity, Abs::from_secs(ts), &target));
            assert_eq!(by_view, by_decide, "ts={ts}");
        }
    }

    // ─── DateOffsetDecider — calendar units ────────────────────────────────

    /// Assemble an `Abs<Seconds>` for `YYYY-MM-DD 00:00:00 UTC`.
    fn ymd_secs(year: i32, month: u32, day: u32) -> Abs<Seconds> {
        use chrono::NaiveDate;
        let secs = NaiveDate::from_ymd_opt(year, month, day)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        Abs::<Seconds>::from_secs(secs)
    }

    fn ymd_to_str(ts: Abs<Seconds>) -> String {
        use chrono::DateTime;
        DateTime::from_timestamp(ts.to_secs(), 0)
            .unwrap()
            .format("%Y-%m-%d")
            .to_string()
    }

    #[test]
    fn test_date_offset_decider_calendar_months() {
        // anchor max_ts = 2024-03-15. Offsets: [-3 months, -1 month].
        // boundaries = [min_ts, max_ts + (-3m), max_ts + (-1m), max_ts + tf + anchor]
        //            = [min_ts, 2023-12-15, 2024-02-15, ...]
        let max_ts = ymd_secs(2024, 3, 15);
        let min_ts = ymd_secs(2023, 1, 1);
        let mut d: DateOffsetDecider<Seconds> = DateOffsetDecider::new_with_units(
            vec![OffsetUnit::Months(-3), OffsetUnit::Months(-1)],
            Dur::<Seconds>::zero(),
            None,
        );
        let entity = Value::Int(0);
        let target = Value::Int(0);
        d.accumulate(&sample_row(&entity, min_ts, &target));
        d.accumulate(&sample_row(&entity, max_ts, &target));
        d.derive_cutoffs();
        // boundaries[1] should be 2023-12-15.
        assert_eq!(d.boundaries.len(), 4);
        assert_eq!(
            ymd_to_str(d.boundaries[1]),
            "2023-12-15",
            "boundaries[1] = max_ts + (-3 months)"
        );
        assert_eq!(
            ymd_to_str(d.boundaries[2]),
            "2024-02-15",
            "boundaries[2] = max_ts + (-1 month)"
        );
    }

    #[test]
    fn test_date_offset_decider_calendar_years() {
        // anchor max_ts = 2024-03-15. Offsets: [-2 years, -1 year].
        let max_ts = ymd_secs(2024, 3, 15);
        let min_ts = ymd_secs(2020, 1, 1);
        let mut d: DateOffsetDecider<Seconds> = DateOffsetDecider::new_with_units(
            vec![OffsetUnit::Years(-2), OffsetUnit::Years(-1)],
            Dur::<Seconds>::zero(),
            None,
        );
        let entity = Value::Int(0);
        let target = Value::Int(0);
        d.accumulate(&sample_row(&entity, min_ts, &target));
        d.accumulate(&sample_row(&entity, max_ts, &target));
        d.derive_cutoffs();
        assert_eq!(
            ymd_to_str(d.boundaries[1]),
            "2022-03-15",
            "boundaries[1] = max_ts - 2 years"
        );
        assert_eq!(
            ymd_to_str(d.boundaries[2]),
            "2023-03-15",
            "boundaries[2] = max_ts - 1 year"
        );
    }

    #[test]
    fn test_date_offset_decider_end_of_month_clamping() {
        // Convention chosen: end-of-month CLAMPING. 2024-03-31 + (-1 month)
        // = 2024-02-29 (leap year, last valid day of Feb). This matches
        // `pd.DateOffset(months=-1)` and `add_calendar_months`'s behavior.
        let anchor = ymd_secs(2024, 3, 31);
        let result = subtract_calendar_offset::<Seconds>(anchor, OffsetUnit::Months(1));
        assert_eq!(
            ymd_to_str(result),
            "2024-02-29",
            "2024 is a leap year; clamp Mar 31 - 1 month to Feb 29"
        );

        // Non-leap year: 2023-03-31 - 1 month = 2023-02-28.
        let anchor_nonleap = ymd_secs(2023, 3, 31);
        let result_nonleap =
            subtract_calendar_offset::<Seconds>(anchor_nonleap, OffsetUnit::Months(1));
        assert_eq!(
            ymd_to_str(result_nonleap),
            "2023-02-28",
            "2023 non-leap; clamp Mar 31 - 1 month to Feb 28"
        );
    }

    #[test]
    fn test_apply_offset_fixed_matches_dur_addition() {
        // Sanity: OffsetUnit::Fixed(d) is identical to anchor + d.
        let anchor = ymd_secs(2024, 6, 15);
        let d = Dur::<Seconds>::from_secs(3600);
        let via_offset = apply_offset(anchor, &OffsetUnit::Fixed(d));
        let via_dur = anchor + d;
        assert_eq!(via_offset.raw(), via_dur.raw());
    }

    // ─── DateOffsetDecider — flexible_boundaries (adjust_boundaries) ─────────

    /// Run derive_cutoffs over `[min_ts, max_ts]` and return the flat boundary
    /// points, for a fixed-duration DateOffsetSplit.
    fn date_offset_boundaries(
        offsets: &[i64],
        tf_days: i64,
        anchor_days: i64,
        step_days: Option<i64>,
        flexible: bool,
        min_ts: i64,
        max_ts: i64,
    ) -> Vec<i64> {
        let day = 86400_i64;
        let offs: Vec<Dur<Seconds>> = offsets.iter().map(|o| Dur::from_raw(o * day)).collect();
        let mut d: DateOffsetDecider<Seconds> = DateOffsetDecider::new(
            offs,
            Dur::<Seconds>::from_raw(anchor_days * day),
            Some(Dur::<Seconds>::from_raw(tf_days * day)),
        )
        .with_flexible_boundaries(flexible, step_days.map(|s| Dur::from_raw(s * day)));
        let entity = Value::Int(0);
        let target = Value::Int(0);
        d.accumulate(&sample_row(&entity, Abs::from_raw(min_ts), &target));
        d.accumulate(&sample_row(&entity, Abs::from_raw(max_ts), &target));
        d.derive_cutoffs();
        d.boundaries.iter().map(|b| b.raw()).collect()
    }

    #[test]
    fn test_flex_boundaries_matches_kumo_offsets_not_grid_aligned() {
        // Ground truth from kumo `DateOffsetSplit.adjust_boundaries`
        // (standalone port): offsets [-35,-13], tf=10, anchor=0,
        // min=2024-01-01, max=2024-04-01. Offsets are NOT multiples of tf,
        // so flexible snapping shifts the val/test starts back to the grid.
        let min_ts = 1_704_067_200; // 2024-01-01
        let max_ts = 1_711_929_600; // 2024-04-01
        let flex =
            date_offset_boundaries(&[-35, -13], 10, 0, None, true, min_ts, max_ts);
        assert_eq!(
            flex,
            vec![1_704_067_200, 1_708_473_600, 1_710_201_600, 1_712_793_600]
        );
        // flexible=False keeps the raw (rigid) offset boundaries.
        let rigid =
            date_offset_boundaries(&[-35, -13], 10, 0, None, false, min_ts, max_ts);
        assert_eq!(
            rigid,
            vec![1_704_067_200, 1_708_905_600, 1_710_806_400, 1_712_793_600]
        );
        assert_ne!(flex, rigid, "flexible must differ from rigid here");
    }

    #[test]
    fn test_flex_boundaries_matches_kumo_with_anchor_and_step() {
        // kumo ground truth: offsets [-30,-10], tf=10, anchor=2,
        // timeframe_step=5, min=2024-01-01, max=2024-04-01.
        let min_ts = 1_704_067_200; // 2024-01-01
        let max_ts = 1_711_929_600; // 2024-04-01
        let flex =
            date_offset_boundaries(&[-30, -10], 10, 2, Some(5), true, min_ts, max_ts);
        assert_eq!(
            flex,
            vec![1_704_067_200, 1_709_337_600, 1_711_065_600, 1_712_966_400]
        );
    }

    #[test]
    fn test_flex_boundaries_grid_aligned_is_noop() {
        // When offsets are already multiples of tf, flexible snapping is a
        // no-op — flexible and rigid agree (matches kumo). offsets [-540,-300],
        // tf=60, min=2000-06-01, max=2014-05-24.
        let min_ts = 959_817_600;
        let max_ts = 1_400_889_600;
        let flex =
            date_offset_boundaries(&[-540, -300], 60, 0, None, true, min_ts, max_ts);
        let rigid =
            date_offset_boundaries(&[-540, -300], 60, 0, None, false, min_ts, max_ts);
        assert_eq!(flex, rigid);
        assert_eq!(
            flex,
            vec![959_817_600, 1_354_233_600, 1_374_969_600, 1_406_073_600]
        );
    }

    #[test]
    fn test_flex_boundaries_rejects_empty_train_window_like_kumo() {
        // Kumo DateOffsetSplit defaults to flexible_boundaries=True and
        // raises when its grid snapping consumes the first train window.
        //
        // min=0, max=20, offsets=[-14,-7], tf=10 produces raw boundaries
        // [0,6,13,30] and snapped boundaries [0,0,10,30]. No timestamp can
        // satisfy both ts >= 0 and ts + 10 <= 0, so publishing this split
        // would silently create an empty training set.
        let mut d: DateOffsetDecider<Seconds> = DateOffsetDecider::new(
            vec![Dur::from_raw(-14), Dur::from_raw(-7)],
            Dur::zero(),
            Some(Dur::from_raw(10)),
        )
        .with_flexible_boundaries(true, None);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        d.accumulate(&sample_row(&entity, Abs::from_raw(0), &target));
        d.accumulate(&sample_row(&entity, Abs::from_raw(20), &target));
        d.derive_cutoffs();

        assert_eq!(
            d.boundaries.iter().map(|b| b.raw()).collect::<Vec<_>>(),
            vec![0, 0, 10, 30],
        );
        let err = d.validate_derived_cutoffs().unwrap_err();
        assert!(matches!(err, TTGenError::InvalidPlan(_)));
        assert!(err.to_string().contains("Not enough data for the split"));
    }

    #[test]
    fn test_flex_boundaries_default_is_flexible() {
        // The decider built straight from `new` defaults to rigid (the
        // builder opt-in is applied by the executor); but the executor's
        // production default IS flexible. Assert the builder toggles it and
        // that the un-toggled decider stays rigid (back-compat).
        let min_ts = 1_704_067_200;
        let max_ts = 1_711_929_600;
        let day = 86400_i64;
        let mut rigid_default: DateOffsetDecider<Seconds> = DateOffsetDecider::new(
            vec![Dur::from_raw(-35 * day), Dur::from_raw(-13 * day)],
            Dur::<Seconds>::zero(),
            Some(Dur::from_raw(10 * day)),
        );
        let entity = Value::Int(0);
        let target = Value::Int(0);
        rigid_default.accumulate(&sample_row(&entity, Abs::from_raw(min_ts), &target));
        rigid_default.accumulate(&sample_row(&entity, Abs::from_raw(max_ts), &target));
        rigid_default.derive_cutoffs();
        assert_eq!(
            rigid_default.boundaries.iter().map(|b| b.raw()).collect::<Vec<_>>(),
            vec![1_704_067_200, 1_708_905_600, 1_710_806_400, 1_712_793_600]
        );
    }

    #[test]
    fn test_flex_boundaries_per_row_decide_parity() {
        // The snapped boundaries must drive per-row stage assignment: a ts in
        // the flexible val window [b1,b2) is VAL, not TRAIN. offsets [-35,-13]
        // tf=10 anchor=0; flex b=[.., 1_708_473_600, 1_710_201_600, ..].
        let min_ts = 1_704_067_200;
        let max_ts = 1_711_929_600;
        let day = 86400_i64;
        let mut d: DateOffsetDecider<Seconds> = DateOffsetDecider::new(
            vec![Dur::from_raw(-35 * day), Dur::from_raw(-13 * day)],
            Dur::<Seconds>::zero(),
            Some(Dur::from_raw(10 * day)),
        )
        .with_flexible_boundaries(true, None);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        d.accumulate(&sample_row(&entity, Abs::from_raw(min_ts), &target));
        d.accumulate(&sample_row(&entity, Abs::from_raw(max_ts), &target));
        d.derive_cutoffs();
        // ts just inside the flex val window (>= b1, +tf <= b2) → VAL(1).
        let val_ts = 1_708_473_600 + day; // b1 + 1 day
        assert_eq!(d.decide(&sample_row(&entity, Abs::from_raw(val_ts), &target)), 1);
        // Same ts under the rigid boundaries would still be TRAIN(0) since the
        // rigid b1 = 1_708_905_600 > val_ts. Confirm the flex snap moved it.
        assert!(val_ts < 1_708_905_600);
    }

    #[test]
    fn test_temporal_decider_uses_quantile_at_directly() {
        // Cleanup test: TemporalDecider.derive_cutoffs() now
        // queries the digest via `quantile_at(q)` directly. With direct
        // queries, the cutoff for q=0.5 over uniform [0..1000] should be
        // within ~0.5% of the empirical median (= 499.5). The OLD path
        // (interpolating between q25/q50/q75 published triple) was
        // accurate at q=0.5 but worse at off-quartile probes — drop the
        // workaround once the test passes.
        let mut d: TemporalDecider<Seconds> = TemporalDecider::new(vec![0.5, 0.5]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for ts in 0..1000_i64 {
            d.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        d.derive_cutoffs();
        let cutoff_raw = d.cutoffs[0].raw();
        // Direct quantile_at(0.5) on a 200-centroid t-digest of 0..1000:
        // tighter than the workaround's 50-row tolerance — within ~5 rows.
        assert!(
            (cutoff_raw - 500).abs() < 5,
            "expected ~500 (within 0.5%), got {cutoff_raw}"
        );
    }

    #[test]
    fn test_temporal_decider_three_way_tighter_off_quartile() {
        // Cleanup test: off-quartile probe q ∈ {0.6, 0.8}.
        // The OLD workaround interpolated linearly between q50/q75 for
        // q in (0.5, 0.75) — large error at off-quartile. With
        // `quantile_at(0.6)` direct, accuracy improves substantially.
        let mut d: TemporalDecider<Seconds> = TemporalDecider::new(vec![0.6, 0.2, 0.2]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for ts in 0..1000_i64 {
            d.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        d.derive_cutoffs();
        // cutoffs at q=0.6 and q=0.8.
        assert_eq!(d.cutoffs.len(), 2);
        let c0 = d.cutoffs[0].raw();
        let c1 = d.cutoffs[1].raw();
        // Within ~1% (10 rows) of the empirical 600/800.
        assert!((c0 - 600).abs() < 15, "cutoff[0] expected ~600, got {c0}");
        assert!((c1 - 800).abs() < 15, "cutoff[1] expected ~800, got {c1}");
    }

    // ─── ApproxDateOffsetDecider ───────────────────────────────────────────

    #[test]
    fn test_approx_date_offset_decider_basic() {
        let mut d: ApproxDateOffsetDecider<Seconds> =
            ApproxDateOffsetDecider::new(vec![0.6, 0.4]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for ts in [10_i64, 20, 30, 40, 50] {
            d.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        d.derive_cutoffs();
        // 5 unique timestamps; 60% threshold → 3 rows in train, 2 in val.
        let mut counts = [0u64; 3];
        for ts in [10_i64, 20, 30, 40, 50] {
            let stage = d.decide(&sample_row(&entity, Abs::from_raw(ts), &target));
            counts[stage as usize] += 1;
        }
        assert_eq!(counts[0], 3);
        assert_eq!(counts[1], 2);
        assert_eq!(counts[2], 0);
    }

    #[test]
    fn test_approx_date_offset_decider_matches_pybind_reference() {
        // Independent rust port of the pybind-side logic — implemented
        // inline so we don't pull a pybind dependency. Same input, same
        // output expected.
        fn pybind_reference(timestamps_ns: Vec<i64>, ratio: Vec<f64>) -> Vec<u8> {
            if timestamps_ns.is_empty() {
                return Vec::new();
            }
            let mut counts: HashMap<i64, usize> = HashMap::new();
            for ts in &timestamps_ns {
                *counts.entry(*ts).or_insert(0) += 1;
            }
            let mut groups: Vec<(i64, usize)> = counts.into_iter().collect();
            groups.sort_by_key(|a| a.0);
            let total_rows: usize = groups.iter().map(|(_, c)| *c).sum();
            let mut cum: Vec<f64> = Vec::with_capacity(groups.len());
            let mut run = 0usize;
            for (_, c) in &groups {
                run += *c;
                cum.push(run as f64 / total_rows as f64);
            }
            let num_split = ratio.len();
            let n = groups.len();
            let mut train_val_idx = cum
                .iter()
                .position(|v| *v >= ratio[0])
                .unwrap_or(n.saturating_sub(1));
            train_val_idx = train_val_idx.min(n.saturating_sub(num_split));
            let mut val_test_idx: Option<usize> = None;
            if num_split == 3 {
                let mut idx = cum
                    .iter()
                    .position(|v| *v >= (ratio[0] + ratio[1]))
                    .unwrap_or(n.saturating_sub(1));
                idx = idx.min(n.saturating_sub(2));
                idx = idx.max(train_val_idx.saturating_add(1));
                val_test_idx = Some(idx);
            }
            let mut stage_by_ts: HashMap<i64, u8> = HashMap::new();
            for (i, (ts, _)) in groups.iter().enumerate() {
                let stage = if i <= train_val_idx {
                    0u8
                } else if num_split == 2 {
                    1u8
                } else {
                    let vt = val_test_idx.unwrap();
                    if i <= vt {
                        1u8
                    } else {
                        2u8
                    }
                };
                stage_by_ts.insert(*ts, stage);
            }
            timestamps_ns
                .iter()
                .map(|ts| *stage_by_ts.get(ts).unwrap_or(&255))
                .collect()
        }

        let raws: Vec<i64> = (1..=20).collect();
        let mut d: ApproxDateOffsetDecider<Seconds> =
            ApproxDateOffsetDecider::new(vec![0.5, 0.3, 0.2]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for &ts in &raws {
            d.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        d.derive_cutoffs();

        let expected = pybind_reference(raws.clone(), vec![0.5, 0.3, 0.2]);
        for (i, &ts) in raws.iter().enumerate() {
            let by_dec = d.decide(&sample_row(&entity, Abs::from_raw(ts), &target));
            assert_eq!(by_dec, expected[i], "ts={ts} idx={i}");
        }
    }

    #[test]
    fn test_approx_date_offset_decider_merge_combines_counts() {
        let mut a: ApproxDateOffsetDecider<Seconds> =
            ApproxDateOffsetDecider::new(vec![0.5, 0.5]);
        let mut b: ApproxDateOffsetDecider<Seconds> =
            ApproxDateOffsetDecider::new(vec![0.5, 0.5]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for ts in [10_i64, 20] {
            a.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        for ts in [10_i64, 30] {
            b.accumulate(&sample_row(&entity, Abs::from_raw(ts), &target));
        }
        a.merge(&b);
        // Counts: ts=10 → 2 (one from a, one from b), ts=20 → 1, ts=30 → 1.
        assert_eq!(*a.counts.get(&Abs::<Seconds>::from_raw(10)).unwrap(), 2);
        assert_eq!(*a.counts.get(&Abs::<Seconds>::from_raw(20)).unwrap(), 1);
        assert_eq!(*a.counts.get(&Abs::<Seconds>::from_raw(30)).unwrap(), 1);
    }

    // ─── Cross-decider helper-function tests ───────────────────────────────

    #[test]
    fn test_ratio_to_cumulative_cutoffs_three_way() {
        let c = ratio_to_cumulative_cutoffs(&[0.7, 0.2, 0.1]);
        assert_eq!(c.len(), 2);
        assert!((c[0] - 0.7).abs() < 1e-9);
        assert!((c[1] - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_ratio_to_cumulative_cutoffs_two_way() {
        let c = ratio_to_cumulative_cutoffs(&[0.8, 0.2]);
        assert_eq!(c.len(), 1);
        assert!((c[0] - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_bucket_by_cutoffs_three_way() {
        let cutoffs = vec![0.7, 0.9];
        assert_eq!(bucket_by_cutoffs(0.5, &cutoffs), 0);
        assert_eq!(bucket_by_cutoffs(0.8, &cutoffs), 1);
        assert_eq!(bucket_by_cutoffs(0.95, &cutoffs), 2);
    }

    #[test]
    fn test_splitmix64_deterministic() {
        assert_eq!(splitmix64(42), splitmix64(42));
        assert_ne!(splitmix64(42), splitmix64(43));
    }

    #[test]
    fn test_hash_to_unit_interval_in_bounds() {
        for ts in 0..100i64 {
            let entity = Value::Int(ts);
            let v = hash_entity_ts_to_unit_interval(&entity, ts, 7);
            assert!((0.0..1.0).contains(&v), "ts={ts} v={v}");
        }
    }

    // ─── ColumnDecider tests ────────────────────────────────────────────────

    use arrow_array::{Float64Array, StringArray};

    fn batch_with_string_column(values: &[Option<&str>], col_name: &str) -> RecordBatch {
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;
        let arr = StringArray::from(values.to_vec());
        let schema = Arc::new(Schema::new(vec![Field::new(col_name, DataType::Utf8, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
    }

    fn batch_with_float_column(values: &[Option<f64>], col_name: &str) -> RecordBatch {
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;
        let arr = Float64Array::from(values.to_vec());
        let schema = Arc::new(Schema::new(vec![Field::new(col_name, DataType::Float64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
    }

    fn batch_with_int_column(values: &[Option<i64>], col_name: &str) -> RecordBatch {
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;
        let arr = Int64Array::from(values.to_vec());
        let schema = Arc::new(Schema::new(vec![Field::new(col_name, DataType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
    }

    #[test]
    fn test_value_to_string_key_variants() {
        assert_eq!(value_to_string_key(&Value::Str("US".to_string())), "US");
        assert_eq!(value_to_string_key(&Value::Int(42)), "42");
        assert_eq!(value_to_string_key(&Value::Bool(true)), "true");
        assert_eq!(value_to_string_key(&Value::Bool(false)), "false");
        // Lists are not valid split-column values.
        assert_eq!(
            value_to_string_key(&Value::List(vec![Value::Int(1)])),
            ""
        );
    }

    #[test]
    fn test_value_to_f64_variants() {
        assert_eq!(value_to_f64(&Value::Int(7)), Some(7.0));
        assert_eq!(value_to_f64(&Value::Float(3.5)), Some(3.5));
        assert_eq!(value_to_f64(&Value::Str("x".to_string())), None);
        assert_eq!(value_to_f64(&Value::Bool(true)), None);
    }

    // (α) ColumnDeciderManual
    #[test]
    fn test_column_decider_manual_decides_by_set_membership() {
        let d = ColumnDeciderManual::new(
            "country".to_string(),
            vec!["US".to_string(), "UK".to_string()],
            vec!["FR".to_string()],
            vec!["DE".to_string()],
        );
        let entity = Value::Int(0);
        let target = Value::Int(0);
        let us = Value::Str("US".to_string());
        let uk = Value::Str("UK".to_string());
        let fr = Value::Str("FR".to_string());
        let de = Value::Str("DE".to_string());
        let row_us = sample_row_with_split(&entity, Abs::from_raw(0), &target, &us);
        let row_uk = sample_row_with_split(&entity, Abs::from_raw(0), &target, &uk);
        let row_fr = sample_row_with_split(&entity, Abs::from_raw(0), &target, &fr);
        let row_de = sample_row_with_split(&entity, Abs::from_raw(0), &target, &de);
        assert_eq!(<ColumnDeciderManual as SplitDecider<Seconds>>::decide(&d, &row_us), 0);
        assert_eq!(<ColumnDeciderManual as SplitDecider<Seconds>>::decide(&d, &row_uk), 0);
        assert_eq!(<ColumnDeciderManual as SplitDecider<Seconds>>::decide(&d, &row_fr), 1);
        assert_eq!(<ColumnDeciderManual as SplitDecider<Seconds>>::decide(&d, &row_de), 2);
    }

    #[test]
    fn test_column_decider_manual_unused_for_unmatched_value() {
        let d = ColumnDeciderManual::new(
            "country".to_string(),
            vec!["US".to_string()],
            vec![],
            vec![],
        );
        let entity = Value::Int(0);
        let target = Value::Int(0);
        let xx = Value::Str("XX".to_string());
        let row_xx = sample_row_with_split(&entity, Abs::from_raw(0), &target, &xx);
        assert_eq!(
            <ColumnDeciderManual as SplitDecider<Seconds>>::decide(&d, &row_xx),
            UNUSED_STAGE
        );
        // Row with no split_column_value also lands UNUSED.
        let row_none = sample_row(&entity, Abs::from_raw(0), &target);
        assert_eq!(
            <ColumnDeciderManual as SplitDecider<Seconds>>::decide(&d, &row_none),
            UNUSED_STAGE
        );
    }

    #[test]
    fn test_column_decider_manual_decide_for_view_matches_decide() {
        let d = ColumnDeciderManual::new(
            "country".to_string(),
            vec!["US".to_string(), "UK".to_string()],
            vec!["FR".to_string()],
            vec!["DE".to_string()],
        );
        let batch = batch_with_string_column(
            &[Some("US"), Some("UK"), Some("FR"), Some("DE"), Some("XX"), None],
            "country",
        );
        let view = <ColumnDeciderManual as SplitDecider<Seconds>>::batch_view(&d, &batch, 0).unwrap();
        let expected = [0u8, 0, 1, 2, UNUSED_STAGE, UNUSED_STAGE];
        for (i, e) in expected.iter().enumerate() {
            assert_eq!(
                <ColumnDeciderManual as SplitDecider<Seconds>>::decide_for_view(&d, &view, i),
                *e,
                "row {i}"
            );
        }
    }

    #[test]
    fn test_column_decider_manual_required_columns_carries_name() {
        let d = ColumnDeciderManual::new("user_country".to_string(), vec![], vec![], vec![]);
        assert_eq!(
            <ColumnDeciderManual as SplitDecider<Seconds>>::required_columns(&d),
            vec!["user_country".to_string()]
        );
    }

    #[test]
    fn test_column_decider_manual_int_value_via_string_key() {
        // Manual sets contain the stringified form; an Int row matches when
        // its string repr is in the train set.
        let d = ColumnDeciderManual::new(
            "level".to_string(),
            vec!["1".to_string(), "2".to_string()],
            vec!["3".to_string()],
            vec![],
        );
        let entity = Value::Int(0);
        let target = Value::Int(0);
        let v1 = Value::Int(1);
        let v3 = Value::Int(3);
        let v9 = Value::Int(9);
        assert_eq!(
            <ColumnDeciderManual as SplitDecider<Seconds>>::decide(
                &d,
                &sample_row_with_split(&entity, Abs::from_raw(0), &target, &v1)
            ),
            0
        );
        assert_eq!(
            <ColumnDeciderManual as SplitDecider<Seconds>>::decide(
                &d,
                &sample_row_with_split(&entity, Abs::from_raw(0), &target, &v3)
            ),
            1
        );
        assert_eq!(
            <ColumnDeciderManual as SplitDecider<Seconds>>::decide(
                &d,
                &sample_row_with_split(&entity, Abs::from_raw(0), &target, &v9)
            ),
            UNUSED_STAGE
        );
    }

    // (β) ColumnDeciderNumericExplicit
    #[test]
    fn test_column_decider_numeric_explicit_at_boundaries() {
        // cutoffs=[0.3, 0.7] → bucket: v<0.3 → 0; 0.3<=v<0.7 → 1; v>=0.7 → 2.
        let d = ColumnDeciderNumericExplicit::new("score".to_string(), vec![0.3, 0.7]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for (v, expected) in [
            (0.0, 0),
            (0.29, 0),
            (0.3, 1),
            (0.5, 1),
            (0.69, 1),
            (0.7, 2),
            (1.0, 2),
        ] {
            let val = Value::Float(v);
            let row = sample_row_with_split(&entity, Abs::from_raw(0), &target, &val);
            assert_eq!(
                <ColumnDeciderNumericExplicit as SplitDecider<Seconds>>::decide(&d, &row),
                expected,
                "v={v}"
            );
        }
    }

    #[test]
    fn test_column_decider_numeric_explicit_sorts_cutoffs() {
        // Constructor should sort unsorted input.
        let d = ColumnDeciderNumericExplicit::new("score".to_string(), vec![0.7, 0.3]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        let val = Value::Float(0.5);
        let row = sample_row_with_split(&entity, Abs::from_raw(0), &target, &val);
        // 0.5 should land in stage 1 (between 0.3 and 0.7), not at the
        // boundary of an unsorted [0.7, 0.3] which would put it in stage 2.
        assert_eq!(
            <ColumnDeciderNumericExplicit as SplitDecider<Seconds>>::decide(&d, &row),
            1
        );
    }

    #[test]
    fn test_column_decider_numeric_explicit_int_column_via_view() {
        let d = ColumnDeciderNumericExplicit::new("age".to_string(), vec![10.0, 20.0]);
        let batch = batch_with_int_column(&[Some(5), Some(10), Some(15), Some(20), None], "age");
        let view =
            <ColumnDeciderNumericExplicit as SplitDecider<Seconds>>::batch_view(&d, &batch, 0)
                .unwrap();
        // 5 → 0; 10 → 1; 15 → 1; 20 → 2; null → UNUSED.
        let expected = [0u8, 1, 1, 2, UNUSED_STAGE];
        for (i, e) in expected.iter().enumerate() {
            assert_eq!(
                <ColumnDeciderNumericExplicit as SplitDecider<Seconds>>::decide_for_view(
                    &d, &view, i
                ),
                *e,
                "row {i}"
            );
        }
    }

    // (γ) ColumnDeciderRatioNumeric
    #[test]
    fn test_column_decider_ratio_numeric_accumulate_then_derive_then_decide() {
        let mut d = ColumnDeciderRatioNumeric::new("score".to_string(), vec![0.5, 0.5]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        // Feed values 0..1000 so the median is ~500.
        for v in 0..1000_i64 {
            let val = Value::Int(v);
            let row = sample_row_with_split(&entity, Abs::from_raw(0), &target, &val);
            <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::accumulate(&mut d, &row);
        }
        <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::derive_cutoffs(&mut d);
        assert_eq!(d.cutoffs.len(), 1, "two-way ratio → one cutoff");
        // Cutoff should land near the median (~500); t-digest tolerance is
        // a few percent, plus the quantile-via-public-api shim's anchor
        // approximation.
        let c = d.cutoffs[0];
        assert!(
            (c - 500.0).abs() < 100.0,
            "expected cutoff near 500, got {c}"
        );
        // Per-row decide: v=100 → 0, v=900 → 1.
        let v100 = Value::Int(100);
        let v900 = Value::Int(900);
        let row100 = sample_row_with_split(&entity, Abs::from_raw(0), &target, &v100);
        let row900 = sample_row_with_split(&entity, Abs::from_raw(0), &target, &v900);
        assert_eq!(
            <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::decide(&d, &row100),
            0
        );
        assert_eq!(
            <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::decide(&d, &row900),
            1
        );
    }

    #[test]
    fn test_column_decider_ratio_numeric_decide_for_view_matches_decide() {
        let mut d = ColumnDeciderRatioNumeric::new("score".to_string(), vec![0.5, 0.5]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for v in 0..100_i64 {
            let val = Value::Int(v);
            let row = sample_row_with_split(&entity, Abs::from_raw(0), &target, &val);
            <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::accumulate(&mut d, &row);
        }
        <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::derive_cutoffs(&mut d);

        let raw_vals: Vec<Option<f64>> = (0..100_i64).map(|v| Some(v as f64)).collect();
        let batch = batch_with_float_column(&raw_vals, "score");
        let view = <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::batch_view(&d, &batch, 0)
            .unwrap();
        for i in 0..100 {
            let v = Value::Float(i as f64);
            let row = sample_row_with_split(&entity, Abs::from_raw(0), &target, &v);
            let by_view = <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::decide_for_view(
                &d, &view, i,
            );
            let by_decide =
                <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::decide(&d, &row);
            assert_eq!(by_view, by_decide, "row {i}");
        }
    }

    #[test]
    fn test_column_decider_ratio_numeric_merge() {
        let mut a = ColumnDeciderRatioNumeric::new("score".to_string(), vec![0.5, 0.5]);
        let mut b = ColumnDeciderRatioNumeric::new("score".to_string(), vec![0.5, 0.5]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        for v in 0..500_i64 {
            let val = Value::Int(v);
            let row = sample_row_with_split(&entity, Abs::from_raw(0), &target, &val);
            <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::accumulate(&mut a, &row);
        }
        for v in 500..1000_i64 {
            let val = Value::Int(v);
            let row = sample_row_with_split(&entity, Abs::from_raw(0), &target, &val);
            <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::accumulate(&mut b, &row);
        }
        <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::merge(&mut a, &b);
        <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::derive_cutoffs(&mut a);
        assert_eq!(a.cutoffs.len(), 1);
        let c = a.cutoffs[0];
        assert!((c - 500.0).abs() < 150.0, "post-merge median ~500, got {c}");
    }

    #[test]
    fn test_column_decider_ratio_numeric_required_columns() {
        let d = ColumnDeciderRatioNumeric::new("level".to_string(), vec![0.7, 0.3]);
        assert_eq!(
            <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::required_columns(&d),
            vec!["level".to_string()]
        );
    }

    #[test]
    fn test_column_decider_ratio_numeric_skips_non_numeric_values() {
        // String / Bool / List rows should be no-ops on accumulate (no
        // panic). Only numeric values feed the digest.
        let mut d = ColumnDeciderRatioNumeric::new("col".to_string(), vec![0.5, 0.5]);
        let entity = Value::Int(0);
        let target = Value::Int(0);
        let s = Value::Str("foo".to_string());
        let row_str = sample_row_with_split(&entity, Abs::from_raw(0), &target, &s);
        // Should not panic.
        <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::accumulate(&mut d, &row_str);
        // Decide on a string row → UNUSED (no f64 conversion).
        assert_eq!(
            <ColumnDeciderRatioNumeric as SplitDecider<Seconds>>::decide(&d, &row_str),
            UNUSED_STAGE
        );
    }

    #[test]
    fn test_column_decider_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ColumnDeciderManual>();
        assert_send_sync::<ColumnDeciderNumericExplicit>();
        assert_send_sync::<ColumnDeciderRatioNumeric>();
    }
}
