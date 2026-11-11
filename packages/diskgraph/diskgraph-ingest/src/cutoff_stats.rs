// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact column statistics over the rows of a table created at or before a
//! cutoff, published under `<index>/col_stats/<cutoff_id>/`.
//!
//! Ingest observes the rows it stores, a built index is rescanned, and Arrow
//! arrays are read directly; all three feed the same [`TableStats`], whose
//! merges are exact and order-independent, so every route writes the same bytes.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{BuildHasher, Hash, RandomState};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Context, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, TimestampSecondArray};
use arrow_schema::DataType;
use diskgraph_core::time::TimeUnit;
use diskgraph_index::graph::{temporal_value_in_clock, write_durable, GraphIndexReader};
use diskgraph_rowpack::{
    ColumnValueObserver, FloatListView, IngestedType, IntListView, RowAccess, RowAccessHints,
    RowAccessStats, StringListView, ValueRef,
};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{ToPrimitive, Zero};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::{json, Value as JsonValue};
use sha2::{Digest, Sha256};

use crate::col_stats::{ValueBins, MAX_CATEGORIES};
use crate::index_metadata::{index_id, read_dataset_metadata, INDEX_METADATA_FILENAME};

const SCHEMA: &str = "diskgraph.col_stats_extension.v1";
const SPEC_VERSION: u32 = 1;
const STATS_DIR: &str = "col_stats";
const SCAN_CHUNK_NODES: u64 = 1 << 16;
const BINS_PER_OCTAVE: u32 = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stat {
    Counts,
    Moments,
    Bins,
}

impl Stat {
    pub fn parse(name: &str) -> Option<Self> {
        [Stat::Counts, Stat::Moments, Stat::Bins]
            .into_iter()
            .find(|stat| stat.name() == name)
    }

    pub fn name(self) -> &'static str {
        match self {
            Stat::Counts => "counts",
            Stat::Moments => "moments",
            Stat::Bins => "bins",
        }
    }
}

/// The statistics ingest computes in the same pass, one file set per cutoff.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngestRequests {
    pub cutoffs: Vec<Option<i64>>,
    pub requests: Vec<(String, String, Stat)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ValueType {
    Bool,
    Int64,
    Uint64,
    Float64,
    String,
    Date,
    Timestamp,
}

impl ValueType {
    fn of(ingested: IngestedType) -> Result<Self> {
        Ok(match ingested {
            IngestedType::Bool => Self::Bool,
            IngestedType::Int8
            | IngestedType::Int16
            | IngestedType::Int32
            | IngestedType::Int64 => Self::Int64,
            IngestedType::UInt8
            | IngestedType::UInt16
            | IngestedType::UInt32
            | IngestedType::UInt64 => Self::Uint64,
            IngestedType::Float32 | IngestedType::Float64 => Self::Float64,
            IngestedType::String => Self::String,
            IngestedType::Date => Self::Date,
            IngestedType::Timestamp => Self::Timestamp,
            other => bail!("column statistics are unsupported for {other:?} columns"),
        })
    }

    fn supports(self, stat: Stat) -> bool {
        stat == Stat::Counts || matches!(self, Self::Int64 | Self::Uint64 | Self::Float64)
    }

    fn number(self, key: i64) -> f64 {
        match self {
            Self::Uint64 => key as u64 as f64,
            Self::Float64 => f64::from_bits(key as u64),
            _ => key as f64,
        }
    }

    fn compare(self, a: i64, b: i64) -> Ordering {
        match self {
            Self::Uint64 => (a as u64).cmp(&(b as u64)),
            Self::Float64 => self.number(a).total_cmp(&self.number(b)),
            _ => a.cmp(&b),
        }
    }

    fn json(self, key: i64) -> JsonValue {
        match self {
            Self::Bool => json!(key != 0),
            Self::Uint64 => json!(key as u64),
            Self::Float64 => json!(self.number(key)),
            _ => json!(key),
        }
    }
}

/// One row's value of a column: integers, booleans, dates and timestamps as
/// `Int` (unsigned values bit-cast), floats as `Float`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Value<'a> {
    Null,
    Int(i64),
    Float(f64),
    Str(&'a str),
}

impl<'a> From<ValueRef<'a>> for Value<'a> {
    fn from(value: ValueRef<'a>) -> Self {
        match value {
            ValueRef::Null => Value::Null,
            ValueRef::Bool(v) => Value::Int(v.into()),
            ValueRef::Int64(v) | ValueRef::Timestamp(v) => Value::Int(v),
            ValueRef::Int32(v) | ValueRef::Date(v) => Value::Int(v.into()),
            ValueRef::UInt64(v) => Value::Int(v as i64),
            ValueRef::UInt32(v) => Value::Int(v.into()),
            ValueRef::Float64(v) => Value::Float(v),
            ValueRef::Float32(v) => Value::Float(v.into()),
            ValueRef::String(v) => Value::Str(v),
            other => unreachable!("column statistics read an unsupported value {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Exact sums
// ---------------------------------------------------------------------------

/// Bit positions above 2^-1074 that a sum of doubles and a sum of their
/// squares can reach, before carries.
const SUM_BITS: usize = 2045 + 53;
const SQUARE_BITS: usize = 4090 + 106;
const SUM_LSB: i64 = -1074;
const SQUARE_LSB: i64 = 2 * SUM_LSB;

/// An exact integer sum in 32-bit digits held by i64 limbs, so carries can
/// wait until `normalize`.
#[derive(Clone)]
struct ExactSum {
    limbs: Vec<i64>,
    adds: u32,
}

impl ExactSum {
    fn new(bits: usize) -> Self {
        Self {
            limbs: vec![0; bits / 32 + 2],
            adds: 0,
        }
    }

    /// Adds `±mantissa · 2^offset`; every limb moves by less than 2^33.
    fn add(&mut self, mut mantissa: u128, offset: usize, negative: bool) {
        let (mut idx, shift) = (offset / 32, offset % 32);
        while mantissa != 0 {
            let wide = (mantissa as u32 as u64) << shift;
            let (low, high) = ((wide & 0xFFFF_FFFF) as i64, (wide >> 32) as i64);
            if negative {
                self.limbs[idx] -= low;
                self.limbs[idx + 1] -= high;
            } else {
                self.limbs[idx] += low;
                self.limbs[idx + 1] += high;
            }
            mantissa >>= 32;
            idx += 1;
        }
        self.adds += 1;
        if self.adds == 1 << 29 {
            self.normalize();
        }
    }

    fn normalize(&mut self) {
        let (top, digits) = self.limbs.split_last_mut().expect("limbs");
        let mut carry = 0;
        for limb in digits {
            let value = *limb + carry;
            *limb = value & 0xFFFF_FFFF;
            carry = value >> 32;
        }
        *top += carry;
        self.adds = 0;
    }

    fn merge(&mut self, other: &Self) {
        self.normalize();
        for (limb, other) in self.limbs.iter_mut().zip(&other.limbs) {
            *limb += other;
        }
        self.normalize();
    }

    fn value(&self) -> BigInt {
        self.limbs
            .iter()
            .rev()
            .fold(BigInt::zero(), |acc, &limb| (acc << 32) + limb)
    }
}

/// `value / n · 2^lsb` rounded once to the nearest double, ties to even.
fn round_quotient(value: &BigInt, n: u64, lsb: i64) -> f64 {
    let scaled = value.magnitude() << 128u32;
    // The remainder sits far below the rounding bit, so a sticky bit stands in for it.
    let magnitude = (&scaled / n) | BigUint::from(u8::from(!(&scaled % n).is_zero()));
    let rounded = round_scaled(&magnitude, lsb - 128);
    if value.sign() == Sign::Minus {
        -rounded
    } else {
        rounded
    }
}

/// `magnitude · 2^lsb` rounded once to the nearest double, ties to even.
fn round_scaled(magnitude: &BigUint, lsb: i64) -> f64 {
    let Some(trailing) = magnitude.trailing_zeros() else {
        return 0.0;
    };
    let low = (magnitude.bits() as i64 - 53).max(-1074 - lsb).max(0) as u64;
    let mut mantissa = (magnitude >> low).to_u64().expect("53-bit mantissa");
    if low > 0 && magnitude.bit(low - 1) && (mantissa & 1 == 1 || trailing < low - 1) {
        mantissa += 1;
    }
    let scale = match low as i64 + lsb {
        1024.. => f64::INFINITY,
        exponent @ -1022.. => f64::from_bits(((exponent + 1023) as u64) << 52),
        exponent => f64::from_bits(1 << (exponent + 1074)),
    };
    mantissa as f64 * scale
}

/// Exact Σx and Σx² of finite doubles.
#[derive(Clone)]
struct Moments {
    sum: ExactSum,
    squares: ExactSum,
}

impl Default for Moments {
    fn default() -> Self {
        Self {
            sum: ExactSum::new(SUM_BITS),
            squares: ExactSum::new(SQUARE_BITS),
        }
    }
}

impl Moments {
    fn add(&mut self, x: f64) {
        let bits = x.to_bits();
        let biased = (bits >> 52 & 0x7FF) as usize;
        let fraction = bits & ((1 << 52) - 1);
        let (mantissa, offset) = if biased == 0 {
            (fraction, 0)
        } else {
            (fraction | 1 << 52, biased - 1)
        };
        let mantissa = u128::from(mantissa);
        self.sum.add(mantissa, offset, x < 0.0);
        self.squares.add(mantissa * mantissa, 2 * offset, false);
    }

    fn merge(&mut self, other: &Self) {
        self.sum.merge(&other.sum);
        self.squares.merge(&other.squares);
    }

    /// The mean and the population sum of squared deviations, `(n·Σx² − (Σx)²) / n`,
    /// each rounded once.
    fn mean_m2(&self, n: u64) -> (f64, f64) {
        if n == 0 {
            return (0.0, 0.0);
        }
        let (sum, squares) = (self.sum.value(), self.squares.value());
        let deviations = squares * n - &sum * &sum;
        (
            round_quotient(&sum, n, SUM_LSB),
            round_quotient(&deviations, n, SQUARE_LSB),
        )
    }
}

// ---------------------------------------------------------------------------
// Per-column statistics
// ---------------------------------------------------------------------------

/// Exact counts of at most `MAX_CATEGORIES` distinct values.
#[derive(Clone)]
struct CountMap<K> {
    counts: gxhash::HashMap<K, u64>,
    truncated: bool,
}

impl<K> Default for CountMap<K> {
    fn default() -> Self {
        Self {
            counts: gxhash::HashMap::default(),
            truncated: false,
        }
    }
}

impl<K: Hash + Eq> CountMap<K> {
    fn add<Q: Hash + Eq + ToOwned<Owned = K> + ?Sized>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
    {
        if let Some(count) = self.counts.get_mut(key) {
            *count += 1;
        } else if (self.counts.len() as u64) < MAX_CATEGORIES {
            self.counts.insert(key.to_owned(), 1);
        } else {
            self.truncated = true;
        }
    }

    fn merge(&mut self, other: &Self, order: impl Fn(&K, &K) -> Ordering)
    where
        K: Clone,
    {
        self.truncated |= other.truncated;
        for (key, count) in &other.counts {
            *self.counts.entry(key.clone()).or_default() += count;
        }
        if self.counts.len() as u64 > MAX_CATEGORIES {
            let mut ranked: Vec<_> = std::mem::take(&mut self.counts).into_iter().collect();
            rank(&mut ranked, order);
            self.counts = ranked.into_iter().take(MAX_CATEGORIES as usize).collect();
            self.truncated = true;
        }
    }

    fn ranked(&self, order: impl Fn(&K, &K) -> Ordering) -> Vec<(&K, u64)> {
        let mut ranked: Vec<_> = self
            .counts
            .iter()
            .map(|(key, count)| (key, *count))
            .collect();
        rank(&mut ranked, |a: &&K, b: &&K| order(a, b));
        ranked
    }
}

/// Count descending, then value ascending.
fn rank<K>(entries: &mut [(K, u64)], order: impl Fn(&K, &K) -> Ordering) {
    entries.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| order(&a.0, &b.0)));
}

#[derive(Clone, Default)]
struct Bins {
    bins: ValueBins<45>,
    zeros: u64,
    range: Option<(i64, i64)>,
}

#[derive(Clone)]
struct ColumnStats {
    value_type: ValueType,
    count: u64,
    nulls: u64,
    non_finite: u64,
    keys: Option<CountMap<i64>>,
    strings: Option<CountMap<String>>,
    moments: Option<Moments>,
    bins: Option<Bins>,
}

#[derive(Serialize)]
struct CountsData {
    count: u64,
    nulls: u64,
    non_finite: u64,
    truncated: bool,
    values: Vec<(JsonValue, u64)>,
}

#[derive(Serialize)]
struct MomentsData {
    count: u64,
    nulls: u64,
    non_finite: u64,
    mean: f64,
    m2: f64,
}

#[derive(Serialize)]
struct BinsData {
    count: u64,
    nulls: u64,
    non_finite: u64,
    zeros: u64,
    min: Option<JsonValue>,
    max: Option<JsonValue>,
    bins_per_octave: u32,
    bins: Vec<(i32, u64)>,
}

impl ColumnStats {
    fn new(value_type: ValueType, stats: &[Stat]) -> Self {
        let counts = stats.contains(&Stat::Counts);
        Self {
            value_type,
            count: 0,
            nulls: 0,
            non_finite: 0,
            keys: (counts && value_type != ValueType::String).then(CountMap::default),
            strings: (counts && value_type == ValueType::String).then(CountMap::default),
            moments: stats.contains(&Stat::Moments).then(Moments::default),
            bins: stats.contains(&Stat::Bins).then(Bins::default),
        }
    }

    fn add(&mut self, value: Value<'_>) {
        let key = match value {
            Value::Null => {
                self.nulls += 1;
                return;
            }
            Value::Float(x) if !x.is_finite() => {
                self.non_finite += 1;
                return;
            }
            Value::Str(s) => {
                self.count += 1;
                if let Some(strings) = &mut self.strings {
                    strings.add(s);
                }
                return;
            }
            // Matches -0.0 too, so both zeros share the key of 0.0.
            Value::Float(0.0) => 0,
            Value::Float(x) => x.to_bits() as i64,
            Value::Int(v) => v,
        };
        self.count += 1;
        if let Some(keys) = &mut self.keys {
            keys.add(&key);
        }
        let x = self.value_type.number(key);
        if let Some(moments) = &mut self.moments {
            moments.add(x);
        }
        if let Some(bins) = &mut self.bins {
            bins.bins.add(x);
            bins.zeros += u64::from(x == 0.0);
            let value_type = self.value_type;
            bins.range = Some(match bins.range {
                None => (key, key),
                Some((min, max)) => (
                    std::cmp::min_by(min, key, |a, b| value_type.compare(*a, *b)),
                    std::cmp::max_by(max, key, |a, b| value_type.compare(*a, *b)),
                ),
            });
        }
    }

    fn merge(&mut self, other: &Self) {
        let value_type = self.value_type;
        self.count += other.count;
        self.nulls += other.nulls;
        self.non_finite += other.non_finite;
        if let (Some(keys), Some(other)) = (&mut self.keys, &other.keys) {
            keys.merge(other, |a, b| value_type.compare(*a, *b));
        }
        if let (Some(strings), Some(other)) = (&mut self.strings, &other.strings) {
            strings.merge(other, Ord::cmp);
        }
        if let (Some(moments), Some(other)) = (&mut self.moments, &other.moments) {
            moments.merge(other);
        }
        if let (Some(bins), Some(other)) = (&mut self.bins, &other.bins) {
            bins.bins.merge(&other.bins);
            bins.zeros += other.zeros;
            bins.range = match (bins.range, other.range) {
                (Some((a_min, a_max)), Some((b_min, b_max))) => Some((
                    std::cmp::min_by(a_min, b_min, |a, b| value_type.compare(*a, *b)),
                    std::cmp::max_by(a_max, b_max, |a, b| value_type.compare(*a, *b)),
                )),
                (range, None) | (None, range) => range,
            };
        }
    }

    /// The compact JSON of `stat`'s data object.
    fn data(&self, stat: Stat) -> Result<String> {
        let (count, nulls, non_finite) = (self.count, self.nulls, self.non_finite);
        let value_type = self.value_type;
        let missing = || anyhow!("{} was not collected", stat.name());
        Ok(match stat {
            Stat::Counts => {
                let (truncated, values) = match (&self.keys, &self.strings) {
                    (Some(keys), _) => (
                        keys.truncated,
                        keys.ranked(|a, b| value_type.compare(*a, *b))
                            .into_iter()
                            .map(|(key, count)| (value_type.json(*key), count))
                            .collect(),
                    ),
                    (_, Some(strings)) => (
                        strings.truncated,
                        strings
                            .ranked(Ord::cmp)
                            .into_iter()
                            .map(|(key, count)| (json!(key), count))
                            .collect(),
                    ),
                    (None, None) => return Err(missing()),
                };
                serde_json::to_string(&CountsData {
                    count,
                    nulls,
                    non_finite,
                    truncated,
                    values,
                })?
            }
            Stat::Moments => {
                let (mean, m2) = self.moments.as_ref().ok_or_else(missing)?.mean_m2(count);
                serde_json::to_string(&MomentsData {
                    count,
                    nulls,
                    non_finite,
                    mean,
                    m2,
                })?
            }
            Stat::Bins => {
                let bins = self.bins.as_ref().ok_or_else(missing)?;
                serde_json::to_string(&BinsData {
                    count,
                    nulls,
                    non_finite,
                    zeros: bins.zeros,
                    min: bins.range.map(|(min, _)| value_type.json(min)),
                    max: bins.range.map(|(_, max)| value_type.json(max)),
                    bins_per_octave: BINS_PER_OCTAVE,
                    bins: bins.bins.sorted(),
                })?
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Per-table collection
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ColumnPlan {
    name: String,
    value_type: ValueType,
    stats: Vec<Stat>,
}

#[derive(Clone, Copy, Default)]
enum Cell {
    #[default]
    Null,
    Int(i64),
    Float(f64),
    Text,
}

/// The requested statistics of one table for every cutoff. Rows arrive one
/// column value at a time ([`TableStats::set`] or the observer methods) and
/// count once [`ColumnValueObserver::row_stored`] confirms them.
#[derive(Clone, Default)]
pub(crate) struct TableStats {
    table: String,
    columns: Vec<ColumnPlan>,
    cutoffs: Vec<Option<i64>>,
    /// Cutoff-major: `stats[cutoff * columns.len() + column]`.
    stats: Vec<ColumnStats>,
    /// Requested column position of each table column.
    slots: Vec<Option<usize>>,
    /// The create-time column and whether it holds dates.
    create_time: Option<(usize, bool)>,
    clock: TimeUnit,
    row: Vec<Cell>,
    texts: Vec<String>,
    row_time: Option<i64>,
}

impl TableStats {
    fn new<'a>(
        table: &str,
        columns: impl IntoIterator<Item = (&'a str, IngestedType)>,
        create_time: Option<&str>,
        requests: impl IntoIterator<Item = (&'a str, Stat)>,
        cutoffs: &[Option<i64>],
        clock: TimeUnit,
    ) -> Result<Self> {
        let columns: Vec<_> = columns.into_iter().collect();
        let index_of = |name: &str| {
            columns
                .iter()
                .position(|(column, _)| *column == name)
                .ok_or_else(|| anyhow!("table {table} has no column {name}"))
        };
        let mut requested: BTreeMap<&str, BTreeSet<Stat>> = BTreeMap::new();
        for (column, stat) in requests {
            requested.entry(column).or_default().insert(stat);
        }
        let mut slots = vec![None; columns.len()];
        let mut plans = Vec::with_capacity(requested.len());
        for (column, stats) in requested {
            let index = index_of(column)?;
            let value_type = ValueType::of(columns[index].1)
                .with_context(|| format!("column {table}.{column}"))?;
            for stat in &stats {
                ensure!(
                    value_type.supports(*stat),
                    "column {table}.{column} ({value_type:?}) does not support {}",
                    stat.name()
                );
            }
            slots[index] = Some(plans.len());
            plans.push(ColumnPlan {
                name: column.to_string(),
                value_type,
                stats: stats.into_iter().collect(),
            });
        }
        let create_time = create_time
            .map(|name| index_of(name).map(|idx| (idx, columns[idx].1 == IngestedType::Date)))
            .transpose()?;
        let stats = cutoffs
            .iter()
            .flat_map(|_| &plans)
            .map(|plan| ColumnStats::new(plan.value_type, &plan.stats))
            .collect();
        Ok(Self {
            table: table.to_string(),
            row: vec![Cell::Null; plans.len()],
            texts: vec![String::new(); plans.len()],
            columns: plans,
            cutoffs: cutoffs.to_vec(),
            stats,
            slots,
            create_time,
            clock,
            row_time: None,
        })
    }

    /// The table columns a row must supply: requested ones and the create time.
    fn read_mask(&self) -> Vec<bool> {
        (0..self.slots.len())
            .map(|idx| {
                self.slots[idx].is_some() || self.create_time.is_some_and(|(ct, _)| ct == idx)
            })
            .collect()
    }

    fn set(&mut self, col_idx: usize, value: Value<'_>) {
        if let Some((_, is_date)) = self.create_time.filter(|(ct, _)| *ct == col_idx) {
            self.row_time = match value {
                Value::Int(days) if is_date => {
                    temporal_value_in_clock(&ValueRef::Date(days as i32), self.clock)
                }
                Value::Int(ticks) => Some(ticks),
                _ => None,
            };
        }
        let Some(Some(slot)) = self.slots.get(col_idx).copied() else {
            return;
        };
        self.row[slot] = match value {
            Value::Null => Cell::Null,
            Value::Int(v) => Cell::Int(v),
            Value::Float(v) => Cell::Float(v),
            Value::Str(v) => {
                self.texts[slot].clear();
                self.texts[slot].push_str(v);
                Cell::Text
            }
        };
    }

    fn commit(&mut self) {
        let width = self.columns.len();
        for (idx, cutoff) in self.cutoffs.iter().enumerate() {
            let included = match (cutoff, self.row_time) {
                (Some(cutoff), Some(time)) => time <= *cutoff,
                (Some(_), None) => self.create_time.is_none(),
                (None, _) => true,
            };
            if !included {
                continue;
            }
            for slot in 0..width {
                let value = match self.row[slot] {
                    Cell::Null => Value::Null,
                    Cell::Int(v) => Value::Int(v),
                    Cell::Float(v) => Value::Float(v),
                    Cell::Text => Value::Str(&self.texts[slot]),
                };
                self.stats[idx * width + slot].add(value);
            }
        }
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        for (stats, other) in self.stats.iter_mut().zip(&other.stats) {
            stats.merge(other);
        }
    }

    /// Publishes every file of cutoff `idx`, returning `(key, path, sha256)`
    /// per file with key `<table>/<column>.<stat>`.
    fn publish(
        &self,
        idx: usize,
        index_dir: &Path,
        index_id: &str,
    ) -> Result<Vec<(String, PathBuf, String)>> {
        let cutoff = CutoffFiles::new(index_dir, index_id, self.cutoffs[idx], self.clock)?;
        cutoff.write_manifest_if_missing()?;
        let width = self.columns.len();
        let mut files = Vec::new();
        for (slot, plan) in self.columns.iter().enumerate() {
            for stat in &plan.stats {
                let data = self.stats[idx * width + slot].data(*stat)?;
                let (path, sha256) =
                    cutoff.write_stats(&self.table, &plan.name, *stat, plan.value_type, data)?;
                files.push((stats_key(&self.table, &plan.name, *stat), path, sha256));
            }
        }
        Ok(files)
    }
}

impl ColumnValueObserver for TableStats {
    fn observe_i64(&mut self, col_idx: usize, value: i64) {
        self.set(col_idx, Value::Int(value));
    }
    fn observe_f64(&mut self, col_idx: usize, value: f64) {
        self.set(col_idx, Value::Float(value));
    }
    fn observe_string(&mut self, col_idx: usize, value: &str) {
        self.set(col_idx, Value::Str(value));
    }
    fn observe_timestamp(&mut self, col_idx: usize, ticks: i64) {
        self.set(col_idx, Value::Int(ticks));
    }
    fn observe_float_list(&mut self, _col_idx: usize, _view: &FloatListView) {}
    fn observe_int64_list(&mut self, _col_idx: usize, _view: &IntListView) {}
    fn observe_string_list(&mut self, _col_idx: usize, _view: &StringListView) {}
    fn observe_null(&mut self, col_idx: usize) {
        self.set(col_idx, Value::Null);
    }
    fn observe_parse_failure(&mut self, _col_idx: usize) {}
    fn row_stored(&mut self) {
        self.commit();
    }
}

fn stats_key(table: &str, column: &str, stat: Stat) -> String {
    format!("{table}/{column}.{}", stat.name())
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct Cutoff {
    value: Option<i64>,
    unit: TimeUnit,
    inclusive: bool,
}

#[derive(Serialize)]
struct Manifest<'a> {
    schema: &'static str,
    spec_version: u32,
    index_id: &'a str,
    cutoff_id: &'a str,
    cutoff: &'a Cutoff,
}

#[derive(Serialize)]
struct StatsFile<'a> {
    schema: &'static str,
    spec_version: u32,
    index_id: &'a str,
    cutoff_id: &'a str,
    cutoff: &'a Cutoff,
    table: &'a str,
    column: &'a str,
    stat: Stat,
    value_type: ValueType,
    data: &'a RawValue,
    sha256: &'a str,
}

#[derive(Deserialize)]
struct StoredStats<'a> {
    schema: String,
    index_id: String,
    cutoff_id: String,
    table: String,
    column: String,
    stat: Stat,
    #[serde(borrow)]
    data: &'a RawValue,
    sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// First 16 hex digits of the sha256 of the cutoff's canonical JSON.
fn cutoff_id(index_id: &str, cutoff: Option<i64>, clock: TimeUnit) -> Result<String> {
    let canonical = json!({
        "index_id": index_id,
        "cutoff": cutoff,
        "unit": clock,
        "spec_version": SPEC_VERSION,
    });
    Ok(sha256_hex(&serde_json::to_vec(&canonical)?)[..16].to_string())
}

fn ensure_file_names(table: &str, column: &str) -> Result<()> {
    for name in [table, column] {
        ensure!(
            !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\']),
            "{name:?} cannot name a statistics file"
        );
    }
    Ok(())
}

/// One cutoff's directory under an index.
struct CutoffFiles<'a> {
    index_dir: &'a Path,
    dir: PathBuf,
    index_id: &'a str,
    cutoff_id: String,
    cutoff: Cutoff,
}

impl<'a> CutoffFiles<'a> {
    fn new(
        index_dir: &'a Path,
        index_id: &'a str,
        cutoff: Option<i64>,
        clock: TimeUnit,
    ) -> Result<Self> {
        let cutoff_id = cutoff_id(index_id, cutoff, clock)?;
        Ok(Self {
            index_dir,
            dir: index_dir.join(STATS_DIR).join(&cutoff_id),
            index_id,
            cutoff_id,
            cutoff: Cutoff {
                value: cutoff,
                unit: clock,
                inclusive: true,
            },
        })
    }

    fn stats_path(&self, table: &str, column: &str, stat: Stat) -> Result<PathBuf> {
        ensure_file_names(table, column)?;
        Ok(self
            .dir
            .join(table)
            .join(format!("{column}.{}.json", stat.name())))
    }

    fn write_manifest_if_missing(&self) -> Result<()> {
        let path = self.dir.join("manifest.json");
        if path.exists() {
            return Ok(());
        }
        let manifest = Manifest {
            schema: SCHEMA,
            spec_version: SPEC_VERSION,
            index_id: self.index_id,
            cutoff_id: &self.cutoff_id,
            cutoff: &self.cutoff,
        };
        self.publish(&path, &manifest)
    }

    /// The sha256 of a valid existing file for this request.
    fn existing(&self, table: &str, column: &str, stat: Stat) -> Result<Option<String>> {
        let Ok(bytes) = std::fs::read(self.stats_path(table, column, stat)?) else {
            return Ok(None);
        };
        let Ok(stored) = serde_json::from_slice::<StoredStats>(&bytes) else {
            return Ok(None);
        };
        let valid = stored.schema == SCHEMA
            && stored.index_id == self.index_id
            && stored.cutoff_id == self.cutoff_id
            && (stored.table.as_str(), stored.column.as_str(), stored.stat)
                == (table, column, stat)
            && sha256_hex(stored.data.get().as_bytes()) == stored.sha256;
        Ok(valid.then_some(stored.sha256))
    }

    fn write_stats(
        &self,
        table: &str,
        column: &str,
        stat: Stat,
        value_type: ValueType,
        data: String,
    ) -> Result<(PathBuf, String)> {
        let path = self.stats_path(table, column, stat)?;
        let sha256 = sha256_hex(data.as_bytes());
        let file = StatsFile {
            schema: SCHEMA,
            spec_version: SPEC_VERSION,
            index_id: self.index_id,
            cutoff_id: &self.cutoff_id,
            cutoff: &self.cutoff,
            table,
            column,
            stat,
            value_type,
            data: &RawValue::from_string(data)?,
            sha256: &sha256,
        };
        self.publish(&path, &file)?;
        Ok((path, sha256))
    }

    /// Atomically replaces `path` through a temp file no other writer uses.
    fn publish(&self, path: &Path, value: &impl Serialize) -> Result<()> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let mut bytes = serde_json::to_vec(value)?;
        bytes.push(b'\n');
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        let nonce = RandomState::new().hash_one(NEXT.fetch_add(1, AtomicOrdering::Relaxed));
        let tmp = path.with_file_name(format!(".{name}.{}.{nonce:x}.tmp", std::process::id()));
        write_durable(path, &tmp, &bytes, self.index_dir)
            .with_context(|| format!("publish {}", path.display()))
    }
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

impl IngestRequests {
    /// One empty [`TableStats`] per requested table; rejects unknown tables,
    /// edge tables, unknown columns and unsupported column types up front.
    pub(crate) fn tables(
        &self,
        tables: &[crate::graph_config::TableSpec],
        clock: TimeUnit,
    ) -> Result<HashMap<String, TableStats>> {
        let mut by_table: BTreeMap<&str, Vec<(&str, Stat)>> = BTreeMap::new();
        for (table, column, stat) in &self.requests {
            ensure_file_names(table, column)?;
            by_table.entry(table).or_default().push((column, *stat));
        }
        by_table
            .into_iter()
            .map(|(name, requests)| {
                let table = tables
                    .iter()
                    .find(|table| table.name == name)
                    .filter(|table| table.edge_table.is_none())
                    .ok_or_else(|| anyhow!("column statistics need a node table, got {name}"))?;
                let columns = crate::rowpack_columns(table)?;
                let stats = TableStats::new(
                    name,
                    columns.iter().map(|c| (c.name.as_str(), c.logical_type)),
                    table.create_time.as_deref(),
                    requests,
                    &self.cutoffs,
                    clock,
                )?;
                Ok((name.to_string(), stats))
            })
            .collect()
    }
}

/// Publishes what ingest collected, after `index_metadata.json` is final.
pub(crate) fn publish_ingested(
    tables: &HashMap<String, TableStats>,
    index_dir: &Path,
    index_id: &str,
) -> Result<()> {
    for stats in tables.values() {
        for idx in 0..stats.cutoffs.len() {
            stats.publish(idx, index_dir, index_id)?;
        }
    }
    Ok(())
}

/// Rescans `table`'s stored rows, decoding only the needed columns, in
/// parallel over chunks of every shard.
fn scan_table(
    reader: &GraphIndexReader,
    table: &str,
    requests: &[(&str, Stat)],
    cutoffs: &[Option<i64>],
) -> Result<TableStats> {
    let table_reader = reader
        .table(table)
        .ok_or_else(|| anyhow!("index has no node table {table}"))?;
    let decoder = table_reader
        .node_data_decoder()
        .ok_or_else(|| anyhow!("table {table} has no node data"))?;
    let empty = TableStats::new(
        table,
        decoder
            .context()
            .columns
            .iter()
            .map(|c| (c.name.as_str(), c.logical_type)),
        table_reader.create_time_col(),
        requests.iter().copied(),
        cutoffs,
        reader.metadata().timestamp_unit,
    )?;
    let mask = empty.read_mask();
    let chunks: Vec<(usize, u64, u64)> = (0..table_reader.shard_count())
        .filter_map(|shard| {
            let (min, max) = table_reader.shard_id_range(shard)?;
            (min <= max).then_some((shard, max - min + 1))
        })
        .flat_map(|(shard, nodes)| {
            (0..nodes)
                .step_by(SCAN_CHUNK_NODES as usize)
                .map(move |start| (shard, start, nodes.min(start + SCAN_CHUNK_NODES)))
        })
        .collect();
    chunks
        .into_par_iter()
        .try_fold(
            || (empty.clone(), Default::default()),
            |(mut stats, mut scratch), (shard, start, end)| {
                let shard = table_reader.shard(shard).expect("shard in range");
                for node in start..end {
                    // Rows stored without node data, oversized ones included, were never observed.
                    let view = shard.edges_for_node(node)?;
                    let Some(row) = view.node_data()?.filter(|row| !row.is_empty()) else {
                        continue;
                    };
                    decoder.for_each_value(row, &mask, &mut scratch, |idx, value| {
                        stats.set(idx, value.into())
                    })?;
                    stats.row_stored();
                }
                anyhow::Ok((stats, scratch))
            },
        )
        .map(|result| result.map(|(stats, _)| stats))
        .try_reduce(
            || empty.clone(),
            |mut stats, other| {
                stats.merge(&other);
                Ok(stats)
            },
        )
}

#[derive(Serialize)]
pub struct EnsuredFile {
    pub path: PathBuf,
    pub sha256: String,
    pub source: &'static str,
}

#[derive(Serialize)]
pub struct Ensured {
    pub cutoff_id: String,
    pub index_id: String,
    pub unit: TimeUnit,
    pub files: BTreeMap<String, EnsuredFile>,
    pub seconds: f64,
}

/// Makes every requested `(table, column, stat)` file of `cutoff` present
/// and valid, recomputing the missing ones from the index.
pub fn ensure_col_stats(
    index_dir: &Path,
    cutoff: Option<i64>,
    requests: &[(String, String, Stat)],
    num_threads: Option<usize>,
) -> Result<Ensured> {
    let start = std::time::Instant::now();
    let index_dir = &std::path::absolute(index_dir)?;
    let metadata = read_dataset_metadata(index_dir)?;
    let index_id = index_id(index_dir, &metadata)?;
    let clock = metadata.timestamp_unit;
    let cutoff_files = CutoffFiles::new(index_dir, &index_id, cutoff, clock)?;
    let mut files = BTreeMap::new();
    let mut missing: BTreeMap<&str, Vec<(&str, Stat)>> = BTreeMap::new();
    for (table, column, stat) in requests {
        match cutoff_files.existing(table, column, *stat)? {
            Some(sha256) => {
                let path = cutoff_files.stats_path(table, column, *stat)?;
                let file = EnsuredFile {
                    path,
                    sha256,
                    source: "existing",
                };
                files.insert(stats_key(table, column, *stat), file);
            }
            None => missing.entry(table).or_default().push((column, *stat)),
        }
    }
    if !missing.is_empty() {
        let tables: HashSet<String> = missing.keys().map(|table| table.to_string()).collect();
        let reader = GraphIndexReader::open_subset(
            &index_dir.join(INDEX_METADATA_FILENAME),
            index_dir,
            &tables,
        )?;
        let mut scan = || -> Result<()> {
            for (table, requests) in &missing {
                let stats = scan_table(&reader, table, requests, &[cutoff])?;
                for (key, path, sha256) in stats.publish(0, index_dir, &index_id)? {
                    let file = EnsuredFile {
                        path,
                        sha256,
                        source: "recompute",
                    };
                    files.insert(key, file);
                }
            }
            Ok(())
        };
        match num_threads {
            Some(threads) => rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?
                .install(scan)?,
            None => scan()?,
        }
    }
    cutoff_files.write_manifest_if_missing()?;
    Ok(Ensured {
        cutoff_id: cutoff_files.cutoff_id.clone(),
        index_id: index_id.clone(),
        unit: clock,
        files,
        seconds: start.elapsed().as_secs_f64(),
    })
}

/// The ingested type a natively typed Arrow array holds.
fn arrow_ingested_type(data_type: &DataType) -> Result<IngestedType> {
    Ok(match data_type {
        DataType::Boolean => IngestedType::Bool,
        DataType::Int8 => IngestedType::Int8,
        DataType::Int16 => IngestedType::Int16,
        DataType::Int32 => IngestedType::Int32,
        DataType::Int64 => IngestedType::Int64,
        DataType::UInt8 => IngestedType::UInt8,
        DataType::UInt16 => IngestedType::UInt16,
        DataType::UInt32 => IngestedType::UInt32,
        DataType::UInt64 => IngestedType::UInt64,
        DataType::Float32 => IngestedType::Float32,
        DataType::Float64 => IngestedType::Float64,
        DataType::Utf8 | DataType::LargeUtf8 => IngestedType::String,
        DataType::Date32 => IngestedType::Date,
        DataType::Timestamp(_, _) => IngestedType::Timestamp,
        other => bail!("column statistics are unsupported for Arrow {other}"),
    })
}

/// Each Arrow timestamp in ticks of `clock`, read as ingest reads it: an
/// error when beyond the clock, `None` when null.
fn ingest_timestamps(
    array: &ArrayRef,
    clock: TimeUnit,
) -> impl Iterator<Item = Result<Option<i64>>> + '_ {
    let mut access = RowAccess::new(vec![RowAccessHints::default()], RowAccessStats::new(None))
        .with_clock(clock);
    access.set_cols(std::slice::from_ref(array));
    (0..array.len()).map(move |row| {
        access.set_row(row);
        access.get_timestamp(0)
    })
}

/// Epoch `seconds` in ticks of the index clock, converted as ingest converts
/// an Arrow `timestamp[s]` create time (rounding up to a coarser clock).
pub fn cutoff_ticks(seconds: i64, clock: TimeUnit) -> Result<i64> {
    ensure!(clock.is_index_clock(), "{clock:?} is not an index clock");
    let array: ArrayRef = Arc::new(TimestampSecondArray::from(vec![seconds]));
    let ticks = ingest_timestamps(&array, clock).next().expect("one value");
    ticks
        .with_context(|| format!("cutoff {seconds} s in the {clock:?} clock"))?
        .ok_or_else(|| anyhow!("cutoff is null"))
}

/// Each row's value as ingest stores it.
fn arrow_values(array: &ArrayRef, clock: TimeUnit) -> Result<Vec<Value<'_>>> {
    fn map<T: ArrowPrimitiveType>(
        array: &dyn Array,
        f: impl Fn(T::Native) -> Value<'static>,
    ) -> Vec<Value<'_>> {
        let array = array.as_primitive::<T>();
        array.iter().map(|v| v.map_or(Value::Null, &f)).collect()
    }
    Ok(match array.data_type() {
        DataType::Boolean => array
            .as_boolean()
            .iter()
            .map(|v| v.map_or(Value::Null, |v| Value::Int(v.into())))
            .collect(),
        DataType::Int8 => map::<Int8Type>(array, |v| Value::Int(v.into())),
        DataType::Int16 => map::<Int16Type>(array, |v| Value::Int(v.into())),
        DataType::Int32 => map::<Int32Type>(array, |v| Value::Int(v.into())),
        DataType::Int64 => map::<Int64Type>(array, Value::Int),
        DataType::UInt8 => map::<UInt8Type>(array, |v| Value::Int(v.into())),
        DataType::UInt16 => map::<UInt16Type>(array, |v| Value::Int(v.into())),
        DataType::UInt32 => map::<UInt32Type>(array, |v| Value::Int(v.into())),
        DataType::UInt64 => map::<UInt64Type>(array, |v| Value::Int(v as i64)),
        DataType::Float32 => map::<Float32Type>(array, |v| Value::Float(v.into())),
        DataType::Float64 => map::<Float64Type>(array, Value::Float),
        DataType::Utf8 => array
            .as_string::<i32>()
            .iter()
            .map(|v| v.map_or(Value::Null, Value::Str))
            .collect(),
        DataType::LargeUtf8 => array
            .as_string::<i64>()
            .iter()
            .map(|v| v.map_or(Value::Null, Value::Str))
            .collect(),
        DataType::Date32 => map::<Date32Type>(array, |v| Value::Int(v.into())),
        DataType::Timestamp(_, _) => ingest_timestamps(array, clock)
            .map(|ticks| ticks.ok().flatten().map_or(Value::Null, Value::Int))
            .collect(),
        other => bail!("column statistics are unsupported for Arrow {other}"),
    })
}

/// `stats` of `values` over the rows whose `create_time` (ticks of `clock`,
/// or Arrow dates/timestamps) is at or before `cutoff`; rows with a null
/// create time are skipped and `None` counts every row. Values and create
/// times convert to `clock` as ingest converts them. Returns each stat's
/// data JSON, as its file would hold it.
pub fn stats_from_arrow(
    values: &ArrayRef,
    create_time: Option<&ArrayRef>,
    cutoff: Option<i64>,
    stats: &[Stat],
    clock: TimeUnit,
) -> Result<Vec<(Stat, String)>> {
    if stats.is_empty() {
        return Ok(Vec::new());
    }
    let value_type = arrow_ingested_type(values.data_type())?;
    let create_time_type = match create_time {
        Some(array) if array.data_type() == &DataType::Date32 => IngestedType::Date,
        Some(array) if array.data_type() == &DataType::Int64 => IngestedType::Timestamp,
        Some(array) => match arrow_ingested_type(array.data_type())? {
            IngestedType::Timestamp => IngestedType::Timestamp,
            other => bail!("create_time must be int64 ticks, a date or a timestamp, got {other:?}"),
        },
        None => IngestedType::Timestamp,
    };
    if let Some(array) = create_time {
        ensure!(
            array.len() == values.len(),
            "create_time has {} rows, values has {}",
            array.len(),
            values.len()
        );
    }
    let mut table = TableStats::new(
        "",
        [("values", value_type), ("create_time", create_time_type)],
        create_time.map(|_| "create_time"),
        stats.iter().map(|stat| ("values", *stat)),
        &[cutoff],
        clock,
    )?;
    let times = create_time
        .map(|array| arrow_values(array, clock))
        .transpose()?;
    for (row, value) in arrow_values(values, clock)?.into_iter().enumerate() {
        if let Some(times) = &times {
            if matches!(times[row], Value::Null) {
                continue;
            }
            table.set(1, times[row]);
        }
        table.set(0, value);
        table.row_stored();
    }
    let column = &table.stats[0];
    let mut requested: Vec<Stat> = stats.to_vec();
    requested.sort_unstable();
    requested.dedup();
    requested
        .into_iter()
        .map(|stat| Ok((stat, column.data(stat)?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact(values: &[f64]) -> (f64, f64) {
        let mut moments = Moments::default();
        for &x in values {
            moments.add(x);
        }
        moments.mean_m2(values.len() as u64)
    }

    #[test]
    fn exact_sum_matches_integer_reference() {
        // k · 2^-10 with |k| < 2^40: Σk and n·Σk² − (Σk)² are exact in i128, and
        // dividing by n = 2^12 is exact too.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let ks: Vec<i64> = (0..4096)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as i64 - (1 << 39)
            })
            .collect();
        let values: Vec<f64> = ks.iter().map(|&k| k as f64 / 1024.0).collect();
        let n = ks.len() as i128;
        let sum: i128 = ks.iter().map(|&k| k as i128).sum();
        let squares: i128 = ks.iter().map(|&k| (k as i128) * (k as i128)).sum();
        let mean = sum as f64 / 1024.0 / n as f64;
        let m2 = (n * squares - sum * sum) as f64 / (1u64 << 20) as f64 / n as f64;
        assert_eq!(exact(&values), (mean, m2));
    }

    #[test]
    fn exact_sum_survives_cancellation_overflow_and_underflow() {
        assert_eq!(exact(&[1e308, 1e308, 1.0, -1e308, -1e308]).0, 1.0 / 5.0);
        assert_eq!(exact(&[1e100, 1.0, -1e100]).0, 1.0 / 3.0);
        assert_eq!(exact(&[f64::MAX, -f64::MAX, f64::MAX]).0, f64::MAX / 3.0);
        assert_eq!(exact(&[5e-324, 5e-324]).0, 5e-324);
        // Squares below the smallest subnormal still cancel exactly.
        assert_eq!(exact(&[1e-200, 1e-200]), (1e-200, 0.0));
        assert_eq!(exact(&[3.0, 3.0, 3.0]), (3.0, 0.0));
        assert_eq!(exact(&[]), (0.0, 0.0));
        // Σx and n·Σx² overflow doubles; the quotients by n do not.
        assert_eq!(exact(&[1e308, 1e308]).0, 1e308);
        let big = 2f64.powi(511);
        assert_eq!(exact(&[big, -big]), (0.0, 2f64.powi(1023)));
    }

    #[test]
    fn quotients_round_once_to_nearest_even() {
        let r = |v: i64, lsb| round_quotient(&BigInt::from(v), 1, lsb);
        assert_eq!(r((1 << 53) + 1, 0), 9007199254740992.0);
        assert_eq!(r((1 << 53) + 3, 0), 9007199254740996.0);
        assert_eq!(r(-((1 << 54) + 3), 0), -18014398509481988.0);
        assert_eq!(r(3, -1075), 1e-323);
        assert_eq!(r(1, -1075), 0.0);
        assert_eq!(r(1, 1024), f64::INFINITY);
        let q = |v: i64, n| round_quotient(&BigInt::from(v), n, 0);
        assert_eq!(q(-10, 3), -10.0 / 3.0);
        // An exact tie rounds to even; a remainder just past it rounds up.
        assert_eq!(q((1 << 54) + 2, 2), 9007199254740992.0);
        assert_eq!(q((1 << 55) + 13, 4), 9007199254740996.0);
    }

    #[test]
    fn merges_are_order_independent() {
        let values: Vec<f64> = (0..3000)
            .map(|i| {
                ((i * 7919) % 1000) as f64 * 1.37e-3 - 0.5 + if i % 97 == 0 { 1e15 } else { 0.0 }
            })
            .collect();
        let data = |chunks: &[&[f64]]| {
            let mut total: Option<ColumnStats> = None;
            for chunk in chunks {
                let mut part = ColumnStats::new(
                    ValueType::Float64,
                    &[Stat::Counts, Stat::Moments, Stat::Bins],
                );
                for &x in *chunk {
                    part.add(Value::Float(x));
                }
                match &mut total {
                    Some(total) => total.merge(&part),
                    None => total = Some(part),
                }
            }
            let total = total.unwrap();
            [Stat::Counts, Stat::Moments, Stat::Bins].map(|stat| total.data(stat).unwrap())
        };
        let mut reversed = values.clone();
        reversed.reverse();
        let whole = data(&[&values]);
        assert_eq!(
            whole,
            data(&[&values[..1234], &values[1234..2000], &values[2000..]])
        );
        assert_eq!(whole, data(&[&reversed[..10], &reversed[10..]]));
    }

    #[test]
    fn bins_and_counts_canonicalize_values() {
        let mut stats = ColumnStats::new(ValueType::Float64, &[Stat::Counts, Stat::Bins]);
        for x in [
            -0.0,
            0.0,
            1.0,
            1.0 + 1.0 / 128.0,
            -3.0,
            f64::NAN,
            f64::INFINITY,
        ] {
            stats.add(Value::Float(x));
        }
        stats.add(Value::Null);
        let bins: JsonValue = serde_json::from_str(&stats.data(Stat::Bins).unwrap()).unwrap();
        let one = (1f64.to_bits() >> 45) as i64;
        let three = (3f64.to_bits() >> 45) as i64;
        assert_eq!(
            bins,
            json!({"count": 5, "nulls": 1, "non_finite": 2, "zeros": 2, "min": -3.0, "max": 1.0078125,
                   "bins_per_octave": 128, "bins": [[-three - 1, 1], [0, 2], [one, 1], [one + 1, 1]]})
        );
        let counts: JsonValue = serde_json::from_str(&stats.data(Stat::Counts).unwrap()).unwrap();
        assert_eq!(
            counts["values"],
            json!([[0.0, 2], [-3.0, 1], [1.0, 1], [1.0078125, 1]])
        );

        let mut unsigned = ColumnStats::new(ValueType::Uint64, &[Stat::Counts, Stat::Bins]);
        for v in [u64::MAX, 1, u64::MAX] {
            unsigned.add(Value::Int(v as i64));
        }
        let counts: JsonValue =
            serde_json::from_str(&unsigned.data(Stat::Counts).unwrap()).unwrap();
        assert_eq!(counts["values"], json!([[u64::MAX, 2], [1, 1]]));
        let bins: JsonValue = serde_json::from_str(&unsigned.data(Stat::Bins).unwrap()).unwrap();
        assert_eq!((&bins["min"], &bins["max"]), (&json!(1), &json!(u64::MAX)));
    }
}
