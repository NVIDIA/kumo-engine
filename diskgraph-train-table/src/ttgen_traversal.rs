//! Edge traversal primitives for TT-gen.
//!
//! `iter_edge` reads edges from a relation segment.
//! `traverse_and_accumulate` recursively walks multi-hop paths.

use diskgraph_core::tt_builder::Value;
use diskgraph_index::graph::TableReader;
use diskgraph_index::layout::RelationLayout;
use diskgraph_rowpack::{IngestedType, RowPackDecodeScratch, ValueRef};
use std::sync::Arc;

use crate::ttgen_resolve::ResolvedEdge;

/// Encoded temporal bounds: an optional exclusive lower bound and an inclusive
/// upper bound. `None` on the lower endpoint represents all prior history.
pub type SortKeyWindow = (Option<u64>, u64);

// ─── Edge iteration ──────────────────────────────────────────────────────────

/// A decoded edge entry: destination node ID and sort key.
#[derive(Debug, Clone, Copy)]
pub struct EdgeEntry {
    pub dst_node: u64,
    pub sort_key: u64,
}

/// Iterate edges from `src_node` in a given relation, optionally restricted
/// to a sort key range (for time window binary search).
///
/// Returns a Vec of (dst_node, sort_key) pairs. A bounded range uses `(lo, hi]`
/// semantics; a missing lower endpoint includes every edge through `hi`.
#[inline]
pub fn iter_edge(
    src_table: &Arc<TableReader>,
    relation_idx: usize,
    layout: &RelationLayout,
    src_node: u64,
    sort_key_range: Option<SortKeyWindow>,
) -> Vec<EdgeEntry> {
    let view = match src_table.edges_for_node(src_node) {
        Ok(v) => v,
        Err(_) => {
            return Vec::new();
        }
    };
    let segments = match view.relation_segments() {
        Ok(s) => s,
        Err(_) => {
            return Vec::new();
        }
    };
    let segment = match segments.get(relation_idx) {
        Some(s) => s,
        None => {
            return Vec::new();
        }
    };

    if segment.count == 0 {
        return Vec::new();
    }

    let entry_bytes = layout.entry_bytes;
    let dst_mask = layout.dst_mask;
    let dst_bits = layout.dst_bits;
    let sk_mask = layout.dest_sort_key_mask.unwrap_or(0);

    let num_entries = segment.count as usize;

    // Decode all edges
    let decode_entry = |idx: usize| -> EdgeEntry {
        let offset = idx * entry_bytes;
        let chunk = &segment.data[offset..offset + entry_bytes];
        let mut buf = [0u8; 16];
        buf[..entry_bytes].copy_from_slice(chunk);
        let val = u128::from_le_bytes(buf);
        let dst_node = (val & dst_mask as u128) as u64;
        let sort_key = if sk_mask > 0 {
            ((val >> dst_bits) & sk_mask as u128) as u64
        } else {
            0
        };
        EdgeEntry { dst_node, sort_key }
    };

    match sort_key_range {
        Some((lo, hi)) => {
            // Binary search for the range (lo, hi] within sorted edges.
            // Matches Kumo's temporal join: start < edge_ts AND end >= edge_ts.
            // Edges are sorted by sort key DESCENDING (newest first).
            // For descending data, `sk > X` is monotonic [true..., false...],
            // so partition_point gives the first index where sk <= X.
            let start = partition_point_by(num_entries, |i| {
                decode_entry(i).sort_key > hi
            });
            let end = lo
                .map(|lo| {
                    partition_point_by(num_entries, |i| decode_entry(i).sort_key > lo)
                })
                .unwrap_or(num_entries);
            (start..end).map(decode_entry).collect()
        }
        None => {
            // Return all edges
            (0..num_entries).map(decode_entry).collect()
        }
    }
}

/// Binary search helper: find the first index where predicate is false.
fn partition_point_by(len: usize, mut pred: impl FnMut(usize) -> bool) -> usize {
    let mut lo = 0;
    let mut hi = len;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pred(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

// ─── Multi-hop recursive traversal ──────────────────────────────────────────

/// Walk a multi-hop path from `start_node`, calling `callback` at each final
/// destination. Time window is applied on the FIRST hop only (matching Kumo's
/// temporal join semantics — the first hop is the reverse FK with time-sorted
/// edges). Subsequent hops iterate ALL edges (no window, no filter).
///
/// The callback returns `bool`: true = continue, false = early exit
/// (for First, ListDistinct with limit).
///
/// Zero heap allocation for intermediates — uses O(path_depth) stack frames
/// via recursion. Path depth is typically 1-2, max ~3 in practice.
#[inline]
pub fn traverse_and_accumulate<F: FnMut(u64, u64) -> bool>(
    start: u64,
    path: &[ResolvedEdge],
    window: Option<SortKeyWindow>,
    is_first_hop: bool,
    callback: &mut F,
) -> bool {
    if path.is_empty() {
        return true;
    }

    let edge = &path[0];
    // Apply window only on the first hop (reverse FK with sort keys).
    let hop_window = if is_first_hop { window } else { None };

    let entries = iter_edge(
        &edge.src_table,
        edge.relation_idx,
        &edge.layout,
        start,
        hop_window,
    );

    if path.len() == 1 {
        // Last hop — deliver results to callback
        for entry in entries {
            if !callback(entry.dst_node, entry.sort_key) {
                return false; // early exit propagates up
            }
        }
        true
    } else {
        // Not last hop — recurse into remaining hops
        for entry in entries {
            if !traverse_and_accumulate(entry.dst_node, &path[1..], window, false, callback) {
                return false;
            }
        }
        true
    }
}

// ─── Column value reading ────────────────────────────────────────────────────

/// Read a single column value from a node's rowpack data.
///
/// Returns None if the node has no data or the column is null.
///
/// Note: we use `view.node_data()` to get raw rowpack bytes directly,
/// NOT `table.decode_node_data()` which converts rowpack→msgpack (for Arrow export).
/// `decode_column_value` expects raw rowpack input.
pub fn read_column_value(
    table: &Arc<TableReader>,
    node_id: u64,
    col_idx: usize,
) -> Option<Value> {
    let view = table.edges_for_node(node_id).ok()?;
    let decoder = table.node_data_decoder()?;
    let raw_data = view.node_data().ok()??;

    let mut scratch = RowPackDecodeScratch::default();
    let value_ref = decoder
        .decode_column_value(raw_data, col_idx, &mut scratch)
        .ok()?;

    match value_ref {
        ValueRef::Null => None,
        _ => Some(value_ref_to_value(&value_ref)),
    }
}

/// Read a column value, distinguishing explicit NULL from missing data.
///
/// Returns:
/// - `Some(Some(value))` — column has a non-null value
/// - `Some(None)` — column is explicitly NULL in rowpack
/// - `None` — node data is missing or undecodable
pub fn read_column_value_nullable(
    table: &Arc<TableReader>,
    node_id: u64,
    col_idx: usize,
) -> Option<Option<Value>> {
    let view = table.edges_for_node(node_id).ok()?;
    let decoder = table.node_data_decoder()?;
    let raw_data = view.node_data().ok()??;

    let mut scratch = RowPackDecodeScratch::default();
    let value_ref = decoder
        .decode_column_value(raw_data, col_idx, &mut scratch)
        .ok()?;

    match value_ref {
        ValueRef::Null => Some(None),
        _ => Some(Some(value_ref_to_value(&value_ref))),
    }
}

/// Read a column as f64 from a node's rowpack data.
/// Returns `None` if the column is NULL or if the node data is missing,
/// matching SQL/Spark NULL-skip semantics for aggregations.
pub fn read_column_f64(
    table: &Arc<TableReader>,
    node_id: u64,
    col_idx: usize,
) -> Option<f64> {
    read_column_value_nullable(table, node_id, col_idx)
        .flatten()
        .and_then(|v| match v {
            Value::Int(i) => Some(i as f64),
            Value::Float(f) => Some(f),
            _ => None,
        })
}

/// Read a column as i64 from a node's rowpack data.
/// Returns None if the column is NULL or if the node data is missing.
pub fn read_column_i64(
    table: &Arc<TableReader>,
    node_id: u64,
    col_idx: usize,
) -> Option<i64> {
    read_column_value_nullable(table, node_id, col_idx)?.and_then(|v| match v {
        Value::Int(i) => Some(i),
        Value::Float(f) => Some(f as i64),
        _ => None,
    })
}

// ─── Multicat dispatch (kumo#28400) ─────────────────────────────────────────
//
// For source columns declared `stype=multicategorical`, every TT-gen read site
// must observe per-token (not per-row-value). Post-#28478 + post-validator
// (graph_config::validate) the executor sees only two multicat-relevant
// shapes; the legacy delimited-string shape is flipped to `StringArray` at
// ingest, and the no-delim scalar shape is rejected at config validation as
// degenerate (use `stype=categorical` for single-token-per-row columns):
//   `IngestedType::StringArray`  → list-aware decode, iterate `&[&str]`.
//   `IngestedType::Int64Array`   → list-aware decode, iterate `&[i64]`.
//   `IngestedType::Float*Array`  → not supported as multicat (precision drift,
//                                   NaN/inf semantics); the resolver rejects
//                                   these plans before reaching here.
//
// For non-multicat sources, the legacy whole-cell `read_column_value` path is
// preserved verbatim. Each closure returns `bool` ("keep going"); the helper
// stops dispatch on the first `false` to support per-site early-exit (e.g.
// `LIST_DISTINCT` limit).

/// One observation passed to the per-row closure by [`dispatch_multicat_source`].
pub enum MulticatToken<'a> {
    /// Per-token observation for a multicat source: borrowed string slice.
    Str(&'a str),
    /// Per-token observation for a multicat source: integer element.
    Int(i64),
    /// Whole-cell legacy observation (non-multicat source, or multicat
    /// declared on a scalar non-string column where per-token semantics
    /// don't apply).
    Legacy(Value),
}

/// Outcome of [`dispatch_multicat_source`]. Distinguishes "source NULL"
/// (caller should treat as no value observed) from "source present, observe
/// invoked some number of times (possibly zero)" (caller may emit a result,
/// even an empty list, depending on its semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulticatDispatchOutcome {
    /// Source row was NULL or had no node data; closure was not invoked.
    /// Direct-column callers (Route A) treat this as "skip this row" rather
    /// than emitting an empty target list. Aggregate callers ignore the
    /// distinction (NULL row contributes nothing, traversal continues).
    SourceNull,
    /// Source row was present and the closure was invoked zero or more
    /// times. The outer traversal should continue (no early-exit signaled).
    Continue,
    /// Source row was present and the closure returned `false` to signal
    /// early-exit (e.g., LIST_DISTINCT limit reached). The outer traversal
    /// should stop.
    Stop,
}

impl MulticatDispatchOutcome {
    /// Returns true when the closure was passed a present source — i.e.,
    /// observation was attempted (even if the list contained zero tokens).
    /// Equivalent to `!matches!(self, SourceNull)`.
    pub fn observed(self) -> bool {
        !matches!(self, MulticatDispatchOutcome::SourceNull)
    }

    /// Returns true when the outer traversal should keep going. Both
    /// `Continue` and `SourceNull` signal "no early-exit"; only `Stop`
    /// signals "break".
    pub fn keep_going(self) -> bool {
        !matches!(self, MulticatDispatchOutcome::Stop)
    }
}

/// Read a source column for one row, dispatching on
/// `(target_semantic_type, IngestedType)`. Calls `observe` once per token
/// (multicat native-list sources) or once per row (legacy / non-multicat
/// sources). Multicat sources are guaranteed to be native-list shaped on
/// disk: ingest's `normalize_multicat_pseudo_list` flips delimited scalars
/// to `StringArray` and `GraphConfig::validate` rejects no-delim scalar
/// multicat as a misdeclaration. The closure returns `false` to signal
/// early-exit (limit reached); the caller receives that as `Stop` and can
/// stop the outer traversal.
///
/// Post-#28478: ingest's `GraphConfig::normalize_multicat_pseudo_list` flips
/// `(stype=Multicategorical, dtype=String, delim=Some)` columns to
/// `IngestedType::StringArray` at convert time. The executor never sees a
/// multicat-with-delimiter scalar source. The `(Multicategorical, String)`
/// arm here covers only the no-delim single-token-per-row case (one cell =
/// one token). No `delim` parameter — ingest consumed it.
///
/// Float-list multicat sources are rejected at resolve time; if the helper
/// ever sees one, it falls through silently rather than emitting wrong data.
pub fn dispatch_multicat_source<F>(
    table: &Arc<TableReader>,
    node_id: u64,
    col_idx: usize,
    is_multicat: bool,
    mut observe: F,
) -> MulticatDispatchOutcome
where
    F: FnMut(MulticatToken<'_>) -> bool,
{
    use MulticatDispatchOutcome::*;
    let Ok(view) = table.edges_for_node(node_id) else {
        return SourceNull;
    };
    let Some(decoder) = table.node_data_decoder() else {
        return SourceNull;
    };
    let Ok(Some(raw)) = view.node_data() else {
        return SourceNull;
    };

    let logical_type = decoder.context().columns[col_idx].logical_type;

    if !is_multicat {
        // Legacy path — whole-cell observe.
        let mut scratch = RowPackDecodeScratch::default();
        let Ok(value_ref) = decoder.decode_column_value(raw, col_idx, &mut scratch) else {
            return SourceNull;
        };
        if matches!(value_ref, ValueRef::Null) {
            return SourceNull;
        }
        return if observe(MulticatToken::Legacy(value_ref_to_value(&value_ref))) {
            Continue
        } else {
            Stop
        };
    }

    match logical_type {
        IngestedType::StringArray => {
            let mut scratch = RowPackDecodeScratch::default();
            // `decode_column_value` errors out for StringArray (cannot return
            // borrowed `&[&str]` safely from a function returning a value);
            // use the closure-form decoder to keep the borrows scoped.
            let result = decoder.with_string_array_column(raw, col_idx, &mut scratch, |toks| {
                let mut keep = true;
                for &tok in toks {
                    if !observe(MulticatToken::Str(tok)) {
                        keep = false;
                        break;
                    }
                }
                keep
            });
            match result {
                Ok(Some(keep)) => if keep { Continue } else { Stop },
                Ok(None) => SourceNull,
                Err(_) => SourceNull,
            }
        }
        IngestedType::Int8Array
        | IngestedType::Int16Array
        | IngestedType::Int32Array
        | IngestedType::Int64Array => {
            let mut scratch = RowPackDecodeScratch::default();
            let Ok(value_ref) = decoder.decode_column_value(raw, col_idx, &mut scratch) else {
                return SourceNull;
            };
            // Each int-array width decodes to its own ValueRef variant; widen
            // to i64 for the shared multicat-token consumer.
            let mut emit = |ints: &[i64]| -> Result<bool, ()> {
                for &n in ints {
                    if !observe(MulticatToken::Int(n)) {
                        return Ok(false);
                    }
                }
                Ok(true)
            };
            let result = match value_ref {
                ValueRef::Int64Array(ints) => emit(ints),
                ValueRef::Int32Array(ints) => {
                    let widened: Vec<i64> = ints.iter().map(|&v| v as i64).collect();
                    emit(&widened)
                }
                ValueRef::Int16Array(ints) => {
                    let widened: Vec<i64> = ints.iter().map(|&v| v as i64).collect();
                    emit(&widened)
                }
                ValueRef::Int8Array(ints) => {
                    let widened: Vec<i64> = ints.iter().map(|&v| v as i64).collect();
                    emit(&widened)
                }
                ValueRef::Null => return SourceNull,
                _ => {
                    // Type mismatch on a multicat-declared column — defensive
                    // fallback. The resolver gate should have rejected this.
                    return SourceNull;
                }
            };
            match result {
                Ok(true) => Continue,
                Ok(false) => Stop,
                Err(_) => SourceNull,
            }
        }
        IngestedType::Float32Array
        | IngestedType::Float64Array
        | IngestedType::Float16Array => {
            // Resolve-time gate should have rejected this; defensive fallback.
            SourceNull
        }
        _ => {
            // Unreachable in practice: graph_config::validate rejects any
            // non-list ingested_dtype paired with stype=multicategorical, so
            // this branch can only fire if a malformed plan + legacy index
            // slip past. Treat as NULL so the caller skips the row instead
            // of falling back to legacy whole-cell observe (which produced
            // the silent zero-token bug pre-PR230).
            SourceNull
        }
    }
}

/// Convert a rowpack ValueRef to our Value type.
fn value_ref_to_value(vr: &ValueRef) -> Value {
    match vr {
        ValueRef::Null => Value::Int(0),
        ValueRef::Int64(v) => Value::Int(*v),
        ValueRef::Int32(v) => Value::Int(*v as i64),
        ValueRef::UInt64(v) => Value::Int(*v as i64),
        ValueRef::UInt32(v) => Value::Int(*v as i64),
        ValueRef::Float64(v) => Value::Float(*v),
        ValueRef::Float32(v) => Value::Float(*v as f64),
        ValueRef::Bool(b) => Value::Bool(*b),
        ValueRef::String(s) => Value::Str(s.to_string()),
        ValueRef::Bytes(b) => Value::Str(String::from_utf8_lossy(b).to_string()),
        ValueRef::Date(d) => Value::Int(*d as i64 * 86400),
        ValueRef::Timestamp(secs) => Value::Int(*secs),
        ValueRef::Decimal128(v) => Value::Float(*v as f64),
        _ => Value::Int(0), // Uuid, Ip, arrays, etc. → default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_point_ascending() {
        let data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let idx = partition_point_by(data.len(), |i| data[i] < 5);
        assert_eq!(idx, 5);
    }

    #[test]
    fn test_partition_point_all_true() {
        let data = [0, 1, 2, 3, 4];
        let idx = partition_point_by(data.len(), |i| data[i] < 100);
        assert_eq!(idx, 5);
    }

    #[test]
    fn test_partition_point_all_false() {
        let idx = partition_point_by(5, |_| false);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_partition_point_empty() {
        let idx = partition_point_by(0, |_| true);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_partition_point_single_true() {
        let idx = partition_point_by(1, |_| true);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_partition_point_single_false() {
        let idx = partition_point_by(1, |_| false);
        assert_eq!(idx, 0);
    }

    // ─── Descending binary search correctness proofs ─────────────────────
    //
    // These tests prove that our binary search predicates correctly find
    // edges in (lo, hi] when edges are sorted DESCENDING by sort key.
    // This matches Kumo's temporal join: start < edge_ts AND end >= edge_ts.

    #[test]
    fn test_descending_search_all_inside() {
        // Edges (descending): [300, 200, 100], window (50, 400]
        // All inside → start=0, end=3
        let keys = [300u64, 200, 100];
        let lo = 50u64;
        let hi = 400u64;
        let start = partition_point_by(keys.len(), |i| keys[i] > hi);
        let end = partition_point_by(keys.len(), |i| keys[i] > lo);
        assert_eq!(start, 0);
        assert_eq!(end, 3);
        assert_eq!(end - start, 3);
    }

    #[test]
    fn test_descending_search_all_outside_above() {
        // Edges: [300, 200, 100], window (0, 50] → all above 50
        let keys = [300u64, 200, 100];
        let start = partition_point_by(keys.len(), |i| keys[i] > 50);
        let end = partition_point_by(keys.len(), |i| keys[i] > 0);
        assert_eq!(start, 3);
        assert_eq!(end, 3);
        assert_eq!(end - start, 0);
    }

    #[test]
    fn test_descending_search_all_outside_below() {
        // Edges: [300, 200, 100], window (400, 500] → all below 400
        let keys = [300u64, 200, 100];
        let start = partition_point_by(keys.len(), |i| keys[i] > 500);
        let end = partition_point_by(keys.len(), |i| keys[i] > 400);
        assert_eq!(start, 0);
        assert_eq!(end, 0);
        assert_eq!(end - start, 0);
    }

    #[test]
    fn test_descending_search_partial_window() {
        // Edges: [3000, 2000, 1000], window (1500, 2500]
        // Only 2000 is in range
        let keys = [3000u64, 2000, 1000];
        let start = partition_point_by(keys.len(), |i| keys[i] > 2500);
        let end = partition_point_by(keys.len(), |i| keys[i] > 1500);
        assert_eq!(start, 1); // first index where sk <= 2500
        assert_eq!(end, 2);   // first index where sk <= 1500
        let in_range: Vec<u64> = (start..end).map(|i| keys[i]).collect();
        assert_eq!(in_range, vec![2000]);
    }

    #[test]
    fn test_descending_search_boundary_exclusive_lo() {
        // Edges: [300, 200, 100], window (100, 250]
        // lo=100 is exclusive → edge 100 excluded
        let keys = [300u64, 200, 100];
        let start = partition_point_by(keys.len(), |i| keys[i] > 250);
        let end = partition_point_by(keys.len(), |i| keys[i] > 100);
        let in_range: Vec<u64> = (start..end).map(|i| keys[i]).collect();
        assert_eq!(in_range, vec![200]);
    }

    #[test]
    fn test_descending_search_boundary_inclusive_hi() {
        // Edges: [300, 200, 100], window (100, 300]
        // hi=300 is inclusive → edge 300 included
        let keys = [300u64, 200, 100];
        let start = partition_point_by(keys.len(), |i| keys[i] > 300);
        let end = partition_point_by(keys.len(), |i| keys[i] > 100);
        // sk > 300 → [false, false, false] → partition_point = 0
        // sk > 100 → [true, true, false] → partition_point = 2
        let in_range: Vec<u64> = (start..end).map(|i| keys[i]).collect();
        assert_eq!(in_range, vec![300, 200]);
    }

    #[test]
    fn test_descending_search_exact_match_single() {
        // Edges: [300, 200, 100], window (199, 200] → only 200
        let keys = [300u64, 200, 100];
        let start = partition_point_by(keys.len(), |i| keys[i] > 200);
        let end = partition_point_by(keys.len(), |i| keys[i] > 199);
        let in_range: Vec<u64> = (start..end).map(|i| keys[i]).collect();
        assert_eq!(in_range, vec![200]);
    }

    #[test]
    fn test_descending_search_empty_window() {
        // Edges: [300, 200, 100], window (150, 150] → empty (lo == hi)
        let keys = [300u64, 200, 100];
        let start = partition_point_by(keys.len(), |i| keys[i] > 150);
        let end = partition_point_by(keys.len(), |i| keys[i] > 150);
        assert_eq!(start, end);
    }

    #[test]
    fn test_descending_search_gap_window() {
        // Edges: [300, 100], window (150, 250] → no edges in gap
        let keys = [300u64, 100];
        let start = partition_point_by(keys.len(), |i| keys[i] > 250);
        let end = partition_point_by(keys.len(), |i| keys[i] > 150);
        assert_eq!(end - start, 0);
    }

    #[test]
    fn test_descending_search_duplicates() {
        // Edges: [300, 200, 200, 200, 100], window (150, 250]
        // All three 200s should be included
        let keys = [300u64, 200, 200, 200, 100];
        let start = partition_point_by(keys.len(), |i| keys[i] > 250);
        let end = partition_point_by(keys.len(), |i| keys[i] > 150);
        assert_eq!(end - start, 3);
    }

    #[test]
    fn test_descending_search_many_edges() {
        // 10 edges descending: 900, 800, ..., 0
        let keys: Vec<u64> = (0..10).rev().map(|i| i * 100).collect();
        // window (250, 750]
        let start = partition_point_by(keys.len(), |i| keys[i] > 750);
        let end = partition_point_by(keys.len(), |i| keys[i] > 250);
        let in_range: Vec<u64> = (start..end).map(|i| keys[i]).collect();
        assert_eq!(in_range, vec![700, 600, 500, 400, 300]);
    }
}
