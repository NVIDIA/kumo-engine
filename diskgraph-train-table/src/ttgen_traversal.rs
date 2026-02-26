//! Edge traversal primitives for TT-gen.
//!
//! `iter_edge` reads edges from a relation segment.
//! `traverse_and_accumulate` recursively walks multi-hop paths.

use diskgraph_core::tt_builder::Value;
use diskgraph_index::graph::TableReader;
use diskgraph_index::layout::RelationLayout;
use diskgraph_rowpack::{RowPackDecodeScratch, ValueRef};
use std::sync::Arc;

use crate::ttgen_resolve::ResolvedEdge;

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
/// Returns a Vec of (dst_node, sort_key) pairs. When `sort_key_range` is
/// Some((lo, hi)), only edges with lo <= sort_key < hi are returned.
#[inline]
pub fn iter_edge(
    src_table: &Arc<TableReader>,
    relation_idx: usize,
    layout: &RelationLayout,
    src_node: u64,
    sort_key_range: Option<(u64, u64)>,
) -> Vec<EdgeEntry> {
    let view = match src_table.edges_for_node(src_node) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let segments = match view.relation_segments() {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let segment = match segments.get(relation_idx) {
        Some(s) => s,
        None => return Vec::new(),
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
            let end = partition_point_by(num_entries, |i| {
                decode_entry(i).sort_key > lo
            });
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
    window: Option<(u64, u64)>,
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

    Some(value_ref_to_value(&value_ref))
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
/// Returns 0.0 if not found or not numeric.
pub fn read_column_f64(
    table: &Arc<TableReader>,
    node_id: u64,
    col_idx: usize,
) -> f64 {
    read_column_value(table, node_id, col_idx)
        .map(|v| match v {
            Value::Int(i) => i as f64,
            Value::Float(f) => f,
            _ => 0.0,
        })
        .unwrap_or(0.0)
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
