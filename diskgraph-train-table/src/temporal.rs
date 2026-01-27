use std::path::Path;

use anyhow::{anyhow, Result};
use diskgraph_core::metadata::{EdgeRelationMetadata, EdgeSortKeyTimestampKind, EdgeSortKeyWeightKind};
use diskgraph_index::builder::compute_relation_layouts;
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_index::layout::RelationSegment;

use crate::ast::{AggFunc, AggTarget, AggregationExpr, Query, TimeUnit};
use crate::exec::{TargetValue, TrainRow, Value};

pub fn execute_temporal(query: &Query, index_path: &Path) -> Result<Vec<TrainRow>> {
    let metadata_path = index_path.join("index_metadata.json");
    let graph = GraphIndexReader::open(&metadata_path, index_path)?;
    let entity_table = graph
        .table(&query.for_each.table)
        .ok_or_else(|| anyhow!("missing entity table {}", query.for_each.table))?;

    let aggs = collect_aggs(query);
    if aggs.is_empty() {
        return Err(anyhow!("no aggregations found"));
    }

    for agg in &aggs {
        if agg.filter.is_some() {
            return Err(anyhow!("temporal aggregations do not support WHERE filters yet"));
        }
    }

    let relation_map = build_relation_map(entity_table.relations(), graph.metadata())?;
    let _layouts = compute_relation_layouts(entity_table.relations(), &graph.metadata().node_types)?;

    let target_agg_index = 0usize;
    let target_window = aggs
        .get(target_agg_index)
        .and_then(|agg| agg.window.as_ref())
        .ok_or_else(|| anyhow!("missing window"))?;
    let target_window_secs = window_seconds(target_window);

    let mut rows = Vec::new();
    let node_count = entity_table.node_count();
    for node_id in 0..node_count {
        let view = entity_table.edges_for_node(node_id)?;
        let segments = view.relation_segments()?;

        let mut per_agg_ranges = Vec::new();
        let mut target_range = None;
        for (idx, agg) in aggs.iter().enumerate() {
            let rel = agg_relation(agg)?;
            let rel_info = relation_map
                .iter()
                .find(|info| info.name == rel)
                .ok_or_else(|| anyhow!("missing relation {rel}"))?;
            let segment = segments
                .get(rel_info.index)
                .ok_or_else(|| anyhow!("missing segment for relation {rel}"))?;
            let (min_ts, max_ts) = scan_edge_range(segment, &rel_info.meta)?;
            let Some(min_ts) = min_ts else { continue };
            let Some(max_ts) = max_ts else { continue };
            let window = agg.window.as_ref().ok_or_else(|| anyhow!("missing window"))?;
            let (start, end) = window_offsets_seconds(window);
            let range_start = min_ts.saturating_sub(end);
            let range_end = max_ts.saturating_sub(start);
            per_agg_ranges.push((range_start, range_end));
            if idx == target_agg_index {
                target_range = Some((range_start, range_end));
            }
        }

        let Some((target_start, target_end)) = target_range else { continue };
        if target_end <= target_start {
            continue;
        }

        let (global_start, global_end) = per_agg_ranges
            .iter()
            .fold((target_start, target_end), |acc, r| (acc.0.min(r.0), acc.1.max(r.1)));
        if global_end <= global_start {
            continue;
        }

        let range_len = (target_end - target_start).max(0) as f64;
        let sample_denom = (target_window_secs as f64 * 1.5).max(1.0);
        let samples = (range_len / sample_denom).floor() as usize;
        if samples == 0 {
            continue;
        }

        let seed = (node_id as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut rng = XorShift64 { state: seed };
        for _ in 0..samples {
            let ts = global_start + (rng.next_f64() * (global_end - global_start) as f64) as i64;
            let mut target_value = None;

            let mut all_ok = true;
            for (idx, agg) in aggs.iter().enumerate() {
                let rel = agg_relation(agg)?;
                let rel_info = relation_map
                    .iter()
                    .find(|info| info.name == rel)
                    .ok_or_else(|| anyhow!("missing relation {rel}"))?;
                let segment = segments
                    .get(rel_info.index)
                    .ok_or_else(|| anyhow!("missing segment for relation {rel}"))?;
                let value = aggregate_for_timestamp(segment, &rel_info.meta, agg, ts)?;
                if idx == target_agg_index {
                    target_value = value;
                } else if value.is_none() {
                    all_ok = false;
                    break;
                }
            }
            if !all_ok {
                continue;
            }
            let Some(target_value) = target_value else { continue };
            if is_empty_target(&target_value) {
                continue;
            }
            rows.push(TrainRow {
                entity: Value::UInt(node_id),
                target: target_value,
                timestamp: ts,
            });
        }
    }

    Ok(rows)
}

struct RelationInfo {
    name: String,
    index: usize,
    meta: EdgeRelationMetadata,
}

fn build_relation_map(relations: &[EdgeRelationMetadata], metadata: &diskgraph_core::metadata::DatasetMetadata) -> Result<Vec<RelationInfo>> {
    let mut out = Vec::new();
    for (idx, rel) in relations.iter().enumerate() {
        if rel.dest_sort_key.as_ref().and_then(|k| k.timestamp.as_ref()).is_none() {
            continue;
        }
        out.push(RelationInfo {
            name: rel.name.clone(),
            index: idx,
            meta: rel.clone(),
        });
    }
    if out.is_empty() && !metadata.edge_relations.is_empty() {
        // no timestamped relations
    }
    Ok(out)
}

fn collect_aggs(query: &Query) -> Vec<AggregationExpr> {
    let mut aggs = vec![query.predict.aggregation.clone()];
    if let Some(pred) = &query.assuming {
        collect_aggs_from_pred(pred, &mut aggs);
    }
    aggs
}

fn collect_aggs_from_pred(pred: &crate::ast::PredicateExpr, out: &mut Vec<AggregationExpr>) {
    match pred {
        crate::ast::PredicateExpr::Comparison { left, right, .. } => {
            collect_aggs_from_expr(left, out);
            collect_aggs_from_expr(right, out);
        }
        crate::ast::PredicateExpr::And(items) | crate::ast::PredicateExpr::Or(items) => {
            for item in items {
                collect_aggs_from_pred(item, out);
            }
        }
    }
}

fn collect_aggs_from_expr(expr: &crate::ast::Expr, out: &mut Vec<AggregationExpr>) {
    match expr {
        crate::ast::Expr::Agg(agg) => out.push((**agg).clone()),
        _ => {}
    }
}

fn agg_relation(agg: &AggregationExpr) -> Result<&str> {
    match &agg.target {
        AggTarget::Column(col) => Ok(&col.table),
        AggTarget::Star(tbl) => Ok(&tbl.table),
    }
}

fn window_seconds(window: &crate::ast::WindowSpec) -> i64 {
    let (start, end) = window_offsets_seconds(window);
    (end - start).abs().max(1)
}

fn window_offsets_seconds(window: &crate::ast::WindowSpec) -> (i64, i64) {
    let scale = match window.unit {
        TimeUnit::Seconds => 1,
        TimeUnit::Minutes => 60,
        TimeUnit::Hours => 3600,
        TimeUnit::Days => 86_400,
        TimeUnit::Weeks => 604_800,
        TimeUnit::Months => 2_592_000,
        TimeUnit::Years => 31_536_000,
    };
    (window.start_offset.saturating_mul(scale), window.end_offset.saturating_mul(scale))
}

fn scan_edge_range(segment: &RelationSegment<'_>, meta: &EdgeRelationMetadata) -> Result<(Option<i64>, Option<i64>)> {
    let mut min_ts = None;
    let mut max_ts = None;
    for edge in decode_edges(segment) {
        let (ts, _weight) = decode_sort_key(meta, edge.sort_key)?;
        let Some(ts) = ts else { continue };
        if min_ts.map_or(true, |v| ts < v) {
            min_ts = Some(ts);
        }
        if max_ts.map_or(true, |v| ts > v) {
            max_ts = Some(ts);
        }
    }
    Ok((min_ts, max_ts))
}

fn aggregate_for_timestamp(
    segment: &RelationSegment<'_>,
    meta: &EdgeRelationMetadata,
    agg: &AggregationExpr,
    ts: i64,
) -> Result<Option<TargetValue>> {
    let window = agg.window.as_ref().ok_or_else(|| anyhow!("missing window"))?;
    let (start_offset, end_offset) = window_offsets_seconds(window);
    let window_start = ts.saturating_add(start_offset);
    let window_end = ts.saturating_add(end_offset);

    let mut count = 0u64;
    let mut sum = 0f64;
    let mut min = None;
    let mut max = None;
    let mut list: std::collections::HashSet<u64> = std::collections::HashSet::new();

    for edge in decode_edges(segment) {
        let (edge_ts, weight) = decode_sort_key(meta, edge.sort_key)?;
        let Some(edge_ts) = edge_ts else { continue };
        if edge_ts < window_start || edge_ts >= window_end {
            continue;
        }
        count += 1;
        if let Some(weight) = weight {
            sum += weight;
            if min.map_or(true, |v| weight < v) {
                min = Some(weight);
            }
            if max.map_or(true, |v| weight > v) {
                max = Some(weight);
            }
        }
        list.insert(edge.dst);
    }

    let out = match agg.func {
        AggFunc::Count => Some(TargetValue::Count(count)),
        AggFunc::ListDistinct => {
            if list.is_empty() {
                None
            } else {
                Some(TargetValue::List(
                    list.into_iter().map(|id| Value::UInt(id)).collect(),
                ))
            }
        }
        AggFunc::Sum => {
            if count == 0 {
                None
            } else {
                Some(TargetValue::Number(sum))
            }
        }
        AggFunc::Avg => {
            if count == 0 {
                None
            } else {
                Some(TargetValue::Number(sum / count as f64))
            }
        }
        AggFunc::Min => min.map(TargetValue::Number),
        AggFunc::Max => max.map(TargetValue::Number),
    };
    Ok(out)
}

fn is_empty_target(target: &TargetValue) -> bool {
    match target {
        TargetValue::Count(c) => *c == 0,
        TargetValue::List(list) => list.is_empty(),
        TargetValue::Number(_) => false,
    }
}

struct EdgeEntry {
    dst: u64,
    sort_key: u64,
}

fn decode_edges<'a>(segment: &'a RelationSegment<'a>) -> impl Iterator<Item = EdgeEntry> + 'a {
    let entry_bytes = segment.layout.entry_bytes;
    let dst_bits = segment.layout.dst_bits;
    let dst_mask = segment.layout.dst_mask;
    let sort_bits = segment.layout.dest_sort_key_bits.unwrap_or(0);
    let sort_mask = segment.layout.dest_sort_key_mask.unwrap_or(0);
    let data = segment.data;
    let count = segment.count as usize;
    (0..count).filter_map(move |i| {
        let start = i * entry_bytes;
        let end = start + entry_bytes;
        if end > data.len() {
            return None;
        }
        let mut buf = [0u8; 16];
        buf[..entry_bytes].copy_from_slice(&data[start..end]);
        let acc = u128::from_le_bytes(buf);
        let dst = (acc as u64) & dst_mask;
        let sort_key = if sort_bits > 0 {
            ((acc as u64) >> dst_bits) & sort_mask
        } else {
            0
        };
        Some(EdgeEntry { dst, sort_key })
    })
}

fn decode_sort_key(
    rel: &EdgeRelationMetadata,
    raw: u64,
) -> Result<(Option<i64>, Option<f64>)> {
    let meta = rel
        .dest_sort_key
        .as_ref()
        .ok_or_else(|| anyhow!("relation {} missing sort key", rel.name))?;
    let weight_bits = meta.weight.as_ref().map(weight_bits_for_meta).unwrap_or(0);
    let weight_mask = if weight_bits > 0 { (1u64 << weight_bits) - 1 } else { 0 };
    let weight_raw = if weight_bits > 0 { raw & weight_mask } else { 0 };
    let ts_raw = if weight_bits > 0 { raw >> weight_bits } else { raw };

    let ts = match meta.timestamp.as_ref() {
        Some(ts_meta) => {
            let bits = timestamp_bits_for_meta(ts_meta);
            let delta = decode_ordered_signed(ts_raw, bits);
            if delta == min_signed_for_bits(bits) {
                None
            } else {
                Some(delta.saturating_mul(ts_meta.encoding.step).saturating_add(ts_meta.encoding.anchor))
            }
        }
        None => None,
    };

    let weight = meta.weight.as_ref().map(|w| decode_weight_value(w, weight_raw));
    Ok((ts, weight))
}

fn timestamp_bits_for_meta(meta: &diskgraph_core::metadata::EdgeSortKeyTimestampMetadata) -> u8 {
    match meta.encoding.kind {
        EdgeSortKeyTimestampKind::Int8 => 8,
        EdgeSortKeyTimestampKind::Int16 => 16,
        EdgeSortKeyTimestampKind::Int32 => 32,
    }
}

fn weight_bits_for_meta(meta: &diskgraph_core::metadata::EdgeSortKeyWeightMetadata) -> u8 {
    match meta.encoding.dtype {
        EdgeSortKeyWeightKind::Byte => 8,
        EdgeSortKeyWeightKind::UInt16 => 16,
        EdgeSortKeyWeightKind::UInt32 => 32,
        EdgeSortKeyWeightKind::Float16 => 16,
        EdgeSortKeyWeightKind::Float32 => 32,
    }
}

fn decode_ordered_signed(raw: u64, bits: u8) -> i64 {
    let mask = 1u64 << (bits - 1);
    let decoded = raw ^ mask;
    match bits {
        8 => (decoded as u8 as i8) as i64,
        16 => (decoded as u16 as i16) as i64,
        32 => (decoded as u32 as i32) as i64,
        64 => decoded as i64,
        _ => decoded as i64,
    }
}

fn min_signed_for_bits(bits: u8) -> i64 {
    match bits {
        8 => i8::MIN as i64,
        16 => i16::MIN as i64,
        32 => i32::MIN as i64,
        64 => i64::MIN,
        _ => i64::MIN,
    }
}

fn decode_weight_value(
    meta: &diskgraph_core::metadata::EdgeSortKeyWeightMetadata,
    raw: u64,
) -> f64 {
    match meta.encoding.dtype {
        EdgeSortKeyWeightKind::Byte => raw as f64,
        EdgeSortKeyWeightKind::UInt16 => raw as f64,
        EdgeSortKeyWeightKind::UInt32 => raw as f64,
        EdgeSortKeyWeightKind::Float16 => {
            let bits = raw as u16;
            half::f16::from_bits(bits).to_f64()
        }
        EdgeSortKeyWeightKind::Float32 => {
            let bits = raw as u32;
            f32::from_bits(bits) as f64
        }
    }
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        let v = self.next_u64();
        (v as f64) / (u64::MAX as f64)
    }
}
