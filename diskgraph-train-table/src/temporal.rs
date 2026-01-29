use std::path::Path;
use std::io::IsTerminal;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use diskgraph_core::metadata::{EdgeRelationMetadata, EdgeSortKeyTimestampKind, EdgeSortKeyWeightKind};
use diskgraph_index::builder::compute_relation_layouts;
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_index::layout::RelationSegment;
use indicatif::{ProgressBar, ProgressStyle};
use diskgraph_rowpack::{IngestedType, RowPackDecodeScratch, RowPackDecoder, ValueRef};

use crate::ast::{AggFunc, AggTarget, AggregationExpr, PredicateExpr, PredictExpr, Query, TimeUnit};
use crate::config::{QueryOptions, TemporalSampling};
use crate::exec::{TargetValue, TrainRow, Value};

pub fn execute_temporal(query: &Query, index_path: &Path, options: &QueryOptions) -> Result<Vec<TrainRow>> {
    let metadata_path = index_path.join("index_metadata.json");
    let graph = GraphIndexReader::open(&metadata_path, index_path)?;
    execute_temporal_with_reader(query, &graph, options, 0, None, None)
}

pub fn execute_temporal_range(
    query: &Query,
    index_path: &Path,
    options: &QueryOptions,
    start: u64,
    end: u64,
) -> Result<Vec<TrainRow>> {
    let metadata_path = index_path.join("index_metadata.json");
    let graph = GraphIndexReader::open(&metadata_path, index_path)?;
    execute_temporal_with_reader(query, &graph, options, start, Some(end), None)
}

pub fn execute_temporal_range_with_progress(
    query: &Query,
    index_path: &Path,
    options: &QueryOptions,
    start: u64,
    end: u64,
    progress: Option<TemporalProgress>,
) -> Result<Vec<TrainRow>> {
    let metadata_path = index_path.join("index_metadata.json");
    let graph = GraphIndexReader::open(&metadata_path, index_path)?;
    execute_temporal_with_reader(query, &graph, options, start, Some(end), progress)
}

#[derive(Clone)]
pub struct TemporalProgress {
    bar: ProgressBar,
    counter: Arc<AtomicU64>,
}

impl TemporalProgress {
    pub fn new(table: &str, total: u64) -> Self {
        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {percent}% {pos}/{len}")
                .unwrap_or_else(|_| ProgressStyle::default_bar()),
        );
        bar.set_message(format!("temporal {table}"));
        Self {
            bar,
            counter: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn advance(&self, delta: u64) {
        let new_pos = self.counter.fetch_add(delta, Ordering::Relaxed) + delta;
        self.bar.set_position(new_pos);
    }

    pub fn finish(&self, rows: usize) {
        self.bar.finish_with_message(format!(
            "{} done rows={}",
            self.bar.message(),
            rows
        ));
    }
}

fn execute_temporal_with_reader(
    query: &Query,
    graph: &GraphIndexReader,
    options: &QueryOptions,
    start: u64,
    end: Option<u64>,
    progress: Option<TemporalProgress>,
) -> Result<Vec<TrainRow>> {
    if !matches!(query.predict, PredictExpr::Aggregation(_) | PredictExpr::Comparison(_)) {
        return Err(anyhow!("temporal execution requires aggregation or comparison predict"));
    }
    let entity_table = graph
        .table(&query.for_each.table)
        .ok_or_else(|| anyhow!("missing entity table {}", query.for_each.table))?;

    let aggs = collect_aggs(query);
    if aggs.is_empty() {
        return Err(anyhow!("no aggregations found"));
    }

    let relation_map = build_relation_map(entity_table.relations(), graph.metadata())?;
    let _layouts = compute_relation_layouts(entity_table.relations(), &graph.metadata().node_types)?;

    let target_agg_index = 0usize;
    let target_window = aggs
        .get(target_agg_index)
        .and_then(|agg| agg.window.as_ref())
        .ok_or_else(|| anyhow!("missing window"))?;
    let target_window_secs = window_seconds(target_window);

    let mut agg_plans = Vec::new();
    for agg in &aggs {
        agg_plans.push(build_agg_plan(agg, &query.for_each.table, &graph, &relation_map)?);
    }

    let Some((global_start, global_end)) = compute_global_time_range(&entity_table, &agg_plans)? else {
        return Ok(Vec::new());
    };

    let entity_decoder = entity_table.node_data_decoder();
    let entity_time_idx = resolve_entity_time_index(entity_decoder, options)?;
    let mut entity_scratch = RowPackDecodeScratch::default();

    let mut rows = Vec::new();
    let node_count = entity_table.node_count();
    let end = end.unwrap_or(node_count).min(node_count);
    let start = start.min(end);
    let (progress, owns_progress) = if let Some(progress) = progress {
        (Some(progress), false)
    } else if std::io::stderr().is_terminal() && start == 0 && end == node_count {
        (Some(TemporalProgress::new(&query.for_each.table, end - start)), true)
    } else {
        (None, false)
    };
    let mut local_progress = 0u64;
    for node_id in start..end {
        let view = entity_table.edges_for_node(node_id)?;
        let segments = view.relation_segments()?;

        let mut sample_start = global_start;
        if let Some(idx) = entity_time_idx {
            let decoder = entity_decoder.ok_or_else(|| {
                anyhow!(
                    "entity time column requires node data for table {}",
                    entity_table.node_type()
                )
            })?;
            let data = view
                .node_data()?
                .ok_or_else(|| anyhow!("missing node data for entity table {}", entity_table.node_type()))?;
            let ts = decode_entity_timestamp(decoder, data, idx, &mut entity_scratch)?;
            let Some(ts) = ts else { continue };
            if ts > sample_start {
                sample_start = ts;
            }
        }
        let sample_end = global_end;
        if sample_end <= sample_start {
            continue;
        }

        match options.temporal_sampling {
            TemporalSampling::Random => {
                let range_len = (sample_end - sample_start).max(0) as f64;
                let sample_denom = (target_window_secs as f64 * 1.5).max(1.0);
                let samples = (range_len / sample_denom).floor().max(1.0) as usize;
                let seed = (node_id as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut rng = XorShift64 { state: seed };
                for _ in 0..samples {
                    let ts =
                        sample_start + (rng.next_f64() * (sample_end - sample_start) as f64) as i64;
                        evaluate_samples(
                            node_id,
                            ts,
                            &segments,
                            &agg_plans,
                            &aggs,
                            target_agg_index,
                            &query.predict,
                            query.assuming.as_ref(),
                            &mut rows,
                        )?;
                }
            }
            TemporalSampling::Timeframe => {
                let mut ts = sample_start;
                while ts < sample_end {
                        evaluate_samples(
                            node_id,
                            ts,
                            &segments,
                            &agg_plans,
                            &aggs,
                            target_agg_index,
                            &query.predict,
                            query.assuming.as_ref(),
                            &mut rows,
                        )?;
                    ts = ts.saturating_add(target_window_secs);
                }
            }
        }
        local_progress += 1;
        if node_id > 0 && node_id % 10_000 == 0 {
            if let Some(bar) = &progress {
                bar.advance(local_progress);
                local_progress = 0;
            }
        }
    }

    if let Some(bar) = &progress {
        if local_progress > 0 {
            bar.advance(local_progress);
        }
        if owns_progress {
            bar.finish(rows.len());
        }
    }
    Ok(rows)
}

fn compute_global_time_range(
    entity_table: &std::sync::Arc<diskgraph_index::graph::TableReader>,
    agg_plans: &[AggPlan],
) -> Result<Option<(i64, i64)>> {
    let mut min_ts: Option<i64> = None;
    let mut max_ts: Option<i64> = None;
    let node_count = entity_table.node_count();
    for node_id in 0..node_count {
        let view = entity_table.edges_for_node(node_id)?;
        let segments = view.relation_segments()?;
        for plan in agg_plans {
            let segment = segments
                .get(plan.rel.index)
                .ok_or_else(|| anyhow!("missing segment for relation {}", plan.rel.name))?;
            let (seg_min, seg_max) = scan_edge_range(segment, &plan.rel.meta)?;
            if let Some(ts) = seg_min {
                min_ts = Some(min_ts.map_or(ts, |v| v.min(ts)));
            }
            if let Some(ts) = seg_max {
                max_ts = Some(max_ts.map_or(ts, |v| v.max(ts)));
            }
        }
    }
    Ok(match (min_ts, max_ts) {
        (Some(min), Some(max)) => Some((min, max)),
        _ => None,
    })
}

fn resolve_entity_time_index(
    decoder: Option<&RowPackDecoder>,
    options: &QueryOptions,
) -> Result<Option<usize>> {
    if let Some(name) = options.entity_time_column.as_deref() {
        let Some(decoder) = decoder else {
            return Err(anyhow!("entity time column requires node data"));
        };
        let idx = decoder
            .context()
            .columns
            .iter()
            .position(|col| col.name == name)
            .ok_or_else(|| anyhow!("missing entity time column {name}"))?;
        return Ok(Some(idx));
    }
    let Some(decoder) = decoder else { return Ok(None); };
    Ok(find_timestamp_index(decoder.context()))
}

fn find_timestamp_index(ctx: &diskgraph_rowpack::RowPackContext) -> Option<usize> {
    for (idx, col) in ctx.columns.iter().enumerate() {
        if matches!(col.logical_type, IngestedType::Timestamp | IngestedType::Date) {
            return Some(idx);
        }
    }
    for name in ["timestamp", "time", "ts"] {
        if let Some(idx) = ctx.columns.iter().position(|col| col.name == name) {
            return Some(idx);
        }
    }
    None
}

fn decode_entity_timestamp(
    decoder: &RowPackDecoder,
    row: &[u8],
    idx: usize,
    scratch: &mut RowPackDecodeScratch,
) -> Result<Option<i64>> {
    let value = decoder.decode_column_value(row, idx, scratch)?;
    Ok(match value {
        ValueRef::Timestamp { secs, .. } => Some(secs),
        ValueRef::Date(days) => Some(days as i64 * 86_400),
        ValueRef::Int64(v) => Some(v),
        ValueRef::Int32(v) => Some(v as i64),
        ValueRef::UInt64(v) => Some(v as i64),
        ValueRef::UInt32(v) => Some(v as i64),
        ValueRef::Null => None,
        other => return Err(anyhow!("unsupported entity time type: {other:?}")),
    })
}

fn evaluate_samples(
    node_id: u64,
    ts: i64,
    segments: &[RelationSegment<'_>],
    agg_plans: &[AggPlan],
    aggs: &[AggregationExpr],
    target_agg_index: usize,
    predict: &PredictExpr,
    assuming: Option<&PredicateExpr>,
    rows: &mut Vec<TrainRow>,
) -> Result<()> {
    let mut values: Vec<Option<TargetValue>> = Vec::with_capacity(agg_plans.len());
    for (_idx, plan) in agg_plans.iter().enumerate() {
        let segment = segments
            .get(plan.rel.index)
            .ok_or_else(|| anyhow!("missing segment for relation {}", plan.rel.name))?;
        let value = aggregate_for_timestamp(segment, &plan.rel.meta, &plan.agg, ts, plan)?;
        values.push(value);
    }
    if let Some(pred) = assuming {
        if !eval_temporal_predicate(pred, aggs, &values)? {
            return Ok(());
        }
    }
    let target_value = match predict {
        PredictExpr::Aggregation(_) => {
            let Some(target_value) = values.get(target_agg_index).and_then(|v| v.clone()) else {
                return Ok(());
            };
            if is_empty_target(&target_value) {
                return Ok(());
            }
            target_value
        }
        PredictExpr::Comparison(pred) => {
            let label = eval_temporal_predicate(pred, aggs, &values)?;
            TargetValue::Scalar(Value::Bool(label))
        }
        PredictExpr::Column(_) => return Err(anyhow!("temporal predict column not supported")),
    };
    rows.push(TrainRow {
        entity: Value::UInt(node_id),
        target: target_value,
        timestamp: ts,
    });
    Ok(())
}

#[derive(Clone)]
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

struct AggPlan {
    agg: AggregationExpr,
    rel: RelationInfo,
    dst_table: Option<std::sync::Arc<diskgraph_index::graph::TableReader>>,
    target_col_idx: Option<usize>,
    filter: Option<PredicateExpr>,
    filter_col_index: Option<std::collections::HashMap<String, usize>>,
}

fn build_agg_plan(
    agg: &AggregationExpr,
    entity_table: &str,
    graph: &GraphIndexReader,
    relation_map: &[RelationInfo],
) -> Result<AggPlan> {
    let (rel_name, dst_table_name) = resolve_relation(entity_table, agg, graph.metadata())?;
    let rel = relation_map
        .iter()
        .find(|info| info.name == rel_name)
        .ok_or_else(|| anyhow!("missing relation {rel_name}"))?
        .clone();

    let mut target_col_idx = None;
    let mut filter_col_index = None;
    let dst_table = dst_table_name
        .as_deref()
        .and_then(|name| graph.table(name));

    let filter = agg.filter.clone().map(|f| *f);
    if let Some(filter) = &filter {
        let dst = dst_table
            .as_ref()
            .ok_or_else(|| anyhow!("temporal filter requires dst node table"))?;
        let decoder = dst
            .node_data_decoder()
            .ok_or_else(|| anyhow!("dst table missing node data decoder"))?;
        let map = decoder
            .context()
            .columns
            .iter()
            .enumerate()
            .map(|(idx, col)| (col.name.clone(), idx))
            .collect::<std::collections::HashMap<_, _>>();
        filter_col_index = Some(map);
        for col in collect_columns_from_pred(filter) {
            if col.table != dst.node_type() {
                return Err(anyhow!("temporal filter only supports dst node columns"));
            }
        }
    }

    match &agg.target {
        AggTarget::Column(col) => {
            let dst = dst_table
                .as_ref()
                .ok_or_else(|| anyhow!("temporal agg requires dst node table"))?;
            let decoder = dst
                .node_data_decoder()
                .ok_or_else(|| anyhow!("dst table missing node data decoder"))?;
            let idx = decoder
                .context()
                .columns
                .iter()
                .position(|c| c.name == col.column)
                .ok_or_else(|| anyhow!("missing target column {}", col.column))?;
            target_col_idx = Some(idx);
        }
        AggTarget::Star(_) => {
            if agg.func != AggFunc::Count {
                return Err(anyhow!("only COUNT(*) supported for relation targets"));
            }
        }
    }

    Ok(AggPlan {
        agg: agg.clone(),
        rel,
        dst_table,
        target_col_idx,
        filter,
        filter_col_index,
    })
}

fn resolve_relation(
    entity_table: &str,
    agg: &AggregationExpr,
    metadata: &diskgraph_core::metadata::DatasetMetadata,
) -> Result<(String, Option<String>)> {
    match &agg.target {
        AggTarget::Star(tbl) => {
            let rel = metadata
                .edge_relations
                .iter()
                .find(|rel| rel.name == tbl.table);
            if let Some(rel) = rel {
                let is_node_table = metadata.node_types.iter().any(|t| t.name == tbl.table);
                return Ok((rel.name.clone(), is_node_table.then(|| tbl.table.clone())));
            }
            let is_node_table = metadata.node_types.iter().any(|t| t.name == tbl.table);
            let mut candidates: Vec<_> = metadata
                .edge_relations
                .iter()
                .filter(|rel| rel.dst_type == tbl.table)
                .collect();
            let by_src: Vec<_> = candidates
                .iter()
                .copied()
                .filter(|rel| rel.src_type == entity_table)
                .collect();
            if by_src.len() == 1 {
                return Ok((by_src[0].name.clone(), is_node_table.then(|| tbl.table.clone())));
            }
            if by_src.is_empty() && candidates.len() == 1 {
                return Ok((candidates[0].name.clone(), is_node_table.then(|| tbl.table.clone())));
            }
            candidates.sort_by(|a, b| a.name.cmp(&b.name));
            if candidates.is_empty() {
                Err(anyhow!("temporal aggregation requires edge relation {}", tbl.table))
            } else {
                Err(anyhow!(
                    "multiple relations point to {} (use relation name): {}",
                    tbl.table,
                    candidates
                        .iter()
                        .map(|rel| rel.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            }
        }
        AggTarget::Column(col) => {
            let is_node_table = metadata.node_types.iter().any(|t| t.name == col.table);
            if metadata.edge_relations.iter().any(|r| r.name == col.table) && !is_node_table {
                return Err(anyhow!("edge tables have no columns for temporal queries"));
            }
            let mut rels = metadata
                .edge_relations
                .iter()
                .filter(|rel| rel.src_type == entity_table && rel.dst_type == col.table)
                .map(|rel| rel.name.clone())
                .collect::<Vec<_>>();
            if rels.is_empty() {
                return Err(anyhow!("no relation from {} to {}", entity_table, col.table));
            }
            if rels.len() > 1 {
                return Err(anyhow!("ambiguous relation from {} to {}", entity_table, col.table));
            }
            Ok((rels.remove(0), Some(col.table.clone())))
        }
    }
}

fn collect_aggs(query: &Query) -> Vec<AggregationExpr> {
    let mut aggs = match &query.predict {
        crate::ast::PredictExpr::Aggregation(agg) => vec![agg.clone()],
        crate::ast::PredictExpr::Comparison(pred) => {
            let mut out = Vec::new();
            collect_aggs_from_pred(pred, &mut out);
            out
        }
        _ => Vec::new(),
    };
    if let Some(pred) = &query.assuming {
        collect_aggs_from_pred(pred, &mut aggs);
    }
    aggs
}

fn eval_temporal_predicate(
    pred: &PredicateExpr,
    aggs: &[AggregationExpr],
    values: &[Option<TargetValue>],
) -> Result<bool> {
    match pred {
        PredicateExpr::Comparison { left, op, right } => {
            let left_num = eval_predicate_expr(left, aggs, values)?;
            let right_num = eval_predicate_expr(right, aggs, values)?;
            let (Some(left_num), Some(right_num)) = (left_num, right_num) else {
                return Ok(false);
            };
            Ok(compare_numeric(left_num, right_num, *op))
        }
        PredicateExpr::And(items) => {
            for item in items {
                if !eval_temporal_predicate(item, aggs, values)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        PredicateExpr::Or(items) => {
            for item in items {
                if eval_temporal_predicate(item, aggs, values)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }
}

fn eval_predicate_expr(
    expr: &crate::ast::Expr,
    aggs: &[AggregationExpr],
    values: &[Option<TargetValue>],
) -> Result<Option<f64>> {
    match expr {
        crate::ast::Expr::Number(n) => Ok(Some(*n)),
        crate::ast::Expr::Agg(agg) => {
            let idx = aggs
                .iter()
                .position(|a| a == agg.as_ref())
                .ok_or_else(|| anyhow!("missing agg for predicate"))?;
            let Some(value) = values.get(idx).and_then(|v| v.clone()) else {
                return Ok(None);
            };
            Ok(target_value_to_number(&value))
        }
        crate::ast::Expr::String(_) | crate::ast::Expr::Column(_) => {
            Err(anyhow!("unsupported predicate expr in temporal assuming"))
        }
    }
}

fn target_value_to_number(value: &TargetValue) -> Option<f64> {
    match value {
        TargetValue::Count(c) => Some(*c as f64),
        TargetValue::Number(v) => Some(*v),
        TargetValue::List(_) => None,
        TargetValue::Scalar(v) => match v {
            Value::Int(v) => Some(*v as f64),
            Value::UInt(v) => Some(*v as f64),
            Value::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            Value::String(_) | Value::Bytes(_) => None,
        },
    }
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
    plan: &AggPlan,
) -> Result<Option<TargetValue>> {
    let window = agg.window.as_ref().ok_or_else(|| anyhow!("missing window"))?;
    let (start_offset, end_offset) = window_offsets_seconds(window);
    let window_start = ts.saturating_add(start_offset);
    let window_end = ts.saturating_add(end_offset);

    let mut count = 0u64;
    let mut sum = 0f64;
    let mut min = None;
    let mut max = None;
    let mut list: std::collections::HashSet<Value> = std::collections::HashSet::new();

    for edge in decode_edges(segment) {
        let (edge_ts, _weight) = decode_sort_key(meta, edge.sort_key)?;
        let Some(edge_ts) = edge_ts else { continue };
        if edge_ts < window_start || edge_ts >= window_end {
            continue;
        }
        if let Some(filter) = &plan.filter {
            if !eval_node_predicate(
                plan.dst_table.as_deref(),
                plan.filter_col_index.as_ref(),
                edge.dst,
                filter,
            )? {
                continue;
            }
        }
        count += 1;
        match agg.func {
            AggFunc::Count => {}
            AggFunc::Sum | AggFunc::Avg | AggFunc::Min | AggFunc::Max => {
                let Some(value) = decode_target_numeric(plan, edge.dst)? else { continue };
                sum += value;
                if min.map_or(true, |v| value < v) {
                    min = Some(value);
                }
                if max.map_or(true, |v| value > v) {
                    max = Some(value);
                }
            }
            AggFunc::ListDistinct => {
                let Some(value) = decode_target_value(plan, edge.dst)? else { continue };
                list.insert(value);
            }
        }
    }

    let out = match agg.func {
        AggFunc::Count => Some(TargetValue::Count(count)),
        AggFunc::ListDistinct => {
            if list.is_empty() {
                None
            } else {
                Some(TargetValue::List(list.into_iter().collect()))
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

fn eval_node_predicate(
    dst_table: Option<&diskgraph_index::graph::TableReader>,
    col_index: Option<&std::collections::HashMap<String, usize>>,
    node_id: u64,
    pred: &PredicateExpr,
) -> Result<bool> {
    let Some(dst_table) = dst_table else { return Err(anyhow!("missing dst table")); };
    let Some(decoder) = dst_table.node_data_decoder() else { return Err(anyhow!("missing node data decoder")); };
    let Some(col_index) = col_index else { return Err(anyhow!("missing predicate column index")); };
    let view = dst_table.edges_for_node(node_id)?;
    let data = view
        .node_data()?
        .ok_or_else(|| anyhow!("missing dst node data"))?;
    let mut scratch = diskgraph_rowpack::RowPackDecodeScratch::default();
    eval_predicate_row(pred, decoder, &data, col_index, &mut scratch)
}

fn eval_predicate_row(
    pred: &PredicateExpr,
    decoder: &diskgraph_rowpack::RowPackDecoder,
    row: &[u8],
    col_index: &std::collections::HashMap<String, usize>,
    scratch: &mut diskgraph_rowpack::RowPackDecodeScratch,
) -> Result<bool> {
    match pred {
        PredicateExpr::Comparison { left, op, right } => {
            let left_val = eval_expr_value(left, decoder, row, col_index, scratch)?;
            let right_val = eval_expr_value(right, decoder, row, col_index, scratch)?;
            Ok(compare_values(&left_val, &right_val, *op))
        }
        PredicateExpr::And(items) => {
            for item in items {
                if !eval_predicate_row(item, decoder, row, col_index, scratch)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        PredicateExpr::Or(items) => {
            for item in items {
                if eval_predicate_row(item, decoder, row, col_index, scratch)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }
}

#[derive(Debug, Clone)]
enum SimpleValue {
    Number(f64),
    String(String),
    Bool(bool),
    Null,
}

fn eval_expr_value(
    expr: &crate::ast::Expr,
    decoder: &diskgraph_rowpack::RowPackDecoder,
    row: &[u8],
    col_index: &std::collections::HashMap<String, usize>,
    scratch: &mut diskgraph_rowpack::RowPackDecodeScratch,
) -> Result<SimpleValue> {
    match expr {
        crate::ast::Expr::Number(n) => Ok(SimpleValue::Number(*n as f64)),
        crate::ast::Expr::String(s) => Ok(SimpleValue::String(s.clone())),
        crate::ast::Expr::Column(col) => {
            let idx = *col_index
                .get(&col.column)
                .ok_or_else(|| anyhow!("missing column {}", col.column))?;
            let value = decoder.decode_column_value(row, idx, scratch)?;
            Ok(match value {
                diskgraph_rowpack::ValueRef::Int64(v) => SimpleValue::Number(v as f64),
                diskgraph_rowpack::ValueRef::Int32(v) => SimpleValue::Number(v as f64),
                diskgraph_rowpack::ValueRef::UInt64(v) => SimpleValue::Number(v as f64),
                diskgraph_rowpack::ValueRef::UInt32(v) => SimpleValue::Number(v as f64),
                diskgraph_rowpack::ValueRef::Float64(v) => SimpleValue::Number(v),
                diskgraph_rowpack::ValueRef::Float32(v) => SimpleValue::Number(v as f64),
                diskgraph_rowpack::ValueRef::String(v) => SimpleValue::String(v.to_string()),
                diskgraph_rowpack::ValueRef::Bool(v) => SimpleValue::Bool(v),
                diskgraph_rowpack::ValueRef::Null => SimpleValue::Null,
                diskgraph_rowpack::ValueRef::Date(v) => SimpleValue::Number(v as f64),
                diskgraph_rowpack::ValueRef::Timestamp { secs, .. } => SimpleValue::Number(secs as f64),
                diskgraph_rowpack::ValueRef::Decimal128(v) => SimpleValue::Number(v as f64),
                diskgraph_rowpack::ValueRef::Decimal256(_) => {
                    return Err(anyhow!("unsupported predicate value"))
                }
                diskgraph_rowpack::ValueRef::Bytes(v) => {
                    SimpleValue::String(hex_encode(v))
                }
                diskgraph_rowpack::ValueRef::Uuid(bytes) => {
                    SimpleValue::String(hex_encode(&bytes))
                }
                diskgraph_rowpack::ValueRef::Uuid256(bytes) => {
                    SimpleValue::String(hex_encode(&bytes))
                }
                diskgraph_rowpack::ValueRef::Ip(bytes) => {
                    SimpleValue::String(hex_encode(&bytes))
                }
                diskgraph_rowpack::ValueRef::Float32Array(_)
                | diskgraph_rowpack::ValueRef::Float64Array(_) => {
                    return Err(anyhow!("unsupported predicate value"))
                }
            })
        }
        crate::ast::Expr::Agg(_) => Err(anyhow!("nested agg in predicate not supported")),
    }
}

fn compare_values(left: &SimpleValue, right: &SimpleValue, op: crate::ast::CompareOp) -> bool {
    if matches!(left, SimpleValue::Null) || matches!(right, SimpleValue::Null) {
        return false;
    }
    match (left, right) {
        (SimpleValue::Number(l), SimpleValue::Number(r)) => compare_numeric(*l, *r, op),
        (SimpleValue::String(l), SimpleValue::String(r)) => match op {
            crate::ast::CompareOp::Eq => l == r,
            crate::ast::CompareOp::Neq => l != r,
            _ => false,
        },
        (SimpleValue::Bool(l), SimpleValue::Bool(r)) => match op {
            crate::ast::CompareOp::Eq => l == r,
            crate::ast::CompareOp::Neq => l != r,
            _ => false,
        },
        _ => false,
    }
}

fn compare_numeric(left: f64, right: f64, op: crate::ast::CompareOp) -> bool {
    match op {
        crate::ast::CompareOp::Eq => left == right,
        crate::ast::CompareOp::Neq => left != right,
        crate::ast::CompareOp::Gt => left > right,
        crate::ast::CompareOp::Gte => left >= right,
        crate::ast::CompareOp::Lt => left < right,
        crate::ast::CompareOp::Lte => left <= right,
    }
}

fn decode_target_numeric(plan: &AggPlan, node_id: u64) -> Result<Option<f64>> {
    let dst_table = plan
        .dst_table
        .as_deref()
        .ok_or_else(|| anyhow!("missing dst table"))?;
    let decoder = dst_table
        .node_data_decoder()
        .ok_or_else(|| anyhow!("missing dst node decoder"))?;
    let idx = plan
        .target_col_idx
        .ok_or_else(|| anyhow!("missing target column index"))?;
    let view = dst_table.edges_for_node(node_id)?;
    let data = view
        .node_data()?
        .ok_or_else(|| anyhow!("missing dst node data"))?;
    let mut scratch = diskgraph_rowpack::RowPackDecodeScratch::default();
    let value = decoder.decode_column_value(data, idx, &mut scratch)?;
    Ok(Some(match value {
        diskgraph_rowpack::ValueRef::Null => return Ok(None),
        diskgraph_rowpack::ValueRef::Int64(v) => v as f64,
        diskgraph_rowpack::ValueRef::Int32(v) => v as f64,
        diskgraph_rowpack::ValueRef::UInt64(v) => v as f64,
        diskgraph_rowpack::ValueRef::UInt32(v) => v as f64,
        diskgraph_rowpack::ValueRef::Float64(v) => v,
        diskgraph_rowpack::ValueRef::Float32(v) => v as f64,
        diskgraph_rowpack::ValueRef::Bool(v) => if v { 1.0 } else { 0.0 },
        _ => return Err(anyhow!("target not numeric")),
    }))
}

fn decode_target_value(plan: &AggPlan, node_id: u64) -> Result<Option<Value>> {
    let dst_table = plan
        .dst_table
        .as_deref()
        .ok_or_else(|| anyhow!("missing dst table"))?;
    let decoder = dst_table
        .node_data_decoder()
        .ok_or_else(|| anyhow!("missing dst node decoder"))?;
    let idx = plan
        .target_col_idx
        .ok_or_else(|| anyhow!("missing target column index"))?;
    let view = dst_table.edges_for_node(node_id)?;
    let data = view
        .node_data()?
        .ok_or_else(|| anyhow!("missing dst node data"))?;
    let mut scratch = diskgraph_rowpack::RowPackDecodeScratch::default();
    let value = decoder.decode_column_value(&data, idx, &mut scratch)?;
    Ok(Some(match value {
        diskgraph_rowpack::ValueRef::Null => return Ok(None),
        diskgraph_rowpack::ValueRef::Int64(v) => Value::Int(v),
        diskgraph_rowpack::ValueRef::Int32(v) => Value::Int(v as i64),
        diskgraph_rowpack::ValueRef::UInt64(v) => Value::UInt(v),
        diskgraph_rowpack::ValueRef::UInt32(v) => Value::UInt(v as u64),
        diskgraph_rowpack::ValueRef::Bool(v) => Value::Bool(v),
        diskgraph_rowpack::ValueRef::String(v) => Value::String(v.to_string()),
        diskgraph_rowpack::ValueRef::Bytes(v) => Value::Bytes(v.to_vec()),
        diskgraph_rowpack::ValueRef::Uuid(bytes) => Value::String(hex_encode(&bytes)),
        diskgraph_rowpack::ValueRef::Uuid256(bytes) => Value::String(hex_encode(&bytes)),
        diskgraph_rowpack::ValueRef::Ip(bytes) => Value::String(hex_encode(&bytes)),
        other => return Err(anyhow!("unsupported target value type: {other:?}")),
    }))
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn is_empty_target(target: &TargetValue) -> bool {
    match target {
        TargetValue::Count(c) => *c == 0,
        TargetValue::List(list) => list.is_empty(),
        TargetValue::Number(_) => false,
        TargetValue::Scalar(_) => false,
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
fn collect_columns_from_pred(pred: &PredicateExpr) -> Vec<crate::ast::ColumnRef> {
    let mut out = Vec::new();
    collect_columns_from_pred_inner(pred, &mut out);
    out
}

fn collect_columns_from_pred_inner(pred: &PredicateExpr, out: &mut Vec<crate::ast::ColumnRef>) {
    match pred {
        PredicateExpr::Comparison { left, right, .. } => {
            collect_columns_from_expr(left, out);
            collect_columns_from_expr(right, out);
        }
        PredicateExpr::And(items) | PredicateExpr::Or(items) => {
            for item in items {
                collect_columns_from_pred_inner(item, out);
            }
        }
    }
}

fn collect_columns_from_expr(expr: &crate::ast::Expr, out: &mut Vec<crate::ast::ColumnRef>) {
    match expr {
        crate::ast::Expr::Column(col) => out.push(col.clone()),
        crate::ast::Expr::Agg(_) | crate::ast::Expr::Number(_) | crate::ast::Expr::String(_) => {}
    }
}
