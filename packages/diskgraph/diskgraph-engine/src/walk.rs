// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use diskgraph_core::indexed_time_range::{unclipped_bucket_end, IndexedTimeRange};
use diskgraph_index::layout::EdgeListView;
use diskgraph_rowpack::{RowPackDecodeScratch, ValueRef};
use std::collections::{BTreeMap, VecDeque};
use std::ops::Range;

#[derive(Default)]
struct WalkResult {
    hits: Vec<(Node, f32, u32)>,
    pushes: u64,
    decoded: u64,
}

type Node = (usize, u64);

struct Step {
    local_rel: u16,
    destination: usize,
    time_range: Option<IndexedTimeRange>,
}

pub(super) struct CompiledWalk {
    name: String,
    // Paths grouped by their starting table. Link rows are traversed internally.
    paths: HashMap<usize, Vec<Vec<Step>>>,
    keep: BTreeMap<usize, usize>,
    clocks: HashMap<usize, usize>,
    restart: f64,
    budget: u32,
}

impl CompiledWalk {
    pub(super) fn new(engine: &DiskGraphEngine, spec: &WalkSpec) -> Result<Self> {
        if spec.name.is_empty()
            || spec.paths.is_empty()
            || spec.keep.is_empty()
            || !(0.0 < spec.restart && spec.restart < 1.0)
            || 1.0 - spec.restart == 1.0
            || spec.budget == 0
        {
            return Err(EngineError::InvalidRequest(
                "invalid walk name, paths, quotas, restart or budget".into(),
            ));
        }
        let type_id =
            |name: &String| {
                engine.node_type_to_id.get(name).copied().ok_or_else(|| {
                    EngineError::InvalidRequest(format!("unknown walk table {name}"))
                })
            };
        let mut clocks = HashMap::new();
        for (table, column) in &spec.clocks {
            let id = type_id(table)?;
            let index = engine.node_decoders[id]
                .as_ref()
                .and_then(|decoder| {
                    decoder
                        .context()
                        .columns
                        .iter()
                        .position(|col| &col.name == column)
                })
                .ok_or_else(|| {
                    EngineError::InvalidRequest(format!("unknown walk clock {table}.{column}"))
                })?;
            clocks.insert(id, index);
        }
        let mut paths = HashMap::<usize, Vec<Vec<Step>>>::new();
        for path in &spec.paths {
            if !(1..=2).contains(&path.len()) {
                return Err(EngineError::InvalidRequest(
                    "walk paths need one or two relations".into(),
                ));
            }
            let mut steps: Vec<Step> = Vec::new();
            let mut origin = 0;
            for name in path {
                let id = *engine.rel_name_to_id.get(name).ok_or_else(|| {
                    EngineError::InvalidRequest(format!("unknown walk relation {name}"))
                })?;
                let src = engine.relation_src_type[id];
                if let Some(prev) = steps.last() {
                    if prev.destination != src {
                        return Err(EngineError::InvalidRequest(format!(
                            "disconnected walk path {path:?}"
                        )));
                    }
                } else {
                    origin = src;
                }
                let destination = engine.relation_info[id];
                let table = &engine.id_to_node_type[destination];
                let time_range = spec.clocks.get(table).and_then(|clock| {
                    IndexedTimeRange::matching(
                        engine.walk_sort_keys.get(id).and_then(Option::as_ref),
                        table,
                        clock,
                    )
                });
                let local_rel = engine.local_to_merged_rel[src]
                    .iter()
                    .position(|&v| usize::from(v) == id)
                    .ok_or_else(|| {
                        EngineError::InvalidRequest(format!("no local adjacency for {name}"))
                    })? as u16;
                steps.push(Step {
                    local_rel,
                    destination,
                    time_range,
                });
            }
            paths.entry(origin).or_default().push(steps);
        }
        let mut keep = BTreeMap::new();
        for (table, &k) in &spec.keep {
            let id = type_id(table)?;
            if k == 0 || !paths.contains_key(&id) {
                return Err(EngineError::InvalidRequest(format!(
                    "invalid walk quota for {table}"
                )));
            }
            keep.insert(id, k);
        }
        Ok(Self {
            name: spec.name.clone(),
            paths,
            keep,
            clocks,
            restart: spec.restart,
            budget: spec.budget,
        })
    }
}

// The first max+1 neighbors suffice to establish that a node is below its
// push threshold. Keep that lower bound for hubs; materialize full lists only
// for nodes that can actually be pushed at this residual.
enum Degree {
    AtLeast(usize),
    Complete(Range<usize>),
}

struct PushNode {
    node: Node,
    degree: Degree,
    residual: f64,
    score: f64,
    queued: bool,
}

// Physical node IDs may be sparse. Intern only nodes reached by this root at
// its cutoff. Once decoded, adjacency and the FIFO use compact local positions.
#[derive(Default)]
struct PushWorkspace {
    positions: GxHashMap<Node, usize>,
    nodes: Vec<PushNode>,
    adjacent: Vec<usize>,
}

impl PushWorkspace {
    fn position(&mut self, node: Node) -> usize {
        *self.positions.entry(node).or_insert_with(|| {
            let i = self.nodes.len();
            self.nodes.push(PushNode {
                node,
                degree: Degree::AtLeast(0),
                residual: 0.0,
                score: 0.0,
                queued: false,
            });
            i
        })
    }

    fn eligible(
        &mut self,
        i: usize,
        budget: u32,
        read: &mut impl FnMut(Node, usize) -> Result<Vec<Node>>,
    ) -> Result<bool> {
        let state = &self.nodes[i];
        let mass = state.residual;
        let threshold = |degree: usize| degree as f64 / f64::from(budget);
        match &state.degree {
            Degree::Complete(range) => {
                return Ok(!range.is_empty() && mass >= threshold(range.len()))
            }
            Degree::AtLeast(lower) if mass < threshold((*lower).max(1)) => return Ok(false),
            _ => {}
        }
        // One extra entry resolves floating-point rounding at integral thresholds.
        let limit = (mass * f64::from(budget)).ceil() as usize + 1;
        let mut neighbors = read(state.node, limit)?;
        let count = neighbors.len();
        if count >= limit {
            self.nodes[i].degree = Degree::AtLeast(count);
            return Ok(false);
        }
        neighbors.sort_unstable();
        let start = self.adjacent.len();
        for neighbor in neighbors {
            let dest = self.position(neighbor);
            self.adjacent.push(dest);
        }
        self.nodes[i].degree = Degree::Complete(start..self.adjacent.len());
        Ok(count > 0 && mass >= threshold(count))
    }
}

fn forward_push(
    start: Node,
    restart: f64,
    budget: u32,
    mut read: impl FnMut(Node, usize) -> Result<Vec<Node>>,
) -> Result<(Vec<(Node, f64)>, u64)> {
    let mut work = PushWorkspace::default();
    let root = work.position(start);
    work.nodes[root].residual = 1.0;
    if !work.eligible(root, budget, &mut read)? {
        return Ok((Vec::new(), 0));
    }
    let mut queue = VecDeque::from([root]);
    let mut pushes = 0;
    while let Some(i) = queue.pop_front() {
        pushes += 1;
        let state = &mut work.nodes[i];
        state.queued = false;
        let mass = std::mem::take(&mut state.residual);
        state.score += restart * mass;
        let Degree::Complete(range) = &state.degree else {
            unreachable!()
        };
        let share = (1.0 - restart) * mass / range.len() as f64;
        for edge in range.clone() {
            let dest = work.adjacent[edge];
            work.nodes[dest].residual += share;
            if !work.nodes[dest].queued && work.eligible(dest, budget, &mut read)? {
                work.nodes[dest].queued = true;
                queue.push_back(dest);
            }
        }
    }
    let scores = work
        .nodes
        .into_iter()
        .filter(|n| n.score > 0.0)
        .map(|n| (n.node, n.score))
        .collect();
    Ok((scores, pushes))
}

impl DiskGraphEngine {
    fn walk_record(&self, (table, id): Node) -> Result<EdgeListView<'_>> {
        let shards = &self.readers[table];
        let shard = &shards[self.find_shard(shards, id)?];
        shard
            .reader
            .edges_for_node(id - shard.min_id)
            .map_err(|e| EngineError::DataCorruption(e.to_string()))
    }

    fn walk_visible(
        &self,
        node: Node,
        walk: &CompiledWalk,
        anchor: Option<i64>,
        scratch: &mut RowPackDecodeScratch,
    ) -> Result<bool> {
        let Some(&column) = walk.clocks.get(&node.0) else {
            return Ok(true);
        };
        let anchor = anchor.ok_or_else(|| {
            EngineError::InvalidRequest("clocked walk needs a seed timestamp".into())
        })?;
        let view = self.walk_record(node)?;
        let row = view
            .node_data()
            .map_err(|e| EngineError::DataCorruption(e.to_string()))?
            .ok_or_else(|| EngineError::DataCorruption("walk row clock is unavailable".into()))?;
        let decoder = self.node_decoders[node.0].as_ref().unwrap();
        let value = decoder
            .decode_column_value(decoder.row_or_null(row), column, scratch)
            .map_err(|e| EngineError::DataCorruption(e.to_string()))?;
        let time = match value {
            ValueRef::Timestamp(ts) => ts,
            ValueRef::Date(days) => convert_trunc(days.into(), TimeUnit::Days, self.timestamp_unit),
            ValueRef::Int64(v) => convert_trunc(v, TimeUnit::Seconds, self.timestamp_unit),
            ValueRef::Int32(v) => {
                convert_trunc(i64::from(v), TimeUnit::Seconds, self.timestamp_unit)
            }
            ValueRef::Null => return Ok(false),
            _ => {
                return Err(EngineError::InvalidRequest(
                    "walk clock is not a date, timestamp or integer".into(),
                ))
            }
        };
        Ok(time <= anchor)
    }

    // Return true to continue, false when the caller has established its degree limit.
    // Each recursive hop shares the walk settings, decode scratch and work counter.
    #[allow(clippy::too_many_arguments)]
    fn walk_path(
        &self,
        node: Node,
        path: &[Step],
        walk: &CompiledWalk,
        anchor: Option<i64>,
        cutoff: Option<&NodeCutoff<'_>>,
        scratch: &mut RowPackDecodeScratch,
        decoded: &mut u64,
        found: &mut impl FnMut(Node) -> bool,
    ) -> Result<bool> {
        let step = &path[0];
        let view = self.walk_record(node)?;
        let Some(segment) = view
            .relation_segment(step.local_rel)
            .map_err(|e| EngineError::DataCorruption(e.to_string()))?
        else {
            return Ok(true);
        };
        let segment_len = segment.count as usize;
        let keys = EdgeUnpacker::new(segment.data, &segment.layout);
        let ranges = match (&step.time_range, anchor) {
            (Some(clock), Some(at)) => {
                clock.candidate_ranges(segment_len, None, Some(at), |i| keys.key_at(i))
            }
            _ => [0..segment_len, 0..0, 0..0],
        };
        for range in ranges {
            let mut edges = EdgeUnpacker::new(segment.data, &segment.layout);
            edges.advance(range.start);
            let mut remaining = range.len();
            while remaining > 0 {
                let mut ids = [0; 32];
                let mut sort_keys = [0; 32];
                let take = remaining.min(ids.len());
                let n = edges.unpack_batch(&mut ids[..take], &mut sort_keys[..take]);
                *decoded += n as u64;
                if n != take {
                    return Err(EngineError::DataCorruption(
                        "walk adjacency length mismatch".into(),
                    ));
                }
                for (&id, &key) in ids[..n].iter().zip(&sort_keys[..n]) {
                    let dest = (step.destination, id);
                    if cutoff.is_some_and(|c| c.type_id == dest.0 && !c.visible(id)) {
                        continue;
                    }
                    // A non-sentinel upper bucket endpoint at or before the
                    // anchor proves visibility, including for a coarse index.
                    let indexed_visible = step
                        .time_range
                        .as_ref()
                        .and_then(|r| unclipped_bucket_end(r.decoder(), key))
                        .zip(anchor)
                        .is_some_and(|(end, at)| end <= at);
                    if !indexed_visible && !self.walk_visible(dest, walk, anchor, scratch)? {
                        continue;
                    }
                    let more = if path.len() == 1 {
                        found(dest)
                    } else {
                        self.walk_path(
                            dest,
                            &path[1..],
                            walk,
                            anchor,
                            cutoff,
                            scratch,
                            decoded,
                            found,
                        )?
                    };
                    if !more {
                        return Ok(false);
                    }
                }
                remaining -= n;
            }
        }
        Ok(true)
    }

    fn walk_hits(
        &self,
        start: Node,
        walk: &CompiledWalk,
        anchor: Option<i64>,
        cutoff: Option<&NodeCutoff<'_>>,
    ) -> Result<WalkResult> {
        let mut scratch = RowPackDecodeScratch::default();
        if !self.walk_visible(start, walk, anchor, &mut scratch)? {
            return Ok(WalkResult::default());
        }
        let mut decoded = 0;
        let (scores, pushes) = forward_push(start, walk.restart, walk.budget, |node, limit| {
            let mut values = Vec::new();
            for path in walk.paths.get(&node.0).into_iter().flatten() {
                if !self.walk_path(
                    node,
                    path,
                    walk,
                    anchor,
                    cutoff,
                    &mut scratch,
                    &mut decoded,
                    &mut |x| {
                        values.push(x);
                        values.len() < limit
                    },
                )? {
                    break;
                }
            }
            Ok(values)
        })?;
        let mut grouped = BTreeMap::<usize, Vec<(Node, f64)>>::new();
        for (node, score) in scores {
            if node != start && walk.keep.contains_key(&node.0) {
                grouped.entry(node.0).or_default().push((node, score));
            }
        }
        let mut hits = Vec::new();
        for (table, mut candidates) in grouped {
            let cmp =
                |a: &(Node, f64), b: &(Node, f64)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
            let k = walk.keep[&table];
            if candidates.len() > k {
                candidates.select_nth_unstable_by(k, cmp);
                candidates.truncate(k);
            }
            candidates.sort_unstable_by(cmp);
            hits.extend(
                candidates
                    .into_iter()
                    .enumerate()
                    .map(|(rank, (node, score))| (node, score as f32, rank as u32)),
            );
        }
        Ok(WalkResult {
            hits,
            pushes,
            decoded,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sample_walks(
        &self,
        layer: usize,
        walks: &[CompiledWalk],
        frontier: &[(usize, Vec<NodeRecord>)],
        anchors: &[Option<&[i64]>],
        state: &mut SamplingState,
        output: &mut impl OutputHandler,
        cutoff: Option<&NodeCutoff<'_>>,
        fused: bool,
    ) -> Result<()> {
        let mut jobs = Vec::new();
        for (table, nodes) in frontier {
            for origin in nodes {
                for walk in walks {
                    if walk.paths.contains_key(table) {
                        jobs.push((*table, origin, walk));
                    }
                }
            }
        }
        let batches = self.thread_pool.install(|| {
            jobs.par_iter()
                .map(|(table, origin, walk)| {
                    let at = anchors
                        .get(origin.seed_type_id as usize)
                        .copied()
                        .flatten()
                        .and_then(|a| a.get(origin.seed_index as usize))
                        .copied();
                    self.walk_hits((*table, origin.global_id), walk, at, cutoff)
                })
                .collect::<Result<Vec<_>>>()
        })?;
        for ((source, origin, walk), result) in jobs.into_iter().zip(batches) {
            state.stats.walk_pushes += result.pushes;
            state.stats.walk_decoded_entries += result.decoded;
            state.stats.total_edges += result.hits.len() as u64;
            for ((table, id), score, rank) in result.hits {
                let key = LookupKey {
                    seed_type_id: origin.seed_type_id,
                    seed_index: origin.seed_index,
                    global_id: id,
                };
                let ts = state.type_states.entry(table).or_default();
                let local = ts.nodes.len() as u32;
                let entry = if self.per_seed_recording(fused) {
                    let slot = (state.seed_slot_base[origin.seed_type_id as usize]
                        + origin.seed_index) as usize;
                    state.seed_maps[slot]
                        .entry(node_key(table, id))
                        .or_insert(local)
                } else {
                    ts.index_map.entry(key).or_insert(local)
                };
                let destination = *entry;
                if let Some(frontier) = &mut state.walk_frontier {
                    frontier.push((table, destination));
                }
                if destination == local {
                    let record = NodeRecord {
                        global_id: id,
                        local_id: local,
                        seed_index: origin.seed_index,
                        seed_type_id: origin.seed_type_id,
                    };
                    ts.nodes.push(record);
                    ts.next_frontier.push(record);
                    state.stats.record_node(layer + 1, table);
                }
                output.handle_walk_edge(
                    &walk.name,
                    &self.id_to_node_type[source],
                    &self.id_to_node_type[table],
                    origin.local_id,
                    destination,
                    score,
                    rank,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alternating_pair_converges_to_geometric_series() {
        let a = (0, 1);
        let b = (1, 1);
        let (scores, _) = forward_push(a, 0.25, 100_000, |n, _| {
            Ok(vec![if n == a { b } else { a }])
        })
        .unwrap();
        let scores: HashMap<_, _> = scores.into_iter().collect();
        assert!((scores[&a] - 1.0 / 1.75).abs() < 1e-5);
        assert!((scores[&b] - 0.75 / 1.75).abs() < 1e-5);
    }

    #[test]
    fn high_degree_checks_stop_and_isolates_terminate() {
        let a = (0, 0);
        assert!(forward_push(a, 0.25, 5, |_, limit| {
            assert_eq!(limit, 6);
            Ok(vec![a; limit])
        })
        .unwrap()
        .0
        .is_empty());
        assert!(forward_push(a, 0.25, 5, |_, _| Ok(Vec::new()))
            .unwrap()
            .0
            .is_empty());
    }
}
