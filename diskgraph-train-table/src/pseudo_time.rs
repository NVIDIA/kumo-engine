//! Static LP fact-row pseudo times (Kumo 2's `PSEUDO_TIME` on the target table).
//!
//! One byte per fact node id: `0` not a label, else the link's stage + 1
//! (train 1, val 2, test 3). A seed at pseudo time `t` sees fact rows `<= t`.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use arrow_array::{Array, ListArray, RecordBatch};
use diskgraph_core::tt_builder::{
    EdgeRef, FactPseudoTimeSpec, Value, PRIMARY_ENTITY_COL, PRIMARY_TARGET_COL,
};
use diskgraph_index::graph::{GraphIndexReader, TableReader};

use crate::custom_tt_chunk_processor::{cell_to_value, value_to_pk_key};
use crate::errors::{TTGenError, TTGenResult};
use crate::ttgen_resolve::{resolve_path, ResolvedEdge};
use crate::ttgen_traversal::{iter_edge, read_column_value};

pub const FACT_PSEUDO_TIME_FILE: &str = "fact_pseudo_time.bin";
pub const FACT_PSEUDO_TIME_MAGIC: &[u8; 8] = b"DGPTIME\0";
pub const FACT_PSEUDO_TIME_VERSION: u32 = 1;
const DTYPE_U8: u32 = 1;
pub const FACT_PSEUDO_TIME_HEADER_LEN: usize = 24;

/// Payload of a `fact_pseudo_time.bin` image; rejects unknown versions.
pub fn fact_pseudo_time_payload(bytes: &[u8]) -> Result<&[u8], String> {
    if bytes.len() < FACT_PSEUDO_TIME_HEADER_LEN || &bytes[..8] != FACT_PSEUDO_TIME_MAGIC {
        return Err("not a fact_pseudo_time sidecar".to_string());
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let (version, dtype) = (word(8), word(12));
    if version != FACT_PSEUDO_TIME_VERSION || dtype != DTYPE_U8 {
        return Err(format!(
            "unsupported fact_pseudo_time version={version} dtype={dtype}"
        ));
    }
    let rows = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let payload = &bytes[FACT_PSEUDO_TIME_HEADER_LEN..];
    if payload.len() as u64 != rows {
        return Err(format!(
            "fact_pseudo_time row_count={rows} but payload has {} bytes",
            payload.len()
        ));
    }
    Ok(payload)
}

#[derive(Hash, PartialEq, Eq)]
enum LinkKey {
    Int(i64),
    Float(u64),
    Str(String),
}

fn link_key(value: &Value) -> Option<LinkKey> {
    match value {
        Value::Int(i) => Some(LinkKey::Int(*i)),
        Value::Bool(b) => Some(LinkKey::Int(*b as i64)),
        Value::Float(f) if f.is_nan() => None,
        Value::Float(f) if f.fract() == 0.0 && f.abs() < 9.0e18 => Some(LinkKey::Int(*f as i64)),
        Value::Float(f) => Some(LinkKey::Float(f.to_bits())),
        Value::Str(s) => Some(LinkKey::Str(s.clone())),
        Value::List(_) => None,
    }
}

fn target_keys(column: &dyn Array, row: usize) -> TTGenResult<HashSet<LinkKey>> {
    if column.is_null(row) {
        return Ok(HashSet::new());
    }
    let mut keys = HashSet::new();
    let mut push = |values: &dyn Array, idx: usize| -> TTGenResult<()> {
        if values.is_null(idx) {
            return Ok(());
        }
        let value = cell_to_value(values, idx).ok_or_else(|| {
            TTGenError::SchemaError(format!(
                "fact_pseudo_time: unsupported {PRIMARY_TARGET_COL} type {:?}",
                values.data_type()
            ))
        })?;
        keys.extend(link_key(&value));
        Ok(())
    };
    match column.as_any().downcast_ref::<ListArray>() {
        Some(list) => {
            let values = list.value(row);
            for idx in 0..values.len() {
                push(values.as_ref(), idx)?;
            }
        }
        None => push(column, row)?,
    }
    Ok(keys)
}

pub struct FactPseudoTime {
    entities: Arc<TableReader>,
    fact: Arc<TableReader>,
    edge: ResolvedEdge,
    target_idx: usize,
    times: Option<Box<[AtomicU8]>>,
}

impl FactPseudoTime {
    pub fn new(
        reader: &GraphIndexReader,
        entity_table: &str,
        entity_column: &str,
        spec: &FactPseudoTimeSpec,
    ) -> TTGenResult<Self> {
        let missing = |name: &str| TTGenError::SchemaError(format!("table '{name}' not found"));
        let entities = reader
            .table(entity_table)
            .ok_or_else(|| missing(entity_table))?;
        let fact = reader
            .table(&spec.table)
            .ok_or_else(|| missing(&spec.table))?;
        entities.pk_map().ok_or_else(|| {
            TTGenError::SchemaError(format!("table '{entity_table}' has no primary key map"))
        })?;
        let path = resolve_path(
            &[EdgeRef {
                src_table: entity_table.to_string(),
                src_col: entity_column.to_string(),
                dst_table: spec.table.clone(),
                dst_col: spec.entity_col.clone(),
            }],
            reader,
        )?;
        let edge = path.into_iter().next().expect("one fact edge");
        let target_idx = fact
            .node_data_decoder()
            .and_then(|d| {
                d.context()
                    .columns
                    .iter()
                    .position(|c| c.name == spec.target_col)
            })
            .ok_or_else(|| {
                TTGenError::SchemaError(format!(
                    "column '{}' not found in table '{}'",
                    spec.target_col, spec.table
                ))
            })?;
        Ok(Self {
            entities,
            fact,
            edge,
            target_idx,
            times: None,
        })
    }

    pub fn initialize(&mut self) {
        let num_rows = (0..self.fact.shard_count())
            .filter_map(|s| self.fact.shard_id_range(s))
            .filter(|(lo, hi)| lo <= hi)
            .map(|(_, hi)| hi + 1)
            .max()
            .unwrap_or(0);
        // Zero is a valid AtomicU8 representation.
        let times = unsafe { Box::<[AtomicU8]>::new_zeroed_slice(num_rows as usize).assume_init() };
        self.times = Some(times);
    }

    pub fn observe(&self, batches: &[RecordBatch; 3]) -> TTGenResult<()> {
        let times = self.times.as_ref().expect("initialized before pass B");
        let pk_map = self.entities.pk_map().expect("validated primary key map");
        let mut links: HashMap<u64, HashMap<LinkKey, u8>> = HashMap::new();
        for (stage, batch) in batches.iter().enumerate() {
            if batch.num_rows() == 0 {
                continue;
            }
            let column = |name| {
                batch.column_by_name(name).ok_or_else(|| {
                    TTGenError::SchemaError(format!("fact_pseudo_time: missing {name}"))
                })
            };
            let entity = column(PRIMARY_ENTITY_COL)?;
            let target = column(PRIMARY_TARGET_COL)?;
            for row in 0..batch.num_rows() {
                let node = cell_to_value(entity.as_ref(), row)
                    .and_then(|v| value_to_pk_key(&v, pk_map.key_type()))
                    .and_then(|key| self.entities.node_id_from_pk(&key).ok().flatten());
                let Some(node) = node else { continue };
                let keys = target_keys(target.as_ref(), row)?;
                let targets = links.entry(node).or_default();
                for key in keys {
                    targets.insert(key, stage as u8 + 1);
                }
            }
        }
        for (node, targets) in links {
            if targets.is_empty() {
                continue;
            }
            for entry in iter_edge(
                &self.edge.src_table,
                self.edge.relation_idx,
                &self.edge.layout,
                node,
                None,
            ) {
                let time = read_column_value(&self.fact, entry.dst_node, self.target_idx)
                    .and_then(|v| link_key(&v))
                    .and_then(|key| targets.get(&key).copied());
                if let (Some(time), Some(slot)) = (time, times.get(entry.dst_node as usize)) {
                    if slot.load(Ordering::Relaxed) < time {
                        slot.fetch_max(time, Ordering::Relaxed);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn finish(
        self,
        output_dir: &Path,
        nonempty: bool,
        generated: bool,
    ) -> TTGenResult<PathBuf> {
        let times = self.times.expect("initialized before pass B");
        let path = output_dir.join(FACT_PSEUDO_TIME_FILE);
        let mut out = std::io::BufWriter::new(std::fs::File::create(&path)?);
        out.write_all(FACT_PSEUDO_TIME_MAGIC)?;
        out.write_all(&FACT_PSEUDO_TIME_VERSION.to_le_bytes())?;
        out.write_all(&DTYPE_U8.to_le_bytes())?;
        out.write_all(&(times.len() as u64).to_le_bytes())?;
        let mut matched = false;
        for chunk in times.chunks(1 << 20) {
            let bytes: Vec<u8> = chunk.iter().map(|t| t.load(Ordering::Relaxed)).collect();
            matched |= bytes.iter().any(|&time| time != 0);
            out.write_all(&bytes)?;
        }
        out.flush()?;
        if nonempty && !matched {
            if generated {
                drop(out);
                std::fs::remove_file(&path)?;
                return Err(TTGenError::SchemaError(
                    "fact_pseudo_time: no generated label link matched a fact row".to_string(),
                ));
            }
            tracing::warn!("fact_pseudo_time: no custom label link matched a fact row; check entity and target types and values");
        }
        Ok(path)
    }
}
