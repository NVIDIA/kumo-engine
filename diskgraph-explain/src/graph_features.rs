use crate::config::{AttributeKind, ExplainConfig, IdLookup, TableAttributeConfig};
use anyhow::{anyhow, Result};
use diskgraph_core::metadata::{EdgeSortKeyTimestampKind, EdgeSortKeyTimestampMetadata};
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use diskgraph_rowpack::{RowPackDecodeScratch, ValueRef};
use std::collections::{HashMap, HashSet};
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FeatureKey {
    RelationCount { rel_id: u32, bucket: u16 },
    NodeAttr { table_id: u32, column_id: u32, value: ValueToken },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueToken {
    Str(u32),
    Bool(bool),
    FloatBucket(u16),
    Missing,
}

#[derive(Clone)]
pub struct SchemaIndex {
    pub strings: Arc<StringTable>,
    pub table_ids: HashMap<String, u32>,
    pub column_ids: HashMap<(String, String), u32>,
    pub relation_ids: HashMap<String, u32>,
}

impl SchemaIndex {
    pub fn new(reader: &GraphIndexReader, attrs: &[TableAttributeConfig]) -> Self {
        let strings = Arc::new(StringTable::new());
        let mut table_ids = HashMap::new();
        let mut column_ids = HashMap::new();
        let mut relation_ids = HashMap::new();

        for table in reader.tables() {
            let t_id = strings.intern(table.node_type());
            table_ids.insert(table.node_type().to_string(), t_id);
            for rel in table.relations() {
                let r_id = strings.intern(&rel.name);
                relation_ids.insert(rel.name.clone(), r_id);
            }
        }

        for table_cfg in attrs {
            let t_id = strings.intern(&table_cfg.table);
            table_ids.insert(table_cfg.table.clone(), t_id);
            for col in &table_cfg.columns {
                let c_id = strings.intern(&col.name);
                column_ids.insert((table_cfg.table.clone(), col.name.clone()), c_id);
            }
        }

        Self { strings, table_ids, column_ids, relation_ids }
    }
}

#[derive(Default)]
pub struct StringTable {
    strings: parking_lot::Mutex<Vec<String>>,
    map: parking_lot::Mutex<HashMap<String, u32>>,
}

impl StringTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&self, value: &str) -> u32 {
        let mut map = self.map.lock();
        if let Some(id) = map.get(value) {
            return *id;
        }
        let mut strings = self.strings.lock();
        let id = strings.len() as u32;
        strings.push(value.to_string());
        map.insert(value.to_string(), id);
        id
    }

    pub fn get(&self, id: u32) -> Option<String> {
        let strings = self.strings.lock();
        strings.get(id as usize).cloned()
    }
}

#[derive(Clone)]
pub struct RelationInfo {
    pub rel_name: String,
    pub rel_id: u32,
    pub dst_table: String,
    pub time_meta: Option<TimeMeta>,
}

#[derive(Clone)]
pub struct TimeMeta {
    pub bits: u8,
    pub anchor: i64,
    pub step: i64,
}

#[derive(Clone)]
pub struct TableAttributeContext {
    pub table_id: u32,
    pub columns: Vec<AttributeColumnContext>,
}

#[derive(Clone)]
pub struct AttributeColumnContext {
    pub column_id: u32,
    pub index: usize,
    pub kind: AttributeKind,
    pub buckets: Vec<f64>,
}

pub struct GraphExtractor {
    pub reader: Arc<GraphIndexReader>,
    pub root_table: Arc<TableReader>,
    pub relations_by_table: HashMap<String, Vec<RelationInfo>>,
    pub schema: SchemaIndex,
    pub attributes: HashMap<String, TableAttributeContext>,
    pub config: ExplainConfig,
}

pub struct RowFeatures {
    pub feature_set: HashSet<FeatureKey>,
    pub node_features: Vec<NodeFeature>,
}

pub struct NodeFeature {
    pub rel_name: String,
    pub dst_table: String,
    pub node_id: u64,
    pub pk_value: Option<String>,
    pub features: Vec<FeatureKey>,
}

struct NodeVisit {
    table: String,
    node_id: u64,
    hop: usize,
}

impl GraphExtractor {
    pub fn new(reader: GraphIndexReader, config: &ExplainConfig) -> Result<Self> {
        let reader = Arc::new(reader);
        let root_table = reader
            .table(&config.root_table)
            .ok_or_else(|| anyhow!("unknown root table {}", config.root_table))?;
        let schema = SchemaIndex::new(&reader, &config.attributes);
        let mut relations_by_table = HashMap::new();
        for table in reader.tables() {
            let rels = build_relation_info(&table, &schema)?;
            relations_by_table.insert(table.node_type().to_string(), rels);
        }
        let attributes = build_attribute_contexts(&reader, &schema, &config.attributes)?;
        Ok(Self {
            reader,
            root_table,
            relations_by_table,
            schema,
            attributes,
            config: config.clone(),
        })
    }

    pub fn extract_row_features(&self, id_value: &str, timestamp: i64) -> Result<RowFeatures> {
        let node_id = self.lookup_node_id(id_value)?;
        let mut feature_set = HashSet::new();
        let mut node_features = Vec::new();
        let mut total_nodes = 0usize;
        let mut visited: HashSet<(String, u64)> = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(NodeVisit {
            table: self.root_table.node_type().to_string(),
            node_id,
            hop: 0,
        });
        visited.insert((self.root_table.node_type().to_string(), node_id));

        while let Some(node) = queue.pop_front() {
            let table = match self.reader.table(&node.table) {
                Some(table) => table,
                None => continue,
            };
            let view = match table.edges_for_node(node.node_id) {
                Ok(view) => view,
                Err(_) => continue,
            };

            if let Some(ctx) = self.attributes.get(&node.table) {
                if let Some(features) = extract_node_attributes(&view, &table, ctx, &self.schema) {
                    for feature in &features {
                        feature_set.insert(feature.clone());
                    }
                }
            }

            if node.hop >= self.config.max_hops {
                continue;
            }

            let rels = match self.relations_by_table.get(&node.table) {
                Some(rels) => rels,
                None => continue,
            };
            let segments = match view.relation_segments() {
                Ok(s) => s,
                Err(_) => continue,
            };

            for (idx, seg) in segments.iter().enumerate() {
                let rel = match rels.get(idx) {
                    Some(rel) => rel,
                    None => continue,
                };
                if rel.time_meta.is_none() && self.config.skip_unbounded_relations {
                    continue;
                }
                let cutoff = rel.time_meta.as_ref().map(|meta| encode_timestamp_value(timestamp, meta));
                let mut count = 0u32;
                let mut collected = 0usize;
                let mut buf = [0u8; 16];
                for chunk in seg.data.chunks(seg.layout.entry_bytes) {
                    buf[..seg.layout.entry_bytes].copy_from_slice(chunk);
                    let val = u128::from_le_bytes(buf);
                    let dst_id = (val & seg.layout.dst_mask as u128) as u64;
                    let sort_key = seg
                        .layout
                        .dest_sort_key_bits
                        .map(|bits| ((val >> seg.layout.dst_bits as u128) & ((1u128 << bits) - 1)) as u64);
                    if let (Some(cut), Some(raw_sort)) = (cutoff, sort_key) {
                        if raw_sort > cut {
                            buf.fill(0);
                            continue;
                        }
                    }
                    count = count.saturating_add(1);
                    if collected < self.config.relation_fanout && total_nodes < self.config.max_nodes {
                        if !visited.contains(&(rel.dst_table.clone(), dst_id)) {
                            visited.insert((rel.dst_table.clone(), dst_id));
                            queue.push_back(NodeVisit {
                                table: rel.dst_table.clone(),
                                node_id: dst_id,
                                hop: node.hop + 1,
                            });
                        }
                        if node.hop == 0 {
                            if let Some(dst_table) = self.reader.table(&rel.dst_table) {
                                if let Ok(dst_view) = dst_table.edges_for_node(dst_id) {
                                    let mut node_feats = Vec::new();
                                    if let Some(ctx) = self.attributes.get(&rel.dst_table) {
                                        if let Some(features) = extract_node_attributes(&dst_view, &dst_table, ctx, &self.schema) {
                                            for feature in &features {
                                                feature_set.insert(feature.clone());
                                            }
                                            node_feats = features;
                                        }
                                    }
                                    let pk_value = extract_pk_value(&dst_view, &dst_table).ok().flatten();
                                    node_features.push(NodeFeature {
                                        rel_name: rel.rel_name.clone(),
                                        dst_table: rel.dst_table.clone(),
                                        node_id: dst_id,
                                        pk_value,
                                        features: node_feats,
                                    });
                                }
                            }
                        }
                        collected += 1;
                        total_nodes += 1;
                    }
                    buf.fill(0);
                    if total_nodes >= self.config.max_nodes {
                        break;
                    }
                }

                if node.hop == 0 {
                    let bucket = bucket_u32(count, &self.config.degree_buckets);
                    let feat = FeatureKey::RelationCount { rel_id: rel.rel_id, bucket };
                    feature_set.insert(feat);
                }
            }
        }

        Ok(RowFeatures { feature_set, node_features })
    }

    fn lookup_node_id(&self, id_value: &str) -> Result<u64> {
        match self.config.id_lookup {
            IdLookup::NodeId => Ok(id_value.parse::<u64>()?),
            IdLookup::PrimaryKey => {
                let key = self.root_table.parse_pk_key(id_value)?;
                let node_id = self
                    .root_table
                    .node_id_from_pk(&key)?
                    .ok_or_else(|| anyhow!("pk not found"))?;
                Ok(node_id)
            }
        }
    }
}

fn extract_pk_value(
    view: &diskgraph_index::layout::EdgeListView<'_>,
    table: &TableReader,
) -> Result<Option<String>> {
    let pk_index = match pk_column_index(table) {
        Some(idx) => idx,
        None => return Ok(None),
    };
    let raw = match view.node_data()? {
        Some(raw) => raw,
        None => return Ok(None),
    };
    let decoder = match table.node_data_decoder() {
        Some(decoder) => decoder,
        None => return Ok(None),
    };
    let mut scratch = RowPackDecodeScratch::default();
    let value = decoder.decode_column_value(raw, pk_index, &mut scratch)?;
    Ok(Some(value_to_string(&value)))
}

fn extract_node_attributes(
    view: &diskgraph_index::layout::EdgeListView<'_>,
    table: &TableReader,
    ctx: &TableAttributeContext,
    schema: &SchemaIndex,
) -> Option<Vec<FeatureKey>> {
    let raw = view.node_data().ok().flatten()?;
    let decoder = table.node_data_decoder()?;
    let mut scratch = RowPackDecodeScratch::default();
    let mut out = Vec::new();
    for col in &ctx.columns {
        let value = match decoder.decode_column_value(raw, col.index, &mut scratch) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let token = value_to_token(&value, col, schema);
        out.push(FeatureKey::NodeAttr {
            table_id: ctx.table_id,
            column_id: col.column_id,
            value: token,
        });
    }
    Some(out)
}

fn value_to_token(value: &ValueRef<'_>, col: &AttributeColumnContext, schema: &SchemaIndex) -> ValueToken {
    match col.kind {
        AttributeKind::Boolean => match value {
            ValueRef::Bool(v) => ValueToken::Bool(*v),
            ValueRef::Null => ValueToken::Missing,
            _ => ValueToken::Missing,
        },
        AttributeKind::Numeric => {
            let num = match value {
                ValueRef::Int64(v) => Some(*v as f64),
                ValueRef::Int32(v) => Some(*v as f64),
                ValueRef::UInt64(v) => Some(*v as f64),
                ValueRef::UInt32(v) => Some(*v as f64),
                ValueRef::Float32(v) => Some(*v as f64),
                ValueRef::Float64(v) => Some(*v),
                _ => None,
            };
            if let Some(v) = num {
                let bucket = bucket_f64(v, &col.buckets);
                ValueToken::FloatBucket(bucket)
            } else {
                ValueToken::Missing
            }
        }
        AttributeKind::Categorical => match value {
            ValueRef::String(v) => {
                let id = schema.strings.intern(v);
                ValueToken::Str(id)
            }
            ValueRef::Bytes(v) => {
                let s = hex_encode(v);
                let id = schema.strings.intern(&s);
                ValueToken::Str(id)
            }
            ValueRef::Null => ValueToken::Missing,
            _ => ValueToken::Missing,
        },
    }
}

fn value_to_string(value: &ValueRef<'_>) -> String {
    match value {
        ValueRef::String(v) => v.to_string(),
        ValueRef::Bytes(v) => hex_encode(v),
        ValueRef::Int64(v) => v.to_string(),
        ValueRef::Int32(v) => v.to_string(),
        ValueRef::UInt64(v) => v.to_string(),
        ValueRef::UInt32(v) => v.to_string(),
        ValueRef::Float32(v) => v.to_string(),
        ValueRef::Float64(v) => v.to_string(),
        ValueRef::Bool(v) => v.to_string(),
        _ => "".to_string(),
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(LUT[(b >> 4) as usize] as char);
        out.push(LUT[(b & 0x0f) as usize] as char);
    }
    out
}

fn build_relation_info(table: &TableReader, schema: &SchemaIndex) -> Result<Vec<RelationInfo>> {
    let mut out = Vec::new();
    for rel in table.relations() {
        let rel_id = *schema
            .relation_ids
            .get(&rel.name)
            .ok_or_else(|| anyhow!("missing relation id {}", rel.name))?;
        let time_meta = rel
            .dest_sort_key
            .as_ref()
            .and_then(|meta| meta.timestamp.as_ref())
            .map(|ts| TimeMeta {
                bits: timestamp_bits_for_meta(ts),
                anchor: ts.encoding.anchor,
                step: ts.encoding.step.max(1),
            });
        out.push(RelationInfo {
            rel_name: rel.name.clone(),
            rel_id,
            dst_table: rel.dst_type.clone(),
            time_meta,
        });
    }
    Ok(out)
}

fn build_attribute_contexts(
    reader: &GraphIndexReader,
    schema: &SchemaIndex,
    attrs: &[TableAttributeConfig],
) -> Result<HashMap<String, TableAttributeContext>> {
    let mut out = HashMap::new();
    for table_cfg in attrs {
        let table = reader
            .table(&table_cfg.table)
            .ok_or_else(|| anyhow!("unknown table {}", table_cfg.table))?;
        let decoder = table
            .node_data_decoder()
            .ok_or_else(|| anyhow!("table {} has no node data", table_cfg.table))?;
        let ctx = decoder.context();
        let table_id = *schema
            .table_ids
            .get(&table_cfg.table)
            .ok_or_else(|| anyhow!("missing table id {}", table_cfg.table))?;
        let mut columns = Vec::new();
        for col in &table_cfg.columns {
            let idx = ctx
                .columns
                .iter()
                .position(|c| c.name == col.name)
                .ok_or_else(|| anyhow!("column {} not found in {}", col.name, table_cfg.table))?;
            let column_id = *schema
                .column_ids
                .get(&(table_cfg.table.clone(), col.name.clone()))
                .ok_or_else(|| anyhow!("missing column id {}.{}", table_cfg.table, col.name))?;
            columns.push(AttributeColumnContext {
                column_id,
                index: idx,
                kind: col.kind.clone(),
                buckets: col.buckets.clone(),
            });
        }
        out.insert(
            table_cfg.table.clone(),
            TableAttributeContext { table_id, columns },
        );
    }
    Ok(out)
}

fn pk_column_index(table: &TableReader) -> Option<usize> {
    let decoder = table.node_data_decoder()?;
    let ctx = decoder.context();
    let pk = table.primary_key_map()?;
    if let Some(idx) = pk.pk_column_index {
        return Some(idx);
    }
    ctx.columns.iter().position(|c| c.name == pk.column)
}

fn timestamp_bits_for_meta(meta: &EdgeSortKeyTimestampMetadata) -> u8 {
    match meta.encoding.kind {
        EdgeSortKeyTimestampKind::Int8 => 8,
        EdgeSortKeyTimestampKind::Int16 => 16,
        EdgeSortKeyTimestampKind::Int32 => 32,
    }
}

fn encode_timestamp_value(value: i64, meta: &TimeMeta) -> u64 {
    let step = meta.step.max(1);
    let delta = value.saturating_sub(meta.anchor).div_euclid(step);
    match meta.bits {
        8 => ordered_signed_to_u64(delta.clamp(i8::MIN as i64, i8::MAX as i64), 8),
        16 => ordered_signed_to_u64(delta.clamp(i16::MIN as i64, i16::MAX as i64), 16),
        32 => ordered_signed_to_u64(delta.clamp(i32::MIN as i64, i32::MAX as i64), 32),
        _ => ordered_signed_to_u64(delta, 64),
    }
}

fn ordered_signed_to_u64(value: i64, bits: u8) -> u64 {
    let mask = 1u64 << (bits - 1);
    let raw = match bits {
        8 => (value as i8 as u8) as u64,
        16 => (value as i16 as u16) as u64,
        32 => (value as i32 as u32) as u64,
        _ => value as u64,
    };
    raw ^ mask
}

fn bucket_u32(value: u32, buckets: &[u32]) -> u16 {
    for (idx, bound) in buckets.iter().enumerate() {
        if value <= *bound {
            return idx as u16;
        }
    }
    buckets.len() as u16
}

fn bucket_f64(value: f64, buckets: &[f64]) -> u16 {
    for (idx, bound) in buckets.iter().enumerate() {
        if value <= *bound {
            return idx as u16;
        }
    }
    buckets.len() as u16
}
