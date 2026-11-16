// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Subgraph extraction from the graph index.
//!
//! Two-pass approach:
//! 1. Extract subgraph structure (nodes, edges, timestamps)
//! 2. Look up attributes for collected nodes

use crate::config::ExplainConfig;
use anyhow::{anyhow, Result};
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Extracted subgraph for a single entity
#[derive(Debug, Clone)]
pub struct ExtractedSubgraph {
    /// Root entity ID
    pub root_id: String,
    /// Root node internal ID
    pub root_node_id: u64,
    /// Anchor timestamp for temporal filtering
    pub anchor_time: i64,
    /// Edges grouped by relation
    pub edges_by_relation: HashMap<String, Vec<ExtractedEdge>>,
    /// Nodes grouped by (table, hop)
    pub nodes_by_table: HashMap<String, Vec<ExtractedNode>>,
    /// Same-type nodes reached at homophily hop (if any)
    pub same_type_nodes: Vec<u64>,
}

/// An extracted edge with decoded timestamp
#[derive(Debug, Clone)]
pub struct ExtractedEdge {
    pub relation_name: String,
    pub src_node_id: u64,
    pub dst_node_id: u64,
    pub dst_table: String,
    pub timestamp: Option<i64>,
    pub hop: usize,
}

/// An extracted node with its attributes
#[derive(Debug, Clone)]
pub struct ExtractedNode {
    pub table: String,
    pub node_id: u64,
    pub hop: usize,
    pub attributes: HashMap<String, AttributeValue>,
}

/// Attribute value (numeric or categorical)
#[derive(Debug, Clone)]
pub enum AttributeValue {
    Numeric(f64),
    Categorical(String),
    Boolean(bool),
    Missing,
}

/// Relation metadata for traversal
#[derive(Clone)]
struct RelationMeta {
    name: String,
    dst_table: String,
    time_bits: Option<u8>,
    time_anchor: Option<i64>,
    time_step: Option<i64>,
}

/// Subgraph extractor using diskgraph-index
pub struct SubgraphExtractor {
    reader: Arc<GraphIndexReader>,
    root_table: Arc<TableReader>,
    root_table_name: String,
    relations_by_table: HashMap<String, Vec<RelationMeta>>,
    config: ExplainConfig,
}

impl SubgraphExtractor {
    pub fn new(reader: GraphIndexReader, config: &ExplainConfig) -> Result<Self> {
        let reader = Arc::new(reader);
        let root_table = reader
            .table(&config.root_table)
            .ok_or_else(|| anyhow!("unknown root table: {}", config.root_table))?;

        let mut relations_by_table: HashMap<String, Vec<RelationMeta>> = HashMap::new();

        for table in reader.tables() {
            let table_name = table.node_type().to_string();
            let mut rels = Vec::new();

            for rel in table.relations() {
                // Check if relation should be included
                if !config.motif_config.include_relations.is_empty()
                    && !config.motif_config.include_relations.contains(&rel.name)
                {
                    continue;
                }
                if config.motif_config.exclude_relations.contains(&rel.name) {
                    continue;
                }

                // Extract timestamp metadata from dest_sort_key
                let time_meta = rel.dest_sort_key.as_ref().and_then(|dsk| {
                    dsk.timestamp
                        .as_ref()
                        .map(|ts| (dsk.width_bits, ts.encoding.anchor, ts.encoding.step))
                });

                rels.push(RelationMeta {
                    name: rel.name.clone(),
                    dst_table: rel.dst_type.clone(),
                    time_bits: time_meta.map(|(b, _, _)| b),
                    time_anchor: time_meta.map(|(_, a, _)| a),
                    time_step: time_meta.map(|(_, _, s)| s),
                });
            }

            relations_by_table.insert(table_name, rels);
        }

        Ok(Self {
            reader,
            root_table: root_table.clone(),
            root_table_name: config.root_table.clone(),
            relations_by_table,
            config: config.clone(),
        })
    }

    /// Look up internal node ID from entity ID string
    pub fn lookup_node_id(&self, entity_id: &str) -> Result<u64> {
        match self.config.id_lookup {
            crate::config::IdLookup::NodeId => entity_id
                .parse::<u64>()
                .map_err(|e| anyhow!("invalid node id: {}", e)),
            crate::config::IdLookup::PrimaryKey => {
                let key = self.root_table.parse_pk_key(entity_id)?;
                self.root_table
                    .node_id_from_pk(&key)?
                    .ok_or_else(|| anyhow!("entity not found: {}", entity_id))
            }
        }
    }

    /// Extract subgraph for an entity (Pass 1: structure + timestamps)
    pub fn extract_subgraph(&self, entity_id: &str, anchor_time: i64) -> Result<ExtractedSubgraph> {
        let root_node_id = self.lookup_node_id(entity_id)?;

        let mut edges_by_relation: HashMap<String, Vec<ExtractedEdge>> = HashMap::new();
        let mut nodes_by_table: HashMap<String, Vec<ExtractedNode>> = HashMap::new();
        let mut same_type_nodes: Vec<u64> = Vec::new();

        let mut visited: HashSet<(String, u64)> = HashSet::new();
        let mut queue: VecDeque<(String, u64, usize)> = VecDeque::new();

        // Start from root
        queue.push_back((self.root_table_name.clone(), root_node_id, 0));
        visited.insert((self.root_table_name.clone(), root_node_id));

        let mut total_nodes = 0usize;

        while let Some((table_name, node_id, hop)) = queue.pop_front() {
            if hop >= self.config.max_hops {
                continue;
            }

            let table = match self.reader.table(&table_name) {
                Some(t) => t,
                None => continue,
            };

            let view = match table.edges_for_node(node_id) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let relations = match self.relations_by_table.get(&table_name) {
                Some(r) => r,
                None => continue,
            };

            let segments = match view.relation_segments() {
                Ok(s) => s,
                Err(_) => continue,
            };

            for (rel_idx, segment) in segments.iter().enumerate() {
                let rel = match relations.get(rel_idx) {
                    Some(r) => r,
                    None => continue,
                };

                // Skip unbounded relations if configured
                if rel.time_bits.is_none() && self.config.skip_unbounded_relations {
                    continue;
                }

                let fanout = self.config.fanout_per_hop.get(hop).copied().unwrap_or(20);
                let cutoff = self.encode_timestamp(anchor_time, rel);

                let mut collected = 0usize;
                let mut buf = [0u8; 16];

                for chunk in segment.data.chunks(segment.layout.entry_bytes) {
                    if collected >= fanout || total_nodes >= self.config.max_nodes {
                        break;
                    }

                    buf[..segment.layout.entry_bytes].copy_from_slice(chunk);
                    let val = u128::from_le_bytes(buf);

                    let dst_id = (val & segment.layout.dst_mask as u128) as u64;
                    let sort_key = segment.layout.dest_sort_key_bits.map(|bits| {
                        ((val >> segment.layout.dst_bits as u128) & ((1u128 << bits) - 1)) as u64
                    });

                    // Temporal filtering
                    if let (Some(cut), Some(raw_sort)) = (cutoff, sort_key) {
                        if raw_sort > cut {
                            buf.fill(0);
                            continue;
                        }
                    }

                    // Decode timestamp
                    let timestamp = self.decode_timestamp(sort_key, rel);

                    // Record edge
                    edges_by_relation
                        .entry(rel.name.clone())
                        .or_default()
                        .push(ExtractedEdge {
                            relation_name: rel.name.clone(),
                            src_node_id: node_id,
                            dst_node_id: dst_id,
                            dst_table: rel.dst_table.clone(),
                            timestamp,
                            hop,
                        });

                    // Check for same-type nodes at homophily hop
                    if hop + 1 == self.config.motif_config.homophily_hop
                        && rel.dst_table == self.root_table_name
                    {
                        same_type_nodes.push(dst_id);
                    }

                    // Queue destination for further traversal
                    let dst_key = (rel.dst_table.clone(), dst_id);
                    if !visited.contains(&dst_key) {
                        visited.insert(dst_key);
                        queue.push_back((rel.dst_table.clone(), dst_id, hop + 1));
                        total_nodes += 1;
                    }

                    collected += 1;
                    buf.fill(0);
                }
            }
        }

        // Pass 2: Look up attributes for all collected nodes
        for (table_name, node_id) in visited.iter() {
            if let Some(table) = self.reader.table(table_name) {
                if let Ok(view) = table.edges_for_node(*node_id) {
                    let attributes = self.extract_node_attributes(&view, &table);
                    let hop = self.compute_hop(&edges_by_relation, table_name, *node_id);

                    nodes_by_table
                        .entry(table_name.clone())
                        .or_default()
                        .push(ExtractedNode {
                            table: table_name.clone(),
                            node_id: *node_id,
                            hop,
                            attributes,
                        });
                }
            }
        }

        Ok(ExtractedSubgraph {
            root_id: entity_id.to_string(),
            root_node_id,
            anchor_time,
            edges_by_relation,
            nodes_by_table,
            same_type_nodes,
        })
    }

    /// Encode timestamp for comparison with sort key
    /// Must match the encoding used in diskgraph-ingest (ordered_signed_to_u64)
    fn encode_timestamp(&self, timestamp: i64, rel: &RelationMeta) -> Option<u64> {
        match (rel.time_anchor, rel.time_step, rel.time_bits) {
            (Some(anchor), Some(step), Some(bits)) if step != 0 => {
                let delta = (timestamp - anchor) / step;
                // Match ingest's ordered_signed_to_u64: clamp to signed range, then XOR with sign bit
                let (clamped, mask): (u64, u64) = match bits {
                    8 => {
                        let v = delta.clamp(i8::MIN as i64, i8::MAX as i64) as i8;
                        ((v as u8) as u64, 1u64 << 7)
                    }
                    16 => {
                        let v = delta.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
                        ((v as u16) as u64, 1u64 << 15)
                    }
                    32 => {
                        let v = delta.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                        ((v as u32) as u64, 1u64 << 31)
                    }
                    _ => (delta.max(0) as u64, 0),
                };
                Some(clamped ^ mask)
            }
            _ => None,
        }
    }

    /// Decode sort key to actual timestamp
    fn decode_timestamp(&self, sort_key: Option<u64>, rel: &RelationMeta) -> Option<i64> {
        match (sort_key, rel.time_anchor, rel.time_step) {
            (Some(key), Some(anchor), Some(step)) => Some(anchor + (key as i64) * step),
            _ => None,
        }
    }

    /// Compute hop distance for a node
    fn compute_hop(
        &self,
        edges: &HashMap<String, Vec<ExtractedEdge>>,
        table: &str,
        node_id: u64,
    ) -> usize {
        if table == self.root_table_name && node_id == 0 {
            return 0;
        }

        for edge_list in edges.values() {
            for edge in edge_list {
                if edge.dst_table == table && edge.dst_node_id == node_id {
                    return edge.hop + 1;
                }
            }
        }

        0
    }

    /// Extract attributes from a node using JSON decoding (Pass 2)
    fn extract_node_attributes(
        &self,
        view: &diskgraph_index::layout::EdgeListView<'_>,
        table: &TableReader,
    ) -> HashMap<String, AttributeValue> {
        let mut attributes = HashMap::new();

        // Get raw node data
        let raw = match view.node_data().ok().flatten() {
            Some(r) => r,
            None => return attributes,
        };

        // Decode to JSON (simplest approach)
        let json_value = match table.decode_node_data_json(view) {
            Ok(Some(v)) => v,
            _ => return attributes,
        };

        // Convert JSON to AttributeValue
        let obj = match json_value.as_object() {
            Some(o) => o,
            None => return attributes,
        };

        let table_name = table.node_type();

        for (col_name, value) in obj {
            // Check if column should be included
            let col_path = crate::config::ColumnPath {
                table: table_name.to_string(),
                column: col_name.clone(),
            };

            if !self.config.motif_config.include_columns.is_empty()
                && !self.config.motif_config.include_columns.contains(&col_path)
            {
                continue;
            }
            if self.config.motif_config.exclude_columns.contains(&col_path) {
                continue;
            }

            let attr_value = json_to_attribute_value(value);
            attributes.insert(col_name.clone(), attr_value);
        }

        let _ = raw; // Silence unused warning
        attributes
    }

    /// Get reader for looking up predictions of same-type neighbors (homophily)
    pub fn reader(&self) -> &GraphIndexReader {
        &self.reader
    }

    /// Get root table name
    pub fn root_table_name(&self) -> &str {
        &self.root_table_name
    }
}

/// Convert a serde_json::Value to AttributeValue
fn json_to_attribute_value(value: &serde_json::Value) -> AttributeValue {
    match value {
        serde_json::Value::Null => AttributeValue::Missing,
        serde_json::Value::Bool(b) => AttributeValue::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                AttributeValue::Numeric(f)
            } else if let Some(i) = n.as_i64() {
                AttributeValue::Numeric(i as f64)
            } else if let Some(u) = n.as_u64() {
                AttributeValue::Numeric(u as f64)
            } else {
                AttributeValue::Missing
            }
        }
        serde_json::Value::String(s) => AttributeValue::Categorical(s.clone()),
        // Arrays and objects are not simple attributes
        _ => AttributeValue::Missing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_value_variants() {
        let num = AttributeValue::Numeric(42.0);
        let cat = AttributeValue::Categorical("test".to_string());
        let boolean = AttributeValue::Boolean(true);
        let missing = AttributeValue::Missing;

        assert!(matches!(num, AttributeValue::Numeric(_)));
        assert!(matches!(cat, AttributeValue::Categorical(_)));
        assert!(matches!(boolean, AttributeValue::Boolean(_)));
        assert!(matches!(missing, AttributeValue::Missing));
    }

    #[test]
    fn test_json_to_attribute_value() {
        assert!(matches!(
            json_to_attribute_value(&serde_json::Value::Null),
            AttributeValue::Missing
        ));
        assert!(matches!(
            json_to_attribute_value(&serde_json::Value::Bool(true)),
            AttributeValue::Boolean(true)
        ));
        assert!(matches!(
            json_to_attribute_value(&serde_json::json!(42.5)),
            AttributeValue::Numeric(_)
        ));
        assert!(matches!(
            json_to_attribute_value(&serde_json::json!("hello")),
            AttributeValue::Categorical(_)
        ));
    }
}
