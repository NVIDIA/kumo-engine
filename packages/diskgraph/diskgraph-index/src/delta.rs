// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Immutable, sorted edge additions. Old snapshots never open a newer log.

use crate::layout::{EdgeListView, RelationLayout};
use anyhow::{ensure, Context, Result};
use memmap2::Mmap;
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"DGDELTA1";
const WIDTH: usize = 26;

pub const MANIFEST: &str = "_incremental/deltas.json";

/// Keep the existing bincode payload readable, with a sorted offset sidecar
/// for disk-backed lookups. Older logs without the sidecar remain supported.
pub fn write_primary_keys(path: &Path, mut records: Vec<(Vec<u8>, u64)>) -> Result<()> {
    records.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    ensure!(
        records.windows(2).all(|pair| pair[0].0 != pair[1].0),
        "duplicate delta primary key"
    );
    let mut file = BufWriter::new(File::options().write(true).create_new(true).open(path)?);
    let mut index = BufWriter::new(
        File::options()
            .write(true)
            .create_new(true)
            .open(path.with_extension("idx"))?,
    );
    file.write_all(&(records.len() as u64).to_le_bytes())?;
    index.write_all(b"DGKEYI01")?;
    index.write_all(&(records.len() as u64).to_le_bytes())?;
    let mut offset = 8u64;
    for (key, id) in records {
        index.write_all(&offset.to_le_bytes())?;
        file.write_all(&key)?;
        file.write_all(&id.to_le_bytes())?;
        offset += key.len() as u64 + 8;
    }
    index.write_all(&offset.to_le_bytes())?;
    file.flush()?;
    index.flush()?;
    file.get_ref().sync_all()?;
    index.get_ref().sync_all()?;
    Ok(())
}

pub fn visit_primary_keys(
    path: &Path,
    mut visit: impl FnMut(crate::pkmap::Key, u64) -> Result<()>,
) -> Result<()> {
    let mut stream = std::io::BufReader::new(File::open(path)?);
    let count: u64 = bincode::deserialize_from(&mut stream)?;
    for _ in 0..count {
        let (key, id) = bincode::deserialize_from(&mut stream)?;
        visit(key, id)?;
    }
    Ok(())
}

pub struct PrimaryKeyLog {
    data: Mmap,
    offsets: Option<Mmap>,
    legacy: std::collections::HashMap<crate::pkmap::Key, u64>,
}

impl PrimaryKeyLog {
    pub fn open(path: &Path, valid_id: impl Fn(u64) -> bool) -> Result<Self> {
        let data = unsafe { Mmap::map(&File::open(path)?)? };
        let index_path = path.with_extension("idx");
        let offsets = if index_path.exists() {
            Some(unsafe { Mmap::map(&File::open(index_path)?)? })
        } else {
            None
        };
        let mut log = Self {
            data,
            offsets,
            legacy: std::collections::HashMap::new(),
        };
        if let Some(index) = &log.offsets {
            ensure!(
                index.len() >= 24 && &index[..8] == b"DGKEYI01" && log.data.len() >= 8,
                "invalid primary key index"
            );
            let count = u64::from_le_bytes(index[8..16].try_into().unwrap());
            ensure!(
                count == u64::from_le_bytes(log.data[..8].try_into().unwrap())
                    && count == ((index.len() - 24) / 8) as u64
                    && (index.len() - 24) % 8 == 0,
                "invalid primary key count"
            );
            ensure!(
                u64::from_le_bytes(index[16..24].try_into().unwrap()) == 8
                    && u64::from_le_bytes(index[index.len() - 8..].try_into().unwrap())
                        == log.data.len() as u64,
                "invalid primary key index bounds"
            );
            let mut previous = None;
            for i in 0..count as usize {
                let (key, id) = log.entry(i)?;
                ensure!(valid_id(id), "delta primary key out of range");
                ensure!(
                    previous.map_or(true, |prior: &[u8]| prior < key),
                    "unsorted or duplicate delta primary key"
                );
                previous = Some(key);
            }
        } else {
            visit_primary_keys(path, |key, id| {
                ensure!(valid_id(id), "delta primary key out of range");
                ensure!(
                    log.legacy.insert(key, id).is_none(),
                    "duplicate delta primary key"
                );
                Ok(())
            })?;
        }
        Ok(log)
    }

    fn entry(&self, i: usize) -> Result<(&[u8], u64)> {
        let index = self.offsets.as_ref().unwrap();
        let offset = 16 + i * 8;
        let start = usize::try_from(u64::from_le_bytes(
            index[offset..offset + 8].try_into().unwrap(),
        ))?;
        let end = usize::try_from(u64::from_le_bytes(
            index[offset + 8..offset + 16].try_into().unwrap(),
        ))?;
        ensure!(
            start >= 8 && end >= start && end - start >= 8 && end <= self.data.len(),
            "invalid primary key offsets"
        );
        Ok((
            &self.data[start..end - 8],
            u64::from_le_bytes(self.data[end - 8..end].try_into().unwrap()),
        ))
    }

    pub fn get(&self, key: &crate::pkmap::Key) -> Result<Option<u64>> {
        let Some(index) = &self.offsets else {
            return Ok(self.legacy.get(key).copied());
        };
        let key = bincode::serialize(key)?;
        let mut lo = 0;
        let mut hi = (index.len() - 24) / 8;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (candidate, id) = self.entry(mid)?;
            match candidate.cmp(&key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Ok(Some(id)),
            }
        }
        Ok(None)
    }
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct DeltaManifest {
    pub version: u32,
    pub tables: std::collections::BTreeMap<String, DeltaTable>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct DeltaTable {
    pub base_shards: usize,
    pub shard_layouts: Vec<Vec<RelationLayout>>,
    pub edge_logs: Vec<String>,
    pub primary_key_logs: Vec<String>,
    #[serde(default)]
    pub feature_file: Option<String>,
    #[serde(default)]
    pub edge_counts: std::collections::BTreeMap<String, u64>,
}

pub type FeatureReplacements = std::collections::BTreeMap<u64, Vec<u8>>;

pub fn load_features(root: &Path, table: &DeltaTable) -> Result<FeatureReplacements> {
    table.feature_file.as_ref().map(|path| {
        Ok(bincode::deserialize(&std::fs::read(root.join(path))?)?)
    }).unwrap_or_else(|| Ok(FeatureReplacements::new()))
}

impl DeltaManifest {
    pub fn load(root: &Path) -> Result<Option<Self>> {
        let path = root.join(MANIFEST);
        if !path.exists() {
            return Ok(None);
        }
        let manifest: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(matches!(manifest.version, 1..=3), "unsupported delta manifest version");
        Ok(Some(manifest))
    }
}

/// New node features use an ordinary shard, with adjacency supplied by logs.
/// Keeping a new shard avoids moving any existing node ID or PK hash slot.
pub struct NodeAppender {
    edges: BufWriter<File>,
    offsets: Vec<u64>,
    header: Vec<u8>,
    edge_count: u64,
}

impl NodeAppender {
    pub fn new(directory: &Path, relation_count: usize) -> Result<Self> {
        std::fs::create_dir_all(directory)?;
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(directory.join(crate::layout::EDGES_FILENAME))?;
        Ok(Self {
            edges: BufWriter::new(file),
            offsets: vec![0],
            header: vec![0; relation_count],
            edge_count: 0,
        })
    }

    pub fn push(&mut self, features: &[u8]) -> Result<()> {
        self.edges.write_all(&self.header)?;
        self.edges
            .write_all(&u32::try_from(features.len())?.to_le_bytes())?;
        self.edges.write_all(features)?;
        let bytes = u32::try_from(self.header.len() + 4 + features.len())?;
        self.offsets.push(
            self.offsets
                .last()
                .unwrap()
                .checked_add(bytes as u64)
                .ok_or_else(|| anyhow::anyhow!("node shard size overflow"))?,
        );
        Ok(())
    }

    pub fn finish(mut self, directory: &Path) -> Result<u64> {
        if self.offsets.len() == 1 {
            self.edges.write_all(&[0])?;
        }
        self.edges.flush()?;
        self.edges.get_ref().sync_all()?;
        let count = self.offsets.len() as u64 - 1;
        let mut header = crate::layout::OffsetFileHeader::new(count, count + 1);
        header.delta_bytes = 4;
        header.node_data_len_bytes = 4;
        header.edge_count = self.edge_count;
        let mut encoded = bincode::serialize(&header)?;
        crate::builder::delta_encode_offsets_into(&self.offsets, &mut encoded, 4)?;
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(directory.join(crate::layout::OFFSETS_FILENAME))?;
        file.write_all(&encoded)?;
        file.sync_all()?;
        Ok(count)
    }

    pub fn push_view(&mut self, view: &EdgeListView<'_>) -> Result<()> {
        ensure!(
            view.node_data_len_bytes == 4,
            "compaction requires a repacked node"
        );
        self.edges.write_all(&view.raw)?;
        self.offsets.push(
            self.offsets
                .last()
                .unwrap()
                .checked_add(view.raw.len() as u64)
                .ok_or_else(|| anyhow::anyhow!("node shard size overflow"))?,
        );
        self.edge_count += view
            .relation_segments()?
            .iter()
            .map(|s| u64::from(s.count))
            .sum::<u64>();
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeDelta {
    pub source: u64,
    pub relation: u16,
    pub destination: u64,
    pub sort_key: u64,
}

pub struct DeltaLog {
    data: Mmap,
}

impl DeltaLog {
    pub fn write(path: &Path, entries: &mut [EdgeDelta]) -> Result<()> {
        entries.sort_unstable_by_key(|e| {
            (
                e.source,
                e.relation,
                std::cmp::Reverse(e.sort_key),
                e.destination,
            )
        });
        let mut out = BufWriter::new(File::options().write(true).create_new(true).open(path)?);
        out.write_all(MAGIC)?;
        for entry in entries {
            out.write_all(&entry.source.to_le_bytes())?;
            out.write_all(&entry.relation.to_le_bytes())?;
            out.write_all(&entry.destination.to_le_bytes())?;
            out.write_all(&entry.sort_key.to_le_bytes())?;
        }
        out.flush()?;
        out.get_ref().sync_all()?;
        Ok(())
    }

    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let data = unsafe { Mmap::map(&file)? };
        ensure!(
            data.len() >= 8 && &data[..8] == MAGIC && (data.len() - 8) % WIDTH == 0,
            "invalid edge delta log"
        );
        let log = Self { data };
        for i in 1..log.len() {
            let a = log.entry(i - 1);
            let b = log.entry(i);
            ensure!(
                (a.source, a.relation) <= (b.source, b.relation),
                "unsorted edge delta log"
            );
        }
        Ok(log)
    }

    fn len(&self) -> usize {
        (self.data.len() - 8) / WIDTH
    }

    fn entry(&self, index: usize) -> EdgeDelta {
        let bytes = &self.data[8 + index * WIDTH..8 + (index + 1) * WIDTH];
        EdgeDelta {
            source: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            relation: u16::from_le_bytes(bytes[8..10].try_into().unwrap()),
            destination: u64::from_le_bytes(bytes[10..18].try_into().unwrap()),
            sort_key: u64::from_le_bytes(bytes[18..26].try_into().unwrap()),
        }
    }

    pub fn for_node(&self, source: u64) -> impl Iterator<Item = EdgeDelta> + '_ {
        let mut lo = 0;
        let mut hi = self.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.entry(mid).source < source {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo..self.len())
            .map(|i| self.entry(i))
            .take_while(move |e| e.source == source)
    }
}

/// Repack only an affected adjacency list. Features retain their exact bytes,
/// and sort keys retain the parent's encoding and newest-first ordering.
pub fn merge<'a>(
    base: EdgeListView<'a>,
    additions: &[EdgeDelta],
    layouts: Arc<Vec<RelationLayout>>,
) -> Result<EdgeListView<'a>> {
    if !base.raw.is_empty()
        && additions.is_empty()
        && base.node_data_len_bytes == 4
        && base.layouts == layouts
    {
        return Ok(base);
    }
    let (mut segments, features) = base.parse()?;
    ensure!(
        segments.len() <= layouts.len(),
        "delta removed existing relations"
    );
    // New relations follow the original relation IDs; old packed records
    // have no count or payload for these appended segments.
    for layout in &layouts[segments.len()..] {
        segments.push(crate::layout::RelationSegment {
            layout: layout.clone(), count: 0, data: &[],
        });
    }
    let mut buckets = vec![Vec::new(); layouts.len()];
    for entry in additions {
        let relation = entry.relation as usize;
        ensure!(
            relation < buckets.len(),
            "delta references unknown relation"
        );
        buckets[relation].push((entry.destination, entry.sort_key));
    }
    let mut raw = Vec::new();
    for (segment, bucket) in segments.iter().zip(&mut buckets) {
        bucket.sort_unstable_by_key(|&(dst, key)| (std::cmp::Reverse(key), dst));
        ensure!(
            segment.layout.entry_bytes > 0 && segment.layout.entry_bytes <= 16,
            "invalid edge width"
        );
        let count = u32::try_from(bucket.len() + segment.data.len() / segment.layout.entry_bytes)?;
        ensure!(count <= 0x3fff_ffff, "too many edges for one node");
        let width = if count <= 63 {
            1
        } else if count <= 0x3fff {
            2
        } else if count <= 0x3f_ffff {
            3
        } else {
            4
        };
        let bytes = count.to_be_bytes();
        let start = raw.len();
        raw.extend_from_slice(&bytes[4 - width..]);
        raw[start] |= ((width - 1) as u8) << 6;
    }
    if let Some(features) = features {
        raw.extend_from_slice(&u32::try_from(features.len())?.to_le_bytes());
    }
    for ((layout, segment), bucket) in layouts.iter().zip(&segments).zip(buckets) {
        let mut original = segment
            .data
            .chunks_exact(segment.layout.entry_bytes)
            .map(|bytes| {
                let mut word = [0u8; 16];
                word[..bytes.len()].copy_from_slice(bytes);
                let word = u128::from_le_bytes(word);
                (
                    (word & segment.layout.dst_mask as u128) as u64,
                    ((word >> segment.layout.dst_bits)
                        & segment.layout.dest_sort_key_mask.unwrap_or(0) as u128)
                        as u64,
                )
            })
            .peekable();
        let mut added = bucket.into_iter().peekable();
        while original.peek().is_some() || added.peek().is_some() {
            let take_original = match (original.peek(), added.peek()) {
                (Some(&(dst, key)), Some(&(new_dst, new_key))) => {
                    (std::cmp::Reverse(key), dst) <= (std::cmp::Reverse(new_key), new_dst)
                }
                (Some(_), None) => true,
                _ => false,
            };
            let (dst, key) = if take_original {
                original.next().unwrap()
            } else {
                added.next().unwrap()
            };
            ensure!(
                dst <= layout.dst_mask && key <= layout.dest_sort_key_mask.unwrap_or(0),
                "delta edge exceeds layout width"
            );
            let word = dst as u128 | ((key as u128) << layout.dst_bits);
            raw.extend_from_slice(&word.to_le_bytes()[..layout.entry_bytes]);
        }
    }
    if let Some(features) = features {
        raw.extend_from_slice(features);
    }
    Ok(EdgeListView {
        raw: Cow::Owned(raw),
        layouts,
        has_node_data: base.has_node_data,
        node_data_len_bytes: 4,
    })
}

/// Check IDs, packed records, temporal ordering, and paired-edge checksums.
/// Four SHA-256 words are accumulated independently, so traversal order does
/// not matter while multiplicity does. Memory is bounded by relation count.
pub fn validate(root: &Path) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    let graph = crate::graph::GraphIndexReader::open(&root.join("index_metadata.json"), root)?;
    let mut node_counts = BTreeMap::new();
    let mut signatures: BTreeMap<String, (u64, [u64; 4])> = BTreeMap::new();
    for table in graph.tables() {
        let mut nodes = 0u64;
        for shard in 0..table.shard_count() {
            let (first, last) = table.shard_id_range(shard).unwrap();
            if first > last {
                continue;
            }
            for id in first..=last {
                nodes += 1;
                let view = table.edges_for_node(id)?;
                let _ = view.node_data()?;
                for segment in view.relation_segments()? {
                    let relation = &table.relations()[segment.layout.relation_id as usize];
                    let destination = graph
                        .table(&relation.dst_type)
                        .context("missing destination table")?;
                    let mut previous_key = u64::MAX;
                    let signature = signatures.entry(relation.name.clone()).or_default();
                    for bytes in segment.data.chunks_exact(segment.layout.entry_bytes) {
                        ensure!(bytes.len() <= 16, "invalid edge width");
                        let mut word = [0u8; 16];
                        word[..bytes.len()].copy_from_slice(bytes);
                        let word = u128::from_le_bytes(word);
                        let dst = (word & segment.layout.dst_mask as u128) as u64;
                        let key = ((word >> segment.layout.dst_bits)
                            & segment.layout.dest_sort_key_mask.unwrap_or(0) as u128)
                            as u64;
                        ensure!(
                            destination.shard_for_node(dst).is_some(),
                            "dangling edge {}:{id}->{dst}",
                            relation.name
                        );
                        ensure!(
                            key <= previous_key,
                            "unsorted temporal adjacency in {}",
                            relation.name
                        );
                        previous_key = key;
                        let reverse = relation
                            .reverse_of
                            .as_ref()
                            .is_some_and(|name| name < &relation.name);
                        let (a, b) = if reverse { (dst, id) } else { (id, dst) };
                        let mut hash = Sha256::new();
                        hash.update(a.to_le_bytes());
                        hash.update(b.to_le_bytes());
                        let hash = hash.finalize();
                        signature.0 += 1;
                        for (sum, bytes) in signature.1.iter_mut().zip(hash.chunks_exact(8)) {
                            *sum = sum.wrapping_add(u64::from_le_bytes(bytes.try_into().unwrap()));
                        }
                    }
                }
            }
        }
        node_counts.insert(table.node_type().to_owned(), nodes);
    }
    for relation in graph.merged_edge_relations() {
        if let Some(reverse) = &relation.reverse_of {
            ensure!(
                signatures.get(&relation.name) == signatures.get(reverse),
                "forward/reverse edge mismatch: {} / {}",
                relation.name,
                reverse
            );
        }
    }
    Ok(
        serde_json::json!({"nodes": node_counts, "edges": signatures.iter()
        .map(|(name, (count, _))| (name, count)).collect::<BTreeMap<_, _>>()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatible_compaction_keeps_the_existing_buffer() {
        let layouts = Arc::new(Vec::new());
        let raw = vec![1, 0, 0, 0, 42];
        let pointer = raw.as_ptr();
        let base = EdgeListView {
            raw: Cow::Owned(raw),
            layouts: layouts.clone(),
            has_node_data: true,
            node_data_len_bytes: 4,
        };
        let view = merge(base, &[], layouts).unwrap();
        assert_eq!(view.raw.as_ptr(), pointer);
        assert_eq!(view.node_data().unwrap(), Some(&[42][..]));
        // Empty base nodes still need their relation headers materialized.
        let empty = EdgeListView {
            raw: Cow::Borrowed(&[]),
            layouts: layout(4),
            has_node_data: false,
            node_data_len_bytes: 4,
        };
        assert_eq!(merge(empty, &[], layout(4)).unwrap().raw.as_ref(), &[0]);
    }

    #[test]
    fn primary_key_logs_roundtrip_and_legacy_compatibility() {
        use crate::pkmap::Key;
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("keys.bin");
        let keys = vec![
            (Key::Int(255), 2),
            (Key::Int(1), 1),
            (Key::Bytes(b"account".to_vec()), 3),
        ];
        write_primary_keys(
            &path,
            keys.iter()
                .map(|(k, id)| (bincode::serialize(k).unwrap(), *id))
                .collect(),
        )
        .unwrap();
        let log = PrimaryKeyLog::open(&path, |id| id <= 3).unwrap();
        for (key, id) in &keys {
            assert_eq!(log.get(key).unwrap(), Some(*id));
        }
        assert_eq!(log.get(&Key::Int(256)).unwrap(), None);
        let mut streamed = Vec::new();
        visit_primary_keys(&path, |key, id| {
            streamed.push((key, id));
            Ok(())
        })
        .unwrap();
        assert_eq!(streamed.len(), keys.len());
        std::fs::remove_file(path.with_extension("idx")).unwrap();
        let legacy = PrimaryKeyLog::open(&path, |id| id <= 3).unwrap();
        for (key, id) in &keys {
            assert_eq!(legacy.get(key).unwrap(), Some(*id));
        }
        assert!(PrimaryKeyLog::open(&path, |id| id < 3).is_err());
    }

    #[test]
    fn primary_key_index_rejects_corrupt_bounds_and_duplicates() {
        use crate::pkmap::Key;
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("keys.bin");
        let key = bincode::serialize(&Key::Int(1)).unwrap();
        assert!(write_primary_keys(&path, vec![(key.clone(), 1), (key.clone(), 2)]).is_err());
        write_primary_keys(&path, vec![(key, 1)]).unwrap();
        let index_path = path.with_extension("idx");
        let mut index = std::fs::read(&index_path).unwrap();
        index[16..24].copy_from_slice(&9u64.to_le_bytes());
        std::fs::write(index_path, index).unwrap();
        assert!(PrimaryKeyLog::open(&path, |_| true).is_err());
    }

    fn layout(bits: u8) -> Arc<Vec<RelationLayout>> {
        Arc::new(vec![RelationLayout {
            relation_id: 0,
            dst_bits: bits,
            dest_sort_key_bits: Some(8),
            entry_bytes: ((bits as usize + 8) + 7) / 8,
            dst_mask: (1u64 << bits) - 1,
            dest_sort_key_mask: Some(255),
        }])
    }

    #[test]
    fn log_roundtrip_and_node_lookup() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("delta.bin");
        let mut entries = vec![
            EdgeDelta {
                source: 9,
                relation: 0,
                destination: 2,
                sort_key: 30,
            },
            EdgeDelta {
                source: 2,
                relation: 1,
                destination: 5,
                sort_key: 10,
            },
            EdgeDelta {
                source: 2,
                relation: 0,
                destination: 4,
                sort_key: 20,
            },
        ];
        DeltaLog::write(&path, &mut entries).unwrap();
        let log = DeltaLog::open(&path).unwrap();
        assert_eq!(log.len(), 3);
        assert_eq!(log.for_node(2).collect::<Vec<_>>(), entries[..2]);
        assert_eq!(log.for_node(9).count(), 1);
        assert_eq!(log.for_node(1).count(), 0);
        assert_eq!(log.for_node(10).count(), 0);
        assert!(DeltaLog::write(&path, &mut entries).is_err());
        std::fs::write(tmp.path().join("bad.bin"), b"DGDELTA1broken").unwrap();
        assert!(DeltaLog::open(&tmp.path().join("bad.bin")).is_err());
    }

    #[test]
    fn merge_preserves_features_and_temporal_order_across_id_width_growth() {
        // One edge (dst=3, timestamp=20), followed by exact rowpack bytes.
        let bytes = [1, 3, 0, 0x43, 0x01, 9, 8, 7];
        let base = EdgeListView {
            raw: Cow::Borrowed(&bytes),
            layouts: layout(4),
            has_node_data: true,
            node_data_len_bytes: 2,
        };
        let additions = [
            EdgeDelta {
                source: 0,
                relation: 0,
                destination: 300,
                sort_key: 30,
            },
            EdgeDelta {
                source: 0,
                relation: 0,
                destination: 8,
                sort_key: 10,
            },
        ];
        let merged = merge(base, &additions, layout(9)).unwrap();
        assert_eq!(merged.node_data().unwrap().unwrap(), &[9, 8, 7]);
        let segments = merged.relation_segments().unwrap();
        assert_eq!(segments[0].count, 3);
        let values: Vec<_> = segments[0]
            .data
            .chunks_exact(3)
            .map(|b| {
                let word = u32::from_le_bytes([b[0], b[1], b[2], 0]);
                (word & 511, word >> 9)
            })
            .collect();
        assert_eq!(values, vec![(300, 30), (3, 20), (8, 10)]);
        // The original bytes and their narrower layout are unchanged.
        assert_eq!(bytes, [1, 3, 0, 0x43, 0x01, 9, 8, 7]);
    }

    #[test]
    fn added_relations_preserve_old_edges_and_features() {
        let bytes = [1, 3, 0, 0x43, 0x01, 9, 8, 7];
        let mut expanded = layout(4).as_ref().clone();
        let mut new_relation = expanded[0].clone();
        new_relation.relation_id = 1;
        expanded.push(new_relation);
        let expanded = Arc::new(expanded);
        for additions in [vec![], vec![EdgeDelta {
            source: 0, relation: 1, destination: 5, sort_key: 40,
        }]] {
            let base = EdgeListView {
                raw: Cow::Borrowed(&bytes), layouts: layout(4),
                has_node_data: true, node_data_len_bytes: 2,
            };
            let merged = merge(base, &additions, expanded.clone()).unwrap();
            assert_eq!(merged.node_data().unwrap().unwrap(), &[9, 8, 7]);
            let segments = merged.relation_segments().unwrap();
            assert_eq!(segments.len(), 2);
            assert_eq!(segments[0].data, &[0x43, 0x01]);
            assert_eq!(segments[1].count as usize, additions.len());
        }
        let temp = tempfile::tempdir().unwrap();
        let mut writer = NodeAppender::new(temp.path(), 1).unwrap();
        writer.push(&[9, 8, 7]).unwrap();
        writer.finish(temp.path()).unwrap();
        let reader = crate::reader::ShardReader::open(temp.path(), layout(4), true)
            .unwrap().with_deltas(0, vec![], expanded);
        let view = reader.edges_for_node(0).unwrap();
        assert_eq!(view.node_data().unwrap().unwrap(), &[9, 8, 7]);
        assert_eq!(view.relation_segments().unwrap().len(), 2);
    }

    #[test]
    fn unknown_relation_is_rejected() {
        let base = EdgeListView {
            raw: Cow::Borrowed(&[]),
            layouts: layout(4),
            has_node_data: false,
            node_data_len_bytes: 2,
        };
        let entry = EdgeDelta {
            source: 0,
            relation: 1,
            destination: 1,
            sort_key: 0,
        };
        assert!(merge(base, &[entry], layout(4)).is_err());
    }
}
