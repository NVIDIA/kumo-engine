// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Immutable, sorted edge additions. Old snapshots never open a newer log.

use crate::layout::{EdgeListView, RelationLayout};
use anyhow::{ensure, Result};
use memmap2::Mmap;
use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

const MAGIC: &[u8; 8] = b"DGDELTA1";
const WIDTH: usize = 26;

pub const MANIFEST: &str = "_expansion/relations.json";

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct DeltaManifest {
    pub version: u32,
    pub tables: std::collections::BTreeMap<String, DeltaTable>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct DeltaTable {
    pub shard_layouts: Vec<Vec<RelationLayout>>,
    pub edge_logs: Vec<String>,
}

impl DeltaManifest {
    pub fn load(root: &Path) -> Result<Option<Self>> {
        let path = root.join(MANIFEST);
        if !path.exists() {
            return Ok(None);
        }
        let manifest: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(manifest.version == 1, "unsupported delta manifest version");
        Ok(Some(manifest))
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

#[cfg(test)]
mod tests {
    use super::*;
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
