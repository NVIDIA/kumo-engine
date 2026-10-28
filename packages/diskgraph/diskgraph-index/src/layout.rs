// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::pkmap::Key;
use diskgraph_core::metadata::PrimaryKeyMap;
use diskgraph_rowpack::{
    ColumnContext, ColumnEncoding, IngestedType, RowPackDecodeScratch, RowPackDecoder, ValueRef,
};
use serde::{Deserialize, Serialize};

/// Logical layout version for on-disk shard artifacts.
/// v4: offset deltas are 16/24/32-bit; node_data_len width stored in header.
/// v3: compact count varints (1–2 bytes, capped at 10k per relation per node).
/// v2: zero-edge nodes write no counts or data; offsets may repeat (delta 0).
pub const LAYOUT_VERSION: u32 = 4;

/// Filename conventions for artifacts emitted per shard.
pub const OFFSETS_FILENAME: &str = "offsets.bin";
pub const EDGES_FILENAME: &str = "edges.bin";

/// Number of offsets to delta-pack in a chunk.
pub const DELTA_CHUNK: usize = 64;

/// Header persisted at the start of the offsets file.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OffsetFileHeader {
    pub version: u32,
    pub nodes: u64,
    pub total_offsets: u64,
    pub delta_chunk: u16,
    /// Offsets refer to byte positions in the edge blob.
    pub edge_blob_word_size: u16,
    /// Number of bytes per delta in offsets.bin (2, 3, or 4).
    pub delta_bytes: u8,
    /// Number of bytes used to store node_data_len in edges.bin (2, 3, or 4).
    pub node_data_len_bytes: u8,
    /// Total edges encoded across all relations (optional; may be zero if not set).
    pub edge_count: u64,
}

impl OffsetFileHeader {
    pub fn new(nodes: u64, total_offsets: u64) -> Self {
        Self {
            version: LAYOUT_VERSION,
            nodes,
            total_offsets,
            delta_chunk: DELTA_CHUNK as u16,
            edge_blob_word_size: 1,
            delta_bytes: 2,
            node_data_len_bytes: 2,
            edge_count: 0,
        }
    }
}

/// Metadata captured per node while building the shard.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct NodeEdgeStats {
    pub edge_count: u64,
    pub byte_len: u64,
}

/// Relation-specific layout information used to decode packed edges.
#[derive(Debug, Clone)]
pub struct RelationLayout {
    pub relation_id: u16,
    pub dst_bits: u8,
    pub dest_sort_key_bits: Option<u8>,
    pub entry_bytes: usize,
    /// Precomputed masks to avoid per-edge recomputation.
    pub dst_mask: u64,
    pub dest_sort_key_mask: Option<u64>,
}

/// A view over an edge list stored in the edge data file. This is a lightweight
/// pointer into an mmapped region; decoding uses the on-disk layout rather than
/// copying data out.
pub struct EdgeListView<'a> {
    pub raw: &'a [u8],
    pub layouts: std::sync::Arc<Vec<RelationLayout>>,
    pub has_node_data: bool,
    pub node_data_len_bytes: u8,
}

impl<'a> EdgeListView<'a> {
    /// Returns the relation segments for this node. Each entry contains the
    /// edge count and raw slice for that relation's edges.
    pub fn relation_segments(&self) -> Result<Vec<RelationSegment<'a>>, std::io::Error> {
        Ok(self.parse()?.0)
    }

    pub fn node_data(&self) -> Result<Option<&'a [u8]>, std::io::Error> {
        if self.raw.is_empty() {
            return if self.has_node_data {
                Err(std::io::ErrorKind::InvalidData.into())
            } else {
                Ok(None)
            };
        }

        // Validate the entire count prefix before lengths or edge sizes, as
        // parse() does. Replaying this small prefix avoids allocating counts
        // and relation segments that a feature-only read would discard.
        let mut cursor = 0usize;
        for _ in self.layouts.iter() {
            let (_, consumed) = decode_varint(&self.raw[cursor..])?;
            cursor = cursor
                .checked_add(consumed)
                .ok_or(std::io::ErrorKind::InvalidData)?;
        }
        let node_data_len = if self.has_node_data {
            let len_bytes = self.node_data_len_bytes as usize;
            if len_bytes == 0 || len_bytes > 4 {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            if cursor + len_bytes > self.raw.len() {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            let len = read_len_le(&self.raw[cursor..cursor + len_bytes]);
            cursor += len_bytes;
            Some(len)
        } else {
            None
        };

        let mut count_cursor = 0usize;
        for layout in self.layouts.iter() {
            let (count, consumed) = decode_varint(&self.raw[count_cursor..])?;
            count_cursor += consumed; // Validated by the first pass.
            let bytes = count
                .checked_mul(layout.entry_bytes as u32)
                .ok_or(std::io::ErrorKind::InvalidData)? as usize;
            cursor = cursor
                .checked_add(bytes)
                .ok_or(std::io::ErrorKind::InvalidData)?;
            if cursor > self.raw.len() {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
        }
        if let Some(len) = node_data_len {
            let end = cursor
                .checked_add(len)
                .ok_or(std::io::ErrorKind::InvalidData)?;
            if end > self.raw.len() {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            if end != self.raw.len() {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            Ok(Some(&self.raw[cursor..end]))
        } else {
            if cursor != self.raw.len() {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            Ok(None)
        }
    }

    pub fn pk_matches(
        &self,
        pk: &PrimaryKeyMap,
        decoder: &RowPackDecoder,
        key: &Key,
    ) -> Result<Option<bool>, std::io::Error> {
        let raw = match self.node_data()? {
            Some(raw) => raw,
            None => return Ok(None),
        };
        let ctx = decoder.context();
        let col_idx = match pk.pk_column_index.and_then(|idx| {
            ctx.columns
                .get(idx)
                .filter(|col| col.name == pk.column)
                .map(|_| idx)
        }) {
            Some(idx) => idx,
            None => match ctx.columns.iter().position(|col| col.name == pk.column) {
                Some(idx) => idx,
                None => return Ok(None),
            },
        };
        let mut scratch = RowPackDecodeScratch::default();
        let value = match decoder.decode_column_value(raw, col_idx, &mut scratch) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        Ok(pk_value_matches(key, &value, &ctx.columns[col_idx]))
    }

    fn parse(&self) -> Result<(Vec<RelationSegment<'a>>, Option<&'a [u8]>), std::io::Error> {
        if self.raw.is_empty() {
            if self.has_node_data {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            // Node has no edges; return empty segments for each relation.
            let segments = self
                .layouts
                .iter()
                .map(|layout| RelationSegment {
                    layout: layout.clone(),
                    count: 0,
                    data: &[],
                })
                .collect();
            return Ok((segments, None));
        }
        let mut segments = Vec::with_capacity(self.layouts.len());
        // First read all counts up front (prefix) to allow quick validation.
        let mut cursor = 0usize;
        let mut counts = Vec::with_capacity(self.layouts.len());
        for layout in self.layouts.iter() {
            let (count, consumed) = decode_varint(&self.raw[cursor..])?;
            cursor = cursor
                .checked_add(consumed)
                .ok_or(std::io::ErrorKind::InvalidData)?;
            counts.push((layout.clone(), count));
        }
        let node_data_len = if self.has_node_data {
            let len_bytes = self.node_data_len_bytes as usize;
            if len_bytes == 0 || len_bytes > 4 {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            if cursor + len_bytes > self.raw.len() {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            let len = read_len_le(&self.raw[cursor..cursor + len_bytes]);
            cursor += len_bytes;
            Some(len)
        } else {
            None
        };
        // Now slice the data segments in the same order.
        for (layout, count) in counts {
            let bytes = count
                .checked_mul(layout.entry_bytes as u32)
                .ok_or(std::io::ErrorKind::InvalidData)? as usize;
            let end = cursor
                .checked_add(bytes)
                .ok_or(std::io::ErrorKind::InvalidData)?;
            if end > self.raw.len() {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            let data = &self.raw[cursor..end];
            segments.push(RelationSegment {
                layout,
                count,
                data,
            });
            cursor = end;
        }
        let node_data = if let Some(len) = node_data_len {
            let end = cursor
                .checked_add(len)
                .ok_or(std::io::ErrorKind::InvalidData)?;
            if end > self.raw.len() {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            if end != self.raw.len() {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            Some(&self.raw[cursor..end])
        } else {
            if cursor != self.raw.len() {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            None
        };
        Ok((segments, node_data))
    }
}

/// Compare a lookup `key` against the node's decoded stored PK `value`.
///
/// Returns `Some(bool)` only for genuinely comparable combinations; `None`
/// when the stored PK's type has no comparison arm (e.g. Date/Timestamp PK
/// columns) — callers must treat `None` as "cannot verify"
/// and TRUST the pkmap hit, never drop, matching the table-level
/// unverifiable fallback in `validate_pk_at_node`. Integer keys are compared
/// in the two's-complement u64 domain the pkmap keys are built in, so
/// negative PK values match their own keys. A `Null` stored PK is a genuine
/// mismatch: the pkmap never maps a key to a node whose PK is NULL.
fn pk_value_matches(key: &Key, value: &ValueRef<'_>, column: &ColumnContext) -> Option<bool> {
    if column.logical_type == IngestedType::Decimal128 {
        let scale = decimal_scale(column)?;
        let stored = match value {
            ValueRef::Decimal128(raw) => decimal128_key_from_raw(*raw, scale)?,
            ValueRef::String(text) => decimal128_key_from_text(text, scale)?,
            ValueRef::Null => return Some(false),
            _ => return None,
        };
        return match key {
            Key::Int(key) => Some(*key == stored),
            _ => Some(false),
        };
    }
    if column.logical_type == IngestedType::Decimal256 {
        let stored = match value {
            ValueRef::String(text) => {
                diskgraph_core::decimal_key::normalize_base10_key(text.as_bytes()).ok()?
            }
            ValueRef::Null => return Some(false),
            _ => return None,
        };
        return match key {
            Key::Bytes(key) => Some(*key == stored),
            _ => Some(false),
        };
    }
    match (key, value) {
        (Key::Int(k), ValueRef::Int64(v)) => Some(*k == *v as u64),
        (Key::Int(k), ValueRef::Int32(v)) => Some(*k == *v as i64 as u64),
        (Key::Int(k), ValueRef::UInt64(v)) => Some(*k == *v),
        (Key::Int(k), ValueRef::UInt32(v)) => Some(*k == *v as u64),
        (Key::Bytes(k), ValueRef::Bytes(v)) => Some(k.as_slice() == *v),
        (Key::Bytes(k), ValueRef::String(v)) => {
            let bytes = v.as_bytes();
            let trimmed = trim_trailing_nuls(bytes);
            Some(k.as_slice() == trimmed)
        }
        (Key::Uuid(k), ValueRef::Uuid(v)) => Some(k == v),
        (Key::Uuid256(k), ValueRef::Uuid256(v)) => Some(k == v),
        (Key::Ip(k), ValueRef::Ip(v)) => Some(k == v),
        (_, ValueRef::Null) => Some(false),
        _ => None,
    }
}

fn decimal_scale(column: &ColumnContext) -> Option<i32> {
    match &column.encoding {
        ColumnEncoding::Decimal { scale, .. } => Some(*scale),
        _ => None,
    }
}

fn decimal128_key_from_raw(mut raw: i128, scale: i32) -> Option<u64> {
    if scale > 0 {
        raw /= 10i128.checked_pow(scale as u32)?;
    }
    Some(i64::try_from(raw).ok()? as u64)
}

fn decimal128_key_from_text(text: &str, scale: i32) -> Option<u64> {
    let (negative, unsigned) = match text.strip_prefix('-') {
        Some(unsigned) => (true, unsigned),
        None => (false, text),
    };
    let mut parts = unsigned.split('.');
    let int_part = parts.next()?;
    let frac_part = parts.next();
    if int_part.is_empty()
        || parts.next().is_some()
        || !int_part.bytes().all(|byte| byte.is_ascii_digit())
        || frac_part.is_some_and(|frac| !frac.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return None;
    }
    let magnitude = int_part.parse::<i128>().ok()?;
    let mut value = if negative { -magnitude } else { magnitude };
    if scale < 0 {
        value /= 10i128.checked_pow(scale.unsigned_abs())?;
    }
    Some(i64::try_from(value).ok()? as u64)
}

fn trim_trailing_nuls(bytes: &[u8]) -> &[u8] {
    match bytes.iter().rposition(|b| *b != 0) {
        Some(idx) => &bytes[..=idx],
        None => &[],
    }
}

/// A relation-specific slice of edges for a node.
pub struct RelationSegment<'a> {
    pub layout: RelationLayout,
    pub count: u32,
    pub data: &'a [u8],
}

fn read_len_le(buf: &[u8]) -> usize {
    match buf.len() {
        2 => u16::from_le_bytes([buf[0], buf[1]]) as usize,
        3 => (buf[0] as u32 | ((buf[1] as u32) << 8) | ((buf[2] as u32) << 16)) as usize,
        4 => u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize,
        _ => 0,
    }
}

fn decode_varint(buf: &[u8]) -> Result<(u32, usize), std::io::Error> {
    if buf.is_empty() {
        return Err(std::io::ErrorKind::UnexpectedEof.into());
    }
    let first = buf[0];
    let tag = first >> 6;
    // Fast path: 1-byte count (0..=63).
    if tag == 0 {
        return Ok(((first & 0x3F) as u32, 1));
    }
    // Fast path: 2-byte count (64..=0x3FFF).
    if tag == 1 {
        if buf.len() < 2 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        let hi = (first & 0x3F) as u32;
        let lo = buf[1] as u32;
        return Ok(((hi << 8) | lo, 2));
    }
    // Extended counts: 3 or 4 bytes.
    if tag == 2 {
        if buf.len() < 3 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        let b0 = (first & 0x3F) as u32;
        let b1 = buf[1] as u32;
        let b2 = buf[2] as u32;
        return Ok(((b0 << 16) | (b1 << 8) | b2, 3));
    }
    if tag == 3 {
        if buf.len() < 4 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        let b0 = (first & 0x3F) as u32;
        let b1 = buf[1] as u32;
        let b2 = buf[2] as u32;
        let b3 = buf[3] as u32;
        return Ok(((b0 << 24) | (b1 << 16) | (b2 << 8) | b3, 4));
    }
    Err(std::io::ErrorKind::InvalidData.into())
}

#[cfg(test)]
mod pk_value_matches_tests {
    use super::pk_value_matches;
    use crate::pkmap::Key;
    use diskgraph_rowpack::{
        ColumnContext, ColumnEncoding, DecimalStorage, IngestedType, ValueRef,
    };

    fn column(logical_type: IngestedType) -> ColumnContext {
        let encoding = match logical_type {
            IngestedType::Decimal128 => ColumnEncoding::Decimal {
                precision: 38,
                scale: 2,
                bytes: 16,
                storage: DecimalStorage::Raw,
            },
            IngestedType::Decimal256 => ColumnEncoding::Decimal {
                precision: 76,
                scale: 2,
                bytes: 32,
                storage: DecimalStorage::Raw,
            },
            _ => ColumnEncoding::Int {
                signed: true,
                varint: false,
                bits: 64,
            },
        };
        ColumnContext {
            name: "id".to_string(),
            logical_type,
            encoding,
        }
    }

    #[test]
    fn negative_int_pk_matches_its_own_key() {
        // Keys for negative PKs are built as two's-complement u64; a member
        // node with a negative id must verify true, never drop.
        let key = Key::Int((-5i64) as u64);
        let column = column(IngestedType::Int64);
        assert_eq!(
            pk_value_matches(&key, &ValueRef::Int64(-5), &column),
            Some(true)
        );
        assert_eq!(
            pk_value_matches(&key, &ValueRef::Int64(-4), &column),
            Some(false)
        );
        let key32 = Key::Int((-7i64) as u64);
        assert_eq!(
            pk_value_matches(&key32, &ValueRef::Int32(-7), &column),
            Some(true)
        );
    }

    #[test]
    fn decimal128_string_value_verifies_the_integer_lookup_key() {
        let column = column(IngestedType::Decimal128);
        assert_eq!(
            pk_value_matches(&Key::Int(123), &ValueRef::String("123.00"), &column),
            Some(true)
        );
        assert_eq!(
            pk_value_matches(
                &Key::Int((-7i64) as u64),
                &ValueRef::String("-7.90"),
                &column,
            ),
            Some(true)
        );
        assert_eq!(
            pk_value_matches(&Key::Int(999), &ValueRef::String("123.00"), &column),
            Some(false)
        );
    }

    #[test]
    fn decimal256_string_value_verifies_the_normalized_lookup_key() {
        let column = column(IngestedType::Decimal256);
        assert_eq!(
            pk_value_matches(
                &Key::Bytes(b"123".to_vec()),
                &ValueRef::String("00123.0000"),
                &column,
            ),
            Some(true)
        );
        assert_eq!(
            pk_value_matches(
                &Key::Bytes(b"-7.5".to_vec()),
                &ValueRef::String("-007.500"),
                &column,
            ),
            Some(true)
        );
        assert_eq!(
            pk_value_matches(
                &Key::Bytes(b"999".to_vec()),
                &ValueRef::String("123.00"),
                &column,
            ),
            Some(false)
        );
    }

    #[test]
    fn uncomparable_pk_types_are_unverifiable_not_mismatch() {
        let key = Key::Int(42);
        let column = column(IngestedType::Date);
        assert_eq!(pk_value_matches(&key, &ValueRef::Date(42), &column), None);
        assert_eq!(
            pk_value_matches(&key, &ValueRef::Timestamp(42), &column),
            None
        );
    }

    #[test]
    fn null_stored_pk_is_a_mismatch() {
        // The pkmap never maps a key to a node whose PK is NULL.
        assert_eq!(
            pk_value_matches(
                &Key::Int(1),
                &ValueRef::Null,
                &column(IngestedType::Int64),
            ),
            Some(false)
        );
    }
}

#[cfg(test)]
mod node_data_tests {
    use super::*;
    use std::io::ErrorKind;
    use std::sync::Arc;

    fn layouts(widths: &[usize]) -> Arc<Vec<RelationLayout>> {
        Arc::new(
            widths
                .iter()
                .enumerate()
                .map(|(i, &entry_bytes)| RelationLayout {
                    relation_id: i as u16,
                    dst_bits: 8,
                    dest_sort_key_bits: None,
                    entry_bytes,
                    dst_mask: 255,
                    dest_sort_key_mask: None,
                })
                .collect(),
        )
    }

    fn check(view: &EdgeListView<'_>) {
        let expected = view.parse().map(|(_, data)| data).map_err(|e| e.kind());
        let actual = view.node_data().map_err(|e| e.kind());
        assert_eq!(
            actual, expected,
            "raw={:?}, length width={}",
            view.raw, view.node_data_len_bytes
        );
        if let Ok(Some(data)) = actual {
            assert_eq!(data.as_ptr(), expected.unwrap().unwrap().as_ptr());
        }
    }

    #[test]
    fn node_data_matches_parse_for_valid_rows_and_all_truncations() {
        for count_width in 1..=4 {
            for len_width in 1..=4 {
                for has_node_data in [false, true] {
                    // Non-minimal count widths are accepted by the existing parser.
                    let mut raw = Vec::new();
                    for count in [2u32, 3] {
                        let encoded = count | (((count_width - 1) as u32) << (count_width * 8 - 2));
                        raw.extend_from_slice(&encoded.to_be_bytes()[4 - count_width..]);
                    }
                    // Preserve the old one-byte length behavior (read_len_le returns 0).
                    let payload_len = if len_width == 1 { 0u32 } else { 7 };
                    if has_node_data {
                        raw.extend_from_slice(&payload_len.to_le_bytes()[..len_width]);
                    }
                    raw.extend_from_slice(&[17; 2 + 3 * 8]);
                    if has_node_data {
                        raw.resize(raw.len() + payload_len as usize, 29);
                    }
                    let view = EdgeListView {
                        raw: &raw,
                        layouts: layouts(&[1, 8]),
                        has_node_data,
                        node_data_len_bytes: len_width as u8,
                    };
                    let expected = if has_node_data {
                        Some(&raw[raw.len() - payload_len as usize..])
                    } else {
                        None
                    };
                    assert_eq!(view.node_data().unwrap(), expected);
                    check(&view);
                    for end in 0..raw.len() {
                        check(&EdgeListView {
                            raw: &raw[..end],
                            layouts: view.layouts.clone(),
                            has_node_data,
                            node_data_len_bytes: len_width as u8,
                        });
                    }
                    raw.push(99);
                    check(&EdgeListView {
                        raw: &raw,
                        layouts: layouts(&[1, 8]),
                        has_node_data,
                        node_data_len_bytes: len_width as u8,
                    });
                }
            }
        }
    }

    #[test]
    fn node_data_preserves_error_precedence_and_empty_rows() {
        type Case<'a> = (&'a [u8], &'a [usize], bool, u8, Option<ErrorKind>);
        let cases: &[Case<'_>] = &[
            (&[], &[8], false, 255, None),
            (&[], &[], true, 2, Some(ErrorKind::InvalidData)),
            (&[0], &[], true, 1, None),
            (&[0, 0], &[8], true, 0, Some(ErrorKind::InvalidData)),
            (&[0, 0], &[8], true, 5, Some(ErrorKind::InvalidData)),
            // Missing second count takes precedence over invalid length width.
            (&[0], &[1, 8], true, 0, Some(ErrorKind::UnexpectedEof)),
            // Missing node-data length takes precedence over edge-size overflow.
            (
                &[255, 255, 255, 255],
                &[8],
                true,
                2,
                Some(ErrorKind::UnexpectedEof),
            ),
            // Earlier segment EOF takes precedence over later count multiplication overflow.
            (
                &[2, 255, 255, 255, 255, 0, 0],
                &[1, 8],
                true,
                2,
                Some(ErrorKind::UnexpectedEof),
            ),
            (
                &[255, 255, 255, 255, 0, 0],
                &[8],
                true,
                2,
                Some(ErrorKind::InvalidData),
            ),
            (&[0, 0, 7], &[], true, 2, Some(ErrorKind::InvalidData)),
            (&[7], &[], false, 2, Some(ErrorKind::InvalidData)),
        ];
        for &(raw, widths, has_node_data, node_data_len_bytes, error) in cases {
            let view = EdgeListView {
                raw,
                layouts: layouts(widths),
                has_node_data,
                node_data_len_bytes,
            };
            check(&view);
            assert_eq!(view.node_data().err().map(|e| e.kind()), error);
        }
    }

    #[test]
    fn node_data_matches_parse_for_arbitrary_buffers() {
        let mut seed = 42u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..20_000 {
            let len = (next() % 128) as usize;
            let raw: Vec<_> = (0..len).map(|_| next() as u8).collect();
            let relation_count = (next() % 9) as usize;
            let widths: Vec<_> = (0..relation_count)
                .map(|_| [0, 1, 8, 64, u32::MAX as usize, usize::MAX][(next() % 6) as usize])
                .collect();
            let view = EdgeListView {
                raw: &raw,
                layouts: layouts(&widths),
                has_node_data: next() & 1 != 0,
                node_data_len_bytes: (next() % 7) as u8,
            };
            check(&view);
        }
    }
}
