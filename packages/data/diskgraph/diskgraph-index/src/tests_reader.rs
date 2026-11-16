// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::builder::delta_encode_offsets_into;
use crate::layout::{OffsetFileHeader, DELTA_CHUNK, EDGES_FILENAME, OFFSETS_FILENAME};
use crate::reader::{decode_offset, ShardReader};
use std::sync::Arc;

struct LegacyChunk {
    base_idx: u64,
    count: u16,
    data_offset: usize,
}

fn legacy_chunks(
    offsets: &[u8],
    total_offsets: u64,
    delta_bytes: u8,
    header_len: usize,
) -> Vec<LegacyChunk> {
    let mut chunks = Vec::new();
    let mut cursor = header_len;
    let mut base_idx = 0u64;
    while cursor < offsets.len() && base_idx < total_offsets {
        if cursor + 10 > offsets.len() {
            break;
        }
        let count = u16::from_le_bytes([offsets[cursor], offsets[cursor + 1]]);
        chunks.push(LegacyChunk {
            base_idx,
            count,
            data_offset: cursor,
        });
        cursor += 10 + (count.saturating_sub(1) as usize) * delta_bytes as usize;
        base_idx += count as u64;
    }
    chunks
}

fn legacy_decode(offsets: &[u8], chunks: &[LegacyChunk], idx: u64, delta_bytes: u8) -> Option<u64> {
    let chunk = chunks.get((idx / DELTA_CHUNK as u64) as usize)?;
    if idx < chunk.base_idx || idx >= chunk.base_idx + chunk.count as u64 {
        return None;
    }
    let mut cursor = chunk.data_offset + 2;
    let mut acc = u64::from_le_bytes(offsets[cursor..cursor + 8].try_into().unwrap());
    cursor += 8;
    for _ in 0..idx - chunk.base_idx {
        let width = delta_bytes as usize;
        if cursor + width > offsets.len() {
            return None;
        }
        let mut buf = [0u8; 8];
        buf[..width].copy_from_slice(&offsets[cursor..cursor + width]);
        acc += u64::from_le_bytes(buf);
        cursor += width;
    }
    Some(acc)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn random_offsets(rng: &mut Rng, total: usize, delta_bytes: u8) -> Vec<u64> {
    let max_delta = (1u64 << (8 * delta_bytes as u32)) - 1;
    let mut offsets = Vec::with_capacity(total);
    let mut cur = rng.next() % 1000;
    for i in 0..total {
        if i > 0 {
            cur += match rng.next() % 4 {
                0 => 0,
                1 => max_delta,
                _ => rng.next() % (max_delta + 1),
            };
        }
        offsets.push(cur);
    }
    offsets
}

fn offsets_file(offsets: &[u64], delta_bytes: u8) -> (Vec<u8>, OffsetFileHeader, usize) {
    let mut header =
        OffsetFileHeader::new(offsets.len().saturating_sub(1) as u64, offsets.len() as u64);
    header.delta_bytes = delta_bytes;
    let mut buf = bincode::serialize(&header).unwrap();
    let header_len = buf.len();
    delta_encode_offsets_into(offsets, &mut buf, delta_bytes).unwrap();
    (buf, header, header_len)
}

const SIZES: [usize; 11] = [1, 2, 63, 64, 65, 127, 128, 129, 191, 4096, 5000];

#[test]
fn computed_chunk_positions_match_legacy_directory() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for delta_bytes in 2..=4u8 {
        for &size in SIZES
            .iter()
            .chain([1 + (rng.next() % 20_000) as usize].iter())
        {
            let offsets = random_offsets(&mut rng, size, delta_bytes);
            let (buf, header, header_len) = offsets_file(&offsets, delta_bytes);
            let legacy = legacy_chunks(&buf, header.total_offsets, delta_bytes, header_len);
            for idx in 0..size as u64 {
                let new = decode_offset(&buf, delta_bytes, header_len, idx).unwrap();
                assert_eq!(Some(new), legacy_decode(&buf, &legacy, idx, delta_bytes));
                assert_eq!(new, offsets[idx as usize]);
            }
            assert!(decode_offset(&buf, delta_bytes, header_len, size as u64).is_err());
        }
    }
}

#[test]
fn truncated_offsets_fail_on_the_same_indices_as_legacy() {
    let mut rng = Rng(42);
    for delta_bytes in 2..=4u8 {
        let offsets = random_offsets(&mut rng, 300, delta_bytes);
        let (full, header, header_len) = offsets_file(&offsets, delta_bytes);
        for cut in [
            header_len,
            header_len + 9,
            header_len + 10,
            full.len() / 2,
            full.len() - 1,
        ] {
            let buf = &full[..cut];
            let legacy = legacy_chunks(buf, header.total_offsets, delta_bytes, header_len);
            for idx in 0..offsets.len() as u64 {
                let new = decode_offset(buf, delta_bytes, header_len, idx).ok();
                assert_eq!(
                    new,
                    legacy_decode(buf, &legacy, idx, delta_bytes),
                    "cut={cut} idx={idx}"
                );
            }
        }
    }
}

#[test]
fn shard_reader_slices_match_offsets() {
    let mut rng = Rng(7);
    for delta_bytes in 2..=4u8 {
        let mut offsets = vec![0u64];
        for _ in 0..2 * DELTA_CHUNK + 17 {
            offsets.push(offsets.last().unwrap() + rng.next() % 300);
        }
        let (buf, _, _) = offsets_file(&offsets, delta_bytes);
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(OFFSETS_FILENAME), &buf).unwrap();
        std::fs::write(
            dir.path().join(EDGES_FILENAME),
            vec![0u8; *offsets.last().unwrap() as usize],
        )
        .unwrap();
        let reader = ShardReader::open(dir.path(), Arc::new(Vec::new()), false).unwrap();
        let base = reader.edges_for_node(0).unwrap().raw.as_ptr() as u64 - offsets[0];
        for node in 0..offsets.len() as u64 - 1 {
            let view = reader.edges_for_node(node).unwrap();
            assert_eq!(view.raw.as_ptr() as u64 - base, offsets[node as usize]);
            assert_eq!(
                view.raw.len() as u64,
                offsets[node as usize + 1] - offsets[node as usize]
            );
        }
        assert!(reader.edges_for_node(offsets.len() as u64 - 1).is_err());
    }
}
