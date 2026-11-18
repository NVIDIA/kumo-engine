// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::consts::*;
use crate::mapping::{KeyType, PkPtrHashOutput};
use gxhash::GxHasher;
use memmap2::Mmap;
use std::fs;
use std::hash::BuildHasherDefault;
use std::path::PathBuf;

#[derive(Default)]
pub struct EdgeMapScratch {
    pub key_buf: Vec<u8>,
    pub fkey_bitmap: Option<Mmap>,
}

pub type PkMapHasher = BuildHasherDefault<GxHasher>;
pub type PkMapHashMap<K, V> = std::collections::HashMap<K, V, PkMapHasher>;

pub fn new_pk_map<K, V>(capacity: usize) -> PkMapHashMap<K, V> {
    std::collections::HashMap::with_capacity_and_hasher(capacity, Default::default())
}

fn pk_map_entry_size(key_type: KeyType) -> u64 {
    (match key_type {
        KeyType::Int64 => std::mem::size_of::<(u64, u32)>(),
        KeyType::VarBytes => std::mem::size_of::<(Vec<u8>, u32)>(),
        KeyType::Uuid => std::mem::size_of::<([u8; 16], u32)>(),
        KeyType::Uuid256 => std::mem::size_of::<([u8; 32], u32)>(),
        KeyType::Ip => std::mem::size_of::<([u8; 16], u32)>(),
    }) as u64
}

pub fn estimate_pk_map_alloc_bytes(key_rows: u64, key_type: KeyType, avg_var_bytes: u64) -> u64 {
    if key_rows == 0 {
        return 0;
    }
    let buckets = key_rows
        .saturating_mul(PK_MAP_LOAD_FACTOR_NUM)
        .div_ceil(PK_MAP_LOAD_FACTOR_DEN)
        .checked_next_power_of_two()
        .unwrap_or(u64::MAX);
    let map_bytes = buckets.saturating_mul(pk_map_entry_size(key_type) + 1);
    let key_bytes = if matches!(key_type, KeyType::VarBytes) {
        key_rows.saturating_mul(avg_var_bytes)
    } else {
        0
    };
    let slots = key_rows
        .saturating_mul(PK_MAP_MPHF_FACTOR_NUM)
        .div_ceil(PK_MAP_MPHF_FACTOR_DEN);
    let slot_bytes = slots
        .saturating_mul((std::mem::size_of::<(u64, u32)>() + std::mem::size_of::<u32>()) as u64);
    let hash_bytes = key_rows.saturating_mul(
        (2 * std::mem::size_of::<PkPtrHashOutput>()) as u64 + PTRHASH_BUCKET_BYTES_PER_KEY,
    );
    map_bytes
        .saturating_add(key_bytes)
        .saturating_add(slot_bytes)
        .saturating_add(hash_bytes)
}

pub fn pk_map_task_estimate(
    union_rows: u64,
    own_rows: u64,
    key_type: KeyType,
    avg_var_bytes: u64,
) -> u64 {
    estimate_pk_map_bytes(union_rows, key_type, avg_var_bytes).max(estimate_pk_map_alloc_bytes(
        own_rows,
        key_type,
        avg_var_bytes,
    ))
}

pub fn estimate_pk_map_bytes(row_count: u64, key_type: KeyType, avg_var_bytes: u64) -> u64 {
    if row_count == 0 {
        return 0;
    }
    let capacity = row_count
        .saturating_mul(PK_MAP_LOAD_FACTOR_NUM)
        .saturating_add(PK_MAP_LOAD_FACTOR_DEN - 1)
        / PK_MAP_LOAD_FACTOR_DEN;
    let entry_size = pk_map_entry_size(key_type);
    let ctrl_bytes = capacity;
    let map_bytes = capacity
        .saturating_mul(entry_size)
        .saturating_add(ctrl_bytes);
    let key_bytes = if matches!(key_type, KeyType::VarBytes) {
        row_count.saturating_mul(avg_var_bytes)
    } else {
        0
    };
    let slots = row_count
        .saturating_mul(PK_MAP_MPHF_FACTOR_NUM)
        .saturating_add(PK_MAP_MPHF_FACTOR_DEN - 1)
        / PK_MAP_MPHF_FACTOR_DEN;
    let entries_bytes = slots.saturating_mul(std::mem::size_of::<(u64, u32)>() as u64);
    let base = map_bytes
        .saturating_add(key_bytes)
        .saturating_add(entries_bytes);
    base.saturating_mul(100 + PK_MAP_ESTIMATE_PAD_PCT) / 100
}

pub fn estimate_varbytes_avg_len_multi(row_count: u64, max_row_len: u32, paths: &[PathBuf]) -> u64 {
    if row_count == 0 {
        return 0;
    }
    let max_row_len = max_row_len as u64;
    let mut total_bytes = 0u64;
    for path in paths {
        if let Ok(meta) = fs::metadata(path) {
            total_bytes = total_bytes.saturating_add(meta.len());
        }
    }
    if total_bytes == 0 {
        return max_row_len;
    }
    let avg_row = total_bytes / row_count;
    let mut avg_key = avg_row.saturating_sub(4);
    if avg_key == 0 {
        avg_key = max_row_len;
    }
    if max_row_len > 0 {
        avg_key = avg_key.min(max_row_len);
    }
    avg_key
}
