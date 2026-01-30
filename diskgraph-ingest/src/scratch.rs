use crate::consts::*;
use crate::mapping::{KeyType, PkPtrHashOutput};
use gxhash::GxHasher;
use memmap2::Mmap;
use ptr_hash::PtrHashScratch;
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

pub fn new_pk_map<K, V>() -> PkMapHashMap<K, V> {
    std::collections::HashMap::with_hasher(Default::default())
}

pub struct PkMapScratch {
    pub seen_u64: PkMapHashMap<u64, u32>,
    pub seen_uuid: PkMapHashMap<[u8; 16], u32>,
    pub seen_uuid256: PkMapHashMap<[u8; 32], u32>,
    pub seen_ip: PkMapHashMap<[u8; 16], u32>,
    pub seen_bytes: PkMapHashMap<Vec<u8>, u32>,
    pub entries: Vec<(u64, u32)>,
    pub ptrhash: PtrHashScratch<PkPtrHashOutput>,
}

impl PkMapScratch {
    pub fn new() -> Self {
        Self {
            seen_u64: new_pk_map(),
            seen_uuid: new_pk_map(),
            seen_uuid256: new_pk_map(),
            seen_ip: new_pk_map(),
            seen_bytes: new_pk_map(),
            entries: Vec::new(),
            ptrhash: PtrHashScratch::new(),
        }
    }

    pub fn clear_others_for(&mut self, key_type: KeyType) {
        match key_type {
            KeyType::Int64 => {
                self.seen_bytes.clear();
                self.seen_bytes.shrink_to_fit();
                self.seen_uuid.clear();
                self.seen_uuid.shrink_to_fit();
                self.seen_uuid256.clear();
                self.seen_uuid256.shrink_to_fit();
                self.seen_ip.clear();
                self.seen_ip.shrink_to_fit();
            }
            KeyType::VarBytes => {
                self.seen_u64.clear();
                self.seen_u64.shrink_to_fit();
                self.seen_uuid.clear();
                self.seen_uuid.shrink_to_fit();
                self.seen_uuid256.clear();
                self.seen_uuid256.shrink_to_fit();
                self.seen_ip.clear();
                self.seen_ip.shrink_to_fit();
            }
            KeyType::Uuid => {
                self.seen_u64.clear();
                self.seen_u64.shrink_to_fit();
                self.seen_bytes.clear();
                self.seen_bytes.shrink_to_fit();
                self.seen_uuid256.clear();
                self.seen_uuid256.shrink_to_fit();
                self.seen_ip.clear();
                self.seen_ip.shrink_to_fit();
            }
            KeyType::Uuid256 => {
                self.seen_u64.clear();
                self.seen_u64.shrink_to_fit();
                self.seen_bytes.clear();
                self.seen_bytes.shrink_to_fit();
                self.seen_uuid.clear();
                self.seen_uuid.shrink_to_fit();
                self.seen_ip.clear();
                self.seen_ip.shrink_to_fit();
            }
            KeyType::Ip => {
                self.seen_u64.clear();
                self.seen_u64.shrink_to_fit();
                self.seen_bytes.clear();
                self.seen_bytes.shrink_to_fit();
                self.seen_uuid.clear();
                self.seen_uuid.shrink_to_fit();
                self.seen_uuid256.clear();
                self.seen_uuid256.shrink_to_fit();
            }
        }
    }
}

impl Default for PkMapScratch {
    fn default() -> Self {
        Self::new()
    }
}

pub fn reserve_map<K: std::cmp::Eq + std::hash::Hash, V, S: std::hash::BuildHasher>(
    map: &mut std::collections::HashMap<K, V, S>,
    target: usize,
) {
    let cap = map.capacity();
    if cap < target {
        map.reserve(target - cap);
    }
}

pub fn estimate_pk_map_bytes(row_count: u64, key_type: KeyType, avg_var_bytes: u64) -> u64 {
    if row_count == 0 {
        return 0;
    }
    let capacity = row_count
        .saturating_mul(PK_MAP_LOAD_FACTOR_NUM)
        .saturating_add(PK_MAP_LOAD_FACTOR_DEN - 1)
        / PK_MAP_LOAD_FACTOR_DEN;
    let entry_size = match key_type {
        KeyType::Int64 => std::mem::size_of::<(u64, u32)>(),
        KeyType::VarBytes => std::mem::size_of::<(Vec<u8>, u32)>(),
        KeyType::Uuid => std::mem::size_of::<([u8; 16], u32)>(),
        KeyType::Uuid256 => std::mem::size_of::<([u8; 32], u32)>(),
        KeyType::Ip => std::mem::size_of::<([u8; 16], u32)>(),
    } as u64;
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
