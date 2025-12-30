use anyhow::{anyhow, Result};
use seahash::SeaHasher;
use std::cmp::Ordering;
use std::hash::Hasher;
use std::io::{self, Write};

const MAGIC: &[u8; 8] = b"DGPKMAP\0";
const VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingKind {
    Hash,
    Sorted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyType {
    Int64,
    FixedBytes { len: usize },
    VarBytes,
    Uuid,
}

#[derive(Debug, Clone)]
pub enum Key {
    Int(u64),
    FixedBytes(Vec<u8>),
    Bytes(Vec<u8>),
    Uuid([u8; 16]),
}

impl Key {
    pub fn key_type(&self) -> KeyType {
        match self {
            Key::Int(_) => KeyType::Int64,
            Key::FixedBytes(v) => KeyType::FixedBytes { len: v.len() },
            Key::Bytes(_) => KeyType::VarBytes,
            Key::Uuid(_) => KeyType::Uuid,
        }
    }

    fn as_bytes(&self) -> &[u8] {
        match self {
            Key::Int(v) => unsafe {
                std::slice::from_raw_parts(v as *const u64 as *const u8, 8)
            },
            Key::FixedBytes(v) => v.as_slice(),
            Key::Bytes(v) => v.as_slice(),
            Key::Uuid(v) => v.as_slice(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PkMap {
    kind: MappingKind,
    key_type: KeyType,
    entries: Vec<(Key, u64)>,
    hash_table: Vec<Option<(Key, u64)>>,
    capacity: usize,
    hash_len: usize,
    min_key: u64,
    max_key: u64,
    direct: Option<DirectMap>,
    identity_base: Option<u64>,
    identity_len: usize,
}

#[derive(Debug, Clone)]
struct DirectMap {
    base: u64,
    values: Vec<u64>,
}

impl PkMap {
    pub fn new(kind: MappingKind, key_type: KeyType, capacity_hint: usize) -> Self {
        let capacity = match kind {
            MappingKind::Hash => next_pow2((capacity_hint as f64 / 0.7).ceil() as usize).max(8),
            MappingKind::Sorted => 0,
        };
        Self {
            kind,
            key_type,
            entries: Vec::with_capacity(capacity_hint),
            hash_table: vec![None; capacity],
            capacity,
            hash_len: 0,
            min_key: u64::MAX,
            max_key: 0,
            direct: None,
            identity_base: None,
            identity_len: 0,
        }
    }

    pub fn insert(&mut self, key: Key, value: u64) -> Result<()> {
        if key.key_type() != self.key_type {
            return Err(anyhow!("key type mismatch"));
        }
        if let Key::Int(v) = key {
            if v < self.min_key {
                self.min_key = v;
            }
            if v > self.max_key {
                self.max_key = v;
            }
            return self.insert_inner(Key::Int(v), value);
        }
        self.insert_inner(key, value)
    }

    fn insert_inner(&mut self, key: Key, value: u64) -> Result<()> {
        match self.kind {
            MappingKind::Hash => {
                if (self.hash_len + 1) * 10 >= self.capacity * 7 {
                    self.grow_hash();
                }
                let mut idx = (hash_key(&key) as usize) & (self.capacity - 1);
                loop {
                    match &self.hash_table[idx] {
                        None => {
                            self.hash_table[idx] = Some((key, value));
                            self.hash_len += 1;
                            return Ok(());
                        }
                        Some((existing, _)) if keys_equal(existing, &key) => {
                            return Err(anyhow!("duplicate primary key"));
                        }
                        _ => {
                            idx = (idx + 1) & (self.capacity - 1);
                        }
                    }
                }
            }
            MappingKind::Sorted => {
                self.entries.push((key, value));
                Ok(())
            }
        }
    }

    pub fn finalize(&mut self) {
        if let MappingKind::Sorted = self.kind {
            self.entries.sort_by(|a, b| compare_keys(&a.0, &b.0));
        }
        if self.kind == MappingKind::Hash && self.key_type == KeyType::Int64 {
            self.try_build_direct();
        }
    }

    pub fn lookup(&self, key: &Key) -> Option<u64> {
        if key.key_type() != self.key_type {
            return None;
        }
        if let (Some(base), Key::Int(v)) = (self.identity_base, key) {
            if *v >= base {
                let idx = (*v - base) as usize;
                if idx < self.identity_len {
                    return Some(idx as u64);
                }
            }
            return None;
        }
        if let (Some(direct), Key::Int(v)) = (&self.direct, key) {
            if *v >= direct.base {
                let idx = (*v - direct.base) as usize;
                if idx < direct.values.len() {
                    return Some(direct.values[idx]);
                }
            }
            return None;
        }
        match self.kind {
            MappingKind::Hash => {
                let mut idx = (hash_key(key) as usize) & (self.capacity - 1);
                loop {
                    match &self.hash_table[idx] {
                        None => return None,
                        Some((existing, value)) if keys_equal(existing, key) => {
                            return Some(*value)
                        }
                        _ => {
                            idx = (idx + 1) & (self.capacity - 1);
                        }
                    }
                }
            }
            MappingKind::Sorted => {
                let idx = self.entries.binary_search_by(|(k, _)| compare_keys(k, key)).ok()?;
                self.entries.get(idx).map(|(_, v)| *v)
            }
        }
    }


    pub fn write_to(&self, mut writer: impl Write) -> io::Result<()> {
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&[match self.kind {
            MappingKind::Hash => 0u8,
            MappingKind::Sorted => 1u8,
        }])?;
        let (key_type_tag, key_len) = match self.key_type {
            KeyType::Int64 => (0u8, 8u32),
            KeyType::FixedBytes { len } => (1u8, len as u32),
            KeyType::VarBytes => (2u8, 0u32),
            KeyType::Uuid => (3u8, 16u32),
        };
        writer.write_all(&[key_type_tag])?;
        writer.write_all(&key_len.to_le_bytes())?;
        let count = match self.kind {
            MappingKind::Hash => self.hash_len,
            MappingKind::Sorted => self.entries.len(),
        } as u64;
        let capacity = if self.kind == MappingKind::Hash {
            self.capacity as u64
        } else {
            count
        };
        writer.write_all(&capacity.to_le_bytes())?;
        writer.write_all(&count.to_le_bytes())?;

        if self.kind == MappingKind::Hash {
            let bitmap_len = (self.capacity + 7) / 8;
            let mut bitmap = vec![0u8; bitmap_len];
            for (idx, entry) in self.hash_table.iter().enumerate() {
                if entry.is_some() {
                    bitmap[idx / 8] |= 1 << (idx % 8);
                }
            }
            writer.write_all(&bitmap)?;
            for entry in &self.hash_table {
                if let Some((key, value)) = entry {
                    if self.key_type == KeyType::VarBytes {
                        let bytes = key.as_bytes();
                        let len = bytes.len() as u32;
                        writer.write_all(&len.to_le_bytes())?;
                        writer.write_all(bytes)?;
                    } else {
                        writer.write_all(key.as_bytes())?;
                    }
                    writer.write_all(&value.to_le_bytes())?;
                } else {
                    if self.key_type == KeyType::VarBytes {
                        writer.write_all(&u32::MAX.to_le_bytes())?;
                    } else {
                        let zero = vec![0u8; key_len as usize];
                        writer.write_all(&zero)?;
                    }
                    writer.write_all(&0u64.to_le_bytes())?;
                }
            }
        } else {
            for (key, value) in &self.entries {
                if self.key_type == KeyType::VarBytes {
                    let bytes = key.as_bytes();
                    let len = bytes.len() as u32;
                    writer.write_all(&len.to_le_bytes())?;
                    writer.write_all(bytes)?;
                } else {
                    writer.write_all(key.as_bytes())?;
                }
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        Ok(())
    }

    fn grow_hash(&mut self) {
        let new_capacity = (self.capacity * 2).max(8);
        let mut new_table = vec![None; new_capacity];
        for entry in self.hash_table.drain(..) {
            if let Some((key, value)) = entry {
                let mut idx = (hash_key(&key) as usize) & (new_capacity - 1);
                loop {
                    if new_table[idx].is_none() {
                        new_table[idx] = Some((key, value));
                        break;
                    }
                    idx = (idx + 1) & (new_capacity - 1);
                }
            }
        }
        self.hash_table = new_table;
        self.capacity = new_capacity;
    }

    fn try_build_direct(&mut self) {
        if self.hash_len == 0 {
            return;
        }
        if self.min_key == u64::MAX {
            return;
        }
        let range = self.max_key.saturating_sub(self.min_key).saturating_add(1);
        if range as usize != self.hash_len {
            return;
        }
        let mut values = vec![0u64; range as usize];
        for entry in &self.hash_table {
            if let Some((Key::Int(k), value)) = entry {
                let idx = (*k - self.min_key) as usize;
                values[idx] = *value;
            }
        }
        let is_identity = values
            .iter()
            .enumerate()
            .all(|(idx, value)| *value == idx as u64);
        if is_identity {
            self.identity_base = Some(self.min_key);
            self.identity_len = values.len();
            self.direct = None;
        } else {
            self.direct = Some(DirectMap {
                base: self.min_key,
                values,
            });
        }
    }
}

fn compare_keys(a: &Key, b: &Key) -> Ordering {
    match (a, b) {
        (Key::Int(a), Key::Int(b)) => a.cmp(b),
        (Key::FixedBytes(a), Key::FixedBytes(b)) => a.cmp(b),
        (Key::Bytes(a), Key::Bytes(b)) => a.cmp(b),
        (Key::Uuid(a), Key::Uuid(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

fn keys_equal(a: &Key, b: &Key) -> bool {
    match (a, b) {
        (Key::Int(a), Key::Int(b)) => a == b,
        (Key::FixedBytes(a), Key::FixedBytes(b)) => a == b,
        (Key::Bytes(a), Key::Bytes(b)) => a == b,
        (Key::Uuid(a), Key::Uuid(b)) => a == b,
        _ => false,
    }
}

fn hash_key(key: &Key) -> u64 {
    match key {
        Key::Int(v) => hash_int(*v),
        Key::FixedBytes(v) | Key::Bytes(v) => {
            let mut hasher = SeaHasher::new();
            hasher.write(v);
            hasher.finish()
        }
        Key::Uuid(v) => {
            let hi = u64::from_be_bytes(v[0..8].try_into().unwrap());
            let lo = u64::from_be_bytes(v[8..16].try_into().unwrap());
            hash_int(hi) ^ hash_int(lo)
        }
    }
}

fn hash_int(mut v: u64) -> u64 {
    v = v.wrapping_add(0x9E3779B97F4A7C15);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D049BB133111EB);
    v ^ (v >> 31)
}

fn next_pow2(value: usize) -> usize {
    value.next_power_of_two()
}
