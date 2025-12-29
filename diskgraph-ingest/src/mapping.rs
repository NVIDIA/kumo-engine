use anyhow::{anyhow, Result};
use seahash::SeaHasher;
use std::cmp::Ordering;
use std::hash::Hasher;
use std::io::{self, Write};

const MAGIC: &[u8; 8] = b"DGPKMAP\0";
const VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingKind {
    Hash,
    Sorted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyType {
    Int64,
    FixedBytes { len: usize },
}

#[derive(Debug, Clone)]
pub enum Key {
    Int(u64),
    Bytes(Vec<u8>),
}

impl Key {
    pub fn key_type(&self) -> KeyType {
        match self {
            Key::Int(_) => KeyType::Int64,
            Key::Bytes(v) => KeyType::FixedBytes { len: v.len() },
        }
    }

    fn as_bytes(&self) -> &[u8] {
        match self {
            Key::Int(v) => unsafe {
                std::slice::from_raw_parts(v as *const u64 as *const u8, 8)
            },
            Key::Bytes(v) => v.as_slice(),
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
        }
    }

    pub fn insert(&mut self, key: Key, value: u64) -> Result<()> {
        if key.key_type() != self.key_type {
            return Err(anyhow!("key type mismatch"));
        }
        match self.kind {
            MappingKind::Hash => {
                let mut idx = (hash_key(&key) as usize) & (self.capacity - 1);
                loop {
                    match &self.hash_table[idx] {
                        None => {
                            self.hash_table[idx] = Some((key, value));
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
    }

    pub fn lookup(&self, key: &Key) -> Option<u64> {
        if key.key_type() != self.key_type {
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
        };
        writer.write_all(&[key_type_tag])?;
        writer.write_all(&key_len.to_le_bytes())?;
        let count = match self.kind {
            MappingKind::Hash => self.hash_table.iter().filter(|e| e.is_some()).count(),
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
                    writer.write_all(key.as_bytes())?;
                    writer.write_all(&value.to_le_bytes())?;
                } else {
                    let zero = vec![0u8; key_len as usize];
                    writer.write_all(&zero)?;
                    writer.write_all(&0u64.to_le_bytes())?;
                }
            }
        } else {
            for (key, value) in &self.entries {
                writer.write_all(key.as_bytes())?;
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        Ok(())
    }
}

fn compare_keys(a: &Key, b: &Key) -> Ordering {
    match (a, b) {
        (Key::Int(a), Key::Int(b)) => a.cmp(b),
        (Key::Bytes(a), Key::Bytes(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

fn keys_equal(a: &Key, b: &Key) -> bool {
    match (a, b) {
        (Key::Int(a), Key::Int(b)) => a == b,
        (Key::Bytes(a), Key::Bytes(b)) => a == b,
        _ => false,
    }
}

fn hash_key(key: &Key) -> u64 {
    match key {
        Key::Int(v) => seahash::hash(&v.to_le_bytes()),
        Key::Bytes(v) => {
            let mut hasher = SeaHasher::new();
            hasher.write(v);
            hasher.finish()
        }
    }
}

fn next_pow2(value: usize) -> usize {
    value.next_power_of_two()
}
