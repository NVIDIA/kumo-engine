use anyhow::{anyhow, Result};
use memmap2::Mmap;
use ptr_hash::hash::{FxHash, Xx128};
use ptr_hash::DefaultPtrHash;
use std::fs::File;
use std::hash::Hasher;
use epserde::prelude::{Deserialize as EpDeserialize, Flags as EpFlags};

const PK_DATA_MAGIC: &[u8; 8] = b"DGPKDAT\0";
const PK_DATA_VERSION: u32 = 1;
const PK_DATA_HEADER_LEN: usize = 8 + 4 + 1 + 1 + 4 + 8 + 8;
const PK_DATA_ENTRY_LEN: usize = 8 + 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyType {
    Int64,
    VarBytes,
    Uuid,
    Uuid256,
    Ip,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    Int(u64),
    Bytes(Vec<u8>),
    Uuid([u8; 16]),
    Uuid256([u8; 32]),
    Ip([u8; 16]),
}

impl Key {
    pub fn key_type(&self) -> KeyType {
        match self {
            Key::Int(_) => KeyType::Int64,
            Key::Bytes(_) => KeyType::VarBytes,
            Key::Uuid(_) => KeyType::Uuid,
            Key::Uuid256(_) => KeyType::Uuid256,
            Key::Ip(_) => KeyType::Ip,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PkDataHeader {
    key_type_tag: u8,
    key_len: u32,
    count: u64,
    fingerprint_seed: u64,
}

impl PkDataHeader {
    fn write_to(&self, mut out: impl std::io::Write) -> std::io::Result<()> {
        out.write_all(PK_DATA_MAGIC)?;
        out.write_all(&PK_DATA_VERSION.to_le_bytes())?;
        out.write_all(&[self.key_type_tag])?;
        out.write_all(&[0u8])?;
        out.write_all(&self.key_len.to_le_bytes())?;
        out.write_all(&self.count.to_le_bytes())?;
        out.write_all(&self.fingerprint_seed.to_le_bytes())?;
        Ok(())
    }

    fn read_from(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < PK_DATA_HEADER_LEN {
            return Err(anyhow!("pk data header truncated"));
        }
        let mut offset = 0usize;
        if &bytes[offset..offset + PK_DATA_MAGIC.len()] != PK_DATA_MAGIC {
            return Err(anyhow!("pk data magic mismatch"));
        }
        offset += PK_DATA_MAGIC.len();
        let version = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        if version != PK_DATA_VERSION {
            return Err(anyhow!("pk data version mismatch"));
        }
        let key_type_tag = bytes[offset];
        offset += 1;
        offset += 1;
        let key_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let count = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        let fingerprint_seed = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        Ok(Self {
            key_type_tag,
            key_len,
            count,
            fingerprint_seed,
        })
    }

    fn key_type(&self) -> Result<KeyType> {
        match self.key_type_tag {
            0 => Ok(KeyType::Int64),
            2 => Ok(KeyType::VarBytes),
            3 => Ok(KeyType::Uuid),
            4 => Ok(KeyType::Uuid256),
            5 => Ok(KeyType::Ip),
            _ => Err(anyhow!("pk data key type invalid")),
        }
    }
}



#[derive(Clone, epserde::prelude::Epserde)]
pub struct PkXxHash;

impl PkXxHash {
    #[inline(always)]
    pub(crate) fn hash<Key: std::hash::Hash + ?Sized>(x: &Key, seed: u64) -> u64 {
        let mut hasher = twox_hash::XxHash64::with_seed(seed);
        x.hash(&mut hasher);
        hasher.finish()
    }
}

type MphfInt = epserde::prelude::MemCase<
    epserde::prelude::DeserType<'static, DefaultPtrHash<FxHash, u64>>,
>;
type MphfBytes = epserde::prelude::MemCase<
    epserde::prelude::DeserType<'static, DefaultPtrHash<Xx128, Vec<u8>>>,
>;
type MphfUuid = epserde::prelude::MemCase<
    epserde::prelude::DeserType<'static, DefaultPtrHash<Xx128, [u8; 16]>>,
>;
type MphfUuid256 = epserde::prelude::MemCase<
    epserde::prelude::DeserType<'static, DefaultPtrHash<Xx128, [u8; 32]>>,
>;
type MphfIp = epserde::prelude::MemCase<
    epserde::prelude::DeserType<'static, DefaultPtrHash<Xx128, [u8; 16]>>,
>;

enum PkHashMphf {
    Int(MphfInt),
    Bytes(MphfBytes),
    Uuid(MphfUuid),
    Uuid256(MphfUuid256),
    Ip(MphfIp),
}

impl PkHashMphf {
    fn index(&self, key: &Key) -> Option<usize> {
        match (self, key) {
            (PkHashMphf::Int(mphf), Key::Int(value)) => {
                let slot = mphf.index_no_remap(value);
                if slot >= mphf.max_index() {
                    None
                } else {
                    Some(slot)
                }
            }
            (PkHashMphf::Bytes(mphf), Key::Bytes(bytes)) => {
                let slot = mphf.index_no_remap(bytes);
                if slot >= mphf.max_index() {
                    None
                } else {
                    Some(slot)
                }
            }
            (PkHashMphf::Uuid(mphf), Key::Uuid(value)) => {
                let slot = mphf.index_no_remap(value);
                if slot >= mphf.max_index() {
                    None
                } else {
                    Some(slot)
                }
            }
            (PkHashMphf::Uuid256(mphf), Key::Uuid256(value)) => {
                let slot = mphf.index_no_remap(value);
                if slot >= mphf.max_index() {
                    None
                } else {
                    Some(slot)
                }
            }
            (PkHashMphf::Ip(mphf), Key::Ip(value)) => {
                let slot = mphf.index_no_remap(value);
                if slot >= mphf.max_index() {
                    None
                } else {
                    Some(slot)
                }
            }
            _ => None,
        }
    }
}

pub struct PkHashShard {
    key_type: KeyType,
    mphf: PkHashMphf,
    data: Mmap,
    header: PkDataHeader,
}

impl PkHashShard {
    pub fn load(hash_path: &std::path::Path, data_path: &std::path::Path, key_type: KeyType) -> Result<Self> {
        let data_file = File::open(data_path)
            .map_err(|e| anyhow!("open pk data {}: {e}", data_path.display()))?;
        let data = unsafe { Mmap::map(&data_file)? };
        advise_random(&data);
        let header = PkDataHeader::read_from(&data)?;
        let header_key_type = header.key_type()?;
        if header_key_type != key_type {
            return Err(anyhow!(
                "pk data key type mismatch: expected {:?}, found {:?}",
                key_type,
                header_key_type
            ));
        }
        let mphf = match key_type {
            KeyType::Int64 => {
                let mphf: MphfInt = DefaultPtrHash::<FxHash, u64>::load_mmap(
                    hash_path,
                    EpFlags::RANDOM_ACCESS | EpFlags::TRANSPARENT_HUGE_PAGES,
                )
                .map_err(|e| anyhow!("load pk hash {}: {e}", hash_path.display()))?;
                PkHashMphf::Int(mphf)
            }
            KeyType::VarBytes => {
                let mphf: MphfBytes = DefaultPtrHash::<Xx128, Vec<u8>>::load_mmap(
                    hash_path,
                    EpFlags::RANDOM_ACCESS | EpFlags::TRANSPARENT_HUGE_PAGES,
                )
                .map_err(|e| anyhow!("load pk hash {}: {e}", hash_path.display()))?;
                PkHashMphf::Bytes(mphf)
            }
            KeyType::Uuid => {
                let mphf: MphfUuid = DefaultPtrHash::<Xx128, [u8; 16]>::load_mmap(
                    hash_path,
                    EpFlags::RANDOM_ACCESS | EpFlags::TRANSPARENT_HUGE_PAGES,
                )
                .map_err(|e| anyhow!("load pk hash {}: {e}", hash_path.display()))?;
                PkHashMphf::Uuid(mphf)
            }
            KeyType::Uuid256 => {
                let mphf: MphfUuid256 = DefaultPtrHash::<Xx128, [u8; 32]>::load_mmap(
                    hash_path,
                    EpFlags::RANDOM_ACCESS | EpFlags::TRANSPARENT_HUGE_PAGES,
                )
                .map_err(|e| anyhow!("load pk hash {}: {e}", hash_path.display()))?;
                PkHashMphf::Uuid256(mphf)
            }
            KeyType::Ip => {
                let mphf: MphfIp = DefaultPtrHash::<Xx128, [u8; 16]>::load_mmap(
                    hash_path,
                    EpFlags::RANDOM_ACCESS | EpFlags::TRANSPARENT_HUGE_PAGES,
                )
                .map_err(|e| anyhow!("load pk hash {}: {e}", hash_path.display()))?;
                PkHashMphf::Ip(mphf)
            }
        };
        Ok(Self {
            key_type,
            mphf,
            data,
            header,
        })
    }

    pub fn lookup(&self, key: &Key) -> Option<u32> {
        if key.key_type() != self.key_type {
            return None;
        }
        let idx = self.mphf.index(key)?;
        if idx >= self.header.count as usize {
            return None;
        }
        let offset = PK_DATA_HEADER_LEN + idx * PK_DATA_ENTRY_LEN;
        if self.data.len() < offset + PK_DATA_ENTRY_LEN {
            return None;
        }
        let fp = u64::from_le_bytes(self.data[offset..offset + 8].try_into().ok()?);
        let value = u32::from_le_bytes(
            self.data[offset + 8..offset + 12]
                .try_into()
                .ok()?,
        );
        if fp != fingerprint_for_key(key, self.header.fingerprint_seed) {
            return None;
        }
        Some(value)
    }
}

pub struct PkHashStore {
    tables: std::collections::HashMap<String, Vec<PkHashShard>>,
}

impl PkHashStore {
    pub fn new(tables: std::collections::HashMap<String, Vec<PkHashShard>>) -> Self {
        Self { tables }
    }

    pub fn shard(&self, table: &str, shard: usize) -> Result<&PkHashShard> {
        let shards = self
            .tables
            .get(table)
            .ok_or_else(|| anyhow!("missing pk hash table {}", table))?;
        shards
            .get(shard)
            .ok_or_else(|| anyhow!("missing pk hash shard {}:{}", table, shard))
    }
}

pub fn key_type_tag(key_type: KeyType) -> (u8, u32) {
    match key_type {
        KeyType::Int64 => (0, 8),
        KeyType::VarBytes => (2, 0),
        KeyType::Uuid => (3, 16),
        KeyType::Uuid256 => (4, 32),
        KeyType::Ip => (5, 16),
    }
}

pub fn write_pk_data_header(
    mut out: impl std::io::Write,
    key_type: KeyType,
    count: u64,
    fingerprint_seed: u64,
) -> std::io::Result<()> {
    let (key_type_tag, key_len) = key_type_tag(key_type);
    let header = PkDataHeader {
        key_type_tag,
        key_len,
        count,
        fingerprint_seed,
    };
    header.write_to(&mut out)
}

pub fn fingerprint_for_key(key: &Key, seed: u64) -> u64 {
    match key {
        Key::Int(value) => PkXxHash::hash(value, seed),
        Key::Bytes(bytes) => PkXxHash::hash(bytes, seed),
        Key::Uuid(value) => PkXxHash::hash(value, seed),
        Key::Uuid256(value) => PkXxHash::hash(value, seed),
        Key::Ip(value) => PkXxHash::hash(value, seed),
    }
}

pub fn fingerprint_seed_for(table: &str, shard: usize) -> u64 {
    let mut hasher = twox_hash::XxHash64::with_seed(0);
    hasher.write(table.as_bytes());
    hasher.write_u64(shard as u64);
    hasher.finish()
}

pub fn write_pk_data_entry(
    mut out: impl std::io::Write,
    fingerprint: u64,
    value: u32,
) -> std::io::Result<()> {
    out.write_all(&fingerprint.to_le_bytes())?;
    out.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn advise_random(map: &Mmap) {
    let ret = unsafe { libc::madvise(map.as_ptr() as *mut libc::c_void, map.len(), libc::MADV_WILLNEED) };
    if ret != 0 {
        let _ = std::io::Error::last_os_error();
    }
}
