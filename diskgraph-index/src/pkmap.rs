use anyhow::{anyhow, Result};
use ahash::RandomState;
use diskgraph_core::metadata::PrimaryKeyMap;
use epserde::prelude::{Deserialize as EpDeserialize, Flags as EpFlags};
use memmap2::Mmap;
use ptr_hash::hash::{FxHash, Xx128};
use ptr_hash::DefaultPtrHash;
use std::fs::File;
use std::hash::{BuildHasher, Hash, Hasher};

const PK_DATA_MAGIC: &[u8; 8] = b"DGPKDAT\0";
const PK_DATA_VERSION: u32 = 1;
const PK_DATA_HEADER_LEN: usize = 8 + 4 + 1 + 1 + 4 + 8 + 8;
const PK_DATA_ENTRY_LEN: usize = 8 + 4;

const AHASH_SEED_1: u64 = 0x9e3779b185ebca87;
const AHASH_SEED_2: u64 = 0xc2b2ae3d27d4eb4f;
const AHASH_SEED_3: u64 = 0x165667b19e3779f9;
const AHASH_SEED_4: u64 = 0xd6e8feb86659fd93;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyType {
    Int64,
    FixedBytes { len: usize },
    VarBytes,
    Uuid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
}

#[derive(Debug, Clone, Copy)]
struct PkDataHeader {
    key_type_tag: u8,
    key_len: u32,
    count: u64,
    fingerprint_seed: u64,
}

impl PkDataHeader {
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
            1 => Ok(KeyType::FixedBytes {
                len: self.key_len as usize,
            }),
            2 => Ok(KeyType::VarBytes),
            3 => Ok(KeyType::Uuid),
            _ => Err(anyhow!("pk data key type invalid")),
        }
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

enum PkHashMphf {
    Int(MphfInt),
    Bytes(MphfBytes),
    Uuid(MphfUuid),
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
            (PkHashMphf::Bytes(mphf), Key::FixedBytes(bytes)) => {
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
            KeyType::FixedBytes { .. } | KeyType::VarBytes => {
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

pub struct PkHashTable {
    key_type: KeyType,
    shards: Vec<PkHashShard>,
}

impl PkHashTable {
    pub fn new(key_type: KeyType, shards: Vec<PkHashShard>) -> Self {
        Self { key_type, shards }
    }

    pub fn key_type(&self) -> KeyType {
        self.key_type
    }

    pub fn shard(&self, index: usize) -> Option<&PkHashShard> {
        self.shards.get(index)
    }
}

pub fn key_type_from_metadata(pk: &PrimaryKeyMap) -> Result<KeyType> {
    if pk.key_mode.as_deref() == Some("uuid") {
        return Ok(KeyType::Uuid);
    }
    match pk.dtype.as_str() {
        "int64"
        | "int32"
        | "int16"
        | "int8"
        | "uint64"
        | "uint32"
        | "uint16"
        | "uint8"
        | "float64"
        | "float32"
        | "decimal128"
        | "decimal256" => Ok(KeyType::Int64),
        "string" | "bytes" => {
            if let Some(len) = pk.fixed_len {
                Ok(KeyType::FixedBytes { len })
            } else {
                Ok(KeyType::VarBytes)
            }
        }
        other => Err(anyhow!("unsupported pk dtype {}", other)),
    }
}

pub fn parse_key_from_str(pk: &PrimaryKeyMap, input: &str) -> Result<Key> {
    let key_type = key_type_from_metadata(pk)?;
    match key_type {
        KeyType::Int64 => parse_int_key(input).map(Key::Int),
        KeyType::Uuid => {
            let value = uuid::Uuid::parse_str(input)
                .map_err(|e| anyhow!("invalid uuid: {e}"))?;
            Ok(Key::Uuid(*value.as_bytes()))
        }
        KeyType::FixedBytes { len } => {
            let bytes = parse_bytes_key(input)?;
            if bytes.len() != len {
                return Err(anyhow!("expected {} bytes, got {}", len, bytes.len()));
            }
            Ok(Key::FixedBytes(bytes))
        }
        KeyType::VarBytes => Ok(Key::Bytes(parse_bytes_key(input)?)),
    }
}

pub fn shard_for_key(key: &Key, num_shards: usize) -> usize {
    if num_shards <= 1 {
        return 0;
    }
    let mut scratch = [0u8; 16];
    let bytes = key_hash_bytes(key, &mut scratch);
    let mut hasher = fixed_hash_state().build_hasher();
    hasher.write(bytes);
    (hasher.finish() as usize) % num_shards
}

pub fn resolve_shard_path(template: &str, shard: usize) -> String {
    template.replace("{shard}", &format!("{:05}", shard))
}

fn parse_int_key(input: &str) -> Result<u64> {
    let raw = input.trim();
    if let Some(rest) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16).map_err(|e| anyhow!("invalid hex int: {e}"))
    } else {
        raw.parse::<u64>().map_err(|e| anyhow!("invalid int: {e}"))
    }
}

fn parse_bytes_key(input: &str) -> Result<Vec<u8>> {
    let raw = input.trim();
    if let Some(rest) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
        parse_hex_bytes(rest)
    } else {
        Ok(raw.as_bytes().to_vec())
    }
}

fn parse_hex_bytes(input: &str) -> Result<Vec<u8>> {
    if input.len() % 2 != 0 {
        return Err(anyhow!("hex input length must be even"));
    }
    let mut out = Vec::with_capacity(input.len() / 2);
    let bytes = input.as_bytes();
    for chunk in bytes.chunks(2) {
        let hi = (chunk[0] as char).to_digit(16).ok_or_else(|| anyhow!("invalid hex"))?;
        let lo = (chunk[1] as char).to_digit(16).ok_or_else(|| anyhow!("invalid hex"))?;
        out.push(((hi << 4) | lo) as u8);
    }
    Ok(out)
}

fn key_hash_bytes<'a>(key: &'a Key, scratch: &'a mut [u8; 16]) -> &'a [u8] {
    match key {
        Key::Int(v) => {
            scratch[..8].copy_from_slice(&v.to_le_bytes());
            &scratch[..8]
        }
        Key::Uuid(v) => v.as_slice(),
        Key::FixedBytes(v) => v.as_slice(),
        Key::Bytes(v) => v.as_slice(),
    }
}

fn fixed_hash_state() -> RandomState {
    RandomState::with_seeds(
        AHASH_SEED_1,
        AHASH_SEED_2,
        AHASH_SEED_3,
        AHASH_SEED_4,
    )
}

fn fingerprint_for_key(key: &Key, seed: u64) -> u64 {
    let mut hasher = twox_hash::XxHash64::with_seed(seed);
    match key {
        Key::Int(v) => v.hash(&mut hasher),
        Key::Uuid(v) => v.hash(&mut hasher),
        Key::FixedBytes(v) | Key::Bytes(v) => v.hash(&mut hasher),
    }
    hasher.finish()
}

fn advise_random(map: &Mmap) {
    let ret = unsafe { libc::madvise(map.as_ptr() as *mut libc::c_void, map.len(), libc::MADV_RANDOM) };
    if ret != 0 {
        let _ = std::io::Error::last_os_error();
    }
}
