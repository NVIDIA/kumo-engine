use anyhow::{anyhow, Result};
use diskgraph_core::metadata::PrimaryKeyMap;
use epserde::prelude::{Deserialize as EpDeserialize, Flags as EpFlags};
#[cfg(not(feature = "pkhash128"))]
use gxhash::GxHasher;
use memmap2::Mmap;
#[cfg(feature = "pkhash128")]
use ptr_hash::hash::GxHash128;
use ptr_hash::hash::PkHash;
use ptr_hash::DefaultPtrHash;
use std::fs::File;
use std::hash::Hasher;
use uuid_simd::{parse_hyphenated, parse_simple, AsOut};

const PK_DATA_MAGIC: &[u8; 8] = b"DGPKDAT\0";
const PK_DATA_VERSION: u32 = 3;
const PK_DATA_HEADER_LEN: usize = 8 + 4 + 1 + 4 + 8;

const GXHASH_SHARD_SEED: u64 = 0x6a09e667f3bcc909;

#[cfg(not(feature = "pkhash128"))]
type PkPtrHash = PkHash;
#[cfg(feature = "pkhash128")]
type PkPtrHash = GxHash128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyType {
    Int64,
    VarBytes,
    Uuid,
    Uuid256,
    Ip,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
        let key_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let count = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        Ok(Self {
            key_type_tag,
            key_len,
            count,
        })
    }

    fn key_type(&self) -> Result<KeyType> {
        match self.key_type_tag {
            0 => {
                self.expect_len(8, "int64")?;
                Ok(KeyType::Int64)
            }
            2 => {
                self.expect_len(0, "var_bytes")?;
                Ok(KeyType::VarBytes)
            }
            3 => {
                self.expect_len(16, "uuid")?;
                Ok(KeyType::Uuid)
            }
            4 => {
                self.expect_len(32, "uuid256")?;
                Ok(KeyType::Uuid256)
            }
            5 => {
                self.expect_len(16, "ip")?;
                Ok(KeyType::Ip)
            }
            _ => Err(anyhow!("pk data key type invalid")),
        }
    }

    fn expect_len(&self, expected: u32, name: &str) -> Result<()> {
        if self.key_len != expected {
            return Err(anyhow!(
                "pk data key length mismatch for {}: expected {}, got {}",
                name,
                expected,
                self.key_len
            ));
        }
        Ok(())
    }
}

type MphfInt =
    epserde::prelude::MemCase<epserde::prelude::DeserType<'static, DefaultPtrHash<PkPtrHash, u64>>>;
type MphfBytes = epserde::prelude::MemCase<
    epserde::prelude::DeserType<'static, DefaultPtrHash<PkPtrHash, Vec<u8>>>,
>;
type MphfUuid = epserde::prelude::MemCase<
    epserde::prelude::DeserType<'static, DefaultPtrHash<PkPtrHash, [u8; 16]>>,
>;
type MphfUuid256 = epserde::prelude::MemCase<
    epserde::prelude::DeserType<'static, DefaultPtrHash<PkPtrHash, [u8; 32]>>,
>;
type MphfIp = epserde::prelude::MemCase<
    epserde::prelude::DeserType<'static, DefaultPtrHash<PkPtrHash, [u8; 16]>>,
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
    entry_len: usize,
    lookup_fn: fn(&PkHashShard, &Key) -> Option<u32>,
}

impl PkHashShard {
    pub fn load(
        hash_path: &std::path::Path,
        data_path: &std::path::Path,
        key_type: KeyType,
    ) -> Result<Self> {
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
        let entry_len = 4;
        let lookup_fn = lookup_id_only;
        let mphf = match key_type {
            KeyType::Int64 => {
                let mphf: MphfInt =
                    DefaultPtrHash::<PkPtrHash, u64>::load_mmap(hash_path, EpFlags::SEQUENTIAL)
                        .map_err(|e| anyhow!("load pk hash {}: {e}", hash_path.display()))?;
                PkHashMphf::Int(mphf)
            }
            KeyType::VarBytes => {
                let mphf: MphfBytes =
                    DefaultPtrHash::<PkPtrHash, Vec<u8>>::load_mmap(hash_path, EpFlags::SEQUENTIAL)
                        .map_err(|e| anyhow!("load pk hash {}: {e}", hash_path.display()))?;
                PkHashMphf::Bytes(mphf)
            }
            KeyType::Uuid => {
                let mphf: MphfUuid = DefaultPtrHash::<PkPtrHash, [u8; 16]>::load_mmap(
                    hash_path,
                    EpFlags::SEQUENTIAL,
                )
                .map_err(|e| anyhow!("load pk hash {}: {e}", hash_path.display()))?;
                PkHashMphf::Uuid(mphf)
            }
            KeyType::Uuid256 => {
                let mphf: MphfUuid256 = DefaultPtrHash::<PkPtrHash, [u8; 32]>::load_mmap(
                    hash_path,
                    EpFlags::SEQUENTIAL,
                )
                .map_err(|e| anyhow!("load pk hash {}: {e}", hash_path.display()))?;
                PkHashMphf::Uuid256(mphf)
            }
            KeyType::Ip => {
                let mphf: MphfIp = DefaultPtrHash::<PkPtrHash, [u8; 16]>::load_mmap(
                    hash_path,
                    EpFlags::SEQUENTIAL,
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
            entry_len,
            lookup_fn,
        })
    }

    pub fn lookup(&self, key: &Key) -> Option<u32> {
        (self.lookup_fn)(self, key)
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

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

pub fn key_type_from_metadata(pk: &PrimaryKeyMap) -> Result<KeyType> {
    match pk.dtype.as_str() {
        "int64" | "int32" | "int16" | "int8" | "uint64" | "uint32" | "uint16" | "uint8"
        | "float64" | "float32" | "decimal128" | "decimal256" => Ok(KeyType::Int64),
        "string" | "bytes" => Ok(KeyType::VarBytes),
        "uuid" => Ok(KeyType::Uuid),
        "uuid256" => Ok(KeyType::Uuid256),
        "ip" => Ok(KeyType::Ip),
        other => Err(anyhow!("unsupported pk dtype {}", other)),
    }
}

pub fn parse_key_from_str(pk: &PrimaryKeyMap, input: &str) -> Result<Key> {
    let key_type = key_type_from_metadata(pk)?;
    match key_type {
        KeyType::Int64 => parse_int_key(input).map(Key::Int),
        KeyType::Uuid => {
            let raw = input.as_bytes();
            let mut out = [0u8; 16];
            let value = match raw.len() {
                32 => parse_simple(raw, out.as_out()).map_err(|e| anyhow!("invalid uuid: {e}"))?,
                36 => {
                    parse_hyphenated(raw, out.as_out()).map_err(|e| anyhow!("invalid uuid: {e}"))?
                }
                _ => return Err(anyhow!("invalid uuid: wrong length")),
            };
            Ok(Key::Uuid(*value))
        }
        KeyType::VarBytes => Ok(Key::Bytes(parse_bytes_key(input)?)),
        KeyType::Uuid256 => {
            let bytes = parse_uuid256_hex(input)?;
            Ok(Key::Uuid256(bytes))
        }
        KeyType::Ip => {
            let ip = parse_ip_text(input)?;
            Ok(Key::Ip(ip))
        }
    }
}

pub fn shard_for_key(key: &Key, num_shards: usize) -> usize {
    if num_shards <= 1 {
        return 0;
    }
    let mut hasher = GxHasher::with_seed(GXHASH_SHARD_SEED as i64);
    match key {
        Key::Int(v) => {
            let mut scratch = [0u8; 8];
            scratch.copy_from_slice(&v.to_le_bytes());
            hasher.write(&scratch);
        }
        Key::Uuid(v) => hasher.write(v),
        Key::Uuid256(v) => hasher.write(v),
        Key::Ip(v) => hasher.write(v),
        Key::Bytes(v) => {
            hasher.write(v);
        }
    }
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
        let hi = (chunk[0] as char)
            .to_digit(16)
            .ok_or_else(|| anyhow!("invalid hex"))?;
        let lo = (chunk[1] as char)
            .to_digit(16)
            .ok_or_else(|| anyhow!("invalid hex"))?;
        out.push(((hi << 4) | lo) as u8);
    }
    Ok(out)
}

fn parse_uuid256_hex(input: &str) -> Result<[u8; 32]> {
    let raw = input.trim();
    if raw.len() != 64 {
        return Err(anyhow!("uuid256 hex must be 64 chars"));
    }
    let bytes = parse_hex_bytes(raw)?;
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn parse_ip_text(input: &str) -> Result<[u8; 16]> {
    let ip: std::net::IpAddr = input
        .trim()
        .parse()
        .map_err(|e| anyhow!("invalid ip: {e}"))?;
    let bytes = match ip {
        std::net::IpAddr::V4(v4) => v4.to_ipv6_mapped().octets(),
        std::net::IpAddr::V6(v6) => v6.octets(),
    };
    Ok(bytes)
}

fn lookup_id_only(shard: &PkHashShard, key: &Key) -> Option<u32> {
    if key.key_type() != shard.key_type {
        return None;
    }
    let idx = shard.mphf.index(key)?;
    if idx >= shard.header.count as usize {
        return None;
    }
    let offset = PK_DATA_HEADER_LEN + idx * shard.entry_len;
    if shard.data.len() < offset + shard.entry_len {
        return None;
    }
    let value = u32::from_le_bytes(shard.data[offset..offset + 4].try_into().ok()?);
    if value == u32::MAX {
        return None;
    }
    Some(value)
}

fn advise_random(map: &Mmap) {
    let ret = unsafe {
        libc::madvise(
            map.as_ptr() as *mut libc::c_void,
            map.len(),
            libc::MADV_RANDOM,
        )
    };
    if ret != 0 {
        let _ = std::io::Error::last_os_error();
    }
}
