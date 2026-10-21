// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use epserde::prelude::{Deserialize as EpDeserialize, Flags as EpFlags};
use memmap2::Mmap;
#[cfg(feature = "pkhash128")]
use ptr_hash::hash::GxHash128;
#[cfg(not(feature = "pkhash128"))]
use ptr_hash::hash::PkHash;
use ptr_hash::hash::Hasher as PtrHasher;
use ptr_hash::DefaultPtrHash;
use std::fs::File;

const PK_DATA_MAGIC: &[u8; 8] = b"DGPKDAT\0";
const PK_DATA_VERSION: u32 = 3;
const PK_DATA_HEADER_LEN: usize = 8 + 4 + 1 + 4 + 8;

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
}

impl PkDataHeader {
    fn write_to(&self, mut out: impl std::io::Write) -> std::io::Result<()> {
        out.write_all(PK_DATA_MAGIC)?;
        out.write_all(&PK_DATA_VERSION.to_le_bytes())?;
        out.write_all(&[self.key_type_tag])?;
        out.write_all(&self.key_len.to_le_bytes())?;
        out.write_all(&self.count.to_le_bytes())?;
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
            0 => Ok(KeyType::Int64),
            2 => Ok(KeyType::VarBytes),
            3 => Ok(KeyType::Uuid),
            4 => Ok(KeyType::Uuid256),
            5 => Ok(KeyType::Ip),
            _ => Err(anyhow!("pk data key type invalid")),
        }
    }
}

#[cfg(not(feature = "pkhash128"))]
pub type PkPtrHash = PkHash;
#[cfg(feature = "pkhash128")]
pub type PkPtrHash = GxHash128;
pub type PkPtrHashOutput = <PkPtrHash as PtrHasher<u64>>::H;

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

#[derive(Clone, Copy, Debug)]
pub enum PkDataAdvice {
    WillNeed,
    Random,
}

impl PkHashShard {
    pub fn load(
        hash_path: &std::path::Path,
        data_path: &std::path::Path,
        key_type: KeyType,
        data_advice: PkDataAdvice,
    ) -> Result<Self> {
        let data_file = File::open(data_path)
            .map_err(|e| anyhow!("open pk data {}: {e}", data_path.display()))?;
        let data = unsafe { Mmap::map(&data_file)? };
        advise_data(&data, data_advice);
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

    /// Shard count for `table`, or `None` if `table` is not loaded. Used by
    /// the edge phase to route external-parent FK keys with the parent's
    /// shard count (which can differ from the local ingest's num_shards —
    /// e.g. a 1-shard seed-table extension against a multi-shard parent).
    pub fn table_shard_count(&self, table: &str) -> Option<usize> {
        self.tables.get(table).map(|v| v.len())
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
) -> std::io::Result<()> {
    let (key_type_tag, key_len) = key_type_tag(key_type);
    let header = PkDataHeader {
        key_type_tag,
        key_len,
        count,
    };
    header.write_to(&mut out)
}

pub fn write_pk_data_entry(mut out: impl std::io::Write, value: u32) -> std::io::Result<()> {
    out.write_all(&value.to_le_bytes())?;
    Ok(())
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

#[cfg(unix)]
fn advise_willneed(map: &Mmap) {
    let ret = unsafe {
        libc::madvise(
            map.as_ptr() as *mut libc::c_void,
            map.len(),
            libc::MADV_WILLNEED,
        )
    };
    if ret != 0 {
        let _ = std::io::Error::last_os_error();
    }
}

#[cfg(unix)]
fn advise_data(map: &Mmap, advice: PkDataAdvice) {
    match advice {
        PkDataAdvice::WillNeed => advise_willneed(map),
        PkDataAdvice::Random => {
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
    }
}

// Memory advice is an optional optimization, not a mapping requirement.
#[cfg(windows)]
fn advise_willneed(_map: &Mmap) {}

#[cfg(windows)]
fn advise_data(_map: &Mmap, _advice: PkDataAdvice) {}

// ---------------------------------------------------------------------------
// External pkmap loader (Phase 1, subtask #2).
//
// Loads a parent DG index's pkmap shards for one table into the same
// `PkHashShard` / `PkHashStore` shape consumed by the resolution phase. Used
// when a seed table's reference columns resolve into a separate, already-built
// parent index (`RefResolver`) — the parent's pkmap is mmap'd without
// rebuilding.
//
// The loader is intentionally minimal: it does not parallelize across shards
// (today's caller loads one parent table per ingest, and shard count is
// bounded by the parent's `num_shards`). If multi-table resolution becomes
// common, this can be parallelized using the same `WorkItem` pattern as
// `build_pk_hash_store_parallel`.
// ---------------------------------------------------------------------------

/// Map a PK column dtype (as stored in `index_metadata.json` /
/// `PrimaryKeyMap.dtype`) to a `KeyType`. Mirrors the IngestedType-based
/// helper used during fresh ingest (`key_type_for_dtype` in lib.rs) but
/// operates on the lowercase dtype string written by the index publisher.
pub fn key_type_from_pk_dtype(dtype: &str) -> Result<KeyType> {
    match dtype {
        "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
        | "float32" | "float64" | "decimal128" => Ok(KeyType::Int64),
        "decimal256" => Ok(KeyType::VarBytes),
        "string" | "bytes" => Ok(KeyType::VarBytes),
        "uuid" => Ok(KeyType::Uuid),
        "uuid256" => Ok(KeyType::Uuid256),
        "ip" => Ok(KeyType::Ip),
        other => Err(anyhow!(
            "unsupported PK dtype `{other}` in parent index metadata"
        )),
    }
}

/// Read `{parent_index_path}/index_metadata.json` and extract the PK dtype
/// for `dst_table`. Returns the parsed `KeyType`.
///
/// Fails loudly when the parent index has no metadata file, the table is
/// missing, or the table has no `primary_key_map`. These are the three
/// scenarios that would silently break a downstream loader call.
pub fn key_type_for_external_parent(
    parent_index_path: &std::path::Path,
    dst_table: &str,
) -> Result<KeyType> {
    use diskgraph_core::metadata::DatasetMetadata;

    let metadata_path = parent_index_path.join("index_metadata.json");
    let bytes = std::fs::read(&metadata_path).map_err(|e| {
        anyhow!(
            "read parent index metadata {}: {e}",
            metadata_path.display()
        )
    })?;
    let meta: DatasetMetadata = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow!("parse parent index metadata {}: {e}", metadata_path.display()))?;
    let node = meta
        .node_types
        .iter()
        .find(|n| n.name == dst_table)
        .ok_or_else(|| {
            anyhow!(
                "external FK dst_table `{dst_table}` not found in parent index at {}",
                parent_index_path.display()
            )
        })?;
    let pk = node.primary_key_map.as_ref().ok_or_else(|| {
        anyhow!(
            "external FK dst_table `{dst_table}` has no primary_key_map in parent index"
        )
    })?;
    key_type_from_pk_dtype(&pk.dtype)
}

/// Discover the `pk_hash-NNNNN.bin` / `pk_data-NNNNN.bin` shard pairs under
/// `{parent_index_path}/nodes/{dst_table}/` and load each into a
/// `PkHashShard`. Returns the shards in shard-index order (0, 1, 2, …).
///
/// On-disk format: shards are written by `dest_node_pkmap_paths` (see
/// `build_pk_hash_store_parallel`) with five-digit zero-padded shard indices.
/// Missing data files or out-of-range shard indices fail loudly so caller
/// sees a clear "stale/incomplete parent" signal rather than silently
/// returning fewer shards than expected.
pub fn load_external_pk_hash_shards(
    parent_index_path: &std::path::Path,
    dst_table: &str,
    key_type: KeyType,
    data_advice: PkDataAdvice,
) -> Result<Vec<PkHashShard>> {
    // Node directories are written lowercased (see `graph_config` table-name
    // normalization), while the FK `dst_table` carries the case-preserved
    // logical node-type name from the parent index metadata. Lowercase it here
    // so a mixed-case table (e.g. `test_TRANS`) resolves to its on-disk
    // `nodes/test_trans` dir instead of failing to find `nodes/test_TRANS`.
    let table_dir = parent_index_path.join("nodes").join(dst_table.to_lowercase());
    if !table_dir.is_dir() {
        return Err(anyhow!(
            "external FK dst_table dir not found: {}",
            table_dir.display()
        ));
    }
    // Collect (shard_idx, hash_path, data_path) triples by scanning the dir.
    let mut entries: Vec<(usize, std::path::PathBuf, std::path::PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(&table_dir)
        .map_err(|e| anyhow!("read dir {}: {e}", table_dir.display()))?
    {
        let entry = entry.map_err(|e| anyhow!("read dir entry {}: {e}", table_dir.display()))?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let Some(shard_suffix) = name_str
            .strip_prefix("pk_hash-")
            .and_then(|s| s.strip_suffix(".bin"))
        else {
            continue;
        };
        let shard_idx: usize = shard_suffix.parse().map_err(|e| {
            anyhow!(
                "unparseable shard index in `{}`: {e}",
                entry.path().display()
            )
        })?;
        let hash_path = entry.path();
        let data_path = table_dir.join(format!("pk_data-{:05}.bin", shard_idx));
        if !data_path.is_file() {
            return Err(anyhow!(
                "external FK shard {} has pk_hash but missing pk_data at {}",
                shard_idx,
                data_path.display()
            ));
        }
        entries.push((shard_idx, hash_path, data_path));
    }
    if entries.is_empty() {
        return Err(anyhow!(
            "no pk_hash-*.bin shards found under {}",
            table_dir.display()
        ));
    }
    entries.sort_by_key(|(idx, _, _)| *idx);
    // Verify shards are contiguous from 0..N — gaps mean stale/incomplete
    // parent or a publishing bug, both of which we want loud about.
    for (expected, (idx, _, _)) in entries.iter().enumerate() {
        if *idx != expected {
            return Err(anyhow!(
                "external FK shards non-contiguous under {}: expected shard {} but found shard {}",
                table_dir.display(),
                expected,
                idx
            ));
        }
    }
    let mut shards = Vec::with_capacity(entries.len());
    for (_, hash_path, data_path) in entries {
        let shard = PkHashShard::load(&hash_path, &data_path, key_type, data_advice)?;
        shards.push(shard);
    }
    Ok(shards)
}

/// Per-dst-table pkmap + global-id offsets, loaded ONCE before the encode
/// phase so the node-writing workers can resolve PK reference values to graph
/// node-ids inline (FK-free LP resolution mode, see
/// `ttng-lp-fkfree-resolution-design.md` §S3). This is the encode-phase
/// analogue of the edges_map FK resolve loop (`dst_maps.get(shard).lookup` +
/// `dst_offsets[shard] + dst_local`), minus the forward-batcher push: it
/// returns the global node-id instead of materializing an edge.
///
/// `PkHashShard::lookup` is `&self`, so a shared `Arc<RefResolver>` can be
/// read concurrently by every encode worker.
pub struct RefResolver {
    /// dst_table -> loaded pkmap shards (shard-index order).
    shards: std::collections::HashMap<String, Vec<PkHashShard>>,
    /// dst_table -> per-shard global-id base offset (parent's
    /// `node_types[].shards[].min_id`), shard-index order.
    offsets: std::collections::HashMap<String, Vec<u64>>,
    /// Membership verifier over the same parent tables — the pkmap lookup is
    /// a blind MPH, so resolution must reject false-positive hits for keys
    /// absent from the parent (cross-graph TT reuse).
    verifier: PkMembershipVerifier,
}

/// Outcome of a membership-verified resolution ([`RefResolver::resolve_verified_in_shard`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveVerified {
    /// Resolved to a parent global node-id (verified, or parent table not
    /// verifiable — in which case the pkmap hit is trusted, never over-drop).
    Resolved(u64),
    /// Genuine pkmap miss (empty/tombstoned slot).
    Miss,
    /// MPH false positive: the slot's node stores a different PK — the key is
    /// an orphan that does not exist in the parent table.
    Orphan,
}

impl RefResolver {
    /// Load the pkmap shards + global-id offsets for every distinct
    /// `dst_table` referenced by the resolution columns, from the parent
    /// index at `parent_index_path`.
    pub fn load(
        parent_index_path: &std::path::Path,
        dst_tables: &[String],
    ) -> Result<Self> {
        use diskgraph_core::metadata::DatasetMetadata;

        let meta_path = parent_index_path.join("index_metadata.json");
        let bytes = std::fs::read(&meta_path)
            .map_err(|e| anyhow!("RefResolver: read parent metadata {}: {e}", meta_path.display()))?;
        let meta: DatasetMetadata = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow!("RefResolver: parse parent metadata {}: {e}", meta_path.display()))?;

        let mut shards = std::collections::HashMap::new();
        let mut offsets = std::collections::HashMap::new();
        let mut seen = std::collections::HashSet::new();
        for dst_table in dst_tables {
            if !seen.insert(dst_table.clone()) {
                continue;
            }
            let key_type = key_type_for_external_parent(parent_index_path, dst_table)?;
            let loaded = load_external_pk_hash_shards(
                parent_index_path,
                dst_table,
                key_type,
                PkDataAdvice::Random,
            )?;
            let node = meta
                .node_types
                .iter()
                .find(|n| n.name == *dst_table)
                .ok_or_else(|| {
                    anyhow!(
                        "RefResolver: dst_table `{dst_table}` not in parent index metadata at {}",
                        parent_index_path.display()
                    )
                })?;
            let table_offsets: Vec<u64> = node.shards.iter().map(|s| s.min_id).collect();
            if table_offsets.len() != loaded.len() {
                return Err(anyhow!(
                    "RefResolver: dst_table `{dst_table}` shard offset count {} != pkmap shard count {}",
                    table_offsets.len(),
                    loaded.len()
                ));
            }
            shards.insert(dst_table.clone(), loaded);
            offsets.insert(dst_table.clone(), table_offsets);
        }
        let verifier = PkMembershipVerifier::open(parent_index_path, dst_tables)?;
        Ok(Self {
            shards,
            offsets,
            verifier,
        })
    }

    /// Shard count for `dst_table` (needed to compute the destination shard
    /// from a key). Returns `None` if the table was not loaded.
    pub fn shard_count(&self, dst_table: &str) -> Option<usize> {
        self.shards.get(dst_table).map(|v| v.len())
    }

    /// Resolve `key` (a parent PK reference value) to the parent's GLOBAL
    /// node-id, given the destination `shard` (computed by the caller from the
    /// key + `shard_count`). Returns `None` on a pkmap miss (orphan) — the
    /// caller decides whether to mask or skip. Mirrors the edges_map math
    /// `dst_offsets[shard] + dst_local`.
    pub fn resolve_in_shard(&self, dst_table: &str, shard: usize, key: &Key) -> Option<u64> {
        let dst_local = self.shards.get(dst_table)?.get(shard)?.lookup(key)?;
        let base = self
            .offsets
            .get(dst_table)?
            .get(shard)
            .copied()
            .unwrap_or(0);
        Some(base.saturating_add(u64::from(dst_local)))
    }

    /// [`Self::resolve_in_shard`] plus membership verification: a pkmap hit
    /// whose parent node stores a different PK is reported as
    /// [`ResolveVerified::Orphan`] instead of a silently wrong node-id.
    pub fn resolve_verified_in_shard(
        &self,
        dst_table: &str,
        shard: usize,
        key: &Key,
    ) -> Result<ResolveVerified> {
        let Some(node_id) = self.resolve_in_shard(dst_table, shard, key) else {
            return Ok(ResolveVerified::Miss);
        };
        match self.verifier.verify(dst_table, node_id, key)? {
            Some(false) => Ok(ResolveVerified::Orphan),
            _ => Ok(ResolveVerified::Resolved(node_id)),
        }
    }
}

/// Membership verifier for external-parent FK / resolution-mode lookups.
///
/// The pkmap used at ingest is a minimal-perfect-hash with no membership
/// verification ([`lookup_id_only`]): a key ABSENT from the parent table does
/// not miss — it collides onto an arbitrary in-range parent node. When a
/// staging bundle is re-ingested into a graph other than the one it was
/// generated against (cross-graph TT reuse), that silently attaches seeds to
/// wrong parent nodes. This verifier re-reads the resolved parent node's
/// stored primary key (rowpack node data) and confirms it equals the
/// reference value, turning MPH false positives into countable misses.
///
/// `TableReader` mmap reads are `Send + Sync`, so one verifier is shared
/// read-only across all edge/encode workers.
pub struct PkMembershipVerifier {
    tables: std::collections::HashMap<String, std::sync::Arc<diskgraph_index::graph::TableReader>>,
}

impl PkMembershipVerifier {
    /// Open the parent index at `parent_index_path` and keep a handle for
    /// each distinct table in `dst_tables`. Only those tables are opened —
    /// a full open would mmap every parent table × shard and load every pk
    /// hash for nothing on a wide production graph.
    pub fn open(parent_index_path: &std::path::Path, dst_tables: &[String]) -> Result<Self> {
        let metadata_path = parent_index_path.join("index_metadata.json");
        let only: std::collections::HashSet<String> = dst_tables.iter().cloned().collect();
        let reader = diskgraph_index::graph::GraphIndexReader::open_subset(
            &metadata_path,
            parent_index_path,
            &only,
        )
        .map_err(|e| {
            anyhow!(
                "membership verifier: open parent index {}: {e}",
                parent_index_path.display()
            )
        })?;
        let mut tables = std::collections::HashMap::new();
        for dst_table in dst_tables {
            if tables.contains_key(dst_table) {
                continue;
            }
            let table = reader.table(dst_table).ok_or_else(|| {
                anyhow!(
                    "membership verifier: dst_table `{dst_table}` not in parent index at {}",
                    parent_index_path.display()
                )
            })?;
            tables.insert(dst_table.clone(), table);
        }
        Ok(Self { tables })
    }

    /// Verify that the parent node `global_node_id` (as resolved through the
    /// pkmap) really stores `key` as its primary key.
    ///
    /// Returns `Ok(None)` when the parent table cannot be verified (no
    /// primary-key map / rowpack node data / decoder) — callers must treat
    /// that as "unknown" and KEEP the edge, never over-drop. `Ok(Some(false))`
    /// means the key is an orphan that falsely collided onto this node.
    pub fn verify(
        &self,
        dst_table: &str,
        global_node_id: u64,
        key: &Key,
    ) -> Result<Option<bool>> {
        let table = self
            .tables
            .get(dst_table)
            .ok_or_else(|| anyhow!("membership verifier: table `{dst_table}` not loaded"))?;
        table.validate_pk_at_node(global_node_id, &to_index_key(key))
    }
}

/// The ingest-side [`Key`] and `diskgraph_index::pkmap::Key` are structurally
/// identical but distinct types; bridge them for verification.
fn to_index_key(key: &Key) -> diskgraph_index::pkmap::Key {
    use diskgraph_index::pkmap::Key as IndexKey;
    match key {
        Key::Int(v) => IndexKey::Int(*v),
        Key::Bytes(b) => IndexKey::Bytes(b.clone()),
        Key::Uuid(b) => IndexKey::Uuid(*b),
        Key::Uuid256(b) => IndexKey::Uuid256(*b),
        Key::Ip(b) => IndexKey::Ip(*b),
    }
}

/// Build a `PkHashStore` containing a single external table, suitable for
/// merging into an existing `PkHashStore` or constructing a standalone one
/// for unit-test inspection.
pub fn load_external_pk_hash_table(
    parent_index_path: &std::path::Path,
    dst_table: &str,
    data_advice: PkDataAdvice,
) -> Result<(PkHashStore, KeyType)> {
    let key_type = key_type_for_external_parent(parent_index_path, dst_table)?;
    let shards = load_external_pk_hash_shards(parent_index_path, dst_table, key_type, data_advice)?;
    let mut tables = std::collections::HashMap::new();
    tables.insert(dst_table.to_string(), shards);
    Ok((PkHashStore::new(tables), key_type))
}
