# Disk Manager

> **Related docs**: Ingest pipeline in [`README.md`](./README.md) | Design decisions in [`docs/agent_design_and_performance.md`](../docs/agent_design_and_performance.md) | Col stats in [`col_stats.md`](./col_stats.md)

Manages local SSD space for diskgraph indexes. Before each ingest, the disk manager scans existing indexes on the local filesystem, estimates whether there is enough room for the new ingest, and evicts stale indexes (LRU) to free space. Introduced in PR #166, hardened in PR #167.

## Context: Where Disk Manager Fits

DiskGraph uses local NVMe SSD storage (`/mnt/localssd` in production) for its indexes. Each graph ingest produces an index directory that can be tens or hundreds of GB. Over time, indexes from previous graph versions accumulate. Without management, the SSD fills up and ingest fails with `ENOSPC`.

The disk manager is called from the Kumo Temporal activity `ensure_diskgraph_space` (in `kumo/rest/backends/data_backend/activities/diskgraph_disk_manager.py`) before the `DiskgraphIngestWorkflow` starts. The call flow:

```
DiskgraphIngestWorkflow
  → ensure_diskgraph_space activity
    → estimate_ingest_size(config_json)       # sum source file sizes
    → DiskManager(data_root).ensure_space(
          needed_bytes = source_bytes * multiplier,
          current_graph_key,
          current_graph_version,
      )
    → Rust scans indexes, plans eviction, executes deletion
  → diskgraph_ingest_activity              # actual ingest starts
```

The `ingest_space_multiplier` (default 4x) accounts for the index being larger than the raw source data (rowpack encoding, pk maps, edge shards, temp files).

## Directory Layout

Indexes live under `data_root` in a two-level directory structure:

```
{data_root}/
  {graph_key}/
    {graph_version}/
      index_metadata.json
      ingest_stats.json
      col_stats.json
      nodes/
      _SUCCESS          ← written after successful ingest
    {graph_version_2}/
      ...
  {graph_key_2}/
    ...
```

- **`_SUCCESS` marker**: A file written at the end of a successful ingest. Its mtime is used for LRU ordering.
- **Valid index**: Has a `_SUCCESS` marker.
- **Orphan**: No `_SUCCESS` marker -- the ingest crashed or was cancelled before completing. Orphans are evicted before valid indexes.

## API

### Rust (`diskgraph_ingest::disk_manager`)

```rust
pub struct DiskManager {
    data_root: PathBuf,  // canonicalized at construction
}

impl DiskManager {
    pub fn new(data_root: &Path) -> Self;
    pub fn ensure_space(&self, req: &EnsureSpaceRequest) -> DiskManagerResult<EnsureSpaceResult>;
    pub fn scan_indexes(&self) -> DiskManagerResult<Vec<IndexInfo>>;
    pub fn disk_usage(&self) -> DiskManagerResult<DiskUsage>;
}

// Free functions
pub fn available_space(path: &Path) -> io::Result<u64>;  // statvfs f_bavail * f_frsize
pub fn total_space(path: &Path) -> io::Result<u64>;       // statvfs f_blocks * f_frsize
```

**Types:**

| Type | Fields | Purpose |
|------|--------|---------|
| `IndexInfo` | path, graph_key, graph_version, size_bytes, has_success_marker, success_mtime_secs | One scanned index directory |
| `DiskUsage` | total_bytes, available_bytes, used_bytes | Filesystem capacity snapshot |
| `EnsureSpaceRequest` | needed_bytes, current_graph_key, current_graph_version | Input to ensure_space |
| `EnsureSpaceResult` | evicted, available_bytes, needed_bytes, freed_bytes | Output from ensure_space |
| `EvictedIndex` | path, graph_key, graph_version, size_bytes, was_orphan | One evicted directory |
| `DiskManagerError` | InvalidPath, InsufficientSpace { needed, available }, Io | Error variants |

### Python (`diskgraph.DiskManager`)

```python
from diskgraph import DiskManager, estimate_ingest_size

dm = DiskManager("/mnt/localssd")

# Ensure space, evicting if needed. Returns JSON string.
result_json = dm.ensure_space(
    needed_bytes=50_000_000_000,
    current_graph_key="my-graph",
    current_graph_version="2025-03-27T12:00:00Z",
)

# Scan all indexes. Returns JSON string (list of IndexInfo).
indexes_json = dm.scan_indexes()

# Get disk usage. Returns JSON string (DiskUsage).
usage_json = dm.disk_usage()

# Estimate source bytes for an ingest (standalone function).
source_bytes = estimate_ingest_size(config_json_str)
```

All three `DiskManager` methods release the GIL (`py.detach()`) during Rust execution so that Python threads (e.g., Temporal heartbeat) continue running. Return values are JSON strings (serde_json serialization of the Rust types).

**Errors:** Methods raise `DiskgraphSystemError` for IO/serialization failures, `DiskgraphConfigError` for invalid `data_root`. `estimate_ingest_size` raises `DiskgraphConfigError` for invalid JSON, `DiskgraphSystemError` if byte computation fails, and `DiskgraphDataError` if computed size is 0.

## Eviction Algorithm

`ensure_space` follows a strict priority order:

1. **Validate** `current_graph_key` and `current_graph_version` via `validate_path_component`.
2. **Check available space** via `statvfs`. If sufficient, return immediately (no eviction).
3. **Scan** all index directories under `data_root` (two-level walk: graph_key → graph_version).
4. **Classify** each as orphan (no `_SUCCESS`) or valid (has `_SUCCESS`).
5. **Plan eviction**:
   - Orphans first, sorted by path for deterministic order.
   - Then valid indexes in LRU order: oldest `_SUCCESS` mtime first.
   - **Never evicts** the current ingest target (matching `current_graph_key` + `current_graph_version`).
   - Stops as soon as cumulative freed bytes >= deficit.
   - If the plan is empty (nothing evictable), return `InsufficientSpace` immediately.
6. **Execute eviction**: Delete each planned directory via `remove_dir_all` on the canonicalized path.
7. **Clean up** empty parent (graph_key) directories after eviction.
8. **Verify**: Re-check available space. If still insufficient after all evictions, return `InsufficientSpace` error.

Eviction failures are logged at error level but do not abort the remaining plan. The final space check catches cases where failed deletions left the disk too full.

## Path Safety

The disk manager validates all path components to prevent directory traversal attacks and symlink-based exploits. This matters because `graph_key` and `graph_version` originate from user-controlled input (graph names, data versions).

| Function | What it does |
|----------|-------------|
| `validate_path_component(name)` | Rejects empty strings, null bytes, path separators (`/`, `\`), `.`, `..`, and hidden files (`.` prefix) |
| `canonicalize_under(parent, child)` | Resolves both paths via `canonicalize()`, then verifies the child is strictly under the parent. Prevents symlink-swap TOCTOU races |
| `safe_index_path(data_root, key, ver)` | Validates both components, returns `data_root/key/ver` |

`scan_indexes` additionally calls `symlink_metadata()` at both the graph-key and version directory levels, skipping any symlinks it finds.

## Security Hardening (PR #167)

PR #167 addressed issues found during adversarial code review:

| Issue | Fix |
|-------|-----|
| GIL held during blocking Rust calls | All three pybind methods use `py.detach()` to release GIL. Prevents Temporal heartbeat timeouts on slow disk operations |
| TOCTOU symlink-swap in eviction | `execute_eviction` canonicalizes via `canonicalize_under(data_root, path)` before `remove_dir_all`. Even if an attacker replaces the directory with a symlink between plan and execute, the canonical path points to the real location |
| `data_root` with `..` segments | `DiskManager::new()` canonicalizes `data_root` at construction time |
| ENOSPC detection across error layers | `is_disk_space_error()` walks the error chain: first tries typed `raw_os_error()` checks for codes 28 (ENOSPC) and 122 (EDQUOT), then falls back to string-matching "No space left on device", "os error 28", "Disk quota exceeded", "os error 122" for errors wrapped inside `ParquetError::General(String)` and similar |
| Arithmetic overflow on large filesystems | `statvfs` arithmetic uses `saturating_mul` instead of `*` |
| Orphan eviction non-determinism | Orphans sorted by path before eviction |
| Dead code (background delete via shell) | Removed ~62 lines of unused `spawn_background_delete` / `ionice` / `nice` code |

## Developing: Changing Disk Manager

**Running tests:**

```bash
cargo test -p diskgraph-ingest disk_manager
```

30 tests across all modules:

| Module | Tests | What they cover |
|--------|-------|-----------------|
| `mod.rs` | 10 | Integration: scan + ensure_space, eviction ordering, insufficient space, bad paths, disk usage, real disk pressure |
| `eviction.rs` | 7 | Plan logic: orphans first, LRU order, skip current target, stop when enough, cross-key eviction, execute + delete |
| `scan.rs` | 3 | Scan: valid + orphan detection, hidden dirs skipped, empty root |
| `path_safety.rs` | 10 | Validation: traversal, hidden, empty, null, symlink escape, canonicalize_under, safe_index_path |

The `test_eviction_under_real_disk_pressure` test creates MB-sized indexes on a real filesystem and verifies that `ensure_space` correctly evicts them in LRU order when the requested space exceeds available space.

**What to watch when changing:**

- **Eviction order must remain deterministic.** Orphans sorted by path, then valid by mtime. Non-determinism in eviction causes flaky tests and makes production debugging harder.
- **Never skip the canonicalize_under check** in execute_eviction. It is the only defense against TOCTOU symlink swaps.
- **GIL release is mandatory** for all pybind methods that call into Rust. Without it, Temporal workers deadlock on heartbeat.
- **The 1-job-at-a-time assumption** means no concurrent eviction races. If concurrency changes, the scan → plan → execute sequence needs locking.

## File Locations

| Component | Path |
|-----------|------|
| Core struct + `ensure_space` | `diskgraph-ingest/src/disk_manager/mod.rs` |
| Data types | `diskgraph-ingest/src/disk_manager/types.rs` |
| Eviction planning + execution | `diskgraph-ingest/src/disk_manager/eviction.rs` |
| Index scanning + `dir_size` | `diskgraph-ingest/src/disk_manager/scan.rs` |
| Path validation + canonicalization | `diskgraph-ingest/src/disk_manager/path_safety.rs` |
| ENOSPC detection helper | `diskgraph-ingest/src/lib.rs` (`is_disk_space_error`) |
| Python bindings | `diskgraph-pybind/src/lib.rs` (`DiskManager` pyclass) |
| Python export | `diskgraph-pybind/python/diskgraph/__init__.py` |
| Kumo Temporal activity caller | `kumo/rest/backends/data_backend/activities/diskgraph_disk_manager.py` |
