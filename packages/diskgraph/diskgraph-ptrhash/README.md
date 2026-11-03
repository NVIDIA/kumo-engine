# diskgraph-ptrhash

Local fork of `ptr_hash` `v1.1.0` (upstream: <https://github.com/RagnarGrootKoerkamp/PtrHash>, crate source: `ptr_hash 1.1.0`).

## Key Changes From Upstream `ptr_hash` 1.1.0

1. Construction API is now scratch-based:
   - Added `PtrHashScratch` and `KeySource`.
   - Main constructor is `try_new_from_source_with_scratch(...)` to allow caller-owned buffer reuse across repeated builds.
2. Build path is single-shard only:
   - Removed sharding config/implementation (`keys_per_shard`, `Sharding`, shard module usage).
   - Removed `new`, `new_from_par_iter`, and `new_with_stats` construction paths.
3. Allocation reuse was pushed deeper into internals:
   - Sorting/building now reuse scratch-owned vectors for hashes, part starts, pilots, taken bits, bucket metadata, slot scratch.
4. Sorting pipeline changed:
   - Removed `rdst` radix sort usage.
   - Added local radix sort implementation in `src/radix.rs`.
5. Parallelism/stats plumbing removed from core build:
   - Removed Rayon-based parallel build/sort paths.
   - Removed `stats` module integration from public construction APIs.
6. Hashing choices were simplified:
   - `DefaultPtrHash` now defaults to `hash::GxHash64` (instead of `FxHash` in upstream 1.1.0).
   - Kept a smaller hasher set focused on `gxhash` and `MulHash`; optional hasher matrix was dropped.
7. Bucket/reduce/storage options were narrowed:
   - `bucket_fn` implementations were reduced to the active path used here (`CubicEps`).
   - `reduce` swapped upstream `MulReduce` path for `FM32` fastmod-style reduction.
   - Removed remapping and its packed-storage backends, including `cacheline-ef`.
8. Public query surface is narrower:
   - Stream/batch/single-part specialized query helpers from upstream are removed.
   - Lookups use `index_no_remap()` with slot bounds from `max_index()`.

## Saved-index compatibility

Removing the remap parameter and table changes the serialized `PtrHash` type
hashes. Existing indexes must be rebuilt from source data with this version;
resuming ingestion from an old primary-key-map checkpoint is unsupported.
Epserde rejects the old hash files rather than interpreting them as the new type.
The epserde upgrade to 0.12.6 and removal of unused `MemSize` derives eliminate
`maligned` without a local dependency patch.

## Testing Notes

Tests in `src/test.rs` were adapted to the scratch-based API, and several heavy tests are marked `#[ignore]` by default.
