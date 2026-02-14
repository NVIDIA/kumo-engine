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
   - Sorting/building now reuse scratch-owned vectors for hashes, part starts, pilots, taken bits, bucket metadata, slot scratch, and remap scratch.
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
   - Removed `EliasFano` packed backend; retained `Vec<...>` and `CachelineEfVec`.
8. Public query surface is narrower:
   - Stream/batch/single-part specialized query helpers from upstream are removed.
   - Primary supported lookups are `index_no_remap()` and `index()`.

## Testing Notes

Tests in `src/test.rs` were adapted to the scratch-based API, and several heavy tests are marked `#[ignore]` by default.
