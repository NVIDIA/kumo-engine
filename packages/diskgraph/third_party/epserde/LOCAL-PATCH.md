# Local patch to epserde 0.8.0

This copy removes the `maligned` dependency without upgrading the serialization library. The saved format version stays at `(1, 1)`.

The source comes from the published `epserde 0.8.0` crate. The upstream commit is `3a78b566ebe088dd8f9af9dbae856415b1791035`. The archive identity and license are recorded in [source-provenance.json](../source-provenance.json).

The local changes are:

- Add sealed `A16` and `A64` byte storage types in `src/utils/alignment.rs`. Each type has equal size and alignment. Rust owns its allocation and deallocation.
- Use those types in `AlignedCursor` and `MemBackend`. Remove the `maligned` dependency and the `mem_dbg/maligned` feature from the Cargo manifest.
- Update upstream tests to import the replacement types. Add tests for alignment during buffer growth.

The manifest uses the compact upstream form. Standalone example programs and the duplicate manifest are omitted. All integration tests and documentation tests remain.

Trailing whitespace is removed from the copied README and deserialization documentation.

Serialized fields, type hashes, writer tags, and format headers stay unchanged. The readers now check bounds and decode the existing ControlFlow writer tags correctly. The index compatibility tests use eight frozen files written with the unpatched release dependencies. They test both hash implementations and all four key storage shapes. They compare lookup results through all three loaders and compare the saved bytes.

The backport also checks truncated inputs, uses owned zeroed storage during loading, and corrects ControlFlow reader tags. Standalone tests run explicitly in CI. Documentation tests use isolated temporary directories.

The package license includes the Apache-2.0-only alignment module alongside the upstream dual license. The upstream copyright and license files remain. New code carries its own Apache-2.0 header. This is a local backport for review, not an upstream release.
