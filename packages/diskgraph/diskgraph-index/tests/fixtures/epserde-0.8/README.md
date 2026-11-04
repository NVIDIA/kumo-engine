# Index compatibility fixtures

These eight files were written before the security backport, with unpatched `epserde 0.8.0` from crates.io. The key index implementation matches the source recorded for Kumo Engine `1.0.0`.

Each index contains 64 keys. The fixtures cover `PkHash` and `GxHash128`. Their key storage types are `u64`, variable byte arrays, 16-byte arrays, and 32-byte arrays. The last two cover the UUID and IP storage shapes.

The JSON files record lookup answers from the original writer. These answers remain as historical evidence. Epserde 0.13.2 rejects these files because their format version is incompatible; the tests verify rejection through all three loaders and round-trip newly constructed indexes separately.

[provenance.json](provenance.json) records the source, crate, and fixture file fingerprints. The files were generated on Linux ARM64. They do not establish compatibility with a separate Linux x86_64 build.

The producer was `examples/write_legacy_index_fixture.rs` in the source history before the epserde upgrade. Do not regenerate these frozen files with the current library; existing indexes must be rebuilt from source data.
