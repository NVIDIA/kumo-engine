# Index compatibility fixtures

These eight files were written before the security backport, with unpatched `epserde 0.8.0` from crates.io. The key index implementation matches the source recorded for Kumo Engine `1.0.0`.

Each index contains 64 keys. The fixtures cover `PkHash` and `GxHash128`. Their key storage types are `u64`, variable byte arrays, 16-byte arrays, and 32-byte arrays. The last two cover the UUID and IP storage shapes.

The JSON files record lookup answers from the original writer. The tests read these fixed answers instead of deriving expected answers from the patched reader. The tests also compare the bytes after loading and saving each index.

[provenance.json](provenance.json) records the source, crate, and fixture file fingerprints. The files were generated on Linux ARM64. They do not establish compatibility with a separate Linux x86_64 build.

The producer is `examples/write_legacy_index_fixture.rs`. Run it only against the unpatched release dependencies when creating replacement fixtures. Do not regenerate these files with the patched library to make a failing compatibility test pass.
