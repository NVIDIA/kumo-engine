# diskgraph-ptrhash

Local copy of ptr_hash v1.1.0 from https://github.com/RagnarGrootKoerkamp/PtrHash.

Contains various performance improvements applied (to be filled out later).

Tests are gated behind the `ptrhash-tests` feature:

```
cargo test -p ptr_hash --features ptrhash-tests
```

They are not run by default since they are slow.
