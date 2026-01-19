# diskgraph-ptrhash

Local copy of ptr_hash v1.1.0 from https://github.com/RagnarGrootKoerkamp/PtrHash.

Contains various performance improvements applied (to be filled out later).

Scratch buffers can be reused across builds to avoid repeated allocations:

```
use ptr_hash::{DefaultPtrHash, PtrHashParams, PtrHashScratch};

let mut scratch = PtrHashScratch::new();
let mphf = DefaultPtrHash::try_new_with_scratch(&keys, PtrHashParams::default(), &mut scratch)
    .expect("ptrhash construction failed");
```

Tests are gated behind the `ptrhash-tests` feature:

```
cargo test -p ptr_hash --features ptrhash-tests
```

They are not run by default since they are slow.
