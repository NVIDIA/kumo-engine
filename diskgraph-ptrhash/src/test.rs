use super::*;

struct KeyVec(Vec<u64>);

impl KeySource<u64> for KeyVec {
    type Iter<'a> = std::slice::Iter<'a, u64>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter()
    }
}

impl KeyVec {
    fn iter(&self) -> std::slice::Iter<'_, u64> {
        self.0.iter()
    }
}

fn make_keys(n: usize) -> KeyVec {
    KeyVec((0..n as u64).collect())
}

/// Construct the MPHF and test all keys are mapped to unique indices.
#[test]
#[ignore]
fn construct_unique() {
    for n in [2, 10, 100, 1000, 10_000, 100_000] {
        let keys = make_keys(n);
        let mut scratch = PtrHashScratch::<u64>::new();
        let ptr_hash = DefaultPtrHash::<hash::GxHash64, u64>::try_new_from_source_with_scratch(
            &keys,
            PtrHashParams::default(),
            &mut scratch,
        )
        .expect("ptrhash construction failed");
        let mut done = bitvec![0; n];
        for key in keys.iter() {
            let idx = ptr_hash.index(key);
            assert!(!done[idx]);
            done.set(idx, true);
        }
    }
}

#[test]
#[ignore]
fn ptr_hash_can_clone() {
    let keys = make_keys(64);
    let mut scratch = PtrHashScratch::<u64>::new();
    let ptr_hash = DefaultPtrHash::<hash::GxHash64, u64>::try_new_from_source_with_scratch(
        &keys,
        PtrHashParams::default(),
        &mut scratch,
    )
    .expect("ptrhash construction failed");

    let _y = ptr_hash.clone();
}
