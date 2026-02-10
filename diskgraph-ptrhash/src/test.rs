use super::*;

struct KeyVec(Vec<u64>);

impl KeySource<u64> for KeyVec {
    type Iter<'a>
        = std::slice::Iter<'a, u64>
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

fn make_low_entropy_keys(n: usize) -> KeyVec {
    let mut keys = Vec::with_capacity(n);
    for i in 0..n as u64 {
        keys.push(i.wrapping_mul(1_u64 << 32));
    }
    KeyVec(keys)
}

fn build_u64(keys: &KeyVec) -> DefaultPtrHash<hash::GxHash64, u64> {
    let mut scratch = PtrHashScratch::<u64>::new();
    DefaultPtrHash::<hash::GxHash64, u64>::try_new_from_source_with_scratch(
        keys,
        PtrHashParams::default(),
        &mut scratch,
    )
    .expect("ptrhash construction failed")
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

#[test]
fn unique_mapping_small_n() {
    for n in [1, 2, 10, 100, 1000, 10_000] {
        let keys = make_keys(n);
        let ptr_hash = build_u64(&keys);
        let mut done = bitvec![0; n];
        for key in keys.iter() {
            let idx = ptr_hash.index(key);
            assert!(idx < n);
            assert!(!done[idx]);
            done.set(idx, true);
        }
    }
}

#[test]
fn index_bounds_small_n() {
    for n in [1, 2, 10, 100, 1000, 10_000] {
        let keys = make_keys(n);
        let ptr_hash = build_u64(&keys);
        for key in keys.iter() {
            let idx = ptr_hash.index_no_remap(key);
            assert!(idx < ptr_hash.max_index());
            assert!(ptr_hash.index(key) < n);
        }
    }
}

#[test]
fn low_entropy_keys_build() {
    let keys = make_low_entropy_keys(10_000);
    let ptr_hash = build_u64(&keys);
    for key in keys.iter() {
        let idx = ptr_hash.index(key);
        assert!(idx < keys.len());
    }
}

#[test]
#[ignore]
fn low_entropy_10k_smoke() {
    let keys = make_low_entropy_keys(10_000);
    let ptr_hash = build_u64(&keys);
    for key in keys.iter().take(5_000) {
        let idx = ptr_hash.index(key);
        assert!(idx < keys.len());
    }
}

#[test]
#[ignore]
fn low_entropy_200k_smoke() {
    let keys = make_low_entropy_keys(200_000);
    let ptr_hash = build_u64(&keys);
    for key in keys.iter().take(10_000) {
        let idx = ptr_hash.index(key);
        assert!(idx < keys.len());
    }
}

#[test]
fn single_part_build() {
    let keys = make_keys(10_000);
    let mut scratch = PtrHashScratch::<u64>::new();
    let mut params = PtrHashParams::default();
    params.single_part = true;
    let ptr_hash = DefaultPtrHash::<hash::GxHash64, u64>::try_new_from_source_with_scratch(
        &keys,
        params,
        &mut scratch,
    )
    .expect("ptrhash construction failed");
    for key in keys.iter() {
        let idx = ptr_hash.index(key);
        assert!(idx < keys.len());
    }
}

#[test]
fn small_n_tuning_applies() {
    let keys = make_keys(10_000);
    let mut scratch = PtrHashScratch::<u64>::new();
    let mut params = PtrHashParams::default();
    params.alpha = 0.8;
    params.lambda = 10.0;
    let ptr_hash = DefaultPtrHash::<hash::GxHash64, u64>::try_new_from_source_with_scratch(
        &keys,
        params,
        &mut scratch,
    )
    .expect("ptrhash construction failed");
    assert!(ptr_hash.params.alpha >= 0.995);
    assert!(ptr_hash.params.lambda <= 2.0);
}

#[test]
fn scratch_reuse_multiple_builds() {
    let mut scratch = PtrHashScratch::<u64>::new();
    for n in [10, 1000, 100, 5000] {
        let keys = make_keys(n);
        let ptr_hash = DefaultPtrHash::<hash::GxHash64, u64>::try_new_from_source_with_scratch(
            &keys,
            PtrHashParams::default(),
            &mut scratch,
        )
        .expect("ptrhash construction failed");
        for key in keys.iter() {
            let idx = ptr_hash.index(key);
            assert!(idx < keys.len());
        }
    }
}

#[test]
#[ignore]
fn large_n_smoke() {
    let keys = make_keys(1_000_000);
    let ptr_hash = build_u64(&keys);
    for key in keys.iter().take(10_000) {
        let idx = ptr_hash.index(key);
        assert!(idx < keys.len());
    }
}

#[test]
#[ignore]
fn near_small_n_threshold_smoke() {
    let keys = make_low_entropy_keys(900_000);
    let ptr_hash = build_u64(&keys);
    for key in keys.iter().take(10_000) {
        let idx = ptr_hash.index(key);
        assert!(idx < keys.len());
    }
}

#[test]
#[ignore]
fn retry_stress_tight_params() {
    let keys = make_keys(100_000);
    let mut scratch = PtrHashScratch::<u64>::new();
    let mut params = PtrHashParams::default();
    params.alpha = 0.995;
    params.lambda = 2.0;
    let ptr_hash = DefaultPtrHash::<hash::GxHash64, u64>::try_new_from_source_with_scratch(
        &keys,
        params,
        &mut scratch,
    )
    .expect("ptrhash construction failed");
    for key in keys.iter().take(10_000) {
        let idx = ptr_hash.index(key);
        assert!(idx < keys.len());
    }
}
