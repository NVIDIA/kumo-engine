use super::*;

struct ByteKeys(Vec<Vec<u8>>);

impl KeySource<Vec<u8>> for ByteKeys {
    type Iter<'a>
        = std::slice::Iter<'a, Vec<u8>>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter()
    }
}

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

fn build_u64(keys: &KeyVec) -> DefaultPtrHash<hash::PkHash, u64> {
    let mut scratch = PtrHashScratch::<u64>::new();
    DefaultPtrHash::<hash::PkHash, u64>::try_new_from_source_with_scratch(
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
        let ptr_hash = DefaultPtrHash::<hash::PkHash, u64>::try_new_from_source_with_scratch(
            &keys,
            PtrHashParams::default(),
            &mut scratch,
        )
        .expect("ptrhash construction failed");
        let mut done = bitvec![0; ptr_hash.max_index()];
        for key in keys.iter() {
            let idx = ptr_hash.index_no_remap(key);
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
    let ptr_hash = DefaultPtrHash::<hash::PkHash, u64>::try_new_from_source_with_scratch(
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
        let mut done = bitvec![0; ptr_hash.max_index()];
        for key in keys.iter() {
            let idx = ptr_hash.index_no_remap(key);
            assert!(idx < ptr_hash.max_index());
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
        }
    }
}

#[test]
fn low_entropy_keys_build() {
    let keys = make_low_entropy_keys(10_000);
    let ptr_hash = build_u64(&keys);
    for key in keys.iter() {
        let idx = ptr_hash.index_no_remap(key);
        assert!(idx < ptr_hash.max_index());
    }
}

#[test]
#[ignore]
fn low_entropy_10k_smoke() {
    let keys = make_low_entropy_keys(10_000);
    let ptr_hash = build_u64(&keys);
    for key in keys.iter().take(5_000) {
        let idx = ptr_hash.index_no_remap(key);
        assert!(idx < ptr_hash.max_index());
    }
}

#[test]
#[ignore]
fn low_entropy_200k_smoke() {
    let keys = make_low_entropy_keys(200_000);
    let ptr_hash = build_u64(&keys);
    for key in keys.iter().take(10_000) {
        let idx = ptr_hash.index_no_remap(key);
        assert!(idx < ptr_hash.max_index());
    }
}

#[test]
fn single_part_build() {
    let keys = make_keys(10_000);
    let mut scratch = PtrHashScratch::<u64>::new();
    let mut params = PtrHashParams::default();
    params.single_part = true;
    let ptr_hash = DefaultPtrHash::<hash::PkHash, u64>::try_new_from_source_with_scratch(
        &keys,
        params,
        &mut scratch,
    )
    .expect("ptrhash construction failed");
    for key in keys.iter() {
        let idx = ptr_hash.index_no_remap(key);
        assert!(idx < ptr_hash.max_index());
    }
}

#[test]
fn small_n_tuning_applies() {
    let keys = make_keys(10_000);
    let mut scratch = PtrHashScratch::<u64>::new();
    let mut params = PtrHashParams::default();
    params.alpha = 0.8;
    params.lambda = 10.0;
    let ptr_hash = DefaultPtrHash::<hash::PkHash, u64>::try_new_from_source_with_scratch(
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
        let ptr_hash = DefaultPtrHash::<hash::PkHash, u64>::try_new_from_source_with_scratch(
            &keys,
            PtrHashParams::default(),
            &mut scratch,
        )
        .expect("ptrhash construction failed");
        for key in keys.iter() {
            let idx = ptr_hash.index_no_remap(key);
            assert!(idx < ptr_hash.max_index());
        }
    }
}

#[test]
#[ignore]
fn large_n_smoke() {
    let keys = make_keys(1_000_000);
    let ptr_hash = build_u64(&keys);
    for key in keys.iter().take(10_000) {
        let idx = ptr_hash.index_no_remap(key);
        assert!(idx < ptr_hash.max_index());
    }
}

#[test]
#[ignore]
fn near_small_n_threshold_smoke() {
    let keys = make_low_entropy_keys(900_000);
    let ptr_hash = build_u64(&keys);
    for key in keys.iter().take(10_000) {
        let idx = ptr_hash.index_no_remap(key);
        assert!(idx < ptr_hash.max_index());
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
    let ptr_hash = DefaultPtrHash::<hash::PkHash, u64>::try_new_from_source_with_scratch(
        &keys,
        params,
        &mut scratch,
    )
    .expect("ptrhash construction failed");
    for key in keys.iter().take(10_000) {
        let idx = ptr_hash.index_no_remap(key);
        assert!(idx < ptr_hash.max_index());
    }
}

/// Keys gxhash cannot separate must build with the fallback.
#[test]
#[cfg(target_arch = "x86_64")]
fn fallback_hash_builds_what_gxhash_cannot() {
    let (base, twin) = hash::gxhash_twins();
    let (base, twin) = (base.into_bytes(), twin.into_bytes());
    assert_ne!(base, twin);
    for seed in [0u64, 1, 12_345, u64::MAX] {
        assert_eq!(
            <hash::PkHash as Hasher<Vec<u8>>>::hash(&base, seed, false),
            <hash::PkHash as Hasher<Vec<u8>>>::hash(&twin, seed, false),
            "the witness stopped colliding under seed {seed:#x}"
        );
    }

    let mut keys = vec![base.clone(), twin.clone()];
    keys.extend((0..2_000u32).map(|i| {
        let mut k = format!("kumo30356-filler-{i:016}").into_bytes();
        k.resize(33, b'.');
        k
    }));
    let mut scratch = PtrHashScratch::<u64>::new();

    assert!(
        DefaultPtrHash::<hash::PkHash, Vec<u8>>::try_new_from_source_with_scratch(
            &ByteKeys(keys.clone()),
            PtrHashParams::default(),
            &mut scratch,
        )
        .is_none(),
        "gxhash must fail here, or this proves nothing about the fallback"
    );

    let mphf = DefaultPtrHash::<hash::PkHash, Vec<u8>>::try_new_from_source_with_scratch(
        &ByteKeys(keys.clone()),
        PtrHashParams {
            prefer_fallback_hash: true,
            ..PtrHashParams::default()
        },
        &mut scratch,
    )
    .expect("the fallback must build it");

    let mut seen = vec![false; mphf.max_index()];
    for k in &keys {
        let idx = mphf.index_no_remap(k);
        assert!(idx < mphf.max_index() && !seen[idx], "slot collision");
        seen[idx] = true;
    }
    assert_ne!(mphf.index_no_remap(&base), mphf.index_no_remap(&twin));
}

#[cfg(feature = "epserde")]
#[test]
fn serialized_slots_round_trip_all_key_types() {
    use epserde::prelude::{AlignedCursor, Deserialize, Serialize};

    macro_rules! check {
        ($name:literal, $keys:expr, $fallback:expr) => {{
            let source: std::collections::HashMap<_, ()> =
                $keys.into_iter().map(|key| (key, ())).collect();
            let mut scratch = PtrHashScratch::<u64>::new();
            let hash = DefaultPtrHash::<hash::PkHash, _>::try_new_from_source_with_scratch(
                &source,
                PtrHashParams {
                    prefer_fallback_hash: $fallback,
                    ..PtrHashParams::default()
                },
                &mut scratch,
            ).unwrap();
            let mut buffer = <AlignedCursor>::new();
            // SAFETY: the constructed hash has valid fields, and the aligned
            // buffer stays immutable and alive while the borrowed view is used.
            let restored = unsafe {
                hash.serialize(&mut buffer).unwrap();
                DefaultPtrHash::<hash::PkHash, _>::deserialize_eps(buffer.as_bytes()).unwrap()
            };
            let output_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../target/test-artifacts/remap-free");
            std::fs::create_dir_all(&output_dir).unwrap();
            let path = output_dir.join(concat!($name, ".bin"));
            std::fs::write(&path, buffer.as_bytes()).unwrap();
            // SAFETY: this test writes the typed hash and keeps the file
            // unchanged for the lifetime of the mapping.
            let mapped = unsafe {
                DefaultPtrHash::<hash::PkHash, _>::load_mmap(
                    &path, epserde::prelude::Flags::empty(),
                ).unwrap()
            };
            assert_eq!(restored.n(), hash.n());
            assert_eq!(restored.max_index(), hash.max_index());
            let mut seen = bitvec![0; hash.max_index()];
            for key in source.keys() {
                let slot = restored.index_no_remap(key);
                assert_eq!(slot, hash.index_no_remap(key));
                assert_eq!(slot, mapped.uncase().index_no_remap(key));
                assert!(!seen[slot], "slot collision after deserialization");
                seen.set(slot, true);
            }
        }};
    }

    check!("int", (0..1_000_u64), false);
    check!(
        "bytes",
        (0..1_000_u64).map(|key| key.to_le_bytes().to_vec()),
        false
    );
    check!("uuid-ip", (0..1_000_u128).map(u128::to_le_bytes), false);
    check!(
        "uuid256",
        (0..1_000_u64).map(|key| {
            let mut bytes = [0; 32];
            bytes[..8].copy_from_slice(&key.to_le_bytes());
            bytes
        }),
        true
    );
}

#[cfg(feature = "epserde")]
#[test]
fn rejects_previous_file_format() {
    use epserde::prelude::{AlignedCursor, Deserialize, Serialize};

    let hash = build_u64(&make_keys(100));
    let mut buffer = <AlignedCursor>::new();
    // SAFETY: the hash was constructed from valid keys.
    let schema = unsafe { hash.serialize_with_schema(&mut buffer) }.unwrap();
    let row = schema
        .0
        .iter()
        .find(|row| row.field == "VERSION_MAJOR")
        .unwrap();
    buffer.as_bytes_mut()[row.offset..row.offset + 2].copy_from_slice(&1_u16.to_ne_bytes());
    // SAFETY: only the header version changed; the serialized payload is valid.
    let result = unsafe { DefaultPtrHash::<hash::PkHash, u64>::deserialize_eps(buffer.as_bytes()) };
    assert!(matches!(
        result,
        Err(epserde::deser::Error::MajorVersionMismatch(1))
    ));
}
