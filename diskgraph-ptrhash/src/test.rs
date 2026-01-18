use super::*;
use crate::util::generate_keys;

/// Construct the MPHF and test all keys are mapped to unique indices.
#[test]
fn construct() {
    for n in [2, 10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000] {
        let keys = generate_keys(n);
        let ptr_hash = <PtrHash>::new(&keys, Default::default());
        let mut done = bitvec![0; n];
        for key in keys {
            let idx = ptr_hash.index(&key);
            assert!(!done[idx]);
            done.set(idx, true);
        }
    }
}

#[test]
fn index_stream() {
    for n in [2, 10, 100, 1000, 10_000, 100_000, 1_000_000] {
        let keys = generate_keys(n);
        let ptr_hash = <PtrHash>::new(&keys, Default::default());
        let sum = ptr_hash.index_stream::<32, true, _>(&keys).sum::<usize>();
        assert_eq!(sum, (n * (n - 1)) / 2, "Failure for n = {n}");
    }
}

#[cfg(feature = "unstable")]
#[test]
fn index_batch() {
    for n in [10usize, 100, 1000, 10_000, 100_000, 1_000_000] {
        let n = n.next_multiple_of(32);
        let keys = generate_keys(n);
        let ptr_hash = <PtrHash>::new(&keys, Default::default());
        let sum = ptr_hash.index_batch_exact::<32, true>(&keys).sum::<usize>();
        assert_eq!(sum, (n * (n - 1)) / 2);
    }
}

#[test]
fn new_par_iter() {
    let n = 10_000_000;
    let keys = generate_keys(n);
    <PtrHash>::new_from_par_iter(n, keys.iter(), Default::default());
}

#[test]
fn ptr_hash_can_clone() {
    let ptr_hash = PtrHash::<_>::new(&["hello", "there"], PtrHashParams::default());

    // test succeeds if this compiles
    let _y = ptr_hash.clone();
}
