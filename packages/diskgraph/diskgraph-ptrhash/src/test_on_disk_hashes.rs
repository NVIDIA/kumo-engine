//! Golden values for hashes that persist in on-disk DiskGraph indexes.
//!
//! This module pins two things:
//!
//! - The MPHF key hash [`PkHash`] computes, on both of its paths. A built
//!   index stores its seed and pilots, and every lookup recomputes the key
//!   hash. Ingest picks the path per index through
//!   `PtrHashParams::prefer_fallback_hash`, set when the longest key exceeds
//!   [`MAX_GXHASH_INJECTIVE_LEN`] (16 bytes):
//!   - gxhash for Int64 (8-byte), Uuid and Ip (16-byte) keys, and for
//!     VarBytes keys whose sampled maximum length is at most 16 bytes. The
//!     maximum comes from a sample, so such an index can still hold longer
//!     keys.
//!   - xxh3 for Uuid256 (32-byte) keys and VarBytes keys longer than 16
//!     bytes.
//! - The type and alignment hashes epserde writes into every index header.
//!   epserde refuses to load a file whose hashes differ.
//!
//! Shard routing, which places each key on a shard with gxhash, is pinned
//! next to the functions that compute it, in diskgraph-ingest's
//! `shard_routing_matches_golden_values`. The optional `pkhash128` mode,
//! which hashes keys with `GxHash128`, is not covered.
//!
//! gxhash documents its output as stable across platforms within a major
//! version. CI runs this test on x86_64.
//!
//! A failure here means existing indexes would route lookups wrongly or fail
//! to load. Do not update these values to make a dependency or toolchain bump
//! pass; that bump breaks on-disk compatibility.
//!
//! [`MAX_GXHASH_INJECTIVE_LEN`]: crate::MAX_GXHASH_INJECTIVE_LEN

use crate::hash::{Hasher, PkHash};

const SEED: u64 = 0x9e37_79b9_7f4a_7c15;

fn gxhash_path<Key: crate::KeyT + ?Sized>(key: &Key, seed: u64) -> u64
where
    PkHash: Hasher<Key, H = u64>,
{
    <PkHash as Hasher<Key>>::hash(key, seed, false)
}

fn fallback<Key: crate::KeyT + ?Sized>(key: &Key, seed: u64) -> u64
where
    PkHash: Hasher<Key, H = u64>,
{
    <PkHash as Hasher<Key>>::hash(key, seed, true)
}

/// Deterministic key bytes of length `len`.
fn bytes(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect()
}

#[test]
fn pk_hash_gxhash_matches_golden_values() {
    // `Hash` writes a `u64` key in one call, but arrays and `Vec`s as an
    // 8-byte length followed by the key bytes, and gxhash compresses each
    // write on its own. The 0-, 8- and 16-byte keys fit in one 16-byte
    // vector, as every key of a gxhash index normally does; the longer ones
    // stand for keys a VarBytes sample missed and reach the 17-32, 49-64 and
    // over 64 byte paths.
    let cases: [(&str, u64, u64); 11] = [
        (
            "u64, seed 0",
            gxhash_path(&0x0123_4567_89ab_cdef_u64, 0),
            0x4bb3_a168_130c_a67d,
        ),
        (
            "u64",
            gxhash_path(&0x0123_4567_89ab_cdef_u64, SEED),
            0x81a3_a15f_9ef8_37c5,
        ),
        (
            "[u8; 16], seed 0",
            gxhash_path(&[0xa5_u8; 16], 0),
            0x89d6_bb2f_e539_9be3,
        ),
        (
            "[u8; 16]",
            gxhash_path(&[0xa5_u8; 16], SEED),
            0x8a1f_de7d_9521_94a9,
        ),
        (
            "0 bytes, seed 0",
            gxhash_path(&bytes(0), 0),
            0x7c27_6c31_7a1a_784f,
        ),
        (
            "0 bytes",
            gxhash_path(&bytes(0), SEED),
            0x8a1a_14d2_0d97_345b,
        ),
        (
            "8 bytes",
            gxhash_path(&bytes(8), SEED),
            0x278d_2161_6a70_21c9,
        ),
        (
            "16 bytes",
            gxhash_path(&bytes(16), SEED),
            0x14c1_742c_422d_ec9a,
        ),
        (
            "20 bytes",
            gxhash_path(&bytes(20), SEED),
            0xf68d_142b_1495_526f,
        ),
        (
            "60 bytes",
            gxhash_path(&bytes(60), SEED),
            0x7196_7502_2840_ac4d,
        ),
        (
            "1000 bytes",
            gxhash_path(&bytes(1000), SEED),
            0xd6ae_f85e_5f52_a7fb,
        ),
    ];
    for (key, got, want) in cases {
        assert_eq!(got, want, "{key}: got {got:#018x}");
    }
}

#[test]
fn pk_hash_fallback_matches_golden_values() {
    // `Hash` prefixes slices, arrays and `Vec`s with an 8-byte length, so the
    // `Vec<u8>` lengths reach every xxh3 input-size path: 4-8, 9-16, 17-128,
    // 129-240, over 240 and over 1024 bytes, where xxh3 starts scrambling its
    // accumulators once per block. Ingest never takes this path for `u64` or
    // `[u8; 16]` keys; those cases cost nothing and stay.
    let cases: [(&str, u64, u64); 12] = [
        (
            "u64, seed 0",
            fallback(&0x0123_4567_89ab_cdef_u64, 0),
            0xb78d_f414_2842_77a6,
        ),
        (
            "u64",
            fallback(&0x0123_4567_89ab_cdef_u64, SEED),
            0x853d_75da_fb24_4901,
        ),
        (
            "[u8; 16], seed 0",
            fallback(&[0xa5_u8; 16], 0),
            0x2c30_72b7_8c58_acb3,
        ),
        (
            "[u8; 16]",
            fallback(&[0xa5_u8; 16], SEED),
            0x93b5_8bc1_ad33_e3ff,
        ),
        (
            "[u8; 32], seed 0",
            fallback(&[0x5a_u8; 32], 0),
            0xb9f5_ffc1_20ef_0b30,
        ),
        (
            "[u8; 32]",
            fallback(&[0x5a_u8; 32], SEED),
            0x052c_8174_70a1_4254,
        ),
        ("0 bytes", fallback(&bytes(0), SEED), 0x1a00_6557_2063_04f9),
        ("8 bytes", fallback(&bytes(8), SEED), 0x7bcd_4557_49be_9b79),
        (
            "100 bytes",
            fallback(&bytes(100), SEED),
            0x9e8f_894f_7f71_27cb,
        ),
        (
            "200 bytes",
            fallback(&bytes(200), SEED),
            0x6845_01d3_866e_0dcf,
        ),
        (
            "1000 bytes",
            fallback(&bytes(1000), SEED),
            0xfafe_4095_9550_08e6,
        ),
        (
            "2000 bytes",
            fallback(&bytes(2000), SEED),
            0x4879_3f62_c434_7704,
        ),
    ];
    for (key, got, want) in cases {
        assert_eq!(got, want, "{key}: got {got:#018x}");
    }
}

#[cfg(feature = "epserde")]
#[test]
fn epserde_header_hashes_match_golden_values() {
    use crate::DefaultPtrHash;
    use epserde::prelude::Serialize;

    /// The `TYPE_HASH` and `REPR_HASH` header fields epserde writes for `T`.
    fn header_hashes<T: Serialize>(value: &T) -> [u64; 2] {
        let mut buf = Vec::new();
        let schema = value.serialize_with_schema(&mut buf).unwrap();
        ["TYPE_HASH", "REPR_HASH"].map(|field| {
            let row = schema.0.iter().find(|row| row.field == field).unwrap();
            u64::from_ne_bytes(buf[row.offset..][..8].try_into().unwrap())
        })
    }

    // The index key types DiskGraph serializes.
    let cases: [(&str, [u64; 2], [u64; 2]); 4] = [
        (
            "u64",
            header_hashes(&DefaultPtrHash::<PkHash, u64>::default()),
            [0x4c5d_2730_c662_a438, 0x9bd4_9f93_9a3a_4229],
        ),
        (
            "Vec<u8>",
            header_hashes(&DefaultPtrHash::<PkHash, Vec<u8>>::default()),
            [0x92f3_8094_abf4_f9ec, 0x9bd4_9f93_9a3a_4229],
        ),
        (
            "[u8; 16]",
            header_hashes(&DefaultPtrHash::<PkHash, [u8; 16]>::default()),
            [0x3f25_4c72_b1c5_a213, 0x9bd4_9f93_9a3a_4229],
        ),
        (
            "[u8; 32]",
            header_hashes(&DefaultPtrHash::<PkHash, [u8; 32]>::default()),
            [0xc71c_d03e_a8ca_12ef, 0x9bd4_9f93_9a3a_4229],
        ),
    ];
    for (key, got, want) in cases {
        assert_eq!(got, want, "{key}: got [{:#018x}, {:#018x}]", got[0], got[1]);
    }
}
