// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read frozen indexes written with the release's unpatched epserde 0.8.0.

mod common;

use epserde::prelude::{Deserialize, Flags, Serialize};
use ptr_hash::hash::{GxHash128, PkHash};
use ptr_hash::DefaultPtrHash;
use std::path::Path;

macro_rules! legacy_index {
    ($name:ident, $hasher:ty, $key:ty, $keys:expr) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            type Index = DefaultPtrHash<$hasher, $key>;
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/epserde-0.8");
            let path = root.join(concat!(stringify!($name), ".bin"));
            let expected: Vec<usize> = serde_json::from_slice(&std::fs::read(
                root.join(concat!(stringify!($name), ".json")),
            )?)?;
            let keys = $keys;
            assert_eq!(expected.len(), keys.0.len());
            let mapped = Index::load_mmap(&path, Flags::empty())?;
            let memory = Index::load_mem(&path)?;
            let full = Index::load_full(&path)?;
            for (key, expected_index) in keys.0.iter().zip(expected) {
                assert_eq!(mapped.index(key), expected_index, "mmap lookup: {key:?}");
                assert_eq!(memory.index(key), expected_index, "heap lookup: {key:?}");
                assert_eq!(full.index(key), expected_index, "full lookup: {key:?}");
            }
            let output = tempfile::NamedTempFile::new()?;
            full.store(output.path())?;
            assert_eq!(
                std::fs::read(output.path())?,
                std::fs::read(path)?,
                "re-saving a legacy index changes its bytes"
            );
            Ok(())
        }
    };
}

legacy_index!(pk_int64, PkHash, u64, common::integers());
legacy_index!(pk_bytes, PkHash, Vec<u8>, common::bytes());
legacy_index!(pk_fixed16, PkHash, [u8; 16], common::fixed::<16>());
legacy_index!(pk_fixed32, PkHash, [u8; 32], common::fixed::<32>());
legacy_index!(gx_int64, GxHash128, u64, common::integers());
legacy_index!(gx_bytes, GxHash128, Vec<u8>, common::bytes());
legacy_index!(gx_fixed16, GxHash128, [u8; 16], common::fixed::<16>());
legacy_index!(gx_fixed32, GxHash128, [u8; 32], common::fixed::<32>());
