// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reject frozen epserde 0.8 indexes and round-trip the current format.

mod common;

use epserde::prelude::{Deserialize, Flags, Serialize};
use ptr_hash::hash::{GxHash128, PkHash};
use ptr_hash::{DefaultPtrHash, PtrHashParams, PtrHashScratch};
use std::path::Path;

macro_rules! legacy_index {
    ($name:ident, $hasher:ty, $key:ty, $keys:expr) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            type Index = DefaultPtrHash<$hasher, $key>;
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/epserde-0.8");
            let path = root.join(concat!(stringify!($name), ".bin"));
            // SAFETY: these are frozen, valid epserde 0.8 files. All loaders
            // must reject their version before interpreting the payload.
            let legacy = unsafe {
                [
                    Index::mmap(&path, Flags::empty()).err(),
                    Index::load_mem(&path).err(),
                    Index::load_full(&path).err(),
                ]
            };
            for error in legacy {
                let error = error.expect("legacy index must require rebuilding");
                assert!(matches!(
                    error.downcast_ref::<epserde::deser::Error>(),
                    Some(epserde::deser::Error::MajorVersionMismatch(1))
                ));
            }

            let keys = $keys;
            let index = Index::try_new_from_source_with_scratch(
                &keys,
                PtrHashParams::default(),
                &mut PtrHashScratch::new(),
            )
            .expect("current index construction failed");
            let output_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../target/test-artifacts/epserde-compat");
            std::fs::create_dir_all(&output_dir)?;
            let output = output_dir.join(concat!(stringify!($name), ".bin"));
            // SAFETY: the index is constructed from valid keys. Its serialized
            // file remains immutable for the lifetime of all mapped views.
            let (mapped, memory, full) = unsafe {
                index.store(&output)?;
                (
                    Index::mmap(&output, Flags::empty())?,
                    Index::load_mem(&output)?,
                    Index::load_full(&output)?,
                )
            };
            for key in &keys.0 {
                let expected = index.index_no_remap(key);
                assert_eq!(mapped.uncase().index_no_remap(key), expected);
                assert_eq!(memory.uncase().index_no_remap(key), expected);
                assert_eq!(full.index_no_remap(key), expected);
            }
            let bytes = std::fs::read(&output)?;
            for missing in 1..=32.min(bytes.len()) {
                let truncated =
                    output_dir.join(format!("{}-truncated-{missing}.bin", stringify!($name),));
                std::fs::write(&truncated, &bytes[..bytes.len() - missing])?;
                // SAFETY: the payload was serialized from valid fields and
                // only shortened. Deserialization must reject missing bytes.
                let result = unsafe { Index::mmap(&truncated, Flags::empty()) };
                assert!(result.is_err(), "accepted {missing} missing bytes");
            }
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
