// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run only against the unpatched release dependencies to create the fixtures.

#[path = "../tests/common/mod.rs"]
mod common;

use epserde::prelude::Serialize;
use ptr_hash::hash::{GxHash128, PkHash};
use ptr_hash::{DefaultPtrHash, PtrHashParams, PtrHashScratch};
use std::path::Path;

macro_rules! write_index {
    ($name:ident, $hasher:ty, $key:ty, $keys:expr, $root:expr) => {{
        let keys = $keys;
        let index = DefaultPtrHash::<$hasher, $key>::try_new_from_source_with_scratch(
            &keys,
            PtrHashParams::default(),
            &mut PtrHashScratch::new(),
        )
        .expect("legacy index construction failed");
        index.store($root.join(concat!(stringify!($name), ".bin")))?;
        let expected: Vec<_> = keys.0.iter().map(|key| index.index(key)).collect();
        std::fs::write(
            $root.join(concat!(stringify!($name), ".json")),
            serde_json::to_vec_pretty(&expected)?,
        )?;
    }};
}

fn main() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/epserde-0.8");
    std::fs::create_dir_all(&root)?;
    write_index!(pk_int64, PkHash, u64, common::integers(), root);
    write_index!(pk_bytes, PkHash, Vec<u8>, common::bytes(), root);
    write_index!(pk_fixed16, PkHash, [u8; 16], common::fixed::<16>(), root);
    write_index!(pk_fixed32, PkHash, [u8; 32], common::fixed::<32>(), root);
    write_index!(gx_int64, GxHash128, u64, common::integers(), root);
    write_index!(gx_bytes, GxHash128, Vec<u8>, common::bytes(), root);
    write_index!(gx_fixed16, GxHash128, [u8; 16], common::fixed::<16>(), root);
    write_index!(gx_fixed32, GxHash128, [u8; 32], common::fixed::<32>(), root);
    Ok(())
}
