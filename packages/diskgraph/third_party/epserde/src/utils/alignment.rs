// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte storage with equal size and alignment. No custom allocator is used.

use mem_dbg::{MemDbg, MemSize};

mod sealed {
    pub trait Sealed {}
}

/// A nonempty byte array with no padding and all byte patterns valid.
///
/// Only this module implements the trait, so AlignedCursor can expose its
/// storage as bytes without allowing invalid values or uninitialized padding.
pub trait Alignment: sealed::Sealed + Sized + Clone + Default + Send + Sync {}

macro_rules! alignment {
    ($name:ident, $size:literal) => {
        #[doc = concat!("A byte array with size and alignment ", stringify!($size), ".")]
        #[repr(C, align($size))]
        #[derive(Clone, Copy, Debug, MemDbg, MemSize)]
        pub struct $name(pub [u8; $size]);

        impl Default for $name {
            fn default() -> Self {
                Self([0; $size])
            }
        }

        impl sealed::Sealed for $name {}
        impl Alignment for $name {}
    };
}

alignment!(A16, 16);
alignment!(A64, 64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::AlignedCursor;
    use std::io::Write;

    fn grows_without_losing_alignment<T: Alignment>() {
        let mut cursor = AlignedCursor::<T>::new();
        for i in 0_u8..64 {
            cursor.write_all(&[i; 37]).unwrap();
            assert_eq!(
                cursor.as_bytes().as_ptr() as usize % std::mem::align_of::<T>(),
                0
            );
        }
        assert_eq!(std::mem::size_of::<T>(), std::mem::align_of::<T>());
        let expected: Vec<u8> = (0_u8..64).flat_map(|i| [i; 37]).collect();
        assert_eq!(cursor.as_bytes(), expected);
    }

    #[test]
    fn aligned_storage_16() {
        grows_without_losing_alignment::<A16>();
    }

    #[test]
    fn aligned_storage_64() {
        grows_without_losing_alignment::<A64>();
    }
}
