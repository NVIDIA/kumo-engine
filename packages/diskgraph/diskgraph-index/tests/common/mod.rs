// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use ptr_hash::{KeySource, KeyT};

pub struct Keys<T>(pub Vec<T>);

impl<T: KeyT> KeySource<T> for Keys<T> {
    type Iter<'a>
        = std::slice::Iter<'a, T>
    where
        Self: 'a,
        T: 'a;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter()
    }
}

pub fn integers() -> Keys<u64> {
    Keys((0..64).map(|i| i * 97).collect())
}

pub fn bytes() -> Keys<Vec<u8>> {
    Keys(
        (0..64)
            .map(|i| format!("legacy-index-long-key-{i:04}").into_bytes())
            .collect(),
    )
}

pub fn fixed<const N: usize>() -> Keys<[u8; N]> {
    Keys(
        (0_u8..64)
            .map(|i| {
                let mut key = [0; N];
                key[0] = i;
                key[N - 1] = 255 - i;
                key
            })
            .collect(),
    )
}
