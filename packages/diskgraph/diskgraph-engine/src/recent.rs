// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::errors::{EngineError, Result};
use diskgraph_rowpack::ValueRef;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Lossless ordering within one declared PK column. Integer keys never pass
/// through floating point; variable-width keys retain their complete bytes.
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum RecentPrimaryKey {
    Integer(i128),
    Bytes(Vec<u8>),
    Float(u64),
}

impl RecentPrimaryKey {
    pub(crate) fn from_value(value: ValueRef<'_>) -> Result<Self> {
        Ok(match value {
            ValueRef::Int64(value) | ValueRef::Timestamp(value) => Self::Integer(value.into()),
            ValueRef::Int32(value) | ValueRef::Date(value) => Self::Integer(value.into()),
            ValueRef::UInt64(value) => Self::Integer(value.into()),
            ValueRef::UInt32(value) => Self::Integer(value.into()),
            ValueRef::Decimal128(value) => Self::Integer(value),
            ValueRef::Bool(value) => Self::Integer(i128::from(value)),
            ValueRef::String(value) => Self::Bytes(value.as_bytes().to_vec()),
            ValueRef::Bytes(value) => Self::Bytes(value.to_vec()),
            ValueRef::Uuid(value) | ValueRef::Ip(value) => Self::Bytes(value.to_vec()),
            ValueRef::Uuid256(value) => Self::Bytes(value.to_vec()),
            ValueRef::Decimal256(mut value) => {
                // Rowpack decimals are little-endian two's-complement integers.
                value.reverse();
                value[0] ^= 0x80;
                Self::Bytes(value.to_vec())
            }
            ValueRef::Float32(value) => Self::float(value.into())?,
            ValueRef::Float64(value) => Self::float(value)?,
            _ => {
                return Err(EngineError::Configuration(
                    "recent sampling requires a non-null scalar primary key".into(),
                ))
            }
        })
    }

    fn float(value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(EngineError::Configuration(
                "recent sampling requires a finite primary key".into(),
            ));
        }
        let bits = value.to_bits();
        Ok(Self::Float(if bits >> 63 == 1 {
            !bits
        } else {
            bits ^ (1 << 63)
        }))
    }
}

#[derive(Debug)]
struct Candidate {
    key: RecentPrimaryKey,
    node_id: u64,
}

// Equal logical identities are equivalent candidates. Do not introduce the
// physical node ID as a secondary ordering rule. Repeated edges retain their
// multiplicity, including when several retained candidates have the same PK.
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

pub(crate) struct RecentGroup {
    capacity: usize,
    nodes: Vec<u64>,
    selected: Option<BinaryHeap<Reverse<Candidate>>>,
}

impl RecentGroup {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            nodes: Vec::new(),
            selected: None,
        }
    }

    pub(crate) fn push(
        &mut self,
        node_id: u64,
        mut primary_key: impl FnMut(u64) -> Result<RecentPrimaryKey>,
    ) -> Result<()> {
        if self.selected.is_none() && self.nodes.len() < self.capacity {
            self.nodes.push(node_id);
            return Ok(());
        }
        if self.selected.is_none() {
            let mut selected = BinaryHeap::new();
            for id in self.nodes.drain(..) {
                selected.push(Reverse(Candidate {
                    key: primary_key(id)?,
                    node_id: id,
                }));
            }
            self.selected = Some(selected);
        }
        let selected = self.selected.as_mut().expect("initialized above");
        let candidate = Candidate {
            key: primary_key(node_id)?,
            node_id,
        };
        if let Some(mut smallest) = selected.peek_mut() {
            if candidate > smallest.0 {
                *smallest = Reverse(candidate);
            }
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Vec<u64> {
        match self.selected {
            Some(selected) => selected
                .into_sorted_vec()
                .into_iter()
                .map(|Reverse(candidate)| candidate.node_id)
                .collect(),
            None => self.nodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recent_group_uses_exact_keys_and_preserves_duplicate_edges() {
        let keys = [1_i128, (1_i128 << 63) + 1, (1_i128 << 63) + 2];
        for order in [[0, 1, 2, 2], [2, 2, 1, 0], [1, 2, 0, 2]] {
            let mut group = RecentGroup::new(3);
            for id in order {
                group
                    .push(id, |id| Ok(RecentPrimaryKey::Integer(keys[id as usize])))
                    .unwrap();
                assert!(group.nodes.len() <= 3);
                assert!(group.selected.as_ref().is_none_or(|heap| heap.len() <= 3));
            }
            assert_eq!(group.finish(), vec![2, 2, 1]);
        }
    }

    #[test]
    fn recent_full_group_does_not_read_primary_keys() {
        let mut group = RecentGroup::new(3);
        for id in [3, 1, 2] {
            group
                .push(id, |_| panic!("uncapped group must not fetch PKs"))
                .unwrap();
        }
        assert_eq!(group.finish(), vec![3, 1, 2]);
    }

    #[test]
    fn recent_primary_keys_preserve_numeric_and_byte_order() {
        let key = |value| RecentPrimaryKey::from_value(value).unwrap();
        assert!(key(ValueRef::UInt64((1 << 63) + 2)) > key(ValueRef::UInt64((1 << 63) + 1)));
        assert!(key(ValueRef::Int64(10)) > key(ValueRef::Int64(9)));
        assert!(key(ValueRef::Int64(0)) > key(ValueRef::Int64(-1)));
        assert!(key(ValueRef::String("z")) > key(ValueRef::String("aa")));
        assert!(key(ValueRef::Bytes(b"a\0z")) > key(ValueRef::Bytes(b"a\0b")));
        assert!(key(ValueRef::Decimal256([0; 32])) > key(ValueRef::Decimal256([255; 32])));
        assert!(key(ValueRef::Float64(-1.0)) > key(ValueRef::Float64(-2.0)));
        assert!(RecentPrimaryKey::from_value(ValueRef::Null).is_err());
        assert!(RecentPrimaryKey::from_value(ValueRef::Float64(f64::NAN)).is_err());
    }
}
