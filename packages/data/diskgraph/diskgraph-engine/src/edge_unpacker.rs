// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use diskgraph_index::layout::RelationLayout;

/// Optimized helper to unpack edges from a binary blob.
pub struct EdgeUnpacker<'a> {
    ptr: *const u8,
    end: *const u8,
    entry_bytes: usize,
    dst_mask: u64,
    dst_bits: u8,
    sort_key_mask: u64,
    // Phantom lifetime to tie ptr to data
    _marker: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> EdgeUnpacker<'a> {
    pub fn new(data: &'a [u8], layout: &RelationLayout) -> Self {
        let entry_bytes = layout.entry_bytes;
        // Safety: We only access within `data` range.
        Self {
            ptr: data.as_ptr(),
            end: unsafe { data.as_ptr().add(data.len()) },
            entry_bytes,
            dst_mask: layout.dst_mask,
            dst_bits: layout.dst_bits,
            sort_key_mask: layout.dest_sort_key_mask.unwrap_or(0),
            _marker: std::marker::PhantomData,
        }
    }

    /// Unpack a batch of edges into the provided buffers.
    /// Returns the number of edges unpacked.
    #[inline(always)]
    pub fn unpack_batch(&mut self, dst_out: &mut [u64], key_out: &mut [u64]) -> usize {
        let count = dst_out.len().min(key_out.len());
        let mut n = 0;

        // Optimization: Branch on entry size once, then loop
        if self.entry_bytes <= 8 {
            // Guard against dst_bits >= 64: shifting a u64 by 64 or more is
            // a debug-mode panic (and UB in release). dst_bits=64 happens
            // when compute_relation_layouts falls back for a dst_type that
            // isn't visible to the layout-computation scope (e.g., a seed
            // sub-index opened standalone where the parent's users table
            // is absent from metadata.node_types). When dst_bits >= 64 the
            // whole u64 is the dst id, and there can't be sort-key bits
            // alongside, so the key is unconditionally 0.
            let sort_key_shift_safe = self.dst_bits < 64;
            while n < count && unsafe { self.ptr.add(self.entry_bytes) <= self.end } {
                let raw = unsafe { read_u64_partial(self.ptr, self.entry_bytes) };

                unsafe {
                    *dst_out.get_unchecked_mut(n) = raw & self.dst_mask;
                    *key_out.get_unchecked_mut(n) = if sort_key_shift_safe {
                        (raw >> self.dst_bits) & self.sort_key_mask
                    } else {
                        0
                    };

                    self.ptr = self.ptr.add(self.entry_bytes);
                }
                n += 1;
            }
        } else {
            // Fallback for > 64 bit entries (rare)
            while n < count && unsafe { self.ptr.add(self.entry_bytes) <= self.end } {
                let raw = unsafe { read_u128_partial(self.ptr, self.entry_bytes) };

                unsafe {
                    *dst_out.get_unchecked_mut(n) = (raw & self.dst_mask as u128) as u64;
                    *key_out.get_unchecked_mut(n) =
                        ((raw >> self.dst_bits) & self.sort_key_mask as u128) as u64;

                    self.ptr = self.ptr.add(self.entry_bytes);
                }
                n += 1;
            }
        }
        n
    }

    /// Number of leading entries whose sort key satisfies `pred`, which must be
    /// true on a prefix and false after it. Entry 0 is checked first, so an empty
    /// prefix only touches the front of the segment.
    pub fn partition_point(&self, mut pred: impl FnMut(u64) -> bool) -> usize {
        if self.entry_bytes == 0 {
            return 0;
        }
        let len = (self.end as usize - self.ptr as usize) / self.entry_bytes;
        if len == 0 || !pred(self.key_at(0)) {
            return 0;
        }
        let (mut lo, mut hi) = (1, len);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pred(self.key_at(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    #[inline]
    fn key_at(&self, index: usize) -> u64 {
        unsafe {
            let ptr = self.ptr.add(index * self.entry_bytes);
            if self.entry_bytes <= 8 {
                if self.dst_bits >= 64 {
                    return 0;
                }
                (read_u64_partial(ptr, self.entry_bytes) >> self.dst_bits) & self.sort_key_mask
            } else {
                ((read_u128_partial(ptr, self.entry_bytes) >> self.dst_bits)
                    & self.sort_key_mask as u128) as u64
            }
        }
    }

    /// Skip N items (advancing pointer)
    #[inline]
    pub fn advance(&mut self, n: usize) {
        let bytes = n * self.entry_bytes;
        unsafe {
            if self.ptr.add(bytes) <= self.end {
                self.ptr = self.ptr.add(bytes);
            } else {
                self.ptr = self.end;
            }
        }
    }
}

// Unsafe helper: reads 1..8 bytes into u64.
// Assumes ptr is valid for `len` bytes.
#[inline(always)]
unsafe fn read_u64_partial(ptr: *const u8, len: usize) -> u64 {
    let mut buf = [0u8; 8];
    // For small constant lens (common in graphs), compiler optimizes this memcpy
    std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), len);
    u64::from_le_bytes(buf)
}

// Unsafe helper: reads 1..16 bytes into u128.
#[inline(always)]
unsafe fn read_u128_partial(ptr: *const u8, len: usize) -> u128 {
    let mut buf = [0u8; 16];
    std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), len);
    u128::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskgraph_index::layout::RelationLayout;

    #[test]
    fn test_unpack_batch() {
        // Layout: 4 bits dst, 4 bits sort_key -> 1 byte entry
        let layout = RelationLayout {
            relation_id: 0,
            dst_bits: 4,
            dst_mask: 0xF,
            dest_sort_key_bits: Some(4),
            dest_sort_key_mask: Some(0xF),
            entry_bytes: 1,
        };

        // Entries:
        // 1. dst=1 (0001), key=2 (0010) -> 0x21
        // 2. dst=15(1111), key=0 (0000) -> 0x0F
        // 3. dst=3 (0011), key=1 (0001) -> 0x13
        let data = vec![0x21, 0x0F, 0x13];

        let mut unpacker = EdgeUnpacker::new(&data, &layout);
        let mut dsts = [0u64; 2];
        let mut keys = [0u64; 2];

        // First batch (2 items)
        let n = unpacker.unpack_batch(&mut dsts, &mut keys);
        assert_eq!(n, 2);
        assert_eq!(dsts, [1, 15]);
        assert_eq!(keys, [2, 0]);

        // Second batch (1 item remaining)
        let n = unpacker.unpack_batch(&mut dsts, &mut keys);
        assert_eq!(n, 1);
        assert_eq!(dsts[0], 3);
        assert_eq!(keys[0], 1);

        // End
        let n = unpacker.unpack_batch(&mut dsts, &mut keys);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_unpack_batch_u128() {
        // Layout: 60 bits dst, 60 bits sort_key -> 15 bytes (needs u128 path)
        let layout = RelationLayout {
            relation_id: 0,
            dst_bits: 60,
            dst_mask: (1u64 << 60) - 1,
            dest_sort_key_bits: Some(60),
            dest_sort_key_mask: Some((1u64 << 60) - 1),
            entry_bytes: 15,
        };

        // Construct 15-byte value
        // Dst: 1, Key: 2
        // Packed: (2 << 60) | 1 = 2305843009213693953 (approx)
        let val: u128 = (2u128 << 60) | 1u128;
        let mut data = vec![0u8; 15];
        // Write little endian 15 bytes
        let bytes = val.to_le_bytes();
        data.copy_from_slice(&bytes[..15]);

        let mut unpacker = EdgeUnpacker::new(&data, &layout);
        let mut dsts = [0u64; 1];
        let mut keys = [0u64; 1];

        let n = unpacker.unpack_batch(&mut dsts, &mut keys);
        assert_eq!(n, 1);
        assert_eq!(dsts[0], 1);
        assert_eq!(keys[0], 2);
    }

    #[test]
    fn test_advance() {
        let layout = RelationLayout {
            relation_id: 0,
            dst_bits: 8,
            dst_mask: 0xFF,
            dest_sort_key_bits: None,
            dest_sort_key_mask: None,
            entry_bytes: 1,
        };
        // 0, 1, 2, 3, 4
        let data = vec![0, 1, 2, 3, 4];
        let mut unpacker = EdgeUnpacker::new(&data, &layout);

        // Skip 2 (points to 2)
        unpacker.advance(2);

        let mut dsts = [0u64; 1];
        let mut keys = [0u64; 1]; // keys will be 0

        // Read 1 (should be 2)
        let n = unpacker.unpack_batch(&mut dsts, &mut keys);
        assert_eq!(n, 1);
        assert_eq!(dsts[0], 2);

        // Skip 1 (points to 4)
        unpacker.advance(1);

        // Read 1 (should be 4)
        let n = unpacker.unpack_batch(&mut dsts, &mut keys);
        assert_eq!(n, 1);
        assert_eq!(dsts[0], 4);
    }

    #[test]
    fn test_partition_point_matches_linear_count() {
        let layout = RelationLayout {
            relation_id: 0,
            dst_bits: 8,
            dst_mask: 0xFF,
            dest_sort_key_bits: Some(8),
            dest_sort_key_mask: Some(0xFF),
            entry_bytes: 2,
        };
        for len in 0..70u64 {
            let data: Vec<u8> = (0..len).flat_map(|i| [0, (len - i) as u8]).collect();
            for cut in 0..=len + 1 {
                let unpacker = EdgeUnpacker::new(&data, &layout);
                let expected = (1..=len).filter(|&key| key >= cut).count();
                assert_eq!(unpacker.partition_point(|key| key >= cut), expected);
            }
        }
    }

    #[test]
    fn test_truncation() {
        let layout = RelationLayout {
            relation_id: 0,
            dst_bits: 8,
            dst_mask: 0xFF,
            dest_sort_key_bits: None,
            dest_sort_key_mask: None,
            entry_bytes: 2, // requires 2 bytes per entry
        };
        // Only 3 bytes provided -> 1 full entry, 1 partial (should be ignored)
        let data = vec![1, 0, 2];

        let mut unpacker = EdgeUnpacker::new(&data, &layout);
        let mut dsts = [0u64; 2];
        let mut keys = [0u64; 2];

        let n = unpacker.unpack_batch(&mut dsts, &mut keys);
        assert_eq!(n, 1); // Only 1 full entry unpacked
        assert_eq!(dsts[0], 1); // 1 | 0<<8 = 1
    }
}
