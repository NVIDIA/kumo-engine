use diskgraph_index::layout::RelationLayout;

/// Optimized helper to unpack edges from a binary blob.
pub struct EdgeUnpacker<'a> {
    ptr: *const u8,
    end: *const u8,
    layout: &'a RelationLayout,
    entry_bytes: usize,
    dst_mask: u64,
    dst_bits: u8,
    sort_key_mask: u64,
    // Phantom lifetime to tie ptr to data
    _marker: std::marker::PhantomData<&'a [u8]>,
}

impl<'a> EdgeUnpacker<'a> {
    pub fn new(data: &'a [u8], layout: &'a RelationLayout) -> Self {
        let entry_bytes = layout.entry_bytes;
        // Safety: We only access within `data` range.
        Self {
            ptr: data.as_ptr(),
            end: unsafe { data.as_ptr().add(data.len()) },
            layout,
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
            while n < count && unsafe { self.ptr.add(self.entry_bytes) <= self.end } {
                let raw = unsafe { read_u64_partial(self.ptr, self.entry_bytes) };

                unsafe {
                    *dst_out.get_unchecked_mut(n) = raw & self.dst_mask;
                    *key_out.get_unchecked_mut(n) = (raw >> self.dst_bits) & self.sort_key_mask;

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
                    *key_out.get_unchecked_mut(n) = ((raw >> self.dst_bits) & self.sort_key_mask as u128) as u64;

                    self.ptr = self.ptr.add(self.entry_bytes);
                }
                n += 1;
            }
        }
        n
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
