use crate::{bucket_idx::BucketIdx, Hash};
use bitvec::vec::BitVec;
use std::collections::BinaryHeap;

pub struct PtrHashScratch<H: Hash> {
    pub hashes: Vec<H>,
    pub hashes_tmp: Vec<H>,
    pub part_starts: Vec<u32>,
    pub pilots: Vec<u8>,
    pub taken: Vec<BitVec>,
    pub bucket_starts: Vec<u32>,
    pub bucket_order: Vec<BucketIdx>,
    pub bucket_len_cnt: Vec<usize>,
    pub slots: Vec<BucketIdx>,
    pub slots_tmp: Vec<usize>,
    pub free_slots: Vec<u64>,
    pub stack: BinaryHeap<(usize, BucketIdx)>,
}

impl<H: Hash> PtrHashScratch<H> {
    pub fn new() -> Self {
        Self {
            hashes: Vec::new(),
            hashes_tmp: Vec::new(),
            part_starts: Vec::new(),
            pilots: Vec::new(),
            taken: Vec::new(),
            bucket_starts: Vec::new(),
            bucket_order: Vec::new(),
            bucket_len_cnt: Vec::new(),
            slots: Vec::new(),
            slots_tmp: Vec::new(),
            free_slots: Vec::new(),
            stack: BinaryHeap::new(),
        }
    }
}
