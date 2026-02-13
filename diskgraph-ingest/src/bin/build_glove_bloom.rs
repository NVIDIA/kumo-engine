//! Offline tool to build a bloom filter from a GloVe vocabulary text file.
//!
//! Usage: cargo run --release --bin build_glove_bloom -- <vocab.txt> <output.bin>
//!
//! The output binary is a raw bit array suitable for `include_bytes!` in col_stats.rs.
//! Hash functions must match `BloomFilter::hash_pair` in col_stats.rs exactly.

use std::io::{BufRead, Write};

/// FNV-1a with 64-bit output and configurable seed.
#[inline]
fn fnv1a_64(bytes: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

const SEED1: u64 = 0x517cc1b727220a95;
const SEED2: u64 = 0x6c62272e07bb0142;

/// Bloom filter parameters (must match col_stats.rs).
const NUM_BITS: u64 = 40_000_000;
const NUM_HASHES: u32 = 7;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <vocab.txt> <output.bin>", args[0]);
        std::process::exit(1);
    }

    let num_bytes = NUM_BITS.div_ceil(8) as usize;
    let mut bits = vec![0u8; num_bytes];

    let file = std::fs::File::open(&args[1]).expect("Failed to open vocab file");
    let reader = std::io::BufReader::new(file);
    let mut count = 0u64;

    for line in reader.lines() {
        let word = line.expect("Failed to read line");
        let word = word.trim();
        if word.is_empty() {
            continue;
        }

        let h1 = fnv1a_64(word.as_bytes(), SEED1);
        let h2 = fnv1a_64(word.as_bytes(), SEED2);

        for i in 0..NUM_HASHES {
            let bit_idx = h1.wrapping_add((i as u64).wrapping_mul(h2)) % NUM_BITS;
            let byte_idx = (bit_idx / 8) as usize;
            let bit_offset = (bit_idx % 8) as u8;
            bits[byte_idx] |= 1 << bit_offset;
        }
        count += 1;
    }

    let mut out = std::fs::File::create(&args[2]).expect("Failed to create output file");
    out.write_all(&bits).expect("Failed to write bloom filter");

    let set_bits: u64 = bits.iter().map(|b| b.count_ones() as u64).sum();
    let fill_ratio = set_bits as f64 / NUM_BITS as f64;
    let estimated_fpr = fill_ratio.powi(NUM_HASHES as i32);

    eprintln!("Inserted {count} words into bloom filter");
    eprintln!("Bits: {NUM_BITS}, Hashes: {NUM_HASHES}");
    eprintln!("Bytes written: {num_bytes}");
    eprintln!("Fill ratio: {fill_ratio:.4}");
    eprintln!("Estimated FPR: {estimated_fpr:.6}");
}
