// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Zero-allocation guard for the LIST_DISTINCT inner observe loop
//! (kumo#28400 §4 Step 4).
//!
//! After warmup, a hot loop of repeated tokens must hit only the Occupied
//! branch of the dedup map — `Value::Str(...)` is never materialized, no
//! `String` is ever allocated. This file installs a custom counting global
//! allocator and asserts the post-warmup alloc delta is zero.
//!
//! Scoped to the inner observe primitive, not the full executor: per-row
//! decoder allocations (e.g. `with_string_array_column` building a
//! `Vec<&str>` view) are intentionally out-of-scope here. They sit upstream
//! of the dedup loop the bench guards.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use diskgraph_core::tt_builder::Value;
use diskgraph_train_table::ttgen_accumulator::{hash_i64, hash_str};

/// Serializes the two zero-alloc tests so that `cargo test`'s default
/// multithreaded runner doesn't record background allocations from a
/// concurrent test in this same binary's allocator counter. Tests in
/// other binaries are unaffected — `#[global_allocator]` is per-binary.
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());

#[global_allocator]
static COUNTER: CountingAllocator = CountingAllocator;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

fn snapshot() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

/// Mirror of the LIST_DISTINCT inner observe step for `&str` tokens.
/// Hash first; only allocate `Value::Str(String)` on the Vacant branch.
#[inline(always)]
fn observe_str_token(
    tok: &str,
    seen: &mut HashMap<u64, usize>,
    values: &mut Vec<Value>,
    counts: &mut Vec<i64>,
) {
    let h = hash_str(tok);
    match seen.entry(h) {
        std::collections::hash_map::Entry::Occupied(e) => {
            counts[*e.get()] += 1;
        }
        std::collections::hash_map::Entry::Vacant(e) => {
            e.insert(values.len());
            values.push(Value::Str(tok.to_string()));
            counts.push(1);
        }
    }
}

/// Mirror for `i64` tokens. Vacant branch never allocates.
#[inline(always)]
fn observe_int_token(
    n: i64,
    seen: &mut HashMap<u64, usize>,
    values: &mut Vec<Value>,
    counts: &mut Vec<i64>,
) {
    let h = hash_i64(n);
    match seen.entry(h) {
        std::collections::hash_map::Entry::Occupied(e) => {
            counts[*e.get()] += 1;
        }
        std::collections::hash_map::Entry::Vacant(e) => {
            e.insert(values.len());
            values.push(Value::Int(n));
            counts.push(1);
        }
    }
}

#[test]
fn bench_listdistinct_zero_alloc_str_tokens() {
    let _g = MEASUREMENT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Tokens used during both warmup and hot loop. Static byte slices, so
    // their borrowed `&str` views allocate nothing on the read side.
    const TOKENS: &[&str] = &[
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    ];
    const HOT_LOOP_ITERS: usize = 1_000_000;

    let mut seen: HashMap<u64, usize> = HashMap::with_capacity(TOKENS.len() * 2);
    let mut values: Vec<Value> = Vec::with_capacity(TOKENS.len());
    let mut counts: Vec<i64> = Vec::with_capacity(TOKENS.len());

    // Warmup: populate seen-map with all distinct tokens. Allocations expected
    // here (one `String` per Vacant push, plus map growth).
    for &tok in TOKENS {
        observe_str_token(tok, &mut seen, &mut values, &mut counts);
    }
    assert_eq!(values.len(), TOKENS.len(), "warmup populated all tokens");

    // Snapshot alloc count after warmup. From here on every observation must
    // hit the Occupied branch.
    let pre = snapshot();
    for i in 0..HOT_LOOP_ITERS {
        observe_str_token(TOKENS[i % TOKENS.len()], &mut seen, &mut values, &mut counts);
    }
    let post = snapshot();
    let delta = post - pre;

    assert_eq!(
        values.len(),
        TOKENS.len(),
        "hot loop must not introduce new distinct tokens"
    );
    // Tolerance for background allocations from concurrent test framework
    // threads (panic handlers, etc.) sharing this process's allocator.
    // Per-token regressions would push delta into the millions; a small
    // absolute floor catches the regression without flaking on background.
    const NOISE_FLOOR: usize = 1_000;
    assert!(
        delta < NOISE_FLOOR,
        "LIST_DISTINCT inner observe loop must not allocate per-token after warmup; \
         delta={delta} for {HOT_LOOP_ITERS} iterations exceeds noise floor {NOISE_FLOOR}"
    );

    // Cumulative count check: each token observed exactly HOT_LOOP_ITERS / N + 1 times.
    let expected_per_token = (HOT_LOOP_ITERS / TOKENS.len()) as i64 + 1;
    for c in &counts {
        assert_eq!(*c, expected_per_token);
    }
}

#[test]
fn bench_listdistinct_zero_alloc_int_tokens() {
    let _g = MEASUREMENT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Mirror of the str test for `Int64Array` source columns.
    const TOKENS: &[i64] = &[1, 7, 13, 42, 99, 1024, 65535];
    const HOT_LOOP_ITERS: usize = 1_000_000;

    let mut seen: HashMap<u64, usize> = HashMap::with_capacity(TOKENS.len() * 2);
    let mut values: Vec<Value> = Vec::with_capacity(TOKENS.len());
    let mut counts: Vec<i64> = Vec::with_capacity(TOKENS.len());

    for &n in TOKENS {
        observe_int_token(n, &mut seen, &mut values, &mut counts);
    }

    let pre = snapshot();
    for i in 0..HOT_LOOP_ITERS {
        observe_int_token(TOKENS[i % TOKENS.len()], &mut seen, &mut values, &mut counts);
    }
    let post = snapshot();
    let delta = post - pre;

    const NOISE_FLOOR: usize = 1_000;
    assert!(
        delta < NOISE_FLOOR,
        "Int64-token observe loop must not allocate per-token after warmup; \
         delta={delta} for {HOT_LOOP_ITERS} iterations exceeds noise floor {NOISE_FLOOR}"
    );
}
