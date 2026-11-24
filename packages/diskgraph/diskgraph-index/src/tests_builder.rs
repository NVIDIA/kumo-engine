// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::builder::*;
use std::sync::Arc;

#[test]
fn edge_buffer_pool_reuses_budget() {
    let budget = Arc::new(MemoryBudget::new(1024 * 1024));
    let pool = Arc::new(EdgeBufferPool::new(Some(Arc::clone(&budget))));
    {
        let mut lease = pool.acquire(128).unwrap();
        lease.set_len_uninit(128).unwrap();
    }
    let snap_after = budget.snapshot();
    assert!(snap_after.in_use > 0);
    {
        let _lease = pool.acquire(64).unwrap();
        let snap_reuse = budget.snapshot();
        assert_eq!(snap_after.in_use, snap_reuse.in_use);
    }
}

#[test]
fn edge_buffer_pool_pair_reuses_budget() {
    let budget = Arc::new(MemoryBudget::new(1024 * 1024));
    let pool = Arc::new(EdgeBufferPool::new(Some(Arc::clone(&budget))));
    {
        let (_lease_a, _lease_b) = pool.acquire_pair(128, None).unwrap();
    }
    let snap_after = budget.snapshot();
    assert!(snap_after.in_use > 0);
    {
        let _lease = pool.acquire(64).unwrap();
        let snap_reuse = budget.snapshot();
        assert_eq!(snap_after.in_use, snap_reuse.in_use);
    }
}

#[test]
fn edge_buffer_pool_releases_budget_on_pool_drop() {
    let budget = Arc::new(MemoryBudget::new(1024 * 1024));
    {
        let pool = Arc::new(EdgeBufferPool::new(Some(Arc::clone(&budget))));
        let mut lease = pool.acquire(1024).unwrap();
        lease.set_len_uninit(1024).unwrap();
    }
    let snap_after = budget.snapshot();
    assert_eq!(snap_after.in_use, 0);
}

#[test]
fn edge_buffer_pool_evicts_smaller_buffers_for_large_request() {
    let record_bytes = std::mem::size_of::<EdgeRecord>() as u64;
    let small_cap = 64usize;
    let large_cap = 1024usize;
    let budget_bytes = (large_cap as u64) * record_bytes;
    let budget = Arc::new(MemoryBudget::new(budget_bytes));
    let pool = Arc::new(EdgeBufferPool::new(Some(Arc::clone(&budget))));
    {
        let _a = pool.acquire(small_cap).unwrap();
        let _b = pool.acquire(small_cap).unwrap();
    }
    let snap_after = budget.snapshot();
    assert!(snap_after.in_use > 0);
    let _large = pool.acquire(large_cap).unwrap();
}

#[test]
fn edge_buffer_pool_uses_reservation_without_touching_global_budget() {
    let record_bytes = std::mem::size_of::<EdgeRecord>() as u64;
    let capacity = 10;
    let total_needed = record_bytes * (capacity as u64) * 2;

    let budget = Arc::new(MemoryBudget::new(total_needed * 2));
    let pool = Arc::new(EdgeBufferPool::new(Some(Arc::clone(&budget))));

    // Acquire reservation upfront (simulating MergedEdgeStream behavior)
    let mut reservation = BudgetReservation::new(Arc::clone(&budget), total_needed).unwrap();

    let initial_in_use = budget.snapshot().in_use;
    assert_eq!(initial_in_use, total_needed);

    // Acquire pair using the reservation
    let (lease_a, lease_b) = pool.acquire_pair(capacity, Some(&mut reservation)).unwrap();

    // Global budget should not have increased; we just moved the reservation ownership
    assert_eq!(budget.snapshot().in_use, initial_in_use);

    // The local reservation handle should be empty/reduced
    assert_eq!(reservation.bytes, 0);

    // Leases should be usable
    assert!(lease_a.buf.capacity() >= capacity);
    assert!(lease_b.buf.capacity() >= capacity);
}

#[test]
fn edge_buffer_pool_distributes_reservation_to_multiple_consumers() {
    let record_bytes = std::mem::size_of::<EdgeRecord>() as u64;
    let capacity_per_consumer = 10;
    let num_consumers = 4;
    let total_needed_per_consumer = record_bytes * (capacity_per_consumer as u64) * 2; // *2 for pair
    let total_needed = total_needed_per_consumer * num_consumers;

    let budget = Arc::new(MemoryBudget::new(total_needed));
    let pool = Arc::new(EdgeBufferPool::new(Some(Arc::clone(&budget))));

    // 1. Reserve total upfront (mimicking MergedEdgeStream)
    let mut reservation = pool.reserve(total_needed).unwrap();

    assert_eq!(budget.snapshot().in_use, total_needed);

    // 2. Distribute to consumers
    let mut leases = Vec::new();
    for _ in 0..num_consumers {
        let (lease_a, lease_b) = pool
            .acquire_pair(capacity_per_consumer, Some(&mut reservation))
            .unwrap();
        leases.push((lease_a, lease_b));
    }

    // 3. Verify all consumers got their buffers
    assert_eq!(leases.len(), num_consumers as usize);

    // 4. Verify reservation is empty (or close to it, exact match expected here)
    assert_eq!(reservation.bytes, 0);

    // 5. Verify budget is still fully in use
    assert_eq!(budget.snapshot().in_use, total_needed);
}

#[test]
fn count_len_correctness() {
    assert_eq!(count_len(0), 1);
    assert_eq!(count_len(63), 1);
    assert_eq!(count_len(64), 2);
    assert_eq!(count_len(0x3FFF), 2);
    assert_eq!(count_len(0x4000), 3);
    assert_eq!(count_len(0x3F_FFFF), 3);
    assert_eq!(count_len(0x40_0000), 4);
    assert_eq!(count_len(0x3FFF_FFFF), 4);
    // Values larger than MAX_VARINT_COUNT might trigger panic or undefined behavior if not checked,
    // but count_len logic just checks ranges.
    assert_eq!(count_len(0x4000_0000), 4);
}

#[test]
fn write_len_le_correctness() {
    let mut buf = Vec::new();

    // 2 bytes
    buf.clear();
    write_len_le(&mut buf, 0x1234, 2).unwrap();
    assert_eq!(buf, vec![0x34, 0x12]);

    // 3 bytes
    buf.clear();
    write_len_le(&mut buf, 0x123456, 3).unwrap();
    assert_eq!(buf, vec![0x56, 0x34, 0x12]);

    // 4 bytes
    buf.clear();
    write_len_le(&mut buf, 0x12345678, 4).unwrap();
    assert_eq!(buf, vec![0x78, 0x56, 0x34, 0x12]);

    // Overflow check 2 bytes
    buf.clear();
    assert!(write_len_le(&mut buf, 0x10000, 2).is_err());

    // Overflow check 3 bytes
    buf.clear();
    assert!(write_len_le(&mut buf, 0x1000000, 3).is_err());
}

#[test]
fn delta_encode_offsets_into_correctness() {
    let mut buf = Vec::new();
    let offsets = vec![100, 105, 120, 122];

    // Base 100
    // Deltas: 105-100=5, 120-105=15, 122-120=2

    // Test with 2-byte deltas
    buf.clear();
    delta_encode_offsets_into(&offsets, &mut buf, 2).unwrap();
    // Layout: [count u16][base u64][delta u16]...
    // count = 4
    // base = 100
    // deltas = 5, 15, 2

    let mut expected = Vec::new();
    expected.extend_from_slice(&4u16.to_le_bytes());
    expected.extend_from_slice(&100u64.to_le_bytes());
    expected.extend_from_slice(&5u16.to_le_bytes());
    expected.extend_from_slice(&15u16.to_le_bytes());
    expected.extend_from_slice(&2u16.to_le_bytes());

    assert_eq!(buf, expected);

    // Test with empty
    buf.clear();
    delta_encode_offsets_into(&[], &mut buf, 2).unwrap();
    assert!(buf.is_empty());
}

fn wait_for_waiters(budget: &MemoryBudget, count: usize) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while budget.snapshot().waiters < count {
        assert!(
            std::time::Instant::now() < deadline,
            "no budget waiter after 10 s"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn finish<T>(rx: &std::sync::mpsc::Receiver<T>) -> T {
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .expect("budget call did not return within 10 s")
}

#[test]
fn guard_resize_waits_until_the_budget_frees() {
    let budget = Arc::new(MemoryBudget::new(100));
    let other = budget.acquire(40).unwrap();
    let mut guard = budget.acquire(60).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        guard.resize(80);
        tx.send(guard).unwrap();
    });
    wait_for_waiters(&budget, 1);
    assert_eq!(budget.snapshot().in_use, 100);
    drop(other);
    let _guard = finish(&rx);
    assert_eq!(budget.snapshot().in_use, 80);
}

#[test]
fn guard_resize_goes_over_budget_when_every_holder_waits_to_grow() {
    let budget = Arc::new(MemoryBudget::new(100));
    let mut first = budget.acquire(60).unwrap();
    let mut second = budget.acquire(40).unwrap();
    let (first_tx, first_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        first.resize(80);
        first_tx.send(first).unwrap();
    });
    wait_for_waiters(&budget, 1);
    let (second_tx, second_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        second.resize(70);
        second_tx.send(second).unwrap();
    });
    let second = finish(&second_rx);
    assert_eq!(budget.snapshot().in_use, 130);
    drop(second);
    let first = finish(&first_rx);
    assert_eq!(budget.snapshot().in_use, 80);
    drop(first);
    assert_eq!(budget.snapshot().in_use, 0);
}

#[test]
fn guard_resize_down_wakes_a_waiting_acquire() {
    let budget = Arc::new(MemoryBudget::new(100));
    let mut guard = budget.acquire(80).unwrap();
    let waiting = Arc::clone(&budget);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        tx.send(waiting.acquire(50).unwrap()).unwrap();
    });
    wait_for_waiters(&budget, 1);
    guard.resize(30);
    let _other = finish(&rx);
    assert_eq!(budget.snapshot().in_use, 80);
}

#[test]
fn release_after_overdraft_frees_only_the_real_room() {
    let budget = Arc::new(MemoryBudget::new(100));
    let _held = budget.acquire(60).unwrap();
    drop(budget.force_acquire(60));
    assert!(budget.try_acquire(41).is_none());
    assert!(budget.try_acquire(40).is_some());
}
