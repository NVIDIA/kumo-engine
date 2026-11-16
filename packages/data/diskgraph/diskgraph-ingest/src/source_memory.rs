// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, bail, Result};
use std::sync::{Arc, Condvar, Mutex};

#[derive(Debug)]
pub(crate) struct ByteBudget {
    max_bytes: u64,
    max_single_bytes: u64,
    max_total_bytes: u64,
    state: Mutex<ByteBudgetState>,
    available: Condvar,
}

#[derive(Debug, Default)]
struct ByteBudgetState {
    used_bytes: u64,
    input_bytes: u64,
    working_bytes: u64,
    high_water_bytes: u64,
}

impl ByteBudget {
    pub(crate) fn new(max_bytes: u64) -> Result<Arc<Self>> {
        Self::with_oversized_limit(max_bytes, max_bytes)
    }

    pub(crate) fn with_oversized_limit(max_bytes: u64, max_single_bytes: u64) -> Result<Arc<Self>> {
        if max_bytes == 0 {
            bail!("Encode memory budget must be positive");
        }
        let max_total_bytes = max_bytes
            .checked_add(max_single_bytes.max(max_bytes))
            .ok_or_else(|| anyhow!("Encode oversized total memory limit overflowed"))?;
        Ok(Arc::new(Self {
            max_bytes,
            max_single_bytes,
            max_total_bytes,
            state: Mutex::new(ByteBudgetState::default()),
            available: Condvar::new(),
        }))
    }

    pub(crate) fn acquire(self: &Arc<Self>, bytes: u64) -> Result<BytePermit> {
        if bytes > self.max_bytes {
            bail!("Encode working allocation exceeds the normal memory budget");
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("Encode memory budget lock was poisoned"))?;
        if state.working_bytes > self.max_bytes.saturating_sub(bytes) {
            bail!("Encode working set exceeds the normal memory budget");
        }
        state.used_bytes += bytes;
        state.working_bytes += bytes;
        state.high_water_bytes = state.high_water_bytes.max(state.used_bytes);
        Ok(BytePermit {
            budget: Arc::clone(self),
            bytes,
            input: false,
        })
    }

    /// Reserve an input batch that will be handed to another thread. Unlike a
    /// producer-owned allocation, queued batches can make progress and release
    /// memory while this caller waits, so budget pressure is backpressure
    /// rather than an immediate self-deadlock error.
    pub(crate) fn acquire_dispatched_batch(self: &Arc<Self>, bytes: u64) -> Result<BytePermit> {
        if bytes > self.max_bytes {
            bail!("Encode batch allocation exceeds the normal memory budget");
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("Encode memory budget lock was poisoned"))?;
        while state.working_bytes > self.max_bytes.saturating_sub(bytes) {
            state = self
                .available
                .wait(state)
                .map_err(|_| anyhow!("Encode memory budget lock was poisoned"))?;
        }
        state.used_bytes += bytes;
        state.working_bytes += bytes;
        state.high_water_bytes = state.high_water_bytes.max(state.used_bytes);
        Ok(BytePermit {
            budget: Arc::clone(self),
            bytes,
            input: false,
        })
    }

    /// Bound complete compressed source objects independently from the
    /// decoded Encode working set by using a separate `ByteBudget` instance.
    pub(crate) fn acquire_resident_object(self: &Arc<Self>, bytes: u64) -> Result<BytePermit> {
        if bytes > self.max_bytes {
            bail!(
                "resident Parquet object requires {bytes} bytes, exceeding the {} byte cloud input budget; raise DISKGRAPH_CLOUD_MAX_BYTES or repartition the source",
                self.max_bytes
            );
        }
        self.acquire_dispatched_batch(bytes)
    }

    /// Reserve one input batch. The caller releases its prior file-batch permit
    /// before requesting the next, so waiting here can only be on another input
    /// lane. The explicit oversized escape is capped at one maximum chunk plus
    /// the independently bounded downstream working set.
    pub(crate) fn acquire_input_chunk(self: &Arc<Self>, bytes: u64) -> Result<BytePermit> {
        if bytes > self.max_single_bytes {
            bail!("Encode input chunk exceeds the oversized chunk limit");
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("Encode memory budget lock was poisoned"))?;
        while state.input_bytes
            > (if state.input_bytes == 0 {
                self.max_single_bytes
            } else {
                self.max_bytes
            })
            .saturating_sub(bytes)
        {
            state = self
                .available
                .wait(state)
                .map_err(|_| anyhow!("Encode memory budget lock was poisoned"))?;
        }
        let total = state
            .used_bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("Encode input memory size overflowed"))?;
        let input_total = state
            .input_bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("Encode input memory size overflowed"))?;
        debug_assert!(total <= self.max_total_bytes);
        state.used_bytes = total;
        state.input_bytes = input_total;
        state.high_water_bytes = state.high_water_bytes.max(total);
        Ok(BytePermit {
            budget: Arc::clone(self),
            bytes,
            input: true,
        })
    }

    pub(crate) fn high_water_bytes(&self) -> u64 {
        self.state
            .lock()
            .map(|state| state.high_water_bytes)
            .unwrap_or(u64::MAX)
    }

    pub(crate) fn max_input_chunk_bytes(&self) -> u64 {
        self.max_single_bytes
    }

    pub(crate) fn normal_limit_bytes(&self) -> u64 {
        self.max_bytes
    }
}

#[derive(Debug)]
pub(crate) struct BytePermit {
    budget: Arc<ByteBudget>,
    bytes: u64,
    input: bool,
}

impl BytePermit {
    pub(crate) fn shrink_to(&mut self, bytes: u64) {
        debug_assert!(bytes <= self.bytes);
        let released = self.bytes.saturating_sub(bytes);
        self.bytes = bytes;
        if released == 0 {
            return;
        }
        if let Ok(mut state) = self.budget.state.lock() {
            state.used_bytes = state.used_bytes.saturating_sub(released);
            if self.input {
                state.input_bytes = state.input_bytes.saturating_sub(released);
            } else {
                state.working_bytes = state.working_bytes.saturating_sub(released);
            }
            self.budget.available.notify_all();
        }
    }
}

impl Drop for BytePermit {
    fn drop(&mut self) {
        if let Ok(mut state) = self.budget.state.lock() {
            state.used_bytes = state.used_bytes.saturating_sub(self.bytes);
            if self.input {
                state.input_bytes = state.input_bytes.saturating_sub(self.bytes);
            } else {
                state.working_bytes = state.working_bytes.saturating_sub(self.bytes);
            }
            self.budget.available.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_input_includes_retained_consumer_working_set() {
        let budget = ByteBudget::with_oversized_limit(10, 20).unwrap();
        let working = budget.acquire(6).unwrap();
        let input = budget.acquire_input_chunk(20).unwrap();
        assert_eq!(budget.high_water_bytes(), 26);
        drop(input);
        drop(working);

        let working = budget.acquire(10).unwrap();
        assert!(budget.acquire_input_chunk(21).is_err());
        drop(working);
    }

    #[test]
    fn working_set_can_coexist_with_oversized_input() {
        let budget = ByteBudget::with_oversized_limit(10, 20).unwrap();
        let input = budget.acquire_input_chunk(20).unwrap();
        let working = budget.acquire(10).unwrap();
        assert_eq!(budget.high_water_bytes(), 30);
        drop(working);
        drop(input);
    }

    #[test]
    fn single_input_limit_can_be_below_the_normal_budget() {
        let budget = ByteBudget::with_oversized_limit(20, 10).unwrap();
        assert!(budget.acquire(20).is_ok());
        assert!(budget.acquire_input_chunk(11).is_err());
    }

    #[test]
    fn resident_object_uses_an_independent_bounded_budget() {
        let resident_budget = ByteBudget::new(10).unwrap();
        let encode_budget = ByteBudget::with_oversized_limit(4, 4).unwrap();
        let _resident = resident_budget.acquire_resident_object(6).unwrap();
        let _decode = encode_budget.acquire_input_chunk(4).unwrap();
        assert_eq!(resident_budget.high_water_bytes(), 6);
        assert_eq!(encode_budget.high_water_bytes(), 4);
        assert!(resident_budget.acquire_resident_object(11).is_err());
    }
}
