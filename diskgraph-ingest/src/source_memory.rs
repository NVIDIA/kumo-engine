use anyhow::{anyhow, bail, Context, Result};
use std::mem::size_of;
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
    producer_bytes: u64,
    writer_bytes: u64,
    high_water_bytes: u64,
    next_ticket: u64,
    serving_ticket: u64,
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
        if state.producer_bytes > self.max_bytes.saturating_sub(bytes) {
            bail!("Encode producer working set exceeds the normal memory budget");
        }
        while state.working_bytes > self.max_bytes.saturating_sub(bytes) {
            state = self
                .available
                .wait(state)
                .map_err(|_| anyhow!("Encode memory budget lock was poisoned"))?;
        }
        state.used_bytes += bytes;
        state.working_bytes += bytes;
        state.producer_bytes += bytes;
        state.high_water_bytes = state.high_water_bytes.max(state.used_bytes);
        Ok(BytePermit {
            budget: Arc::clone(self),
            bytes,
            input: false,
            writer: false,
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
        state.producer_bytes += bytes;
        state.high_water_bytes = state.high_water_bytes.max(state.used_bytes);
        Ok(BytePermit {
            budget: Arc::clone(self),
            bytes,
            input: false,
            writer: false,
        })
    }

    /// Bound complete compressed source objects independently from the
    /// decoded Encode working set by using a separate `ByteBudget` instance.
    pub(crate) fn acquire_resident_object(
        self: &Arc<Self>,
        bytes: u64,
    ) -> Result<BytePermit> {
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
            writer: false,
        })
    }

    pub(crate) fn issue_ticket(&self, bytes: u64) -> Result<u64> {
        if bytes > self.max_single_bytes {
            bail!("Encode allocation exceeds the memory budget");
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("Encode memory budget lock was poisoned"))?;
        let ticket = state.next_ticket;
        state.next_ticket = state
            .next_ticket
            .checked_add(1)
            .ok_or_else(|| anyhow!("Encode memory budget ticket overflowed"))?;
        Ok(ticket)
    }

    pub(crate) fn acquire_ticket(self: &Arc<Self>, ticket: u64, bytes: u64) -> Result<BytePermit> {
        if bytes > self.max_single_bytes {
            bail!("Encode input chunk exceeds the oversized chunk limit");
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("Encode memory budget lock was poisoned"))?;
        while ticket != state.serving_ticket || {
            let input_limit = if state.input_bytes == 0 {
                // The first input object may use the explicit oversized escape
                // together with retained consumer scratch. Later input
                // lookahead objects remain within the normal input budget.
                self.max_single_bytes
            } else {
                self.max_bytes
            };
            state.input_bytes > input_limit.saturating_sub(bytes)
        } {
            state = self
                .available
                .wait(state)
                .map_err(|_| anyhow!("Encode memory budget lock was poisoned"))?;
        }
        state.used_bytes += bytes;
        state.input_bytes += bytes;
        state.high_water_bytes = state.high_water_bytes.max(state.used_bytes);
        state.serving_ticket = state
            .serving_ticket
            .checked_add(1)
            .ok_or_else(|| anyhow!("Encode memory budget ticket overflowed"))?;
        self.available.notify_all();
        Ok(BytePermit {
            budget: Arc::clone(self),
            bytes,
            input: true,
            writer: false,
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
    writer: bool,
}

impl BytePermit {
    pub(crate) fn max_input_chunk_bytes(&self) -> u64 {
        self.budget.max_single_bytes
    }

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
            } else if self.writer {
                state.working_bytes = state.working_bytes.saturating_sub(released);
                state.writer_bytes = state.writer_bytes.saturating_sub(released);
            } else {
                state.working_bytes = state.working_bytes.saturating_sub(released);
                state.producer_bytes = state.producer_bytes.saturating_sub(released);
            }
            self.budget.available.notify_all();
        }
    }

    fn transfer_to_writer(&mut self) {
        if self.input || self.writer || self.bytes == 0 {
            return;
        }
        if let Ok(mut state) = self.budget.state.lock() {
            state.producer_bytes = state.producer_bytes.saturating_sub(self.bytes);
            state.writer_bytes = state.writer_bytes.saturating_add(self.bytes);
            self.writer = true;
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
            } else if self.writer {
                state.working_bytes = state.working_bytes.saturating_sub(self.bytes);
                state.writer_bytes = state.writer_bytes.saturating_sub(self.bytes);
            } else {
                state.working_bytes = state.working_bytes.saturating_sub(self.bytes);
                state.producer_bytes = state.producer_bytes.saturating_sub(self.bytes);
            }
            self.budget.available.notify_all();
        }
    }
}

/// A Vec whose reported capacity is reserved from the shared Encode budget
/// before the allocator is allowed to grow. `take` moves both the allocation
/// and its permits into an asynchronous writer message.
pub(crate) struct BudgetedVec<T> {
    data: Vec<T>,
    budget: Arc<ByteBudget>,
    permits: Vec<BytePermit>,
}

impl<T> BudgetedVec<T> {
    pub(crate) fn new(budget: Arc<ByteBudget>) -> Self {
        Self {
            data: Vec::new(),
            budget,
            permits: Vec::new(),
        }
    }

    fn ensure_additional(&mut self, additional: usize) -> Result<()> {
        let required = self
            .data
            .len()
            .checked_add(additional)
            .ok_or_else(|| anyhow!("Encode buffer length overflowed"))?;
        if required <= self.data.capacity() || size_of::<T>() == 0 {
            return Ok(());
        }
        let old_capacity = self.data.capacity();
        let target_capacity = required.max(old_capacity.saturating_mul(2)).max(8);
        // Reserve conservatively before allocation. `Vec::try_reserve_exact`
        // may round for allocator alignment; twice the requested capacity is
        // an upper guard, then the permit is shrunk to the capacity Rust
        // actually reports.
        let guarded_capacity = target_capacity
            .checked_mul(2)
            .ok_or_else(|| anyhow!("Encode buffer capacity overflowed"))?;
        let guarded_elements = guarded_capacity.saturating_sub(old_capacity);
        let guarded_bytes = guarded_elements
            .checked_mul(size_of::<T>())
            .ok_or_else(|| anyhow!("Encode buffer byte size overflowed"))?
            as u64;
        let mut permit = self.budget.acquire(guarded_bytes)?;
        self.data
            .try_reserve_exact(target_capacity.saturating_sub(self.data.len()))
            .context("allocate Encode buffer")?;
        let actual_elements = self.data.capacity().saturating_sub(old_capacity);
        let actual_bytes = actual_elements
            .checked_mul(size_of::<T>())
            .ok_or_else(|| anyhow!("Encode buffer byte size overflowed"))?
            as u64;
        if actual_bytes > guarded_bytes {
            bail!("allocator exceeded the guarded Encode buffer capacity");
        }
        permit.shrink_to(actual_bytes);
        self.permits.push(permit);
        Ok(())
    }

    pub(crate) fn push(&mut self, value: T) -> Result<()> {
        self.ensure_additional(1)?;
        self.data.push(value);
        Ok(())
    }

    pub(crate) fn extend_from_slice(&mut self, values: &[T]) -> Result<()>
    where
        T: Clone,
    {
        self.ensure_additional(values.len())?;
        self.data.extend_from_slice(values);
        Ok(())
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.data.capacity()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }

    pub(crate) fn take(&mut self) -> (Vec<T>, Vec<BytePermit>) {
        let replacement = Self::new(Arc::clone(&self.budget));
        let mut allocation = std::mem::replace(self, replacement);
        for permit in &mut allocation.permits {
            permit.transfer_to_writer();
        }
        (allocation.data, allocation.permits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgeted_vec_reserves_before_growth_and_transfers_permits() {
        let budget = ByteBudget::new(1024).unwrap();
        let mut values = BudgetedVec::new(Arc::clone(&budget));
        values.extend_from_slice(&[1u64, 2, 3]).unwrap();
        assert!(budget.high_water_bytes() >= values.capacity() as u64 * 8);
        let (data, permits) = values.take();
        assert_eq!(data, vec![1, 2, 3]);
        assert!(!permits.is_empty());
        assert!(values.is_empty());
        drop(permits);
        assert!(budget.acquire(1024).is_ok());
    }

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
    fn downstream_growth_does_not_wait_on_its_own_oversized_input() {
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
        assert!(budget.issue_ticket(11).is_err());
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
