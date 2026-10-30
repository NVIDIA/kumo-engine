// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit 4 KiB page I/O used by the experimental sampler backends.
//!
//! The production mmap reader remains separate.  This module deliberately
//! owns every destination buffer until its CQE arrives and accepts a short
//! completion only for the logical final page of an immutable file.

use io_uring::{opcode, types, IoUring};
use std::cell::Cell;
use std::cell::RefCell;
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, RawFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU16, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

fn other_error(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::Other, error)
}

pub const PAGE_SIZE: usize = 4096;
const MAX_CACHE_FILE_ID: u64 = (1 << 31) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeIoBackend {
    Mmap,
    Buffered,
    Direct,
}

impl EdgeIoBackend {
    pub fn from_env() -> io::Result<Self> {
        match std::env::var("DG_EDGE_BACKEND")
            .unwrap_or_else(|_| "mmap".to_string())
            .as_str()
        {
            "mmap" => Ok(Self::Mmap),
            "buffered" => Ok(Self::Buffered),
            "direct" => Ok(Self::Direct),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("DG_EDGE_BACKEND must be mmap|buffered|direct, got `{other}`"),
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mmap => "mmap",
            Self::Buffered => "buffered",
            Self::Direct => "direct",
        }
    }
}

#[repr(align(4096))]
pub struct AlignedPage(pub [u8; PAGE_SIZE]);

pub struct PageData {
    // Keep the 4K-aligned allocation separate from the small descriptor so
    // moving completed pages between the I/O owner and sampler stays cheap.
    bytes: Option<Box<AlignedPage>>,
    valid_len: usize,
    recycler: Option<&'static SyncSender<Box<AlignedPage>>>,
}

impl PageData {
    #[cfg(test)]
    fn zeroed() -> Self {
        Self {
            bytes: Some(Box::new(AlignedPage([0; PAGE_SIZE]))),
            valid_len: 0,
            recycler: None,
        }
    }

    fn recycled(bytes: Box<AlignedPage>, recycler: &'static SyncSender<Box<AlignedPage>>) -> Self {
        Self {
            bytes: Some(bytes),
            valid_len: 0,
            recycler: Some(recycler),
        }
    }

    fn bytes_mut(&mut self) -> &mut [u8; PAGE_SIZE] {
        &mut self.bytes.as_mut().unwrap().0
    }

    pub fn valid(&self) -> &[u8] {
        &self.bytes.as_ref().unwrap().0[..self.valid_len]
    }

    pub fn valid_len(&self) -> usize {
        self.valid_len
    }
}

impl Drop for PageData {
    fn drop(&mut self) {
        if let (Some(bytes), Some(recycler)) = (self.bytes.take(), self.recycler.as_ref()) {
            // Never make a sampling worker wait for pool capacity. A full pool
            // simply releases this frame back to the allocator.
            if recycler.try_send(bytes).is_ok() {
                record_frame_recycled();
            } else {
                COUNTERS.frames_dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct PageKey {
    file_id: u64,
    page_number: u64,
}

impl PageKey {
    fn packed(self) -> u64 {
        debug_assert!(self.file_id <= MAX_CACHE_FILE_ID);
        debug_assert!(self.page_number <= u32::MAX as u64);
        (self.file_id << 32) | self.page_number
    }
}

#[derive(Default)]
struct Counters {
    submitted: AtomicU64,
    io_ops: AtomicU64,
    multi_page_ops: AtomicU64,
    multi_page_pages: AtomicU64,
    runtime_jobs: AtomicU64,
    runtime_envelopes: AtomicU64,
    runtime_batches: AtomicU64,
    max_batch_pages: AtomicU64,
    bytes: AtomicU64,
    frames_allocated: AtomicU64,
    frames_reused: AtomicU64,
    frames_recycled: AtomicU64,
    frames_dropped: AtomicU64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PageIoStats {
    pub submitted: u64,
    pub io_ops: u64,
    pub multi_page_ops: u64,
    pub multi_page_pages: u64,
    pub runtime_jobs: u64,
    pub runtime_envelopes: u64,
    pub runtime_batches: u64,
    pub max_batch_pages: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub coalesced: u64,
    pub evictions: u64,
    pub bytes: u64,
    pub frames_allocated: u64,
    pub frames_reused: u64,
    pub frames_recycled: u64,
    pub frames_dropped: u64,
    pub cache_capacity_pages: u64,
    pub cache_entries: u64,
}

static COUNTERS: Counters = Counters {
    submitted: AtomicU64::new(0),
    io_ops: AtomicU64::new(0),
    multi_page_ops: AtomicU64::new(0),
    multi_page_pages: AtomicU64::new(0),
    runtime_jobs: AtomicU64::new(0),
    runtime_envelopes: AtomicU64::new(0),
    runtime_batches: AtomicU64::new(0),
    max_batch_pages: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    frames_allocated: AtomicU64::new(0),
    frames_reused: AtomicU64::new(0),
    frames_recycled: AtomicU64::new(0),
    frames_dropped: AtomicU64::new(0),
};

// Cache events are collected once per planned shard, not once per 4K lookup.
// Give each sampling thread a private cache line so benchmark diagnostics do
// not serialize the cache algorithm they are intended to measure.
const EVENT_COUNTER_SHARDS: usize = 256;

#[repr(align(64))]
struct CacheEventCounter {
    hits: AtomicU64,
    misses: AtomicU64,
    coalesced: AtomicU64,
    evictions: AtomicU64,
}

static CACHE_EVENT_COUNTERS: [CacheEventCounter; EVENT_COUNTER_SHARDS] = [const {
    CacheEventCounter {
        hits: AtomicU64::new(0),
        misses: AtomicU64::new(0),
        coalesced: AtomicU64::new(0),
        evictions: AtomicU64::new(0),
    }
}; EVENT_COUNTER_SHARDS];
static NEXT_EVENT_COUNTER: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static EVENT_COUNTER_SLOT: Cell<usize> = Cell::new(
        NEXT_EVENT_COUNTER.fetch_add(1, Ordering::Relaxed) as usize % EVENT_COUNTER_SHARDS
    );
    static RECYCLED_PENDING: Cell<u64> = const { Cell::new(0) };
}

#[inline]
fn record_cache_events(events: CacheEvents) {
    EVENT_COUNTER_SLOT.with(|slot| {
        let counters = &CACHE_EVENT_COUNTERS[slot.get()];
        if events.hits != 0 {
            counters.hits.fetch_add(events.hits, Ordering::Relaxed);
        }
        if events.misses != 0 {
            counters.misses.fetch_add(events.misses, Ordering::Relaxed);
        }
        if events.coalesced != 0 {
            counters
                .coalesced
                .fetch_add(events.coalesced, Ordering::Relaxed);
        }
        if events.evictions != 0 {
            counters
                .evictions
                .fetch_add(events.evictions, Ordering::Relaxed);
        }
    });
}

#[inline]
fn record_frame_recycled() {
    const FLUSH_EVERY: u64 = 256;
    RECYCLED_PENDING.with(|pending| {
        let count = pending.get() + 1;
        if count == FLUSH_EVERY {
            COUNTERS
                .frames_recycled
                .fetch_add(FLUSH_EVERY, Ordering::Relaxed);
            pending.set(0);
        } else {
            pending.set(count);
        }
    });
}

#[derive(Default, Clone, Copy)]
struct CacheEvents {
    hits: u64,
    misses: u64,
    coalesced: u64,
    evictions: u64,
}

fn cache_events() -> CacheEvents {
    CACHE_EVENT_COUNTERS
        .iter()
        .fold(CacheEvents::default(), |mut total, counters| {
            total.hits += counters.hits.load(Ordering::Relaxed);
            total.misses += counters.misses.load(Ordering::Relaxed);
            total.coalesced += counters.coalesced.load(Ordering::Relaxed);
            total.evictions += counters.evictions.load(Ordering::Relaxed);
            total
        })
}

pub fn stats() -> PageIoStats {
    let cache = cache_events();
    PageIoStats {
        submitted: COUNTERS.submitted.load(Ordering::Relaxed),
        io_ops: COUNTERS.io_ops.load(Ordering::Relaxed),
        multi_page_ops: COUNTERS.multi_page_ops.load(Ordering::Relaxed),
        multi_page_pages: COUNTERS.multi_page_pages.load(Ordering::Relaxed),
        runtime_jobs: COUNTERS.runtime_jobs.load(Ordering::Relaxed),
        runtime_envelopes: COUNTERS.runtime_envelopes.load(Ordering::Relaxed),
        runtime_batches: COUNTERS.runtime_batches.load(Ordering::Relaxed),
        max_batch_pages: COUNTERS.max_batch_pages.load(Ordering::Relaxed),
        cache_hits: cache.hits,
        cache_misses: cache.misses,
        coalesced: cache.coalesced,
        evictions: cache.evictions,
        bytes: COUNTERS.bytes.load(Ordering::Relaxed),
        frames_allocated: COUNTERS.frames_allocated.load(Ordering::Relaxed),
        frames_reused: COUNTERS.frames_reused.load(Ordering::Relaxed),
        frames_recycled: COUNTERS.frames_recycled.load(Ordering::Relaxed),
        frames_dropped: COUNTERS.frames_dropped.load(Ordering::Relaxed),
        cache_capacity_pages: CACHE_CAPACITY_PAGES.load(Ordering::Relaxed),
        cache_entries: CACHE_ENTRIES.load(Ordering::Relaxed),
    }
}

pub fn reset_stats() {
    COUNTERS.submitted.store(0, Ordering::Relaxed);
    COUNTERS.io_ops.store(0, Ordering::Relaxed);
    COUNTERS.multi_page_ops.store(0, Ordering::Relaxed);
    COUNTERS.multi_page_pages.store(0, Ordering::Relaxed);
    COUNTERS.runtime_jobs.store(0, Ordering::Relaxed);
    COUNTERS.runtime_envelopes.store(0, Ordering::Relaxed);
    COUNTERS.runtime_batches.store(0, Ordering::Relaxed);
    COUNTERS.max_batch_pages.store(0, Ordering::Relaxed);
    for counter in &CACHE_EVENT_COUNTERS {
        counter.hits.store(0, Ordering::Relaxed);
        counter.misses.store(0, Ordering::Relaxed);
        counter.coalesced.store(0, Ordering::Relaxed);
        counter.evictions.store(0, Ordering::Relaxed);
    }
    COUNTERS.bytes.store(0, Ordering::Relaxed);
    COUNTERS.frames_allocated.store(0, Ordering::Relaxed);
    COUNTERS.frames_reused.store(0, Ordering::Relaxed);
    COUNTERS.frames_recycled.store(0, Ordering::Relaxed);
    COUNTERS.frames_dropped.store(0, Ordering::Relaxed);
}

pub struct PreparedPages {
    page_numbers: Vec<u64>,
    pages: Vec<PreparedPage>,
}

pub struct PreparedPageCursor<'a> {
    prepared: &'a PreparedPages,
    index: usize,
}

impl PreparedPageCursor<'_> {
    /// Borrow a one-page range while advancing monotonically through the
    /// prepared page numbers. Sorted sampled indices pay one lower-bound per
    /// relation instead of one binary search per emitted edge.
    pub fn contiguous_range(&mut self, start: u64, end: u64) -> io::Result<Option<&[u8]>> {
        if end < start {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        if start == end {
            return Ok(Some(&[]));
        }
        let first_page = start / PAGE_SIZE as u64;
        let last_page = (end - 1) / PAGE_SIZE as u64;
        if first_page != last_page {
            return Ok(None);
        }
        while self.index < self.prepared.page_numbers.len()
            && self.prepared.page_numbers[self.index] < first_page
        {
            self.index += 1;
        }
        if self.prepared.page_numbers.get(self.index) != Some(&first_page) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("page {first_page} was not prepared"),
            ));
        }
        let in_page = (start % PAGE_SIZE as u64) as usize;
        let len: usize = (end - start)
            .try_into()
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let page = &self.prepared.pages[self.index];
        let range_end = in_page
            .checked_add(len)
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        if range_end > page.valid_len() {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        Ok(Some(&page.valid()[in_page..range_end]))
    }
}

enum PreparedPage {
    Transient(Box<PageData>),
    Cached(CacheLease),
    Missing,
}

impl PreparedPage {
    fn valid(&self) -> &[u8] {
        match self {
            Self::Transient(page) => page.valid(),
            Self::Cached(page) => page.valid(),
            Self::Missing => unreachable!("unfinished prepared page"),
        }
    }

    fn valid_len(&self) -> usize {
        match self {
            Self::Transient(page) => page.valid_len(),
            Self::Cached(page) => page.valid_len(),
            Self::Missing => unreachable!("unfinished prepared page"),
        }
    }
}

pub enum PendingPreparedPages {
    Ready(PreparedPages),
    Runtime {
        page_numbers: Vec<u64>,
        result: RuntimeResponse,
    },
    DirectCache {
        page_numbers: Vec<u64>,
        pending: PendingDirectRead,
    },
}

impl PendingPreparedPages {
    pub fn finish(self) -> io::Result<PreparedPages> {
        match self {
            Self::Ready(pages) => Ok(pages),
            Self::Runtime {
                page_numbers,
                result,
            } => match result
                .recv()
                .map_err(|_| other_error("edge-I/O owner dropped response"))?
            {
                Ok(batch)
                    if batch.groups.as_slice() == [(0, batch.pages.len())]
                        && batch.pages.len() == page_numbers.len() =>
                {
                    Ok(PreparedPages::new_transient(page_numbers, batch.pages))
                }
                Ok(batch) => Err(other_error(format!(
                    "single edge-I/O request returned {} page groups",
                    batch.groups.len()
                ))),
                Err(message) => Err(other_error(message)),
            },
            Self::DirectCache {
                page_numbers,
                pending,
            } => Ok(PreparedPages::new_cached(page_numbers, pending.finish()?)),
        }
    }
}

impl PreparedPages {
    pub(crate) fn empty() -> Self {
        Self {
            page_numbers: Vec::new(),
            pages: Vec::new(),
        }
    }

    fn new_transient(page_numbers: Vec<u64>, data: Vec<PageData>) -> Self {
        debug_assert_eq!(page_numbers.len(), data.len());
        Self {
            page_numbers,
            pages: data
                .into_iter()
                .map(Box::new)
                .map(PreparedPage::Transient)
                .collect(),
        }
    }

    fn new_cached(page_numbers: Vec<u64>, pages: Vec<PreparedPage>) -> Self {
        debug_assert_eq!(page_numbers.len(), pages.len());
        Self {
            page_numbers,
            pages,
        }
    }

    pub fn page_count(&self) -> usize {
        self.page_numbers.len()
    }

    pub fn contains_page(&self, page_number: u64) -> bool {
        self.page_numbers.binary_search(&page_number).is_ok()
    }

    pub fn cursor_at(&self, start: u64) -> PreparedPageCursor<'_> {
        let page = start / PAGE_SIZE as u64;
        let index = self.page_numbers.partition_point(|number| *number < page);
        PreparedPageCursor {
            prepared: self,
            index,
        }
    }

    pub fn merge(self, other: PreparedPages) -> Self {
        if other.page_numbers.is_empty() {
            return self;
        }
        if self.page_numbers.is_empty() {
            return other;
        }
        let total = self.page_numbers.len() + other.page_numbers.len();
        let mut numbers = Vec::with_capacity(total);
        let mut pages = Vec::with_capacity(total);
        let mut left = self.page_numbers.into_iter().zip(self.pages).peekable();
        let mut right = other.page_numbers.into_iter().zip(other.pages).peekable();
        while left.peek().is_some() || right.peek().is_some() {
            let take_left = match (left.peek(), right.peek()) {
                (Some((a, _)), Some((b, _))) => a <= b,
                (Some(_), None) => true,
                _ => false,
            };
            let (number, page) = if take_left {
                left.next().unwrap()
            } else {
                right.next().unwrap()
            };
            if numbers.last() == Some(&number) {
                continue;
            }
            numbers.push(number);
            pages.push(page);
        }
        Self {
            page_numbers: numbers,
            pages,
        }
    }

    /// Borrow a range without copying when it resides in one prepared page.
    /// Callers can fall back to `copy_range` for the rare page-straddling
    /// record. Sampled fixed-width edge entries are contiguous almost always.
    pub fn contiguous_range(&self, start: u64, end: u64) -> io::Result<Option<&[u8]>> {
        if end < start {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        if start == end {
            return Ok(Some(&[]));
        }
        let first_page = start / PAGE_SIZE as u64;
        let last_page = (end - 1) / PAGE_SIZE as u64;
        if first_page != last_page {
            return Ok(None);
        }
        let in_page = (start % PAGE_SIZE as u64) as usize;
        let len: usize = (end - start)
            .try_into()
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let page = self
            .page_numbers
            .binary_search(&first_page)
            .ok()
            .map(|idx| &self.pages[idx])
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("page {first_page} was not prepared"),
                )
            })?;
        let range_end = in_page
            .checked_add(len)
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        if range_end > page.valid_len() {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        Ok(Some(&page.valid()[in_page..range_end]))
    }

    pub fn copy_range(&self, start: u64, end: u64) -> io::Result<Vec<u8>> {
        if end < start {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        let len: usize = (end - start)
            .try_into()
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let mut out = Vec::with_capacity(len);
        let mut cursor = start;
        while cursor < end {
            let page_number = cursor / PAGE_SIZE as u64;
            let in_page = (cursor % PAGE_SIZE as u64) as usize;
            let page = self
                .page_numbers
                .binary_search(&page_number)
                .ok()
                .map(|idx| &self.pages[idx])
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("page {page_number} was not prepared"),
                    )
                })?;
            if in_page >= page.valid_len() {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            let take = ((end - cursor) as usize).min(page.valid_len() - in_page);
            out.extend_from_slice(&page.valid()[in_page..in_page + take]);
            cursor += take as u64;
        }
        Ok(out)
    }

    /// Append a prepared byte range to caller-owned scratch. Multi-page node
    /// features use one arena per decode chunk instead of one allocation per
    /// row; one-page rows can continue to use [`Self::contiguous_range`].
    pub fn append_range(&self, start: u64, end: u64, out: &mut Vec<u8>) -> io::Result<()> {
        if end < start {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        let additional: usize = (end - start)
            .try_into()
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        out.reserve(additional);
        let mut cursor = start;
        while cursor < end {
            let page_number = cursor / PAGE_SIZE as u64;
            let in_page = (cursor % PAGE_SIZE as u64) as usize;
            let page = self
                .page_numbers
                .binary_search(&page_number)
                .ok()
                .map(|idx| &self.pages[idx])
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("page {page_number} was not prepared"),
                    )
                })?;
            if in_page >= page.valid_len() {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            let take = ((end - cursor) as usize).min(page.valid_len() - in_page);
            out.extend_from_slice(&page.valid()[in_page..in_page + take]);
            cursor += take as u64;
        }
        Ok(())
    }
}

pub fn pages_for_ranges(ranges: &[(u64, u64)], file_len: u64) -> io::Result<Vec<u64>> {
    let mut page_capacity = 0usize;
    for &(start, end) in ranges {
        if start > end || end > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid edge range {start}..{end} for file length {file_len}"),
            ));
        }
        if start == end {
            continue;
        }
        let first = start / PAGE_SIZE as u64;
        let last = (end - 1) / PAGE_SIZE as u64;
        let span: usize = (last - first + 1)
            .try_into()
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        page_capacity = page_capacity
            .checked_add(span)
            .ok_or_else(|| io::Error::from(io::ErrorKind::OutOfMemory))?;
    }
    let mut pages = Vec::with_capacity(page_capacity);
    for &(start, end) in ranges {
        if start == end {
            continue;
        }
        let first = start / PAGE_SIZE as u64;
        let last = (end - 1) / PAGE_SIZE as u64;
        pages.extend(first..=last);
    }
    pages.sort_unstable();
    pages.dedup();
    Ok(pages)
}

pub fn read_prepared_pages(
    buffered_file: &Arc<File>,
    direct_file: Option<&Arc<File>>,
    file_id: u64,
    file_len: u64,
    backend: EdgeIoBackend,
    page_numbers: &[u64],
) -> io::Result<PreparedPages> {
    submit_prepared_pages(
        buffered_file,
        direct_file,
        file_id,
        file_len,
        backend,
        page_numbers,
    )?
    .finish()
}

pub struct PageReadRequest {
    buffered_file: Arc<File>,
    direct_file: Option<Arc<File>>,
    file_id: u64,
    file_len: u64,
    backend: EdgeIoBackend,
    page_numbers: Vec<u64>,
}

pub fn page_read_request(
    buffered_file: &Arc<File>,
    direct_file: Option<&Arc<File>>,
    file_id: u64,
    file_len: u64,
    backend: EdgeIoBackend,
    page_numbers: &[u64],
) -> io::Result<PageReadRequest> {
    page_read_request_owned(
        buffered_file,
        direct_file,
        file_id,
        file_len,
        backend,
        page_numbers.to_vec(),
    )
}

pub fn page_read_request_owned(
    buffered_file: &Arc<File>,
    direct_file: Option<&Arc<File>>,
    file_id: u64,
    file_len: u64,
    backend: EdgeIoBackend,
    mut page_numbers: Vec<u64>,
) -> io::Result<PageReadRequest> {
    if file_id > MAX_CACHE_FILE_ID {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "edge file id exceeds packed-cache key width",
        ));
    }
    page_numbers.sort_unstable();
    page_numbers.dedup();
    if page_numbers
        .iter()
        .any(|&page| page > u32::MAX as u64 || page.saturating_mul(PAGE_SIZE as u64) >= file_len)
    {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "page begins at or beyond edge-file EOF",
        ));
    }
    if backend == EdgeIoBackend::Mmap {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "explicit page read requested for mmap backend",
        ));
    }
    if backend == EdgeIoBackend::Direct && direct_file.is_none() && !page_numbers.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "O_DIRECT file is not open",
        ));
    }
    Ok(PageReadRequest {
        buffered_file: Arc::clone(buffered_file),
        direct_file: direct_file.cloned(),
        file_id,
        file_len,
        backend,
        page_numbers,
    })
}

enum PendingBatchKind {
    Ready(Vec<PreparedPages>),
    Runtime(Vec<RuntimeResponse>),
    DirectCache {
        plans: Vec<DirectReadPlan>,
        responses: Vec<RuntimeResponse>,
    },
    Individual(Vec<PendingPreparedPages>),
}

pub struct PendingPreparedPageBatch {
    page_numbers: Vec<Vec<u64>>,
    kind: PendingBatchKind,
}

impl PendingPreparedPageBatch {
    pub fn finish(mut self) -> io::Result<Vec<PreparedPages>> {
        match self.kind {
            PendingBatchKind::Ready(pages) => Ok(pages),
            PendingBatchKind::Individual(pending) => pending
                .into_iter()
                .map(PendingPreparedPages::finish)
                .collect(),
            PendingBatchKind::Runtime(responses) => {
                let mut out: Vec<Option<PreparedPages>> =
                    (0..self.page_numbers.len()).map(|_| None).collect();
                for (idx, pages) in self.page_numbers.iter().enumerate() {
                    if pages.is_empty() {
                        out[idx] = Some(PreparedPages::empty());
                    }
                }
                for response in responses {
                    let batch = response
                        .recv()
                        .map_err(|_| other_error("edge-I/O owner dropped batch response"))?
                        .map_err(other_error)?;
                    let grouped_pages = batch
                        .groups
                        .iter()
                        .try_fold(0usize, |total, (_, count)| total.checked_add(*count))
                        .ok_or_else(|| other_error("edge-I/O batch page count overflow"))?;
                    if grouped_pages != batch.pages.len() {
                        return Err(other_error(
                            "edge-I/O batch returned an invalid grouped page count",
                        ));
                    }
                    let mut pages = batch.pages.into_iter();
                    for (request_idx, page_count) in batch.groups {
                        let page_numbers =
                            self.page_numbers.get_mut(request_idx).ok_or_else(|| {
                                other_error("edge-I/O batch returned an invalid request index")
                            })?;
                        if out[request_idx].is_some() {
                            return Err(other_error(
                                "edge-I/O batch returned a duplicate request index",
                            ));
                        }
                        out[request_idx] = Some(PreparedPages::new_transient(
                            std::mem::take(page_numbers),
                            pages.by_ref().take(page_count).collect(),
                        ));
                    }
                    if pages.next().is_some() {
                        return Err(other_error("edge-I/O batch returned ungrouped pages"));
                    }
                }
                out.into_iter()
                    .map(|pages| {
                        pages.ok_or_else(|| other_error("edge-I/O batch omitted a request result"))
                    })
                    .collect()
            }
            PendingBatchKind::DirectCache {
                mut plans,
                responses,
            } => {
                let mut completed = Vec::with_capacity(responses.len());
                let mut miss_counts = vec![usize::MAX; plans.len()];
                for response in responses {
                    let batch = match response
                        .recv()
                        .map_err(|_| other_error("edge-I/O owner dropped cache-batch response"))?
                    {
                        Ok(batch) => batch,
                        Err(message) => {
                            for plan in &plans {
                                plan.fail(&message);
                            }
                            return Err(other_error(message));
                        }
                    };
                    let grouped_pages = batch
                        .groups
                        .iter()
                        .try_fold(0usize, |total, (_, count)| total.checked_add(*count))
                        .ok_or_else(|| other_error("cache batch page count overflow"))?;
                    if grouped_pages != batch.pages.len() {
                        let message = "cache batch returned an invalid grouped page count";
                        for plan in &plans {
                            plan.fail(message);
                        }
                        return Err(other_error(message));
                    }
                    for &(request_idx, page_count) in &batch.groups {
                        let Some(count) = miss_counts.get_mut(request_idx) else {
                            let message = "cache batch returned an invalid request index";
                            for plan in &plans {
                                plan.fail(message);
                            }
                            return Err(other_error(message));
                        };
                        if *count != usize::MAX {
                            let message = "cache batch returned a duplicate request index";
                            for plan in &plans {
                                plan.fail(message);
                            }
                            return Err(other_error(message));
                        }
                        *count = page_count;
                    }
                    completed.push(batch);
                }
                for (idx, plan) in plans.iter().enumerate() {
                    let actual = if miss_counts[idx] == usize::MAX {
                        0
                    } else {
                        miss_counts[idx]
                    };
                    if actual != plan.reads.len() {
                        let message = format!(
                            "edge-I/O cache batch returned {actual} pages for {} misses",
                            plan.reads.len()
                        );
                        for plan in &plans {
                            plan.fail(&message);
                        }
                        return Err(other_error(message));
                    }
                }
                // Publish every completed read before assembling the prepared
                // page lists. A LOADING duplicate is read transiently rather
                // than blocking a sampling thread on cache publication.
                for batch in completed {
                    let mut pages = batch.pages.into_iter();
                    for (request_idx, page_count) in batch.groups {
                        plans[request_idx].publish_reads(pages.by_ref().take(page_count));
                    }
                    debug_assert!(pages.next().is_none());
                }
                plans
                    .into_iter()
                    .zip(self.page_numbers)
                    .map(|(plan, page_numbers)| {
                        Ok(PreparedPages::new_cached(
                            page_numbers,
                            plan.finish_ready()?,
                        ))
                    })
                    .collect()
            }
        }
    }
}

pub fn submit_prepared_page_batch(
    requests: Vec<PageReadRequest>,
) -> io::Result<PendingPreparedPageBatch> {
    if direct_cache_enabled()
        && requests
            .iter()
            .all(|request| request.backend == EdgeIoBackend::Direct)
    {
        let mut plans = Vec::with_capacity(requests.len());
        let mut parts = Vec::with_capacity(requests.len());
        let mut page_numbers = Vec::with_capacity(requests.len());
        for (request_idx, mut request) in requests.into_iter().enumerate() {
            let plan = direct_cache()?.plan_read(request.file_id, &request.page_numbers);
            if !plan.reads.is_empty() {
                parts.push(ReadPart {
                    request_idx,
                    file: request.direct_file.unwrap(),
                    file_len: request.file_len,
                    pages: plan.reads.iter().map(|read| read.key.page_number).collect(),
                });
            }
            plans.push(plan);
            page_numbers.push(std::mem::take(&mut request.page_numbers));
        }
        if plans.iter().all(|plan| plan.reads.is_empty()) {
            let ready = plans
                .into_iter()
                .zip(page_numbers)
                .map(|(plan, pages)| Ok(PreparedPages::new_cached(pages, plan.finish_ready()?)))
                .collect::<io::Result<_>>()?;
            return Ok(PendingPreparedPageBatch {
                page_numbers: Vec::new(),
                kind: PendingBatchKind::Ready(ready),
            });
        }
        let responses = if parts.is_empty() {
            Vec::new()
        } else {
            page_runtime().submit_batch(parts)?
        };
        return Ok(PendingPreparedPageBatch {
            page_numbers,
            kind: PendingBatchKind::DirectCache { plans, responses },
        });
    }
    if requests
        .iter()
        .any(|request| request.backend == EdgeIoBackend::Direct && direct_cache_enabled())
    {
        let page_numbers: Vec<Vec<u64>> = requests
            .iter()
            .map(|request| request.page_numbers.clone())
            .collect();
        let pending = requests
            .iter()
            .map(|request| {
                submit_prepared_pages(
                    &request.buffered_file,
                    request.direct_file.as_ref(),
                    request.file_id,
                    request.file_len,
                    request.backend,
                    &request.page_numbers,
                )
            })
            .collect::<io::Result<Vec<_>>>()?;
        return Ok(PendingPreparedPageBatch {
            page_numbers,
            kind: PendingBatchKind::Individual(pending),
        });
    }

    let page_numbers: Vec<Vec<u64>> = requests
        .iter()
        .map(|request| request.page_numbers.clone())
        .collect();

    let parts = requests
        .into_iter()
        .enumerate()
        .filter(|(_, request)| !request.page_numbers.is_empty())
        .map(|(request_idx, request)| {
            let file = match request.backend {
                EdgeIoBackend::Buffered => request.buffered_file,
                EdgeIoBackend::Direct => request.direct_file.unwrap(),
                EdgeIoBackend::Mmap => unreachable!(),
            };
            ReadPart {
                request_idx,
                file,
                file_len: request.file_len,
                pages: request.page_numbers,
            }
        })
        .collect();
    let responses = page_runtime().submit_batch(parts)?;
    Ok(PendingPreparedPageBatch {
        page_numbers,
        kind: PendingBatchKind::Runtime(responses),
    })
}

pub fn submit_prepared_pages(
    buffered_file: &Arc<File>,
    direct_file: Option<&Arc<File>>,
    file_id: u64,
    file_len: u64,
    backend: EdgeIoBackend,
    page_numbers: &[u64],
) -> io::Result<PendingPreparedPages> {
    if file_id > MAX_CACHE_FILE_ID {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "edge file id exceeds packed-cache key width",
        ));
    }
    let mut unique = page_numbers.to_vec();
    unique.sort_unstable();
    unique.dedup();
    if unique.is_empty() {
        return Ok(PendingPreparedPages::Ready(PreparedPages::empty()));
    }
    if unique
        .iter()
        .any(|&p| p > u32::MAX as u64 || p.saturating_mul(PAGE_SIZE as u64) >= file_len)
    {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "page begins at or beyond edge-file EOF",
        ));
    }
    match backend {
        EdgeIoBackend::Mmap => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "explicit page read requested for mmap backend",
        )),
        EdgeIoBackend::Buffered => {
            let result =
                page_runtime().submit(Arc::clone(buffered_file), file_id, file_len, &unique)?;
            Ok(PendingPreparedPages::Runtime {
                page_numbers: unique,
                result,
            })
        }
        EdgeIoBackend::Direct => {
            let direct = direct_file.ok_or_else(|| {
                io::Error::new(io::ErrorKind::Unsupported, "O_DIRECT file is not open")
            })?;
            if direct_cache_enabled() {
                let pending =
                    direct_cache()?.begin_read(Arc::clone(direct), file_id, file_len, &unique)?;
                Ok(PendingPreparedPages::DirectCache {
                    page_numbers: unique,
                    pending,
                })
            } else {
                let result =
                    page_runtime().submit(Arc::clone(direct), file_id, file_len, &unique)?;
                Ok(PendingPreparedPages::Runtime {
                    page_numbers: unique,
                    result,
                })
            }
        }
    }
}

fn configured_qd() -> usize {
    std::env::var("DG_IO_QD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256usize)
        .clamp(1, 4096)
}

fn direct_cache_enabled() -> bool {
    configured_direct_cache_gib() > 0
}

fn configured_direct_cache_gib() -> usize {
    static CACHE_GIB: OnceLock<usize> = OnceLock::new();
    *CACHE_GIB.get_or_init(|| {
        std::env::var("DG_DIRECT_CACHE_GB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

fn configured_rings() -> usize {
    std::env::var("DG_IO_RINGS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8)
        .clamp(1, 64)
}

fn configured_batch_micros() -> u64 {
    std::env::var("DG_IO_BATCH_US")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
        .min(1_000)
}

fn configured_batch_depth_multiplier() -> usize {
    std::env::var("DG_IO_BATCH_MULT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 32)
}

fn configured_coalesce_pages() -> usize {
    static COALESCE_PAGES: OnceLock<usize> = OnceLock::new();
    *COALESCE_PAGES.get_or_init(|| {
        std::env::var("DG_IO_COALESCE_PAGES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(16)
            .clamp(1, 256)
    })
}

pub struct RuntimePageBatch {
    /// Compact `(original request index, page count)` spans in `pages` order.
    groups: Vec<(usize, usize)>,
    pages: Vec<PageData>,
}

type RuntimeResult = Result<RuntimePageBatch, String>;
type RuntimeResponse = Receiver<RuntimeResult>;

struct ReadPart {
    request_idx: usize,
    file: Arc<File>,
    file_len: u64,
    pages: Vec<u64>,
}

struct ReadJob {
    parts: Vec<ReadPart>,
    response: SyncSender<RuntimeResult>,
}

struct PageRuntime {
    senders: Vec<SyncSender<ReadJob>>,
    next_owner: AtomicU64,
}

impl PageRuntime {
    fn new() -> Self {
        let rings = configured_rings();
        let qd = configured_qd();
        let batch_micros = configured_batch_micros();
        let batch_page_target = qd.saturating_mul(configured_batch_depth_multiplier());
        let mut senders = Vec::with_capacity(rings);
        for owner in 0..rings {
            let (sender, receiver) = mpsc::sync_channel::<ReadJob>(8192);
            std::thread::Builder::new()
                .name(format!("dg-edge-io-{owner}"))
                .spawn(move || page_owner_loop(receiver, qd, batch_page_target, batch_micros))
                .expect("failed to spawn explicit edge-I/O owner");
            senders.push(sender);
        }
        Self {
            senders,
            next_owner: AtomicU64::new(0),
        }
    }

    fn submit(
        &self,
        file: Arc<File>,
        _file_id: u64,
        file_len: u64,
        pages: &[u64],
    ) -> io::Result<RuntimeResponse> {
        let owner = self.next_owner.fetch_add(1, Ordering::Relaxed) as usize % self.senders.len();
        let (response, result) = mpsc::sync_channel(1);
        self.senders[owner]
            .send(ReadJob {
                parts: vec![ReadPart {
                    request_idx: 0,
                    file,
                    file_len,
                    pages: pages.to_vec(),
                }],
                response,
            })
            .map_err(|_| other_error("edge-I/O owner stopped"))?;
        COUNTERS.runtime_jobs.fetch_add(1, Ordering::Relaxed);
        COUNTERS.runtime_envelopes.fetch_add(1, Ordering::Relaxed);
        Ok(result)
    }

    fn submit_batch(&self, parts: Vec<ReadPart>) -> io::Result<Vec<RuntimeResponse>> {
        let part_count = parts.len();
        let mut by_owner: Vec<Vec<ReadPart>> =
            (0..self.senders.len()).map(|_| Vec::new()).collect();
        // Reserve the round-robin range once for the whole batch rather than
        // bouncing one global atomic cache line for every shard part.
        let owner_base = self
            .next_owner
            .fetch_add(part_count as u64, Ordering::Relaxed) as usize;
        for (offset, part) in parts.into_iter().enumerate() {
            let owner = (owner_base + offset) % self.senders.len();
            by_owner[owner].push(part);
        }

        let mut responses = Vec::with_capacity(self.senders.len());
        for (owner, parts) in by_owner.into_iter().enumerate() {
            if parts.is_empty() {
                continue;
            }
            let (response, result) = mpsc::sync_channel(1);
            self.senders[owner]
                .send(ReadJob { parts, response })
                .map_err(|_| other_error("edge-I/O owner stopped"))?;
            responses.push(result);
        }
        COUNTERS
            .runtime_jobs
            .fetch_add(part_count as u64, Ordering::Relaxed);
        COUNTERS
            .runtime_envelopes
            .fetch_add(responses.len() as u64, Ordering::Relaxed);
        Ok(responses)
    }
}

fn page_owner_loop(
    receiver: Receiver<ReadJob>,
    qd: usize,
    batch_page_target: usize,
    batch_micros: u64,
) {
    // Completed pages remain live until the sampling worker consumes its
    // prepared batch.  A QD-sized recycle reserve is therefore much too small
    // when many graph batches overlap: sustained runs repeatedly free and
    // fault anonymous 4K frames even though the maximum live demand is stable.
    // Keep a bounded reserve of several completed owner batches.  At the
    // tuned 8 x (QD128, batch_mult8) geometry this is 512 MiB process-wide.
    let recycle_capacity = batch_page_target
        .saturating_mul(16)
        .max(qd.saturating_mul(16))
        .max(1);
    let (recycle_sender, recycle_receiver) = mpsc::sync_channel(recycle_capacity);
    // PageRuntime is process-global and its owner threads never shut down.
    // Leaking these few sender handles lets every 4K PageData borrow one
    // immutable handle instead of cloning/dropping an Arc per physical I/O.
    let recycle_sender: &'static SyncSender<Box<AlignedPage>> = Box::leak(Box::new(recycle_sender));
    let mut free_frames = Vec::with_capacity(recycle_capacity);
    while let Ok(first) = receiver.recv() {
        while let Ok(frame) = recycle_receiver.try_recv() {
            free_frames.push(frame);
        }
        let mut page_count: usize = first.parts.iter().map(|part| part.pages.len()).sum();
        let mut jobs = vec![first];
        let batch_deadline = Instant::now() + Duration::from_micros(batch_micros);
        while page_count < batch_page_target {
            match receiver.try_recv() {
                Ok(job) => {
                    page_count += job.parts.iter().map(|part| part.pages.len()).sum::<usize>();
                    jobs.push(job);
                }
                Err(mpsc::TryRecvError::Empty)
                    if batch_micros != 0 && Instant::now() < batch_deadline =>
                {
                    std::hint::spin_loop();
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }

        COUNTERS.runtime_batches.fetch_add(1, Ordering::Relaxed);
        COUNTERS
            .max_batch_pages
            .fetch_max(page_count as u64, Ordering::Relaxed);

        let mut specs = Vec::with_capacity(page_count);
        let mut part_lengths = Vec::with_capacity(jobs.len());
        for job in &jobs {
            let mut lengths = Vec::with_capacity(job.parts.len());
            for part in &job.parts {
                lengths.push((part.request_idx, part.pages.len()));
                specs.extend(part.pages.iter().map(|&page_number| PageReadSpec {
                    fd: part.file.as_raw_fd(),
                    file_len: part.file_len,
                    page_number,
                }));
            }
            part_lengths.push(lengths);
        }
        let mut reused = 0u64;
        let mut allocated = 0u64;
        let buffers = (0..specs.len())
            .map(|_| {
                let frame = if let Some(frame) = free_frames.pop() {
                    reused += 1;
                    frame
                } else {
                    allocated += 1;
                    Box::new(AlignedPage([0; PAGE_SIZE]))
                };
                PageData::recycled(frame, recycle_sender)
            })
            .collect();
        if reused != 0 {
            COUNTERS.frames_reused.fetch_add(reused, Ordering::Relaxed);
        }
        if allocated != 0 {
            COUNTERS
                .frames_allocated
                .fetch_add(allocated, Ordering::Relaxed);
        }
        match read_specs_local_with_buffers(&specs, qd, buffers) {
            Ok(pages) => {
                let mut pages = pages.into_iter();
                for (job, lengths) in jobs.into_iter().zip(part_lengths) {
                    let job_page_count = lengths.iter().map(|(_, len)| *len).sum();
                    let result = RuntimePageBatch {
                        groups: lengths,
                        pages: pages.by_ref().take(job_page_count).collect(),
                    };
                    let _ = job.response.send(Ok(result));
                }
            }
            Err(error) => {
                let message = error.to_string();
                for job in jobs {
                    let _ = job.response.send(Err(message.clone()));
                }
            }
        }
    }
}

fn page_runtime() -> &'static PageRuntime {
    static RUNTIME: OnceLock<PageRuntime> = OnceLock::new();
    RUNTIME.get_or_init(PageRuntime::new)
}

#[cfg(test)]
fn read_pages_local(
    file: Arc<File>,
    file_len: u64,
    page_numbers: &[u64],
    qd: usize,
) -> io::Result<Vec<PageData>> {
    let specs: Vec<PageReadSpec> = page_numbers
        .iter()
        .map(|&page_number| PageReadSpec {
            fd: file.as_raw_fd(),
            file_len,
            page_number,
        })
        .collect();
    read_specs_local(&specs, qd)
}

struct PageReadSpec {
    // The owning ReadJob (or the test helper's Arc<File>) remains live until
    // every CQE has arrived, so page-sized specs need only carry the stable fd.
    fd: RawFd,
    file_len: u64,
    page_number: u64,
}

#[cfg(test)]
fn read_specs_local(specs: &[PageReadSpec], qd: usize) -> io::Result<Vec<PageData>> {
    let buffers = specs.iter().map(|_| PageData::zeroed()).collect();
    read_specs_local_with_buffers(specs, qd, buffers)
}

fn read_specs_local_with_buffers(
    specs: &[PageReadSpec],
    qd: usize,
    buffers: Vec<PageData>,
) -> io::Result<Vec<PageData>> {
    if specs.is_empty() {
        return Ok(Vec::new());
    }
    let entries = qd.min(specs.len()).max(1) as u32;
    thread_local! {
        static RING: RefCell<Option<(u32, IoUring)>> = const { RefCell::new(None) };
    }
    RING.with(|slot| {
        let mut slot = slot.borrow_mut();
        let needs_ring = slot
            .as_ref()
            .map_or(true, |(capacity, _)| *capacity < entries);
        if needs_ring {
            // Drop the old ring before allocating its replacement so its
            // kernel-accounted pages cannot overlap the new allocation.
            *slot = None;
            let ring = IoUring::new(entries).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!("io_uring setup with {entries} entries failed: {error}"),
                )
            })?;
            *slot = Some((entries, ring));
        }
        if configured_coalesce_pages() >= 2 && has_consecutive_pages(specs) {
            read_pages_on_ring_coalesced(
                &mut slot.as_mut().unwrap().1,
                specs,
                entries,
                buffers,
                configured_coalesce_pages(),
            )
        } else {
            // Preserve the original one-page SQE loop exactly for sparse and
            // cache-hit-heavy workloads. The vectored path is entered only
            // when this physical miss batch actually contains adjacent pages.
            read_pages_on_ring(&mut slot.as_mut().unwrap().1, specs, entries, buffers)
        }
    })
}

fn has_consecutive_pages(specs: &[PageReadSpec]) -> bool {
    specs.windows(2).any(|pair| {
        pair[0].fd == pair[1].fd
            && pair[0].file_len == pair[1].file_len
            && pair[0].page_number.checked_add(1) == Some(pair[1].page_number)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CoalescedRead {
    first: usize,
    pages: usize,
    iovec_start: usize,
}

fn plan_coalesced_reads(specs: &[PageReadSpec], max_pages: usize) -> Vec<CoalescedRead> {
    let mut reads = Vec::with_capacity(specs.len());
    let mut first = 0usize;
    while first < specs.len() {
        let mut pages = 1usize;
        while first + pages < specs.len() && pages < max_pages {
            let previous = &specs[first + pages - 1];
            let next = &specs[first + pages];
            if previous.fd != next.fd
                || previous.file_len != next.file_len
                || previous.page_number.checked_add(1) != Some(next.page_number)
            {
                break;
            }
            pages += 1;
        }
        reads.push(CoalescedRead {
            first,
            pages,
            iovec_start: usize::MAX,
        });
        first += pages;
    }
    reads
}

fn read_pages_on_ring_coalesced(
    ring: &mut IoUring,
    specs: &[PageReadSpec],
    entries: u32,
    mut buffers: Vec<PageData>,
    max_pages: usize,
) -> io::Result<Vec<PageData>> {
    debug_assert_eq!(buffers.len(), specs.len());
    let mut reads = plan_coalesced_reads(specs, max_pages);
    let mut iovecs = Vec::with_capacity(specs.len());
    for read in &mut reads {
        if read.pages == 1 {
            continue;
        }
        read.iovec_start = iovecs.len();
        for buffer in &mut buffers[read.first..read.first + read.pages] {
            iovecs.push(libc::iovec {
                iov_base: buffer.bytes_mut().as_mut_ptr().cast(),
                iov_len: PAGE_SIZE,
            });
        }
    }

    let mut next = 0usize;
    let mut in_flight = 0usize;
    let mut completed = 0usize;
    let mut first_error: Option<io::Error> = None;
    let mut completed_bytes = 0u64;

    while completed < reads.len() {
        while next < reads.len() && in_flight < entries as usize {
            let read = reads[next];
            let spec = &specs[read.first];
            let offset = spec
                .page_number
                .checked_mul(PAGE_SIZE as u64)
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
            let entry = if read.pages == 1 {
                opcode::Read::new(
                    types::Fd(spec.fd),
                    buffers[read.first].bytes_mut().as_mut_ptr(),
                    PAGE_SIZE as u32,
                )
                .offset(offset)
                .build()
                .user_data(next as u64)
            } else {
                opcode::Readv::new(
                    types::Fd(spec.fd),
                    iovecs[read.iovec_start..].as_ptr(),
                    read.pages as u32,
                )
                .offset(offset)
                .build()
                .user_data(next as u64)
            };
            unsafe {
                ring.submission().push(&entry).map_err(|_| {
                    io::Error::new(io::ErrorKind::WouldBlock, "io_uring SQ unexpectedly full")
                })?;
            }
            next += 1;
            in_flight += 1;
        }

        loop {
            match ring.submit_and_wait(1) {
                Ok(_) => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
        let mut completion = ring.completion();
        for cqe in &mut completion {
            let read_idx = cqe.user_data() as usize;
            let read = reads[read_idx];
            let spec = &specs[read.first];
            let offset = spec.page_number * PAGE_SIZE as u64;
            let requested = (read.pages as u64)
                .checked_mul(PAGE_SIZE as u64)
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
            let expected = (spec.file_len - offset).min(requested) as usize;
            let result = cqe.result();
            if result < 0 {
                if first_error.is_none() {
                    first_error = Some(io::Error::new(
                        io::Error::from_raw_os_error(-result).kind(),
                        format!(
                            "io_uring {}-page read at offset {offset} failed: {}",
                            read.pages,
                            io::Error::from_raw_os_error(-result)
                        ),
                    ));
                }
            } else if result as usize != expected {
                if first_error.is_none() {
                    first_error = Some(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!(
                            "{}-page read at {offset} completed with {result} bytes, expected {expected}",
                            read.pages
                        ),
                    ));
                }
            } else {
                let mut remaining = expected;
                for buffer in &mut buffers[read.first..read.first + read.pages] {
                    let valid = remaining.min(PAGE_SIZE);
                    buffer.valid_len = valid;
                    remaining -= valid;
                }
                completed_bytes += expected as u64;
            }
            in_flight -= 1;
            completed += 1;
        }
    }

    if let Some(error) = first_error {
        return Err(error);
    }
    let multi_page_ops = reads.iter().filter(|read| read.pages > 1).count() as u64;
    let multi_page_pages = reads
        .iter()
        .filter(|read| read.pages > 1)
        .map(|read| read.pages as u64)
        .sum::<u64>();
    COUNTERS.bytes.fetch_add(completed_bytes, Ordering::Relaxed);
    COUNTERS
        .submitted
        .fetch_add(specs.len() as u64, Ordering::Relaxed);
    COUNTERS
        .io_ops
        .fetch_add(reads.len() as u64, Ordering::Relaxed);
    COUNTERS
        .multi_page_ops
        .fetch_add(multi_page_ops, Ordering::Relaxed);
    COUNTERS
        .multi_page_pages
        .fetch_add(multi_page_pages, Ordering::Relaxed);
    Ok(buffers)
}

fn read_pages_on_ring(
    ring: &mut IoUring,
    specs: &[PageReadSpec],
    entries: u32,
    mut buffers: Vec<PageData>,
) -> io::Result<Vec<PageData>> {
    debug_assert_eq!(buffers.len(), specs.len());
    let mut next = 0usize;
    let mut in_flight = 0usize;
    let mut completed = 0usize;
    let mut first_error: Option<io::Error> = None;
    let mut completed_bytes = 0u64;

    while completed < specs.len() {
        while next < specs.len() && in_flight < entries as usize {
            let offset = specs[next]
                .page_number
                .checked_mul(PAGE_SIZE as u64)
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
            let entry = opcode::Read::new(
                types::Fd(specs[next].fd),
                buffers[next].bytes_mut().as_mut_ptr(),
                PAGE_SIZE as u32,
            )
            .offset(offset)
            .build()
            .user_data(next as u64);
            unsafe {
                ring.submission().push(&entry).map_err(|_| {
                    io::Error::new(io::ErrorKind::WouldBlock, "io_uring SQ unexpectedly full")
                })?;
            }
            next += 1;
            in_flight += 1;
        }

        // SIGPROF and other process signals may interrupt io_uring_enter after
        // the kernel has accepted submissions. Returning here would drop
        // buffers that are still in flight and leave their CQEs for the next
        // owner batch. Retry EINTR until those owned buffers are completed.
        loop {
            match ring.submit_and_wait(1) {
                Ok(_) => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
        let mut completion = ring.completion();
        for cqe in &mut completion {
            let idx = cqe.user_data() as usize;
            let offset = specs[idx].page_number * PAGE_SIZE as u64;
            let expected = (specs[idx].file_len - offset).min(PAGE_SIZE as u64) as usize;
            let result = cqe.result();
            if result < 0 {
                if first_error.is_none() {
                    first_error = Some(io::Error::new(
                        io::Error::from_raw_os_error(-result).kind(),
                        format!(
                            "io_uring page read at offset {offset} failed: {}",
                            io::Error::from_raw_os_error(-result)
                        ),
                    ));
                }
            } else if result as usize != expected {
                if first_error.is_none() {
                    first_error = Some(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!(
                            "page read at {offset} completed with {result} bytes, expected {expected}"
                        ),
                    ));
                }
            } else {
                buffers[idx].valid_len = expected;
                completed_bytes += expected as u64;
            }
            in_flight -= 1;
            completed += 1;
        }
    }

    if let Some(error) = first_error {
        return Err(error);
    }
    COUNTERS.bytes.fetch_add(completed_bytes, Ordering::Relaxed);
    COUNTERS
        .submitted
        .fetch_add(specs.len() as u64, Ordering::Relaxed);
    COUNTERS
        .io_ops
        .fetch_add(specs.len() as u64, Ordering::Relaxed);
    Ok(buffers)
}

const CACHE_TARGET_WAYS: usize = 6;
const CACHE_LOADING_BIT: u64 = 1 << 63;
const CACHE_MISS_LOCKS: usize = 16 * 1024;
const PIN_EXCLUSIVE: u16 = 1 << 15;
const PIN_COUNT_MASK: u16 = PIN_EXCLUSIVE - 1;

#[repr(align(64))]
struct PaddedMutex(Mutex<()>);

static CACHE_CAPACITY_PAGES: AtomicU64 = AtomicU64::new(0);
static CACHE_ENTRIES: AtomicU64 = AtomicU64::new(0);

struct CacheArena {
    ptr: NonNull<u8>,
    len: usize,
}

unsafe impl Send for CacheArena {}
unsafe impl Sync for CacheArena {}

impl CacheArena {
    fn new(page_count: usize) -> io::Result<Self> {
        let len = page_count
            .checked_mul(PAGE_SIZE)
            .ok_or_else(|| other_error("direct-cache arena size overflow"))?;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let ptr: NonNull<u8> = NonNull::new(ptr.cast()).unwrap();
        unsafe {
            libc::madvise(ptr.as_ptr().cast(), len, libc::MADV_DONTDUMP);
            libc::madvise(ptr.as_ptr().cast(), len, libc::MADV_DONTFORK);
            libc::madvise(ptr.as_ptr().cast(), len, libc::MADV_NOHUGEPAGE);
        }
        Ok(Self { ptr, len })
    }

    fn page_ptr(&self, slot: usize) -> *mut u8 {
        unsafe { self.ptr.as_ptr().add(slot * PAGE_SIZE) }
    }

    fn copy_page(&self, slot: usize, bytes: &[u8]) {
        debug_assert!(bytes.len() <= PAGE_SIZE);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.page_ptr(slot), bytes.len());
        }
    }

    fn page(&self, slot: usize, valid_len: usize) -> &[u8] {
        debug_assert!(valid_len <= PAGE_SIZE);
        unsafe { std::slice::from_raw_parts(self.page_ptr(slot), valid_len) }
    }
}

impl Drop for CacheArena {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast(), self.len);
        }
    }
}

struct DirectCache {
    arena: CacheArena,
    set_count: usize,
    ways: usize,
    tags: Box<[AtomicU64]>,
    pins: Box<[AtomicU16]>,
    valid_lens: Box<[AtomicU16]>,
    referenced: Box<[AtomicU8]>,
    rotors: Box<[AtomicU8]>,
    miss_locks: Box<[PaddedMutex]>,
}

struct CacheLease {
    cache: &'static DirectCache,
    slot: usize,
}

impl CacheLease {
    fn valid(&self) -> &[u8] {
        self.cache.arena.page(self.slot, self.valid_len())
    }

    fn valid_len(&self) -> usize {
        self.cache.valid_lens[self.slot].load(Ordering::Relaxed) as usize
    }
}

impl Drop for CacheLease {
    fn drop(&mut self) {
        let previous = self.cache.pins[self.slot].fetch_sub(1, Ordering::Release);
        debug_assert!(previous > 0);
    }
}

enum Lookup {
    Hit(CacheLease),
    Read { slot: Option<usize> },
}

struct PlannedRead {
    idx: usize,
    key: PageKey,
    slot: Option<usize>,
}

struct DirectReadPlan {
    cache: &'static DirectCache,
    out: Vec<PreparedPage>,
    reads: Vec<PlannedRead>,
}

impl DirectReadPlan {
    fn fail(&self, _message: &str) {
        for read in &self.reads {
            if let Some(slot) = read.slot {
                self.cache.abort_fill(slot, read.key.packed());
            }
        }
    }

    fn publish_reads<I>(&mut self, pages: I)
    where
        I: IntoIterator<Item = PageData>,
        I::IntoIter: ExactSizeIterator,
    {
        let pages = pages.into_iter();
        debug_assert_eq!(pages.len(), self.reads.len());
        for (read, page) in self.reads.iter_mut().zip(pages) {
            self.out[read.idx] = match read.slot.take() {
                Some(slot) => PreparedPage::Cached(self.cache.publish_fill(
                    slot,
                    read.key.packed(),
                    page.valid(),
                )),
                None => PreparedPage::Transient(Box::new(page)),
            };
        }
    }

    fn finish_ready(mut self) -> io::Result<Vec<PreparedPage>> {
        if self
            .out
            .iter()
            .any(|page| matches!(page, PreparedPage::Missing))
        {
            return Err(other_error("direct cache failed to produce page"));
        }
        Ok(std::mem::take(&mut self.out))
    }

    fn finish(mut self, pages: Vec<PageData>) -> io::Result<Vec<PreparedPage>> {
        if pages.len() != self.reads.len() {
            let message = format!(
                "edge-I/O cache batch returned {} pages for {} misses",
                pages.len(),
                self.reads.len()
            );
            self.fail(&message);
            return Err(other_error(message));
        }
        self.publish_reads(pages);
        self.finish_ready()
    }
}

impl Drop for DirectReadPlan {
    fn drop(&mut self) {
        // A cancelled batch still owns every reservation it has not
        // published. Return those LOADING slots so they cannot become
        // permanent transient-read bypasses.
        for read in &mut self.reads {
            if let Some(slot) = read.slot.take() {
                self.cache.abort_fill(slot, read.key.packed());
            }
        }
    }
}

pub struct PendingDirectRead {
    plan: Option<DirectReadPlan>,
    result: Option<RuntimeResponse>,
}

impl PendingDirectRead {
    fn finish(mut self) -> io::Result<Vec<PreparedPage>> {
        let plan = self.plan.take().unwrap();
        let pages = if let Some(result) = self.result.take() {
            let received = result
                .recv()
                .map_err(|_| other_error("edge-I/O owner dropped direct-cache response"))?;
            match received {
                Ok(batch)
                    if batch.groups.as_slice() == [(0, batch.pages.len())]
                        && batch.pages.len() == plan.reads.len() =>
                {
                    batch.pages
                }
                Ok(batch) => {
                    let error = other_error(format!(
                        "edge-I/O owner returned {} direct-cache pages for {} misses",
                        batch.pages.len(),
                        plan.reads.len()
                    ));
                    plan.fail(&error.to_string());
                    return Err(error);
                }
                Err(message) => {
                    plan.fail(&message);
                    return Err(other_error(message));
                }
            }
        } else {
            Vec::new()
        };
        plan.finish(pages)
    }
}

impl DirectCache {
    fn reserve_pin(&self, slot: usize) -> bool {
        let mut current = self.pins[slot].load(Ordering::Relaxed);
        loop {
            if current & PIN_EXCLUSIVE != 0 || current == PIN_COUNT_MASK {
                return false;
            }
            match self.pins[slot].compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    fn from_env() -> io::Result<Self> {
        let cache_gib = configured_direct_cache_gib();
        let requested_pages = cache_gib
            .checked_mul(1024 * 1024 * 1024usize / PAGE_SIZE)
            .ok_or_else(|| other_error("direct-cache size overflow"))?;
        let target_sets = (requested_pages / CACHE_TARGET_WAYS).max(1);
        let upper_sets = target_sets.next_power_of_two();
        let upper_ways = requested_pages / upper_sets;
        let set_count = if upper_ways >= 4 {
            upper_sets
        } else {
            upper_sets / 2
        }
        .max(1);
        let ways = (requested_pages / set_count).max(1);
        Self::new(set_count, ways)
    }

    fn new(set_count: usize, ways: usize) -> io::Result<Self> {
        if !set_count.is_power_of_two() || ways == 0 || ways > u8::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "direct-cache set count must be a power of two and ways must fit in u8",
            ));
        }
        let slot_count = set_count
            .checked_mul(ways)
            .ok_or_else(|| other_error("direct-cache slot count overflow"))?;
        let atomic_u64s = |count| {
            std::iter::repeat_with(|| AtomicU64::new(0))
                .take(count)
                .collect::<Vec<_>>()
                .into_boxed_slice()
        };
        let atomic_u16s = |count| {
            std::iter::repeat_with(|| AtomicU16::new(0))
                .take(count)
                .collect::<Vec<_>>()
                .into_boxed_slice()
        };
        let atomic_u8s = |count| {
            std::iter::repeat_with(|| AtomicU8::new(0))
                .take(count)
                .collect::<Vec<_>>()
                .into_boxed_slice()
        };
        let arena = CacheArena::new(slot_count)?;
        CACHE_CAPACITY_PAGES.store(slot_count as u64, Ordering::Relaxed);
        CACHE_ENTRIES.store(0, Ordering::Relaxed);
        eprintln!(
            "[diskgraph] fixed direct cache: {:.1} GiB data, {} sets x {} ways, {:.1}M slots",
            slot_count as f64 * PAGE_SIZE as f64 / (1u64 << 30) as f64,
            set_count,
            ways,
            slot_count as f64 / 1_000_000.0,
        );
        Ok(Self {
            arena,
            set_count,
            ways,
            tags: atomic_u64s(slot_count),
            pins: atomic_u16s(slot_count),
            valid_lens: atomic_u16s(slot_count),
            referenced: atomic_u8s(slot_count),
            rotors: atomic_u8s(set_count),
            miss_locks: std::iter::repeat_with(|| PaddedMutex(Mutex::new(())))
                .take(CACHE_MISS_LOCKS)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    }

    fn set_for(&self, key: u64) -> usize {
        let mut mixed = key;
        mixed ^= mixed >> 30;
        mixed = mixed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed ^= mixed >> 27;
        mixed = mixed.wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^= mixed >> 31;
        mixed as usize & (self.set_count - 1)
    }

    fn try_pin_matching(&'static self, slot: usize, key: u64) -> Option<CacheLease> {
        // scan() already observed the matching tag with Acquire. Pin, then
        // revalidate once in case replacement retired it in between.
        if !self.reserve_pin(slot) {
            return None;
        }
        if self.tags[slot].load(Ordering::Acquire) == key {
            // Popular pages normally remain referenced. Avoid dirtying this
            // densely packed shared metadata line on every cache hit.
            if self.referenced[slot].load(Ordering::Relaxed) == 0 {
                self.referenced[slot].store(1, Ordering::Relaxed);
            }
            Some(CacheLease { cache: self, slot })
        } else {
            self.pins[slot].fetch_sub(1, Ordering::Release);
            None
        }
    }

    fn scan(&'static self, base: usize, key: u64, events: &mut CacheEvents) -> Option<Lookup> {
        let loading = key | CACHE_LOADING_BIT;
        for slot in base..base + self.ways {
            let tag = self.tags[slot].load(Ordering::Acquire);
            if tag == key {
                if let Some(lease) = self.try_pin_matching(slot, key) {
                    return Some(Lookup::Hit(lease));
                }
            } else if tag == loading {
                // Waiting and notifying costs vastly more than the rare
                // duplicate 4K read on this workload. Bypass admission for a
                // key that is already loading; only its owner will publish.
                events.coalesced += 1;
                events.misses += 1;
                return Some(Lookup::Read { slot: None });
            }
        }
        None
    }

    fn lookup(&'static self, page_key: PageKey, events: &mut CacheEvents) -> Lookup {
        let key = page_key.packed();
        let set = self.set_for(key);
        let base = set * self.ways;
        if let Some(found) = self.scan(base, key, events) {
            return found;
        }

        let _miss_guard = self.miss_locks[set & (CACHE_MISS_LOCKS - 1)]
            .0
            .lock()
            .unwrap();
        if let Some(found) = self.scan(base, key, events) {
            return found;
        }

        let loading = key | CACHE_LOADING_BIT;
        for slot in base..base + self.ways {
            if self.tags[slot]
                .compare_exchange(0, loading, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                debug_assert_eq!(self.pins[slot].load(Ordering::Relaxed), 0);
                self.pins[slot].store(PIN_EXCLUSIVE, Ordering::Relaxed);
                CACHE_ENTRIES.fetch_add(1, Ordering::Relaxed);
                events.misses += 1;
                return Lookup::Read { slot: Some(slot) };
            }
        }

        let rotor = self.rotors[set].load(Ordering::Relaxed);
        self.rotors[set].store(rotor.wrapping_add(1), Ordering::Relaxed);
        let start = rotor as usize % self.ways;
        for clear_referenced in [true, false] {
            for step in 0..self.ways {
                let slot = base + (start + step) % self.ways;
                let old = self.tags[slot].load(Ordering::Acquire);
                if old & CACHE_LOADING_BIT != 0 {
                    continue;
                }
                if clear_referenced && self.referenced[slot].load(Ordering::Relaxed) != 0 {
                    self.referenced[slot].store(0, Ordering::Relaxed);
                    continue;
                }
                if self.pins[slot]
                    .compare_exchange(0, PIN_EXCLUSIVE, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    // The exclusive pin is the single synchronization word:
                    // no reader can acquire the retiring frame after this.
                    debug_assert_eq!(self.tags[slot].load(Ordering::Relaxed), old);
                    self.tags[slot].store(loading, Ordering::Release);
                    events.evictions += 1;
                    events.misses += 1;
                    return Lookup::Read { slot: Some(slot) };
                }
            }
        }

        events.misses += 1;
        Lookup::Read { slot: None }
    }

    fn publish_fill(&'static self, slot: usize, key: u64, bytes: &[u8]) -> CacheLease {
        debug_assert_eq!(
            self.tags[slot].load(Ordering::Acquire),
            key | CACHE_LOADING_BIT
        );
        self.arena.copy_page(slot, bytes);
        self.valid_lens[slot].store(bytes.len() as u16, Ordering::Relaxed);
        self.referenced[slot].store(1, Ordering::Relaxed);
        // LOADING is exclusively owned by this fill and replacement waited
        // for every old lease before installing it, so publication cannot
        // race a pin. Avoid an atomic RMW on every physical miss.
        debug_assert_eq!(self.pins[slot].load(Ordering::Relaxed), PIN_EXCLUSIVE);
        self.pins[slot].store(1, Ordering::Relaxed);
        self.tags[slot].store(key, Ordering::Release);
        CacheLease { cache: self, slot }
    }

    fn abort_fill(&self, slot: usize, key: u64) {
        // A LOADING slot has exactly one owner and its exclusive pin prevents
        // all readers. Clear that pin before making the slot empty.
        if self.tags[slot].load(Ordering::Acquire) == (key | CACHE_LOADING_BIT) {
            debug_assert_eq!(self.pins[slot].load(Ordering::Relaxed), PIN_EXCLUSIVE);
            self.pins[slot].store(0, Ordering::Relaxed);
            self.tags[slot].store(0, Ordering::Release);
            CACHE_ENTRIES.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn begin_read(
        &'static self,
        file: Arc<File>,
        file_id: u64,
        file_len: u64,
        page_numbers: &[u64],
    ) -> io::Result<PendingDirectRead> {
        let plan = self.plan_read(file_id, page_numbers);
        let missing: Vec<u64> = plan.reads.iter().map(|read| read.key.page_number).collect();
        let result = if missing.is_empty() {
            None
        } else {
            match page_runtime().submit(Arc::clone(&file), file_id, file_len, &missing) {
                Ok(result) => Some(result),
                Err(error) => {
                    plan.fail(&error.to_string());
                    return Err(error);
                }
            }
        };
        Ok(PendingDirectRead {
            plan: Some(plan),
            result,
        })
    }

    fn plan_read(&'static self, file_id: u64, page_numbers: &[u64]) -> DirectReadPlan {
        let mut out: Vec<PreparedPage> = (0..page_numbers.len())
            .map(|_| PreparedPage::Missing)
            .collect();
        // The full-page capacity avoids several growth allocations on the
        // miss-heavy path. It is temporary and bounded by the already-owned
        // page-number list; cache hits simply leave unused capacity.
        let mut reads = Vec::with_capacity(page_numbers.len());
        let mut events = CacheEvents::default();
        for (idx, &page_number) in page_numbers.iter().enumerate() {
            let page_key = PageKey {
                file_id,
                page_number,
            };
            match self.lookup(page_key, &mut events) {
                Lookup::Hit(page) => {
                    events.hits += 1;
                    out[idx] = PreparedPage::Cached(page);
                }
                Lookup::Read { slot } => reads.push(PlannedRead {
                    idx,
                    key: page_key,
                    slot,
                }),
            }
        }
        record_cache_events(events);
        DirectReadPlan {
            cache: self,
            out,
            reads,
        }
    }
}

fn direct_cache() -> io::Result<&'static DirectCache> {
    static CACHE: OnceLock<Result<DirectCache, String>> = OnceLock::new();
    match CACHE.get_or_init(|| DirectCache::from_env().map_err(|error| error.to_string())) {
        Ok(cache) => Ok(cache),
        Err(message) => Err(other_error(message.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    use std::sync::Barrier;

    fn lookup(cache: &'static DirectCache, key: PageKey) -> Lookup {
        cache.lookup(key, &mut CacheEvents::default())
    }

    #[test]
    fn range_page_planning_deduplicates_and_handles_boundaries() {
        let pages = pages_for_ranges(&[(4093, 4100), (4096, 8192), (0, 0)], 8192).unwrap();
        assert_eq!(pages, vec![0, 1]);
    }

    #[test]
    fn prepared_pages_respects_valid_len() {
        let mut page = PageData::zeroed();
        page.bytes_mut()[..3].copy_from_slice(&[1, 2, 3]);
        page.valid_len = 3;
        let prepared = PreparedPages::new_transient(vec![0], vec![page]);
        assert_eq!(prepared.copy_range(0, 3).unwrap(), vec![1, 2, 3]);
        assert_eq!(
            prepared.copy_range(0, 4).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn contiguous_range_borrows_one_page_and_rejects_a_straddle() {
        let mut first = PageData::zeroed();
        first.bytes_mut()[PAGE_SIZE - 2..].copy_from_slice(&[7, 8]);
        first.valid_len = PAGE_SIZE;
        let mut second = PageData::zeroed();
        second.bytes_mut()[..2].copy_from_slice(&[9, 10]);
        second.valid_len = 2;
        let prepared = PreparedPages::new_transient(vec![0, 1], vec![first, second]);

        assert_eq!(
            prepared
                .contiguous_range(PAGE_SIZE as u64 - 2, PAGE_SIZE as u64)
                .unwrap(),
            Some(&[7, 8][..])
        );
        assert!(prepared
            .contiguous_range(PAGE_SIZE as u64 - 1, PAGE_SIZE as u64 + 1)
            .unwrap()
            .is_none());
        assert_eq!(
            prepared
                .copy_range(PAGE_SIZE as u64 - 1, PAGE_SIZE as u64 + 1)
                .unwrap(),
            vec![8, 9]
        );
    }

    #[test]
    fn prepared_page_cursor_advances_without_per_range_search() {
        let mut first = PageData::zeroed();
        first.bytes_mut()[7] = 17;
        first.valid_len = PAGE_SIZE;
        let mut second = PageData::zeroed();
        second.bytes_mut()[11] = 23;
        second.valid_len = PAGE_SIZE;
        let prepared = PreparedPages::new_transient(vec![3, 9], vec![first, second]);
        let mut cursor = prepared.cursor_at(3 * PAGE_SIZE as u64 + 7);
        assert_eq!(
            cursor
                .contiguous_range(3 * PAGE_SIZE as u64 + 7, 3 * PAGE_SIZE as u64 + 8)
                .unwrap(),
            Some(&[17][..])
        );
        assert_eq!(
            cursor
                .contiguous_range(9 * PAGE_SIZE as u64 + 11, 9 * PAGE_SIZE as u64 + 12)
                .unwrap(),
            Some(&[23][..])
        );
    }

    #[test]
    fn coalesced_read_plan_respects_files_gaps_and_cap() {
        let specs = [
            PageReadSpec {
                fd: 7,
                file_len: 20 * PAGE_SIZE as u64,
                page_number: 2,
            },
            PageReadSpec {
                fd: 7,
                file_len: 20 * PAGE_SIZE as u64,
                page_number: 3,
            },
            PageReadSpec {
                fd: 7,
                file_len: 20 * PAGE_SIZE as u64,
                page_number: 4,
            },
            PageReadSpec {
                fd: 7,
                file_len: 20 * PAGE_SIZE as u64,
                page_number: 9,
            },
            PageReadSpec {
                fd: 8,
                file_len: 20 * PAGE_SIZE as u64,
                page_number: 10,
            },
        ];
        assert!(has_consecutive_pages(&specs));
        assert_eq!(
            plan_coalesced_reads(&specs, 2),
            vec![
                CoalescedRead {
                    first: 0,
                    pages: 2,
                    iovec_start: usize::MAX,
                },
                CoalescedRead {
                    first: 2,
                    pages: 1,
                    iovec_start: usize::MAX,
                },
                CoalescedRead {
                    first: 3,
                    pages: 1,
                    iovec_start: usize::MAX,
                },
                CoalescedRead {
                    first: 4,
                    pages: 1,
                    iovec_start: usize::MAX,
                },
            ]
        );
    }

    #[test]
    fn io_uring_accepts_exact_eof_short_completion() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let mut bytes = vec![0u8; PAGE_SIZE + 1373];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }
        tmp.write_all(&bytes).unwrap();
        tmp.as_file().sync_all().unwrap();

        let buffered_file = Arc::new(tmp.as_file().try_clone().unwrap());
        let buffered = match read_pages_local(buffered_file, bytes.len() as u64, &[0, 1], 2) {
            Ok(pages) => pages,
            // Some CI/seccomp sandboxes reject io_uring_setup despite the host
            // kernel supporting it. The same test is run unsandboxed on the
            // benchmark host.
            Err(error)
                if error.kind() == io::ErrorKind::PermissionDenied
                    || error.to_string().contains("Operation not permitted") =>
            {
                return;
            }
            Err(error) => panic!("buffered io_uring read failed: {error}"),
        };
        assert_eq!(buffered[0].valid_len(), PAGE_SIZE);
        assert_eq!(buffered[1].valid_len(), 1373);
        assert_eq!(buffered[1].valid(), &bytes[PAGE_SIZE..]);

        let direct = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT | libc::O_CLOEXEC)
            .open(tmp.path())
            .unwrap();
        let direct_pages =
            read_pages_local(Arc::new(direct), bytes.len() as u64, &[0, 1], 2).unwrap();
        assert_eq!(direct_pages[0].valid_len(), PAGE_SIZE);
        assert_eq!(direct_pages[0].valid(), &bytes[..PAGE_SIZE]);
        assert_eq!(direct_pages[1].valid_len(), 1373);
        assert_eq!(direct_pages[1].valid(), &bytes[PAGE_SIZE..]);
    }

    #[test]
    fn batched_requests_preserve_request_and_page_mapping() {
        let mut first = tempfile::NamedTempFile::new().unwrap();
        let mut first_bytes = vec![0u8; PAGE_SIZE + 29];
        first_bytes[..PAGE_SIZE].fill(11);
        first_bytes[PAGE_SIZE..].fill(12);
        first.write_all(&first_bytes).unwrap();
        first.as_file().sync_all().unwrap();

        let mut second = tempfile::NamedTempFile::new().unwrap();
        let second_bytes = vec![23u8; PAGE_SIZE];
        second.write_all(&second_bytes).unwrap();
        second.as_file().sync_all().unwrap();

        let first_file = Arc::new(first.as_file().try_clone().unwrap());
        let second_file = Arc::new(second.as_file().try_clone().unwrap());
        let requests = vec![
            page_read_request(
                &first_file,
                None,
                101,
                first_bytes.len() as u64,
                EdgeIoBackend::Buffered,
                &[1, 0, 1],
            )
            .unwrap(),
            page_read_request(
                &second_file,
                None,
                202,
                second_bytes.len() as u64,
                EdgeIoBackend::Buffered,
                &[0],
            )
            .unwrap(),
            page_read_request(
                &second_file,
                None,
                303,
                second_bytes.len() as u64,
                EdgeIoBackend::Buffered,
                &[],
            )
            .unwrap(),
        ];

        let prepared =
            match submit_prepared_page_batch(requests).and_then(|pending| pending.finish()) {
                Ok(prepared) => prepared,
                // Some CI/seccomp sandboxes reject io_uring_setup despite the host
                // kernel supporting it. The benchmark host runs this test natively.
                Err(error)
                    if error.kind() == io::ErrorKind::PermissionDenied
                        || error.to_string().contains("Operation not permitted") =>
                {
                    return;
                }
                Err(error) => panic!("batched buffered io_uring read failed: {error}"),
            };
        assert_eq!(prepared.len(), 3);
        assert_eq!(prepared[0].page_count(), 2);
        assert_eq!(prepared[0].copy_range(0, 1).unwrap(), vec![11]);
        assert_eq!(
            prepared[0]
                .copy_range(PAGE_SIZE as u64, PAGE_SIZE as u64 + 29)
                .unwrap(),
            vec![12; 29]
        );
        assert_eq!(prepared[1].page_count(), 1);
        assert_eq!(prepared[1].copy_range(0, 1).unwrap(), vec![23]);
        assert_eq!(prepared[2].page_count(), 0);
    }

    #[test]
    fn concurrent_loading_lookups_bypass_without_waiting() {
        let cache = Box::leak(Box::new(DirectCache::new(32, 6).unwrap()));
        let barrier = Arc::new(Barrier::new(200));
        let bypasses = Arc::new(AtomicU64::new(0));
        let key = PageKey {
            file_id: 7,
            page_number: 11,
        };

        let owner_slot = match lookup(cache, key) {
            Lookup::Read { slot: Some(slot) } => slot,
            _ => panic!("first lookup did not reserve an empty cache slot"),
        };
        std::thread::scope(|scope| {
            for _ in 0..200 {
                let cache = &*cache;
                let barrier = Arc::clone(&barrier);
                let bypasses = Arc::clone(&bypasses);
                scope.spawn(move || {
                    barrier.wait();
                    match lookup(cache, key) {
                        Lookup::Read { slot: None } => {
                            bypasses.fetch_add(1, Ordering::Relaxed);
                        }
                        _ => panic!("loading lookup did not bypass cache admission"),
                    }
                });
            }
        });
        assert_eq!(bypasses.load(Ordering::Relaxed), 200);

        let mut page = PageData::zeroed();
        page.valid_len = 17;
        drop(cache.publish_fill(owner_slot, key.packed(), page.valid()));
        match lookup(cache, key) {
            Lookup::Hit(page) => assert_eq!(page.valid_len(), 17),
            _ => panic!("published page was not a cache hit"),
        }
    }

    #[test]
    fn dropping_unfinished_plan_releases_loading_reservation() {
        let cache = Box::leak(Box::new(DirectCache::new(32, 6).unwrap()));
        let key = PageKey {
            file_id: 9,
            page_number: 27,
        };
        let plan = cache.plan_read(key.file_id, &[key.page_number]);
        assert_eq!(plan.reads.len(), 1);
        let slot = plan.reads[0].slot.unwrap();
        assert_eq!(
            cache.tags[slot].load(Ordering::Acquire),
            key.packed() | CACHE_LOADING_BIT
        );
        drop(plan);
        assert_eq!(cache.tags[slot].load(Ordering::Acquire), 0);
        assert_eq!(cache.pins[slot].load(Ordering::Acquire), 0);

        match lookup(cache, key) {
            Lookup::Read { slot: Some(slot) } => cache.abort_fill(slot, key.packed()),
            _ => panic!("cancelled cache reservation was not reusable"),
        }
    }
}

#[cfg(test)]
#[path = "page_io_readiness_tests.rs"]
mod readiness_tests;
