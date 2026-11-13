// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit 4 KiB page I/O used by the experimental sampler backends.
//!
//! The production mmap reader remains separate.  This module deliberately
//! owns every destination buffer until its CQE arrives and accepts a short
//! completion only for the logical final page of an immutable file.

#[cfg(target_os = "linux")]
use io_uring::{opcode, types, IoUring};
use std::cell::Cell;
#[cfg(target_os = "linux")]
use std::cell::RefCell;
use std::fs::File;
use std::io;
use std::mem::ManuallyDrop;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, OnceLock};
#[cfg(target_os = "linux")]
use std::time::{Duration, Instant};

fn other_error(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::Other, error)
}

pub const PAGE_SIZE: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeIoBackend {
    Mmap,
    Buffered,
}

impl EdgeIoBackend {
    pub fn from_env() -> io::Result<Self> {
        let configured = match std::env::var("DG_EDGE_BACKEND") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "mmap".to_string(),
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "DG_EDGE_BACKEND must be valid UTF-8 and one of mmap|buffered",
                ));
            }
        };
        let backend = match configured.as_str() {
            "mmap" => Self::Mmap,
            "buffered" => Self::Buffered,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("DG_EDGE_BACKEND must be mmap|buffered, got `{other}`"),
                ))
            }
        };
        backend.ensure_supported()?;
        Ok(backend)
    }

    pub fn ensure_supported(self) -> io::Result<()> {
        if self != Self::Mmap && !cfg!(target_os = "linux") {
            return Err(unsupported_backend());
        }
        Ok(())
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mmap => "mmap",
            Self::Buffered => "buffered",
        }
    }
}

fn unsupported_backend() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "buffered edge I/O requires Linux io_uring; use DG_EDGE_BACKEND=mmap on this platform",
    )
}

#[repr(align(4096))]
pub struct AlignedPage(pub [u8; PAGE_SIZE]);

pub struct PageData {
    // Keep the 4K-aligned allocation separate from the small descriptor so
    // moving completed pages between the I/O owner and sampler stays cheap.
    bytes: Option<Box<AlignedPage>>,
    valid_len: usize,
    recycler: Option<(&'static SyncSender<Box<AlignedPage>>, u32)>,
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
            recycler: Some((recycler, std::process::id())),
        }
    }

    #[cfg(any(target_os = "linux", test))]
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
        if let (Some(bytes), Some((recycler, pid))) = (self.bytes.take(), self.recycler.as_ref()) {
            // A forked child has no owner for the inherited recycle channel.
            if *pid != std::process::id() {
                return;
            }
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
    pub bytes: u64,
    pub frames_allocated: u64,
    pub frames_reused: u64,
    pub frames_recycled: u64,
    pub frames_dropped: u64,
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

thread_local! {
    static RECYCLED_PENDING: Cell<u64> = const { Cell::new(0) };
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

pub fn stats() -> PageIoStats {
    PageIoStats {
        submitted: COUNTERS.submitted.load(Ordering::Relaxed),
        io_ops: COUNTERS.io_ops.load(Ordering::Relaxed),
        multi_page_ops: COUNTERS.multi_page_ops.load(Ordering::Relaxed),
        multi_page_pages: COUNTERS.multi_page_pages.load(Ordering::Relaxed),
        runtime_jobs: COUNTERS.runtime_jobs.load(Ordering::Relaxed),
        runtime_envelopes: COUNTERS.runtime_envelopes.load(Ordering::Relaxed),
        runtime_batches: COUNTERS.runtime_batches.load(Ordering::Relaxed),
        max_batch_pages: COUNTERS.max_batch_pages.load(Ordering::Relaxed),
        bytes: COUNTERS.bytes.load(Ordering::Relaxed),
        frames_allocated: COUNTERS.frames_allocated.load(Ordering::Relaxed),
        frames_reused: COUNTERS.frames_reused.load(Ordering::Relaxed),
        frames_recycled: COUNTERS.frames_recycled.load(Ordering::Relaxed),
        frames_dropped: COUNTERS.frames_dropped.load(Ordering::Relaxed),
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
    COUNTERS.bytes.store(0, Ordering::Relaxed);
    COUNTERS.frames_allocated.store(0, Ordering::Relaxed);
    COUNTERS.frames_reused.store(0, Ordering::Relaxed);
    COUNTERS.frames_recycled.store(0, Ordering::Relaxed);
    COUNTERS.frames_dropped.store(0, Ordering::Relaxed);
}

pub struct PreparedPages {
    page_numbers: Vec<u64>,
    pages: Vec<PageData>,
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

pub enum PendingPreparedPages {
    Ready(PreparedPages),
    Runtime {
        page_numbers: Vec<u64>,
        result: RuntimeResponse,
    },
}

impl PendingPreparedPages {
    pub fn finish(self) -> io::Result<PreparedPages> {
        match self {
            Self::Ready(pages) => Ok(pages),
            Self::Runtime {
                page_numbers,
                result,
            } => match result.recv()? {
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
            pages: data,
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
    file_len: u64,
    backend: EdgeIoBackend,
    page_numbers: &[u64],
) -> io::Result<PreparedPages> {
    submit_prepared_pages(buffered_file, file_len, backend, page_numbers)?.finish()
}

pub struct PageReadRequest {
    buffered_file: Arc<File>,
    file_len: u64,
    page_numbers: Vec<u64>,
}

pub fn page_read_request(
    buffered_file: &Arc<File>,
    file_len: u64,
    backend: EdgeIoBackend,
    page_numbers: &[u64],
) -> io::Result<PageReadRequest> {
    page_read_request_owned(buffered_file, file_len, backend, page_numbers.to_vec())
}

pub fn page_read_request_owned(
    buffered_file: &Arc<File>,
    file_len: u64,
    backend: EdgeIoBackend,
    mut page_numbers: Vec<u64>,
) -> io::Result<PageReadRequest> {
    backend.ensure_supported()?;
    page_numbers.sort_unstable();
    page_numbers.dedup();
    if page_numbers.iter().any(|&page| {
        page.checked_mul(PAGE_SIZE as u64)
            .map_or(true, |offset| offset >= file_len)
    }) {
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
    Ok(PageReadRequest {
        buffered_file: Arc::clone(buffered_file),
        file_len,
        page_numbers,
    })
}

pub struct PendingPreparedPageBatch {
    page_numbers: Vec<Vec<u64>>,
    responses: Vec<RuntimeResponse>,
}

impl PendingPreparedPageBatch {
    pub fn finish(mut self) -> io::Result<Vec<PreparedPages>> {
        let responses = self.responses;
        let mut out: Vec<Option<PreparedPages>> =
            (0..self.page_numbers.len()).map(|_| None).collect();
        for (idx, pages) in self.page_numbers.iter().enumerate() {
            if pages.is_empty() {
                out[idx] = Some(PreparedPages::empty());
            }
        }
        for response in responses {
            let batch = response.recv()?.map_err(other_error)?;
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
                let page_numbers = self.page_numbers.get_mut(request_idx).ok_or_else(|| {
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
}

pub fn submit_prepared_page_batch(
    requests: Vec<PageReadRequest>,
) -> io::Result<PendingPreparedPageBatch> {
    let page_numbers: Vec<Vec<u64>> = requests
        .iter()
        .map(|request| request.page_numbers.clone())
        .collect();

    let parts = requests
        .into_iter()
        .enumerate()
        .filter(|(_, request)| !request.page_numbers.is_empty())
        .map(|(request_idx, request)| {
            let file = request.buffered_file;
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
        responses,
    })
}

pub fn submit_prepared_pages(
    buffered_file: &Arc<File>,
    file_len: u64,
    backend: EdgeIoBackend,
    page_numbers: &[u64],
) -> io::Result<PendingPreparedPages> {
    backend.ensure_supported()?;
    let mut unique = page_numbers.to_vec();
    unique.sort_unstable();
    unique.dedup();
    if unique.is_empty() {
        return Ok(PendingPreparedPages::Ready(PreparedPages::empty()));
    }
    if unique.iter().any(|&p| {
        p.checked_mul(PAGE_SIZE as u64)
            .map_or(true, |offset| offset >= file_len)
    }) {
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
            let result = page_runtime().submit(Arc::clone(buffered_file), file_len, &unique)?;
            Ok(PendingPreparedPages::Runtime {
                page_numbers: unique,
                result,
            })
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

#[cfg(target_os = "linux")]
fn configured_coalesce_pages() -> usize {
    *process_state().coalesce_pages.get_or_init(|| {
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

pub struct RuntimeResponse {
    pid: u32,
    receiver: ManuallyDrop<Receiver<RuntimeResult>>,
}

impl RuntimeResponse {
    fn new(receiver: Receiver<RuntimeResult>) -> Self {
        Self {
            pid: std::process::id(),
            receiver: ManuallyDrop::new(receiver),
        }
    }

    fn recv(&self) -> io::Result<RuntimeResult> {
        check_process(self.pid)?;
        self.receiver
            .recv()
            .map_err(|_| other_error("edge-I/O owner dropped response"))
    }
}

impl Drop for RuntimeResponse {
    fn drop(&mut self) {
        if self.pid == std::process::id() {
            // A receiver inherited at fork can contain a channel lock held by
            // a vanished thread. Only its creating process may run Drop.
            unsafe { ManuallyDrop::drop(&mut self.receiver) };
        }
    }
}

fn check_process(pid: u32) -> io::Result<()> {
    if pid == std::process::id() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "prepared edge I/O belongs to another process; submit a new read after fork",
        ))
    }
}

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

    fn submit(&self, file: Arc<File>, file_len: u64, pages: &[u64]) -> io::Result<RuntimeResponse> {
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
        Ok(RuntimeResponse::new(result))
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
            responses.push(RuntimeResponse::new(result));
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

#[cfg(target_os = "linux")]
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
        if OWNER_RETIRED.with(Cell::get) {
            let _ = first.response.send(Err(
                "edge-I/O owner retired after incomplete read; restart the process".into(),
            ));
            continue;
        }
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
        let mut files = Vec::new();
        let mut part_lengths = Vec::with_capacity(jobs.len());
        for job in &jobs {
            let mut lengths = Vec::with_capacity(job.parts.len());
            for part in &job.parts {
                lengths.push((part.request_idx, part.pages.len()));
                files.push(Arc::clone(&part.file));
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
        match read_specs_local_with_buffers(&specs, qd, buffers, files) {
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

#[cfg(not(target_os = "linux"))]
fn page_owner_loop(
    receiver: Receiver<ReadJob>,
    _qd: usize,
    _batch_page_target: usize,
    _batch_micros: u64,
) {
    for job in receiver {
        let _ = job.response.send(Err(unsupported_backend().to_string()));
    }
}

fn page_runtime() -> &'static PageRuntime {
    process_state().runtime.get_or_init(PageRuntime::new)
}

struct ProcessState {
    pid: u32,
    runtime: OnceLock<PageRuntime>,
    #[cfg(target_os = "linux")]
    coalesce_pages: OnceLock<usize>,
}

fn process_state() -> &'static ProcessState {
    static CURRENT: AtomicPtr<ProcessState> = AtomicPtr::new(std::ptr::null_mut());
    let pid = std::process::id();
    loop {
        let current = CURRENT.load(Ordering::Acquire);
        // Published nodes live until process exit. Reading their immutable PID
        // never enters an inherited OnceLock or channel after fork.
        if !current.is_null() && unsafe { (*current).pid } == pid {
            return unsafe { &*current };
        }
        let fresh = Box::into_raw(Box::new(ProcessState {
            pid,
            runtime: OnceLock::new(),
            #[cfg(target_os = "linux")]
            coalesce_pages: OnceLock::new(),
        }));
        if CURRENT
            .compare_exchange(current, fresh, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // Do not destroy inherited state: its locks may have been held by
            // other threads at fork.
            return unsafe { &*fresh };
        }
        // This allocation was never published and contains only empty cells.
        unsafe { drop(Box::from_raw(fresh)) };
    }
}

#[cfg(all(target_os = "linux", test))]
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
    read_specs_local(&specs, qd, vec![file])
}

#[cfg(any(target_os = "linux", test))]
struct PageReadSpec {
    // The owning ReadJob (or the test helper's Arc<File>) remains live until
    // every CQE has arrived, so page-sized specs need only carry the stable fd.
    fd: std::os::raw::c_int,
    file_len: u64,
    page_number: u64,
}

#[cfg(all(target_os = "linux", test))]
fn read_specs_local(
    specs: &[PageReadSpec],
    qd: usize,
    files: Vec<Arc<File>>,
) -> io::Result<Vec<PageData>> {
    let buffers = specs.iter().map(|_| PageData::zeroed()).collect();
    read_specs_local_with_buffers(specs, qd, buffers, files)
}

#[cfg(target_os = "linux")]
thread_local! {
    static RING: RefCell<Option<(u32, IoUring)>> = const { RefCell::new(None) };
    static OWNER_RETIRED: Cell<bool> = const { Cell::new(false) };
    #[cfg(test)]
    static FAIL_AFTER_SUBMIT: Cell<bool> = const { Cell::new(false) };
    #[cfg(test)]
    static PANIC_AFTER_SUBMIT: Cell<bool> = const { Cell::new(false) };
    #[cfg(test)]
    static SUBMITTED_RING_FD: Cell<i32> = const { Cell::new(-1) };
}

/// Retain every kernel-visible allocation on an undrainable submission error.
/// Closing a ring cancels requests but does not prove their buffers are idle.
#[cfg(target_os = "linux")]
struct ActiveRead {
    ring: Option<IoUring>,
    buffers: Vec<PageData>,
    iovecs: Vec<libc::iovec>,
    files: Vec<Arc<File>>,
    pending: usize,
}

#[cfg(target_os = "linux")]
impl ActiveRead {
    fn new(ring: IoUring, buffers: Vec<PageData>, files: Vec<Arc<File>>) -> Self {
        Self {
            ring: Some(ring),
            buffers,
            iovecs: Vec::new(),
            files,
            pending: 0,
        }
    }

    fn wait(&mut self) -> io::Result<()> {
        #[cfg(test)]
        if FAIL_AFTER_SUBMIT.with(|fail| fail.replace(false)) {
            SUBMITTED_RING_FD.with(|fd| fd.set(self.ring.as_ref().unwrap().as_raw_fd()));
            let submitted = self.ring.as_mut().unwrap().submit()?;
            assert!(
                submitted > 0,
                "fault injection must follow actual submission"
            );
            assert!(
                !PANIC_AFTER_SUBMIT.with(|panic| panic.replace(false)),
                "injected panic after submission"
            );
            return Err(other_error("injected terminal failure after submission"));
        }
        loop {
            match self.ring.as_mut().unwrap().submit_and_wait(1) {
                Ok(_) => return Ok(()),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
    }

    fn finish(mut self) -> (IoUring, Vec<PageData>) {
        debug_assert_eq!(self.pending, 0);
        (self.ring.take().unwrap(), std::mem::take(&mut self.buffers))
    }
}

#[cfg(target_os = "linux")]
impl Drop for ActiveRead {
    fn drop(&mut self) {
        if self.pending == 0 {
            return;
        }
        // This owner rejects future work, bounding retention to one failed
        // batch. Keep the ring, frames, iovecs and file owners until process
        // exit rather than recycle memory that the kernel may still access.
        OWNER_RETIRED.with(|retired| retired.set(true));
        std::mem::forget(self.ring.take());
        std::mem::forget(std::mem::take(&mut self.buffers));
        std::mem::forget(std::mem::take(&mut self.iovecs));
        std::mem::forget(std::mem::take(&mut self.files));
    }
}

#[cfg(target_os = "linux")]
fn read_specs_local_with_buffers(
    specs: &[PageReadSpec],
    qd: usize,
    buffers: Vec<PageData>,
    files: Vec<Arc<File>>,
) -> io::Result<Vec<PageData>> {
    if OWNER_RETIRED.with(Cell::get) {
        return Err(other_error(
            "edge-I/O owner retired after incomplete read; restart the process",
        ));
    }
    if specs.is_empty() {
        return Ok(Vec::new());
    }
    let entries = qd.min(specs.len()).max(1) as u32;
    RING.with(|slot| {
        // A checked-out ring cannot be reused after an early return or panic.
        let (capacity, ring) = match slot.borrow_mut().take() {
            Some((capacity, ring)) if capacity >= entries => (capacity, ring),
            previous => {
                drop(previous);
                let ring = IoUring::new(entries).map_err(|error| {
                    io::Error::new(
                        error.kind(),
                        format!("io_uring setup with {entries} entries failed: {error}"),
                    )
                })?;
                (entries, ring)
            }
        };
        let owned = ActiveRead::new(ring, buffers, files);
        let result = if configured_coalesce_pages() >= 2 && has_consecutive_pages(specs) {
            read_pages_on_ring_coalesced(owned, specs, entries, configured_coalesce_pages())
        } else {
            read_pages_on_ring(owned, specs, entries)
        };
        match result {
            Ok((ring, pages)) => {
                *slot.borrow_mut() = Some((capacity, ring));
                Ok(pages)
            }
            Err(error) if OWNER_RETIRED.with(Cell::get) => Err(io::Error::new(
                error.kind(),
                format!("{error}; edge-I/O owner retired; restart the process"),
            )),
            Err(error) => Err(error),
        }
    })
}

#[cfg(any(target_os = "linux", test))]
fn has_consecutive_pages(specs: &[PageReadSpec]) -> bool {
    specs.windows(2).any(|pair| {
        pair[0].fd == pair[1].fd
            && pair[0].file_len == pair[1].file_len
            && pair[0].page_number.checked_add(1) == Some(pair[1].page_number)
    })
}

#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CoalescedRead {
    first: usize,
    pages: usize,
    iovec_start: usize,
}

#[cfg(any(target_os = "linux", test))]
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

#[cfg(target_os = "linux")]
fn read_pages_on_ring_coalesced(
    mut owned: ActiveRead,
    specs: &[PageReadSpec],
    entries: u32,
    max_pages: usize,
) -> io::Result<(IoUring, Vec<PageData>)> {
    debug_assert_eq!(owned.buffers.len(), specs.len());
    let mut reads = plan_coalesced_reads(specs, max_pages);
    owned.iovecs = Vec::with_capacity(specs.len());
    for read in &mut reads {
        if read.pages == 1 {
            continue;
        }
        read.iovec_start = owned.iovecs.len();
        for buffer in &mut owned.buffers[read.first..read.first + read.pages] {
            owned.iovecs.push(libc::iovec {
                iov_base: buffer.bytes_mut().as_mut_ptr().cast(),
                iov_len: PAGE_SIZE,
            });
        }
    }

    let mut next = 0usize;
    let mut completed = 0usize;
    let mut first_error: Option<io::Error> = None;
    let mut completed_bytes = 0u64;

    while completed < reads.len() {
        while next < reads.len() && owned.pending < entries as usize {
            let read = reads[next];
            let spec = &specs[read.first];
            let offset = spec
                .page_number
                .checked_mul(PAGE_SIZE as u64)
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
            let entry = if read.pages == 1 {
                opcode::Read::new(
                    types::Fd(spec.fd),
                    owned.buffers[read.first].bytes_mut().as_mut_ptr(),
                    PAGE_SIZE as u32,
                )
                .offset(offset)
                .build()
                .user_data(next as u64)
            } else {
                opcode::Readv::new(
                    types::Fd(spec.fd),
                    owned.iovecs[read.iovec_start..].as_ptr(),
                    read.pages as u32,
                )
                .offset(offset)
                .build()
                .user_data(next as u64)
            };
            owned.pending += 1;
            unsafe {
                owned
                    .ring
                    .as_mut()
                    .unwrap()
                    .submission()
                    .push(&entry)
                    .map_err(|_| {
                        io::Error::new(io::ErrorKind::WouldBlock, "io_uring SQ unexpectedly full")
                    })?;
            }
            next += 1;
        }

        owned.wait()?;
        let mut completion = owned.ring.as_mut().unwrap().completion();
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
                for buffer in &mut owned.buffers[read.first..read.first + read.pages] {
                    let valid = remaining.min(PAGE_SIZE);
                    buffer.valid_len = valid;
                    remaining -= valid;
                }
                completed_bytes += expected as u64;
            }
            owned.pending -= 1;
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
    Ok(owned.finish())
}

#[cfg(target_os = "linux")]
fn read_pages_on_ring(
    mut owned: ActiveRead,
    specs: &[PageReadSpec],
    entries: u32,
) -> io::Result<(IoUring, Vec<PageData>)> {
    debug_assert_eq!(owned.buffers.len(), specs.len());
    let mut next = 0usize;
    let mut completed = 0usize;
    let mut first_error: Option<io::Error> = None;
    let mut completed_bytes = 0u64;

    while completed < specs.len() {
        while next < specs.len() && owned.pending < entries as usize {
            let offset = specs[next]
                .page_number
                .checked_mul(PAGE_SIZE as u64)
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
            let entry = opcode::Read::new(
                types::Fd(specs[next].fd),
                owned.buffers[next].bytes_mut().as_mut_ptr(),
                PAGE_SIZE as u32,
            )
            .offset(offset)
            .build()
            .user_data(next as u64);
            owned.pending += 1;
            unsafe {
                owned
                    .ring
                    .as_mut()
                    .unwrap()
                    .submission()
                    .push(&entry)
                    .map_err(|_| {
                        io::Error::new(io::ErrorKind::WouldBlock, "io_uring SQ unexpectedly full")
                    })?;
            }
            next += 1;
        }

        // SIGPROF and other process signals may interrupt io_uring_enter after
        // the kernel has accepted submissions. Returning here would drop
        // buffers that are still in flight and leave their CQEs for the next
        // owner batch. Retry EINTR until those owned buffers are completed.
        owned.wait()?;
        let mut completion = owned.ring.as_mut().unwrap().completion();
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
                owned.buffers[idx].valid_len = expected;
                completed_bytes += expected as u64;
            }
            owned.pending -= 1;
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
    Ok(owned.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "linux")]
    use std::io::Write;

    #[test]
    fn range_page_planning_deduplicates_and_handles_boundaries() {
        let pages = pages_for_ranges(&[(4093, 4100), (4096, 8192), (0, 0)], 8192).unwrap();
        assert_eq!(pages, vec![0, 1]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn request_bounds_allow_large_pages() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let file = Arc::new(tmp.reopen().unwrap());
        let page = 1u64 << 32;
        let file_len = (page + 1) * PAGE_SIZE as u64;
        let request = page_read_request(&file, file_len, EdgeIoBackend::Buffered, &[page]).unwrap();
        assert_eq!(request.page_numbers, [page]);
        for invalid in [page + 1, u64::MAX / PAGE_SIZE as u64 + 1] {
            assert!(
                page_read_request(&file, file_len, EdgeIoBackend::Buffered, &[invalid]).is_err()
            );
            assert!(
                submit_prepared_pages(&file, file_len, EdgeIoBackend::Buffered, &[invalid])
                    .is_err()
            );
        }
        assert!(page_read_request(&file, file_len, EdgeIoBackend::Mmap, &[0]).is_err());
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

    #[cfg(target_os = "linux")]
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
    }

    #[cfg(target_os = "linux")]
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
                first_bytes.len() as u64,
                EdgeIoBackend::Buffered,
                &[1, 0, 1],
            )
            .unwrap(),
            page_read_request(
                &second_file,
                second_bytes.len() as u64,
                EdgeIoBackend::Buffered,
                &[0],
            )
            .unwrap(),
            page_read_request(
                &second_file,
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
}

#[cfg(test)]
#[path = "page_io_readiness_tests.rs"]
mod readiness_tests;
