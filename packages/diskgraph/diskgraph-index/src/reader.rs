// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0


use crate::layout::{
    EdgeListView, OffsetFileHeader, RelationLayout, DELTA_CHUNK, EDGES_FILENAME, OFFSETS_FILENAME,
};
use crate::page_io::{
    page_read_request, page_read_request_owned, pages_for_ranges, read_prepared_pages,
    submit_prepared_pages, EdgeIoBackend, PageReadRequest, PendingPreparedPages, PreparedPages,
    PAGE_SIZE,
};
use crate::prefetch;
use anyhow::Result;
use diskgraph_core::errors::DiskgraphError;
use libc::madvise;
use memmap2::Mmap;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

static NEXT_EDGE_FILE_ID: AtomicU64 = AtomicU64::new(1);

fn configured_eager_node_pages() -> usize {
    static EAGER_NODE_PAGES: OnceLock<usize> = OnceLock::new();
    *EAGER_NODE_PAGES.get_or_init(|| {
        std::env::var("DG_IO_EAGER_NODE_PAGES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
            .clamp(0, 256)
    })
}

#[inline]
fn planned_header_end(start: u64, end: u64, max_prefix: u64, eager_node_pages: usize) -> u64 {
    let prefix_end = end.min(start.saturating_add(max_prefix));
    if start == end || eager_node_pages == 0 {
        return prefix_end;
    }
    let first_page = start / PAGE_SIZE as u64;
    let last_page = (end - 1) / PAGE_SIZE as u64;
    let record_pages = (last_page - first_page + 1) as usize;
    if record_pages <= eager_node_pages {
        end
    } else {
        prefix_end
    }
}

#[derive(Debug, Clone)]
pub struct RelationByteRange {
    pub layout_index: usize,
    pub count: u32,
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone)]
pub struct ParsedNodeHeader {
    pub local_id: u64,
    pub record_start: u64,
    pub record_end: u64,
    pub relations: SmallVec<[RelationByteRange; 4]>,
    pub topology_end: u64,
    pub node_data: Option<std::ops::Range<u64>>,
}

pub struct PreparedHeaders {
    pub headers: Vec<ParsedNodeHeader>,
    pub pages: PreparedPages,
}

pub struct PendingHeaders {
    plan: HeaderReadPlan,
    pages: PendingPreparedPages,
}

#[derive(Default)]
pub struct HeaderReadPlan {
    records: Vec<HeaderRecordRange>,
    pub page_numbers: Vec<u64>,
}

struct HeaderRecordRange {
    local_id: u64,
    start: u64,
    end: u64,
}

pub struct PendingPages {
    existing: PreparedPages,
    missing: PendingPreparedPages,
}

pub struct RangeReadPlan {
    existing: PreparedPages,
    pub page_numbers: Vec<u64>,
}

/// Memory-mapped shard reader. Provides zero-copy edge list access by decoding
/// offsets and slicing into the edge data file.
pub struct ShardReader {
    offsets_mmap: Mmap,
    edges: Mmap,
    edges_file: Arc<File>,
    direct_edges_file: Option<Arc<File>>,
    edge_file_id: u64,
    edge_backend: EdgeIoBackend,
    header: OffsetFileHeader,
    offsets_header_len: usize,
    layouts: Arc<Vec<RelationLayout>>,
    edge_count: u64,
    has_node_data: bool,
}

impl ShardReader {
    pub fn open(
        dir: &Path,
        layouts: Arc<Vec<RelationLayout>>,
        has_node_data: bool,
    ) -> Result<Self> {
        let offsets_path = dir.join(OFFSETS_FILENAME);
        let edges_path = dir.join(EDGES_FILENAME);
        let offsets_file = File::open(&offsets_path)?;
        let edges_file = File::open(&edges_path)?;
        let offsets = unsafe { Mmap::map(&offsets_file)? };
        let edges = unsafe { Mmap::map(&edges_file)? };
        advise_random(&edges);
        let edge_backend = EdgeIoBackend::from_env()?;
        if edge_backend == EdgeIoBackend::Buffered {
            advise_file_random(&edges_file)?;
        }
        let direct_edges_file =
            if edge_backend == EdgeIoBackend::Direct && edges_file.metadata()?.len() > 0 {
                Some(Arc::new(
                    OpenOptions::new()
                        .read(true)
                        .custom_flags(libc::O_DIRECT | libc::O_CLOEXEC)
                        .open(&edges_path)?,
                ))
            } else {
                None
            };
        let header_len = bincode::serialized_size(&OffsetFileHeader::new(0, 0))
            .map_err(|_| DiskgraphError::InvalidMetadata("invalid offsets header".into()))?
            as usize;
        if offsets.len() < header_len {
            return Err(DiskgraphError::InvalidMetadata("invalid offsets header".into()).into());
        }
        let header: OffsetFileHeader = bincode::deserialize(&offsets[..header_len])
            .map_err(|_| DiskgraphError::InvalidMetadata("invalid offsets header".into()))?;
        if !matches!(header.delta_bytes, 2..=4) {
            return Err(
                DiskgraphError::InvalidMetadata("offset delta width invalid".into()).into(),
            );
        }
        if !matches!(header.node_data_len_bytes, 2..=4) {
            return Err(
                DiskgraphError::InvalidMetadata("node data len width invalid".into()).into(),
            );
        }
        if header.delta_chunk as usize != crate::layout::DELTA_CHUNK {
            return Err(
                DiskgraphError::InvalidMetadata("offset delta chunk invalid".into()).into(),
            );
        }
        validate_offset_storage_len(
            offsets.len(),
            header.total_offsets,
            header.delta_bytes,
            header_len,
        )?;
        let edge_count = header.edge_count;
        tracing::info!(
            dir = %dir.display(),
            edge_count,
            total_offsets = header.total_offsets,
            delta_bytes = header.delta_bytes,
            node_data_len_bytes = header.node_data_len_bytes,
            "Shard reader opened"
        );
        Ok(Self {
            offsets_mmap: offsets,
            edges,
            edges_file: Arc::new(edges_file),
            direct_edges_file,
            edge_file_id: NEXT_EDGE_FILE_ID.fetch_add(1, Ordering::Relaxed),
            edge_backend,
            header,
            offsets_header_len: header_len,
            layouts,
            edge_count,
            has_node_data,
        })
    }

    pub fn header(&self) -> &OffsetFileHeader {
        &self.header
    }

    pub fn edge_count(&self) -> u64 {
        self.edge_count
    }

    pub fn offsets_len(&self) -> u64 {
        self.offsets_mmap.len() as u64
    }

    /// Fault every offsets page before a measured explicit-edge-I/O phase.
    /// The returned checksum keeps the reads observable to the optimizer.
    pub fn prefault_offsets(&self) -> u64 {
        let mut checksum = 0u64;
        for offset in (0..self.offsets_mmap.len()).step_by(4096) {
            checksum = checksum.wrapping_add(self.offsets_mmap[offset] as u64);
        }
        if let Some(last) = self.offsets_mmap.last() {
            checksum = checksum.wrapping_add(*last as u64);
        }
        checksum
    }

    pub fn edges_len(&self) -> u64 {
        self.edges.len() as u64
    }

    pub fn edge_backend(&self) -> EdgeIoBackend {
        self.edge_backend
    }

    pub fn layout(&self, index: usize) -> Option<&RelationLayout> {
        self.layouts.get(index)
    }

    pub fn record_range(&self, node_idx: u64) -> Result<(u64, u64)> {
        if node_idx + 1 >= self.header.total_offsets {
            return Err(DiskgraphError::InvalidMetadata("node index out of range".into()).into());
        }
        let (start, end) = decode_offset_pair(
            &self.offsets_mmap,
            node_idx,
            self.header.total_offsets,
            self.header.delta_bytes,
            self.offsets_header_len,
        )?;
        if start > end || end > self.edges.len() as u64 {
            return Err(
                DiskgraphError::InvalidMetadata("offset points outside edge blob".into()).into(),
            );
        }
        Ok((start, end))
    }

    /// Read and parse only the bounded record headers needed to discover
    /// relation byte ranges. The returned headers own no page leases.
    pub fn prepare_headers(&self, local_ids: &[u64]) -> Result<PreparedHeaders> {
        let ranges: Vec<(u64, u64)> = local_ids
            .iter()
            .map(|&id| self.record_range(id))
            .collect::<Result<_>>()?;
        if self.edge_backend == EdgeIoBackend::Mmap {
            let headers = local_ids
                .iter()
                .copied()
                .zip(ranges)
                .map(|(id, range)| self.parse_header(id, range, None))
                .collect::<Result<_>>()?;
            return Ok(PreparedHeaders {
                headers,
                pages: PreparedPages::empty(),
            });
        }

        self.finish_prepare_headers(self.begin_prepare_headers_from_ranges(local_ids, ranges)?)
    }

    pub fn begin_prepare_headers(&self, local_ids: &[u64]) -> Result<PendingHeaders> {
        if self.edge_backend == EdgeIoBackend::Mmap {
            return Err(DiskgraphError::InvalidMetadata(
                "async header preparation is not used for mmap".into(),
            )
            .into());
        }
        let plan = self.plan_headers(local_ids)?;
        let pages = self.submit_page_numbers(&plan.page_numbers)?;
        Ok(PendingHeaders { plan, pages })
    }

    fn begin_prepare_headers_from_ranges(
        &self,
        local_ids: &[u64],
        ranges: Vec<(u64, u64)>,
    ) -> Result<PendingHeaders> {
        let plan = self.plan_headers_from_ranges(local_ids, ranges)?;
        let pages = self.submit_page_numbers(&plan.page_numbers)?;
        Ok(PendingHeaders { plan, pages })
    }

    pub fn plan_headers(&self, local_ids: &[u64]) -> Result<HeaderReadPlan> {
        self.plan_headers_owned(local_ids.to_vec())
    }

    pub fn plan_headers_owned(&self, local_ids: Vec<u64>) -> Result<HeaderReadPlan> {
        if self.edge_backend == EdgeIoBackend::Mmap {
            return Err(DiskgraphError::InvalidMetadata(
                "explicit header planning is not used for mmap".into(),
            )
            .into());
        }
        let mut records = Vec::with_capacity(local_ids.len());
        for local_id in local_ids {
            let (start, end) = self.record_range(local_id)?;
            records.push(HeaderRecordRange {
                local_id,
                start,
                end,
            });
        }
        self.plan_header_records(records)
    }

    /// Plan headers directly from sorted local IDs, removing the caller's
    /// temporary ID vector. Duplicate IDs must be adjacent and are folded into
    /// one record read while preserving the existing hub semantics downstream.
    pub fn plan_headers_sorted<I>(&self, local_ids: I) -> Result<HeaderReadPlan>
    where
        I: ExactSizeIterator<Item = u64>,
    {
        if self.edge_backend == EdgeIoBackend::Mmap {
            return Err(DiskgraphError::InvalidMetadata(
                "explicit header planning is not used for mmap".into(),
            )
            .into());
        }
        let mut plan = HeaderReadPlan::default();
        self.plan_headers_sorted_reusing(local_ids, &mut plan)?;
        Ok(plan)
    }

    /// Fill a retained header plan from sorted local IDs. The record array is
    /// request scratch and can be reused after header parsing; page numbers
    /// are moved onward into `PreparedPages` because their lifetime spans the
    /// topology read.
    pub fn plan_headers_sorted_reusing<I>(
        &self,
        local_ids: I,
        plan: &mut HeaderReadPlan,
    ) -> Result<()>
    where
        I: ExactSizeIterator<Item = u64>,
    {
        if self.edge_backend == EdgeIoBackend::Mmap {
            return Err(DiskgraphError::InvalidMetadata(
                "explicit header planning is not used for mmap".into(),
            )
            .into());
        }
        plan.records.clear();
        plan.records.reserve(local_ids.len());
        let mut previous = None;
        for local_id in local_ids {
            if previous == Some(local_id) {
                continue;
            }
            let (start, end) = self.record_range(local_id)?;
            plan.records.push(HeaderRecordRange {
                local_id,
                start,
                end,
            });
            previous = Some(local_id);
        }
        self.plan_header_records_reusing(plan)
    }

    fn plan_headers_from_ranges(
        &self,
        local_ids: &[u64],
        ranges: Vec<(u64, u64)>,
    ) -> Result<HeaderReadPlan> {
        let mut records = Vec::with_capacity(local_ids.len());
        for (&local_id, (start, end)) in local_ids.iter().zip(ranges) {
            records.push(HeaderRecordRange {
                local_id,
                start,
                end,
            });
        }
        self.plan_header_records(records)
    }

    fn plan_header_records(&self, records: Vec<HeaderRecordRange>) -> Result<HeaderReadPlan> {
        let mut plan = HeaderReadPlan {
            records,
            page_numbers: Vec::new(),
        };
        self.plan_header_records_reusing(&mut plan)?;
        Ok(plan)
    }

    fn plan_header_records_reusing(&self, plan: &mut HeaderReadPlan) -> Result<()> {
        let max_prefix = self
            .layouts
            .len()
            .checked_mul(4)
            .and_then(|v| {
                v.checked_add(if self.has_node_data {
                    self.header.node_data_len_bytes as usize
                } else {
                    0
                })
            })
            .ok_or_else(|| DiskgraphError::InvalidMetadata("record prefix overflow".into()))?
            as u64;
        let file_len = self.edges.len() as u64;
        let eager_node_pages = configured_eager_node_pages();
        let mut page_capacity = 0usize;
        for record in &plan.records {
            let (start, end) = (record.start, record.end);
            let planned_end = planned_header_end(start, end, max_prefix, eager_node_pages);
            if start > planned_end || planned_end > file_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "invalid explicit header range",
                )
                .into());
            }
            if start != planned_end {
                let first = start / PAGE_SIZE as u64;
                let last = (planned_end - 1) / PAGE_SIZE as u64;
                let span: usize = (last - first + 1).try_into().map_err(|_| {
                    DiskgraphError::InvalidMetadata("header page span overflow".into())
                })?;
                page_capacity = page_capacity.checked_add(span).ok_or_else(|| {
                    DiskgraphError::InvalidMetadata("header page count overflow".into())
                })?;
            }
        }
        plan.page_numbers.clear();
        plan.page_numbers.reserve(page_capacity);
        for record in &plan.records {
            let (start, end) = (record.start, record.end);
            let planned_end = planned_header_end(start, end, max_prefix, eager_node_pages);
            if start != planned_end {
                plan.page_numbers
                    .extend(start / PAGE_SIZE as u64..=(planned_end - 1) / PAGE_SIZE as u64);
            }
        }
        plan.page_numbers.sort_unstable();
        plan.page_numbers.dedup();
        Ok(())
    }

    pub fn finish_prepare_headers(&self, pending: PendingHeaders) -> Result<PreparedHeaders> {
        let prepared = pending.pages.finish()?;
        self.finish_header_plan(pending.plan, prepared)
    }

    pub fn finish_header_plan(
        &self,
        mut plan: HeaderReadPlan,
        prepared: PreparedPages,
    ) -> Result<PreparedHeaders> {
        let mut headers = Vec::new();
        let pages = self.finish_header_plan_reusing(&mut plan, prepared, &mut headers)?;
        Ok(PreparedHeaders { headers, pages })
    }

    /// Parse a completed header read into caller-owned scratch. Explicit graph
    /// sampling invokes this on every hop; retaining the largest header vector
    /// in `SamplingState` avoids allocating and faulting a large array of
    /// inline relation descriptors for every request.
    pub fn finish_header_plan_reusing(
        &self,
        plan: &mut HeaderReadPlan,
        prepared: PreparedPages,
        headers: &mut Vec<ParsedNodeHeader>,
    ) -> Result<PreparedPages> {
        headers.clear();
        headers.reserve(plan.records.len());
        for record in plan.records.drain(..) {
            headers.push(self.parse_header(
                record.local_id,
                (record.start, record.end),
                Some(&prepared),
            )?);
        }
        Ok(prepared)
    }

    pub fn prepare_ranges(&self, ranges: &[(u64, u64)]) -> Result<PreparedPages> {
        let pages = pages_for_ranges(ranges, self.edges.len() as u64)?;
        self.prepare_page_numbers(&pages)
    }

    pub fn prepare_ranges_reusing(
        &self,
        ranges: &[(u64, u64)],
        existing: PreparedPages,
    ) -> Result<PreparedPages> {
        let mut missing = pages_for_ranges(ranges, self.edges.len() as u64)?;
        missing.retain(|&page| !existing.contains_page(page));
        Ok(existing.merge(self.prepare_page_numbers(&missing)?))
    }

    pub fn begin_prepare_ranges_reusing(
        &self,
        ranges: &[(u64, u64)],
        existing: PreparedPages,
    ) -> Result<PendingPages> {
        let mut missing = pages_for_ranges(ranges, self.edges.len() as u64)?;
        missing.retain(|&page| !existing.contains_page(page));
        Ok(PendingPages {
            existing,
            missing: self.submit_page_numbers(&missing)?,
        })
    }

    pub fn plan_ranges_reusing(
        &self,
        ranges: &[(u64, u64)],
        existing: PreparedPages,
    ) -> Result<RangeReadPlan> {
        let mut missing = pages_for_ranges(ranges, self.edges.len() as u64)?;
        missing.retain(|&page| !existing.contains_page(page));
        Ok(RangeReadPlan {
            existing,
            page_numbers: missing,
        })
    }

    pub fn finish_range_plan(&self, plan: RangeReadPlan, pages: PreparedPages) -> PreparedPages {
        plan.existing.merge(pages)
    }

    pub fn finish_prepare_ranges(&self, pending: PendingPages) -> Result<PreparedPages> {
        Ok(pending.existing.merge(pending.missing.finish()?))
    }

    pub fn prepare_page_numbers(&self, pages: &[u64]) -> Result<PreparedPages> {
        Ok(read_prepared_pages(
            &self.edges_file,
            self.direct_edges_file.as_ref(),
            self.edge_file_id,
            self.edges.len() as u64,
            self.edge_backend,
            pages,
        )?)
    }

    pub fn submit_page_numbers(&self, pages: &[u64]) -> Result<PendingPreparedPages> {
        Ok(submit_prepared_pages(
            &self.edges_file,
            self.direct_edges_file.as_ref(),
            self.edge_file_id,
            self.edges.len() as u64,
            self.edge_backend,
            pages,
        )?)
    }

    pub fn page_read_request(&self, pages: &[u64]) -> Result<PageReadRequest> {
        Ok(page_read_request(
            &self.edges_file,
            self.direct_edges_file.as_ref(),
            self.edge_file_id,
            self.edges.len() as u64,
            self.edge_backend,
            pages,
        )?)
    }

    pub fn page_read_request_owned(&self, pages: Vec<u64>) -> Result<PageReadRequest> {
        Ok(page_read_request_owned(
            &self.edges_file,
            self.direct_edges_file.as_ref(),
            self.edge_file_id,
            self.edges.len() as u64,
            self.edge_backend,
            pages,
        )?)
    }

    fn parse_header(
        &self,
        local_id: u64,
        (record_start, record_end): (u64, u64),
        prepared: Option<&PreparedPages>,
    ) -> Result<ParsedNodeHeader> {
        if record_start == record_end {
            if self.has_node_data {
                return Err(DiskgraphError::InvalidMetadata(
                    "empty record cannot contain declared node data".into(),
                )
                .into());
            }
            return Ok(ParsedNodeHeader {
                local_id,
                record_start,
                record_end,
                relations: self
                    .layouts
                    .iter()
                    .enumerate()
                    .map(|(layout_index, _)| RelationByteRange {
                        layout_index,
                        count: 0,
                        start: record_start,
                        end: record_start,
                    })
                    .collect(),
                topology_end: record_start,
                node_data: None,
            });
        }

        let max_prefix = self.layouts.len() * 4
            + if self.has_node_data {
                self.header.node_data_len_bytes as usize
            } else {
                0
            };
        let prefix_end = record_end.min(record_start.saturating_add(max_prefix as u64));
        let raw: Cow<'_, [u8]> = if let Some(prepared) = prepared {
            match prepared.contiguous_range(record_start, prefix_end)? {
                Some(bytes) => Cow::Borrowed(bytes),
                None => Cow::Owned(prepared.copy_range(record_start, prefix_end)?),
            }
        } else {
            Cow::Borrowed(&self.edges[record_start as usize..prefix_end as usize])
        };
        let raw = raw.as_ref();
        let mut cursor = 0usize;
        let mut counts = SmallVec::<[u32; 4]>::with_capacity(self.layouts.len());
        for _ in self.layouts.iter() {
            let (count, consumed) = decode_count(&raw[cursor..])?;
            cursor = cursor
                .checked_add(consumed)
                .ok_or_else(|| DiskgraphError::InvalidMetadata("header overflow".into()))?;
            counts.push(count);
        }
        let node_data_len = if self.has_node_data {
            let width = self.header.node_data_len_bytes as usize;
            let end = cursor
                .checked_add(width)
                .ok_or_else(|| DiskgraphError::InvalidMetadata("header overflow".into()))?;
            if end > raw.len() {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
            }
            let len = read_len_le(&raw[cursor..end]);
            cursor = end;
            Some(len as u64)
        } else {
            None
        };
        let mut absolute = record_start
            .checked_add(cursor as u64)
            .ok_or_else(|| DiskgraphError::InvalidMetadata("header overflow".into()))?;
        let mut relations = SmallVec::<[RelationByteRange; 4]>::with_capacity(self.layouts.len());
        for (layout_index, (layout, count)) in self.layouts.iter().zip(counts).enumerate() {
            let bytes = (count as u64)
                .checked_mul(layout.entry_bytes as u64)
                .ok_or_else(|| {
                    DiskgraphError::InvalidMetadata("relation length overflow".into())
                })?;
            let end = absolute.checked_add(bytes).ok_or_else(|| {
                DiskgraphError::InvalidMetadata("relation length overflow".into())
            })?;
            if end > record_end {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
            }
            relations.push(RelationByteRange {
                layout_index,
                count,
                start: absolute,
                end,
            });
            absolute = end;
        }
        let topology_end = absolute;
        let node_data = if let Some(len) = node_data_len {
            let end = topology_end
                .checked_add(len)
                .ok_or_else(|| DiskgraphError::InvalidMetadata("node data overflow".into()))?;
            if end != record_end {
                return Err(
                    DiskgraphError::InvalidMetadata("node data length mismatch".into()).into(),
                );
            }
            Some(topology_end..end)
        } else {
            if topology_end != record_end {
                return Err(
                    DiskgraphError::InvalidMetadata("record length mismatch".into()).into(),
                );
            }
            None
        };
        Ok(ParsedNodeHeader {
            local_id,
            record_start,
            record_end,
            relations,
            topology_end,
            node_data,
        })
    }

    /// Lookup a node's edge list by index within the shard.
    pub fn edges_for_node(&self, node_idx: u64) -> Result<EdgeListView<'_>> {
        let (start, end) = self.record_range(node_idx)?;
        let start_byte = start as usize;
        let end_byte = end as usize;
        if end_byte > self.edges.len() || start_byte > end_byte {
            return Err(
                DiskgraphError::InvalidMetadata("offset points outside edge blob".into()).into(),
            );
        }
        let raw = &self.edges[start_byte..end_byte];
        Ok(EdgeListView {
            raw,
            layouts: self.layouts.clone(),
            has_node_data: self.has_node_data,
            node_data_len_bytes: self.header.node_data_len_bytes,
        })
    }

    /// Start reading in every page the subsequent `edges_for_node` calls for
    /// `node_indices` will touch, so the batch's reads overlap instead of
    /// serializing one blocking major fault at a time.
    ///
    /// Two stages, because an edge list's location is not known until its
    /// offsets are decoded:
    ///   1. the offset-chunk pages — addressable straight from the in-RAM chunk
    ///      table, so this costs no faults of its own;
    ///   2. the edge-blob ranges those offsets decode to — issued while stage 1's
    ///      reads are still landing.
    ///
    /// A hint only: it never changes which bytes are read. Callers pass a whole
    /// frontier, which the traversal already materializes before walking it.
    /// No-op unless `DG_PREFETCH=1`.
    ///
    /// Callers must gate on [`prefetch::admit`] first — that is what keeps the
    /// cache-resident case free — and pass its `is_probe` result as `probing`.
    pub fn prefetch_nodes(&self, node_indices: &[u64], probing: bool) {
        if node_indices.is_empty() {
            return;
        }

        let mut ranges = Vec::with_capacity(node_indices.len());
        for &idx in node_indices {
            // Both `idx` and `idx + 1` are decoded by `edges_for_node`, and a
            // chunk boundary puts them in different chunks.
            for i in [idx, idx + 1] {
                if let Some(r) = self.offset_chunk_range(i) {
                    ranges.push(r);
                }
            }
        }
        prefetch::willneed(&self.offsets_mmap, &mut ranges);

        ranges.clear();
        for &idx in node_indices {
            if idx + 1 >= self.header.total_offsets {
                continue;
            }
            let delta = self.header.delta_bytes;
            let (Ok(start), Ok(end)) = (
                decode_offset(&self.offsets_mmap, delta, self.offsets_header_len, idx),
                decode_offset(&self.offsets_mmap, delta, self.offsets_header_len, idx + 1),
            ) else {
                continue; // a corrupt offset is the read path's problem, not ours
            };
            let (start, end) = (start as usize, end as usize);
            if start <= end && end <= self.edges.len() {
                ranges.push((start, end));
            }
        }
        // Residency is measured on the EDGE blob — the offsets file is small
        // enough to stay cached even when the edges are nowhere near it, so
        // gating on the offsets would switch prefetch off exactly when it is
        // needed. Must observe before advising, or the advice makes the pages
        // look resident.
        if probing {
            prefetch::observe(&self.edges, &ranges);
        }
        prefetch::willneed(&self.edges, &mut ranges);
    }

    /// Byte range of the offset chunk holding `idx`, from the in-RAM chunk
    /// table alone (no access to the mapping, so it cannot fault).
    fn offset_chunk_range(&self, idx: u64) -> Option<(usize, usize)> {
        offset_chunk_bounds(
            idx,
            self.header.total_offsets,
            self.header.delta_bytes,
            self.offsets_header_len,
        )
        .filter(|&(_, end)| end <= self.offsets_mmap.len())
    }
}

fn decode_offset_pair(
    offsets: &Mmap,
    idx: u64,
    total_offsets: u64,
    delta_bytes: u8,
    header_len: usize,
) -> Result<(u64, u64)> {
    let start = decode_offset(offsets, delta_bytes, header_len, idx)?;
    let within = idx % crate::layout::DELTA_CHUNK as u64;
    if within + 1 == crate::layout::DELTA_CHUNK as u64 {
        let end = decode_offset(offsets, delta_bytes, header_len, idx + 1)?;
        return Ok((start, end));
    }

    let (chunk_start, chunk_end) = offset_chunk_bounds(idx, total_offsets, delta_bytes, header_len)
        .ok_or_else(|| DiskgraphError::InvalidMetadata("offset chunk missing".into()))?;
    let delta_offset = chunk_start
        .checked_add(10)
        .and_then(|base| base.checked_add(within as usize * delta_bytes as usize))
        .ok_or_else(|| DiskgraphError::InvalidMetadata("offset delta overflow".into()))?;
    let delta_end = delta_offset
        .checked_add(delta_bytes as usize)
        .ok_or_else(|| DiskgraphError::InvalidMetadata("offset delta overflow".into()))?;
    if delta_end > chunk_end || delta_end > offsets.len() {
        return Err(DiskgraphError::InvalidMetadata("offset delta out of bounds".into()).into());
    }
    let delta = match delta_bytes {
        2 => u16::from_le_bytes([offsets[delta_offset], offsets[delta_offset + 1]]) as u64,
        3 => {
            offsets[delta_offset] as u64
                | (offsets[delta_offset + 1] as u64) << 8
                | (offsets[delta_offset + 2] as u64) << 16
        }
        4 => u32::from_le_bytes(
            offsets[delta_offset..delta_end]
                .try_into()
                .map_err(|_| DiskgraphError::InvalidMetadata("offset delta slice error".into()))?,
        ) as u64,
        _ => {
            return Err(DiskgraphError::InvalidMetadata("offset delta width invalid".into()).into())
        }
    };
    let end = start
        .checked_add(delta)
        .ok_or_else(|| DiskgraphError::InvalidMetadata("offset delta overflow".into()))?;
    Ok((start, end))
}

fn advise_random(map: &Mmap) {
    let ret = unsafe {
        madvise(
            map.as_ptr() as *mut libc::c_void,
            map.len(),
            libc::MADV_RANDOM,
        )
    };
    if ret != 0 {
        let _ = std::io::Error::last_os_error();
    }
}

fn advise_file_random(file: &File) -> std::io::Result<()> {
    let ret = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::from_raw_os_error(ret))
    }
}

fn read_len_le(buf: &[u8]) -> usize {
    match buf.len() {
        2 => u16::from_le_bytes([buf[0], buf[1]]) as usize,
        3 => (buf[0] as u32 | (buf[1] as u32) << 8 | (buf[2] as u32) << 16) as usize,
        4 => u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize,
        _ => 0,
    }
}

fn decode_count(buf: &[u8]) -> std::io::Result<(u32, usize)> {
    if buf.is_empty() {
        return Err(std::io::ErrorKind::UnexpectedEof.into());
    }
    let first = buf[0];
    let width = (first >> 6) as usize + 1;
    if buf.len() < width {
        return Err(std::io::ErrorKind::UnexpectedEof.into());
    }
    let mut out = (first & 0x3f) as u32;
    for &byte in &buf[1..width] {
        out = (out << 8) | byte as u32;
    }
    Ok((out, width))
}

fn validate_offset_storage_len(
    actual_len: usize,
    total_offsets: u64,
    delta_bytes: u8,
    header_len: usize,
) -> Result<()> {
    let chunk = crate::layout::DELTA_CHUNK as u64;
    let full_chunks = total_offsets / chunk;
    let remainder = total_offsets % chunk;
    let full_chunk_bytes = offset_chunk_bytes(crate::layout::DELTA_CHUNK as u16, delta_bytes)?;
    let mut expected = header_len
        .checked_add(
            (full_chunks as usize)
                .checked_mul(full_chunk_bytes)
                .ok_or_else(|| {
                    DiskgraphError::InvalidMetadata("offset file length overflow".into())
                })?,
        )
        .ok_or_else(|| DiskgraphError::InvalidMetadata("offset file length overflow".into()))?;
    if remainder != 0 {
        expected = expected
            .checked_add(offset_chunk_bytes(remainder as u16, delta_bytes)?)
            .ok_or_else(|| DiskgraphError::InvalidMetadata("offset file length overflow".into()))?;
    }
    if actual_len != expected {
        return Err(DiskgraphError::InvalidMetadata(format!(
            "offset file length mismatch: expected {expected}, found {actual_len}"
        ))
        .into());
    }
    Ok(())
}

fn offset_chunk_bytes(count: u16, delta_bytes: u8) -> Result<usize> {
    if count == 0 || count as usize > crate::layout::DELTA_CHUNK {
        return Err(DiskgraphError::InvalidMetadata("offset chunk count invalid".into()).into());
    }
    Ok(2 + 8 + count.saturating_sub(1) as usize * delta_bytes as usize)
}

fn expected_offset_chunk_count(total_offsets: u64, chunk_idx: u64) -> Option<u16> {
    let base = chunk_idx.checked_mul(crate::layout::DELTA_CHUNK as u64)?;
    if base >= total_offsets {
        return None;
    }
    Some((total_offsets - base).min(crate::layout::DELTA_CHUNK as u64) as u16)
}

fn offset_chunk_bounds(
    idx: u64,
    total_offsets: u64,
    delta_bytes: u8,
    header_len: usize,
) -> Option<(usize, usize)> {
    if idx >= total_offsets {
        return None;
    }
    let chunk_idx = idx / crate::layout::DELTA_CHUNK as u64;
    let count = expected_offset_chunk_count(total_offsets, chunk_idx)?;
    let full_chunk_bytes = 2 + 8 + (crate::layout::DELTA_CHUNK - 1) * delta_bytes as usize;
    let start = header_len.checked_add((chunk_idx as usize).checked_mul(full_chunk_bytes)?)?;
    let bytes = 2 + 8 + count.saturating_sub(1) as usize * delta_bytes as usize;
    Some((start, start.checked_add(bytes)?))
}

pub(crate) fn decode_offset(
    offsets: &[u8],
    delta_bytes: u8,
    header_len: usize,
    idx: u64,
) -> Result<u64> {
    let (mut cursor, within) = locate_chunk(offsets, delta_bytes, header_len, idx)?;
    cursor += 2;
    let base_bytes: [u8; 8] = offsets[cursor..cursor + 8]
        .try_into()
        .map_err(|_| DiskgraphError::InvalidMetadata("offset base slice error".into()))?;
    let base = u64::from_le_bytes(base_bytes);
    if within == 0 {
        return Ok(base);
    }
    cursor += 8;
    let mut acc = base;
    match delta_bytes {
        2 => {
            for _ in 0..within {
                if cursor + 2 > offsets.len() {
                    return Err(DiskgraphError::InvalidMetadata(
                        "offset delta out of bounds".into(),
                    )
                    .into());
                }
                let delta = u16::from_le_bytes([offsets[cursor], offsets[cursor + 1]]);
                acc = acc.checked_add(delta as u64).ok_or_else(|| {
                    DiskgraphError::InvalidMetadata("offset delta overflow".into())
                })?;
                cursor += 2;
            }
        }
        3 => {
            let mut remaining = within;
            while remaining >= 4 {
                if cursor + 12 > offsets.len() {
                    return Err(DiskgraphError::InvalidMetadata(
                        "offset delta out of bounds".into(),
                    )
                    .into());
                }
                for _ in 0..4 {
                    let delta = offsets[cursor] as u64
                        | ((offsets[cursor + 1] as u64) << 8)
                        | ((offsets[cursor + 2] as u64) << 16);
                    acc = acc.checked_add(delta).ok_or_else(|| {
                        DiskgraphError::InvalidMetadata("offset delta overflow".into())
                    })?;
                    cursor += 3;
                }
                remaining -= 4;
            }
            for _ in 0..remaining {
                if cursor + 3 > offsets.len() {
                    return Err(DiskgraphError::InvalidMetadata(
                        "offset delta out of bounds".into(),
                    )
                    .into());
                }
                let delta = offsets[cursor] as u64
                    | ((offsets[cursor + 1] as u64) << 8)
                    | ((offsets[cursor + 2] as u64) << 16);
                acc = acc.checked_add(delta).ok_or_else(|| {
                    DiskgraphError::InvalidMetadata("offset delta overflow".into())
                })?;
                cursor += 3;
            }
        }
        4 => {
            for _ in 0..within {
                if cursor + 4 > offsets.len() {
                    return Err(DiskgraphError::InvalidMetadata(
                        "offset delta out of bounds".into(),
                    )
                    .into());
                }
                let delta = u32::from_le_bytes([
                    offsets[cursor],
                    offsets[cursor + 1],
                    offsets[cursor + 2],
                    offsets[cursor + 3],
                ]);
                acc = acc.checked_add(delta as u64).ok_or_else(|| {
                    DiskgraphError::InvalidMetadata("offset delta overflow".into())
                })?;
                cursor += 4;
            }
        }
        _ => {
            return Err(
                DiskgraphError::InvalidMetadata("offset delta width invalid".into()).into(),
            );
        }
    }
    Ok(acc)
}

/// Chunks are fixed-size (`DELTA_CHUNK` offsets, last one partial), so a
/// chunk's byte position follows from its index without a directory.
fn locate_chunk(
    offsets: &[u8],
    delta_bytes: u8,
    header_len: usize,
    idx: u64,
) -> Result<(usize, u64)> {
    let chunk = DELTA_CHUNK as u64;
    let chunk_idx = idx / chunk;
    let stride = 10 + (chunk - 1) * delta_bytes as u64;
    let cursor = header_len as u64 + chunk_idx * stride;
    if cursor + 10 > offsets.len() as u64 {
        return Err(DiskgraphError::InvalidMetadata("offset chunk missing".into()).into());
    }
    let cursor = cursor as usize;
    let count = u16::from_le_bytes([offsets[cursor], offsets[cursor + 1]]) as u64;
    let within = idx % chunk;
    if within >= count {
        return Err(DiskgraphError::InvalidMetadata("offset chunk range mismatch".into()).into());
    }
    Ok((cursor, within))
}

#[cfg(test)]
mod explicit_reader_tests {
    use super::*;

    #[test]
    fn compact_counts_all_widths_match_layout_format() {
        assert_eq!(decode_count(&[0x3f]).unwrap(), (63, 1));
        assert_eq!(decode_count(&[0x40, 0x40]).unwrap(), (64, 2));
        assert_eq!(decode_count(&[0x80, 0x01, 0x02]).unwrap(), (0x0102, 3));
        assert_eq!(
            decode_count(&[0xc0, 0x01, 0x02, 0x03]).unwrap(),
            (0x010203, 4)
        );
        assert_eq!(
            decode_count(&[0x40]).unwrap_err().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn canonical_offset_chunk_positions_need_no_heap_directory() {
        let header_len = 32;
        let delta_bytes = 2;
        let full_bytes = 2 + 8 + 63 * delta_bytes as usize;
        assert_eq!(
            offset_chunk_bounds(0, 130, delta_bytes, header_len),
            Some((header_len, header_len + full_bytes))
        );
        assert_eq!(
            offset_chunk_bounds(64, 130, delta_bytes, header_len),
            Some((header_len + full_bytes, header_len + 2 * full_bytes))
        );
        let final_start = header_len + 2 * full_bytes;
        assert_eq!(
            offset_chunk_bounds(129, 130, delta_bytes, header_len),
            Some((final_start, final_start + 2 + 8 + delta_bytes as usize))
        );
        assert!(offset_chunk_bounds(130, 130, delta_bytes, header_len).is_none());
        validate_offset_storage_len(
            final_start + 2 + 8 + delta_bytes as usize,
            130,
            delta_bytes,
            header_len,
        )
        .unwrap();
    }

    #[test]
    fn eager_header_plan_uses_complete_records_only_within_page_limit() {
        let prefix = 16;
        // Unaligned two-page record: eager=2 includes its whole body.
        assert_eq!(planned_header_end(4090, 5000, prefix, 0), 4106);
        assert_eq!(planned_header_end(4090, 5000, prefix, 1), 4106);
        assert_eq!(planned_header_end(4090, 5000, prefix, 2), 5000);

        // Three- and four-page policies are exact, configurable extensions.
        let three_page_end = 2 * PAGE_SIZE as u64 + 100;
        assert_eq!(planned_header_end(4090, three_page_end, prefix, 2), 4106);
        assert_eq!(
            planned_header_end(4090, three_page_end, prefix, 3),
            three_page_end
        );
        let four_page_end = 3 * PAGE_SIZE as u64 + 100;
        assert_eq!(planned_header_end(4090, four_page_end, prefix, 3), 4106);
        assert_eq!(
            planned_header_end(4090, four_page_end, prefix, 4),
            four_page_end
        );
    }
}
