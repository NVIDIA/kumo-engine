// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::errors::{DiskgraphError, Result};

/// Interface for compressing adjacency blocks on disk.
///
/// Implementations should be streaming-friendly and support both CPU-bound and
/// IO-bound deployments. The compressor is responsible for encoding adjacency
/// lists and optional attributes (dest_sort_key, edge_id) into contiguous
/// buffers that the sampler can mmap or cache. The chosen format must allow
/// efficient slicing to retrieve a single node's neighbors without scanning
/// unrelated data.
pub trait CompressionCodec: Send + Sync {
    /// Compress a raw adjacency blob. The caller provides already-packed bytes
    /// for one logical unit (e.g., a shard or node bucket). The compressor must
    /// return a new owned buffer.
    fn compress(&self, input: &[u8]) -> Result<Vec<u8>>;

    /// Decompress a buffer produced by `compress`. Implementations must be
    /// deterministic and backwards compatible across versions.
    fn decompress(&self, input: &[u8]) -> Result<Vec<u8>>;
}

/// Placeholder codec used until a real compression strategy is selected. The
/// methods return an explicit NotImplemented error so callers can differentiate
/// between missing algorithm and runtime bugs.
pub struct PlaceholderCompressionCodec;

impl CompressionCodec for PlaceholderCompressionCodec {
    fn compress(&self, _input: &[u8]) -> Result<Vec<u8>> {
        Err(DiskgraphError::NotImplemented(
            "compression codec not implemented",
        ))
    }

    fn decompress(&self, _input: &[u8]) -> Result<Vec<u8>> {
        Err(DiskgraphError::NotImplemented(
            "compression codec not implemented",
        ))
    }
}
