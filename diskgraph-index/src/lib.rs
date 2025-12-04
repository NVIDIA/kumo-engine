//! Index builder and shard reader for diskgraph.
//!
//! This crate focuses on per-shard ingestion and lookup:
//! - `ShardBuilder` ingests a node shard and its edge shards into a compact
//!   on-disk representation (offset file + edge data file).
//! - `ShardReader` mmaps the on-disk representation and provides zero-copy
//!   edge list views for lookup.
//!
//! Compression and bit-packing of edge payloads are intentionally left
//! unimplemented; the APIs document the intended layout so the implementation
//! can be swapped in later without changing callers.

pub mod builder;
pub mod layout;
pub mod reader;
pub mod streaming;
