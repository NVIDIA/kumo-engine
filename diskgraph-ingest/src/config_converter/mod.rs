//! Kumo GraphConfig to DiskGraph GraphConfig converter.
//!
//! This module converts Kumo's graph configuration format to DiskGraph's GraphConfig format,
//! allowing existing Kumo data to be ingested by the new graph engine.

mod types;
mod mapping;
mod convert;

pub use types::*;
pub use mapping::{map_kumo_dtype, map_timestamp_unit};
pub use convert::{convert_kumo_to_diskgraph, ConvertOptions, ConvertError};

#[cfg(test)]
mod tests;
