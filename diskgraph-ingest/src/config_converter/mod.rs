//! Kumo GraphConfig to DiskGraph GraphConfig converter.
//!
//! This module converts Kumo's graph configuration format to DiskGraph's GraphConfig format,
//! allowing existing Kumo data to be ingested by the new graph engine.

mod convert;
pub mod inference;
mod mapping;
mod types;

pub use crate::errors::ConvertError;
pub use convert::{convert_kumo_to_diskgraph, ConvertOptions};
pub use inference::detect_timestamp_formats;
pub use mapping::{map_kumo_dtype, map_timestamp_unit};
pub use types::*;

#[cfg(test)]
mod tests;
