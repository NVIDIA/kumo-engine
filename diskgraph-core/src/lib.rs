//! Core data types and interfaces shared by the diskgraph ingestion pipeline,
//! runtime sampling engine, and HTTP/Python frontends.
//!
//! This crate intentionally contains no heavy implementations of compression,
//! on-disk layout, or sampling algorithms; instead it defines contracts and
//! glue types that the runtime and service layers rely on. The goal is to keep
//! a single source of truth for sampling semantics, metadata layout, and
//! dataset lifecycle state.

pub mod dataset;
pub mod errors;
pub mod features;
pub mod metadata;
pub mod sampling;
pub mod simulation_test;
pub mod sortkey;
pub mod storage;
