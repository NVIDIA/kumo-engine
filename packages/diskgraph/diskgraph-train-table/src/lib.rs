// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// TT-gen executor (plan-based, driven by DiskGraphTTPlan from PyO3 builder)
pub mod ttgen_accumulator;
pub mod ttgen_executor;
pub mod ttgen_filter;
pub mod ttgen_output;
pub mod ttgen_resolve;
pub mod ttgen_stats;
pub mod ttgen_traversal;
pub mod tt_stage_ingest;
pub mod errors;
pub mod pass_b;
pub mod parquet_dataset;
pub mod split_decider;
pub mod custom_tt_chunk_processor;
pub mod regroup;
pub mod route;
pub mod manifest;
pub mod pseudo_time;
pub mod split_stats;
pub mod kumo_contract_bridge;
