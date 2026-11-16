// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The custom training-table column contract, as Python sees it.
//!
//! A supplied training table is validated before it reaches the reader --
//! twice, in two places, on two backends. Each of those checks needs the same
//! two facts: what a column has to be called, and what Arrow types the reader
//! will actually read it as. Those facts live in
//! `diskgraph_train_table::custom_tt_chunk_processor`, beside the arms that
//! enforce them; this exports them so that no caller has to write them down a
//! second time.
//!
//! Every type list is a list of `pyarrow.types.is_<token>` predicate names,
//! and every list is asserted against the reader it describes by
//! `the contract tests` in that module -- acceptance is exactly membership,
//! in both directions.

use diskgraph_train_table::custom_tt_chunk_processor as contract;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let spec = PyDict::new(py);

    spec.set_item("entity_column_prefix", contract::ENTITY_COLUMN_PREFIX)?;
    spec.set_item("target_column_prefix", contract::TARGET_COLUMN_PREFIX)?;
    spec.set_item("timestamp_column", contract::TIMESTAMP_COLUMN)?;
    spec.set_item("weight_output_column", contract::WEIGHT_OUTPUT_COLUMN)?;

    spec.set_item("entity_types", PyList::new(py, contract::ENTITY_TYPES)?)?;
    spec.set_item("scalar_types", PyList::new(py, contract::SCALAR_TYPES)?)?;
    spec.set_item(
        "list_target_containers",
        PyList::new(py, contract::LIST_TARGET_CONTAINERS)?,
    )?;
    spec.set_item(
        "list_target_item_types",
        PyList::new(py, contract::LIST_TARGET_ITEM_TYPES)?,
    )?;
    spec.set_item(
        "timestamp_types",
        PyList::new(py, contract::TIMESTAMP_TYPES)?,
    )?;
    spec.set_item("weight_types", PyList::new(py, contract::WEIGHT_TYPES)?)?;
    spec.set_item(
        "weight_list_containers",
        PyList::new(py, contract::WEIGHT_LIST_CONTAINERS)?,
    )?;
    spec.set_item(
        "weight_list_item_types",
        PyList::new(py, contract::WEIGHT_LIST_ITEM_TYPES)?,
    )?;

    m.add("CUSTOM_TT_CONTRACT", spec)
}
