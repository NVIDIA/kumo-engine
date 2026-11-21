// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::DiskgraphConfigError;
use diskgraph_core::sampling::{
    AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingMode, SamplingPlan, Weighting,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping, PyTuple};
use std::collections::HashMap;

/// Exponential edge weighting.
///
/// Args:
///     rate: Decay rate per second, corresponding to Rust's lambda field.
#[pyclass(
    frozen,
    skip_from_py_object,
    eq,
    get_all,
    name = "ExponentialDecay",
    module = "diskgraph._c"
)]
#[derive(Clone, PartialEq)]
struct PyExponentialDecay {
    rate: f64,
}

#[pymethods]
impl PyExponentialDecay {
    #[new]
    fn new(rate: f64) -> Self {
        Self { rate }
    }

    fn __getnewargs__(&self) -> (f64,) {
        (self.rate,)
    }
}

/// Linear edge weighting.
///
/// Args:
///     half_life_seconds: Native linear-decay time scale in seconds.
#[pyclass(
    frozen,
    skip_from_py_object,
    eq,
    get_all,
    name = "LinearDecay",
    module = "diskgraph._c"
)]
#[derive(Clone, PartialEq)]
struct PyLinearDecay {
    half_life_seconds: u64,
}

#[pymethods]
impl PyLinearDecay {
    #[new]
    fn new(half_life_seconds: u64) -> Self {
        Self { half_life_seconds }
    }

    fn __getnewargs__(&self) -> (u64,) {
        (self.half_life_seconds,)
    }
}

/// Native custom weighting expression.
///
/// Args:
///     id: Expression identifier interpreted by the engine.
#[pyclass(
    frozen,
    skip_from_py_object,
    eq,
    get_all,
    name = "CustomExpression",
    module = "diskgraph._c"
)]
#[derive(Clone, PartialEq)]
struct PyCustomExpression {
    id: String,
}

#[pymethods]
impl PyCustomExpression {
    #[new]
    fn new(id: String) -> Self {
        Self { id }
    }

    fn __getnewargs__(&self) -> (&str,) {
        (&self.id,)
    }
}

/// Weighted or temporal-weighted neighbor selection.
///
/// Args:
///     weighting: ExponentialDecay, LinearDecay, or CustomExpression.
///     temporal: Select temporal_weighted instead of weighted.
#[pyclass(
    frozen,
    skip_from_py_object,
    eq,
    name = "WeightedSampling",
    module = "diskgraph._c"
)]
#[derive(Clone, PartialEq)]
struct PyWeightedSampling {
    inner: SamplingMode,
}

#[pymethods]
impl PyWeightedSampling {
    #[new]
    #[pyo3(signature = (weighting, temporal=false))]
    fn new(weighting: &Bound<'_, PyAny>, temporal: bool) -> PyResult<Self> {
        let weighting = if let Ok(value) = weighting.extract::<PyRef<'_, PyExponentialDecay>>() {
            Weighting::ExponentialDecay { lambda: value.rate }
        } else if let Ok(value) = weighting.extract::<PyRef<'_, PyLinearDecay>>() {
            Weighting::LinearDecay {
                half_life_seconds: value.half_life_seconds,
            }
        } else {
            let value: PyRef<'_, PyCustomExpression> = weighting.extract()?;
            Weighting::CustomExpression {
                id: value.id.clone(),
            }
        };
        Ok(Self {
            inner: if temporal {
                SamplingMode::TemporalWeighted { weighting }
            } else {
                SamplingMode::Weighted { weighting }
            },
        })
    }

    #[getter]
    fn weighting<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (SamplingMode::Weighted { weighting } | SamplingMode::TemporalWeighted { weighting }) =
            &self.inner
        else {
            unreachable!("WeightedSampling only stores weighted modes")
        };
        match weighting {
            Weighting::ExponentialDecay { lambda } => {
                Ok(Py::new(py, PyExponentialDecay { rate: *lambda })?
                    .into_bound(py)
                    .into_any())
            }
            Weighting::LinearDecay { half_life_seconds } => Ok(Py::new(
                py,
                PyLinearDecay {
                    half_life_seconds: *half_life_seconds,
                },
            )?
            .into_bound(py)
            .into_any()),
            Weighting::CustomExpression { id } => {
                Ok(Py::new(py, PyCustomExpression { id: id.clone() })?
                    .into_bound(py)
                    .into_any())
            }
        }
    }

    #[getter]
    fn temporal(&self) -> bool {
        matches!(self.inner, SamplingMode::TemporalWeighted { .. })
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, bool)> {
        Ok((self.weighting(py)?, self.temporal()))
    }
}

fn sampling_mode(value: &Bound<'_, PyAny>) -> PyResult<SamplingMode> {
    if let Ok(mode) = value.extract::<String>() {
        return match mode.as_str() {
            "uniform" => Ok(SamplingMode::Uniform),
            "deterministic_uniform" => Ok(SamplingMode::DeterministicUniform),
            "temporal_most_recent" => Ok(SamplingMode::TemporalMostRecent),
            _ => Err(DiskgraphConfigError::new_err(format!(
                "invalid sampling mode: {mode}"
            ))),
        };
    }
    Ok(value
        .extract::<PyRef<'_, PyWeightedSampling>>()?
        .inner
        .clone())
}

/// Neighbor count, selection strategy, and missing-key policy for one relation.
///
/// Args:
///     fanout: Maximum neighbor count; zero disables the relation.
///     mode: A sampling-mode string or WeightedSampling.
///     absent_key_policy: reject_plan or uniform_fallback.
///
/// Raises:
///     DiskgraphConfigError: If the count, mode, or policy is invalid.
#[pyclass(
    frozen,
    skip_from_py_object,
    eq,
    name = "FanoutSpec",
    module = "diskgraph._c"
)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyFanoutSpec {
    inner: FanoutSpec,
}

#[pymethods]
impl PyFanoutSpec {
    #[new]
    fn new(
        fanout: &Bound<'_, PyAny>,
        mode: &Bound<'_, PyAny>,
        absent_key_policy: &str,
    ) -> PyResult<Self> {
        let fanout = fanout.extract::<u32>().map_err(|error| {
            DiskgraphConfigError::new_err(format!("invalid fanout: expected a u32: {error}"))
        })?;
        let absent_key_policy = match absent_key_policy {
            "reject_plan" => AbsentKeyPolicy::RejectPlan,
            "uniform_fallback" => AbsentKeyPolicy::UniformFallback,
            _ => {
                return Err(DiskgraphConfigError::new_err(format!(
                    "invalid absent_key_policy: {absent_key_policy}"
                )))
            }
        };
        Ok(Self {
            inner: FanoutSpec {
                fanout,
                mode: sampling_mode(mode)?,
                absent_key_policy,
            },
        })
    }

    #[getter]
    fn fanout(&self) -> u32 {
        self.inner.fanout
    }

    #[getter]
    fn mode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let name = match &self.inner.mode {
            SamplingMode::Uniform => "uniform",
            SamplingMode::DeterministicUniform => "deterministic_uniform",
            SamplingMode::TemporalMostRecent => "temporal_most_recent",
            mode => {
                return Ok(Py::new(
                    py,
                    PyWeightedSampling {
                        inner: mode.clone(),
                    },
                )?
                .into_bound(py)
                .into_any())
            }
        };
        Ok(name.into_pyobject(py)?.into_any())
    }

    #[getter]
    fn absent_key_policy(&self) -> &'static str {
        match self.inner.absent_key_policy {
            AbsentKeyPolicy::RejectPlan => "reject_plan",
            AbsentKeyPolicy::UniformFallback => "uniform_fallback",
        }
    }

    fn __getnewargs__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(u32, Bound<'py, PyAny>, &'static str)> {
        Ok((self.fanout(), self.mode(py)?, self.absent_key_policy()))
    }
}

/// Immutable per-relation fanouts and an optional fallback for one hop.
///
/// Args:
///     fanouts: Overrides keyed by native relation name; copied on construction.
///     default_fanout: Fallback for unspecified relations; None skips them.
#[pyclass(
    frozen,
    skip_from_py_object,
    eq,
    name = "LayerSpec",
    module = "diskgraph._c"
)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyLayerSpec {
    inner: LayerSpec,
}

impl PyLayerSpec {
    fn fanout_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let values = PyDict::new(py);
        for (name, inner) in &self.inner.fanouts {
            values.set_item(
                name,
                PyFanoutSpec {
                    inner: inner.clone(),
                },
            )?;
        }
        Ok(values)
    }
}

#[pymethods]
impl PyLayerSpec {
    #[new]
    #[pyo3(signature = (fanouts=None, default_fanout=None))]
    fn new(
        fanouts: Option<&Bound<'_, PyMapping>>,
        default_fanout: Option<PyRef<'_, PyFanoutSpec>>,
    ) -> PyResult<Self> {
        let mut overrides =
            HashMap::with_capacity(fanouts.map(|values| values.len()).transpose()?.unwrap_or(0));
        if let Some(values) = fanouts {
            for item in values.items()?.iter() {
                let (name, spec): (String, PyRef<'_, PyFanoutSpec>) = item.extract()?;
                overrides.insert(name, spec.inner.clone());
            }
        }
        Ok(Self {
            inner: LayerSpec {
                fanouts: overrides,
                default_fanout: default_fanout.map(|spec| spec.inner.clone()),
            },
        })
    }

    /// Read-only mapping of relation overrides.
    #[getter]
    fn fanouts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let values = self.fanout_values(py)?;
        py.import("types")?
            .getattr("MappingProxyType")?
            .call1((values,))
    }

    #[getter]
    fn default_fanout(&self) -> Option<PyFanoutSpec> {
        self.inner
            .default_fanout
            .as_ref()
            .map(|inner| PyFanoutSpec {
                inner: inner.clone(),
            })
    }

    fn __getnewargs__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyDict>, Option<PyFanoutSpec>)> {
        Ok((self.fanout_values(py)?, self.default_fanout()))
    }
}

/// Immutable, Rust-owned traversal hops, from the seed nodes outward.
///
/// Args:
///     layers: Ordered LayerSpec values; copied on construction.
#[pyclass(
    frozen,
    skip_from_py_object,
    eq,
    name = "SamplingPlan",
    module = "diskgraph._c"
)]
#[derive(Clone, PartialEq)]
pub struct PySamplingPlan {
    pub(crate) inner: SamplingPlan,
}

#[pymethods]
impl PySamplingPlan {
    #[new]
    #[pyo3(signature = (layers=None))]
    fn new(layers: Option<Vec<PyRef<'_, PyLayerSpec>>>) -> Self {
        Self {
            inner: SamplingPlan {
                layers: layers
                    .unwrap_or_default()
                    .into_iter()
                    .map(|layer| layer.inner.clone())
                    .collect(),
            },
        }
    }

    /// Ordered, immutable traversal layers.
    #[getter]
    fn layers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.inner.layers.iter().map(|inner| PyLayerSpec {
                inner: inner.clone(),
            }),
        )
    }

    fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyTuple>,)> {
        Ok((self.layers(py)?,))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExponentialDecay>()?;
    m.add_class::<PyLinearDecay>()?;
    m.add_class::<PyCustomExpression>()?;
    m.add_class::<PyWeightedSampling>()?;
    m.add_class::<PyFanoutSpec>()?;
    m.add_class::<PyLayerSpec>()?;
    m.add_class::<PySamplingPlan>()?;
    Ok(())
}
