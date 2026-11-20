// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::DiskgraphConfigError;
use diskgraph_core::sampling::{
    AbsentKeyPolicy, FanoutSpec, LayerSpec, SamplingMode, SamplingPlan, Weighting,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

fn require_type(value: &Bound<'_, PyAny>, types: &Bound<'_, PyModule>, name: &str) -> PyResult<()> {
    if !value.is_instance(&types.getattr(name)?)? {
        return Err(PyTypeError::new_err(format!(
            "sampling_plan must contain diskgraph.{name} values"
        )));
    }
    Ok(())
}

pub(crate) fn extract_sampling_plan(value: &Bound<'_, PyAny>) -> PyResult<SamplingPlan> {
    let types = value.py().import("diskgraph.sampling_plan")?;
    require_type(value, &types, "SamplingPlan")?;
    let layers = value.getattr("layers")?;
    let layers = layers.cast::<PyList>()?;
    let mut result = Vec::with_capacity(layers.len());
    for layer in layers {
        require_type(&layer, &types, "LayerSpec")?;
        let fanouts = layer.getattr("fanouts")?;
        let fanouts = fanouts.cast::<PyDict>()?;
        let mut overrides = std::collections::HashMap::with_capacity(fanouts.len());
        for (name, spec) in fanouts {
            overrides.insert(name.extract()?, extract_fanout(&spec, &types)?);
        }
        let default = layer.getattr("default_fanout")?;
        result.push(LayerSpec {
            fanouts: overrides,
            default_fanout: if default.is_none() {
                None
            } else {
                Some(extract_fanout(&default, &types)?)
            },
        });
    }
    Ok(SamplingPlan { layers: result })
}

fn extract_fanout(value: &Bound<'_, PyAny>, types: &Bound<'_, PyModule>) -> PyResult<FanoutSpec> {
    require_type(value, types, "FanoutSpec")?;
    let fanout = value.getattr("fanout")?.extract::<u32>().map_err(|error| {
        DiskgraphConfigError::new_err(format!("invalid fanout: expected a u32: {error}"))
    })?;
    let policy: String = value.getattr("absent_key_policy")?.extract()?;
    let absent_key_policy = match policy.as_str() {
        "reject_plan" => AbsentKeyPolicy::RejectPlan,
        "uniform_fallback" => AbsentKeyPolicy::UniformFallback,
        _ => {
            return Err(DiskgraphConfigError::new_err(format!(
                "invalid absent_key_policy: {policy}"
            )))
        }
    };
    Ok(FanoutSpec {
        fanout,
        mode: extract_mode(&value.getattr("mode")?, types)?,
        absent_key_policy,
    })
}

fn extract_mode(value: &Bound<'_, PyAny>, types: &Bound<'_, PyModule>) -> PyResult<SamplingMode> {
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
    require_type(value, types, "WeightedSampling")?;
    let weighting = extract_weighting(&value.getattr("weighting")?, types)?;
    Ok(if value.getattr("temporal")?.extract::<bool>()? {
        SamplingMode::TemporalWeighted { weighting }
    } else {
        SamplingMode::Weighted { weighting }
    })
}

fn extract_weighting(value: &Bound<'_, PyAny>, types: &Bound<'_, PyModule>) -> PyResult<Weighting> {
    if value.is_instance(&types.getattr("ExponentialDecay")?)? {
        return Ok(Weighting::ExponentialDecay {
            lambda: value.getattr("rate")?.extract()?,
        });
    }
    if value.is_instance(&types.getattr("LinearDecay")?)? {
        return Ok(Weighting::LinearDecay {
            half_life_seconds: value.getattr("half_life_seconds")?.extract()?,
        });
    }
    require_type(value, types, "CustomExpression")?;
    Ok(Weighting::CustomExpression {
        id: value.getattr("id")?.extract()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn extract_expression(py: Python<'_>, expression: &str) -> PyResult<SamplingPlan> {
        let types = py.import("diskgraph.sampling_plan")?;
        let expression = CString::new(expression).unwrap();
        let value = py.eval(&expression, Some(&types.dict()), None)?;
        extract_sampling_plan(&value)
    }

    #[test]
    fn extracts_layers_overrides_modes_and_policies() {
        Python::initialize();
        Python::attach(|py| {
            let plan = extract_expression(
                py,
                r#"SamplingPlan([
                LayerSpec(
                    {'edge': FanoutSpec(0, 'uniform', 'reject_plan')},
                    FanoutSpec(7, 'deterministic_uniform', 'uniform_fallback'),
                ),
                LayerSpec(default_fanout=FanoutSpec(
                    3, 'temporal_most_recent', 'reject_plan',
                )),
                LayerSpec(),
            ])"#,
            )
            .unwrap();
            assert_eq!(plan.layers.len(), 3);
            let first = &plan.layers[0];
            assert_eq!(first.fanouts["edge"].fanout, 0);
            assert!(matches!(first.fanouts["edge"].mode, SamplingMode::Uniform));
            assert!(matches!(
                first.fanouts["edge"].absent_key_policy,
                AbsentKeyPolicy::RejectPlan
            ));
            let default = first.default_fanout.as_ref().unwrap();
            assert_eq!(default.fanout, 7);
            assert!(matches!(default.mode, SamplingMode::DeterministicUniform));
            assert!(matches!(
                default.absent_key_policy,
                AbsentKeyPolicy::UniformFallback
            ));
            assert!(matches!(
                plan.layers[1].default_fanout.as_ref().unwrap().mode,
                SamplingMode::TemporalMostRecent
            ));
            assert!(plan.layers[2].fanouts.is_empty());
            assert!(plan.layers[2].default_fanout.is_none());
        });
    }

    #[test]
    fn extracts_all_weighting_variants_and_temporal_flags() {
        Python::initialize();
        Python::attach(|py| {
            for temporal in [false, true] {
                for (expression, expected) in [
                    (
                        "ExponentialDecay(0.125)",
                        serde_json::json!({"exponential_decay": {"lambda": 0.125}}),
                    ),
                    (
                        "LinearDecay(3600)",
                        serde_json::json!({"linear_decay": {"half_life_seconds": 3600}}),
                    ),
                    (
                        "CustomExpression('score')",
                        serde_json::json!({"custom_expression": {"id": "score"}}),
                    ),
                ] {
                    let expression = format!(
                        "SamplingPlan([LayerSpec(default_fanout=FanoutSpec(\
                         3, WeightedSampling({expression}, temporal={}), \
                         'reject_plan'))])",
                        if temporal { "True" } else { "False" },
                    );
                    let plan = extract_expression(py, &expression).unwrap();
                    let mode = &plan.layers[0].default_fanout.as_ref().unwrap().mode;
                    let weighting = match (temporal, mode) {
                        (false, SamplingMode::Weighted { weighting }) => weighting,
                        (true, SamplingMode::TemporalWeighted { weighting }) => weighting,
                        _ => panic!("incorrect sampling mode: {mode:?}"),
                    };
                    assert_eq!(serde_json::to_value(weighting).unwrap(), expected);
                }
            }
        });
    }

    #[test]
    fn rejects_untyped_plans_and_nested_values() {
        Python::initialize();
        Python::attach(|py| {
            for expression in [
                "{'layers': []}",
                "None",
                "SamplingPlan([{'fanouts': {}}])",
                "SamplingPlan([LayerSpec(default_fanout={'fanout': 1})])",
                "SamplingPlan([LayerSpec(default_fanout=FanoutSpec(1, {}, 'reject_plan'))])",
                "SamplingPlan([LayerSpec(default_fanout=FanoutSpec(\
                 1, WeightedSampling({}), 'reject_plan'))])",
            ] {
                let error = extract_expression(py, expression).unwrap_err();
                assert!(
                    error.is_instance_of::<PyTypeError>(py),
                    "{expression}: {error}"
                );
            }
        });
    }

    #[test]
    fn rejects_invalid_fanouts_modes_and_policies() {
        Python::initialize();
        Python::attach(|py| {
            for spec in [
                "FanoutSpec(-1, 'uniform', 'reject_plan')",
                "FanoutSpec(4294967296, 'uniform', 'reject_plan')",
                "FanoutSpec(1, 'unknown', 'reject_plan')",
                "FanoutSpec(1, 'uniform', 'unknown')",
            ] {
                let expression = format!("SamplingPlan([LayerSpec(default_fanout={spec})])");
                let error = extract_expression(py, &expression).unwrap_err();
                assert!(
                    error.is_instance_of::<DiskgraphConfigError>(py),
                    "{spec}: {error}"
                );
            }
        });
    }
}
