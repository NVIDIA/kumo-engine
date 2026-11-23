// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Result, RuntimeError};
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

/// Caller intent for replacement seed sampling; sidecars come from the extension.
#[derive(Clone, Debug)]
pub struct WeightedSeedSampling {
    pub num_samples: u64,
    pub false_weight: Option<f64>,
    pub true_weight: Option<f64>,
    pub use_sample_weights: bool,
}

/// Resolved sidecars and masses consumed by the graph-independent seed scheduler.
#[derive(Clone, Debug)]
pub struct WeightedSeedSamplingConfig {
    pub buckets: Vec<WeightedSeedBucketConfig>,
    pub num_samples: u64,
}

/// One uniform or CDF-weighted seed bucket.
#[derive(Clone, Debug)]
pub struct WeightedSeedBucketConfig {
    pub name: String,
    pub node_ids_path: String,
    pub count: u64,
    pub mass: f64,
    pub cdf_path: Option<String>,
    pub total_weight: Option<f64>,
}

#[derive(Deserialize)]
struct BinaryLabelIndex {
    false_node_ids: String,
    false_count: u64,
    true_node_ids: String,
    true_count: u64,
}

#[derive(Deserialize)]
struct WeightedBucketIndex {
    node_ids: String,
    cdf: String,
    count: u64,
    total_weight: f64,
}

pub(crate) fn read_manifest(root: &Path) -> Result<Value> {
    let path = root.join("diskgraph_manifest.json");
    let bytes = std::fs::read(&path)
        .map_err(|error| RuntimeError::system(format!("{}: {error}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| RuntimeError::data(format!("{}: {error}", path.display())))
}

pub(crate) fn resolve(
    manifest: &Value,
    root: &Path,
    table: &str,
    options: WeightedSeedSampling,
) -> Result<WeightedSeedSamplingConfig> {
    let stages = manifest["task_dataset"]["stages"]
        .as_object()
        .ok_or_else(|| {
            RuntimeError::data("weighted_seed_sampling manifest is missing task_dataset.stages")
        })?;
    let stage = stages
        .values()
        .find(|stage| stage["table_name"].as_str() == Some(table))
        .ok_or_else(|| {
            RuntimeError::config(format!(
                "weighted_seed_sampling has no manifest stage for table {table:?}"
            ))
        })?;
    let path = |relative: String| -> Result<String> {
        if relative.is_empty() {
            return Err(RuntimeError::config(format!(
                "weighted_seed_sampling for table {table:?} has an empty sidecar path"
            )));
        }
        Ok(root.join(relative).to_string_lossy().into_owned())
    };
    let metadata = |key: &str| -> Result<&Value> {
        stage
            .get(key)
            .filter(|value| value.is_object())
            .ok_or_else(|| {
                RuntimeError::config(format!(
                    "weighted_seed_sampling requires {key} for table {table:?}"
                ))
            })
    };
    let invalid_index = |error| {
        RuntimeError::data(format!(
            "invalid weighted_seed_sampling metadata for table {table:?}: {error}"
        ))
    };
    let false_weight = options.false_weight.unwrap_or(1.0);
    let true_weight = options.true_weight.unwrap_or(1.0);
    let buckets = if options.use_sample_weights {
        let index = metadata("sample_weight_index")?;
        let weighted_bucket = |key: &str, multiplier: f64| -> Result<WeightedSeedBucketConfig> {
            let entry = index
                .get(key)
                .filter(|entry| entry.is_object())
                .ok_or_else(|| {
                    RuntimeError::config(format!(
                        "sample_weight_index for table {table:?} missing {key:?} bucket"
                    ))
                })?;
            let bucket: WeightedBucketIndex =
                serde_json::from_value(entry.clone()).map_err(invalid_index)?;
            Ok(WeightedSeedBucketConfig {
                name: key.to_string(),
                node_ids_path: path(bucket.node_ids)?,
                count: bucket.count,
                mass: bucket.total_weight * multiplier,
                cdf_path: Some(path(bucket.cdf)?),
                total_weight: Some(bucket.total_weight),
            })
        };
        if options.false_weight.is_some() || options.true_weight.is_some() {
            vec![
                weighted_bucket("false_bucket", false_weight)?,
                weighted_bucket("true_bucket", true_weight)?,
            ]
        } else {
            vec![weighted_bucket("all", 1.0)?]
        }
    } else {
        let index: BinaryLabelIndex =
            serde_json::from_value(metadata("binary_label_index")?.clone())
                .map_err(invalid_index)?;
        [
            (
                "false",
                index.false_node_ids,
                index.false_count,
                false_weight,
            ),
            ("true", index.true_node_ids, index.true_count, true_weight),
        ]
        .into_iter()
        .map(|(name, node_ids, count, weight)| {
            Ok(WeightedSeedBucketConfig {
                name: name.to_string(),
                node_ids_path: path(node_ids)?,
                count,
                mass: count as f64 * weight,
                cdf_path: None,
                total_weight: None,
            })
        })
        .collect::<Result<Vec<_>>>()?
    };
    Ok(WeightedSeedSamplingConfig {
        buckets,
        num_samples: options.num_samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn options() -> WeightedSeedSampling {
        WeightedSeedSampling {
            num_samples: 17,
            false_weight: None,
            true_weight: None,
            use_sample_weights: false,
        }
    }

    #[test]
    fn class_buckets_use_table_name_and_opened_extension_root() {
        let manifest = json!({"task_dataset": {"stages": {
            "validation": {"table_name": "validation_seeds"},
            "train": {
                "table_name": "train_seeds", "index_path": "/old/location",
                "binary_label_index": {
                    "false_node_ids": "train_seeds/false.bin", "false_count": 5,
                    "true_node_ids": "train_seeds/true.bin", "true_count": 2
                }
            }
        }}});
        let mut intent = options();
        intent.true_weight = Some(3.0);
        let config = resolve(
            &manifest,
            Path::new("/moved/extension"),
            "train_seeds",
            intent,
        )
        .unwrap();
        assert_eq!(config.num_samples, 17);
        assert_eq!(config.buckets.len(), 2);
        assert_eq!(config.buckets[0].name, "false");
        assert_eq!(config.buckets[0].mass, 5.0);
        assert_eq!(config.buckets[1].mass, 6.0);
        assert_eq!(config.buckets[1].count, 2);
        assert_eq!(
            config.buckets[1].node_ids_path,
            "/moved/extension/train_seeds/true.bin"
        );
        assert!(config.buckets[0].cdf_path.is_none());
    }

    #[test]
    fn row_weights_select_all_or_ordered_class_buckets() {
        let bucket = |total| {
            json!({
                "node_ids": "ids.bin", "cdf": "cdf.bin", "count": 3, "total_weight": total
            })
        };
        let manifest = json!({"task_dataset": {"stages": {"train": {
            "table_name": "train_seeds", "sample_weight_index": {
                "all": bucket(11.5), "false_bucket": bucket(2.0), "true_bucket": bucket(9.5)
            }
        }}}});
        let mut intent = options();
        intent.use_sample_weights = true;
        let config = resolve(
            &manifest,
            Path::new("/extension"),
            "train_seeds",
            intent.clone(),
        )
        .unwrap();
        assert_eq!(config.buckets.len(), 1);
        assert_eq!(config.buckets[0].name, "all");
        assert_eq!(config.buckets[0].mass, 11.5);
        assert_eq!(
            config.buckets[0].cdf_path.as_deref(),
            Some("/extension/cdf.bin")
        );
        assert_eq!(config.buckets[0].total_weight, Some(11.5));
        intent.false_weight = Some(0.0);
        intent.true_weight = Some(2.0);
        let config = resolve(&manifest, Path::new("/extension"), "train_seeds", intent).unwrap();
        assert_eq!(config.buckets.len(), 2);
        assert_eq!(config.buckets[0].name, "false_bucket");
        assert_eq!(config.buckets[0].mass, 0.0);
        assert_eq!(config.buckets[1].name, "true_bucket");
        assert_eq!(config.buckets[1].mass, 19.0);
        assert_eq!(config.buckets[1].total_weight, Some(9.5));
    }

    #[test]
    fn missing_and_malformed_metadata_fail_with_typed_errors() {
        let root = Path::new("/extension");
        let mut manifest = json!({"task_dataset": {"stages": {"train": {"table_name": "seeds"}}}});
        assert!(matches!(resolve(&manifest, root, "missing", options()),
            Err(RuntimeError::Config(message)) if message.contains("missing")));
        assert!(matches!(resolve(&manifest, root, "seeds", options()),
            Err(RuntimeError::Config(message)) if message.contains("binary_label_index")));
        let mut intent = options();
        intent.use_sample_weights = true;
        assert!(matches!(resolve(&manifest, root, "seeds", intent),
            Err(RuntimeError::Config(message)) if message.contains("sample_weight_index")));
        manifest["task_dataset"]["stages"]["train"]["binary_label_index"] = json!({
            "false_node_ids": "", "false_count": 1, "true_node_ids": "true.bin", "true_count": 1
        });
        assert!(matches!(resolve(&manifest, root, "seeds", options()),
            Err(RuntimeError::Config(message)) if message.contains("empty sidecar path")));
        manifest["task_dataset"]["stages"]["train"]["binary_label_index"]["false_count"] =
            json!(-1);
        assert!(matches!(
            resolve(&manifest, root, "seeds", options()),
            Err(RuntimeError::Data(_))
        ));
        assert!(matches!(
            resolve(&json!({}), root, "seeds", options()),
            Err(RuntimeError::Data(_))
        ));
    }

    #[test]
    fn extension_manifest_read_errors_are_classified() {
        let root = tempfile::tempdir().unwrap();
        assert!(matches!(
            read_manifest(root.path()),
            Err(RuntimeError::System(_))
        ));
        let path = root.path().join("diskgraph_manifest.json");
        std::fs::write(&path, b"invalid json").unwrap();
        assert!(matches!(
            read_manifest(root.path()),
            Err(RuntimeError::Data(_))
        ));
        std::fs::write(&path, b"{\"task_dataset\":{\"stages\":{}}}").unwrap();
        assert!(read_manifest(root.path()).unwrap()["task_dataset"]["stages"].is_object());
    }
}
