// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Result, RuntimeError};
use serde::Deserialize;
use std::{collections::HashMap, path::Path};

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

#[derive(Debug, Deserialize)]
struct Manifest {
    task_dataset: TaskDataset,
}

#[derive(Debug, Deserialize)]
struct TaskDataset {
    stages: HashMap<String, StageMetadata>,
}

#[derive(Debug, Deserialize)]
struct StageMetadata {
    table_name: String,
    binary_label_index: Option<BinaryLabelIndex>,
    sample_weight_index: Option<SampleWeightIndex>,
}

#[derive(Debug, Deserialize)]
struct BinaryLabelIndex {
    false_node_ids: String,
    false_count: u64,
    true_node_ids: String,
    true_count: u64,
}

#[derive(Debug, Deserialize)]
struct SampleWeightIndex {
    all: WeightedBucketIndex,
    false_bucket: Option<WeightedBucketIndex>,
    true_bucket: Option<WeightedBucketIndex>,
}

#[derive(Debug, Deserialize)]
struct WeightedBucketIndex {
    node_ids: String,
    cdf: String,
    count: u64,
    total_weight: f64,
}

#[derive(Debug)]
pub(crate) struct WeightedSamplingMetadata {
    stages: HashMap<String, StageMetadata>,
}

pub(crate) fn read_manifest(root: &Path) -> Result<Option<WeightedSamplingMetadata>> {
    let path = root.join("diskgraph_manifest.json");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(RuntimeError::system(format!("{}: {error}", path.display())));
        }
    };
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .map_err(|error| RuntimeError::data(format!("{}: {error}", path.display())))?;
    resolve_paths(manifest, root).map(Some)
}

fn resolve_paths(manifest: Manifest, root: &Path) -> Result<WeightedSamplingMetadata> {
    let mut stages = HashMap::with_capacity(manifest.task_dataset.stages.len());
    for mut stage in manifest.task_dataset.stages.into_values() {
        let path = |relative: &mut String| -> Result<()> {
            if relative.is_empty() {
                return Err(RuntimeError::data(format!(
                    "weighted_seed_sampling for table {:?} has an empty sidecar path",
                    stage.table_name,
                )));
            }
            *relative = root.join(&*relative).to_string_lossy().into_owned();
            Ok(())
        };
        if let Some(index) = &mut stage.binary_label_index {
            path(&mut index.false_node_ids)?;
            path(&mut index.true_node_ids)?;
        }
        if let Some(index) = &mut stage.sample_weight_index {
            for bucket in std::iter::once(&mut index.all)
                .chain(index.false_bucket.iter_mut())
                .chain(index.true_bucket.iter_mut())
            {
                path(&mut bucket.node_ids)?;
                path(&mut bucket.cdf)?;
            }
        }
        if let Some(previous) = stages.insert(stage.table_name.clone(), stage) {
            return Err(RuntimeError::data(format!(
                "weighted_seed_sampling manifest has duplicate table {:?}",
                previous.table_name,
            )));
        }
    }
    Ok(WeightedSamplingMetadata { stages })
}

impl WeightedSamplingMetadata {
    pub(crate) fn resolve(
        &self,
        table: &str,
        options: WeightedSeedSampling,
    ) -> Result<WeightedSeedSamplingConfig> {
        let stage = self.stages.get(table).ok_or_else(|| {
            RuntimeError::config(format!(
                "weighted_seed_sampling has no manifest stage for table {table:?}"
            ))
        })?;
        let missing = |key: &str| {
            RuntimeError::config(format!(
                "weighted_seed_sampling requires {key} for table {table:?}"
            ))
        };
        let false_weight = options.false_weight.unwrap_or(1.0);
        let true_weight = options.true_weight.unwrap_or(1.0);
        let buckets = if options.use_sample_weights {
            let index = stage
                .sample_weight_index
                .as_ref()
                .ok_or_else(|| missing("sample_weight_index"))?;
            let weighted_bucket = |key: &str,
                                   bucket: Option<&WeightedBucketIndex>,
                                   multiplier: f64|
             -> Result<WeightedSeedBucketConfig> {
                let bucket = bucket.ok_or_else(|| missing(key))?;
                Ok(WeightedSeedBucketConfig {
                    name: key.to_string(),
                    node_ids_path: bucket.node_ids.clone(),
                    count: bucket.count,
                    mass: bucket.total_weight * multiplier,
                    cdf_path: Some(bucket.cdf.clone()),
                    total_weight: Some(bucket.total_weight),
                })
            };
            if options.false_weight.is_some() || options.true_weight.is_some() {
                vec![
                    weighted_bucket("false_bucket", index.false_bucket.as_ref(), false_weight)?,
                    weighted_bucket("true_bucket", index.true_bucket.as_ref(), true_weight)?,
                ]
            } else {
                vec![weighted_bucket("all", Some(&index.all), 1.0)?]
            }
        } else {
            let index = stage
                .binary_label_index
                .as_ref()
                .ok_or_else(|| missing("binary_label_index"))?;
            [
                (
                    "false",
                    &index.false_node_ids,
                    index.false_count,
                    false_weight,
                ),
                ("true", &index.true_node_ids, index.true_count, true_weight),
            ]
            .into_iter()
            .map(|(name, node_ids, count, weight)| WeightedSeedBucketConfig {
                name: name.to_string(),
                node_ids_path: node_ids.clone(),
                count,
                mass: count as f64 * weight,
                cdf_path: None,
                total_weight: None,
            })
            .collect()
        };
        Ok(WeightedSeedSamplingConfig {
            buckets,
            num_samples: options.num_samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn metadata(manifest: Value, root: &Path) -> Result<WeightedSamplingMetadata> {
        let manifest = serde_json::from_value(manifest)
            .map_err(|error| RuntimeError::data(format!("invalid manifest: {error}")))?;
        resolve_paths(manifest, root)
    }

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
        let metadata = metadata(manifest, Path::new("/moved/extension")).unwrap();
        let config = metadata.resolve("train_seeds", intent).unwrap();
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
        let metadata = metadata(manifest, Path::new("/extension")).unwrap();
        let config = metadata.resolve("train_seeds", intent.clone()).unwrap();
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
        let config = metadata.resolve("train_seeds", intent).unwrap();
        assert_eq!(config.buckets.len(), 2);
        assert_eq!(config.buckets[0].name, "false_bucket");
        assert_eq!(config.buckets[0].mass, 0.0);
        assert_eq!(config.buckets[1].name, "true_bucket");
        assert_eq!(config.buckets[1].mass, 19.0);
        assert_eq!(config.buckets[1].total_weight, Some(9.5));
    }

    #[test]
    fn missing_metadata_fails_when_weighted_sampling_is_requested() {
        let manifest = json!({"task_dataset": {"stages": {"train": {"table_name": "seeds"}}}});
        let metadata = metadata(manifest, Path::new("/extension")).unwrap();
        assert!(matches!(metadata.resolve("missing", options()),
            Err(RuntimeError::Config(message)) if message.contains("missing")));
        assert!(matches!(metadata.resolve("seeds", options()),
            Err(RuntimeError::Config(message)) if message.contains("binary_label_index")));
        let mut intent = options();
        intent.use_sample_weights = true;
        assert!(matches!(metadata.resolve("seeds", intent),
            Err(RuntimeError::Config(message)) if message.contains("sample_weight_index")));
    }

    #[test]
    fn malformed_metadata_fails_during_initialization() {
        let root = Path::new("/extension");
        let mut manifest = json!({"task_dataset": {"stages": {"train": {
            "table_name": "seeds", "binary_label_index": {
                "false_node_ids": "", "false_count": 1,
                "true_node_ids": "true.bin", "true_count": 1
            }
        }}}});
        assert!(matches!(metadata(manifest.clone(), root),
            Err(RuntimeError::Data(message)) if message.contains("empty sidecar path")));
        manifest["task_dataset"]["stages"]["train"]["binary_label_index"]["false_count"] =
            json!(-1);
        assert!(matches!(
            metadata(manifest, root),
            Err(RuntimeError::Data(_))
        ));
        assert!(matches!(
            metadata(json!({}), root),
            Err(RuntimeError::Data(_))
        ));
    }

    #[test]
    fn duplicate_stage_tables_are_rejected() {
        let manifest = json!({"task_dataset": {"stages": {
            "train": {"table_name": "seeds"}, "validation": {"table_name": "seeds"}
        }}});
        assert!(matches!(metadata(manifest, Path::new("/extension")),
            Err(RuntimeError::Data(message)) if message.contains("duplicate table")));
    }

    #[test]
    fn extension_manifest_is_optional_but_present_metadata_is_validated() {
        let root = tempfile::tempdir().unwrap();
        assert!(read_manifest(root.path()).unwrap().is_none());
        let path = root.path().join("diskgraph_manifest.json");
        std::fs::write(&path, b"invalid json").unwrap();
        assert!(matches!(
            read_manifest(root.path()),
            Err(RuntimeError::Data(_))
        ));
        std::fs::write(&path, b"{\"task_dataset\":{\"stages\":{}}}").unwrap();
        assert!(read_manifest(root.path())
            .unwrap()
            .unwrap()
            .stages
            .is_empty());
        std::fs::write(
            &path,
            b"{\"task_dataset\":{\"stages\":{\"train\":{\"table_name\":4}}}}",
        )
        .unwrap();
        assert!(matches!(
            read_manifest(root.path()),
            Err(RuntimeError::Data(_))
        ));
    }
}
