//! TT-NG row-weight sidecars for replacement seed sampling.
//!
//! We spill one f64 per surviving train-stage node in node-write order. After
//! final shard min ids are known, the spill is compacted into mmap-friendly
//! node-id + cumulative-weight sidecars. Optional binary-label spills let the
//! same pass also publish false/true class buckets.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

pub const NODE_IDS_FILENAME: &str = "sample_weight_node_ids.bin";
pub const CDF_FILENAME: &str = "sample_weight_cdf_f64.bin";
pub const FALSE_NODE_IDS_FILENAME: &str = "sample_weight_false_node_ids.bin";
pub const FALSE_CDF_FILENAME: &str = "sample_weight_false_cdf_f64.bin";
pub const TRUE_NODE_IDS_FILENAME: &str = "sample_weight_true_node_ids.bin";
pub const TRUE_CDF_FILENAME: &str = "sample_weight_true_cdf_f64.bin";

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WeightedBucketIndex {
    pub node_ids: String,
    pub cdf: String,
    pub count: u64,
    pub total_weight: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SampleWeightIndex {
    pub all: WeightedBucketIndex,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub false_bucket: Option<WeightedBucketIndex>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub true_bucket: Option<WeightedBucketIndex>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct WeightCounts {
    pub row_count: u64,
    pub valid_count: u64,
    pub invalid_count: u64,
    pub total_weight: f64,
}

impl WeightCounts {
    pub(crate) fn add(&mut self, weight: f64) {
        self.row_count = self.row_count.saturating_add(1);
        if weight.is_finite() && weight >= 0.0 {
            self.valid_count = self.valid_count.saturating_add(1);
            self.total_weight += weight;
        } else {
            self.invalid_count = self.invalid_count.saturating_add(1);
        }
    }
}

pub(crate) struct ShardSpill {
    pub min_id: u64,
    pub counts: WeightCounts,
    pub spill_path: PathBuf,
    pub label_spill_path: Option<PathBuf>,
    pub label_counts: Option<crate::binary_label::LabelCounts>,
}

pub(crate) fn spill_rel_path(table: &str, shard: usize) -> String {
    format!(
        "{}/sample_weight/{}/shard-{:05}.weights",
        crate::consts::TEMP_DIR_NAME,
        table.to_lowercase(),
        shard
    )
}

pub(crate) fn append_spill(file: &mut std::fs::File, weights: &[f64]) -> Result<()> {
    for weight in weights {
        file.write_all(&weight.to_le_bytes())
            .context("append sample-weight spill")?;
    }
    Ok(())
}

fn write_u64(out: &mut BufWriter<std::fs::File>, value: u64) -> Result<()> {
    out.write_all(&value.to_le_bytes())
        .context("write sample-weight node id")
}

fn write_f64(out: &mut BufWriter<std::fs::File>, value: f64) -> Result<()> {
    out.write_all(&value.to_le_bytes())
        .context("write sample-weight cdf")
}

struct BucketWriter {
    node_out: BufWriter<std::fs::File>,
    cdf_out: BufWriter<std::fs::File>,
    node_rel: String,
    cdf_rel: String,
    count: u64,
    total_weight: f64,
}

impl BucketWriter {
    fn new(node_path: &Path, node_rel: String, cdf_path: &Path, cdf_rel: String) -> Result<Self> {
        if let Some(parent) = node_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir for {}", node_path.display()))?;
        }
        if let Some(parent) = cdf_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir for {}", cdf_path.display()))?;
        }
        Ok(Self {
            node_out: BufWriter::new(
                std::fs::File::create(node_path)
                    .with_context(|| format!("create {}", node_path.display()))?,
            ),
            cdf_out: BufWriter::new(
                std::fs::File::create(cdf_path)
                    .with_context(|| format!("create {}", cdf_path.display()))?,
            ),
            node_rel,
            cdf_rel,
            count: 0,
            total_weight: 0.0,
        })
    }

    fn push(&mut self, node_id: u64, weight: f64) -> Result<()> {
        self.total_weight += weight;
        self.count = self.count.saturating_add(1);
        write_u64(&mut self.node_out, node_id)?;
        write_f64(&mut self.cdf_out, self.total_weight)
    }

    fn finish(mut self) -> Result<WeightedBucketIndex> {
        self.node_out
            .flush()
            .context("flush sample-weight node ids")?;
        self.cdf_out.flush().context("flush sample-weight cdf")?;
        Ok(WeightedBucketIndex {
            node_ids: self.node_rel,
            cdf: self.cdf_rel,
            count: self.count,
            total_weight: self.total_weight,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn merge_stage(
    shards: &[ShardSpill],
    all_node_path: &Path,
    all_node_rel: String,
    all_cdf_path: &Path,
    all_cdf_rel: String,
    false_node_path: &Path,
    false_node_rel: String,
    false_cdf_path: &Path,
    false_cdf_rel: String,
    true_node_path: &Path,
    true_node_rel: String,
    true_cdf_path: &Path,
    true_cdf_rel: String,
) -> Result<Option<SampleWeightIndex>> {
    let mut totals = WeightCounts::default();
    let mut labels_valid = !shards.is_empty();
    for shard in shards {
        totals.row_count = totals.row_count.saturating_add(shard.counts.row_count);
        totals.valid_count = totals.valid_count.saturating_add(shard.counts.valid_count);
        totals.invalid_count = totals
            .invalid_count
            .saturating_add(shard.counts.invalid_count);
        totals.total_weight += shard.counts.total_weight;
        labels_valid &= match &shard.label_counts {
            Some(c) => {
                c.invalid_count == 0
                    && c.false_count.saturating_add(c.true_count) == shard.counts.row_count
                    && shard.label_spill_path.as_ref().is_some_and(|p| p.exists())
            }
            None => false,
        };
    }
    if totals.invalid_count > 0 || totals.valid_count == 0 || totals.total_weight <= 0.0 {
        for shard in shards {
            let _ = std::fs::remove_file(&shard.spill_path);
        }
        return Ok(None);
    }

    let mut all = BucketWriter::new(all_node_path, all_node_rel, all_cdf_path, all_cdf_rel)?;
    let mut false_bucket = if labels_valid {
        Some(BucketWriter::new(
            false_node_path,
            false_node_rel,
            false_cdf_path,
            false_cdf_rel,
        )?)
    } else {
        None
    };
    let mut true_bucket = if labels_valid {
        Some(BucketWriter::new(
            true_node_path,
            true_node_rel,
            true_cdf_path,
            true_cdf_rel,
        )?)
    } else {
        None
    };

    let mut weight_raw = [0u8; 8];
    let mut label_raw = [0u8; 1];
    for shard in shards {
        let file = std::fs::File::open(&shard.spill_path)
            .with_context(|| format!("open sample-weight spill {}", shard.spill_path.display()))?;
        let mut weights = BufReader::new(file);
        let mut labels = match &shard.label_spill_path {
            Some(path) if labels_valid => Some(BufReader::new(
                std::fs::File::open(path)
                    .with_context(|| format!("open binary-label spill {}", path.display()))?,
            )),
            _ => None,
        };
        for local in 0..shard.counts.row_count {
            weights
                .read_exact(&mut weight_raw)
                .with_context(|| format!("read {}", shard.spill_path.display()))?;
            let weight = f64::from_le_bytes(weight_raw);
            if !weight.is_finite() || weight < 0.0 {
                return Err(anyhow!(
                    "invalid sample weight {weight} in {}",
                    shard.spill_path.display()
                ));
            }
            let node_id = shard.min_id.saturating_add(local);
            all.push(node_id, weight)?;
            if let Some(label_reader) = labels.as_mut() {
                label_reader
                    .read_exact(&mut label_raw)
                    .with_context(|| "read binary-label spill for sample weights")?;
                match label_raw[0] {
                    crate::binary_label::LABEL_FALSE => {
                        if let Some(bucket) = false_bucket.as_mut() {
                            bucket.push(node_id, weight)?;
                        }
                    }
                    crate::binary_label::LABEL_TRUE => {
                        if let Some(bucket) = true_bucket.as_mut() {
                            bucket.push(node_id, weight)?;
                        }
                    }
                    other => {
                        return Err(anyhow!("unexpected binary-label spill value {other}"));
                    }
                }
            }
        }
        let mut extra = [0u8; 1];
        if weights.read(&mut extra)? != 0 {
            return Err(anyhow!(
                "sample-weight spill {} has more than {} records",
                shard.spill_path.display(),
                shard.counts.row_count
            ));
        }
    }

    let all = all.finish()?;
    let false_bucket = match false_bucket {
        Some(bucket) if bucket.total_weight > 0.0 => Some(bucket.finish()?),
        Some(bucket) => {
            let _ = bucket.finish()?;
            None
        }
        None => None,
    };
    let true_bucket = match true_bucket {
        Some(bucket) if bucket.total_weight > 0.0 => Some(bucket.finish()?),
        Some(bucket) => {
            let _ = bucket.finish()?;
            None
        }
        None => None,
    };
    for shard in shards {
        let _ = std::fs::remove_file(&shard.spill_path);
    }
    Ok(Some(SampleWeightIndex {
        all,
        false_bucket,
        true_bucket,
    }))
}

pub(crate) fn output_rel_paths(table: &str) -> [(String, String); 3] {
    [
        (
            format!("{table}/{NODE_IDS_FILENAME}"),
            format!("{table}/{CDF_FILENAME}"),
        ),
        (
            format!("{table}/{FALSE_NODE_IDS_FILENAME}"),
            format!("{table}/{FALSE_CDF_FILENAME}"),
        ),
        (
            format!("{table}/{TRUE_NODE_IDS_FILENAME}"),
            format!("{table}/{TRUE_CDF_FILENAME}"),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_label::{LabelCounts, LABEL_FALSE, LABEL_TRUE};

    fn read_u64s(path: &Path) -> Vec<u64> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(8)
            .map(|chunk| {
                let mut raw = [0u8; 8];
                raw.copy_from_slice(chunk);
                u64::from_le_bytes(raw)
            })
            .collect()
    }

    fn read_f64s(path: &Path) -> Vec<f64> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(8)
            .map(|chunk| {
                let mut raw = [0u8; 8];
                raw.copy_from_slice(chunk);
                f64::from_le_bytes(raw)
            })
            .collect()
    }

    #[test]
    fn merge_stage_writes_all_and_class_cdfs_by_final_node_id() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("s0.weights");
        let labels = dir.path().join("s0.labels");
        append_spill(
            &mut std::fs::File::create(&weights).unwrap(),
            &[0.5, 2.0, 4.0],
        )
        .unwrap();
        std::fs::write(&labels, [LABEL_TRUE, LABEL_FALSE, LABEL_TRUE]).unwrap();

        let [(all_ids, all_cdf), (false_ids, false_cdf), (true_ids, true_cdf)] =
            output_rel_paths("train");
        let index = merge_stage(
            &[ShardSpill {
                min_id: 10,
                counts: WeightCounts {
                    row_count: 3,
                    valid_count: 3,
                    invalid_count: 0,
                    total_weight: 6.5,
                },
                spill_path: weights,
                label_spill_path: Some(labels),
                label_counts: Some(LabelCounts {
                    false_count: 1,
                    true_count: 2,
                    invalid_count: 0,
                }),
            }],
            &dir.path().join(&all_ids),
            all_ids.clone(),
            &dir.path().join(&all_cdf),
            all_cdf.clone(),
            &dir.path().join(&false_ids),
            false_ids.clone(),
            &dir.path().join(&false_cdf),
            false_cdf.clone(),
            &dir.path().join(&true_ids),
            true_ids.clone(),
            &dir.path().join(&true_cdf),
            true_cdf.clone(),
        )
        .unwrap()
        .unwrap();

        assert_eq!(index.all.count, 3);
        assert_eq!(index.all.total_weight, 6.5);
        assert_eq!(
            read_u64s(&dir.path().join(index.all.node_ids)),
            vec![10, 11, 12]
        );
        assert_eq!(
            read_f64s(&dir.path().join(index.all.cdf)),
            vec![0.5, 2.5, 6.5]
        );
        let false_bucket = index.false_bucket.unwrap();
        assert_eq!(read_u64s(&dir.path().join(false_bucket.node_ids)), vec![11]);
        assert_eq!(read_f64s(&dir.path().join(false_bucket.cdf)), vec![2.0]);
        let true_bucket = index.true_bucket.unwrap();
        assert_eq!(
            read_u64s(&dir.path().join(true_bucket.node_ids)),
            vec![10, 12]
        );
        assert_eq!(read_f64s(&dir.path().join(true_bucket.cdf)), vec![0.5, 4.5]);
    }
}
