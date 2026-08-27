//! TT-NG binary-label sidecars for replacement seed sampling.
//!
//! Labels are captured in node-write order, not source parquet order. For each
//! shard, the k-th spill record corresponds to final node id `shard.min_id + k`.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

pub const FALSE_NODE_IDS_FILENAME: &str = "binary_label_false_node_ids.bin";
pub const TRUE_NODE_IDS_FILENAME: &str = "binary_label_true_node_ids.bin";

pub const LABEL_FALSE: u8 = 0;
pub const LABEL_TRUE: u8 = 1;
pub const LABEL_INVALID: u8 = 2;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BinaryLabelIndex {
    pub false_node_ids: String,
    pub false_count: u64,
    pub true_node_ids: String,
    pub true_count: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct LabelCounts {
    pub false_count: u64,
    pub true_count: u64,
    pub invalid_count: u64,
}

impl LabelCounts {
    pub(crate) fn add(&mut self, label: u8) {
        match label {
            LABEL_FALSE => self.false_count = self.false_count.saturating_add(1),
            LABEL_TRUE => self.true_count = self.true_count.saturating_add(1),
            _ => self.invalid_count = self.invalid_count.saturating_add(1),
        }
    }

    fn valid_total(&self) -> u64 {
        self.false_count.saturating_add(self.true_count)
    }
}

pub(crate) struct ShardSpill {
    pub min_id: u64,
    pub counts: LabelCounts,
    pub spill_path: PathBuf,
}

pub(crate) fn spill_rel_path(table: &str, shard: usize) -> String {
    format!(
        "{}/binary_label/{}/shard-{:05}.labels",
        crate::consts::TEMP_DIR_NAME,
        table.to_lowercase(),
        shard
    )
}

pub(crate) fn append_spill(file: &mut std::fs::File, labels: &[u8]) -> Result<()> {
    file.write_all(labels).context("append binary-label spill")
}

fn write_node_id(out: &mut BufWriter<std::fs::File>, node_id: u64) -> Result<()> {
    out.write_all(&node_id.to_le_bytes())
        .context("write binary-label node id")
}

pub(crate) fn merge_stage(
    shards: &[ShardSpill],
    false_path: &Path,
    false_rel_path: String,
    true_path: &Path,
    true_rel_path: String,
) -> Result<Option<BinaryLabelIndex>> {
    let mut totals = LabelCounts::default();
    for shard in shards {
        totals.false_count = totals.false_count.saturating_add(shard.counts.false_count);
        totals.true_count = totals.true_count.saturating_add(shard.counts.true_count);
        totals.invalid_count = totals
            .invalid_count
            .saturating_add(shard.counts.invalid_count);
    }
    if totals.invalid_count > 0 || totals.valid_total() == 0 {
        for shard in shards {
            let _ = std::fs::remove_file(&shard.spill_path);
        }
        return Ok(None);
    }

    if let Some(parent) = false_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("mkdir for {}", false_path.display()))?;
    }
    if let Some(parent) = true_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("mkdir for {}", true_path.display()))?;
    }
    let mut false_out = BufWriter::new(
        std::fs::File::create(false_path)
            .with_context(|| format!("create {}", false_path.display()))?,
    );
    let mut true_out = BufWriter::new(
        std::fs::File::create(true_path)
            .with_context(|| format!("create {}", true_path.display()))?,
    );

    let mut observed = LabelCounts::default();
    let mut buf = [0u8; 64 * 1024];
    for shard in shards {
        let file = std::fs::File::open(&shard.spill_path)
            .with_context(|| format!("open binary-label spill {}", shard.spill_path.display()))?;
        let mut reader = BufReader::new(file);
        let mut local = 0u64;
        loop {
            let n = reader
                .read(&mut buf)
                .with_context(|| format!("read {}", shard.spill_path.display()))?;
            if n == 0 {
                break;
            }
            for label in &buf[..n] {
                let node_id = shard.min_id.saturating_add(local);
                match *label {
                    LABEL_FALSE => {
                        write_node_id(&mut false_out, node_id)?;
                        observed.false_count = observed.false_count.saturating_add(1);
                    }
                    LABEL_TRUE => {
                        write_node_id(&mut true_out, node_id)?;
                        observed.true_count = observed.true_count.saturating_add(1);
                    }
                    other => {
                        return Err(anyhow!(
                            "unexpected binary-label spill value {other} in {}",
                            shard.spill_path.display()
                        ));
                    }
                }
                local = local.saturating_add(1);
            }
        }
        if local != shard.counts.valid_total() {
            return Err(anyhow!(
                "binary-label spill {} has {local} records, expected {}",
                shard.spill_path.display(),
                shard.counts.valid_total()
            ));
        }
    }
    false_out
        .flush()
        .context("flush binary-label false node ids")?;
    true_out
        .flush()
        .context("flush binary-label true node ids")?;
    for shard in shards {
        let _ = std::fs::remove_file(&shard.spill_path);
    }

    Ok(Some(BinaryLabelIndex {
        false_node_ids: false_rel_path,
        false_count: observed.false_count,
        true_node_ids: true_rel_path,
        true_count: observed.true_count,
    }))
}

pub(crate) fn output_rel_paths(table: &str) -> (String, String) {
    (
        format!("{table}/{FALSE_NODE_IDS_FILENAME}"),
        format!("{table}/{TRUE_NODE_IDS_FILENAME}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u64s(path: &Path) -> Vec<u64> {
        std::fs::read(path)
            .unwrap()
            .as_chunks::<8>().0.iter()
            .map(|chunk| {
                let mut raw = [0u8; 8];
                raw.copy_from_slice(chunk);
                u64::from_le_bytes(raw)
            })
            .collect()
    }

    #[test]
    fn merge_stage_maps_labels_from_shard_write_order_to_final_node_ids() {
        let dir = tempfile::tempdir().unwrap();
        let s0 = dir.path().join("s0.labels");
        let s1 = dir.path().join("s1.labels");
        std::fs::write(&s0, [LABEL_TRUE, LABEL_FALSE, LABEL_TRUE]).unwrap();
        std::fs::write(&s1, [LABEL_FALSE, LABEL_TRUE]).unwrap();
        let false_path = dir.path().join(FALSE_NODE_IDS_FILENAME);
        let true_path = dir.path().join(TRUE_NODE_IDS_FILENAME);
        let index = merge_stage(
            &[
                ShardSpill {
                    min_id: 10,
                    counts: LabelCounts {
                        false_count: 1,
                        true_count: 2,
                        invalid_count: 0,
                    },
                    spill_path: s0,
                },
                ShardSpill {
                    min_id: 20,
                    counts: LabelCounts {
                        false_count: 1,
                        true_count: 1,
                        invalid_count: 0,
                    },
                    spill_path: s1,
                },
            ],
            &false_path,
            FALSE_NODE_IDS_FILENAME.to_string(),
            &true_path,
            TRUE_NODE_IDS_FILENAME.to_string(),
        )
        .unwrap()
        .unwrap();

        assert_eq!(index.false_count, 2);
        assert_eq!(index.true_count, 3);
        assert_eq!(read_u64s(&false_path), vec![11, 20]);
        assert_eq!(read_u64s(&true_path), vec![10, 12, 21]);
    }

    #[test]
    fn merge_stage_omits_index_when_any_label_is_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let spill = dir.path().join("s0.labels");
        std::fs::write(&spill, [LABEL_TRUE, LABEL_INVALID]).unwrap();
        let result = merge_stage(
            &[ShardSpill {
                min_id: 0,
                counts: LabelCounts {
                    false_count: 0,
                    true_count: 1,
                    invalid_count: 1,
                },
                spill_path: spill,
            }],
            &dir.path().join(FALSE_NODE_IDS_FILENAME),
            FALSE_NODE_IDS_FILENAME.to_string(),
            &dir.path().join(TRUE_NODE_IDS_FILENAME),
            TRUE_NODE_IDS_FILENAME.to_string(),
        )
        .unwrap();

        assert!(result.is_none());
    }
}
