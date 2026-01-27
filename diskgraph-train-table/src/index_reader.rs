use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use diskgraph_core::metadata::{DatasetMetadata, NodeShardMetadata};
use diskgraph_rowpack::RowPackContext;

pub struct IndexReader {
    root: PathBuf,
    pub metadata: DatasetMetadata,
    contexts: HashMap<String, RowPackContext>,
}

impl IndexReader {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let metadata_path = root.join("index_metadata.json");
        let bytes = std::fs::read(&metadata_path)
            .with_context(|| format!("read index metadata at {}", metadata_path.display()))?;
        let metadata: DatasetMetadata = serde_json::from_slice(&bytes)?;

        let mut contexts = HashMap::new();
        for node in &metadata.node_types {
            if let Some(node_data) = &node.node_data_compression {
                let ctx_path = root.join(&node_data.context_path);
                if ctx_path.exists() {
                    let ctx_bytes = std::fs::read(&ctx_path)?;
                    let ctx = RowPackContext::from_bytes(&ctx_bytes)?;
                    contexts.insert(node.name.clone(), ctx);
                }
            }
        }

        Ok(Self {
            root,
            metadata,
            contexts,
        })
    }

    pub fn node_context(&self, table: &str) -> Option<&RowPackContext> {
        self.contexts.get(table)
    }

    pub fn node_shards(&self, table: &str) -> Option<&[NodeShardMetadata]> {
        self.metadata
            .node_types
            .iter()
            .find(|n| n.name == table)
            .map(|n| n.shards.as_slice())
    }

    pub fn node_data_path(&self, table: &str, shard: usize) -> PathBuf {
        let table_lower = table.to_lowercase();
        self.root
            .join("nodes")
            .join(table_lower)
            .join(format!("shard-{shard:05}"))
            .join("node_data.bin")
    }

    pub fn iter_node_rows(&self, table: &str, shard: usize) -> Result<NodeRowIter> {
        let path = self.node_data_path(table, shard);
        let file = File::open(&path)
            .with_context(|| format!("open node data at {}", path.display()))?;
        Ok(NodeRowIter { file })
    }
}

pub struct NodeRowIter {
    file: File,
}

impl NodeRowIter {
    pub fn next_row(&mut self) -> Result<Option<Vec<u8>>> {
        let mut len_buf = [0u8; 4];
        match self.file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        if len == 0 {
            return Ok(Some(Vec::new()));
        }
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf)?;
        Ok(Some(buf))
    }

    pub fn reset(&mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskgraph_core::metadata::{DatasetMetadata, NodeDataCompression, NodeTypeMetadata};
    use diskgraph_rowpack::{
        ColumnContext, ColumnEncoding, IngestedType, RowPackContext, RowPackEncoder,
        RowPackScratch, ValueRef,
    };
    use std::io::Write;

    fn temp_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("diskgraph_train_table_index_{}", name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn build_context() -> RowPackContext {
        RowPackContext {
            version: 3,
            columns: vec![ColumnContext {
                name: "id".to_string(),
                logical_type: IngestedType::Int64,
                encoding: ColumnEncoding::Int {
                    signed: true,
                    varint: true,
                    bits: 64,
                },
            }],
            estimated_row_bytes: 8,
        }
    }

    fn write_rowpack_ctx(path: &Path, ctx: &RowPackContext) {
        let bytes = ctx.to_bytes().expect("ctx bytes");
        std::fs::create_dir_all(path.parent().unwrap()).expect("ctx dir");
        std::fs::write(path, bytes).expect("write ctx");
    }

    fn write_node_data(path: &Path, rows: Vec<Vec<u8>>) {
        std::fs::create_dir_all(path.parent().unwrap()).expect("node data dir");
        let mut file = File::create(path).expect("node data");
        for row in rows {
            let len = row.len() as u32;
            file.write_all(&len.to_le_bytes()).expect("write len");
            file.write_all(&row).expect("write row");
        }
    }

    #[test]
    fn read_node_rows() {
        let dir = temp_dir("rows");
        let ctx = build_context();
        let ctx_path = dir.join("nodes/users/node_data.ctx");
        write_rowpack_ctx(&ctx_path, &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "users".to_string(),
                max_id: 1,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/users/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 1,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        std::fs::write(dir.join("index_metadata.json"), serde_json::to_vec_pretty(&metadata).unwrap())
            .expect("write metadata");

        let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut row_buf = Vec::new();
        encoder
            .encode_row_into(&[ValueRef::Int64(1)], &mut row_buf, &mut scratch)
            .expect("encode row");

        let node_data_path = dir.join("nodes/users/shard-00000/node_data.bin");
        write_node_data(&node_data_path, vec![row_buf.clone()]);

        let reader = IndexReader::open(&dir).expect("reader");
        let mut iter = reader.iter_node_rows("users", 0).expect("iter");
        let row = iter.next_row().expect("read row").expect("row");
        assert_eq!(row, row_buf);
        assert!(iter.next_row().expect("read row").is_none());
    }
}
