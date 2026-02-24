use anyhow::{anyhow, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize)]
pub struct ExportConfig {
    pub train_table_path: String,
    pub index_path: String,
    pub output_dir: String,
    #[serde(default = "default_row_idx_col")]
    pub row_idx_col: String,
    #[serde(default = "default_train_table_name")]
    pub train_table_name: String,
    pub fkeys: Vec<FkeyConfig>,
}

fn default_row_idx_col() -> String {
    "arange".to_string()
}

fn default_train_table_name() -> String {
    "custom_train_table".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FkeyConfig {
    pub column_name: String,
    pub target_table: String,
}

pub trait IdResolver {
    fn resolve(&self, fkey_idx: usize, key_str: &str) -> Result<Option<u64>>;
}

pub struct IndexResolver {
    pub tables: Vec<Arc<TableReader>>,
}

impl IdResolver for IndexResolver {
    fn resolve(&self, fkey_idx: usize, key_str: &str) -> Result<Option<u64>> {
        let table = &self.tables[fkey_idx];
        let key = table.parse_pk_key(key_str)?;
        table.node_id_from_pk(&key)
    }
}

fn write_parquet(path: &Path, batch: &RecordBatch) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

pub fn process_export_internal<R: IdResolver>(config: &ExportConfig, resolver: &R) -> Result<()> {
    let input_file = File::open(&config.train_table_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file)?;
    let mut reader = builder.build()?;

    let output_dir = Path::new(&config.output_dir);
    std::fs::create_dir_all(output_dir)?;

    let mut batch_index = 0;

    while let Some(maybe_batch) = reader.next() {
        let batch = maybe_batch?;
        let num_rows = batch.num_rows();

        let arange_col = batch
            .column_by_name(&config.row_idx_col)
            .ok_or_else(|| anyhow!("Column {} not found", config.row_idx_col))?;
        let arange_array = arange_col.as_primitive::<Int64Type>();

        let mut mask_builder = BooleanArray::builder(num_rows);
        let mut fkey_arrays = Vec::new();
        let mut fkey_fields = Vec::new();

        let mut resolved_matrix: Vec<Vec<i64>> = vec![Vec::with_capacity(num_rows); config.fkeys.len()];

        for (i, fkey_conf) in config.fkeys.iter().enumerate() {
            let col = batch
                .column_by_name(&fkey_conf.column_name)
                .ok_or_else(|| anyhow!("Column {} not found", fkey_conf.column_name))?;

            let str_array = col.as_string::<i32>();

            for row in 0..num_rows {
                if str_array.is_null(row) {
                    resolved_matrix[i].push(-1);
                } else {
                    let key_str = str_array.value(row);
                    if let Some(local_id) = resolver.resolve(i, key_str)? {
                        resolved_matrix[i].push(local_id as i64);
                    } else {
                        resolved_matrix[i].push(-1);
                    }
                }
            }

            let array = Int64Array::from(resolved_matrix[i].clone());
            fkey_arrays.push(Arc::new(array) as Arc<dyn Array>);
            fkey_fields.push(Field::new(&fkey_conf.column_name, DataType::Int64, true));
        }

        for row in 0..num_rows {
            let mut valid = true;
            for i in 0..config.fkeys.len() {
                if resolved_matrix[i][row] == -1 {
                    valid = false;
                    break;
                }
            }
            mask_builder.append_value(valid);
        }

        let mask_array = Arc::new(mask_builder.finish()) as Arc<dyn Array>;

        let fkey_schema = Arc::new(Schema::new(fkey_fields));
        let fkey_batch = RecordBatch::try_new(fkey_schema, fkey_arrays)?;
        let fkey_dir = output_dir.join(format!("{}_fkey_cols.parquet", config.train_table_name));
        write_parquet(&fkey_dir.join(format!("part-{:04}.parquet", batch_index)), &fkey_batch)?;

        let mask_schema = Arc::new(Schema::new(vec![Field::new("__mask__", DataType::Boolean, false)]));
        let mask_batch = RecordBatch::try_new(mask_schema, vec![mask_array])?;
        let mask_dir = output_dir.join(format!("{}_mask.parquet", config.train_table_name));
        write_parquet(&mask_dir.join(format!("part-{:04}.parquet", batch_index)), &mask_batch)?;

        for (i, fkey_conf) in config.fkeys.iter().enumerate() {
            let mut rows = Vec::new();
            let mut cols = Vec::new();

            for row in 0..num_rows {
                let target_id = resolved_matrix[i][row];
                if target_id != -1 {
                    let arange_val = arange_array.value(row);
                    rows.push(target_id);
                    cols.push(arange_val);
                }
            }

            let edge_schema = Arc::new(Schema::new(vec![
                Field::new("row", DataType::Int64, false),
                Field::new("col", DataType::Int64, false),
            ]));

            let edge_batch = RecordBatch::try_new(
                edge_schema,
                vec![
                    Arc::new(Int64Array::from(rows)) as Arc<dyn Array>,
                    Arc::new(Int64Array::from(cols)) as Arc<dyn Array>,
                ],
            )?;

            let edge_dir = output_dir
                .join(format!("{}__rev_{}__{}", fkey_conf.target_table, fkey_conf.column_name, config.train_table_name))
                .join("edge.parquet");
            write_parquet(&edge_dir.join(format!("part-{:04}.parquet", batch_index)), &edge_batch)?;
        }

        batch_index += 1;
    }

    Ok(())
}

pub fn process_export(config: &ExportConfig) -> Result<()> {
    let index_reader = GraphIndexReader::open(
        &Path::new(&config.index_path).join("index_metadata.json"),
        Path::new(&config.index_path),
    )?;

    let mut tables = Vec::new();
    for fkey in &config.fkeys {
        let table = index_reader
            .table(&fkey.target_table)
            .ok_or_else(|| anyhow!("Target table {} not found in index", fkey.target_table))?;
        tables.push(table);
    }

    let resolver = IndexResolver { tables };
    process_export_internal(config, &resolver)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::{Int64Builder, StringBuilder};
    use tempfile::TempDir;
    use std::collections::HashMap;

    struct MockResolver {
        // [fkey_index][key_str] -> local_id
        maps: Vec<HashMap<String, u64>>,
    }

    impl IdResolver for MockResolver {
        fn resolve(&self, fkey_idx: usize, key_str: &str) -> Result<Option<u64>> {
            Ok(self.maps[fkey_idx].get(key_str).cloned())
        }
    }

    fn write_test_parquet(path: &Path) {
        let mut arange = Int64Builder::new();
        let mut entity = StringBuilder::new();
        let mut target = StringBuilder::new();

        arange.append_value(100);
        entity.append_value("user_1");
        target.append_value("prod_1");

        arange.append_value(101);
        entity.append_value("user_2");
        target.append_value("prod_999");

        arange.append_value(102);
        entity.append_value("user_1");
        target.append_null();

        let schema = Arc::new(Schema::new(vec![
            Field::new("arange", DataType::Int64, false),
            Field::new("ENTITY", DataType::Utf8, true),
            Field::new("TARGET", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arange.finish()) as Arc<dyn Array>,
                Arc::new(entity.finish()) as Arc<dyn Array>,
                Arc::new(target.finish()) as Arc<dyn Array>,
            ],
        ).unwrap();

        write_parquet(path, &batch).unwrap();
    }

    #[test]
    fn test_export_tt_artifacts_mock() {
        let dir = TempDir::new().unwrap();
        let tt_path = dir.path().join("train_table.parquet");
        write_test_parquet(&tt_path);

        let output_dir = dir.path().join("output");

        let config = ExportConfig {
            train_table_path: tt_path.to_str().unwrap().to_string(),
            index_path: "dummy".to_string(),
            output_dir: output_dir.to_str().unwrap().to_string(),
            row_idx_col: "arange".to_string(),
            train_table_name: "custom_train_table".to_string(),
            fkeys: vec![
                FkeyConfig { column_name: "ENTITY".to_string(), target_table: "users".to_string() },
                FkeyConfig { column_name: "TARGET".to_string(), target_table: "products".to_string() }
            ]
        };

        let mut user_map = HashMap::new();
        user_map.insert("user_1".to_string(), 0);
        user_map.insert("user_2".to_string(), 1);

        let mut prod_map = HashMap::new();
        prod_map.insert("prod_1".to_string(), 0);

        let resolver = MockResolver {
            maps: vec![user_map, prod_map],
        };

        process_export_internal(&config, &resolver).unwrap();

        assert!(output_dir.join("custom_train_table_fkey_cols.parquet").exists());
        assert!(output_dir.join("custom_train_table_mask.parquet").exists());
        assert!(output_dir.join("users__rev_ENTITY__custom_train_table").join("edge.parquet").exists());
        assert!(output_dir.join("products__rev_TARGET__custom_train_table").join("edge.parquet").exists());

        let mask_file = File::open(output_dir.join("custom_train_table_mask.parquet/part-0000.parquet")).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(mask_file).unwrap().build().unwrap();
        let batch = reader.next().unwrap().unwrap();
        let mask = batch.column(0).as_boolean();
        assert_eq!(mask.len(), 3);
        assert_eq!(mask.value(0), true);  // user_1, prod_1 (both valid)
        assert_eq!(mask.value(1), false); // user_2, prod_999 (invalid target)
        assert_eq!(mask.value(2), false); // user_1, null (invalid target)

        let edge_file = File::open(output_dir.join("products__rev_TARGET__custom_train_table/edge.parquet/part-0000.parquet")).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(edge_file).unwrap().build().unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
        let row_col = batch.column(0).as_primitive::<Int64Type>(); // target_id
        let col_col = batch.column(1).as_primitive::<Int64Type>(); // arange
        assert_eq!(row_col.value(0), 0); // prod_1 -> local_id 0
        assert_eq!(col_col.value(0), 100); // arange -> 100
    }

    #[test]
    fn test_export_tt_artifacts_e2e_real_index() {
        use std::process::Command;

        let dir = TempDir::new().unwrap();

        // 1. Create raw source data for ingestion
        let source_dir = dir.path().join("source");
        std::fs::create_dir_all(&source_dir).unwrap();

        // Users source
        let u_schema = Arc::new(Schema::new(vec![Field::new("user_id", DataType::Utf8, false)]));
        let mut u_builder = StringBuilder::new();
        u_builder.append_value("user_1");
        u_builder.append_value("user_2");
        let u_batch = RecordBatch::try_new(u_schema, vec![Arc::new(u_builder.finish()) as Arc<dyn Array>]).unwrap();
        write_parquet(&source_dir.join("users.parquet"), &u_batch).unwrap();

        // Products source
        let p_schema = Arc::new(Schema::new(vec![Field::new("product_id", DataType::Utf8, false)]));
        let mut p_builder = StringBuilder::new();
        p_builder.append_value("prod_1");
        p_builder.append_value("prod_2");
        let p_batch = RecordBatch::try_new(p_schema, vec![Arc::new(p_builder.finish()) as Arc<dyn Array>]).unwrap();
        write_parquet(&source_dir.join("products.parquet"), &p_batch).unwrap();

        // 2. Write Kumo Config
        let config_json = serde_json::json!({
            "dataset_name": "test_e2e",
            "num_shards": 1,
            "tables": [
                {
                    "name": "users",
                    "columns": [{ "name": "user_id", "ingested_dtype": "string" }],
                    "primary_key": "user_id",
                    "source": { "type": "parquet", "paths": [source_dir.join("users.parquet").to_str().unwrap()] }
                },
                {
                    "name": "products",
                    "columns": [{ "name": "product_id", "ingested_dtype": "string" }],
                    "primary_key": "product_id",
                    "source": { "type": "parquet", "paths": [source_dir.join("products.parquet").to_str().unwrap()] }
                }
            ]
        });
        let config_path = dir.path().join("config.json");
        std::fs::write(&config_path, serde_json::to_string(&config_json).unwrap()).unwrap();

        // 3. Build Index using ingest binary
        let index_dir = dir.path().join("compiled_index");

        // Find cargo project root
        let mut current_dir = std::env::current_dir().unwrap();
        while !current_dir.join("Cargo.toml").exists() {
            if !current_dir.pop() {
                panic!("Could not find workspace root");
            }
        }

        let status = Command::new("cargo")
            .args(["run", "--release", "-p", "diskgraph-ingest", "--bin", "ingest", "--", "--source", config_path.to_str().unwrap(), "--dest", "local", "--output", index_dir.to_str().unwrap()])
            .current_dir(current_dir)
            .status()
            .expect("failed to execute ingest binary");

        assert!(status.success(), "ingestion failed");

        // 4. Create Mock Train Table
        let tt_path = dir.path().join("train_table.parquet");
        write_test_parquet(&tt_path);

        // 5. Run real export utility
        let output_dir = dir.path().join("output");
        let config = ExportConfig {
            train_table_path: tt_path.to_str().unwrap().to_string(),
            index_path: index_dir.to_str().unwrap().to_string(),
            output_dir: output_dir.to_str().unwrap().to_string(),
            row_idx_col: "arange".to_string(),
            train_table_name: "custom_train_table".to_string(),
            fkeys: vec![
                FkeyConfig { column_name: "ENTITY".to_string(), target_table: "users".to_string() },
                FkeyConfig { column_name: "TARGET".to_string(), target_table: "products".to_string() }
            ]
        };

        // Use the real process_export which loads the GraphIndexReader
        process_export(&config).unwrap();

        // 6. Verify Outputs
        let mask_file = File::open(output_dir.join("custom_train_table_mask.parquet/part-0000.parquet")).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(mask_file).unwrap().build().unwrap();
        let batch = reader.next().unwrap().unwrap();
        let mask = batch.column(0).as_boolean();
        assert_eq!(mask.len(), 3);
        assert_eq!(mask.value(0), true);  // user_1, prod_1 (both valid)
        assert_eq!(mask.value(1), false); // user_2, prod_999 (invalid target)
        assert_eq!(mask.value(2), false); // user_1, null (invalid target)

        let edge_file = File::open(output_dir.join("products__rev_TARGET__custom_train_table/edge.parquet/part-0000.parquet")).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(edge_file).unwrap().build().unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
        let row_col = batch.column(0).as_primitive::<Int64Type>(); // target_id
        let col_col = batch.column(1).as_primitive::<Int64Type>(); // arange

        // prod_1 should be local_id 0 based on our source parquet
        assert_eq!(row_col.value(0), 0);
        assert_eq!(col_col.value(0), 100);
    }
}
