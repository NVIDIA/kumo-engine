use anyhow::{anyhow, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type, UInt32Type, UInt64Type};
use arrow_array::{Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use diskgraph_index::pkmap::{Key, KeyType};
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
    /// Resolve a native Key to a node ID.
    fn resolve(&self, fkey_idx: usize, key: &Key) -> Result<Option<u64>>;
}

pub struct IndexResolver {
    pub tables: Vec<Arc<TableReader>>,
}

impl IdResolver for IndexResolver {
    fn resolve(&self, fkey_idx: usize, key: &Key) -> Result<Option<u64>> {
        self.tables[fkey_idx].node_id_from_pk(key)
    }
}

/// Read a native Key from an Arrow column at the given row.
///
/// Dispatches on the target `KeyType` (from the index) and the column's Arrow
/// dtype to construct the correct `Key` variant without string round-tripping.
/// For Int64 keys with integer columns, this avoids allocation entirely.
fn read_key_from_column(col: &dyn Array, row: usize, key_type: KeyType) -> Option<Key> {
    if col.is_null(row) {
        return None;
    }
    match key_type {
        KeyType::Int64 => {
            let val: u64 = match col.data_type() {
                DataType::Int64 => col.as_primitive::<Int64Type>().value(row) as u64,
                DataType::Int32 => col.as_primitive::<Int32Type>().value(row) as u64,
                DataType::UInt64 => col.as_primitive::<UInt64Type>().value(row),
                DataType::UInt32 => col.as_primitive::<UInt32Type>().value(row) as u64,
                DataType::Utf8 => col.as_string::<i32>().value(row).parse::<u64>().ok()?,
                DataType::LargeUtf8 => col.as_string::<i64>().value(row).parse::<u64>().ok()?,
                _ => return None,
            };
            Some(Key::Int(val))
        }
        KeyType::VarBytes => {
            let bytes = match col.data_type() {
                DataType::Utf8 => col.as_string::<i32>().value(row).as_bytes().to_vec(),
                DataType::LargeUtf8 => col.as_string::<i64>().value(row).as_bytes().to_vec(),
                DataType::Binary => col.as_binary::<i32>().value(row).to_vec(),
                DataType::LargeBinary => col.as_binary::<i64>().value(row).to_vec(),
                _ => return None,
            };
            Some(Key::Bytes(bytes))
        }
        KeyType::Uuid => {
            let s = match col.data_type() {
                DataType::Utf8 => col.as_string::<i32>().value(row),
                DataType::LargeUtf8 => col.as_string::<i64>().value(row),
                _ => return None,
            };
            use uuid_simd::{parse_hyphenated, parse_simple, AsOut};
            let raw = s.as_bytes();
            let mut out = [0u8; 16];
            let parsed = match raw.len() {
                32 => parse_simple(raw, out.as_out()).ok(),
                36 => parse_hyphenated(raw, out.as_out()).ok(),
                _ => None,
            }?;
            Some(Key::Uuid(*parsed))
        }
        KeyType::Uuid256 => {
            let s = match col.data_type() {
                DataType::Utf8 => col.as_string::<i32>().value(row),
                DataType::LargeUtf8 => col.as_string::<i64>().value(row),
                _ => return None,
            };
            let bytes = parse_hex_bytes(s)?;
            if bytes.len() != 32 {
                return None;
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            Some(Key::Uuid256(arr))
        }
        KeyType::Ip => {
            let s = match col.data_type() {
                DataType::Utf8 => col.as_string::<i32>().value(row),
                DataType::LargeUtf8 => col.as_string::<i64>().value(row),
                _ => return None,
            };
            use std::net::IpAddr;
            let ip: IpAddr = s.parse().ok()?;
            let bytes = match ip {
                IpAddr::V4(v4) => {
                    let mut b = [0u8; 16];
                    b[12..].copy_from_slice(&v4.octets());
                    b
                }
                IpAddr::V6(v6) => v6.octets(),
            };
            Some(Key::Ip(bytes))
        }
    }
}

/// Simple hex string decoder.
fn parse_hex_bytes(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
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

pub fn process_export_internal<R: IdResolver>(
    config: &ExportConfig,
    resolver: &R,
    key_types: &[KeyType],
) -> Result<()> {
    let train_table_path = Path::new(&config.train_table_path);
    let mut input_files = Vec::new();

    if train_table_path.is_dir() {
        for entry in std::fs::read_dir(train_table_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file()
                && path
                    .extension()
                    .map(|ext| ext == "parquet")
                    .unwrap_or(false)
            {
                input_files.push(path);
            }
        }
        // Sort files to ensure deterministic batch_index if needed
        input_files.sort();
    } else {
        input_files.push(train_table_path.to_path_buf());
    }

    let output_dir = Path::new(&config.output_dir);
    std::fs::create_dir_all(output_dir)?;

    let mut batch_index = 0;

    for file_path in input_files {
        let input_file = File::open(file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(input_file)?;
        let mut reader = builder.build()?;

        while let Some(maybe_batch) = reader.next() {
            let batch = maybe_batch?;
            let num_rows = batch.num_rows();

            let arange_col = batch
                .column_by_name(&config.row_idx_col)
                .ok_or_else(|| anyhow!("Column {} not found", config.row_idx_col))?;
            let arange_array = arange_col.as_primitive::<Int64Type>();

            let mut mask_values: Vec<i32> = Vec::with_capacity(num_rows);
            let mut fkey_arrays = Vec::new();
            let mut fkey_fields = Vec::new();

            let mut resolved_matrix: Vec<Vec<i64>> =
                vec![Vec::with_capacity(num_rows); config.fkeys.len()];

            for (i, fkey_conf) in config.fkeys.iter().enumerate() {
                let col = batch
                    .column_by_name(&fkey_conf.column_name)
                    .ok_or_else(|| anyhow!("Column {} not found", fkey_conf.column_name))?;

                let kt = key_types[i];

                for row in 0..num_rows {
                    match read_key_from_column(col.as_ref(), row, kt) {
                        None => resolved_matrix[i].push(-1),
                        Some(key) => match resolver.resolve(i, &key)? {
                            Some(local_id) => resolved_matrix[i].push(local_id as i64),
                            None => resolved_matrix[i].push(-1),
                        },
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
                mask_values.push(if valid { 1 } else { 0 });
            }

            let mask_array = Arc::new(Int32Array::from(mask_values)) as Arc<dyn Array>;

            let fkey_schema = Arc::new(Schema::new(fkey_fields));
            let fkey_batch = RecordBatch::try_new(fkey_schema, fkey_arrays)?;
            let fkey_dir =
                output_dir.join(format!("{}_fkey_cols.parquet", config.train_table_name));
            write_parquet(
                &fkey_dir.join(format!("part-{:04}.parquet", batch_index)),
                &fkey_batch,
            )?;

            let mask_schema = Arc::new(Schema::new(vec![Field::new(
                "__mask__",
                DataType::Int32,
                false,
            )]));
            let mask_batch = RecordBatch::try_new(mask_schema, vec![mask_array])?;
            let mask_dir =
                output_dir.join(format!("{}_mask.parquet", config.train_table_name));
            write_parquet(
                &mask_dir.join(format!("part-{:04}.parquet", batch_index)),
                &mask_batch,
            )?;

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
                    .join(format!(
                        "{}__rev_{}__{}",
                        fkey_conf.target_table, fkey_conf.column_name, config.train_table_name
                    ))
                    .join("edge.parquet");
                write_parquet(
                    &edge_dir.join(format!("part-{:04}.parquet", batch_index)),
                    &edge_batch,
                )?;
            }

            batch_index += 1;
        }
    }

    Ok(())
}

pub fn process_export(config: &ExportConfig) -> Result<()> {
    let index_reader = GraphIndexReader::open(
        &Path::new(&config.index_path).join("index_metadata.json"),
        Path::new(&config.index_path),
    )?;

    let mut tables = Vec::new();
    let mut key_types = Vec::new();
    for fkey in &config.fkeys {
        let table = index_reader
            .table(&fkey.target_table)
            .ok_or_else(|| anyhow!("Target table {} not found in index", fkey.target_table))?;
        let kt = table
            .pk_map()
            .ok_or_else(|| {
                anyhow!("Target table {} has no PK map", fkey.target_table)
            })?
            .key_type();
        key_types.push(kt);
        tables.push(table);
    }

    let resolver = IndexResolver { tables };
    process_export_internal(config, &resolver, &key_types)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::{Int64Builder, StringBuilder};
    use tempfile::TempDir;

    struct MockResolver {
        maps: Vec<Vec<(Key, u64)>>,
    }

    impl IdResolver for MockResolver {
        fn resolve(&self, fkey_idx: usize, key: &Key) -> Result<Option<u64>> {
            Ok(self.maps[fkey_idx]
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| *v))
        }
    }

    fn write_test_parquet_string(path: &Path) {
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
        )
        .unwrap();

        write_parquet(path, &batch).unwrap();
    }

    fn write_test_parquet_int(path: &Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("arange", DataType::Int64, false),
            Field::new("ENTITY", DataType::Int64, true),
            Field::new("TARGET", DataType::Int64, true),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![100, 101, 102])) as Arc<dyn Array>,
                Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(1)])) as Arc<dyn Array>,
                Arc::new(Int64Array::from(vec![Some(10), Some(999), None])) as Arc<dyn Array>,
            ],
        )
        .unwrap();

        write_parquet(path, &batch).unwrap();
    }

    #[test]
    fn test_export_string_keys() {
        let dir = TempDir::new().unwrap();
        let tt_path = dir.path().join("train_table.parquet");
        write_test_parquet_string(&tt_path);

        let output_dir = dir.path().join("output");

        let config = ExportConfig {
            train_table_path: tt_path.to_str().unwrap().to_string(),
            index_path: "dummy".to_string(),
            output_dir: output_dir.to_str().unwrap().to_string(),
            row_idx_col: "arange".to_string(),
            train_table_name: "custom_train_table".to_string(),
            fkeys: vec![
                FkeyConfig {
                    column_name: "ENTITY".to_string(),
                    target_table: "users".to_string(),
                },
                FkeyConfig {
                    column_name: "TARGET".to_string(),
                    target_table: "products".to_string(),
                },
            ],
        };

        let resolver = MockResolver {
            maps: vec![
                vec![
                    (Key::Bytes(b"user_1".to_vec()), 0),
                    (Key::Bytes(b"user_2".to_vec()), 1),
                ],
                vec![
                    (Key::Bytes(b"prod_1".to_vec()), 0),
                ],
            ],
        };

        let key_types = vec![KeyType::VarBytes, KeyType::VarBytes];
        process_export_internal(&config, &resolver, &key_types).unwrap();

        assert!(output_dir
            .join("custom_train_table_fkey_cols.parquet")
            .exists());
        assert!(output_dir
            .join("custom_train_table_mask.parquet")
            .exists());

        let mask_file = File::open(
            output_dir.join("custom_train_table_mask.parquet/part-0000.parquet"),
        )
        .unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(mask_file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        let mask = batch.column(0).as_primitive::<Int32Type>();
        assert_eq!(mask.len(), 3);
        assert_eq!(mask.value(0), 1); // user_1, prod_1 (both valid)
        assert_eq!(mask.value(1), 0); // user_2, prod_999 (invalid target)
        assert_eq!(mask.value(2), 0); // user_1, null (invalid target)
    }

    #[test]
    fn test_export_int_keys() {
        let dir = TempDir::new().unwrap();
        let tt_path = dir.path().join("train_table.parquet");
        write_test_parquet_int(&tt_path);

        let output_dir = dir.path().join("output");

        let config = ExportConfig {
            train_table_path: tt_path.to_str().unwrap().to_string(),
            index_path: "dummy".to_string(),
            output_dir: output_dir.to_str().unwrap().to_string(),
            row_idx_col: "arange".to_string(),
            train_table_name: "custom_train_table".to_string(),
            fkeys: vec![
                FkeyConfig {
                    column_name: "ENTITY".to_string(),
                    target_table: "users".to_string(),
                },
                FkeyConfig {
                    column_name: "TARGET".to_string(),
                    target_table: "products".to_string(),
                },
            ],
        };

        let resolver = MockResolver {
            maps: vec![
                vec![(Key::Int(1), 0), (Key::Int(2), 1)],
                vec![(Key::Int(10), 0)],
            ],
        };

        let key_types = vec![KeyType::Int64, KeyType::Int64];
        process_export_internal(&config, &resolver, &key_types).unwrap();

        // Verify mask
        let mask_file = File::open(
            output_dir.join("custom_train_table_mask.parquet/part-0000.parquet"),
        )
        .unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(mask_file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        let mask = batch.column(0).as_primitive::<Int32Type>();
        assert_eq!(mask.len(), 3);
        assert_eq!(mask.value(0), 1); // entity=1, target=10 (both valid)
        assert_eq!(mask.value(1), 0); // entity=2, target=999 (invalid target)
        assert_eq!(mask.value(2), 0); // entity=1, target=null

        // Verify fkey_cols
        let fkey_file = File::open(
            output_dir.join("custom_train_table_fkey_cols.parquet/part-0000.parquet"),
        )
        .unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(fkey_file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        let entity_col = batch.column(0).as_primitive::<Int64Type>();
        assert_eq!(entity_col.value(0), 0); // entity 1 -> local_id 0
        assert_eq!(entity_col.value(1), 1); // entity 2 -> local_id 1

        // Verify edge parquet
        let edge_file = File::open(
            output_dir
                .join("products__rev_TARGET__custom_train_table/edge.parquet/part-0000.parquet"),
        )
        .unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(edge_file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
        let row_col = batch.column(0).as_primitive::<Int64Type>();
        let col_col = batch.column(1).as_primitive::<Int64Type>();
        assert_eq!(row_col.value(0), 0); // target 10 -> local_id 0
        assert_eq!(col_col.value(0), 100); // arange -> 100
    }

    #[test]
    fn test_export_e2e_real_index() {
        use std::process::Command;

        let dir = TempDir::new().unwrap();

        let source_dir = dir.path().join("source");
        std::fs::create_dir_all(&source_dir).unwrap();

        let u_schema =
            Arc::new(Schema::new(vec![Field::new("user_id", DataType::Utf8, false)]));
        let mut u_builder = StringBuilder::new();
        u_builder.append_value("user_1");
        u_builder.append_value("user_2");
        let u_batch = RecordBatch::try_new(
            u_schema,
            vec![Arc::new(u_builder.finish()) as Arc<dyn Array>],
        )
        .unwrap();
        write_parquet(&source_dir.join("users.parquet"), &u_batch).unwrap();

        let p_schema = Arc::new(Schema::new(vec![Field::new(
            "product_id",
            DataType::Utf8,
            false,
        )]));
        let mut p_builder = StringBuilder::new();
        p_builder.append_value("prod_1");
        p_builder.append_value("prod_2");
        let p_batch = RecordBatch::try_new(
            p_schema,
            vec![Arc::new(p_builder.finish()) as Arc<dyn Array>],
        )
        .unwrap();
        write_parquet(&source_dir.join("products.parquet"), &p_batch).unwrap();

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
        std::fs::write(
            &config_path,
            serde_json::to_string(&config_json).unwrap(),
        )
        .unwrap();

        let index_dir = dir.path().join("compiled_index");

        let mut current_dir = std::env::current_dir().unwrap();
        while !current_dir.join("Cargo.toml").exists() {
            if !current_dir.pop() {
                panic!("Could not find workspace root");
            }
        }

        let status = Command::new("cargo")
            .args([
                "run",
                "--release",
                "-p",
                "diskgraph-ingest",
                "--bin",
                "ingest",
                "--",
                "--source",
                config_path.to_str().unwrap(),
                "--dest",
                "local",
                "--output",
                index_dir.to_str().unwrap(),
            ])
            .current_dir(current_dir)
            .status()
            .expect("failed to execute ingest binary");

        assert!(status.success(), "ingestion failed");

        let tt_path = dir.path().join("train_table.parquet");
        write_test_parquet_string(&tt_path);

        let output_dir = dir.path().join("output");
        let config = ExportConfig {
            train_table_path: tt_path.to_str().unwrap().to_string(),
            index_path: index_dir.to_str().unwrap().to_string(),
            output_dir: output_dir.to_str().unwrap().to_string(),
            row_idx_col: "arange".to_string(),
            train_table_name: "custom_train_table".to_string(),
            fkeys: vec![
                FkeyConfig {
                    column_name: "ENTITY".to_string(),
                    target_table: "users".to_string(),
                },
                FkeyConfig {
                    column_name: "TARGET".to_string(),
                    target_table: "products".to_string(),
                },
            ],
        };

        process_export(&config).unwrap();

        let mask_file = File::open(
            output_dir.join("custom_train_table_mask.parquet/part-0000.parquet"),
        )
        .unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(mask_file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        let mask = batch.column(0).as_primitive::<Int32Type>();
        assert_eq!(mask.len(), 3);
        assert_eq!(mask.value(0), 1);
        assert_eq!(mask.value(1), 0);
        assert_eq!(mask.value(2), 0);
    }
}
