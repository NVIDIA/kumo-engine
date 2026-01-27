use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};

use crate::config::JobConfig;
use crate::exec::execute_simple;
use crate::index_reader::IndexReader;
use crate::parquet_writer::{write_train_table, TrainTableRow};
use crate::parser::parse_query;
use crate::report::{ValidationReport, ValidationStatus};
use crate::split::SplitAssigner;
use crate::temporal::execute_temporal;
use crate::validate::{validate_query, IndexSchema};

pub fn run_job(config: &JobConfig) -> Result<()> {
    let query = parse_query(&config.query)?;
    let schema = IndexSchema::from_index_path(&config.index_path)?;
    let outcome = validate_query(&query, &schema);

    if !outcome.is_ok() {
        if let Some(path) = &config.validation_output {
            write_validation_report(path, &outcome.errors, &outcome.warnings, None)?;
        }
        return Err(anyhow!("validation failed"));
    }

    let rows = if predict_has_window(&query) {
        eprintln!("run_job: executing temporal query");
        execute_temporal(&query, Path::new(&config.index_path), &config.options)?
    } else {
        eprintln!("run_job: executing non-temporal query");
        let reader = IndexReader::open(&config.index_path)?;
        execute_simple(&query, &reader)?
    };
    if rows.is_empty() {
        if let Some(path) = &config.validation_output {
            write_validation_report(path, &[], &[], Some(0))?;
        }
        return Err(anyhow!("no rows produced"));
    }

    let (min_ts, max_ts) = min_max_timestamp(&rows);
    let assigner = SplitAssigner::from_config(&config.split, min_ts, max_ts)?;

    let mut out_rows = Vec::new();
    for row in rows {
        let split = assigner.assign(&row.entity, Some(row.timestamp));
        if let Some(split) = split {
            out_rows.push(TrainTableRow {
                entity: row.entity,
                target: row.target,
                timestamp: row.timestamp,
                weight: None,
                split,
            });
        }
    }

    write_train_table(&config.output_parquet, &out_rows)?;

    if let Some(path) = &config.validation_output {
        write_validation_report(path, &[], &[], Some(out_rows.len() as u64))?;
    }

    Ok(())
}

fn predict_has_window(query: &crate::ast::Query) -> bool {
    match &query.predict {
        crate::ast::PredictExpr::Aggregation(agg) => agg.window.is_some(),
        crate::ast::PredictExpr::Comparison(pred) => pred_has_window(pred),
        _ => false,
    }
}

fn pred_has_window(pred: &crate::ast::PredicateExpr) -> bool {
    match pred {
        crate::ast::PredicateExpr::Comparison { left, right, .. } => {
            expr_has_window(left) || expr_has_window(right)
        }
        crate::ast::PredicateExpr::And(items) | crate::ast::PredicateExpr::Or(items) => {
            items.iter().any(pred_has_window)
        }
    }
}

fn expr_has_window(expr: &crate::ast::Expr) -> bool {
    match expr {
        crate::ast::Expr::Agg(agg) => agg.window.is_some(),
        _ => false,
    }
}

fn min_max_timestamp(rows: &[crate::exec::TrainRow]) -> (Option<i64>, Option<i64>) {
    let mut min_ts = None;
    let mut max_ts = None;
    for row in rows {
        if min_ts.map_or(true, |v| row.timestamp < v) {
            min_ts = Some(row.timestamp);
        }
        if max_ts.map_or(true, |v| row.timestamp > v) {
            max_ts = Some(row.timestamp);
        }
    }
    (min_ts, max_ts)
}

fn write_validation_report(
    path: impl AsRef<Path>,
    errors: &[crate::report::ValidationMessage],
    warnings: &[crate::report::ValidationMessage],
    rows: Option<u64>,
) -> Result<()> {
    let mut report = ValidationReport {
        status: if errors.is_empty() {
            ValidationStatus::Ok
        } else {
            ValidationStatus::Error
        },
        errors: errors.to_vec(),
        warnings: warnings.to_vec(),
        stats: None,
    };
    if let Some(rows) = rows {
        report = report.with_stats(rows);
    }
    let bytes = serde_json::to_vec_pretty(&report)?;
    fs::write(path, bytes).context("write validation report")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SplitConfig, SplitType};
    use diskgraph_core::metadata::{DatasetMetadata, NodeDataCompression, NodeShardMetadata, NodeTypeMetadata};
    use diskgraph_rowpack::{ColumnContext, ColumnEncoding, IngestedType, RowPackContext, RowPackEncoder, RowPackScratch, ValueRef};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;
    use tempfile::tempdir;

    fn build_context() -> RowPackContext {
        RowPackContext {
            version: 3,
            columns: vec![
                ColumnContext {
                    name: "customer_id".to_string(),
                    logical_type: IngestedType::Int64,
                    encoding: ColumnEncoding::Int {
                        signed: true,
                        varint: true,
                        bits: 64,
                    },
                },
                ColumnContext {
                    name: "price".to_string(),
                    logical_type: IngestedType::Float64,
                    encoding: ColumnEncoding::Float {
                        storage: diskgraph_rowpack::FloatStorage::F64,
                    },
                },
            ],
            estimated_row_bytes: 32,
        }
    }

    fn write_rowpack_ctx(path: &Path, ctx: &RowPackContext) {
        let bytes = ctx.to_bytes().expect("ctx bytes");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    fn write_node_data(path: &Path, rows: Vec<Vec<u8>>) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut file = File::create(path).unwrap();
        for row in rows {
            let len = row.len() as u32;
            file.write_all(&len.to_le_bytes()).unwrap();
            file.write_all(&row).unwrap();
        }
    }

    #[test]
    fn run_job_writes_parquet() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("index");
        fs::create_dir_all(&index_path).unwrap();

        let ctx = build_context();
        write_rowpack_ctx(&index_path.join("nodes/transactions/node_data.ctx"), &ctx);

        let metadata = DatasetMetadata {
            dataset_name: "test".to_string(),
            node_types: vec![NodeTypeMetadata {
                name: "transactions".to_string(),
                max_id: 2,
                has_node_data: true,
                node_data_compression: Some(NodeDataCompression {
                    codec: "rowpack".to_string(),
                    context_path: "nodes/transactions/node_data.ctx".to_string(),
                }),
                primary_key_map: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 2,
                    node_data_paths: None,
                }],
            }],
            edge_relations: Vec::new(),
        };
        fs::write(
            index_path.join("index_metadata.json"),
            serde_json::to_vec_pretty(&metadata).unwrap(),
        )
        .unwrap();

        let encoder = RowPackEncoder::new(ctx.clone()).unwrap();
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut row1 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Float64(2.5)],
                &mut row1,
                &mut scratch,
            )
            .unwrap();
        let mut row2 = Vec::new();
        encoder
            .encode_row_into(
                &[ValueRef::Int64(1), ValueRef::Float64(1.5)],
                &mut row2,
                &mut scratch,
            )
            .unwrap();
        write_node_data(
            &index_path.join("nodes/transactions/shard-00000/node_data.bin"),
            vec![row1, row2],
        );

        let output_path = dir.path().join("train.parquet");
        let config = JobConfig {
            job_name: None,
            index_path: index_path.to_string_lossy().to_string(),
            query: "PREDICT SUM(transactions.price) FOR EACH transactions.customer_id".to_string(),
            output_parquet: output_path.to_string_lossy().to_string(),
            split: SplitConfig {
                split_type: SplitType::Random,
                fractions: vec![1.0, 0.0, 0.0],
                seed: Some(0),
                time: None,
                range: None,
            },
            options: Default::default(),
            validation_output: None,
        };

        run_job(&config).unwrap();

        let file = File::open(&output_path).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
    }
}
