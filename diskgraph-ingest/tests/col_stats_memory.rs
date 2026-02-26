/// Integration test: verify col-stats ingest does not retain excessive memory
/// from LocalColumnCounts HashMap capacity in the mpsc channel.
///
/// Creates many parquet files × categorical columns each.
/// Before the fix, the retained HashMap bucket arrays would accumulate
/// in the unbounded mpsc channel during encode, consuming massive memory.
/// After the fix (dropping locals in flush_categories), peak RSS during
/// encode should be close to the no-col-stats baseline.
use anyhow::Result;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_ingest::col_stats::CategoryCountStrategy;
use diskgraph_ingest::io::Dest;
use diskgraph_ingest::{ingest_with_stats, IngestOptions};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(path)?;
    let batch = RecordBatch::try_new(Arc::new(schema), columns)?;
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn rss_anon_kb() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("RssAnon:") {
            if let Some(val) = line.split_whitespace().nth(1) {
                return val.parse().unwrap_or(0);
            }
        }
    }
    0
}

/// Spawns a background thread that polls peak RSS every 50ms.
/// Returns (stop_flag, peak_kb_value).
fn start_peak_rss_tracker() -> (Arc<AtomicBool>, Arc<AtomicU64>) {
    let stop = Arc::new(AtomicBool::new(false));
    let peak_kb = Arc::new(AtomicU64::new(0));
    let stop_clone = Arc::clone(&stop);
    let peak_clone = Arc::clone(&peak_kb);
    thread::spawn(move || {
        while !stop_clone.load(Ordering::Relaxed) {
            let kb = rss_anon_kb();
            peak_clone.fetch_max(kb, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(50));
        }
    });
    (stop, peak_kb)
}

#[test]
fn test_col_stats_memory_bounded() -> Result<()> {
    let num_files = 2000;
    let num_cat_cols = 10;
    let rows_per_file = 5000; // Enough to fill the local HashMap (HYBRID_LOCAL_CAP=5000)

    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_base = dir.path().join("output_baseline");
    let output_cs = dir.path().join("output_colstats");
    fs::create_dir_all(&input_root)?;

    // Build schema: 1 PK column + N categorical string columns
    let mut fields = vec![Field::new("id", DataType::Int64, false)];
    for i in 0..num_cat_cols {
        fields.push(Field::new(format!("cat_{i}"), DataType::Utf8, false));
    }
    let schema = Schema::new(fields);

    // Generate parquet files
    let mut paths = Vec::new();
    for file_idx in 0..num_files {
        let path = input_root.join(format!("data_{file_idx:04}.parquet"));

        let id_start = (file_idx * rows_per_file) as i64;
        let ids: Vec<i64> = (id_start..id_start + rows_per_file as i64).collect();
        let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(ids))];

        for col_idx in 0..num_cat_cols {
            // Generate categorical values: mix of repeated and unique values
            let values: Vec<String> = (0..rows_per_file)
                .map(|row| format!("cat{col_idx}_val_{}", row % 3000))
                .collect();
            columns.push(Arc::new(StringArray::from(values)));
        }

        write_parquet(&path, schema.clone(), columns)?;
        paths.push(path);
    }

    // Build config JSON
    let mut col_specs = vec![serde_json::json!({"name": "id", "ingested_dtype": "int64"})];
    for i in 0..num_cat_cols {
        col_specs.push(serde_json::json!({
            "name": format!("cat_{i}"),
            "ingested_dtype": "string",
            "semantic_type": "categorical"
        }));
    }

    let source = serde_json::json!({
        "dataset_name": "col_stats_mem_test",
        "tables": [{
            "name": "Items",
            "source": {
                "type": "parquet",
                "paths": paths.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>()
            },
            "primary_key": "id",
            "columns": col_specs
        }]
    });

    let source_path = dir.path().join("config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    // Run ingest WITHOUT col-stats — measure peak RSS
    let (stop1, peak1) = start_peak_rss_tracker();
    let _ = ingest_with_stats(
        &source_path,
        output_base.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;
    stop1.store(true, Ordering::Relaxed);
    thread::sleep(Duration::from_millis(100));
    let baseline_peak_mb = peak1.load(Ordering::Relaxed) as f64 / 1024.0;

    // Run ingest WITH col-stats + Hybrid strategy — measure peak RSS
    let (stop2, peak2) = start_peak_rss_tracker();
    let _ = ingest_with_stats(
        &source_path,
        output_cs.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            col_stats: true,
            cat_count_strategy: CategoryCountStrategy::Hybrid,
            ..IngestOptions::default()
        },
    )?;
    stop2.store(true, Ordering::Relaxed);
    thread::sleep(Duration::from_millis(100));
    let colstats_peak_mb = peak2.load(Ordering::Relaxed) as f64 / 1024.0;

    let excess_mb = colstats_peak_mb - baseline_peak_mb;

    println!("=== Col-stats PEAK memory test ===");
    println!("Files: {num_files}, Cat cols: {num_cat_cols}, Rows/file: {rows_per_file}");
    println!("Baseline peak RSS: {baseline_peak_mb:.1} MB");
    println!("Col-stats peak RSS: {colstats_peak_mb:.1} MB");
    println!("Excess:             {excess_mb:.1} MB");

    // Theoretical pre-fix: 2000 files × 10 cols × ~400KB = ~8000 MB excess
    // (HashMap bucket arrays retained in unbounded mpsc channel)
    // After fix: should be much smaller (HLLs ~16KB × 11 cols × 2000 = ~344 MB + overhead)
    // Use 2000 MB as generous threshold — well below the ~8000 MB pre-fix
    assert!(
        excess_mb < 2000.0,
        "Col-stats peak RSS exceeded baseline by {excess_mb:.0} MB — \
         likely HashMap capacity retention bug (expected < 2000 MB, \
         pre-fix would be ~8000 MB for {num_files} files × {num_cat_cols} cols)"
    );

    Ok(())
}
