use anyhow::{anyhow, Result};
use arrow_array::Array;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::cmp::Ordering;

#[derive(Debug, Clone, Deserialize)]
struct FeatureExplanation {
    feature: String,
    lift: f32,
    delta: f32,
    support: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct EventExplanation {
    relation: String,
    table: String,
    node_id: u64,
    pk: Option<String>,
    feature: String,
    lift: f32,
    delta: f32,
    support: u64,
}

#[derive(Debug, Clone)]
struct ExplainRow {
    id: String,
    timestamp: i64,
    prediction: f32,
    embedding: Vec<f32>,
    cohort_id: i32,
    cohort_mean_pred: f32,
    cohort_std_pred: f32,
    global_mean_pred: f32,
    local_lift: f32,
    contrast_cohort_id: i32,
    top_features: Vec<FeatureExplanation>,
    top_events: Vec<EventExplanation>,
}

#[derive(Debug, Clone)]
struct CohortSummary {
    count: i64,
    mean_pred: f32,
    std_pred: f32,
}

#[derive(Debug, Clone)]
struct CohortFeature {
    feature_key: String,
    lift: f32,
    support: i64,
}

#[derive(Debug, Clone)]
struct TemporalTrend {
    bucket: i32,
    mean_pred: f32,
    count: i64,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(anyhow!("usage: explain_cli <explanations.parquet> <global_trends.parquet>"));
    }
    let explain_path = &args[1];
    let trends_path = &args[2];

    let (rows, cohort_index) = load_explanations(explain_path)?;
    let (cohort_summary, cohort_features, cohort_trends) = load_trends(trends_path)?;

    println!("{}", title("DiskGraph Explain Inspector"));
    println!("{} {}", label("rows"), rows.len());
    println!("{} {}", label("cohorts"), cohort_summary.len());
    println!("{}", dim("type 'help' for commands"));

    let stdin = io::stdin();
    loop {
        print!("explain> ");
        io::stdout().flush()?;
        let mut input = String::new();
        if stdin.read_line(&mut input)? == 0 {
            break;
        }
        let cmd = input.trim();
        if cmd.is_empty() {
            continue;
        }
        let mut parts = cmd.split_whitespace();
        let Some(head) = parts.next() else { continue };
        match head {
            "help" => print_help(),
            "exit" | "quit" => break,
            "stats" => print_stats(&rows, &cohort_summary),
            "show" => {
                let Some(id) = parts.next() else { println!("{}", err("usage: show <id>")); continue; };
                match rows.get(id) {
                    Some(row) => print_row(row),
                    None => println!("{} {}", err("unknown id"), dim(id)),
                }
            }
            "reason" => {
                let Some(id) = parts.next() else { println!("{}", err("usage: reason <id>")); continue; };
                match rows.get(id) {
                    Some(row) => print_reason(row),
                    None => println!("{} {}", err("unknown id"), dim(id)),
                }
            }
            "similar" => {
                let Some(id) = parts.next() else { println!("{}", err("usage: similar <id> [k]")); continue; };
                let k = parts.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or(5);
                match rows.get(id) {
                    Some(row) => print_similar(row, &cohort_index, k),
                    None => println!("{} {}", err("unknown id"), dim(id)),
                }
            }
            "cohort" => {
                let Some(id) = parts.next() else { println!("{}", err("usage: cohort <id>")); continue; };
                match rows.get(id) {
                    Some(row) => print_cohort(row.cohort_id, &cohort_summary, &cohort_features),
                    None => println!("{} {}", err("unknown id"), dim(id)),
                }
            }
            "trend" => {
                let Some(id) = parts.next() else { println!("{}", err("usage: trend <cohort_id>")); continue; };
                let cohort_id = id.parse::<i32>().unwrap_or(-1);
                print_trend(cohort_id, &cohort_trends);
            }
            "list" => {
                let Some(id) = parts.next() else { println!("{}", err("usage: list <cohort_id> [n]")); continue; };
                let cohort_id = id.parse::<i32>().unwrap_or(-1);
                let n = parts.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or(10);
                print_list(cohort_id, &cohort_index, n);
            }
            _ => println!("{} {}", err("unknown command"), dim(head)),
        }
    }
    Ok(())
}

fn load_explanations(path: &str) -> Result<(HashMap<String, ExplainRow>, HashMap<i32, Vec<ExplainRow>>)> {
    let file = File::open(path)?;
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut rows = HashMap::new();
    let mut cohort_index: HashMap<i32, Vec<ExplainRow>> = HashMap::new();
    while let Some(batch) = reader.next() {
        let batch = batch?;
        let schema = batch.schema();
        let id_idx = schema.index_of("id")?;
        let ts_idx = schema.index_of("timestamp")?;
        let pred_idx = schema.index_of("prediction")?;
        let emb_idx = schema.index_of("embedding")?;
        let cohort_idx = schema.index_of("cohort_id")?;
        let cohort_mean_idx = schema.index_of("cohort_mean_pred")?;
        let cohort_std_idx = schema.index_of("cohort_std_pred")?;
        let global_idx = schema.index_of("global_mean_pred")?;
        let local_lift_idx = schema.index_of("local_lift")?;
        let contrast_idx = schema.index_of("contrast_cohort_id")?;
        let top_feat_idx = schema.index_of("top_features")?;
        let top_event_idx = schema.index_of("top_events")?;

        let ids = batch.column(id_idx).as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
        let timestamps = batch.column(ts_idx).as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
        let preds = batch.column(pred_idx).as_any().downcast_ref::<arrow_array::Float32Array>().unwrap();
        let cohorts = batch.column(cohort_idx).as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
        let cohort_means = batch.column(cohort_mean_idx).as_any().downcast_ref::<arrow_array::Float32Array>().unwrap();
        let cohort_stds = batch.column(cohort_std_idx).as_any().downcast_ref::<arrow_array::Float32Array>().unwrap();
        let globals = batch.column(global_idx).as_any().downcast_ref::<arrow_array::Float32Array>().unwrap();
        let local_lifts = batch.column(local_lift_idx).as_any().downcast_ref::<arrow_array::Float32Array>().unwrap();
        let contrast = batch.column(contrast_idx).as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
        let top_feats = batch.column(top_feat_idx).as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
        let top_events = batch.column(top_event_idx).as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
        let embeddings = batch.column(emb_idx).as_any().downcast_ref::<arrow_array::ListArray>().unwrap();

        for i in 0..batch.num_rows() {
            let emb_vals = embeddings.value(i);
            let emb = emb_vals.as_any().downcast_ref::<arrow_array::Float32Array>().unwrap();
            let embedding = (0..emb.len()).map(|j| emb.value(j)).collect::<Vec<_>>();
            let top_features: Vec<FeatureExplanation> = serde_json::from_str(top_feats.value(i)).unwrap_or_default();
            let top_events: Vec<EventExplanation> = serde_json::from_str(top_events.value(i)).unwrap_or_default();
            let row = ExplainRow {
                id: ids.value(i).to_string(),
                timestamp: timestamps.value(i),
                prediction: preds.value(i),
                embedding,
                cohort_id: cohorts.value(i),
                cohort_mean_pred: cohort_means.value(i),
                cohort_std_pred: cohort_stds.value(i),
                global_mean_pred: globals.value(i),
                local_lift: local_lifts.value(i),
                contrast_cohort_id: contrast.value(i),
                top_features,
                top_events,
            };
            rows.insert(row.id.clone(), row.clone());
            cohort_index.entry(row.cohort_id).or_default().push(row);
        }
    }
    Ok((rows, cohort_index))
}

fn load_trends(path: &str) -> Result<(HashMap<i32, CohortSummary>, HashMap<i32, Vec<CohortFeature>>, HashMap<i32, Vec<TemporalTrend>>)> {
    let file = File::open(path)?;
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut summary = HashMap::new();
    let mut features: HashMap<i32, Vec<CohortFeature>> = HashMap::new();
    let mut trends: HashMap<i32, Vec<TemporalTrend>> = HashMap::new();
    while let Some(batch) = reader.next() {
        let batch = batch?;
        let schema = batch.schema();
        let type_idx = schema.index_of("artifact_type")?;
        let cohort_idx = schema.index_of("cohort_id")?;
        let count_idx = schema.index_of("count")?;
        let mean_idx = schema.index_of("mean_pred")?;
        let std_idx = schema.index_of("std_pred")?;
        let feature_idx = schema.index_of("feature_key")?;
        let lift_idx = schema.index_of("lift")?;
        let support_idx = schema.index_of("support")?;
        let bucket_idx = schema.index_of("time_bucket")?;

        let types = batch.column(type_idx).as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
        let cohorts = batch.column(cohort_idx).as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
        let counts = batch.column(count_idx).as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
        let means = batch.column(mean_idx).as_any().downcast_ref::<arrow_array::Float32Array>().unwrap();
        let stds = batch.column(std_idx).as_any().downcast_ref::<arrow_array::Float32Array>().unwrap();
        let feature_keys = batch.column(feature_idx).as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
        let lifts = batch.column(lift_idx).as_any().downcast_ref::<arrow_array::Float32Array>().unwrap();
        let supports = batch.column(support_idx).as_any().downcast_ref::<arrow_array::Int64Array>().unwrap();
        let buckets = batch.column(bucket_idx).as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();

        for i in 0..batch.num_rows() {
            let kind = types.value(i);
            if !cohorts.is_valid(i) {
                continue;
            }
            let cohort_id = cohorts.value(i);
            match kind {
                "cohort_summary" => {
                    summary.insert(
                        cohort_id,
                        CohortSummary {
                            count: counts.value(i),
                            mean_pred: means.value(i),
                            std_pred: stds.value(i),
                        },
                    );
                }
                "cohort_feature_lift" => {
                    if feature_keys.is_valid(i) {
                        features.entry(cohort_id).or_default().push(CohortFeature {
                            feature_key: feature_keys.value(i).to_string(),
                            lift: lifts.value(i),
                            support: supports.value(i),
                        });
                    }
                }
                "temporal_trend" => {
                    if buckets.is_valid(i) {
                        trends.entry(cohort_id).or_default().push(TemporalTrend {
                            bucket: buckets.value(i),
                            mean_pred: means.value(i),
                            count: counts.value(i),
                        });
                    }
                }
                _ => {}
            }
        }
    }
    for rows in features.values_mut() {
        rows.sort_by(|a, b| b.lift.partial_cmp(&a.lift).unwrap_or(Ordering::Equal));
    }
    for rows in trends.values_mut() {
        rows.sort_by(|a, b| a.bucket.cmp(&b.bucket));
    }
    Ok((summary, features, trends))
}

fn print_help() {
    println!("{}", title("Commands"));
    println!("  {}", cmd("help"));
    println!("  {}", cmd("stats"));
    println!("  {}", cmd("show <id>"));
    println!("  {}", cmd("reason <id>"));
    println!("  {}", cmd("similar <id> [k]"));
    println!("  {}", cmd("cohort <id>"));
    println!("  {}", cmd("trend <cohort_id>"));
    println!("  {}", cmd("list <cohort_id> [n]"));
    println!("  {}", cmd("exit | quit"));
}

fn print_stats(rows: &HashMap<String, ExplainRow>, cohorts: &HashMap<i32, CohortSummary>) {
    println!("{} {}", label("rows"), rows.len());
    println!("{} {}", label("cohorts"), cohorts.len());
}

fn print_row(row: &ExplainRow) {
    println!("{} {}", label("id"), bright(&row.id));
    println!("{} {}", label("timestamp"), row.timestamp);
    println!("{} {}", label("prediction"), value_color(row.prediction));
    println!("{} {}", label("cohort"), row.cohort_id);
    println!("{} {:.4} ± {:.4}", label("cohort mean"), row.cohort_mean_pred, row.cohort_std_pred);
    println!("{} {:.4}", label("global mean"), row.global_mean_pred);
    println!("{} {:.4}", label("local lift"), row.local_lift);
    println!("{} {}", label("contrast cohort"), row.contrast_cohort_id);
}

fn print_reason(row: &ExplainRow) {
    let polarity = if row.prediction >= row.global_mean_pred { "positive" } else { "negative" };
    let headline = format!("Prediction is {} ({:.4})", polarity, row.prediction);
    println!("{}", good(&headline));
    println!(
        "{} {:.4} (cohort mean {:.4}, global mean {:.4})",
        label("context"),
        row.local_lift,
        row.cohort_mean_pred,
        row.global_mean_pred
    );
    if !row.top_features.is_empty() {
        println!("{}", title("Top features"));
        for feat in row.top_features.iter().take(5) {
            println!(
                "  {} {} {} (lift {:.2}, delta {:.2}, support {})",
                bullet("feature"),
                bright(&feat.feature),
                dim(""),
                feat.lift,
                feat.delta,
                feat.support
            );
        }
    }
    if !row.top_events.is_empty() {
        println!("{}", title("Top events"));
        for ev in row.top_events.iter().take(5) {
            let pk = ev.pk.clone().unwrap_or_else(|| ev.node_id.to_string());
            println!(
                "  {} {}:{} ({}) lift {:.2}, delta {:.2}, support {}",
                bullet("event"),
                bright(&ev.table),
                pk,
                ev.relation,
                ev.lift,
                ev.delta,
                ev.support
            );
            println!("    {} {}", dim("feature"), bright(&ev.feature));
        }
    }
}

fn print_similar(row: &ExplainRow, cohort_index: &HashMap<i32, Vec<ExplainRow>>, k: usize) {
    let Some(cohort_rows) = cohort_index.get(&row.cohort_id) else {
        println!("{}", err("cohort not found"));
        return;
    };
    let mut scored: Vec<(f32, &ExplainRow)> = cohort_rows
        .iter()
        .filter(|r| r.id != row.id)
        .map(|r| (l2(&row.embedding, &r.embedding), r))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    println!("{}", title("Similar entities"));
    for (dist, r) in scored.into_iter().take(k) {
        println!("  {} {} pred {:.4} dist {:.4}", bullet("id"), bright(&r.id), r.prediction, dist);
    }
}

fn print_cohort(cohort_id: i32, summary: &HashMap<i32, CohortSummary>, features: &HashMap<i32, Vec<CohortFeature>>) {
    println!("{} {}", label("cohort"), cohort_id);
    if let Some(info) = summary.get(&cohort_id) {
        println!("{} {}", label("count"), info.count);
        println!("{} {:.4} ± {:.4}", label("mean"), info.mean_pred, info.std_pred);
    }
    if let Some(list) = features.get(&cohort_id) {
        println!("{}", title("Top cohort features"));
        for feat in list.iter().take(10) {
            println!("  {} {} (lift {:.2}, support {})", bullet("feature"), bright(&feat.feature_key), feat.lift, feat.support);
        }
    }
}

fn print_trend(cohort_id: i32, trends: &HashMap<i32, Vec<TemporalTrend>>) {
    println!("{} {}", label("temporal trend for cohort"), cohort_id);
    if let Some(rows) = trends.get(&cohort_id) {
        for row in rows {
            println!("  bucket {} mean {:.4} count {}", row.bucket, row.mean_pred, row.count);
        }
    } else {
        println!("{}", dim("no trend data"));
    }
}

fn print_list(cohort_id: i32, cohort_index: &HashMap<i32, Vec<ExplainRow>>, n: usize) {
    let Some(rows) = cohort_index.get(&cohort_id) else {
        println!("{}", err("cohort not found"));
        return;
    };
    println!("{} {}", label("cohort"), cohort_id);
    for row in rows.iter().take(n) {
        println!("  {} {} pred {:.4}", bullet("id"), bright(&row.id), row.prediction);
    }
}

fn value_color(v: f32) -> String {
    if v >= 0.7 {
        good(&format!("{v:.4}"))
    } else if v <= 0.3 {
        warn(&format!("{v:.4}"))
    } else {
        info(&format!("{v:.4}"))
    }
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum.sqrt()
}

fn color(code: &str, value: &str) -> String {
    format!("\x1b[{code}m{value}\x1b[0m")
}

fn title(value: &str) -> String {
    color("1;36", value)
}

fn label(value: &str) -> String {
    color("1;34", value)
}

fn cmd(value: &str) -> String {
    color("1;33", value)
}

fn bullet(value: &str) -> String {
    color("1;35", value)
}

fn bright(value: &str) -> String {
    color("1;37", value)
}

fn dim(value: &str) -> String {
    color("2;37", value)
}

fn good(value: &str) -> String {
    color("1;32", value)
}

fn warn(value: &str) -> String {
    color("1;31", value)
}

fn info(value: &str) -> String {
    color("1;36", value)
}

fn err(value: &str) -> String {
    color("1;31", value)
}
