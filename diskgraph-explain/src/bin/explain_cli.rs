//! Interactive CLI for exploring diskgraph-explain output.
//!
//! Usage: explain_cli <explanations.parquet>
//!
//! Commands:
//!   help                  Show this help
//!   stats                 Show summary statistics
//!   show <id>             Show full explanation for entity
//!   reason <id>           Show top drivers for entity's prediction
//!   similar <id> [k]      Show k most similar entities in same cluster
//!   cluster <id>          Show cluster profile for entity
//!   list [n]              List first n entities
//!   exit | quit           Exit the CLI

use anyhow::{anyhow, Result};
use diskgraph_explain::{EntityExplanation, FeatureContext};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::cmp::Ordering;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        return Err(anyhow!("usage: explain_cli <explanations.parquet>"));
    }
    let explain_path = &args[1];

    let explanations = load_explanations(explain_path)?;

    println!("{}", title("DiskGraph Explain Inspector"));
    println!("{} {}", label("entities"), explanations.len());
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
            "stats" => print_stats(&explanations),
            "show" => {
                let Some(id) = parts.next() else {
                    println!("{}", err("usage: show <id>"));
                    continue;
                };
                match explanations.get(id) {
                    Some(exp) => print_explanation(exp),
                    None => println!("{} {}", err("unknown id"), dim(id)),
                }
            }
            "reason" => {
                let Some(id) = parts.next() else {
                    println!("{}", err("usage: reason <id>"));
                    continue;
                };
                match explanations.get(id) {
                    Some(exp) => print_reason(exp),
                    None => println!("{} {}", err("unknown id"), dim(id)),
                }
            }
            "similar" => {
                let Some(id) = parts.next() else {
                    println!("{}", err("usage: similar <id> [k]"));
                    continue;
                };
                let k = parts.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or(5);
                match explanations.get(id) {
                    Some(exp) => print_similar(exp, &explanations, k),
                    None => println!("{} {}", err("unknown id"), dim(id)),
                }
            }
            "cluster" => {
                let Some(id) = parts.next() else {
                    println!("{}", err("usage: cluster <id>"));
                    continue;
                };
                match explanations.get(id) {
                    Some(exp) => print_cluster(exp),
                    None => println!("{} {}", err("unknown id"), dim(id)),
                }
            }
            "list" => {
                let n = parts.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or(10);
                print_list(&explanations, n);
            }
            _ => println!("{} {}", err("unknown command"), dim(head)),
        }
    }
    Ok(())
}

fn load_explanations(path: &str) -> Result<HashMap<String, EntityExplanation>> {
    let file = File::open(path)?;
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut explanations = HashMap::new();

    while let Some(batch) = reader.next() {
        let batch = batch?;
        let schema = batch.schema();
        let id_idx = schema.index_of("entity_id")?;
        let json_idx = schema.index_of("explanation_json")?;

        let ids = batch
            .column(id_idx)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| anyhow!("entity_id column not string"))?;
        let jsons = batch
            .column(json_idx)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .ok_or_else(|| anyhow!("explanation_json column not string"))?;

        for i in 0..batch.num_rows() {
            let id = ids.value(i).to_string();
            let json_str = jsons.value(i);
            let exp: EntityExplanation = serde_json::from_str(json_str)?;
            explanations.insert(id, exp);
        }
    }

    Ok(explanations)
}

fn print_help() {
    println!("{}", title("Commands"));
    println!("  {}", cmd("help"));
    println!("  {}", cmd("stats"));
    println!("  {}", cmd("show <id>"));
    println!("  {}", cmd("reason <id>"));
    println!("  {}", cmd("similar <id> [k]"));
    println!("  {}", cmd("cluster <id>"));
    println!("  {}", cmd("list [n]"));
    println!("  {}", cmd("exit | quit"));
}

fn print_stats(explanations: &HashMap<String, EntityExplanation>) {
    if explanations.is_empty() {
        println!("{}", dim("no explanations loaded"));
        return;
    }

    let predictions: Vec<f64> = explanations.values().map(|e| e.prediction).collect();
    let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
    let min = predictions.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = predictions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Count entities by cluster
    let mut cluster_counts: HashMap<usize, usize> = HashMap::new();
    for exp in explanations.values() {
        if let Some(cluster) = &exp.cluster {
            *cluster_counts.entry(cluster.cluster_id).or_default() += 1;
        }
    }

    println!("{} {}", label("entities"), explanations.len());
    println!("{} {:.4}", label("mean prediction"), mean);
    println!("{} {:.4} - {:.4}", label("range"), min, max);
    println!("{} {}", label("clusters"), cluster_counts.len());
}

fn print_explanation(exp: &EntityExplanation) {
    println!("{} {}", label("id"), bright(&exp.entity_id));
    println!("{} {}", label("prediction"), value_color(exp.prediction));
    println!();

    // Population context
    println!("{}", title("Population Context"));
    println!(
        "  {} mean {:.4}, std {:.4}",
        label("population"),
        exp.population.mean_prediction,
        exp.population.std_prediction
    );
    println!(
        "  {} {:.4} - {:.4}",
        label("range"),
        exp.population.min_prediction,
        exp.population.max_prediction
    );
    println!();

    // Top drivers
    if !exp.top_drivers.is_empty() {
        println!("{}", title("Top Drivers"));
        for driver in &exp.top_drivers {
            print_feature_context(driver);
        }
        println!();
    }

    // Cluster
    if let Some(cluster) = &exp.cluster {
        println!("{}", title("Cluster"));
        println!("  {} {}", label("cluster_id"), cluster.cluster_id);
        println!("  {} {}", label("size"), cluster.cluster_size);
        println!(
            "  {} {:.4} ± {:.4}",
            label("mean prediction"),
            cluster.cluster_mean_prediction,
            cluster.cluster_std_prediction
        );
        println!("  {} {:.2}", label("fit score"), cluster.fit_score);

        if !cluster.entity_matches.is_empty() {
            println!("  {} {:?}", label("matches"), cluster.entity_matches);
        }
        if !cluster.entity_differs.is_empty() {
            println!("  {} {:?}", label("differs"), cluster.entity_differs);
        }
    }
}

fn print_reason(exp: &EntityExplanation) {
    let polarity = if exp.prediction >= exp.population.mean_prediction {
        "above average"
    } else {
        "below average"
    };
    let headline = format!(
        "Prediction {:.4} ({}, mean {:.4})",
        exp.prediction, polarity, exp.population.mean_prediction
    );
    println!("{}", good(&headline));
    println!();

    if exp.top_drivers.is_empty() {
        println!("{}", dim("no top drivers identified"));
        return;
    }

    println!("{}", title("Top Prediction Drivers"));
    for driver in &exp.top_drivers {
        print_feature_context(driver);
    }
}

fn print_feature_context(ctx: &FeatureContext) {
    println!(
        "  {} {} = {:.4}",
        bullet("feature"),
        bright(&ctx.feature),
        ctx.value
    );
    println!(
        "    {} z={:.2}, decile={}, cond_rate={:.4}",
        dim("context"),
        ctx.z_score,
        ctx.decile,
        ctx.conditional_rate
    );
    println!(
        "    {} pop_mean={:.4}, pop_rate={:.4}",
        dim("population"),
        ctx.population_mean,
        ctx.population_rate
    );
}

fn print_similar(exp: &EntityExplanation, all: &HashMap<String, EntityExplanation>, k: usize) {
    let Some(cluster) = &exp.cluster else {
        println!("{}", err("entity has no cluster assignment"));
        return;
    };

    // Find entities in same cluster
    let mut same_cluster: Vec<&EntityExplanation> = all
        .values()
        .filter(|e| {
            e.entity_id != exp.entity_id
                && e.cluster.as_ref().map(|c| c.cluster_id) == Some(cluster.cluster_id)
        })
        .collect();

    // Sort by prediction similarity
    same_cluster.sort_by(|a, b| {
        let a_diff = (a.prediction - exp.prediction).abs();
        let b_diff = (b.prediction - exp.prediction).abs();
        a_diff.partial_cmp(&b_diff).unwrap_or(Ordering::Equal)
    });

    println!("{} (cluster {})", title("Similar Entities"), cluster.cluster_id);
    for e in same_cluster.into_iter().take(k) {
        println!(
            "  {} {} pred {:.4}",
            bullet("id"),
            bright(&e.entity_id),
            e.prediction
        );
    }
}

fn print_cluster(exp: &EntityExplanation) {
    let Some(cluster) = &exp.cluster else {
        println!("{}", err("entity has no cluster assignment"));
        return;
    };

    println!("{} {}", label("cluster_id"), cluster.cluster_id);
    println!("{} {}", label("size"), cluster.cluster_size);
    println!(
        "{} {:.4} ± {:.4}",
        label("mean prediction"),
        cluster.cluster_mean_prediction,
        cluster.cluster_std_prediction
    );
    println!("{} {:.2}", label("fit score"), cluster.fit_score);
    println!();

    if !cluster.profile.is_empty() {
        println!("{}", title("Cluster Profile"));
        for feat in &cluster.profile {
            println!(
                "  {} {} mean={:.4} cond_rate={:.4}",
                bullet("feature"),
                bright(&feat.feature),
                feat.cluster_mean,
                feat.conditional_rate
            );
        }
    }
}

fn print_list(explanations: &HashMap<String, EntityExplanation>, n: usize) {
    // Sort by prediction descending
    let mut sorted: Vec<&EntityExplanation> = explanations.values().collect();
    sorted.sort_by(|a, b| b.prediction.partial_cmp(&a.prediction).unwrap_or(Ordering::Equal));

    println!("{}", title("Entities (by prediction)"));
    for exp in sorted.into_iter().take(n) {
        let cluster_info = exp
            .cluster
            .as_ref()
            .map(|c| format!("cluster={}", c.cluster_id))
            .unwrap_or_else(|| "no cluster".to_string());
        println!(
            "  {} {} pred {:.4} ({})",
            bullet("id"),
            bright(&exp.entity_id),
            exp.prediction,
            dim(&cluster_info)
        );
    }
}

fn value_color(v: f64) -> String {
    if v >= 0.7 {
        good(&format!("{v:.4}"))
    } else if v <= 0.3 {
        warn(&format!("{v:.4}"))
    } else {
        info(&format!("{v:.4}"))
    }
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
