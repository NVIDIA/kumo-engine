use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use diskgraph_index::graph::GraphIndexReader;
use rayon::prelude::*;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{Context as ReadlineContext, Editor, Helper};

use diskgraph_train_table::config::{QueryOptions, SplitConfig, SplitType, TemporalSampling};
use diskgraph_train_table::exec::{execute_simple, TargetValue, TrainRow, Value};
use diskgraph_train_table::index_reader::IndexReader;
use diskgraph_train_table::parquet_writer::{write_train_table, TrainTableRow};
use diskgraph_train_table::parser::parse_query;
use diskgraph_train_table::split::{SplitAssigner, SplitLabel};
use diskgraph_train_table::validate::{validate_query, IndexSchema};

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const BLUE: &str = "\x1b[34m";
const CYAN: &str = "\x1b[36m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const MAGENTA: &str = "\x1b[35m";

fn colorize(value: &str, color: &str) -> String {
    format!("{color}{value}{RESET}")
}

fn dim(value: &str) -> String {
    format!("{DIM}{value}{RESET}")
}

fn warn(value: &str) -> String {
    colorize(value, YELLOW)
}

fn err(value: &str) -> String {
    colorize(value, RED)
}

fn label(value: &str) -> String {
    colorize(value, CYAN)
}

fn name(value: &str) -> String {
    colorize(value, BLUE)
}

fn highlight(value: &str) -> String {
    colorize(value, MAGENTA)
}

#[derive(Clone)]
struct TtuiHelper {
    commands: Vec<String>,
}

impl TtuiHelper {
    fn new(reader: &GraphIndexReader) -> Self {
        let commands = vec![
            "help", "tables", "relations", "run", "split", "sampling", "output", "show", "last", "quit", "exit",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();
        let _ = reader;
        Self { commands }
    }
}

impl Completer for TtuiHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &ReadlineContext<'_>,
    ) -> Result<(usize, Vec<Pair>), ReadlineError> {
        let input = &line[..pos];
        let mut parts = input.split_whitespace();
        let Some(first) = parts.next() else {
            return Ok((0, self
                .commands
                .iter()
                .map(|cmd| Pair { display: cmd.clone(), replacement: cmd.clone() })
                .collect()));
        };
        if parts.next().is_none() {
            let mut out = Vec::new();
            for cmd in &self.commands {
                if cmd.starts_with(first) {
                    out.push(Pair {
                        display: cmd.clone(),
                        replacement: cmd.clone(),
                    });
                }
            }
            return Ok((0, out));
        }
        if first == "run" || first == "output" || first == "split" || first == "sampling" {
            return Ok((pos, Vec::new()));
        }
        Ok((pos, Vec::new()))
    }
}

impl Hinter for TtuiHelper {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &ReadlineContext<'_>) -> Option<String> {
        None
    }
}

impl Highlighter for TtuiHelper {}
impl Validator for TtuiHelper {}
impl Helper for TtuiHelper {}

#[derive(Clone)]
struct SessionConfig {
    index_path: PathBuf,
    output_dir: PathBuf,
    split: SplitConfig,
    options: QueryOptions,
    last_output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = parse_args();
    if !args.index_path.exists() {
        return Err(anyhow!("index path not found: {}", args.index_path.display()));
    }

    let default_output_dir = args
        .output_dir
        .unwrap_or_else(|| args.index_path.join("train-tables"));

    let split = SplitConfig {
        split_type: SplitType::Random,
        fractions: vec![0.8, 0.1, 0.1],
        seed: Some(0),
        time: None,
        range: None,
    };

    let mut session = SessionConfig {
        index_path: args.index_path,
        output_dir: default_output_dir,
        split,
        options: QueryOptions::default(),
        last_output: None,
    };

    if let Some(command) = args.command {
        let output = run_query(&mut session, &command)?;
        println!("{} {}", label("output:"), name(&output.display().to_string()));
        return Ok(());
    }

    let reader = Arc::new(GraphIndexReader::open(
        &session.index_path.join("index_metadata.json"),
        &session.index_path,
    )?);

    repl(reader, &mut session)
}

struct Args {
    index_path: PathBuf,
    output_dir: Option<PathBuf>,
    command: Option<String>,
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut index_path: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut command: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-C" => {
                if let Some(value) = args.next() {
                    command = Some(value);
                }
            }
            "-o" | "--output-dir" => {
                if let Some(value) = args.next() {
                    output_dir = Some(PathBuf::from(value));
                }
            }
            _ if index_path.is_none() => {
                index_path = Some(PathBuf::from(arg));
            }
            _ => {}
        }
    }

    let index_path = index_path.unwrap_or_else(|| {
        eprintln!("usage: ttui <index_path> [-o output_dir] [-C \"PREDICT ...\"]");
        std::process::exit(2);
    });

    Args {
        index_path,
        output_dir,
        command,
    }
}

fn repl(reader: Arc<GraphIndexReader>, session: &mut SessionConfig) -> Result<()> {
    let mut rl = Editor::<TtuiHelper, DefaultHistory>::new().context("init editor")?;
    rl.set_helper(Some(TtuiHelper::new(&reader)));

    println!(
        "{} {} | {} | {} | {} | {} | {} | {} | {}",
        label("Commands:"),
        name("help"),
        name("tables"),
        name("relations"),
        name("run <query>"),
        name("split random <train> <val> <test> [seed]"),
        name("sampling <random|timeframe>"),
        name("output <dir>"),
        name("show"),
    );

    loop {
        let prompt = format!("{BOLD}{CYAN}train-table>{RESET} ");
        match rl.readline(&prompt) {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                rl.add_history_entry(trimmed).ok();
                if trimmed.eq_ignore_ascii_case("quit") || trimmed.eq_ignore_ascii_case("exit") {
                    break;
                }
                if trimmed.eq_ignore_ascii_case("help") {
                    println!(
                        "{} {} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
                        label("Commands:"),
                        name("help"),
                        name("tables"),
                        name("relations"),
                        name("run <query>"),
                        name("split random <train> <val> <test> [seed]"),
                        name("sampling <random|timeframe>"),
                        name("output <dir>"),
                        name("show"),
                        name("last"),
                        name("quit")
                    );
                    continue;
                }
                if trimmed.eq_ignore_ascii_case("tables") {
                    print_tables(&reader);
                    continue;
                }
                if trimmed.eq_ignore_ascii_case("relations") {
                    print_relations(&reader);
                    continue;
                }
                if trimmed.eq_ignore_ascii_case("show") {
                    print_config(session);
                    continue;
                }
                if trimmed.eq_ignore_ascii_case("last") {
                    if let Some(path) = &session.last_output {
                        println!("{} {}", label("last output:"), name(&path.display().to_string()));
                    } else {
                        println!("{}", warn("no output yet"));
                    }
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("output ") {
                    let dir = rest.trim();
                    if dir.is_empty() {
                        println!("{}", warn("missing output dir"));
                        continue;
                    }
                    session.output_dir = PathBuf::from(dir);
                    println!("{} {}", label("output dir:"), name(dir));
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("sampling ") {
                    let value = rest.trim();
                    session.options.temporal_sampling = match value {
                        "random" => TemporalSampling::Random,
                        "timeframe" => TemporalSampling::Timeframe,
                        _ => {
                            println!("{} {value}", err("invalid sampling:"));
                            continue;
                        }
                    };
                    println!("{} {}", label("temporal sampling:"), name(value));
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("split ") {
                    if let Err(split_err) = set_split_config(session, rest) {
                        println!("{} {split_err}", err("invalid split:"));
                    }
                    continue;
                }
                if let Some(rest) = trimmed.strip_prefix("run ") {
                    let query = rest.trim();
                    if query.is_empty() {
                        println!("{}", warn("missing query"));
                        continue;
                    }
                    match run_query(session, query) {
                        Ok(output) => {
                            println!("{} {}", label("output:"), name(&output.display().to_string()));
                        }
                        Err(run_err) => {
                            println!("{} {run_err}", err("run failed:"));
                        }
                    }
                    continue;
                }
                println!("{} {trimmed}", warn("unknown command:"));
            }
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        }
    }

    Ok(())
}

fn print_tables(reader: &GraphIndexReader) {
    println!("{}", label("Tables:"));
    let mut tables: Vec<_> = reader.tables();
    tables.sort_by(|a, b| a.node_type().cmp(b.node_type()));
    for table in tables {
        println!("  {}", name(table.node_type()));
    }
}

fn print_relations(reader: &GraphIndexReader) {
    println!("{}", label("Relations:"));
    for rel in &reader.metadata().edge_relations {
        println!(
            "  {} {} {}",
            name(&rel.name),
            dim("from"),
            highlight(&format!("{} -> {}", rel.src_type, rel.dst_type))
        );
    }
}

fn print_config(session: &SessionConfig) {
    println!("{} {}", label("index:"), name(&session.index_path.display().to_string()));
    println!("{} {}", label("output dir:"), name(&session.output_dir.display().to_string()));
    println!("{} {:?}", label("split:"), session.split);
    println!("{} {:?}", label("temporal sampling:"), session.options.temporal_sampling);
}

fn set_split_config(session: &mut SessionConfig, input: &str) -> Result<()> {
    let mut parts = input.split_whitespace();
    let kind = parts.next().ok_or_else(|| anyhow!("missing split kind"))?;
    if kind != "random" {
        return Err(anyhow!("only random split is supported in this UI"));
    }
    let train = parts
        .next()
        .ok_or_else(|| anyhow!("missing train fraction"))?
        .parse::<f64>()?;
    let val = parts
        .next()
        .ok_or_else(|| anyhow!("missing val fraction"))?
        .parse::<f64>()?;
    let test = parts
        .next()
        .ok_or_else(|| anyhow!("missing test fraction"))?
        .parse::<f64>()?;
    let seed = parts.next().map(|s| s.parse::<u64>()).transpose()?;
    session.split = SplitConfig {
        split_type: SplitType::Random,
        fractions: vec![train, val, test],
        seed,
        time: None,
        range: None,
    };
    println!(
        "{} train={} val={} test={} seed={}",
        label("split set:"),
        train,
        val,
        test,
        seed.map(|s| s.to_string()).unwrap_or_else(|| "none".to_string())
    );
    Ok(())
}

fn run_query(session: &mut SessionConfig, query_str: &str) -> Result<PathBuf> {
    let query = parse_query(query_str)?;
    let schema = IndexSchema::from_index_path(&session.index_path)?;
    let outcome = validate_query(&query, &schema);
    if !outcome.is_ok() {
        println!("{}", err("validation failed"));
        for msg in outcome.errors {
            println!("  {} {}", err(&msg.code), msg.message);
        }
        return Err(anyhow!("validation failed"));
    }

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1);

    let shard_rows: Vec<Vec<TrainRow>> = if predict_has_window(&query) {
        let graph = GraphIndexReader::open(
            &session.index_path.join("index_metadata.json"),
            &session.index_path,
        )?;
        let table = graph
            .table(&query.for_each.table)
            .ok_or_else(|| anyhow!("missing entity table {}", query.for_each.table))?;
        let node_count = table.node_count();
        let chunk = (node_count + threads as u64 - 1) / threads as u64;
        let mut ranges = Vec::new();
        for i in 0..threads {
            let start = i as u64 * chunk;
            if start >= node_count {
                break;
            }
            let end = (start + chunk).min(node_count);
            ranges.push((start, end));
        }
        let progress = diskgraph_train_table::temporal::TemporalProgress::new(
            &query.for_each.table,
            node_count,
        );
        let results: Vec<Result<Vec<TrainRow>>> = ranges
            .par_iter()
            .map(|(start, end)| {
                diskgraph_train_table::temporal::execute_temporal_range_with_progress(
                    &query,
                    Path::new(&session.index_path),
                    &session.options,
                    *start,
                    *end,
                    Some(progress.clone()),
                )
            })
            .collect();
        let shards = collect_parallel_results(results)?;
        let total_rows: usize = shards.iter().map(|rows| rows.len()).sum();
        progress.finish(total_rows);
        shards
    } else {
        let reader = IndexReader::open(&session.index_path)?;
        let rows = execute_simple(&query, &reader)?;
        if rows.is_empty() {
            return Err(anyhow!("no rows produced"));
        }
        split_rows(rows, threads)
    };

    if shard_rows.is_empty() || shard_rows.iter().all(|s| s.is_empty()) {
        return Err(anyhow!("no rows produced"));
    }

    let (min_ts, max_ts) = min_max_timestamp_shards(&shard_rows);
    let assigner = SplitAssigner::from_config(&session.split, min_ts, max_ts)?;

    let assigner = Arc::new(assigner);
    let out_shards: Vec<Vec<TrainTableRow>> = shard_rows
        .into_par_iter()
        .map(|rows| {
            let mut out = Vec::with_capacity(rows.len());
            for row in rows {
                let split = assigner.assign(&row.entity, Some(row.timestamp));
                if let Some(split) = split {
                    out.push(TrainTableRow {
                        entity: row.entity,
                        target: row.target,
                        timestamp: row.timestamp,
                        weight: None,
                        split,
                    });
                }
            }
            out
        })
        .collect();

    if out_shards.iter().all(|s| s.is_empty()) {
        return Err(anyhow!("no rows after split assignment"));
    }

    let shard_dir = session.output_dir.join(format!(
        "train_table_{}_shards",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    ));
    fs::create_dir_all(&shard_dir).context("create output dir")?;

    out_shards
        .par_iter()
        .enumerate()
        .try_for_each(|(idx, rows)| -> Result<()> {
            if rows.is_empty() {
                return Ok(());
            }
            let path = shard_dir.join(format!("part-{idx:05}.parquet"));
            write_train_table(&path, rows)?;
            Ok(())
        })?;

    print_stats_sharded(&out_shards);
    print_samples_sharded(&out_shards, 10);

    session.last_output = Some(shard_dir.clone());
    Ok(shard_dir)
}

fn collect_parallel_results<T>(results: Vec<Result<T>>) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(results.len());
    for result in results {
        out.push(result?);
    }
    Ok(out)
}

fn split_rows(rows: Vec<TrainRow>, shards: usize) -> Vec<Vec<TrainRow>> {
    let shards = shards.max(1);
    let mut out = vec![Vec::new(); shards];
    for (idx, row) in rows.into_iter().enumerate() {
        out[idx % shards].push(row);
    }
    out
}

fn predict_has_window(query: &diskgraph_train_table::ast::Query) -> bool {
    match &query.predict {
        diskgraph_train_table::ast::PredictExpr::Aggregation(agg) => agg.window.is_some(),
        diskgraph_train_table::ast::PredictExpr::Comparison(pred) => pred_has_window(pred),
        _ => false,
    }
}

fn pred_has_window(pred: &diskgraph_train_table::ast::PredicateExpr) -> bool {
    match pred {
        diskgraph_train_table::ast::PredicateExpr::Comparison { left, right, .. } => {
            expr_has_window(left) || expr_has_window(right)
        }
        diskgraph_train_table::ast::PredicateExpr::And(items)
        | diskgraph_train_table::ast::PredicateExpr::Or(items) => items.iter().any(pred_has_window),
    }
}

fn expr_has_window(expr: &diskgraph_train_table::ast::Expr) -> bool {
    match expr {
        diskgraph_train_table::ast::Expr::Agg(agg) => agg.window.is_some(),
        _ => false,
    }
}

fn min_max_timestamp_shards(rows: &[Vec<TrainRow>]) -> (Option<i64>, Option<i64>) {
    let mut min_ts = None;
    let mut max_ts = None;
    for shard in rows {
        for row in shard {
            if min_ts.map_or(true, |v| row.timestamp < v) {
                min_ts = Some(row.timestamp);
            }
            if max_ts.map_or(true, |v| row.timestamp > v) {
                max_ts = Some(row.timestamp);
            }
        }
    }
    (min_ts, max_ts)
}

fn print_stats_sharded(shards: &[Vec<TrainTableRow>]) {
    let mut split_counts = [0u64; 3];
    let mut total = 0usize;
    for shard in shards {
        total += shard.len();
        for row in shard {
            split_counts[split_index(row.split)] += 1;
        }
    }

    println!("{} {}", label("rows:"), total);
    println!("{}", label("split counts:"));
    for split in [SplitLabel::Train, SplitLabel::Val, SplitLabel::Test] {
        let count = split_counts[split_index(split)];
        println!("  {} {}", name(split.as_str()), format_number(count));
    }
    let Some(first) = shards.iter().find(|s| !s.is_empty()).and_then(|s| s.first()) else {
        return;
    };
    let target_kind = match &first.target {
        TargetValue::List(_) => TargetKind::List,
        TargetValue::Number(_) | TargetValue::Count(_) => TargetKind::Numeric,
        TargetValue::Scalar(v) => match v {
            Value::Int(_) | Value::UInt(_) => TargetKind::Numeric,
            Value::Bool(_) => TargetKind::Bool,
            Value::String(_) => TargetKind::Text,
            Value::Bytes(_) => TargetKind::Bytes,
        },
    };

    match target_kind {
        TargetKind::Numeric => {
            let mut overall = NumericAgg::new();
            let mut per_split = [NumericAgg::new(), NumericAgg::new(), NumericAgg::new()];
            for shard in shards {
                for row in shard {
                    let value = target_as_f64(&row.target).unwrap_or(0.0);
                    overall.add(value);
                    per_split[split_index(row.split)].add(value);
                }
            }
            println!("{}", label("label stats (numeric):"));
            print_numeric_stats("overall", &overall);
            for split in [SplitLabel::Train, SplitLabel::Val, SplitLabel::Test] {
                print_numeric_stats(split.as_str(), &per_split[split_index(split)]);
            }
        }
        TargetKind::Bool => {
            let mut overall = BoolAgg::new();
            let mut per_split = [BoolAgg::new(), BoolAgg::new(), BoolAgg::new()];
            for shard in shards {
                for row in shard {
                    let value = matches!(row.target, TargetValue::Scalar(Value::Bool(true)));
                    overall.add(value);
                    per_split[split_index(row.split)].add(value);
                }
            }
            println!("{}", label("label stats (bool):"));
            print_bool_stats("overall", &overall);
            for split in [SplitLabel::Train, SplitLabel::Val, SplitLabel::Test] {
                print_bool_stats(split.as_str(), &per_split[split_index(split)]);
            }
        }
        TargetKind::Text | TargetKind::Bytes => {
            let mut overall = DistinctAgg::new(10_000);
            let mut per_split = [
                DistinctAgg::new(10_000),
                DistinctAgg::new(10_000),
                DistinctAgg::new(10_000),
            ];
            for shard in shards {
                for row in shard {
                    let value = match &row.target {
                        TargetValue::Scalar(Value::String(v)) => v.clone(),
                        TargetValue::Scalar(Value::Bytes(v)) => format!("0x{}", hex_encode(v)),
                        _ => String::new(),
                    };
                    overall.add(&value);
                    per_split[split_index(row.split)].add(&value);
                }
            }
            println!("{}", label("label stats (distinct):"));
            print_distinct_stats("overall", &overall);
            for split in [SplitLabel::Train, SplitLabel::Val, SplitLabel::Test] {
                print_distinct_stats(split.as_str(), &per_split[split_index(split)]);
            }
        }
        TargetKind::List => {
            let mut overall = ListAgg::new();
            let mut per_split = [ListAgg::new(), ListAgg::new(), ListAgg::new()];
            for shard in shards {
                for row in shard {
                    let len = match &row.target {
                        TargetValue::List(values) => values.len() as u64,
                        _ => 0,
                    };
                    overall.add(len);
                    per_split[split_index(row.split)].add(len);
                }
            }
            println!("{}", label("label stats (list length):"));
            print_list_stats("overall", &overall);
            for split in [SplitLabel::Train, SplitLabel::Val, SplitLabel::Test] {
                print_list_stats(split.as_str(), &per_split[split_index(split)]);
            }
        }
    }
}

fn print_samples_sharded(shards: &[Vec<TrainTableRow>], limit: usize) {
    let mut samples = Vec::new();
    for shard in shards {
        for row in shard {
            samples.push(row);
            if samples.len() >= limit {
                break;
            }
        }
        if samples.len() >= limit {
            break;
        }
    }
    let count = samples.len();
    println!("{} {}", label("sample rows:"), count);
    for row in samples {
        let entity = format_value(&row.entity);
        let target = format_target(&row.target);
        println!(
            "  entity={} target={} ts={} split={}",
            entity,
            target,
            row.timestamp,
            row.split.as_str()
        );
    }
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Int(v) => v.to_string(),
        Value::UInt(v) => v.to_string(),
        Value::Bool(v) => v.to_string(),
        Value::String(v) => format!("\"{v}\""),
        Value::Bytes(v) => format!("0x{}", hex_encode(v)),
    }
}

fn format_target(target: &TargetValue) -> String {
    match target {
        TargetValue::Number(v) => format!("{v:.6}"),
        TargetValue::Count(v) => v.to_string(),
        TargetValue::List(values) => {
            let mut out = String::new();
            out.push('[');
            for (idx, value) in values.iter().enumerate() {
                if idx > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format_value(value));
            }
            out.push(']');
            out
        }
        TargetValue::Scalar(value) => format_value(value),
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn format_number(value: u64) -> String {
    let s = value.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let mut count = 0usize;
    for ch in s.chars().rev() {
        if count != 0 && count % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
        count += 1;
    }
    out.chars().rev().collect()
}

#[derive(Clone, Copy)]
enum TargetKind {
    Numeric,
    List,
    Bool,
    Text,
    Bytes,
}

#[derive(Default, Clone)]
struct NumericAgg {
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
    zeros: u64,
}

impl NumericAgg {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            min: 0.0,
            max: 0.0,
            zeros: 0,
        }
    }

    fn add(&mut self, value: f64) {
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            if value < self.min {
                self.min = value;
            }
            if value > self.max {
                self.max = value;
            }
        }
        if value == 0.0 {
            self.zeros += 1;
        }
        self.count += 1;
        self.sum += value;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

#[derive(Default, Clone)]
struct ListAgg {
    count: u64,
    sum: u64,
    min: u64,
    max: u64,
    empty: u64,
}

impl ListAgg {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0,
            min: 0,
            max: 0,
            empty: 0,
        }
    }

    fn add(&mut self, value: u64) {
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            if value < self.min {
                self.min = value;
            }
            if value > self.max {
                self.max = value;
            }
        }
        if value == 0 {
            self.empty += 1;
        }
        self.count += 1;
        self.sum += value;
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum as f64 / self.count as f64
        }
    }
}

fn print_numeric_stats(label_text: &str, agg: &NumericAgg) {
    println!(
        "  {} count={} min={:.4} max={:.4} mean={:.4} zeros={}",
        highlight(label_text),
        format_number(agg.count),
        agg.min,
        agg.max,
        agg.mean(),
        format_number(agg.zeros)
    );
}

fn print_list_stats(label_text: &str, agg: &ListAgg) {
    println!(
        "  {} count={} min_len={} max_len={} mean_len={:.4} empty={}",
        highlight(label_text),
        format_number(agg.count),
        agg.min,
        agg.max,
        agg.mean(),
        format_number(agg.empty)
    );
}

#[derive(Default, Clone)]
struct BoolAgg {
    count: u64,
    true_count: u64,
}

impl BoolAgg {
    fn new() -> Self {
        Self { count: 0, true_count: 0 }
    }

    fn add(&mut self, value: bool) {
        self.count += 1;
        if value {
            self.true_count += 1;
        }
    }
}

fn print_bool_stats(label_text: &str, agg: &BoolAgg) {
    let false_count = agg.count.saturating_sub(agg.true_count);
    println!(
        "  {} count={} true={} false={}",
        highlight(label_text),
        format_number(agg.count),
        format_number(agg.true_count),
        format_number(false_count)
    );
}

#[derive(Clone)]
struct DistinctAgg {
    values: std::collections::HashSet<String>,
    max: usize,
    truncated: bool,
    count: u64,
}

impl DistinctAgg {
    fn new(max: usize) -> Self {
        Self {
            values: std::collections::HashSet::new(),
            max,
            truncated: false,
            count: 0,
        }
    }

    fn add(&mut self, value: &str) {
        self.count += 1;
        if self.truncated {
            return;
        }
        if self.values.len() >= self.max {
            self.truncated = true;
            return;
        }
        self.values.insert(value.to_string());
    }
}

fn print_distinct_stats(label_text: &str, agg: &DistinctAgg) {
    let distinct = agg.values.len();
    let mut sample: Vec<_> = agg.values.iter().take(5).cloned().collect();
    sample.sort();
    let suffix = if agg.truncated { "+" } else { "" };
    println!(
        "  {} count={} distinct={}{suffix} sample={:?}",
        highlight(label_text),
        format_number(agg.count),
        distinct,
        sample
    );
}

fn target_as_f64(target: &TargetValue) -> Option<f64> {
    match target {
        TargetValue::Number(v) => Some(*v),
        TargetValue::Count(v) => Some(*v as f64),
        TargetValue::List(_) => None,
        TargetValue::Scalar(v) => match v {
            Value::Int(v) => Some(*v as f64),
            Value::UInt(v) => Some(*v as f64),
            Value::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            Value::String(_) | Value::Bytes(_) => None,
        },
    }
}

fn split_index(split: SplitLabel) -> usize {
    match split {
        SplitLabel::Train => 0,
        SplitLabel::Val => 1,
        SplitLabel::Test => 2,
    }
}
