use anyhow::{anyhow, Result};
use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use diskgraph_core::metadata::DatasetMetadata;
use diskgraph_core::sampling::*;
use diskgraph_engine::engine::DiskGraphEngine;
use diskgraph_engine::output::InMemoryOutput;
use diskgraph_engine::state::StatePool;
use hdrhistogram::Histogram;
use rand::Rng;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use std::fs;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use sysinfo::System;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the 'diskgraph_output' directory containing index_metadata.json
    #[arg(short, long)]
    data: PathBuf,

    /// Number of concurrent sampling workers
    #[arg(short, long, default_value_t = 16)]
    concurrency: usize,

    /// Batch size (seeds per request)
    #[arg(short, long, default_value_t = 1024)]
    batch_size: usize,

    /// Number of hops
    #[arg(long, default_value_t = 2)]
    hops: usize,

    /// Fanout per hop
    #[arg(long, default_value_t = 10)]
    fanout: u32,

    /// Seed node type (optional, auto-detected if None)
    #[arg(long)]
    seed_type: Option<String>,

    /// Enable exhaustive correctness validation for each sample (impacts performance)
    #[arg(long, default_value_t = false)]
    validate: bool,
}

struct AppStats {
    requests_completed: u64,
    nodes_sampled: u64,
    edges_sampled: u64,
    start_time: Instant,
    latency_hist: Histogram<u64>,
    memory_bytes: u64,
    validation_failures: u64,
}

fn load_graph(path: &std::path::Path) -> Result<(DiskGraphEngine, DatasetMetadata)> {
    let index_path = path.join("index_metadata.json");
    if !index_path.exists() {
        return Err(anyhow!(
            "Metadata not found at {:?}. Run ingestion first.",
            index_path
        ));
    }
    let metadata_bytes = fs::read(&index_path)?;
    let metadata: DatasetMetadata = serde_json::from_slice(&metadata_bytes)?;

    let engine = DiskGraphEngine::load(path, None, None)?;

    Ok((engine, metadata))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let (sampler, metadata) = match load_graph(&args.data) {
        Ok(v) => v,
        Err(e) => {
            println!("Failed to load graph: {}", e);
            return Err(e);
        }
    };

    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let sampler_arc = Arc::new(sampler);
    let rt = tokio::runtime::Runtime::new()?;
    let res = rt.block_on(run_app(&mut terminal, args, sampler_arc, metadata));

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Ok(r) = res {
        println!("\nBenchmark Finished.");
        println!("Total Requests: {}", r.requests_completed);
        println!("Nodes Sampled: {}", r.nodes_sampled);
        println!("Edges Sampled: {}", r.edges_sampled);
        println!("Validation Failures: {}", r.validation_failures);
        println!("P50 Latency: {} us", r.latency_hist.value_at_quantile(0.5));
        println!("P99 Latency: {} us", r.latency_hist.value_at_quantile(0.99));
    }

    Ok(())
}

async fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    args: Args,
    sampler: Arc<DiskGraphEngine>,
    metadata: DatasetMetadata,
) -> Result<AppStats> {
    let running = Arc::new(AtomicBool::new(true));
    let stats = Arc::new(Mutex::new(AppStats {
        requests_completed: 0,
        nodes_sampled: 0,
        edges_sampled: 0,
        start_time: Instant::now(),
        latency_hist: Histogram::<u64>::new(3).unwrap(),
        memory_bytes: 0,
        validation_failures: 0,
    }));

    let pool = Arc::new(StatePool::new());
    let mut handles = Vec::new();

    let seed_type = args.seed_type.clone().unwrap_or_else(|| {
        metadata
            .node_types
            .first()
            .map(|n| n.name.clone())
            .unwrap_or_default()
    });
    let max_seed_id = metadata
        .node_types
        .iter()
        .find(|n| n.name == seed_type)
        .map(|n| n.max_id)
        .unwrap_or(0);

    let plan_base = make_plan(&args, &metadata, &seed_type);

    for _ in 0..args.concurrency {
        let running = running.clone();
        let stats = stats.clone();
        let sampler = sampler.clone();
        let pool = pool.clone();
        let seed_type = seed_type.clone();
        let plan = plan_base.clone();
        let do_validate = args.validate;

        handles.push(tokio::task::spawn_blocking(move || {
            let mut rng = rand::thread_rng();
            while running.load(Ordering::Relaxed) {
                let seeds: Vec<u64> = (0..args.batch_size)
                    .map(|_| rng.gen_range(0..=max_seed_id))
                    .collect();

                let request = BatchSamplingRequest {
                    dataset_id: diskgraph_core::dataset::DatasetId::nil(),
                    seed_nodes: [(seed_type.clone(), seeds)].into(),
                    seed_timestamps: None,
                    seed_timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
                    plan: plan.clone(),
                    flags: SamplingFlags {
                        replace: false,
                        directed: true,
                        disjoint: true,
                        return_edge_id: false,
                        adaptive: false,
                        fused_feature_fetching: false,
                        swap_edge_direction: true,
                    },
                };

                let start = Instant::now();
                let mut state = pool.acquire();
                let mut output = InMemoryOutput::new();

                if sampler.sample(&request, &mut state, &mut output).is_ok() {
                    let duration = start.elapsed().as_micros() as u64;
                    let nodes_count = state.stats.total_nodes;
                    let edges_count = state.stats.total_edges;

                    let mut v_failures = 0;
                    if do_validate {
                        // Validation needs to inspect output now
                        if let Err(_e) = validate_sample(&output, &state) {
                            v_failures = 1;
                        }
                    }

                    let mut s = stats.lock().unwrap();
                    s.requests_completed += 1;
                    s.nodes_sampled += nodes_count;
                    s.edges_sampled += edges_count;
                    s.validation_failures += v_failures;
                    s.latency_hist.record(duration).ok();
                }
            }
        }));
    }

    let mut sys = System::new_all();
    let pid = sysinfo::get_current_pid().map_err(|e| anyhow!(e))?;

    loop {
        // Update memory usage
        sys.refresh_process(pid);
        let mem = sys
            .process(pid)
            .map(|p: &sysinfo::Process| p.memory())
            .unwrap_or(0);
        {
            let mut s = stats.lock().unwrap();
            s.memory_bytes = mem;
        }

        terminal.draw(|f| {
            let s = stats.lock().unwrap();
            let elapsed = s.start_time.elapsed().as_secs_f64();
            let req_rate = s.requests_completed as f64 / elapsed;
            let node_rate = s.nodes_sampled as f64 / elapsed;
            let edge_rate = s.edges_sampled as f64 / elapsed;
            let p50 = s.latency_hist.value_at_quantile(0.5);
            let p99 = s.latency_hist.value_at_quantile(0.99);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([Constraint::Length(7), Constraint::Min(0)].as_ref())
                .split(f.size());

            let text = vec![
                ratatui::text::Line::from(format!(
                    "Requests: {} | Throughput: {:.2} req/s",
                    s.requests_completed, req_rate
                )),
                ratatui::text::Line::from(format!(
                    "Nodes: {} | Throughput: {:.2} nodes/s",
                    s.nodes_sampled, node_rate
                )),
                ratatui::text::Line::from(format!(
                    "Edges: {} | Throughput: {:.2} edges/s",
                    s.edges_sampled, edge_rate
                )),
                ratatui::text::Line::from(format!("Latency (us): P50={} P99={}", p50, p99)),
                ratatui::text::Line::from(format!(
                    "Memory Usage: {:.2} MB",
                    s.memory_bytes as f64 / 1024.0 / 1024.0
                )),
                ratatui::text::Line::from(format!(
                    "Validation Failures: {}",
                    s.validation_failures
                )),
                ratatui::text::Line::from(format!(
                    "Seed Type: {} | Max ID: {}",
                    seed_type, max_seed_id
                )),
            ];

            let p = Paragraph::new(text).block(
                Block::default()
                    .title("Sampler TUI - Real-time Performance & Correctness")
                    .borders(Borders::ALL),
            );
            f.render_widget(p, chunks[0]);
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    running.store(false, Ordering::Relaxed);
                    break;
                }
            }
        }
    }

    for h in handles {
        let _ = h.await;
    }

    let final_stats = stats.lock().unwrap();
    Ok(AppStats {
        requests_completed: final_stats.requests_completed,
        nodes_sampled: final_stats.nodes_sampled,
        edges_sampled: final_stats.edges_sampled,
        start_time: final_stats.start_time,
        latency_hist: final_stats.latency_hist.clone(),
        memory_bytes: final_stats.memory_bytes,
        validation_failures: final_stats.validation_failures,
    })
}

fn validate_sample(
    output: &diskgraph_engine::output::InMemoryOutput,
    state: &diskgraph_engine::state::SamplingState,
) -> Result<()> {
    // Basic structural validation
    for (_hop, _rel, src_local, dst_local) in &output.edges {
        // Check if src/dst local IDs are within range of total nodes
        if *src_local as u64 >= state.stats.total_nodes {
            return Err(anyhow!("src_local_id out of range"));
        }
        if *dst_local as u64 >= state.stats.total_nodes {
            return Err(anyhow!("dst_local_id out of range"));
        }
    }
    Ok(())
}

fn make_plan(args: &Args, meta: &DatasetMetadata, seed_type: &str) -> SamplingPlan {
    let candidates: Vec<&str> = meta
        .edge_relations
        .iter()
        .filter(|r| r.name.starts_with(seed_type))
        .map(|r| r.name.as_str())
        .collect();

    let rel_name = candidates.first().unwrap_or(&"Unknown");

    let mut layers = Vec::new();
    for _ in 0..args.hops {
        layers.push(LayerSpec {
            fanouts: [(
                rel_name.to_string(),
                FanoutSpec {
                    fanout: args.fanout,
                    mode: SamplingMode::TemporalMostRecent,
                    absent_key_policy: diskgraph_core::sampling::AbsentKeyPolicy::UniformFallback,
                },
            )]
            .into_iter()
            .collect(),
            default_fanout: None,
        });
    }
    SamplingPlan { layers }
}
