use anyhow::{anyhow, Result};
use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use hdrhistogram::Histogram;
use rand::Rng;
use diskgraph_core::sampling::*;
use diskgraph_index::reader::ShardReader;
use diskgraph_sampler::sampler::Sampler;
use diskgraph_sampler::state::StatePool;
use diskgraph_core::metadata::DatasetMetadata;
use std::fs;

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
}

struct AppStats {
    requests_completed: u64,
    nodes_sampled: u64,
    start_time: Instant,
    latency_hist: Histogram<u64>,
}

fn load_graph(path: &PathBuf) -> Result<(Sampler, DatasetMetadata)> {
    let index_path = path.join("index_metadata.json");
    if !index_path.exists() {
        return Err(anyhow!("Metadata not found at {:?}. Run ingestion first.", index_path));
    }
    let metadata_bytes = fs::read(&index_path)?;
    let metadata: DatasetMetadata = serde_json::from_slice(&metadata_bytes)?;

    let layouts = Arc::new(diskgraph_index::builder::compute_relation_layouts(
        &metadata.edge_relations,
        &metadata.node_types,
    )?);

    let mut readers = HashMap::new();

    for node_type in &metadata.node_types {
        let mut type_readers = Vec::new();
        // Assume shards are in output_root/shard-NNNNN/
        for (shard_idx, _) in node_type.shards.iter().enumerate() {
            let shard_dir_name = format!("shard-{:05}", shard_idx);
            let shard_path = path.join(&shard_dir_name);

            // Check if it exists (if single shard, it might be root?)
            let shard_path = if shard_path.exists() {
                shard_path
            } else {
                path.clone()
            };

            let reader = ShardReader::open(&shard_path, layouts.clone(), node_type.has_node_data)?;
            type_readers.push(Arc::new(reader));
        }
        readers.insert(node_type.name.clone(), type_readers);
    }

    Ok((Sampler::new(readers), metadata))
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

    res
}

async fn run_app<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, args: Args, sampler: Arc<Sampler>, metadata: DatasetMetadata) -> Result<()> {
    let running = Arc::new(AtomicBool::new(true));
    let stats = Arc::new(Mutex::new(AppStats {
        requests_completed: 0,
        nodes_sampled: 0,
        start_time: Instant::now(),
        latency_hist: Histogram::<u64>::new(3).unwrap(),
    }));

    let pool = Arc::new(StatePool::new());
    let mut handles = Vec::new();

    let seed_type = args.seed_type.clone().unwrap_or_else(|| {
        metadata.node_types.first().map(|n| n.name.clone()).unwrap_or_default()
    });
    let max_seed_id = metadata.node_types.iter()
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

        handles.push(tokio::task::spawn_blocking(move || {
            let mut rng = rand::thread_rng();
            while running.load(Ordering::Relaxed) {
                let seeds: Vec<u64> = (0..args.batch_size).map(|_| rng.gen_range(0..=max_seed_id)).collect();

                let request = BatchSamplingRequest {
                    dataset_id: diskgraph_core::dataset::DatasetId::nil(),
                    seed_nodes: [(seed_type.clone(), seeds)].into(),
                    seed_timestamps: None,
                    plan: plan.clone(),
                    flags: SamplingFlags {
                        replace: false,
                        directed: true,
                        disjoint: true,
                        return_edge_id: false,
                        adaptive: false,
                    },
                };

                let start = Instant::now();
                let mut state = pool.acquire();
                match sampler.sample(&request, &mut state) {
                    Ok(_) => {
                        let duration = start.elapsed().as_micros() as u64;
                        let nodes_count: usize = state.type_states.values().map(|ts| ts.nodes.len()).sum();

                        let mut s = stats.lock().unwrap();
                        s.requests_completed += 1;
                        s.nodes_sampled += nodes_count as u64;
                        s.latency_hist.record(duration).ok();
                    }
                    Err(_) => {
                        // Error handling
                    }
                }
            }
        }));
    }

    loop {
        terminal.draw(|f| {
            let s = stats.lock().unwrap();
            let elapsed = s.start_time.elapsed().as_secs_f64();
            let req_rate = s.requests_completed as f64 / elapsed;
            let node_rate = s.nodes_sampled as f64 / elapsed;
            let p50 = s.latency_hist.value_at_quantile(0.5);
            let p99 = s.latency_hist.value_at_quantile(0.99);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([Constraint::Length(5), Constraint::Min(0)].as_ref())
                .split(f.size());

            let text = vec![
                ratatui::text::Line::from(format!("Requests: {} | Throughput: {:.2} req/s", s.requests_completed, req_rate)),
                ratatui::text::Line::from(format!("Nodes: {} | Throughput: {:.2} nodes/s", s.nodes_sampled, node_rate)),
                ratatui::text::Line::from(format!("Latency (us): P50={} P99={}", p50, p99)),
                ratatui::text::Line::from(format!("Seed Type: {} | Max ID: {}", seed_type, max_seed_id)),
            ];

            let p = Paragraph::new(text)
                .block(Block::default().title("Sampler Stats").borders(Borders::ALL));
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

    Ok(())
}

fn make_plan(args: &Args, meta: &DatasetMetadata, seed_type: &str) -> SamplingPlan {
    // Find relations that start with seed_type
    // Assuming format "Src_Rel_to_Dst" or "Src__Rel__Dst"
    // We check if relation.name starts with seed_type
    // Also need dst_type.
    // EdgeRelationMetadata has `dst_type`.
    // It does not have `src_type` explicitly, but usually it matches or we guess.

    // Heuristic: Pick the first relation that *seems* to start with seed_type or just pick the first one.
    // Let's iterate all relations.
    let candidates: Vec<&str> = meta.edge_relations.iter()
        .filter(|r| r.name.starts_with(seed_type))
        .map(|r| r.name.as_str())
        .collect();

    let _rel_name = candidates.first().unwrap_or(&"Unknown");

    // Construct valid "Sampler" relation string: "Src__Rel__Dst".
    // If metadata name is "User_click_Item", we might need to convert.
    // For now, use the raw name and rely on Sampler logic (which splits by "__").
    // If real data uses single "_", we might need to patch Sampler or this logic.
    // I'll assume for this tool we use "User__click__Item" if possible.

    let mut layers = Vec::new();
    for _ in 0..args.hops {
        layers.push(LayerSpec {
            fanouts: [(
                format!("{}__{}__{}", seed_type, "rel", "dst"), // Placeholder
                FanoutSpec {
                    fanout: args.fanout,
                    mode: SamplingMode::TemporalMostRecent,
                    absent_key_policy: diskgraph_core::sampling::AbsentKeyPolicy::UniformFallback,
                }
            )].into_iter().collect(),
        });
    }
    SamplingPlan { layers }
}
