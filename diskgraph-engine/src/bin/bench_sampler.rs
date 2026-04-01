use anyhow::Result;
use diskgraph_core::metadata::DatasetMetadata;
use diskgraph_core::sampling::*;
use diskgraph_engine::engine::DiskGraphEngine;
use diskgraph_engine::output::InMemoryOutput;
use diskgraph_engine::state::StatePool;
use rand::Rng;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<()> {
    // 1. Setup Environment
    let data_path =
        PathBuf::from("/mnt/data/data/index/ifood_bp-job-8dd9d07e93bf4dbf8fe2845687bd96ef_index");
    let index_path = data_path.join("index_metadata.json");
    let metadata_bytes = fs::read(&index_path)?;
    let metadata: DatasetMetadata = serde_json::from_slice(&metadata_bytes)?;

    println!("Loading graph from {:?}...", data_path);
    let engine = DiskGraphEngine::load(&data_path, None, None)?;
    let sampler = Arc::new(engine); // Bench code expects sampler var name
    let pool = StatePool::new();

    // 2. Define Request
    let seed_type = "orders_parquet";
    let max_id = metadata
        .node_types
        .iter()
        .find(|n| n.name == seed_type)
        .unwrap()
        .max_id;

    let batch_size = 100;
    let mut rng = rand::thread_rng();
    let seeds: Vec<u64> = (0..batch_size).map(|_| rng.gen_range(0..=max_id)).collect();

    // Plan: orders_parquet -> merchants -> reviews_reverse
    let plan = SamplingPlan {
        layers: vec![
            LayerSpec {
                fanouts: [(
                    "orders_parquet_merchant_id_numerical_to_merchants_merchant_id_numerical"
                        .to_string(),
                    FanoutSpec {
                        fanout: 10,
                        mode: SamplingMode::TemporalMostRecent,
                        absent_key_policy: AbsentKeyPolicy::UniformFallback,
                    },
                )]
                .into(),
                default_fanout: None,
            },
            LayerSpec {
                fanouts: [(
                    "reviews_merchant_id_numerical_to_merchants_merchant_id_numerical_reverse"
                        .to_string(),
                    FanoutSpec {
                        fanout: 10,
                        mode: SamplingMode::TemporalMostRecent,
                        absent_key_policy: AbsentKeyPolicy::UniformFallback,
                    },
                )]
                .into(),
                default_fanout: None,
            },
        ],
    };

    let request = BatchSamplingRequest {
        dataset_id: diskgraph_core::dataset::DatasetId::nil(),
        seed_nodes: [(seed_type.to_string(), seeds)].into(),
        seed_timestamps: None,
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
        plan,
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

    // 3. Run
    println!("Starting sampling (Batch=100, Hops=2)...");
    let start = Instant::now();
    let mut pooled_state = pool.acquire();
    let mut output = InMemoryOutput::new();

    sampler.sample(&request, &mut pooled_state, &mut output)?;

    let duration = start.elapsed();

    println!("Done in {:?}", duration);
    for (id, ts) in &pooled_state.type_states {
        let name = sampler
            .id_to_node_type
            .get(*id)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        println!("Type {}: {} nodes", name, ts.nodes.len());
    }
    println!("Total Unique Nodes: {}", pooled_state.stats.total_nodes);
    println!("Total Edges: {}", pooled_state.stats.total_edges);
    println!("Output Edges: {}", output.edges.len());

    Ok(())
}
