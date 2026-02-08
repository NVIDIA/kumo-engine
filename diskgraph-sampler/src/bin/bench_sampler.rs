use anyhow::{anyhow, Result};
use diskgraph_core::sampling::*;
use diskgraph_sampler::sampler::Sampler;
use diskgraph_sampler::state::StatePool;
use diskgraph_sampler::output::InMemoryOutput;
use diskgraph_index::reader::ShardReader;
use diskgraph_core::metadata::DatasetMetadata;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use std::path::PathBuf;
use rand::Rng;
use std::fs;

use diskgraph_sampler::sampler::ShardHandle;

fn main() -> Result<()> {
    // 1. Setup Environment
    let data_path = PathBuf::from("/mnt/data/data/index/ifood_bp-job-8dd9d07e93bf4dbf8fe2845687bd96ef_index");
    let index_path = data_path.join("index_metadata.json");
    let metadata_bytes = fs::read(&index_path)?;
    let metadata: DatasetMetadata = serde_json::from_slice(&metadata_bytes)?;
    
    println!("Loading graph from {:?}...", data_path);
    let all_layouts = diskgraph_index::builder::compute_relation_layouts(
        &metadata.edge_relations,
        &metadata.node_types,
    )?;
    
    let mut readers = HashMap::new();
    for node_type in &metadata.node_types {
        // Filter layouts for this node type
        let mut type_layouts = Vec::new();
        for layout in &all_layouts {
            let rel_meta = &metadata.edge_relations[layout.relation_id as usize];
            if rel_meta.src_type == node_type.name {
                type_layouts.push(layout.clone());
            }
        }
        let type_layouts = Arc::new(type_layouts);

        let mut type_shards = Vec::new();
        for (shard_idx, shard_meta) in node_type.shards.iter().enumerate() {
            let shard_dir_name = format!("shard-{:05}", shard_idx);
            let shard_path = data_path.join("nodes").join(&node_type.name).join(&shard_dir_name);
            if shard_path.exists() {
                let reader = ShardReader::open(&shard_path, type_layouts.clone(), node_type.has_node_data)?;
                type_shards.push(ShardHandle {
                    reader: Arc::new(reader),
                    min_id: shard_meta.min_id,
                    max_id: shard_meta.max_id,
                });
            }
        }
        readers.insert(node_type.name.clone(), type_shards);
    }
    
    let mut relation_info = HashMap::new();
    for (idx, rel) in metadata.edge_relations.iter().enumerate() {
        relation_info.insert(idx, (rel.name.clone(), rel.dst_type.clone()));
    }
    
    let decoders = HashMap::new();
    let sampler = Sampler::new(readers, relation_info, decoders);
    let pool = StatePool::new();

    // 2. Define Request
    let seed_type = "orders_parquet";
    let max_id = metadata.node_types.iter().find(|n| n.name == seed_type).unwrap().max_id;
    
    let batch_size = 100;
    let mut rng = rand::thread_rng();
    let seeds: Vec<u64> = (0..batch_size).map(|_| rng.gen_range(0..=max_id)).collect();
    
    // Plan: orders_parquet -> merchants -> reviews_reverse
    let plan = SamplingPlan {
        layers: vec![
            LayerSpec {
                fanouts: [
                    ("orders_parquet_merchant_id_numerical_to_merchants_merchant_id_numerical".to_string(), FanoutSpec {
                        fanout: 10,
                        mode: SamplingMode::TemporalMostRecent,
                        absent_key_policy: AbsentKeyPolicy::UniformFallback,
                    })
                ].into(),
                default_fanout: None,
            },
            LayerSpec {
                fanouts: [
                    ("reviews_merchant_id_numerical_to_merchants_merchant_id_numerical_reverse".to_string(), FanoutSpec {
                        fanout: 10,
                        mode: SamplingMode::TemporalMostRecent,
                        absent_key_policy: AbsentKeyPolicy::UniformFallback,
                    })
                ].into(),
                default_fanout: None,
            },
        ]
    };

    let request = BatchSamplingRequest {
        dataset_id: diskgraph_core::dataset::DatasetId::nil(),
        seed_nodes: [(seed_type.to_string(), seeds)].into(),
        seed_timestamps: None,
        plan,
        flags: SamplingFlags {
            replace: false, directed: true, disjoint: true, return_edge_id: false, adaptive: false,
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
        let name = sampler.id_to_node_type.get(*id).map(|s| s.as_str()).unwrap_or("unknown");
        println!("Type {}: {} nodes", name, ts.nodes.len());
    }
    println!("Total Unique Nodes: {}", pooled_state.stats.total_nodes);
    println!("Total Edges: {}", pooled_state.stats.total_edges);
    println!("Output Edges: {}", output.edges.len());

    Ok(())
}