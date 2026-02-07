use anyhow::Result;
use diskgraph_core::sampling::*;
use diskgraph_sampler::sampler::Sampler;
use diskgraph_sampler::state::StatePool;
use diskgraph_index::reader::ShardReader;
use diskgraph_index::layout::RelationLayout;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use std::path::PathBuf;
use rand::Rng;

fn main() -> Result<()> {
    // 1. Setup Environment
    let num_nodes = 100_000;
    let num_edges_per_node = 20;
    let tmp_dir = tempfile::tempdir()?;
    let path = tmp_dir.path().to_path_buf();

    println!("Generating synthetic graph in {:?}...", path);
    // TODO: Use diskgraph-datagenerator or manual generation.
    // For now, I'll manually write offsets.bin and edges.bin to simulate a shard.
    generate_synthetic_shard(&path, num_nodes, num_edges_per_node)?;

    // 2. Load Sampler
    let layout = RelationLayout {
        relation_id: 0,
        dst_bits: 32, // Support up to 4B nodes
        dst_mask: 0xFFFF_FFFF,
        dest_sort_key_bits: Some(32),
        dest_sort_key_mask: Some(0xFFFF_FFFF),
        entry_bytes: 8, // 32-bit dst + 32-bit sort key
    };
    let layouts = Arc::new(vec![layout]);

    let reader = Arc::new(ShardReader::open(&path, layouts, false)?);
    println!("Reader opened successfully");
    let mut readers = HashMap::new();
    readers.insert("User".to_string(), vec![reader]); // 1 shard

    let sampler = Sampler::new(readers);
    let pool = StatePool::new();

    // 3. Define Request
    let batch_size = 1024;
    let mut rng = rand::thread_rng();
    let seeds: Vec<u64> = (0..batch_size).map(|_| rng.gen_range(0..num_nodes)).collect();

    let plan = SamplingPlan {
        layers: vec![
            LayerSpec {
                fanouts: [
                    ("User__Click__User".to_string(), FanoutSpec {
                        fanout: 10,
                        mode: SamplingMode::TemporalMostRecent,
                        absent_key_policy: AbsentKeyPolicy::UniformFallback,
                    })
                ].into(),
            },
            LayerSpec {
                fanouts: [
                    ("User__Click__User".to_string(), FanoutSpec {
                        fanout: 10,
                        mode: SamplingMode::TemporalMostRecent,
                        absent_key_policy: AbsentKeyPolicy::UniformFallback,
                    })
                ].into(),
            },
        ]
    };

    let request = BatchSamplingRequest {
        dataset_id: diskgraph_core::dataset::DatasetId::nil(),
        seed_nodes: [("User".to_string(), seeds)].into(),
        seed_timestamps: None,
        plan,
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
        },
    };

    // 4. Run Benchmark
    println!("Starting sampling benchmark (Batch={}, Hops=2, Fanout=10)...", batch_size);
    let start = Instant::now();

    let mut pooled_state = pool.acquire();
    sampler.sample(&request, &mut pooled_state)?;

    let duration = start.elapsed();
    let nodes_sampled = pooled_state.type_states.get("User").map(|s| s.nodes.len()).unwrap_or(0);

    println!("Done in {:?}", duration);
    println!("Nodes Sampled: {}", nodes_sampled);
    println!("Throughput: {:.2} reqs/sec", batch_size as f64 / duration.as_secs_f64());
    println!("Throughput: {:.2} nodes/sec", nodes_sampled as f64 / duration.as_secs_f64());

    Ok(())
}

fn generate_synthetic_shard(path: &PathBuf, num_nodes: u64, edges_per_node: usize) -> Result<()> {
    use std::io::Write;
    use byteorder::{WriteBytesExt, LittleEndian};

    std::fs::create_dir_all(path)?;

    // Write Edges (simple packing: [dst: u32][sort: u32])
    let mut edges_file = std::fs::File::create(path.join("edges.bin"))?;
    let mut offsets_file = std::fs::File::create(path.join("offsets.bin"))?;

    // Header for Offsets (OffsetFileHeader)
    // We cheat and use bincode to match reader expectation
    let mut header = diskgraph_index::layout::OffsetFileHeader::new(num_nodes, num_nodes + 1);
    header.edge_count = num_nodes * edges_per_node as u64;
    header.delta_bytes = 4;
    header.node_data_len_bytes = 4;
    bincode::serialize_into(&mut offsets_file, &header)?;

    // Write Offsets (Delta Encoded Chunks)
    // Simple: 1 chunk of size DELTA_CHUNK
    // Format: [count: u16][base: u64][deltas...]
    let chunk_size = diskgraph_index::layout::DELTA_CHUNK as u64;
    let mut current_offset = 0u64;
    let mut written_nodes = 0;

    while written_nodes <= num_nodes {
        let count = (num_nodes + 1 - written_nodes).min(chunk_size);
        offsets_file.write_u16::<LittleEndian>(count as u16)?;
        offsets_file.write_u64::<LittleEndian>(current_offset)?;

        // Write count-1 deltas
        // Each delta represents one edge list (or 0 if end)
        // Note: The last offset is an end marker.
        // If count=1, we write 0 deltas.
        let count_len = 1; // 20 fits in 1 byte varint
        let edge_bytes = (edges_per_node * 8) as u32;
        let node_bytes = count_len + edge_bytes;

        for _ in 0..count.saturating_sub(1) {
            offsets_file.write_u32::<LittleEndian>(node_bytes)?;
            current_offset += node_bytes as u64;
        }

        // Advance current_offset to the start of the next chunk
        // (Add the delta that would have been the next one)
        if written_nodes + count <= num_nodes {
             current_offset += node_bytes as u64;
        }

        written_nodes += count;
    }

    // Write Edges
    // Random dst, Random sort key
    let mut rng = rand::thread_rng();
    let mut buf = Vec::with_capacity(1024 * 1024);
    for _ in 0..num_nodes {
        // Write count (20) as varint (1 byte)
        buf.push(edges_per_node as u8); // 20 < 63

        for _ in 0..edges_per_node {
            let dst = rng.gen_range(0..num_nodes as u32);
            let sort = rng.gen::<u32>();

            // Packed u64: dst | (sort << 32)
            let packed = (dst as u64) | ((sort as u64) << 32);
            buf.write_u64::<LittleEndian>(packed)?;
        }
        if buf.len() > 100_000 {
            edges_file.write_all(&buf)?;
            buf.clear();
        }
    }
    edges_file.write_all(&buf)?;

    offsets_file.flush()?;
    edges_file.flush()?;

    let meta_off = offsets_file.metadata()?;
    let meta_edge = edges_file.metadata()?;
    println!("Generated: Offsets={} bytes, Edges={} bytes", meta_off.len(), meta_edge.len());

    Ok(())
}
