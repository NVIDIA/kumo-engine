use clap::Parser;
use diskgraph_gateway::SamplerGateway;
use diskgraph_sampler::sampler::Sampler;
use diskgraph_sampler::state::StatePool;
use diskgraph_core::metadata::DatasetMetadata;
use diskgraph_index::reader::ShardReader;
use diskgraph_sampler::sampler::ShardHandle;
use arrow_flight::flight_service_server::FlightServiceServer;
use tonic::transport::Server;
use std::collections::HashMap;
use std::sync::Arc;
use std::path::PathBuf;
use std::fs;
use anyhow::{anyhow, Result};
use tracing_subscriber;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the 'diskgraph_output' directory containing index_metadata.json
    #[arg(short, long)]
    data: PathBuf,

    /// Port to listen on (TCP)
    #[arg(short, long, default_value_t = 50051)]
    port: u16,

    /// Path to Unix Domain Socket (UDS) to listen on. If provided, port is ignored.
    #[arg(short, long)]
    uds_path: Option<PathBuf>,
}

// ... (load_graph logic continues below)

fn load_graph(path: &PathBuf) -> Result<Sampler> {
    let index_path = path.join("index_metadata.json");
    if !index_path.exists() {
        return Err(anyhow!("Metadata not found at {:?}. Run ingestion first.", index_path));
    }
    let metadata_bytes = fs::read(&index_path)?;
    let metadata: DatasetMetadata = serde_json::from_slice(&metadata_bytes)?;

    println!("Loading graph metadata for {} node types...", metadata.node_types.len());

    let all_layouts = diskgraph_index::builder::compute_relation_layouts(
        &metadata.edge_relations,
        &metadata.node_types,
    )?;

    let mut relation_info = HashMap::new();
    let mut decoders = HashMap::new();
    for (idx, rel) in metadata.edge_relations.iter().enumerate() {
        relation_info.insert(idx, (rel.name.clone(), rel.dst_type.clone()));
        if let Some(sort_key_meta) = &rel.dest_sort_key {
            decoders.insert(idx, diskgraph_core::sortkey::SortKeyDecoder::new(sort_key_meta));
        }
    }

    let mut readers = HashMap::new();

    for node_type in &metadata.node_types {
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
            let shard_path = path.join("nodes").join(&node_type.name).join(&shard_dir_name);

            // Only open if shard exists
            if shard_path.exists() {
                let reader = ShardReader::open(&shard_path, type_layouts.clone(), node_type.has_node_data)?;
                type_shards.push(ShardHandle {
                    reader: Arc::new(reader),
                    min_id: shard_meta.min_id,
                    max_id: shard_meta.max_id,
                });
            }
        }
        if !type_shards.is_empty() {
             readers.insert(node_type.name.clone(), type_shards);
        }
    }

    println!("Graph loaded. Ready to sample.");
    Ok(Sampler::new(readers, relation_info, decoders))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let sampler = Arc::new(load_graph(&args.data)?);
    let pool = StatePool::new();

    let service = SamplerGateway::new(sampler, pool);
    let server = Server::builder().add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service));

    if let Some(path) = args.uds_path {
        // Clean up existing socket file if any
        if path.exists() {
            std::fs::remove_file(&path)?;
        }

        let listener = tokio::net::UnixListener::bind(&path)?;
        let stream = tokio_stream::wrappers::UnixListenerStream::new(listener);

        println!("Listening on UDS: {:?}", path);
        server.serve_with_incoming(stream).await?;
    } else {
        let addr = format!("0.0.0.0:{}", args.port).parse()?;
        println!("Listening on TCP: {}", addr);
        server.serve(addr).await?;
    }

    Ok(())
}
