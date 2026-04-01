use anyhow::Result;
use clap::Parser;
use diskgraph_engine::engine::DiskGraphEngine;
use diskgraph_engine::state::StatePool;
use diskgraph_gateway::SamplerGateway;
use std::path::PathBuf;
use std::sync::Arc;
use tonic::transport::Server;

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

    /// Optional global random seed for deterministic sampling
    #[arg(long)]
    seed: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    if let Some(seed) = args.seed {
        println!("Using deterministic random seed: {}", seed);
    }
    let engine = Arc::new(DiskGraphEngine::load(&args.data, args.seed, None)?);
    let pool = StatePool::new();

    let service = SamplerGateway::new(engine, pool);
    let server = Server::builder()
        .add_service(arrow_flight::flight_service_server::FlightServiceServer::new(service));

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
