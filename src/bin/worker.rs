//! Worker binary for distributed ANN testing
//!
//! This binary runs a WorkerNode gRPC server that participates in distributed
//! Protean network simulations coordinated by a central coordinator.
//!
//! ## Usage
//! ```bash
//! cargo run --bin worker -- \
//!     --worker-id worker0 \
//!     --bind-address 0.0.0.0:50051 \
//!     --coordinator-address localhost:50050
//! ```

use std::error::Error;
use std::net::SocketAddr;
use std::time::Duration;

use clap::Parser;
use protean::address::Address;
use protean::embedding_space::F32L2Space;
use tonic::transport::Server;
use tonic::Request;
use tracing::info;

use protean_dist_sim::proto::dist_sim::worker_node_server::WorkerNodeServer;
use protean_dist_sim::proto::dist_sim::coordinator_node_client::CoordinatorNodeClient;
use protean_dist_sim::proto::dist_sim::WorkerInfo;
use protean_dist_sim::worker::worker::Worker;

/// Command-line arguments for worker binary
#[derive(Parser, Debug)]
#[command(name = "worker")]
#[command(about = "Distributed ANN worker node", long_about = None)]
struct Args {
    /// Worker ID (unique identifier)
    #[arg(long)]
    worker_id: String,

    /// Bind address for gRPC server (e.g., "0.0.0.0:50051")
    #[arg(long)]
    bind_address: String,

    /// Advertise address for other nodes to connect (e.g., "worker0:50051")
    /// If not provided, uses bind_address
    #[arg(long)]
    advertise_address: Option<String>,

    /// Coordinator address (e.g., "localhost:50050" or "coordinator:50050")
    #[arg(long)]
    coordinator_address: String,
}

type Sift1MSpace = F32L2Space<128>;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .with_level(true)
        .init();

    let args = Args::parse();
    info!("Starting worker: id={}", args.worker_id);
    info!("  Bind address: {}", args.bind_address);
    info!("  Coordinator: {}", args.coordinator_address);

    let bind_addr: SocketAddr = args.bind_address.parse()?;
    let advertise_addr = args.advertise_address.as_deref().unwrap_or(&args.bind_address);
    let my_address = Address::from(advertise_addr.to_string());
    let coordinator_address = Address::from(args.coordinator_address.clone());

    // Create worker (connects to coordinator)
    info!("Connecting to coordinator...");
    let worker = Worker::<Sift1MSpace>::new(
        args.worker_id.clone(),
        my_address,
        coordinator_address,
    ).await?;

    // Create gRPC service
    let service = WorkerNodeServer::new(worker)
        .max_decoding_message_size(100 * 1024 * 1024)
        .max_encoding_message_size(100 * 1024 * 1024);

    // Spawn gRPC server in background
    info!("Starting gRPC server on {}...", bind_addr);
    let server_handle = tokio::spawn(async move {
        Server::builder()
            .add_service(service)
            .serve(bind_addr)
            .await
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Register with coordinator
    info!("Registering with coordinator at {}...", args.coordinator_address);
    let coordinator_url = format!("http://{}", args.coordinator_address);
    let mut client = CoordinatorNodeClient::connect(coordinator_url).await?;
    client.register_worker(Request::new(WorkerInfo {
        address: advertise_addr.to_string(),
    })).await?;

    info!("Registered with coordinator");

    // Wait for server
    server_handle.await??;

    info!("Worker shutting down");
    Ok(())
}
