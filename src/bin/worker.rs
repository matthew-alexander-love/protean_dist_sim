//! Standalone worker binary for distributed ANN testing
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
//!
//! ## Embedding Space Selection
//! The embedding space is selected at compile time via feature flags:
//! - Default: f32l2 (L2 distance with 128 dimensions)

use std::error::Error;
use std::net::SocketAddr;

use clap::Parser;
use protean::address::Address;
use protean::embedding_space::EmbeddingSpace;
use tonic::transport::Server;
use tracing::{info, error};

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

// Compile-time embedding space selection via feature flags
type EmbeddingSpaceImpl = protean::embedding_space::spaces::f32_l2::F32L2Space<128>;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .with_level(true)
        .init();

    // Parse command-line arguments
    let args = Args::parse();

    info!(
        "Starting worker: id={}, bind_address={}, coordinator={}",
        args.worker_id, args.bind_address, args.coordinator_address
    );
    info!("Using embedding space: F32L2Space<128>");

    // Run the worker
    run_worker::<EmbeddingSpaceImpl>(args).await
}

/// Main worker logic
async fn run_worker<S: EmbeddingSpace + Send + Sync + 'static>(args: Args) -> Result<(), Box<dyn Error>>
where
    S::EmbeddingData: Send + Sync + protean::embedding_space::Embedding<Scalar = f32>,
{
    // Parse bind address
    let bind_addr: SocketAddr = args
        .bind_address
        .parse()
        .map_err(|e| format!("Invalid bind address '{}': {}", args.bind_address, e))?;

    // Use advertise address for peer-to-peer communication (not bind address!)
    // Bind address is 0.0.0.0:50051 (local only), advertise is worker0:50051 (routable)
    let my_address = Address::from(
        args.advertise_address.as_deref().unwrap_or(&args.bind_address).to_string()
    );

    info!("Creating worker instance...");

    // Create worker instance with coordinator address for lazy connection
    let worker = Worker::<S>::new(
        args.worker_id.clone(),
        my_address,
        Some(args.coordinator_address.clone()),
    );

    // Create gRPC service (Worker implements WorkerNode trait)
    let service = WorkerNodeServer::new(worker)
        .max_decoding_message_size(100 * 1024 * 1024) // 100MB
        .max_encoding_message_size(100 * 1024 * 1024); // 100MB

    info!("Starting gRPC server on {}...", bind_addr);

    // Start serving in background
    let server_handle = tokio::spawn(async move {
        Server::builder()
            .add_service(service)
            .serve(bind_addr)
            .await
    });

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    info!("Worker gRPC server started");

    // Register with coordinator (coordinator connects to us)
    // This must happen AFTER our gRPC server is running because coordinator will connect back to us
    info!("Registering with coordinator at {}...", args.coordinator_address);
    let advertise_addr = args.advertise_address.as_deref().unwrap_or(&args.bind_address);
    match register_with_coordinator(
        advertise_addr,
        &args.coordinator_address,
    )
    .await
    {
        Ok(assigned_id) => info!("Successfully registered with coordinator, assigned ID: {}", assigned_id),
        Err(e) => {
            error!("Failed to register with coordinator: {}. Continuing anyway...", e);
            // Don't fail - coordinator might be starting up
        }
    }

    // Wait for server to complete
    server_handle.await
        .map_err(|e| format!("Server task failed: {}", e))?
        .map_err(|e| format!("Failed to start gRPC server: {}", e))?;

    info!("Worker shutting down");
    Ok(())
}

/// Register this worker with the coordinator
async fn register_with_coordinator(
    bind_address: &str,
    coordinator_address: &str,
) -> Result<String, Box<dyn Error>> {
    // Connect to coordinator
    let coordinator_url = format!("http://{}", coordinator_address);
    let mut client = CoordinatorNodeClient::connect(coordinator_url).await?;

    // Send registration request
    let request = tonic::Request::new(WorkerInfo {
        address: bind_address.to_string(),
    });

    let response = client.register_worker(request).await?;
    let ack = response.into_inner();

    if !ack.success {
        return Err(format!("Registration failed: {}", ack.message).into());
    }

    Ok(ack.message)
}
