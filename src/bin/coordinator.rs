//! Standalone coordinator binary for distributed ANN testing
//!
//! This binary orchestrates distributed Protean network simulations by:
//! - Loading datasets and test plans
//! - Managing worker nodes via gRPC
//! - Distributing data across workers
//! - Executing test phases (bootstrap, churn, queries, snapshots)
//! - Collecting and analyzing results
//!
//! ## Usage
//! ```bash
//! cargo run --bin coordinator -- --config config.yaml --num-workers 4
//! ```

use std::error::Error;
use std::net::SocketAddr;
use std::time::Duration;

use clap::Parser;
use protean::embedding_space::F32L2Space;
use tonic::transport::Server;
use tracing::{error, info};

use protean_dist_sim::coordinator::{
    Config, Coordinator, CoordinatorConfig, Sift1MDataset,
};
use protean_dist_sim::proto::dist_sim::coordinator_node_server::CoordinatorNodeServer;

/// Command-line arguments for coordinator binary
#[derive(Parser, Debug)]
#[command(name = "coordinator")]
#[command(about = "Distributed ANN test coordinator", long_about = None)]
struct Args {
    /// Path to unified config YAML file (contains dataset, sim_config, and test plan)
    #[arg(long)]
    config: String,

    /// Number of workers to wait for
    #[arg(long, default_value = "1")]
    num_workers: usize,

    /// Maximum peers per worker (overrides config if provided)
    #[arg(long)]
    workers_capacity: Option<usize>,

    /// gRPC bind address (overrides config if provided)
    #[arg(long)]
    bind_address: Option<String>,

    /// Output directory for results (overrides config if provided)
    #[arg(long)]
    output_dir: Option<String>,
}

// Sift1MDataset uses F32L2Space<128>
type Sift1MEmbeddingSpace = F32L2Space<128>;

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

    info!("Starting coordinator");
    info!("  Config file: {}", args.config);
    info!("  Num workers: {}", args.num_workers);
    info!("  Dataset: SIFT1M (F32L2Space<128>)");

    run_coordinator(args).await
}

/// Main coordinator logic using SIFT1M dataset
async fn run_coordinator(args: Args) -> Result<(), Box<dyn Error>> {
    // Load unified config
    info!("Loading config from {}...", args.config);
    let unified_config = Config::from_yaml(&args.config)
        .map_err(|e| format!("Failed to load config: {}", e))?;

    info!("Loaded config with {} phases", unified_config.phases.len());
    info!("  Dataset base: {}", unified_config.dataset.base_path);
    info!("  Dataset query: {}", unified_config.dataset.query_path);
    info!("  Peer count: {}", unified_config.peer_count);

    // Apply CLI overrides
    let output_dir = args.output_dir.unwrap_or(unified_config.output_dir.clone());
    let bind_address = args.bind_address.unwrap_or(unified_config.coordinator_bind_address.clone());
    let workers_capacity = args.workers_capacity.unwrap_or(unified_config.sim_config.max_peers);

    info!("  Output dir: {}", output_dir);
    info!("  Bind address: {}", bind_address);

    // Create coordinator configuration
    let config = CoordinatorConfig {
        workers_capacity,
        num_workers: args.num_workers,
        snv_config: unified_config.snv_config(),
        output_dir: output_dir.clone(),
        coordinator_bind_address: bind_address.clone(),
    };

    // Get data directory from dataset base_path (strip sift/sift_base.fvecs to get parent)
    // e.g., /app/data/sift_base.fvecs -> /app/data, then we look for sift/ subdirectory
    let base_path = std::path::Path::new(&unified_config.dataset.base_path);
    let data_dir = base_path
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "./data".to_string());
    info!("  Data directory: {}", data_dir);

    // Extract test plan from unified config
    let test_plan = unified_config.into_test_plan();

    let dataloader = Sift1MDataset::new(&data_dir);

    // Create coordinator - embedding space is determined by the dataloader
    info!("Creating coordinator and loading dataset...");
    let coordinator = Coordinator::<Sift1MEmbeddingSpace>::new(dataloader, test_plan, config.clone())
        .map_err(|e| format!("Failed to create coordinator: {}", e))?;

    // Parse bind address for gRPC server
    let bind_addr: SocketAddr = bind_address
        .parse()
        .map_err(|e| format!("Invalid bind address '{}': {}", bind_address, e))?;

    info!("Starting coordinator gRPC server on {}...", bind_addr);

    // Wrap coordinator in Arc for sharing with gRPC server
    let coordinator = std::sync::Arc::new(coordinator);
    let coordinator_for_run = coordinator.clone();

    let grpc_server = CoordinatorNodeServer::new(coordinator)
        .max_decoding_message_size(100 * 1024 * 1024) // 100MB
        .max_encoding_message_size(100 * 1024 * 1024); // 100MB

    // Start gRPC server in background
    let server_handle = tokio::spawn(async move {
        Server::builder()
            .add_service(grpc_server)
            .serve(bind_addr)
            .await
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(500)).await;
    info!("Coordinator gRPC server started");

    // Execute test plan in background
    let run_handle = tokio::spawn(async move {
        if let Err(e) = coordinator_for_run.run().await {
            error!("Test plan execution failed: {}", e);
        }
    });

    // Wait for either the test plan to complete or the server to error
    tokio::select! {
        result = run_handle => {
            match result {
                Ok(()) => info!("Test plan completed"),
                Err(e) => error!("Test plan task error: {}", e),
            }
        }
        result = server_handle => {
            match result {
                Ok(Ok(())) => info!("Server shutdown gracefully"),
                Ok(Err(e)) => error!("Server error: {}", e),
                Err(e) => error!("Server task error: {}", e),
            }
        }
    }

    info!("Coordinator shutting down");
    Ok(())
}
