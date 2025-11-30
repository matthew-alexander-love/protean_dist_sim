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
//! cargo run --bin coordinator -- \
//!     --test-plan test_plan.yaml \
//!     --data-dir ./data
//! ```

use std::error::Error;
use std::net::SocketAddr;
use std::time::Duration;

use clap::Parser;
use protean::embedding_space::F32L2Space;
use protean::SnvConfig;
use tonic::transport::Server;
use tracing::{error, info};

use protean_dist_sim::coordinator::{
    Coordinator, CoordinatorConfig, Sift1MDataset, TestPlan,
};
use protean_dist_sim::proto::dist_sim::coordinator_node_server::CoordinatorNodeServer;

/// Command-line arguments for coordinator binary
#[derive(Parser, Debug)]
#[command(name = "coordinator")]
#[command(about = "Distributed ANN test coordinator", long_about = None)]
struct Args {
    /// Path to test plan YAML file
    #[arg(long)]
    test_plan: String,

    /// Path to data directory (for SIFT1M dataset)
    #[arg(long, default_value = "./data")]
    data_dir: String,

    /// Number of workers to wait for
    #[arg(long, default_value = "1")]
    num_workers: usize,

    /// Maximum peers per worker
    #[arg(long, default_value = "100000")]
    workers_capacity: usize,

    /// gRPC bind address
    #[arg(long, default_value = "0.0.0.0:50050")]
    bind_address: String,

    /// Output directory for results
    #[arg(long, default_value = "./output")]
    output_dir: String,
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
    info!("  Test plan file: {}", args.test_plan);
    info!("  Data directory: {}", args.data_dir);
    info!("  Num workers: {}", args.num_workers);
    info!("  Bind address: {}", args.bind_address);
    info!("  Dataset: SIFT1M (F32L2Space<128>)");

    run_coordinator(args).await
}

/// Main coordinator logic using SIFT1M dataset
async fn run_coordinator(args: Args) -> Result<(), Box<dyn Error>> {
    // Load test plan
    info!("Loading test plan from {}...", args.test_plan);
    let test_plan = TestPlan::from_yaml(&args.test_plan)
        .map_err(|e| format!("Failed to load test plan: {}", e))?;

    info!("Loaded test plan with {} phases", test_plan.phases.len());

    // Create configuration
    let config = CoordinatorConfig {
        workers_capacity: args.workers_capacity,
        num_workers: args.num_workers,
        snv_config: SnvConfig::default(),
        output_dir: args.output_dir.clone(),
        coordinator_bind_address: args.bind_address.clone(),
    };

    let dataloader = Sift1MDataset::new(&args.data_dir);

    // Create coordinator - embedding space is determined by the dataloader
    info!("Creating coordinator and loading dataset...");
    let coordinator = Coordinator::<Sift1MEmbeddingSpace>::new(dataloader, test_plan, config.clone())
        .map_err(|e| format!("Failed to create coordinator: {}", e))?;

    // Parse bind address for gRPC server
    let bind_addr: SocketAddr = args
        .bind_address
        .parse()
        .map_err(|e| format!("Invalid bind address '{}': {}", args.bind_address, e))?;

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
