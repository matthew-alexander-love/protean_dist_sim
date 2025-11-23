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
//!     --config config.yaml \
//!     --test-plan test_plan.yaml
//! ```
//!
//! ## Embedding Space Selection
//! The embedding space is selected at compile time via feature flags:
//! - Default: f32l2 (L2 distance with 128 dimensions)

use std::error::Error;
use std::fs;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use protean::embedding_space::EmbeddingSpace;
use tonic::transport::Server;
use tracing::{info, warn, error};

use protean_dist_sim::coordinator::{Coordinator, CoordinatorConfig, CoordinatorService, TestPlan};
use protean_dist_sim::proto::dist_sim::coordinator_server::CoordinatorServer;

/// Command-line arguments for coordinator binary
#[derive(Parser, Debug)]
#[command(name = "coordinator")]
#[command(about = "Distributed ANN test coordinator", long_about = None)]
struct Args {
    /// Path to coordinator configuration YAML file
    #[arg(long)]
    config: String,

    /// Path to test plan YAML file
    #[arg(long)]
    test_plan: String,

    /// Timeout for waiting for workers to register (seconds)
    #[arg(long, default_value = "30")]
    worker_timeout: u64,
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

    info!("Starting coordinator");
    info!("  Config file: {}", args.config);
    info!("  Test plan file: {}", args.test_plan);
    info!("  Embedding space: F32L2Space<128>");

    // Run the coordinator
    run_coordinator::<EmbeddingSpaceImpl>(args).await
}

/// Main coordinator logic
async fn run_coordinator<S>(args: Args) -> Result<(), Box<dyn Error>>
where
    S: EmbeddingSpace,
    S::EmbeddingData: protean::embedding_space::Embedding<Scalar = f32>,
{
    // Load configuration
    info!("Loading configuration from {}...", args.config);
    let config_str = fs::read_to_string(&args.config)
        .map_err(|e| format!("Failed to read config file '{}': {}", args.config, e))?;

    let config: CoordinatorConfig = serde_yaml::from_str(&config_str)
        .map_err(|e| format!("Failed to parse config YAML: {}", e))?;

    info!("Loaded configuration:");
    info!("  Workers: {}", config.workers.len());
    info!("  Base dataset: {}", config.dataset.base_path);
    info!("  Query dataset: {}", config.dataset.query_path);
    info!("  Output dir: {}", config.output_dir);

    // Load test plan
    info!("Loading test plan from {}...", args.test_plan);
    let test_plan_str = fs::read_to_string(&args.test_plan)
        .map_err(|e| format!("Failed to read test plan file '{}': {}", args.test_plan, e))?;

    let test_plan: TestPlan = serde_yaml::from_str(&test_plan_str)
        .map_err(|e| format!("Failed to parse test plan YAML: {}", e))?;

    info!("Loaded test plan with {} phases", test_plan.phases.len());

    // Create coordinator
    info!("Creating coordinator...");
    let coordinator = Coordinator::<S>::new(config.clone())
        .await
        .map_err(|e| format!("Failed to create coordinator: {}", e))?;

    let coordinator_arc = Arc::new(coordinator);

    // Parse bind address for gRPC server
    let bind_addr: SocketAddr = config
        .coordinator_bind_address
        .parse()
        .map_err(|e| format!("Invalid bind address '{}': {}", config.coordinator_bind_address, e))?;

    info!("Starting coordinator gRPC server on {}...", bind_addr);

    // Create gRPC service
    let service = CoordinatorService::new(coordinator_arc.clone());
    let grpc_server = CoordinatorServer::new(service)
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

    // Wait for workers to register
    info!(
        "Waiting for {} workers to register (timeout: {}s)...",
        config.workers.len(),
        args.worker_timeout
    );

    let worker_timeout = Duration::from_secs(args.worker_timeout);
    match coordinator_arc.wait_for_workers(worker_timeout).await {
        Ok(_) => info!("All workers registered successfully"),
        Err(e) => {
            warn!("Worker registration issue: {}. Proceeding with registered workers...", e);
        }
    }

    info!("Current worker count: {}", coordinator_arc.workers().len());

    // Load datasets
    info!("Loading datasets...");
    coordinator_arc
        .load_datasets()
        .await
        .map_err(|e| format!("Failed to load datasets: {}", e))?;

    info!("Datasets loaded successfully");

    // Distribute embeddings to workers
    info!("Distributing embeddings to workers...");
    coordinator_arc
        .distribute_embeddings()
        .await
        .map_err(|e| format!("Failed to distribute embeddings: {}", e))?;

    info!("Embeddings distributed successfully");

    // Execute test plan
    info!("Executing test plan...");
    match coordinator_arc.run_test_plan(test_plan).await {
        Ok(_) => info!("Test plan completed successfully"),
        Err(e) => {
            error!("Test plan execution failed: {}", e);
            return Err(e);
        }
    }

    // Write results
    info!("Writing results to {}...", config.output_dir);
    coordinator_arc
        .write_results()
        .await
        .map_err(|e| format!("Failed to write results: {}", e))?;

    info!("Results written successfully");
    info!("Coordinator shutting down");

    // Shutdown gRPC server
    server_handle.abort();

    Ok(())
}
