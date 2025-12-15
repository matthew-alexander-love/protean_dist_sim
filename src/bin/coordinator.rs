//! Coordinator binary for distributed ANN testing
//!
//! This binary orchestrates distributed Protean network simulations by:
//! - Loading datasets and test plans from config
//! - Starting the gRPC server and waiting for workers
//! - Running test phases (join, leave, churn, queries, snapshots)
//!
//! ## Usage
//! ```bash
//! cargo run --bin coordinator -- --config deploy/config.yaml
//! ```

use std::error::Error;
use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;
use protean::embedding_space::F32L2Space;
use tracing::info;

use protean_dist_sim::coordinator::{
    Config, Coordinator, DataLoader, Sift1MDataset,
};

/// Command-line arguments for coordinator binary
#[derive(Parser, Debug)]
#[command(name = "coordinator")]
#[command(about = "Distributed ANN test coordinator", long_about = None)]
struct Args {
    /// Path to config YAML file
    #[arg(long)]
    config: String,

    /// gRPC bind address (e.g., "0.0.0.0:50050")
    #[arg(long, default_value = "0.0.0.0:50050")]
    bind_address: String,
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
    info!("Starting coordinator");
    info!("  Config: {}", args.config);
    info!("  Bind address: {}", args.bind_address);

    // Load config
    let config = Config::from_yaml(&args.config)?;
    info!("Loaded config with {} phases", config.phases.len());
    info!("  Data dir: {}", config.sim_config.data_dir);
    info!("  Output dir: {}", config.sim_config.output_dir);
    info!("  Num workers: {}", config.sim_config.num_workers);

    // Load dataset
    let data_dir = &config.sim_config.data_dir;
    let dataset = Sift1MDataset::new(data_dir);
    if !dataset.verify_available() {
        info!("Dataset not found, downloading...");
        dataset.download()?;
    }
    let data_set = dataset.load_data()?;
    info!(
        "Loaded dataset: {} train vectors, {} test vectors",
        data_set.train.len(),
        data_set.test.len()
    );

    // Create coordinator
    let coordinator = Coordinator::<Sift1MSpace>::new(config, data_set);
    let coordinator = Arc::new(coordinator);

    // Start gRPC server and wait for workers
    let bind_addr: SocketAddr = args.bind_address.parse()?;
    let coordinator = coordinator.spawn_worker_overlay(bind_addr).await
        .map_err(|e| -> Box<dyn Error> { e })?;

    // Run test plan
    info!("Starting test plan execution...");
    coordinator.run_test_plan().await;

    info!("Coordinator complete");
    Ok(())
}
