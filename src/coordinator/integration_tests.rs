//! Integration tests for distributed coordinator orchestration
//!
//! These tests create real worker instances and test the full distributed workflow:
//! - Worker setup and registration
//! - Data distribution
//! - Network bootstrapping
//! - Query execution with recall verification
//! - Churn operations
//! - Snapshot collection

use std::fs::File;
use std::io::Write;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;
use tonic::transport::Server;

use protean::embedding_space::spaces::f32_l2::F32L2Space;
use protean::address::Address;

use super::*;
use crate::proto::dist_sim::worker_node_server::WorkerNodeServer;
use crate::proto::dist_sim::coordinator_server::CoordinatorServer;
use crate::proto::dist_sim::coordinator_client::CoordinatorClient;
use crate::proto::dist_sim::WorkerInfo;
use crate::worker::worker::Worker;

type TestSpace = F32L2Space<4>;

/// Helper to create test coordinator config for integration tests
fn create_integration_config(num_workers: usize) -> CoordinatorConfig {
    let mut workers = Vec::new();

    // Create worker configs with sequential ports starting from 60051
    for i in 0..num_workers {
        workers.push(WorkerConfig {
            worker_id: format!("worker{}", i),
            address: format!("127.0.0.1:{}", 60051 + i),
        });
    }

    CoordinatorConfig {
        workers,
        dataset: DatasetConfig {
            base_path: "./tmp/integration_test_base.fvecs".to_string(),
            query_path: "./tmp/integration_test_query.fvecs".to_string(),
        },
        sim_config: config::SimConfig {
            snv_config: config::SnvConfig {
                concurrency_limit: 10,
                timeout_ms: 5000,
                occlusion_threshold: 0.5,
                drift_threshold: 0.1,
                target_degree_ratio: 1.0,
                dynamism_threshold: 1.2,
                exploration_config: config::ExplorationConfig {
                    converge_k: 50,
                    converge_config: config::QueryConfig {
                        search_list_size: 50,
                        concurrency_limit: 10,
                        share_floor: 3,
                        timeout: 5000,
                    },
                    explore_k: 0,
                    explore_config: config::QueryConfig {
                        search_list_size: 0,
                        concurrency_limit: 10,
                        share_floor: 3,
                        timeout: 5000,
                    },
                },
                max_exploration_interval_secs: 0,
            },
            max_peers: 1000,
            region: "local".to_string(),
        },
        output_dir: "./tmp".to_string(),
        coordinator_bind_address: "127.0.0.1:60050".to_string(),
    }
}

/// Generate test data with `num_nodes` vectors in 4 dimensions
fn generate_test_data(num_nodes: usize) -> Vec<[f32; 4]> {
    (0..num_nodes)
        .map(|i| {
            let base = (i as f32) * 0.04;
            [base, base + 0.01, base + 0.02, base + 0.03]
        })
        .collect()
}

/// Generate test queries distinct from base dataset
fn generate_test_queries(num_queries: usize) -> Vec<[f32; 4]> {
    (0..num_queries)
        .map(|i| {
            let base = 1000.0 + (i as f32) * 0.04;
            [base, base + 0.01, base + 0.02, base + 0.03]
        })
        .collect()
}

/// Write vectors to .fvecs format
fn write_fvecs(path: &str, vectors: &[[f32; 4]]) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    for vec in vectors {
        // Write dimension count (4 as u32 little-endian)
        file.write_all(&4u32.to_le_bytes())?;
        // Write vector components
        for &component in vec {
            file.write_all(&component.to_le_bytes())?;
        }
    }
    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_staged_bootstrap() {
    const NUM_WORKERS: usize = 5;
    const TOTAL_NODES: usize = 1000;

    println!("\n=== Starting Staged Bootstrap Test ===");
    println!("Workers: {}", NUM_WORKERS);
    println!("Total nodes: {}", TOTAL_NODES);

    // 1. Create coordinator config
    let config = create_integration_config(NUM_WORKERS);
    let coordinator = Arc::new(Coordinator::<TestSpace>::new(config.clone()).await.expect("Failed to create coordinator"));

    // 2. Generate test data
    let base_vectors = generate_test_data(TOTAL_NODES);
    let query_vectors = generate_test_queries(10);

    write_fvecs(&config.dataset.base_path, &base_vectors).expect("Failed to write base dataset");
    write_fvecs(&config.dataset.query_path, &query_vectors).expect("Failed to write query dataset");

    // 3. Start coordinator server
    let coordinator_clone = coordinator.clone();
    let coordinator_svc = service::CoordinatorService::new(coordinator_clone);
    let coord_addr: SocketAddr = config.coordinator_bind_address.parse().unwrap();

    let coord_handle = tokio::spawn(async move {
        Server::builder()
            .add_service(
                CoordinatorServer::new(coordinator_svc)
                    .max_decoding_message_size(100 * 1024 * 1024) // 100MB
                    .max_encoding_message_size(100 * 1024 * 1024) // 100MB
            )
            .serve(coord_addr)
            .await
            .expect("Coordinator server failed");
    });

    tokio::time::sleep(Duration::from_millis(100)).await;
    println!("Coordinator server started at {}", config.coordinator_bind_address);

    // 4. Start workers
    let mut worker_handles = Vec::new();

    for i in 0..NUM_WORKERS {
        let worker_id = format!("worker{}", i);
        let worker_address: Address = format!("127.0.0.1:{}", 60051 + i).parse().unwrap();
        let coordinator_address = "127.0.0.1:60050".to_string();

        let worker = Worker::<TestSpace>::new(
            worker_id.clone(),
            worker_address.clone(),
            NUM_WORKERS,
            200, // max_actors per worker
        );

        // Connect worker to coordinator
        worker.set_coordinator(coordinator_address.clone()).await.expect("Failed to connect");

        // Start worker server
        let worker_addr: SocketAddr = worker_address.to_string().parse().unwrap();
        let worker_server_handle = tokio::spawn(async move {
            Server::builder()
                .add_service(
                    WorkerNodeServer::new(worker)
                        .max_decoding_message_size(100 * 1024 * 1024) // 100MB
                        .max_encoding_message_size(100 * 1024 * 1024) // 100MB
                )
                .serve(worker_addr)
                .await
                .expect("Worker server failed");
        });

        worker_handles.push(worker_server_handle);
        tokio::time::sleep(Duration::from_millis(50)).await;
        println!("Started {}", worker_id);
    }

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Register workers with coordinator
    let coord_client_addr = format!("http://{}", config.coordinator_bind_address);
    let mut coord_client = CoordinatorClient::connect(coord_client_addr).await.expect("Failed to connect to coordinator");
    for i in 0..NUM_WORKERS {
        let worker_address = format!("127.0.0.1:{}", 60051 + i);
        coord_client.register_worker(WorkerInfo {
            address: worker_address.clone(),
            capacity: 1000,
            region: "local".to_string(),
            version: "test".to_string(),
        }).await.expect("Failed to register worker");
        println!("Registered worker{} at {}", i, worker_address);
    }

    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(coordinator.workers().len(), NUM_WORKERS);

    // Load and distribute
    coordinator.load_datasets().await.expect("Failed to load");
    coordinator.distribute_embeddings().await.expect("Failed to distribute");

    println!("Successfully registered {} workers and distributed {} nodes", NUM_WORKERS, TOTAL_NODES);

    // Cleanup
    coord_handle.abort();
    for handle in worker_handles {
        handle.abort();
    }

    std::fs::remove_file(&config.dataset.base_path).ok();
    std::fs::remove_file(&config.dataset.query_path).ok();

    println!("=== Test Passed ===\n");
}

/// Test dynamic bootstrap of 1000 nodes across 5 workers with snapshots
#[tokio::test]
#[ignore]
async fn test_sequential_bootstrap_with_snapshots() {
    const NUM_WORKERS: usize = 5;
    const TOTAL_NODES: usize = 1000;

    println!("\n=== Starting Dynamic Bootstrap Test ===");
    println!("Workers: {}", NUM_WORKERS);
    println!("Total nodes: {}", TOTAL_NODES);
    println!("Strategy: Dynamic flow control with max_bootstrapping = max(1, floor(0.01 * current_active))\n");

    // 1. Create coordinator config
    let config = create_integration_config(NUM_WORKERS);
    let coordinator = Arc::new(Coordinator::<TestSpace>::new(config.clone()).await.expect("Failed to create coordinator"));

    // 2. Generate test data
    let base_vectors = generate_test_data(TOTAL_NODES);
    let query_vectors = generate_test_queries(5);

    write_fvecs(&config.dataset.base_path, &base_vectors).expect("Failed to write base dataset");
    write_fvecs(&config.dataset.query_path, &query_vectors).expect("Failed to write query dataset");

    // 3. Start coordinator server
    let coordinator_clone = coordinator.clone();
    let coordinator_svc = service::CoordinatorService::new(coordinator_clone);
    let coord_addr: SocketAddr = config.coordinator_bind_address.parse().unwrap();

    let coord_handle = tokio::spawn(async move {
        Server::builder()
            .add_service(
                CoordinatorServer::new(coordinator_svc)
                    .max_decoding_message_size(100 * 1024 * 1024) // 100MB
                    .max_encoding_message_size(100 * 1024 * 1024) // 100MB
            )
            .serve(coord_addr)
            .await
            .expect("Coordinator server failed");
    });

    tokio::time::sleep(Duration::from_millis(100)).await;
    println!("Coordinator server started at {}", config.coordinator_bind_address);

    // 4. Start workers
    let mut worker_handles = Vec::new();

    for i in 0..NUM_WORKERS {
        let worker_id = format!("worker{}", i);
        let worker_address: Address = format!("127.0.0.1:{}", 60051 + i).parse().unwrap();
        let coordinator_address = "127.0.0.1:60050".to_string();

        let worker = Worker::<TestSpace>::new(
            worker_id.clone(),
            worker_address.clone(),
            NUM_WORKERS,
            250, // max_actors per worker (1000 nodes / 5 workers = 200, give some headroom)
        );

        // Connect worker to coordinator
        worker.set_coordinator(coordinator_address.clone()).await.expect("Failed to connect");

        // Start worker server
        let worker_addr: SocketAddr = worker_address.to_string().parse().unwrap();
        let worker_server_handle = tokio::spawn(async move {
            Server::builder()
                .add_service(
                    WorkerNodeServer::new(worker)
                        .max_decoding_message_size(100 * 1024 * 1024) // 100MB
                        .max_encoding_message_size(100 * 1024 * 1024) // 100MB
                )
                .serve(worker_addr)
                .await
                .expect("Worker server failed");
        });

        worker_handles.push(worker_server_handle);
        tokio::time::sleep(Duration::from_millis(50)).await;
        println!("Started {}", worker_id);
    }

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Register workers with coordinator
    let coord_client_addr = format!("http://{}", config.coordinator_bind_address);
    let mut coord_client = CoordinatorClient::connect(coord_client_addr).await.expect("Failed to connect to coordinator");
    for i in 0..NUM_WORKERS {
        let worker_address = format!("127.0.0.1:{}", 60051 + i);
        coord_client.register_worker(WorkerInfo {
            address: worker_address.clone(),
            capacity: 1000,
            region: "local".to_string(),
            version: "test".to_string(),
        }).await.expect("Failed to register worker");
        println!("Registered worker{} at {}", i, worker_address);
    }

    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(coordinator.workers().len(), NUM_WORKERS);

    // 5. Load and distribute embeddings
    coordinator.load_datasets().await.expect("Failed to load");
    coordinator.distribute_embeddings().await.expect("Failed to distribute");

    println!("\n6. Dynamic bootstrap with flow control:");
    println!("   Using max_bootstrapping = max(1, floor(0.01 * current_active))");
    println!("   Taking snapshots at logarithmic intervals: 10, 100, 1000 nodes\n");

    let bootstrap_target_idx = 0; // All peers bootstrap to peer 0 (seed peer)

    // Stage 1: Bootstrap first 10 nodes
    println!("   Stage 1: Bootstrapping first 10 nodes...");
    let stage1_peers: Vec<u64> = (0..10).collect();
    coordinator.execute_staged_bootstrap(stage1_peers, bootstrap_target_idx, true).await
        .expect("Failed to execute stage 1 bootstrap");

    println!("   Stage 1 complete. Taking snapshot...");
    let snapshot_10 = test_plan::SnapshotPhase {
        output_path: "bootstrap_10_nodes.json".to_string(),
    };
    coordinator.collect_snapshots(&snapshot_10).await
        .expect("Failed to collect 10-node snapshot");
    println!("   Snapshot saved: {}/bootstrap_10_nodes*.json", config.output_dir);

    // Stage 2: Bootstrap up to 100 nodes (incremental, don't reset)
    println!("\n   Stage 2: Bootstrapping to 100 nodes (90 more)...");
    let stage2_peers: Vec<u64> = (10..100).collect();
    coordinator.execute_staged_bootstrap(stage2_peers, bootstrap_target_idx, false).await
        .expect("Failed to execute stage 2 bootstrap");

    println!("   Stage 2 complete. Taking snapshot...");
    let snapshot_100 = test_plan::SnapshotPhase {
        output_path: "bootstrap_100_nodes.json".to_string(),
    };
    coordinator.collect_snapshots(&snapshot_100).await
        .expect("Failed to collect 100-node snapshot");
    println!("   Snapshot saved: {}/bootstrap_100_nodes*.json", config.output_dir);

    // Stage 3: Bootstrap up to 1000 nodes (incremental, don't reset)
    println!("\n   Stage 3: Bootstrapping to 1000 nodes (900 more)...");
    let stage3_peers: Vec<u64> = (100..TOTAL_NODES as u64).collect();
    coordinator.execute_staged_bootstrap(stage3_peers, bootstrap_target_idx, false).await
        .expect("Failed to execute stage 3 bootstrap");

    println!("\n7. Final snapshot:");
    let snapshot_1000 = test_plan::SnapshotPhase {
        output_path: "bootstrap_1000_nodes.json".to_string(),
    };
    coordinator.collect_snapshots(&snapshot_1000).await
        .expect("Failed to collect final snapshot");
    println!("   Snapshot saved: {}/bootstrap_1000_nodes*.json", config.output_dir);

    // Cleanup
    coord_handle.abort();
    for handle in worker_handles {
        handle.abort();
    }

    std::fs::remove_file(&config.dataset.base_path).ok();
    std::fs::remove_file(&config.dataset.query_path).ok();

    println!("\n=== Test Passed ===\n");
}
