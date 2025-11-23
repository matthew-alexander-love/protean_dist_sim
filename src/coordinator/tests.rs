use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

use protean::embedding_space::spaces::f32_l2::F32L2Space;

use super::*;
use crate::proto::dist_sim::*;

type TestSpace = F32L2Space<4>;

/// Helper to create a test coordinator config
fn create_test_config() -> CoordinatorConfig {
    CoordinatorConfig {
        workers: vec![
            WorkerConfig {
                worker_id: "worker0".to_string(),
                address: "127.0.0.1:50051".to_string(),
            },
            WorkerConfig {
                worker_id: "worker1".to_string(),
                address: "127.0.0.1:50052".to_string(),
            },
        ],
        dataset: DatasetConfig {
            base_path: "/tmp/test_base.fvecs".to_string(),
            query_path: "/tmp/test_query.fvecs".to_string(),
        },
        sim_config: config::SimConfig {
            snv_config: config::SnvConfig {
                concurrency_limit: 5,
                timeout_ms: 1000,
                occlusion_threshold: 0.5,
                drift_threshold: 0.1,
                target_degree_ratio: 1.5,
                dynamism_threshold: 0.2,
                exploration_config: config::ExplorationConfig {
                    converge_k: 5,
                    converge_config: config::QueryConfig {
                        search_list_size: 10,
                        concurrency_limit: 3,
                        share_floor: 2,
                        timeout: 500,
                    },
                    explore_k: 3,
                    explore_config: config::QueryConfig {
                        search_list_size: 5,
                        concurrency_limit: 2,
                        share_floor: 1,
                        timeout: 250,
                    },
                },
                max_exploration_interval_secs: 30,
            },
            max_peers: 100,
            region: "test".to_string(),
        },
        output_dir: "/tmp/test_output".to_string(),
        coordinator_bind_address: "0.0.0.0:50050".to_string(),
    }
}

#[tokio::test]
async fn test_coordinator_creation() {
    let config = create_test_config();
    let coordinator = Coordinator::<TestSpace>::new(config.clone())
        .await
        .expect("Failed to create coordinator");

    assert_eq!(coordinator.workers().len(), 0); // No workers registered yet
}

#[tokio::test]
async fn test_global_index_to_uuid() {
    let coordinator = Coordinator::<TestSpace>::new(create_test_config()).await.unwrap();

    let uuid = coordinator.global_index_to_uuid(42);

    assert_eq!(uuid.len(), 64);
    assert_eq!(
        u64::from_be_bytes([uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7]]),
        42
    );
}

#[tokio::test]
async fn test_calculate_recall_perfect() {
    let coordinator = Coordinator::<TestSpace>::new(create_test_config()).await.unwrap();

    let ground_truth = vec![1, 2, 3, 4, 5];
    let results = vec![1, 2, 3, 4, 5];

    let recall = coordinator.calculate_recall(&ground_truth, &results, 5);
    assert_eq!(recall, 1.0);
}

#[tokio::test]
async fn test_calculate_recall_partial() {
    let coordinator = Coordinator::<TestSpace>::new(create_test_config()).await.unwrap();

    let ground_truth = vec![1, 2, 3, 4, 5];
    let results = vec![1, 2, 6, 7, 8]; // Only 2 matches

    let recall = coordinator.calculate_recall(&ground_truth, &results, 5);
    assert!((recall - 0.4).abs() < 0.01); // 2/5 = 0.4
}

#[tokio::test]
async fn test_calculate_recall_no_match() {
    let coordinator = Coordinator::<TestSpace>::new(create_test_config()).await.unwrap();

    let ground_truth = vec![1, 2, 3, 4, 5];
    let results = vec![6, 7, 8, 9, 10]; // No matches

    let recall = coordinator.calculate_recall(&ground_truth, &results, 5);
    assert_eq!(recall, 0.0);
}

#[tokio::test]
async fn test_calculate_recall_empty() {
    let coordinator = Coordinator::<TestSpace>::new(create_test_config()).await.unwrap();

    let ground_truth: Vec<u64> = vec![];
    let results: Vec<u64> = vec![];

    let recall = coordinator.calculate_recall(&ground_truth, &results, 0);
    assert!(recall.is_nan() || recall == 0.0);
}

#[tokio::test]
async fn test_assign_peers_to_workers() {
    let mut config = create_test_config();
    let base_path = "/tmp/test_assign_peers_base.fvecs";
    let query_path = "/tmp/test_assign_peers_query.fvecs";

    // Create a small dataset
    let mut base_file = File::create(base_path).unwrap();
    for i in 0..10 {
        base_file.write_all(&4i32.to_le_bytes()).unwrap();
        for j in 0..4 {
            base_file.write_all(&((i * 4 + j) as f32).to_le_bytes()).unwrap();
        }
    }

    let mut query_file = File::create(query_path).unwrap();
    for i in 0..2 {
        query_file.write_all(&4i32.to_le_bytes()).unwrap();
        for j in 0..4 {
            query_file.write_all(&((i * 4 + j) as f32).to_le_bytes()).unwrap();
        }
    }

    config.dataset.base_path = base_path.to_string();
    config.dataset.query_path = query_path.to_string();

    let coordinator = Coordinator::<TestSpace>::new(config).await.unwrap();
    coordinator.load_datasets().await.unwrap();

    let peer_indices = vec![0, 1, 2, 3, 4, 5];
    let assignments = coordinator
        .assign_peers_to_workers(&peer_indices)
        .await
        .expect("Failed to assign peers");

    // With 2 workers, peers should be distributed
    assert!(assignments.len() <= 2);

    // All peers should be assigned
    let total_assigned: usize = assignments.values().map(|v| v.len()).sum();
    assert_eq!(total_assigned, 6);

    // Cleanup
    std::fs::remove_file(base_path).ok();
    std::fs::remove_file(query_path).ok();
}

#[tokio::test]
async fn test_config_to_proto() {
    let config = create_test_config();
    let coordinator = Coordinator::<TestSpace>::new(config.clone()).await.unwrap();

    let proto = coordinator.config_to_proto();

    assert!(proto.snv_config.is_some());
    assert_eq!(proto.max_peers, 100);
    assert_eq!(proto.region, "test");

    let snv = proto.snv_config.unwrap();
    assert_eq!(snv.concurrency_limit, 5);
    assert_eq!(snv.timeout_ms, 1000);

    assert!(snv.exploration_config.is_some());
    let exploration = snv.exploration_config.unwrap();
    assert_eq!(exploration.converge_k, 5);
    assert_eq!(exploration.explore_k, 3);
}

#[tokio::test]
async fn test_wait_for_workers_timeout() {
    let config = create_test_config();
    let coordinator = Coordinator::<TestSpace>::new(config)
        .await
        .expect("Failed to create coordinator");

    // Should timeout since no workers will register
    let result = coordinator.wait_for_workers(Duration::from_millis(100)).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Timeout"));
}

#[test]
fn test_query_result_serialization() {
    let result = coordinator::QueryResult {
        query_idx: 0,
        source_peer_idx: 42,
        k: 10,
        results: vec![1, 2, 3],
        ground_truth: vec![1, 2, 4],
        recall: 0.67,
        latency_ms: 100,
    };

    // Should be serializable to JSON
    let json = serde_json::to_string(&result).expect("Failed to serialize");
    assert!(json.contains("query_idx"));
    assert!(json.contains("recall"));

    // Should be deserializable
    let deserialized: coordinator::QueryResult =
        serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(deserialized.query_idx, 0);
    assert_eq!(deserialized.k, 10);
}

// Test dataset loading requires actual .fvecs files, so we'll create temp files
#[tokio::test]
async fn test_load_datasets() {
    use std::fs::File;
    use std::io::Write;

    // Create temp .fvecs files
    let base_path = "/tmp/test_load_base.fvecs";
    let query_path = "/tmp/test_load_query.fvecs";

    // Write simple 4D vectors
    let mut base_file = File::create(base_path).unwrap();
    for i in 0..5 {
        base_file.write_all(&4i32.to_le_bytes()).unwrap();
        for j in 0..4 {
            base_file
                .write_all(&((i * 4 + j) as f32).to_le_bytes())
                .unwrap();
        }
    }

    let mut query_file = File::create(query_path).unwrap();
    for i in 0..2 {
        query_file.write_all(&4i32.to_le_bytes()).unwrap();
        for j in 0..4 {
            query_file
                .write_all(&((i * 4 + j + 100) as f32).to_le_bytes())
                .unwrap();
        }
    }

    let mut config = create_test_config();
    config.dataset.base_path = base_path.to_string();
    config.dataset.query_path = query_path.to_string();

    let coordinator = Coordinator::<TestSpace>::new(config)
        .await
        .expect("Failed to create coordinator");

    // Test that dataset loading succeeds - we can't directly access the private fields
    // but we can verify the operation succeeded
    coordinator
        .load_datasets()
        .await
        .expect("Failed to load datasets");

    // Cleanup
    std::fs::remove_file(base_path).ok();
    std::fs::remove_file(query_path).ok();
}

#[tokio::test]
async fn test_get_worker_for_peer() {
    let mut config = create_test_config();
    let base_path = "/tmp/test_worker_assignment_base.fvecs";
    let query_path = "/tmp/test_worker_assignment_query.fvecs";

    // Create dataset with 100 vectors
    let mut base_file = File::create(base_path).unwrap();
    for i in 0..100 {
        base_file.write_all(&4i32.to_le_bytes()).unwrap();
        for j in 0..4 {
            base_file.write_all(&((i * 4 + j) as f32).to_le_bytes()).unwrap();
        }
    }

    // Create query file with 2 vectors
    let mut query_file = File::create(query_path).unwrap();
    for i in 0..2 {
        query_file.write_all(&4i32.to_le_bytes()).unwrap();
        for j in 0..4 {
            query_file
                .write_all(&((i * 4 + j + 100) as f32).to_le_bytes())
                .unwrap();
        }
    }

    config.dataset.base_path = base_path.to_string();
    config.dataset.query_path = query_path.to_string();
    let coordinator = Coordinator::<TestSpace>::new(config)
        .await
        .expect("Failed to create coordinator");

    coordinator.load_datasets().await.unwrap();

    // With 100 embeddings and 2 workers, first 50 go to worker0, next 50 to worker1
    let worker0 = coordinator.get_worker_for_peer(0).await.unwrap();
    assert_eq!(worker0, "worker0");

    let worker1 = coordinator.get_worker_for_peer(75).await.unwrap();
    assert_eq!(worker1, "worker1");

    std::fs::remove_file(base_path).ok();
    std::fs::remove_file(query_path).ok();
}
