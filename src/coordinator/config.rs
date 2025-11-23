use serde::{Deserialize, Serialize};

/// Main coordinator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// List of worker configurations
    pub workers: Vec<WorkerConfig>,

    /// Dataset paths (cloud-compatible)
    pub dataset: DatasetConfig,

    /// Simulation configuration (passed to all workers)
    pub sim_config: SimConfig,

    /// Output directory for results
    pub output_dir: String,

    /// Coordinator gRPC bind address
    pub coordinator_bind_address: String,
}

/// Worker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Worker identifier
    pub worker_id: String,

    /// Worker gRPC address (e.g., "worker0:50051" or "127.0.0.1:50051")
    pub address: String,
}

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Path to base dataset (.fvecs format)
    pub base_path: String,

    /// Path to query dataset (.fvecs format)
    pub query_path: String,

    // NOTE: No groundtruth_path - calculated dynamically via TrueQuery
}

/// Simulation configuration (mirrors SimConfigProto)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    /// SNV configuration
    pub snv_config: SnvConfig,

    /// Maximum peers this worker can host
    pub max_peers: u32,

    /// Region/zone identifier
    pub region: String,
}

/// SNV configuration (mirrors SnvConfigProto)
/// All fields exposed for hyperparameter sweep configurability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnvConfig {
    /// Concurrency limit for SNV operations
    pub concurrency_limit: u32,

    /// Timeout in milliseconds for operations
    pub timeout_ms: u64,

    /// Occlusion threshold for pruning
    pub occlusion_threshold: f32,

    /// Drift threshold for detecting embedding changes
    pub drift_threshold: f32,

    /// Target degree ratio for network topology
    pub target_degree_ratio: f32,

    /// Dynamism threshold for triggering rebalancing
    pub dynamism_threshold: f32,

    /// Exploration configuration
    pub exploration_config: ExplorationConfig,

    /// Maximum interval between periodic explorations (0 = disabled)
    pub max_exploration_interval_secs: u64,
}

/// Exploration configuration for converge and explore queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationConfig {
    /// k for convergence queries (k-NN)
    pub converge_k: u32,

    /// Configuration for convergence queries
    pub converge_config: QueryConfig,

    /// k for exploration queries (k-FN)
    pub explore_k: u32,

    /// Configuration for exploration queries
    pub explore_config: QueryConfig,
}

/// Query execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    /// Search list size for query execution
    pub search_list_size: u32,

    /// Concurrency limit for parallel queries
    pub concurrency_limit: u32,

    /// Share floor parameter
    pub share_floor: u32,

    /// Timeout in milliseconds
    pub timeout: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_config_yaml_roundtrip() {
        let config = CoordinatorConfig {
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
                base_path: "/data/base.fvecs".to_string(),
                query_path: "/data/query.fvecs".to_string(),
            },
            sim_config: SimConfig {
                snv_config: SnvConfig {
                    concurrency_limit: 10,
                    timeout_ms: 5000,
                    occlusion_threshold: 0.5,
                    drift_threshold: 0.1,
                    target_degree_ratio: 1.5,
                    dynamism_threshold: 0.2,
                    exploration_config: ExplorationConfig {
                        converge_k: 10,
                        converge_config: QueryConfig {
                            search_list_size: 20,
                            concurrency_limit: 5,
                            share_floor: 3,
                            timeout: 1000,
                        },
                        explore_k: 5,
                        explore_config: QueryConfig {
                            search_list_size: 10,
                            concurrency_limit: 3,
                            share_floor: 2,
                            timeout: 500,
                        },
                    },
                    max_exploration_interval_secs: 60,
                },
                max_peers: 1000,
                region: "us-east-1".to_string(),
            },
            output_dir: "/results".to_string(),
            coordinator_bind_address: "0.0.0.0:50050".to_string(),
        };

        // Serialize to YAML
        let yaml = serde_yaml::to_string(&config).expect("Failed to serialize");

        // Verify it contains expected fields
        assert!(yaml.contains("worker0"));
        assert!(yaml.contains("127.0.0.1:50051"));
        assert!(yaml.contains("base_path"));
        assert!(yaml.contains("max_peers"));
        assert!(yaml.contains("us-east-1"));

        // Deserialize back
        let deserialized: CoordinatorConfig = serde_yaml::from_str(&yaml)
            .expect("Failed to deserialize");

        // Verify round-trip
        assert_eq!(deserialized.workers.len(), 2);
        assert_eq!(deserialized.workers[0].worker_id, "worker0");
        assert_eq!(deserialized.dataset.base_path, "/data/base.fvecs");
        assert_eq!(deserialized.sim_config.max_peers, 1000);
        assert_eq!(deserialized.sim_config.region, "us-east-1");
    }

    #[test]
    fn test_worker_config_deserialization() {
        let yaml = r#"
worker_id: "test_worker"
address: "192.168.1.100:8080"
"#;

        let config: WorkerConfig = serde_yaml::from_str(yaml)
            .expect("Failed to deserialize WorkerConfig");

        assert_eq!(config.worker_id, "test_worker");
        assert_eq!(config.address, "192.168.1.100:8080");
    }

    #[test]
    fn test_dataset_config_no_groundtruth() {
        let yaml = r#"
base_path: "/path/to/base.fvecs"
query_path: "/path/to/query.fvecs"
"#;

        let config: DatasetConfig = serde_yaml::from_str(yaml)
            .expect("Failed to deserialize DatasetConfig");

        assert_eq!(config.base_path, "/path/to/base.fvecs");
        assert_eq!(config.query_path, "/path/to/query.fvecs");
        // Verify no groundtruth field exists (compile-time check)
    }

    #[test]
    fn test_snv_config_all_fields() {
        let config = SnvConfig {
            concurrency_limit: 15,
            timeout_ms: 10000,
            occlusion_threshold: 0.75,
            drift_threshold: 0.15,
            target_degree_ratio: 2.0,
            dynamism_threshold: 0.3,
            exploration_config: ExplorationConfig {
                converge_k: 20,
                converge_config: QueryConfig {
                    search_list_size: 50,
                    concurrency_limit: 10,
                    share_floor: 5,
                    timeout: 2000,
                },
                explore_k: 15,
                explore_config: QueryConfig {
                    search_list_size: 30,
                    concurrency_limit: 8,
                    share_floor: 4,
                    timeout: 1500,
                },
            },
            max_exploration_interval_secs: 120,
        };

        // Serialize and deserialize
        let yaml = serde_yaml::to_string(&config).unwrap();
        let deserialized: SnvConfig = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(deserialized.concurrency_limit, 15);
        assert_eq!(deserialized.timeout_ms, 10000);
        assert!((deserialized.occlusion_threshold - 0.75).abs() < 0.001);
        assert_eq!(deserialized.exploration_config.converge_k, 20);
        assert_eq!(deserialized.max_exploration_interval_secs, 120);
    }

    #[test]
    fn test_query_config_serialization() {
        let config = QueryConfig {
            search_list_size: 100,
            concurrency_limit: 20,
            share_floor: 10,
            timeout: 5000,
        };

        let yaml = serde_yaml::to_string(&config).unwrap();
        let deserialized: QueryConfig = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(deserialized.search_list_size, 100);
        assert_eq!(deserialized.concurrency_limit, 20);
        assert_eq!(deserialized.share_floor, 10);
        assert_eq!(deserialized.timeout, 5000);
    }
}
