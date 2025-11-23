use protean_dist_sim::coordinator::*;
use protean_dist_sim::coordinator::test_plan::ChurnPatternType;

#[test]
fn test_full_coordinator_config_from_yaml() {
    let yaml = r#"
workers:
  - worker_id: "worker0"
    address: "worker0:50051"
  - worker_id: "worker1"
    address: "worker1:50051"
  - worker_id: "worker2"
    address: "worker2:50052"

dataset:
  base_path: "/data/sift_base.fvecs"
  query_path: "/data/sift_query.fvecs"

sim_config:
  snv_config:
    concurrency_limit: 10
    timeout_ms: 5000
    occlusion_threshold: 0.5
    drift_threshold: 0.1
    target_degree_ratio: 1.5
    dynamism_threshold: 0.2
    exploration_config:
      converge_k: 10
      converge_config:
        search_list_size: 20
        concurrency_limit: 5
        share_floor: 3
        timeout: 1000
      explore_k: 5
      explore_config:
        search_list_size: 10
        concurrency_limit: 3
        share_floor: 2
        timeout: 500
    max_exploration_interval_secs: 60
  max_peers: 1000
  region: "us-east-1"

output_dir: "/results"
coordinator_bind_address: "0.0.0.0:50050"
"#;

    let config: CoordinatorConfig = serde_yaml::from_str(yaml)
        .expect("Failed to deserialize coordinator config");

    // Verify workers
    assert_eq!(config.workers.len(), 3);
    assert_eq!(config.workers[0].worker_id, "worker0");
    assert_eq!(config.workers[0].address, "worker0:50051");
    assert_eq!(config.workers[2].address, "worker2:50052");

    // Verify dataset
    assert_eq!(config.dataset.base_path, "/data/sift_base.fvecs");
    assert_eq!(config.dataset.query_path, "/data/sift_query.fvecs");

    // Verify sim_config
    assert_eq!(config.sim_config.max_peers, 1000);
    assert_eq!(config.sim_config.region, "us-east-1");

    // Verify SNV config
    let snv = &config.sim_config.snv_config;
    assert_eq!(snv.concurrency_limit, 10);
    assert_eq!(snv.timeout_ms, 5000);
    assert!((snv.occlusion_threshold - 0.5).abs() < 0.001);
    assert_eq!(snv.max_exploration_interval_secs, 60);

    // Verify exploration config
    assert_eq!(snv.exploration_config.converge_k, 10);
    assert_eq!(snv.exploration_config.converge_config.search_list_size, 20);
    assert_eq!(snv.exploration_config.explore_k, 5);

    // Verify output settings
    assert_eq!(config.output_dir, "/results");
    assert_eq!(config.coordinator_bind_address, "0.0.0.0:50050");
}

#[test]
fn test_full_test_plan_from_yaml() {
    let yaml = r#"
phases:
  # Initial bootstrap
  - type: Bootstrap
    peer_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    bootstrap_index: 0

  # Wait for convergence
  - type: Wait
    duration_ms: 10000

  # Baseline queries
  - type: Query
    query_indices: [0, 1, 2, 3, 4]
    source_peer_indices: [0, 2, 4, 6, 8]
    k: 10

  # Take baseline snapshot
  - type: Snapshot
    output_path: "baseline_network.json"

  # Add more peers gradually
  - type: Churn
    pattern: GradualJoin
    global_indices: [10, 11, 12, 13, 14]
    bootstrap_indices: [0, 0, 0, 0, 0]
    duration_ms: 15000
    rate_per_second: 0.33

  # Query during/after churn
  - type: Query
    query_indices: [5, 6, 7]
    source_peer_indices: [0, 5, 10]
    k: 10

  # Embedding drift test
  - type: Churn
    pattern: EmbeddingDrift
    global_indices: [0, 1, 2]
    bootstrap_indices: []
    drift_target_indices: [100, 101, 102]
    drift_steps: 10
    duration_ms: 20000
    rate_per_second: 0.0

  # Second bootstrap (adding more peers)
  - type: Bootstrap
    peer_indices: [15, 16, 17, 18, 19]
    bootstrap_index: 0

  # Mass departure
  - type: Churn
    pattern: MassDeparture
    global_indices: [10, 11, 12, 13, 14]
    bootstrap_indices: []
    duration_ms: 0
    rate_per_second: 0.0

  # Final queries
  - type: Query
    query_indices: [8, 9]
    source_peer_indices: [0, 5, 15]
    k: 10

  # Final snapshot
  - type: Snapshot
    output_path: "final_network.json"
"#;

    let plan: TestPlan = serde_yaml::from_str(yaml)
        .expect("Failed to deserialize test plan");

    // Verify phase count and ordering
    assert_eq!(plan.phases.len(), 11);

    // Verify phase types in order
    assert!(matches!(plan.phases[0], TestPhase::Bootstrap(_)));
    assert!(matches!(plan.phases[1], TestPhase::Wait(_)));
    assert!(matches!(plan.phases[2], TestPhase::Query(_)));
    assert!(matches!(plan.phases[3], TestPhase::Snapshot(_)));
    assert!(matches!(plan.phases[4], TestPhase::Churn(_)));
    assert!(matches!(plan.phases[5], TestPhase::Query(_)));
    assert!(matches!(plan.phases[6], TestPhase::Churn(_)));
    assert!(matches!(plan.phases[7], TestPhase::Bootstrap(_))); // Second bootstrap
    assert!(matches!(plan.phases[8], TestPhase::Churn(_)));
    assert!(matches!(plan.phases[9], TestPhase::Query(_)));
    assert!(matches!(plan.phases[10], TestPhase::Snapshot(_)));

    // Verify specific phase details
    if let TestPhase::Bootstrap(b) = &plan.phases[0] {
        assert_eq!(b.peer_indices.len(), 10);
        assert_eq!(b.bootstrap_index, 0);
    } else {
        panic!("Expected Bootstrap phase");
    }

    if let TestPhase::Churn(c) = &plan.phases[4] {
        assert!(matches!(c.pattern, ChurnPatternType::GradualJoin));
        assert_eq!(c.global_indices.len(), 5);
        assert_eq!(c.duration_ms, 15000);
    } else {
        panic!("Expected Churn phase");
    }

    if let TestPhase::Churn(c) = &plan.phases[6] {
        assert!(matches!(c.pattern, ChurnPatternType::EmbeddingDrift));
        assert_eq!(c.drift_target_indices, Some(vec![100, 101, 102]));
        assert_eq!(c.drift_steps, Some(10));
    } else {
        panic!("Expected EmbeddingDrift phase");
    }

    // Verify repeated Bootstrap phase
    if let TestPhase::Bootstrap(b) = &plan.phases[7] {
        assert_eq!(b.peer_indices, vec![15, 16, 17, 18, 19]);
    } else {
        panic!("Expected second Bootstrap phase");
    }
}

#[test]
fn test_config_and_plan_together() {
    // Test that we can load both config and plan for a complete test setup
    let config_yaml = r#"
workers:
  - worker_id: "w0"
    address: "localhost:50051"

dataset:
  base_path: "/tmp/base.fvecs"
  query_path: "/tmp/query.fvecs"

sim_config:
  snv_config:
    concurrency_limit: 5
    timeout_ms: 1000
    occlusion_threshold: 0.5
    drift_threshold: 0.1
    target_degree_ratio: 1.5
    dynamism_threshold: 0.2
    exploration_config:
      converge_k: 5
      converge_config:
        search_list_size: 10
        concurrency_limit: 3
        share_floor: 2
        timeout: 500
      explore_k: 3
      explore_config:
        search_list_size: 5
        concurrency_limit: 2
        share_floor: 1
        timeout: 250
    max_exploration_interval_secs: 30
  max_peers: 100
  region: "local"

output_dir: "/tmp/results"
coordinator_bind_address: "0.0.0.0:50050"
"#;

    let plan_yaml = r#"
phases:
  - type: Bootstrap
    peer_indices: [0, 1, 2]
    bootstrap_index: 0
  - type: Query
    query_indices: [0]
    source_peer_indices: [0]
    k: 5
"#;

    let config: CoordinatorConfig = serde_yaml::from_str(config_yaml).unwrap();
    let plan: TestPlan = serde_yaml::from_str(plan_yaml).unwrap();

    // Verify both loaded successfully
    assert_eq!(config.workers.len(), 1);
    assert_eq!(plan.phases.len(), 2);

    // Verify they're compatible
    assert!(config.dataset.base_path.ends_with(".fvecs"));
    assert!(config.dataset.query_path.ends_with(".fvecs"));
}

#[test]
fn test_minimal_config() {
    // Test minimal valid config
    let yaml = r#"
workers:
  - worker_id: "w0"
    address: "localhost:50051"

dataset:
  base_path: "/data/base.fvecs"
  query_path: "/data/query.fvecs"

sim_config:
  snv_config:
    concurrency_limit: 1
    timeout_ms: 1000
    occlusion_threshold: 0.5
    drift_threshold: 0.1
    target_degree_ratio: 1.0
    dynamism_threshold: 0.1
    exploration_config:
      converge_k: 1
      converge_config:
        search_list_size: 1
        concurrency_limit: 1
        share_floor: 1
        timeout: 100
      explore_k: 1
      explore_config:
        search_list_size: 1
        concurrency_limit: 1
        share_floor: 1
        timeout: 100
    max_exploration_interval_secs: 0
  max_peers: 1
  region: "test"

output_dir: "/tmp"
coordinator_bind_address: "0.0.0.0:50050"
"#;

    let config: CoordinatorConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(config.workers.len(), 1);
    assert_eq!(config.sim_config.max_peers, 1);
}

#[test]
fn test_empty_test_plan() {
    let yaml = r#"
phases: []
"#;

    let plan: TestPlan = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(plan.phases.len(), 0);
}
