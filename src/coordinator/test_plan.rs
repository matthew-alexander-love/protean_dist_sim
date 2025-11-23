use serde::{Deserialize, Serialize};

use super::config::SnvConfig;

/// Peer index specification - supports both explicit lists and ranges
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PeerIndices {
    /// Explicit list of peer indices
    List(Vec<u64>),
    /// Range of peer indices (inclusive)
    Range { start: u64, end: u64 },
}

impl PeerIndices {
    /// Convert to a Vec of indices
    pub fn to_vec(&self) -> Vec<u64> {
        match self {
            PeerIndices::List(list) => list.clone(),
            PeerIndices::Range { start, end } => (*start..=*end).collect(),
        }
    }

    /// Get the number of indices
    pub fn len(&self) -> usize {
        match self {
            PeerIndices::List(list) => list.len(),
            PeerIndices::Range { start, end } => {
                if end >= start {
                    (end - start + 1) as usize
                } else {
                    0
                }
            }
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for PeerIndices {
    fn default() -> Self {
        PeerIndices::List(Vec::new())
    }
}

// Allow comparing PeerIndices with Vec<u64> for tests
impl PartialEq<Vec<u64>> for PeerIndices {
    fn eq(&self, other: &Vec<u64>) -> bool {
        self.to_vec() == *other
    }
}

impl PartialEq<&[u64]> for PeerIndices {
    fn eq(&self, other: &&[u64]) -> bool {
        self.to_vec().as_slice() == *other
    }
}

/// Test plan defining a sequence of operations to execute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestPlan {
    /// Total number of peers available for this test (0 to peer_count-1)
    /// Supports large-scale tests without listing individual indices
    pub peer_count: u64,

    /// Phases to execute in order (can repeat and intermix phase types)
    pub phases: Vec<TestPhase>,
}

/// Test phase - can be Bootstrap, Churn, Query, Snapshot, or Wait
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TestPhase {
    Bootstrap(BootstrapPhase),
    Churn(ChurnPhase),
    Query(QueryPhase),
    Snapshot(SnapshotPhase),
    Wait(WaitPhase),
}

/// Bootstrap phase: Create and connect peers to the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapPhase {
    /// Global indices of peers to create (supports list or range)
    pub peer_indices: PeerIndices,

    /// Bootstrap peer global index (0 = no bootstrap, create network seed)
    pub bootstrap_index: u64,

    /// Optional SNV configuration override for this bootstrap phase
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snv_config_override: Option<SnvConfig>,
}

/// Churn phase: Execute churn patterns (join, leave, drift)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChurnPhase {
    /// Churn pattern type
    pub pattern: ChurnPatternType,

    /// Global indices of peers affected by churn (supports list or range)
    pub global_indices: PeerIndices,

    /// Bootstrap indices for joining peers (supports list or range, empty for non-join patterns)
    #[serde(default)]
    pub bootstrap_indices: PeerIndices,

    /// Duration in milliseconds (for GRADUAL patterns)
    #[serde(default)]
    pub duration_ms: u64,

    /// Rate per second (for GRADUAL patterns)
    #[serde(default)]
    pub rate_per_second: f32,

    /// Target indices for EMBEDDING_DRIFT pattern
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drift_target_indices: Option<Vec<u64>>,

    /// Number of steps for EMBEDDING_DRIFT pattern
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drift_steps: Option<u32>,
}

/// Churn pattern types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ChurnPatternType {
    /// Instant spawn of multiple peers
    FlashCrowd,

    /// Instant deletion of multiple peers
    MassDeparture,

    /// Gradual peer joining over duration
    GradualJoin,

    /// Gradual peer leaving over duration
    GradualLeave,

    /// Gradual embedding drift (LERP interpolation)
    EmbeddingDrift,
}

/// Query phase: Execute distributed queries and calculate recall
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPhase {
    /// Indices into the query dataset (supports list or range)
    pub query_indices: PeerIndices,

    /// Global indices of source peers to query from (supports list or range)
    pub source_peer_indices: PeerIndices,

    /// k for k-NN queries
    pub k: usize,
}

/// Snapshot phase: Collect network state from all workers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotPhase {
    /// Output path for snapshot (relative to output_dir)
    pub output_path: String,
}

/// Wait phase: Pause execution for a duration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaitPhase {
    /// Duration to wait in milliseconds
    pub duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_plan_arbitrary_ordering() {
        let plan = TestPlan {
            peer_count: 6,
            phases: vec![
                TestPhase::Bootstrap(BootstrapPhase {
                    peer_indices: PeerIndices::List(vec![0, 1, 2]),
                    bootstrap_index: 0,
                    snv_config_override: None,
                }),
                TestPhase::Snapshot(SnapshotPhase {
                    output_path: "snapshot1.json".to_string(),
                }),
                TestPhase::Query(QueryPhase {
                    query_indices: PeerIndices::List(vec![0, 1]),
                    source_peer_indices: PeerIndices::List(vec![0, 1]),
                    k: 10,
                }),
                TestPhase::Bootstrap(BootstrapPhase {
                    peer_indices: PeerIndices::List(vec![3, 4, 5]),
                    bootstrap_index: 0,
                    snv_config_override: None,
                }),
                TestPhase::Wait(WaitPhase { duration_ms: 5000 }),
                TestPhase::Query(QueryPhase {
                    query_indices: PeerIndices::List(vec![2, 3]),
                    source_peer_indices: PeerIndices::List(vec![3, 4]),
                    k: 10,
                }),
            ],
        };

        // Serialize to YAML
        let yaml = serde_yaml::to_string(&plan).unwrap();

        // Deserialize back
        let deserialized: TestPlan = serde_yaml::from_str(&yaml).unwrap();

        // Verify phases are in correct order
        assert_eq!(deserialized.phases.len(), 6);
        assert!(matches!(deserialized.phases[0], TestPhase::Bootstrap(_)));
        assert!(matches!(deserialized.phases[1], TestPhase::Snapshot(_)));
        assert!(matches!(deserialized.phases[2], TestPhase::Query(_)));
        assert!(matches!(deserialized.phases[3], TestPhase::Bootstrap(_)));
        assert!(matches!(deserialized.phases[4], TestPhase::Wait(_)));
        assert!(matches!(deserialized.phases[5], TestPhase::Query(_)));
    }

    #[test]
    fn test_bootstrap_phase_yaml() {
        let yaml = r#"
type: Bootstrap
peer_indices: [0, 1, 2, 3, 4]
bootstrap_index: 0
"#;

        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();

        match phase {
            TestPhase::Bootstrap(b) => {
                assert_eq!(b.peer_indices, vec![0, 1, 2, 3, 4]);
                assert_eq!(b.bootstrap_index, 0);
                assert!(b.snv_config_override.is_none());
            }
            _ => panic!("Expected Bootstrap phase"),
        }
    }

    #[test]
    fn test_churn_phase_all_patterns() {
        let patterns = vec![
            (ChurnPatternType::FlashCrowd, "FlashCrowd"),
            (ChurnPatternType::MassDeparture, "MassDeparture"),
            (ChurnPatternType::GradualJoin, "GradualJoin"),
            (ChurnPatternType::GradualLeave, "GradualLeave"),
            (ChurnPatternType::EmbeddingDrift, "EmbeddingDrift"),
        ];

        for (pattern, name) in patterns {
            let phase = ChurnPhase {
                pattern,
                global_indices: PeerIndices::List(vec![10, 11, 12]),
                bootstrap_indices: PeerIndices::List(vec![0, 0, 0]),
                duration_ms: 10000,
                rate_per_second: 0.5,
                drift_target_indices: Some(vec![100, 101, 102]),
                drift_steps: Some(20),
            };

            let yaml = serde_yaml::to_string(&phase).unwrap();
            assert!(yaml.contains(name), "Pattern {} not in YAML", name);

            let deserialized: ChurnPhase = serde_yaml::from_str(&yaml).unwrap();
            assert_eq!(deserialized.global_indices, vec![10, 11, 12]);
            assert_eq!(deserialized.duration_ms, 10000);
        }
    }

    #[test]
    fn test_churn_phase_gradual_join() {
        let yaml = r#"
type: Churn
pattern: GradualJoin
global_indices: [20, 21, 22, 23, 24]
bootstrap_indices: [0, 0, 0, 0, 0]
duration_ms: 15000
rate_per_second: 0.33
"#;

        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();

        match phase {
            TestPhase::Churn(c) => {
                assert!(matches!(c.pattern, ChurnPatternType::GradualJoin));
                assert_eq!(c.global_indices.len(), 5);
                assert_eq!(c.duration_ms, 15000);
                assert!((c.rate_per_second - 0.33).abs() < 0.01);
            }
            _ => panic!("Expected Churn phase"),
        }
    }

    #[test]
    fn test_churn_phase_embedding_drift() {
        let yaml = r#"
type: Churn
pattern: EmbeddingDrift
global_indices: [0, 1, 2]
bootstrap_indices: []
drift_target_indices: [100, 101, 102]
drift_steps: 10
duration_ms: 20000
rate_per_second: 0.0
"#;

        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();

        match phase {
            TestPhase::Churn(c) => {
                assert!(matches!(c.pattern, ChurnPatternType::EmbeddingDrift));
                assert_eq!(c.drift_target_indices, Some(vec![100, 101, 102]));
                assert_eq!(c.drift_steps, Some(10));
                assert_eq!(c.bootstrap_indices, Vec::<u64>::new());
            }
            _ => panic!("Expected Churn phase"),
        }
    }

    #[test]
    fn test_query_phase_yaml() {
        let yaml = r#"
type: Query
query_indices: [0, 1, 2, 3, 4]
source_peer_indices: [0, 5, 10, 15]
k: 10
"#;

        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();

        match phase {
            TestPhase::Query(q) => {
                assert_eq!(q.query_indices, vec![0, 1, 2, 3, 4]);
                assert_eq!(q.source_peer_indices, vec![0, 5, 10, 15]);
                assert_eq!(q.k, 10);
            }
            _ => panic!("Expected Query phase"),
        }
    }

    #[test]
    fn test_snapshot_phase_yaml() {
        let yaml = r#"
type: Snapshot
output_path: "baseline_network.json"
"#;

        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();

        match phase {
            TestPhase::Snapshot(s) => {
                assert_eq!(s.output_path, "baseline_network.json");
            }
            _ => panic!("Expected Snapshot phase"),
        }
    }

    #[test]
    fn test_wait_phase_yaml() {
        let yaml = r#"
type: Wait
duration_ms: 10000
"#;

        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();

        match phase {
            TestPhase::Wait(w) => {
                assert_eq!(w.duration_ms, 10000);
            }
            _ => panic!("Expected Wait phase"),
        }
    }

    #[test]
    fn test_full_test_plan_yaml() {
        let yaml = r#"
peer_count: 8

phases:
  - type: Bootstrap
    peer_indices: [0, 1, 2, 3, 4]
    bootstrap_index: 0

  - type: Wait
    duration_ms: 5000

  - type: Query
    query_indices: [0, 1, 2]
    source_peer_indices: [0, 2, 4]
    k: 10

  - type: Churn
    pattern: GradualJoin
    global_indices: [5, 6, 7]
    bootstrap_indices: [0, 0, 0]
    duration_ms: 10000
    rate_per_second: 0.3

  - type: Snapshot
    output_path: "final_state.json"
"#;

        let plan: TestPlan = serde_yaml::from_str(yaml).unwrap();

        assert_eq!(plan.phases.len(), 5);
        assert!(matches!(plan.phases[0], TestPhase::Bootstrap(_)));
        assert!(matches!(plan.phases[1], TestPhase::Wait(_)));
        assert!(matches!(plan.phases[2], TestPhase::Query(_)));
        assert!(matches!(plan.phases[3], TestPhase::Churn(_)));
        assert!(matches!(plan.phases[4], TestPhase::Snapshot(_)));
    }

    #[test]
    fn test_repeated_phases() {
        // Test that phases can be repeated
        let plan = TestPlan {
            peer_count: 4,
            phases: vec![
                TestPhase::Bootstrap(BootstrapPhase {
                    peer_indices: PeerIndices::List(vec![0, 1]),
                    bootstrap_index: 0,
                    snv_config_override: None,
                }),
                TestPhase::Query(QueryPhase {
                    query_indices: PeerIndices::List(vec![0]),
                    source_peer_indices: PeerIndices::List(vec![0]),
                    k: 10,
                }),
                TestPhase::Bootstrap(BootstrapPhase {
                    peer_indices: PeerIndices::List(vec![2, 3]),
                    bootstrap_index: 0,
                    snv_config_override: None,
                }),
                TestPhase::Query(QueryPhase {
                    query_indices: PeerIndices::List(vec![1]),
                    source_peer_indices: PeerIndices::List(vec![2]),
                    k: 10,
                }),
                TestPhase::Query(QueryPhase {
                    query_indices: PeerIndices::List(vec![2]),
                    source_peer_indices: PeerIndices::List(vec![3]),
                    k: 10,
                }),
            ],
        };

        let yaml = serde_yaml::to_string(&plan).unwrap();
        let deserialized: TestPlan = serde_yaml::from_str(&yaml).unwrap();

        // Count phase types
        let bootstrap_count = deserialized.phases.iter()
            .filter(|p| matches!(p, TestPhase::Bootstrap(_)))
            .count();
        let query_count = deserialized.phases.iter()
            .filter(|p| matches!(p, TestPhase::Query(_)))
            .count();

        assert_eq!(bootstrap_count, 2);
        assert_eq!(query_count, 3);
    }
}
