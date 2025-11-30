use std::fmt;
use std::path::Path;
use std::time::Duration;

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

use protean::{ExplorationConfig, QueryConfig, SnvConfig};


/// Deserialize QueryConfig from YAML (timeout_ms -> Duration)
fn deserialize_query_config<'de, D>(deserializer: D) -> Result<QueryConfig, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct QueryConfigHelper {
        #[serde(default = "default_search_list_size")]
        search_list_size: usize,
        #[serde(default = "default_query_concurrency")]
        concurrency_limit: usize,
        #[serde(default = "default_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_share_floor")]
        share_floor: usize,
    }

    let helper = QueryConfigHelper::deserialize(deserializer)?;
    Ok(QueryConfig {
        search_list_size: helper.search_list_size,
        concurrency_limit: helper.concurrency_limit,
        timeout: Duration::from_millis(helper.timeout_ms),
        share_floor: helper.share_floor,
    })
}

/// Deserialize Option<QueryConfig> from YAML
fn deserialize_optional_query_config<'de, D>(deserializer: D) -> Result<Option<QueryConfig>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct QueryConfigHelper {
        #[serde(default = "default_search_list_size")]
        search_list_size: usize,
        #[serde(default = "default_query_concurrency")]
        concurrency_limit: usize,
        #[serde(default = "default_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_share_floor")]
        share_floor: usize,
    }

    let helper: Option<QueryConfigHelper> = Option::deserialize(deserializer)?;
    Ok(helper.map(|h| QueryConfig {
        search_list_size: h.search_list_size,
        concurrency_limit: h.concurrency_limit,
        timeout: Duration::from_millis(h.timeout_ms),
        share_floor: h.share_floor,
    }))
}

/// Deserialize ExplorationConfig from YAML
fn deserialize_exploration_config<'de, D>(deserializer: D) -> Result<ExplorationConfig, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct QueryConfigHelper {
        #[serde(default = "default_search_list_size")]
        search_list_size: usize,
        #[serde(default = "default_query_concurrency")]
        concurrency_limit: usize,
        #[serde(default = "default_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_share_floor")]
        share_floor: usize,
    }

    #[derive(Deserialize)]
    struct ExplorationConfigHelper {
        #[serde(default = "default_converge_k")]
        converge_k: usize,
        #[serde(default)]
        converge_config: Option<QueryConfigHelper>,
        #[serde(default)]
        explore_k: usize,
        #[serde(default)]
        explore_config: Option<QueryConfigHelper>,
    }

    let helper = ExplorationConfigHelper::deserialize(deserializer)?;

    let converge_config = helper.converge_config.map(|h| QueryConfig {
        search_list_size: h.search_list_size,
        concurrency_limit: h.concurrency_limit,
        timeout: Duration::from_millis(h.timeout_ms),
        share_floor: h.share_floor,
    }).unwrap_or_default();

    let explore_config = helper.explore_config.map(|h| QueryConfig {
        search_list_size: h.search_list_size,
        concurrency_limit: h.concurrency_limit,
        timeout: Duration::from_millis(h.timeout_ms),
        share_floor: h.share_floor,
    }).unwrap_or_else(|| QueryConfig::default().with_search_list_size(0));

    Ok(ExplorationConfig {
        converge_k: helper.converge_k,
        converge_config,
        explore_k: helper.explore_k,
        explore_config,
    })
}

/// Deserialize SnvConfig from YAML (timeout_ms, max_exploration_interval_ms -> Duration)
fn deserialize_snv_config<'de, D>(deserializer: D) -> Result<SnvConfig, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct QueryConfigHelper {
        #[serde(default = "default_search_list_size")]
        search_list_size: usize,
        #[serde(default = "default_query_concurrency")]
        concurrency_limit: usize,
        #[serde(default = "default_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_share_floor")]
        share_floor: usize,
    }

    #[derive(Deserialize)]
    struct ExplorationConfigHelper {
        #[serde(default = "default_converge_k")]
        converge_k: usize,
        #[serde(default)]
        converge_config: Option<QueryConfigHelper>,
        #[serde(default)]
        explore_k: usize,
        #[serde(default)]
        explore_config: Option<QueryConfigHelper>,
    }

    #[derive(Deserialize)]
    struct SnvConfigHelper {
        #[serde(default = "default_snv_concurrency")]
        concurrency_limit: usize,
        #[serde(default = "default_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_occlusion_threshold")]
        occlusion_threshold: f32,
        #[serde(default = "default_drift_threshold")]
        drift_threshold: f32,
        #[serde(default = "default_target_degree_ratio")]
        target_degree_ratio: f32,
        #[serde(default = "default_dynamism_threshold")]
        dynamism_threshold: f32,
        #[serde(default)]
        exploration_config: Option<ExplorationConfigHelper>,
        #[serde(default = "default_max_exploration_interval_ms")]
        max_exploration_interval_ms: Option<u64>,
    }

    let helper = SnvConfigHelper::deserialize(deserializer)?;

    let exploration_config = if let Some(exp) = helper.exploration_config {
        let converge_config = exp.converge_config.map(|h| QueryConfig {
            search_list_size: h.search_list_size,
            concurrency_limit: h.concurrency_limit,
            timeout: Duration::from_millis(h.timeout_ms),
            share_floor: h.share_floor,
        }).unwrap_or_default();

        let explore_config = exp.explore_config.map(|h| QueryConfig {
            search_list_size: h.search_list_size,
            concurrency_limit: h.concurrency_limit,
            timeout: Duration::from_millis(h.timeout_ms),
            share_floor: h.share_floor,
        }).unwrap_or_else(|| QueryConfig::default().with_search_list_size(0));

        ExplorationConfig {
            converge_k: exp.converge_k,
            converge_config,
            explore_k: exp.explore_k,
            explore_config,
        }
    } else {
        ExplorationConfig::default()
    };

    Ok(SnvConfig {
        concurrency_limit: helper.concurrency_limit,
        timeout: Duration::from_millis(helper.timeout_ms),
        occlusion_threshold: helper.occlusion_threshold,
        drift_threshold: helper.drift_threshold,
        target_degree_ratio: helper.target_degree_ratio,
        dynamism_threshold: helper.dynamism_threshold,
        exploration_config,
        max_exploration_interval: helper.max_exploration_interval_ms.map(Duration::from_millis),
    })
}

/// Deserialize Option<SnvConfig> from YAML
fn deserialize_optional_snv_config<'de, D>(deserializer: D) -> Result<Option<SnvConfig>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct QueryConfigHelper {
        #[serde(default = "default_search_list_size")]
        search_list_size: usize,
        #[serde(default = "default_query_concurrency")]
        concurrency_limit: usize,
        #[serde(default = "default_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_share_floor")]
        share_floor: usize,
    }

    #[derive(Deserialize)]
    struct ExplorationConfigHelper {
        #[serde(default = "default_converge_k")]
        converge_k: usize,
        #[serde(default)]
        converge_config: Option<QueryConfigHelper>,
        #[serde(default)]
        explore_k: usize,
        #[serde(default)]
        explore_config: Option<QueryConfigHelper>,
    }

    #[derive(Deserialize)]
    struct SnvConfigHelper {
        #[serde(default = "default_snv_concurrency")]
        concurrency_limit: usize,
        #[serde(default = "default_timeout_ms")]
        timeout_ms: u64,
        #[serde(default = "default_occlusion_threshold")]
        occlusion_threshold: f32,
        #[serde(default = "default_drift_threshold")]
        drift_threshold: f32,
        #[serde(default = "default_target_degree_ratio")]
        target_degree_ratio: f32,
        #[serde(default = "default_dynamism_threshold")]
        dynamism_threshold: f32,
        #[serde(default)]
        exploration_config: Option<ExplorationConfigHelper>,
        #[serde(default = "default_max_exploration_interval_ms")]
        max_exploration_interval_ms: Option<u64>,
    }

    let helper: Option<SnvConfigHelper> = Option::deserialize(deserializer)?;

    Ok(helper.map(|helper| {
        let exploration_config = if let Some(exp) = helper.exploration_config {
            let converge_config = exp.converge_config.map(|h| QueryConfig {
                search_list_size: h.search_list_size,
                concurrency_limit: h.concurrency_limit,
                timeout: Duration::from_millis(h.timeout_ms),
                share_floor: h.share_floor,
            }).unwrap_or_default();

            let explore_config = exp.explore_config.map(|h| QueryConfig {
                search_list_size: h.search_list_size,
                concurrency_limit: h.concurrency_limit,
                timeout: Duration::from_millis(h.timeout_ms),
                share_floor: h.share_floor,
            }).unwrap_or_else(|| QueryConfig::default().with_search_list_size(0));

            ExplorationConfig {
                converge_k: exp.converge_k,
                converge_config,
                explore_k: exp.explore_k,
                explore_config,
            }
        } else {
            ExplorationConfig::default()
        };

        SnvConfig {
            concurrency_limit: helper.concurrency_limit,
            timeout: Duration::from_millis(helper.timeout_ms),
            occlusion_threshold: helper.occlusion_threshold,
            drift_threshold: helper.drift_threshold,
            target_degree_ratio: helper.target_degree_ratio,
            dynamism_threshold: helper.dynamism_threshold,
            exploration_config,
            max_exploration_interval: helper.max_exploration_interval_ms.map(Duration::from_millis),
        }
    }))
}

// Default value functions
fn default_search_list_size() -> usize { 50 }
fn default_query_concurrency() -> usize { 5 }
fn default_timeout_ms() -> u64 { 30000 }
fn default_share_floor() -> usize { 5 }
fn default_converge_k() -> usize { 50 }
fn default_snv_concurrency() -> usize { 3 }
fn default_occlusion_threshold() -> f32 { 1.2 }
fn default_drift_threshold() -> f32 { 0.1 }
fn default_target_degree_ratio() -> f32 { 2.0 }
fn default_dynamism_threshold() -> f32 { 0.2 }
fn default_max_exploration_interval_ms() -> Option<u64> { Some(300000) }

fn default_query_config() -> QueryConfig {
    QueryConfig::default()
}

// ============================================================================
// Index range deserializers
// ============================================================================

/// Helper to deserialize either a list [1, 2, 3] or a range {start: 0, end: 100}
fn deserialize_index_range<'de, D>(deserializer: D) -> Result<Vec<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    struct IndexRangeVisitor;

    impl<'de> Visitor<'de> for IndexRangeVisitor {
        type Value = Vec<u64>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a list of indices or a range with start/end")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut indices = Vec::new();
            while let Some(val) = seq.next_element()? {
                indices.push(val);
            }
            Ok(indices)
        }

        fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut start: Option<u64> = None;
            let mut end: Option<u64> = None;

            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "start" => start = Some(map.next_value()?),
                    "end" => end = Some(map.next_value()?),
                    _ => {
                        let _ = map.next_value::<de::IgnoredAny>()?;
                    }
                }
            }

            let start = start.ok_or_else(|| de::Error::missing_field("start"))?;
            let end = end.ok_or_else(|| de::Error::missing_field("end"))?;

            Ok((start..=end).collect())
        }
    }

    deserializer.deserialize_any(IndexRangeVisitor)
}

/// Helper to deserialize Option<Vec<u64>> with range support
fn deserialize_optional_index_range<'de, D>(deserializer: D) -> Result<Option<Vec<u64>>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptionalIndexRangeVisitor;

    impl<'de> Visitor<'de> for OptionalIndexRangeVisitor {
        type Value = Option<Vec<u64>>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("null, a list of indices, or a range with start/end")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut indices = Vec::new();
            while let Some(val) = seq.next_element()? {
                indices.push(val);
            }
            Ok(Some(indices))
        }

        fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut start: Option<u64> = None;
            let mut end: Option<u64> = None;

            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "start" => start = Some(map.next_value()?),
                    "end" => end = Some(map.next_value()?),
                    _ => {
                        let _ = map.next_value::<de::IgnoredAny>()?;
                    }
                }
            }

            let start = start.ok_or_else(|| de::Error::missing_field("start"))?;
            let end = end.ok_or_else(|| de::Error::missing_field("end"))?;

            Ok(Some((start..=end).collect()))
        }
    }

    deserializer.deserialize_any(OptionalIndexRangeVisitor)
}

/// Test phase - can be Bootstrap, Churn, Query, Snapshot, or Wait
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum TestPhase {
    Bootstrap(BootstrapPhase),
    Churn(ChurnPhase),
    Query(QueryPhase),
    Snapshot(SnapshotPhase),
    Wait(WaitPhase),
}

/// Test plan defining a sequence of operations to execute
#[derive(Debug, Clone, Deserialize)]
pub struct TestPlan {
    /// Total number of peers available for this test (0 to peer_count-1)
    pub peer_count: u64,

    /// Bootstrap server indices - created first before any phases, used as entry points.
    /// First server is isolated, each subsequent one bootstraps to the previous.
    #[serde(default, deserialize_with = "deserialize_index_range")]
    pub bootstrap_server_indices: Vec<u64>,

    /// Phases to execute in order (can repeat and intermix phase types)
    pub phases: Vec<TestPhase>,
}

impl TestPlan {
    /// Load a TestPlan from a YAML file
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let plan: TestPlan = serde_yaml::from_str(&contents)?;
        Ok(plan)
    }

    /// Parse a TestPlan from a YAML string
    pub fn from_yaml_str(yaml: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }
}

/// Bootstrap phase: Create and connect peers to the network
#[derive(Debug, Clone, Deserialize)]
pub struct BootstrapPhase {
    /// Global indices of peers to create (supports list or range)
    #[serde(deserialize_with = "deserialize_index_range")]
    pub peer_indices: Vec<u64>,

    /// Bootstrap peer global index (0 = no bootstrap, create network seed)
    pub bootstrap_index: u64,

    /// Optional SNV configuration override for this bootstrap phase
    #[serde(default, deserialize_with = "deserialize_optional_snv_config")]
    pub snv_config_override: Option<SnvConfig>,
}

/// Churn phase: Execute churn patterns (join, leave, drift)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChurnPhase {
    /// Churn pattern type
    pub pattern: ChurnPatternType,

    /// Global indices of peers affected by churn (supports list or range)
    #[serde(deserialize_with = "deserialize_index_range")]
    pub global_indices: Vec<u64>,

    /// Bootstrap indices for joining peers (supports list or range, empty for non-join patterns)
    #[serde(default, deserialize_with = "deserialize_index_range")]
    pub bootstrap_indices: Vec<u64>,

    /// Duration in milliseconds (for GRADUAL patterns)
    #[serde(default)]
    pub duration_ms: u64,

    /// Rate per second (for GRADUAL patterns)
    #[serde(default)]
    pub rate_per_second: f32,

    /// Target indices for EMBEDDING_DRIFT pattern
    #[serde(default, deserialize_with = "deserialize_optional_index_range", skip_serializing_if = "Option::is_none")]
    pub drift_target_indices: Option<Vec<u64>>,

    /// Number of steps for EMBEDDING_DRIFT pattern
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
#[derive(Debug, Clone, Deserialize)]
pub struct QueryPhase {
    /// Indices into the query dataset (supports list or range)
    #[serde(deserialize_with = "deserialize_index_range")]
    pub query_indices: Vec<u64>,

    /// Global indices of source peers to query from (supports list or range)
    #[serde(deserialize_with = "deserialize_index_range")]
    pub source_peer_indices: Vec<u64>,

    /// k for k-NN queries
    pub k: usize,

    /// Query execution configuration (search_list_size, concurrency_limit, timeout_ms, share_floor)
    #[serde(default = "default_query_config", deserialize_with = "deserialize_query_config")]
    pub config: QueryConfig,
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
    fn test_parse_bootstrap_phase() {
        let yaml = r#"
type: Bootstrap
peer_indices: [0, 1, 2, 3, 4]
bootstrap_index: 0
"#;
        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();
        match phase {
            TestPhase::Bootstrap(bp) => {
                assert_eq!(bp.peer_indices, vec![0, 1, 2, 3, 4]);
                assert_eq!(bp.bootstrap_index, 0);
                assert!(bp.snv_config_override.is_none());
            }
            _ => panic!("Expected Bootstrap phase"),
        }
    }

    #[test]
    fn test_parse_bootstrap_with_range() {
        let yaml = r#"
type: Bootstrap
peer_indices:
  start: 0
  end: 99
bootstrap_index: 0
"#;
        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();
        match phase {
            TestPhase::Bootstrap(bp) => {
                assert_eq!(bp.peer_indices.len(), 100);
                assert_eq!(bp.peer_indices[0], 0);
                assert_eq!(bp.peer_indices[99], 99);
            }
            _ => panic!("Expected Bootstrap phase"),
        }
    }

    #[test]
    fn test_parse_query_phase_with_config() {
        let yaml = r#"
type: Query
query_indices: [0, 1, 2]
source_peer_indices: [10, 20, 30]
k: 10
config:
  search_list_size: 100
  concurrency_limit: 10
  timeout_ms: 5000
  share_floor: 3
"#;
        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();
        match phase {
            TestPhase::Query(qp) => {
                assert_eq!(qp.query_indices, vec![0, 1, 2]);
                assert_eq!(qp.source_peer_indices, vec![10, 20, 30]);
                assert_eq!(qp.k, 10);
                assert_eq!(qp.config.search_list_size, 100);
                assert_eq!(qp.config.concurrency_limit, 10);
                assert_eq!(qp.config.timeout, Duration::from_millis(5000));
                assert_eq!(qp.config.share_floor, 3);
            }
            _ => panic!("Expected Query phase"),
        }
    }

    #[test]
    fn test_parse_query_phase_default_config() {
        let yaml = r#"
type: Query
query_indices: [0]
source_peer_indices: [0]
k: 5
"#;
        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();
        match phase {
            TestPhase::Query(qp) => {
                // Check defaults from default_* functions
                assert_eq!(qp.config.search_list_size, 50);
                assert_eq!(qp.config.concurrency_limit, 5);
                assert_eq!(qp.config.timeout, Duration::from_millis(30000));
                assert_eq!(qp.config.share_floor, 5);
            }
            _ => panic!("Expected Query phase"),
        }
    }

    #[test]
    fn test_parse_churn_flash_crowd() {
        let yaml = r#"
type: Churn
pattern: FlashCrowd
global_indices: [100, 101, 102]
bootstrap_indices: [0]
"#;
        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();
        match phase {
            TestPhase::Churn(cp) => {
                assert!(matches!(cp.pattern, ChurnPatternType::FlashCrowd));
                assert_eq!(cp.global_indices, vec![100, 101, 102]);
                assert_eq!(cp.bootstrap_indices, vec![0]);
            }
            _ => panic!("Expected Churn phase"),
        }
    }

    #[test]
    fn test_parse_churn_gradual_join() {
        let yaml = r#"
type: Churn
pattern: GradualJoin
global_indices:
  start: 100
  end: 199
bootstrap_indices: [0, 1, 2]
duration_ms: 60000
rate_per_second: 1.5
"#;
        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();
        match phase {
            TestPhase::Churn(cp) => {
                assert!(matches!(cp.pattern, ChurnPatternType::GradualJoin));
                assert_eq!(cp.global_indices.len(), 100);
                assert_eq!(cp.duration_ms, 60000);
                assert!((cp.rate_per_second - 1.5).abs() < 0.001);
            }
            _ => panic!("Expected Churn phase"),
        }
    }

    #[test]
    fn test_parse_churn_embedding_drift() {
        let yaml = r#"
type: Churn
pattern: EmbeddingDrift
global_indices: [0, 1, 2]
drift_target_indices: [100, 101, 102]
drift_steps: 50
duration_ms: 10000
"#;
        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();
        match phase {
            TestPhase::Churn(cp) => {
                assert!(matches!(cp.pattern, ChurnPatternType::EmbeddingDrift));
                assert_eq!(cp.global_indices, vec![0, 1, 2]);
                assert_eq!(cp.drift_target_indices, Some(vec![100, 101, 102]));
                assert_eq!(cp.drift_steps, Some(50));
            }
            _ => panic!("Expected Churn phase"),
        }
    }

    #[test]
    fn test_parse_snapshot_phase() {
        let yaml = r#"
type: Snapshot
output_path: "snapshot_001.json"
"#;
        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();
        match phase {
            TestPhase::Snapshot(sp) => {
                assert_eq!(sp.output_path, "snapshot_001.json");
            }
            _ => panic!("Expected Snapshot phase"),
        }
    }

    #[test]
    fn test_parse_wait_phase() {
        let yaml = r#"
type: Wait
duration_ms: 5000
"#;
        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();
        match phase {
            TestPhase::Wait(wp) => {
                assert_eq!(wp.duration_ms, 5000);
            }
            _ => panic!("Expected Wait phase"),
        }
    }

    #[test]
    fn test_parse_complete_test_plan() {
        let yaml = r#"
peer_count: 1000
phases:
  - type: Bootstrap
    peer_indices:
      start: 0
      end: 99
    bootstrap_index: 0
  - type: Wait
    duration_ms: 1000
  - type: Query
    query_indices: [0, 1, 2]
    source_peer_indices: [10, 20, 30]
    k: 10
  - type: Snapshot
    output_path: "final.json"
"#;
        let plan = TestPlan::from_yaml_str(yaml).unwrap();
        assert_eq!(plan.peer_count, 1000);
        assert_eq!(plan.phases.len(), 4);
        assert!(matches!(plan.phases[0], TestPhase::Bootstrap(_)));
        assert!(matches!(plan.phases[1], TestPhase::Wait(_)));
        assert!(matches!(plan.phases[2], TestPhase::Query(_)));
        assert!(matches!(plan.phases[3], TestPhase::Snapshot(_)));
    }

    #[test]
    fn test_parse_empty_test_plan() {
        let yaml = r#"
peer_count: 0
phases: []
"#;
        let plan = TestPlan::from_yaml_str(yaml).unwrap();
        assert_eq!(plan.peer_count, 0);
        assert!(plan.phases.is_empty());
    }

    #[test]
    fn test_invalid_yaml_missing_required_field() {
        let yaml = r#"
type: Bootstrap
bootstrap_index: 0
"#;
        // Missing peer_indices
        let result: Result<TestPhase, _> = serde_yaml::from_str(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_range_list_format() {
        let yaml = r#"
type: Bootstrap
peer_indices: [5, 10, 15, 20]
bootstrap_index: 5
"#;
        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();
        match phase {
            TestPhase::Bootstrap(bp) => {
                assert_eq!(bp.peer_indices, vec![5, 10, 15, 20]);
            }
            _ => panic!("Expected Bootstrap phase"),
        }
    }

    #[test]
    fn test_index_range_object_format() {
        let yaml = r#"
type: Bootstrap
peer_indices:
  start: 10
  end: 15
bootstrap_index: 10
"#;
        let phase: TestPhase = serde_yaml::from_str(yaml).unwrap();
        match phase {
            TestPhase::Bootstrap(bp) => {
                assert_eq!(bp.peer_indices, vec![10, 11, 12, 13, 14, 15]);
            }
            _ => panic!("Expected Bootstrap phase"),
        }
    }

    #[test]
    fn test_all_churn_pattern_types() {
        let patterns = [
            ("FlashCrowd", ChurnPatternType::FlashCrowd),
            ("MassDeparture", ChurnPatternType::MassDeparture),
            ("GradualJoin", ChurnPatternType::GradualJoin),
            ("GradualLeave", ChurnPatternType::GradualLeave),
            ("EmbeddingDrift", ChurnPatternType::EmbeddingDrift),
        ];

        for (name, expected) in patterns {
            let yaml = format!(
                r#"
type: Churn
pattern: {}
global_indices: [0]
"#,
                name
            );
            let phase: TestPhase = serde_yaml::from_str(&yaml).unwrap();
            match phase {
                TestPhase::Churn(cp) => {
                    assert!(std::mem::discriminant(&cp.pattern) == std::mem::discriminant(&expected));
                }
                _ => panic!("Expected Churn phase for {}", name),
            }
        }
    }
}