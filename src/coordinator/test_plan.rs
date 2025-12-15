//! Configuration parsing for the distributed simulation coordinator.
//!
//! This module provides types for parsing YAML configuration files that define
//! simulation parameters and test phases.

use std::error::Error;
use std::time::Duration;

use protean::{ExplorationConfig, QueryConfig, SnvConfig};
use serde::{Deserialize, Serialize};

/// Top-level configuration parsed from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub sim_config: SimConfig,
    pub phases: Vec<TestPhase>,
}

impl Config {
    /// Parse configuration from a YAML file.
    pub fn from_yaml(path: &str) -> Result<Self, Box<dyn Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&contents)?;
        Ok(config)
    }
}

/// Simulation configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    pub max_peers_per_worker: usize,
    pub num_workers: usize,
    pub data_dir: String,
    pub output_dir: String,
    pub protean_config: ProteanConfigYaml,
}

/// Protean library configuration (YAML-parseable form).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteanConfigYaml {
    /// Timeout in milliseconds for protean operations.
    pub timeout: u64,
    /// Maximum concurrent queries a peer can handle.
    pub max_concurrent_queries: usize,
    /// SNV (Sparse Neighbor View) configuration.
    pub snv_config: SnvConfigYaml,
}

/// SNV configuration (YAML-parseable form, converts to protean::SnvConfig).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnvConfigYaml {
    /// Maximum concurrent ping requests.
    pub concurrency_limit: usize,
    /// Occlusion threshold for spatial filtering (e.g., 1.2).
    pub occlusion_threshold: f32,
    /// Drift threshold for peer embedding changes (e.g., 0.1 = 10%).
    pub drift_threshold: f32,
    /// Target ratio of total view size to routable set size.
    pub target_degree_ratio: f32,
    /// Dynamism threshold for triggering rebalancing.
    pub dynamism_threshold: f32,
    /// Maximum interval between exploration rounds in seconds.
    pub max_exploration_interval_secs: u64,
    /// Exploration query configuration.
    pub exploration_config: ExplorationConfigYaml,
}

impl SnvConfigYaml {
    /// Convert to protean::SnvConfig.
    pub fn into_snv_config(self, timeout_ms: u64) -> SnvConfig {
        SnvConfig {
            concurrency_limit: self.concurrency_limit,
            timeout: Duration::from_millis(timeout_ms),
            occlusion_threshold: self.occlusion_threshold,
            drift_threshold: self.drift_threshold,
            target_degree_ratio: self.target_degree_ratio,
            dynamism_threshold: self.dynamism_threshold,
            exploration_config: self.exploration_config.into_exploration_config(timeout_ms),
            max_exploration_interval: Some(Duration::from_secs(self.max_exploration_interval_secs)),
        }
    }
}

/// Exploration configuration (YAML-parseable form).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationConfigYaml {
    /// K value for converging query (finds peers near ourselves).
    pub converge_k: usize,
    /// Query configuration for converging query.
    pub converge_config: QueryConfigYaml,
    /// K value for exploration query (0 = disabled).
    pub explore_k: usize,
    /// Query configuration for exploration query.
    pub explore_config: QueryConfigYaml,
}

impl ExplorationConfigYaml {
    /// Convert to protean::ExplorationConfig.
    pub fn into_exploration_config(self, timeout_ms: u64) -> ExplorationConfig {
        ExplorationConfig {
            converge_k: self.converge_k,
            converge_config: self.converge_config.into_query_config(timeout_ms),
            explore_k: self.explore_k,
            explore_config: self.explore_config.into_query_config(timeout_ms),
        }
    }
}

/// Query configuration (YAML-parseable form).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfigYaml {
    /// Maximum size of the search list (candidate buffer).
    pub search_list_size: usize,
    /// Maximum concurrent in-flight requests.
    pub concurrency_limit: usize,
    /// Minimum candidates to request from each peer.
    pub share_floor: usize,
}

impl QueryConfigYaml {
    /// Convert to protean::QueryConfig.
    pub fn into_query_config(self, timeout_ms: u64) -> QueryConfig {
        QueryConfig {
            search_list_size: self.search_list_size,
            concurrency_limit: self.concurrency_limit,
            timeout: Duration::from_millis(timeout_ms),
            share_floor: self.share_floor,
        }
    }
}

/// A test phase in the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TestPhase {
    /// Gradually join peers to the network.
    GradualJoin {
        global_indices: IndexRange,
        bootstrap_indices: Vec<u64>,
        rate_per_sec: f64,
        bootstrap_timeout_sec: u64,
    },
    /// Gradually remove peers from the network.
    GradualLeave {
        global_indices: IndexRange,
        rate_per_sec: f64,
        bootstrap_timeout_sec: u64,
    },
    /// Simultaneously join and leave peers.
    GradualJoinLeave {
        join_global_indices: IndexRange,
        leave_global_indices: IndexRange,
        bootstrap_indices: Vec<u64>,
        join_rate_per_sec: f64,
        leave_rate_per_sec: f64,
        bootstrap_timeout_sec: u64,
    },
    /// Gradually drift peer embeddings to new positions.
    EmbeddingDrift {
        start_indices: IndexRange,
        end_indices: IndexRange,
        drift_steps: u32,
        duration_per_step_sec: u64,
    },
    /// Wait phase - pause execution for a duration.
    Wait {
        duration_sec: u64,
    },
    /// Snapshot phase - collect network state.
    Snapshot {
        output_path: String,
    },
    /// Query phase - execute distributed queries.
    Query {
        /// Number of random peers to execute each query from
        num_source_peers: usize,
        /// Indices into DataSet::test for query vectors
        query_indices: Vec<usize>,
        /// K values to calculate recall for
        k: Vec<usize>,
        query_config: QueryConfigYaml,
        /// Output filename (placed in output_dir from SimConfig)
        output_path: String,
    },
}

/// A range of global peer indices (inclusive start, exclusive end).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRange {
    pub start: u64,
    pub end: u64,
}

impl IndexRange {
    /// Convert to a Vec of indices.
    pub fn to_vec(&self) -> Vec<u64> {
        (self.start..self.end).collect()
    }

    /// Get the count of indices in the range.
    pub fn len(&self) -> usize {
        (self.end - self.start) as usize
    }

    /// Check if the range is empty.
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_range() {
        let range = IndexRange { start: 0, end: 10 };
        assert_eq!(range.len(), 10);
        assert!(!range.is_empty());
        assert_eq!(range.to_vec(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_empty_index_range() {
        let range = IndexRange { start: 5, end: 5 };
        assert!(range.is_empty());
        assert_eq!(range.len(), 0);
    }
}
