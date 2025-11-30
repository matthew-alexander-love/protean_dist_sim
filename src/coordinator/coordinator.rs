use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use protean::embedding_space::{Embedding, EmbeddingSpace};
use protean::SnvConfig;
use tokio::sync::Mutex;
use tonic::transport::Channel;
use tonic::{Request, Response, Status};

use crate::proto::dist_sim::coordinator_node_server::CoordinatorNode;
use crate::proto::dist_sim::worker_node_client::WorkerNodeClient;
use crate::proto::dist_sim::*;
use crate::proto::protean::TensorProto;

use super::constrained_kmeans::ClusteredData;
use super::dataloader::DataLoader;
use super::snapshot::{GlobalSnapshot, ParsedSnapshot};
use super::test_plan::{
    BootstrapPhase, ChurnPhase, ChurnPatternType, QueryPhase, SnapshotPhase, TestPhase, TestPlan, WaitPhase,
};

/// Query identifier: (query_index, source_peer_index)
pub type QueryId = (usize, u64);

/// Maximum number of events to keep in the event queue
const MAX_EVENT_QUEUE_SIZE: usize = 10_000;

/// Query result from a distributed query (legacy - kept for compatibility)
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub query_id: QueryId,
    pub candidates: Vec<u64>,
    pub hops: u32,
    pub latency_ms: u64,
}

/// Pending query awaiting result
struct PendingQuery {
    query_idx: usize,
    source_peer_idx: u64,
    ground_truth: Vec<u64>,
    #[allow(dead_code)]
    sent_at: Instant,
}

/// Completed query with recall metrics
#[derive(Debug, Clone)]
pub struct CompletedQuery {
    pub query_idx: usize,
    pub source_peer_idx: u64,
    pub ground_truth: Vec<u64>,
    pub returned_candidates: Vec<u64>,
    pub recall: f32,
    pub hops: u32,
    pub latency_ms: u64,
}

/// Main coordinator configuration (not serializable - use YAML parsing separately)
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Maximum peers per worker (cluster size limit)
    pub workers_capacity: usize,

    /// Number of workers we will use
    pub num_workers: usize,

    /// SNV configuration (passed to all workers)
    pub snv_config: SnvConfig,

    /// Output directory for results
    pub output_dir: String,

    /// Coordinator gRPC bind address
    pub coordinator_bind_address: String,
}

/// Worker entry with gRPC client and cluster assignment
struct WorkerEntry {
    client: WorkerNodeClient<Channel>,
    address: String,
    cluster_idx: usize,
}

/// Main coordinator that orchestrates distributed ANN tests
pub struct Coordinator<S: EmbeddingSpace> {
    // Worker registry (address -> worker entry)
    workers: Arc<DashMap<String, WorkerEntry>>,

    // Dataset (full dataset loaded and clustered in coordinator)
    clustered_data: ClusteredData<S>,

    // Index of next cluster to assign
    assigned_cluster_idx: Arc<AtomicUsize>,

    // Test Plan
    test_plan: TestPlan,

    // Event aggregation
    events: Arc<Mutex<VecDeque<ProteanEventProto>>>,

    // Query tracking for recall calculation
    pending_queries: Arc<DashMap<Vec<u8>, PendingQuery>>,
    completed_queries: Arc<Mutex<Vec<CompletedQuery>>>,

    // Active peer tracking for staged bootstrap
    active_peer_count: Arc<AtomicUsize>,

    // Dynamic bootstrapping flow control
    current_bootstrapping: Arc<DashMap<u64, Instant>>,
    active: Arc<DashMap<u64, ()>>,

    // Bootstrap servers (indices that are designated entry points)
    bootstrap_servers: Arc<DashMap<u64, ()>>,

    // Config
    config: CoordinatorConfig,
}

impl<S: EmbeddingSpace> Coordinator<S>
where
    S::EmbeddingData: Embedding<Scalar = f32>,
{
    /// Create new coordinator with config
    pub fn new<D: DataLoader<S>>(
        dataloader: D,
        test_plan: TestPlan,
        config: CoordinatorConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Prepare data set
        dataloader.download()?;

        let dataset = if dataloader.verify_available() {
            dataloader.load_data()?
        } else {
            return Err("Dataset not available after download".into());
        };

        let mut clustered_data = ClusteredData::new(
            config.workers_capacity,
            config.num_workers,
            dataset,
        );

        clustered_data.fit();

        Ok(Self {
            workers: Arc::new(DashMap::new()),
            clustered_data,
            assigned_cluster_idx: Arc::new(AtomicUsize::new(0)),
            test_plan,
            events: Arc::new(Mutex::new(VecDeque::new())),
            pending_queries: Arc::new(DashMap::new()),
            completed_queries: Arc::new(Mutex::new(Vec::new())),
            active_peer_count: Arc::new(AtomicUsize::new(0)),

            current_bootstrapping: Arc::new(DashMap::new()),
            active: Arc::new(DashMap::new()),
            bootstrap_servers: Arc::new(DashMap::new()),

            config,
        })
    }

    /// Main orchestration loop - execute test plan phases
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Wait for all workers to register
        self.wait_for_workers().await?;

        // Create bootstrap servers before any phases
        self.create_bootstrap_servers().await?;

        // Execute test plan phases
        for phase in self.test_plan.phases.clone() {
            match phase {
                TestPhase::Bootstrap(cfg) => self.bootstrap(&cfg).await?,
                TestPhase::Query(cfg) => self.query(&cfg).await?,
                TestPhase::Snapshot(cfg) => self.snapshot(&cfg).await?,
                TestPhase::Wait(cfg) => self.wait(&cfg).await?,
                TestPhase::Churn(cfg) => self.churn(&cfg).await?,
            }
        }

        Ok(())
    }

    /// Create bootstrap servers before any test phases
    /// First server is isolated, each subsequent one bootstraps to the previous
    async fn create_bootstrap_servers(&self) -> Result<(), Box<dyn std::error::Error>> {
        let server_indices = &self.test_plan.bootstrap_server_indices;

        if server_indices.is_empty() {
            tracing::info!("No bootstrap servers configured");
            return Ok(());
        }

        tracing::info!("Creating {} bootstrap servers: {:?}", server_indices.len(), server_indices);

        for (i, &server_idx) in server_indices.iter().enumerate() {
            // Find which worker should host this server
            let cluster_idx = self.get_cluster_for_peer(server_idx);

            if let Some(worker) = self.find_worker_for_cluster(cluster_idx) {
                // Mark as bootstrapping
                self.current_bootstrapping.insert(server_idx, Instant::now());

                // First server is isolated (bootstrap_index = 0 means no bootstrap)
                // Subsequent servers bootstrap to the previous one
                let bootstrap_index = if i == 0 {
                    0 // Isolated - no bootstrap peer
                } else {
                    server_indices[i - 1]
                };

                let request = CreatePeersRequest {
                    global_indices: vec![server_idx],
                    bootstrap_index,
                    bootstrap_peer: None,
                };

                let mut client = worker.client.clone();
                match client.create_peers(request).await {
                    Ok(_) => {
                        tracing::info!(
                            "Created bootstrap server {} on cluster {} (bootstrap_to: {})",
                            server_idx, cluster_idx,
                            if i == 0 { "none".to_string() } else { format!("{}", bootstrap_index) }
                        );
                    }
                    Err(e) => {
                        tracing::error!("Failed to create bootstrap server {}: {}", server_idx, e);
                        self.current_bootstrapping.remove(&server_idx);
                        return Err(format!("Failed to create bootstrap server {}: {}", server_idx, e).into());
                    }
                }

                // Wait for this server to complete before creating next
                let timeout = Duration::from_secs(60);
                let start = Instant::now();
                while self.current_bootstrapping.contains_key(&server_idx) {
                    if start.elapsed() > timeout {
                        tracing::error!("Bootstrap server {} timed out", server_idx);
                        self.current_bootstrapping.remove(&server_idx);
                        return Err(format!("Bootstrap server {} timed out", server_idx).into());
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }

                // Record as bootstrap server
                self.bootstrap_servers.insert(server_idx, ());
                tracing::info!("Bootstrap server {} is ready", server_idx);
            } else {
                return Err(format!("No worker found for cluster {}", cluster_idx).into());
            }
        }

        tracing::info!("All {} bootstrap servers ready", server_indices.len());
        Ok(())
    }

    /// Wait for all expected workers to register
    async fn wait_for_workers(&self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Waiting for {} workers to register...", self.config.num_workers);

        while self.workers.len() < self.config.num_workers {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        tracing::info!("All {} workers registered", self.config.num_workers);
        Ok(())
    }

    /// Bootstrap phase: Create and connect peers to the network
    async fn bootstrap(&self, config: &BootstrapPhase) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!(
            "Bootstrap phase: creating {} peers with bootstrap_index={}",
            config.peer_indices.len(),
            config.bootstrap_index
        );

        // Rate limit: max 1% of current network bootstrapping at once (min 1)
        let max_concurrent = std::cmp::max(1, self.active_peer_count.load(Ordering::SeqCst) / 100);

        for &peer_idx in &config.peer_indices {
            // Wait until we have room to bootstrap
            while self.current_bootstrapping.len() >= max_concurrent {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }

            // Find which worker should host this peer (round-robin by cluster)
            let cluster_idx = self.get_cluster_for_peer(peer_idx);

            // Find worker for this cluster
            if let Some(worker) = self.find_worker_for_cluster(cluster_idx) {
                // Mark as bootstrapping
                self.current_bootstrapping.insert(peer_idx, Instant::now());

                // Send CreatePeers RPC
                let request = CreatePeersRequest {
                    global_indices: vec![peer_idx],
                    bootstrap_index: config.bootstrap_index,
                    bootstrap_peer: None, // TODO: populate if needed
                };

                let mut client = worker.client.clone();
                match client.create_peers(request).await {
                    Ok(_) => {
                        tracing::debug!("Created peer {} on cluster {}", peer_idx, cluster_idx);
                    }
                    Err(e) => {
                        tracing::error!("Failed to create peer {}: {}", peer_idx, e);
                        self.current_bootstrapping.remove(&peer_idx);
                    }
                }
            }
        }

        // Wait for all bootstrapping to complete (with timeout)
        let timeout = Duration::from_secs(300); // 5 minute timeout
        let start = std::time::Instant::now();

        while !self.current_bootstrapping.is_empty() {
            if start.elapsed() > timeout {
                let pending: Vec<_> = self.current_bootstrapping.iter().map(|e| *e.key()).collect();
                tracing::warn!(
                    "Bootstrap phase timed out after {:?} with {} peers still bootstrapping: {:?}",
                    timeout,
                    pending.len(),
                    &pending[..std::cmp::min(10, pending.len())]
                );
                // Clear stuck peers to continue
                for peer_idx in pending {
                    self.current_bootstrapping.remove(&peer_idx);
                }
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        tracing::info!("Bootstrap phase complete");
        Ok(())
    }

    /// Query phase: Execute distributed queries and calculate recall
    async fn query(&self, config: &QueryPhase) -> Result<(), Box<dyn std::error::Error>> {
        let total_queries = config.query_indices.len() * config.source_peer_indices.len();
        tracing::info!(
            "Query phase: {} queries x {} source peers = {} total, k={}",
            config.query_indices.len(),
            config.source_peer_indices.len(),
            total_queries,
            config.k
        );

        let mut queries_sent = 0;

        // For each query index
        for &query_idx in &config.query_indices {
            // Get query embedding from test set
            let query_embedding = self.clustered_data.test_set()
                .get(query_idx as usize)
                .ok_or_else(|| format!("Query index {} out of bounds", query_idx))?;

            // Compute ground truth via brute force on active peers
            let ground_truth = self.compute_ground_truth(query_embedding, config.k);

            // Send query to each source peer
            for &source_peer_idx in &config.source_peer_indices {
                // Find worker hosting this peer
                let cluster_idx = self.get_cluster_for_peer(source_peer_idx);

                if let Some(worker) = self.find_worker_for_cluster(cluster_idx) {
                    let request = QueryRequest {
                        source_peer_uuid: Self::global_index_to_uuid(source_peer_idx),
                        query_embedding: Some(Self::embedding_to_tensor_proto(query_embedding)),
                        k: config.k as u32,
                        config: None,
                    };

                    let mut client = worker.client.clone();
                    match client.execute_query(request).await {
                        Ok(response) => {
                            let query_uuid = response.into_inner().query_uuid;

                            // Store pending query for result tracking
                            self.pending_queries.insert(
                                query_uuid,
                                PendingQuery {
                                    query_idx: query_idx as usize,
                                    source_peer_idx,
                                    ground_truth: ground_truth.clone(),
                                    sent_at: Instant::now(),
                                },
                            );
                            queries_sent += 1;
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to execute query {} from peer {}: {}",
                                query_idx, source_peer_idx, e
                            );
                        }
                    }
                }
            }
        }

        tracing::info!("Sent {} queries, waiting for results...", queries_sent);

        // Wait for all pending queries (with timeout)
        let timeout = Duration::from_secs(60);
        let start = Instant::now();
        while !self.pending_queries.is_empty() && start.elapsed() < timeout {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Log any timed-out queries
        if !self.pending_queries.is_empty() {
            let pending_count = self.pending_queries.len();
            tracing::warn!("{} queries timed out after {:?}", pending_count, timeout);
            self.pending_queries.clear();
        }

        // Calculate and log summary statistics
        let completed = self.completed_queries.lock().await;
        if !completed.is_empty() {
            let avg_recall: f32 = completed.iter().map(|q| q.recall).sum::<f32>() / completed.len() as f32;
            let avg_latency: f64 = completed.iter().map(|q| q.latency_ms as f64).sum::<f64>() / completed.len() as f64;
            let avg_hops: f64 = completed.iter().map(|q| q.hops as f64).sum::<f64>() / completed.len() as f64;

            tracing::info!(
                "Query phase complete: {} queries, avg recall: {:.3}, avg latency: {:.1}ms, avg hops: {:.1}",
                completed.len(), avg_recall, avg_latency, avg_hops
            );
        } else {
            tracing::warn!("Query phase complete: no queries completed");
        }

        Ok(())
    }

    /// Snapshot phase: Collect network state from all workers
    async fn snapshot(&self, config: &SnapshotPhase) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Snapshot phase: collecting state to {}", config.output_path);

        let mut snapshots = Vec::new();

        // Collect snapshots from all workers
        for entry in self.workers.iter() {
            let worker = entry.value();
            let mut client = worker.client.clone();

            let request = SnapshotRequest {
                peer_uuids: vec![], // Empty = all peers
            };

            // Add timeout to prevent hanging on unresponsive workers
            let snapshot_timeout = Duration::from_secs(30);
            match tokio::time::timeout(snapshot_timeout, client.get_snapshot(request)).await {
                Ok(Ok(response)) => {
                    let network_snapshot = response.into_inner();
                    match ParsedSnapshot::from_proto(network_snapshot) {
                        Ok(parsed) => snapshots.push(parsed),
                        Err(e) => tracing::error!("Failed to parse snapshot: {}", e),
                    }
                }
                Ok(Err(e)) => {
                    tracing::error!("Failed to get snapshot from worker {}: {}", worker.address, e);
                }
                Err(_) => {
                    tracing::error!("Timeout getting snapshot from worker {} after {:?}", worker.address, snapshot_timeout);
                }
            }
        }

        // Merge snapshots into global view
        if !snapshots.is_empty() {
            let global_snapshot = GlobalSnapshot::merge_snapshots(snapshots)?;

            // Save to output directory
            let output_path = format!("{}/{}", self.config.output_dir, config.output_path);
            global_snapshot.save_to_file(&output_path)?;

            tracing::info!("Snapshot saved to {}", output_path);
        }

        Ok(())
    }

    /// Wait phase: Pause execution for a duration
    async fn wait(&self, config: &WaitPhase) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Wait phase: sleeping for {}ms", config.duration_ms);
        tokio::time::sleep(Duration::from_millis(config.duration_ms)).await;
        Ok(())
    }

    /// Churn phase: Execute churn patterns (join, leave, drift)
    async fn churn(&self, config: &ChurnPhase) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Churn phase: {:?} on {} peers", config.pattern, config.global_indices.len());

        match config.pattern {
            ChurnPatternType::FlashCrowd => {
                // Instant spawn of multiple peers
                self.flash_crowd(config).await?;
            }
            ChurnPatternType::MassDeparture => {
                // Instant deletion of multiple peers
                self.mass_departure(config).await?;
            }
            ChurnPatternType::GradualJoin => {
                // Gradual peer joining over duration
                self.gradual_join(config).await?;
            }
            ChurnPatternType::GradualLeave => {
                // Gradual peer leaving over duration
                self.gradual_leave(config).await?;
            }
            ChurnPatternType::EmbeddingDrift => {
                // Gradual embedding drift
                self.embedding_drift(config).await?;
            }
        }

        Ok(())
    }

    // Churn pattern implementations

    async fn flash_crowd(&self, config: &ChurnPhase) -> Result<(), Box<dyn std::error::Error>> {
        let mut success_count = 0;
        let mut error_count = 0;

        // Group peers by cluster and batch create
        for &peer_idx in &config.global_indices {
            let cluster_idx = self.get_cluster_for_peer(peer_idx);

            if let Some(worker) = self.find_worker_for_cluster(cluster_idx) {
                let bootstrap_idx = config.bootstrap_indices.first().copied().unwrap_or(0);
                let request = CreatePeersRequest {
                    global_indices: vec![peer_idx],
                    bootstrap_index: bootstrap_idx,
                    bootstrap_peer: None,
                };

                let mut client = worker.client.clone();
                if let Err(e) = client.create_peers(request).await {
                    tracing::error!("FlashCrowd: Failed to create peer {}: {}", peer_idx, e);
                    error_count += 1;
                } else {
                    success_count += 1;
                }
            } else {
                tracing::warn!("FlashCrowd: No worker found for cluster {}", cluster_idx);
            }
        }

        tracing::info!("FlashCrowd complete: {} succeeded, {} failed", success_count, error_count);
        Ok(())
    }

    async fn mass_departure(&self, config: &ChurnPhase) -> Result<(), Box<dyn std::error::Error>> {
        let mut success_count = 0;
        let mut error_count = 0;

        // Group peers by cluster and batch delete
        for &peer_idx in &config.global_indices {
            let cluster_idx = self.get_cluster_for_peer(peer_idx);

            if let Some(worker) = self.find_worker_for_cluster(cluster_idx) {
                let request = DeletePeersRequest {
                    global_indices: vec![peer_idx],
                };

                let mut client = worker.client.clone();
                if let Err(e) = client.delete_peers(request).await {
                    tracing::error!("MassDeparture: Failed to delete peer {}: {}", peer_idx, e);
                    error_count += 1;
                } else {
                    success_count += 1;
                }
            } else {
                tracing::warn!("MassDeparture: No worker found for cluster {}", cluster_idx);
            }
        }

        tracing::info!("MassDeparture complete: {} succeeded, {} failed", success_count, error_count);
        Ok(())
    }

    async fn gradual_join(&self, config: &ChurnPhase) -> Result<(), Box<dyn std::error::Error>> {
        let num_peers = config.global_indices.len();
        if num_peers == 0 {
            return Ok(());
        }

        let duration = Duration::from_millis(config.duration_ms);
        let interval = duration / num_peers as u32;

        let mut success_count = 0;
        let mut error_count = 0;

        for &peer_idx in &config.global_indices {
            let cluster_idx = self.get_cluster_for_peer(peer_idx);

            if let Some(worker) = self.find_worker_for_cluster(cluster_idx) {
                let bootstrap_idx = config.bootstrap_indices.first().copied().unwrap_or(0);
                let request = CreatePeersRequest {
                    global_indices: vec![peer_idx],
                    bootstrap_index: bootstrap_idx,
                    bootstrap_peer: None,
                };

                let mut client = worker.client.clone();
                if let Err(e) = client.create_peers(request).await {
                    tracing::error!("GradualJoin: Failed to create peer {}: {}", peer_idx, e);
                    error_count += 1;
                } else {
                    success_count += 1;
                }
            } else {
                tracing::warn!("GradualJoin: No worker found for cluster {}", cluster_idx);
            }

            tokio::time::sleep(interval).await;
        }

        tracing::info!("GradualJoin complete: {} succeeded, {} failed", success_count, error_count);
        Ok(())
    }

    async fn gradual_leave(&self, config: &ChurnPhase) -> Result<(), Box<dyn std::error::Error>> {
        let num_peers = config.global_indices.len();
        if num_peers == 0 {
            return Ok(());
        }

        let duration = Duration::from_millis(config.duration_ms);
        let interval = duration / num_peers as u32;

        let mut success_count = 0;
        let mut error_count = 0;

        for &peer_idx in &config.global_indices {
            let cluster_idx = self.get_cluster_for_peer(peer_idx);

            if let Some(worker) = self.find_worker_for_cluster(cluster_idx) {
                let request = DeletePeersRequest {
                    global_indices: vec![peer_idx],
                };

                let mut client = worker.client.clone();
                if let Err(e) = client.delete_peers(request).await {
                    tracing::error!("GradualLeave: Failed to delete peer {}: {}", peer_idx, e);
                    error_count += 1;
                } else {
                    success_count += 1;
                }
            } else {
                tracing::warn!("GradualLeave: No worker found for cluster {}", cluster_idx);
            }

            tokio::time::sleep(interval).await;
        }

        tracing::info!("GradualLeave complete: {} succeeded, {} failed", success_count, error_count);
        Ok(())
    }

    async fn embedding_drift(&self, config: &ChurnPhase) -> Result<(), Box<dyn std::error::Error>> {
        let drift_targets = config.drift_target_indices.as_ref()
            .ok_or("EmbeddingDrift requires drift_target_indices")?;
        let num_steps = config.drift_steps.unwrap_or(10);

        if num_steps == 0 {
            return Ok(());
        }

        let update_interval_ms = config.duration_ms / num_steps as u64;

        // Group drift requests by cluster
        let mut drifts_by_cluster: HashMap<usize, Vec<DriftPeerProto>> = HashMap::new();

        for (&peer_idx, &target_idx) in config.global_indices.iter().zip(drift_targets.iter()) {
            let cluster_idx = self.get_cluster_for_peer(peer_idx);

            drifts_by_cluster.entry(cluster_idx).or_default().push(DriftPeerProto {
                peer_uuid: Self::global_index_to_uuid(peer_idx),
                target_idx,
                num_steps: num_steps as u64,
                update_interval_ms,
            });
        }

        let mut success_count = 0;
        let mut error_count = 0;

        // Send drift requests to each worker
        for (cluster_idx, drifts) in drifts_by_cluster {
            if let Some(worker) = self.find_worker_for_cluster(cluster_idx) {
                let request = DriftPeerRequest { drifts };

                let mut client = worker.client.clone();
                if let Err(e) = client.drift_peer(request).await {
                    tracing::error!("EmbeddingDrift: Failed to send drift request to cluster {}: {}", cluster_idx, e);
                    error_count += 1;
                } else {
                    success_count += 1;
                }
            } else {
                tracing::warn!("EmbeddingDrift: No worker found for cluster {}", cluster_idx);
                error_count += 1;
            }
        }

        tracing::info!(
            "EmbeddingDrift complete: {} clusters succeeded, {} failed ({} peers total)",
            success_count, error_count, config.global_indices.len()
        );

        Ok(())
    }

    // Helper methods

    /// Get cluster index for a peer based on k-means assignment
    fn get_cluster_for_peer(&self, peer_idx: u64) -> usize {
        // Use actual k-means cluster assignment
        self.clustered_data
            .get_assignment(peer_idx as usize)
            .unwrap_or_else(|| {
                // Fallback to modulo for indices outside training set
                (peer_idx as usize) % self.config.num_workers
            })
    }

    /// Find worker entry for a given cluster
    fn find_worker_for_cluster(&self, cluster_idx: usize) -> Option<WorkerEntry> {
        for entry in self.workers.iter() {
            if entry.value().cluster_idx == cluster_idx {
                return Some(WorkerEntry {
                    client: entry.value().client.clone(),
                    address: entry.value().address.clone(),
                    cluster_idx: entry.value().cluster_idx,
                });
            }
        }
        None
    }

    /// Compute ground truth k-NN via brute force on active peers
    fn compute_ground_truth(&self, query: &S::EmbeddingData, k: usize) -> Vec<u64> {
        // Collect (peer_idx, distance) for all active peers
        let mut distances: Vec<(u64, S::DistanceValue)> = self
            .active
            .iter()
            .filter_map(|entry| {
                let peer_idx = *entry.key();
                // Get embedding from train set
                self.clustered_data
                    .train_set()
                    .get(peer_idx as usize)
                    .map(|emb| (peer_idx, S::distance(query, emb)))
            })
            .collect();

        // Sort by distance ascending
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k peer indices
        distances.into_iter().take(k).map(|(idx, _)| idx).collect()
    }

    /// Convert global index to UUID (index stored in first 8 bytes, big-endian)
    fn global_index_to_uuid(idx: u64) -> Vec<u8> {
        let mut uuid = vec![0u8; 16];
        uuid[0..8].copy_from_slice(&idx.to_be_bytes());
        uuid
    }

    /// Extract global index from UUID (first 8 bytes, big-endian)
    fn uuid_to_global_index(uuid: &[u8]) -> u64 {
        if uuid.len() >= 8 {
            u64::from_be_bytes([
                uuid[0], uuid[1], uuid[2], uuid[3],
                uuid[4], uuid[5], uuid[6], uuid[7],
            ])
        } else {
            0
        }
    }

    /// Convert an embedding to TensorProto format
    fn embedding_to_tensor_proto(embedding: &S::EmbeddingData) -> TensorProto {
        let values: Vec<f32> = embedding.as_slice().to_vec();
        TensorProto {
            dims: vec![values.len() as i64],
            data_type: 1, // FLOAT
            float_data: values,
            ..Default::default()
        }
    }
}

#[tonic::async_trait]
impl<S: EmbeddingSpace + Send + Sync + 'static> CoordinatorNode for Coordinator<S>
where
    S::EmbeddingData: Embedding<Scalar = f32> + Send + Sync,
{
    async fn register_worker(
        &self,
        request: Request<WorkerInfo>,
    ) -> Result<Response<Ack>, Status> {
        <Self as CoordinatorNodeImpl>::register_worker_impl(self, request).await
    }

    async fn forward_event(
        &self,
        request: Request<ProteanEventProto>,
    ) -> Result<Response<Ack>, Status> {
        <Self as CoordinatorNodeImpl>::forward_event_impl(self, request).await
    }
}

#[tonic::async_trait]
impl<S: EmbeddingSpace + Send + Sync + 'static> CoordinatorNode for Arc<Coordinator<S>>
where
    S::EmbeddingData: Embedding<Scalar = f32> + Send + Sync,
{
    async fn register_worker(
        &self,
        request: Request<WorkerInfo>,
    ) -> Result<Response<Ack>, Status> {
        <Coordinator<S> as CoordinatorNodeImpl>::register_worker_impl(self.as_ref(), request).await
    }

    async fn forward_event(
        &self,
        request: Request<ProteanEventProto>,
    ) -> Result<Response<Ack>, Status> {
        <Coordinator<S> as CoordinatorNodeImpl>::forward_event_impl(self.as_ref(), request).await
    }
}

/// Internal trait for shared implementation
#[tonic::async_trait]
trait CoordinatorNodeImpl {
    async fn register_worker_impl(
        &self,
        request: Request<WorkerInfo>,
    ) -> Result<Response<Ack>, Status>;

    async fn forward_event_impl(
        &self,
        request: Request<ProteanEventProto>,
    ) -> Result<Response<Ack>, Status>;
}

#[tonic::async_trait]
impl<S: EmbeddingSpace + Send + Sync + 'static> CoordinatorNodeImpl for Coordinator<S>
where
    S::EmbeddingData: Embedding<Scalar = f32> + Send + Sync,
{
    async fn register_worker_impl(
        &self,
        request: Request<WorkerInfo>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();

        // Get next cluster assignment
        let cluster_idx = self.assigned_cluster_idx.fetch_add(1, Ordering::SeqCst);

        if cluster_idx >= self.config.num_workers {
            return Ok(Response::new(Ack {
                success: false,
                message: "Maximum workers reached".into(),
            }));
        }

        // Connect to worker
        let client = WorkerNodeClient::connect(format!("http://{}", req.address))
            .await
            .map_err(|e| Status::internal(format!("Failed to connect to worker: {}", e)))?;

        // Store worker entry
        let worker = WorkerEntry {
            client: client.clone(),
            address: req.address.clone(),
            cluster_idx,
        };

        self.workers.insert(req.address.clone(), worker);

        tracing::info!(
            "Registered worker {} as cluster {}",
            req.address, cluster_idx
        );

        // Send LoadEmbeddings for this cluster
        let embeddings: Vec<IndexedEmbeddingProto> = self
            .clustered_data
            .cluster_indexed_embeddings(cluster_idx)
            .map(|iter| {
                iter.map(|(idx, emb)| IndexedEmbeddingProto {
                    embeddings: Some(Self::embedding_to_tensor_proto(emb)),
                    global_idx: idx as u64,
                })
                .collect()
            })
            .unwrap_or_default();

        if !embeddings.is_empty() {
            let load_req = LoadEmbeddingsRequest { embeddings };
            let mut worker_client = client.clone();
            match worker_client.load_embeddings(Request::new(load_req)).await {
                Ok(_) => {
                    tracing::info!(
                        "Loaded embeddings for cluster {} to worker {}",
                        cluster_idx, req.address
                    );
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to load embeddings for cluster {} to worker {}: {}",
                        cluster_idx, req.address, e
                    );
                    return Err(Status::internal(format!(
                        "Failed to load embeddings: {}", e
                    )));
                }
            }
        }

        // TODO: Register with existing workers

        Ok(Response::new(Ack {
            success: true,
            message: format!("Assigned to cluster {}", cluster_idx),
        }))
    }

    async fn forward_event_impl(
        &self,
        request: Request<ProteanEventProto>,
    ) -> Result<Response<Ack>, Status> {
        let event = request.into_inner();

        // Handle specific event types
        if let Some(event_data) = &event.event {
            match event_data {
                protean_event_proto::Event::BootstrapCompleted(bc) => {
                    let peer_idx = Self::uuid_to_global_index(&bc.peer_uuid);
                    self.current_bootstrapping.remove(&peer_idx);
                    self.active.insert(peer_idx, ());
                    self.active_peer_count.fetch_add(1, Ordering::SeqCst);
                    tracing::debug!("Peer {} bootstrap completed", peer_idx);
                }
                protean_event_proto::Event::QueryCompleted(qc) => {
                    let peer_idx = Self::uuid_to_global_index(&qc.peer_uuid);

                    // Look up pending query by query_uuid and calculate recall
                    if let Some((_, pending)) = self.pending_queries.remove(&qc.query_uuid) {
                        // Extract returned candidate peer indices
                        // QueryCandidateProto has peer field (PeerProto) which has uuid field
                        let returned: Vec<u64> = qc.candidates.iter()
                            .filter_map(|c| {
                                c.peer.as_ref().map(|p| Self::uuid_to_global_index(&p.uuid))
                            })
                            .collect();

                        // Calculate recall: |intersection| / |ground_truth|
                        let intersection_count = returned.iter()
                            .filter(|r| pending.ground_truth.contains(r))
                            .count();
                        let recall = if pending.ground_truth.is_empty() {
                            tracing::warn!("Query {} had empty ground truth - setting recall to 0.0", pending.query_idx);
                            0.0 // Empty ground truth indicates an issue, don't inflate metrics
                        } else {
                            intersection_count as f32 / pending.ground_truth.len() as f32
                        };

                        // Store completed query result
                        if let Ok(mut completed) = self.completed_queries.try_lock() {
                            completed.push(CompletedQuery {
                                query_idx: pending.query_idx,
                                source_peer_idx: pending.source_peer_idx,
                                ground_truth: pending.ground_truth,
                                returned_candidates: returned.clone(),
                                recall,
                                hops: qc.hops,
                                latency_ms: qc.latency_ms,
                            });
                        }

                        tracing::debug!(
                            "Query {} from peer {} completed: recall={:.3}, {} candidates, {}ms, {} hops",
                            pending.query_idx, peer_idx, recall, returned.len(), qc.latency_ms, qc.hops
                        );
                    } else {
                        tracing::debug!(
                            "Query completed on peer {} with {} candidates (untracked query)",
                            peer_idx, qc.candidates.len()
                        );
                    }
                }
                protean_event_proto::Event::StateChanged(sc) => {
                    tracing::debug!(
                        "Peer state changed: {} -> {}",
                        sc.from_state, sc.to_state
                    );
                }
                protean_event_proto::Event::BootstrapConvergingCompleted(bc) => {
                    let peer_idx = Self::uuid_to_global_index(&bc.peer_uuid);
                    tracing::debug!("Peer {} converging completed", peer_idx);
                }
            }
        }

        // Store event for analysis (bounded queue)
        {
            let mut events = self.events.lock().await;
            // Drop oldest events if queue is full
            let mut dropped_count = 0;
            while events.len() >= MAX_EVENT_QUEUE_SIZE {
                events.pop_front();
                dropped_count += 1;
            }
            if dropped_count > 0 {
                tracing::warn!(
                    "Event queue full: dropped {} oldest events (queue size: {})",
                    dropped_count, MAX_EVENT_QUEUE_SIZE
                );
            }
            events.push_back(event);
        }

        Ok(Response::new(Ack {
            success: true,
            message: String::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protean::embedding_space::F32L2Space;

    type TestSpace = F32L2Space<4>;

    #[test]
    fn test_uuid_conversion_roundtrip() {
        for idx in [0u64, 1, 100, 1000, u64::MAX] {
            let uuid = Coordinator::<TestSpace>::global_index_to_uuid(idx);
            let recovered = Coordinator::<TestSpace>::uuid_to_global_index(&uuid);
            assert_eq!(idx, recovered, "Failed roundtrip for index {}", idx);
        }
    }

    #[test]
    fn test_global_index_to_uuid_format() {
        let uuid = Coordinator::<TestSpace>::global_index_to_uuid(0x0102030405060708);

        // First 8 bytes should be big-endian representation
        assert_eq!(uuid[0], 0x01);
        assert_eq!(uuid[1], 0x02);
        assert_eq!(uuid[2], 0x03);
        assert_eq!(uuid[3], 0x04);
        assert_eq!(uuid[4], 0x05);
        assert_eq!(uuid[5], 0x06);
        assert_eq!(uuid[6], 0x07);
        assert_eq!(uuid[7], 0x08);

        // Rest should be zeros
        for i in 8..16 {
            assert_eq!(uuid[i], 0);
        }
    }

    #[test]
    fn test_uuid_to_global_index_short_input() {
        // Should return 0 for too-short input
        let short = vec![0u8; 4];
        let result = Coordinator::<TestSpace>::uuid_to_global_index(&short);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_uuid_to_global_index_exact_8_bytes() {
        let bytes = 12345u64.to_be_bytes();
        let result = Coordinator::<TestSpace>::uuid_to_global_index(&bytes);
        assert_eq!(result, 12345);
    }

    #[test]
    fn test_global_index_to_uuid_deterministic() {
        let uuid1 = Coordinator::<TestSpace>::global_index_to_uuid(42);
        let uuid2 = Coordinator::<TestSpace>::global_index_to_uuid(42);
        assert_eq!(uuid1, uuid2);
    }

    #[test]
    fn test_coordinator_config_creation() {
        let config = CoordinatorConfig {
            workers_capacity: 1000,
            num_workers: 4,
            snv_config: SnvConfig::default(),
            output_dir: "/tmp/output".to_string(),
            coordinator_bind_address: "0.0.0.0:50050".to_string(),
        };

        assert_eq!(config.workers_capacity, 1000);
        assert_eq!(config.num_workers, 4);
        assert_eq!(config.output_dir, "/tmp/output");
    }

    #[test]
    fn test_query_result_creation() {
        let result = QueryResult {
            query_id: (5, 100),
            candidates: vec![1, 2, 3, 4, 5],
            hops: 3,
            latency_ms: 150,
        };

        assert_eq!(result.query_id, (5, 100));
        assert_eq!(result.candidates.len(), 5);
        assert_eq!(result.hops, 3);
        assert_eq!(result.latency_ms, 150);
    }
}
