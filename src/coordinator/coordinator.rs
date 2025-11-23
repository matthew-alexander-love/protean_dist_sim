use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use protean::embedding_space::{Embedding, EmbeddingSpace};
use tokio::sync::{Mutex, RwLock};
use tonic::transport::Channel;

use crate::proto::dist_sim::worker_node_client::WorkerNodeClient;
use crate::proto::dist_sim::*;

use super::config::{CoordinatorConfig, WorkerConfig};
use super::dataset::{
    convert_to_embedding_data, embedding_to_tensor_proto, embeddings_to_tensor_protos,
    partition_embeddings, read_fvecs,
};
use super::snapshot::{GlobalSnapshot, ParsedSnapshot};
use super::test_plan::{
    self, BootstrapPhase, ChurnPhase, QueryPhase, SnapshotPhase, TestPhase, TestPlan,
};

/// Query identifier: (query_index, source_peer_index)
pub type QueryId = (usize, u64);

/// Query result with ground truth and recall
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryResult {
    pub query_idx: usize,
    pub source_peer_idx: u64,
    pub k: usize,
    pub results: Vec<u64>,
    pub ground_truth: Vec<u64>,
    pub recall: f64,
    pub latency_ms: u64,
}

/// Main coordinator that orchestrates distributed ANN tests
pub struct Coordinator<S: EmbeddingSpace> {
    // Worker registry
    workers: Arc<DashMap<String, WorkerNodeClient<Channel>>>,
    worker_configs: Vec<WorkerConfig>,

    // Dataset (full dataset loaded in coordinator)
    embeddings: Arc<RwLock<Vec<S::EmbeddingData>>>,
    query_embeddings: Arc<RwLock<Vec<S::EmbeddingData>>>,

    // Event/result aggregation
    events: Arc<Mutex<Vec<ProteanEventProto>>>,
    query_results: Arc<DashMap<QueryId, QueryResult>>,

    // Active peer tracking for staged bootstrap
    active_peer_count: Arc<std::sync::atomic::AtomicUsize>,

    // Dynamic bootstrapping flow control
    current_bootstrapping: Arc<DashMap<u64, Instant>>,
    pending_bootstrap: Arc<Mutex<HashSet<u64>>>, // Set of pending peer indices
    bootstrap_target_idx: Arc<RwLock<u64>>, // Bootstrap target for all peers

    // Config
    config: CoordinatorConfig,

    _phantom: PhantomData<S>,
}

impl<S: EmbeddingSpace> Coordinator<S> {
    /// Create new coordinator with config
    pub async fn new(config: CoordinatorConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let workers = Arc::new(DashMap::new());

        Ok(Self {
            workers,
            worker_configs: config.workers.clone(),
            embeddings: Arc::new(RwLock::new(Vec::new())),
            query_embeddings: Arc::new(RwLock::new(Vec::new())),
            events: Arc::new(Mutex::new(Vec::new())),
            query_results: Arc::new(DashMap::new()),
            active_peer_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            current_bootstrapping: Arc::new(DashMap::new()),
            pending_bootstrap: Arc::new(Mutex::new(HashSet::new())),
            bootstrap_target_idx: Arc::new(RwLock::new(0)),
            config,
            _phantom: PhantomData,
        })
    }

    /// Get reference to workers for service to use
    pub fn workers(&self) -> &Arc<DashMap<String, WorkerNodeClient<Channel>>> {
        &self.workers
    }

    /// Get reference to events for service to use
    pub fn events(&self) -> &Arc<Mutex<Vec<ProteanEventProto>>> {
        &self.events
    }

    /// Get reference to query results
    pub fn query_results(&self) -> &Arc<DashMap<QueryId, QueryResult>> {
        &self.query_results
    }

    /// Get reference to worker configs
    pub fn worker_configs(&self) -> &[WorkerConfig] {
        &self.worker_configs
    }

    /// Get reference to coordinator config
    pub fn config(&self) -> &CoordinatorConfig {
        &self.config
    }

    /// Get current active peer count
    pub fn active_peer_count(&self) -> usize {
        self.active_peer_count.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Increment active peer count (called when BootstrapCompleted event received)
    pub fn increment_active_peers(&self) {
        self.active_peer_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get coordinator bind address
    pub fn coordinator_address(&self) -> &str {
        &self.config.coordinator_bind_address
    }
}

impl<S: EmbeddingSpace> Coordinator<S>
where
    S::EmbeddingData: Embedding<Scalar = f32>,
{

    /// Load base and query datasets from .fvecs files
    pub async fn load_datasets(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Read base dataset
        let base_vecs = read_fvecs(&self.config.dataset.base_path)?;
        let base_embeddings = convert_to_embedding_data::<S>(base_vecs);
        *self.embeddings.write().await = base_embeddings;

        // Read query dataset
        let query_vecs = read_fvecs(&self.config.dataset.query_path)?;
        let query_embeddings = convert_to_embedding_data::<S>(query_vecs);
        *self.query_embeddings.write().await = query_embeddings;

        Ok(())
    }

    /// Handle bootstrap completion: remove from current_bootstrapping and trigger next bootstrap
    pub async fn on_bootstrap_complete(&self, peer_idx: u64) {
        // Remove from current_bootstrapping map
        self.current_bootstrapping.remove(&peer_idx);

        // Increment active peer count
        self.increment_active_peers();

        tracing::debug!(
            "Peer {} bootstrap completed. Active: {}, Bootstrapping: {}",
            peer_idx,
            self.active_peer_count(),
            self.current_bootstrapping.len()
        );

        // Try to start next bootstrap
        if let Err(e) = self.try_bootstrap_next().await {
            tracing::error!("Error starting next bootstrap: {:?}", e);
        }
    }

    /// Try to start next bootstrap(s) if capacity allows
    /// max_bootstrapping = floor(0.01 * current_active)
    pub async fn try_bootstrap_next(&self) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            let current_active = self.active_peer_count();
            let max_bootstrapping = std::cmp::max(1, (current_active as f64 * 0.01).floor() as usize);
            let current_bootstrapping = self.current_bootstrapping.len();

            // Check if we have capacity
            if current_bootstrapping >= max_bootstrapping {
                tracing::trace!(
                    "Bootstrap capacity full: {}/{} (active: {})",
                    current_bootstrapping,
                    max_bootstrapping,
                    current_active
                );
                break;
            }

            // Try to get a peer from pending set
            let peer_idx = {
                let mut pending = self.pending_bootstrap.lock().await;
                if pending.is_empty() {
                    break; // No more peers to bootstrap
                }
                // Take any peer from the set
                let peer_idx = *pending.iter().next().unwrap();
                pending.remove(&peer_idx);
                peer_idx
            };

            // Get bootstrap target
            let bootstrap_target_idx = *self.bootstrap_target_idx.read().await;

            // Mark as bootstrapping
            self.current_bootstrapping.insert(peer_idx, Instant::now());

            tracing::debug!(
                "Starting bootstrap for peer {} -> {} (active: {}, bootstrapping: {}/{})",
                peer_idx,
                bootstrap_target_idx,
                current_active,
                current_bootstrapping + 1,
                max_bootstrapping
            );

            // Create a single-peer bootstrap phase
            let is_seed = peer_idx == bootstrap_target_idx;
            self.create_and_bootstrap_batch(&[peer_idx], bootstrap_target_idx, is_seed).await?;

            // Continue loop to try starting more bootstraps if capacity allows
        }

        Ok(())
    }

    /// Wait for workers to register (via RegisterWorker RPC)
    pub async fn wait_for_workers(
        &self,
        timeout: Duration,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();
        let expected_count = self.worker_configs.len();

        while self.workers.len() < expected_count {
            if start.elapsed() > timeout {
                return Err(format!(
                    "Timeout waiting for workers to register. Expected {}, got {}",
                    expected_count,
                    self.workers.len()
                )
                .into());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Ok(())
    }

    /// Partition and distribute embeddings to workers
    pub async fn distribute_embeddings(&self) -> Result<(), Box<dyn std::error::Error>> {
        let embeddings = self.embeddings.read().await;
        let partitions = partition_embeddings(embeddings.clone(), self.workers.len());

        for ((offset, slice), worker_cfg) in partitions.into_iter().zip(&self.worker_configs) {
            let client = self
                .workers
                .get(&worker_cfg.worker_id)
                .ok_or("Worker not registered")?;

            let tensor_protos = embeddings_to_tensor_protos::<S>(&slice);

            client
                .value()
                .clone()
                .load_embeddings(LoadEmbeddingsRequest {
                    global_offset: offset,
                    embeddings: tensor_protos,
                })
                .await?;
        }

        Ok(())
    }

    /// Send configuration to all workers
    pub async fn configure_workers(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Convert our config types to proto types
        let sim_config_proto = self.config_to_proto();

        for worker in self.workers.iter() {
            worker
                .value()
                .clone()
                .set_config(sim_config_proto.clone())
                .await?;
        }
        Ok(())
    }

    /// Setup worker-to-worker mesh network
    pub async fn setup_worker_network(&self) -> Result<(), Box<dyn std::error::Error>> {
        // For each worker, register all OTHER workers
        for worker_cfg in &self.worker_configs {
            let client = self
                .workers
                .get(&worker_cfg.worker_id)
                .ok_or("Worker not registered")?;

            for other_cfg in &self.worker_configs {
                if other_cfg.worker_id != worker_cfg.worker_id {
                    client
                        .value()
                        .clone()
                        .register_worker(WorkerInfo {
                            address: other_cfg.address.clone(),
                            capacity: 0, // Not used in current implementation
                            region: String::new(),
                            version: String::new(),
                        })
                        .await?;
                }
            }
        }
        Ok(())
    }

    /// Execute bootstrap phase with dynamic flow control
    pub async fn execute_bootstrap(
        &self,
        phase: &BootstrapPhase,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let peer_indices = phase.peer_indices.to_vec();
        println!("[Coordinator] execute_bootstrap: peer_indices={:?}, bootstrap_index={}", peer_indices, phase.bootstrap_index);

        // Use staged bootstrap with dynamic flow control
        // Reset state = true for first bootstrap, false for subsequent ones
        let reset_state = self.active_peer_count() == 0;
        self.execute_staged_bootstrap(peer_indices, phase.bootstrap_index, reset_state).await?;

        Ok(())
    }

    /// Execute bootstrap phase (old implementation without flow control)
    /// Kept for reference but not used
    #[allow(dead_code)]
    async fn execute_bootstrap_all_at_once(
        &self,
        phase: &BootstrapPhase,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let peer_indices = phase.peer_indices.to_vec();
        println!("[Coordinator] execute_bootstrap_all_at_once: peer_indices={:?}, bootstrap_index={}", peer_indices, phase.bootstrap_index);
        // Group peer indices by worker
        let peer_assignments = self.assign_peers_to_workers(&peer_indices).await?;
        println!("[Coordinator] Peer assignments: {:?}", peer_assignments);

        // Prepare bootstrap peer info if bootstrap_index is provided
        // Only provide bootstrap info if we're NOT creating the bootstrap seed itself
        let bootstrap_peer_info = if !peer_indices.contains(&phase.bootstrap_index) {
            // Peers in this batch need to bootstrap from the seed peer
            // Generate UUID for bootstrap peer
            let bootstrap_uuid = self.global_index_to_uuid(phase.bootstrap_index);

            // Find which worker owns the bootstrap peer
            let bootstrap_worker_address = self.get_worker_address_for_peer(phase.bootstrap_index).await?;

            // Don't send embedding - worker will look it up locally by global_index
            Some(BootstrapPeerInfo {
                uuid: bootstrap_uuid,
                embedding: None,  // Workers have all embeddings loaded, use global_index instead
                worker_address: bootstrap_worker_address,
                global_index: phase.bootstrap_index,
            })
        } else {
            // This batch contains the seed peer itself - no bootstrap needed
            None
        };

        for (worker_id, peer_indices) in peer_assignments {
            println!("[Coordinator] Sending create_peers to worker {}, indices: {:?}", worker_id, peer_indices);
            let client = self.workers.get(&worker_id).ok_or("Worker not found")?;

            client
                .value()
                .clone()
                .create_peers(CreatePeersRequest {
                    global_indices: peer_indices.clone(),
                    bootstrap_index: phase.bootstrap_index,
                    bootstrap_peer: bootstrap_peer_info.clone(),
                })
                .await
                .map_err(|e| {
                    println!("[Coordinator] Failed to send create_peers to worker {}: {}", worker_id, e);
                    e
                })?;
            println!("[Coordinator] Successfully sent create_peers to worker {}", worker_id);
        }

        // TODO: Wait for bootstrap completion by monitoring events
        // For now, just return immediately

        Ok(())
    }

    /// Execute dynamic bootstrap with flow control
    /// Bootstraps peers dynamically, allowing max_bootstrapping = floor(0.01 * current_active) concurrent bootstraps
    /// This ensures later peers can discover the full network during bootstrap
    pub async fn execute_staged_bootstrap(
        &self,
        all_peer_indices: Vec<u64>,
        bootstrap_index: u64,
        reset_state: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if all_peer_indices.is_empty() {
            return Ok(());
        }

        println!("   Starting dynamic bootstrap for {} peers...", all_peer_indices.len());

        // Reset active peer count and bootstrapping state if requested
        if reset_state {
            self.active_peer_count.store(0, std::sync::atomic::Ordering::SeqCst);
            self.current_bootstrapping.clear();
        }

        // Set bootstrap target
        *self.bootstrap_target_idx.write().await = bootstrap_index;

        // Build pending set, but prioritize seed peer to bootstrap first
        let mut pending = self.pending_bootstrap.lock().await;
        if reset_state {
            pending.clear();
        }

        // Remove bootstrap_index from the list (will bootstrap it first)
        let mut remaining_peers: Vec<u64> = all_peer_indices.iter()
            .filter(|&&idx| idx != bootstrap_index)
            .copied()
            .collect();

        // Check if bootstrap_index is in the list
        let has_seed = all_peer_indices.contains(&bootstrap_index);

        // Add remaining peers to pending
        pending.extend(&remaining_peers);
        drop(pending);

        println!("   Queued {} peers for bootstrap", all_peer_indices.len());

        // Create seed peer first (does NOT bootstrap - just gets created and becomes active)
        if has_seed {
            println!("   Creating seed peer {} (no bootstrap needed)...", bootstrap_index);
            self.create_and_bootstrap_batch(&[bootstrap_index], bootstrap_index, true).await?;

            // Seed peer becomes active immediately (doesn't go through bootstrap process)
            self.increment_active_peers();
            println!("   Seed peer {} is now active", bootstrap_index);
        }

        // Start initial bootstrap(s) for remaining peers
        self.try_bootstrap_next().await?;

        // Wait for all peers to complete
        // When reset_state=false, we're adding to existing peers, so calculate incremental target
        let current_active_at_start = self.active_peer_count();
        let target_count = if reset_state {
            all_peer_indices.len()
        } else {
            current_active_at_start + all_peer_indices.len()
        };
        let mut wait_time = 0;
        // Scale timeout with number of peers (minimum 60s, add 100ms per peer)
        let max_wait_ms = std::cmp::max(60000, all_peer_indices.len() * 100);

        while self.active_peer_count() < target_count {
            tokio::time::sleep(Duration::from_millis(100)).await;
            wait_time += 100;

            if wait_time >= max_wait_ms {
                let pending = self.pending_bootstrap.lock().await.len();
                let bootstrapping = self.current_bootstrapping.len();
                return Err(format!(
                    "Timeout waiting for bootstrap completion. Expected {}, got {} active, {} pending, {} bootstrapping",
                    target_count, self.active_peer_count(), pending, bootstrapping
                ).into());
            }

            if wait_time % 5000 == 0 {
                let active = self.active_peer_count();
                let pending = self.pending_bootstrap.lock().await.len();
                let bootstrapping = self.current_bootstrapping.len();
                let max_boot = std::cmp::max(1, (active as f64 * 0.01).floor() as usize);
                println!("   Progress: {}/{} active, {} pending, {}/{} bootstrapping",
                         active, target_count, pending, bootstrapping, max_boot);
            }
        }

        println!("   Dynamic bootstrap completed. Total active peers: {}", self.active_peer_count());
        Ok(())
    }

    /// Create and bootstrap a batch of peers in a single RPC call per worker
    /// This distributes load across workers as they can process requests in parallel
    async fn create_and_bootstrap_batch(
        &self,
        peer_indices: &[u64],
        bootstrap_target: u64,
        is_seed: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Get bootstrap peer info (None for seed peer)
        let bootstrap_peer = if is_seed {
            None
        } else {
            Some(self.get_bootstrap_peer_info(bootstrap_target).await?)
        };

        // Assign peers to workers
        let assignments = self.assign_peers_to_workers(peer_indices).await?;

        // Send create_peers to all workers in parallel (distributes load)
        for (worker_id, indices) in assignments {
            let client = self.workers.get(&worker_id)
                .ok_or_else(|| format!("Worker {} not found", worker_id))?;

            let request = tonic::Request::new(CreatePeersRequest {
                global_indices: indices.clone(),
                bootstrap_index: 0, // Unused (legacy field)
                bootstrap_peer: bootstrap_peer.clone(),
            });

            client.value().clone()
                .create_peers(request)
                .await
                .map_err(|e| {
                    println!("[Coordinator] Failed to create peers on worker {}: {}", worker_id, e);
                    e
                })?;
        }

        Ok(())
    }


    /// Find the closest active peer in embedding space to the given peer
    async fn find_closest_active_peer(
        &self,
        peer_index: u64,
        active_peers: &[u64],
    ) -> Result<u64, Box<dyn std::error::Error>> {
        let embeddings = self.embeddings.read().await;
        let query_embedding = embeddings.get(peer_index as usize)
            .ok_or_else(|| format!("Embedding {} not found", peer_index))?;

        let mut best_peer = active_peers[0];
        let mut best_distance = S::distance(
            query_embedding,
            embeddings.get(best_peer as usize).unwrap()
        );

        for &candidate_idx in &active_peers[1..] {
            let candidate_embedding = embeddings.get(candidate_idx as usize).unwrap();
            let distance = S::distance(query_embedding, candidate_embedding);
            if distance < best_distance {
                best_distance = distance;
                best_peer = candidate_idx;
            }
        }

        Ok(best_peer)
    }

    /// Get bootstrap peer info for a given global index
    async fn get_bootstrap_peer_info(
        &self,
        global_index: u64,
    ) -> Result<BootstrapPeerInfo, Box<dyn std::error::Error>> {
        use crate::proto::dist_sim::BootstrapPeerInfo;

        // Generate UUID
        let bootstrap_uuid = self.global_index_to_uuid(global_index);

        // Find which worker owns the bootstrap peer
        let bootstrap_worker_address = self.get_worker_address_for_peer(global_index).await?;

        // Don't send embedding - worker will look it up locally by global_index
        Ok(BootstrapPeerInfo {
            uuid: bootstrap_uuid,
            embedding: None,  // Workers have all embeddings loaded, use global_index instead
            worker_address: bootstrap_worker_address,
            global_index,
        })
    }

    /// Execute churn phase
    pub async fn execute_churn(
        &self,
        phase: &ChurnPhase,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let pattern_type = match phase.pattern {
            test_plan::ChurnPatternType::FlashCrowd => ChurnPatternType::FlashCrowd,
            test_plan::ChurnPatternType::MassDeparture => ChurnPatternType::MassDeparture,
            test_plan::ChurnPatternType::GradualJoin => ChurnPatternType::GradualJoin,
            test_plan::ChurnPatternType::GradualLeave => ChurnPatternType::GradualLeave,
            test_plan::ChurnPatternType::EmbeddingDrift => ChurnPatternType::EmbeddingDrift,
        };

        let global_indices = phase.global_indices.to_vec();
        let bootstrap_indices = phase.bootstrap_indices.to_vec();
        let peer_assignments = self.assign_peers_to_workers(&global_indices).await?;

        for (worker_id, peer_indices) in peer_assignments {
            let client = self.workers.get(&worker_id).ok_or("Worker not found")?;

            client
                .value()
                .clone()
                .churn(ChurnPatternRequest {
                    pattern: pattern_type as i32,
                    config: Some(ChurnConfig {
                        global_indices: peer_indices,
                        bootstrap_indices: bootstrap_indices.clone(),
                        duration_ms: phase.duration_ms,
                        rate_per_second: phase.rate_per_second,
                        drift_target_indices: phase.drift_target_indices.clone().unwrap_or_default(),
                        drift_steps: phase.drift_steps.unwrap_or(0),
                    }),
                })
                .await?;
        }

        Ok(())
    }

    /// Execute query phase - CRITICAL METHOD
    /// Calculates ground truth via TrueQuery across all workers, then executes distributed queries
    pub async fn execute_queries(
        &self,
        phase: &QueryPhase,
    ) -> Result<Vec<QueryResult>, Box<dyn std::error::Error>> {
        let query_embeddings = self.query_embeddings.read().await;
        let mut results = Vec::new();

        let query_indices = phase.query_indices.to_vec();
        let source_peer_indices = phase.source_peer_indices.to_vec();

        // Ensure query_indices and source_peer_indices have the same length
        if query_indices.len() != source_peer_indices.len() {
            return Err(format!(
                "query_indices ({}) and source_peer_indices ({}) must have same length",
                query_indices.len(),
                source_peer_indices.len()
            ).into());
        }

        // Zip query_indices with source_peer_indices to execute one query from each source
        for (&query_idx, &source_peer_idx) in query_indices.iter().zip(source_peer_indices.iter()) {
            let query_embedding = &query_embeddings
                .get(query_idx as usize)
                .ok_or(format!("Query index {} out of bounds", query_idx))?;

            // STEP 1: Get ground truth from ALL workers via TrueQuery
            // This queries all ACTIVE peers across all workers
            let mut all_true_results: Vec<(u64, f32)> = Vec::new();

            for worker in self.workers.iter() {
                let response = worker
                    .value()
                    .clone()
                    .true_query(QueryRequest {
                        source_peer_uuid: vec![], // Not used for TrueQuery
                        query_embedding: Some(embedding_to_tensor_proto::<S>(query_embedding)),
                        k: phase.k as u32,
                        config: None, // Use default config
                    })
                    .await?
                    .into_inner();

                // Collect results: Vec<(global_index, distance)>
                for result in response.results {
                    if let Some(peer) = result.peer {
                        // Extract global index from UUID (first 8 bytes)
                        let global_index = u64::from_be_bytes([
                            peer.uuid[0],
                            peer.uuid[1],
                            peer.uuid[2],
                            peer.uuid[3],
                            peer.uuid[4],
                            peer.uuid[5],
                            peer.uuid[6],
                            peer.uuid[7],
                        ]);

                        // Extract distance from DistanceProto
                        if let Some(dist) = result.distance {
                            let distance = if !dist.float_data.is_empty() {
                                dist.float_data[0]
                            } else {
                                0.0
                            };
                            all_true_results.push((global_index, distance));
                        }
                    }
                }
            }

            // STEP 2: Merge and get true top-k from active peers
            all_true_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let ground_truth: Vec<u64> = all_true_results
                .iter()
                .take(phase.k)
                .map(|(idx, _)| *idx)
                .collect();

            // STEP 3: Execute distributed query from this source peer
            {
                let worker_id = self.get_worker_for_peer(source_peer_idx).await?;
                let client = self.workers.get(&worker_id).ok_or("Worker not found")?;

                // Convert source_peer_idx to UUID
                let source_uuid = self.global_index_to_uuid(source_peer_idx);

                let start = Instant::now();
                let response = client
                    .value()
                    .clone()
                    .execute_query(QueryRequest {
                        source_peer_uuid: source_uuid,
                        query_embedding: Some(embedding_to_tensor_proto::<S>(query_embedding)),
                        k: phase.k as u32,
                        config: None,
                    })
                    .await?
                    .into_inner();
                let latency = start.elapsed();

                // STEP 4: Calculate recall vs ground truth
                let query_results: Vec<u64> = response
                    .results
                    .iter()
                    .filter_map(|r| {
                        r.peer.as_ref().map(|p| {
                            u64::from_be_bytes([
                                p.uuid[0], p.uuid[1], p.uuid[2], p.uuid[3],
                                p.uuid[4], p.uuid[5], p.uuid[6], p.uuid[7],
                            ])
                        })
                    })
                    .collect();

                let recall = self.calculate_recall(&ground_truth, &query_results, phase.k);

                let result = QueryResult {
                    query_idx: query_idx as usize,
                    source_peer_idx,
                    k: phase.k,
                    results: query_results,
                    ground_truth: ground_truth.clone(),
                    recall,
                    latency_ms: latency.as_millis() as u64,
                };

                results.push(result.clone());
                self.query_results
                    .insert((query_idx as usize, source_peer_idx), result);
            }
        }

        Ok(results)
    }

    /// Calculate recall: intersection(ground_truth, results) / k
    pub fn calculate_recall(&self, ground_truth: &[u64], results: &[u64], k: usize) -> f64 {
        let gt_set: HashSet<_> = ground_truth.iter().take(k).collect();
        let result_set: HashSet<_> = results.iter().take(k).collect();
        let intersection = gt_set.intersection(&result_set).count();

        intersection as f64 / k as f64
    }

    /// Collect snapshots from all workers
    pub async fn collect_snapshots(
        &self,
        phase: &SnapshotPhase,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut all_parsed_snapshots = Vec::new();

        for worker in self.workers.iter() {
            let snapshot = worker
                .value()
                .clone()
                .get_snapshot(SnapshotRequest { peer_uuids: vec![] })
                .await?
                .into_inner();

            // Parse the NetworkSnapshot into structured format
            let parsed = ParsedSnapshot::from_proto(snapshot)?;
            all_parsed_snapshots.push(parsed);
        }

        std::fs::create_dir_all(&self.config.output_dir)?;

        // Create base filename without extension
        let base_name = phase
            .output_path
            .strip_suffix(".json")
            .unwrap_or(&phase.output_path);

        // Save each worker's snapshot with adjacency matrix (local view)
        for parsed_snapshot in all_parsed_snapshots.iter() {
            // Save full snapshot with statistics
            let snapshot_path = format!(
                "{}/{}_{}.json",
                self.config.output_dir, base_name, parsed_snapshot.worker_id
            );
            parsed_snapshot.save_to_file(&snapshot_path)?;

            // Save local adjacency matrix (only shows connections between peers on this worker)
            let adj_matrix_path = format!(
                "{}/{}_{}_{}.json",
                self.config.output_dir, base_name, parsed_snapshot.worker_id, "adjacency"
            );
            parsed_snapshot.save_adjacency_matrix(&adj_matrix_path)?;
        }

        // Create and save GLOBAL snapshot (aggregates all workers)
        let global_snapshot = GlobalSnapshot::merge_snapshots(all_parsed_snapshots.clone())?;

        // Save global snapshot
        let global_snapshot_path = format!(
            "{}/{}_global.json",
            self.config.output_dir, base_name
        );
        global_snapshot.save_to_file(&global_snapshot_path)?;

        // Save GLOBAL adjacency matrix (shows all connections across all workers)
        let global_adj_matrix_path = format!(
            "{}/{}_global_adjacency.json",
            self.config.output_dir, base_name
        );
        global_snapshot.save_adjacency_matrix(&global_adj_matrix_path)?;

        println!("  Global snapshot saved: {}", global_snapshot_path);
        println!("  Global adjacency matrix saved: {}", global_adj_matrix_path);
        println!("  Total peers: {}", global_snapshot.total_peers);
        println!("  Total edges (connections): {}", global_snapshot.summary.total_edges);
        println!("  Average degree: {:.2}", global_snapshot.summary.avg_degree);

        // Save combined summary for backwards compatibility
        let summary_path = format!("{}/{}", self.config.output_dir, phase.output_path);
        let combined_summary = serde_json::json!({
            "snapshot_count": global_snapshot.worker_count,
            "total_peers": global_snapshot.total_peers,
            "total_edges": global_snapshot.summary.total_edges,
            "avg_degree": global_snapshot.summary.avg_degree,
            "max_degree": global_snapshot.summary.max_degree,
            "min_degree": global_snapshot.summary.min_degree,
            "avg_dynamism": global_snapshot.summary.avg_dynamism,
        });

        std::fs::write(
            summary_path,
            serde_json::to_string_pretty(&combined_summary)?,
        )?;

        Ok(())
    }

    /// Execute full test plan sequentially
    pub async fn run_test_plan(
        &self,
        plan: TestPlan,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (idx, phase) in plan.phases.iter().enumerate() {
            println!("Executing phase {}/{}: {:?}", idx + 1, plan.phases.len(), phase);

            match phase {
                TestPhase::Bootstrap(p) => {
                    self.execute_bootstrap(p).await?;
                    println!("  Bootstrap completed: {} peers created", p.peer_indices.to_vec().len());
                }
                TestPhase::Churn(p) => {
                    self.execute_churn(p).await?;
                    println!("  Churn completed: {:?} pattern", p.pattern);
                }
                TestPhase::Query(p) => {
                    let results = self.execute_queries(p).await?;
                    let avg_recall = results.iter().map(|r| r.recall).sum::<f64>() / results.len() as f64;
                    println!(
                        "  Query phase completed: {} queries, avg recall: {:.4}",
                        results.len(),
                        avg_recall
                    );
                }
                TestPhase::Snapshot(p) => {
                    self.collect_snapshots(p).await?;
                    println!("  Snapshot saved: {}", p.output_path);
                }
                TestPhase::Wait(p) => {
                    tokio::time::sleep(Duration::from_millis(p.duration_ms)).await;
                    println!("  Wait completed: {}ms", p.duration_ms);
                }
            }
        }
        Ok(())
    }

    /// Write results to output directory
    pub async fn write_results(&self) -> Result<(), Box<dyn std::error::Error>> {
        let events_count = self.events.lock().await.len();

        let output = serde_json::json!({
            "config": self.config,
            "query_results": self.query_results.iter()
                .map(|entry| entry.value().clone())
                .collect::<Vec<_>>(),
            "events_count": events_count,
            // TODO: Serialize events properly (protobuf to json conversion needed)
        });

        let output_path = format!("{}/results.json", self.config.output_dir);
        std::fs::create_dir_all(&self.config.output_dir)?;
        std::fs::write(output_path, serde_json::to_string_pretty(&output)?)?;

        Ok(())
    }

    /// Helper: Determine which worker owns a peer by global index
    pub async fn get_worker_for_peer(&self, global_index: u64) -> Result<String, Box<dyn std::error::Error>> {
        let embeddings_len = self.embeddings.read().await.len() as u64;
        if embeddings_len == 0 {
            return Err("No embeddings loaded".into());
        }

        // Round-robin distribution: worker_idx = global_index % n_workers
        let n_workers = self.worker_configs.len() as u64;
        let worker_idx = global_index % n_workers;
        Ok(self.worker_configs[worker_idx as usize].worker_id.clone())
    }

    /// Get the worker address that owns a specific peer index
    pub async fn get_worker_address_for_peer(&self, global_index: u64) -> Result<String, Box<dyn std::error::Error>> {
        let embeddings_len = self.embeddings.read().await.len() as u64;
        if embeddings_len == 0 {
            return Err("No embeddings loaded".into());
        }

        // Round-robin distribution: worker_idx = global_index % n_workers
        let n_workers = self.worker_configs.len() as u64;
        let worker_idx = global_index % n_workers;
        Ok(self.worker_configs[worker_idx as usize].address.clone())
    }

    /// Helper: Assign peer indices to workers based on embedding partitioning
    /// Assign peers to workers using round-robin distribution
    /// Must match the partitioning used in distribute_embeddings
    ///
    /// Example: peer index 7 with 3 workers -> worker_idx = 7 % 3 = 1
    pub async fn assign_peers_to_workers(
        &self,
        peer_indices: &[u64],
    ) -> Result<HashMap<String, Vec<u64>>, Box<dyn std::error::Error>> {
        let mut assignments: HashMap<String, Vec<u64>> = HashMap::new();

        // Get total embeddings count
        let embeddings_len = self.embeddings.read().await.len() as u64;

        if embeddings_len == 0 {
            return Err("No embeddings loaded".into());
        }

        let num_workers = self.worker_configs.len() as u64;
        println!("[Coordinator] assign_peers: {} peers across {} workers (embeddings_len={})",
            peer_indices.len(), num_workers, embeddings_len);

        // Round-robin assignment: peer index N goes to worker (N % num_workers)
        for &peer_idx in peer_indices {
            if peer_idx >= embeddings_len {
                return Err(format!("Peer index {} exceeds embeddings length {}", peer_idx, embeddings_len).into());
            }

            // Determine which worker owns this peer index using round-robin
            let worker_idx = (peer_idx % num_workers) as usize;
            let worker_id = &self.worker_configs[worker_idx].worker_id;

            println!("[Coordinator] Peer {} assigned to worker {} (round-robin)",
                peer_idx, worker_id);

            assignments
                .entry(worker_id.clone())
                .or_insert_with(Vec::new)
                .push(peer_idx);
        }

        Ok(assignments)
    }

    /// Helper: Convert global index to UUID (deterministic mapping)
    pub fn global_index_to_uuid(&self, global_index: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; 64];
        bytes[0..8].copy_from_slice(&global_index.to_be_bytes());
        bytes
    }

    /// Helper: Convert our config types to proto SimConfigProto
    pub fn config_to_proto(&self) -> SimConfigProto {
        use crate::proto::protean::*;

        let snv = &self.config.sim_config.snv_config;

        SimConfigProto {
            snv_config: Some(SnvConfigProto {
                concurrency_limit: snv.concurrency_limit,
                timeout_ms: snv.timeout_ms,
                occlusion_threshold: snv.occlusion_threshold,
                drift_threshold: snv.drift_threshold,
                target_degree_ratio: snv.target_degree_ratio,
                dynamism_threshold: snv.dynamism_threshold,
                exploration_config: Some(ExplorationConfigProto {
                    converge_k: snv.exploration_config.converge_k,
                    converge_config: Some(QueryConfigProto {
                        search_list_size: snv.exploration_config.converge_config.search_list_size,
                        concurrency_limit: snv.exploration_config.converge_config.concurrency_limit,
                        share_floor: snv.exploration_config.converge_config.share_floor,
                        timeout: snv.exploration_config.converge_config.timeout,
                    }),
                    explore_k: snv.exploration_config.explore_k,
                    explore_config: Some(QueryConfigProto {
                        search_list_size: snv.exploration_config.explore_config.search_list_size,
                        concurrency_limit: snv.exploration_config.explore_config.concurrency_limit,
                        share_floor: snv.exploration_config.explore_config.share_floor,
                        timeout: snv.exploration_config.explore_config.timeout,
                    }),
                }),
                max_exploration_interval_secs: snv.max_exploration_interval_secs,
            }),
            max_peers: self.config.sim_config.max_peers,
            region: self.config.sim_config.region.clone(),
        }
    }
}

// Make Coordinator cloneable for sharing between service and main
impl<S: EmbeddingSpace> Clone for Coordinator<S> {
    fn clone(&self) -> Self {
        Self {
            workers: self.workers.clone(),
            worker_configs: self.worker_configs.clone(),
            embeddings: self.embeddings.clone(),
            query_embeddings: self.query_embeddings.clone(),
            events: self.events.clone(),
            query_results: self.query_results.clone(),
            active_peer_count: self.active_peer_count.clone(),
            current_bootstrapping: self.current_bootstrapping.clone(),
            pending_bootstrap: self.pending_bootstrap.clone(),
            bootstrap_target_idx: self.bootstrap_target_idx.clone(),
            config: self.config.clone(),
            _phantom: PhantomData,
        }
    }
}
