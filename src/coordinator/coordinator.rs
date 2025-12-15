use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashSet;
use protean::address::Address;
use protean::embedding_space::EmbeddingSpace;
use rand::seq::IteratorRandom;
use serde::Serialize;

use tokio::sync::{Mutex, Notify, RwLock};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::time::{interval, Instant};

use tonic::transport::{Channel, Server};
use tonic::{Request, Response, Status};

use crate::{
    proto::dist_sim::{
        coordinator_node_server::{CoordinatorNode, CoordinatorNodeServer},
        worker_node_client::WorkerNodeClient,
        Ack,
        BootstrapPeerRequest,
        CreatePeersRequest,
        DeletePeersRequest,
        DriftPeerProto,
        DriftPeerRequest,
        ProteanEventProto,
        QueryRequest,
        SnapshotRequest,
        WorkerInfo,
        protean_event_proto,
    },
    proto::protean::PeerProto,
    coordinator::test_plan::{
        Config,
        IndexRange,
        QueryConfigYaml,
        TestPhase,
    },
    coordinator::dataloader::DataSet,
    coordinator::snapshot::{ParsedSnapshot, GlobalSnapshot},
};

struct Worker {
    address: Address,
    client: WorkerNodeClient<Channel>,
}

/// Result of a gradual join operation
#[derive(Debug)]
pub struct GradualJoinResult {
    pub total_requested: usize,
    pub completed: usize,
    pub timed_out: usize,
    pub duration: Duration,
}

/// Result of a gradual leave operation
#[derive(Debug)]
pub struct GradualLeaveResult {
    pub total_requested: usize,
    pub completed: usize,
    pub timed_out: usize,
    pub duration: Duration,
}

/// Result of a combined gradual join and leave operation
#[derive(Debug)]
pub struct GradualJoinLeaveResult {
    pub join_total: usize,
    pub join_completed: usize,
    pub join_timed_out: usize,
    pub leave_total: usize,
    pub leave_completed: usize,
    pub leave_timed_out: usize,
    pub duration: Duration,
}

/// Result of an embedding drift operation
#[derive(Debug)]
pub struct EmbeddingDriftResult {
    pub total_peers: usize,
    pub requests_sent: usize,
    pub duration: Duration,
}

/// Result of a single query from one source peer
#[derive(Debug, Serialize)]
pub struct SingleQueryResult {
    pub source_peer: Vec<u8>,
    pub result_uuids: Vec<Vec<u8>>,
    pub hops: u32,
}

/// Result of a query across multiple source peers
#[derive(Debug, Serialize)]
pub struct QueryResult {
    pub query_idx: usize,
    pub ground_truth: Vec<Vec<u8>>,
    pub source_results: Vec<SingleQueryResult>,
    pub k_recalls: Vec<(usize, f64)>,  // (k, avg_recall) pairs
    pub duration: Duration,
}

/// Result of the entire query phase
#[derive(Debug, Serialize)]
pub struct QueryPhaseResult {
    pub results: Vec<QueryResult>,
    pub total_duration: Duration,
}

/// Result of a snapshot operation
#[derive(Debug, Serialize)]
pub struct SnapshotResult {
    pub num_workers: usize,
    pub total_peers: usize,
    pub duration: Duration,
}

pub struct Coordinator<S: EmbeddingSpace> {
    config: Config,

    data_set: DataSet<S>,

    event_tx: UnboundedSender<ProteanEventProto>,

    event_rx: Mutex<UnboundedReceiver<ProteanEventProto>>,

    workers: Arc<RwLock<Vec<Worker>>>,

    active_peers: Arc<DashSet<Vec<u8>>>,

    /// Notifier signaled when a worker connects
    worker_connected: Arc<Notify>,
}

impl<S: EmbeddingSpace> Coordinator<S> {
    /// Create a new Coordinator instance.
    pub fn new(config: Config, data_set: DataSet<S>) -> Self {
        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();

        Self {
            config,
            data_set,
            event_tx,
            event_rx: Mutex::new(event_rx),
            workers: Arc::new(RwLock::new(Vec::new())),
            active_peers: Arc::new(DashSet::new()),
            worker_connected: Arc::new(Notify::new()),
        }
    }

    /// Spawn the gRPC server and wait for all workers to connect.
    ///
    /// This starts the CoordinatorNode gRPC service in the background and blocks
    /// until `config.sim_config.num_workers` workers have registered.
    pub async fn spawn_worker_overlay(
        self: Arc<Self>,
        bind_addr: SocketAddr,
    ) -> Result<Arc<Self>, Box<dyn std::error::Error + Send + Sync>>
    where
        S: Send + Sync + 'static,
        S::EmbeddingData: Send + Sync,
    {
        let expected_workers = self.config.sim_config.num_workers;

        // Clone Arc for the gRPC server
        let server_self = self.clone();

        // Start gRPC server in background
        tokio::spawn(async move {
            if let Err(e) = Server::builder()
                .add_service(CoordinatorNodeServer::new(server_self))
                .serve(bind_addr)
                .await
            {
                tracing::error!("Coordinator gRPC server error: {}", e);
            }
        });

        tracing::info!("Coordinator gRPC server started on {}", bind_addr);

        // Wait for each worker to connect via notification
        for i in 0..expected_workers {
            self.worker_connected.notified().await;
            tracing::info!("Worker connected ({}/{})", i + 1, expected_workers);
        }

        tracing::info!("All {} workers connected", expected_workers);
        Ok(self)
    }

    /// Run all phases in the test plan sequentially.
    pub async fn run_test_plan(&self) {
        let phases = self.config.phases.clone();

        for (i, phase) in phases.into_iter().enumerate() {
            tracing::info!("Starting phase {}: {:?}", i + 1, phase);

            match phase {
                TestPhase::GradualJoin {
                    global_indices,
                    bootstrap_indices,
                    rate_per_sec,
                    bootstrap_timeout_sec,
                } => {
                    let result = self
                        .gradual_join(global_indices, bootstrap_indices, rate_per_sec, bootstrap_timeout_sec)
                        .await;
                    tracing::info!("GradualJoin complete: {:?}", result);
                }
                TestPhase::GradualLeave {
                    global_indices,
                    rate_per_sec,
                    bootstrap_timeout_sec,
                } => {
                    let result = self
                        .gradual_leave(global_indices, rate_per_sec, bootstrap_timeout_sec)
                        .await;
                    tracing::info!("GradualLeave complete: {:?}", result);
                }
                TestPhase::GradualJoinLeave {
                    join_global_indices,
                    leave_global_indices,
                    bootstrap_indices,
                    join_rate_per_sec,
                    leave_rate_per_sec,
                    bootstrap_timeout_sec,
                } => {
                    let result = self
                        .gradual_join_leave(
                            join_global_indices,
                            leave_global_indices,
                            bootstrap_indices,
                            join_rate_per_sec,
                            leave_rate_per_sec,
                            bootstrap_timeout_sec,
                        )
                        .await;
                    tracing::info!("GradualJoinLeave complete: {:?}", result);
                }
                TestPhase::EmbeddingDrift {
                    start_indices,
                    end_indices,
                    drift_steps,
                    duration_per_step_sec,
                } => {
                    let result = self
                        .embedding_drift(start_indices, end_indices, drift_steps, duration_per_step_sec)
                        .await;
                    tracing::info!("EmbeddingDrift complete: {:?}", result);
                }
                TestPhase::Wait { duration_sec } => {
                    self.wait(duration_sec).await;
                }
                TestPhase::Snapshot { output_path } => {
                    let result = self.snapshot(output_path).await;
                    tracing::info!("Snapshot complete: {:?}", result);
                }
                TestPhase::Query {
                    num_source_peers,
                    query_indices,
                    k,
                    query_config,
                    output_path,
                } => {
                    let result = self
                        .query(num_source_peers, query_indices, k, query_config, output_path)
                        .await;
                    tracing::info!("Query complete: {:?}", result);
                }
            }
        }

        tracing::info!("Test plan complete");
    }

    fn create_actor_uuid(embedding_idx: usize) -> Vec<u8> {
        (embedding_idx as u64).to_be_bytes().to_vec()
    }

    async fn get_num_workers(&self) -> usize {
        let workers_guard = self.workers.read().await;
        workers_guard.len()
    }

    async fn get_worker_for_embedding(&self, embedding_idx: usize) -> Option<usize> {
        if embedding_idx >= self.data_set.train.len() {
            return None;
        }

        let worker_count = self.get_num_workers().await;
        if worker_count == 0 {
            return None;
        }
        Some(embedding_idx % worker_count)
    }

    fn make_create_peer_requests(
        &self,
        embedding_idx: &[usize],
        worker_address: &str,
    ) -> CreatePeersRequest {
        let peers = embedding_idx
            .iter()
            .filter_map(|&idx| {
                self.data_set.train.get(idx).map(|embedding| {
                    PeerProto {
                        embedding: Some(embedding.clone().into()),
                        uuid: Self::create_actor_uuid(idx),
                        address: worker_address.to_string(),
                    }
                })
            })
            .collect();

        CreatePeersRequest { peers }
    }

    /// Create PeerProtos for bootstrap servers
    fn make_bootstrap_peer_protos(
        &self,
        bootstrap_idx: &[u64],
        worker_address: &str,
    ) -> Vec<PeerProto> {
        bootstrap_idx
            .iter()
            .filter_map(|&idx| {
                let idx = idx as usize;
                self.data_set.train.get(idx).map(|embedding| {
                    PeerProto {
                        embedding: Some(embedding.clone().into()),
                        uuid: Self::create_actor_uuid(idx),
                        address: worker_address.to_string(),
                    }
                })
            })
            .collect()
    }

    /// Gradually join peers to the network at a specified rate
    ///
    /// Sends CreatePeersRequests round-robin to workers, one peer per tick.
    /// Monitors for BootstrapCompleted events and returns when all peers complete
    /// or the timeout is reached after the last request.
    async fn gradual_join(
        &self,
        embedding_idx: IndexRange,
        bootstrap_idx: Vec<u64>,
        rate_per_sec: f64,
        bootstrap_timeout_sec: u64,
    ) -> GradualJoinResult {
        let start = Instant::now();
        let indices: Vec<usize> = (embedding_idx.start as usize..embedding_idx.end as usize).collect();
        let total_peers = indices.len();

        // Handle edge cases
        if total_peers == 0 {
            return GradualJoinResult {
                total_requested: 0,
                completed: 0,
                timed_out: 0,
                duration: start.elapsed(),
            };
        }

        let workers_guard = self.workers.read().await;
        let num_workers = workers_guard.len();

        if num_workers == 0 {
            return GradualJoinResult {
                total_requested: total_peers,
                completed: 0,
                timed_out: total_peers,
                duration: start.elapsed(),
            };
        }

        // Get first worker's address for bootstrap peers
        let bootstrap_worker_address = workers_guard[0].address.to_string();
        drop(workers_guard);

        // Build and send bootstrap server config to all workers
        let bootstrap_peers = self.make_bootstrap_peer_protos(&bootstrap_idx, &bootstrap_worker_address);
        let bootstrap_request = BootstrapPeerRequest {
            bs_server: bootstrap_peers,
        };

        {
            let mut workers_guard = self.workers.write().await;
            for worker in workers_guard.iter_mut() {
                if let Err(e) = worker.client.set_bootstrap_servers(Request::new(bootstrap_request.clone())).await {
                    tracing::error!("Failed to set bootstrap servers on worker {}: {}", worker.address, e);
                }
            }
        }

        // Calculate tick duration from rate
        let tick_duration = Duration::from_secs_f64(1.0 / rate_per_sec);

        // Initialize tracking with DashSet for thread-safe access
        let pending_peers: Arc<DashSet<Vec<u8>>> = Arc::new(DashSet::new());
        let mut completed_count = 0usize;
        let mut send_idx = 0usize;
        let mut worker_idx = 0usize;
        let mut last_send_time = Instant::now();
        let mut all_sent = false;

        let mut send_interval = interval(tick_duration);

        loop {
            tokio::select! {
                // Send next peer on interval tick
                _ = send_interval.tick(), if !all_sent => {
                    let idx = indices[send_idx];

                    // Get worker and send request
                    let mut workers_guard = self.workers.write().await;
                    let worker = &mut workers_guard[worker_idx];
                    let worker_address = worker.address.to_string();

                    let request = self.make_create_peer_requests(&[idx], &worker_address);

                    // Track the peer UUID
                    let peer_uuid = Self::create_actor_uuid(idx);
                    pending_peers.insert(peer_uuid);

                    // Send to worker
                    if let Err(e) = worker.client.create_peers(Request::new(request)).await {
                        tracing::error!("Failed to create peer {} on worker {}: {}", idx, worker.address, e);
                    }

                    drop(workers_guard);

                    // Advance indices
                    send_idx += 1;
                    worker_idx = (worker_idx + 1) % num_workers;
                    last_send_time = Instant::now();

                    if send_idx >= total_peers {
                        all_sent = true;
                    }
                }

                // Process incoming events
                Some(event) = async { self.event_rx.lock().await.recv().await } => {
                    if let Some(protean_event_proto::Event::BootstrapCompleted(bc)) = event.event {
                        if pending_peers.remove(&bc.peer_uuid).is_some() {
                            self.active_peers.insert(bc.peer_uuid);
                            completed_count += 1;
                        }
                    }

                    // Check if done
                    if all_sent && pending_peers.is_empty() {
                        break;
                    }
                }

                // Timeout check - poll periodically
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    if all_sent && last_send_time.elapsed() > Duration::from_secs(bootstrap_timeout_sec) {
                        tracing::warn!(
                            "Gradual join timed out with {} peers still pending",
                            pending_peers.len()
                        );
                        break;
                    }
                }
            }
        }

        GradualJoinResult {
            total_requested: total_peers,
            completed: completed_count,
            timed_out: pending_peers.len(),
            duration: start.elapsed(),
        }
    }

    /// Gradually remove peers from the network at a specified rate
    ///
    /// Sends DeletePeersRequests round-robin to workers, one peer per tick.
    /// Monitors for StateChanged events with to_state="Shutdown" and returns
    /// when all peers complete or the timeout is reached after the last request.
    async fn gradual_leave(
        &self,
        embedding_idx: IndexRange,
        rate_per_sec: f64,
        bootstrap_timeout_sec: u64,
    ) -> GradualLeaveResult {
        let start = Instant::now();
        let indices: Vec<usize> = (embedding_idx.start as usize..embedding_idx.end as usize).collect();
        let total_peers = indices.len();

        // Handle edge cases
        if total_peers == 0 {
            return GradualLeaveResult {
                total_requested: 0,
                completed: 0,
                timed_out: 0,
                duration: start.elapsed(),
            };
        }

        let workers_guard = self.workers.read().await;
        let num_workers = workers_guard.len();
        drop(workers_guard);

        if num_workers == 0 {
            return GradualLeaveResult {
                total_requested: total_peers,
                completed: 0,
                timed_out: total_peers,
                duration: start.elapsed(),
            };
        }

        // Calculate tick duration from rate
        let tick_duration = Duration::from_secs_f64(1.0 / rate_per_sec);

        // Initialize tracking with DashSet for thread-safe access
        let pending_peers: Arc<DashSet<Vec<u8>>> = Arc::new(DashSet::new());
        let mut completed_count = 0usize;
        let mut send_idx = 0usize;
        let mut worker_idx = 0usize;
        let mut last_send_time = Instant::now();
        let mut all_sent = false;

        let mut send_interval = interval(tick_duration);

        loop {
            tokio::select! {
                // Send delete request on interval tick (one peer per tick, round-robin workers)
                _ = send_interval.tick(), if !all_sent => {
                    let idx = indices[send_idx];
                    let peer_uuid = Self::create_actor_uuid(idx);

                    // Get worker and send delete request
                    let mut workers_guard = self.workers.write().await;
                    let worker = &mut workers_guard[worker_idx];

                    let request = DeletePeersRequest {
                        uuids: vec![peer_uuid.clone()],
                    };

                    pending_peers.insert(peer_uuid);

                    if let Err(e) = worker.client.delete_peers(Request::new(request)).await {
                        tracing::error!("Failed to delete peer {} on worker {}: {}", idx, worker.address, e);
                    }

                    drop(workers_guard);

                    // Advance indices
                    send_idx += 1;
                    worker_idx = (worker_idx + 1) % num_workers;
                    last_send_time = Instant::now();

                    if send_idx >= total_peers {
                        all_sent = true;
                    }
                }

                // Process incoming events - look for StateChanged with to_state = "Shutdown"
                Some(event) = async { self.event_rx.lock().await.recv().await } => {
                    if let Some(protean_event_proto::Event::StateChanged(sc)) = event.event {
                        if sc.to_state == "Shutdown" {
                            if pending_peers.remove(&sc.peer_uuid).is_some() {
                                self.active_peers.remove(&sc.peer_uuid);
                                completed_count += 1;
                            }
                        }
                    }

                    // Check if done
                    if all_sent && pending_peers.is_empty() {
                        break;
                    }
                }

                // Timeout check - poll periodically
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    if all_sent && last_send_time.elapsed() > Duration::from_secs(bootstrap_timeout_sec) {
                        tracing::warn!(
                            "Gradual leave timed out with {} peers still pending",
                            pending_peers.len()
                        );
                        break;
                    }
                }
            }
        }

        GradualLeaveResult {
            total_requested: total_peers,
            completed: completed_count,
            timed_out: pending_peers.len(),
            duration: start.elapsed(),
        }
    }

    /// Gradually join and leave peers simultaneously at specified rates
    ///
    /// Combines join and leave operations with independent rates.
    /// Monitors for both BootstrapCompleted (joins) and StateChanged/Shutdown (leaves).
    /// Returns when all operations complete or timeout is reached.
    async fn gradual_join_leave(
        &self,
        join_idx: IndexRange,
        leave_idx: IndexRange,
        bootstrap_idx: Vec<u64>,
        join_rate_per_sec: f64,
        leave_rate_per_sec: f64,
        bootstrap_timeout_sec: u64,
    ) -> GradualJoinLeaveResult {
        let start = Instant::now();
        let join_indices: Vec<usize> = (join_idx.start as usize..join_idx.end as usize).collect();
        let leave_indices: Vec<usize> = (leave_idx.start as usize..leave_idx.end as usize).collect();
        let total_joins = join_indices.len();
        let total_leaves = leave_indices.len();

        // Handle edge case: nothing to do
        if total_joins == 0 && total_leaves == 0 {
            return GradualJoinLeaveResult {
                join_total: 0,
                join_completed: 0,
                join_timed_out: 0,
                leave_total: 0,
                leave_completed: 0,
                leave_timed_out: 0,
                duration: start.elapsed(),
            };
        }

        let workers_guard = self.workers.read().await;
        let num_workers = workers_guard.len();

        if num_workers == 0 {
            return GradualJoinLeaveResult {
                join_total: total_joins,
                join_completed: 0,
                join_timed_out: total_joins,
                leave_total: total_leaves,
                leave_completed: 0,
                leave_timed_out: total_leaves,
                duration: start.elapsed(),
            };
        }

        // Get first worker's address for bootstrap peers
        let bootstrap_worker_address = workers_guard[0].address.to_string();
        drop(workers_guard);

        // Build and send bootstrap server config to all workers
        let bootstrap_peers = self.make_bootstrap_peer_protos(&bootstrap_idx, &bootstrap_worker_address);
        let bootstrap_request = BootstrapPeerRequest {
            bs_server: bootstrap_peers,
        };

        {
            let mut workers_guard = self.workers.write().await;
            for worker in workers_guard.iter_mut() {
                if let Err(e) = worker.client.set_bootstrap_servers(Request::new(bootstrap_request.clone())).await {
                    tracing::error!("Failed to set bootstrap servers on worker {}: {}", worker.address, e);
                }
            }
        }

        // Calculate tick durations from rates
        let join_tick_duration = Duration::from_secs_f64(1.0 / join_rate_per_sec);
        let leave_tick_duration = Duration::from_secs_f64(1.0 / leave_rate_per_sec);

        // Initialize tracking with DashSets for thread-safe access
        let pending_joins: Arc<DashSet<Vec<u8>>> = Arc::new(DashSet::new());
        let pending_leaves: Arc<DashSet<Vec<u8>>> = Arc::new(DashSet::new());
        let mut join_completed_count = 0usize;
        let mut leave_completed_count = 0usize;

        let mut join_send_idx = 0usize;
        let mut leave_send_idx = 0usize;
        let mut join_worker_idx = 0usize;
        let mut leave_worker_idx = 0usize;

        let mut last_activity_time = Instant::now();
        let mut all_joins_sent = total_joins == 0;
        let mut all_leaves_sent = total_leaves == 0;

        let mut join_interval = interval(join_tick_duration);
        let mut leave_interval = interval(leave_tick_duration);

        loop {
            tokio::select! {
                // Send next join request on interval tick
                _ = join_interval.tick(), if !all_joins_sent => {
                    let idx = join_indices[join_send_idx];

                    // Get worker and send request
                    let mut workers_guard = self.workers.write().await;
                    let worker = &mut workers_guard[join_worker_idx];
                    let worker_address = worker.address.to_string();

                    let request = self.make_create_peer_requests(&[idx], &worker_address);

                    // Track the peer UUID
                    let peer_uuid = Self::create_actor_uuid(idx);
                    pending_joins.insert(peer_uuid);

                    // Send to worker
                    if let Err(e) = worker.client.create_peers(Request::new(request)).await {
                        tracing::error!("Failed to create peer {} on worker {}: {}", idx, worker.address, e);
                    }

                    drop(workers_guard);

                    // Advance indices
                    join_send_idx += 1;
                    join_worker_idx = (join_worker_idx + 1) % num_workers;
                    last_activity_time = Instant::now();

                    if join_send_idx >= total_joins {
                        all_joins_sent = true;
                    }
                }

                // Send next leave request on interval tick
                _ = leave_interval.tick(), if !all_leaves_sent => {
                    let idx = leave_indices[leave_send_idx];
                    let peer_uuid = Self::create_actor_uuid(idx);

                    // Get worker and send delete request
                    let mut workers_guard = self.workers.write().await;
                    let worker = &mut workers_guard[leave_worker_idx];

                    let request = DeletePeersRequest {
                        uuids: vec![peer_uuid.clone()],
                    };

                    pending_leaves.insert(peer_uuid);

                    if let Err(e) = worker.client.delete_peers(Request::new(request)).await {
                        tracing::error!("Failed to delete peer {} on worker {}: {}", idx, worker.address, e);
                    }

                    drop(workers_guard);

                    // Advance indices
                    leave_send_idx += 1;
                    leave_worker_idx = (leave_worker_idx + 1) % num_workers;
                    last_activity_time = Instant::now();

                    if leave_send_idx >= total_leaves {
                        all_leaves_sent = true;
                    }
                }

                // Process incoming events
                Some(event) = async { self.event_rx.lock().await.recv().await } => {
                    match event.event {
                        Some(protean_event_proto::Event::BootstrapCompleted(bc)) => {
                            if pending_joins.remove(&bc.peer_uuid).is_some() {
                                self.active_peers.insert(bc.peer_uuid);
                                join_completed_count += 1;
                                last_activity_time = Instant::now();
                            }
                        }
                        Some(protean_event_proto::Event::StateChanged(sc)) => {
                            if sc.to_state == "Shutdown" {
                                if pending_leaves.remove(&sc.peer_uuid).is_some() {
                                    self.active_peers.remove(&sc.peer_uuid);
                                    leave_completed_count += 1;
                                    last_activity_time = Instant::now();
                                }
                            }
                        }
                        _ => {}
                    }

                    // Check if done
                    if all_joins_sent && all_leaves_sent && pending_joins.is_empty() && pending_leaves.is_empty() {
                        break;
                    }
                }

                // Timeout check - poll periodically
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    if all_joins_sent && all_leaves_sent &&
                       last_activity_time.elapsed() > Duration::from_secs(bootstrap_timeout_sec) {
                        tracing::warn!(
                            "Gradual join/leave timed out with {} joins and {} leaves still pending",
                            pending_joins.len(),
                            pending_leaves.len()
                        );
                        break;
                    }
                }
            }
        }

        GradualJoinLeaveResult {
            join_total: total_joins,
            join_completed: join_completed_count,
            join_timed_out: pending_joins.len(),
            leave_total: total_leaves,
            leave_completed: leave_completed_count,
            leave_timed_out: pending_leaves.len(),
            duration: start.elapsed(),
        }
    }

    /// Drift peer embeddings toward target positions over time
    ///
    /// Sends drift commands to all specified peers, then waits for the drift
    /// to complete. No event feedback is provided, so we wait 1.5x the total
    /// drift duration to allow the network to settle.
    async fn embedding_drift(
        &self,
        start_indices: IndexRange,
        end_indices: IndexRange,
        drift_steps: u32,
        duration_per_step_sec: u64,
    ) -> EmbeddingDriftResult {
        let start = Instant::now();

        let start_idx_vec: Vec<usize> = (start_indices.start as usize..start_indices.end as usize).collect();
        let end_idx_vec: Vec<usize> = (end_indices.start as usize..end_indices.end as usize).collect();
        let total_peers = start_idx_vec.len();

        // Validate 1:1 mapping
        if start_idx_vec.len() != end_idx_vec.len() {
            tracing::error!(
                "Embedding drift requires 1:1 mapping: start has {} indices, end has {}",
                start_idx_vec.len(),
                end_idx_vec.len()
            );
            return EmbeddingDriftResult {
                total_peers,
                requests_sent: 0,
                duration: start.elapsed(),
            };
        }

        // Handle edge case: nothing to do
        if total_peers == 0 {
            return EmbeddingDriftResult {
                total_peers: 0,
                requests_sent: 0,
                duration: start.elapsed(),
            };
        }

        let workers_guard = self.workers.read().await;
        let num_workers = workers_guard.len();
        drop(workers_guard);

        if num_workers == 0 {
            tracing::error!("No workers available for embedding drift");
            return EmbeddingDriftResult {
                total_peers,
                requests_sent: 0,
                duration: start.elapsed(),
            };
        }

        // Group drifts by worker
        let mut worker_drifts: Vec<Vec<DriftPeerProto>> = vec![Vec::new(); num_workers];

        for (&start_idx, &end_idx) in start_idx_vec.iter().zip(end_idx_vec.iter()) {
            let worker_idx = start_idx % num_workers;

            // Get target embedding from dataset
            let target_embedding = match self.data_set.train.get(end_idx) {
                Some(emb) => emb.clone(),
                None => {
                    tracing::warn!("Target embedding index {} out of bounds, skipping", end_idx);
                    continue;
                }
            };

            let drift_proto = DriftPeerProto {
                uuid: Self::create_actor_uuid(start_idx),
                target_embedding: Some(target_embedding.into()),
                drift_steps: drift_steps as u64,
                duration_per_step_sec,
            };

            worker_drifts[worker_idx].push(drift_proto);
        }

        // Send one DriftPeerRequest per worker
        let mut requests_sent = 0usize;
        {
            let mut workers_guard = self.workers.write().await;
            for (worker_idx, drifts) in worker_drifts.into_iter().enumerate() {
                if drifts.is_empty() {
                    continue;
                }

                let num_drifts = drifts.len();
                let request = DriftPeerRequest { drifts };

                let worker = &mut workers_guard[worker_idx];
                if let Err(e) = worker.client.drift_peer(Request::new(request)).await {
                    tracing::error!(
                        "Failed to send drift request to worker {}: {}",
                        worker.address,
                        e
                    );
                } else {
                    requests_sent += num_drifts;
                }
            }
        }

        // Wait for drift to complete: drift_steps * duration_per_step_sec * 1.5
        let total_drift_duration_secs = (drift_steps as u64) * duration_per_step_sec;
        let wait_duration = Duration::from_secs_f64(total_drift_duration_secs as f64 * 1.5);

        tracing::info!(
            "Embedding drift started for {} peers, waiting {} seconds for completion",
            requests_sent,
            wait_duration.as_secs()
        );

        tokio::time::sleep(wait_duration).await;

        EmbeddingDriftResult {
            total_peers,
            requests_sent,
            duration: start.elapsed(),
        }
    }

    /// Wait for a specified duration
    ///
    /// Simple pause to allow the network to stabilize between phases.
    async fn wait(&self, duration_sec: u64) {
        tracing::info!("Waiting for {} seconds", duration_sec);
        tokio::time::sleep(Duration::from_secs(duration_sec)).await;
        tracing::info!("Wait complete");
    }

    /// Convert UUID bytes back to embedding index
    fn uuid_to_embedding_idx(uuid: &[u8]) -> Option<usize> {
        if uuid.len() == std::mem::size_of::<usize>() {
            let mut bytes = [0u8; std::mem::size_of::<usize>()];
            bytes.copy_from_slice(uuid);
            Some(usize::from_ne_bytes(bytes))
        } else {
            None
        }
    }

    /// Compute ground truth by brute-forcing distances to all active peers
    fn compute_ground_truth(&self, query_embedding: &S::EmbeddingData, k: usize) -> Vec<Vec<u8>> {
        let mut distances: Vec<(Vec<u8>, S::DistanceValue)> = self
            .active_peers
            .iter()
            .filter_map(|uuid_ref| {
                let uuid = uuid_ref.clone();
                let idx = Self::uuid_to_embedding_idx(&uuid)?;
                let peer_embedding = self.data_set.train.get(idx)?;
                let distance = S::distance(query_embedding, peer_embedding);
                Some((uuid, distance))
            })
            .collect();

        // Sort by distance (ascending)
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k UUIDs
        distances.into_iter().take(k).map(|(uuid, _)| uuid).collect()
    }

    /// Calculate recall: intersection of results and ground truth divided by k
    fn calculate_recall(results: &[Vec<u8>], groundtruth: &[Vec<u8>], k: usize) -> f64 {
        if k == 0 {
            return 1.0;
        }

        let gt_set: HashSet<&Vec<u8>> = groundtruth.iter().take(k).collect();
        let result_set: Vec<&Vec<u8>> = results.iter().take(k).collect();

        let intersection = result_set.iter().filter(|r| gt_set.contains(*r)).count();
        intersection as f64 / k as f64
    }

    /// Execute queries from multiple source peers and calculate recall
    async fn query(
        &self,
        num_source_peers: usize,
        query_indices: Vec<usize>,
        k_values: Vec<usize>,
        query_config: QueryConfigYaml,
        output_path: String,
    ) -> QueryPhaseResult {
        let phase_start = Instant::now();
        let mut results = Vec::new();

        // Get max k for queries
        let max_k = *k_values.iter().max().unwrap_or(&10);

        // Convert query config to proto
        let timeout_ms = self.config.sim_config.protean_config.timeout;
        let query_config_proto = query_config.into_query_config(timeout_ms);

        for query_idx in query_indices {
            let query_start = Instant::now();

            // Get query embedding
            let query_embedding = match self.data_set.test.get(query_idx) {
                Some(emb) => emb.clone(),
                None => {
                    tracing::warn!("Query index {} out of bounds, skipping", query_idx);
                    continue;
                }
            };

            // Compute ground truth
            let ground_truth = self.compute_ground_truth(&query_embedding, max_k);

            // Select N random source peers
            let mut rng = rand::thread_rng();
            let source_peers: Vec<Vec<u8>> = self
                .active_peers
                .iter()
                .map(|r| r.clone())
                .choose_multiple(&mut rng, num_source_peers);

            if source_peers.is_empty() {
                tracing::warn!("No active peers for query {}, skipping", query_idx);
                continue;
            }

            // Send queries and collect query_uuids
            let mut pending_queries: HashSet<Vec<u8>> = HashSet::new();
            let mut query_to_source: std::collections::HashMap<Vec<u8>, Vec<u8>> = std::collections::HashMap::new();

            {
                let mut workers_guard = self.workers.write().await;
                let num_workers = workers_guard.len();

                for source_uuid in &source_peers {
                    // Determine which worker has this peer
                    let source_idx = match Self::uuid_to_embedding_idx(source_uuid) {
                        Some(idx) => idx,
                        None => continue,
                    };
                    let worker_idx = source_idx % num_workers;

                    let request = QueryRequest {
                        source_peer_uuid: source_uuid.clone(),
                        query_embedding: Some(query_embedding.clone().into()),
                        k: max_k as u32,
                        config: Some(query_config_proto.clone().into()),
                    };

                    let worker = &mut workers_guard[worker_idx];
                    match worker.client.execute_query(Request::new(request)).await {
                        Ok(response) => {
                            let query_uuid = response.into_inner().query_uuid;
                            pending_queries.insert(query_uuid.clone());
                            query_to_source.insert(query_uuid, source_uuid.clone());
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to send query from peer {:?} to worker {}: {}",
                                source_uuid,
                                worker.address,
                                e
                            );
                        }
                    }
                }
            }

            // Collect results from QueryCompletedEvents
            let mut source_results: Vec<SingleQueryResult> = Vec::new();
            let timeout = Duration::from_secs(30); // Query timeout
            let deadline = Instant::now() + timeout;

            while !pending_queries.is_empty() && Instant::now() < deadline {
                tokio::select! {
                    Some(event) = async { self.event_rx.lock().await.recv().await } => {
                        if let Some(protean_event_proto::Event::QueryCompleted(qc)) = event.event {
                            if pending_queries.remove(&qc.query_uuid) {
                                let source_peer = query_to_source
                                    .remove(&qc.query_uuid)
                                    .unwrap_or_default();

                                let result_uuids: Vec<Vec<u8>> = qc
                                    .candidates
                                    .into_iter()
                                    .filter_map(|c| c.peer.map(|p| p.uuid))
                                    .collect();

                                source_results.push(SingleQueryResult {
                                    source_peer,
                                    result_uuids,
                                    hops: qc.hops,
                                });
                            }
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {}
                }
            }

            if !pending_queries.is_empty() {
                tracing::warn!(
                    "Query {} timed out with {} pending results",
                    query_idx,
                    pending_queries.len()
                );
            }

            // Calculate recall for each k value
            let k_recalls: Vec<(usize, f64)> = k_values
                .iter()
                .map(|&k| {
                    if source_results.is_empty() {
                        return (k, 0.0);
                    }
                    let avg_recall: f64 = source_results
                        .iter()
                        .map(|sr| Self::calculate_recall(&sr.result_uuids, &ground_truth, k))
                        .sum::<f64>()
                        / source_results.len() as f64;
                    (k, avg_recall)
                })
                .collect();

            tracing::info!(
                "Query {} completed: {} sources, recalls: {:?}",
                query_idx,
                source_results.len(),
                k_recalls
            );

            results.push(QueryResult {
                query_idx,
                ground_truth,
                source_results,
                k_recalls,
                duration: query_start.elapsed(),
            });
        }

        // Write results to output file
        let full_path = format!("{}/{}_query_results.json", self.config.sim_config.output_dir, output_path);
        match serde_json::to_string_pretty(&results) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&full_path, json) {
                    tracing::error!("Failed to write query results to {}: {}", full_path, e);
                } else {
                    tracing::info!("Query results written to: {}", full_path);
                }
            }
            Err(e) => tracing::error!("Failed to serialize query results: {}", e),
        }

        QueryPhaseResult {
            results,
            total_duration: phase_start.elapsed(),
        }
    }

    /// Take a snapshot of the network state and save adjacency matrices
    async fn snapshot(&self, output_path: String) -> SnapshotResult {
        let start = Instant::now();

        // Collect all active peer UUIDs
        let peer_uuids: Vec<Vec<u8>> = self
            .active_peers
            .iter()
            .map(|r| r.clone())
            .collect();

        let total_peers = peer_uuids.len();

        if total_peers == 0 {
            tracing::warn!("No active peers to snapshot");
            return SnapshotResult {
                num_workers: 0,
                total_peers: 0,
                duration: start.elapsed(),
            };
        }

        // Group peer UUIDs by worker
        let workers_guard = self.workers.read().await;
        let num_workers = workers_guard.len();
        drop(workers_guard);

        if num_workers == 0 {
            tracing::error!("No workers available for snapshot");
            return SnapshotResult {
                num_workers: 0,
                total_peers,
                duration: start.elapsed(),
            };
        }

        let mut worker_peers: Vec<Vec<Vec<u8>>> = vec![Vec::new(); num_workers];
        for uuid in &peer_uuids {
            if let Some(idx) = Self::uuid_to_embedding_idx(uuid) {
                let worker_idx = idx % num_workers;
                worker_peers[worker_idx].push(uuid.clone());
            }
        }

        // Send snapshot requests to each worker
        let mut parsed_snapshots = Vec::new();

        {
            let mut workers_guard = self.workers.write().await;
            for (worker_idx, peers) in worker_peers.into_iter().enumerate() {
                if peers.is_empty() {
                    continue;
                }

                let request = SnapshotRequest {
                    peer_uuids: peers,
                    name: output_path.clone(),
                };

                let worker = &mut workers_guard[worker_idx];
                match worker.client.get_snapshot(Request::new(request)).await {
                    Ok(response) => {
                        let network_snapshot = response.into_inner();
                        match ParsedSnapshot::from_proto(network_snapshot) {
                            Ok(parsed) => {
                                tracing::info!(
                                    "Received snapshot from worker {} with {} peers",
                                    worker.address,
                                    parsed.peers.len()
                                );
                                parsed_snapshots.push(parsed);
                            }
                            Err(e) => {
                                tracing::error!(
                                    "Failed to parse snapshot from worker {}: {}",
                                    worker.address,
                                    e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            "Failed to get snapshot from worker {}: {}",
                            worker.address,
                            e
                        );
                    }
                }
            }
        }

        // Merge snapshots into global view
        if parsed_snapshots.is_empty() {
            tracing::error!("No snapshots received from workers");
            return SnapshotResult {
                num_workers,
                total_peers,
                duration: start.elapsed(),
            };
        }

        let global_snapshot = match GlobalSnapshot::merge_snapshots(parsed_snapshots) {
            Ok(gs) => gs,
            Err(e) => {
                tracing::error!("Failed to merge snapshots: {}", e);
                return SnapshotResult {
                    num_workers,
                    total_peers,
                    duration: start.elapsed(),
                };
            }
        };

        // Save adjacency matrices
        let output_dir = &self.config.sim_config.output_dir;

        // Routable adjacency matrix
        let routable_path = format!("{}/{}_routable_adjacency.json", output_dir, output_path);
        if let Err(e) = global_snapshot.save_adjacency_matrix(&routable_path) {
            tracing::error!("Failed to save routable adjacency matrix: {}", e);
        } else {
            tracing::info!("Saved routable adjacency matrix to: {}", routable_path);
        }

        // All peers adjacency matrix
        let all_path = format!("{}/{}_all_adjacency.json", output_dir, output_path);
        if let Err(e) = global_snapshot.save_all_peers_adjacency_matrix(&all_path) {
            tracing::error!("Failed to save all-peers adjacency matrix: {}", e);
        } else {
            tracing::info!("Saved all-peers adjacency matrix to: {}", all_path);
        }

        // Also save the full snapshot data
        let snapshot_path = format!("{}/{}_snapshot.json", output_dir, output_path);
        if let Err(e) = global_snapshot.save_to_file(&snapshot_path) {
            tracing::error!("Failed to save snapshot: {}", e);
        } else {
            tracing::info!("Saved snapshot to: {}", snapshot_path);
        }

        tracing::info!(
            "Snapshot complete: {} workers, {} peers, {} edges",
            global_snapshot.worker_count,
            global_snapshot.total_peers,
            global_snapshot.summary.total_edges
        );

        SnapshotResult {
            num_workers: global_snapshot.worker_count,
            total_peers: global_snapshot.total_peers,
            duration: start.elapsed(),
        }
    }
}

#[tonic::async_trait]
impl<S: EmbeddingSpace + Send + Sync + 'static> CoordinatorNode for Coordinator<S>
where
    S::EmbeddingData: Send + Sync,
{
    async fn register_worker(
        &self,
        request: Request<WorkerInfo>,
    ) -> Result<Response<Ack>, Status> {
        let worker_info = request.into_inner();
        let address: Address = worker_info.address.parse().map_err(|e| {
            Status::invalid_argument(format!("Invalid worker address: {}", e))
        })?;

        // Create client connection to new worker
        let channel = tonic::transport::Channel::from_shared(format!("http://{}", address))
            .map_err(|e| Status::internal(format!("Failed to create channel: {}", e)))?
            .connect()
            .await
            .map_err(|e| Status::unavailable(format!("Failed to connect to worker: {}", e)))?;

        let mut new_client = WorkerNodeClient::new(channel);

        // Get write lock to access existing workers
        let mut workers_guard = self.workers.write().await;

        // Tell new worker about all existing workers
        for existing_worker in workers_guard.iter() {
            let existing_info = WorkerInfo {
                address: existing_worker.address.to_string(),
            };
            if let Err(e) = new_client.register_worker(Request::new(existing_info)).await {
                tracing::error!(
                    "Failed to inform new worker {} about existing worker {}: {}",
                    address,
                    existing_worker.address,
                    e
                );
            }
        }

        // Tell all existing workers about the new worker
        let new_worker_info = WorkerInfo {
            address: address.to_string(),
        };
        for existing_worker in workers_guard.iter_mut() {
            if let Err(e) = existing_worker.client.register_worker(Request::new(new_worker_info.clone())).await {
                tracing::error!(
                    "Failed to inform existing worker {} about new worker {}: {}",
                    existing_worker.address,
                    address,
                    e
                );
            }
        }

        // Add new worker to list
        let worker = Worker {
            address: address.clone(),
            client: new_client,
        };
        tracing::info!("Registered worker: {}", address);
        workers_guard.push(worker);

        // Notify that a worker has connected
        self.worker_connected.notify_one();

        Ok(Response::new(Ack {}))
    }

    async fn forward_event(
        &self,
        request: Request<ProteanEventProto>,
    ) -> Result<Response<Ack>, Status> {
        let event = request.into_inner();

        self.event_tx
            .send(event)
            .map_err(|e| Status::internal(format!("Failed to forward event: {}", e)))?;

        Ok(Response::new(Ack {}))
    }
}

#[tonic::async_trait]
impl<S: EmbeddingSpace + Send + Sync + 'static> CoordinatorNode for Arc<Coordinator<S>>
where
    S::EmbeddingData: Send + Sync,
{
    async fn register_worker(
        &self,
        request: Request<WorkerInfo>,
    ) -> Result<Response<Ack>, Status> {
        (**self).register_worker(request).await
    }

    async fn forward_event(
        &self,
        request: Request<ProteanEventProto>,
    ) -> Result<Response<Ack>, Status> {
        (**self).forward_event(request).await
    }
}