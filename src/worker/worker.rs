//! Worker implementation for distributed Protean simulation
//!
//! ## Architecture
//! - **Broker Thread**: Routes protocol messages to other workers via gRPC
//! - **Event Processor Thread**: Processes local events and forwards to coordinator via ForwardEvent RPC
//! - **Actor Management**: Spawns and manages ActorProtean instances for each peer
//!

use std::sync::{Arc, RwLock};
use std::time::Duration;

use tokio::sync::mpsc::{UnboundedSender, UnboundedReceiver, unbounded_channel};
use tokio::sync::oneshot;
use tokio::runtime::{Runtime, Builder, Handle};
use tokio::task::JoinHandle;
use tonic::{transport::Channel, Request, Response, Status};

use dashmap::DashMap;
use crate::proto::dist_sim::{
    worker_node_server::WorkerNode, worker_node_client::WorkerNodeClient,
    coordinator_client::CoordinatorClient, RouteMessageRequest, Ack,
    SimConfigProto, LoadEmbeddingsRequest, CreatePeersRequest, DeletePeersRequest,
    BootstrapPeerRequest, ChurnPatternRequest, ChurnPatternType, BootstrapPeerInfo,
    QueryRequest, QueryResponse, SnapshotRequest, NetworkSnapshot, WorkerInfo,
    ProteanEventProto, PingRequest,
    protean_event_proto, ProteanEventType, StateChangedEvent, QueryCompletedEvent,
    BootstrapConvergingCompletedEvent, BootstrapCompletedEvent,
};
use crate::proto::protean::{
    QueryConfigProto, PeerProto,
    QueryCandidateProto, ProteanMessageProto,
};
use crate::worker::actor::{ControlMessage, ActorProtean};

use protean::{
    ProteanConfig,
    Peer,
    OutMessage,
    SerializationError,
    ProteanMessage,
    ProteanPeer,
    QueryConfig,
    SnvConfig,
    embedding_space::EmbeddingSpace,
    address::Address,
    protean::ProteanEvent,
    proto::TensorProto,
    uuid::Uuid,
    query::QueryResult,
};

/// Commands sent to the churn background thread
enum PeerControlerCommand {
    Start(ChurnPatternRequest),
    Stop,
}

pub struct EmbeddingPool<S: EmbeddingSpace> {
    embeddings: Vec<S::EmbeddingData>,
    offset: usize,
}

pub struct Worker<S: EmbeddingSpace> {
    worker_id: String,
    my_address: Arc<RwLock<Address>>,

    /// gRPC connections to other worker
    workers: Arc<DashMap<Address, WorkerNodeClient<Channel>>>,

    /// gRPC connection to the coordinator
    coordinator: Arc<RwLock<Option<CoordinatorClient<Channel>>>>,

    /// Channels to deliver message to other nodes
    actor_protocol_channels: Arc<DashMap<Uuid, UnboundedSender<ProteanMessage<S>>>>,
    actor_control_channels: Arc<DashMap<Uuid, UnboundedSender<ControlMessage<S>>>>,

    /// Allow peers to send remote messages
    remote_msg_tx: UnboundedSender<OutMessage<S>>,

    /// Event channel for peers to send events
    event_tx: UnboundedSender<ProteanEvent<S>>,

    /// Simulation configuration from coordinator
    config: Arc<RwLock<Option<SimConfigProto>>>,

    /// Tokio runtime for async tasks
    runtime: Arc<Runtime>,

    /// Actors on worker
    actor_handles: Vec<JoinHandle<()>>,
    /// Msg Broker to other handlers
    broker_handle: Option<JoinHandle<()>>,
    /// Event Q to Coordinator fwding thread
    event_processor_handle: Option<JoinHandle<()>>,
    /// Handles mutating actors
    ///     Creation and Bootstrap
    ///     Drift
    ///     Initiate Queries
    ///     Deletion
    ///     Getting Snapshots
    ///     Pausing / Unpauseing
    peer_controller_handle: Option<JoinHandle<()>>,

    /// Channel to send churn commands to background thread
    peer_controller_tx: UnboundedSender<ChurnCommand>,

    /// Config for making new peers
    actor_config: Arc<RwLock<ProteanConfig>>,

    /// Embedding pool for this worker (indexed by local index)
    embedding_pool: Arc<RwLock<EmbeddingPool<S>>>,

    /// Total number of workers
    n_workers: usize,

    // Phase tracking for event matching, query tracking, churn tracking
    // etc. to know progress from monitor
    /// Use to track churn, and different phases
    bootstrapping_actors: DashMap<Uuid, ()>,

    /// Active Nodes
    active_actors: DashMap<Uuid, ()>,

    /// Shutting Down Nodes
    shuttingdown_actors: DashMap<Uuid, ()>,

    /// Shutdown Nodes:
    shutdown_actors: DashMap<Uuid, ()>,
}

// Snapshot request, create directory with timestamped at the time,
// send to each peer needed, pass said directory, each timestamp there
// own snv protos

impl<S: EmbeddingSpace> Worker<S> {
    pub fn new(worker_id: String, my_address: Address, n_workers: usize) -> Self {
        let runtime = Builder::new_multi_thread()
            .worker_threads(n_workers)
            .thread_name(format!("worker-{}", worker_id))
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");

        let (remote_msg_tx, remote_msg_rx) = unbounded_channel();
        let (event_tx, event_rx) = unbounded_channel();

        let workers = Arc::new(DashMap::new());
        let actor_protocol_channels = Arc::new(DashMap::new());
        let actor_control_channels = Arc::new(DashMap::new());

        let actor_config = Arc::new(RwLock::new(ProteanConfig::default()));
        let config = Arc::new(RwLock::new(None));

        let coordinator = Arc::new(RwLock::new(None));

        let broker_handle = Self::spawn_broker_thread(
            Arc::clone(&workers),
            remote_msg_rx,
            runtime.handle().clone(),
        );

        let pending_queries = Arc::new(DashMap::new());

        let event_processor_handle = Self::spawn_event_processor_thread(
            worker_id.clone(),
            Arc::clone(&coordinator),
            Arc::clone(&pending_queries),
            event_rx,
            runtime.handle().clone(),
        );

        let embedding_pool = Arc::new(RwLock::new(Vec::new()));
        let global_offset = Arc::new(RwLock::new(0));

        let runtime_arc = Arc::new(runtime);

        let churn_running = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let (churn_tx, churn_rx) = unbounded_channel();
        let peer_controller_handle = Self::spawn_peer_controller_thread(
            worker_id.clone(),
            Arc::clone(&actor_control_channels),
            Arc::clone(&actor_protocol_channels),
            Arc::new(RwLock::new(my_address.clone())),
            Arc::clone(&embedding_pool),
            Arc::clone(&global_offset),
            remote_msg_tx.clone(),
            event_tx.clone(),
            Arc::clone(&actor_config),
            Arc::clone(&runtime_arc),
            Arc::clone(&churn_running),
            churn_rx,
            runtime_arc.handle().clone(),
        );

        Self {
            worker_id,
            my_address: Arc::new(RwLock::new(my_address)),
            workers,
            coordinator,
            actor_protocol_channels,
            actor_control_channels,
            remote_msg_tx,
            event_tx,
            config,
            runtime: runtime_arc,
            actor_handles: Vec::new(),
            broker_handle: Some(broker_handle),
            event_processor_handle: Some(event_processor_handle),
            peer_controller_handle: Some(peer_controller_handle),
            churn_tx,
            churn_running,
            actor_config,
            embedding_pool,
            global_offset,
            n_workers,
            bootstrapping_actors: DashMap::new(),
            active_actors: DashMap::new(),
            shuttingdown_actors: DashMap::new(),
            shutdown_actors: DashMap::new(),
            pending_queries,
        }
    }

    /// Set coordinator connection for event forwarding
    pub async fn set_coordinator(&self, coordinator_address: String) -> Result<(), Box<dyn std::error::Error>> {
        println!("[{}] Attempting to connect to coordinator at {}", self.worker_id, coordinator_address);
        let addr = format!("http://{}", coordinator_address);
        let client = CoordinatorClient::connect(addr.clone()).await
            .map_err(|e| {
                println!("[{}] Failed to connect to coordinator at {}: {}", self.worker_id, addr, e);
                e
            })?
            .max_decoding_message_size(100 * 1024 * 1024) // 100MB
            .max_encoding_message_size(100 * 1024 * 1024); // 100MB
        *self.coordinator.write().unwrap() = Some(client);
        println!("[{}] Successfully connected to coordinator at {}", self.worker_id, coordinator_address);
        tracing::info!("Connected to coordinator at {}", coordinator_address);
        Ok(())
    }

    fn spawn_event_processor_thread(
        worker_id: String,
        coordinator: Arc<RwLock<Option<CoordinatorClient<Channel>>>>,
        pending_queries: Arc<DashMap<Uuid, oneshot::Sender<QueryResult<S>>>>,
        mut event_rx: UnboundedReceiver<ProteanEvent<S>>,
        runtime_handle: Handle,
    ) -> JoinHandle<()> {
        runtime_handle.spawn(async move {
            println!("[{}] Event processor thread started", worker_id);
            while let Some(event) = event_rx.recv().await {
                println!("[{}] Received event: {:?}", worker_id, event);
                // Extract peer UUID from event
                let peer_uuid = match &event {
                    ProteanEvent::QueryCompleted { local_uuid, .. } => *local_uuid,
                    ProteanEvent::BootstrapConvergingCompleted { local_uuid } => *local_uuid,
                    ProteanEvent::BootstrapCompleted { local_uuid } => *local_uuid,
                    ProteanEvent::StateChanged { .. } => {
                        // StateChanged doesn't have UUID - would need to be passed differently
                        // For now, use a zero UUID as placeholder
                        Uuid::from_bytes([0u8; 64])
                    }
                };

                // Handle QueryCompleted events - send results to pending queries
                if let ProteanEvent::QueryCompleted { result, .. } = &event {
                    // The query UUID is stored in the result
                    if let Some((_, sender)) = pending_queries.remove(&result.query_uuid) {
                        // Send the result to the waiting execute_query call
                        let _ = sender.send(result.clone());
                    }
                }

                // Forward event to coordinator if connected
                let client_option = {
                    let guard = coordinator.read().unwrap();
                    guard.clone()
                };

                if let Some(mut client) = client_option {
                    println!("[{}] Forwarding event to coordinator", worker_id);
                    // Convert event to proto
                    let event_proto = Self::event_to_proto(&event, &worker_id, &peer_uuid);

                    // Forward to coordinator via ForwardEvent RPC
                    match client.forward_event(Request::new(event_proto)).await {
                        Ok(_) => {
                            println!("[{}] Successfully forwarded event to coordinator", worker_id);
                            tracing::trace!("Forwarded event to coordinator");
                        }
                        Err(e) => {
                            println!("[{}] Failed to forward event to coordinator: {}", worker_id, e);
                            tracing::error!("Failed to forward event to coordinator: {}", e);
                        }
                    }
                } else {
                    println!("[{}] No coordinator connection, cannot forward event", worker_id);
                }
            }
        })
    }

    fn spawn_churn_thread(
        _worker_id: String,
        actor_control_channels: Arc<DashMap<Uuid, UnboundedSender<ControlMessage<S>>>>,
        actor_protocol_channels: Arc<DashMap<Uuid, UnboundedSender<ProteanMessage<S>>>>,
        my_address: Arc<RwLock<Address>>,
        embedding_pool: Arc<RwLock<Vec<S::EmbeddingData>>>,
        global_offset: Arc<RwLock<u64>>,
        remote_msg_tx: UnboundedSender<OutMessage<S>>,
        event_tx: UnboundedSender<ProteanEvent<S>>,
        actor_config: Arc<RwLock<ProteanConfig>>,
        runtime: Arc<Runtime>,
        churn_running: Arc<std::sync::atomic::AtomicBool>,
        mut churn_rx: UnboundedReceiver<ChurnCommand>,
        runtime_handle: Handle,
    ) -> JoinHandle<()> {
        runtime_handle.spawn(async move {
            // Helper closure to spawn a peer actor
            let spawn_peer_actor = |uuid: Uuid, embedding: S::EmbeddingData, original_embedding_index: u64, bootstrap_peer: Option<ProteanPeer<S>>| {
                let (msg_tx, msg_rx) = unbounded_channel();
                let (control_tx, control_rx) = unbounded_channel();

                let cloned_config = actor_config.read().unwrap().clone();
                let max_interval = cloned_config.snv_config.max_exploration_interval;

                let actor = ActorProtean::new(
                    my_address.clone(),
                    uuid,
                    embedding,
                    original_embedding_index,
                    cloned_config.clone(),
                    msg_rx,
                    control_rx,
                    actor_protocol_channels.clone(),
                    remote_msg_tx.clone(),
                    event_tx.clone(),
                    Duration::from_millis(5),
                    max_interval,
                );

                actor_protocol_channels.insert(uuid, msg_tx);
                actor_control_channels.insert(uuid, control_tx.clone());

                let handle = runtime.spawn(async move {
                    actor.run().await;
                });

                drop(handle);

                // Bootstrap the peer if bootstrap info provided
                if let Some(bootstrap_peer) = bootstrap_peer {
                    let bs_msg = ControlMessage::Bootstrap {
                        contact_point: bootstrap_peer,
                        config: Some(cloned_config.snv_config.exploration_config.clone()),
                    };

                    if let Err(e) = control_tx.send(bs_msg) {
                        tracing::error!("Churn thread: Failed to send bootstrap message to peer {}: {:?}", uuid, e);
                    }
                }
            };

            while let Some(command) = churn_rx.recv().await {
                match command {
                    ChurnCommand::Start(request) => {
                        // Set flag to indicate churn is running
                        churn_running.store(true, std::sync::atomic::Ordering::SeqCst);

                        let pattern = request.pattern();
                        let config = request.config.unwrap_or_default();

                        tracing::info!("Starting churn pattern: {:?}", pattern);

                        match pattern {
                            ChurnPatternType::FlashCrowd => {
                                // FLASH_CROWD: Spawn all peers immediately
                                tracing::info!("Executing FLASH_CROWD pattern for {} peers", config.global_indices.len());

                                // Pick random bootstrap from bootstrap_indices (if provided)
                                let bootstrap_peer = if !config.bootstrap_indices.is_empty() {
                                    use rand::seq::SliceRandom;
                                    let mut rng = rand::thread_rng();
                                    let bootstrap_index = *config.bootstrap_indices.choose(&mut rng).unwrap();

                                    // Get embedding from pool
                                    let global_offset_val = *global_offset.read().unwrap();
                                    if bootstrap_index < global_offset_val {
                                        tracing::error!("Bootstrap index {} below offset {}", bootstrap_index, global_offset_val);
                                        None
                                    } else {
                                        let pool = embedding_pool.read().unwrap();
                                        let local_index = (bootstrap_index - global_offset_val) as usize;
                                        if let Some(embedding) = pool.get(local_index).cloned() {
                                            let bootstrap_uuid = Self::global_index_to_uuid(bootstrap_index);
                                            let bootstrap_address = my_address.read().unwrap().clone();
                                            Some(ProteanPeer {
                                                embedding,
                                                peer: Peer {
                                                    uuid: bootstrap_uuid,
                                                    address: bootstrap_address,
                                                },
                                            })
                                        } else {
                                            tracing::error!("Bootstrap index {} out of range", bootstrap_index);
                                            None
                                        }
                                    }
                                } else {
                                    None
                                };

                                // Spawn all peers
                                // Acquire locks once and clone all embeddings to avoid lock contention
                                let global_offset_val = *global_offset.read().unwrap();
                                let pool = embedding_pool.read().unwrap();

                                // Collect (uuid, embedding, global_index) tuples while holding lock
                                let peers_to_spawn: Vec<_> = config.global_indices.iter()
                                    .filter_map(|&global_index| {
                                        if global_index < global_offset_val {
                                            tracing::error!("Global index {} below offset {}", global_index, global_offset_val);
                                            return None;
                                        }

                                        let local_index = (global_index - global_offset_val) as usize;
                                        pool.get(local_index).cloned().map(|embedding| {
                                            (Self::global_index_to_uuid(global_index), embedding, global_index)
                                        }).or_else(|| {
                                            tracing::error!("Global index {} out of range", global_index);
                                            None
                                        })
                                    })
                                    .collect();

                                drop(pool); // Release lock before spawning

                                // Spawn all peers without holding locks
                                for (uuid, embedding, global_index) in peers_to_spawn {
                                    spawn_peer_actor(uuid, embedding, global_index, bootstrap_peer.clone());
                                    tracing::info!("FLASH_CROWD: Spawned peer {}", uuid);
                                }

                                tracing::info!("FLASH_CROWD pattern completed");
                            }
                            ChurnPatternType::MassDeparture => {
                                // MASS_DEPARTURE: Delete all peers immediately
                                tracing::info!("Executing MASS_DEPARTURE pattern for {} peers", config.global_indices.len());

                                for global_index in config.global_indices {
                                    let uuid = Self::global_index_to_uuid(global_index);

                                    // Send shutdown command
                                    if let Some(control_tx) = actor_control_channels.get(&uuid) {
                                        let _ = control_tx.send(ControlMessage::Shutdown);
                                    }

                                    // Remove from channel maps
                                    actor_protocol_channels.remove(&uuid);
                                    actor_control_channels.remove(&uuid);

                                    tracing::info!("MASS_DEPARTURE: Deleted peer {}", uuid);
                                }

                                tracing::info!("MASS_DEPARTURE pattern completed");
                            }
                            ChurnPatternType::GradualJoin => {
                                // GRADUAL_JOIN: Spawn peers over time
                                tracing::info!("Executing GRADUAL_JOIN pattern for {} peers over {}ms",
                                    config.global_indices.len(), config.duration_ms);

                                if config.global_indices.is_empty() {
                                    tracing::warn!("GRADUAL_JOIN: No peers to spawn");
                                    continue;
                                }

                                // Calculate interval between spawns
                                let interval_ms = if config.duration_ms > 0 && config.global_indices.len() > 1 {
                                    config.duration_ms / (config.global_indices.len() as u64 - 1)
                                } else {
                                    0
                                };

                                // Pick random bootstrap from bootstrap_indices (if provided)
                                let bootstrap_peer = if !config.bootstrap_indices.is_empty() {
                                    use rand::seq::SliceRandom;
                                    let mut rng = rand::thread_rng();
                                    let bootstrap_index = *config.bootstrap_indices.choose(&mut rng).unwrap();

                                    let global_offset_val = *global_offset.read().unwrap();
                                    if bootstrap_index < global_offset_val {
                                        tracing::error!("Bootstrap index {} below offset {}", bootstrap_index, global_offset_val);
                                        None
                                    } else {
                                        let pool = embedding_pool.read().unwrap();
                                        let local_index = (bootstrap_index - global_offset_val) as usize;
                                        if let Some(embedding) = pool.get(local_index).cloned() {
                                            let bootstrap_uuid = Self::global_index_to_uuid(bootstrap_index);
                                            let bootstrap_address = my_address.read().unwrap().clone();
                                            Some(ProteanPeer {
                                                embedding,
                                                peer: Peer {
                                                    uuid: bootstrap_uuid,
                                                    address: bootstrap_address,
                                                },
                                            })
                                        } else {
                                            tracing::error!("Bootstrap index {} out of range", bootstrap_index);
                                            None
                                        }
                                    }
                                } else {
                                    None
                                };

                                // Spawn peers gradually
                                // Pre-fetch all embeddings to avoid repeated lock acquisition
                                let peers_to_spawn: Vec<_> = {
                                    let global_offset_val = *global_offset.read().unwrap();
                                    let pool = embedding_pool.read().unwrap();

                                    config.global_indices.iter()
                                        .filter_map(|&global_index| {
                                            if global_index < global_offset_val {
                                                tracing::error!("Global index {} below offset {}", global_index, global_offset_val);
                                                return None;
                                            }

                                            let local_index = (global_index - global_offset_val) as usize;
                                            pool.get(local_index).cloned().map(|embedding| {
                                                (Self::global_index_to_uuid(global_index), embedding, global_index)
                                            }).or_else(|| {
                                                tracing::error!("Global index {} out of range", global_index);
                                                None
                                            })
                                        })
                                        .collect()
                                    // Lock dropped at end of block
                                };

                                // Spawn peers with time delays
                                for (i, (uuid, embedding, global_index)) in peers_to_spawn.into_iter().enumerate() {
                                    if i > 0 && interval_ms > 0 {
                                        tokio::time::sleep(tokio::time::Duration::from_millis(interval_ms)).await;
                                    }

                                    spawn_peer_actor(uuid, embedding, global_index, bootstrap_peer.clone());
                                    tracing::info!("GRADUAL_JOIN: Spawned peer {} ({}/{})", uuid, i + 1, config.global_indices.len());
                                }

                                tracing::info!("GRADUAL_JOIN pattern completed");
                            }
                            ChurnPatternType::GradualLeave => {
                                // GRADUAL_LEAVE: Delete peers over time
                                tracing::info!("Executing GRADUAL_LEAVE pattern for {} peers over {}ms",
                                    config.global_indices.len(), config.duration_ms);

                                if config.global_indices.is_empty() {
                                    tracing::warn!("GRADUAL_LEAVE: No peers to delete");
                                    continue;
                                }

                                // Calculate interval between deletions
                                let interval_ms = if config.duration_ms > 0 && config.global_indices.len() > 1 {
                                    config.duration_ms / (config.global_indices.len() as u64 - 1)
                                } else {
                                    0
                                };

                                // Delete peers gradually
                                for (i, global_index) in config.global_indices.iter().enumerate() {
                                    if i > 0 && interval_ms > 0 {
                                        tokio::time::sleep(tokio::time::Duration::from_millis(interval_ms)).await;
                                    }

                                    let uuid = Self::global_index_to_uuid(*global_index);

                                    // Send shutdown command
                                    if let Some(control_tx) = actor_control_channels.get(&uuid) {
                                        let _ = control_tx.send(ControlMessage::Shutdown);
                                    }

                                    // Remove from channel maps
                                    actor_protocol_channels.remove(&uuid);
                                    actor_control_channels.remove(&uuid);

                                    tracing::info!("GRADUAL_LEAVE: Deleted peer {} ({}/{})", uuid, i + 1, config.global_indices.len());
                                }

                                tracing::info!("GRADUAL_LEAVE pattern completed");
                            }
                            ChurnPatternType::EmbeddingDrift => {
                                // EMBEDDING_DRIFT: Cause existing peers' embeddings to drift
                                tracing::info!(
                                    "Executing EMBEDDING_DRIFT pattern for {} peers with {} steps over {}ms",
                                    config.global_indices.len(),
                                    config.drift_steps,
                                    config.duration_ms
                                );

                                // Validate configuration
                                if config.drift_steps == 0 {
                                    tracing::error!("EMBEDDING_DRIFT requires drift_steps > 0");
                                    continue;
                                }

                                if config.drift_target_indices.len() != config.global_indices.len() {
                                    tracing::error!(
                                        "EMBEDDING_DRIFT requires equal number of drift targets ({}) and peers ({})",
                                        config.drift_target_indices.len(),
                                        config.global_indices.len()
                                    );
                                    continue;
                                }

                                // Calculate update interval
                                let update_interval = if config.drift_steps > 1 {
                                    std::time::Duration::from_millis(config.duration_ms / config.drift_steps as u64)
                                } else {
                                    std::time::Duration::from_millis(config.duration_ms)
                                };

                                // Fetch all original and target embeddings in one lock acquisition
                                let drift_data: Vec<_> = {
                                    let global_offset_val = *global_offset.read().unwrap();
                                    let pool = embedding_pool.read().unwrap();

                                    config.global_indices.iter()
                                        .zip(config.drift_target_indices.iter())
                                        .filter_map(|(&peer_index, &target_index)| {
                                            let peer_uuid = Self::global_index_to_uuid(peer_index);

                                            // Validate peer exists in actor_control_channels
                                            if !actor_control_channels.contains_key(&peer_uuid) {
                                                tracing::error!("Peer {} not found for drift", peer_uuid);
                                                return None;
                                            }

                                            // Fetch original embedding (from peer's original index)
                                            if peer_index < global_offset_val {
                                                tracing::error!("Peer index {} below offset {}", peer_index, global_offset_val);
                                                return None;
                                            }
                                            let original_local_index = (peer_index - global_offset_val) as usize;
                                            let original_embedding = pool.get(original_local_index).cloned()?;

                                            // Fetch target embedding
                                            if target_index < global_offset_val {
                                                tracing::error!("Target index {} below offset {}", target_index, global_offset_val);
                                                return None;
                                            }
                                            let target_local_index = (target_index - global_offset_val) as usize;
                                            let target_embedding = pool.get(target_local_index).cloned()
                                                .or_else(|| {
                                                    tracing::error!("Target index {} out of range", target_index);
                                                    None
                                                })?;

                                            Some((peer_uuid, original_embedding, target_embedding))
                                        })
                                        .collect()
                                }; // Lock dropped here

                                // Send StartDrift messages to all peers
                                for (peer_uuid, original_embedding, target_embedding) in drift_data {
                                    if let Some(control_tx) = actor_control_channels.get(&peer_uuid) {
                                        let drift_msg = ControlMessage::StartDrift {
                                            original_embedding,
                                            target_embedding,
                                            update_interval,
                                            total_steps: config.drift_steps,
                                        };

                                        if let Err(e) = control_tx.send(drift_msg) {
                                            tracing::error!("Failed to send drift message to peer {}: {:?}", peer_uuid, e);
                                        } else {
                                            tracing::debug!("Initiated drift for peer {}", peer_uuid);
                                        }
                                    }
                                }

                                tracing::info!("EMBEDDING_DRIFT pattern initiated for {} peers", config.global_indices.len());
                            }
                            ChurnPatternType::Unspecified => {
                                tracing::error!("Unspecified churn pattern");
                            }
                        }

                        // Clear flag to indicate churn is no longer running
                        churn_running.store(false, std::sync::atomic::Ordering::SeqCst);
                    }
                    ChurnCommand::Stop => {
                        tracing::info!("Stopping churn");
                        churn_running.store(false, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            }
        })
    }

    

    fn spawn_broker_thread(
        workers: Arc<DashMap<Address, WorkerNodeClient<Channel>>>,
        mut remote_msg_rx: UnboundedReceiver<OutMessage<S>>,
        runtime_handle: Handle,
    ) -> JoinHandle<()> {
        runtime_handle.spawn(async move {
            tracing::info!("[Broker] Broker thread started");
            while let Some(out_msg) = remote_msg_rx.recv().await {
                let dest_address = out_msg.destination.address.clone();
                tracing::info!("[Broker] Routing message to {} at {}", out_msg.destination.uuid, dest_address);

                let mut client = if let Some(entry) = workers.get(&dest_address) {
                    entry.value().clone()
                } else {
                    // Try to make connection with worker we don't know
                    match Self::create_client(&dest_address).await {
                        Ok(new_client) => {
                            workers.insert(dest_address.clone(), new_client.clone());
                            tracing::info!("Connected to worker at {}", dest_address);
                            new_client
                        }
                        Err(e) => {
                            tracing::error!("Failed to connect to worker {}: {}", dest_address, e);
                            continue;
                        }
                    }
                };

                // Convert Uuid to bytes for proto
                let uuid_bytes = out_msg.destination.uuid.as_bytes().to_vec();

                // Convert ProteanMessage to ProteanMessageProto
                let message_proto = ProteanMessageProto::from(out_msg.message.clone());

                // Log message being sent
                tracing::debug!(
                    "[Broker] SENDING message to peer {} at worker {}, type: {:?}",
                    out_msg.destination.uuid,
                    dest_address,
                    Self::message_type_name(&out_msg.message)
                );

                let route_msg_req = tonic::Request::new(RouteMessageRequest {
                    destination_uuid: uuid_bytes,
                    message: Some(message_proto),
                });

                // Send message (TODO: Add retry logic)
                match client.route_message(route_msg_req).await {
                    Ok(response) => {
                        let ack = response.into_inner();
                        if ack.success {
                            tracing::trace!("Message delivered to {}", dest_address);
                        } else {
                            tracing::warn!("Bad ack from {}: {}", dest_address, ack.message);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to route message to {}: {}", dest_address, e);
                    }
                }
            }
        })
    }

    // ============ UUID/Index Conversion Helpers ============

    /// Convert global dataset index to UUID
    /// UUIDs are derived deterministically from global indices
    pub(crate) fn global_index_to_uuid(global_index: u64) -> Uuid {
        let mut bytes = [0u8; 64];
        bytes[0..8].copy_from_slice(&global_index.to_be_bytes());
        Uuid::from_bytes(bytes)
    }

    /// Convert UUID back to global dataset index
    pub(crate) fn uuid_to_global_index(uuid: &Uuid) -> u64 {
        let bytes = uuid.as_bytes();
        u64::from_be_bytes(bytes[0..8].try_into().unwrap())
    }

    /// Get message type name for logging
    fn message_type_name(message: &ProteanMessage<S>) -> &'static str {
        match message {
            ProteanMessage::KnnRequest { .. } => "KnnRequest",
            ProteanMessage::KnnResponse { .. } => "KnnResponse",
            ProteanMessage::KfnRequest { .. } => "KfnRequest",
            ProteanMessage::KfnResponse { .. } => "KfnResponse",
            ProteanMessage::PingRequest { .. } => "PingRequest",
            ProteanMessage::PingResponse { .. } => "PingResponse",
        }
    }

    /// Get embedding from pool by global index
    /// Uses round-robin distribution: worker i owns indices [i, i+N, i+2N, ...]
    pub(crate) fn get_embedding(&self, global_index: u64) -> Result<S::EmbeddingData, Status> {
        let global_offset = *self.global_offset.read().unwrap(); // This is the worker index
        let pool = self.embedding_pool.read().unwrap();

        // Check if embedding pool is empty
        if pool.is_empty() {
            return Err(Status::internal(
                format!("Embedding pool is empty for worker {}. Embeddings must be distributed before creating peers.",
                    self.worker_id)
            ));
        }

        let n_workers = self.n_workers as u64;

        // Check if this global index belongs to this worker (round-robin)
        let owner_worker_idx = global_index % n_workers;
        if owner_worker_idx != global_offset {
            return Err(Status::invalid_argument(
                format!("Global index {} belongs to worker {} (this is worker {})",
                    global_index, owner_worker_idx, global_offset)
            ));
        }

        // Calculate local index for round-robin distribution
        let local_index = (global_index / n_workers) as usize;

        // Validate local index is within pool bounds
        if local_index >= pool.len() {
            return Err(Status::invalid_argument(
                format!("Global index {} maps to local index {} which exceeds pool size {}",
                    global_index, local_index, pool.len())
            ));
        }

        Ok(pool[local_index].clone())
    }

    /// Get any embedding from the full dataset by global index
    /// This is used for bootstrap peer lookups where we need embeddings from other workers
    /// All workers have the full dataset loaded (each worker loads all embeddings, but only owns a subset)
    pub(crate) fn get_any_embedding(&self, global_index: u64) -> Result<S::EmbeddingData, Status> {
        let pool = self.embedding_pool.read().unwrap();

        // Check if embedding pool is empty
        if pool.is_empty() {
            return Err(Status::internal(
                format!("Embedding pool is empty for worker {}. Embeddings must be distributed before creating peers.",
                    self.worker_id)
            ));
        }

        // Since all workers have the full dataset, we can directly index
        let index = global_index as usize;
        if index >= pool.len() {
            return Err(Status::invalid_argument(
                format!("Global index {} out of bounds (pool size: {})", global_index, pool.len())
            ));
        }

        Ok(pool[index].clone())
    }

    async fn create_client(address: &Address) -> Result<WorkerNodeClient<Channel>, Box<dyn std::error::Error>> {
        let endpoint = Channel::from_shared(format!("http://{}", address))?
            .connect()
            .await?;

        Ok(WorkerNodeClient::new(endpoint))
    }

    async fn add_worker_connection(&self, address: Address) -> bool {
        match Self::create_client(&address).await {
            Ok(client) => {
                self.workers.insert(address.clone(), client);
                tracing::info!("Connected to worker at {}", address);
                true
            }
            Err(e) => {
                tracing::error!("Failed to connect to worker {}: {}", address, e);
                false
            }
        }
    }

    // Helper methods

    // Proto conversion helpers

    pub(crate) fn parse_embedding(tensor_proto: Option<TensorProto>) -> Result<S::EmbeddingData, Status> {
        let tensor = tensor_proto.ok_or_else(|| Status::invalid_argument("Missing embedding tensor"))?;
        S::EmbeddingData::try_from(tensor)
            .map_err(|e: SerializationError| Status::invalid_argument(format!("Invalid embedding: {}", e)))
    }

    fn parse_query_config(proto: Option<QueryConfigProto>) -> QueryConfig {
        proto.map(|p| QueryConfig::from(p)).unwrap_or_default()
    }

    fn config_proto_to_protean_config(proto: &SimConfigProto) -> Result<ProteanConfig, Status> {
        let snv_config_proto = proto.snv_config.clone()
            .ok_or_else(|| Status::invalid_argument("Missing snv_config"))?;

        let snv_config = SnvConfig::from(snv_config_proto);

        Ok(ProteanConfig {
            snv_config,
            timeout: Duration::from_secs(30),
            max_concurrent_queries: 100,
        })
    }

    pub fn parse_uuid(bytes: &[u8]) -> Result<Uuid, Status> {
        if bytes.len() != 64 {
            return Err(Status::invalid_argument(format!(
                "Invalid UUID length: expected 64 bytes, got {}",
                bytes.len()
            )));
        }
        let uuid_bytes: [u8; 64] = bytes.try_into()
            .map_err(|_| Status::invalid_argument("Failed to convert UUID bytes"))?;
        Ok(Uuid::from_bytes(uuid_bytes))
    }

    async fn route_local_message(&self, dest_uuid: Uuid, message: ProteanMessage<S>) -> Result<(), Status> {
        if let Some(tx) = self.actor_protocol_channels.get(&dest_uuid) {
            tracing::debug!(
                "[Worker {} Router] DELIVERING message to actor {}, type: {:?}",
                self.worker_id,
                dest_uuid,
                Self::message_type_name(&message)
            );
            tx.send(message)
                .map_err(|_| Status::internal("Failed to send message to peer"))?;
            tracing::debug!("[Worker {} Router] Message successfully delivered to actor's channel", self.worker_id);
            Ok(())
        } else {
            tracing::error!("[Worker {} Router] ERROR: Peer {} NOT FOUND on this worker", self.worker_id, dest_uuid);
            Err(Status::not_found(format!("Peer {} not found on this worker", dest_uuid)))
        }
    }

    pub(crate) fn local_peer_uuids(&self) -> Vec<Uuid> {
        self.actor_protocol_channels.iter().map(|entry| *entry.key()).collect()
    }

    pub(crate) async fn delete_peer(&self, uuid: &Uuid) -> bool {
        // Send shutdown command
        if let Some(control_tx) = self.actor_control_channels.get(uuid) {
            let _ = control_tx.send(ControlMessage::Shutdown);
        }

        // Remove from channel maps
        self.actor_protocol_channels.remove(uuid);
        self.actor_control_channels.remove(uuid);

        // Remove from tracking maps
        self.bootstrapping_actors.remove(uuid);
        self.active_actors.remove(uuid);
        self.shuttingdown_actors.remove(uuid);
        self.shutdown_actors.remove(uuid);

        true
    }

    pub fn spawn_peer(&self, uuid: Uuid, local_embedding: S::EmbeddingData, original_embedding_index: u64, bootstrap_peer: Option<ProteanPeer<S>>) {
        let (msg_tx, msg_rx) = unbounded_channel();
        let (control_tx, control_rx) = unbounded_channel();

        let cloned_config = self.actor_config.read().unwrap().clone();
        let max_interval = cloned_config.snv_config.max_exploration_interval;

        let actor = ActorProtean::new(
            self.my_address.clone(),
            uuid,
            local_embedding,
            original_embedding_index,
            cloned_config.clone(),
            msg_rx,
            control_rx,
            self.actor_protocol_channels.clone(),
            self.remote_msg_tx.clone(),
            self.event_tx.clone(),
            Duration::from_millis(5),
            max_interval,
        );

        self.actor_protocol_channels.insert(uuid, msg_tx);
        self.actor_control_channels.insert(uuid, control_tx.clone());

        let handle = self.runtime.spawn(async move {
            actor.run().await;
        });

        // Note: We can't push to actor_handles with &self, consider using Arc<Mutex<Vec<...>>>
        // For now, handles will be managed by the runtime
        drop(handle);

        // Bootstrap the peer if bootstrap info provided
        if let Some(bootstrap_peer) = bootstrap_peer {
            println!("[Worker {}] Sending Bootstrap message to peer {}", self.worker_id, uuid);
            println!("  Contact UUID: {}", bootstrap_peer.peer.uuid);
            println!("  Contact Address: {}", bootstrap_peer.peer.address);
            let bs_msg = ControlMessage::Bootstrap {
                contact_point: bootstrap_peer.clone(),
                config: Some(cloned_config.snv_config.exploration_config.clone()),
            };

            if let Err(e) = control_tx.send(bs_msg) {
                println!("[Worker {}] Failed to send bootstrap message to peer {}: {:?}", self.worker_id, uuid, e);
                tracing::error!("Failed to send bootstrap message to peer {}: {:?}", uuid, e);
            } else {
                println!("[Worker {}] Successfully sent Bootstrap message to peer {}", self.worker_id, uuid);
            }
        } else {
            println!("[Worker {}] No bootstrap peer for {}, skipping bootstrap", self.worker_id, uuid);
        }
    }

    /// Gracefully shutdown all actors/peers
    /// This should be called before dropping the worker to ensure clean shutdown
    pub async fn shutdown(&self) {
        println!("[Worker {}] Shutting down all actors...", self.worker_id);

        // Get all peer UUIDs
        let peer_uuids = self.local_peer_uuids();
        println!("[Worker {}] Deleting {} actors", self.worker_id, peer_uuids.len());

        // Delete all peers
        for uuid in peer_uuids {
            self.delete_peer(&uuid).await;
        }

        // Send stop command to churn thread
        let _ = self.churn_tx.send(ChurnCommand::Stop);

        println!("[Worker {}] All actors deleted, shutdown complete", self.worker_id);

        // Note: Background threads (broker, event_processor, churn) will automatically
        // stop when their channels close, which happens when Worker is dropped
    }

}

// Implement Drop to ensure clean shutdown when Worker is dropped
impl<S: EmbeddingSpace> Drop for Worker<S> {
    fn drop(&mut self) {
        println!("[Worker {}] Drop called - cleaning up background threads", self.worker_id);

        // Send stop command to churn thread
        let _ = self.churn_tx.send(ChurnCommand::Stop);

        // Channels will be dropped automatically which will cause background threads to exit
        // We don't have access to async here, so we can't wait for threads to finish,
        // but dropping the senders will cause the receivers to return None and threads will exit

        println!("[Worker {}] Drop complete", self.worker_id);
    }
}

impl<S: EmbeddingSpace> Worker<S> {
    // Public testable methods that RPC handlers will wrap

    /// Handle ping request - returns success
    pub fn handle_ping(&self) -> Ack {
        Ack {
            success: true,
            message: "pong".to_string(),
        }
    }

    /// Handle worker registration
    pub async fn handle_register_worker(&self, address: String) -> Ack {
        if self.add_worker_connection(address.clone()).await {
            Ack {
                success: true,
                message: format!("Registered worker at {}", address),
            }
        } else {
            Ack {
                success: false,
                message: format!("Failed to connect to worker at {}", address),
            }
        }
    }

    /// Handle message routing to local peer
    pub async fn handle_route_message(
        &self,
        destination_uuid: &[u8],
        message_proto: ProteanMessageProto,
    ) -> Result<Ack, Status> {
        // Parse UUID
        let uuid = Self::parse_uuid(destination_uuid)?;

        // Convert ProteanMessageProto to ProteanMessage
        let message: ProteanMessage<S> = ProteanMessage::try_from(message_proto)
            .map_err(|e| Status::invalid_argument(format!("Failed to convert message: {}", e)))?;

        // Log message being received
        tracing::debug!(
            "[Worker {} gRPC] RECEIVED message for peer {}, type: {:?}",
            self.worker_id,
            uuid,
            Self::message_type_name(&message)
        );

        // Route to local peer
        self.route_local_message(uuid, message).await?;

        Ok(Ack {
            success: true,
            message: "Message routed".to_string(),
        })
    }

    /// Handle configuration update
    pub fn handle_set_config(&self, config_proto: SimConfigProto) -> Result<Ack, Status> {
        // Store config
        *self.config.write().unwrap() = Some(config_proto.clone());

        // Update actor_config from proto config
        let protean_config = Self::config_proto_to_protean_config(&config_proto)?;
        *self.actor_config.write().unwrap() = protean_config;

        Ok(Ack {
            success: true,
            message: "Configuration set successfully".to_string(),
        })
    }

    /// Handle loading embeddings into the pool
    pub fn handle_load_embeddings(
        &self,
        global_offset: u64,
        embeddings: Vec<TensorProto>,
    ) -> Result<Ack, Status> {
        // Set global offset
        *self.global_offset.write().unwrap() = global_offset;

        // Clear existing pool and parse new embeddings
        let mut pool = self.embedding_pool.write().unwrap();
        pool.clear();

        for tensor_proto in embeddings {
            let embedding = Self::parse_embedding(Some(tensor_proto))?;
            pool.push(embedding);
        }

        let count = pool.len();
        drop(pool); // Release lock

        tracing::info!(
            "Loaded {} embeddings starting at global index {}",
            count,
            global_offset
        );

        Ok(Ack {
            success: true,
            message: format!(
                "Loaded {} embeddings starting at global index {}",
                count, global_offset
            ),
        })
    }

    /// Handle peer creation
    pub fn handle_create_peers(
        &self,
        global_indices: Vec<u64>,
        bootstrap_peer_info: Option<BootstrapPeerInfo>,
    ) -> Result<Ack, Status> {
        let mut created_count = 0;
        for global_index in global_indices {
            // Convert global index to UUID
            let uuid = Self::global_index_to_uuid(global_index);

            // Get embedding from pool
            let embedding = self.get_embedding(global_index)?;

            // Determine bootstrap peer for this specific peer
            // Don't bootstrap if this peer IS the bootstrap peer (avoid self-bootstrap)
            let bootstrap_peer = if let Some(ref info) = bootstrap_peer_info {
                if global_index != info.global_index {
                    // Parse bootstrap UUID from bytes
                    let bootstrap_uuid = Self::parse_uuid(&info.uuid)?;

                    // Get bootstrap embedding - either from proto or look up locally by global_index
                    let bootstrap_embedding = if let Some(ref embedding_proto) = info.embedding {
                        // Old path: embedding provided in message (deprecated)
                        Self::parse_embedding(Some(embedding_proto.clone()))?
                    } else {
                        // New path: look up embedding locally by global_index (much faster!)
                        self.get_any_embedding(info.global_index)?
                    };

                    // Use the worker address from the info
                    let bootstrap_address = info.worker_address.clone();

                    Some(ProteanPeer {
                        embedding: bootstrap_embedding,
                        peer: Peer {
                            uuid: bootstrap_uuid,
                            address: bootstrap_address,
                        },
                    })
                } else {
                    // This is the bootstrap seed peer - no bootstrap needed
                    tracing::debug!("Peer {} is the bootstrap seed, no bootstrap needed", global_index);
                    None
                }
            } else {
                // No bootstrap info provided
                None
            };

            // Spawn peer with original embedding index
            self.spawn_peer(uuid, embedding, global_index, bootstrap_peer);
            created_count += 1;
        }

        Ok(Ack {
            success: true,
            message: format!("Created {} peers", created_count),
        })
    }

    /// Handle bootstrap request for an existing peer
    pub fn handle_bootstrap_peer(
        &self,
        peer_index: u64,
        bootstrap_peer_info: BootstrapPeerInfo,
    ) -> Result<Ack, Status> {
        // Convert global index to UUID
        let uuid = Self::global_index_to_uuid(peer_index);

        // Parse bootstrap UUID from bytes
        let bootstrap_uuid = Self::parse_uuid(&bootstrap_peer_info.uuid)?;

        // Get bootstrap embedding - either from proto or look up locally by global_index
        let bootstrap_embedding = if let Some(embedding_proto) = bootstrap_peer_info.embedding.clone() {
            // Old path: embedding provided in message (deprecated)
            Self::parse_embedding(Some(embedding_proto))?
        } else {
            // New path: look up embedding locally by global_index (much faster!)
            self.get_any_embedding(bootstrap_peer_info.global_index)?
        };

        // Use the worker address from the info
        let bootstrap_address = bootstrap_peer_info.worker_address.clone();

        let bootstrap_peer = ProteanPeer {
            embedding: bootstrap_embedding,
            peer: Peer {
                uuid: bootstrap_uuid,
                address: bootstrap_address.clone(),
            },
        };

        // Get the control channel for this peer
        if let Some(control_tx) = self.actor_control_channels.get(&uuid) {
            println!("[Worker {}] Sending Bootstrap message to peer {}
  Contact UUID: {}
  Contact Address: {}",
                self.worker_id,
                uuid,
                bootstrap_uuid,
                &bootstrap_address
            );

            // Send bootstrap command
            control_tx.send(ControlMessage::Bootstrap {
                contact_point: bootstrap_peer,
                config: None,
            }).map_err(|_| Status::internal("Failed to send bootstrap message to peer"))?;

            println!("[Worker {}] Successfully sent Bootstrap message to peer {}",
                self.worker_id, uuid);

            Ok(Ack {
                success: true,
                message: format!("Bootstrapped peer {}", peer_index),
            })
        } else {
            Err(Status::not_found(format!("Peer {} not found on this worker", peer_index)))
        }
    }

    /// Handle peer deletion
    pub async fn handle_delete_peers(&self, global_indices: Vec<u64>) -> Result<Ack, Status> {
        let mut deleted_count = 0;
        for global_index in global_indices {
            let uuid = Self::global_index_to_uuid(global_index);
            if self.delete_peer(&uuid).await {
                deleted_count += 1;
            }
        }

        Ok(Ack {
            success: true,
            message: format!("Deleted {} peers", deleted_count),
        })
    }

    /// Handle churn pattern execution
    pub fn handle_churn(&self, request: ChurnPatternRequest) -> Result<Ack, Status> {
        // Check if churn is already running
        if self.churn_running.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(Ack {
                success: false,
                message: "Churn pattern already running. Please wait for it to complete.".to_string(),
            });
        }

        // Send command to churn thread
        self.churn_tx
            .send(ChurnCommand::Start(request))
            .map_err(|_| Status::internal("Churn thread not available"))?;

        Ok(Ack {
            success: true,
            message: "Churn pattern started".to_string(),
        })
    }

    /// Handle query execution on a specific peer
    pub async fn handle_execute_query(
        &self,
        source_peer_uuid: &[u8],
        query_embedding: Option<TensorProto>,
        k: u32,
        config: Option<QueryConfigProto>,
    ) -> Result<QueryResponse, Status> {
        // Parse peer UUID
        let peer_uuid = Self::parse_uuid(source_peer_uuid)?;

        // Get control channel for this peer
        let control_tx = self.actor_control_channels.get(&peer_uuid)
            .ok_or_else(|| Status::not_found(format!("Peer {} not found", peer_uuid)))?
            .clone();

        // Parse query embedding and config
        let query_embedding = Self::parse_embedding(query_embedding)?;
        let k = k as usize;
        let config = Self::parse_query_config(config);

        // Create oneshot channel for query UUID
        let (uuid_tx, uuid_rx) = oneshot::channel();

        // Send query command to peer
        let query_msg = ControlMessage::Query {
            embedding: query_embedding,
            k,
            config,
            response: uuid_tx,
        };

        control_tx.send(query_msg)
            .map_err(|_| Status::internal("Failed to send query"))?;

        // Await query UUID from peer
        let query_uuid = uuid_rx.await
            .map_err(|_| Status::internal("Failed to receive query UUID"))?
            .ok_or_else(|| Status::internal("Peer rejected query"))?;

        // Create oneshot channel for query result and register it
        let (result_tx, result_rx) = oneshot::channel();
        self.pending_queries.insert(query_uuid, result_tx);

        // Wait for query to complete (with timeout)
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            result_rx
        ).await
            .map_err(|_| Status::deadline_exceeded("Query timed out"))?
            .map_err(|_| Status::internal("Failed to receive query result"))?;

        // Convert result to QueryResponse proto
        let results: Vec<QueryCandidateProto> = result.candidates
            .iter()
            .map(|candidate| {
                // Convert embedding to TensorProto
                let embedding_proto: TensorProto = candidate.protean_peer.embedding.clone().into();

                QueryCandidateProto {
                    peer: Some(PeerProto {
                        embedding: Some(embedding_proto),
                        uuid: candidate.protean_peer.peer.uuid.as_bytes().to_vec(),
                        address: candidate.protean_peer.peer.address.clone(),
                    }),
                    distance: Some(candidate.distance.into()),
                }
            })
            .collect();

        Ok(QueryResponse {
            query_uuid: query_uuid.as_bytes().to_vec(),
            results,
            hops: 0, // Not tracked in QueryResult
            latency_ms: 0, // Not tracked in QueryResult (could be calculated from start time if needed)
            success: true,
            error_message: String::new(),
        })
    }

    /// Handle brute-force k-NN query across all peers
    pub async fn handle_true_query(
        &self,
        query_embedding: Option<TensorProto>,
        k: u32,
    ) -> Result<QueryResponse, Status> {
        // Parse query embedding
        let query_embedding = Self::parse_embedding(query_embedding)?;
        let k = k as usize;

        // Get all peer UUIDs
        let peer_uuids = self.local_peer_uuids();

        // Collect embeddings from all peers and compute distances
        let mut distances: Vec<(Uuid, S::DistanceValue)> = Vec::new();

        for peer_uuid in peer_uuids {
            if let Some(control_tx) = self.actor_control_channels.get(&peer_uuid) {
                let (response_tx, response_rx) = oneshot::channel();
                let msg = ControlMessage::GetEmbedding { response: response_tx };

                if control_tx.send(msg).is_ok() {
                    // Wait for embedding with timeout
                    match tokio::time::timeout(Duration::from_secs(5), response_rx).await {
                        Ok(Ok(embedding)) => {
                            let distance = S::distance(&query_embedding, &embedding);
                            distances.push((peer_uuid, distance));
                        }
                        _ => {
                            tracing::warn!("Failed to get embedding from peer {}", peer_uuid);
                        }
                    }
                }
            }
        }

        // Sort by distance and take top k
        distances.sort_by(|a, b| a.1.cmp(&b.1));
        let top_k_uuids: Vec<Uuid> = distances.into_iter().take(k).map(|(uuid, _)| uuid).collect();

        // Convert to QueryCandidateProto format (simplified - just UUIDs for now)
        let results: Vec<QueryCandidateProto> = top_k_uuids.iter().map(|uuid| {
            QueryCandidateProto {
                peer: Some(PeerProto {
                    uuid: uuid.as_bytes().to_vec(),
                    embedding: None, // Could include embedding if needed
                    address: "".to_string(), // Address not available in this context
                }),
                distance: None, // Could calculate distance if needed
            }
        }).collect();

        // Generate a unique query UUID from the query embedding
        let query_uuid = Uuid::from_embedding::<S>(&query_embedding);

        Ok(QueryResponse {
            query_uuid: query_uuid.as_bytes().to_vec(),
            results,
            hops: 0,
            latency_ms: 0,
            success: true,
            error_message: String::new(),
        })
    }

    /// Handle getting network snapshot
    pub async fn handle_get_snapshot(
        &self,
        peer_uuids: Vec<Vec<u8>>,
    ) -> Result<NetworkSnapshot, Status> {
        // Get peer UUIDs - either from request or all local peers
        let peer_uuids = if peer_uuids.is_empty() {
            self.local_peer_uuids()
        } else {
            peer_uuids.iter()
                .map(|bytes| Self::parse_uuid(bytes))
                .collect::<Result<Vec<_>, _>>()?
        };

        // Send requests to all peers in parallel (don't wait for responses yet)
        let mut response_channels = Vec::new();
        for peer_uuid in &peer_uuids {
            if let Some(control_tx) = self.actor_control_channels.get(peer_uuid) {
                let (response_tx, response_rx) = oneshot::channel();

                // Send GetSnvProto command to get full SNV proto
                let msg = ControlMessage::GetSnvProto {
                    response: response_tx,
                };

                if control_tx.send(msg).is_ok() {
                    response_channels.push((*peer_uuid, response_rx));
                } else {
                    tracing::warn!("Failed to send GetSnvProto to peer {}", peer_uuid);
                }
            }
        }

        // Now collect all responses with timeout
        let mut snapshots = Vec::new();
        for (peer_uuid, response_rx) in response_channels {
            match tokio::time::timeout(Duration::from_secs(5), response_rx).await {
                Ok(Ok(snv_proto)) => {
                    snapshots.push(snv_proto);
                    tracing::debug!("Got full SNV proto from peer {}", peer_uuid);
                }
                Ok(Err(_)) => {
                    tracing::warn!("Peer {} failed to send SNV proto", peer_uuid);
                }
                Err(_) => {
                    tracing::warn!("Timeout waiting for SNV proto from peer {}", peer_uuid);
                }
            }
        }

        // Create network snapshot with full SNV protos
        let snapshot = NetworkSnapshot {
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            worker_id: self.worker_id.clone(),
            peer_snapshots: snapshots,
        };

        Ok(snapshot)
    }

    /// Handle loading network snapshot
    pub async fn handle_load_snapshot(
        &self,
        snapshot: NetworkSnapshot,
    ) -> Result<Ack, Status> {
        // Clear all existing peers
        let existing_uuids = self.local_peer_uuids();
        for uuid in existing_uuids {
            self.delete_peer(&uuid).await;
        }

        // Restore peers from snapshot
        let mut restored_count = 0;
        for snv_proto in snapshot.peer_snapshots {
            // Extract peer UUID and embedding from proto
            let uuid = Self::parse_uuid(&snv_proto.local_uuid)?;
            let embedding = Self::parse_embedding(snv_proto.local_embedding.clone())?;

            // Spawn peer without bootstrap
            // Note: Using 0 as original_embedding_index since we're loading from snapshot
            self.spawn_peer(uuid, embedding, 0, None);

            // Send LoadSnvSnapshot command to restore SNV state
            if let Some(control_tx) = self.actor_control_channels.get(&uuid) {
                let (response_tx, response_rx) = oneshot::channel();
                let msg = ControlMessage::LoadSnvSnapshot {
                    snapshot: snv_proto,
                    response: response_tx,
                };

                if control_tx.send(msg).is_ok() {
                    // Wait for confirmation
                    match tokio::time::timeout(Duration::from_secs(5), response_rx).await {
                        Ok(Ok(success)) if success => {
                            restored_count += 1;
                            tracing::debug!("Restored peer {}", uuid);
                        }
                        _ => {
                            tracing::warn!("Failed to restore peer {}", uuid);
                        }
                    }
                }
            }
        }

        Ok(Ack {
            success: true,
            message: format!("Restored {} peers from snapshot", restored_count),
        })
    }
}

// WorkerNode gRPC trait implementation

#[tonic::async_trait]
impl<S: EmbeddingSpace + Send + Sync + 'static> WorkerNode for Worker<S>
where
    S::EmbeddingData: Send + Sync,
{
    // 1. Ping - Health check
    async fn ping(
        &self,
        _request: Request<PingRequest>,
    ) -> Result<Response<Ack>, Status> {
        Ok(Response::new(self.handle_ping()))
    }

    // 2. RegisterWorker - Add connection to another worker
    async fn register_worker(
        &self,
        request: Request<WorkerInfo>,
    ) -> Result<Response<Ack>, Status> {
        let worker_info = request.into_inner();
        let ack = self.handle_register_worker(worker_info.address).await;
        Ok(Response::new(ack))
    }

    // 3. RouteMessage - Route message to local peer
    async fn route_message(
        &self,
        request: Request<RouteMessageRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let message_proto = req.message
            .ok_or_else(|| Status::invalid_argument("Missing message field"))?;
        let ack = self.handle_route_message(&req.destination_uuid, message_proto).await?;
        Ok(Response::new(ack))
    }

    // 4. SetConfig - Store configuration
    async fn set_config(
        &self,
        request: Request<SimConfigProto>,
    ) -> Result<Response<Ack>, Status> {
        let config_proto = request.into_inner();
        let ack = self.handle_set_config(config_proto)?;
        Ok(Response::new(ack))
    }

    // LoadEmbeddings - Load embedding pool from coordinator
    async fn load_embeddings(
        &self,
        request: Request<LoadEmbeddingsRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let ack = self.handle_load_embeddings(req.global_offset, req.embeddings)?;
        Ok(Response::new(ack))
    }

    // 5. CreatePeers - Spawn multiple peers using global indices
    async fn create_peers(
        &self,
        request: Request<CreatePeersRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        println!("[{} gRPC] Received create_peers request: indices={:?}, has_bootstrap_peer={}",
            self.worker_id, req.global_indices, req.bootstrap_peer.is_some());
        let ack = self.handle_create_peers(req.global_indices, req.bootstrap_peer)?;
        println!("[{} gRPC] create_peers completed: {}", self.worker_id, ack.message);
        Ok(Response::new(ack))
    }

    // BootstrapPeer - Bootstrap an existing peer
    async fn bootstrap_peer(
        &self,
        request: Request<BootstrapPeerRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        println!("[Worker {} gRPC] Received bootstrap_peer RPC: peer_index={}, has_bootstrap_peer={}",
                 self.worker_id, req.peer_index, req.bootstrap_peer.is_some());
        let bootstrap_peer = req.bootstrap_peer
            .ok_or_else(|| Status::invalid_argument("Missing bootstrap_peer field"))?;
        let ack = self.handle_bootstrap_peer(req.peer_index, bootstrap_peer)?;
        println!("[Worker {} gRPC] bootstrap_peer RPC completed successfully", self.worker_id);
        Ok(Response::new(ack))
    }

    // 6. DeletePeers - Remove peers using global indices
    async fn delete_peers(
        &self,
        request: Request<DeletePeersRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let ack = self.handle_delete_peers(req.global_indices).await?;
        Ok(Response::new(ack))
    }

    // 7. ExecuteQuery - Start query on a peer
    async fn execute_query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let req = request.into_inner();
        let response = self.handle_execute_query(
            &req.source_peer_uuid,
            req.query_embedding,
            req.k,
            req.config,
        ).await?;
        Ok(Response::new(response))
    }

    // 8. TrueQuery - Brute-force k-NN
    async fn true_query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let req = request.into_inner();
        let response = self.handle_true_query(req.query_embedding, req.k).await?;
        Ok(Response::new(response))
    }

    // 9. GetSnapshot - Collect SNV snapshots
    async fn get_snapshot(
        &self,
        request: Request<SnapshotRequest>,
    ) -> Result<Response<NetworkSnapshot>, Status> {
        let req = request.into_inner();
        let snapshot = self.handle_get_snapshot(req.peer_uuids).await?;
        Ok(Response::new(snapshot))
    }

    // 10. LoadSnapshot - Restore network state
    async fn load_snapshot(
        &self,
        request: Request<NetworkSnapshot>,
    ) -> Result<Response<Ack>, Status> {
        let snapshot = request.into_inner();
        let ack = self.handle_load_snapshot(snapshot).await?;
        Ok(Response::new(ack))
    }

    // 11. Churn - Apply churn pattern
    async fn churn(
        &self,
        request: Request<ChurnPatternRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let ack = self.handle_churn(req)?;
        Ok(Response::new(ack))
    }
}