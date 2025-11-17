//! Worker implementation for distributed Protean simulation
//!
//! ## Architecture
//! - **Broker Thread**: Routes protocol messages to other workers via gRPC
//! - **Event Processor Thread**: Processes local events and forwards to coordinator via ForwardEvent RPC
//! - **Actor Management**: Spawns and manages ActorProtean instances for each peer
//!
//! ## WorkerNode gRPC Service - Implementation Status
//!
//! ### ✅ Fully Implemented
//! 1. **Ping** - Health check (returns "pong")
//! 2. **RegisterWorker** - Registers connection to another worker
//! 3. **RouteMessage** - Routes protocol messages to local peers
//! 4. **SetConfig** - Stores simulation configuration
//! 5. **DeletePeers** - Removes peers and cleans up all state
//! 6. **GetSnapshot** - Collects SNV snapshots from peers
//! 7. **Churn** - Returns "not implemented" stub
//!
//! ### 🔧 Partially Implemented (needs proto parsing)
//! 8. **CreatePeers** - Needs embedding tensor parsing
//! 9. **ExecuteQuery** - Needs embedding/config parsing
//! 10. **TrueQuery** - Needs embedding parsing and ControlMessage::GetEmbedding
//! 11. **LoadSnapshot** - Needs snapshot proto parsing and ControlMessage::LoadSnvSnapshot
//!
//! ## TODOs
//! - Implement embedding tensor proto conversion (TensorProto <-> S::EmbeddingData)
//! - Implement config proto conversion (SimConfigProto -> ProteanConfig)
//! - Add ControlMessage::GetEmbedding variant for TrueQuery
//! - Add ControlMessage::LoadSnvSnapshot variant for snapshot restoration
//! - Implement event_to_proto() conversion for event forwarding
//! - Handle actor_handles tracking with interior mutability

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
    ChurnPatternRequest, ChurnPatternType,
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
};

/// Commands sent to the churn background thread
enum ChurnCommand {
    Start(ChurnPatternRequest),
    Stop,
}

pub struct Worker<S: EmbeddingSpace> {
    worker_id: String,
    my_address: Arc<RwLock<Address>>,

    /// gRPC connections to other workers (we keep connections to all other workers in sim (scale is only up to 1k))
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

    /// Handles for all running threads
    actor_handles: Vec<JoinHandle<()>>,
    broker_handle: Option<JoinHandle<()>>,
    event_processor_handle: Option<JoinHandle<()>>,
    churn_handle: Option<JoinHandle<()>>,

    /// Channel to send churn commands to background thread
    churn_tx: UnboundedSender<ChurnCommand>,

    /// Atomic flag to track if churn is currently running
    churn_running: Arc<std::sync::atomic::AtomicBool>,

    /// Config for making new peers
    actor_config: Arc<RwLock<ProteanConfig>>,

    /// The max number of actors this worker is allowed to host
    max_actors: usize,

    /// Embedding pool for this worker (indexed by local index)
    pub(crate) embedding_pool: Arc<RwLock<Vec<S::EmbeddingData>>>,

    /// Global offset for this worker's embeddings (e.g., 100 if this worker owns embeddings 100-9999)
    pub(crate) global_offset: Arc<RwLock<u64>>,

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
    pub fn new(worker_id: String, my_address: Address, n_workers: usize, max_actors: usize) -> Self {
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

        let event_processor_handle = Self::spawn_event_processor_thread(
            worker_id.clone(),
            Arc::clone(&coordinator),
            event_rx,
            runtime.handle().clone(),
        );

        let embedding_pool = Arc::new(RwLock::new(Vec::new()));
        let global_offset = Arc::new(RwLock::new(0));

        let runtime_arc = Arc::new(runtime);

        let churn_running = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let (churn_tx, churn_rx) = unbounded_channel();
        let churn_handle = Self::spawn_churn_thread(
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
            churn_handle: Some(churn_handle),
            churn_tx,
            churn_running,
            actor_config,
            max_actors,
            embedding_pool,
            global_offset,
            bootstrapping_actors: DashMap::new(),
            active_actors: DashMap::new(),
            shuttingdown_actors: DashMap::new(),
            shutdown_actors: DashMap::new(),
        }
    }

    fn spawn_event_processor_thread(
        worker_id: String,
        coordinator: Arc<RwLock<Option<CoordinatorClient<Channel>>>>,
        mut event_rx: UnboundedReceiver<ProteanEvent<S>>,
        runtime_handle: Handle,
    ) -> JoinHandle<()> {
        runtime_handle.spawn(async move {
            while let Some(event) = event_rx.recv().await {
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

                // TODO: Process event locally (update tracking maps for bootstrap/active/shutdown)
                // Could update bootstrapping_actors, active_actors, etc. based on event type

                // Forward event to coordinator if connected
                let client_option = {
                    let guard = coordinator.read().unwrap();
                    guard.clone()
                };

                if let Some(mut client) = client_option {
                    // Convert event to proto
                    let event_proto = Self::event_to_proto(&event, &worker_id, &peer_uuid);

                    // Forward to coordinator via ForwardEvent RPC
                    match client.forward_event(Request::new(event_proto)).await {
                        Ok(_) => {
                            tracing::trace!("Forwarded event to coordinator");
                        }
                        Err(e) => {
                            tracing::error!("Failed to forward event to coordinator: {}", e);
                        }
                    }
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

    fn event_to_proto(event: &ProteanEvent<S>, worker_id: &str, peer_uuid: &Uuid) -> ProteanEventProto {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        match event {
            ProteanEvent::StateChanged { from_state, to_state } => {
                ProteanEventProto {
                    event_type: ProteanEventType::StateChanged as i32,
                    worker_id: worker_id.to_string(),
                    timestamp_ms,
                    event: Some(protean_event_proto::Event::StateChanged(StateChangedEvent {
                        peer_uuid: peer_uuid.as_bytes().to_vec(),
                        from_state: format!("{:?}", from_state),
                        to_state: format!("{:?}", to_state),
                    })),
                }
            },
            ProteanEvent::QueryCompleted { local_uuid, result } => {
                ProteanEventProto {
                    event_type: ProteanEventType::QueryCompleted as i32,
                    worker_id: worker_id.to_string(),
                    timestamp_ms,
                    event: Some(protean_event_proto::Event::QueryCompleted(QueryCompletedEvent {
                        peer_uuid: local_uuid.as_bytes().to_vec(),
                        query_uuid: result.query_uuid.as_bytes().to_vec(),
                        candidates: result.candidates.iter().map(|c| QueryCandidateProto::from(c.clone())).collect(),
                        hops: 0, // TODO: Track hops in QueryResult
                        latency_ms: 0, // TODO: Track latency in QueryResult
                    })),
                }
            },
            ProteanEvent::BootstrapConvergingCompleted { local_uuid } => {
                ProteanEventProto {
                    event_type: ProteanEventType::BootstrapConvergingCompleted as i32,
                    worker_id: worker_id.to_string(),
                    timestamp_ms,
                    event: Some(protean_event_proto::Event::BootstrapConvergingCompleted(
                        BootstrapConvergingCompletedEvent {
                            peer_uuid: local_uuid.as_bytes().to_vec(),
                        }
                    )),
                }
            },
            ProteanEvent::BootstrapCompleted { local_uuid } => {
                ProteanEventProto {
                    event_type: ProteanEventType::BootstrapCompleted as i32,
                    worker_id: worker_id.to_string(),
                    timestamp_ms,
                    event: Some(protean_event_proto::Event::BootstrapCompleted(
                        BootstrapCompletedEvent {
                            peer_uuid: local_uuid.as_bytes().to_vec(),
                        }
                    )),
                }
            },
        }
    }

    fn spawn_broker_thread(
        workers: Arc<DashMap<Address, WorkerNodeClient<Channel>>>,
        mut remote_msg_rx: UnboundedReceiver<OutMessage<S>>,
        runtime_handle: Handle,
    ) -> JoinHandle<()> {
        runtime_handle.spawn(async move {
            while let Some(out_msg) = remote_msg_rx.recv().await {
                let dest_address = out_msg.destination.address.clone();

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

    /// Get embedding from pool by global index
    pub(crate) fn get_embedding(&self, global_index: u64) -> Result<S::EmbeddingData, Status> {
        let global_offset = *self.global_offset.read().unwrap();
        let pool = self.embedding_pool.read().unwrap();

        // Validate lower bound
        if global_index < global_offset {
            return Err(Status::invalid_argument(
                format!("Global index {} is below this worker's offset {}",
                    global_index, global_offset)
            ));
        }

        // Validate upper bound
        let max_index = global_offset + pool.len() as u64;
        if global_index >= max_index {
            return Err(Status::invalid_argument(
                format!("Global index {} out of range (pool range: {}-{})",
                    global_index, global_offset, max_index - 1)
            ));
        }

        let local_index = (global_index - global_offset) as usize;
        Ok(pool[local_index].clone())
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

    fn parse_uuid(bytes: &[u8]) -> Result<Uuid, Status> {
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
            tx.send(message)
                .map_err(|_| Status::internal("Failed to send message to peer"))?;
            Ok(())
        } else {
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
            let bs_msg = ControlMessage::Bootstrap {
                contact_point: bootstrap_peer,
                config: Some(cloned_config.snv_config.exploration_config.clone()),
            };

            if let Err(e) = control_tx.send(bs_msg) {
                tracing::error!("Failed to send bootstrap message to peer {}: {:?}", uuid, e);
            }
        }
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
        bootstrap_index: u64,
    ) -> Result<Ack, Status> {
        // Get bootstrap peer if index provided (0 = no bootstrap)
        let bootstrap_peer = if bootstrap_index != 0 {
            let bootstrap_uuid = Self::global_index_to_uuid(bootstrap_index);
            let bootstrap_embedding = self.get_embedding(bootstrap_index)?;
            let bootstrap_address = self.my_address.read().unwrap().clone();

            Some(ProteanPeer {
                embedding: bootstrap_embedding,
                peer: Peer {
                    uuid: bootstrap_uuid,
                    address: bootstrap_address,
                },
            })
        } else {
            None
        };

        let mut created_count = 0;
        for global_index in global_indices {
            // Convert global index to UUID
            let uuid = Self::global_index_to_uuid(global_index);

            // Get embedding from pool
            let embedding = self.get_embedding(global_index)?;

            // Spawn peer with original embedding index
            self.spawn_peer(uuid, embedding, global_index, bootstrap_peer.clone());
            created_count += 1;
        }

        Ok(Ack {
            success: true,
            message: format!("Created {} peers", created_count),
        })
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
    pub fn handle_churn(&self, pattern: ChurnPatternType, config: ChurnConfig) -> Result<Ack, Status> {
        let request = ChurnPatternRequest {
            pattern: pattern.into(),
            config: Some(config),
        };

        self.churn_tx
            .send(ChurnCommand::Start(request))
            .map_err(|_| Status::internal("Failed to send churn command"))?;

        Ok(Ack {
            success: true,
            message: format!("Churn pattern {:?} initiated", pattern),
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
        let address = worker_info.address;

        if self.add_worker_connection(address.clone()).await {
            Ok(Response::new(Ack {
                success: true,
                message: format!("Registered worker at {}", address),
            }))
        } else {
            Ok(Response::new(Ack {
                success: false,
                message: format!("Failed to connect to worker at {}", address),
            }))
        }
    }

    // 3. RouteMessage - Route message to local peer
    async fn route_message(
        &self,
        request: Request<RouteMessageRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();

        // Parse UUID
        let uuid = Self::parse_uuid(&req.destination_uuid)?;

        // Convert ProteanMessageProto to ProteanMessage
        let message_proto = req.message
            .ok_or_else(|| Status::invalid_argument("Missing message field"))?;
        let message: ProteanMessage<S> = ProteanMessage::try_from(message_proto)
            .map_err(|e| Status::invalid_argument(format!("Failed to convert message: {}", e)))?;

        // Route to local peer
        self.route_local_message(uuid, message).await?;

        Ok(Response::new(Ack {
            success: true,
            message: "Message routed".to_string(),
        }))
    }

    // 4. SetConfig - Store configuration
    async fn set_config(
        &self,
        request: Request<SimConfigProto>,
    ) -> Result<Response<Ack>, Status> {
        let config_proto = request.into_inner();

        // Store config
        *self.config.write().unwrap() = Some(config_proto.clone());

        // Update actor_config from proto config
        let protean_config = Self::config_proto_to_protean_config(&config_proto)?;
        *self.actor_config.write().unwrap() = protean_config;

        Ok(Response::new(Ack {
            success: true,
            message: "Configuration set successfully".to_string(),
        }))
    }

    // LoadEmbeddings - Load embedding pool from coordinator
    async fn load_embeddings(
        &self,
        request: Request<LoadEmbeddingsRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();

        // Set global offset
        *self.global_offset.write().unwrap() = req.global_offset;

        // Clear existing pool and parse new embeddings
        let mut pool = self.embedding_pool.write().unwrap();
        pool.clear();

        for tensor_proto in req.embeddings {
            let embedding = Self::parse_embedding(Some(tensor_proto))?;
            pool.push(embedding);
        }

        let count = pool.len();
        drop(pool); // Release lock

        tracing::info!(
            "Loaded {} embeddings starting at global index {}",
            count,
            req.global_offset
        );

        Ok(Response::new(Ack {
            success: true,
            message: format!(
                "Loaded {} embeddings starting at global index {}",
                count, req.global_offset
            ),
        }))
    }

    // 5. CreatePeers - Spawn multiple peers using global indices
    async fn create_peers(
        &self,
        request: Request<CreatePeersRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();

        // Get bootstrap peer if index provided (0 = no bootstrap)
        let bootstrap_peer = if req.bootstrap_index != 0 {
            let bootstrap_uuid = Self::global_index_to_uuid(req.bootstrap_index);
            let bootstrap_embedding = self.get_embedding(req.bootstrap_index)?;
            let bootstrap_address = self.my_address.read().unwrap().clone();

            Some(ProteanPeer {
                embedding: bootstrap_embedding,
                peer: Peer {
                    uuid: bootstrap_uuid,
                    address: bootstrap_address,
                },
            })
        } else {
            None
        };

        let mut created_count = 0;
        for global_index in req.global_indices {
            // Convert global index to UUID
            let uuid = Self::global_index_to_uuid(global_index);

            // Get embedding from pool
            let embedding = self.get_embedding(global_index)?;

            // Spawn peer with original embedding index
            self.spawn_peer(uuid, embedding, global_index, bootstrap_peer.clone());
            created_count += 1;
        }

        Ok(Response::new(Ack {
            success: true,
            message: format!("Created {} peers", created_count),
        }))
    }

    // 6. DeletePeers - Remove peers using global indices
    async fn delete_peers(
        &self,
        request: Request<DeletePeersRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();

        let mut deleted_count = 0;
        for global_index in req.global_indices {
            // Convert global index to UUID
            let uuid = Self::global_index_to_uuid(global_index);

            // Delete peer
            if self.delete_peer(&uuid).await {
                deleted_count += 1;
            }
        }

        Ok(Response::new(Ack {
            success: true,
            message: format!("Deleted {} peers", deleted_count),
        }))
    }

    // 7. ExecuteQuery - Start query on a peer
    async fn execute_query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let req = request.into_inner();

        // Parse peer UUID
        let peer_uuid = Self::parse_uuid(&req.source_peer_uuid)?;

        // Get control channel for this peer
        let control_tx = self.actor_control_channels.get(&peer_uuid)
            .ok_or_else(|| Status::not_found(format!("Peer {} not found", peer_uuid)))?
            .clone();

        // Parse query embedding and config
        let query_embedding = Self::parse_embedding(req.query_embedding)?;
        let k = req.k as usize;
        let config = Self::parse_query_config(req.config);

        // Create oneshot channel for response
        let (response_tx, response_rx) = oneshot::channel();

        // Send query command to peer
        let query_msg = ControlMessage::Query {
            embedding: query_embedding,
            k,
            config,
            response: response_tx,
        };

        control_tx.send(query_msg)
            .map_err(|_| Status::internal("Failed to send query"))?;

        // Await query UUID from peer
        let query_uuid = response_rx.await
            .map_err(|_| Status::internal("Failed to receive query UUID"))?
            .ok_or_else(|| Status::internal("Peer rejected query"))?;

        Ok(Response::new(QueryResponse {
            query_uuid: query_uuid.as_bytes().to_vec(),
            results: vec![],
            hops: 0,
            latency_ms: 0,
            success: true,
            error_message: String::new(),
        }))
    }

    // 8. TrueQuery - Brute-force k-NN
    async fn true_query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let req = request.into_inner();

        // Parse query embedding
        let query_embedding = Self::parse_embedding(req.query_embedding)?;
        let k = req.k as usize;

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

        Ok(Response::new(QueryResponse {
            query_uuid: query_uuid.as_bytes().to_vec(),
            results,
            hops: 0,
            latency_ms: 0,
            success: true,
            error_message: String::new(),
        }))
    }

    // 9. GetSnapshot - Collect SNV snapshots
    async fn get_snapshot(
        &self,
        request: Request<SnapshotRequest>,
    ) -> Result<Response<NetworkSnapshot>, Status> {
        let req = request.into_inner();

        // Get peer UUIDs - either from request or all local peers
        let peer_uuids = if req.peer_uuids.is_empty() {
            self.local_peer_uuids()
        } else {
            req.peer_uuids.iter()
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

        Ok(Response::new(snapshot))
    }

    // 10. LoadSnapshot - Restore network state
    async fn load_snapshot(
        &self,
        request: Request<NetworkSnapshot>,
    ) -> Result<Response<Ack>, Status> {
        let snapshot = request.into_inner();

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

        Ok(Response::new(Ack {
            success: true,
            message: format!("Restored {} peers from snapshot", restored_count),
        }))
    }

    // 11. Churn - Apply churn pattern
    // 11. Churn - Apply churn pattern
    async fn churn(
        &self,
        request: Request<ChurnPatternRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();

        // Check if churn is already running
        if self.churn_running.load(std::sync::atomic::Ordering::SeqCst) {
            return Ok(Response::new(Ack {
                success: false,
                message: "Churn pattern already running. Please wait for it to complete.".to_string(),
            }));
        }

        // Validate pattern (EMBEDDING_DRIFT not supported)
        if req.pattern() == ChurnPatternType::EmbeddingDrift {
            return Ok(Response::new(Ack {
                success: false,
                message: "EMBEDDING_DRIFT pattern not yet implemented".to_string(),
            }));
        }

        // Send command to churn thread
        self.churn_tx.send(ChurnCommand::Start(req))
            .map_err(|_| Status::internal("Churn thread not available"))?;

        Ok(Response::new(Ack {
            success: true,
            message: "Churn pattern started".to_string(),
        }))
    }
}