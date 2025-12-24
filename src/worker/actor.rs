use std::path::PathBuf;
use std::sync::{Arc};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::mpsc::{UnboundedSender, UnboundedReceiver};
use tokio::sync::oneshot;
use tokio::sync::RwLock;
use tokio::time::{interval, MissedTickBehavior};

use protean::{
    Protean,
    ProteanConfig,
    ProteanMessage,
    ProteanError,
    OutMessage,
    ProteanPeer,
    QueryConfig,
    QueryStatus,
    SnvSnapshot,
    ExplorationConfig,
    embedding_space::{EmbeddingSpace, Embedding},
    protean::ProteanEvent,
    address::Address,
    uuid::Uuid,
    proto::QueryCandidateProto,
};

use tonic::{transport::Channel, Request};

use protean::proto::{ProteanMessageProto, SparseNeighborViewProto};

use crate::proto::dist_sim::{
    coordinator_node_client::CoordinatorNodeClient,
    worker_node_client::WorkerNodeClient,
    RouteMessageRequest,
    protean_event_proto, ProteanEventType, StateChangedEvent, QueryCompletedEvent,
    BootstrapCompletedEvent, BootstrapConvergingCompletedEvent,
    ProteanEventProto,
};

pub struct ActorSnapshot {
    pub stats: SnvSnapshot,
    pub proto: SparseNeighborViewProto,
}

/// State for embedding drift functionality
pub(crate) struct DriftState<S: EmbeddingSpace> {
    /// The original embedding (from the pool at original_embedding_index)
    pub original_embedding: S::EmbeddingData,
    /// The target embedding to drift towards
    pub target_embedding: S::EmbeddingData,
    /// Deadline for the next drift update
    pub next_update_deadline: Instant,
    /// Interval between drift updates in sec
    pub update_interval: Duration,
    /// Current drift step (0 = at original, total_steps = at target)
    pub current_step: u32,
    /// Total number of drift steps
    pub total_steps: u32,
}

pub(crate) enum ControlMessage<S>
where
    S: EmbeddingSpace,
{
    /// Bootstrap this peer using another peer as entry point
    Bootstrap {
        contact_point: ProteanPeer<S>,
        config: Option<ExplorationConfig>,
    },
    /// Execute a k-NN query
    Query {
        embedding: S::EmbeddingData,
        k: usize,
        config: QueryConfig,
        /// Channel to send back the query UUID
        response: oneshot::Sender<Option<Uuid>>,
    },
    /// Starts a peer shutting down if it is not already, peer will emit an event when it is shutdown
    Shutdown,
    /// Saves protean to path and returns full path
    Save {
        save_path: PathBuf,
        response: oneshot::Sender<String>,
    },
    /// Get query status
    GetQueryStatus {
        query_uuid: Uuid,
        reply: oneshot::Sender<Option<QueryStatus>>,
    },
    /// Get the SNV snapshot (stats only)
    GetSnvSnapshot {
        response: oneshot::Sender<ActorSnapshot>,
    },
    /// Get the local embedding
    GetEmbedding {
        response: oneshot::Sender<S::EmbeddingData>,
    },
    /// Start embedding drift towards a target
    StartDrift {
        target_embedding: S::EmbeddingData,
        update_interval: Duration,
        total_steps: u32,
    },
}

pub(crate) struct ActorProtean<S: EmbeddingSpace> {
    /// The Protean peer instance (owned by this actor)
    protean: Protean<S>,

    /// Incoming protocol message channel from other peers
    msg_rx: UnboundedReceiver<ProteanMessage<S>>,

    /// Incoming control messages from controller thread
    control_rx: UnboundedReceiver<ControlMessage<S>>,

    /// Shared map of all LOCAL peer channels for message routing
    peer_channels: Arc<DashMap<Uuid, UnboundedSender<ProteanMessage<S>>>>,

    /// gRPC connection to the coordinator
    /// Forward Events here
    coordinator: CoordinatorNodeClient<Channel>,

    /// gRPC connections to other worker to send messages intra worker
    workers: Arc<DashMap<Address, WorkerNodeClient<Channel>>>,

    /// Minimum step interval (when active)
    min_step_interval: Duration,

    /// Maximum step interval (when idle)
    max_step_interval: Option<Duration>,

    /// Current drift state (None if not drifting)
    drift_state: Option<DriftState<S>>,
}

impl<S: EmbeddingSpace> ActorProtean<S>
where
    S::EmbeddingData: Embedding<Scalar = f32>,
{
    pub(crate) fn new(
        local_address: Arc<RwLock<Address>>,
        uuid: Uuid,
        local_embedding: S::EmbeddingData,
        config: ProteanConfig,
        msg_rx: UnboundedReceiver<ProteanMessage<S>>,
        control_rx: UnboundedReceiver<ControlMessage<S>>,
        peer_channels: Arc<DashMap<Uuid, UnboundedSender<ProteanMessage<S>>>>,
        coordinator: CoordinatorNodeClient<Channel>,
        workers: Arc<DashMap<Address, WorkerNodeClient<Channel>>>,
        min_step_interval: Duration,
        max_step_interval: Option<Duration>,
    ) -> Self {
        // Use blocking_read for sync context (tokio RwLock)
        let address = local_address.blocking_read().clone();
        let protean = Protean::new(
            address,
            uuid,
            local_embedding,
            config
        );

        Self {
            protean,
            msg_rx,
            control_rx,
            peer_channels,
            coordinator,
            workers,
            min_step_interval,
            max_step_interval,
            drift_state: None,
        }
    }

    pub(crate) fn from_proto(
        local_address: Arc<RwLock<Address>>,
        proto: SparseNeighborViewProto,
        config: ProteanConfig,
        msg_rx: UnboundedReceiver<ProteanMessage<S>>,
        control_rx: UnboundedReceiver<ControlMessage<S>>,
        peer_channels: Arc<DashMap<Uuid, UnboundedSender<ProteanMessage<S>>>>,
        coordinator: CoordinatorNodeClient<Channel>,
        workers: Arc<DashMap<Address, WorkerNodeClient<Channel>>>,
        min_step_interval: Duration,
        max_step_interval: Option<Duration>,
    ) -> Result<Self, ProteanError> {
        // Use blocking_read for sync context (tokio RwLock)
        let address = local_address.blocking_read().clone();
        let protean = Protean::from_proto(address, proto, config)?;

        Ok(Self {
            protean,
            msg_rx,
            control_rx,
            peer_channels,
            coordinator,
            workers,
            min_step_interval,
            max_step_interval,
            drift_state: None,
        })
    }


    /// Main actor event loop
    ///
    /// Runs indefinitely until channels are closed, processing:
    /// - Incoming protocol messages
    /// - Control commands (bootstrap/query/etc)
    /// - Periodic auto-steps with adaptive backoff
    pub(crate) async fn run(mut self) {
        let mut current_interval = self.min_step_interval;
        let mut step_interval = interval(current_interval);
        step_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                Some(msg) = self.control_rx.recv() => {
                    self.process_control(msg).await;
                },

                Some(msg) = self.msg_rx.recv() => {
                    self.process_message(msg).await;

                    // Reset Timer on Work done at this node
                    current_interval = self.min_step_interval;
                    step_interval = interval(current_interval);
                    step_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
                },

                _ = step_interval.tick() => {
                    let was_productive = self.auto_step().await;
                    if was_productive {
                        // Reset Timer on Work done at this node
                        current_interval = self.min_step_interval;
                        step_interval = interval(current_interval);
                        step_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
                    } else {
                        // Exonential increase on no-op on peer
                        current_interval *= 2;
                        if let Some(max_step_interval) = self.max_step_interval {
                            current_interval = current_interval.min(max_step_interval);
                        }
                        step_interval = interval(current_interval);
                        step_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
                    }
                },

                else => break,
            }
        }

    }


    fn event_to_proto(event: &ProteanEvent<S>, peer_uuid: &Uuid) -> ProteanEventProto {
        match event {
            ProteanEvent::StateChanged { from_state, to_state } => {
                ProteanEventProto {
                    event_type: ProteanEventType::StateChanged as i32,
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
                    event: Some(protean_event_proto::Event::QueryCompleted(QueryCompletedEvent {
                        peer_uuid: local_uuid.as_bytes().to_vec(),
                        query_uuid: result.query_uuid.as_bytes().to_vec(),
                        candidates: result.candidates.iter().map(|c| QueryCandidateProto::from(c.clone())).collect(),
                        num_seen: result.query_stats.num_seen as u32,
                        num_requests: result.query_stats.num_requests as u32,
                        num_received: result.query_stats.num_received as u32,
                    })),
                }
            },
            ProteanEvent::BootstrapConvergingCompleted { local_uuid } => {
                ProteanEventProto {
                    event_type: ProteanEventType::BootstrapConvergingCompleted as i32,
                    event: Some(protean_event_proto::Event::BootstrapConvergingCompleted(BootstrapConvergingCompletedEvent {
                        peer_uuid: local_uuid.as_bytes().to_vec(),
                    })),
                }
            },
            ProteanEvent::BootstrapCompleted { local_uuid } => {
                ProteanEventProto {
                    event_type: ProteanEventType::BootstrapCompleted as i32,
                    event: Some(protean_event_proto::Event::BootstrapCompleted(
                        BootstrapCompletedEvent {
                            peer_uuid: local_uuid.as_bytes().to_vec(),
                        }
                    )),
                }
            },
        }
    }

    async fn process_message(&mut self, msg: ProteanMessage<S>) {
        tracing::trace!("[Actor {}] Received protocol message: {:?}", self.protean.uuid(), msg);
        self.update_drift_if_needed();

        let step = self.protean.event(msg);
        tracing::trace!("[Actor {}] After processing message: {} messages, {} events",
            self.protean.uuid(), step.messages().len(), step.events().len());
        
        self.route_messages(step.messages()).await;
        self.forward_events(&step.events()).await;
    }

    async fn process_control(&mut self, msg: ControlMessage<S>) {
        // Check drift before processing control messages that trigger steps
        match &msg {
            ControlMessage::Bootstrap { .. } | ControlMessage::Query { .. } => {
                self.update_drift_if_needed();
            },
            _ => {}
        }

        match msg {
            ControlMessage::Bootstrap { contact_point, config } => {
                tracing::debug!("[Actor {}] Received Bootstrap command, contact: {}", self.protean.uuid(), contact_point.peer.uuid);
                let success = self.protean.bootstrap(contact_point, config);
                tracing::info!("[Actor {}] Bootstrap result: {}", self.protean.uuid(), success);
                if success {
                    // Bootstrap started, step to get initial messages
                    let step = self.protean.step();
                    tracing::trace!("[Actor {}] Bootstrap step: {} messages, {} events", self.protean.uuid(), step.messages().len(), step.events().len());
                    self.route_messages(step.messages()).await;
                    self.forward_events(step.events()).await;
                }
            },
            ControlMessage::Query { embedding, k, config, response } => {
                let query_uuid = self.protean.query(embedding, k, config);
                let _ = response.send(query_uuid);

                let step = self.protean.step();
                self.route_messages(step.messages()).await;
                self.forward_events(step.events()).await;
            },
            ControlMessage::Shutdown => {
                if self.protean.is_connected() {
                    self.protean.start_shutdown();
                }
            },
            ControlMessage::Save { save_path, response } => {
                match self.protean.save_snv(&save_path) {
                    Ok(_) => {
                        let _ = response.send(save_path.to_string_lossy().to_string());
                    },
                    Err(e) => {
                        let _ = response.send(format!("Error saving: {:?}", e));
                    }
                }
            },
            ControlMessage::GetQueryStatus { query_uuid, reply } => {
                let status = self.protean.query_status(&query_uuid);
                let _ = reply.send(status);
            },
            ControlMessage::GetSnvSnapshot { response } => {
                let stats = self.protean.snv_snapshot();
                let proto = self.protean.to_proto();
                let ss = ActorSnapshot {
                    stats,
                    proto
                };
                let _ = response.send(ss);
            },
            ControlMessage::GetEmbedding { response } => {
                let embedding = self.protean.embedding().clone();
                let _ = response.send(embedding);
            },
            ControlMessage::StartDrift { target_embedding, update_interval, total_steps } => {
                let original_embedding = self.protean.embedding().clone();
                // Initialize drift state
                self.drift_state = Some(DriftState {
                    original_embedding,
                    target_embedding,
                    next_update_deadline: Instant::now() + update_interval,
                    update_interval,
                    current_step: 0,
                    total_steps,
                });
                tracing::debug!(
                    "Started drift for peer {} with {} steps over {:?}",
                    self.protean.uuid(),
                    total_steps,
                    update_interval * total_steps
                );
            },
        }
    }

    async fn auto_step(&mut self) -> bool {
        // Check if we need to update embedding due to drift
        self.update_drift_if_needed();

        let step = self.protean.step();

        let has_messages = !step.messages().is_empty();
        let has_events = !step.events().is_empty();

        if has_messages || has_events {
            tracing::debug!("[Actor {}] Auto-step: {} messages, {} events",
                self.protean.uuid(), step.messages().len(), step.events().len());
        }

        self.route_messages(step.messages()).await;
        self.forward_events(step.events()).await;

        has_messages || has_events
    }

    /// Check drift state and update embedding if deadline has passed
    fn update_drift_if_needed(&mut self) {
        if let Some(drift_state) = &mut self.drift_state {
            let now = Instant::now();

            if now >= drift_state.next_update_deadline {
                // Time to update embedding (saturating add prevents overflow)
                drift_state.current_step = drift_state.current_step.saturating_add(1);

                if drift_state.current_step >= drift_state.total_steps {
                    // Final step - set to exact target and clear drift state
                    let target = drift_state.target_embedding.clone();
                    self.protean.set_embedding(target);
                    self.drift_state = None;
                    tracing::debug!("Drift complete for peer {}", self.protean.uuid());
                } else {
                    // Compute progress ratio
                    let t = drift_state.current_step as f32 / drift_state.total_steps as f32;

                    // LERP to compute: original + (target - original) * t
                    // This implements the offset model: current = original + offset
                    // Where offset = (target - original) * t
                    let new_embedding = Self::lerp_embedding(
                        &drift_state.original_embedding,
                        &drift_state.target_embedding,
                        t
                    );

                    self.protean.set_embedding(new_embedding);

                    // Schedule next update
                    drift_state.next_update_deadline = now + drift_state.update_interval;

                    tracing::trace!(
                        "Drift update for peer {}: step {}/{}",
                        self.protean.uuid(),
                        drift_state.current_step,
                        drift_state.total_steps
                    );
                }
            }
        }
    }

    /// Linear interpolation between two embeddings
    /// Returns: a + (b - a) * t
    fn lerp_embedding(a: &S::EmbeddingData, b: &S::EmbeddingData, t: f32) -> S::EmbeddingData {
        let a_slice = a.as_slice();
        let b_slice = b.as_slice();

        let lerped: Vec<f32> = a_slice.iter()
            .zip(b_slice.iter())
            .map(|(a_val, b_val)| a_val + (b_val - a_val) * t)
            .collect();

        S::EmbeddingData::from_slice(&lerped)
    }

    async fn route_messages(&self, msgs: &[OutMessage<S>]) {
        for outgoing in msgs {
            if &outgoing.destination.address == self.protean.address() {
                // Local routing
                if let Some(channel) = self.peer_channels.get(&outgoing.destination.uuid) {
                    let _ = channel.send(outgoing.message.clone());
                }
            } else {
                tracing::trace!("[Actor {}] Sending REMOTE message to {} at {}",
                    self.protean.uuid(), outgoing.destination.uuid, outgoing.destination.address);
                if let Some(mut worker_entry) = self.workers.get_mut(&outgoing.destination.address) {
                    let route_message_proto = RouteMessageRequest {
                        destination_uuid: outgoing.destination.uuid.to_bytes(),
                        message: Some(ProteanMessageProto::from(outgoing.message.clone())),
                    };

                    match worker_entry.route_message(Request::new(route_message_proto)).await {
                        Ok(_) => {}
                        Err(e) => {
                            tracing::error!("Failed to route message to {}: {}", outgoing.destination.address, e);
                        }
                    }
                } else {
                    tracing::error!("Worker not found for address: {}", outgoing.destination.address);
                }
            }
        }
    }

    async fn forward_events(&mut self, events: &[ProteanEvent<S>]) {
        for event in events {
            tracing::debug!("[Actor {}] Event from message processing: {:?}", self.protean.uuid(), event);
            let event_proto = Self::event_to_proto(&event, &self.protean.uuid());

            match self.coordinator.forward_event(Request::new(event_proto)).await {
                Ok(_) => {
                    tracing::trace!("[{}] Successfully forwarded event to coordinator", self.protean.uuid());
                }
                Err(e) => {
                    tracing::error!(
                        "[{}] Failed to forward event to coordinator: {}",
                        self.protean.uuid(), e
                    );
                }
            }
        }
    }
}