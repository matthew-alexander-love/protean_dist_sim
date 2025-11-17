use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::mpsc::{UnboundedSender, UnboundedReceiver};
use tokio::sync::oneshot;
use tokio::time::{interval, MissedTickBehavior};

use protean::{
    Protean,
    ProteanConfig,
    ProteanMessage,
    OutMessage,
    ProteanPeer,
    QueryConfig,
    QueryStatus,
    SnvSnapshot,
    ExplorationConfig,
    embedding_space::EmbeddingSpace,
    protean::ProteanEvent,
    address::Address,
    uuid::Uuid,
};

use crate::proto::protean::SparseNeighborViewProto;

/// State for embedding drift functionality
pub(crate) struct DriftState<S: EmbeddingSpace> {
    /// The original embedding (from the pool at original_embedding_index)
    pub original_embedding: S::EmbeddingData,
    /// The target embedding to drift towards
    pub target_embedding: S::EmbeddingData,
    /// Deadline for the next drift update
    pub next_update_deadline: Instant,
    /// Interval between drift updates
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
        response: oneshot::Sender<SnvSnapshot>,
    },
    /// Get the full SNV as proto (for snapshot/restore)
    GetSnvProto {
        response: oneshot::Sender<SparseNeighborViewProto>,
    },
    /// Get the local embedding
    GetEmbedding {
        response: oneshot::Sender<S::EmbeddingData>,
    },
    /// Load a complete SNV snapshot
    LoadSnvSnapshot {
        snapshot: SparseNeighborViewProto,
        response: oneshot::Sender<bool>,
    },
    /// Start embedding drift towards a target
    StartDrift {
        original_embedding: S::EmbeddingData,
        target_embedding: S::EmbeddingData,
        update_interval: Duration,
        total_steps: u32,
    },
}

pub(crate) struct ActorProtean<S: EmbeddingSpace> {
    /// The Protean peer instance (owned by this actor)
    protean: Protean<S>,

    /// Incoming protocol message channel
    msg_rx: UnboundedReceiver<ProteanMessage<S>>,
    /// Incoming control messages
    control_rx: UnboundedReceiver<ControlMessage<S>>,

    /// Shared map of all LOCAL peer channels for message routing
    peer_channels: Arc<DashMap<Uuid, UnboundedSender<ProteanMessage<S>>>>,
    /// Messages for peers ota
    remote_msg_tx: UnboundedSender<OutMessage<S>>,
    /// Event Capture channel
    event_tx: UnboundedSender<ProteanEvent<S>>,

    /// Minimum step interval (when active)
    min_step_interval: Duration,
    /// Maximum step interval (when idle)
    max_step_interval: Option<Duration>,

    /// Original embedding index (for drift functionality)
    original_embedding_index: u64,
    /// Current drift state (None if not drifting)
    drift_state: Option<DriftState<S>>,
}

impl<S: EmbeddingSpace> ActorProtean<S> {
    pub(crate) fn new(
        local_address: Arc<RwLock<Address>>,
        uuid: Uuid,
        local_embedding: S::EmbeddingData,
        original_embedding_index: u64,
        config: ProteanConfig,
        msg_rx: UnboundedReceiver<ProteanMessage<S>>,
        control_rx: UnboundedReceiver<ControlMessage<S>>,
        peer_channels: Arc<DashMap<Uuid, UnboundedSender<ProteanMessage<S>>>>,
        remote_msg_tx: UnboundedSender<OutMessage<S>>,
        event_tx: UnboundedSender<ProteanEvent<S>>,
        min_step_interval: Duration,
        max_step_interval: Option<Duration>,
    ) -> Self {
        let address = local_address.read().unwrap().clone();
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
            remote_msg_tx,
            event_tx,
            min_step_interval,
            max_step_interval,
            original_embedding_index,
            drift_state: None,
        }
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
                    self.process_control(msg);
                },

                Some(msg) = self.msg_rx.recv() => {
                    self.process_message(msg);

                    // Reset Timer on Work done at this node
                    current_interval = self.min_step_interval;
                    step_interval = interval(current_interval);
                    step_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
                },

                _ = step_interval.tick() => {
                    let was_productive = self.auto_step();
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

    fn process_message(&mut self, msg: ProteanMessage<S>) {
        // Check drift before processing message
        self.update_drift_if_needed();

        let step = self.protean.event(msg);
        self.route_messages(step.messages());

        for event in step.events() {
            if let Err(e) = self.event_tx.send(event.clone()) {
                tracing::error!("Failed to send event: {:?}", e);
            }
        }
    }

    fn process_control(&mut self, msg: ControlMessage<S>) {
        // Check drift before processing control messages that trigger steps
        match &msg {
            ControlMessage::Bootstrap { .. } | ControlMessage::Query { .. } => {
                self.update_drift_if_needed();
            },
            _ => {}
        }

        match msg {
            ControlMessage::Bootstrap { contact_point, config } => {
                let success = self.protean.bootstrap(contact_point, config);
                if success {
                    // Bootstrap started, step to get initial messages
                    let step = self.protean.step();
                    self.route_messages(step.messages());

                    for event in step.events() {
                        let _ = self.event_tx.send(event.clone());
                    }
                }
            },
            ControlMessage::Query { embedding, k, config, response } => {
                let query_uuid = self.protean.query(embedding, k, config);
                let _ = response.send(query_uuid);

                let step = self.protean.step();
                self.route_messages(step.messages());

                for event in step.events() {
                    let _ = self.event_tx.send(event.clone());
                }
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
                let snapshot = self.protean.snv_snapshot();
                let _ = response.send(snapshot);
            },
            ControlMessage::GetSnvProto { response } => {
                let snv_proto = self.protean.to_proto();
                let _ = response.send(snv_proto);
            },
            ControlMessage::GetEmbedding { response } => {
                let embedding = self.protean.embedding().clone();
                let _ = response.send(embedding);
            },
            ControlMessage::LoadSnvSnapshot { snapshot: _, response } => {
                // TODO: Implement load_snv_snapshot in Protean
                // For now, just return false
                let _ = response.send(false);
            },
            ControlMessage::StartDrift { original_embedding, target_embedding, update_interval, total_steps } => {
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

    fn auto_step(&mut self) -> bool {
        // Check if we need to update embedding due to drift
        self.update_drift_if_needed();

        let step = self.protean.step();

        let has_messages = !step.messages().is_empty();
        let has_events = !step.events().is_empty();

        self.route_messages(step.messages());

        for event in step.events() {
            if let Err(e) = self.event_tx.send(event.clone()) {
                tracing::error!("Failed to send event: {:?}", e);
            }
        }

        has_messages || has_events
    }

    /// Check drift state and update embedding if deadline has passed
    fn update_drift_if_needed(&mut self) {
        if let Some(drift_state) = &mut self.drift_state {
            let now = Instant::now();

            if now >= drift_state.next_update_deadline {
                // Time to update embedding
                drift_state.current_step += 1;

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
    ///
    /// This assumes the embedding scalar type is f32 (true for F32L2Space and similar).
    /// For other scalar types, this would need to be generic over the scalar type.
    fn lerp_embedding(a: &S::EmbeddingData, b: &S::EmbeddingData, t: f32) -> S::EmbeddingData {
        use protean::embedding_space::Embedding;

        // Get the raw vector data
        let a_slice = a.as_slice();
        let b_slice = b.as_slice();

        // Compute LERP for each dimension: a + (b - a) * t
        // This works because Scalar is f32 for F32L2Space
        let lerped: Vec<<S::EmbeddingData as Embedding>::Scalar> = a_slice.iter()
            .zip(b_slice.iter())
            .map(|(a_val, b_val)| {
                // Cast to f32 (should be no-op for f32 scalar types)
                let a_f32 = unsafe { *(a_val as *const _ as *const f32) };
                let b_f32 = unsafe { *(b_val as *const _ as *const f32) };
                let result = a_f32 + (b_f32 - a_f32) * t;
                // Cast back to Scalar type
                unsafe { *((&result) as *const f32 as *const <S::EmbeddingData as Embedding>::Scalar) }
            })
            .collect();

        // Convert back to EmbeddingData
        S::EmbeddingData::from_slice(&lerped)
    }

    fn route_messages(&self, msgs: &[OutMessage<S>]) {
        for outgoing in msgs {
            if &outgoing.destination.address == self.protean.address() {
                if let Some(channel) = self.peer_channels.get(&outgoing.destination.uuid) {
                    let _ = channel.send(outgoing.message.clone());
                }
            } else {
                let _ = self.remote_msg_tx.send(outgoing.clone());
            }
        }
    }
}