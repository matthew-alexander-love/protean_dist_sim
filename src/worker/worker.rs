use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::sync::oneshot;
use tokio::runtime::Handle;
use tonic::{transport::Channel, Request, Response, Status};

use dashmap::DashMap;
use crate::proto::dist_sim::{
    worker_node_server::WorkerNode, worker_node_client::WorkerNodeClient,
    coordinator_node_client::CoordinatorNodeClient, RouteMessageRequest, Ack,
    LoadEmbeddingsRequest, CreatePeersRequest, DeletePeersRequest,
    DriftPeerRequest,
    QueryRequest, QueryResponse, SnapshotRequest, NetworkSnapshot, WorkerInfo,
    PingRequest,
};
use crate::proto::protean::SnvConfigProto;
use crate::worker::actor::{ControlMessage, ActorProtean};

use protean::{
    ProteanConfig,
    Peer,
    ProteanMessage,
    ProteanPeer,
    QueryConfig,
    SnvConfig,
    embedding_space::{EmbeddingSpace, Embedding},
    address::Address,
    uuid::Uuid,
};

struct EmbeddingPool<S: EmbeddingSpace> {
    embeddings: HashMap<u64, S::EmbeddingData>,
}

impl<S: EmbeddingSpace> EmbeddingPool<S> {
    fn new() -> Self {
        Self {
            embeddings: HashMap::new(),
        }
    }

    fn get(&self, global_idx: u64) -> Option<S::EmbeddingData> {
        self.embeddings.get(&global_idx).cloned()
    }

    fn insert(&mut self, global_idx: u64, embedding: S::EmbeddingData) {
        self.embeddings.insert(global_idx, embedding);
    }
}

pub struct Worker<S: EmbeddingSpace> {
    worker_id: String,
    my_address: Arc<RwLock<Address>>,

    /// gRPC connections to other worker
    workers: Arc<DashMap<Address, WorkerNodeClient<Channel>>>,

    /// gRPC connection to the coordinator (lazily connected)
    coordinator: Arc<RwLock<Option<CoordinatorNodeClient<Channel>>>>,

    /// Address of the coordinator for lazy connection
    coordinator_address: Option<String>,

    /// Channels to deliver message to other nodes
    actor_protocol_channels: Arc<DashMap<Uuid, UnboundedSender<ProteanMessage<S>>>>,
    actor_control_channels: Arc<DashMap<Uuid, UnboundedSender<ControlMessage<S>>>>,

    /// Handle to the tokio runtime (uses existing runtime, doesn't create a new one)
    runtime_handle: Handle,

    /// Config for making new peers
    actor_config: Arc<RwLock<ProteanConfig>>,

    /// Embedding pool for this worker (indexed by local index)
    embedding_pool: Arc<RwLock<EmbeddingPool<S>>>,
}

impl<S: EmbeddingSpace> Worker<S> {
    pub fn new(worker_id: String, my_address: Address, coordinator_address: Option<String>) -> Self {
        // Use the current tokio runtime handle instead of creating a new one
        // This avoids the "Cannot drop a runtime in a context where blocking is not allowed" panic
        let runtime_handle = Handle::current();

        Self {
            worker_id,
            my_address: Arc::new(RwLock::new(my_address)),
            workers: Arc::new(DashMap::new()),
            coordinator: Arc::new(RwLock::new(None)),
            coordinator_address,
            actor_protocol_channels: Arc::new(DashMap::new()),
            actor_control_channels: Arc::new(DashMap::new()),
            runtime_handle,
            actor_config: Arc::new(RwLock::new(ProteanConfig::default())),
            embedding_pool: Arc::new(RwLock::new(EmbeddingPool::new())),
        }
    }

    /// Get or create coordinator client (lazy connection)
    async fn get_coordinator_client(&self) -> Result<CoordinatorNodeClient<Channel>, Status> {
        // Check if we already have a connection
        {
            let guard = self.coordinator.read()
                .map_err(|_| Status::internal("Failed to acquire coordinator read lock"))?;
            if let Some(ref client) = *guard {
                return Ok(client.clone());
            }
        }

        // Need to connect - get the address
        let addr = self.coordinator_address.as_ref()
            .ok_or_else(|| Status::failed_precondition("Coordinator address not configured"))?;

        let url = format!("http://{}", addr);
        tracing::info!("[{}] Connecting to coordinator at {}", self.worker_id, url);

        let client = CoordinatorNodeClient::connect(url.clone())
            .await
            .map_err(|e| Status::internal(format!("Failed to connect to coordinator: {}", e)))?
            .max_decoding_message_size(100 * 1024 * 1024)
            .max_encoding_message_size(100 * 1024 * 1024);

        // Store the client
        {
            let mut guard = self.coordinator.write()
                .map_err(|_| Status::internal("Failed to acquire coordinator write lock"))?;
            *guard = Some(client.clone());
        }

        tracing::info!("[{}] Connected to coordinator", self.worker_id);
        Ok(client)
    }

    /// Convert global dataset index to UUID
    /// UUIDs are derived deterministically from global indices
    pub(crate) fn global_index_to_uuid(global_index: u64) -> Uuid {
        let mut bytes = [0u8; 64];
        bytes[0..8].copy_from_slice(&global_index.to_be_bytes());
        Uuid::from_bytes(bytes)
    }

    /// Convert UUID back to global dataset index
    #[allow(dead_code)]
    pub(crate) fn uuid_to_global_index(uuid: &Uuid) -> u64 {
        let bytes = uuid.as_bytes();
        // Safe: we know bytes is at least 64 bytes, so first 8 bytes always exist
        u64::from_be_bytes(bytes[0..8].try_into().unwrap_or([0u8; 8]))
    }

}

#[tonic::async_trait]
impl<S: EmbeddingSpace + Send + Sync + 'static> WorkerNode for Worker<S>
where
    S::EmbeddingData: Send + Sync + Embedding<Scalar = f32>,
{
    async fn ping(&self, _request: Request<PingRequest>) -> Result<Response<Ack>, Status> {
        Ok(Response::new(Ack {
            success: true,
            message: format!("Worker {} is healthy", self.worker_id),
        }))
    }

    async fn set_config(&self, request: Request<SnvConfigProto>) -> Result<Response<Ack>, Status> {
        let snv_config: SnvConfig = request.into_inner().into();
        match self.actor_config.write() {
            Ok(mut config) => {
                config.snv_config = snv_config;
                Ok(Response::new(Ack { success: true, message: "Configuration updated".to_string() }))
            }
            Err(e) => Err(Status::internal(format!("Failed to acquire config lock: {}", e))),
        }
    }

    async fn load_embeddings(&self, request: Request<LoadEmbeddingsRequest>) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();

        match self.embedding_pool.write() {
            Ok(mut pool) => {
                let mut count = 0;
                for indexed_emb in req.embeddings.into_iter() {
                    let global_idx = indexed_emb.global_idx;
                    match indexed_emb.embeddings {
                        Some(tensor_proto) => {
                            match S::EmbeddingData::try_from(tensor_proto) {
                                Ok(embedding) => {
                                    pool.insert(global_idx, embedding);
                                    count += 1;
                                }
                                Err(e) => return Err(Status::invalid_argument(
                                    format!("Failed to convert embedding at index {}: {:?}", global_idx, e)
                                )),
                            }
                        }
                        None => return Err(Status::invalid_argument(
                            format!("Missing embedding data at index {}", global_idx)
                        )),
                    }
                }
                Ok(Response::new(Ack {
                    success: true,
                    message: format!("Loaded {} embeddings", count)
                }))
            }
            Err(e) => Err(Status::internal(format!("Failed to acquire embedding pool lock: {}", e))),
        }
    }

    async fn create_peers(&self, request: Request<CreatePeersRequest>) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let mut created_count = 0;

        // Get coordinator client (lazy connection)
        let coordinator = self.get_coordinator_client().await?;

        let config = match self.actor_config.read() {
            Ok(guard) => guard.clone(),
            Err(e) => return Err(Status::internal(format!("Config lock error: {}", e))),
        };

        let max_step_interval = config.snv_config.max_exploration_interval;

        for global_index in req.global_indices.iter() {
            let uuid = Self::global_index_to_uuid(*global_index);

            let embedding = match self.embedding_pool.read() {
                Ok(pool) => match pool.get(*global_index) {
                    Some(e) => e,
                    None => { tracing::warn!("Embedding not found for index {}", global_index); continue; }
                },
                Err(e) => {
                    tracing::warn!("Failed to read embedding pool: {}", e);
                    continue;
                }
            };

            let (protocol_tx, protocol_rx) = unbounded_channel();
            let (control_tx, control_rx) = unbounded_channel();

            self.actor_protocol_channels.insert(uuid.clone(), protocol_tx);
            self.actor_control_channels.insert(uuid.clone(), control_tx.clone());

            let actor = ActorProtean::new(
                self.my_address.clone(), uuid.clone(), embedding, config.clone(),
                protocol_rx, control_rx, self.actor_protocol_channels.clone(),
                coordinator.clone(), self.workers.clone(),
                Duration::from_millis(10), max_step_interval,
            );

            self.runtime_handle.spawn(actor.run());
            created_count += 1;

            if let Some(ref bootstrap_info) = req.bootstrap_peer {
                match bootstrap_info.embedding.as_ref() {
                    Some(emb_proto) => {
                        match S::EmbeddingData::try_from(emb_proto.clone()) {
                            Ok(bootstrap_emb) => {
                                if control_tx.send(ControlMessage::Bootstrap {
                                    contact_point: ProteanPeer {
                                        embedding: bootstrap_emb,
                                        peer: Peer {
                                            uuid: Uuid::from_slice(&bootstrap_info.uuid),
                                            address: bootstrap_info.worker_address.clone(),
                                        },
                                    },
                                    config: None,
                                }).is_err() {
                                    tracing::warn!("Failed to send bootstrap command to peer {}", uuid);
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Failed to convert bootstrap embedding for peer {}: {:?}", uuid, e);
                            }
                        }
                    }
                    None => {
                        tracing::warn!("Bootstrap peer info missing embedding for peer {}", uuid);
                    }
                }
            }
        }

        Ok(Response::new(Ack { success: true, message: format!("Created {} peers", created_count) }))
    }

    async fn delete_peers(&self, request: Request<DeletePeersRequest>) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let mut deleted_count = 0;
        let mut not_found_count = 0;

        // First, send shutdown signals to all peers
        for global_index in req.global_indices.iter() {
            let uuid = Self::global_index_to_uuid(*global_index);
            if let Some(tx) = self.actor_control_channels.get(&uuid) {
                if tx.send(ControlMessage::Shutdown).is_ok() {
                    deleted_count += 1;
                }
            } else {
                not_found_count += 1;
            }
        }

        // Give actors a moment to process shutdown before removing channels
        // This helps ensure the shutdown message is delivered
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Now remove channels (actors should have received shutdown by now)
        for global_index in req.global_indices.iter() {
            let uuid = Self::global_index_to_uuid(*global_index);
            self.actor_protocol_channels.remove(&uuid);
            self.actor_control_channels.remove(&uuid);
        }

        if not_found_count > 0 {
            tracing::warn!("delete_peers: {} peers not found", not_found_count);
        }

        Ok(Response::new(Ack { success: true, message: format!("Deleted {} peers ({} not found)", deleted_count, not_found_count) }))
    }

    async fn drift_peer(&self, request: Request<DriftPeerRequest>) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let mut drift_count = 0;

        for drift in req.drifts.iter() {
            let peer_uuid = Uuid::from_slice(&drift.peer_uuid);

            // Get the peer's current embedding as the original
            let original_embedding = {
                let control_tx = match self.actor_control_channels.get(&peer_uuid) {
                    Some(tx) => tx,
                    None => { tracing::warn!("Peer {} not found for drift", peer_uuid); continue; }
                };
                let (resp_tx, resp_rx) = oneshot::channel();
                if control_tx.send(ControlMessage::GetEmbedding { response: resp_tx }).is_err() {
                    tracing::warn!("Failed to send GetEmbedding request to peer {}", peer_uuid);
                    continue;
                }
                match resp_rx.await {
                    Ok(emb) => emb,
                    Err(_) => {
                        tracing::warn!("Failed to receive embedding from peer {}", peer_uuid);
                        continue;
                    }
                }
            };

            // Get target embedding from pool
            let target_embedding = match self.embedding_pool.read() {
                Ok(pool) => match pool.get(drift.target_idx) {
                    Some(e) => e,
                    None => { tracing::warn!("Target embedding {} not found", drift.target_idx); continue; }
                },
                Err(e) => {
                    tracing::warn!("Failed to read embedding pool: {}", e);
                    continue;
                }
            };

            if let Some(control_tx) = self.actor_control_channels.get(&peer_uuid) {
                // Use update_interval_ms from proto, default to 100ms if not specified
                let update_interval_ms = if drift.update_interval_ms > 0 { drift.update_interval_ms } else { 100 };
                let _ = control_tx.send(ControlMessage::StartDrift {
                    original_embedding,
                    target_embedding,
                    update_interval: Duration::from_millis(update_interval_ms),
                    total_steps: drift.num_steps as u32,
                });
                drift_count += 1;
            }
        }

        Ok(Response::new(Ack { success: true, message: format!("Started drift for {} peers", drift_count) }))
    }

    async fn execute_query(&self, request: Request<QueryRequest>) -> Result<Response<QueryResponse>, Status> {
        let req = request.into_inner();
        let source_uuid = Uuid::from_slice(&req.source_peer_uuid);

        let query_embedding = match req.query_embedding {
            Some(tp) => S::EmbeddingData::try_from(tp).map_err(|e| Status::invalid_argument(format!("Invalid embedding: {:?}", e)))?,
            None => return Err(Status::invalid_argument("Missing query embedding")),
        };

        let query_config: QueryConfig = req.config.map(QueryConfig::from).unwrap_or_default();

        let control_tx = self.actor_control_channels.get(&source_uuid)
            .ok_or_else(|| Status::not_found(format!("Peer {} not found", source_uuid)))?;

        let (resp_tx, resp_rx) = oneshot::channel();
        control_tx.send(ControlMessage::Query { embedding: query_embedding, k: req.k as usize, config: query_config, response: resp_tx })
            .map_err(|_| Status::internal("Failed to send query"))?;

        match resp_rx.await {
            Ok(Some(query_uuid)) => Ok(Response::new(QueryResponse { query_uuid: query_uuid.to_bytes() })),
            Ok(None) => Err(Status::internal("Query failed to start")),
            Err(_) => Err(Status::internal("Query channel closed")),
        }
    }

    async fn get_snapshot(&self, request: Request<SnapshotRequest>) -> Result<Response<NetworkSnapshot>, Status> {
        let req = request.into_inner();
        let mut peer_snapshots = Vec::new();

        let peer_uuids: Vec<Uuid> = if req.peer_uuids.is_empty() {
            self.actor_control_channels.iter().map(|e| e.key().clone()).collect()
        } else {
            req.peer_uuids.iter().map(|b| Uuid::from_slice(b)).collect()
        };

        for uuid in peer_uuids {
            if let Some(control_tx) = self.actor_control_channels.get(&uuid) {
                let (resp_tx, resp_rx) = oneshot::channel();
                if control_tx.send(ControlMessage::GetSnvSnapshot { response: resp_tx }).is_ok() {
                    if let Ok(snapshot) = resp_rx.await {
                        peer_snapshots.push(snapshot.proto);
                    }
                }
            }
        }

        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Ok(Response::new(NetworkSnapshot { timestamp_ms, worker_id: self.worker_id.clone(), peer_snapshots }))
    }

    async fn load_snapshot(&self, request: Request<NetworkSnapshot>) -> Result<Response<Ack>, Status> {
        let snapshot = request.into_inner();
        let mut loaded_count = 0;

        // Get coordinator client (lazy connection)
        let coordinator = self.get_coordinator_client().await?;

        let config = match self.actor_config.read() {
            Ok(guard) => guard.clone(),
            Err(e) => return Err(Status::internal(format!("Config lock error: {}", e))),
        };

        let max_step_interval = config.snv_config.max_exploration_interval;

        for peer_snapshot in snapshot.peer_snapshots {
            let uuid = Uuid::from_slice(&peer_snapshot.local_uuid);

            let (protocol_tx, protocol_rx) = unbounded_channel();
            let (control_tx, control_rx) = unbounded_channel();

            self.actor_protocol_channels.insert(uuid.clone(), protocol_tx);
            self.actor_control_channels.insert(uuid.clone(), control_tx);

            match ActorProtean::from_proto(
                self.my_address.clone(), peer_snapshot, config.clone(),
                protocol_rx, control_rx, self.actor_protocol_channels.clone(),
                coordinator.clone(), self.workers.clone(),
                Duration::from_millis(10), max_step_interval,
            ) {
                Ok(actor) => { self.runtime_handle.spawn(actor.run()); loaded_count += 1; }
                Err(e) => {
                    tracing::error!("Failed to restore peer {}: {:?}", uuid, e);
                    self.actor_protocol_channels.remove(&uuid);
                    self.actor_control_channels.remove(&uuid);
                }
            }
        }

        Ok(Response::new(Ack { success: loaded_count > 0, message: format!("Loaded {} peers", loaded_count) }))
    }

    async fn register_worker(&self, request: Request<WorkerInfo>) -> Result<Response<Ack>, Status> {
        let address: Address = request.into_inner().address;

        match WorkerNodeClient::connect(format!("http://{}", address)).await {
            Ok(client) => {
                let client = client.max_decoding_message_size(100 * 1024 * 1024).max_encoding_message_size(100 * 1024 * 1024);
                self.workers.insert(address.clone(), client);
                tracing::info!("[{}] Registered worker at {}", self.worker_id, address);
                Ok(Response::new(Ack { success: true, message: format!("Registered worker at {}", address) }))
            }
            Err(e) => {
                tracing::error!("[{}] Failed to connect to worker {}: {}", self.worker_id, address, e);
                Err(Status::unavailable(format!("Failed to connect: {}", e)))
            }
        }
    }

    async fn route_message(&self, request: Request<RouteMessageRequest>) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let dest_uuid = Uuid::from_slice(&req.destination_uuid);

        let message: ProteanMessage<S> = match req.message {
            Some(proto) => ProteanMessage::try_from(proto).map_err(|e| Status::invalid_argument(format!("Invalid message: {:?}", e)))?,
            None => return Err(Status::invalid_argument("Missing message")),
        };

        match self.actor_protocol_channels.get(&dest_uuid) {
            Some(ch) => {
                ch.send(message).map_err(|_| Status::internal("Failed to send message"))?;
                Ok(Response::new(Ack { success: true, message: String::new() }))
            }
            None => Err(Status::not_found(format!("Peer {} not found", dest_uuid))),
        }
    }
}