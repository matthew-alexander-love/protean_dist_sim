use std::collections::HashMap;
use std::sync::{Arc};
use std::time::Duration;

use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::sync::oneshot;
use tokio::sync::RwLock;
use tokio::runtime::Handle;
use tonic::{transport::Channel, Request, Response, Status};

use dashmap::DashMap;
use crate::proto::dist_sim::{
    worker_node_server::WorkerNode, 
    worker_node_client::WorkerNodeClient,
    coordinator_node_client::CoordinatorNodeClient,
    Ack,
    ProteanConfigProto,
    BootstrapPeerRequest,
    CreatePeersRequest, DeletePeersRequest, ChurnPeersRequest,
    DriftPeerProto, DriftPeerRequest,
    QueryRequest, QueryResponse,
    SnapshotRequest, NetworkSnapshot,
    WorkerInfo, RouteMessageRequest,
    ProteanEventType, 
    StateChangedEvent,
    QueryCompletedEvent,
    BootstrapCompletedEvent,
    ProteanEventProto,
};
use crate::proto::protean::{SnvConfigProto, PeerProto};
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

pub struct Worker<S: EmbeddingSpace> {
    worker_id: String,
    my_address: Arc<RwLock<Address>>,

    /// gRPC connections to other worker
    workers: Arc<DashMap<Address, WorkerNodeClient<Channel>>>,

    /// gRPC connection to the coordinator
    coordinator: CoordinatorNodeClient<Channel>,

    coordinator_address: Address,

    /// Channels to deliver message to other nodes
    actor_protocol_channels: Arc<DashMap<Uuid, UnboundedSender<ProteanMessage<S>>>>,
    actor_control_channels: Arc<DashMap<Uuid, UnboundedSender<ControlMessage<S>>>>,

    /// Handle to the tokio runtime (uses existing runtime, doesn't create a new one)
    runtime_handle: Handle,

    /// Config for making new peers
    actor_config: Arc<RwLock<ProteanConfig>>,

    /// Cached bootstrap peers for entering peers
    bootstrap_servers: Arc<RwLock<Vec<ProteanPeer<S>>>>,
}

impl<S: EmbeddingSpace> Worker<S> {
    pub async fn new(worker_id: String, my_address: Address, coordinator_address: Address) -> Result<Self, Status> {
        // Use the current tokio runtime handle instead of creating a new one
        // This avoids the "Cannot drop a runtime in a context where blocking is not allowed" panic
        let runtime_handle = Handle::current();

        let coordinator_client=  Self::get_coordinator_client(coordinator_address.clone()).await?;

        Ok(Self {
            worker_id,
            my_address: Arc::new(RwLock::new(my_address)),
            workers: Arc::new(DashMap::new()),
            coordinator: coordinator_client,
            coordinator_address,
            actor_protocol_channels: Arc::new(DashMap::new()),
            actor_control_channels: Arc::new(DashMap::new()),
            runtime_handle,
            actor_config: Arc::new(RwLock::new(ProteanConfig::default())),
            bootstrap_servers: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Get or create coordinator client (lazy connection)
    async fn get_coordinator_client(coordinator_address: Address) -> Result<CoordinatorNodeClient<Channel>, Status> {

        let url = format!("http://{}", coordinator_address);
        tracing::info!("Connecting to coordinator at {}", url);

        let client = CoordinatorNodeClient::connect(url.clone())
            .await
            .map_err(|e| Status::internal(format!("Failed to connect to coordinator: {}", e)))?
            .max_decoding_message_size(100 * 1024 * 1024)
            .max_encoding_message_size(100 * 1024 * 1024);

        tracing::info!("Connected to coordinator at {}", coordinator_address);
        Ok(client)
    }

    fn new_actor(&mut self, uuid: &Uuid, embedding: S::EmbeddingData, config: ProteanConfig) -> ActorProtean<S> {
        let (protocol_tx, protocol_rx) = unbounded_channel();
        let (control_tx, control_rx) = unbounded_channel();

        self.actor_protocol_channels.insert(uuid.clone(), protocol_tx);
        self.actor_control_channels.insert(uuid.clone(), control_tx.clone());

        ActorProtean::new(
            self.my_address.clone(), 
            uuid.clone(), 
            embedding, 
            config,
            protocol_rx, 
            control_rx, 
            self.actor_protocol_channels.clone(),
            self.coordinator.clone(), 
            self.workers.clone(),
            Duration::from_millis(10), 
            config.snv_config.max_exploration_interval,
        )
    }

}

#[tonic::async_trait]
impl<S: EmbeddingSpace + Send + Sync + 'static> WorkerNode for Worker<S>
where
    S::EmbeddingData: Send + Sync + Embedding<Scalar = f32>,
{
    async fn set_config(&self, request: Request<ProteanConfigProto>) -> Result<Response<Ack>, Status> {
        let protean_config_proto: ProteanConfigProto = request.into_inner().into();
        if let Some(snv_config_proto) = protean_config_proto.snv_config {
            let snv_config: SnvConfig = snv_config_proto.into();

            let mut local_config = self.actor_config.write().await;
            *local_config = ProteanConfig { 
                timeout: Duration::from_secs(protean_config_proto.timeout_sec),
                snv_config, 
                max_concurrent_queries: protean_config_proto.max_concurrent_queries as usize,
            };
            Ok(Response::new(Ack {}))
        } else {
            Err(Status::invalid_argument("SnvConfig missing from ProteanConfigProto"))
        }
    }

    async fn set_bootstrap_servers(&self, request: Request<BootstrapPeerRequest>) -> Result<Response<Ack>, Status> {
        let bootstrap_peer_request = request.into_inner();
        
        let bootstrap_servers = self.bootstrap_servers.write().await;
        *bootstrap_servers = bootstrap_peer_request.bs_server.iter().map(|peer_proto| peer_proto.into()).collect();
        Ok(Response::new(Ack {}))
    }

    async fn create_peers(&self, request: Request<CreatePeersRequest>) -> Result<Response<Ack>, Status> {
        let create_peers_request = request.into_inner();
        let mut created_count = 0;

        let config = self.actor_config.read().await.clone();

        // for peer:
        //      create peer
        //      send bootstrap command
        
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