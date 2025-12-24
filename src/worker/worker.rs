use std::sync::{Arc, Mutex};
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
    DriftPeerRequest,
    QueryRequest, QueryResponse,
    SnapshotRequest, NetworkSnapshot, SnvSnapshotProto,
    WorkerInfo, RouteMessageRequest,
};
use crate::proto::protean::SparseNeighborViewProto;
use crate::worker::actor::{ControlMessage, ActorProtean};

use protean::{
    ProteanConfig,
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

    /// Config for making new peers
    actor_config: Arc<RwLock<ProteanConfig>>,

    /// Cached bootstrap peers for entering peers
    bootstrap_servers: Arc<RwLock<Vec<ProteanPeer<S>>>>,
    bootstrap_peer_idx: Mutex<usize>,
}

impl<S: EmbeddingSpace> Worker<S>
where
    S::EmbeddingData: Embedding<Scalar = f32>,
{
    pub async fn new(worker_id: String, my_address: Address, coordinator_address: Address) -> Result<Self, Status> {

        let coordinator_client=  Self::get_coordinator_client(coordinator_address.clone()).await?;

        Ok(Self {
            worker_id,
            my_address: Arc::new(RwLock::new(my_address)),
            workers: Arc::new(DashMap::new()),
            coordinator: coordinator_client,
            coordinator_address,
            actor_protocol_channels: Arc::new(DashMap::new()),
            actor_control_channels: Arc::new(DashMap::new()),
            actor_config: Arc::new(RwLock::new(ProteanConfig::default())),
            bootstrap_servers: Arc::new(RwLock::new(Vec::new())),
            bootstrap_peer_idx: Mutex::new(0),
        })
    }

    /// Get or create coordinator client
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

    fn new_actor(&self, uuid: &Uuid, embedding: S::EmbeddingData, config: ProteanConfig) -> UnboundedSender<ControlMessage<S>> {
        let (protocol_tx, protocol_rx) = unbounded_channel();
        let (control_tx, control_rx) = unbounded_channel();

        self.actor_protocol_channels.insert(uuid.clone(), protocol_tx);
        self.actor_control_channels.insert(uuid.clone(), control_tx.clone());

        let max_exploration_interval = config.snv_config.max_exploration_interval;
        let actor = ActorProtean::new(
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
            max_exploration_interval,
        );

        tokio::spawn(async move {
            actor.run();
        });

        control_tx
    }

    async fn new_actor_from_proto(&self, uuid: &Uuid, proto: SparseNeighborViewProto) -> Result<UnboundedSender<ControlMessage<S>>, Status> {
        let (protocol_tx, protocol_rx) = unbounded_channel();
        let (control_tx, control_rx) = unbounded_channel();

        self.actor_protocol_channels.insert(uuid.clone(), protocol_tx);
        self.actor_control_channels.insert(uuid.clone(), control_tx.clone());

        let config = self.actor_config.read().await.clone();
        let actor = ActorProtean::from_proto(
            self.my_address.clone(),
            proto,
            config.clone(),
            protocol_rx,
            control_rx,
            self.actor_protocol_channels.clone(),
            self.coordinator.clone(),
            self.workers.clone(),
            Duration::from_millis(10),
            config.snv_config.max_exploration_interval,
        ).map_err(|e| Status::internal(format!("Failed to create actor from snapshot: {:?}", e)))?;

        tokio::spawn(async move {
            actor.run();
        });

        Ok(control_tx)
    }


    async fn get_bootstrap_server(&self) -> Result<ProteanPeer<S>, Status> {
        let bootstrap_servers = self.bootstrap_servers.read().await;
        let bootstrap_len: usize = bootstrap_servers.len();
        if bootstrap_len == 0 {
            return Err(Status::failed_precondition("No bootstrap servers configured"));
        }
        let mut idx_guard = self.bootstrap_peer_idx.lock().unwrap();
        let idx = *idx_guard % bootstrap_len;
        *idx_guard = idx_guard.wrapping_add(1);
        let bootstrap_peer = bootstrap_servers[idx].clone();
        Ok(bootstrap_peer)
    }

    async fn bootstrap_peer(&self, peer_control_tx: UnboundedSender<ControlMessage<S>>) -> Result<(), Status> {
        let bootstrap_peer = self.get_bootstrap_server().await?;
        let bs_cmd = ControlMessage::Bootstrap {
            contact_point: bootstrap_peer,
            config: None,
        };
        peer_control_tx.send(bs_cmd).map_err(|_| Status::internal("Failed to send bootstrap command"))?;
        Ok(())
    }

    async fn create_peers(&self, peers: Vec<ProteanPeer<S>>) -> Result<(), Status> {
        let config = self.actor_config.read().await.clone();
        let bootstrap_servers = self.bootstrap_servers.read().await.clone();
        let bootstrap_len = bootstrap_servers.len();

        if bootstrap_len == 0 {
            return Err(Status::failed_precondition("No bootstrap servers configured"));
        }

        for (idx, peer) in peers.into_iter().enumerate() {
            let peer_control_tx: UnboundedSender<ControlMessage<S>> = self.new_actor(&peer.peer.uuid, peer.embedding, config.clone());

            self.bootstrap_peer(peer_control_tx).await?;

        }

        Ok(())
    }

    async fn delete_peers(&self, uuids: Vec<Uuid>) -> Result<(), Status> {
        for uuid in uuids.iter() {
            if let Some(control_tx) = self.actor_control_channels.get(&uuid) {
                control_tx.send(ControlMessage::Shutdown).map_err(|_| Status::internal("Failed to send shutdown command"))?;
            }
        }
        Ok(())
    }

}

#[tonic::async_trait]
impl<S: EmbeddingSpace + Send + Sync + 'static> WorkerNode for Worker<S>
where
    S::EmbeddingData: Send + Sync + Embedding<Scalar = f32>,
{
    async fn set_config(&self, request: Request<ProteanConfigProto>) -> Result<Response<Ack>, Status> {
        tracing::info!("[Worker {}] set_config called", self.worker_id);
        let protean_config_proto: ProteanConfigProto = request.into_inner().into();
        if let Some(snv_config_proto) = protean_config_proto.snv_config {
            let snv_config: SnvConfig = snv_config_proto.into();

            let mut local_config = self.actor_config.write().await;
            *local_config = ProteanConfig { 
                timeout: Duration::from_secs(protean_config_proto.timeout_sec),
                snv_config, 
                max_concurrent_queries: protean_config_proto.max_concurrent_queries as usize,
            };
            tracing::info!(
                "[Worker {}] Config updated: timeout={}s, max_concurrent_queries={}",
                self.worker_id,
                protean_config_proto.timeout_sec,
                protean_config_proto.max_concurrent_queries
            );
            Ok(Response::new(Ack {}))
        } else {
            tracing::warn!("[Worker {}] set_config failed: SnvConfig missing from ProteanConfigProto", self.worker_id);
            Err(Status::invalid_argument("SnvConfig missing from ProteanConfigProto"))
        }
    }

    async fn set_bootstrap_servers(&self, request: Request<BootstrapPeerRequest>) -> Result<Response<Ack>, Status> {
        let bootstrap_peer_request = request.into_inner();
        let num_servers = bootstrap_peer_request.bs_server.len();
        tracing::info!("[Worker {}] set_bootstrap_servers called with {} servers", self.worker_id, num_servers);
        
        let mut bootstrap_servers = self.bootstrap_servers.write().await;
        *bootstrap_servers = bootstrap_peer_request.bs_server.into_iter()
            .filter_map(|peer_proto| ProteanPeer::try_from(peer_proto).ok())
            .collect();
        tracing::info!("[Worker {}] Bootstrap servers updated: {} servers configured", self.worker_id, bootstrap_servers.len());
        Ok(Response::new(Ack {}))
    }

    async fn create_peers(&self, request: Request<CreatePeersRequest>) -> Result<Response<Ack>, Status> {
        let create_peers_request = request.into_inner();
        let num_peers = create_peers_request.peers.len();
        tracing::info!("[Worker {}] create_peers called: creating {} peers", self.worker_id, num_peers);

        let peers: Vec<ProteanPeer<S>> = create_peers_request.peers.into_iter()
            .filter_map(|peer| ProteanPeer::try_from(peer).ok())
            .collect();
        match self.create_peers(peers).await {
            Ok(_) => {
                tracing::info!("[Worker {}] Successfully created {} peers", self.worker_id, num_peers);
                Ok(Response::new(Ack {}))
            }
            Err(e) => {
                tracing::error!("[Worker {}] Failed to create peers: {}", self.worker_id, e);
                Err(e)
            }
        }
    }

    async fn delete_peers(&self, request: Request<DeletePeersRequest>) -> Result<Response<Ack>, Status> {
        let delete_peers_request: DeletePeersRequest = request.into_inner();
        let num_uuids = delete_peers_request.uuids.len();
        tracing::info!("[Worker {}] delete_peers called: deleting {} peers", self.worker_id, num_uuids);
        
        let uuids: Vec<Uuid> = delete_peers_request.uuids.iter()
            .map(|uuid_bytes| Uuid::from_slice(uuid_bytes))
            .collect();

        match self.delete_peers(uuids.clone()).await {
            Ok(_) => {
                tracing::info!("[Worker {}] Successfully deleted {} peers", self.worker_id, uuids.len());
                Ok(Response::new(Ack {}))
            }
            Err(e) => {
                tracing::error!("[Worker {}] Failed to delete peers: {}", self.worker_id, e);
                Err(e)
            }
        }
    }

    async fn churn_peers(&self, request: Request<ChurnPeersRequest>) -> Result<Response<Ack>, Status> {
        let churn_peers_request = request.into_inner();
        let create_request = churn_peers_request.create.unwrap_or_default();
        let delete_request = churn_peers_request.delete.unwrap_or_default();
        let num_create = create_request.peers.len();
        let num_delete = delete_request.uuids.len();
        tracing::info!(
            "[Worker {}] churn_peers called: creating {} peers, deleting {} peers",
            self.worker_id,
            num_create,
            num_delete
        );

        let peers: Vec<ProteanPeer<S>> = create_request.peers.into_iter()
            .filter_map(|peer| ProteanPeer::try_from(peer).ok())
            .collect();
        match self.create_peers(peers).await {
            Ok(_) => {
                tracing::debug!("[Worker {}] Successfully created {} peers during churn", self.worker_id, num_create);
            }
            Err(e) => {
                tracing::error!("[Worker {}] Failed to create peers during churn: {}", self.worker_id, e);
                return Err(e);
            }
        }

        let uuids: Vec<Uuid> = delete_request.uuids.iter()
            .map(|uuid_bytes| Uuid::from_slice(uuid_bytes))
            .collect();

        match self.delete_peers(uuids.clone()).await {
            Ok(_) => {
                tracing::info!(
                    "[Worker {}] Successfully churned peers: created {}, deleted {}",
                    self.worker_id,
                    num_create,
                    uuids.len()
                );
                Ok(Response::new(Ack {}))
            }
            Err(e) => {
                tracing::error!("[Worker {}] Failed to delete peers during churn: {}", self.worker_id, e);
                Err(e)
            }
        }
    }

    async fn drift_peer(&self, request: Request<DriftPeerRequest>) -> Result<Response<Ack>, Status> {
        let drift_peer_request = request.into_inner();
        let num_drifts = drift_peer_request.drifts.len();
        tracing::info!("[Worker {}] drift_peer called: {} drift requests", self.worker_id, num_drifts);

        for drift in drift_peer_request.drifts.iter() {
            let uuid = Uuid::from_slice(&drift.uuid);
            if drift.drift_steps == 0 {
                tracing::warn!("[Worker {}] drift_peer failed: drift_steps must be > 0 for peer {}", self.worker_id, uuid);
                return Err(Status::invalid_argument("drift_steps must be > 0"));
            }
            let embedding_proto = drift.target_embedding.clone().ok_or_else(|| {
                tracing::warn!("[Worker {}] drift_peer failed: Missing target embedding for peer {}", self.worker_id, uuid);
                Status::invalid_argument("Missing target embedding")
            })?;
            let target_embedding = S::EmbeddingData::try_from(embedding_proto).map_err(|e| {
                tracing::warn!("[Worker {}] drift_peer failed: Invalid target embedding for peer {}: {:?}", self.worker_id, uuid, e);
                Status::invalid_argument(format!("Invalid target embedding: {:?}", e))
            })?;
            let control_tx = self.actor_control_channels.get(&uuid)
                .ok_or_else(|| {
                    tracing::warn!("[Worker {}] drift_peer failed: Peer {} not found", self.worker_id, uuid);
                    Status::not_found(format!("Peer {} not found", uuid))
                })?;
            control_tx.send(ControlMessage::StartDrift {
                target_embedding,
                update_interval: Duration::from_secs(drift.duration_per_step_sec),
                total_steps: drift.drift_steps as u32,
            }).map_err(|e| {
                tracing::error!("[Worker {}] drift_peer failed: Failed to send drift command to peer {}: {:?}", self.worker_id, uuid, e);
                Status::internal("Failed to send drift command")
            })?;
            tracing::debug!(
                "[Worker {}] Started drift for peer {}: {} steps, {}ms per step",
                self.worker_id,
                uuid,
                drift.drift_steps,
                drift.duration_per_step_sec
            );
        }
        tracing::info!("[Worker {}] Successfully initiated {} drift requests", self.worker_id, num_drifts);
        Ok(Response::new(Ack {}))
    }

    async fn execute_query(&self, request: Request<QueryRequest>) -> Result<Response<QueryResponse>, Status> {
        let query_request: QueryRequest = request.into_inner();
        let source_uuid = Uuid::from_slice(&query_request.source_peer_uuid);
        tracing::info!(
            "[Worker {}] execute_query called: peer={}, k={}",
            self.worker_id,
            source_uuid,
            query_request.k
        );
        let embedding_proto = query_request.query_embedding.ok_or_else(|| {
            tracing::warn!("[Worker {}] execute_query failed: Missing query embedding for peer {}", self.worker_id, source_uuid);
            Status::invalid_argument("Missing query embedding")
        })?;
        let query_embedding = S::EmbeddingData::try_from(embedding_proto).map_err(|e| {
            tracing::warn!("[Worker {}] execute_query failed: Invalid embedding for peer {}: {:?}", self.worker_id, source_uuid, e);
            Status::invalid_argument(format!("Invalid embedding: {:?}", e))
        })?;
        let query_config: QueryConfig = query_request.config.map(QueryConfig::from).unwrap_or_default();
        let control_tx = self.actor_control_channels.get(&source_uuid)
            .ok_or_else(|| {
                tracing::warn!("[Worker {}] execute_query failed: Peer {} not found", self.worker_id, source_uuid);
                Status::not_found(format!("Peer {} not found", source_uuid))
            })?;
        let (resp_tx, resp_rx) = oneshot::channel();
        control_tx.send(ControlMessage::Query { embedding: query_embedding, k: query_request.k as usize, config: query_config, response: resp_tx })
            .map_err(|e| {
                tracing::error!("[Worker {}] execute_query failed: Failed to send query to peer {}: {:?}", self.worker_id, source_uuid, e);
                Status::internal("Failed to send query")
            })?;
        match resp_rx.await {
            Ok(Some(query_uuid)) => {
                tracing::info!("[Worker {}] Query started successfully: peer={}, query_uuid={}", self.worker_id, source_uuid, query_uuid);
                Ok(Response::new(QueryResponse { query_uuid: query_uuid.to_bytes() }))
            }
            Ok(None) => {
                tracing::error!("[Worker {}] Query failed to start: peer={}", self.worker_id, source_uuid);
                Err(Status::internal("Query failed to start"))
            }
            Err(e) => {
                tracing::error!("[Worker {}] Query channel closed: peer={}, error={:?}", self.worker_id, source_uuid, e);
                Err(Status::internal("Query channel closed"))
            }
        }
    }

    async fn get_snapshot(&self, request: Request<SnapshotRequest>) -> Result<Response<NetworkSnapshot>, Status> {
        let snapshot_request: SnapshotRequest = request.into_inner();
        let num_requested = snapshot_request.peer_uuids.len();
        let snapshot_name = if snapshot_request.name.is_empty() { "unnamed" } else { &snapshot_request.name };
        tracing::info!(
            "[Worker {}] get_snapshot called: name={}, {} peers requested",
            self.worker_id,
            snapshot_name,
            num_requested
        );

        let mut snapshot = NetworkSnapshot {
            worker_id: self.worker_id.clone(),
            peer_snapshots: Vec::new(),
            peer_uuids: Vec::new(),
            snv_snapshots: Vec::new(),
        };
        let mut successful = 0;
        let mut failed = 0;
        for uuid_bytes in snapshot_request.peer_uuids.iter() {
            let uuid = Uuid::from_slice(uuid_bytes);
            let (resp_tx, resp_rx) = oneshot::channel();
            if let Some(control_tx) = self.actor_control_channels.get(&uuid) {
                control_tx.send(ControlMessage::GetSnvSnapshot { response: resp_tx }).map_err(|e| {
                    tracing::warn!("[Worker {}] get_snapshot failed: Failed to send get snv snapshot command to peer {}: {:?}", self.worker_id, uuid, e);
                    Status::internal("Failed to send get snv snapshot command")
                })?;
            } else {
                tracing::warn!("[Worker {}] get_snapshot: Peer {} not found, skipping", self.worker_id, uuid);
                failed += 1;
                continue;
            }
            match resp_rx.await {
                Ok(actor_snapshot) => {
                    snapshot.peer_uuids.push(uuid.to_bytes());
                    snapshot.snv_snapshots.push(SnvSnapshotProto {
                        total_peers: actor_snapshot.stats.total_peers as u32,
                        routable_peers: actor_snapshot.stats.routable_peers as u32,
                        suspect_peers: actor_snapshot.stats.suspect_peers as u32,
                        pending_pings: actor_snapshot.stats.pending_pings as u32,
                        inflight_pings: actor_snapshot.stats.inflight_pings as u32,
                        dynamism: actor_snapshot.stats.dynamism as u32,
                    });
                    snapshot.peer_snapshots.push(actor_snapshot.proto);
                    successful += 1;
                }
                Err(e) => {
                    tracing::warn!("[Worker {}] get_snapshot: Failed to get snapshot from peer {}: {:?}", self.worker_id, uuid, e);
                    failed += 1;
                }
            }
        }

        tracing::info!(
            "[Worker {}] Snapshot '{}' completed: {} successful, {} failed",
            self.worker_id,
            snapshot_name,
            successful,
            failed
        );
        Ok(Response::new(snapshot))
    }

    async fn load_snapshot(&self, request: Request<NetworkSnapshot>) -> Result<Response<Ack>, Status> {
        let network_snapshot: NetworkSnapshot = request.into_inner();
        let num_peers = network_snapshot.peer_snapshots.len();
        tracing::info!(
            "[Worker {}] load_snapshot called: worker_id={}, {} peers to load",
            self.worker_id,
            network_snapshot.worker_id,
            num_peers
        );
        if network_snapshot.peer_snapshots.len() != network_snapshot.peer_uuids.len() {
            tracing::error!(
                "[Worker {}] load_snapshot failed: peer_snapshots ({}) and peer_uuids ({}) length mismatch",
                self.worker_id,
                network_snapshot.peer_snapshots.len(),
                network_snapshot.peer_uuids.len()
            );
            return Err(Status::invalid_argument("peer_snapshots and peer_uuids length mismatch"));
        }
        let mut successful = 0;
        let mut failed = 0;
        for (idx, snv_proto) in network_snapshot.peer_snapshots.into_iter().enumerate() {
            let uuid = Uuid::from_slice(&network_snapshot.peer_uuids[idx]);
            match self.new_actor_from_proto(&uuid, snv_proto).await {
                Ok(peer_control_tx) => {
                    match self.bootstrap_peer(peer_control_tx).await {
                        Ok(_) => {
                            tracing::debug!("[Worker {}] Successfully loaded peer {} from snapshot", self.worker_id, uuid);
                            successful += 1;
                        }
                        Err(e) => {
                            tracing::warn!("[Worker {}] load_snapshot: Failed to bootstrap peer {}: {}", self.worker_id, uuid, e);
                            failed += 1;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("[Worker {}] load_snapshot: Failed to create actor for peer {}: {}", self.worker_id, uuid, e);
                    failed += 1;
                }
            }
        }
        tracing::info!(
            "[Worker {}] Snapshot load completed: {} successful, {} failed",
            self.worker_id,
            successful,
            failed
        );
        Ok(Response::new(Ack {}))
    }

    async fn register_worker(&self, request: Request<WorkerInfo>) -> Result<Response<Ack>, Status> {
        let worker_info: WorkerInfo = request.into_inner();
        tracing::info!("[Worker {}] register_worker called: address={}", self.worker_id, worker_info.address);

        match WorkerNodeClient::connect(format!("http://{}", worker_info.address)).await {
            Ok(client) => {
                self.workers.insert(worker_info.address.clone(), client);
                tracing::info!("[Worker {}] Successfully registered worker at {}", self.worker_id, worker_info.address);
                Ok(Response::new(Ack {}))
            }
            Err(e) => {
                tracing::error!("[Worker {}] Failed to register worker at {}: {}", self.worker_id, worker_info.address, e);
                Err(Status::internal(format!("Failed to connect to worker: {}", e)))
            }
        }
    }

    async fn route_message(&self, request: Request<RouteMessageRequest>) -> Result<Response<Ack>, Status> {
        let route_message_request: RouteMessageRequest = request.into_inner();
        let dest_uuid = Uuid::from_slice(&route_message_request.destination_uuid);

        let message: ProteanMessage<S> = match route_message_request.message {
            Some(proto) => ProteanMessage::try_from(proto).map_err(|e| {
                tracing::warn!("[Worker {}] route_message failed: Invalid message for peer {}: {:?}", self.worker_id, dest_uuid, e);
                Status::invalid_argument(format!("Invalid message: {:?}", e))
            })?,
            None => {
                tracing::warn!("[Worker {}] route_message failed: Missing message for peer {}", self.worker_id, dest_uuid);
                return Err(Status::invalid_argument("Missing message"));
            }
        };

        tracing::debug!("[Worker {}] Routing message to peer {}", self.worker_id, dest_uuid);
        match self.actor_protocol_channels.get(&dest_uuid) {
            Some(ch) => {
                ch.send(message).map_err(|e| {
                    tracing::error!("[Worker {}] route_message failed: Failed to send message to peer {}: {:?}", self.worker_id, dest_uuid, e);
                    Status::internal("Failed to send message")
                })?;
                tracing::debug!("[Worker {}] Successfully routed message to peer {}", self.worker_id, dest_uuid);
                Ok(Response::new(Ack {}))
            }
            None => {
                tracing::warn!("[Worker {}] route_message failed: Peer {} not found", self.worker_id, dest_uuid);
                Err(Status::not_found(format!("Peer {} not found", dest_uuid)))
            }
        }
    }
}