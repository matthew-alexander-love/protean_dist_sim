use std::sync::Arc;

use tonic::{Request, Response, Status};

use protean::embedding_space::{Embedding, EmbeddingSpace};

use crate::proto::dist_sim::coordinator_server::Coordinator as CoordinatorTrait;
use crate::proto::dist_sim::worker_node_client::WorkerNodeClient;
use crate::proto::dist_sim::*;

use super::coordinator::Coordinator;

/// gRPC service implementation for Coordinator
pub struct CoordinatorService<S: EmbeddingSpace> {
    coordinator: Arc<Coordinator<S>>,
}

impl<S: EmbeddingSpace> CoordinatorService<S> {
    pub fn new(coordinator: Arc<Coordinator<S>>) -> Self {
        Self { coordinator }
    }
}

#[tonic::async_trait]
impl<S: EmbeddingSpace + Send + Sync + 'static> CoordinatorTrait for CoordinatorService<S>
where
    S::EmbeddingData: Embedding<Scalar = f32>,
{
    /// Worker registers with coordinator
    async fn register_worker(
        &self,
        request: Request<WorkerInfo>,
    ) -> Result<Response<WorkerIdResponse>, Status> {
        let req = request.into_inner();

        // Create gRPC client to worker
        let worker_address = format!("http://{}", req.address);
        let client = WorkerNodeClient::connect(worker_address)
            .await
            .map_err(|e| Status::internal(format!("Failed to connect to worker: {}", e)))?
            .max_decoding_message_size(100 * 1024 * 1024) // 100MB
            .max_encoding_message_size(100 * 1024 * 1024); // 100MB

        // Determine worker ID from address by looking up in config
        let worker_id = self.coordinator
            .worker_configs()
            .iter()
            .find(|w| w.address == req.address)
            .map(|w| w.worker_id.clone())
            .ok_or_else(|| Status::invalid_argument(format!("Unknown worker address: {}", req.address)))?;

        // Store in registry
        self.coordinator.workers().insert(worker_id.clone(), client);

        tracing::info!("Worker registered: {}", worker_id);

        // Get coordinator address from config
        let coordinator_address = self.coordinator.coordinator_address().to_string();

        Ok(Response::new(WorkerIdResponse {
            worker_id,
            success: true,
            message: "Worker registered successfully".to_string(),
            coordinator_address,
        }))
    }

    /// Periodic heartbeat from worker
    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();

        // Update last-seen timestamp (TODO: implement tracking)
        // For now, just acknowledge

        tracing::debug!(
            "Heartbeat from worker {}: {} active peers",
            req.worker_id,
            req.active_peers
        );

        Ok(Response::new(Ack {
            success: true,
            message: "Heartbeat acknowledged".to_string(),
        }))
    }

    /// Worker unregisters (graceful shutdown)
    async fn unregister_worker(
        &self,
        request: Request<WorkerIdRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();

        // Remove from registry
        self.coordinator.workers().remove(&req.worker_id);

        tracing::info!("Worker unregistered: {}", req.worker_id);

        Ok(Response::new(Ack {
            success: true,
            message: "Worker unregistered successfully".to_string(),
        }))
    }

    /// Worker forwards an event to coordinator
    async fn forward_event(
        &self,
        request: Request<ProteanEventProto>,
    ) -> Result<Response<Ack>, Status> {
        let event = request.into_inner();

        // Track BootstrapCompleted events for dynamic bootstrap flow control
        if event.event_type == ProteanEventType::BootstrapCompleted as i32 {
            // Access the event via the oneof field
            if let Some(protean_event_proto::Event::BootstrapCompleted(ref bootstrap_event)) = event.event {
                // Extract peer UUID and convert to global index
                let peer_idx = Self::uuid_bytes_to_global_index(&bootstrap_event.peer_uuid);

                // Handle bootstrap completion: remove from current_bootstrapping and trigger next
                self.coordinator.on_bootstrap_complete(peer_idx).await;

                tracing::debug!("Peer {} bootstrapped. Active peers: {}", peer_idx, self.coordinator.active_peer_count());
            } else {
                tracing::warn!("BootstrapCompleted event missing bootstrap_completed data");
            }
        }
        // When explore_k=0, BootstrapConvergingCompleted is the final event, treat it as completion
        else if event.event_type == ProteanEventType::BootstrapConvergingCompleted as i32 {
            // Check if explore is disabled
            let explore_k = self.coordinator.config().sim_config.snv_config.exploration_config.explore_k;
            if explore_k == 0 {
                if let Some(protean_event_proto::Event::BootstrapConvergingCompleted(ref bootstrap_event)) = event.event {
                    let peer_idx = Self::uuid_bytes_to_global_index(&bootstrap_event.peer_uuid);
                    self.coordinator.on_bootstrap_complete(peer_idx).await;
                    tracing::debug!("Peer {} converged (exploration disabled). Active peers: {}", peer_idx, self.coordinator.active_peer_count());
                }
            }
        }

        // Store event for later analysis
        self.coordinator.events().lock().await.push(event);

        Ok(Response::new(Ack {
            success: true,
            message: "Event received".to_string(),
        }))
    }
}

impl<S: EmbeddingSpace> CoordinatorService<S> {
    /// Convert UUID bytes back to global dataset index
    /// The UUID is constructed from the global index in the first 8 bytes
    fn uuid_bytes_to_global_index(uuid_bytes: &[u8]) -> u64 {
        if uuid_bytes.len() >= 8 {
            u64::from_be_bytes(uuid_bytes[0..8].try_into().unwrap())
        } else {
            tracing::error!("Invalid UUID bytes length: {}", uuid_bytes.len());
            0
        }
    }
}
