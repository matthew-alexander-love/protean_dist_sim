//! Unit and RPC tests for Worker

use super::helpers::*;
use crate::proto::dist_sim::*;
use crate::worker::worker::Worker;
use protean::embedding_space::spaces::f32_l2::F32L2Space;
use tonic::Request;

type TestSpace = F32L2Space<4>;

// ============================================================================
// Helper Method Tests
// ============================================================================

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn test_parse_uuid_valid() {
        let uuid = Uuid::new_v4();
        let bytes = uuid.as_bytes();
        let parsed = Worker::<TestSpace>::parse_uuid(bytes).unwrap();
        assert_eq!(parsed, uuid);
    }

    #[test]
    fn test_parse_uuid_invalid_length() {
        let bytes = [0u8; 8]; // Wrong length
        let result = Worker::<TestSpace>::parse_uuid(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_embedding_valid() {
        let tensor = create_test_embedding_proto(vec![1.0, 2.0, 3.0, 4.0]);
        let result = Worker::<TestSpace>::parse_embedding(Some(tensor));
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_embedding_missing() {
        let result = Worker::<TestSpace>::parse_embedding(None);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_embedding_wrong_dimension() {
        let tensor = create_test_embedding_proto(vec![1.0, 2.0]); // Wrong dim
        let result = Worker::<TestSpace>::parse_embedding(Some(tensor));
        assert!(result.is_err());
    }
}

// ============================================================================
// RPC Tests - Simple Endpoints
// ============================================================================

#[cfg(test)]
mod rpc_simple_tests {
    use super::*;

    #[tokio::test]
    async fn test_ping_rpc() {
        let worker = create_test_worker();
        let ack = worker.handle_ping();

        assert!(ack.success);
        assert_eq!(ack.message, "pong");
    }

    #[tokio::test]
    async fn test_register_worker_rpc() {
        let worker = create_test_worker();
        let ack = worker.handle_register_worker("127.0.0.1:8081".to_string()).await;

        assert!(ack.success);
        assert!(ack.message.contains("Registered"));
    }

    #[tokio::test]
    async fn test_set_config_rpc() {
        let worker = create_test_worker();

        // Create a minimal config
        let config = SimConfigProto {
            snv_config: Some(Default::default()),
            max_peers: 100,
            region: "us-east-1".to_string(),
        };

        let request = Request::new(config);
        let response = worker.set_config(request).await.unwrap();
        let ack = response.into_inner();

        assert!(ack.success);
    }

    #[tokio::test]
    async fn test_churn_rpc_not_implemented() {
        let worker = create_test_worker();
        let request = Request::new(ChurnPatternRequest::default());

        let response = worker.churn(request).await.unwrap();
        let ack = response.into_inner();

        assert!(!ack.success);
        assert!(ack.message.contains("not yet implemented"));
    }
}

// ============================================================================
// RPC Tests - Peer Management
// ============================================================================

#[cfg(test)]
mod rpc_peer_management_tests {
    use super::*;

    #[tokio::test]
    async fn test_create_peers_rpc() {
        let worker = create_test_worker();

        // Load embeddings first (required for index-based peer creation)
        let embeddings = vec![
            create_test_embedding_proto(vec![1.0, 0.0, 0.0, 0.0]),
            create_test_embedding_proto(vec![0.0, 1.0, 0.0, 0.0]),
        ];
        let load_req = Request::new(LoadEmbeddingsRequest {
            global_offset: 100,
            embeddings,
        });
        worker.load_embeddings(load_req).await.unwrap();

        // Create peers using global indices
        let request = Request::new(CreatePeersRequest {
            global_indices: vec![100, 101],
            bootstrap_index: 0, // No bootstrap
        });

        let response = worker.create_peers(request).await.unwrap();
        let ack = response.into_inner();

        assert!(ack.success);
        assert!(ack.message.contains("2"));

        // Verify peers were created
        assert_eq!(worker.local_peer_uuids().len(), 2);
    }

    #[tokio::test]
    async fn test_delete_peers_rpc() {
        let worker = create_test_worker();

        // Load embeddings first
        let embeddings = vec![create_test_embedding_proto(vec![1.0, 0.0, 0.0, 0.0])];
        let load_req = Request::new(LoadEmbeddingsRequest {
            global_offset: 100,
            embeddings,
        });
        worker.load_embeddings(load_req).await.unwrap();

        // Create a peer using global index
        let create_req = Request::new(CreatePeersRequest {
            global_indices: vec![100],
            bootstrap_index: 0,
        });
        worker.create_peers(create_req).await.unwrap();

        // Verify peer was created
        assert_eq!(worker.local_peer_uuids().len(), 1);

        // Now delete it using global index
        let delete_req = Request::new(DeletePeersRequest {
            global_indices: vec![100],
        });

        let response = worker.delete_peers(delete_req).await.unwrap();
        let ack = response.into_inner();

        assert!(ack.success);
        assert!(ack.message.contains("1"));

        // Verify peer was deleted
        assert_eq!(worker.local_peer_uuids().len(), 0);
    }
}

// ============================================================================
// RPC Tests - Queries
// ============================================================================

#[cfg(test)]
mod rpc_query_tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_query_rpc() {
        let worker = create_test_worker();

        // Load embeddings
        let embeddings = vec![create_test_embedding_proto(vec![1.0, 0.0, 0.0, 0.0])];
        let load_req = Request::new(LoadEmbeddingsRequest {
            global_offset: 100,
            embeddings,
        });
        worker.load_embeddings(load_req).await.unwrap();

        // Create a peer using global index
        let create_req = Request::new(CreatePeersRequest {
            global_indices: vec![100],
            bootstrap_index: 0,
        });
        worker.create_peers(create_req).await.unwrap();

        // Get the peer UUID (generated from global index)
        let peer_uuid = Worker::<TestSpace>::global_index_to_uuid(100);

        // Execute query
        let query_req = Request::new(QueryRequest {
            source_peer_uuid: peer_uuid.as_bytes().to_vec(),
            query_embedding: Some(create_known_embedding(0.5, 0.5, 0.5, 0.5)),
            k: 5,
            config: None,
        });

        let response = worker.execute_query(query_req).await.unwrap();
        let query_response = response.into_inner();

        assert!(query_response.success);
        assert!(!query_response.query_uuid.is_empty());
    }

    #[tokio::test]
    async fn test_true_query_rpc() {
        let worker = create_test_worker();

        // Load embeddings for multiple peers
        let embeddings = vec![
            create_test_embedding_proto(vec![1.0, 0.0, 0.0, 0.0]),
            create_test_embedding_proto(vec![0.0, 1.0, 0.0, 0.0]),
            create_test_embedding_proto(vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let load_req = Request::new(LoadEmbeddingsRequest {
            global_offset: 100,
            embeddings,
        });
        worker.load_embeddings(load_req).await.unwrap();

        // Create multiple peers using global indices
        let create_req = Request::new(CreatePeersRequest {
            global_indices: vec![100, 101, 102],
            bootstrap_index: 0,
        });
        worker.create_peers(create_req).await.unwrap();

        // Execute true query
        let query_req = Request::new(QueryRequest {
            source_peer_uuid: vec![], // Not used in true query
            query_embedding: Some(create_known_embedding(1.0, 0.1, 0.1, 0.1)),
            k: 2,
            config: None,
        });

        let response = worker.true_query(query_req).await.unwrap();
        let query_response = response.into_inner();

        assert!(query_response.success);
        assert_eq!(query_response.results.len(), 2);
    }

    #[tokio::test]
    async fn test_execute_query_peer_not_found() {
        let worker = create_test_worker();

        let query_req = Request::new(QueryRequest {
            source_peer_uuid: Uuid::new_v4().as_bytes().to_vec(),
            query_embedding: Some(create_known_embedding(0.5, 0.5, 0.5, 0.5)),
            k: 5,
            config: None,
        });

        let result = worker.execute_query(query_req).await;
        assert!(result.is_err());
    }
}

// ============================================================================
// RPC Tests - Snapshots
// ============================================================================

#[cfg(test)]
mod rpc_snapshot_tests {
    use super::*;

    #[tokio::test]
    async fn test_get_snapshot_rpc() {
        let worker = create_test_worker();

        // Load embeddings and create a peer
        let embeddings = vec![create_test_embedding_proto(vec![1.0, 0.0, 0.0, 0.0])];
        let load_req = Request::new(LoadEmbeddingsRequest {
            global_offset: 100,
            embeddings,
        });
        worker.load_embeddings(load_req).await.unwrap();

        let create_req = Request::new(CreatePeersRequest {
            global_indices: vec![100],
            bootstrap_index: 0,
        });
        worker.create_peers(create_req).await.unwrap();

        // Get snapshot
        let snapshot_req = Request::new(SnapshotRequest {
            peer_uuids: vec![],
        });

        let response = worker.get_snapshot(snapshot_req).await.unwrap();
        let snapshot = response.into_inner();

        assert_eq!(snapshot.worker_id, "test-worker-1");
        assert!(snapshot.timestamp > 0);
    }

    #[tokio::test]
    async fn test_load_snapshot_rpc() {
        let worker = create_test_worker();

        // Create empty snapshot
        let snapshot = NetworkSnapshot {
            timestamp: 123456,
            worker_id: "test-worker-1".to_string(),
            snapshots: vec![],
            ..Default::default()
        };

        let request = Request::new(snapshot);
        let response = worker.load_snapshot(request).await.unwrap();
        let ack = response.into_inner();

        assert!(ack.success);
    }
}

// ============================================================================
// RPC Tests - Message Routing
// ============================================================================

#[cfg(test)]
mod rpc_routing_tests {
    use super::*;

    #[tokio::test]
    async fn test_route_message_peer_not_found() {
        let worker = create_test_worker();

        let route_req = Request::new(RouteMessageRequest {
            destination_uuid: Uuid::new_v4().as_bytes().to_vec(),
            message: Some(vec![1, 2, 3]),
        });

        let result = worker.route_message(route_req).await;
        // Should fail because peer doesn't exist
        assert!(result.is_err());
    }
}
