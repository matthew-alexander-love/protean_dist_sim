//! Unit and RPC tests for Worker

use super::helpers::*;
use crate::proto::dist_sim::*;
use crate::worker::worker::Worker;
use protean::embedding_space::spaces::f32_l2::F32L2Space;
use protean::uuid::Uuid;

type TestSpace = F32L2Space<4>;

// ============================================================================
// Helper Method Tests
// ============================================================================

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn test_parse_uuid_valid() {
        let uuid = Uuid::from_data(12345u64);
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
        // Try to register a non-existent worker - should fail
        let ack = worker.handle_register_worker("127.0.0.1:8081".to_string()).await;

        // Connection should fail since no worker is actually running at that address
        assert!(!ack.success);
        assert!(ack.message.contains("Failed"));
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

        let ack = worker.handle_set_config(config).unwrap();

        assert!(ack.success);
    }

    #[tokio::test]
    async fn test_churn_rpc() {
        let worker = create_test_worker();
        let request = ChurnPatternRequest::default();

        let ack = worker.handle_churn(request).unwrap();

        assert!(ack.success);
        assert!(ack.message.contains("started"));
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
        worker.handle_load_embeddings(100, embeddings).unwrap();

        // Create peers using global indices
        let ack = worker.handle_create_peers(vec![100, 101], None).unwrap();

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
        worker.handle_load_embeddings(100, embeddings).unwrap();

        // Create a peer using global index
        worker.handle_create_peers(vec![100], None).unwrap();

        // Verify peer was created
        assert_eq!(worker.local_peer_uuids().len(), 1);

        // Now delete it using global index
        let ack = worker.handle_delete_peers(vec![100]).await.unwrap();

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
        worker.handle_load_embeddings(100, embeddings).unwrap();

        // Create a peer using global index
        worker.handle_create_peers(vec![100], None).unwrap();

        // Get the peer UUID (generated from global index)
        let peer_uuid = Worker::<TestSpace>::global_index_to_uuid(100);

        // Execute query
        let query_response = worker.handle_execute_query(
            peer_uuid.as_bytes(),
            Some(create_known_embedding(0.5, 0.5, 0.5, 0.5)),
            5,
            None,
        ).await.unwrap();

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
        worker.handle_load_embeddings(100, embeddings).unwrap();

        // Create multiple peers using global indices
        worker.handle_create_peers(vec![100, 101, 102], None).unwrap();

        // Execute true query
        let query_response = worker.handle_true_query(
            Some(create_known_embedding(1.0, 0.1, 0.1, 0.1)),
            2,
        ).await.unwrap();

        assert!(query_response.success);
        assert_eq!(query_response.results.len(), 2);
    }

    #[tokio::test]
    async fn test_execute_query_peer_not_found() {
        let worker = create_test_worker();

        let random_uuid = Uuid::from_data(99999u64);
        let result = worker.handle_execute_query(
            random_uuid.as_bytes(),
            Some(create_known_embedding(0.5, 0.5, 0.5, 0.5)),
            5,
            None,
        ).await;

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
        worker.handle_load_embeddings(100, embeddings).unwrap();

        worker.handle_create_peers(vec![100], None).unwrap();

        // Get snapshot
        let snapshot = worker.handle_get_snapshot(vec![]).await.unwrap();

        assert_eq!(snapshot.worker_id, "test-worker-1");
        assert!(snapshot.timestamp_ms > 0);
    }

    #[tokio::test]
    async fn test_load_snapshot_rpc() {
        let worker = create_test_worker();

        // Create empty snapshot
        let snapshot = NetworkSnapshot {
            timestamp_ms: 123456,
            worker_id: "test-worker-1".to_string(),
            peer_snapshots: vec![],
        };

        let ack = worker.handle_load_snapshot(snapshot).await.unwrap();

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
        use crate::proto::protean::ProteanMessageProto;

        let worker = create_test_worker();

        // Create a minimal valid message (content doesn't matter since peer doesn't exist)
        let message = ProteanMessageProto::default();
        let random_uuid = Uuid::from_data(99999u64);

        let result = worker.handle_route_message(
            random_uuid.as_bytes(),
            message,
        ).await;

        // Should fail because peer doesn't exist
        assert!(result.is_err());
    }
}
