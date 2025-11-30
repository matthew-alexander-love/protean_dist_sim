//! Tests for ActorProtean
//!
//! Tests the ControlMessage variants used for actor communication.

use super::helpers::*;
use crate::worker::actor::ControlMessage;
use protean::embedding_space::spaces::f32_l2::F32L2Space;
use protean::ProteanPeer;
use protean::uuid::Uuid;
use tokio::sync::oneshot;
use std::path::PathBuf;
use std::time::Duration;

type TestSpace = F32L2Space<4>;

#[cfg(test)]
mod control_message_tests {
    use super::*;

    // NOTE: These are structural tests that verify the ControlMessage API.
    // Full integration tests would require a running ActorProtean instance.

    #[test]
    fn test_control_message_bootstrap() {
        use protean::Peer;

        let peer = ProteanPeer {
            embedding: create_test_embedding_proto(vec![1.0, 0.0, 0.0, 0.0]).try_into().unwrap(),
            peer: Peer {
                uuid: Uuid::from_data("test-peer"),
                address: "test".to_string(),
            },
        };

        let msg = ControlMessage::<TestSpace>::Bootstrap {
            contact_point: peer,
            config: None,
        };

        // Verify we can construct the message
        match msg {
            ControlMessage::Bootstrap { .. } => {}
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_control_message_query() {
        let (tx, _rx) = oneshot::channel();

        let embedding = create_test_embedding_proto(vec![1.0, 0.0, 0.0, 0.0]).try_into().unwrap();

        let msg = ControlMessage::<TestSpace>::Query {
            embedding,
            k: 5,
            config: Default::default(),
            response: tx,
        };

        match msg {
            ControlMessage::Query { k, .. } => {
                assert_eq!(k, 5);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_control_message_shutdown() {
        let msg = ControlMessage::<TestSpace>::Shutdown;

        match msg {
            ControlMessage::Shutdown => {}
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_control_message_get_embedding() {
        let (tx, _rx) = oneshot::channel();

        let msg = ControlMessage::<TestSpace>::GetEmbedding {
            response: tx,
        };

        match msg {
            ControlMessage::GetEmbedding { .. } => {}
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_control_message_get_snv_snapshot() {
        let (tx, _rx) = oneshot::channel();

        let msg = ControlMessage::<TestSpace>::GetSnvSnapshot {
            response: tx,
        };

        match msg {
            ControlMessage::GetSnvSnapshot { .. } => {}
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_control_message_save() {
        let (tx, _rx) = oneshot::channel();

        let msg = ControlMessage::<TestSpace>::Save {
            save_path: PathBuf::from("/tmp/test_snapshot"),
            response: tx,
        };

        match msg {
            ControlMessage::Save { save_path, .. } => {
                assert_eq!(save_path, PathBuf::from("/tmp/test_snapshot"));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_control_message_get_query_status() {
        let (tx, _rx) = oneshot::channel();
        let query_uuid = Uuid::from_data("test-query");

        let msg = ControlMessage::<TestSpace>::GetQueryStatus {
            query_uuid: query_uuid.clone(),
            reply: tx,
        };

        match msg {
            ControlMessage::GetQueryStatus { query_uuid: uuid, .. } => {
                assert_eq!(uuid, query_uuid);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_control_message_start_drift() {
        let original = create_test_embedding_proto(vec![1.0, 0.0, 0.0, 0.0]).try_into().unwrap();
        let target = create_test_embedding_proto(vec![0.0, 1.0, 0.0, 0.0]).try_into().unwrap();

        let msg = ControlMessage::<TestSpace>::StartDrift {
            original_embedding: original,
            target_embedding: target,
            update_interval: Duration::from_millis(100),
            total_steps: 10,
        };

        match msg {
            ControlMessage::StartDrift { total_steps, .. } => {
                assert_eq!(total_steps, 10);
            }
            _ => panic!("Wrong variant"),
        }
    }
}

// Note: Full ActorProtean integration tests would require:
// - Spawning actual actors with coordinator connection
// - Sending control messages and verifying responses
// - Checking event emissions
// - Testing message routing between workers
// These are covered by the coordinator integration tests
