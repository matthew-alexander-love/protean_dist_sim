//! Tests for ActorProtean

use super::helpers::*;
use crate::worker::actor::{ControlMessage, ActorProtean};
use protean::embedding_space::spaces::f32_l2::F32L2Space;
use protean::ProteanPeer;
use protean::uuid::Uuid;
use tokio::sync::oneshot;

type TestSpace = F32L2Space<4>;

#[cfg(test)]
mod control_message_tests {
    use super::*;

    // NOTE: These are integration-style tests that would require
    // a fully running ActorProtean instance. For now, these are
    // structural tests to verify the ControlMessage API.

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
    fn test_control_message_load_snv_snapshot() {
        let (tx, _rx) = oneshot::channel();

        let msg = ControlMessage::<TestSpace>::LoadSnvSnapshot {
            snapshot: Default::default(),
            response: tx,
        };

        match msg {
            ControlMessage::LoadSnvSnapshot { .. } => {}
            _ => panic!("Wrong variant"),
        }
    }
}

// Note: Full ActorProtean integration tests would require:
// - Spawning actual actors
// - Sending control messages and verifying responses
// - Checking event emissions
// - Testing message routing
// These would be better suited for integration tests with full runtime
