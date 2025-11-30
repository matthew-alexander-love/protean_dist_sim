//! Unit tests for Worker
//!
//! Tests the Worker's internal helper functions and UUID mapping.
//!
//! Note: RPC endpoint tests that call gRPC methods are disabled because
//! the Worker contains its own tokio runtime which conflicts with the
//! test framework's async runtime. Testing gRPC endpoints requires
//! full integration tests with actual network connections.

use crate::worker::worker::Worker;
use protean::embedding_space::spaces::f32_l2::F32L2Space;

type TestSpace = F32L2Space<4>;

// ============================================================================
// Helper Method Tests
// ============================================================================

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn test_global_index_to_uuid_deterministic() {
        let uuid1 = Worker::<TestSpace>::global_index_to_uuid(12345);
        let uuid2 = Worker::<TestSpace>::global_index_to_uuid(12345);
        assert_eq!(uuid1, uuid2);
    }

    #[test]
    fn test_global_index_to_uuid_different_indices() {
        let uuid1 = Worker::<TestSpace>::global_index_to_uuid(100);
        let uuid2 = Worker::<TestSpace>::global_index_to_uuid(101);
        assert_ne!(uuid1, uuid2);
    }

    #[test]
    fn test_uuid_roundtrip() {
        let original_index = 12345u64;
        let uuid = Worker::<TestSpace>::global_index_to_uuid(original_index);
        let recovered = Worker::<TestSpace>::uuid_to_global_index(&uuid);
        assert_eq!(original_index, recovered);
    }

    #[test]
    fn test_uuid_roundtrip_zero() {
        let original_index = 0u64;
        let uuid = Worker::<TestSpace>::global_index_to_uuid(original_index);
        let recovered = Worker::<TestSpace>::uuid_to_global_index(&uuid);
        assert_eq!(original_index, recovered);
    }

    #[test]
    fn test_uuid_roundtrip_max() {
        let original_index = u64::MAX;
        let uuid = Worker::<TestSpace>::global_index_to_uuid(original_index);
        let recovered = Worker::<TestSpace>::uuid_to_global_index(&uuid);
        assert_eq!(original_index, recovered);
    }

    #[test]
    fn test_uuid_uniqueness() {
        // Test that sequential indices produce unique UUIDs
        let uuids: Vec<_> = (0..100u64)
            .map(|i| Worker::<TestSpace>::global_index_to_uuid(i))
            .collect();

        for (i, uuid1) in uuids.iter().enumerate() {
            for (j, uuid2) in uuids.iter().enumerate() {
                if i != j {
                    assert_ne!(uuid1, uuid2, "UUIDs for indices {} and {} should differ", i, j);
                }
            }
        }
    }
}

// Note: Full RPC endpoint tests (ping, set_config, load_embeddings, etc.)
// require integration testing with actual gRPC connections. The Worker
// creates its own tokio runtime internally, which conflicts with
// #[tokio::test] async test contexts.
//
// RPC functionality is covered by the coordinator integration tests which
// set up full distributed scenarios with actual network communication.
