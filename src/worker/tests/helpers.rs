//! Test helpers and utilities

use crate::proto::protean::TensorProto;
use crate::worker::worker::Worker;
use protean::embedding_space::spaces::f32_l2::F32L2Space;

/// Test embedding space type
pub type TestSpace = F32L2Space<4>;

/// Create a test worker instance
pub fn create_test_worker() -> Worker<TestSpace> {
    Worker::new(
        "test-worker-1".to_string(),
        "127.0.0.1:8080".to_string(),
        4,
        100,
    )
}

/// Create a test embedding tensor proto
pub fn create_test_embedding_proto(values: Vec<f32>) -> TensorProto {
    TensorProto {
        dims: vec![values.len() as i64],
        data_type: 1, // FLOAT
        float_data: values,
        ..Default::default()
    }
}

/// Create a specific embedding for testing
pub fn create_known_embedding(a: f32, b: f32, c: f32, d: f32) -> TensorProto {
    create_test_embedding_proto(vec![a, b, c, d])
}
