//! Test helpers and utilities

use crate::proto::protean::TensorProto;
use protean::embedding_space::spaces::f32_l2::F32L2Space;

/// Test embedding space type (4-dimensional F32 with L2 distance)
#[allow(dead_code)]
pub type TestSpace = F32L2Space<4>;

/// Create a test embedding tensor proto from f32 values
pub fn create_test_embedding_proto(values: Vec<f32>) -> TensorProto {
    TensorProto {
        dims: vec![values.len() as i64],
        data_type: 1, // FLOAT
        float_data: values,
        ..Default::default()
    }
}
