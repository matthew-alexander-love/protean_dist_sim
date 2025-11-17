//! Distributed Protean Simulation Library

pub mod proto {
    // Include generated protobuf code for dist_sim
    pub mod dist_sim {
        tonic::include_proto!("protean_proto.sim");
    }

    // Re-export protean proto types from the protean library
    pub use protean::proto as protean;
}

pub mod worker;
