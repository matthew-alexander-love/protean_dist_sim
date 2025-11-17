fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile dist_sim proto, mapping imported protean types to use the protean library
    // Note: .protean_proto.sim is our own package, so don't map it
    //       .protean_proto (without .sim) refers to the imported protean.proto file
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        // Map the imported protean_proto package to use types from protean library
        .extern_path(".protean_proto.SnvConfigProto", "::protean::proto::SnvConfigProto")
        .extern_path(".protean_proto.PeerProto", "::protean::proto::PeerProto")
        .extern_path(".protean_proto.QueryConfigProto", "::protean::proto::QueryConfigProto")
        .extern_path(".protean_proto.QueryCandidateProto", "::protean::proto::QueryCandidateProto")
        .extern_path(".protean_proto.SparseNeighborViewProto", "::protean::proto::SparseNeighborViewProto")
        .extern_path(".protean_proto.ProteanMessageProto", "::protean::proto::ProteanMessageProto")
        .extern_path(".onnx.TensorProto", "::protean::proto::onnx::TensorProto")
        .compile_protos(
            &[
                "src/proto/dist_sim.proto",
            ],
            &["src/proto/"]
        )?;

    Ok(())
}
