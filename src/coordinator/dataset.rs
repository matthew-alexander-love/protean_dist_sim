use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use protean::embedding_space::{Embedding, EmbeddingSpace};
use protean::proto::TensorProto;

/// Read .fvecs file format
///
/// .fvecs format: [dim:i32][values:f32*dim] repeated for each vector
/// All values are little-endian
pub fn read_fvecs<P: AsRef<Path>>(filepath: P) -> io::Result<Vec<Vec<f32>>> {
    let mut file = File::open(filepath)?;
    let mut vectors = Vec::new();

    loop {
        // Read dimension (4 bytes, little-endian i32)
        let mut dim_bytes = [0u8; 4];
        match file.read_exact(&mut dim_bytes) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let dim = i32::from_le_bytes(dim_bytes) as usize;

        // Read vector data (dim * 4 bytes, little-endian f32s)
        let mut buffer = vec![0u8; dim * 4];
        file.read_exact(&mut buffer)?;

        let vector: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        vectors.push(vector);

        // Optional: check if we've hit max_vectors limit (for testing)
        // This can be added as a parameter if needed
    }

    Ok(vectors)
}

/// Convert Vec<Vec<f32>> to Vec<S::EmbeddingData>
///
/// Uses the EmbeddingSpace's create_embedding method to convert raw f32 vectors
/// into the appropriate embedding type for the space.
pub fn convert_to_embedding_data<S: EmbeddingSpace>(
    vectors: Vec<Vec<f32>>,
) -> Vec<S::EmbeddingData>
where
    S::EmbeddingData: Embedding<Scalar = f32>,
{
    vectors
        .into_iter()
        .map(|v| S::create_embedding(v))
        .collect()
}

/// Partition embeddings across workers using round-robin distribution
/// This ensures equal distribution regardless of access patterns
///
/// Example with 10 embeddings and 3 workers:
/// - Worker 0: [0, 3, 6, 9] (global indices)
/// - Worker 1: [1, 4, 7]
/// - Worker 2: [2, 5, 8]
///
/// Returns: Vec<(global_offset, embeddings)> where global_offset is the worker index
pub fn partition_embeddings<T>(embeddings: Vec<T>, num_workers: usize) -> Vec<(u64, Vec<T>)> {
    if num_workers == 0 {
        return vec![];
    }

    // Initialize partitions for each worker
    let mut partitions: Vec<Vec<T>> = (0..num_workers).map(|_| Vec::new()).collect();

    // Round-robin distribution
    for (idx, embedding) in embeddings.into_iter().enumerate() {
        let worker_idx = idx % num_workers;
        partitions[worker_idx].push(embedding);
    }

    // Convert to (offset, slice) format
    // Each worker's first embedding has global index = worker_idx
    partitions
        .into_iter()
        .enumerate()
        .map(|(worker_idx, slice)| (worker_idx as u64, slice))
        .collect()
}

/// Convert S::EmbeddingData to TensorProto for gRPC transmission
pub fn embedding_to_tensor_proto<S: EmbeddingSpace>(
    embedding: &S::EmbeddingData,
) -> TensorProto {
    // Use the Into<TensorProto> conversion provided by the Embedding trait
    embedding.clone().into()
}

/// Convert Vec<S::EmbeddingData> to Vec<TensorProto> for batch transmission
pub fn embeddings_to_tensor_protos<S: EmbeddingSpace>(
    embeddings: &[S::EmbeddingData],
) -> Vec<TensorProto> {
    embeddings
        .iter()
        .map(|e| embedding_to_tensor_proto::<S>(e))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use protean::embedding_space::spaces::f32_l2::F32L2Space;

    fn create_test_fvecs_file(path: &str, vectors: &[Vec<f32>]) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        for vector in vectors {
            // Write dimension as little-endian i32
            let dim = vector.len() as i32;
            file.write_all(&dim.to_le_bytes())?;

            // Write vector data as little-endian f32s
            for &value in vector {
                file.write_all(&value.to_le_bytes())?;
            }
        }

        Ok(())
    }

    #[test]
    fn test_read_fvecs_single_vector() {
        let temp_path = "/tmp/test_single.fvecs";
        let test_data = vec![vec![1.0, 2.0, 3.0, 4.0]];

        create_test_fvecs_file(temp_path, &test_data).unwrap();

        let result = read_fvecs(temp_path).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 4);
        assert!((result[0][0] - 1.0).abs() < 0.001);
        assert!((result[0][1] - 2.0).abs() < 0.001);
        assert!((result[0][2] - 3.0).abs() < 0.001);
        assert!((result[0][3] - 4.0).abs() < 0.001);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_read_fvecs_multiple_vectors() {
        let temp_path = "/tmp/test_multiple.fvecs";
        let test_data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        create_test_fvecs_file(temp_path, &test_data).unwrap();

        let result = read_fvecs(temp_path).unwrap();

        assert_eq!(result.len(), 3);
        for i in 0..3 {
            assert_eq!(result[i].len(), 3);
            for j in 0..3 {
                let expected = (i * 3 + j + 1) as f32;
                assert!((result[i][j] - expected).abs() < 0.001);
            }
        }

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_read_fvecs_different_dimensions() {
        let temp_path = "/tmp/test_diff_dims.fvecs";
        let test_data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0, 5.0, 6.0],
            vec![7.0],
        ];

        create_test_fvecs_file(temp_path, &test_data).unwrap();

        let result = read_fvecs(temp_path).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].len(), 2);
        assert_eq!(result[1].len(), 4);
        assert_eq!(result[2].len(), 1);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_read_fvecs_empty_file() {
        let temp_path = "/tmp/test_empty.fvecs";
        std::fs::File::create(temp_path).unwrap();

        let result = read_fvecs(temp_path).unwrap();
        assert_eq!(result.len(), 0);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_convert_to_embedding_data() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        // Test with 4D F32L2Space
        let embeddings = convert_to_embedding_data::<F32L2Space<4>>(vectors);

        assert_eq!(embeddings.len(), 2);

        // Verify the data is correctly converted
        use protean::embedding_space::Embedding;
        assert_eq!(embeddings[0].as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(embeddings[1].as_slice(), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_embedding_to_tensor_proto() {
        use protean::embedding_space::EmbeddingSpace;

        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let embeddings = convert_to_embedding_data::<F32L2Space<4>>(vectors);

        let tensor_proto = embedding_to_tensor_proto::<F32L2Space<4>>(&embeddings[0]);

        assert_eq!(tensor_proto.data_type, 1); // FLOAT type
        assert_eq!(tensor_proto.float_data.len(), 4);
        assert_eq!(tensor_proto.float_data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor_proto.dims, vec![4]);
    }

    #[test]
    fn test_embeddings_to_tensor_protos_batch() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let embeddings = convert_to_embedding_data::<F32L2Space<3>>(vectors);
        let tensor_protos = embeddings_to_tensor_protos::<F32L2Space<3>>(&embeddings);

        assert_eq!(tensor_protos.len(), 3);
        for (i, proto) in tensor_protos.iter().enumerate() {
            assert_eq!(proto.data_type, 1);
            assert_eq!(proto.float_data.len(), 3);
            assert_eq!(proto.dims, vec![3]);

            // Verify values
            let base = (i * 3 + 1) as f32;
            assert_eq!(proto.float_data[0], base);
            assert_eq!(proto.float_data[1], base + 1.0);
            assert_eq!(proto.float_data[2], base + 2.0);
        }
    }

    #[test]
    fn test_partition_embeddings() {
        let data: Vec<u32> = (0..10).collect();

        // 3 workers: should be [0,1,2,3], [4,5,6], [7,8,9]
        let partitions = partition_embeddings(data, 3);
        assert_eq!(partitions.len(), 3);
        assert_eq!(partitions[0].0, 0);
        assert_eq!(partitions[0].1.len(), 4); // Gets remainder
        assert_eq!(partitions[1].0, 4);
        assert_eq!(partitions[1].1.len(), 3);
        assert_eq!(partitions[2].0, 7);
        assert_eq!(partitions[2].1.len(), 3);
    }

    #[test]
    fn test_partition_embeddings_even() {
        let data: Vec<u32> = (0..9).collect();

        // 3 workers: should be [0,1,2], [3,4,5], [6,7,8]
        let partitions = partition_embeddings(data, 3);
        assert_eq!(partitions.len(), 3);
        for partition in partitions {
            assert_eq!(partition.1.len(), 3);
        }
    }

    #[test]
    fn test_partition_embeddings_empty() {
        let data: Vec<u32> = vec![];
        let partitions = partition_embeddings(data, 3);
        assert_eq!(partitions.len(), 3);
        for partition in partitions {
            assert_eq!(partition.1.len(), 0);
        }
    }

    #[test]
    fn test_partition_embeddings_single_worker() {
        let data: Vec<u32> = (0..10).collect();
        let partitions = partition_embeddings(data, 1);

        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].0, 0);
        assert_eq!(partitions[0].1.len(), 10);
    }

    #[test]
    fn test_partition_embeddings_offsets() {
        let data: Vec<u32> = (0..100).collect();
        let partitions = partition_embeddings(data, 5);

        // Verify offsets are cumulative
        assert_eq!(partitions[0].0, 0);
        assert_eq!(partitions[1].0, 20);
        assert_eq!(partitions[2].0, 40);
        assert_eq!(partitions[3].0, 60);
        assert_eq!(partitions[4].0, 80);

        // Verify all have size 20
        for partition in partitions {
            assert_eq!(partition.1.len(), 20);
        }
    }

    #[test]
    fn test_full_pipeline_fvecs_to_tensor_protos() {
        // Test the complete pipeline: .fvecs -> Vec<Vec<f32>> -> EmbeddingData -> TensorProto
        let temp_path = "/tmp/test_pipeline.fvecs";
        let test_data = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        create_test_fvecs_file(temp_path, &test_data).unwrap();

        // Read .fvecs
        let vectors = read_fvecs(temp_path).unwrap();

        // Convert to embeddings
        let embeddings = convert_to_embedding_data::<F32L2Space<4>>(vectors);

        // Convert to TensorProtos
        let protos = embeddings_to_tensor_protos::<F32L2Space<4>>(&embeddings);

        assert_eq!(protos.len(), 2);
        assert_eq!(protos[0].float_data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(protos[1].float_data, vec![5.0, 6.0, 7.0, 8.0]);

        std::fs::remove_file(temp_path).ok();
    }
}
