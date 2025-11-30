use std::iter;

use rand::Rng;

use protean::embedding_space::{Embedding, EmbeddingSpace};

use super::dataloader::DataSet;

#[derive(Debug, Clone)]
struct AssignmentPreference<S: EmbeddingSpace> {
    point_index: usize,
    cluster_index: usize,
    distance: S::DistanceValue,
}

#[derive(Debug, Clone)]
struct Cluster<S: EmbeddingSpace> {
    /// Calculated Centroid Embedding
    centroid: S::EmbeddingData,
    /// Index assignments from dataset
    assignments: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ClusteredData<S: EmbeddingSpace> {
    cluster_size_limit: usize,
    k: usize,
    dataset: DataSet<S>,
    clusters: Vec<Cluster<S>>,
}

impl<S: EmbeddingSpace> ClusteredData<S>
where
    S::EmbeddingData: Embedding<Scalar = f32>,
{
    pub fn new(cluster_size_limit: usize, k: usize, dataset: DataSet<S>) -> Self {
        let template_cluster = Cluster::<S> {
            centroid: S::zero_vector(),
            assignments: Vec::new(),
        };

        let clusters: Vec<Cluster<S>> = iter::repeat(template_cluster).take(k).collect();

        Self {
            cluster_size_limit,
            k,
            dataset,
            clusters,
        }
    }

    /// Compute the centroid (mean) of a set of embeddings
    fn compute_centroid(embeddings: &[&S::EmbeddingData]) -> S::EmbeddingData {
        if embeddings.is_empty() {
            return S::zero_vector();
        }

        let dim = S::length();
        let mut sum = vec![0.0f32; dim];

        for emb in embeddings {
            let slice = emb.as_slice();
            for (i, &val) in slice.iter().enumerate() {
                sum[i] += val;
            }
        }

        let count = embeddings.len() as f32;
        for val in &mut sum {
            *val /= count;
        }

        S::create_embedding(sum)
    }

    /// Find the index of the nearest cluster to a point
    fn find_nearest_cluster(&self, point: &S::EmbeddingData) -> usize {
        let mut nearest_idx = 0;
        let mut nearest_dist = S::distance(point, &self.clusters[0].centroid);

        for (idx, cluster) in self.clusters.iter().enumerate().skip(1) {
            let dist = S::distance(point, &cluster.centroid);
            if dist < nearest_dist {
                nearest_dist = dist;
                nearest_idx = idx;
            }
        }

        nearest_idx
    }

    /// Initialize centroids using k-means++ algorithm
    fn initialize_centroids_kmeanspp(&mut self) {
        let mut rng = rand::thread_rng();
        let n = self.dataset.train.len();

        if n == 0 || self.k == 0 {
            return;
        }

        // Pick first centroid uniformly at random
        let first_idx = rng.gen_range(0..n);
        self.clusters[0].centroid = self.dataset.train[first_idx].clone();

        // For each remaining centroid
        for c_idx in 1..self.k {
            // Compute D(x)² for each point (distance to nearest existing centroid)
            let mut distances_sq: Vec<f32> = Vec::with_capacity(n);
            let mut total_dist_sq = 0.0f32;

            for point in &self.dataset.train {
                let mut min_dist = S::distance(point, &self.clusters[0].centroid);
                for existing in self.clusters.iter().take(c_idx).skip(1) {
                    let dist = S::distance(point, &existing.centroid);
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                let dist_f32: f32 = min_dist.into();
                let dist_sq = dist_f32 * dist_f32;
                distances_sq.push(dist_sq);
                total_dist_sq += dist_sq;
            }

            // Select next centroid with probability proportional to D(x)²
            if total_dist_sq == 0.0 {
                // All points are at existing centroids, pick randomly
                let idx = rng.gen_range(0..n);
                self.clusters[c_idx].centroid = self.dataset.train[idx].clone();
            } else {
                let threshold = rng.gen::<f32>() * total_dist_sq;
                let mut cumulative = 0.0f32;
                let mut selected_idx = n - 1;

                for (idx, &dist_sq) in distances_sq.iter().enumerate() {
                    cumulative += dist_sq;
                    if cumulative >= threshold {
                        selected_idx = idx;
                        break;
                    }
                }

                self.clusters[c_idx].centroid = self.dataset.train[selected_idx].clone();
            }
        }
    }

    /// Update all centroids based on current assignments
    fn update_centroids(&mut self) {
        for cluster in &mut self.clusters {
            if cluster.assignments.is_empty() {
                continue;
            }

            let points: Vec<&S::EmbeddingData> = cluster
                .assignments
                .iter()
                .map(|&idx| &self.dataset.train[idx])
                .collect();

            cluster.centroid = Self::compute_centroid(&points);
        }
    }

    /// Check if centroids have converged (unchanged from previous iteration)
    fn centroids_unchanged(&self, old_centroids: &[S::EmbeddingData]) -> bool {
        for (cluster, old) in self.clusters.iter().zip(old_centroids.iter()) {
            if cluster.centroid != *old {
                return false;
            }
        }
        true
    }

    /// Apply bounded assignment after convergence
    /// Uses greedy assignment sorted by distance to enforce cluster_size_limit
    fn apply_bounded_assignment(&mut self) {
        let n = self.dataset.train.len();

        // Build preference list: all (point, cluster, distance) tuples
        let mut all_preferences: Vec<AssignmentPreference<S>> =
            Vec::with_capacity(self.k * n);

        for (p_idx, point) in self.dataset.train.iter().enumerate() {
            for (c_idx, cluster) in self.clusters.iter().enumerate() {
                let dist = S::distance(point, &cluster.centroid);
                all_preferences.push(AssignmentPreference {
                    point_index: p_idx,
                    cluster_index: c_idx,
                    distance: dist,
                });
            }
        }

        // Sort by distance ascending (NaN values are treated as equal)
        all_preferences.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Clear all assignments
        for cluster in &mut self.clusters {
            cluster.assignments.clear();
        }

        // Greedy assignment respecting cluster_size_limit
        let mut is_assigned = vec![false; n];
        let mut points_assigned_count = 0;

        for preference in all_preferences.iter() {
            let p_idx = preference.point_index;
            let c_idx = preference.cluster_index;

            if is_assigned[p_idx] {
                continue;
            }

            if self.clusters[c_idx].assignments.len() < self.cluster_size_limit {
                self.clusters[c_idx].assignments.push(p_idx);
                is_assigned[p_idx] = true;
                points_assigned_count += 1;
            }

            if points_assigned_count == n {
                break;
            }
        }
    }

    /// Fit the k-means model: k-means++ init, Lloyd's algorithm, then bounded assignment
    pub fn fit(&mut self) {
        self.initialize_centroids_kmeanspp();

        loop {
            for cluster in &mut self.clusters {
                cluster.assignments.clear();
            }

            for (p_idx, point) in self.dataset.train.iter().enumerate() {
                let nearest = self.find_nearest_cluster(point);
                self.clusters[nearest].assignments.push(p_idx);
            }

            let old_centroids: Vec<S::EmbeddingData> =
                self.clusters.iter().map(|c| c.centroid.clone()).collect();
            self.update_centroids();

            if self.centroids_unchanged(&old_centroids) {
                break;
            }
        }

        self.apply_bounded_assignment();
    }

    /// Get the k nearest centroids to a query embedding
    pub fn predict(&self, query: &S::EmbeddingData, k: usize) -> Vec<(usize, &S::EmbeddingData)> {
        let mut distances: Vec<(usize, S::DistanceValue)> = self
            .clusters
            .iter()
            .enumerate()
            .map(|(idx, cluster)| (idx, S::distance(query, &cluster.centroid)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        distances
            .into_iter()
            .take(k)
            .map(|(idx, _)| (idx, &self.clusters[idx].centroid))
            .collect()
    }

    /// Get all cluster centroids as (index, centroid) pairs
    pub fn labels(&self) -> Vec<(usize, &S::EmbeddingData)> {
        self.clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, &c.centroid))
            .collect()
    }

    /// Get an iterator over embeddings in a specific cluster
    pub fn cluster_embeddings(&self, cluster_idx: usize) -> Option<impl Iterator<Item = &S::EmbeddingData>> {
        if self.clusters.len() <= cluster_idx {
            return None;
        }

        Some(self.clusters[cluster_idx]
            .assignments
            .iter()
            .map(move |&idx| &self.dataset.train[idx]))
    }

    /// Get an iterator over (global_index, embedding) pairs in a specific cluster
    pub fn cluster_indexed_embeddings(&self, cluster_idx: usize) -> Option<impl Iterator<Item = (usize, &S::EmbeddingData)>> {
        if self.clusters.len() <= cluster_idx {
            return None;
        }

        Some(self.clusters[cluster_idx]
            .assignments
            .iter()
            .map(move |&idx| (idx, &self.dataset.train[idx])))
    }

    /// Get a reference to the test set (query data)
    pub fn test_set(&self) -> &[S::EmbeddingData] {
        &self.dataset.test
    }

    /// Get an iterator over the test set
    pub fn test_iter(&self) -> impl Iterator<Item = &S::EmbeddingData> {
        self.dataset.test.iter()
    }

    /// Get the cluster index that a point was assigned to
    pub fn get_assignment(&self, point_idx: usize) -> Option<usize> {
        for (cluster_idx, cluster) in self.clusters.iter().enumerate() {
            if cluster.assignments.contains(&point_idx) {
                return Some(cluster_idx);
            }
        }
        None
    }

    /// Get the number of clusters
    pub fn num_clusters(&self) -> usize {
        self.k
    }

    /// Get the train set
    pub fn train_set(&self) -> &[S::EmbeddingData] {
        &self.dataset.train
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protean::embedding_space::F32L2Space;

    type TestSpace = F32L2Space<4>;

    fn create_test_dataset(train: Vec<Vec<f32>>, test: Vec<Vec<f32>>) -> DataSet<TestSpace> {
        DataSet {
            train: train.into_iter().map(|v| TestSpace::create_embedding(v)).collect(),
            test: test.into_iter().map(|v| TestSpace::create_embedding(v)).collect(),
        }
    }

    #[test]
    fn test_clustered_data_new() {
        let dataset = create_test_dataset(
            vec![vec![1.0, 0.0, 0.0, 0.0], vec![2.0, 0.0, 0.0, 0.0]],
            vec![],
        );
        let clustered = ClusteredData::<TestSpace>::new(10, 2, dataset);

        assert_eq!(clustered.k, 2);
        assert_eq!(clustered.cluster_size_limit, 10);
        assert_eq!(clustered.num_clusters(), 2);
    }

    #[test]
    fn test_compute_centroid() {
        let emb1 = TestSpace::create_embedding(vec![0.0, 0.0, 0.0, 0.0]);
        let emb2 = TestSpace::create_embedding(vec![2.0, 4.0, 6.0, 8.0]);
        let embeddings: Vec<&_> = vec![&emb1, &emb2];

        let centroid = ClusteredData::<TestSpace>::compute_centroid(&embeddings);
        let slice = centroid.as_slice();

        assert!((slice[0] - 1.0).abs() < 0.001);
        assert!((slice[1] - 2.0).abs() < 0.001);
        assert!((slice[2] - 3.0).abs() < 0.001);
        assert!((slice[3] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_centroid_empty() {
        let embeddings: Vec<&<TestSpace as EmbeddingSpace>::EmbeddingData> = vec![];
        let centroid = ClusteredData::<TestSpace>::compute_centroid(&embeddings);
        let slice = centroid.as_slice();

        // Should return zero vector
        for &val in slice {
            assert!((val - 0.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_fit_basic() {
        // Create two clearly separated clusters
        let dataset = create_test_dataset(
            vec![
                // Cluster 1: near origin
                vec![0.1, 0.1, 0.0, 0.0],
                vec![0.2, 0.1, 0.0, 0.0],
                vec![0.1, 0.2, 0.0, 0.0],
                // Cluster 2: far from origin
                vec![10.0, 10.0, 0.0, 0.0],
                vec![10.1, 10.0, 0.0, 0.0],
                vec![10.0, 10.1, 0.0, 0.0],
            ],
            vec![],
        );

        let mut clustered = ClusteredData::<TestSpace>::new(10, 2, dataset);
        clustered.fit();

        // Each cluster should have 3 points
        let mut sizes: Vec<usize> = clustered.clusters.iter().map(|c| c.assignments.len()).collect();
        sizes.sort();
        assert_eq!(sizes, vec![3, 3]);
    }

    #[test]
    fn test_bounded_assignment_respects_capacity() {
        // 6 points, 2 clusters, limit of 2 per cluster
        // Only 4 points can be assigned
        let dataset = create_test_dataset(
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.1, 0.0, 0.0, 0.0],
                vec![0.2, 0.0, 0.0, 0.0],
                vec![10.0, 0.0, 0.0, 0.0],
                vec![10.1, 0.0, 0.0, 0.0],
                vec![10.2, 0.0, 0.0, 0.0],
            ],
            vec![],
        );

        let mut clustered = ClusteredData::<TestSpace>::new(2, 2, dataset);
        clustered.fit();

        // Each cluster should have at most 2 points
        for cluster in &clustered.clusters {
            assert!(cluster.assignments.len() <= 2);
        }

        // Total assignments should be min(n, k * limit) = min(6, 4) = 4
        let total: usize = clustered.clusters.iter().map(|c| c.assignments.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_predict_nearest() {
        let dataset = create_test_dataset(
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![10.0, 0.0, 0.0, 0.0],
            ],
            vec![],
        );

        let mut clustered = ClusteredData::<TestSpace>::new(10, 2, dataset);
        clustered.fit();

        // Query point near first cluster
        let query = TestSpace::create_embedding(vec![0.1, 0.0, 0.0, 0.0]);
        let nearest = clustered.predict(&query, 1);

        assert_eq!(nearest.len(), 1);
        // The nearest centroid should be close to [0,0,0,0]
        let centroid_slice = nearest[0].1.as_slice();
        assert!(centroid_slice[0].abs() < 1.0); // Near origin
    }

    #[test]
    fn test_labels() {
        let dataset = create_test_dataset(
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![10.0, 0.0, 0.0, 0.0],
            ],
            vec![],
        );

        let mut clustered = ClusteredData::<TestSpace>::new(10, 2, dataset);
        clustered.fit();

        let labels = clustered.labels();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].0, 0);
        assert_eq!(labels[1].0, 1);
    }

    #[test]
    fn test_cluster_embeddings() {
        let dataset = create_test_dataset(
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.1, 0.0, 0.0, 0.0],
                vec![10.0, 0.0, 0.0, 0.0],
            ],
            vec![],
        );

        let mut clustered = ClusteredData::<TestSpace>::new(10, 2, dataset);
        clustered.fit();

        // Should be able to get embeddings for valid cluster
        assert!(clustered.cluster_embeddings(0).is_some());
        assert!(clustered.cluster_embeddings(1).is_some());

        // Should return None for invalid cluster index
        assert!(clustered.cluster_embeddings(99).is_none());
    }

    #[test]
    fn test_test_set() {
        let dataset = create_test_dataset(
            vec![vec![0.0, 0.0, 0.0, 0.0]],
            vec![vec![1.0, 1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0, 2.0]],
        );

        let clustered = ClusteredData::<TestSpace>::new(10, 1, dataset);

        let test_set = clustered.test_set();
        assert_eq!(test_set.len(), 2);
    }

    #[test]
    fn test_empty_dataset() {
        let dataset = create_test_dataset(vec![], vec![]);
        let mut clustered = ClusteredData::<TestSpace>::new(10, 2, dataset);

        // Should not panic on empty dataset
        clustered.fit();

        // All clusters should be empty
        for cluster in &clustered.clusters {
            assert!(cluster.assignments.is_empty());
        }
    }

    #[test]
    fn test_single_point() {
        let dataset = create_test_dataset(
            vec![vec![5.0, 5.0, 5.0, 5.0]],
            vec![],
        );

        let mut clustered = ClusteredData::<TestSpace>::new(10, 2, dataset);
        clustered.fit();

        // Single point should be assigned to exactly one cluster
        let total: usize = clustered.clusters.iter().map(|c| c.assignments.len()).sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn test_get_assignment() {
        let dataset = create_test_dataset(
            vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.1, 0.0, 0.0, 0.0],
                vec![10.0, 0.0, 0.0, 0.0],
            ],
            vec![],
        );

        let mut clustered = ClusteredData::<TestSpace>::new(10, 2, dataset);
        clustered.fit();

        // Each point should have an assignment
        assert!(clustered.get_assignment(0).is_some());
        assert!(clustered.get_assignment(1).is_some());
        assert!(clustered.get_assignment(2).is_some());

        // Invalid index should return None
        assert!(clustered.get_assignment(99).is_none());
    }
}
