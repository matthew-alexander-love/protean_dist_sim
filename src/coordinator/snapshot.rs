use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::proto::dist_sim::NetworkSnapshot;
use crate::proto::protean::SparseNeighborViewProto;

/// Parsed snapshot with adjacency matrices and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedSnapshot {
    pub timestamp_ms: u64,
    pub worker_id: String,
    pub peers: Vec<PeerSnapshot>,
    pub summary: SnapshotSummary,
}

/// Individual peer snapshot with adjacency information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerSnapshot {
    /// Peer UUID (as hex string)
    pub uuid: String,

    /// Global index from UUID
    pub global_index: u64,

    /// Adjacency matrix indices for routable peers (bidirectional connections)
    pub routable_neighbors: Vec<String>,

    /// All known peers (routable + passive)
    pub all_known_peers: Vec<KnownPeer>,

    /// Number of routable peers
    pub num_routable: usize,

    /// Dynamism counter
    pub dynamism: u32,

    /// Statistics
    pub stats: PeerStats,
}

/// Known peer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownPeer {
    pub uuid: String,
    pub global_index: u64,
    pub status: String,  // "CONNECTED" or "SUSPECT"
    pub distance: f32,
    pub is_routable: bool,
}

/// Statistics for a peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerStats {
    pub avg_distance_routable: f32,
    pub avg_distance_all: f32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub routable_ratio: f32,
}

/// Summary statistics for entire snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotSummary {
    pub total_peers: usize,
    pub total_edges: usize,  // Total routable connections
    pub avg_degree: f32,
    pub max_degree: usize,
    pub min_degree: usize,
    pub avg_dynamism: f32,
}

impl ParsedSnapshot {
    /// Parse a NetworkSnapshot proto into structured data
    pub fn from_proto(snapshot: NetworkSnapshot) -> Result<Self, Box<dyn std::error::Error>> {
        let mut peers = Vec::new();
        let mut total_edges = 0;
        let mut total_dynamism = 0u64;
        let mut degrees = Vec::new();

        for peer_proto in &snapshot.peer_snapshots {
            let peer_snapshot = Self::parse_peer(peer_proto)?;
            degrees.push(peer_snapshot.num_routable);
            total_edges += peer_snapshot.num_routable;
            total_dynamism += peer_snapshot.dynamism as u64;
            peers.push(peer_snapshot);
        }

        let total_peers = peers.len();
        let avg_degree = if total_peers > 0 {
            total_edges as f32 / total_peers as f32
        } else {
            0.0
        };

        let max_degree = degrees.iter().max().copied().unwrap_or(0);
        let min_degree = degrees.iter().min().copied().unwrap_or(0);
        let avg_dynamism = if total_peers > 0 {
            total_dynamism as f32 / total_peers as f32
        } else {
            0.0
        };

        let summary = SnapshotSummary {
            total_peers,
            total_edges,
            avg_degree,
            max_degree,
            min_degree,
            avg_dynamism,
        };

        Ok(Self {
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            worker_id: snapshot.worker_id,
            peers,
            summary,
        })
    }

    /// Parse a single peer's SNV
    fn parse_peer(
        snv: &SparseNeighborViewProto,
    ) -> Result<PeerSnapshot, Box<dyn std::error::Error>> {
        // Extract UUID and global index
        let uuid_hex = hex::encode(&snv.local_uuid);
        let global_index = Self::uuid_to_global_index(&snv.local_uuid);

        let num_routable = snv.num_routable as usize;
        let mut routable_neighbors = Vec::new();
        let mut all_known_peers = Vec::new();

        // Parse peers
        for (idx, peer_entry) in snv.peers.iter().enumerate() {
            if let Some(peer) = &peer_entry.peer {
                let peer_uuid_hex = hex::encode(&peer.uuid);
                let peer_global_index = Self::uuid_to_global_index(&peer.uuid);

                let status = match peer_entry.status {
                    0 => "CONNECTED",
                    1 => "SUSPECT",
                    _ => "UNKNOWN",
                };

                let distance = if let Some(dist) = &peer_entry.dist_to_local {
                    if !dist.float_data.is_empty() {
                        dist.float_data[0]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let is_routable = idx < num_routable;

                if is_routable {
                    routable_neighbors.push(peer_uuid_hex.clone());
                }

                all_known_peers.push(KnownPeer {
                    uuid: peer_uuid_hex,
                    global_index: peer_global_index,
                    status: status.to_string(),
                    distance,
                    is_routable,
                });
            }
        }

        // Calculate statistics
        let stats = Self::calculate_stats(&all_known_peers, num_routable);

        Ok(PeerSnapshot {
            uuid: uuid_hex,
            global_index,
            routable_neighbors,
            all_known_peers,
            num_routable,
            dynamism: snv.dynamism,
            stats,
        })
    }

    /// Calculate statistics for a peer
    fn calculate_stats(known_peers: &[KnownPeer], num_routable: usize) -> PeerStats {
        if known_peers.is_empty() {
            return PeerStats {
                avg_distance_routable: 0.0,
                avg_distance_all: 0.0,
                min_distance: 0.0,
                max_distance: 0.0,
                routable_ratio: 0.0,
            };
        }

        let routable_distances: Vec<f32> = known_peers
            .iter()
            .filter(|p| p.is_routable)
            .map(|p| p.distance)
            .collect();

        let all_distances: Vec<f32> = known_peers.iter().map(|p| p.distance).collect();

        let avg_distance_routable = if !routable_distances.is_empty() {
            routable_distances.iter().sum::<f32>() / routable_distances.len() as f32
        } else {
            0.0
        };

        let avg_distance_all = all_distances.iter().sum::<f32>() / all_distances.len() as f32;

        let min_distance = all_distances
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let max_distance = all_distances
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let routable_ratio = if !known_peers.is_empty() {
            num_routable as f32 / known_peers.len() as f32
        } else {
            0.0
        };

        PeerStats {
            avg_distance_routable,
            avg_distance_all,
            min_distance,
            max_distance,
            routable_ratio,
        }
    }

    /// Extract global index from UUID (first 8 bytes, big-endian)
    fn uuid_to_global_index(uuid: &[u8]) -> u64 {
        if uuid.len() >= 8 {
            u64::from_be_bytes([
                uuid[0], uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7],
            ])
        } else {
            0
        }
    }

    /// Create adjacency matrix for routable peers (local worker only)
    pub fn create_adjacency_matrix(&self) -> AdjacencyMatrix {
        let mut uuid_to_index = HashMap::new();
        for (idx, peer) in self.peers.iter().enumerate() {
            uuid_to_index.insert(peer.uuid.clone(), idx);
        }

        let n = self.peers.len();
        let mut matrix = vec![vec![0u8; n]; n];

        for (i, peer) in self.peers.iter().enumerate() {
            for neighbor_uuid in &peer.routable_neighbors {
                if let Some(&j) = uuid_to_index.get(neighbor_uuid) {
                    matrix[i][j] = 1;
                }
            }
        }

        AdjacencyMatrix {
            peer_uuids: self.peers.iter().map(|p| p.uuid.clone()).collect(),
            global_indices: self.peers.iter().map(|p| p.global_index).collect(),
            matrix,
        }
    }

    /// Save to JSON file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Save adjacency matrix to separate file
    pub fn save_adjacency_matrix(
        &self,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let adj_matrix = self.create_adjacency_matrix();
        let json = serde_json::to_string_pretty(&adj_matrix)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Adjacency matrix representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjacencyMatrix {
    /// Ordered list of peer UUIDs (matrix indices)
    pub peer_uuids: Vec<String>,

    /// Global indices corresponding to peer_uuids
    pub global_indices: Vec<u64>,

    /// Adjacency matrix: matrix[i][j] = 1 if peer i has peer j as routable neighbor
    pub matrix: Vec<Vec<u8>>,
}

/// Global snapshot aggregating all workers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalSnapshot {
    pub timestamp_ms: u64,
    pub worker_count: usize,
    pub total_peers: usize,
    pub peers: Vec<PeerSnapshot>,
    pub summary: SnapshotSummary,
}

impl GlobalSnapshot {
    /// Merge multiple worker snapshots into a single global view
    pub fn merge_snapshots(snapshots: Vec<ParsedSnapshot>) -> Result<Self, Box<dyn std::error::Error>> {
        if snapshots.is_empty() {
            return Err("Cannot merge empty snapshot list".into());
        }

        // Collect all peers from all workers
        let mut all_peers = Vec::new();
        let mut timestamp_ms = 0u64;

        for snapshot in &snapshots {
            all_peers.extend(snapshot.peers.clone());
            // Use the latest timestamp
            timestamp_ms = timestamp_ms.max(snapshot.timestamp_ms);
        }

        // Calculate global statistics
        let total_peers = all_peers.len();
        let mut total_edges = 0;
        let mut total_dynamism = 0u64;
        let mut degrees = Vec::new();

        for peer in &all_peers {
            degrees.push(peer.num_routable);
            total_edges += peer.num_routable;
            total_dynamism += peer.dynamism as u64;
        }

        let avg_degree = if total_peers > 0 {
            total_edges as f32 / total_peers as f32
        } else {
            0.0
        };

        let max_degree = degrees.iter().max().copied().unwrap_or(0);
        let min_degree = degrees.iter().min().copied().unwrap_or(0);
        let avg_dynamism = if total_peers > 0 {
            total_dynamism as f32 / total_peers as f32
        } else {
            0.0
        };

        let summary = SnapshotSummary {
            total_peers,
            total_edges,
            avg_degree,
            max_degree,
            min_degree,
            avg_dynamism,
        };

        Ok(Self {
            timestamp_ms,
            worker_count: snapshots.len(),
            total_peers,
            peers: all_peers,
            summary,
        })
    }

    /// Create global adjacency matrix that includes all peers across all workers
    pub fn create_global_adjacency_matrix(&self) -> AdjacencyMatrix {
        // Create mapping from UUID to matrix index
        let mut uuid_to_index = HashMap::new();
        for (idx, peer) in self.peers.iter().enumerate() {
            uuid_to_index.insert(peer.uuid.clone(), idx);
        }

        let n = self.peers.len();
        let mut matrix = vec![vec![0u8; n]; n];

        // Fill adjacency matrix
        for (i, peer) in self.peers.iter().enumerate() {
            for neighbor_uuid in &peer.routable_neighbors {
                // Key difference: we look up neighbors across ALL peers, not just local ones
                if let Some(&j) = uuid_to_index.get(neighbor_uuid) {
                    matrix[i][j] = 1;
                }
            }
        }

        AdjacencyMatrix {
            peer_uuids: self.peers.iter().map(|p| p.uuid.clone()).collect(),
            global_indices: self.peers.iter().map(|p| p.global_index).collect(),
            matrix,
        }
    }

    /// Save global snapshot to JSON file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Save global adjacency matrix to separate file
    pub fn save_adjacency_matrix(
        &self,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let adj_matrix = self.create_global_adjacency_matrix();
        let json = serde_json::to_string_pretty(&adj_matrix)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Create adjacency matrix for ALL known peers (not just routable)
    pub fn create_all_peers_adjacency_matrix(&self) -> AdjacencyMatrix {
        // Create mapping from UUID to matrix index
        let mut uuid_to_index = HashMap::new();
        for (idx, peer) in self.peers.iter().enumerate() {
            uuid_to_index.insert(peer.uuid.clone(), idx);
        }

        let n = self.peers.len();
        let mut matrix = vec![vec![0u8; n]; n];

        // Fill adjacency matrix using all_known_peers (not just routable)
        for (i, peer) in self.peers.iter().enumerate() {
            for known_peer in &peer.all_known_peers {
                if let Some(&j) = uuid_to_index.get(&known_peer.uuid) {
                    matrix[i][j] = 1;
                }
            }
        }

        AdjacencyMatrix {
            peer_uuids: self.peers.iter().map(|p| p.uuid.clone()).collect(),
            global_indices: self.peers.iter().map(|p| p.global_index).collect(),
            matrix,
        }
    }

    /// Save all-peers adjacency matrix to file
    pub fn save_all_peers_adjacency_matrix(
        &self,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let adj_matrix = self.create_all_peers_adjacency_matrix();
        let json = serde_json::to_string_pretty(&adj_matrix)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uuid_to_global_index() {
        let mut uuid = vec![0u8; 64];
        uuid[0..8].copy_from_slice(&42u64.to_be_bytes());

        assert_eq!(ParsedSnapshot::uuid_to_global_index(&uuid), 42);
    }

    #[test]
    fn test_calculate_stats_empty() {
        let stats = ParsedSnapshot::calculate_stats(&[], 0);
        assert_eq!(stats.avg_distance_routable, 0.0);
        assert_eq!(stats.avg_distance_all, 0.0);
    }

    #[test]
    fn test_calculate_stats() {
        let peers = vec![
            KnownPeer {
                uuid: "peer1".to_string(),
                global_index: 1,
                status: "CONNECTED".to_string(),
                distance: 1.0,
                is_routable: true,
            },
            KnownPeer {
                uuid: "peer2".to_string(),
                global_index: 2,
                status: "CONNECTED".to_string(),
                distance: 2.0,
                is_routable: true,
            },
            KnownPeer {
                uuid: "peer3".to_string(),
                global_index: 3,
                status: "CONNECTED".to_string(),
                distance: 3.0,
                is_routable: false,
            },
        ];

        let stats = ParsedSnapshot::calculate_stats(&peers, 2);
        assert!((stats.avg_distance_routable - 1.5).abs() < 0.01);
        assert!((stats.avg_distance_all - 2.0).abs() < 0.01);
        assert_eq!(stats.min_distance, 1.0);
        assert_eq!(stats.max_distance, 3.0);
        assert!((stats.routable_ratio - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_adjacency_matrix_creation() {
        let peers = vec![
            PeerSnapshot {
                uuid: "peer0".to_string(),
                global_index: 0,
                routable_neighbors: vec!["peer1".to_string(), "peer2".to_string()],
                all_known_peers: vec![],
                num_routable: 2,
                dynamism: 0,
                stats: PeerStats {
                    avg_distance_routable: 0.0,
                    avg_distance_all: 0.0,
                    min_distance: 0.0,
                    max_distance: 0.0,
                    routable_ratio: 0.0,
                },
            },
            PeerSnapshot {
                uuid: "peer1".to_string(),
                global_index: 1,
                routable_neighbors: vec!["peer0".to_string()],
                all_known_peers: vec![],
                num_routable: 1,
                dynamism: 0,
                stats: PeerStats {
                    avg_distance_routable: 0.0,
                    avg_distance_all: 0.0,
                    min_distance: 0.0,
                    max_distance: 0.0,
                    routable_ratio: 0.0,
                },
            },
            PeerSnapshot {
                uuid: "peer2".to_string(),
                global_index: 2,
                routable_neighbors: vec![],
                all_known_peers: vec![],
                num_routable: 0,
                dynamism: 0,
                stats: PeerStats {
                    avg_distance_routable: 0.0,
                    avg_distance_all: 0.0,
                    min_distance: 0.0,
                    max_distance: 0.0,
                    routable_ratio: 0.0,
                },
            },
        ];

        let snapshot = ParsedSnapshot {
            timestamp_ms: 0,
            worker_id: "test".to_string(),
            peers,
            summary: SnapshotSummary {
                total_peers: 3,
                total_edges: 3,
                avg_degree: 1.0,
                max_degree: 2,
                min_degree: 0,
                avg_dynamism: 0.0,
            },
        };

        let adj_matrix = snapshot.create_adjacency_matrix();

        assert_eq!(adj_matrix.peer_uuids.len(), 3);
        assert_eq!(adj_matrix.matrix.len(), 3);

        // peer0 -> peer1, peer2
        assert_eq!(adj_matrix.matrix[0][1], 1);
        assert_eq!(adj_matrix.matrix[0][2], 1);

        // peer1 -> peer0
        assert_eq!(adj_matrix.matrix[1][0], 1);

        // peer2 -> none
        assert_eq!(adj_matrix.matrix[2][0], 0);
        assert_eq!(adj_matrix.matrix[2][1], 0);
    }
}
