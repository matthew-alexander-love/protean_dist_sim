

use dashmap::DashMap;
use protean::embedding_space::{Embedding, EmbeddingSpace};
use protean::SnvConfig;

use tokio::sync::Mutex;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender}

use tonic::transport::Channel;
use tonic::{Request, Response, Status};

use crate::worker;
use crate::{
    test_plan::{
        Config, 
        SimConfig, 
        TestPhase,
    },
    dataloader::{
        DataSet,
        DataLoader,
    },
}

struct Worker {
    address: Address,
    WorkerNodeClient<Channel>,
}

pub struct Coordinator<S: EmbeddingSpace> {
    config: Config,

    data_set: DataSet<S>,

    event_rx: UnboundedReceiver<ProteanEvent>,

    workers: Arc<RwLock<Vec<Worker>>>,
}

impl<S: EmbeddingSpace> Coordinator<S> {
    fn create_actor_uuid(embedding_idx: usize) -> Vec<u8> {
        embedding_idx.into()
    }

    async fn get_num_workers(&self) {
        let workers_guard = self.workers.read().await;
        workers_guard.len();
    }

    async fn get_worker_for_embedding(&self, embedding_idx: usize) -> Option<usize> {
        if embedding_idx >= self.data_set.train.len() {
            return None;
        }

        let worker_count = self.get_num_workers().await;
        if worker_count == 0 {
            return None;
        }
        embedding_idx % worker_count;
    }

    async fn gradual_join(&self, global_indices: IndexRange, bootstrap_indices: Vec<u64>, rate_per_sec: f64) {

        let mut worker_assignments: Vec<Vec<usize>> = Vec::with_capacity();
        for idx in global_indices {
            if let Some(worker_idx) = self.get_worker_for_embedding(idx) {
                worker_assignments 
            }
        }
    }
}