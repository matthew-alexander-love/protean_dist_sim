# Quick Start Guide

## Prerequisites

✅ All required files are in place:
- `coordinator_config.yaml` - Coordinator configuration (3 workers)
- `test_plan.yaml` - Test plan with bootstrap, churn, and queries
- `run.sh` - Interactive deployment script
- `../docker-compose.yml` - Service orchestration
- `../Dockerfile` - Multi-stage build configuration
- `../data/sift_base.fvecs` - Base dataset (493MB)
- `../data/sift_query.fvecs` - Query dataset (5MB)
- `../output/` - Results directory (auto-created)

## Running the Simulation

### Option 1: Full Setup and Start (Recommended)

```bash
cd /home/matt/Projects/gossipann/protean_dist_sim/deploy
./run.sh start
```

This will:
1. Check Docker prerequisites
2. Verify dataset files
3. Build Docker images
4. Start 3 workers + 1 coordinator
5. Wait for services to be healthy
6. Show service status

### Option 2: Individual Commands

```bash
cd /home/matt/Projects/gossipann/protean_dist_sim/deploy

# Just build images
./run.sh build

# Start services (assumes images are built)
./run.sh up

# View logs in real-time
./run.sh logs

# Check service status
./run.sh status

# View test results
./run.sh results

# Stop services
./run.sh down

# Clean up everything (including volumes)
./run.sh clean
```

## Test Plan Overview

The test plan executes these phases:

1. **Bootstrap** (9000 peers) - Initial network setup
2. **Wait** (5s) - Stabilization
3. **Initial Snapshot** - Capture baseline topology
4. **Queries** (5 queries) - Test initial network
5. **Flash Crowd Churn** (1000 new peers join) - Stress test
6. **Wait** (10s) - Stabilization
7. **Post-Churn Snapshot** - Capture after flash crowd
8. **Queries** (5 queries) - Test network performance
9. **Mixed Churn** (500 leave, 500 join) - Realistic churn
10. **Wait** (5s) - Stabilization
11. **Queries from New Peers** (5 queries) - Test new peer integration
12. **Final Snapshot** - Capture final state

## Expected Output

After the test completes, check `../output/`:

```bash
ls -lh ../output/
```

You should see:
- `initial_network_global.json` - Global snapshot of initial network
- `initial_network_global_adjacency.json` - Complete adjacency matrix
- `after_flash_crowd_churn_global.json` - After flash crowd
- `final_network_global.json` - Final network state
- `snapshot_worker*.json` - Per-worker snapshots
- Query results and metrics

## Analyzing Results

### View Global Snapshot Summary
```bash
cat ../output/initial_network_global.json | jq '.summary'
```

### Count Total Connections
```bash
cat ../output/initial_network_global_adjacency.json | jq '.matrix | map(add) | add'
```

### Find Peer with Most Connections
```bash
cat ../output/initial_network_global.json | jq '[.peers[] | {uuid: .uuid, degree: .num_routable}] | sort_by(.degree) | reverse | .[0]'
```

## Troubleshooting

### Services Won't Start
```bash
# Check Docker is running
docker info

# Check service logs
./run.sh logs

# Rebuild from scratch
./run.sh clean
./run.sh build
./run.sh up
```

### Missing Dataset Files
The script will detect missing files and prompt you. Files should be at:
- `../data/sift_base.fvecs`
- `../data/sift_query.fvecs`

### Configuration Mismatch
Ensure `coordinator_config.yaml` has exactly 3 workers matching `docker-compose.yml`:
- worker0, worker1, worker2

## Scaling to More Workers

To add more workers, edit both:

1. `docker-compose.yml` - Add worker3, worker4, etc.
2. `coordinator_config.yaml` - Add worker entries

Example for 5 workers:
```yaml
# In docker-compose.yml
  worker3:
    # ... (copy worker2 config, change worker-id and ports)

# In coordinator_config.yaml
  - worker_id: "worker3"
    address: "worker3:50051"
```

## Next Steps

- Review test results in `../output/`
- Modify `test_plan.yaml` to test different scenarios
- Scale to more workers for larger simulations
- Deploy to GCP using `../deploy/gcp/` scripts
