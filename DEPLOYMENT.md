# Protean Distributed Simulation - Deployment Guide

This guide covers building Docker images and deploying the protean_dist_sim system in both local and cloud environments.

## Overview

The deployment system consists of:
- **Coordinator**: Central node that manages the simulation and distributes data
- **Workers**: Compute nodes that run the distributed simulation
- **Docker images**: Pre-built images for both coordinator and worker
- **Deployment scripts**: Automated deployment for local and cloud environments

## Architecture

```
┌─────────────────┐
│   Coordinator   │ (Port 50050)
│   - Manages sim │
│   - Loads data  │
│   - Distributes │
└────────┬────────┘
         │
    ┌────┴─────┬─────────┬─────────┐
    │          │         │         │
┌───▼───┐  ┌──▼───┐  ┌──▼───┐  ┌──▼───┐
│Worker0│  │Worker1│ │Worker2│ │ ...  │
│:50051 │  │:50052 │ │:50053 │ │      │
└───────┘  └───────┘ └───────┘ └──────┘
```

## Quick Start

### Prerequisites

1. **Docker** installed (20.10 or later)
2. **Docker Compose** installed (for local deployment)
3. **Dataset files** in `./data/` directory:
   - `sift_base.fvecs` (516 MB)
   - `sift_query.fvecs` (5 MB)
4. **Configuration files** in `./deploy/` directory:
   - `coordinator_config.yaml`
   - `test_plan.yaml`

### Local Deployment (3 Steps)

```bash
# 1. Build Docker images
./build.sh

# 2. Deploy locally with 11 workers
cd deploy
./deploy-local.sh start --workers 11

# 3. View logs
./deploy-local.sh logs
```

### Cloud Deployment

See [Cloud Deployment](#cloud-deployment) section below.

---

## Building Docker Images

The `build.sh` script builds both coordinator and worker images from a single Dockerfile using multi-stage builds.

### Basic Build

```bash
./build.sh
```

This creates:
- `protean-dist-sim-coordinator:latest`
- `protean-dist-sim-worker:latest`

### Build with Custom Tag

```bash
./build.sh --tag v1.0.0
```

### Build and Push to Registry

```bash
./build.sh --registry gcr.io/my-project --tag v1.0.0 --push
```

This creates and pushes:
- `gcr.io/my-project/protean-dist-sim-coordinator:v1.0.0`
- `gcr.io/my-project/protean-dist-sim-worker:v1.0.0`

### Build Options

| Option | Description | Example |
|--------|-------------|---------|
| `--tag TAG` | Image tag | `--tag v1.0.0` |
| `--push` | Push to registry after build | `--push` |
| `--registry URL` | Container registry URL | `--registry gcr.io/project-id` |
| `--help` | Show help | `--help` |

---

## Local Deployment

The `deploy-local.sh` script manages local deployment using Docker Compose.

### Start Services

```bash
cd deploy
./deploy-local.sh start --workers 11
```

This will:
1. Validate that dataset and config files exist
2. Check that Docker images are built
3. Start coordinator and workers using Docker Compose
4. Scale workers to the specified count

### Configuration

The deployment uses these paths (relative to project root):

| Component | Host Path | Container Path |
|-----------|-----------|----------------|
| Dataset | `./data/` | `/app/data/` |
| Configs | `./deploy/*.yaml` | `/app/config/` |
| Output | `./output/` | `/app/output/` |

### Deployment Commands

```bash
# Start services
./deploy-local.sh start --workers 11

# Stop services
./deploy-local.sh stop

# Restart services
./deploy-local.sh restart --workers 11

# View logs (follow mode)
./deploy-local.sh logs

# Check status
./deploy-local.sh status
```

### Custom Paths

```bash
./deploy-local.sh start \
  --workers 11 \
  --data-dir /path/to/data \
  --config-dir /path/to/configs \
  --output-dir /path/to/output
```

### Docker Compose Direct Usage

If you prefer to use `docker compose` directly:

```bash
cd protean_dist_sim

# Start with 11 workers
WORKER_COUNT=11 docker compose up -d --scale worker=11

# View logs
docker compose logs -f

# Stop
docker compose down
```

---

## Cloud Deployment

The `deploy-cloud.sh` script provides a cloud-agnostic deployment approach that works on any VMs with Docker installed.

### Prerequisites

1. **VMs provisioned** with Docker installed
2. **Cloud storage mounted** on all VMs (e.g., GCS, S3, Azure Blob)
3. **Network connectivity** between coordinator and workers
4. **Docker images** available (built locally or in a registry)

### Cloud Storage Setup

Mount your cloud storage at a consistent path (e.g., `/mnt/data`) on all VMs and ensure it contains:

```
/mnt/data/
├── sift_base.fvecs           # Dataset files
├── sift_query.fvecs
├── coordinator_config.yaml   # Configuration files
├── test_plan.yaml
└── output/                   # Results directory (created automatically)
```

#### Example: Mounting GCS Bucket

```bash
# Install gcsfuse
sudo apt-get update
sudo apt-get install -y gcsfuse

# Mount bucket
mkdir -p /mnt/data
gcsfuse my-bucket /mnt/data

# Verify
ls -la /mnt/data
```

#### Example: Mounting S3 Bucket

```bash
# Install s3fs
sudo apt-get install -y s3fs

# Configure credentials
echo "ACCESS_KEY:SECRET_KEY" > ~/.passwd-s3fs
chmod 600 ~/.passwd-s3fs

# Mount bucket
mkdir -p /mnt/data
s3fs my-bucket /mnt/data -o passwd_file=~/.passwd-s3fs

# Verify
ls -la /mnt/data
```

### Deployment Scenarios

#### Scenario 1: Single VM Deployment

Deploy everything (coordinator + workers) on a single VM:

```bash
./deploy-cloud.sh deploy \
  --workers 11 \
  --cloud-mount /mnt/data
```

#### Scenario 2: Multi-VM Deployment

**On the coordinator VM:**

```bash
./deploy-cloud.sh coordinator \
  --cloud-mount /mnt/data
```

Note the coordinator's IP address (e.g., `10.0.1.5`).

**On each worker VM:**

```bash
./deploy-cloud.sh worker \
  --coordinator-host 10.0.1.5 \
  --cloud-mount /mnt/data
```

For multiple workers per VM, increase the worker count:

```bash
./deploy-cloud.sh worker \
  --coordinator-host 10.0.1.5 \
  --workers 5 \
  --cloud-mount /mnt/data
```

#### Scenario 3: Using a Container Registry

If images are in a registry (recommended for multi-VM deployments):

```bash
# On coordinator VM
./deploy-cloud.sh coordinator \
  --cloud-mount /mnt/data \
  --registry gcr.io/my-project

# On worker VMs
./deploy-cloud.sh worker \
  --coordinator-host 10.0.1.5 \
  --cloud-mount /mnt/data \
  --registry gcr.io/my-project
```

### Cloud Deployment Commands

```bash
# Deploy full stack on single VM
./deploy-cloud.sh deploy --workers 11 --cloud-mount /mnt/data

# Deploy coordinator only
./deploy-cloud.sh coordinator --cloud-mount /mnt/data

# Deploy workers only
./deploy-cloud.sh worker \
  --coordinator-host 10.0.1.5 \
  --cloud-mount /mnt/data

# Stop all services
./deploy-cloud.sh stop

# Check status
./deploy-cloud.sh status
```

### Cloud Deployment Options

| Option | Description | Example |
|--------|-------------|---------|
| `--workers N` | Number of workers | `--workers 11` |
| `--coordinator-host IP` | Coordinator IP (for worker deployment) | `--coordinator-host 10.0.1.5` |
| `--cloud-mount PATH` | Cloud storage mount path | `--cloud-mount /mnt/data` |
| `--registry URL` | Container registry URL | `--registry gcr.io/project-id` |

### Monitoring Cloud Deployments

```bash
# View coordinator logs
docker logs -f protean-coordinator

# View worker logs
docker logs -f protean-worker0

# Check all containers
docker ps | grep protean

# View network
docker network inspect protean-network
```

---

## Configuration Files

### coordinator_config.yaml

Located in `./deploy/coordinator_config.yaml`. Key sections:

```yaml
# Worker list (can be dynamic if workers self-register)
workers:
  - worker_id: "worker0"
    address: "worker0:50051"
  - worker_id: "worker1"
    address: "worker1:50051"

# Dataset paths (container paths)
dataset:
  base_path: "/app/data/sift_base.fvecs"
  query_path: "/app/data/sift_query.fvecs"

# Output directory
output_dir: "/app/output"

# Coordinator bind address
coordinator_bind_address: "0.0.0.0:50050"
```

### test_plan.yaml

Located in `./deploy/test_plan.yaml`. Defines simulation phases:

```yaml
phases:
  - type: Bootstrap
    duration_secs: 10
    peer_count: 100

  - type: Wait
    duration_secs: 5

  - type: Query
    duration_secs: 30
    query_count: 100
```

See the actual files for complete examples.

---

## Networking

### Ports

| Service | Port | Protocol |
|---------|------|----------|
| Coordinator | 50050 | gRPC |
| Worker 0 | 50051 | gRPC |
| Worker 1 | 50052 | gRPC |
| Worker N | 50050+N+1 | gRPC |

### Firewall Rules

For cloud deployments, ensure these ports are open:

```bash
# Allow coordinator port
sudo ufw allow 50050/tcp

# Allow worker ports (adjust range as needed)
sudo ufw allow 50051:50061/tcp
```

For GCP:

```bash
gcloud compute firewall-rules create allow-protean-coordinator \
  --allow tcp:50050 \
  --source-ranges 0.0.0.0/0

gcloud compute firewall-rules create allow-protean-workers \
  --allow tcp:50051-50061 \
  --source-ranges 0.0.0.0/0
```

---

## Troubleshooting

### Images Not Found

```bash
# Build images
./build.sh

# Or pull from registry
docker pull gcr.io/my-project/protean-dist-sim-coordinator:latest
docker pull gcr.io/my-project/protean-dist-sim-worker:latest
```

### Dataset Files Missing

```bash
# Check data directory
ls -lh ./data/

# Expected output:
# -rw-r--r-- 1 user user 516M sift_base.fvecs
# -rw-r--r-- 1 user user 5.1M sift_query.fvecs
```

Download SIFT dataset if missing (see main README).

### Workers Can't Connect to Coordinator

1. Check coordinator is running:
   ```bash
   docker ps | grep coordinator
   ```

2. Check coordinator logs:
   ```bash
   docker logs protean-coordinator
   ```

3. Verify network connectivity:
   ```bash
   # From worker VM
   telnet <coordinator-ip> 50050
   ```

4. Check firewall rules (see [Networking](#networking))

### Cloud Storage Not Mounted

```bash
# Check mount
df -h | grep /mnt/data

# Remount if needed
sudo mount /mnt/data
```

### View Detailed Logs

```bash
# Local deployment
docker compose logs -f

# Cloud deployment
docker logs -f protean-coordinator
docker logs -f protean-worker0

# All containers
docker ps -a
docker logs <container-id>
```

---

## Performance Tuning

### Resource Limits

Adjust resource limits in `docker-compose.yml` or when starting containers:

```bash
docker run ... \
  --cpus 4 \
  --memory 8g \
  protean-dist-sim-worker:latest
```

### Worker Count

More workers = more parallelism, but also more overhead:

```bash
# Light workload
./deploy-local.sh start --workers 5

# Heavy workload
./deploy-local.sh start --workers 20
```

### Network Performance

For cloud deployments, ensure VMs are in the same region/zone to minimize latency.

---

## Cleanup

### Local Deployment

```bash
# Stop services
cd deploy
./deploy-local.sh stop

# Remove volumes (this deletes output data!)
docker compose down -v

# Remove images
docker rmi protean-dist-sim-coordinator:latest
docker rmi protean-dist-sim-worker:latest
```

### Cloud Deployment

```bash
# Stop all services
./deploy-cloud.sh stop

# Remove network
docker network rm protean-network

# Remove images
docker rmi protean-dist-sim-coordinator:latest
docker rmi protean-dist-sim-worker:latest
```

---

## Next Steps

- Review simulation results in `./output/`
- Adjust test plan in `./deploy/test_plan.yaml`
- Scale workers based on workload
- Set up monitoring and alerting for production deployments

For more information, see the main README.md
