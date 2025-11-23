# Protean Distributed Simulation - Docker Deployment Guide

This directory contains everything needed to deploy the Protean distributed ANN testing system using Docker and Docker Compose, ready for GCP deployment.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [GCP Deployment](#gcp-deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Overview

The containerized deployment consists of:
- **1 Coordinator**: Orchestrates the distributed simulation
- **3 Workers** (default): Execute peer actors and run queries
- Shared configuration and data volumes
- Network isolation for secure communication

## Prerequisites

### Local Development
```bash
# Required
- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 20GB+ disk space

# Optional
- docker-compose CLI (for older Docker versions)
```

### Dataset Files
You need to provide SIFT dataset files:
```bash
mkdir -p data
# Place your dataset files in ./data/:
# - sift_base.fvecs
# - sift_query.fvecs
```

Download SIFT datasets from: http://corpus-texmex.irisa.fr/

## Quick Start

### 1. Prepare Dataset
```bash
cd protean_dist_sim
mkdir -p data output

# Copy or symlink your dataset files
cp /path/to/sift_base.fvecs data/
cp /path/to/sift_query.fvecs data/
```

### 2. Build Images
```bash
# Build all services
docker-compose build

# Or build individually
docker-compose build coordinator
docker-compose build worker0
```

### 3. Start Services
```bash
# Start all services
docker-compose up

# Or run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

### 4. Check Status
```bash
# Check all services
docker-compose ps

# Check specific service logs
docker-compose logs coordinator
docker-compose logs worker0
```

### 5. Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Directory Structure

```
protean_dist_sim/
├── Dockerfile                  # Multi-stage build for coordinator & worker
├── docker-compose.yml          # Service orchestration
├── .dockerignore              # Build context optimization
├── deploy/
│   ├── README.md              # This file
│   ├── coordinator_config.yaml # Coordinator configuration
│   └── test_plan.yaml         # Test execution plan
├── data/                      # Dataset files (user-provided)
│   ├── sift_base.fvecs
│   └── sift_query.fvecs
└── output/                    # Test results (auto-created)
    ├── initial_network.json
    ├── after_churn.json
    └── final_network.json
```

## Configuration

### Coordinator Configuration

Edit `deploy/coordinator_config.yaml`:

```yaml
# Workers (must match docker-compose service names)
workers:
  - worker_id: "worker0"
    address: "worker0:50051"
  - worker_id: "worker1"
    address: "worker1:50051"
  - worker_id: "worker2"
    address: "worker2:50051"

# Dataset paths (container paths)
dataset:
  base_path: "/app/data/sift_base.fvecs"
  query_path: "/app/data/sift_query.fvecs"

# Simulation parameters
sim_config:
  max_peers: 1000
  region: "us-central1"
  snv_config:
    concurrency_limit: 10
    timeout_ms: 5000
    # ... (see file for full config)
```

### Test Plan Configuration

Edit `deploy/test_plan.yaml`:

```yaml
# Number of peers to simulate
peer_count: 100

phases:
  # Bootstrap phase
  - type: Bootstrap
    peer_indices:
      start: 0
      end: 49
    bootstrap_index: 0

  # Query phase
  - type: Query
    query_indices: [0, 1, 2, 3, 4]
    source_peer_indices: [0, 10, 20, 30, 40]
    k: 10

  # ... (see file for full plan)
```

### Scaling Workers

To add more workers, edit `docker-compose.yml`:

```yaml
  worker3:
    build:
      context: ..
      dockerfile: protean_dist_sim/Dockerfile
      target: worker
    image: protean-dist-sim-worker:latest
    container_name: protean-worker3
    hostname: worker3
    command:
      - "--worker-id=worker3"
      - "--bind-address=0.0.0.0:50051"
      - "--coordinator-address=coordinator:50050"
      - "--n-workers=4"  # Update this!
      - "--max-actors=1000"
    ports:
      - "50054:50051"  # Unique host port
    # ... rest of config
```

**Don't forget to:**
1. Update `--n-workers` for ALL workers
2. Add worker to coordinator config
3. Update coordinator dependencies

## Running Tests

### Standard Test Run
```bash
# Start services and watch logs
docker-compose up

# The coordinator will:
# 1. Wait for workers to register
# 2. Load and distribute datasets
# 3. Execute test plan phases
# 4. Write results to ./output/
```

### Custom Test Plan
```bash
# Create custom test plan
cp deploy/test_plan.yaml deploy/my_test.yaml
# Edit deploy/my_test.yaml

# Update docker-compose.yml coordinator volume:
volumes:
  - ./deploy/my_test.yaml:/app/config/test_plan.yaml:ro

# Run
docker-compose up
```

### Development Mode
```bash
# Build with verbose logging
RUST_LOG=debug docker-compose up

# Run specific test
docker-compose run --rm coordinator \
  --config /app/config/coordinator_config.yaml \
  --test-plan /app/config/test_plan.yaml \
  --worker-timeout 60
```

## GCP Deployment

### Option 1: Google Kubernetes Engine (GKE)

1. **Convert to Kubernetes**:
```bash
# Install kompose
curl -L https://github.com/kubernetes/kompose/releases/download/v1.31.2/kompose-linux-amd64 -o kompose
chmod +x kompose
sudo mv kompose /usr/local/bin/

# Convert docker-compose to k8s manifests
cd protean_dist_sim
kompose convert

# Review generated files
ls -la *-deployment.yaml *-service.yaml
```

2. **Deploy to GKE**:
```bash
# Create cluster
gcloud container clusters create protean-cluster \
  --num-nodes=4 \
  --machine-type=n2-standard-4 \
  --region=us-central1

# Build and push images
docker build -t gcr.io/YOUR_PROJECT/protean-coordinator:latest --target coordinator .
docker build -t gcr.io/YOUR_PROJECT/protean-worker:latest --target worker .

docker push gcr.io/YOUR_PROJECT/protean-coordinator:latest
docker push gcr.io/YOUR_PROJECT/protean-worker:latest

# Update image names in k8s manifests
# Deploy
kubectl apply -f coordinator-deployment.yaml
kubectl apply -f worker0-deployment.yaml
kubectl apply -f worker1-deployment.yaml
kubectl apply -f worker2-deployment.yaml
```

### Option 2: Google Compute Engine (GCE)

1. **Create VM**:
```bash
gcloud compute instances create protean-host \
  --machine-type=n2-standard-8 \
  --boot-disk-size=50GB \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --zone=us-central1-a
```

2. **SSH and Deploy**:
```bash
gcloud compute ssh protean-host

# Install docker-compose
# Copy files
# Run: docker-compose up -d
```

### Option 3: Cloud Run (for smaller tests)

```bash
# Build images
gcloud builds submit --tag gcr.io/YOUR_PROJECT/protean-coordinator
gcloud builds submit --tag gcr.io/YOUR_PROJECT/protean-worker

# Deploy coordinator
gcloud run deploy protean-coordinator \
  --image gcr.io/YOUR_PROJECT/protean-coordinator \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2

# Note: Cloud Run may not be ideal for multi-service deployments
```

## Monitoring

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f coordinator
docker-compose logs -f worker0

# Last 100 lines
docker-compose logs --tail=100 coordinator
```

### Check Health
```bash
# Service status
docker-compose ps

# Health checks
docker inspect protean-coordinator | jq '.[].State.Health'
docker inspect protean-worker0 | jq '.[].State.Health'
```

### Resource Usage
```bash
# Container stats
docker stats

# Specific containers
docker stats protean-coordinator protean-worker0 protean-worker1
```

### Results
```bash
# View output files
ls -lh output/

# Pretty-print JSON results
cat output/initial_network.json | jq .
cat output/final_network.json | jq .
```

## Troubleshooting

### Workers Not Registering
```bash
# Check coordinator logs
docker-compose logs coordinator | grep -i register

# Check worker logs
docker-compose logs worker0 | grep -i coordinator

# Verify network
docker network inspect protean_protean-network

# Test connectivity
docker-compose exec worker0 nc -zv coordinator 50050
```

### Out of Memory
```bash
# Increase Docker memory limit
# Edit docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 8G  # Increase

# Or reduce peer count in test_plan.yaml
peer_count: 50  # Reduce from 100
```

### Build Failures
```bash
# Clear build cache
docker-compose build --no-cache

# Check .dockerignore
cat .dockerignore

# Verify protean dependency
ls -la ../protean/
```

### Dataset Not Found
```bash
# Verify files exist
ls -lh data/

# Check permissions
chmod 644 data/*.fvecs

# Verify mount in container
docker-compose exec coordinator ls -lh /app/data/
```

### Port Conflicts
```bash
# Check if ports are in use
lsof -i :50050
lsof -i :50051

# Change ports in docker-compose.yml
ports:
  - "60050:50050"  # Use different host port
```

## Performance Tuning

### For Large-Scale Tests

1. **Increase Resources**:
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
```

2. **Add More Workers**:
- Scale horizontally by adding worker services
- Update `--n-workers` parameter
- Distribute peers evenly

3. **Optimize Test Plan**:
- Reduce concurrent queries
- Increase wait times between phases
- Use smaller churn rates

### For GCP Deployment

1. **Use N2 Machine Types**: Better performance-to-cost ratio
2. **Enable Preemptible VMs**: Cost savings for non-critical tests
3. **Use Regional Clusters**: Better availability
4. **Set up Cloud Logging**: Integrate with Stackdriver

## Security Notes

- Containers run as non-root user (uid 1000)
- Network isolation via bridge network
- Read-only mounts for config and data
- No privileged containers
- Health checks enabled

## Next Steps

1. Test locally with small peer count
2. Verify results in `./output/`
3. Scale up peer count gradually
4. Deploy to GCP for production tests
5. Set up monitoring and alerting

## Support

For issues and questions:
- Check logs: `docker-compose logs`
- Review test plan syntax
- Verify dataset format (.fvecs)
- Check coordinator config worker addresses
