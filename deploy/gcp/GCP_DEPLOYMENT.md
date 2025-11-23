# GCP Multi-Node Deployment Guide

This guide shows how to deploy workers on **separate physical hardware** in GCP for maximum resource utilization.

## Deployment Options

### Option 1: GKE (Kubernetes) - **RECOMMENDED** ⭐

**Best for**: Production, auto-scaling, easy management

- Each worker runs as a separate Pod
- Kubernetes scheduler distributes Pods across different Nodes (physical machines)
- Auto-scaling and self-healing
- Easy to add/remove workers

**Resource isolation**: ✅ Each worker on different physical node

### Option 2: Multiple GCE VMs - Manual Control

**Best for**: Fine-grained control, custom machine types

- Each worker runs on its own dedicated VM
- Full control over machine types and resources
- Manual scaling

**Resource isolation**: ✅ Each worker on dedicated VM

### Option 3: Managed Instance Groups

**Best for**: Auto-scaling workers, cost optimization

- Template-based worker deployment
- Auto-scaling based on load
- Load balancing

**Resource isolation**: ✅ Each instance on separate VM

## Quick Comparison

| Feature | GKE | Multi-VM | Instance Groups |
|---------|-----|----------|-----------------|
| Setup Complexity | Medium | Low | Medium |
| Physical Isolation | ✅ Yes | ✅ Yes | ✅ Yes |
| Auto-scaling | ✅ Yes | ❌ No | ✅ Yes |
| Cost Efficiency | High | Medium | High |
| Management | Easy | Manual | Medium |
| **Recommended for** | Production | Development | Large-scale |

---

# Option 1: GKE Deployment (Recommended)

## Architecture

```
GKE Cluster (us-central1)
├── Node Pool: coordinator-pool (1 node, n2-standard-4)
│   └── coordinator-pod
└── Node Pool: worker-pool (3+ nodes, n2-standard-8)
    ├── worker0-pod (on node-1)
    ├── worker1-pod (on node-2)
    └── worker2-pod (on node-3)
```

## Prerequisites

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Install kubectl
gcloud components install kubectl
```

## Step 1: Create GKE Cluster with Separate Node Pools

```bash
# Set variables
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export CLUSTER_NAME="protean-cluster"

# Create cluster with coordinator node pool
gcloud container clusters create $CLUSTER_NAME \
  --region=$REGION \
  --num-nodes=1 \
  --machine-type=n2-standard-4 \
  --node-labels=role=coordinator \
  --disk-size=50 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=1 \
  --node-pool-name=coordinator-pool

# Add worker node pool (separate physical nodes)
gcloud container clusters node-pools create worker-pool \
  --cluster=$CLUSTER_NAME \
  --region=$REGION \
  --num-nodes=3 \
  --machine-type=n2-standard-8 \
  --node-labels=role=worker \
  --disk-size=100 \
  --enable-autoscaling \
  --min-nodes=3 \
  --max-nodes=10

# Get credentials
gcloud container clusters get-credentials $CLUSTER_NAME --region=$REGION
```

**Key points:**
- Coordinator runs on `coordinator-pool` (smaller machines)
- Workers run on `worker-pool` (larger machines)
- Anti-affinity rules ensure workers spread across nodes

## Step 2: Build and Push Images

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build and tag images
cd protean_dist_sim
docker build -t gcr.io/$PROJECT_ID/protean-coordinator:latest \
  --target coordinator \
  -f Dockerfile \
  ..

docker build -t gcr.io/$PROJECT_ID/protean-worker:latest \
  --target worker \
  -f Dockerfile \
  ..

# Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/protean-coordinator:latest
docker push gcr.io/$PROJECT_ID/protean-worker:latest
```

## Step 3: Deploy to GKE

```bash
# Apply Kubernetes manifests
cd deploy/gcp/gke

# Create namespace
kubectl create namespace protean

# Create ConfigMaps (datasets and configs)
kubectl create configmap coordinator-config \
  --from-file=coordinator_config.yaml=../../coordinator_config.yaml \
  -n protean

kubectl create configmap test-plan \
  --from-file=test_plan.yaml=../../test_plan.yaml \
  -n protean

# Deploy services
kubectl apply -f coordinator-service.yaml
kubectl apply -f worker-service.yaml

# Deploy workloads
kubectl apply -f coordinator-deployment.yaml
kubectl apply -f worker-deployment.yaml

# Check status
kubectl get pods -n protean -o wide
```

## Step 4: Verify Physical Isolation

```bash
# Check which nodes pods are on
kubectl get pods -n protean -o wide

# Should show something like:
# NAME                          NODE
# coordinator-xxx               gke-node-1
# worker0-xxx                   gke-node-2  ← Different node
# worker1-xxx                   gke-node-3  ← Different node
# worker2-xxx                   gke-node-4  ← Different node

# Verify anti-affinity
kubectl describe pod worker0-xxx -n protean | grep -A5 "Affinity"
```

## Step 5: Monitor and Scale

```bash
# View logs
kubectl logs -f deployment/coordinator -n protean
kubectl logs -f deployment/worker0 -n protean

# Scale workers
kubectl scale deployment worker-pool --replicas=10 -n protean

# Auto-scaling (HPA)
kubectl autoscale deployment worker-pool \
  --cpu-percent=70 \
  --min=3 \
  --max=20 \
  -n protean
```

---

# Option 2: Multiple GCE VMs

## Architecture

```
VMs in us-central1
├── protean-coordinator (n2-standard-4, 10.0.0.10)
├── protean-worker-0 (n2-standard-8, 10.0.0.11)
├── protean-worker-1 (n2-standard-8, 10.0.0.12)
└── protean-worker-2 (n2-standard-8, 10.0.0.13)
```

## Deployment Script

```bash
# See deploy/gcp/gce/deploy-multi-vm.sh
cd deploy/gcp/gce
./deploy-multi-vm.sh
```

The script will:
1. Create VPC network
2. Create 1 coordinator VM + N worker VMs
3. Install Docker on each VM
4. Deploy containers
5. Configure networking

## Manual Steps

```bash
export PROJECT_ID="your-project-id"
export ZONE="us-central1-a"

# 1. Create coordinator VM
gcloud compute instances create protean-coordinator \
  --zone=$ZONE \
  --machine-type=n2-standard-4 \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --boot-disk-size=50GB \
  --tags=protean-coordinator \
  --metadata=startup-script='#!/bin/bash
    docker pull gcr.io/'$PROJECT_ID'/protean-coordinator:latest
    docker run -d \
      --name coordinator \
      -p 50050:50050 \
      gcr.io/'$PROJECT_ID'/protean-coordinator:latest'

# 2. Create worker VMs (repeat for each worker)
for i in {0..2}; do
  gcloud compute instances create protean-worker-$i \
    --zone=$ZONE \
    --machine-type=n2-standard-8 \
    --image-family=cos-stable \
    --image-project=cos-cloud \
    --boot-disk-size=100GB \
    --tags=protean-worker \
    --metadata=startup-script='#!/bin/bash
      docker pull gcr.io/'$PROJECT_ID'/protean-worker:latest
      docker run -d \
        --name worker \
        -p 50051:50051 \
        gcr.io/'$PROJECT_ID'/protean-worker:latest \
        --worker-id=worker'$i' \
        --bind-address=0.0.0.0:50051 \
        --coordinator-address=protean-coordinator:50050 \
        --n-workers=3 \
        --max-actors=2000'
done

# 3. Configure firewall
gcloud compute firewall-rules create protean-internal \
  --allow=tcp:50050-50053 \
  --source-tags=protean-coordinator,protean-worker \
  --target-tags=protean-coordinator,protean-worker

# 4. Check status
gcloud compute instances list --filter="name~protean"
```

---

# Option 3: Managed Instance Groups

## Setup

```bash
# 1. Create instance template for workers
gcloud compute instance-templates create protean-worker-template \
  --machine-type=n2-standard-8 \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --boot-disk-size=100GB \
  --tags=protean-worker \
  --metadata=startup-script='#!/bin/bash
    docker pull gcr.io/'$PROJECT_ID'/protean-worker:latest
    docker run -d --name worker \
      gcr.io/'$PROJECT_ID'/protean-worker:latest \
      --worker-id=$(hostname) \
      --bind-address=0.0.0.0:50051 \
      --coordinator-address=COORDINATOR_IP:50050'

# 2. Create managed instance group
gcloud compute instance-groups managed create protean-workers \
  --template=protean-worker-template \
  --size=3 \
  --zone=us-central1-a

# 3. Set up auto-scaling
gcloud compute instance-groups managed set-autoscaling protean-workers \
  --max-num-replicas=10 \
  --min-num-replicas=3 \
  --target-cpu-utilization=0.7 \
  --cool-down-period=300 \
  --zone=us-central1-a
```

---

# Resource Recommendations

## Machine Types by Scale

### Small Scale (< 1000 peers)
- **Coordinator**: n2-standard-2 (2 vCPU, 8 GB)
- **Workers**: n2-standard-4 (4 vCPU, 16 GB) × 3

### Medium Scale (1000-10000 peers)
- **Coordinator**: n2-standard-4 (4 vCPU, 16 GB)
- **Workers**: n2-standard-8 (8 vCPU, 32 GB) × 5-10

### Large Scale (10000+ peers)
- **Coordinator**: n2-standard-8 (8 vCPU, 32 GB)
- **Workers**: n2-highmem-16 (16 vCPU, 128 GB) × 10-20

## Cost Optimization

```bash
# Use preemptible VMs (60-90% cost savings)
--preemptible

# Use spot VMs (even cheaper)
--provisioning-model=SPOT

# Use committed use discounts
# Purchase 1-year or 3-year commitments for 37-57% off

# Use sustained use discounts
# Automatic discounts for running >25% of month
```

---

# Monitoring & Debugging

## Check Physical Node Distribution

### GKE
```bash
# Show which nodes pods are on
kubectl get pods -n protean -o custom-columns=\
POD:.metadata.name,NODE:.spec.nodeName,IP:.status.podIP

# Describe node
kubectl describe node NODE_NAME
```

### GCE
```bash
# List all VMs with internal IPs
gcloud compute instances list \
  --filter="name~protean" \
  --format="table(name,zone,machineType,networkInterfaces[0].networkIP)"
```

## Performance Monitoring

```bash
# GKE: Resource usage
kubectl top nodes
kubectl top pods -n protean

# GCE: SSH and check
gcloud compute ssh protean-worker-0
docker stats
```

## Common Issues

### Workers on Same Node (GKE)

**Fix**: Add pod anti-affinity to worker deployment:

```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchExpressions:
        - key: app
          operator: In
          values:
          - worker
      topologyKey: kubernetes.io/hostname
```

### Network Latency

**Fix**: Deploy in single zone or use regional clusters

```bash
# Single zone (lowest latency)
--zone=us-central1-a

# Regional (high availability, slightly higher latency)
--region=us-central1
```

---

# Next Steps

1. **Choose deployment option** (GKE recommended)
2. **Set up monitoring** (Cloud Logging, Cloud Monitoring)
3. **Configure datasets** (Use GCS for large datasets)
4. **Test scaling** (Start with 3 workers, scale to 10+)
5. **Optimize costs** (Use preemptible VMs for testing)

See specific guides:
- `deploy/gcp/gke/README.md` - Full GKE guide
- `deploy/gcp/gce/README.md` - Multi-VM guide
