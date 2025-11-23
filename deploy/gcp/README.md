# GCP Multi-Machine Deployment

This directory contains scripts and manifests to deploy workers on **separate physical hardware** in Google Cloud Platform.

## ✅ Physical Resource Isolation Guaranteed

Both deployment options ensure each worker runs on separate physical machines:

### Option 1: GKE (Kubernetes) - Recommended
- **Pod anti-affinity** forces workers on different nodes
- Each node = separate physical machine
- Auto-scaling and management built-in
- **Deploy**: `cd gke && ./deploy.sh`

### Option 2: Multi-VM (GCE)
- Each worker = dedicated VM
- Each VM = separate physical machine
- Direct control over machine types
- **Deploy**: `cd gce && ./deploy-multi-vm.sh`

## Quick Start

### GKE Deployment (Recommended for Production)

```bash
cd gke

# Set project
export GCP_PROJECT=your-project-id

# Deploy (creates cluster + builds images + deploys)
./deploy.sh

# Verify physical distribution
kubectl get pods -n protean -o wide
# Each worker should be on different NODE

# Scale workers (auto-distributes across nodes)
kubectl scale statefulset worker --replicas=10 -n protean
```

**Result**: 10 workers across 10 physical machines

### GCE Multi-VM Deployment (Direct Control)

```bash
cd gce

# Deploy 5 workers on 5 separate VMs
export GCP_PROJECT=your-project-id
export NUM_WORKERS=5
./deploy-multi-vm.sh

# Check deployment
gcloud compute instances list --filter="name~protean"

# Clean up
./cleanup-multi-vm.sh
```

**Result**: 1 coordinator VM + 5 worker VMs = 6 physical machines

## Architecture Comparison

### GKE Architecture
```
GKE Cluster
├── Node Pool: coordinator-pool
│   └── Physical Node 1: n2-standard-4
│       └── coordinator-pod
│
└── Node Pool: worker-pool
    ├── Physical Node 2: n2-standard-8 → worker-0
    ├── Physical Node 3: n2-standard-8 → worker-1
    ├── Physical Node 4: n2-standard-8 → worker-2
    └── ...can scale to 20+ nodes
```

### GCE Architecture
```
VPC Network: protean-network
├── VM: protean-coordinator (n2-standard-4)
│   └── Physical Machine 1
│
├── VM: protean-worker-0 (n2-standard-8)
│   └── Physical Machine 2
│
├── VM: protean-worker-1 (n2-standard-8)
│   └── Physical Machine 3
│
└── VM: protean-worker-2 (n2-standard-8)
    └── Physical Machine 4
```

## Resource Scaling Examples

### Small Scale (3 workers)
```bash
# GKE
kubectl scale statefulset worker --replicas=3 -n protean

# GCE
NUM_WORKERS=3 ./deploy-multi-vm.sh
```

**Physical Machines**: 4 total (1 coordinator + 3 workers)
**Total vCPUs**: 28 (4 + 3×8)

### Medium Scale (10 workers)
```bash
# GKE
kubectl scale statefulset worker --replicas=10 -n protean

# GCE
NUM_WORKERS=10 ./deploy-multi-vm.sh
```

**Physical Machines**: 11 total (1 coordinator + 10 workers)
**Total vCPUs**: 84 (4 + 10×8)

### Large Scale (50 workers)
```bash
# GKE - Auto-scales node pool
kubectl scale statefulset worker --replicas=50 -n protean

# GCE - Creates 50 VMs
NUM_WORKERS=50 ./deploy-multi-vm.sh
```

**Physical Machines**: 51 total (1 coordinator + 50 workers)
**Total vCPUs**: 404 (4 + 50×8)

## Cost Estimates

### GKE (Regional, Standard Tier)
```
Coordinator Pool: 1 × n2-standard-4
  ~$120/month

Worker Pool: 3 × n2-standard-8
  ~$450/month (3 × $150)

GKE Management: Free (first cluster)

Total: ~$570/month
```

### GCE (Separate VMs)
```
Coordinator: 1 × n2-standard-4
  ~$100/month

Workers: 3 × n2-standard-8
  ~$450/month (3 × $150)

Networking: ~$10/month

Total: ~$560/month
```

### Cost Optimization
```bash
# Use preemptible VMs (60-90% savings)
--preemptible

# Use spot VMs (even cheaper, more interruptions)
--provisioning-model=SPOT

# Example: 10 preemptible workers
# Normal: ~$1500/month
# Preemptible: ~$300/month
```

## Verification Scripts

### Check Physical Distribution (GKE)
```bash
# Show which node each pod is on
kubectl get pods -n protean -o custom-columns=\
POD:.metadata.name,NODE:.spec.nodeName,IP:.status.podIP

# Verify anti-affinity
kubectl describe statefulset worker -n protean | grep -A10 "Anti-Affinity"

# Count unique nodes
kubectl get pods -n protean -l app=worker -o jsonpath='{.items[*].spec.nodeName}' | \
  tr ' ' '\n' | sort | uniq -c
```

### Check Physical Distribution (GCE)
```bash
# List all VMs with machine types
gcloud compute instances list \
  --filter="name~protean" \
  --format="table(name,zone,machineType,status)"

# Each VM = separate physical machine (GCP guarantee)
```

## Monitoring

### GKE
```bash
# Resource usage
kubectl top nodes
kubectl top pods -n protean

# Logs
kubectl logs -f statefulset/worker -n protean

# Events
kubectl get events -n protean --sort-by='.lastTimestamp'
```

### GCE
```bash
# SSH into VM
gcloud compute ssh protean-worker-0 --zone=us-central1-a

# Check Docker stats
docker stats

# View logs
docker logs -f worker
```

## Troubleshooting

### Workers on Same Node (GKE)

**Check**:
```bash
kubectl get pods -n protean -o wide
```

**Fix**: Ensure anti-affinity is set in `worker-statefulset.yaml`:
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

**Or**: Scale up node pool:
```bash
gcloud container clusters node-pools update worker-pool \
  --cluster=protean-cluster \
  --num-nodes=10 \
  --region=us-central1
```

### VMs Not Communicating (GCE)

**Check firewall**:
```bash
gcloud compute firewall-rules list --filter="name~protean"
```

**Test connectivity**:
```bash
# From worker to coordinator
gcloud compute ssh protean-worker-0 --zone=us-central1-a
ping COORDINATOR_IP
nc -zv COORDINATOR_IP 50050
```

## Files

```
gcp/
├── README.md (this file)
├── GCP_DEPLOYMENT.md (full guide)
│
├── gke/ (Kubernetes deployment)
│   ├── deploy.sh ← Run this
│   ├── namespace.yaml
│   ├── coordinator-deployment.yaml
│   ├── worker-statefulset.yaml ← Anti-affinity configured
│   ├── services.yaml
│   ├── configmaps.yaml
│   └── persistent-volumes.yaml
│
└── gce/ (Multi-VM deployment)
    ├── deploy-multi-vm.sh ← Run this
    └── cleanup-multi-vm.sh
```

## Next Steps

1. **Choose deployment method**:
   - GKE: Better for production, auto-scaling, easier management
   - GCE: Direct control, simpler architecture, predictable costs

2. **Test locally first**: Use docker-compose to validate configs

3. **Start small**: Deploy with 3 workers, verify distribution

4. **Scale up**: Gradually increase to 10, 20, 50+ workers

5. **Monitor costs**: Use GCP cost calculator and billing alerts

6. **Optimize**: Use preemptible VMs for development/testing

For detailed instructions, see:
- **GKE**: `gke/deploy.sh` and `GCP_DEPLOYMENT.md`
- **GCE**: `gce/deploy-multi-vm.sh` and `GCP_DEPLOYMENT.md`
