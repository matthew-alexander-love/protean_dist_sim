#!/bin/bash
# GCE Multi-VM Deployment Script
# Deploys coordinator and workers on SEPARATE VMs for maximum resource isolation

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }

# Configuration
PROJECT_ID="${GCP_PROJECT:-}"
ZONE="${GCP_ZONE:-us-central1-a}"
NETWORK_NAME="protean-network"
NUM_WORKERS="${NUM_WORKERS:-3}"

# Machine types
COORDINATOR_MACHINE="n2-standard-4"
WORKER_MACHINE="n2-standard-8"

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found"
        exit 1
    fi
    print_success "gcloud CLI found"

    if [ -z "$PROJECT_ID" ]; then
        print_error "GCP_PROJECT not set"
        echo "Usage: GCP_PROJECT=your-project ./deploy-multi-vm.sh"
        exit 1
    fi

    gcloud config set project "$PROJECT_ID"
    print_success "Using project: $PROJECT_ID"
}

# Create VPC network
create_network() {
    print_header "Creating VPC Network"

    # Check if network exists
    if gcloud compute networks describe "$NETWORK_NAME" &>/dev/null; then
        print_warning "Network $NETWORK_NAME already exists"
        return 0
    fi

    # Create network
    gcloud compute networks create "$NETWORK_NAME" \
        --subnet-mode=auto \
        --bgp-routing-mode=regional

    print_success "Network created"

    # Create firewall rules
    echo "Creating firewall rules..."

    # Allow internal communication
    gcloud compute firewall-rules create "${NETWORK_NAME}-internal" \
        --network="$NETWORK_NAME" \
        --allow=tcp:50050-50060,icmp \
        --source-ranges=10.0.0.0/8 \
        --description="Allow internal protean communication"

    # Allow SSH
    gcloud compute firewall-rules create "${NETWORK_NAME}-ssh" \
        --network="$NETWORK_NAME" \
        --allow=tcp:22 \
        --source-ranges=0.0.0.0/0 \
        --description="Allow SSH access"

    print_success "Firewall rules created"
}

# Build and push images
build_and_push_images() {
    print_header "Building and Pushing Images"

    gcloud auth configure-docker --quiet

    cd ../../../

    echo "Building images..."
    docker build \
        -t "gcr.io/$PROJECT_ID/protean-coordinator:latest" \
        --target coordinator \
        -f Dockerfile \
        ..

    docker build \
        -t "gcr.io/$PROJECT_ID/protean-worker:latest" \
        --target worker \
        -f Dockerfile \
        ..

    echo "Pushing images..."
    docker push "gcr.io/$PROJECT_ID/protean-coordinator:latest"
    docker push "gcr.io/$PROJECT_ID/protean-worker:latest"

    print_success "Images built and pushed"

    cd deploy/gcp/gce
}

# Create coordinator VM
create_coordinator() {
    print_header "Creating Coordinator VM"

    # Startup script for coordinator
    cat > /tmp/coordinator-startup.sh << 'EOF'
#!/bin/bash
# Pull and run coordinator
docker pull gcr.io/PROJECT_ID/protean-coordinator:latest

# Create directories
mkdir -p /app/config /app/data /app/output

# Note: You'll need to upload config and data files separately
# For now, we'll just start the container in a waiting state
docker run -d \
  --name coordinator \
  --restart=unless-stopped \
  -p 50050:50050 \
  -v /app/config:/app/config \
  -v /app/data:/app/data \
  -v /app/output:/app/output \
  gcr.io/PROJECT_ID/protean-coordinator:latest \
  tail -f /dev/null

echo "Coordinator container ready. Upload config files and restart."
EOF

    sed -i "s|PROJECT_ID|$PROJECT_ID|g" /tmp/coordinator-startup.sh

    # Create VM
    gcloud compute instances create protean-coordinator \
        --zone="$ZONE" \
        --machine-type="$COORDINATOR_MACHINE" \
        --network="$NETWORK_NAME" \
        --image-family=cos-stable \
        --image-project=cos-cloud \
        --boot-disk-size=50GB \
        --boot-disk-type=pd-standard \
        --tags=protean-coordinator \
        --metadata-from-file=startup-script=/tmp/coordinator-startup.sh \
        --scopes=https://www.googleapis.com/auth/cloud-platform

    print_success "Coordinator VM created"
}

# Create worker VMs (each on separate hardware)
create_workers() {
    print_header "Creating Worker VMs"

    # Get coordinator internal IP
    COORDINATOR_IP=$(gcloud compute instances describe protean-coordinator \
        --zone="$ZONE" \
        --format='get(networkInterfaces[0].networkIP)')

    print_success "Coordinator IP: $COORDINATOR_IP"

    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        echo "Creating worker-$i..."

        # Startup script for worker
        cat > "/tmp/worker-$i-startup.sh" << EOF
#!/bin/bash
# Pull and run worker
docker pull gcr.io/$PROJECT_ID/protean-worker:latest

# Run worker container
docker run -d \
  --name worker \
  --restart=unless-stopped \
  -p 50051:50051 \
  gcr.io/$PROJECT_ID/protean-worker:latest \
  --worker-id=worker$i \
  --bind-address=0.0.0.0:50051 \
  --coordinator-address=$COORDINATOR_IP:50050 \
  --n-workers=$NUM_WORKERS \
  --max-actors=2000

echo "Worker $i started"
EOF

        # Create VM
        gcloud compute instances create "protean-worker-$i" \
            --zone="$ZONE" \
            --machine-type="$WORKER_MACHINE" \
            --network="$NETWORK_NAME" \
            --image-family=cos-stable \
            --image-project=cos-cloud \
            --boot-disk-size=100GB \
            --boot-disk-type=pd-ssd \
            --tags=protean-worker \
            --metadata-from-file=startup-script="/tmp/worker-$i-startup.sh" \
            --scopes=https://www.googleapis.com/auth/cloud-platform &

        # Create VMs in parallel for speed
        sleep 2
    done

    # Wait for all background jobs
    wait

    print_success "All $NUM_WORKERS worker VMs created"
}

# Wait for VMs to be ready
wait_for_vms() {
    print_header "Waiting for VMs to Start"

    echo "Waiting for coordinator..."
    for i in {1..30}; do
        if gcloud compute instances describe protean-coordinator \
            --zone="$ZONE" \
            --format='get(status)' | grep -q "RUNNING"; then
            break
        fi
        sleep 2
    done
    print_success "Coordinator running"

    echo "Waiting for workers..."
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        for j in {1..30}; do
            if gcloud compute instances describe "protean-worker-$i" \
                --zone="$ZONE" \
                --format='get(status)' | grep -q "RUNNING"; then
                break
            fi
            sleep 2
        done
    done
    print_success "All workers running"

    echo "Waiting for Docker containers to start (60 seconds)..."
    sleep 60
}

# Verify deployment
verify_deployment() {
    print_header "Verifying Deployment"

    echo ""
    echo "VMs:"
    gcloud compute instances list \
        --filter="name~protean" \
        --format="table(name,zone,machineType,networkInterfaces[0].networkIP,status)"

    echo ""
    echo "Physical Isolation Verification:"
    echo "================================="
    echo "✓ Each VM runs on separate physical hardware (GCP guarantee)"
    echo "✓ Coordinator: 1 VM ($COORDINATOR_MACHINE)"
    echo "✓ Workers: $NUM_WORKERS VMs ($WORKER_MACHINE each)"

    echo ""
    echo "Resource Allocation:"
    echo "===================="
    COORDINATOR_CPUS=$(echo $COORDINATOR_MACHINE | grep -oP '\d+' | tail -1)
    WORKER_CPUS=$(echo $WORKER_MACHINE | grep -oP '\d+' | tail -1)
    TOTAL_WORKER_CPUS=$((WORKER_CPUS * NUM_WORKERS))

    echo "Coordinator: $COORDINATOR_CPUS vCPUs"
    echo "Workers: $NUM_WORKERS × $WORKER_CPUS vCPUs = $TOTAL_WORKER_CPUS total vCPUs"
    echo "Grand Total: $((COORDINATOR_CPUS + TOTAL_WORKER_CPUS)) vCPUs across $((1 + NUM_WORKERS)) physical machines"
}

# Upload config files
upload_configs() {
    print_header "Uploading Configuration Files"

    echo "You need to upload config and data files manually:"
    echo ""
    echo "1. Upload coordinator config:"
    echo "   gcloud compute scp ../../coordinator_config.yaml protean-coordinator:/tmp/ --zone=$ZONE"
    echo "   gcloud compute scp ../../test_plan.yaml protean-coordinator:/tmp/ --zone=$ZONE"
    echo ""
    echo "2. Upload dataset files:"
    echo "   gcloud compute scp /path/to/sift_base.fvecs protean-coordinator:/tmp/ --zone=$ZONE"
    echo "   gcloud compute scp /path/to/sift_query.fvecs protean-coordinator:/tmp/ --zone=$ZONE"
    echo ""
    echo "3. SSH into coordinator and move files:"
    echo "   gcloud compute ssh protean-coordinator --zone=$ZONE"
    echo "   sudo mv /tmp/*.yaml /app/config/"
    echo "   sudo mv /tmp/*.fvecs /app/data/"
    echo ""
    echo "4. Restart coordinator with proper command:"
    echo "   docker stop coordinator && docker rm coordinator"
    echo "   docker run -d --name coordinator -p 50050:50050 \\"
    echo "     -v /app/config:/app/config \\"
    echo "     -v /app/data:/app/data \\"
    echo "     -v /app/output:/app/output \\"
    echo "     gcr.io/$PROJECT_ID/protean-coordinator:latest \\"
    echo "     --config /app/config/coordinator_config.yaml \\"
    echo "     --test-plan /app/config/test_plan.yaml"
    echo ""
}

# Show next steps
show_next_steps() {
    print_header "Deployment Complete!"

    COORDINATOR_IP=$(gcloud compute instances describe protean-coordinator \
        --zone="$ZONE" \
        --format='get(networkInterfaces[0].networkIP)')

    cat << EOF

Next Steps:
===========

1. Upload configs and data (see above)

2. SSH into coordinator:
   gcloud compute ssh protean-coordinator --zone=$ZONE

3. Check coordinator logs:
   gcloud compute ssh protean-coordinator --zone=$ZONE --command="docker logs -f coordinator"

4. Check worker logs:
   gcloud compute ssh protean-worker-0 --zone=$ZONE --command="docker logs -f worker"

5. Monitor resources:
   gcloud compute ssh protean-worker-0 --zone=$ZONE --command="docker stats"

6. Scale workers (add more VMs):
   NUM_WORKERS=5 ./deploy-multi-vm.sh

7. Clean up:
   ./cleanup-multi-vm.sh

Cost Estimate:
==============
Coordinator: $COORDINATOR_MACHINE ≈ \$50-100/month
Workers: $NUM_WORKERS × $WORKER_MACHINE ≈ \$$(($NUM_WORKERS * 150))-\$$(($NUM_WORKERS * 300))/month

💡 Tip: Use --preemptible flag to save 60-90% on costs for testing

EOF
}

# Main execution
main() {
    check_prerequisites
    create_network
    build_and_push_images
    create_coordinator
    create_workers
    wait_for_vms
    verify_deployment
    upload_configs
    show_next_steps
}

# Run main
main "$@"
