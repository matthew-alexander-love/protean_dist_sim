#!/bin/bash
# GKE Deployment Script for Protean Distributed Simulation
# Deploys coordinator and workers across separate physical nodes

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
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
CLUSTER_NAME="${CLUSTER_NAME:-protean-cluster}"
NUM_WORKERS="${NUM_WORKERS:-3}"

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    print_success "gcloud CLI found"

    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Installing..."
        gcloud components install kubectl
    fi
    print_success "kubectl found"

    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker."
        exit 1
    fi
    print_success "Docker found"

    if [ -z "$PROJECT_ID" ]; then
        print_error "GCP_PROJECT not set"
        echo "Usage: GCP_PROJECT=your-project ./deploy.sh"
        exit 1
    fi

    gcloud config set project "$PROJECT_ID"
    print_success "Using project: $PROJECT_ID"
}

# Create GKE cluster with separate node pools
create_cluster() {
    print_header "Creating GKE Cluster"

    # Check if cluster exists
    if gcloud container clusters describe "$CLUSTER_NAME" --region="$REGION" &>/dev/null; then
        print_warning "Cluster $CLUSTER_NAME already exists"
        read -p "Delete and recreate? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            gcloud container clusters delete "$CLUSTER_NAME" --region="$REGION" --quiet
        else
            print_success "Using existing cluster"
            return 0
        fi
    fi

    echo "Creating cluster with coordinator node pool..."
    gcloud container clusters create "$CLUSTER_NAME" \
        --region="$REGION" \
        --num-nodes=1 \
        --machine-type=n2-standard-4 \
        --node-labels=role=coordinator \
        --disk-size=50 \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=1 \
        --node-pool-name=coordinator-pool \
        --enable-stackdriver-kubernetes \
        --no-enable-cloud-logging \
        --no-enable-cloud-monitoring

    print_success "Coordinator node pool created"

    echo "Adding worker node pool..."
    gcloud container clusters node-pools create worker-pool \
        --cluster="$CLUSTER_NAME" \
        --region="$REGION" \
        --num-nodes="$NUM_WORKERS" \
        --machine-type=n2-standard-8 \
        --node-labels=role=worker \
        --disk-size=100 \
        --enable-autoscaling \
        --min-nodes="$NUM_WORKERS" \
        --max-nodes=20

    print_success "Worker node pool created with $NUM_WORKERS nodes"

    # Get credentials
    gcloud container clusters get-credentials "$CLUSTER_NAME" --region="$REGION"
    print_success "Cluster credentials configured"
}

# Build and push Docker images
build_and_push_images() {
    print_header "Building and Pushing Images"

    # Configure Docker for GCR
    gcloud auth configure-docker --quiet

    cd ../../../

    # Build coordinator
    echo "Building coordinator image..."
    docker build \
        -t "gcr.io/$PROJECT_ID/protean-coordinator:latest" \
        --target coordinator \
        -f Dockerfile \
        ..

    # Push coordinator
    echo "Pushing coordinator image to GCR..."
    docker push "gcr.io/$PROJECT_ID/protean-coordinator:latest"
    print_success "Coordinator image pushed"

    # Build worker
    echo "Building worker image..."
    docker build \
        -t "gcr.io/$PROJECT_ID/protean-worker:latest" \
        --target worker \
        -f Dockerfile \
        ..

    # Push worker
    echo "Pushing worker image to GCR..."
    docker push "gcr.io/$PROJECT_ID/protean-worker:latest"
    print_success "Worker image pushed"

    cd deploy/gcp/gke
}

# Deploy to Kubernetes
deploy_to_k8s() {
    print_header "Deploying to Kubernetes"

    # Update image references in manifests
    echo "Updating image references..."
    sed -i.bak "s|gcr.io/PROJECT_ID|gcr.io/$PROJECT_ID|g" *.yaml
    rm -f *.yaml.bak

    # Create namespace
    echo "Creating namespace..."
    kubectl apply -f namespace.yaml
    print_success "Namespace created"

    # Create ConfigMaps from local files
    echo "Creating ConfigMaps..."
    kubectl create configmap coordinator-config \
        --from-file=coordinator_config.yaml=../../coordinator_config.yaml \
        -n protean \
        --dry-run=client -o yaml | kubectl apply -f -

    kubectl create configmap test-plan \
        --from-file=test_plan.yaml=../../test_plan.yaml \
        -n protean \
        --dry-run=client -o yaml | kubectl apply -f -

    print_success "ConfigMaps created"

    # Create PVCs
    echo "Creating Persistent Volume Claims..."
    kubectl apply -f persistent-volumes.yaml
    print_success "PVCs created"

    # Deploy services
    echo "Creating services..."
    kubectl apply -f services.yaml
    print_success "Services created"

    # Deploy coordinator
    echo "Deploying coordinator..."
    kubectl apply -f coordinator-deployment.yaml
    print_success "Coordinator deployed"

    # Update worker StatefulSet with correct replica count
    sed -i.bak "s/replicas: 3/replicas: $NUM_WORKERS/" worker-statefulset.yaml
    sed -i.bak "s/--n-workers=3/--n-workers=$NUM_WORKERS/" worker-statefulset.yaml
    rm -f worker-statefulset.yaml.bak

    # Deploy workers
    echo "Deploying workers..."
    kubectl apply -f worker-statefulset.yaml
    print_success "Workers deployed"
}

# Wait for pods to be ready
wait_for_pods() {
    print_header "Waiting for Pods"

    echo "Waiting for coordinator..."
    kubectl wait --for=condition=ready pod \
        -l app=coordinator \
        -n protean \
        --timeout=300s

    print_success "Coordinator ready"

    echo "Waiting for workers..."
    kubectl wait --for=condition=ready pod \
        -l app=worker \
        -n protean \
        --timeout=300s

    print_success "All workers ready"
}

# Verify deployment
verify_deployment() {
    print_header "Verifying Deployment"

    echo ""
    echo "Pod Distribution:"
    kubectl get pods -n protean -o wide

    echo ""
    echo "Checking node distribution..."
    UNIQUE_NODES=$(kubectl get pods -n protean -l app=worker -o jsonpath='{.items[*].spec.nodeName}' | tr ' ' '\n' | sort | uniq | wc -l)
    TOTAL_WORKERS=$(kubectl get pods -n protean -l app=worker --no-headers | wc -l)

    if [ "$UNIQUE_NODES" -eq "$TOTAL_WORKERS" ]; then
        print_success "✓ Each worker is on a separate physical node!"
        print_success "  Total workers: $TOTAL_WORKERS"
        print_success "  Unique nodes: $UNIQUE_NODES"
    else
        print_warning "! Workers are not fully distributed"
        print_warning "  Total workers: $TOTAL_WORKERS"
        print_warning "  Unique nodes: $UNIQUE_NODES"
        echo ""
        echo "This may happen if there aren't enough nodes in the cluster."
        echo "Consider scaling the worker node pool."
    fi

    echo ""
    echo "Services:"
    kubectl get svc -n protean

    echo ""
    echo "Resource Usage:"
    kubectl top nodes || print_warning "Metrics server may not be installed"
}

# Show next steps
show_next_steps() {
    print_header "Deployment Complete!"

    cat << EOF

Next Steps:
===========

1. View logs:
   kubectl logs -f deployment/coordinator -n protean
   kubectl logs -f statefulset/worker -n protean

2. Check status:
   kubectl get pods -n protean -o wide

3. Scale workers:
   kubectl scale statefulset worker --replicas=10 -n protean

4. Get coordinator external IP (if LoadBalancer created):
   kubectl get svc coordinator-external -n protean

5. Monitor:
   kubectl top pods -n protean
   kubectl top nodes

6. Shell into pods:
   kubectl exec -it deployment/coordinator -n protean -- /bin/bash
   kubectl exec -it worker-0 -n protean -- /bin/bash

7. Delete deployment:
   kubectl delete namespace protean
   gcloud container clusters delete $CLUSTER_NAME --region=$REGION

For more info, see deploy/gcp/GCP_DEPLOYMENT.md

EOF
}

# Main execution
main() {
    check_prerequisites
    create_cluster
    build_and_push_images
    deploy_to_k8s
    wait_for_pods
    verify_deployment
    show_next_steps
}

# Run main
main "$@"
