#!/bin/bash
# Cloud deployment script for protean_dist_sim
# Generic script for deploying to any cloud VMs with Docker

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
WORKER_COUNT=11
COORDINATOR_HOST=""
CLOUD_MOUNT_PATH="/mnt/data"
COORDINATOR_IMAGE="protean-dist-sim-coordinator:latest"
WORKER_IMAGE="protean-dist-sim-worker:latest"
NETWORK_NAME="protean-network"
ACTION="deploy"
REGISTRY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKER_COUNT="$2"
            shift 2
            ;;
        --coordinator-host)
            COORDINATOR_HOST="$2"
            shift 2
            ;;
        --cloud-mount)
            CLOUD_MOUNT_PATH="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        deploy|coordinator|worker|stop|status)
            ACTION="$1"
            shift
            ;;
        --help)
            echo "Usage: $0 [ACTION] [OPTIONS]"
            echo ""
            echo "Cloud deployment manager for protean_dist_sim"
            echo ""
            echo "Actions:"
            echo "  deploy             Deploy both coordinator and workers (default)"
            echo "  coordinator        Deploy coordinator only"
            echo "  worker             Deploy worker only (requires --coordinator-host)"
            echo "  stop               Stop all services"
            echo "  status             Show service status"
            echo ""
            echo "Options:"
            echo "  --workers N            Number of worker nodes (default: 11)"
            echo "  --coordinator-host IP  Coordinator host/IP (required for worker action)"
            echo "  --cloud-mount PATH     Cloud storage mount path (default: /mnt/data)"
            echo "  --registry URL         Container registry URL (e.g., gcr.io/project-id)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Prerequisites:"
            echo "  - Docker installed on all VMs"
            echo "  - Cloud storage mounted at CLOUD_MOUNT_PATH containing:"
            echo "      - sift_base.fvecs"
            echo "      - sift_query.fvecs"
            echo "      - coordinator_config.yaml"
            echo "      - test_plan.yaml"
            echo "  - Network connectivity between coordinator and workers"
            echo "  - Docker images built and available (locally or in registry)"
            echo ""
            echo "Examples:"
            echo "  # On coordinator VM:"
            echo "  $0 coordinator --cloud-mount /mnt/gcs-data"
            echo ""
            echo "  # On each worker VM:"
            echo "  $0 worker --coordinator-host 10.0.1.5 --cloud-mount /mnt/gcs-data"
            echo ""
            echo "  # Deploy everything on single VM:"
            echo "  $0 deploy --workers 11 --cloud-mount /mnt/data"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Add registry prefix if specified
if [ -n "$REGISTRY" ]; then
    COORDINATOR_IMAGE="${REGISTRY}/${COORDINATOR_IMAGE}"
    WORKER_IMAGE="${REGISTRY}/${WORKER_IMAGE}"
fi

# Validate prerequisites
validate_prerequisites() {
    echo -e "${YELLOW}Validating prerequisites...${NC}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi

    # Check cloud mount
    if [ ! -d "$CLOUD_MOUNT_PATH" ]; then
        echo -e "${RED}Error: Cloud mount path not found: $CLOUD_MOUNT_PATH${NC}"
        echo "Please ensure cloud storage is mounted at this path"
        exit 1
    fi

    echo -e "${GREEN}✓ Prerequisites validated${NC}"
}

# Validate cloud storage contents
validate_cloud_storage() {
    echo -e "${YELLOW}Validating cloud storage contents...${NC}"

    # Check dataset files
    if [ ! -f "$CLOUD_MOUNT_PATH/sift_base.fvecs" ]; then
        echo -e "${RED}Error: Dataset file not found: $CLOUD_MOUNT_PATH/sift_base.fvecs${NC}"
        exit 1
    fi

    if [ ! -f "$CLOUD_MOUNT_PATH/sift_query.fvecs" ]; then
        echo -e "${RED}Error: Dataset file not found: $CLOUD_MOUNT_PATH/sift_query.fvecs${NC}"
        exit 1
    fi

    # Check config files
    if [ ! -f "$CLOUD_MOUNT_PATH/coordinator_config.yaml" ]; then
        echo -e "${RED}Error: Config file not found: $CLOUD_MOUNT_PATH/coordinator_config.yaml${NC}"
        exit 1
    fi

    if [ ! -f "$CLOUD_MOUNT_PATH/test_plan.yaml" ]; then
        echo -e "${RED}Error: Config file not found: $CLOUD_MOUNT_PATH/test_plan.yaml${NC}"
        exit 1
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$CLOUD_MOUNT_PATH/output"

    echo -e "${GREEN}✓ Cloud storage validated${NC}"
}

# Pull images from registry
pull_images() {
    if [ -n "$REGISTRY" ]; then
        echo -e "${YELLOW}Pulling images from registry...${NC}"

        if [ "$ACTION" = "coordinator" ] || [ "$ACTION" = "deploy" ]; then
            docker pull "$COORDINATOR_IMAGE"
        fi

        if [ "$ACTION" = "worker" ] || [ "$ACTION" = "deploy" ]; then
            docker pull "$WORKER_IMAGE"
        fi

        echo -e "${GREEN}✓ Images pulled${NC}"
    else
        echo -e "${YELLOW}Using local images (no registry specified)${NC}"
    fi
}

# Create Docker network if it doesn't exist
create_network() {
    if ! docker network inspect "$NETWORK_NAME" &> /dev/null; then
        echo -e "${YELLOW}Creating Docker network: $NETWORK_NAME${NC}"
        docker network create "$NETWORK_NAME"
        echo -e "${GREEN}✓ Network created${NC}"
    else
        echo -e "${GREEN}✓ Network already exists${NC}"
    fi
}

# Deploy coordinator
deploy_coordinator() {
    echo -e "${GREEN}==============================================================${NC}"
    echo -e "${GREEN}Deploying Coordinator${NC}"
    echo -e "${GREEN}==============================================================${NC}"
    echo ""

    validate_prerequisites
    validate_cloud_storage
    pull_images
    create_network

    echo -e "${YELLOW}Starting coordinator container...${NC}"

    # Stop and remove existing coordinator if running
    docker rm -f protean-coordinator 2>/dev/null || true

    # Start coordinator
    docker run -d \
        --name protean-coordinator \
        --hostname coordinator \
        --network "$NETWORK_NAME" \
        -p 50050:50050 \
        -v "$CLOUD_MOUNT_PATH:/app/data:ro" \
        -v "$CLOUD_MOUNT_PATH/coordinator_config.yaml:/app/config/coordinator_config.yaml:ro" \
        -v "$CLOUD_MOUNT_PATH/test_plan.yaml:/app/config/test_plan.yaml:ro" \
        -v "$CLOUD_MOUNT_PATH/output:/app/output" \
        -e RUST_LOG=info \
        -e RUST_BACKTRACE=1 \
        --restart unless-stopped \
        "$COORDINATOR_IMAGE"

    echo ""
    echo -e "${GREEN}✓ Coordinator deployed successfully${NC}"
    echo ""
    echo "Coordinator details:"
    echo "  Container:     protean-coordinator"
    echo "  Port:          50050"
    echo "  Data mount:    $CLOUD_MOUNT_PATH"
    echo ""
    echo "To view logs:  docker logs -f protean-coordinator"
    echo "To check status: docker ps | grep coordinator"
}

# Deploy worker
deploy_worker() {
    if [ -z "$COORDINATOR_HOST" ] && [ "$ACTION" != "deploy" ]; then
        echo -e "${RED}Error: --coordinator-host is required when deploying workers${NC}"
        exit 1
    fi

    echo -e "${GREEN}==============================================================${NC}"
    echo -e "${GREEN}Deploying Workers${NC}"
    echo -e "${GREEN}==============================================================${NC}"
    echo ""

    validate_prerequisites
    pull_images
    create_network

    # Determine coordinator address
    if [ "$ACTION" = "deploy" ]; then
        # Same VM deployment - use container name
        COORD_ADDR="coordinator:50050"
    else
        # Multi-VM deployment - use provided host
        COORD_ADDR="${COORDINATOR_HOST}:50050"
    fi

    echo "Configuration:"
    echo "  Worker count:      $WORKER_COUNT"
    echo "  Coordinator addr:  $COORD_ADDR"
    echo ""
    echo -e "${YELLOW}Starting worker containers...${NC}"

    # Start workers
    for i in $(seq 0 $((WORKER_COUNT - 1))); do
        WORKER_NAME="protean-worker$i"
        WORKER_ID="worker$i"
        WORKER_PORT=$((50051 + i))

        # Stop and remove existing worker if running
        docker rm -f "$WORKER_NAME" 2>/dev/null || true

        # Start worker
        docker run -d \
            --name "$WORKER_NAME" \
            --hostname "$WORKER_ID" \
            --network "$NETWORK_NAME" \
            -p "$WORKER_PORT:50051" \
            -e RUST_LOG=info \
            -e RUST_BACKTRACE=1 \
            --restart unless-stopped \
            "$WORKER_IMAGE" \
            --worker-id="$WORKER_ID" \
            --bind-address=0.0.0.0:50051 \
            --coordinator-address="$COORD_ADDR" \
            --n-workers="$WORKER_COUNT" \
            --max-actors=1000

        echo -e "${GREEN}✓${NC} Started $WORKER_NAME (port $WORKER_PORT)"
    done

    echo ""
    echo -e "${GREEN}✓ All workers deployed successfully${NC}"
    echo ""
    echo "Worker details:"
    echo "  Count:         $WORKER_COUNT"
    echo "  Ports:         50051-$((50050 + WORKER_COUNT))"
    echo ""
    echo "To view logs:  docker logs -f protean-worker0"
    echo "To check status: docker ps | grep worker"
}

# Deploy full stack
deploy_full() {
    echo -e "${GREEN}==============================================================${NC}"
    echo -e "${GREEN}Deploying Full Stack${NC}"
    echo -e "${GREEN}==============================================================${NC}"
    echo ""

    deploy_coordinator
    echo ""
    deploy_worker

    echo ""
    echo -e "${GREEN}==============================================================${NC}"
    echo -e "${GREEN}Deployment Complete${NC}"
    echo -e "${GREEN}==============================================================${NC}"
}

# Stop all services
stop_services() {
    echo -e "${YELLOW}Stopping all services...${NC}"

    # Stop coordinator
    docker rm -f protean-coordinator 2>/dev/null || true

    # Stop all workers
    docker ps -a | grep protean-worker | awk '{print $1}' | xargs -r docker rm -f

    echo -e "${GREEN}✓ All services stopped${NC}"
}

# Show status
show_status() {
    echo -e "${GREEN}==============================================================${NC}"
    echo -e "${GREEN}Service Status${NC}"
    echo -e "${GREEN}==============================================================${NC}"
    echo ""

    docker ps -a | grep -E "CONTAINER|protean-" || echo "No services running"

    echo ""
    echo -e "${GREEN}==============================================================${NC}"
    echo -e "${GREEN}Network Status${NC}"
    echo -e "${GREEN}==============================================================${NC}"
    echo ""

    if docker network inspect "$NETWORK_NAME" &> /dev/null; then
        docker network inspect "$NETWORK_NAME" | grep -A 20 "Containers"
    else
        echo "Network $NETWORK_NAME not found"
    fi
}

# Execute action
case $ACTION in
    deploy)
        deploy_full
        ;;
    coordinator)
        deploy_coordinator
        ;;
    worker)
        deploy_worker
        ;;
    stop)
        stop_services
        ;;
    status)
        show_status
        ;;
    *)
        echo -e "${RED}Error: Unknown action $ACTION${NC}"
        exit 1
        ;;
esac
