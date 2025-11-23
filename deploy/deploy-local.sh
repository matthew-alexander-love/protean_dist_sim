#!/bin/bash
# Local deployment script for protean_dist_sim
# Deploys coordinator and workers using Docker Compose

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
WORKER_COUNT=11
DATA_DIR="../data"
CONFIG_DIR="."
OUTPUT_DIR="../output"
ACTION="start"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKER_COUNT="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        start|stop|restart|logs|status)
            ACTION="$1"
            shift
            ;;
        --help)
            echo "Usage: $0 [ACTION] [OPTIONS]"
            echo ""
            echo "Local deployment manager for protean_dist_sim"
            echo ""
            echo "Actions:"
            echo "  start              Start the coordinator and workers (default)"
            echo "  stop               Stop all services"
            echo "  restart            Restart all services"
            echo "  logs               View service logs"
            echo "  status             Show service status"
            echo ""
            echo "Options:"
            echo "  --workers N        Number of worker nodes (default: 11)"
            echo "  --data-dir PATH    Path to dataset directory (default: ../data)"
            echo "  --config-dir PATH  Path to config directory (default: .)"
            echo "  --output-dir PATH  Path to output directory (default: ../output)"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 start --workers 11              # Start with 11 workers"
            echo "  $0 stop                             # Stop all services"
            echo "  $0 logs                             # View logs"
            echo "  $0 status                           # Check status"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to project root directory
cd "$(dirname "$0")/.."

# Validate paths
validate_paths() {
    echo -e "${YELLOW}Validating paths...${NC}"

    # Check data directory
    if [ ! -d "$DATA_DIR" ]; then
        echo -e "${RED}Error: Data directory not found: $DATA_DIR${NC}"
        exit 1
    fi

    # Check dataset files
    if [ ! -f "$DATA_DIR/sift_base.fvecs" ]; then
        echo -e "${RED}Error: Dataset file not found: $DATA_DIR/sift_base.fvecs${NC}"
        echo "Please download the SIFT dataset first"
        exit 1
    fi

    if [ ! -f "$DATA_DIR/sift_query.fvecs" ]; then
        echo -e "${RED}Error: Dataset file not found: $DATA_DIR/sift_query.fvecs${NC}"
        echo "Please download the SIFT dataset first"
        exit 1
    fi

    # Check config files
    if [ ! -f "$CONFIG_DIR/coordinator_config.yaml" ]; then
        echo -e "${RED}Error: Config file not found: $CONFIG_DIR/coordinator_config.yaml${NC}"
        exit 1
    fi

    if [ ! -f "$CONFIG_DIR/test_plan.yaml" ]; then
        echo -e "${RED}Error: Config file not found: $CONFIG_DIR/test_plan.yaml${NC}"
        exit 1
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    echo -e "${GREEN}✓ All paths validated${NC}"
}

# Check Docker images
check_images() {
    echo -e "${YELLOW}Checking Docker images...${NC}"

    if ! docker image inspect protean-dist-sim-coordinator:latest &> /dev/null; then
        echo -e "${RED}Error: Coordinator image not found${NC}"
        echo "Please build images first: ./build.sh"
        exit 1
    fi

    if ! docker image inspect protean-dist-sim-worker:latest &> /dev/null; then
        echo -e "${RED}Error: Worker image not found${NC}"
        echo "Please build images first: ./build.sh"
        exit 1
    fi

    echo -e "${GREEN}✓ Docker images found${NC}"
}

# Start services
start_services() {
    echo -e "${GREEN}==============================================================${NC}"
    echo -e "${GREEN}Starting Protean Distributed Simulation${NC}"
    echo -e "${GREEN}==============================================================${NC}"
    echo ""
    echo "Configuration:"
    echo "  Workers:       $WORKER_COUNT"
    echo "  Data dir:      $DATA_DIR"
    echo "  Config dir:    $CONFIG_DIR"
    echo "  Output dir:    $OUTPUT_DIR"
    echo ""

    validate_paths
    check_images

    echo ""
    echo -e "${YELLOW}Starting services...${NC}"

    # Export environment variables for docker-compose
    export WORKER_COUNT

    # Start services with scaling
    docker compose up -d --scale worker=$WORKER_COUNT

    echo ""
    echo -e "${GREEN}✓ Services started successfully${NC}"
    echo ""
    echo "Next steps:"
    echo "  - View logs:     docker compose logs -f"
    echo "  - Check status:  $0 status"
    echo "  - Stop services: $0 stop"
}

# Stop services
stop_services() {
    echo -e "${YELLOW}Stopping services...${NC}"
    docker compose down
    echo -e "${GREEN}✓ Services stopped${NC}"
}

# Restart services
restart_services() {
    stop_services
    echo ""
    start_services
}

# View logs
view_logs() {
    echo -e "${BLUE}Viewing logs (Ctrl+C to exit)...${NC}"
    docker compose logs -f
}

# Show status
show_status() {
    echo -e "${GREEN}==============================================================${NC}"
    echo -e "${GREEN}Service Status${NC}"
    echo -e "${GREEN}==============================================================${NC}"
    echo ""

    docker compose ps

    echo ""
    echo -e "${GREEN}==============================================================${NC}"
    echo -e "${GREEN}Recent Logs${NC}"
    echo -e "${GREEN}==============================================================${NC}"
    echo ""

    docker compose logs --tail=20
}

# Execute action
case $ACTION in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        view_logs
        ;;
    status)
        show_status
        ;;
    *)
        echo -e "${RED}Error: Unknown action $ACTION${NC}"
        exit 1
        ;;
esac
