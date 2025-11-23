#!/bin/bash
# Protean Distributed Simulation - Quick Start Script
# This script helps run the containerized simulation with proper setup

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/output"

# Functions
print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Protean Distributed Simulation${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_prerequisites() {
    echo "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker."
        exit 1
    fi
    print_success "Docker found: $(docker --version)"

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose not found. Please install Docker Compose V2."
        exit 1
    fi
    print_success "Docker Compose found: $(docker compose version)"

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    print_success "Docker daemon is running"
}

check_datasets() {
    echo ""
    echo "Checking dataset files..."

    if [ ! -d "$DATA_DIR" ]; then
        mkdir -p "$DATA_DIR"
        print_warning "Created data directory: $DATA_DIR"
    fi

    MISSING_FILES=0

    if [ ! -f "$DATA_DIR/sift_base.fvecs" ]; then
        print_warning "Missing: $DATA_DIR/sift_base.fvecs"
        MISSING_FILES=1
    else
        SIZE=$(du -h "$DATA_DIR/sift_base.fvecs" | cut -f1)
        print_success "Found base dataset: $SIZE"
    fi

    if [ ! -f "$DATA_DIR/sift_query.fvecs" ]; then
        print_warning "Missing: $DATA_DIR/sift_query.fvecs"
        MISSING_FILES=1
    else
        SIZE=$(du -h "$DATA_DIR/sift_query.fvecs" | cut -f1)
        print_success "Found query dataset: $SIZE"
    fi

    if [ $MISSING_FILES -eq 1 ]; then
        echo ""
        print_warning "Dataset files are missing!"
        echo "Please download SIFT datasets from: http://corpus-texmex.irisa.fr/"
        echo "And place them in: $DATA_DIR/"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

setup_output_dir() {
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
        print_success "Created output directory: $OUTPUT_DIR"
    fi
}

build_images() {
    echo ""
    echo "Building Docker images..."
    echo "This may take a few minutes on first run..."

    cd "$PROJECT_ROOT"

    if docker compose build; then
        print_success "Images built successfully"
    else
        print_error "Failed to build images"
        exit 1
    fi
}

start_services() {
    echo ""
    echo "Starting services..."

    cd "$PROJECT_ROOT"

    # Check if already running
    if docker compose ps | grep -q "Up"; then
        print_warning "Services are already running"
        read -p "Restart services? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker compose down
        else
            return
        fi
    fi

    # Read worker count from coordinator config
    WORKER_COUNT=$(grep -c "worker_id:" "$PROJECT_ROOT/deploy/coordinator_config.yaml" || echo "11")
    echo "Starting coordinator with $WORKER_COUNT workers..."

    # Start in detached mode with correct worker count
    if docker compose up -d --scale worker=$WORKER_COUNT; then
        print_success "Services started"

        echo ""
        echo "Waiting for services to be healthy..."
        sleep 5

        # Check health
        docker compose ps

        echo ""
        print_success "All services are running"
        echo ""
        echo "To view logs:"
        echo "  docker compose logs -f"
        echo ""
        echo "To stop services:"
        echo "  docker compose down"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

view_logs() {
    cd "$PROJECT_ROOT"
    echo "Viewing logs (Ctrl+C to exit)..."
    docker compose logs -f
}

stop_services() {
    cd "$PROJECT_ROOT"
    echo "Stopping services..."
    docker compose down
    print_success "Services stopped"
}

show_status() {
    cd "$PROJECT_ROOT"
    echo "Service Status:"
    echo "==============="
    docker compose ps

    echo ""
    echo "Recent Logs:"
    echo "============"
    docker compose logs --tail=20
}

show_results() {
    echo "Test Results:"
    echo "============="

    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
        ls -lh "$OUTPUT_DIR"
        echo ""

        # Show summary if JSON files exist
        for file in "$OUTPUT_DIR"/*.json; do
            if [ -f "$file" ]; then
                echo "$(basename "$file"):"
                # Try to use jq if available, otherwise just show file
                if command -v jq &> /dev/null; then
                    echo "  Timestamp: $(jq -r '.timestamp_ms // "N/A"' "$file" 2>/dev/null)"
                    echo "  Peers: $(jq -r '.peer_snapshots | length // "N/A"' "$file" 2>/dev/null)"
                else
                    head -3 "$file"
                fi
                echo ""
            fi
        done
    else
        print_warning "No results found in $OUTPUT_DIR"
    fi
}

show_help() {
    cat << EOF
Protean Distributed Simulation - Quick Start

Usage: $0 [COMMAND]

Commands:
  start       Check prerequisites, build images, and start services
  build       Build Docker images only
  up          Start services (assumes images are built)
  down        Stop all services
  logs        View service logs
  status      Show service status and recent logs
  results     Show test results
  clean       Stop services and remove volumes
  help        Show this help message

Examples:
  $0 start       # Full setup and start
  $0 logs        # View logs
  $0 results     # View test results
  $0 down        # Stop services

Environment Variables:
  COMPOSE_FILE   Path to docker-compose.yml (default: ./docker-compose.yml)

EOF
}

# Main script
main() {
    print_header

    case "${1:-start}" in
        start)
            check_prerequisites
            check_datasets
            setup_output_dir
            build_images
            start_services
            ;;
        build)
            check_prerequisites
            build_images
            ;;
        up)
            check_prerequisites
            setup_output_dir
            start_services
            ;;
        down|stop)
            stop_services
            ;;
        logs)
            view_logs
            ;;
        status)
            show_status
            ;;
        results)
            show_results
            ;;
        clean)
            cd "$PROJECT_ROOT"
            echo "Stopping services and removing volumes..."
            docker compose down -v
            print_success "Cleaned up"
            ;;
        help|-h|--help)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
