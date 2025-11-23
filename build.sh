#!/bin/bash
# Build script for protean_dist_sim Docker images
# Builds coordinator and worker images for local and cloud deployment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Configuration
DOCKERFILE="$PROJECT_ROOT/Dockerfile"
BUILD_CONTEXT="$PROJECT_ROOT/.."
COORDINATOR_IMAGE="protean-dist-sim-coordinator"
WORKER_IMAGE="protean-dist-sim-worker"
TAG="${TAG:-latest}"
PUSH="${PUSH:-false}"
REGISTRY="${REGISTRY:-}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --push)
            PUSH="true"
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Build Docker images for protean_dist_sim coordinator and worker"
            echo ""
            echo "Options:"
            echo "  --tag TAG          Image tag (default: latest)"
            echo "  --push             Push images to registry after build"
            echo "  --registry URL     Container registry URL (e.g., gcr.io/project-id)"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Build with default settings"
            echo "  $0 --tag v1.0.0                       # Build with specific tag"
            echo "  $0 --registry gcr.io/my-project --push # Build and push to GCR"
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

echo -e "${GREEN}==============================================================${NC}"
echo -e "${GREEN}Building protean_dist_sim Docker Images${NC}"
echo -e "${GREEN}==============================================================${NC}"
echo ""
echo "Coordinator image: ${COORDINATOR_IMAGE}:${TAG}"
echo "Worker image:      ${WORKER_IMAGE}:${TAG}"
echo ""

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${RED}Error: Dockerfile not found at $DOCKERFILE${NC}"
    exit 1
fi

# Build coordinator image
echo -e "${YELLOW}Building coordinator image...${NC}"
docker build \
    --target coordinator \
    --tag "${COORDINATOR_IMAGE}:${TAG}" \
    --tag "${COORDINATOR_IMAGE}:latest" \
    -f "$DOCKERFILE" \
    "$BUILD_CONTEXT"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Coordinator image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build coordinator image${NC}"
    exit 1
fi

echo ""

# Build worker image
echo -e "${YELLOW}Building worker image...${NC}"
docker build \
    --target worker \
    --tag "${WORKER_IMAGE}:${TAG}" \
    --tag "${WORKER_IMAGE}:latest" \
    -f "$DOCKERFILE" \
    "$BUILD_CONTEXT"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Worker image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build worker image${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}==============================================================${NC}"
echo -e "${GREEN}Build Summary${NC}"
echo -e "${GREEN}==============================================================${NC}"

# Show image sizes
docker images | grep -E "REPOSITORY|${COORDINATOR_IMAGE}|${WORKER_IMAGE}" | grep -E "REPOSITORY|${TAG}|latest"

# Push images if requested
if [ "$PUSH" = "true" ]; then
    echo ""
    echo -e "${YELLOW}Pushing images to registry...${NC}"

    if [ -z "$REGISTRY" ]; then
        echo -e "${RED}Error: --registry must be specified when using --push${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Pushing ${COORDINATOR_IMAGE}:${TAG}...${NC}"
    docker push "${COORDINATOR_IMAGE}:${TAG}"

    echo -e "${YELLOW}Pushing ${WORKER_IMAGE}:${TAG}...${NC}"
    docker push "${WORKER_IMAGE}:${TAG}"

    echo -e "${GREEN}✓ Images pushed successfully${NC}"
fi

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "Next steps:"
echo "  - Local deployment:  ./deploy/deploy-local.sh --workers 11"
echo "  - Cloud deployment:  ./deploy/deploy-cloud.sh --help"
