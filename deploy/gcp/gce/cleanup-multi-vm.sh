#!/bin/bash
# Cleanup script for GCE multi-VM deployment

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ID="${GCP_PROJECT:-}"
ZONE="${GCP_ZONE:-us-central1-a}"
NETWORK_NAME="protean-network"

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP_PROJECT not set${NC}"
    exit 1
fi

gcloud config set project "$PROJECT_ID"

echo -e "${YELLOW}This will delete ALL protean VMs and network resources${NC}"
read -p "Are you sure? (yes/NO) " -r
if [[ ! $REPLY == "yes" ]]; then
    echo "Aborted"
    exit 0
fi

echo "Deleting VMs..."
gcloud compute instances delete \
    $(gcloud compute instances list --filter="name~protean" --format="value(name)") \
    --zone="$ZONE" \
    --quiet || echo "No VMs to delete"

echo "Deleting firewall rules..."
gcloud compute firewall-rules delete \
    "${NETWORK_NAME}-internal" \
    "${NETWORK_NAME}-ssh" \
    --quiet || echo "No firewall rules to delete"

echo "Deleting network..."
gcloud compute networks delete "$NETWORK_NAME" --quiet || echo "No network to delete"

echo -e "${GREEN}Cleanup complete${NC}"
