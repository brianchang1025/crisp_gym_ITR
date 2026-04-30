#!/bin/bash

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"

# Available Dockerfiles
declare -a DOCKERFILES=("Dockerfile.groot" "Dockerfile.pi05")

echo "======================================"
echo "  Docker Build and Run Script"
echo "======================================"
echo ""

# Menu to select Dockerfile
echo "Available Docker configurations:"
for i in "${!DOCKERFILES[@]}"; do
    echo "$((i+1)). ${DOCKERFILES[$i]}"
done
echo ""

read -p "Select Dockerfile (1-${#DOCKERFILES[@]}): " dockerfile_choice

# Validate choice
if ! [[ "$dockerfile_choice" =~ ^[0-9]+$ ]] || [ "$dockerfile_choice" -lt 1 ] || [ "$dockerfile_choice" -gt ${#DOCKERFILES[@]} ]; then
    echo "Invalid choice. Exiting."
    exit 1
fi

DOCKERFILE="${DOCKERFILES[$((dockerfile_choice-1))]}"
echo "Selected: $DOCKERFILE"
echo ""

# Get image name
read -p "Enter Docker image name (default: crisp-gym): " IMAGE_NAME
IMAGE_NAME="${IMAGE_NAME:-crisp-gym}"

# Get container name
read -p "Enter Docker container name (default: crisp-gym-container): " CONTAINER_NAME
CONTAINER_NAME="${CONTAINER_NAME:-crisp-gym-container}"

echo ""
echo "Configuration Summary:"
echo "  Dockerfile: $DOCKERFILE"
echo "  Image Name: $IMAGE_NAME:latest"
echo "  Container Name: $CONTAINER_NAME"
echo "  Workspace: $WORKSPACE_PATH"
echo ""

read -p "Proceed with build and run? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Building Docker image: $IMAGE_NAME"
docker build -f "$SCRIPT_DIR/$DOCKERFILE" -t "$IMAGE_NAME:latest" "$SCRIPT_DIR/.."

if [ $? -eq 0 ]; then
    echo ""
    echo "Docker image built successfully."
    echo "Running Docker container: $CONTAINER_NAME"
    docker run -it \
        --gpus all \
        --name "$CONTAINER_NAME" \
        -v "$WORKSPACE_PATH:/workspace" \
        -w /workspace \
        "$IMAGE_NAME:latest"
else
    echo "Docker build failed. Exiting."
    exit 1
fi
