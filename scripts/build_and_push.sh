#!/bin/bash

# Check if suffix is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <suffix>"
  echo "Example: $0 dev"
  exit 1
fi

SUFFIX=$1
IMAGE_NAME="chuross/acestep-runpod:1.5-$SUFFIX"

echo "Building Docker image: $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" .

if [ $? -ne 0 ]; then
  echo "Docker build failed."
  exit 1
fi

echo "Pushing Docker image: $IMAGE_NAME..."
docker push "$IMAGE_NAME"

if [ $? -ne 0 ]; then
  echo "Docker push failed."
  exit 1
fi

echo "Successfully built and pushed $IMAGE_NAME"
