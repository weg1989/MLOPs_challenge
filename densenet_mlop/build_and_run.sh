#!/bin/bash
set -e

# Default arguments
OUTPUT_DIR=${1:-"./results"}
GPU_ENABLED=${2:-"true"}

echo ">>> Building Docker image..."
docker compose build

echo ">>> Running benchmark suite..."
docker compose run --rm \
  -e OUTPUT_DIR=$OUTPUT_DIR \
  -e GPU_ENABLED=$GPU_ENABLED \
  benchmark

echo ">>> Starting TensorBoard..."
docker compose up -d tensorboard

echo ">>> Done!"
echo "Results saved in $OUTPUT_DIR/benchmark_results.csv"
echo "TensorBoard running at http://localhost:6006"
