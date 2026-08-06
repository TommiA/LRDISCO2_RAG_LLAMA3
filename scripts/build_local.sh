#!/bin/bash

set -e

source .env

echo "Downloading ChromaDB contents..."

rm -rf chromadb_contents
mkdir -p chromadb_contents

python3 -m gdown --folder "$CHROMADB_CONTENTS" -O chromadb_contents

echo "Building Docker image..."

sudo docker build \
  --file Dockerfile \
  --tag lrdisco2rag:latest \
  .

echo "Done"