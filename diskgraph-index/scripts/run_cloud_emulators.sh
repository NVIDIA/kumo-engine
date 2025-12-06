#!/usr/bin/env bash
set -euo pipefail

# Spin up local cloud emulators (MinIO, fake-gcs-server, Azurite), create buckets,
# set env vars, and run cloud_ingest tests. Requires Docker.

MINIO_IMAGE="minio/minio:RELEASE.2024-01-31T20-20-33Z"
GCS_IMAGE="fsouza/fake-gcs-server:1.47.3"
AZURITE_IMAGE="mcr.microsoft.com/azure-storage/azurite:3.29.0"
MINIO_PORT=9000
GCS_PORT=4443
AZURITE_PORT=10000
AZURITE_ACCOUNT="devstoreaccount1"
AZURITE_ACCESS_KEY="Eby8vdM02xnoz0DU3+UEiFZdLFhCydS+z+EUz0MBAhc="
AZURITE_CONTAINER="test-container"

RUN_ID="run-$(date +%s)-$RANDOM"
S3_PREFIX_BASE=${S3_PREFIX_BASE:-"diskgraph-index/emulator/s3"}
GCS_PREFIX_BASE=${GCS_PREFIX_BASE:-"diskgraph-index/emulator/gcs"}
AZURE_PREFIX_BASE=${AZURE_PREFIX_BASE:-"diskgraph-index/emulator/az"}

cleanup() {
  docker rm -f cloudtest-minio >/dev/null 2>&1 || true
  docker rm -f cloudtest-gcs >/dev/null 2>&1 || true
  docker rm -f cloudtest-azurite >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup
echo "Starting MinIO..."
docker run -d --name cloudtest-minio -p ${MINIO_PORT}:9000 \
  -e MINIO_ACCESS_KEY=minioadmin -e MINIO_SECRET_KEY=minioadmin \
  ${MINIO_IMAGE} server /data --console-address :9001 >/dev/null
sleep 3

echo "Creating MinIO bucket..."
docker run --rm --network host \
  -e MC_HOST_minio="http://minioadmin:minioadmin@127.0.0.1:${MINIO_PORT}" \
  minio/mc mb --ignore-existing minio/test-bucket >/dev/null

echo "Starting fake-gcs-server..."
docker run -d --name cloudtest-gcs -p ${GCS_PORT}:4443 \
  ${GCS_IMAGE} -scheme http >/dev/null
sleep 3
echo "Creating fake GCS bucket..."
curl -s -X POST -H "Content-Type: application/json" \
  -d "{\"name\":\"gcs-bucket\"}" \
  "http://127.0.0.1:${GCS_PORT}/storage/v1/b?project=fake" >/dev/null

echo "Starting Azurite..."
docker run -d --name cloudtest-azurite -p ${AZURITE_PORT}:10000 \
  -e AZURITE_ACCOUNTS="${AZURITE_ACCOUNT}:${AZURITE_ACCESS_KEY}" \
  ${AZURITE_IMAGE} azurite-blob --blobHost 0.0.0.0 --blobPort 10000 >/dev/null
sleep 3

echo "Creating Azurite container..."
docker run --rm --network host -e AZURE_STORAGE_ACCOUNT=${AZURITE_ACCOUNT} \
  -e AZURE_STORAGE_KEY=${AZURITE_ACCESS_KEY} \
  mcr.microsoft.com/azure-cli az storage container create \
  --name ${AZURITE_CONTAINER} \
  --connection-string "DefaultEndpointsProtocol=http;AccountName=${AZURITE_ACCOUNT};AccountKey=${AZURITE_ACCESS_KEY};BlobEndpoint=http://127.0.0.1:${AZURITE_PORT}/${AZURITE_ACCOUNT};" >/dev/null

export RUN_CLOUD_TESTS=1
export S3_BUCKET=test-bucket
export S3_ENDPOINT="http://127.0.0.1:${MINIO_PORT}"
export S3_REGION=us-east-1
export S3_ACCESS_KEY=minioadmin
export S3_SECRET_KEY=minioadmin
export S3_VIRTUAL_HOSTED=false
export AWS_ALLOW_HTTP=true
export S3_ALLOW_HTTP=1

export GCS_BUCKET=gcs-bucket
export GCS_ENDPOINT="http://127.0.0.1:${GCS_PORT}"

export AZURE_ACCOUNT=${AZURITE_ACCOUNT}
export AZURE_CONTAINER=${AZURITE_CONTAINER}
export AZURE_ENDPOINT="http://127.0.0.1:${AZURITE_PORT}/${AZURITE_ACCOUNT}"
export AZURE_ACCESS_KEY="${AZURITE_ACCESS_KEY}"
export AZURE_USE_EMULATOR=1

echo "Running cloud ingest tests against local emulators..."
cargo test -p diskgraph-index --test cloud_ingest

# Cleanup via trap
