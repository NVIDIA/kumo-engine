# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

THIS_FILE := $(lastword $(MAKEFILE_LIST))
.PHONY: help pull up start down destroy stop restart logs
KUMO_IMAGE_TAG ?= $(if $(STACK),$(STACK),local)
KUMO_BACKEND_VARIANT ?= cpu
DOCKER_HOST_ARCH ?= $(subst aarch64,arm64,$(subst x86_64,amd64,$(shell uname -m)))
DOCKER_BUILD_PLATFORM ?= linux/$(DOCKER_HOST_ARCH)
CUDA_ARCH_LIST ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';' -)
DOCKER_BUILD_ULIMIT ?= nofile=65536:65536

help:## prints help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build:## Builds kumo containers from local source and tags with local
	docker build -t quay.io/kumo/kumo-ui:local -f docker/Dockerfile.ui .

build-backend-local:
	@echo "==> Generating kumo_version.txt"
	git rev-parse HEAD > kumo_version.txt 2>/dev/null || echo "0.0.0+local" > kumo_version.txt
	@./scripts/diskgraph stage-source --out docker/diskgraph-source
	@set -e; \
	if [ "$(KUMO_BACKEND_VARIANT)" = "cpu" ]; then \
	    echo "==> Building kumo-backend-cpu:$(KUMO_IMAGE_TAG) from source with staged DiskGraph"; \
	    docker build \
	      --platform $(DOCKER_BUILD_PLATFORM) \
	      --ulimit $(DOCKER_BUILD_ULIMIT) \
	      --build-arg BAKE_GLOVE_LEVEL=none \
	      --build-arg KUMO_VERSION=0.0.0+local \
	      -t kumo/kumo-backend-cpu:$(KUMO_IMAGE_TAG) \
	      -f docker/Dockerfile.kumo_backend_cpu .; \
	elif [ "$(KUMO_BACKEND_VARIANT)" = "gpu" ]; then \
	    echo "==> Ensuring CUDA 13 Faiss wheel is available"; \
	    if [ "$(DOCKER_BUILD_PLATFORM)" = "linux/arm64" ]; then \
	        :; \
	    else \
	        ./scripts/build_faiss_gpu_amd64_wheel.sh; \
	    fi; \
	    echo "==> Building kumo-backend-gpu:$(KUMO_IMAGE_TAG) from source with staged DiskGraph"; \
	    docker build \
	      --platform $(DOCKER_BUILD_PLATFORM) \
	      --ulimit $(DOCKER_BUILD_ULIMIT) \
	      --build-arg BAKE_GLOVE_LEVEL=none \
	      --build-arg KUMO_VERSION=0.0.0+local \
	      --build-arg CUDA_ARCH_LIST="$(CUDA_ARCH_LIST)" \
	      -t kumo/kumo-backend-gpu:$(KUMO_IMAGE_TAG) \
	      -f docker/Dockerfile.kumo_backend_gpu .; \
	else \
	    echo "ERROR: KUMO_BACKEND_VARIANT must be cpu or gpu, got '$(KUMO_BACKEND_VARIANT)'"; \
	    exit 1; \
	fi
build-ui-local:
	@echo "==> Generating kumo_version.txt"
	git rev-parse HEAD > kumo_version.txt 2>/dev/null || echo "0.0.0+local" > kumo_version.txt
	$(eval _UI_CTX := $(shell mktemp -d))
	cp -r docker/. $(_UI_CTX)/
	mkdir -p $(_UI_CTX)/kumo
	cp LICENSE $(_UI_CTX)/kumo/LICENSE
	rsync -a --exclude=node_modules --exclude=build --exclude=.next --exclude='stacktracer/source_map/*.map' ui/ $(_UI_CTX)/kumo/ui/
	mkdir -p $(_UI_CTX)/kumo/packages
	rsync -a --exclude=node_modules --exclude=dist --exclude=storybook-static --exclude=coverage packages/kumo-ui-catalog/ $(_UI_CTX)/kumo/packages/kumo-ui-catalog/
	docker build \
	  --ulimit $(DOCKER_BUILD_ULIMIT) \
	  -t kumo/kumo-ui:$(KUMO_IMAGE_TAG) \
	  -f docker/Dockerfile.ui \
	  $(_UI_CTX)/
	rm -rf $(_UI_CTX)

build-compose-local: build-backend-local build-ui-local
	@docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" \
	  | grep -E "^kumo/(kumo-backend-(cpu|gpu)|kumo-ui)" | grep -E ":$(KUMO_IMAGE_TAG)"
