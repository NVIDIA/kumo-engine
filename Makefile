# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage
# -----
# macOS defaults to port 12000; other platforms default to port 10000.
# CGROUP_CONFIG=auto NUMS=2 make devserver      # Two equal fenced devservers
# CGROUP_CONFIG='{"train-a":{"cpus":32,"memory_gib":200}}' make devserver
# UNFENCED=1 NUMS=1 make devserver              # Explicit shared-host opt-out
# DISABLE_AUTH=true is set in rest/devserver/.env by default
# UI_CATALOG_DEV=1 make devserver               # Use default catalog path (packages/kumo-ui-catalog)
# UI_CATALOG_DEV=1 UI_CATALOG_PATH=/path/to/catalog make devserver
x ?= $()
DEVSERVER_PLATFORM ?= $(shell uname -s)
devserver_platform_base_port = $(if $(filter Darwin,$(DEVSERVER_PLATFORM)),12000,)
# RANDOM_PORT wins over both the explicit and platform-default base ports.
devserver_base_port = $(if $(RANDOM_PORT),,$(or $(BASE_PORT),$(devserver_platform_base_port)))
ifdef RANDOM_PORT
  x = --random_port
endif
ifdef DETACHED
  x += --detached
endif
ifneq ($(strip $(devserver_base_port)),)
  x += --base_port $(devserver_base_port)
endif

DEVSERVER_PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
DOCKER_COMPOSE ?= $(shell if command -v docker-compose >/dev/null 2>&1; then printf 'docker-compose'; elif docker compose version >/dev/null 2>&1; then printf 'docker compose'; else printf 'docker compose'; fi)
devserver_default_kumo_home = $(if $(filter local-postgres,$(or $(variant),default)),.cache/kumo/devserver-postgres,.cache/kumo/devserver)
devserver_kumo_home = $(or $(KUMO_HOME),$(DEVSERVER_KUMO_HOME),$(devserver_default_kumo_home))
# An explicit deploy config is intentionally independent of CGROUP_CONFIG.
# It is useful for a locally supplied connector/identity policy without
# introducing another named devserver profile.
devserver_config_file = $(or $(DEVSERVER_CONFIG_FILE),rest/devserver/configs/$(or $(variant),default).json)

# UI Catalog development mode
# UI_CATALOG_DEV=1 enables catalog development with hot-reload
# UI_CATALOG_PATH overrides the default path (packages/kumo-ui-catalog)
DEFAULT_UI_CATALOG_PATH := $(abspath packages/kumo-ui-catalog)
catalog_path_arg ?= $()
ifdef UI_CATALOG_DEV
  ifeq ($(UI_CATALOG_DEV),1)
    ifdef UI_CATALOG_PATH
      catalog_path_arg = $(UI_CATALOG_PATH)
    else
      catalog_path_arg = $(DEFAULT_UI_CATALOG_PATH)
    endif
  endif
endif




# Clean NPM cache
clean-npm-modules:
	@if [ "$(CLEAN_MODULES)" = "1" ]; then \
		sudo rm -rf ui/node_modules; \
	fi

install-venv:
	./scripts/install_venv.sh

install-tools:
	mise install

KUMO_CORE_PYTHON ?= $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/python,python3)
KUMO_CORE_MIGRATED_PATHS := \
	src/kumo_core/encoders \
	src/kumo_core/evaluation \
	src/kumo_core/explain \
	src/kumo_core/features \
	src/kumo_core/modeling \
	src/kumo_core/sampling \
	src/kumo_core/training/supervision \
	src/kumo_core/training/task_config.py \
	src/kumo_core/types/features.py \
	tests/conftest.py \
	tests/encoders \
	tests/evaluation \
	tests/explain \
	tests/features \
	tests/modeling \
	tests/sampling \
	tests/training/supervision \
	tests/training/test_task_config.py \
	tests/support

.PHONY: test-kumo-core check-kumo-core test-kumo-core-artifact \
	test-kumo-core-consumers test-kumo-core-gpu

test-kumo-core:
	cd packages/kumo-core && $(KUMO_CORE_PYTHON) -m pytest --cov=kumo_core --cov-report=term-missing --cov-report=xml

check-kumo-core:
	cd packages/kumo-core && uvx --from flake8==6.1.0 flake8 --config ../../setup.cfg --select F src tests
	cd packages/kumo-core && uvx ty==0.0.23 check --python ../../.venv --extra-search-path ../kumo-api $(KUMO_CORE_MIGRATED_PATHS)
	uvx pyroma==5.0.0 --min=9 packages/kumo-core

test-kumo-core-artifact:
	@set -eu; \
	tmp="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp"' EXIT; \
	uv build --quiet --wheel --sdist --out-dir "$$tmp/dist" packages/kumo-core; \
	uv build --quiet --wheel --out-dir "$$tmp/dist" packages/kumo-api; \
	test -n "$$(find "$$tmp/dist" -maxdepth 1 -name '*.tar.gz' -print -quit)"; \
	uv pip install --quiet --target "$$tmp/site" --no-deps --force-reinstall "$$tmp"/dist/*.whl; \
	cd "$$tmp"; \
	MPLCONFIGDIR="$$tmp/matplotlib" PYTHONPATH="$$tmp/site" "$(KUMO_CORE_PYTHON)" "$(CURDIR)/tools/ci/kumo_core_artifact_smoke.py" --forbid-prefix "$(CURDIR)"

test-kumo-core-consumers:
	PYTHONPATH=packages/kumo-core/src $(KUMO_CORE_PYTHON) -m pytest \
		test/automl/test_searcher.py \
		test/config/test_encoder.py \
		test/config/test_mixin.py \
		test/data/generic/test_diskgraph_runtime_adapter.py \
		test/engine/test_diskgraph_category_counts.py \
		test/engine/test_diskgraph_task_dataset_spec.py \
		test/explain \
		test/rest/test_evaluation_statistics.py \
		test/train/test_diskgraph_nested_trainer.py \
		test/train/test_trainer.py \
		test/pool/test_images.py \
		test/pool/test_worker_main.py

test-kumo-core-gpu:
	cd packages/kumo-core && $(KUMO_CORE_PYTHON) -m pytest -k cuda --cov=kumo_core --cov-report=term-missing --cov-report=xml

# Updates Python packages required by Kumo to their latest versions and
# reinstalls Kumo.
update-packages:
	./scripts/update_packages.sh

# Build and install DiskGraph from packages/diskgraph.
install-diskgraph:
	./scripts/diskgraph install

# SQL console for local SQLite DB
sqlite:
	./scripts/sqlite_sql_console.sh

# Spin up a local Temporal cluster, no-op if it's already running.
up-local-temporal:
	./scripts/up_local_temporal_cli.sh

# Spin down a local Temporal cluster if it's already running.
down-local-temporal:
	tmux kill-session -t $${TEMPORAL_SESSION_NAME:-local_temporal} || true

# Deploy config for each variant is in rest/devserver/configs/<variant>.json.
# Usage:
#   make devserver                    # default = singlehost diskgraph
#   make devserver variant=k8s        # Diskgraph pool + K8s (runs setup wizard)
#   make devserver variant=local      # local Temporal (offline)
#
#   ls rest/devserver/configs/        # see all available variants
#
# variant=k8s: devserver itself runs the setup wizard once it has
# loaded .env + resolved deploy_env(), so the canonical tenant name
# (which the controller derives from deploy_env().id_suffix) is
# already known when provisioning the cluster resources. The wizard
# is idempotent — first run sets up the dev tenant, subsequent runs
# no-op all checks.
devserver-env:
	@if [ -z "$(DEVSERVER_ID)" ]; then \
		echo 'make devserver-env requires DEVSERVER_ID=<child-id>.'; \
		exit 2; \
	fi
	# A client may already export a child KUMO_HOME.  Manifest lookup must use
	# the coordinator root, not that unrelated child path.
	@KUMO_HOME="$(or $(DEVSERVER_KUMO_HOME),$(devserver_default_kumo_home))" \
		$(DEVSERVER_PYTHON) -m rest.devserver.multi_main \
		--print-child-env "$(DEVSERVER_ID)"

devserver: clean-npm-modules
	@if [ -n "$(BASE_PORT)$(RANDOM_PORT)" ]; then \
		echo 'BASE_PORT/RANDOM_PORT are replaced by BASE_PORTS for coordinated devservers.'; \
		exit 2; \
	fi
	@if [ -z "$(CGROUP_CONFIG)" ] && [ "$(UNFENCED)" != "1" ]; then \
		echo 'make devserver requires CGROUP_CONFIG=auto, JSON CGROUP_CONFIG, or UNFENCED=1.'; \
		exit 2; \
	fi
	KUMO_HOME="$(devserver_kumo_home)" \
	DEPLOY_STAGE=$(if $(filter local local-%,$(or $(variant),default)),dev_local,dev) \
	DEPLOY_ENV_CONFIG_FILE="$(devserver_config_file)" \
		UI_CATALOG_PATH="$(catalog_path_arg)" \
		CGROUP_CONFIG='$(CGROUP_CONFIG)' NUMS='$(NUMS)' \
		DEVSERVER_IDS='$(DEVSERVER_IDS)' BASE_PORTS='$(BASE_PORTS)' \
		UNFENCED='$(UNFENCED)' $(DEVSERVER_PYTHON) -m rest.devserver.multi_main

# Aliases for variants that need local Temporal
devserver-local: up-local-temporal
	$(MAKE) devserver variant=local

devserver-local-postgres: up-local-temporal up-local-postgres
	$(MAKE) devserver variant=local-postgres

# Direct Connect OIDC (Entra ID) variant. Prompts for the admin email and
# renders an untracked config; the OIDC client secret is pulled from AWS
# Secrets Manager (devserver-oidc-client-secret) at startup. Runs as a single
# attached devserver (interactive tmux) like the legacy path; pass DETACHED=1
# to run it in the background instead.
devserver-oidc: up-local-temporal clean-npm-modules
	@bash rest/devserver/scripts/setup_oidc.sh
	KUMO_HOME="$(devserver_kumo_home)" \
	DEPLOY_STAGE=dev_local \
	DEPLOY_ENV_CONFIG_FILE=/tmp/kumo-oidc-secrets/oidc.rendered.json \
	KUMO_DEV_TRUST_ADMIN_EMAILS=1 \
	UI_CATALOG_PATH="$(catalog_path_arg)" \
		$(DEVSERVER_PYTHON) -m rest.devserver.main $(x)

# Local Postgres for the venv devserver variants. This is a small
# Docker-backed helper, not the full Docker devserver/compose-local stack.
up-local-postgres:
	./scripts/up_local_postgres.sh

down-local-postgres:
	./scripts/down_local_postgres.sh

# Mirrors the env-file stacking that scripts/compose-local-preflight.sh
# performs so that targets which invoke $(_COMPOSE_LOCAL) directly — most
# importantly compose-local-down and compose-local-logs — see the same
# environment as the preflight-launched stack. Without this, local override
# settings are silently ignored when bringing the stack down or tailing logs.
_COMPOSE_LOCAL_BASE_ENV_FILE := $(or $(COMPOSE_LOCAL_BASE_ENV_FILE),docker/compose/.env.compose-local)
_COMPOSE_LOCAL_EXTRA_ENV_FILES := $(foreach f,$(COMPOSE_LOCAL_EXTRA_ENV_FILES),--env-file $(f))
_COMPOSE_LOCAL_ENV_FILES := --env-file $(_COMPOSE_LOCAL_BASE_ENV_FILE) \
  $(if $(wildcard docker/compose/.env.compose-local.local),--env-file docker/compose/.env.compose-local.local,) \
  $(_COMPOSE_LOCAL_EXTRA_ENV_FILES)
_COMPOSE_LOCAL_EXTRA_COMPOSE_FILES := $(foreach f,$(COMPOSE_LOCAL_EXTRA_COMPOSE_FILES),-f $(f))
_COMPOSE_LOCAL_POSTGRES_MODE := $(shell scripts/lib/resolve-postgres-mode.sh docker/compose/.env.compose-local docker/compose/.env.compose-local.local $(COMPOSE_LOCAL_EXTRA_ENV_FILES))
ifeq ($(_COMPOSE_LOCAL_POSTGRES_MODE),)
$(error Failed to resolve KUMO_POSTGRES_MODE)
endif
_COMPOSE_LOCAL_POSTGRES_FILES := $(if $(filter external,$(_COMPOSE_LOCAL_POSTGRES_MODE)),,-f docker/compose/docker-compose.singlehost.postgres.yml)

_DOCKER_COMPOSE := $(or $(DOCKER_COMPOSE),$(shell \
	if docker compose version >/dev/null 2>&1; then \
		printf 'docker compose'; \
	elif command -v docker-compose >/dev/null 2>&1; then \
		printf 'docker-compose'; \
	elif [ -x /opt/homebrew/lib/docker/cli-plugins/docker-compose ]; then \
		printf '/opt/homebrew/lib/docker/cli-plugins/docker-compose'; \
	else \
		printf 'docker compose'; \
	fi))

_COMPOSE_LOCAL = $(_DOCKER_COMPOSE) \
  -f docker/compose/docker-compose.singlehost.yml \
  $(_COMPOSE_LOCAL_POSTGRES_FILES) \
  $(_COMPOSE_LOCAL_GPU_FILES) \
  -f docker/compose/docker-compose.production-local.override.yml \
  $(_COMPOSE_LOCAL_EXTRA_COMPOSE_FILES) \
  $(_COMPOSE_LOCAL_ENV_FILES) \
  -p $${STACK:-$${COMPOSE_PROJECT_NAME:-kumo}}

_COMPOSE_LOCAL_STACK_SUFFIX := $(or $(STACK),$(COMPOSE_PROJECT_NAME),kumo)
_COMPOSE_LOCAL_BACKEND_HASH := .compose-local-backend-hash.$(_COMPOSE_LOCAL_STACK_SUFFIX)
_COMPOSE_LOCAL_UI_HASH     := .compose-local-ui-hash.$(_COMPOSE_LOCAL_STACK_SUFFIX)
_COMPOSE_LOCAL_IMAGE_TAG   := $(or $(KUMO_IMAGE_TAG),$(shell scripts/lib/resolve-image-tag.sh $(_COMPOSE_LOCAL_BASE_ENV_FILE) docker/compose/.env.compose-local.local))
_ECR_REGISTRY              := 926922431314.dkr.ecr.us-west-2.amazonaws.com
# Tags that signal "build from local source"; anything else (commit hash,
# nightly-<date>, ...) triggers an ECR pull.
_COMPOSE_LOCAL_BUILD_TAGS  := local latest
_COMPOSE_LOCAL_BACKEND_VARIANT := $(shell \
	if [ -n "$$KUMO_BACKEND_VARIANT" ]; then \
		printf '%s' "$$KUMO_BACKEND_VARIANT"; \
	elif [ "$$BUILD_GPU" = "0" ]; then \
		printf cpu; \
	elif [ "$$BUILD_GPU" = "1" ]; then \
		printf gpu; \
	elif command -v nvidia-smi >/dev/null 2>&1 \
		&& nvidia-smi >/dev/null 2>&1 \
		&& docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then \
		printf gpu; \
	else \
		printf cpu; \
	fi)
_COMPOSE_LOCAL_GPU_FILES = $(if $(filter gpu,$(_COMPOSE_LOCAL_BACKEND_VARIANT)),-f docker/compose/docker-compose.singlehost.gpu.yml,)

# Paths whose contents are baked into the backend image. An explicit allowlist
# (rather than "everything except ui/") so unrelated repo files — docs,
# scripts, agent config — don't force a backend rebuild.
_COMPOSE_LOCAL_BACKEND_PATHS := \
    kumo packages/kumo-api packages/kumo-core packages/kumo-snowflake \
    packages/kumo-bigquery packages/kumo-databricks \
    packages/kumo-s3 \
    rest temporalio_commons pool statedb service_framework \
    pyproject.toml setup.py setup.cfg bin \
    docker/Dockerfile.kumo_backend_cpu docker/Dockerfile.kumo_backend_gpu \
    docker/kumodeps_base \
    docker/install_postgres_ca_bundles docker/postgres-ca-bundles \
    docker/build_diskgraph_wheel scripts/diskgraph \
    scripts/build_faiss_gpu_amd64_wheel.sh scripts/build_faiss_gpu_arm64_wheel.sh \
    docker/diskgraph-source \
    docker/install_glove_files \
    docker/busyloop.sh

# Paths whose contents are baked into the UI image, including the docker-side
# entrypoint and nginx configuration.
_COMPOSE_LOCAL_UI_PATHS := \
    ui packages/kumo-ui-catalog \
    docker/Dockerfile.ui docker/docker-entrypoint-ui.sh \
    docker/nginx.ui.conf docker/busyloop.sh

# Hash commands. Each emits a one-line shell pipeline computing a content
# digest over the allowlist plus build-time dependency selections.
_COMPOSE_LOCAL_BACKEND_HASH_CMD = { \
    git log --format=%H -1 -- $(_COMPOSE_LOCAL_BACKEND_PATHS) 2>/dev/null; \
    git status --porcelain -- $(_COMPOSE_LOCAL_BACKEND_PATHS) 2>/dev/null; \
    ./scripts/diskgraph hash-source; \
} | md5sum | cut -d' ' -f1

_COMPOSE_LOCAL_UI_HASH_CMD = { \
    git log --format=%H -1 -- $(_COMPOSE_LOCAL_UI_PATHS) 2>/dev/null; \
    git status --porcelain -- $(_COMPOSE_LOCAL_UI_PATHS) 2>/dev/null; \
} | md5sum | cut -d' ' -f1

compose-local-build-backend:
	KUMO_BACKEND_VARIANT=cpu KUMO_IMAGE_TAG=$(_COMPOSE_LOCAL_IMAGE_TAG) $(MAKE) -f docker/Makefile build-backend-local
	@if [ "$(_COMPOSE_LOCAL_BACKEND_VARIANT)" = "gpu" ]; then \
	    KUMO_BACKEND_VARIANT=gpu KUMO_IMAGE_TAG=$(_COMPOSE_LOCAL_IMAGE_TAG) $(MAKE) -f docker/Makefile build-backend-local; \
	fi
	@$(_COMPOSE_LOCAL_BACKEND_HASH_CMD) > $(_COMPOSE_LOCAL_BACKEND_HASH)

compose-local-build-ui:
	KUMO_IMAGE_TAG=$(_COMPOSE_LOCAL_IMAGE_TAG) UI_CATALOG_DEV="$(UI_CATALOG_DEV)" UI_CATALOG_PATH="$(UI_CATALOG_PATH)" $(MAKE) -f docker/Makefile build-ui-local
	@$(_COMPOSE_LOCAL_UI_HASH_CMD) > $(_COMPOSE_LOCAL_UI_HASH)

compose-local-build: compose-local-build-backend compose-local-build-ui

compose-local:
	@command -v docker >/dev/null 2>&1 || (echo "ERROR: docker is required for compose-local. Install Docker Engine or provide a Docker-compatible CLI on PATH." && exit 1)
	@_avail_kb=$$(df -Pk / | awk 'NR == 2 {print $$4}'); \
	case "$$_avail_kb" in ''|*[!0-9]*) echo "ERROR: unable to determine free space on /"; exit 1;; esac; \
	if [ "$$_avail_kb" -lt 20971520 ]; then \
	    _avail=$$(( _avail_kb / 1048576 )); \
	    echo "ERROR: less than 20 GB free on / ($$_avail GB). Free space before building:"; \
	    echo "    docker image prune -f                        # removes untagged images only"; \
	    echo "    docker builder prune -f --filter 'until=1h' # removes old build cache"; \
	    exit 1; \
	fi
	@set -e; \
	_docker_info_output=$$(docker info 2>&1) || { \
	    if printf '%s\n' "$$_docker_info_output" | grep -q "permission denied" && [ "$$(uname -s)" = "Linux" ]; then \
	        echo "==> Adding $$USER to docker group and re-launching..."; \
	        sudo usermod -aG docker "$$USER"; \
	        sg docker -c "$(MAKE) compose-local"; \
	    else \
	        echo "ERROR: docker is installed but not accessible. Start Docker Desktop and verify 'docker info' succeeds."; \
	        printf '%s\n' "$$_docker_info_output"; \
	        exit 1; \
	    fi; \
	    exit 0; \
	}; \
	if ! echo "$(_COMPOSE_LOCAL_BUILD_TAGS)" | tr ' ' '\n' | grep -qx "$(_COMPOSE_LOCAL_IMAGE_TAG)"; then \
	    echo "==> Image tag '$(_COMPOSE_LOCAL_IMAGE_TAG)' is not a build-locally tag ($(_COMPOSE_LOCAL_BUILD_TAGS)) — pulling from ECR"; \
	    echo "==> AWS profile: $${AWS_PROFILE:-default} — run 'aws sso login --profile <name>' first if not logged in"; \
	    aws ecr get-login-password --region us-west-2 \
	        | docker login --username AWS --password-stdin $(_ECR_REGISTRY); \
	    _backend_variants="cpu"; \
	    if [ "$(_COMPOSE_LOCAL_BACKEND_VARIANT)" = "gpu" ]; then _backend_variants="cpu gpu"; fi; \
	    _ui_ecr="$(_ECR_REGISTRY)/kumo/kumo-ui:$(_COMPOSE_LOCAL_IMAGE_TAG)"; \
	    _ui_local="kumo/kumo-ui:$(_COMPOSE_LOCAL_IMAGE_TAG)"; \
	    for _backend_variant in $$_backend_variants; do \
	        _backend_ecr="$(_ECR_REGISTRY)/kumo/kumo-backend-$$_backend_variant:$(_COMPOSE_LOCAL_IMAGE_TAG)"; \
	        _backend_local="kumo/kumo-backend-$$_backend_variant:$(_COMPOSE_LOCAL_IMAGE_TAG)"; \
	        docker pull "$$_backend_ecr"; \
	        docker tag "$$_backend_ecr" "$$_backend_local"; \
	    done; \
	    docker pull "$$_ui_ecr"; \
	    docker tag "$$_ui_ecr" "$$_ui_local"; \
	else \
	_backend_cur=$$( $(_COMPOSE_LOCAL_BACKEND_HASH_CMD) ); \
	_backend_last=$$(cat $(_COMPOSE_LOCAL_BACKEND_HASH) 2>/dev/null || true); \
	_ui_cur=$$( $(_COMPOSE_LOCAL_UI_HASH_CMD) ); \
	_ui_last=$$(cat $(_COMPOSE_LOCAL_UI_HASH) 2>/dev/null || true); \
	if [ "$${SKIP_COMPOSE_LOCAL_BUILD:-}" = "1" ]; then \
	    echo "==> SKIP_COMPOSE_LOCAL_BUILD=1 — using existing images"; \
	else \
	    _backend_images="kumo/kumo-backend-cpu:$(_COMPOSE_LOCAL_IMAGE_TAG)"; \
	    if [ "$(_COMPOSE_LOCAL_BACKEND_VARIANT)" = "gpu" ]; then \
	        _backend_images="$$_backend_images kumo/kumo-backend-gpu:$(_COMPOSE_LOCAL_IMAGE_TAG)"; \
	    fi; \
	    _backend_missing=0; \
	    for _backend_image in $$_backend_images; do \
	        docker image inspect "$$_backend_image" >/dev/null 2>&1 || _backend_missing=1; \
	    done; \
	    if [ "$$_backend_missing" = "1" ] || [ "$$_backend_cur" != "$$_backend_last" ]; then \
	        echo "==> Backend changed or missing — rebuilding backend"; \
	        $(MAKE) compose-local-build-backend; \
	    else \
	        echo "==> Backend up to date — skipping rebuild"; \
	    fi; \
	    if ! docker image inspect kumo/kumo-ui:$(_COMPOSE_LOCAL_IMAGE_TAG) >/dev/null 2>&1 \
	        || [ "$$_ui_cur" != "$$_ui_last" ]; then \
	        echo "==> UI changed or missing — rebuilding UI"; \
	        $(MAKE) compose-local-build-ui; \
	    else \
	        echo "==> UI up to date — skipping rebuild"; \
	    fi; \
	fi; \
	fi; \
	COMPOSE_LOCAL_BASE_ENV_FILE="$(_COMPOSE_LOCAL_BASE_ENV_FILE)" \
	KUMO_IMAGE_TAG="$(_COMPOSE_LOCAL_IMAGE_TAG)" \
	KUMO_BACKEND_VARIANT="$(_COMPOSE_LOCAL_BACKEND_VARIANT)" \
	KUMO_POOL_WORKER_IMAGE="kumo/kumo-backend-$(_COMPOSE_LOCAL_BACKEND_VARIANT):$(_COMPOSE_LOCAL_IMAGE_TAG)" \
	bash scripts/compose-local-preflight.sh

compose-local-down:
	$(_COMPOSE_LOCAL) down

compose-local-down-volumes:
	$(_COMPOSE_LOCAL) down -v

compose-local-logs:
	$(_COMPOSE_LOCAL) logs -f

compose-local-logs-once:
	$(_COMPOSE_LOCAL) logs --no-color

BATCH_VALIDATION_LEGACY_TAG ?= kumo-legacy
BATCH_VALIDATION_TTNG_TAG ?= kumo-ttng
BATCH_VALIDATION_UI_TAG ?= kumo-validation-ui
BATCH_VALIDATION_WORK_ROOT ?= $(abspath .cache/kumo/batch-validation)
BATCH_VALIDATION_DISKGRAPH_DATA_ROOT ?= $(BATCH_VALIDATION_WORK_ROOT)/diskgraph-data
BATCH_VALIDATION_DATAPLANE_HOST_ROOT ?= $(BATCH_VALIDATION_WORK_ROOT)/dataplane
BATCH_VALIDATION_BACKEND_ROOT ?= /tmp/kumo-batch-validation
BATCH_VALIDATION_DATAPLANE_CONTAINER_ROOT ?= $(BATCH_VALIDATION_BACKEND_ROOT)/dataplane
BATCH_VALIDATION_LEGACY_PORT ?= 8001
BATCH_VALIDATION_TTNG_PORT ?= 8888
BATCH_VALIDATION_BUILD_UI ?= 1

_BATCH_VALIDATION_COMPOSE_FILE := docker/compose/docker-compose.batch-validation.yml
_BATCH_VALIDATION_ENV_FILE := docker/compose/env.batch-validation
_BATCH_VALIDATION_EXTRA := \
  COMPOSE_LOCAL_EXTRA_COMPOSE_FILES=$(_BATCH_VALIDATION_COMPOSE_FILE) \
  COMPOSE_LOCAL_EXTRA_ENV_FILES=$(_BATCH_VALIDATION_ENV_FILE)
_BATCH_VALIDATION_COMPOSE = $(_DOCKER_COMPOSE) \
  -f docker/compose/docker-compose.singlehost.yml \
  $(_COMPOSE_LOCAL_POSTGRES_FILES) \
  -f docker/compose/docker-compose.production-local.override.yml \
  -f $(_BATCH_VALIDATION_COMPOSE_FILE) \
  --env-file docker/compose/.env.compose-local \
  --env-file $(_BATCH_VALIDATION_ENV_FILE)

.PHONY: batch-validation-build batch-validation-build-ui \
  batch-validation-build-legacy batch-validation-build-ttng \
  batch-validation-prepare-root batch-validation-wait-healthy \
  batch-validation-up batch-validation-up-legacy batch-validation-up-ttng \
  batch-validation-down batch-validation-down-legacy batch-validation-down-ttng \
  batch-validation-down-volumes batch-validation-reset \
  batch-validation-restart \
  batch-validation-logs-legacy batch-validation-logs-ttng

batch-validation-build: batch-validation-build-ui batch-validation-build-legacy batch-validation-build-ttng

batch-validation-build-ui:
	@if [ "$(BATCH_VALIDATION_BUILD_UI)" = "0" ]; then \
	    echo "==> Skipping validation UI build"; \
	else \
	    KUMO_IMAGE_TAG=$(BATCH_VALIDATION_UI_TAG) $(MAKE) compose-local-build-ui; \
	fi

batch-validation-build-legacy:
	KUMO_IMAGE_TAG=$(BATCH_VALIDATION_LEGACY_TAG) \
	KUMO_BACKEND_VARIANT=cpu \
	    $(MAKE) compose-local-build-backend

batch-validation-build-ttng:
	KUMO_IMAGE_TAG=$(BATCH_VALIDATION_TTNG_TAG) \
	KUMO_BACKEND_VARIANT=cpu \
	    $(MAKE) compose-local-build-backend

batch-validation-prepare-root:
	@set -e; \
	for d in \
	    "$(BATCH_VALIDATION_WORK_ROOT)" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)" \
	    "$(BATCH_VALIDATION_DISKGRAPH_DATA_ROOT)" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-legacy" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-legacy/test" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-legacy/test/kumouser-kumo-legacy" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-legacy/test/artifacts" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-legacy/test/artifacts/kumouser-kumo-legacy" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-legacy/kumo_parity" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-legacy/custom_tt_staging" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-ttng" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-ttng/test" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-ttng/test/kumouser-kumo-ttng" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-ttng/test/artifacts" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-ttng/test/artifacts/kumouser-kumo-ttng" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-ttng/kumo_parity" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-ttng/custom_tt_staging" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-legacy/custom_tt_staging/legacy" \
	    "$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT)/kumo-ttng/custom_tt_staging/ttng"; do \
	    mkdir -p "$$d"; \
	    chmod a+rwX "$$d" 2>/dev/null || true; \
	done

batch-validation-up: batch-validation-prepare-root batch-validation-up-legacy batch-validation-up-ttng

batch-validation-up-legacy:
	STACK=kumo-legacy \
	KUMO_IMAGE_TAG=$(BATCH_VALIDATION_LEGACY_TAG) \
	KUMO_BATCH_VALIDATION_UI_TAG=$(BATCH_VALIDATION_UI_TAG) \
	KUMO_DISKGRAPH_TT_NG_ENABLED=false \
	ALLOW_ARRAY_TARGETS=false \
	KUMO_BACKEND_VARIANT=cpu \
	PRODUCTION_LOCAL_DATAPLANE_ROOT=$(BATCH_VALIDATION_DATAPLANE_CONTAINER_ROOT)/{instance_id}/ \
	PRODUCTION_LOCAL_ARTIFACT_ROOT=$(BATCH_VALIDATION_DATAPLANE_CONTAINER_ROOT)/artifacts/{instance_id}/ \
	LOCAL_DATAPLANE_HOST_MOUNT=$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT) \
	LOCAL_DATAPLANE_MOUNT=$(BATCH_VALIDATION_DATAPLANE_CONTAINER_ROOT) \
	DISKGRAPH_DATA_ROOT=$(BATCH_VALIDATION_DISKGRAPH_DATA_ROOT) \
	NGINX_PORT=$(BATCH_VALIDATION_LEGACY_PORT) \
	$(_BATCH_VALIDATION_EXTRA) \
	    bash scripts/compose-local-preflight.sh

batch-validation-up-ttng:
	STACK=kumo-ttng \
	KUMO_IMAGE_TAG=$(BATCH_VALIDATION_TTNG_TAG) \
	KUMO_BATCH_VALIDATION_UI_TAG=$(BATCH_VALIDATION_UI_TAG) \
	KUMO_DISKGRAPH_TT_NG_ENABLED=true \
	ALLOW_ARRAY_TARGETS=true \
	KUMO_BACKEND_VARIANT=cpu \
	PRODUCTION_LOCAL_DATAPLANE_ROOT=$(BATCH_VALIDATION_DATAPLANE_CONTAINER_ROOT)/{instance_id}/ \
	PRODUCTION_LOCAL_ARTIFACT_ROOT=$(BATCH_VALIDATION_DATAPLANE_CONTAINER_ROOT)/artifacts/{instance_id}/ \
	LOCAL_DATAPLANE_HOST_MOUNT=$(BATCH_VALIDATION_DATAPLANE_HOST_ROOT) \
	LOCAL_DATAPLANE_MOUNT=$(BATCH_VALIDATION_DATAPLANE_CONTAINER_ROOT) \
	DISKGRAPH_DATA_ROOT=$(BATCH_VALIDATION_DISKGRAPH_DATA_ROOT) \
	NGINX_PORT=$(BATCH_VALIDATION_TTNG_PORT) \
	$(_BATCH_VALIDATION_EXTRA) \
	    bash scripts/compose-local-preflight.sh

batch-validation-down: batch-validation-down-legacy batch-validation-down-ttng

batch-validation-down-legacy:
	STACK=kumo-legacy \
	KUMO_IMAGE_TAG=$(BATCH_VALIDATION_LEGACY_TAG) \
	KUMO_BATCH_VALIDATION_UI_TAG=$(BATCH_VALIDATION_UI_TAG) \
	$(_BATCH_VALIDATION_COMPOSE) -p kumo-legacy down --remove-orphans

batch-validation-down-ttng:
	STACK=kumo-ttng \
	KUMO_IMAGE_TAG=$(BATCH_VALIDATION_TTNG_TAG) \
	KUMO_BATCH_VALIDATION_UI_TAG=$(BATCH_VALIDATION_UI_TAG) \
	$(_BATCH_VALIDATION_COMPOSE) -p kumo-ttng down --remove-orphans

batch-validation-down-volumes:
	STACK=kumo-legacy \
	KUMO_IMAGE_TAG=$(BATCH_VALIDATION_LEGACY_TAG) \
	KUMO_BATCH_VALIDATION_UI_TAG=$(BATCH_VALIDATION_UI_TAG) \
	$(_BATCH_VALIDATION_COMPOSE) -p kumo-legacy down -v --remove-orphans
	STACK=kumo-ttng \
	KUMO_IMAGE_TAG=$(BATCH_VALIDATION_TTNG_TAG) \
	KUMO_BATCH_VALIDATION_UI_TAG=$(BATCH_VALIDATION_UI_TAG) \
	$(_BATCH_VALIDATION_COMPOSE) -p kumo-ttng down -v --remove-orphans

batch-validation-reset:
	$(MAKE) batch-validation-down-volumes
	$(MAKE) batch-validation-up

batch-validation-restart: batch-validation-down batch-validation-up

batch-validation-wait-healthy:
	@set -e; \
	for i in $$(seq 1 60); do \
	    not_ready="$$(docker ps --filter 'name=kumo-legacy' --filter 'name=kumo-ttng' --format '{{.Names}} {{.Status}}' | awk '/health: starting|unhealthy/ {print}')"; \
	    if [ -z "$$not_ready" ]; then \
	        docker ps --filter 'name=kumo-legacy' --filter 'name=kumo-ttng' --format 'table {{.Names}}\t{{.Status}}' | sort; \
	        exit 0; \
	    fi; \
	    if [ "$$i" = 1 ] || [ $$((i % 10)) = 0 ]; then \
	        echo "Waiting for validation stack health checks..."; \
	        echo "$$not_ready"; \
	    fi; \
	    sleep 5; \
	done; \
	echo "Timed out waiting for validation stack health checks" >&2; \
	docker ps --filter 'name=kumo-legacy' --filter 'name=kumo-ttng' --format 'table {{.Names}}\t{{.Status}}' | sort >&2; \
	exit 1

batch-validation-logs-legacy:
	STACK=kumo-legacy \
	KUMO_IMAGE_TAG=$(BATCH_VALIDATION_LEGACY_TAG) \
	KUMO_BATCH_VALIDATION_UI_TAG=$(BATCH_VALIDATION_UI_TAG) \
	$(_BATCH_VALIDATION_COMPOSE) -p kumo-legacy logs -f

batch-validation-logs-ttng:
	STACK=kumo-ttng \
	KUMO_IMAGE_TAG=$(BATCH_VALIDATION_TTNG_TAG) \
	KUMO_BATCH_VALIDATION_UI_TAG=$(BATCH_VALIDATION_UI_TAG) \
	$(_BATCH_VALIDATION_COMPOSE) -p kumo-ttng logs -f

playwright-local-stack:
	./scripts/run_playwright_local_stack.sh

playwright-local-stack-headed:
	./scripts/run_playwright_local_stack.sh -- --headed

playwright-local-stack-seed:
	./scripts/run_playwright_local_stack.sh --skip-tests

playwright-local-stack-seed-existing:
	./scripts/run_playwright_local_stack.sh --reuse-stack --skip-build --skip-tests

playwright-local-stack-down:
	COMPOSE_LOCAL_BASE_ENV_FILE=docker/compose/.env.compose-playwright-ci STACK=playwright $(MAKE) compose-local-down

# Delete all persistent data needed to run the REST API and UI. It is necessary
# to delete this when a backwards-incompatible change is made to the DB schema.
reset-devserver:
	$(MAKE) CLEAN_MODULES=1 clean-npm-modules
	python3 rest/scripts/reset_devserver.py
	@if minikube -p minikube status >/dev/null 2>&1; then \
		minikube update-context && kubectl config current-context && kubectl -n devtest delete jobs --all; \
	else \
		echo "Skipping minikube cleanup: no running 'minikube' profile found."; \
	fi

reset-devserver-local:
	$(MAKE) CLEAN_MODULES=1 clean-npm-modules
	python3 rest/scripts/reset_devserver.py --local
	@if minikube -p minikube status >/dev/null 2>&1; then \
		minikube update-context && kubectl config current-context && kubectl -n devtest delete jobs --all; \
	else \
		echo "Skipping minikube cleanup: no running 'minikube' profile found."; \
	fi

telemetry:
	./monitoring/monitoring.sh

minikube-get-pods:
	minikube update-context && kubectl config current-context && kubectl -n devtest get pods

install-all-dependencies:
	./scripts/install_all_dependencies.sh $(INSTALL_FLAGS)

generate-ui-client-code:
	DEPLOY_STAGE=dev_local python3 rest/scripts/download_schema.py
	cd ui && rm -rf src/gen/v0/
	cd ui && rm -rf src/gen/v1/
	cd ui && npx orval
	cd ui && npm run inject-zod-schemas
	cd ui && npm run lint-format-codegen
