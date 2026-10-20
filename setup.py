# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup


def parse_project_version(project_file_path: Path) -> str:
    """Read the static ``[project].version`` from a pyproject file."""
    text = project_file_path.read_text(encoding="utf8")
    match = re.search(
        r'^\[project\]\s*$.*?^version\s*=\s*[\'\"]([^\'\"]+)[\'\"]',
        text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"No [project].version found in {project_file_path}")
    return match.group(1)


# NOTE that for major dependencies, we will check the latest update every year
# in May and November to ensure that our dependencies are up to date.
# Please include performance and efficiency critical packages here.
major_dependencies = [
    # Keep in sync with docker/kumodeps_base.
    "torch==2.12.0",
    # numpy 2.2.x / pandas 2.2.x ship no cp314 wheel (first cp314 wheels:
    # numpy 2.3.2, pandas 2.3.3); older pins force a fragile from-sdist build
    # on 3.14. Both stay within kumo-core's numpy>=2.1,<3 / pandas>=2.2,<3.
    "numpy==2.3.3",
    "pandas==2.3.3",
    # Used directly by kumo.utils.dataclass and REST integration tests.
    "httpx<0.28",
    "backoff==2.2.1",
]

observability_requires = [
    # opentelemetry <1.35 (proto) capped protobuf<5, conflicting with the cp314
    # protobuf>=6.33 floor; the 1.43.0 / 0.64b0 line relaxed that cap. Kept in
    # sync with the opentelemetry requirement here.
    "opentelemetry-api==1.43.0",
    "opentelemetry-sdk==1.43.0",
    "opentelemetry-exporter-otlp==1.43.0",
    "pyinstrument==4.6.2",
]

# Source and consumers change atomically in this repository. Built Kumo wheels
# retain an ordinary versioned dependency; developer and image builds install
# packages/kumo-api first so they exercise the colocated source.
kumo_api_ver = parse_project_version(
    Path(__file__).parent / "packages" / "kumo-api" / "pyproject.toml")
kumo_api_dep = f"kumo-api=={kumo_api_ver}"

install_requires = major_dependencies + observability_requires + [
    # Internal component distributions imported by the platform package.
    # scripts/update_packages.sh installs their colocated sources before the
    # root resolve; keep ordinary requirements here so built metadata records
    # the real runtime ownership even though they are not public PyPI releases.
    # The faiss-cpu/faiss-gpu choice is NOT made here: a build-time env var
    # would get frozen into uv.lock at whatever value was active when `uv
    # lock` last ran, so a later `uv sync --locked --extra cu130` would
    # silently keep installing faiss-cpu. It's wired through the "cpu"/
    # "cu130" extras below instead, exactly like torch's build selection,
    # so it's a real per-sync `--extra` choice uv.lock can represent for
    # both profiles at once.
    "kumo-core==0.1.0.dev1",
    "kumo-bigquery==0.1.0.dev1",
    "kumo-databricks==0.1.0.dev1",
    "kumo-gcs==0.1.0.dev1",
    "kumo-s3==0.1.0.dev1",
    "kumo-snowflake==0.1.0.dev1",
    # Sole implementation of temporal detection/parsing and the multicategorical
    # delimiter decision; kumo imports it directly, not only via kumo-core.
    "diskgraph>=0.1.0",
    "cachetools==5.3.1",
    "scipy<2.0.0",  # Last version to support Python 3.8 :(
    "jsons==1.6.3",
    "setuptools-scm==8.3.1",
    "typing_extensions>=4.12.2",
    # protobuf <6 ships no cp314 wheel (and the pure-python fallback breaks the
    # otel/grpc import chain), so a cp314-capable floor is required on py3.14.
    "protobuf>=6.33,<8",
    "pydantic>=2.11",
    "pydantic-core>=2.33,<3",
    "tabulate==0.8.10",
    "omegaconf==2.3.0",
    "PyYAML==6.0.2",
    # ``tools`` is part of the root wheel, and its memory utilities import
    # psutil at runtime.
    "psutil==7.2.2",
    "tensorboard==2.16.2",
    "prometheus_client==0.14.1",
    "scikit-learn>=1.7.2",
    "smart-open==5.2.1",
    # pyarrow version must be kept in sync with snowflake-connector-python.
    # pyarrow <18 has no cp314 wheel and still ships the legacy ParquetDataset
    # API; >=18 is the cp314-capable floor. PyArrow 25 is not yet compatible
    # with the DiskGraph extension's parquet statistics path (it raises
    # ImportError while computing BP column stats).
    "pyarrow>=18,<25",
    "kubernetes==24.2.0",
    # filelock >=3.20.3 fixes CVE-2025-68146 / CVE-2026-22701. (3.29.5+ are
    # newer than the uv exclude-newer quarantine window at time of writing.)
    "filelock==3.29.4",
    # cryptography 50 clears CVE-2026-27459 (critical) and the cryptography
    # 38.x CVE set. The historical "newer cryptography rejects legacy-cipher
    # Snowflake keys" concern no longer reproduces: 50.0.0 loads DES3-,
    # MD5-DES-, RC2- and AES-encrypted PKCS#8 and traditional PEM keys
    # identically to 38.0.4. test/connector/test_legacy_private_key.py guards
    # this. It is a base dependency because io_utils/encryption.py and
    # connector code import it outside the REST-only surface.
    "cryptography==50.0.0",
    "inflect==7.5.0",
    # 2026.6.0 sandboxes ReferenceFileSystem Jinja templates and no longer
    # unpickles cache metadata (GHSA-27vj-qcqg-25rc).
    "fsspec==2026.6.0",
    "optuna==4.3.0",
    "watchtower==3.0.1",
    "asgiref==3.5.2",
    "blockbuster>=1.5,<1.6",
    "urllib3>=1.26.18,<3",
    "requests==2.33.1",
    "tenacity>=8,<10",
    # google-api-python-client permits much older httplib2 releases; pin the
    # patched version so the backend image cannot resolve back to 0.20.4.
    "httplib2==0.32.0",
    "orjson>=3.10",
    "ec2-metadata==2.10.0",
    "deprecated==1.3.1",
    # NOTE: historically we relied on internal details of the wrapt
    # `@synchronized` decorator in kumo/connector/snowflake_connector.py;
    # that usage has since been removed. wrapt <1.17 ships no cp314 wheel, so
    # bump to a cp314-capable floor.
    "wrapt>=1.17,<2",
    # boto3 >=1.42.22 is required by redshift_connector >=2.1.14 (see below).
    "boto3==1.42.90",
    "botocore==1.42.90",
    "boto3-stubs[s3]==1.42.90",
    "azure-core>=1.31,<2",
    "azure-storage-blob>=12.27,<13",
    "azure-storage-file-datalake==12.22.0",
    "azure-identity==1.25.1",
    "mypy-boto3-s3==1.42.94",
    "s3fs==0.4.2",
    "sqlalchemy>=2.0.14,<2.1",
    "aiosqlite==0.17.0",
    # cp314-capable floors (the exact old pins ship no cp314 wheel).
    "asyncpg>=0.30",
    # Required for BigQuery connector
    "google-cloud-bigquery==3.36.0",
    "google-cloud-bigquery-storage>=2.39.0",
    "google-cloud-storage-transfer==1.17.0",
    # gcsfs 2026.6.0 (below) requires google-cloud-storage>=3.11.0.
    "google-cloud-storage==3.11.0",
    "google-api-core>=2,<3",
    "google-auth>=2.14.1,<3",
    # Required for GCP Secret Manager (IS_GCP_DEPLOY)
    "google-cloud-secret-manager==2.26.0",
    # Required for GCS filesystem in PySpark
    "gcsfs==2026.6.0",
    # Required for reading data from BigQuery to a Pandas dataframe
    "db-dtypes==1.1.1",
    # Some runtime deps still import pkg_resources (for example Snowpark).
    # Pin >=80.10.2 for vendored CVE fixes, <82 because 82 removed
    # pkg_resources.
    "setuptools>=80.10.2,<82",
    "snowflake-connector-python>=4.3",
    # Floor >=1.49: first release admitting Python 3.14 (1.11.0/1.24 cap at
    # <3.12, so on 3.14 the resolver must pick >=1.49). Keep in sync with
    # scripts/update_packages.sh.
    "snowflake-snowpark-python>=1.49",
    "databricks-sql-connector==4.5.0",
    "databricks-sdk==0.73.0",
    "thrift==0.24.0",
    "certifi==2025.1.31",  # sf support case 01012966
    "python-logging-loki==0.3.1",
    # temporalio 1.17.0 capped protobuf<6, conflicting with the cp314
    # protobuf>=6.33 floor (kumo-api requires protobuf>=5.27 on 3.14).
    # 1.18.0+ relaxed the cap to protobuf<7; abi3 wheels install on cp314.
    # 1.30.0 raises the floor further: its Rust bridge bundles quinn-proto
    # 0.11.15 (RUSTSEC-2026-0185) and opentelemetry_sdk 0.32.1
    # (GHSA-w9wp-h8wv-79jx). 1.31.0 fixes a cancel racing a completed sync
    # activity killing an idle pool thread (sdk-python#1648).
    "temporalio>=1.31,<2",
    "zstd>=1.5.6",
    "aiocache==0.12.3",
    # aiobotocore 3.7.0 supports the botocore 1.42.x line pulled in by the boto3
    # pin above. Every released aioboto3 version caps botocore below
    # redshift_connector 2.1.14's boto3 floor.
    "aiobotocore==3.7.0",
    "types-aiobotocore[s3]==3.7.0",
    kumo_api_dep,
    "kumo-pql>=0.1.0,<0.2",
]

drawing_requires = [
    # seaborn<0.12 imports `matplotlib.cm.register_cmap`, removed in
    # matplotlib 3.9; 0.13.2 uses `matplotlib.colormaps.register` and
    # imports cleanly against matplotlib>=3.10.5 on Python 3.14.
    "seaborn==0.13.2",
    "graphviz==0.20.1",
    "matplotlib>=3.10.5",
    # Bundles the MapLibre fix for CVE-2026-85061.
    "plotly==7.1.0",
]

rest_requires = [
    # fastapi >=0.137 allows starlette >=1.3.1, which fixes
    # CVE-2026-48818 / CVE-2026-54283 (high) and the starlette 1.x medium
    # CVE set.
    "fastapi==0.138.0",
    "starlette==1.3.1",
    "uvicorn==0.38.0",
    "watchdog[watchmedo]",
    # Paired with the base cryptography pin above; both versions are
    # single-sourced here.
    "pyOpenSSL==26.4.0",
    "readerwriterlock==1.0.9",
    "greenlet>=3.1",
    # 0.64b0 pairs with the opentelemetry-api/sdk 1.43.0 core above; kept in
    # sync with the opentelemetry/fastapi requirements here.
    # >=0.62b0 is REQUIRED with starlette 1.x / fastapi >=0.137: older
    # instrumentation iterates app.routes and crashes on _IncludedRouter
    # entries (500s every /v0/auth/* request); fixed upstream by
    # opentelemetry-python-contrib#4700 (_flatten_routes).
    "opentelemetry-instrumentation==0.64b0",
    "opentelemetry-instrumentation-fastapi==0.64b0",
    "opentelemetry-distro[otlp]==0.64b0",
    "fifolock==0.0.20",
    # 6.0.0 reads route.path unguarded while resolving handler names and
    # crashes on starlette 1.x's _IncludedRouter entries; 8.x supports
    # starlette>=1,<2.
    "prometheus-fastapi-instrumentator==8.0.2",
    "humanize==4.6.0",
    "markdown==3.10",
    "python-dateutil>=2.9,<3",
    # 4.14.2 fixes GHSA-82r6-8w77-94w6 and GHSA-5p39-cfhj-2xmp.
    "anyio==4.14.2",
    "pyright==1.1.283",
    # PyJWT replaces python-jose: jose hard-depends on ecdsa
    # (CVE-2024-23342, Minerva timing attack — no fixed release exists) and
    # caps pyasn1 <0.5 (CVE-2026-30922).
    "pyjwt[crypto]==2.15.1",
]

# TODO(frg100): Store stats in S3 so we don't have to calculate them with
# polars
tools_requires = [
    "polars==1.19.0",
    "tqdm>=4.66",
]

# sql-metadata 2.x caps sqlparse below 0.6. Version 3.x uses sqlglot instead,
# allowing the patched sqlparse 0.6.0 release while preserving Parser.columns.
data_platform_requires = [
    "sql_metadata==3.0.1",
    "sqlparse==0.6.0",
    # Custom SQL Tables parse/normalize user SQL via sqlglot
    # (kumo/data/custom_sql.py). One minor, not a range: sqlglot normalizes
    # Custom SQL text into a digest that lands in `graph_id`. Must stay
    # byte-identical to the sqlglot pin in `packages/kumo-core/pyproject.toml`;
    # enforced by `packages/kumo-core/tests/test_custom_sql_golden.py`.
    "sqlglot==30.17.*",
]

full_requires = [
    *drawing_requires,
    *rest_requires,
    *data_platform_requires,
    *tools_requires,
]

test_requires = [
    "moto==4.2.14",
    "pytest==9.0.3",
    "pytest-asyncio==1.3.0",
    "pytest-benchmark==4.0.0",
    "pytest-cov",
    "pytest-env==0.6.2",
    "pytest-mock",
    "pytest-timeout",
    "pytest-split",
    "coverage",
    "category_encoders",
    "rocksdict",
    "respx==0.22.0",
    "pytorch-frame==0.3.0",
    "pytorch-lightning==2.6.5",
]

dev_requires = test_requires + [
    "pre-commit",
    "enum-actions",
    "pick",
    "packaging>=23",
    "kubernetes-stubs",
    # Editor/lint tooling; versions match CI and .pre-commit-config.yaml.
    "flake8==7.1.2",
    "isort==5.12.0",
    "pyright==1.1.283",
    "yapf==0.40.2",
    "autoflake==2.3.1",
    "autopep8",
    "pip",
]

setup(
    author="Kumo.ai",
    author_email="support@kumo.ai",
    url="https://kumo.ai",
    keywords=[
        "deep-learning",
        "graph-neural-networks",
        "cloud-data-warehouse",
    ],
    # Keep in sync with pyproject.toml [project].requires-python.
    python_requires=">=3.11",
    install_requires=install_requires,
    tests_require=test_requires,
    extras_require={
        "tracing": observability_requires,
        "drawing": drawing_requires,
        "rest": rest_requires,
        "full": full_requires,
        "test": test_requires,
        "dev": dev_requires,
        "data_platform": data_platform_requires,
        # Torch build selectors for `uv sync --extra <name>`; see
        # pyproject.toml's [tool.uv.sources] torch index mapping. Also
        # selects kumo-core's matching faiss build -- see the comment on
        # install_requires above for why this lives here and not in a
        # CUDA-env-var branch.
        "cpu": ["torch", "kumo-core[faiss-cpu]==0.1.0.dev1"],
        "cu130": ["torch", "kumo-core[faiss-gpu]==0.1.0.dev1"],
        "cu132": ["torch", "kumo-core[faiss-gpu]==0.1.0.dev1"],
    },
    # The platform's version providers for `kumo_core.version.get_kumo_version`:
    # the image version in ~/kumo_version.txt, then the release version from
    # the root VERSION file. Providers run in entry-point name order, so
    # `container-file` wins over kumo-engine's provider, which wins over
    # `release-version`.
    entry_points={
        "kumo_core.version_providers": [
            "container-file = common_utils.version_file:container_version",
            "release-version = common_utils.version_file:release_version",
        ],
    },
    scripts=[
        "bin/kumo-auto",
        "bin/kumo-pred",
    ],
    # `kumo` lost its __init__.py so it can be a shared PEP 420 namespace
    # package with packages/kumo-engine's own `kumo.engine` -- find_packages()
    # alone can no longer see it or anything under it (it won't recurse into
    # a directory without an __init__.py at all), so union in a
    # namespace-aware pass scoped to just `kumo`/`kumo.*`.
    packages=find_packages() +
    find_namespace_packages(include=["kumo", "kumo.*"]),
    setup_requires=["setuptools-scm==8.3.1"],
    setup_scm_version={
        'git_describe_command':
        'git describe --dirty --tags --long --match v[1-9]*'
    },
)
