# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# 1st party (Kumo) dependencies live under packages/ and are owned by the monorepo.
TORCH_GEOMETRIC_VERSION = "be5e445f73517440a92f26c956753e7f29d45b16"

python_requirement(
    name="diskgraph",
    requirements=[],
    modules=["diskgraph"],
)

# 3rd party major dependencies
# numpy 2.2.x / pandas 2.2.x ship no cp314 wheel (first cp314 wheels: numpy
# 2.3.2, pandas 2.3.3) — leaving them pinned forces a fragile from-sdist
# build on 3.14. Bumped to the cp314-wheel floors; both stay within kumo-core's
# numpy>=2.1,<3 / pandas>=2.2,<3 bounds.
PANDAS_VERSION = "2.3.3"
NUMPY_VERSION = "2.3.3"
# pyarrow <18 has no cp314 wheel and still ships the legacy ParquetDataset
# API; >=18 is the cp314-capable floor. This is a full version specifier (it
# includes the comparison operator), so consumers interpolate it as
# ``pyarrow{PYARROW_VERSION}`` rather than ``pyarrow=={PYARROW_VERSION}``.
# PyArrow 25 is not yet compatible with the DiskGraph extension's parquet
# statistics path (it raises ImportError while computing BP column stats).
PYARROW_VERSION = ">=18,<25"
# temporalio 1.17.0 capped protobuf<6, conflicting with the cp314 protobuf>=6.33
# floor (kumo-api requires protobuf>=5.27 on 3.14). 1.18.0+ relaxed the cap to
# protobuf<7; abi3 (cp39-abi3) wheels install on cp314. Like PYARROW_VERSION,
# this is a full specifier interpolated as ``temporalio{TEMPORALIO_VERSION}``.
# 1.30.0 raises the floor further: its Rust bridge bundles quinn-proto 0.11.15
# (RUSTSEC-2026-0185) and opentelemetry_sdk 0.32.1 (GHSA-w9wp-h8wv-79jx).
TEMPORALIO_VERSION = ">=1.30,<2"
TORCH_VERSION = "2.12.0"
# Security-pinned packages, single-sourced here and parsed by setup.py
# (parse_version_constants) so the Pants requirements and the pip/Docker
# install cannot drift. See the python_requirement blocks below for the
# CVE rationale behind each pin.
FASTAPI_VERSION = "0.138.0"
STARLETTE_VERSION = "1.3.1"
CRYPTOGRAPHY_VERSION = "50.0.0"
PYOPENSSL_VERSION = "26.4.0"
PYJWT_VERSION = "2.13.0"
SQL_METADATA_VERSION = "3.0.1"
SQLPARSE_VERSION = "0.6.0"
# One minor, not a range: sqlglot normalizes Custom SQL text into a digest
# that lands in `graph_id`. Must stay byte-identical to the sqlglot pin in
# `packages/kumo-core/pyproject.toml`; enforced by
# `packages/kumo-core/tests/test_custom_sql_golden.py`.
SQLGLOT_VERSION = "==30.17.*"
HTTPLIB2_VERSION = "0.32.0"
# 6.0.0 reads route.path unguarded while resolving handler names and crashes
# on starlette 1.x's _IncludedRouter entries; 8.x supports starlette>=1,<2.
PROM_FASTAPI_INSTRUMENTATOR_VERSION = "8.0.2"

torch_geometric_git_url = (
    f"git+https://github.com/pyg-team/pytorch_geometric.git@"
    f"{TORCH_GEOMETRIC_VERSION}")
cuda = env("CUDA", "cpu")
platform = env("PANTS_PLATFORM", "Linux")
arch = env("PANTS_ARCH", "x86_64")
torch_version = TORCH_VERSION
if cuda != "cpu":
    torch_version += f"+{cuda}"
elif platform == "Darwin":
    torch_version += f",!={TORCH_VERSION}+cpu"
python_requirement(
    name="torch", requirements=[f"torch=={torch_version}"])

python_requirement(
    name="pytorch_frame",
    requirements=["pytorch-frame==0.3.0"],
    modules=["torch_frame"],
)

python_requirement(
    name="torch_geometric",
    # Preserve the exact unreleased implementation selected before cutover
    # without leaking a build-host-local wheel URL into the Pants lockfile.
    # Pip and Docker use the identical wheel from 3rdparty/kumo-wheels.
    requirements=[f"torch-geometric@ {torch_geometric_git_url}"],
    modules=["torch_geometric"],
)

python_requirement(
    name="pytorch_lightning",
    requirements=["pytorch-lightning==2.6.5"],
    modules=["lightning_fabric", "pytorch_lightning"],
)

python_requirement(name="optuna", requirements=["optuna==4.3.0"])
python_requirement(
    name="matplotlib",
    requirements=["matplotlib>=3.10.5"],
)
python_requirement(name="filelock", requirements=["filelock>=3.29.4"])
python_requirement(name="torchmetrics", requirements=["torchmetrics==1.8.1"])
python_requirement(name="tabulate", requirements=["tabulate==0.8.10"])
python_requirement(name="psutil", requirements=["psutil==7.2.2"])
python_requirement(
    name="python_dateutil",
    requirements=["python-dateutil>=2.9,<3"],
    modules=["dateutil"],
)
python_requirement(name="tenacity", requirements=["tenacity>=8,<10"])
device = "cpu" if cuda == "cpu" else "gpu"
is_linux_arm64_cuda = (
    cuda != "cpu" and platform != "Darwin" and
    (arch == "aarch64" or arch == "arm64")
)
# The published faiss-gpu wheel is x86_64-only. ARM runtime images install
# their separately built FAISS wheel out of band.
faiss_requirements = (
    [] if is_linux_arm64_cuda else
    ["faiss-cpu==1.14.2"] if device == "cpu" else
    ["faiss-gpu==1.14.3"]
)
python_requirement(
    name="faiss",
    requirements=faiss_requirements,
    modules=["faiss"],
)

python_requirement(
    name="antlr4",
    requirements=["antlr4-python3-runtime==4.9.3"],
    modules=["antlr4"],
)

python_requirement(
    name="fastapi",
    requirements=[
        # fastapi >=0.137 allows starlette >=1.3.1 (CVE-2026-48818 /
        # CVE-2026-54283 fixes). PyJWT replaces python-jose, which
        # hard-depends on ecdsa (CVE-2024-23342, no fixed release) and caps
        # pyasn1 <0.5 (CVE-2026-30922).
        f"fastapi=={FASTAPI_VERSION}",
        f"starlette=={STARLETTE_VERSION}",
        "uvicorn==0.38.0",
        f"pyjwt[crypto]=={PYJWT_VERSION}",
        "opentelemetry-instrumentation==0.64b0",
        "opentelemetry-instrumentation-fastapi==0.64b0",
        f"prometheus-fastapi-instrumentator=="
        f"{PROM_FASTAPI_INSTRUMENTATOR_VERSION}",
    ],
)

python_requirement(name="pydantic", requirements=["pydantic>=2.11"])
python_requirement(
    name="pydantic_core",
    requirements=["pydantic-core"],
    modules=["pydantic_core"],
)
python_requirement(name="kubernetes", requirements=["kubernetes==29.0.0"])
python_requirement(name="ec2_metadata", requirements=["ec2-metadata==2.10.0"])
python_requirement(name="aiocache", requirements=["aiocache==0.12.3"])
python_requirement(name="blockbuster", requirements=["blockbuster>=1.5,<1.6"])
python_requirement(name="asgiref", requirements=["asgiref==3.5.2",])
python_requirement(name="pyinstrument", requirements=["pyinstrument==4.6.2"])
python_requirement(
    name="typing_extensions", requirements=["typing_extensions>=4.12.2"])
# opentelemetry-proto <=1.34 capped protobuf<6, conflicting with the cp314
# protobuf>=6.33 floor; 1.35+ relaxed it to protobuf<7. Bump the whole
# opentelemetry family to the 1.43.0 / 0.64b0 line (core 1.43 pairs with the
# 0.64b0 instrumentation/distro series — >=0.62b0 required for starlette 1.x /
# fastapi >=0.137 route trees, opentelemetry-python-contrib#4700); all
# pure-python py3-none-any wheels.
python_requirement(name="opentelemetry", requirements=[
    "opentelemetry-api==1.43.0",
    "opentelemetry-sdk==1.43.0",
    "opentelemetry-exporter-otlp==1.43.0",
    "opentelemetry-distro[otlp]==0.64b0",
])

python_requirement(name="numpy", requirements=[f"numpy=={NUMPY_VERSION}"])
python_requirement(
    name="pandas",
    requirements=[f"pandas=={PANDAS_VERSION}"],
    dependencies=[":numpy"])
python_requirement(
    name="logging_loki", requirements=["python-logging-loki==0.3.1"])
python_requirement(
    name="http",
    requirements=["urllib3>=1.26.18,<3", "requests==2.32.5", "httpx<0.28"])
# Pin httplib2 directly so google-api-python-client cannot retain the
# vulnerable 0.20.4 transitive version in the shipped backend images.
python_requirement(
    name="httplib2", requirements=[f"httplib2=={HTTPLIB2_VERSION}"])
python_requirement(
    name="kumo_sdk_runtime",
    requirements=[
        "aiohttp>=3.10.0",
        "graphviz",
        "plotly",
    ],
    modules=["aiohttp", "graphviz", "plotly"],
)
python_requirement(name="tqdm", requirements=["tqdm>=4.66.0"])
# cryptography 50 / pyOpenSSL 26.4 clear CVE-2026-27459 (critical) and the
# cryptography 38.x CVE set. The historical "newer cryptography rejects
# legacy-cipher Snowflake keys" concern no longer reproduces: 50.0.0 loads
# DES3-, MD5-DES-, RC2- and AES-encrypted PKCS#8 and traditional PEM keys
# identically to 38.0.4 (key decryption is implemented in Rust and no longer
# depends on the bundled OpenSSL's legacy provider).
python_requirement(
    name="cryptography",
    requirements=[
        f"cryptography=={CRYPTOGRAPHY_VERSION}",
        f"pyOpenSSL=={PYOPENSSL_VERSION}",
    ],
)
python_requirement(
    name="yaml", requirements=["pyyaml==6.0.2", "omegaconf==2.3.0"])
python_requirement(name="pyarrow", requirements=[f"pyarrow{PYARROW_VERSION}"])
python_requirement(
    name="snowflake",
    requirements=[
        "snowflake-connector-python>=4.3",
        "snowflake-snowpark-python>=1.24",
    ]
)
python_requirement(name="databricks", requirements=[
    "databricks-sql-connector==4.5.0",
    "databricks-sdk==0.73.0",
    "thrift==0.24.0",
])
python_requirement(name="packaging", requirements=["packaging>=23"])
python_requirement(name="deprecated", requirements=["deprecated==1.3.1"])
python_requirement(name="wrapt", requirements=["wrapt>=1.17,<2"])
python_requirement(
    name="temporalio", requirements=[f"temporalio{TEMPORALIO_VERSION}"])
python_requirement(name="protobuf", requirements=["protobuf>=6.33,<8"])
python_requirement(name="smart_open", requirements=["smart-open==5.2.1"])
# boto3/botocore 1.42.x is the floor for redshift_connector >=2.1.14
# (CVE-2026-8838). aiobotocore 3.7.0 supports botocore 1.42.x; every released
# aioboto3 version caps botocore below that floor.
python_requirement(name="botocore", requirements=["botocore==1.42.90"])
python_requirement(
    name="azure_core",
    requirements=["azure-core>=1.31,<2"],
    modules=["azure.core"],
)
python_requirement(
    name="azure_storage_blob",
    requirements=["azure-storage-blob>=12.27,<13"],
    modules=["azure.storage.blob"],
)
python_requirement(
    name="boto_s3",
    requirements=[
        "boto3==1.42.90",
        "boto3-stubs[s3]==1.42.90",
        "mypy-boto3-s3==1.42.94",
        "s3fs==0.4.2",
        "aiobotocore==3.7.0",
        "types-aiobotocore[s3]==3.7.0",
    ],
    modules=["types_aiobotocore_s3", "aiobotocore", "boto3", "mypy_boto3_s3"],
)
python_requirement(
    name="fsspec",
    requirements=["fsspec>=2025.3.0,<=2025.5.1"],
    modules=["fsspec"],
)
python_requirement(
    name="google_auth",
    requirements=["google-auth>=2.14.1"],
    modules=["google.auth", "google.oauth2"],
)
python_requirement(
    name="gcp_secretmanager",
    requirements=["google-cloud-secret-manager==2.26.0"],
    modules=["google.cloud.secretmanager"],
)
python_requirement(
    name="google_api_core",
    requirements=["google-api-core>=2.0.0"],
    modules=["google.api_core", "google.api_core.exceptions"],
)
python_requirement(
    name="gcp_storage",
    requirements=["google-cloud-storage==3.1.0"],
    modules=["google.cloud.storage"],
)
python_requirement(
    name="gcsfs",
    requirements=["gcsfs==2025.5.1"],
    modules=["gcsfs"],
)
python_requirement(name="zstd", requirements=["zstd>=1.5.6"])
python_requirement(name="decorator", requirements=["decorator==5.2.1"])
python_requirement(name="cachetools", requirements=["cachetools==5.3.1"])
python_requirement(
    name="prometheus_client", requirements=["prometheus_client==0.14.1"])
python_requirement(name="markdown", requirements=["markdown==3.10"])
python_requirement(name="humanize", requirements=["humanize==4.6.0"])
# opentelemetry-instrumentation requires setuptools (for pkg_resources).
# Pin >=80.10.2 for vendored CVE fixes, <82 because 82 removed pkg_resources.
python_requirement(name="setuptools", requirements=["setuptools>=80.10.2,<82"])

# sql-metadata 3.x moved from sqlparse to sqlglot and therefore permits the
# sqlparse 0.6.0 security release. Keep all three requirements explicit so
# image and Pants resolves cannot silently retain an older transitive pin.
python_requirement(
    name="sql_metadata",
    requirements=[f"sql-metadata=={SQL_METADATA_VERSION}"],
)
python_requirement(
    name="sqlparse", requirements=[f"sqlparse=={SQLPARSE_VERSION}"])
python_requirement(name="sqlglot", requirements=[f"sqlglot{SQLGLOT_VERSION}"])
python_requirement(name="sqlalchemy", requirements=["SQLAlchemy>=2.0.14,<2.1"])
# Test-only dependencies
python_requirement(name="pytest", requirements=["pytest>=8.0"])
python_requirement(name="moto", requirements=["moto[s3]==4.2.14"])
python_requirement(name="pytest_asyncio", requirements=["pytest-asyncio==0.23.8"])
