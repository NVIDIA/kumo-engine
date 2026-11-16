# Paper Benchmarks

Benchmarking framework for comparing models on tabular and forecasting
datasets. Benchmark scripts share a common dataset interface. Adding a
new dataset enables the generic benchmarks it has the right structure for.

## Project layout

```
examples/paper/
├── datasets/
│   ├── benchmarking_dataset.py   # Abstract base class
│   ├── ibm_aml.py                # IBM AML reference implementation
│   ├── m5.py                     # M5 Walmart daily retail sales forecasting
│   ├── y9c.py                    # FR Y-9C deposit-yield forecasting
│   ├── YOUR_NEW_DATASET.py
│   └── __init__.py               # Dataset registry (DATASETS dict)
├── benchmark_single_table.py     # LightGBM / XGBoost / TabICL / TabPFN / AutoGluon
├── benchmark_forecasting.py      # Forecasting baselines
├── benchmark_enterprise.py       # Kumo Enterprise
```

## Adding a new dataset

### 1. Create a dataset module

Add a file under `datasets/`, e.g. `my_dataset.py`.
Subclass `BenchmarkingDataset` and implement the abstract methods.
Each method's docstring in `benchmarking_dataset.py` explains what to return
and which benchmark script uses it. Forecasting datasets can additionally
implement `get_forecasting_tables()`.
See `ibm_aml.py` for a complete reference implementation.

### 2. Register the dataset

In `examples/paper/datasets/__init__.py`, import the class and add it
to the `DATASETS` dict:

```python
from .my_dataset import MyDataset

DATASETS: dict[str, type[BenchmarkingDataset]] = {
    "ibm_aml": IBMAMLBenchmarkingDataset,
    "my_dataset": MyDataset,              # <-- add here
}
```

The key (`"my_dataset"`) is the name used on the command line with
`--dataset`.

## Running benchmarks

All commands are run from the **repo root**.

### Single-table (LightGBM, XGBoost, TabICL, TabPFN, AutoGluon)

```bash
python -m examples.paper.benchmark_single_table \
    --dataset my_dataset \
    --table-type entity \
    --model lightgbm

# Subsample for quick iteration or memory-constrained models:
python -m examples.paper.benchmark_single_table \
    --dataset my_dataset \
    --model tabicl \
    --max-train-rows 10000 \
    --max-test-rows 10000
```

| Flag               | Default     | Description                                            |
| ------------------ | ----------- | ------------------------------------------------------ |
| `--dataset`        | `ibm_aml`   | Dataset name (must be in `DATASETS`)                   |
| `--table-type`     | `entity`    | `entity` or `joined`                                   |
| `--model`          | `lightgbm`  | `lightgbm`, `xgboost`, `tabicl`, `tabpfn`, `autogluon` |
| `--max-train-rows` | all         | Cap training rows                                      |
| `--max-test-rows`  | all         | Cap test rows                                          |
| `--val-size`       | `0.2`       | Validation fraction of training set                    |
| `--data-root`      | per-dataset | Override data location                                 |

Running with `--model autogluon` requires `pip install autogluon`.

### Forecasting

Forecasting baselines require the dataset to implement
`get_forecasting_tables()`: train/test rows with entity, timestamp, and target
columns.

```bash
python -m examples.paper.benchmark_forecasting \
    --dataset metr_la \
    --model arima \
    --arima-order 2,0,1

python -m examples.paper.benchmark_forecasting \
    --dataset metr_la \
    --model tabpfn_ts

python -m examples.paper.benchmark_forecasting \
    --dataset metr_la \
    --model chronos2

python -m examples.paper.benchmark_forecasting \
    --dataset metr_la \
    --model timesfm2_5
```

| Flag                        | Default                 | Description                                                    |
| --------------------------- | ----------------------- | -------------------------------------------------------------- |
| `--dataset`                 | `metr_la`               | Dataset name (must be in `DATASETS`)                           |
| `--data-root`               | per-dataset             | Override data location                                         |
| `--model`                   | `arima`                 | Forecasting model to evaluate; see `--help` for choices        |
| `--save`                    | off                     | Write `metrics/` CSV and `predictions/` parquet under this dir |
| `--max-train-rows`          | all                     | Cap training rows, distributed across entities                 |
| `--max-test-rows`           | all                     | Cap test rows, distributed across entities                     |
| `--max-entities`            | all                     | Cap entities for quick smoke tests                             |
| `--n-jobs`                  | `1`                     | Parallel workers                                               |
| `--arima-order`             | `2,0,1`                 | ARIMA order as `p,d,q`                                         |
| `--no-arima-exog`           | off                     | Disable circular temporal exogenous features                   |
| `--tabpfn-ts-features`      | `default`               | TabPFN-TS temporal feature preset                              |
| `--chronos2-features`       | `default`               | Chronos-2 temporal feature preset                              |
| `--chronos2-cross-learning` | off                     | Enable Chronos-2 cross-learning mode                           |
| `--timesfm-features`        | `default`               | TimesFM 2.5 temporal feature preset                            |
| `--toto2-model`             | `Datadog/Toto-2.0-313m` | Toto 2.0 checkpoint (HF repo id), e.g. `Datadog/Toto-2.0-2.5B` |
| `--toto2-context-length`    | `4096`                  | Toto 2.0 history steps fed per entity                          |

`benchmark_forecasting.py` imports forecasting backends at startup, so optional
model packages must be installed before running the script.

TimesFM 2.5 requires:
`pip install "timesfm[torch,xreg] @ git+https://github.com/google-research/timesfm.git"`.

Toto 2.0 requires:
`pip install "toto-2 @ git+https://github.com/DataDog/toto.git#subdirectory=toto2"`.

### Enterprise

Requires a running Kumo API (`KUMO_API_URL`, `KUMO_API_KEY`).

Each dataset defines a `get_pquery()` used for training.

```bash
python -m examples.paper.benchmark_enterprise \
    --dataset my_dataset
```

| Flag             | Default     | Description                                          |
| ---------------- | ----------- | ---------------------------------------------------- |
| `--dataset`      | `ibm_aml`   | Dataset name, or `all`                               |
| `--data-root`    | per-dataset | Override data location                               |
| `--non-blocking` | off         | Launch training and return immediately               |
| `--max-entities` | all         | Report/save eval metrics on first N holdout entities |
| `--save`         | off         | Base dir for holdout `metrics/` and `predictions/`   |
