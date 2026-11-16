# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy registry for paper benchmark datasets.

Importing one dataset must not import every optional benchmark dependency.
In particular, METR-LA and the other built-in datasets remain usable without
installing RelBench. Dataset classes are imported only when their registry
entry (or their package-level class export) is accessed.
"""

from collections.abc import Iterator, Mapping
from importlib import import_module
from typing import Any

from .benchmarking_dataset import (
    BenchmarkingDataset,
    ForecastingTables,
    SupervisedTables,
    forecast_benchmark_entity_ids,
)

_DatasetSpec = tuple[str, str]

_DATASET_SPECS: dict[str, _DatasetSpec] = {
    "ibm_aml": ("ibm_aml", "IBMAMLBenchmarkingDataset"),
    "elliptic_one": ("elliptic", "EllipticBenchmarkingDataset"),
    "figraph": ("figraph", "FiGraphBenchmarkingDataset"),
    "figraph_projected": (
        "figraph_projected",
        "FiGraphProjectedBenchmarkingDataset",
    ),
    "ieee_cis_fraud": ("ieee_cis_fraud", "IEEECISFraudBenchmarkingDataset"),
    "yelp_chi": ("yelp_chi", "YelpChiBenchmarkingDataset"),
    "dataco_supply_chain": (
        "dataco_supply_chain",
        "DataCoSupplyChainBenchmarkingDataset",
    ),
    "m5": ("m5", "M5BenchmarkingDataset"),
    "metr_la": ("metr_la", "MetrLABenchmarkingDataset"),
    "y9c": ("y9c", "Y9CBenchmarkingDataset"),
    "dbb_seznam_charge": ("dbinfer", "DBBSeznamCharge"),
    "dbb_seznam_prepay": ("dbinfer", "DBBSeznamPrepay"),
    "dbb_avs_repeater": ("dbinfer", "DBBAVSRepeater"),
    "dbb_amazon_churn": ("dbinfer", "DBBAmazonChurn"),
    "dbb_amazon_rating": ("dbinfer", "DBBAmazonRating"),
    "dbb_diginetica_ctr": ("dbinfer", "DBBDigineticaCTR"),
    "dbb_outbrain_ctr": ("dbinfer", "DBBOutbrainCTR"),
    "dbb_retailrocket_cvr": ("dbinfer", "DBBRetailRocketCVR"),
    "dbb_stackexchange_churn": ("dbinfer", "DBBStackExchangeChurn"),
    "dbb_stackexchange_upvote": ("dbinfer", "DBBStackExchangeUpvote"),
    "salt_item_plant": ("salt", "SALTItemPlantBenchmarkingDataset"),
    "salt_item_shippoint": ("salt", "SALTItemShipPointBenchmarkingDataset"),
    "salt_item_incoterms": ("salt", "SALTItemIncotermsBenchmarkingDataset"),
    "salt_sales_incoterms": (
        "salt",
        "SALTSalesIncotermsBenchmarkingDataset",
    ),
    "salt_sales_office": ("salt", "SALTSalesOfficeBenchmarkingDataset"),
    "salt_sales_group": ("salt", "SALTSalesGroupBenchmarkingDataset"),
    "salt_sales_payterms": ("salt", "SALTSalesPaytermsBenchmarkingDataset"),
    "salt_sales_shipcond": ("salt", "SALTSalesShipcondBenchmarkingDataset"),
}

_EXPORTED_CLASS_SPECS: dict[str, _DatasetSpec] = {
    class_name: spec
    for spec in _DATASET_SPECS.values()
    for class_name in (spec[1], )
}


def _load_class(spec: _DatasetSpec) -> type[BenchmarkingDataset]:
    module_name, class_name = spec
    module = import_module(f"{__name__}.{module_name}")
    dataset_type = getattr(module, class_name)
    if not isinstance(dataset_type, type):
        raise TypeError(f"{module.__name__}.{class_name} is not a class")
    return dataset_type


class _DatasetRegistry(Mapping[str, type[BenchmarkingDataset]]):
    """Mapping that imports a dataset implementation only on value access."""
    def __init__(self, specs: Mapping[str, _DatasetSpec]) -> None:
        self._specs = dict(specs)
        self._classes: dict[str, type[BenchmarkingDataset]] = {}
        self._relbench_loaded = False

    def _load_relbench(self) -> None:
        if self._relbench_loaded:
            return
        self._relbench_loaded = True
        try:
            module = import_module(f"{__name__}.relbench")
        except ModuleNotFoundError as error:
            if error.name == "relbench" or (error.name
                                            or "").startswith("relbench."):
                return
            raise
        self._classes.update(module.get_all_relbench_datasets())

    def __getitem__(self, name: str) -> type[BenchmarkingDataset]:
        cached = self._classes.get(name)
        if cached is not None:
            return cached
        spec = self._specs.get(name)
        if spec is not None:
            dataset_type = _load_class(spec)
            self._classes[name] = dataset_type
            return dataset_type
        self._load_relbench()
        try:
            return self._classes[name]
        except KeyError:
            raise KeyError(name) from None

    def __iter__(self) -> Iterator[str]:
        # Preserve dynamic RelBench task discovery when the optional package
        # is installed. A missing package is ignored, so CLI help can still
        # list every built-in dataset.
        self._load_relbench()
        return iter(dict.fromkeys((*self._specs, *self._classes)))

    def __len__(self) -> int:
        self._load_relbench()
        return len(dict.fromkeys((*self._specs, *self._classes)))


DATASETS: Mapping[str,
                  type[BenchmarkingDataset]] = _DatasetRegistry(_DATASET_SPECS)


def get_dataset(name: str, **kwargs: Any) -> BenchmarkingDataset:
    """Construct one named dataset without importing unrelated benchmarks."""
    try:
        dataset_type = DATASETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(DATASETS)}") from None
    return dataset_type(**kwargs)


def __getattr__(name: str) -> object:
    """Lazily resolve the historical package-level dataset class exports."""
    spec = _EXPORTED_CLASS_SPECS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    dataset_type = _load_class(spec)
    globals()[name] = dataset_type
    return dataset_type


__all__ = [
    "BenchmarkingDataset",
    "ForecastingTables",
    "DATASETS",
    "DataCoSupplyChainBenchmarkingDataset",
    "DBBAVSRepeater",
    "DBBAmazonChurn",
    "DBBAmazonRating",
    "DBBDigineticaCTR",
    "DBBOutbrainCTR",
    "DBBRetailRocketCVR",
    "DBBSeznamCharge",
    "DBBSeznamPrepay",
    "DBBStackExchangeChurn",
    "DBBStackExchangeUpvote",
    "EllipticBenchmarkingDataset",
    "FiGraphBenchmarkingDataset",
    "FiGraphProjectedBenchmarkingDataset",
    "IBMAMLBenchmarkingDataset",
    "IEEECISFraudBenchmarkingDataset",
    "M5BenchmarkingDataset",
    "MetrLABenchmarkingDataset",
    "Y9CBenchmarkingDataset",
    "SALTItemIncotermsBenchmarkingDataset",
    "SALTItemPlantBenchmarkingDataset",
    "SALTItemShipPointBenchmarkingDataset",
    "SALTSalesGroupBenchmarkingDataset",
    "SALTSalesIncotermsBenchmarkingDataset",
    "SALTSalesOfficeBenchmarkingDataset",
    "SALTSalesPaytermsBenchmarkingDataset",
    "SALTSalesShipcondBenchmarkingDataset",
    "YelpChiBenchmarkingDataset",
    "forecast_benchmark_entity_ids",
    "SupervisedTables",
    "get_dataset",
]
