# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selected dataset/task/model bindings and their pinned Arena provenance.

Registration describes the requested workload, not successful execution. The
suite's existing required-test inventory remains the execution authority.
"""

import importlib
import json
from copy import deepcopy
from functools import lru_cache

from .data import ROOT


@lru_cache(maxsize=1)
def _catalog():
    return json.loads((ROOT / "workloads/bindings.json").read_text())


def load_bindings():
    records = deepcopy(_catalog()["bindings"])
    result = {row["ref"]: row for row in records}
    if len(result) != len(records) or len({r["binding_id"] for r in records}) != len(
        records
    ):
        raise ValueError("Binding references and registered IDs must be unique")
    return result


def selected_bindings(*, dataset=None, campaign=None):
    return [
        binding
        for binding in load_bindings().values()
        if binding["campaigns"]
        and (dataset is None or binding["selector"]["dataset_id"] == dataset)
        and (campaign is None or campaign in binding["campaigns"])
    ]


def selection(binding):
    """The existing contract/model/check interfaces consume this catalog view."""
    catalog = _catalog()
    details = deepcopy(catalog["semantics"].get(binding.get("semantics"), {}))
    details.setdefault(
        "anchor_binding",
        {
            "split_column": binding["split_column"],
            "targets": {
                name: value["column"] for name, value in binding["targets"].items()
            },
        },
    )
    return {
        **details,
        "case_name": binding.get("case", binding["ref"]),
        "selector": deepcopy(binding["selector"]),
        "models": [binding["representation"]],
        "binding": deepcopy(binding),
    }


def selections(dataset="aml-data/hi-small"):
    """Group shared selectors for existing fixtures and campaign implementations."""
    catalog = _catalog()
    records = selected_bindings(dataset=dataset)
    order = {key: i for i, key in enumerate(catalog["semantics"])}
    records.sort(key=lambda b: order.get(b.get("semantics"), len(order)))
    grouped = {}
    for binding in records:
        row = selection(binding)
        key = json.dumps(row["selector"], sort_keys=True)
        if key in grouped:
            grouped[key]["models"].extend(row["models"])
        else:
            grouped[key] = row
    return list(grouped.values())


def binding_for(selected, family):
    if selected.get("binding", {}).get("representation") == family:
        binding = deepcopy(selected["binding"])
        binding["selector"] = deepcopy(selected["selector"])
        return binding
    matches = [
        b
        for b in load_bindings().values()
        if b["selector"] == selected["selector"] and b["representation"] == family
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one registered dataset/anchors/task/model binding for {selected['selector']} / {family}"
        )
    return matches[0]


def dataset_definition(binding):
    if "dataset" in binding:
        return deepcopy(binding["dataset"])
    catalog = _catalog()
    definition = deepcopy(catalog["datasets"][binding["selector"]["dataset_id"]])
    views = definition.pop("views", {})
    if "source_view" in binding:
        view = views[binding["source_view"]]
        options = {
            **definition.get("source_options", {}),
            **view.get("source_options", {}),
        }
        definition.update(view)
        definition["source_options"] = options
    return definition


def binding_label(binding):
    definition = dataset_definition(binding)
    dataset = definition.get("display_name", binding["selector"]["dataset_id"])
    model = {
        "combined": "Combined",
        "graphsage": "GraphSAGE",
        "nuformer": "nuFormer",
        "pragma": "PRAGMA",
    }.get(binding["representation"], binding["representation"])
    return f"{dataset} / {binding['ref']} {model}"


def task_definition(value):
    if isinstance(value, dict) and "kind" in value:
        return deepcopy(value)
    catalog = _catalog()
    key = (
        value.get("task", value["selector"]["workload_id"])
        if isinstance(value, dict)
        else value
    )
    return deepcopy(key if isinstance(key, dict) else catalog["tasks"][key])


def profile(binding, kind):
    catalog = _catalog()
    key = binding["inputs" if kind == "input" else "anchors"]
    return deepcopy(key if isinstance(key, dict) else catalog[f"{kind}_profiles"][key])


def resolve(path):
    module, name = path.split(":", 1)
    return getattr(importlib.import_module(module), name)


def prepare_binding(data_dir, directory, binding, files, *, limit, recorder=None):
    """Source-family conversion returns common inputs for every campaign."""
    from .data import prepare_binding as prepare

    return prepare(data_dir, directory, binding, files, limit=limit, recorder=recorder)


def contract_for(schema, binding, *, max_events):
    from .contracts import binding_contract

    return binding_contract(schema, binding, max_events=max_events)


def sequence_capacity(binding, events):
    """Bound encoded fields, calendar/age values and the marker per event."""
    fields = max(
        (len(family[3]) for family in profile(binding, "input").get("families", ())),
        default=0,
    )
    return (fields + 5) * events


def graph_layout(inputs):
    """Resolve node fields and root relations for single- or multi-table graphs."""
    graph = inputs["graph"]
    if "nodes" in graph:
        return graph["nodes"], graph["relations"], graph.get("reverse_hop", False)
    node = graph["node_table"]
    return {node: graph}, dict.fromkeys(graph["relations"], node), True


def aggregate_fields(inputs):
    return [entry[0] for entry in inputs["aggregates"]] + [
        name
        for name, query in inputs.get("aggregate_queries", {}).items()
        if query.get("model", True)
    ]
