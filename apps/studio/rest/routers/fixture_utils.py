# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from kumo_core.types.schema import FPID

from async_utils.sync import blocking_func
from kumo.config.model_plan_yaml import model_plan_from_yaml
from rest.config.api_command import FixtureDefinition
from rest.config.predictive_query import pquery_runtime_options_or_none
from rest.config.predictive_query_v2 import PQueryResource


@blocking_func()
def fixture_to_pquery_resources_v2(
        fixture: FixtureDefinition) -> List[PQueryResource]:
    """Convert credential-free fixture definitions to PQuery resources.

    Fixtures contain no identities, connectors, groups, or credentials. Their
    referenced graph/table resources must already exist in the target project.
    """
    graphs_by_fpid = {graph.fpid: graph for graph in fixture.graphs}
    resources: List[PQueryResource] = []
    for pquery_args in fixture.queries:
        graph = graphs_by_fpid[FPID(pquery_args.graph_fpid)]
        if pquery_args.model_planner_yaml:
            model_plan_from_yaml(pquery_args.model_planner_yaml)
        resources.append(
            PQueryResource(
                name=pquery_args.pquery_name,
                query_string=pquery_args.query_str,
                desc=pquery_args.description,
                graph_display_name=f'graph_{graph.fpid[:8]}',
                graph=graph,
                advanced_options_yaml=pquery_args.model_planner_yaml or None,
                runtime_options=pquery_runtime_options_or_none(
                    list_distinct_target_is_array=(
                        pquery_args.list_distinct_target_is_array),
                    lp_bp_layout=pquery_args.lp_bp_layout,
                ),
            ))
    return resources
