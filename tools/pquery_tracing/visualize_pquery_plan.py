# SPDX-FileCopyrightText: Copyright (c) 2024 Gerhard Brueckl
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT

# Adapted from https://github.com/gbrueckl/Fabric.Toolbox/blob/58212be03380d45c9798ee610d61f2808776941b/DataEngineering/Library/VisualizeExecutionPlan.py
# takes a directory with pquery traces (either from running kumo with
# ADD_DF_TRACING=True or using `trace_hm_query.py`) and generates and saves
# html visualization of each physical trace (entity, target, combined) saved
# in that location.

import argparse
import contextlib
import io
import math
import os.path as osp
import re
from typing import Any, Optional, Union

from graphviz import Digraph
from pyspark.sql import DataFrame

from io_utils import fsutil


def get_execution_plan(df: DataFrame) -> str:
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        df.explain(mode="cost")

    plan = stdout.getvalue()
    return plan


class PlanNode:
    """PlanNode is a GraphViz node which also represents a step in a sql
    execution plan, including any information from the plan about the
    dataframe or operation details.
    Takes a line from the text output of the plan as input, and parses out all
    information/attributes, including tree level based on indentation.
    """
    id: str
    level: int
    node_type: str
    node_label: str
    parent: Any
    node_tooltip: Optional[str]
    operation: str

    def __init__(self, id, level, node_type, details):
        raise NotImplementedError

    def get_level(self) -> int:
        raise NotImplementedError

    def get_parent(self, skip_ops: list[str] = []) -> Any:
        raise NotImplementedError

    def get_id(self) -> str:
        raise NotImplementedError

    def get_node_type(self) -> str:
        return self.node_type

    def get_node_label(self) -> str:
        raise NotImplementedError

    def get_node_tooltip(self) -> Optional[str]:
        return None

    def get_edge_label(self) -> Optional[str]:
        return None

    def get_edge_tooltip(self) -> Optional[str]:
        return None


def insert_newlines(text: str, length: int = 15) -> str:
    chars = list(text)
    current_length = 0
    for i in range(len(chars)):
        current_length += 1
        if chars[i] == ' ' and current_length >= length:
            chars[i] = '\n'
            current_length = 0
    return ''.join(chars)


class SparkPlanNode(PlanNode):
    plan_type: str
    line: str
    text: str
    line_number: int
    table: Optional[str]
    sub_operation: str

    size_in_bytes: int
    size: Optional[str]

    edge_label: Optional[str]
    edge_tooltip: Optional[str]

    node_matching_text: Optional[str]

    def __init__(self, line, line_number, plan_type):
        if line.startswith("*"):  # replace whole-stage code gen prefix
            line = re.sub(r'([^*]*)\*\([0-9]*\)\s(.*)', r'\1\2', line)

        self.line = line
        self.line_number = line_number
        self.plan_type = plan_type

        self.level = self.get_level()

        self.parent = None

        self.populate_fields()

    def populate_fields(self):
        self.text = self.get_text()
        self.id = self.get_id()
        self.table = self.get_table()
        self.operation = self.get_operation()
        self.sub_operation = self.get_sub_operation()
        self.size_in_bytes = self.get_size_in_bytes()
        self.size = self.get_size()
        self.node_type = self.get_node_type()
        self.node_label = self.get_node_label()
        self.node_tooltip = self.get_node_tooltip()
        self.edge_label = self.get_edge_label()
        self.edge_tooltip = self.get_edge_tooltip()
        self.node_matching_text = self.get_node_matching_text()

    def get_level(self) -> int:
        """
        Infer level based on indentation in text plan
        """
        m = re.search(r'[A-Z]', self.line)
        assert m
        return int(m.start() / 3)

    def get_parent(self, skip_ops: list[str] = []) -> Any:
        if not self.parent:
            return None

        it = self.parent
        while it.get_operation() in skip_ops:
            it = it.parent

        return it

    def get_id(self) -> str:
        return str(self.line_number)

    def get_operation(self) -> str:
        m = re.search(r'([A-Za-z]+)', self.line)
        if m:
            return m.group(1)
        else:
            return self.line

    def get_sub_operation(self) -> str:
        m = re.search(r'^[:\s+-]*([A-Za-z]+)\s([A-Za-z]+)', self.line)
        if m:
            if m.group(1) in ["Join", "FileScan"]:
                return m.group(2)
            return m.group(1)
        else:
            return self.get_operation()

    def get_table(self) -> Optional[str]:
        m = re.search(r'([a-z0-9-_]*)\.([a-z0-9-_]*)\.([a-z0-9-_]*)',
                      self.line)
        if m:
            return f"{m.group(2)}.{m.group(3)}"
        else:
            return None

    def get_size_in_bytes(self) -> int:
        # from https://semyonsinchenko.github.io/ssinchenko/post/estimation-spark-df-size/ # noqa: E501
        m = re.search(r'sizeInBytes\s*=\s*([0-9.]*)\s(.*?)[),]', self.line)
        if m:
            size = float(m.group(1))
            units = m.group(2)
        else:
            return -1

        units = units.replace("i", "")

        size_names = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        i = size_names.index(units)
        p = math.pow(1024, i)

        size = size * p

        return int(size)  # size in byte

    def get_size(self) -> Optional[str]:
        # TODO (Alex): remove since this isn't included in the kumo tracing
        # plans?
        size_bytes = self.get_size_in_bytes()
        if size_bytes == -1:
            return None

        if size_bytes == 0:
            return "0B"

        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    def get_node_type(self) -> str:
        if self.table:
            return "table"
        else:
            if "Join" in self.operation:
                return "join"
            return self.operation.lower()

    def get_node_label(self) -> str:
        if not self.parent:
            return "RESULT"
        return insert_newlines(self.text)

    # Text and mouseover text for HTML renditions of the plan graph:
    def get_node_tooltip(self) -> Optional[str]:
        return self.text

    def get_edge_label(self) -> Optional[str]:
        return self.get_size()

    def get_edge_tooltip(self) -> Optional[str]:
        return None

    def get_text(self) -> str:
        m = re.search(r'^[:\s+-]*(.*)', self.line)
        assert m
        return m.group(1)

    def get_node_matching_text(self) -> Optional[str]:
        if self.node_type == "table":
            return self.table
        if self.operation == "Filter":
            if self.text:
                return self.text.split(',')[0]

        return None


class SnowflakePlanNode(PlanNode):
    details: str
    node_tooltip: Optional[str] = None
    operation: str = ''

    def __init__(self, id, level, node_type, details):
        self.id = id
        self.level = level
        self.node_type = node_type
        self.details = details

    def get_level(self) -> int:
        return self.level

    def get_parent(self, skip_ops: list[str] = []) -> Any:
        return self.parent

    def get_id(self) -> str:
        return self.id

    def get_node_label(self) -> str:
        return insert_newlines(f'{self.node_type}: {self.details}')


def spark_execution_plan_to_nodes(
        exec_plan: str, plan_type: str = "combined") -> list[SparkPlanNode]:
    assert plan_type in ["logical", "physical", "combined"]

    if plan_type == "combined":
        logical_nodes = spark_execution_plan_to_nodes(exec_plan, "logical")
        physical_nodes = spark_execution_plan_to_nodes(exec_plan, "physical")

        for node in physical_nodes:
            log_node = next(
                (nd for nd in logical_nodes
                 if nd.node_matching_text == node.node_matching_text), None)
            if log_node:
                node.size = log_node.size
                node.size_in_bytes = log_node.size_in_bytes
                node.edge_label = log_node.edge_label
                node.edge_tooltip = log_node.edge_tooltip

        return physical_nodes

    lines = [line for line in exec_plan.split("\n") if len(line.strip()) > 0]

    nodes = []
    line_number = 1
    capture_started = False
    for line in lines:
        node = SparkPlanNode(line, line_number, plan_type)
        line_number += 1

        if line.startswith("==") and line.endswith("=="):
            if not capture_started:
                if plan_type == "logical":
                    if "== Optimized Logical Plan ==" in line:
                        capture_started = True
                        node.line = "RESULT"
                        node.level = -1
                elif plan_type == "physical":
                    if line == "== Physical Plan ==":
                        capture_started = True
                        node.line = "RESULT"
                        node.level = -1
                else:
                    raise Exception("Invalid plan_type! Only 'logical'"
                                    " and 'physical' are supported!")
            else:
                break

        if not capture_started:
            continue

        parent = next(
            (nd for nd in reversed(nodes) if nd.level == node.level - 1), None)
        node.parent = parent
        node.populate_fields()

        nodes.append(node)

    return nodes


def snowflake_execution_plan_to_nodes(
        exec_plan: str) -> list[SnowflakePlanNode]:
    # get the part EXPLAIN USING TEXT
    exec_plan = exec_plan.split("Operations:")[1]
    lines = [line for line in exec_plan.split("\n") if len(line.strip()) > 0]

    nodes = []
    for line in lines:
        # Match the line to extract node details
        match = re.match(r'(\d+:\d+)(\s+)->\s*(\w+)\s*(.*)', line)
        if match:
            node = SnowflakePlanNode(id=match.group(1)[2:],
                                     level=len(match.group(2)),
                                     node_type=match.group(3),
                                     details=match.group(4))
            parent = next(
                (nd for nd in reversed(nodes) if nd.level == node.level - 5),
                None)
            node.parent = parent
            nodes.append(node)
    return nodes


# https://graphviz.readthedocs.io/en/stable/examples.html
def get_graph_from_nodes(nodes: Union[list[SparkPlanNode],
                                      list[SnowflakePlanNode]], backend: str,
                         skip_operations: list[str] = []):
    g = Digraph(name="Execution Plan", comment='Execution Plan')

    g.attr(label='')
    g.attr(fontsize='12')

    g.attr('node', shape='box')
    g.attr('node', color='black')

    for node in nodes:
        if node.operation in skip_operations:
            continue

        if "table" in node.get_node_type().lower():
            color = 'lightgreen'
        elif "join" in node.get_node_type().lower():
            color = 'lightblue'
        elif "filter" in node.get_node_type().lower():
            color = 'lightgoldenrodyellow'
        else:
            color = 'white'

        g.node(node.get_id(), node.get_node_label(), style='filled',
               fillcolor=color, tooltip=node.get_node_tooltip())

        if node.get_parent(skip_operations):
            assert node.id
            parent = node.get_parent(skip_operations)
            assert isinstance(parent, PlanNode)
            g.edge(node.id, parent.get_id(), node.get_edge_label(),
                   tooltip=node.get_node_tooltip())
    return g


def load_plan_from_path(file_path: str):
    return fsutil.read_text(file_path)


def get_plan_viz_textfile(file_path: str, backend: str,
                          skip_operations: list[str] = []):
    exec_plan = load_plan_from_path(file_path)
    if backend == 'spark':
        nodes = spark_execution_plan_to_nodes(exec_plan, plan_type="physical")
    else:
        assert backend == 'snowflake'
        nodes = snowflake_execution_plan_to_nodes(exec_plan)

    plan_viz = get_graph_from_nodes(nodes, backend, skip_operations)

    return plan_viz


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--src_dir',
        type=str,
        help='Pquery debug dir (full s3 path \'s3://*debug_pquery/\')',
        required=True,
    )

    parser.add_argument(
        '--dst_dir',
        type=str,
        help='Output dir (defaults to src_dir)',
        default=None,
    )

    parser.add_argument('--backend', type=str,
                        help='spark or snowflake compute backend',
                        choices=['spark', 'snowflake'], default='spark')

    args = parser.parse_args()

    trace_job_dir = args.src_dir
    out_dir = args.dst_dir or trace_job_dir

    files = fsutil.listdir(trace_job_dir)
    plan_files = [f for f in files if 'execute_plan.txt' in f]
    for plan_file in plan_files:
        print(f"Using plan file: {plan_file}")
        plan_vis = get_plan_viz_textfile(plan_file, args.backend)
        plan_name = '.'.join(
            plan_file.split('.')[0:-1]).split('/')[-1].split('_')[1:4]
        print(f"Parsed plan name: {plan_name}")
        plan_name = '_'.join(plan_name)
        plan_vis.format = 'svg'
        html_output = plan_vis.pipe().decode()
        output_path = osp.join(out_dir, plan_name + '.html')
        print(f"Saving visualized graph: {output_path}\n\n")
        fsutil.rm(output_path, ignore_errors=True)
        fsutil.write_text(output_path, html_output)
