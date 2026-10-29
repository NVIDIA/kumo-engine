# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported prepared graph requests before reading data."""

import runpy
import unittest
from pathlib import Path

BUILDER = (Path(__file__).resolve().parents[2] /
           'packages/kumo-engine/test/pr172/kumo/build_graph.py')


class PreparedGraphTest(unittest.TestCase):
    def test_rejects_unknown_graph_task_and_variant_without_source_reads(self):
        self.assertTrue(BUILDER.is_file(), 'Public graph adapter is missing')
        build = runpy.run_path(str(BUILDER))['kumo_graph']
        for graph, task, variant in (
            ('unknown', None, 'full'),
            ('adtech', 'unknown', 'full'),
            ('adtech', 'user_converts', 'unknown'),
        ):
            with self.subTest(graph=graph, task=task, variant=variant):
                with self.assertRaisesRegex(ValueError, 'Unsupported'):
                    build(graph, None, variant=variant, task=task)


if __name__ == '__main__':
    unittest.main()
