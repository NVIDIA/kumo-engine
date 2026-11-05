# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise learning and prediction failures without fitting a model."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('nightly_gates', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)


class LearningAndPredictions(unittest.TestCase):
    def test_ranks_only_change_between_close_scores(self):
        before = dict(ranked_ids=[[1, 2, 3]], ranked_scores=[[.9, .5001, .5]])
        after = dict(ranked_ids=[[1, 3, 2]], ranked_scores=[[.9, .5001, .5]])
        result = nightly.ranked_difference(before, after, atol=.0004)
        self.assertEqual(result['status'], 'PASS')
        for damaged in [
                dict(ranked_ids=[[2, 1, 3]], ranked_scores=[[.9, .5001, .5]]),
                dict(ranked_ids=[[1, 3, 4]], ranked_scores=[[.9, .5001, .5]])
        ]:
            with self.subTest(damaged=damaged), self.assertRaises(ValueError):
                nightly.ranked_difference(before, damaged, atol=.0004)

    def test_rank_measurement_retains_failed_score_and_inversion_bounds(self):
        before = dict(ranked_ids=[[1, 2]], ranked_scores=[[.9, .5]])
        after = dict(ranked_ids=[[2, 1]], ranked_scores=[[.9, .5]])
        result = nightly.ranked_difference(before, after, atol=.0004,
                                           measure_only=True)
        self.assertEqual(result['status'], 'FAIL')
        self.assertAlmostEqual(result['max_abs_aligned_score_difference'], .4)
        self.assertAlmostEqual(result['max_inverted_score_gap'], .4)
        with self.assertRaises(ValueError):
            nightly.ranked_difference(before, dict(after, ranked_ids=[[2, 2]]),
                                      atol=.0004, measure_only=True)

    def test_rank_measurement_reports_missing_cutoff_proof_as_failure(self):
        before = dict(ranked_ids=[[1, 2]], ranked_scores=[[.5, .5]],
                      boundary_ids=[[1, 2, 5]], boundary_scores=[[.5, .5, .5]])
        after = dict(ranked_ids=[[3, 4]], ranked_scores=[[.5, .5]],
                     boundary_ids=[[3, 4, 6]], boundary_scores=[[.5, .5, .5]])
        result = nightly.ranked_difference(before, after, atol=.0004,
                                           measure_only=True)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['unproven_membership_changes'], 1)
        with self.assertRaises(ValueError):
            nightly.ranked_difference(before, after, atol=.0004)

    def test_multiple_cutoff_swaps_use_all_retained_boundary_coordinates(self):
        before = dict(ranked_ids=[[1, 2]], ranked_scores=[[.5001, .5]],
                      boundary_ids=[[1, 2, 3, 4]],
                      boundary_scores=[[.5001, .5, .4999, .4998]])
        after = dict(ranked_ids=[[3, 4]], ranked_scores=[[.5001, .5]],
                     boundary_ids=[[3, 4, 1, 2]],
                     boundary_scores=[[.5001, .5, .4999, .4998]])
        self.assertEqual(
            nightly.ranked_difference(before, after, atol=.0004)['status'],
            'PASS')

    def test_rank_cutoff_changes_need_both_sides_of_the_boundary(self):
        before = dict(ranked_ids=[[1, 2]], ranked_scores=[[.9, .5001]],
                      boundary_ids=[[1, 2,
                                     3]], boundary_scores=[[.9, .5001, .5]])
        after = dict(ranked_ids=[[1, 3]], ranked_scores=[[.9, .5001]],
                     boundary_ids=[[1, 3, 2]], boundary_scores=[[.9, .5001,
                                                                 .5]])
        result = nightly.ranked_difference(before, after, atol=.0004)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['added_items'], 1)
        self.assertEqual(result['removed_items'], 1)
        for damage in [
                dict(after, boundary_ids=[[1, 3, 4]]),
                dict(after, boundary_scores=[[.9, .5001, .4]])
        ]:
            with self.subTest(damage=damage), self.assertRaises(ValueError):
                nightly.ranked_difference(before, damage, atol=.0004)

    def test_installation_binds_one_checked_runtime_to_the_build_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nightly.save(
                root / 'setup/build.json',
                dict(revision='a' * 40, tree='b' * 40, version='1.2.0+abc',
                     wheel_sha256='c' * 64))
            runtime = dict(version='1.2.0+abc', wheel_sha256='c' * 64,
                           installed_files=10)
            nightly.check_install(root, runtime,
                                  build_receipt=root / 'setup/build.json')
            for bad in [
                    dict(runtime, installed_files=0),
                    dict(runtime, wheel_sha256='d' * 64)
            ]:
                with self.assertRaises(ValueError):
                    nightly.check_install(
                        root, bad, build_receipt=root / 'setup/build.json')


if __name__ == '__main__':
    unittest.main()


def test_weekly_retention_accepts_an_exact_decimal_boundary():
    result = nightly.sibling('nightly_gates').case_learning(
        dict(task='binary', arm='normal', score=.725),
        dict(baseline=.5, score=.8),
        dict(min_normal_gain=.01, max_control_drift=.01), tier='weekly')
    assert result['status'] == 'PASS'
