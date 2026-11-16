# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp

from result_analysis import Analysis

fixture = '${0}'
version1 = '${1}'.replace(".", "_")
version2 = '${2}'.replace(".", "_")

output_rootdir = (
    f"s3://kumocloud-test-bucket/ux_e2e_stability_tests"
    f"/predictions/doordash/doordash_acc_testing_{version1}_{version2}")

path_template = (
    "s3://kumocloud-test-bucket/ux_e2e_stability_tests/predictions/"
    f"doordash/{version1}/{fixture}/{fixture}_predictions")

summary_file = 'result_summary.txt'

output_path = osp.join(output_rootdir, f'pred_{fixture}_{version1}')
analysis = Analysis(output_path)
jaccard_avg = analysis.prediction_comp(
    job1=path_template.format(fixture, version1),
    entity1='CONSUMER_ID',
    target1='CLASS',
    job2=path_template.format(fixture, version1) + "_2",
    entity2='CONSUMER_ID',
    target2='CLASS',
    comp_type='jaccard',
)

assert jaccard_avg > 0.8

pair = (version2, version1)
path_template = (
    "s3://kumocloud-test-bucket/ux_e2e_stability_tests/"
    f"predictions/doordash/{version1}/{fixture}/{fixture}_predictions")

old_version, new_version = pair
output_path = osp.join(output_rootdir,
                       f'pred_{fixture}_{old_version}_{new_version}')
analysis = Analysis(output_path)

jaccard_avg = analysis.prediction_comp(
    job1=path_template.format(fixture, old_version),
    entity1='CONSUMER_ID',
    target1='CLASS',
    job2=path_template.format(fixture, new_version),
    entity2='CONSUMER_ID',
    target2='CLASS',
    comp_type='jaccard',
)

assert jaccard_avg > 0.25

path_template = (
    "s3://kumocloud-test-bucket/ux_e2e_stability_tests/"
    f"predictions/doordash/{version1}/{fixture}/{fixture}_embeddings")

# note: COMMON_ID is treated as a special case inside Analysis for which
# connector id isn't used in in filtering for store or user embeddings

output_path = osp.join(output_rootdir, f'embed_stores_{fixture}_{version2}')
analysis = Analysis(output_path)
quantile_values = analysis.embedding_dist(
    job1=path_template.format(fixture, version2),
    type1="COMMON_ID.stores_v2",
    job2=path_template.format(fixture, version2) + "_2",
    type2="COMMON_ID.stores_v2",
    dist_type='cosine',
)
# 90% quantile > 0.99
assert quantile_values[1] > 0.99

analysis = Analysis(
    osp.join(output_rootdir, f'embed_users_{fixture}_{version2}'))
quantile_values = analysis.embedding_dist(
    job1=path_template.format(fixture, version2),
    type1="COMMON_ID.users",
    job2=path_template.format(fixture, version2) + "_2",
    type2="COMMON_ID.users",
    dist_type='cosine',
)
# 90% quantile > 0.99
assert quantile_values[1] > 0.99
