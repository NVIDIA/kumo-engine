# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os.path as osp
import sys
import traceback

from result_analysis import Analysis

fixture = '${0}'
version1 = '${1}'.replace(".", "_")
version2 = '${2}'.replace(".", "_")

# Configure logging to stdout so messages appear locally and in EMR step logs
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logging.info(
    f"Starting H&M acceptance: fixture={fixture} version1={version1} version2={version2}"
)

output_rootdir = (f"s3://kumocloud-test-bucket/ux_e2e_stability_tests"
                  f"/predictions/test/hnm_acc_testing_{version1}_{version2}")

path_template = (
    "s3://kumocloud-test-bucket/ux_e2e_stability_tests/predictions/"
    f"test/{version1}/{fixture}/{fixture}_predictions")

summary_file = 'result_summary.txt'

logging.info(f"Output root dir: {output_rootdir}")
logging.info(f"Predictions path template: {path_template}")
try:
    failures = []
    output_path = osp.join(output_rootdir, f'pred_{fixture}_{version1}')
    logging.info(f"Creating Analysis at: {output_path}")
    analysis = Analysis(output_path)
    logging.info("Running same-version stability (jaccard > 0.78)...")
    jaccard_avg = analysis.prediction_comp(
        job1=path_template.format(fixture, version1),
        entity1='customer_id',
        target1='CLASS',
        job2=path_template.format(fixture, version1) + "_2",
        entity2='customer_id',
        target2='CLASS',
        comp_type='jaccard',
    )
    logging.info(f"Same-version jaccard_avg: {jaccard_avg}")
    if not (jaccard_avg > 0.78):
        msg = (f"Same-version jaccard_avg too low: {jaccard_avg} (<= 0.78)")
        logging.warning(msg)
        failures.append(msg)

    pair = (version2, version1)
    path_template = (
        "s3://kumocloud-test-bucket/ux_e2e_stability_tests/"
        f"predictions/test/{version1}/{fixture}/{fixture}_predictions")

    old_version, new_version = pair
    output_path = osp.join(output_rootdir,
                           f'pred_{fixture}_{old_version}_{new_version}')
    logging.info(f"Creating Analysis at: {output_path}")
    analysis = Analysis(output_path)

    logging.info("Running cross-version stability (jaccard > 0.25)...")
    jaccard_avg = analysis.prediction_comp(
        job1=path_template.format(fixture, old_version),
        entity1='customer_id',
        target1='CLASS',
        job2=path_template.format(fixture, new_version),
        entity2='customer_id',
        target2='CLASS',
        comp_type='jaccard',
    )
    logging.info(f"Cross-version jaccard_avg: {jaccard_avg}")
    if not (jaccard_avg > 0.25):
        msg = (f"Cross-version jaccard_avg too low: {jaccard_avg} (<= 0.25)")
        logging.warning(msg)
        failures.append(msg)

    path_template = (
        "s3://kumocloud-test-bucket/ux_e2e_stability_tests/"
        f"predictions/test/{version1}/{fixture}/{fixture}_embeddings")

    # note: COMMON_ID is treated as a special case inside Analysis for which
    # connector id isn't used in in filtering for article or customer embeddings

    output_path = osp.join(output_rootdir,
                           f'embed_articles_{fixture}_{version2}')
    logging.info(f"Creating Analysis at: {output_path}")
    analysis = Analysis(output_path)
    logging.info(
        "Running article embeddings stability (90% quantile cosine > 0.99)...")
    quantile_values = analysis.embedding_dist(
        job1=path_template.format(fixture, version2),
        type1="COMMON_ID.articles",
        job2=path_template.format(fixture, version2) + "_2",
        type2="COMMON_ID.articles",
        dist_type='cosine',
    )
    logging.info(f"Article embeddings quantiles: {quantile_values}")
    if not (quantile_values and len(quantile_values) > 1
            and quantile_values[1] > 0.99):
        msg = (
            "Article embeddings 90% quantile cosine too low: "
            f"{quantile_values[1] if quantile_values and len(quantile_values) > 1 else 'N/A'} (<= 0.99)"
        )
        logging.warning(msg)
        failures.append(msg)

    output_path = osp.join(output_rootdir,
                           f'embed_customers_{fixture}_{version2}')
    logging.info(f"Creating Analysis at: {output_path}")
    analysis = Analysis(output_path)
    logging.info(
        "Running customer embeddings stability (90% quantile cosine > 0.99)..."
    )
    quantile_values = analysis.embedding_dist(
        job1=path_template.format(fixture, version2),
        type1="COMMON_ID.customers",
        job2=path_template.format(fixture, version2) + "_2",
        type2="COMMON_ID.customers",
        dist_type='cosine',
    )
    logging.info(f"Customer embeddings quantiles: {quantile_values}")
    if not (quantile_values and len(quantile_values) > 1
            and quantile_values[1] > 0.99):
        msg = (
            "Customer embeddings 90% quantile cosine too low: "
            f"{quantile_values[1] if quantile_values and len(quantile_values) > 1 else 'N/A'} (<= 0.99)"
        )
        logging.warning(msg)
        failures.append(msg)

    if failures:
        logging.error("Acceptance checks failed:")
        for f in failures:
            logging.error(f"- {f}")
        print("\nAcceptance checks failed:", file=sys.stderr)
        for f in failures:
            print(f"- {f}", file=sys.stderr)
        sys.exit(1)
    else:
        logging.info("All acceptance checks passed.")
except Exception as e:
    logging.exception("Acceptance test failed")
    traceback.print_exc()
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
