# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to submit the H&M acceptance tests (defined in hnm_acc.py) to an
# EMR cluster. hnm_acc.py has placeholders ${i} for versions to use, so this
# script should be used as, e.g.
# python analysis/run_acc_script.py  --cluster_name preprod
#  --script_args 'hnm_partitioned_store_rank_channels128 v1_20_3 v1_20_0'
# additional placeholders can be added to `hnm_acc` and the values added to
# the --script_args string separated by whitespace.
import argparse
import os.path as osp
import re
import shutil
import time

import boto3
import pandas as pd

from io_utils import fsutil

root_dir = osp.dirname(osp.realpath(__file__))

# Location where current version of analysis zip and evaluation script will be
# placed in S3 so that they can be used on EMR
bucket = 'kumocloud-test-bucket'
prefix = 'ux_e2e_stability_tests/acc_testing'


def print_output(s3_path):
    outputs = fsutil.listdir(s3_path)
    for output_object in outputs:
        df = pd.read_parquet(output_object)
        print(f"\nOutput Artifact: {output_object.split('/')[-1]}")
        print(df)


def update_step_script(template_path, out_path, args_str):
    args_list = args_str.split()
    template_file = open(template_path, 'r')
    template_lines = template_file.readlines()
    template_file.close()
    with open(out_path, 'w') as outfile:
        for line in template_lines:
            # replace ${(i)} with the ith arg
            line = re.sub(r"\${(.*?)}",
                          lambda x: args_list[int(x.group()[2:-1])], line)
            outfile.write(line)

    print(f"Args used to fill in template at {out_path}: {args_list}")


def upload_step_script(local_path):
    s3_client = boto3.client('s3')
    local_name = local_path.split('/')[-1]
    response = s3_client.upload_file(local_path, bucket,
                                     f'{prefix}/{local_name}')
    upload_path = f's3://{bucket}/{prefix}/{local_name}'
    print(response)
    print(f"Script uploaded to {upload_path}.")
    return upload_path


def upload_analysis_zip():
    s3_client = boto3.client('s3')
    local_dir = osp.dirname(osp.realpath(__file__))
    root_dir, base_dir = osp.split(local_dir)
    shutil.make_archive(base_name=local_dir, format='zip', root_dir=root_dir,
                        base_dir=base_dir)
    zip_path = local_dir + '.zip'
    response = s3_client.upload_file(zip_path, bucket,
                                     f'{prefix}/result_analysis.zip')

    upload_path = f's3://{bucket}/{prefix}/result_analysis.zip'
    print(response)
    print(f"Script uploaded to {upload_path}.")
    fsutil.rm(zip_path)
    return upload_path


def get_cluster_id(cluster_name: str, client):
    response = client.list_clusters(ClusterStates=['RUNNING', 'WAITING'])
    while "Clusters" in response and len(response["Clusters"]) > 0:
        print(f'response: {response} num clusters:{len(response["Clusters"])}')
        for cluster in response['Clusters']:
            if cluster['Name'] == cluster_name:
                return cluster['Id']
        marker = response.get('Marker')
        response = client.list_clusters(ClusterStates=['RUNNING', 'WAITING'],
                                        Marker=marker)

    raise ValueError(f'No running cluster with name {cluster_name} found.')


def run(args):
    eval_script_template_path = osp.join(root_dir, args.template_name + '.py')
    eval_script_filled_path = osp.join(root_dir,
                                       args.template_name + '_filled.py')
    update_step_script(eval_script_template_path, eval_script_filled_path,
                       args.script_args)

    if args.dry_run:
        print(
            f"DRY-RUN: run_acc_script.py --cluster_name {args.cluster_name} --template_name {args.template_name} --script_args '{args.script_args}'"
        )
        return

    client = boto3.client('emr', region_name='us-west-2')
    uploaded_script_location = upload_step_script(eval_script_filled_path)
    package_location = upload_analysis_zip()
    cluster_id = get_cluster_id(args.cluster_name, client)
    response = client.add_job_flow_steps(
        JobFlowId=cluster_id,
        Steps=[
            {
                'Name': 'acceptance_test',
                'ActionOnFailure': 'CONTINUE',
                'HadoopJarStep': {
                    'Jar':
                    'command-runner.jar',
                    'Args': [
                        'spark-submit',
                        '--conf',
                        'spark.yarn.dist.archives=s3://kumo-pyspark-venv/env/kumo_pyspark_env-latest.tar.gz#env',  # noqa: E501
                        '--conf',
                        'spark.yarn.appMasterEnv.PYSPARK_PYTHON=./env/bin/python',  # noqa: E501
                        '--conf',
                        'spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=./env/bin/python',  # noqa: E501
                        '--conf',
                        'spark.executorEnv.PYSPARK_PYTHON=./env/bin/python',
                        '--deploy-mode',
                        'cluster',
                        '--master',
                        'yarn',
                        '--jars',
                        's3://kumo-pyspark-venv/jars/*.jar',
                        '--py-files',
                        f'{package_location}',
                        f'{uploaded_script_location}'
                    ]
                }
            },
        ],
    )
    step_id = response['StepIds'][0]
    print(f'START STEP: {step_id}')

    response = client.describe_step(
        ClusterId=cluster_id,
        StepId=step_id,
    )
    step_status = response['Step']['Status']['State']
    print(f"Status {step_status}")
    while step_status == 'PENDING' or step_status == 'RUNNING':
        response = client.describe_step(
            ClusterId=cluster_id,
            StepId=step_id,
        )
        step_status = response['Step']['Status']['State']
        print(f"Status {step_status}")
        if step_status in ['FAILED', 'CANCELLED', 'CANCEL_PENDING']:
            raise ValueError(f"Step {step_id} failed, check the EMR cluster "
                             "in AWS for details. If stdout logs are not "
                             "available on the step, check the application "
                             "logs in the YARN Timeline server for the "
                             "cluster and get logs from the corresponding "
                             "spark application.")
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Preprod EMR cluster to submit to - must have the IAM role to access all
    # files used in the script being run
    parser.add_argument('--cluster_name', type=str, required=True)

    # template name without extension, e.g. 'hnm_acc' or 'dd_acc'
    parser.add_argument('--template_name', type=str, required=True)
    # all args to be filled into the template script as one string, separated
    # by spaces e.g.
    # --script_args 'hnm_partitioned_store_rank_channels128 v1.20.3 v1.20.0'
    # to replace ${0} ${1} and {2}
    parser.add_argument('--script_args', type=str, required=True)
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Print planned actions without executing')

    args = parser.parse_args()
    run(args)
