# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This tool is used to delete data accumulated in the persist directory of
# older training and batch prediction jobs in S3. It is okay to delete the
# persist_dir of older jobs (even though they are active in StateDB) for the
# following reasons:
# 1. Training_job: Batch prediction job never uses the `persist` directory of
#    the corresponding training job. So, it is okay to delete it.
# 2. Batch_prediction_job: Batch prediction job `persist` directory is not
#    useful after the prediction is complete. So, it is okay to delete it.
# 3. Retraining a pquery: Retraining a pquery will create a new training job,
#    which will have a new persist directory. So, it is okay to delete the
#    old persist directory.
#
# By default, we delete any persist directory older than 30 days. This can be
# changed by passing the --num_days_older flag.
#
# Usage:
# python tools/cleanup_s3_persist.py --s3_path <s3 path> --dry_run
# where s3 path can be s3://kumo-dev-dataplane-bucket/test/kiriti-i-0e7e948954ab7f809/jobs/ # noqa: E501
# Use the --dry_run flag to check how much space will be freed up without
# actually deleting anything.
#
# If the s3 path is a non-jobs directory, the script will recursively search
# all the subdirectories for 'jobs' directories and delete the persist
# directories of older jobs.
# E.g.
# python tools/cleanup_s3_persist.py --s3_path s3://kumo-dev-dataplane-bucket --dry_run # noqa: E501
#
# Delete persist directories older than 60 days:
# python tools/cleanup_s3_persist.py --s3_path <s3 path> --num_days_older 60 # noqa: E501
import argparse
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List

import boto3

from io_utils import fsutil, kumo_dataclass
from io_utils.fsutil import s3_filesystem

DAY = 60 * 60 * 24
s3_client = boto3.client('s3')
_global_total_space_saved = 0

IGNORE_DIRS = {'db'}


@kumo_dataclass
class PersistDirStats:
    """Stats about persist directory of a job in S3."""
    # The total size of all files in the directory.
    total_size: int
    last_modified_timestamp: int


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_path', type=str, required=True)
    # Dry_run flag allows the user to check how much memory will be freed up
    # without actually deleting anything.
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument(
        '--num_days_older',
        type=int,
        default=30,
        help="Any persist directory older than this number of days will be "
        "deleted. (Default: 30 days)",
    )
    return parser.parse_args()


def analyze_persist_dir(persist_dir: str) -> PersistDirStats:
    bucket, prefix = s3_filesystem.parse_s3(persist_dir)
    # List all the objects in the persist directory and compute the total size
    # and the maximum last modified time of the files.
    total_size = 0
    last_modified_timestamp = 0
    # NOTE: Max keys is set to 10000 to avoid pagination. If the number of
    # files in that persist_dir is more than 10000, we approximate the size
    # and last modified time.
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix,
                                     MaxKeys=10000)
    if 'Contents' not in resp:
        print("Error with response: ", resp)
        return PersistDirStats(total_size=0, last_modified_timestamp=0)
    for obj in resp['Contents']:  # type: ignore
        total_size += obj['Size']  # type: ignore
        last_modified_timestamp = max(
            last_modified_timestamp,
            int(obj['LastModified'].timestamp()))  # type: ignore
    return PersistDirStats(total_size=total_size,
                           last_modified_timestamp=last_modified_timestamp)


def try_clean_up_dir(dir_path: str, num_days_older: int,
                     dry_run: bool) -> float:
    if not fsutil.exists(dir_path):
        return 0
    stats = analyze_persist_dir(dir_path)
    if stats.total_size == 0:
        return 0
    total_space_saved = 0
    # Delete the persist directory if it is older than specified no.of days
    if (datetime.now().timestamp() - stats.last_modified_timestamp
            > num_days_older * DAY):
        total_space_saved += stats.total_size
        if not dry_run:
            print(f"Deleting {dir_path} of size "
                  f"{stats.total_size/(1024*1024.0)} MB")
            try:
                fsutil.rm(dir_path)
            except Exception as e:
                # Ignore error and continue.
                print(e)
        else:
            print(f"Dryrun deleting {dir_path} of size "
                  f"{stats.total_size/(1024*1024.0)} MB")
    return total_space_saved


def maybe_cleanup_jobs_dir(jobs_dir: str, num_days_older: int,
                           dry_run: bool) -> float:
    """Delete persist directories for jobs older than :obj:`num_days_older`
    Returns the space saved in GB."""
    total_space_saved = 0
    if jobs_dir.endswith('jobs'):
        for job_dir in fsutil.listdir(jobs_dir):
            total_space_saved += try_clean_up_dir(osp.join(job_dir, 'persist'),
                                                  num_days_older, dry_run)

            total_space_saved += try_clean_up_dir(
                osp.join(job_dir, 'prediction_output'), num_days_older, dry_run)

            total_space_saved += try_clean_up_dir(
                osp.join(job_dir, 'intermediate_train_tables'), num_days_older,
                dry_run)

            total_space_saved += try_clean_up_dir(
                osp.join(job_dir, 'train_table_intermediate'), num_days_older,
                dry_run)

    if jobs_dir.endswith('data-share'):
        pass
    if jobs_dir.endswith('data-persist-dir'):
        # Clean up ingested data.
        tables = fsutil.listdir(osp.join(jobs_dir, 'ingest'))
        for table in tables:
            versions = fsutil.listdir(table)
            for version in versions:
                total_space_saved += try_clean_up_dir(version, num_days_older,
                                                      dry_run)

        # Clean up shared materalized feature store.
        tables = fsutil.listdir(osp.join(jobs_dir, 'materialize', 'tables'))
        for table in tables:
            versions = fsutil.listdir(table)
            for version in versions:
                total_space_saved += try_clean_up_dir(version, num_days_older,
                                                      dry_run)
        # Clean up shared materalized edges store.
        edges = fsutil.listdir(osp.join(jobs_dir, 'materialize', 'edges'))
        for edge in edges:
            versions = fsutil.listdir(edge)
            for version in versions:
                total_space_saved += try_clean_up_dir(version, num_days_older,
                                                      dry_run)

    global _global_total_space_saved
    _global_total_space_saved += total_space_saved / (1024 * 1024 * 1024.0)
    print(f"Analyzing jobs_dir: {jobs_dir}, "
          f"accumulated saving: {_global_total_space_saved}GB")
    return total_space_saved / (1024 * 1024 * 1024.0)


FINDERS = []


# Keep calling fsutil.listdir recursively until we don't find any
# jobs directories.
def find_jobs_helper(path: str, jobs_dirs: List[str], depth: int,
                     max_depth: int, executor: ThreadPoolExecutor) -> int:
    if depth > max_depth:
        return 0

    # Exit early if `jobs` directory exists
    subfolders = fsutil.listdir(path)
    subbasenames = [osp.basename(v) for v in subfolders]
    if 'jobs' in subbasenames or 'data-persist-dir' in subbasenames:
        jobs_dirs.append(osp.join(path, 'jobs'))
        jobs_dirs.append(osp.join(path, 'data-persist-dir'))
        return 0

    for i, dir_path in enumerate(subfolders):
        if dir_path == path:
            continue

        # Prune some known directories
        basename = subbasenames[i]
        if basename in IGNORE_DIRS:
            continue
        elif (basename.startswith('training')
              or basename.startswith('batch-prediction')
              or basename.startswith('bp-job')):
            # Ignore the training and batch-prediction directories since
            # each job internally contains a 'jobs' folder.
            continue
        else:
            FINDERS.append(
                executor.submit(
                    find_jobs_helper,
                    path=dir_path,
                    jobs_dirs=jobs_dirs,
                    depth=depth + 1,
                    max_depth=max_depth,
                    executor=executor,
                ))
    return 1


def find_all_jobs_dirs(path: str, max_depth: int = 5):
    r"""Finds all the 'jobs' directories under the given path which contain
    entries for training and prediction jobs.
    Args:
        path(str): The path to search for 'jobs' directories.
        max_depth(int): The maximum depth to search for 'jobs' directories.
            If set to 0, only the given path will be searched."""
    path = path.rstrip("/")
    if osp.basename(path) == 'jobs':
        return [path]

    result = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        FINDERS.append(
            executor.submit(
                find_jobs_helper,
                path=path,
                jobs_dirs=result,
                depth=0,
                max_depth=max_depth,
                executor=executor,
            ))
        for f in FINDERS:
            f.result()
    return result


if __name__ == "__main__":
    args = parse_args()

    assert fsutil.exists(args.s3_path), "s3 bucket doesn't exist"
    # Find all the jobs directories in that bucket.
    jobs_dirs = find_all_jobs_dirs(args.s3_path)
    print(f"Found {len(jobs_dirs)} jobs directories")
    tasks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for jobs_dir in jobs_dirs:
            task = executor.submit(
                maybe_cleanup_jobs_dir,
                jobs_dir=jobs_dir,
                num_days_older=args.num_days_older,
                dry_run=args.dry_run,
            )
            tasks.append(task)
        for task in tasks:
            task.result()

    if args.dry_run:
        print(f"Estimated space savings: {_global_total_space_saved} GB")
    else:
        print(f"Total space saved: {_global_total_space_saved} GB")
