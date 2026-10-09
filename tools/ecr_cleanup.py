# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage:
# python tools/ecr_cleanup.py --repo <repository-name> --delete
# where repository-name is the name of the ECR repository to cleanup.
# Use the --delete flag to actually delete the images.
#
# Example:
# python tools/ecr_cleanup.py --repo kumo/kumo-rest --delete
#
# This will delete all images in the kumo/kumo-rest repository that are older than 30 days.
#

import argparse
import re
from datetime import datetime, timedelta, timezone

import boto3

# --- Config ---
REGION = "us-west-2"
RETENTION_DAYS = 30

# --- Tag exclusion patterns ---
# Exclude pattern for versions and any image tag that is used for a BP run over the last 60 days
# as of October 20, 2025.
EXCLUDE_TAG_PATTERNS = [
    re.compile(r"^v?\d+\.\d+\.\d+(-[a-zA-Z0-9]+(\.\d+)*)?(?:_databricks)?$"),  # v1.2.3, 1.0.0-beta, v2.3.4-rc.1[_databricks]
    re.compile(r"^v\d+\.\d+(?:_databricks)?$"),                                 # vX.Y[_databricks]
    re.compile(r"^v\d+\.\d+\.0(?:_databricks)?$"),                              # vX.Y.0[_databricks]
    re.compile(r"^v\d{3,}(?:_databricks)?$"),                                   # vXYZ[_databricks]
    re.compile(r"^base-.*"),                                                     # base-*
    re.compile(r"^852560652af9446381f6bd1ea96b101816bca899$"),
    re.compile(r"^bumble_sparse_cross_entropy$"),
    re.compile(r"^237f0f922e25f23a03aea5e4ba487fdb0f4ca9f8$"),
    re.compile(r"^3041555ac97e51f0d801dcd15a8ab411d0d40df1$"),
    re.compile(r"^5b1b755fa3450c86036af3f4e5501f66e671301f.*"),
    re.compile(r"^8f869be9a7ae6737166b9c78b6e688fcc5f607a1.*"),
    re.compile(r"^92a0f54153b5fa100386dca417d0e9f843028757.*"),
    re.compile(r"^c20b25383b511b79cd0fb7322ea37ece000a4377.*"),
    re.compile(r"^1d8ca50c4aac1a66cae5fb00b38d32935f3e9bd1.*"),
    re.compile(r"^v2.7.3_incl_temporal_fix.*"),
    re.compile(r"^v2.7.6_incl_temporal_fix.*"),
    re.compile(r"^v2.9.4_incl_temporal_fix.*"),
    re.compile(r"^v2.9.7_incl_temporal_fix.*"),
    re.compile(r"^v210_reddit_fix_ge.*"),
    re.compile(r"^latest.*"),
]


# --- AWS setup ---
def get_ecr_client(region):
    return boto3.client("ecr", region_name=region)


cutoff_date = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)


def should_exclude(tags):
    if not tags:
        return False
    for tag in tags:
        for pattern in EXCLUDE_TAG_PATTERNS:
            if pattern.match(tag):
                return True
    return False


def list_old_images_excluding_tag_patterns(ecr, repo_name):
    paginator = ecr.get_paginator("describe_images")
    page_iterator = paginator.paginate(repositoryName=repo_name)

    images_to_delete = []
    excluded_images = []
    for page in page_iterator:
        for image_detail in page.get("imageDetails", []):
            pushed_at = image_detail.get("imagePushedAt")
            tags = image_detail.get("imageTags", [])

            if not pushed_at or pushed_at > cutoff_date:
                continue

            if should_exclude(tags):
                excluded_images.append({
                    "imageDigest": image_detail["imageDigest"],
                    "imageTags": tags
                })
                continue

            images_to_delete.append({
                "imageDigest": image_detail["imageDigest"],
                "imageTags": tags
            })

    return images_to_delete, excluded_images


def delete_images(ecr, repo_name, images, dry_run=True):
    if not images:
        print("No images to delete.")
        return

    print(f"{'Dry run:' if dry_run else 'Deleting'} {len(images)} images in repo '{repo_name}'...")
    if dry_run:
        for img in images:
            print(f"Would delete: Digest={img['imageDigest']}, Tags={img['imageTags']}")
        return

    response = ecr.batch_delete_image(
        repositoryName=repo_name,
        imageIds=[{"imageDigest": img["imageDigest"]} for img in images]
    )
    print("Deleted images:", response.get("imageIds", []))
    print("Failed to delete images:", response.get("failures", []))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECR image cleanup script")
    parser.add_argument("--delete", action="store_true", help="Actually delete images (default is False)")
    parser.add_argument("--repo", default="kumo/kumo-rest", help="ECR repository name (default: kumo/kumo-rest)")

    args = parser.parse_args()
    ecr_client = get_ecr_client(REGION)
    images, excluded_images = list_old_images_excluding_tag_patterns(ecr_client, args.repo)
    print(f"Excluded {len(excluded_images)} images")
    print(f"Would delete {len(images)} images")
    delete_images(ecr_client, args.repo, images, dry_run=not args.delete)
