#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi

if [[ "$DRY_RUN" == "false" ]]; then
    # Copy the Kumo 3 hotfix script before checking out any release branch so
    # the version we run cannot depend on the branch being swept.
    temp_kumo3_hotfix_script=$(mktemp)
    cp ./scripts/release/make_kumo3_hotfix_release.sh "$temp_kumo3_hotfix_script"
    chmod +x "$temp_kumo3_hotfix_script"
    echo "temp_kumo3_hotfix_script: $temp_kumo3_hotfix_script"
fi

# Get the date 3 months ago
three_months_ago=$(date -d '3 months ago' +%Y-%m-%d)

# Get the date 1 week ago
one_week_ago=$(date -d '1 week ago' +%Y-%m-%d)

# Check if branch arg is set and valid version format
branch_choice=$1
if [[ -n "$branch_choice" ]]; then
	if [[ $branch_choice =~ v[1-9]+\.[0-9]+$ ]]; then
		branches='origin/'$branch_choice
	else
		echo "invalid branch choice"
		exit 1
	fi
else
	echo "branch not specified"
    # Get all branches matching "v<some number>.<some number>"
    branches=$(git branch -r | grep -E 'origin/v[0-9]+\.[0-9]+$')
fi

if [[ "$DRY_RUN" == "false" ]]; then
    touch ../hotfix_versions.txt
fi

# Iterate over each branch
for branch in $branches; do
    # Strip the 'origin/' prefix to get the local branch name
    local_branch=${branch#origin/}

    # Kumo 2 releases are no longer created from this repository. The final
    # supported line, v2.21, has a private, approval-gated maintenance path.
    # Older Kumo 2 branches are unsupported and must not fall through to the
    # Kumo 3 release script.
    if [[ "$local_branch" == "v2.21" ]]; then
        echo "Skipping $branch: v2.21 maintenance moved to:"
        echo "  https://github.com/kumo-ai/kumo-v2-maintenance"
        continue
    fi
    if [[ $local_branch =~ ^v([0-9]+)\.[0-9]+$ ]] \
        && (( BASH_REMATCH[1] < 3 )); then
        echo "Skipping $branch: pre-v3 releases are not maintained from this repository"
        continue
    fi

    # Get the commit date of the branch
    commit_date=$(git log -1 --format=%ci $branch)

    # Convert commit date to a comparable format
    commit_date=$(date -d "$commit_date" +%Y-%m-%d)

    echo 'branch: '$branch' commit date: '$commit_date' three months ago: '$three_months_ago' one week ago: '$one_week_ago
    # Check if the branch was created within the last 3 months
    if [[ "$commit_date" > "$three_months_ago" ]]; then
        # Get the latest commit hash
        latest_commit=$(git log -1 --format=%H $branch)

        # Check if the latest commit is merged within the last week
        merge_date=$(git show -s --format=%ci $latest_commit)
        merge_date=$(date -d "$merge_date" +%Y-%m-%d)
        echo 'latest commit: '$latest_commit' merge date: '$merge_date
        if [[ "$merge_date" > "$one_week_ago" ]]; then
            # Check if the latest commit has no tags
            tags=$(git tag --contains $latest_commit)
            if [[ -z "$tags" ]]; then
                if [[ "$DRY_RUN" == "true" ]]; then
                    echo "DRY-RUN: Would tag a new hotfix release on $branch (latest commit: ${latest_commit:0:7}, merged: $merge_date)"
                    continue
                fi

                echo "Tagging a new minor release on $branch"

                # Checkout the branch
                git checkout --track $branch

                # The only release lines still maintained here are Kumo 3+.
                output=$(bash "$temp_kumo3_hotfix_script" true)
                new_version=$(echo "$output" | grep "New version is" | awk '{print $4}')

                if [[ -n "$new_version" ]]; then
                    echo "$new_version" >> ../hotfix_versions.txt
                    echo "Created new hotfix version: $new_version"
                fi

                # Switch back to master first
                git switch master
            else
                echo "Skipping release for $branch because latest commit $latest_commit already has tags: $tags"
            fi
        else
            echo "Skipping release for $branch because latest commit $latest_commit was merged more than a week ago"
        fi
    fi
done
