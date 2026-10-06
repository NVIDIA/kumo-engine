# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta
from typing import List

import pytest

from rest.backends.ml_backend.job_models import (
    GenericJob,
    JobFilterMetadata,
    JobSortField,
    ListJobsProcessor,
    ListJobsQuery,
)
from statedb.config.common import (
    JobEventLogEntry,
    JobExecution,
    JobSource,
    JobStatus,
    JobType,
    ProjectID,
)

_PROJECT_ID = ProjectID.generate()


# Override the heavy test/rest/conftest.py autouse fixtures (which spin up the
# full REST app/driver) with no-ops; these tests only exercise job list helpers.
@pytest.fixture(scope="session", autouse=True)
def setup_rest_test_session() -> None:
    pass


@pytest.fixture(autouse=True)
def auto_cleanup_per_func() -> None:
    pass


@pytest.fixture
def sample_jobs() -> List[GenericJob]:
    """Create a list of sample jobs for testing."""
    base_time = datetime(2024, 1, 1, 12, 0)
    jobs = []

    for i in range(10):
        job_type = (JobType.TRAINING_JOB if i %
                    2 == 0 else JobType.BATCH_PREDICTION_JOB)
        status = (JobStatus.DONE if i % 3 == 0 else JobStatus.FAILED if i %
                  3 == 1 else JobStatus.RUNNING)

        # Create tags for every third job
        tags = {
            "environment": "production",
            "owner": f"user{i}"
        } if i % 3 == 0 else {}

        execution = JobExecution(
            status=status, log=JobEventLogEntry(
                stage_name='Done',
                last_updated_at=datetime.now(),
            ), last_updated_at=base_time + timedelta(hours=i + 1),
            start_time=base_time + timedelta(hours=i))

        source = JobSource.SDK if i % 2 == 0 else JobSource.UI

        # Even-index jobs are training (Models), odd-index are batch
        # prediction (Predictions); assign a distinct user to each group so
        # user search can be exercised across both job types.
        user = "Alice Anderson" if i % 2 == 0 else "Bob Brown"

        jobs.append(
            GenericJob(
                project_id=_PROJECT_ID, job_id=f"job_{i+1}", job_type=job_type,
                execution=execution, metadata=JobFilterMetadata(
                    start_time=base_time + timedelta(hours=i),
                    end_time=base_time + timedelta(hours=i + 1)
                    if status != JobStatus.RUNNING else None,
                    created_at=base_time + timedelta(hours=i), job_type=job_type,
                    status=status, tags=tags), source=source, user=user))

    return jobs


def test_apply_filters_time_range(sample_jobs):
    base_time = datetime(2024, 1, 1, 12, 0)
    query = ListJobsQuery(project_id=_PROJECT_ID,
                          time_range_start=base_time + timedelta(hours=2),
                          time_range_end=base_time + timedelta(hours=5))

    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert len(filtered_jobs) == 4  # Jobs starting between hour 2 and 5

    assert all(job.metadata.start_time >= query.time_range_start
               for job in filtered_jobs)
    assert all(job.metadata.start_time <= query.time_range_end
               for job in filtered_jobs)


def test_apply_filters_status(sample_jobs):
    query = ListJobsQuery(project_id=_PROJECT_ID, status=[JobStatus.DONE])

    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert len(filtered_jobs) == 4  # Every third job is DONE
    assert all(job.metadata.status == JobStatus.DONE for job in filtered_jobs)


def test_apply_filters_job_type(sample_jobs):
    query = ListJobsQuery(project_id=_PROJECT_ID,
                          job_type=[JobType.TRAINING_JOB])

    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert len(filtered_jobs) == 5  # Every second job is TRAINING_JOB
    assert all(job.metadata.job_type == JobType.TRAINING_JOB
               for job in filtered_jobs)


def test_apply_filters_tags(sample_jobs):
    query = ListJobsQuery(project_id=_PROJECT_ID, tags_filter=["production"])

    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert len(filtered_jobs) == 4  # Every third job has tags
    assert all("production" in job.metadata.tags.values()
               for job in filtered_jobs)


def test_apply_filters_search_keyword(sample_jobs):
    # Test job ID match - Note: job_1 and job_10 both match search term "job_1"
    query = ListJobsQuery(project_id=_PROJECT_ID, search_keyword="job_1")
    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert [job.job_id for job in filtered_jobs] == ["job_1", "job_10"]

    # Test exact job ID match
    query = ListJobsQuery(project_id=_PROJECT_ID, search_keyword="job_2")
    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert len(filtered_jobs) == 1
    assert filtered_jobs[0].job_id == "job_2"

    # Test job type match
    query = ListJobsQuery(project_id=_PROJECT_ID, search_keyword="training")
    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert all(job.metadata.job_type == JobType.TRAINING_JOB
               for job in filtered_jobs)

    # Test tag match
    query = ListJobsQuery(project_id=_PROJECT_ID, search_keyword="production")
    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert all("production" in job.metadata.tags.values()
               for job in filtered_jobs)


def test_apply_filters_search_keyword_user(sample_jobs):
    # Full-name match on user (Alice -> training/Models jobs)
    query = ListJobsQuery(project_id=_PROJECT_ID,
                          search_keyword="Alice Anderson")
    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert len(filtered_jobs) == 5
    assert all(job.user == "Alice Anderson" for job in filtered_jobs)
    assert all(job.metadata.job_type == JobType.TRAINING_JOB
               for job in filtered_jobs)

    # Case-insensitive partial match on user
    query = ListJobsQuery(project_id=_PROJECT_ID, search_keyword="alice")
    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert len(filtered_jobs) == 5
    assert all(job.user == "Alice Anderson" for job in filtered_jobs)

    # User search must also reach batch-prediction (Predictions) jobs
    query = ListJobsQuery(project_id=_PROJECT_ID, search_keyword="bob")
    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert len(filtered_jobs) == 5
    assert all(job.user == "Bob Brown" for job in filtered_jobs)
    assert all(job.metadata.job_type == JobType.BATCH_PREDICTION_JOB
               for job in filtered_jobs)

    # Non-matching user returns nothing
    query = ListJobsQuery(project_id=_PROJECT_ID, search_keyword="Charlie")
    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert len(filtered_jobs) == 0


def test_apply_sorting_default(sample_jobs):
    query = ListJobsQuery(project_id=_PROJECT_ID,
                          )  # Default sort is by start_time descending
    sorted_jobs = ListJobsProcessor.apply_sorting(query, sample_jobs)
    start_times = [job.metadata.start_time for job in sorted_jobs]
    # Default sort should be newest first
    assert start_times == sorted(start_times, reverse=True)


def test_apply_sorting_job_id(sample_jobs):
    query = ListJobsQuery(project_id=_PROJECT_ID, sort_by=JobSortField.JOB_ID,
                          sort_desc=True)

    sorted_jobs = ListJobsProcessor.apply_sorting(query, sample_jobs)
    job_ids = [job.job_id for job in sorted_jobs]
    assert job_ids == sorted(job_ids, reverse=True)


def test_apply_sorting_start_time(sample_jobs):
    query = ListJobsQuery(project_id=_PROJECT_ID,
                          sort_by=JobSortField.START_TIME)

    sorted_jobs = ListJobsProcessor.apply_sorting(query, sample_jobs)
    start_times = [job.metadata.start_time for job in sorted_jobs]
    assert start_times == sorted(start_times)


def test_apply_sorting_duration(sample_jobs):
    # In our implementation, jobs with no end_time return float('-inf') for
    # duration. Let's check the running jobs in the sample data.
    running_jobs = [
        job for job in sample_jobs if job.metadata.status == JobStatus.RUNNING
    ]
    assert len(running_jobs
               ) > 0, "Test requires at least one running job in sample_jobs"

    # Test ascending sort
    query = ListJobsQuery(project_id=_PROJECT_ID, sort_by=JobSortField.DURATION,
                          sort_desc=False)
    sorted_jobs = ListJobsProcessor.apply_sorting(query, sample_jobs)

    # Running jobs (no end_time) should be first with ascending sort.
    assert sorted_jobs[0].metadata.status == JobStatus.RUNNING
    assert sorted_jobs[0].metadata.end_time is None

    # Test descending sort
    query = ListJobsQuery(project_id=_PROJECT_ID, sort_by=JobSortField.DURATION,
                          sort_desc=True)
    sorted_jobs = ListJobsProcessor.apply_sorting(query, sample_jobs)

    # Check the sorted order - running jobs should be at the end
    # (We know that running jobs should be placed last in the list)
    for job in sorted_jobs[:-len(running_jobs)]:
        assert job.metadata.status != JobStatus.RUNNING

    # Check that all running jobs are at the end
    for job in sorted_jobs[-len(running_jobs):]:
        assert job.metadata.status == JobStatus.RUNNING
        assert job.metadata.end_time is None


def test_queued_job_uses_submission_time_for_list_operations(sample_jobs) -> None:
    queued, completed = sample_jobs[:2]
    submitted_at = queued.metadata.created_at
    assert completed.metadata.start_time is not None
    assert submitted_at < completed.metadata.start_time
    queued.metadata.start_time = None
    queued.metadata.end_time = None
    queued.metadata.status = JobStatus.QUEUED
    jobs = [queued, completed]

    assert ListJobsProcessor.apply_sorting(
        ListJobsQuery(project_id=_PROJECT_ID), jobs) == [completed, queued]
    assert ListJobsProcessor.apply_sorting(
        ListJobsQuery(
            project_id=_PROJECT_ID,
            sort_by=JobSortField.DURATION,
        ), jobs) == [queued, completed]
    assert ListJobsProcessor.apply_filters(
        ListJobsQuery(
            project_id=_PROJECT_ID,
            time_range_start=submitted_at,
            time_range_end=submitted_at,
        ), jobs) == [queued]


def test_apply_pagination(sample_jobs):
    # Test first page
    query = ListJobsQuery(project_id=_PROJECT_ID, limit=3, page=0)
    paginated_jobs = ListJobsProcessor.apply_pagination(query, sample_jobs)
    assert len(paginated_jobs) == 3
    assert paginated_jobs == sample_jobs[:3]

    # Test second page
    query = ListJobsQuery(project_id=_PROJECT_ID, limit=3, page=1)
    paginated_jobs = ListJobsProcessor.apply_pagination(query, sample_jobs)
    assert len(paginated_jobs) == 3
    assert paginated_jobs == sample_jobs[3:6]

    # Test last incomplete page
    query = ListJobsQuery(project_id=_PROJECT_ID, limit=3, page=3)
    paginated_jobs = ListJobsProcessor.apply_pagination(query, sample_jobs)
    assert len(paginated_jobs) == 1
    assert paginated_jobs == sample_jobs[9:]

    # Test empty page (beyond available data)
    query = ListJobsQuery(project_id=_PROJECT_ID, limit=3, page=4)
    paginated_jobs = ListJobsProcessor.apply_pagination(query, sample_jobs)
    assert len(paginated_jobs) == 0


def test_combined_operations(sample_jobs):
    # Filter for completed training jobs, sort by start time, and get first
    # page
    query = ListJobsQuery(project_id=_PROJECT_ID, status=[JobStatus.DONE],
                          job_type=[JobType.TRAINING_JOB],
                          sort_by=JobSortField.START_TIME, limit=2, page=0)

    # Apply operations in sequence
    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    sorted_jobs = ListJobsProcessor.apply_sorting(query, filtered_jobs)
    paginated_jobs = ListJobsProcessor.apply_pagination(query, sorted_jobs)

    assert len(paginated_jobs) <= 2
    assert all(job.metadata.status == JobStatus.DONE for job in paginated_jobs)
    assert all(job.metadata.job_type == JobType.TRAINING_JOB
               for job in paginated_jobs)

    # Verify sorting
    start_times = [job.metadata.start_time for job in paginated_jobs]
    assert start_times == sorted(start_times)
