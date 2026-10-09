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
from statedb.config.job_tag import TAG_KEY_ARCHIVED, TAG_VALUE_ARCHIVED

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

    # Create jobs with different types, statuses, start times and tags
    for i in range(10):
        job_type = (JobType.TRAINING_JOB if i %
                    2 == 0 else JobType.BATCH_PREDICTION_JOB)
        status = (JobStatus.DONE if i % 3 == 0 else JobStatus.FAILED if i %
                  3 == 1 else JobStatus.RUNNING)

        # Create a JobExecution instance for each job
        execution = JobExecution(
            status=status, log=JobEventLogEntry(
                stage_name='Done',
                last_updated_at=datetime.now(),
            ), last_updated_at=base_time + timedelta(hours=i + 1),
            start_time=base_time + timedelta(hours=i))

        # Use proper JobSource enum values
        source = JobSource.SDK if i % 2 == 0 else JobSource.UI

        # Add archived tag to every 5th job (jobs 5 and 10)
        tags = {}
        if i % 5 == 4:  # i=4 (job_5) and i=9 (job_10)
            tags[TAG_KEY_ARCHIVED] = TAG_VALUE_ARCHIVED

        jobs.append(
            GenericJob(
                project_id=_PROJECT_ID, job_id=f"job_{i+1}", job_type=job_type,
                execution=execution, metadata=JobFilterMetadata(
                    start_time=base_time + timedelta(hours=i),
                    end_time=base_time + timedelta(hours=i + 1)
                    if status != JobStatus.RUNNING else None,
                    created_at=base_time + timedelta(hours=i), job_type=job_type,
                    status=status, tags=tags), source=source))

    return jobs


def test_apply_filters_time_range(sample_jobs: List[GenericJob]) -> None:
    """Test filtering by time range."""
    base_time = datetime(2024, 1, 1, 12, 0)
    query = ListJobsQuery(project_id=_PROJECT_ID,
                          time_range_start=base_time + timedelta(hours=2),
                          time_range_end=base_time + timedelta(hours=5))

    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    # Jobs starting between hour 2 and 5: job_3, job_4, job_5, job_6
    # But job_5 is archived and filtered out by default, so we get 3 jobs
    assert len(filtered_jobs) == 3

    # Fix datetime comparison with optional values
    assert all(job.metadata.start_time >= query.time_range_start
               for job in filtered_jobs if query.time_range_start is not None)
    assert all(job.metadata.start_time <= query.time_range_end
               for job in filtered_jobs if query.time_range_end is not None)


def test_apply_filters_status(sample_jobs: List[GenericJob]) -> None:
    """Test filtering by job status."""
    query = ListJobsQuery(project_id=_PROJECT_ID, status=[JobStatus.DONE])

    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    # Every third job is DONE: job_1, job_4, job_7, job_10
    # But job_10 is archived and filtered out by default, so we get 3 jobs
    assert len(filtered_jobs) == 3
    assert all(job.metadata.status == JobStatus.DONE for job in filtered_jobs)


def test_apply_filters_job_type(sample_jobs: List[GenericJob]) -> None:
    """Test filtering by job type."""
    query = ListJobsQuery(project_id=_PROJECT_ID,
                          job_type=[JobType.TRAINING_JOB])

    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    # Every second job is TRAINING_JOB: job_1, job_3, job_5, job_7, job_9
    # But job_5 is archived and filtered out by default, so we get 4 jobs
    assert len(filtered_jobs) == 4
    assert all(job.metadata.job_type == JobType.TRAINING_JOB
               for job in filtered_jobs)


def test_apply_filters_search_keyword(sample_jobs: List[GenericJob]) -> None:
    """Test filtering by search keyword."""
    query = ListJobsQuery(project_id=_PROJECT_ID, search_keyword="job_10")

    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    assert all(job.job_id == "job_10" for job in filtered_jobs)


def test_apply_sorting_job_id(sample_jobs: List[GenericJob]) -> None:
    """Test sorting by job ID."""
    query = ListJobsQuery(project_id=_PROJECT_ID, sort_by=JobSortField.JOB_ID,
                          sort_desc=True)

    sorted_jobs = ListJobsProcessor.apply_sorting(query, sample_jobs)
    job_ids = [job.job_id for job in sorted_jobs]
    assert job_ids == sorted(job_ids, reverse=True)


def test_apply_sorting_start_time(sample_jobs: List[GenericJob]) -> None:
    """Test sorting by start time."""
    query = ListJobsQuery(project_id=_PROJECT_ID,
                          sort_by=JobSortField.START_TIME)

    sorted_jobs = ListJobsProcessor.apply_sorting(query, sample_jobs)
    start_times = [job.metadata.start_time for job in sorted_jobs]
    assert start_times == sorted(start_times)


def test_apply_pagination(sample_jobs: List[GenericJob]) -> None:
    """Test pagination."""
    # Test first page
    query = ListJobsQuery(project_id=_PROJECT_ID, limit=3, page=0)
    paginated_jobs = ListJobsProcessor.apply_pagination(query, sample_jobs)
    assert len(paginated_jobs) == 3
    assert paginated_jobs == list(
        sample_jobs)[:3]  # Convert to list before slicing

    # Test second page
    query = ListJobsQuery(project_id=_PROJECT_ID, limit=3, page=1)
    paginated_jobs = ListJobsProcessor.apply_pagination(query, sample_jobs)
    assert len(paginated_jobs) == 3
    assert paginated_jobs == list(sample_jobs)[
        3:6]  # Convert to list before slicing

    # Test last incomplete page
    query = ListJobsQuery(project_id=_PROJECT_ID, limit=3, page=3)
    paginated_jobs = ListJobsProcessor.apply_pagination(query, sample_jobs)
    assert len(paginated_jobs) == 1
    assert paginated_jobs == list(sample_jobs)[
        9:]  # Convert to list before slicing

    # Test empty page (beyond available data)
    query = ListJobsQuery(project_id=_PROJECT_ID, limit=3, page=4)
    paginated_jobs = ListJobsProcessor.apply_pagination(query, sample_jobs)
    assert len(paginated_jobs) == 0


def test_combined_operations(sample_jobs: List[GenericJob]) -> None:
    """Test filters, sorting and pagination working together."""
    # Filter for completed training jobs, sort by start time, and get
    # first page
    query = ListJobsQuery(project_id=_PROJECT_ID, status=[JobStatus.DONE],
                          job_type=[JobType.TRAINING_JOB],
                          sort_by=JobSortField.START_TIME, limit=2, page=0)

    # Apply operations in sequence
    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)
    sorted_jobs = ListJobsProcessor.apply_sorting(query, filtered_jobs)
    paginated_jobs = ListJobsProcessor.apply_pagination(query, sorted_jobs)

    assert len(paginated_jobs) == 2
    assert all(job.metadata.status == JobStatus.DONE for job in paginated_jobs)
    assert all(job.metadata.job_type == JobType.TRAINING_JOB
               for job in paginated_jobs)

    # Verify sorting
    start_times = [job.metadata.start_time for job in paginated_jobs]
    assert start_times == sorted(start_times)


def test_apply_filters_archived_jobs_included(
        sample_jobs: List[GenericJob]) -> None:
    """Test that when TAG_VALUE_ARCHIVED is in tags_filter, only archived jobs are returned."""
    query = ListJobsQuery(project_id=_PROJECT_ID,
                          tags_filter=[TAG_VALUE_ARCHIVED])

    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)

    # Should only return jobs with archived tag (job_5 and job_10)
    assert len(filtered_jobs) == 2
    assert all(
        job.metadata.tags.get(TAG_KEY_ARCHIVED) == TAG_VALUE_ARCHIVED
        for job in filtered_jobs)
    # Verify the specific job IDs
    job_ids = {job.job_id for job in filtered_jobs}
    assert job_ids == {"job_5", "job_10"}


def test_apply_filters_archived_jobs_excluded(
        sample_jobs: List[GenericJob]) -> None:
    """Test that when TAG_VALUE_ARCHIVED is NOT in tags_filter, only non-archived jobs are returned."""
    # Empty tags_filter means we don't want archived jobs
    query = ListJobsQuery(project_id=_PROJECT_ID, tags_filter=[])

    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)

    # Should return all jobs WITHOUT the archived tag (8 jobs)
    assert len(filtered_jobs) == 8
    assert all(
        job.metadata.tags.get(TAG_KEY_ARCHIVED) != TAG_VALUE_ARCHIVED
        for job in filtered_jobs)
    # Verify archived jobs are excluded
    job_ids = {job.job_id for job in filtered_jobs}
    assert "job_5" not in job_ids
    assert "job_10" not in job_ids


def test_apply_filters_archived_with_other_filters(
        sample_jobs: List[GenericJob]) -> None:
    """Test archived filtering combined with other filters."""
    # Filter for non-archived training jobs
    query = ListJobsQuery(project_id=_PROJECT_ID,
                          job_type=[JobType.TRAINING_JOB], tags_filter=[])

    filtered_jobs = ListJobsProcessor.apply_filters(query, sample_jobs)

    # Should return training jobs without archived tag
    # Out of 5 training jobs (job_1, job_3, job_5, job_7, job_9),
    # only job_5 is archived, so we should get 4 jobs
    assert len(filtered_jobs) == 4
    assert all(job.metadata.job_type == JobType.TRAINING_JOB
               for job in filtered_jobs)
    assert all(
        job.metadata.tags.get(TAG_KEY_ARCHIVED) != TAG_VALUE_ARCHIVED
        for job in filtered_jobs)
